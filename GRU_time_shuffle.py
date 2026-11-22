"""Seeded time permutations and nearest-snapped nested inner LOPO for the GRU."""

import argparse
import json
import fcntl
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import balanced_accuracy_score
from GRU import ArousalDataset, run_fold, get_segment_dir_name
from nested_LOPO_GRU import SEEDS, scientific, fold_metrics, training_args as gru_training_args
from nested_LOPO_GRU_new import load_data as load_nearest_data, PARTICIPANTS


ROOT = Path("plots/GRU_time_shuffle/nested_LOPO,BP=nearest")
OUTPUT_DIR = ROOT / "segments"
IDENTITY_COLUMNS = ["participant", "run", "label", "segment_start_TR", "segment_end_TR_exclusive"]
HIDDEN_SIZES = (16, 32, 64)
WEIGHT_DECAYS = (1e-2, 3e-3, 1e-3)
COLUMNS = ["outer_held_out_participant", "roi", "hidden", "WD", "seed",
           "n_validation_segments", "segment_pooled_balanced_accuracy"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", action="store_true", help="Summarize all ten outer-evaluation seeds and bootstrap participants")
    parser.add_argument("--get_best_starting_config", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Run outer LOPO using saved best configurations")
    parser.add_argument("--run_inner", action="store_true", help="Run all nine inner-LOPO configurations")
    parser.add_argument("--outer-held-out-participant", choices=PARTICIPANTS)
    parser.add_argument("--seed", type=int, choices=SEEDS,
                        help="Generate one seed only; default generates all ten seeds.")
    args = parser.parse_args()
    if args.summarize and (args.run_inner or args.eval or args.get_best_starting_config or args.seed is not None):
        parser.error("--summarize is a standalone mode")
    if args.run_inner and (args.eval or args.get_best_starting_config):
        parser.error("--run_inner cannot be combined with selection or eval")
    if args.eval and args.seed is None:
        parser.error("--eval requires --seed")
    if args.run_inner and (args.outer_held_out_participant is None or args.seed is None):
        parser.error("--run_inner requires --outer-held-out-participant and --seed")
    if not args.run_inner and args.outer_held_out_participant is not None:
        parser.error("--outer-held-out-participant requires --run_inner")
    return args


def generate(seed):
    from all_rois import all_rois
    args = gru_training_args(argparse.Namespace(rois=list(all_rois), roi_label="all_rois",
                                               bp_TR_snap="nearest"), 64, .001, seed)
    metadata_path = Path("plots/GRU/segments") / get_segment_dir_name(args) / "all_segment_metadata.csv"
    metadata = pd.read_csv(metadata_path).reset_index(drop=True)
    if not (metadata.segment_end_TR_exclusive - metadata.segment_start_TR).eq(40).all():
        raise ValueError("Expected only 40-TR nearest-snapped segments")
    participants = metadata["participant"].astype(str)
    unique_participants = np.unique(participants)
    folds = {participant: fold for fold, participant in enumerate(unique_participants, start=1)}
    rng = np.random.default_rng(seed)
    permutations = np.argsort(rng.random((len(metadata), 40)), axis=1)
    output = pd.DataFrame({
        "segment_index": metadata.index.to_numpy(),
        "fold": participants.map(folds).to_numpy(),
        "participant": metadata["participant"].to_numpy(),
        "run": metadata["run"].to_numpy(),
        "permuted_TR_indices": [json.dumps(row.tolist()) for row in permutations],
    })
    for column in IDENTITY_COLUMNS:
        output[column] = metadata[column].to_numpy()
    output["BP"] = "nearest"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"seed={seed}.csv"
    output.to_csv(path, index=False)
    print(f"Saved {len(output)} segment permutations to {path}.")


def shuffle_segments(X, seed, metadata):
    """Apply saved nearest-cohort permutations after checking segment identities."""
    path = OUTPUT_DIR / f"seed={seed}.csv"
    saved = pd.read_csv(path)
    if saved.segment_index.duplicated().any() or set(saved.segment_index) != set(range(len(X))):
        raise ValueError(f"Saved permutation indices must cover the nearest cohort exactly: {path}")
    saved = saved.set_index("segment_index").loc[np.arange(len(X))]
    if not saved[IDENTITY_COLUMNS].reset_index(drop=True).equals(metadata[IDENTITY_COLUMNS].reset_index(drop=True)):
        raise ValueError(f"Saved permutations do not match nearest segment identities: {path}")
    if not saved.BP.eq("nearest").all():
        raise ValueError(f"Expected nearest-snapped permutations: {path}")
    permutations = np.asarray([json.loads(value) for value in saved.permuted_TR_indices])
    if (permutations.shape != (len(X), X.shape[1])
            or not np.issubdtype(permutations.dtype, np.integer)
            or not np.all(np.sort(permutations, axis=1) == np.arange(X.shape[1]))):
        raise ValueError(f"Invalid saved time permutations: {path}")
    return np.take_along_axis(X, permutations[:, :, None], axis=1)


def run_outer_participant_seed(outer_held_out_participant, seed):
    """Evaluate 9 configurations x 19 inner folds for one outer participant/seed."""
    outer = str(outer_held_out_participant)
    if outer not in PARTICIPANTS or seed not in SEEDS:
        raise ValueError("Use a canonical outer participant and seed")
    X, y, metadata = load_nearest_data()
    y = np.asarray(y, dtype=int)
    metadata = metadata.reset_index(drop=True)
    participants = metadata.participant.astype(str).to_numpy()
    if (X.ndim != 3 or X.shape[1:] != (40, 42) or len(X) != len(y)
            or len(y) != len(metadata) or set(participants) != set(PARTICIPANTS)):
        raise ValueError("Expected aligned nearest-snapped canonical 40-TR, all-42-ROI data")
    dataset = ArousalDataset(shuffle_segments(X, seed, metadata), y)
    inner_participants = [participant for participant in PARTICIPANTS if participant != outer]
    outputs = []
    for hidden in HIDDEN_SIZES:
        for wd in WEIGHT_DECAYS:
            output = (ROOT / "inner_LOPO" / outer / "all_rois"
                      / f"hidden={hidden},WD={scientific(wd)}" / f"seed={seed}.csv")
            if output.is_file() and output.stat().st_size:
                saved = pd.read_csv(output)
                if (list(saved.columns) != COLUMNS or len(saved) != 1
                        or saved.iloc[0].outer_held_out_participant != outer
                        or saved.iloc[0].roi != "all_rois" or saved.iloc[0].hidden != hidden
                        or saved.iloc[0].WD != wd or saved.iloc[0].seed != seed
                        or saved.iloc[0].n_validation_segments != int((participants != outer).sum())
                        or not 0 <= saved.iloc[0].segment_pooled_balanced_accuracy <= 1):
                    raise ValueError(f"Invalid saved inner result: {output}")
                print(f"Skipping completed configuration: {output}", flush=True)
                outputs.append(output)
                continue
            from all_rois import all_rois
            args = gru_training_args(argparse.Namespace(rois=list(all_rois), roi_label="all_rois",
                                                       bp_TR_snap="nearest"), hidden, wd, seed)
            labels, predictions, seen = [], [], []
            for fold, validation_participant in enumerate(inner_participants, start=1):
                train = np.flatnonzero((participants != outer) & (participants != validation_participant))
                validation = np.flatnonzero(participants == validation_participant)
                print(f"Outer={outer} | seed/shuffle_seed={seed} | HD={hidden} | WD={scientific(wd)} | "
                      f"inner {fold}/19: {validation_participant}", flush=True)
                result, _ = run_fold(dataset, train, validation, participants, args,
                                     evaluation_split="validation")
                pred = np.asarray(result["predictions"])
                if pred.shape != y[validation].shape or not np.isin(pred, [0, 1]).all():
                    raise ValueError("Invalid inner validation predictions")
                labels.append(y[validation]); predictions.append(pred); seen.extend(validation.tolist())
            if sorted(seen) != np.flatnonzero(participants != outer).tolist():
                raise RuntimeError("Expected exactly one validation prediction per non-outer segment")
            labels, predictions = np.concatenate(labels), np.concatenate(predictions)
            if set(np.unique(labels)) != {0, 1}:
                raise ValueError("Pooled validation requires both classes")
            row = dict(outer_held_out_participant=outer, roi="all_rois", hidden=hidden, WD=wd,
                       seed=seed, n_validation_segments=len(labels),
                       segment_pooled_balanced_accuracy=balanced_accuracy_score(labels, predictions))
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".csv.tmp")
            pd.DataFrame([row], columns=COLUMNS).to_csv(temporary, index=False)
            temporary.replace(output)
            outputs.append(output)
            print(f"Saved {output}", flush=True)
    return outputs


def get_best_starting_config():
    """Rank nine configurations using ten-seed inner means, separately per outer fold."""
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / ".best_configurations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rankings = {}
        for outer in PARTICIPANTS:
            rows, all_counts = [], set()
            for hidden in HIDDEN_SIZES:
                for wd in WEIGHT_DECAYS:
                    scores, counts = [], []
                    for seed in SEEDS:
                        path = (ROOT / "inner_LOPO" / outer / "all_rois"
                                / f"hidden={hidden},WD={scientific(wd)}" / f"seed={seed}.csv")
                        frame = pd.read_csv(path)
                        if list(frame.columns) != COLUMNS or len(frame) != 1:
                            raise ValueError(f"Invalid inner result: {path}")
                        row = frame.iloc[0]
                        score, count = float(row.segment_pooled_balanced_accuracy), float(row.n_validation_segments)
                        if (row.outer_held_out_participant != outer or row.roi != "all_rois"
                                or row.hidden != hidden or row.WD != wd or row.seed != seed
                                or not np.isfinite(score) or not 0 <= score <= 1
                                or not np.isfinite(count) or count <= 0 or not count.is_integer()):
                            raise ValueError(f"Invalid inner result: {path}")
                        scores.append(score)
                        counts.append(int(count))
                    all_counts.update(counts)
                    rows.append(dict(outer_held_out_participant=outer, roi="all_rois", hidden=hidden,
                                     WD=wd, n_seeds=len(SEEDS), n_validation_segments=counts[0],
                                     mean_segment_pooled_balanced_accuracy=float(np.mean(scores))))
            if len(all_counts) != 1:
                raise ValueError(f"Inconsistent validation counts for {outer}")
            ranking = pd.DataFrame(rows).sort_values(
                ["mean_segment_pooled_balanced_accuracy", "hidden", "WD"],
                ascending=[False, True, True]).reset_index(drop=True)
            ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
            rankings[outer] = ranking
        for outer, ranking in rankings.items():
            path = ROOT / "inner_LOPO" / outer / "best_configurations.csv"
            temporary = path.with_suffix(".csv.tmp")
            ranking.to_csv(temporary, index=False)
            temporary.replace(path)
            best = ranking.iloc[0]
            print(f"{outer}: HD={best.hidden}, WD={scientific(best.WD)}, mean BA={best.mean_segment_pooled_balanced_accuracy:.6f}", flush=True)
        combined = pd.concat([frame.iloc[[0]] for frame in rankings.values()], ignore_index=True)
        path = ROOT / "best_configurations.csv"
        temporary = path.with_suffix(".csv.tmp")
        combined.to_csv(temporary, index=False)
        temporary.replace(path)
        return path


def run_evaluation(seed):
    """Evaluate all 20 outer folds with inner-selected HD/WD and seed-matched shuffles."""
    if seed not in SEEDS:
        raise ValueError("Use a canonical evaluation seed")
    roi_label = "all_rois"
    path = ROOT / "best_configurations.csv"
    selected_frame = pd.read_csv(path)
    if (len(selected_frame) != len(PARTICIPANTS)
            or selected_frame.outer_held_out_participant.duplicated().any()
            or set(selected_frame.outer_held_out_participant) != set(PARTICIPANTS)
            or not selected_frame.roi.eq("all_rois").all()
            or not selected_frame.hidden.isin(HIDDEN_SIZES).all()
            or not selected_frame.WD.isin(WEIGHT_DECAYS).all()
            or not selected_frame.n_seeds.eq(len(SEEDS)).all()
            or not selected_frame.mean_segment_pooled_balanced_accuracy.between(0, 1).all()):
        raise ValueError(f"Invalid best configurations: {path}")
    choices = [row for _, row in selected_frame.iterrows()]
    selected = pd.DataFrame(choices).set_index("outer_held_out_participant")
    X, y, metadata = load_nearest_data()
    metadata = metadata.reset_index(drop=True)
    y = np.asarray(y, dtype=int)
    dataset = ArousalDataset(shuffle_segments(X, seed, metadata), y)
    participants = metadata.participant.astype(str).to_numpy()
    if set(participants) != set(PARTICIPANTS):
        raise ValueError("Evaluation data must contain all canonical outer participants")
    rows, fold_rows = [], []
    for fold, outer in enumerate(PARTICIPANTS, start=1):
        choice = selected.loc[outer]
        from all_rois import all_rois
        args = gru_training_args(argparse.Namespace(rois=list(all_rois), roi_label="all_rois",
                                                   bp_TR_snap="nearest"), int(choice.hidden), float(choice.WD), seed)
        train = np.flatnonzero(participants != outer)
        test = np.flatnonzero(participants == outer)
        print(f"Outer seed={seed} | fold {fold}/{len(PARTICIPANTS)} | held out={outer} | "
              f"ROI={args.roi_label} | WD={scientific(args.weight_decay)}", flush=True)
        result, _ = run_fold(dataset, train, test, participants, args, evaluation_split="test")
        predicted = np.asarray(result["predictions"])
        probabilities = np.asarray(result["probabilities"])
        if (predicted.shape != y[test].shape or probabilities.shape != y[test].shape
                or not np.isin(predicted, [0, 1]).all()
                or not np.isfinite(probabilities).all()
                or not ((probabilities >= 0) & (probabilities <= 1)).all()):
            raise ValueError(f"Invalid held-out predictions for {outer}")
        fold_rows.append({
            "fold": fold, "outer_held_out_participant": outer,
            "roi": args.roi_label, "hidden": int(choice.hidden), "WD": args.weight_decay,
            "best_mean_segment_pooled_balanced_accuracy": float(choice.mean_segment_pooled_balanced_accuracy),
            "seed": seed, "n_train_segments": len(train), "n_test_segments": len(test),
            "train_accuracy": float(result["train_accuracy"]),
            **fold_metrics(y[test], predicted),
        })
        frame = metadata.iloc[test].copy()
        frame["segment_index"] = test
        frame["fold"] = fold
        frame["outer_held_out_participant"] = outer
        frame["roi"] = args.roi_label
        frame["hidden"] = int(choice.hidden)
        frame["WD"] = args.weight_decay
        frame["seed"] = seed
        frame["best_mean_segment_pooled_balanced_accuracy"] = float(choice.mean_segment_pooled_balanced_accuracy)
        frame["label_binary"] = y[test]
        frame["prediction"] = predicted.astype(int)
        frame["probability_nonresponsive"] = probabilities
        rows.append(frame)
    predictions = pd.concat(rows, ignore_index=True).sort_values("segment_index").reset_index(drop=True)
    if sorted(predictions.segment_index) != list(range(len(y))):
        raise RuntimeError("Outer LOPO must predict every segment exactly once")
    output_dir = ROOT / "outer_LOPO" / roi_label / f"seed={seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(fold_rows)
    labels, predicted = predictions.label_binary, predictions.prediction
    participant_accuracy = predictions.assign(correct=predicted.eq(labels)).groupby("participant").correct.mean()
    summary = pd.DataFrame([{
        "seed": seed,
        "pooled_train_accuracy": float(np.average(folds.train_accuracy, weights=folds.n_train_segments)),
        "pooled_test_accuracy": float(predicted.eq(labels).mean()),
        "participant_averaged_accuracy": float(participant_accuracy.mean()),
        "responsive_recall": float(predicted[labels.eq(0)].eq(0).mean()),
        "nonresponsive_recall": float(predicted[labels.eq(1)].eq(1).mean()),
        "segment_weighted_balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }])
    for name, frame in (("predictions.csv", predictions), ("fold_results.csv", folds), ("summary.csv", summary)):
        output = output_dir / name
        temporary = output.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(output)
    print(f"Saved outer LOPO seed results to {output_dir}", flush=True)
    print(summary.to_string(index=False), flush=True)
    return output_dir




def summarize_outer():
    """Write per-seed, mean/SD, and participant-bootstrap summaries per ROI."""
    import subprocess
    import sys

    selections = ("all_rois",)
    outputs = []
    for roi in selections:
        rows, predictions = [], []
        reference = None
        keys = ["participant", "run", "segment_start_TR", "segment_end_TR_exclusive"]
        for seed in SEEDS:
            folder = ROOT / "outer_LOPO" / roi / f"seed={seed}"
            summary = pd.read_csv(folder / "summary.csv")
            if len(summary) != 1 or summary.iloc[0]["seed"] != seed:
                raise ValueError(f"Expected one summary row for seed {seed}: {folder}")
            if not np.isfinite(summary.select_dtypes(include="number").to_numpy()).all():
                raise ValueError(f"Nonfinite summary metrics: {folder}")
            frame = pd.read_csv(folder / "predictions.csv")
            if (not frame.seed.eq(seed).all() or not frame.roi.eq(roi).all()
                    or frame.duplicated(keys).any()
                    or not frame.participant.eq(frame.outer_held_out_participant).all()
                    or not frame.label_binary.isin([0, 1]).all()
                    or not frame.prediction.isin([0, 1]).all()):
                raise ValueError(f"Invalid held-out predictions: {folder}")
            identity = frame.set_index(keys).label_binary.sort_index()
            if reference is None:
                reference = identity
            elif not identity.equals(reference):
                raise ValueError(f"Segment identities/labels differ across seeds: {folder}")
            rows.append(summary.iloc[0].to_dict())
            predictions.append(frame)
        seed_summary = pd.DataFrame(rows)
        seed_summary["seed"] = seed_summary.seed.astype(int)
        metrics = [column for column in seed_summary if column != "seed"]
        mean_std = seed_summary[metrics].agg(["mean", "std"]).rename_axis("statistic").reset_index()
        participants = sorted(predictions[0].participant.astype(str).unique())
        if set(participants) != set(PARTICIPANTS):
            raise ValueError(f"Expected all canonical participants for {roi}")
        evaluation = ROOT / "evaluation" / roi
        evaluation.mkdir(parents=True, exist_ok=True)
        for name, frame in (("seed_summary.csv", seed_summary), ("seed_mean_std.csv", mean_std)):
            output = evaluation / name
            temporary = output.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(output)
            outputs.append(output)
            print(f"Saved {output}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve().with_name("bootstrap.py")),
                        "--predictions-folder", str(ROOT / "outer_LOPO" / roi)], check=True)
        outputs.append(evaluation / "bootstrap_ci.csv")
    return outputs



if __name__ == "__main__":
    args = parse_args()
    if args.summarize:
        summarize_outer()
    elif args.run_inner:
        run_outer_participant_seed(args.outer_held_out_participant, args.seed)
    elif args.get_best_starting_config or args.eval:
        if args.get_best_starting_config:
            get_best_starting_config()
        if args.eval:
            run_evaluation(args.seed)
    else:
        for selected_seed in (SEEDS if args.seed is None else (args.seed,)):
            generate(selected_seed)
