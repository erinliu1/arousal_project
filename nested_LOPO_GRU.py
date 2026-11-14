"""Run ROI-specific nested LOPO model selection and evaluation for the GRU."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from GRU import ArousalDataset, load_segments, resolve_rois, run_fold


SEEDS = (42, 123, 456, 789, 2026, 12, 314, 258, 1024, 2048)
HIDDEN_SIZES = (16, 32, 64)
WEIGHT_DECAYS = (1e-2, 3e-3, 1e-3)
ROOT = Path("plots/GRU/nested_LOPO_2")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("inner_lopo", "eval"), default="inner_lopo")
    parser.add_argument("--summarize", action="store_true",
                        help="Summarize completed outer-LOPO seeds for the selected ROI grouping.")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Bootstrap the ten completed outer-LOPO seed predictions.")
    parser.add_argument("--n-bootstrap", type=int, default=10_000,
                        help="Number of participant bootstrap replicates.")
    parser.add_argument("--bootstrap-seed", type=int, default=0,
                        help="Random seed used to sample bootstrap participants.")
    parser.add_argument(
        "--rois", type=str, nargs="+", default=["all_rois"],
        help=("ROI groups (all_rois, aan_rois, cortical_rois, thalamic_rois) "
              "and/or individual ROI names, as in GRU.py."),
    )
    parser.add_argument("--held-out-participant",
                        help="Optional outer participant filter for inner_lopo.")
    parser.add_argument("--hidden-size", type=int, choices=HIDDEN_SIZES,
                        help="Optional hidden-size filter for inner_lopo.")
    parser.add_argument("--weight-decay", type=float, choices=WEIGHT_DECAYS,
                        help="Optional weight-decay filter for inner_lopo.")
    parser.add_argument("--seed", type=int, choices=SEEDS,
                        help="Optional seed filter for inner_lopo or eval.")
    args = parser.parse_args()
    try:
        args.rois, args.roi_label = resolve_rois(args.rois)
    except ValueError as error:
        parser.error(str(error))
    if args.summarize and args.bootstrap:
        parser.error("Choose either --summarize or --bootstrap, not both.")
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be at least 1.")
    if (args.summarize or args.bootstrap) and any(value is not None for value in (
            args.held_out_participant, args.hidden_size, args.weight_decay, args.seed)):
        parser.error("--summarize and --bootstrap do not accept training filters.")
    if args.run == "eval" and any(value is not None for value in (
            args.held_out_participant, args.hidden_size, args.weight_decay)):
        parser.error("--run eval accepts only the optional --seed filter.")
    return args


def scientific(value):
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def configuration_name(hidden_size, weight_decay):
    return f"hidden={hidden_size},WD={scientific(weight_decay)}"


def roi_dir(args):
    return ROOT / args.roi_label


def training_args(args, hidden_size, weight_decay, seed):
    """Build the GRU arguments shared by inner and outer folds."""
    return argparse.Namespace(
        seed=seed,
        hidden_size=hidden_size,
        weight_decay=weight_decay,
        epochs=30,
        batch_size=32,
        learning_rate=1e-3,
        dropout=0.2,
        pooling="final",
        balance="participant_then_class",
        rois=args.rois,
        roi_label=args.roi_label,
        segment_length=40,
        segment_stride=40,
        responsive_window_size=30,
        responsive_window_stride=1,
        responsive_min_bp_trs=8,
        responsive_max_bp_gap=4,
        trim_responsive_intervals=False,
        nonresponsive_min_gap=20,
        rebuild_segment_arrays=False,
        inner_cv_folds=0,
    )


def load_data(args):
    """Load the canonical 40-TR segment cache and select the requested ROIs."""
    data_args = training_args(args, HIDDEN_SIZES[0], WEIGHT_DECAYS[0], SEEDS[0])
    X, y, metadata = load_segments(data_args)
    return X, np.asarray(y, dtype=int), metadata.reset_index(drop=True)


def inner_output(args, held_out, hidden_size, weight_decay, seed):
    return (roi_dir(args) / "inner_LOPO" / str(held_out)
            / configuration_name(hidden_size, weight_decay) / f"seed={seed}.csv")


def run_inner_configuration(args, X, y, metadata, held_out, hidden_size, weight_decay, seed):
    output = inner_output(args, held_out, hidden_size, weight_decay, seed)
    if output.exists() and output.stat().st_size:
        print(f"Skipping completed inner LOPO: {output}", flush=True)
        return

    participants = metadata["participant"].astype(str).to_numpy()
    unique_participants = np.unique(participants)
    held_out = str(held_out)
    if held_out not in unique_participants:
        raise ValueError(f"Unknown held-out participant {held_out!r}.")
    inner_participants = unique_participants[unique_participants != held_out]
    model_args = training_args(args, hidden_size, weight_decay, seed)
    dataset = ArousalDataset(X, y)
    labels, predictions = [], []

    for fold, validation_participant in enumerate(inner_participants, start=1):
        train = np.flatnonzero(
            (participants != held_out) & (participants != validation_participant))
        validation = np.flatnonzero(participants == validation_participant)
        print(
            f"Held out={held_out} | hidden={hidden_size} | WD={scientific(weight_decay)} "
            f"| seed={seed} | inner fold {fold}/{len(inner_participants)}: "
            f"{validation_participant}",
            flush=True,
        )
        result, _ = run_fold(
            dataset, train, validation, participants, model_args,
            evaluation_split="validation",
        )
        labels.append(y[validation])
        predictions.append(result["predictions"].astype(int))

    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    score = balanced_accuracy_score(labels, predictions)
    row = {
        "held_out_participant": held_out,
        "roi": args.roi_label,
        "hidden": hidden_size,
        "WD": weight_decay,
        "seed": seed,
        "inner_folds": len(inner_participants),
        "n_validation_segments": len(labels),
        "segment_pooled_balanced_accuracy": score,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output, index=False)
    print(f"Saved {output}", flush=True)


def run_inner_lopo(args):
    X, y, metadata = load_data(args)
    participants = sorted(metadata["participant"].astype(str).unique())
    held_outs = [str(args.held_out_participant)] if args.held_out_participant else participants
    hidden_sizes = [args.hidden_size] if args.hidden_size is not None else HIDDEN_SIZES
    weight_decays = [args.weight_decay] if args.weight_decay is not None else WEIGHT_DECAYS
    seeds = [args.seed] if args.seed is not None else SEEDS
    total = len(held_outs) * len(hidden_sizes) * len(weight_decays) * len(seeds)
    run_number = 0
    for held_out in held_outs:
        for hidden_size in hidden_sizes:
            for weight_decay in weight_decays:
                for seed in seeds:
                    run_number += 1
                    print(f"Inner run {run_number}/{total}", flush=True)
                    run_inner_configuration(
                        args, X, y, metadata, held_out, hidden_size, weight_decay, seed)


def select_best_configurations(args, expected_participants=None):
    """Select each outer participant's configuration by its mean over ten seeds."""
    inner_dir = roi_dir(args) / "inner_LOPO"
    paths = sorted(inner_dir.glob("*/hidden=*,WD=*/seed=*.csv"))
    if not paths:
        raise FileNotFoundError(f"No inner-LOPO results found under {inner_dir}.")
    results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    results["held_out_participant"] = results["held_out_participant"].astype(str)
    keys = ["held_out_participant", "hidden", "WD", "seed"]
    if results.duplicated(keys).any():
        raise ValueError("Duplicate participant/configuration/seed inner results found.")

    expected_seeds = set(SEEDS)
    expected_configurations = {
        (hidden_size, weight_decay)
        for hidden_size in HIDDEN_SIZES for weight_decay in WEIGHT_DECAYS
    }
    rows = []
    for held_out, participant_results in results.groupby("held_out_participant", sort=True):
        completed = set()
        scores = []
        for (hidden_size, weight_decay), group in participant_results.groupby(
                ["hidden", "WD"], sort=True):
            seeds = set(group["seed"].astype(int))
            if seeds != expected_seeds:
                missing = sorted(expected_seeds - seeds)
                raise ValueError(
                    f"{held_out}/{configuration_name(int(hidden_size), weight_decay)} "
                    f"is missing seeds: {missing}")
            configuration = (int(hidden_size), float(weight_decay))
            completed.add(configuration)
            scores.append((
                float(group["segment_pooled_balanced_accuracy"].mean()),
                configuration[0], configuration[1],
            ))
        if not all(any(
                hidden == expected_hidden and np.isclose(wd, expected_wd)
                for hidden, wd in completed)
                for expected_hidden, expected_wd in expected_configurations):
            missing = [configuration_name(hidden, wd)
                       for hidden, wd in expected_configurations
                       if not any(h == hidden and np.isclose(w, wd) for h, w in completed)]
            raise ValueError(f"{held_out} is missing configurations: {sorted(missing)}")
        best_score, best_hidden, best_wd = max(scores, key=lambda item: item[0])
        rows.append({
            "held_out_participant": str(held_out),
            "best_hidden": best_hidden,
            "best_WD": best_wd,
            "best_mean_segment_pooled_balanced_accuracy": best_score,
        })

    best = pd.DataFrame(rows).sort_values("held_out_participant").reset_index(drop=True)
    if expected_participants is not None:
        expected = {str(participant) for participant in expected_participants}
        completed = set(best["held_out_participant"])
        if completed != expected:
            raise ValueError(
                "Inner-LOPO participant set mismatch; "
                f"missing={sorted(expected - completed)}, extra={sorted(completed - expected)}")
    output = roi_dir(args) / "best_configurations.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(output, index=False)
    print(f"Saved {output}", flush=True)
    return best


def fold_metrics(labels, predictions):
    responsive = labels == 0
    nonresponsive = labels == 1
    return {
        "test_accuracy": float((predictions == labels).mean()),
        "responsive_recall": float((predictions[responsive] == 0).mean()),
        "nonresponsive_recall": float((predictions[nonresponsive] == 1).mean()),
        "balanced_test_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def run_outer_seed(args, best, X, y, metadata, seed):
    output_dir = roi_dir(args) / "outer_LOPO" / f"seed={seed}"
    required = [output_dir / name for name in
                ("predictions.csv", "fold_results.csv", "summary.csv")]
    if all(path.exists() and path.stat().st_size for path in required):
        print(f"Skipping completed outer LOPO: {output_dir}", flush=True)
        return

    participants = metadata["participant"].astype(str).to_numpy()
    unique_participants = np.unique(participants)
    selected = best.copy()
    selected["held_out_participant"] = selected["held_out_participant"].astype(str)
    selected = selected.set_index("held_out_participant")
    if set(selected.index) != set(unique_participants):
        missing = sorted(set(unique_participants) - set(selected.index))
        extra = sorted(set(selected.index) - set(unique_participants))
        raise ValueError(f"Best-configuration participants mismatch; missing={missing}, extra={extra}")

    dataset = ArousalDataset(X, y)
    fold_rows, prediction_rows = [], []
    for fold, held_out in enumerate(unique_participants, start=1):
        choice = selected.loc[held_out]
        hidden_size = int(choice["best_hidden"])
        weight_decay = float(choice["best_WD"])
        train = np.flatnonzero(participants != held_out)
        test = np.flatnonzero(participants == held_out)
        model_args = training_args(args, hidden_size, weight_decay, seed)
        print(
            f"Outer seed={seed} | fold {fold}/{len(unique_participants)} | held out={held_out} "
            f"| hidden={hidden_size} | WD={scientific(weight_decay)}",
            flush=True,
        )
        result, history = run_fold(dataset, train, test, participants, model_args)
        labels = y[test]
        predictions = result["predictions"].astype(int)
        fold_rows.append({
            "fold": fold,
            "held_out_participant": held_out,
            "best_hidden": hidden_size,
            "best_WD": weight_decay,
            "best_mean_segment_pooled_balanced_accuracy": float(
                choice["best_mean_segment_pooled_balanced_accuracy"]),
            "seed": seed,
            "n_train_segments": len(train),
            "n_test_segments": len(test),
            "train_accuracy": float(result["train_accuracy"]),
            **fold_metrics(labels, predictions),
        })
        for segment_index, label, prediction, probability in zip(
                test, labels, predictions, result["probabilities"]):
            row = metadata.iloc[segment_index].to_dict()
            row.update({
                "segment_index": int(segment_index),
                "fold": fold,
                "held_out_participant": held_out,
                "best_hidden": hidden_size,
                "best_WD": weight_decay,
                "seed": seed,
                "label_binary": int(label),
                "prediction": int(prediction),
                "probability_nonresponsive": float(probability),
            })
            prediction_rows.append(row)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    labels = predictions["label_binary"]
    predicted = predictions["prediction"]
    participant_accuracy = predictions.assign(
        correct=predicted.eq(labels)).groupby("participant")["correct"].mean()
    summary = pd.DataFrame([{
        "seed": seed,
        "pooled_train_accuracy": np.average(
            folds["train_accuracy"], weights=folds["n_train_segments"]),
        "pooled_test_accuracy": float(predicted.eq(labels).mean()),
        "participant_averaged_accuracy": float(participant_accuracy.mean()),
        "responsive_recall": float(predicted[labels.eq(0)].eq(0).mean()),
        "nonresponsive_recall": float(predicted[labels.eq(1)].eq(1).mean()),
        "segment_weighted_balanced_accuracy": float(
            balanced_accuracy_score(labels, predicted)),
    }])
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    folds.to_csv(output_dir / "fold_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    print(f"Saved outer LOPO seed results to {output_dir}", flush=True)


def run_evaluation(args):
    X, y, metadata = load_data(args)
    best = select_best_configurations(args, metadata["participant"].astype(str).unique())
    seeds = [args.seed] if args.seed is not None else SEEDS
    for run_number, seed in enumerate(seeds, start=1):
        print(f"Outer evaluation seed {run_number}/{len(seeds)}: {seed}", flush=True)
        run_outer_seed(args, best, X, y, metadata, seed)


def summarize_outer(args):
    """Write per-seed metrics and their across-seed mean and sample SD."""
    outer_dir = roi_dir(args) / "outer_LOPO"
    rows = []
    missing = []
    for seed in SEEDS:
        path = outer_dir / f"seed={seed}" / "summary.csv"
        if not path.exists():
            missing.append(seed)
            continue
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(f"Expected one summary row in {path}, found {len(frame)}.")
        rows.append(frame.iloc[0].to_dict())
    if missing:
        raise FileNotFoundError(f"Missing completed outer-LOPO seeds: {missing}")

    seed_summary = pd.DataFrame(rows)
    seed_summary["seed"] = seed_summary["seed"].astype(int)
    seed_summary = seed_summary.set_index("seed").loc[list(SEEDS)].reset_index()
    metrics = [column for column in seed_summary.columns if column != "seed"]
    mean_std = seed_summary[metrics].agg(["mean", "std"]).rename_axis(
        "statistic").reset_index()
    evaluation_dir = roi_dir(args) / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    seed_output = evaluation_dir / "seed_summary.csv"
    mean_std_output = evaluation_dir / "seed_mean_std.csv"
    seed_summary.to_csv(seed_output, index=False)
    mean_std.to_csv(mean_std_output, index=False)
    print(f"Saved {seed_output}", flush=True)
    print(f"Saved {mean_std_output}", flush=True)
    return seed_output, mean_std_output


def bootstrap_outer(args):
    """Bootstrap participant-clustered metrics across all ten outer model seeds."""
    from bootstrap import main as run_bootstrap

    run_bootstrap([
        "--predictions-folder", str(roi_dir(args) / "outer_LOPO"),
        "--n-bootstrap", str(args.n_bootstrap),
        "--bootstrap-seed", str(args.bootstrap_seed),
    ])


def main():
    args = parse_args()
    if args.bootstrap:
        bootstrap_outer(args)
    elif args.summarize:
        summarize_outer(args)
    elif args.run == "inner_lopo":
        run_inner_lopo(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
