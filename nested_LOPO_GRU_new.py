"""Nearest inner LOPO: HD16/32/64 across all seven ROI groups.

One Slurm task runs one outer participant and seed (21 completed CSVs).
Nearest-snapped canonical segments; 30 epochs, LR=.001, dropout=.2,
final pooling and participant-first/class-second training balance.
"""

import argparse
from functools import lru_cache
from pathlib import Path
import fcntl

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from GRU import ArousalDataset, get_segment_dir, load_segments, resolve_rois, run_fold
from nested_LOPO_GRU_old import SEEDS, scientific, training_args, fold_metrics
from all_rois import all_rois

ROOT = Path("plots/GRU/nested_LOPO,BP=nearest")
WEIGHT_DECAYS = (1e-2, 3e-3, 1e-3)
ROI_SELECTIONS = (
    "all_rois", "aan_rois", "cortical_rois", "thalamic_rois",
    "aan_rois+cortical_rois", "aan_rois+thalamic_rois",
    "thalamic_rois+cortical_rois",
)
PARTICIPANTS = tuple(f"AAN_{i:02d}" for i in range(1, 22) if i != 13)
COLUMNS = ["outer_held_out_participant", "roi", "hidden", "WD", "seed",
           "n_validation_segments", "segment_pooled_balanced_accuracy"]


EXPANDED_ROI_SELECTIONS = ROI_SELECTIONS
REMAINING_ROI_SELECTIONS = tuple(roi for roi in ROI_SELECTIONS
                                 if roi not in ("all_rois", "thalamic_rois+cortical_rois"))


def model_args(roi, wd, seed, hidden_size=64):
    values = [roi] if isinstance(roi, str) else roi
    rois, label = resolve_rois([token for value in values for token in value.split("+")])
    if label not in ROI_SELECTIONS:
        raise ValueError(f"Expected one of the seven ROI selections; got {roi!r}")
    if hidden_size not in (16, 32, 64) or (label not in EXPANDED_ROI_SELECTIONS and hidden_size != 64):
        raise ValueError("Hidden size must be 16, 32 or 64")
    if wd not in WEIGHT_DECAYS or seed not in SEEDS:
        raise ValueError("Use a canonical weight decay and model seed")
    return training_args(argparse.Namespace(rois=rois, roi_label=label, bp_TR_snap="nearest"),
                         hidden_size, wd, seed)


@lru_cache(maxsize=1)
def load_data():
    """Load all ROI columns once per process; serialize first-time cache builds."""
    args = model_args("all_rois", WEIGHT_DECAYS[0], SEEDS[0])
    segment_dir = get_segment_dir(args)
    # Different array tasks may start simultaneously with an absent segment cache.
    with (segment_dir / ".nested_inner_cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        X, y, metadata = load_segments(args)
    y = np.asarray(y, dtype=int)
    metadata = metadata.reset_index(drop=True)
    if len(X) != len(y) or len(y) != len(metadata):
        raise ValueError("Segment arrays and metadata are not aligned")
    if not np.array_equal(y, metadata.label.map({"responsive": 0, "non-responsive": 1})):
        raise ValueError("Cached labels disagree with metadata")
    if set(metadata.participant.astype(str)) != set(PARTICIPANTS):
        raise ValueError("Expected exactly the canonical 20 participants")
    return X, y, metadata


def inner_folder(outer, roi, hidden, wd):
    base = ROOT / "inner_LOPO" / outer / roi
    return base / f"hidden={hidden},WD={scientific(wd)}"


def run_inner_lopo(outer_held_out_participant, roi, wd, seed, hidden_size=64):
    """Run 19 inner folds, pool validation predictions, and save exactly one CSV."""
    outer = str(outer_held_out_participant)
    if outer not in PARTICIPANTS:
        raise ValueError(f"Unknown outer participant: {outer}")
    args = model_args(roi, wd, seed, hidden_size)
    output = inner_folder(outer, args.roi_label, hidden_size, wd) / f"seed={seed}.csv"
    if output.is_file() and output.stat().st_size:
        saved = pd.read_csv(output)
        if (list(saved.columns) != COLUMNS or len(saved) != 1
                or saved.iloc[0]["outer_held_out_participant"] != outer
                or saved.iloc[0]["roi"] != args.roi_label
                or saved.iloc[0]["hidden"] != hidden_size or saved.iloc[0]["WD"] != wd
                or saved.iloc[0]["seed"] != seed
                or not 0 <= saved.iloc[0]["segment_pooled_balanced_accuracy"] <= 1
                or saved.iloc[0]["n_validation_segments"] <= 0):
            raise ValueError(f"Invalid completed result: {output}")
        print(f"Skipping completed result: {output}", flush=True)
        return output
    X, y, metadata = load_data()
    participants = metadata.participant.astype(str).to_numpy()
    columns = [all_rois.index(name) for name in args.rois]
    dataset = ArousalDataset(X[:, :, columns], y)
    pooled_labels, pooled_predictions, validation_indices = [], [], []
    inner_participants = [p for p in PARTICIPANTS if p != outer]
    for fold, validation_participant in enumerate(inner_participants, start=1):
        train = np.flatnonzero((participants != outer) & (participants != validation_participant))
        validation = np.flatnonzero(participants == validation_participant)
        if not len(train) or not len(validation):
            raise ValueError("Empty inner training or validation split")
        print(f"Outer={outer} | ROI={args.roi_label} | HD={args.hidden_size} | WD={scientific(wd)} | seed={seed} "
              f"| validation {fold}/19: {validation_participant}", flush=True)
        result, _ = run_fold(dataset, train, validation, participants, args,
                             evaluation_split="validation")
        predictions = np.asarray(result["predictions"], dtype=int)
        if predictions.shape != y[validation].shape or not np.isin(predictions, [0, 1]).all():
            raise ValueError("Invalid validation predictions")
        pooled_labels.append(y[validation])
        pooled_predictions.append(predictions)
        validation_indices.extend(validation.tolist())
    expected = np.flatnonzero(participants != outer)
    if sorted(validation_indices) != expected.tolist():
        raise RuntimeError("Inner validation must cover every non-outer segment exactly once")
    labels, predictions = np.concatenate(pooled_labels), np.concatenate(pooled_predictions)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Pooled validation labels must contain both classes")
    row = dict(outer_held_out_participant=outer, roi=args.roi_label, hidden=hidden_size,
               WD=wd, seed=seed, n_validation_segments=len(labels),
               segment_pooled_balanced_accuracy=balanced_accuracy_score(labels, predictions))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    pd.DataFrame([row], columns=COLUMNS).to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"Saved {output}", flush=True)
    return output


def run_outer_participant_seed(outer_held_out_participant, seed):
    """Run seven ROI selections by three WDs for one outer participant/seed."""
    return [run_inner_lopo(outer_held_out_participant, roi, wd, seed)
            for roi in ROI_SELECTIONS for wd in WEIGHT_DECAYS]


def run_remaining_roi_groups(outer_held_out_participant, seed, hidden_sizes=(16,32)):
    """One array task: five remaining groups x six configs x 19 inner folds."""
    return [run_inner_lopo(outer_held_out_participant, roi, wd, seed, hidden)
            for roi in REMAINING_ROI_SELECTIONS
            for hidden in dict.fromkeys(hidden_sizes)
            for wd in WEIGHT_DECAYS]


def get_best_all_rois_config(roi="all_rois"):
    """Rank nine HD/WD configurations for one ROI group; preserve pruning rankings."""
    roi = model_args(roi, WEIGHT_DECAYS[0], SEEDS[0]).roi_label
    if roi not in EXPANDED_ROI_SELECTIONS:
        raise ValueError("Expected one of the seven canonical ROI groups")
    rankings = {}
    for outer in PARTICIPANTS:
        rows, counts = [], set()
        for hidden in (16, 32, 64):
            for wd in WEIGHT_DECAYS:
                scores = []
                for seed in SEEDS:
                    path = inner_folder(outer, roi, hidden, wd) / f"seed={seed}.csv"
                    frame = pd.read_csv(path)
                    if list(frame.columns) != COLUMNS or len(frame) != 1:
                        raise ValueError(f"Invalid result schema: {path}")
                    row = frame.iloc[0]
                    count = float(row.n_validation_segments)
                    score = float(row.segment_pooled_balanced_accuracy)
                    if (row.outer_held_out_participant != outer or row.roi != roi
                            or row.hidden != hidden or row.WD != wd or row.seed != seed
                            or not np.isfinite(score) or not 0 <= score <= 1
                            or not np.isfinite(count) or count <= 0 or not count.is_integer()):
                        raise ValueError(f"Invalid result identifiers or metrics: {path}")
                    scores.append(score); counts.add(int(count))
                rows.append(dict(outer_held_out_participant=outer, roi=roi, hidden=hidden,
                                 WD=wd, n_seeds=len(SEEDS), n_validation_segments=int(count),
                                 mean_segment_pooled_balanced_accuracy=float(np.mean(scores))))
        if len(counts) != 1:
            raise ValueError(f"Validation counts differ for {outer}")
        ranking = pd.DataFrame(rows).sort_values(
            ["mean_segment_pooled_balanced_accuracy", "hidden", "WD"],
            ascending=[False, True, True]).reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1,10))
        rankings[outer] = ranking
    for outer, ranking in rankings.items():
        path = ROOT / "inner_LOPO" / outer / roi / "best_configurations.csv"
        temporary = path.with_suffix(".csv.tmp")
        ranking.to_csv(temporary, index=False); temporary.replace(path)
        top = ranking.iloc[0]
        print(f"{outer}: {roi} HD={top.hidden}, WD={scientific(top.WD)}", flush=True)
    path = ROOT / f"{roi}_best_configurations.csv"
    temporary = path.with_suffix(".csv.tmp")
    pd.concat([r.iloc[[0]] for r in rankings.values()], ignore_index=True).to_csv(temporary,index=False)
    temporary.replace(path)
    return path


def get_best_starting_config():
    """Rank all 21 configurations per participant using all ten canonical seeds.

    Validate the complete experiment before writing any ranking files. No data
    loading, model fitting, or outer-held-out predictions are used for ranking.
    """
    rankings = {}
    for outer in PARTICIPANTS:
        rows = []
        participant_counts = set()
        for roi in ROI_SELECTIONS:
            for wd in WEIGHT_DECAYS:
                scores, counts = [], []
                folder = inner_folder(outer, roi, 64, wd)
                for seed in SEEDS:
                    path = folder / f"seed={seed}.csv"
                    if not path.is_file() or not path.stat().st_size:
                        raise FileNotFoundError(f"Missing required inner result: {path}")
                    frame = pd.read_csv(path)
                    if list(frame.columns) != COLUMNS or len(frame) != 1:
                        raise ValueError(f"Invalid result schema: {path}")
                    row = frame.iloc[0]
                    if (row.outer_held_out_participant != outer or row.roi != roi
                            or row.hidden != 64 or row.WD != wd or row.seed != seed):
                        raise ValueError(f"Result identifiers do not match the path: {path}")
                    score = float(row.segment_pooled_balanced_accuracy)
                    count = float(row.n_validation_segments)
                    if (not np.isfinite(score) or not 0 <= score <= 1
                            or not np.isfinite(count) or count <= 0 or not count.is_integer()):
                        raise ValueError(f"Invalid score or validation count: {path}")
                    scores.append(score)
                    counts.append(int(count))
                if len(set(counts)) != 1:
                    raise ValueError(f"Validation segment counts differ across seeds: {folder}")
                participant_counts.update(counts)
                rows.append(dict(outer_held_out_participant=outer, roi=roi, hidden=64,
                                 WD=wd, n_seeds=len(SEEDS), n_validation_segments=counts[0],
                                 mean_segment_pooled_balanced_accuracy=float(np.mean(scores))))
        if len(participant_counts) != 1:
            raise ValueError(f"Validation segment counts differ across configurations for {outer}")
        ranking = pd.DataFrame(rows).sort_values(
            ["mean_segment_pooled_balanced_accuracy", "roi", "WD"],
            ascending=[False, True, True], kind="stable").reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
        rankings[outer] = ranking
    outputs = []
    for outer, ranking in rankings.items():
        output = ROOT / "inner_LOPO" / outer / "best_configurations.csv"
        temporary = output.with_suffix(".csv.tmp")
        ranking.to_csv(temporary, index=False)
        temporary.replace(output)
        best = ranking.iloc[0]
        print(f"{outer}: top ROI={best.roi} | WD={scientific(best.WD)} | "
              f"mean pooled BA={best.mean_segment_pooled_balanced_accuracy:.6f} | "
              f"{len(SEEDS)} seeds", flush=True)
        outputs.append(output)
    combined = pd.concat([ranking.iloc[[0]] for ranking in rankings.values()], ignore_index=True)
    combined_path = ROOT / "overall_best_configuration.csv"
    temporary = combined_path.with_suffix(".csv.tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(combined_path)
    print(f"Saved top configuration per participant: {combined_path}", flush=True)
    outputs.append(combined_path)
    return outputs


def run_evaluation(seed, roi):
    """Evaluate one ROI group using each participant's best saved WD ranking.

    Ranking generation is standalone; this reads inner_LOPO/<participant>/
    best_configurations.csv and never refreshes it or selects a different ROI.
    """
    if seed not in SEEDS:
        raise ValueError("Use a canonical evaluation seed")
    roi_label = model_args(roi, WEIGHT_DECAYS[0], seed).roi_label
    choices = []
    for outer in PARTICIPANTS:
        path = ROOT / "inner_LOPO" / outer
        path = path / roi_label / "best_configurations.csv" if roi_label in EXPANDED_ROI_SELECTIONS else path / "best_configurations.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ranking: {path}. Run --get_best_starting_config first.")
        ranking = pd.read_csv(path)
        required = {"outer_held_out_participant", "roi", "hidden", "WD", "n_seeds",
                    "mean_segment_pooled_balanced_accuracy"}
        if not required.issubset(ranking.columns):
            raise ValueError(f"Invalid ranking schema: {path}")
        candidates = ranking.loc[ranking.roi.eq(roi_label)].copy()
        scores = candidates.mean_segment_pooled_balanced_accuracy
        expected = {(h,w) for h in ((16,32,64) if roi_label in EXPANDED_ROI_SELECTIONS else (64,)) for w in WEIGHT_DECAYS}
        if (len(candidates) != len(expected) or candidates.duplicated(["hidden", "WD"]).any()
                or set(zip(candidates.hidden, candidates.WD)) != expected
                or not candidates.outer_held_out_participant.eq(outer).all()
                or not candidates.n_seeds.eq(len(SEEDS)).all()
                or not np.isfinite(scores).all() or not scores.between(0, 1).all()):
            raise ValueError(f"Expected complete ten-seed HD/WD rankings for {roi_label}: {path}")
        choice = candidates.sort_values(["mean_segment_pooled_balanced_accuracy", "hidden", "WD"],
                                       ascending=[False, True, True]).iloc[0]
        choices.append(choice)
    selected = pd.DataFrame(choices).set_index("outer_held_out_participant")
    X, y, metadata = load_data()
    participants = metadata.participant.astype(str).to_numpy()
    if set(participants) != set(PARTICIPANTS):
        raise ValueError("Evaluation data must contain all canonical outer participants")
    rows, fold_rows = [], []
    for fold, outer in enumerate(PARTICIPANTS, start=1):
        choice = selected.loc[outer]
        args = model_args(choice.roi, float(choice.WD), seed, int(choice.hidden))
        columns = [all_rois.index(roi) for roi in args.rois]
        dataset = ArousalDataset(X[:, :, columns], y)
        train = np.flatnonzero(participants != outer)
        test = np.flatnonzero(participants == outer)
        print(f"Outer seed={seed} | fold {fold}/{len(PARTICIPANTS)} | held out={outer} | "
              f"ROI={args.roi_label} | HD={args.hidden_size} | WD={scientific(args.weight_decay)}", flush=True)
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
            "roi": args.roi_label, "hidden": args.hidden_size, "WD": args.weight_decay,
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
        frame["hidden"] = args.hidden_size
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



def summarize_outer(rois=None):
    """Write per-seed, mean/SD, and participant-bootstrap summaries per ROI."""
    from bootstrap import main as run_bootstrap

    selections = (ROI_SELECTIONS if rois is None else
                  [model_args(rois, WEIGHT_DECAYS[0], SEEDS[0]).roi_label])
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
        run_bootstrap([str(ROOT / "outer_LOPO" / roi)])
        outputs.append(evaluation / "bootstrap_ci.csv")
    return outputs


def visualize_results():
    """Plot seed means and participant-bootstrap intervals for completed ROI selections."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = ROOT
    preferred_order = (
        "all_rois",
        "aan_rois",
        "cortical_rois",
        "thalamic_rois",
        "aan_rois+cortical_rois",
        "aan_rois+thalamic_rois",
        "thalamic_rois+cortical_rois",
    )
    display_names = {
        "all_rois": "All ROIs",
        "aan_rois": "AAN",
        "cortical_rois": "Cortical",
        "thalamic_rois": "Thalamic",
        "aan_rois+cortical_rois": "AAN + Cortical",
        "aan_rois+thalamic_rois": "AAN + Thalamic",
        "thalamic_rois+cortical_rois": "Cortical + Thalamic",
    }
    metric_specs = (
        ("segment_weighted_balanced_accuracy", "Balanced Accuracy", "ci_2.5", "ci_97.5"),
        ("responsive_recall", "Responsive Recall",
         "responsive_recall_ci_2.5", "responsive_recall_ci_97.5"),
        ("nonresponsive_recall", "Non-Responsive Recall",
         "nonresponsive_recall_ci_2.5", "nonresponsive_recall_ci_97.5"),
    )
    colors = ("tab:blue", "tab:green", "#df65a0")

    discovered = []
    for roi_folder in (root / "evaluation").iterdir() if (root / "evaluation").exists() else ():
        if not roi_folder.is_dir():
            continue
        evaluation = roi_folder
        mean_std_path = evaluation / "seed_mean_std.csv"
        bootstrap_path = evaluation / "bootstrap_ci.csv"
        if not mean_std_path.exists() or not bootstrap_path.exists():
            continue
        mean_std = pd.read_csv(mean_std_path)
        bootstrap = pd.read_csv(bootstrap_path)
        if "statistic" not in mean_std or "mean" not in set(mean_std["statistic"]):
            raise ValueError(f"Missing mean row in {mean_std_path}")
        if len(bootstrap) != 1:
            raise ValueError(f"Expected one bootstrap row in {bootstrap_path}, found {len(bootstrap)}")
        discovered.append((
            roi_folder.name,
            mean_std.set_index("statistic").loc["mean"],
            bootstrap.iloc[0],
        ))
    if not discovered:
        raise FileNotFoundError(
            f"No completed seed_mean_std.csv/bootstrap_ci.csv pairs found under {root}")

    order = {name: index for index, name in enumerate(preferred_order)}
    discovered.sort(key=lambda item: (
        -float(item[1]["segment_weighted_balanced_accuracy"]),
        order.get(item[0], len(order)),
        item[0],
    ))
    positions = np.arange(len(discovered))
    labels = [display_names.get(name, name.replace("_rois", "").replace("+", " + ").title())
              for name, _, _ in discovered]

    with plt.rc_context({
            "font.size": 26, "axes.labelsize": 30, "axes.titlesize": 32,
            "axes.linewidth": 3, "xtick.labelsize": 26, "ytick.labelsize": 28}):
        fig, axes = plt.subplots(1, 3, figsize=(26, 9), sharex=True, sharey=True,
                                 layout="constrained")
        for ax, (metric, title, lower_column, upper_column), color in zip(
                axes, metric_specs, colors):
            points = np.array([float(mean[metric]) for _, mean, _ in discovered])
            lower = np.array([float(ci[lower_column]) for _, _, ci in discovered])
            upper = np.array([float(ci[upper_column]) for _, _, ci in discovered])
            ax.hlines(positions, lower, upper, color=color, linewidth=7)
            ax.plot(lower, positions, "|", color=color, markersize=24, markeredgewidth=5)
            ax.plot(upper, positions, "|", color=color, markersize=24, markeredgewidth=5)
            ax.plot(points, positions, "o", color=color, markersize=19, zorder=3)
            if metric == "segment_weighted_balanced_accuracy":
                ax.axvline(0.5, color="0.4", linestyle=":", linewidth=4.5, zorder=0)
            ax.set_title(title, pad=20)
            ax.set_xlabel(title)
            ax.set_yticks(positions, labels)
            ax.set_ylim(len(discovered) - 0.5, -0.5)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 6))
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.grid(axis="both", alpha=0.25, linewidth=2.5)
            ax.tick_params(axis="both", width=3, length=10)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Nested LOPO GRU Performance by ROI Groups", fontsize=32)

    root.mkdir(parents=True, exist_ok=True)
    output = root / "nested_lopo_roi_comparison.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualize", action="store_true", help="Plot current ROI-group seed means and participant-bootstrap CIs")
    parser.add_argument("--run_inner", action="store_true",
                        help="Enable inner training (disabled when omitted)")
    parser.add_argument("--summarize", action="store_true",
                        help="Summarize all ROI groups (or --rois) across ten seeds, including bootstrap CIs")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate --rois and --seed using saved participant-specific HD/WD rankings")
    parser.add_argument("--get_best_starting_config", action="store_true",
                        help="Rank all completed inner configurations for every participant; no training")
    parser.add_argument("--outer-held-out-participant", choices=PARTICIPANTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--rois", nargs="+", help="ROI groups, e.g. thalamic_rois cortical_rois; required for --eval")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, choices=(16,32,64), default=None,
                        help="Inner training HD list; defaults to 64. HD16/32/64 supported for all seven ROI groups")
    parser.add_argument("--remaining-roi-groups", action="store_true",
                        help="Run HD16/32 x all three WDs for the five ROI groups excluding all_rois and cortical+thalamic")
    parser.add_argument("--weight-decay", type=float, choices=WEIGHT_DECAYS)
    args = parser.parse_args(argv)
    if sum((args.run_inner, args.eval, args.get_best_starting_config, args.summarize, args.visualize)) > 1:
        parser.error("Choose only one of --run_inner, --eval, --summarize, or --get_best_starting_config")
    if args.remaining_roi_groups and (not args.run_inner or args.rois is not None or args.weight_decay is not None):
        parser.error("--remaining-roi-groups requires --run_inner and cannot use --rois or --weight-decay")
    if args.visualize:
        if any(value is not None for value in (args.seed,args.rois,args.outer_held_out_participant,args.hidden_sizes,args.weight_decay)):
            parser.error("--visualize runs standalone")
        return visualize_results()
    if args.hidden_sizes is not None and not args.run_inner:
        parser.error("--hidden-sizes is only for --run_inner")
    if args.summarize:
        if any(value is not None for value in (args.seed, args.outer_held_out_participant, args.weight_decay)):
            parser.error("--summarize accepts only the optional --rois filter")
        return summarize_outer(args.rois)
    if args.eval:
        if args.seed is None or args.rois is None:
            parser.error("--eval requires --rois and --seed")
        if args.outer_held_out_participant is not None or args.weight_decay is not None:
            parser.error("--eval accepts --rois and --seed; HD/WD are selected from saved rankings")
        try:
            roi_label = model_args(args.rois, WEIGHT_DECAYS[0], args.seed).roi_label
        except ValueError as error:
            parser.error(str(error))
        return run_evaluation(args.seed, roi_label)
    if args.get_best_starting_config:
        if args.run_inner or any(value is not None for value in (
                args.outer_held_out_participant, args.seed, args.weight_decay)):
            parser.error("--get_best_starting_config accepts only the optional --rois filter")
        if args.rois is not None:
            roi = model_args(args.rois, WEIGHT_DECAYS[0], SEEDS[0]).roi_label
            if roi not in EXPANDED_ROI_SELECTIONS:
                parser.error("Filtered ranking requires a canonical ROI group")
            return get_best_all_rois_config(roi)
        return get_best_starting_config()
    if not args.run_inner:
        print("Inner training disabled. Use --run_inner to train, or --eval --seed SEED to evaluate, or --get_best_starting_config to rank.")
        return
    if args.outer_held_out_participant is None or args.seed is None:
        parser.error("--run_inner requires --outer-held-out-participant and --seed")
    if args.remaining_roi_groups:
        return run_remaining_roi_groups(args.outer_held_out_participant, args.seed, args.hidden_sizes or (16,32))
    if args.rois is None:
        if args.weight_decay is not None or args.hidden_sizes is not None:
            parser.error("--weight-decay/--hidden-sizes require --rois")
        return run_outer_participant_seed(args.outer_held_out_participant, args.seed)
    roi = model_args(args.rois, WEIGHT_DECAYS[0], args.seed).roi_label
    hidden_sizes = args.hidden_sizes or [64]
    if roi not in EXPANDED_ROI_SELECTIONS and hidden_sizes != [64]:
        parser.error("HD16/32/64 supported for all seven ROI groups")
    return [run_inner_lopo(args.outer_held_out_participant, roi, wd, args.seed, hidden)
            for hidden in dict.fromkeys(hidden_sizes)
            for wd in (WEIGHT_DECAYS if args.weight_decay is None else [args.weight_decay])]



if __name__ == "__main__":
    main()
