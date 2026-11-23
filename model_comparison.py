"""Compare saved model predictions and visualize nested-LOPO performance.

Default cohort: BP=nearest. No model training or hyperparameter selection.
    python model_comparison.py --visualize
    python model_comparison.py --visualize --visible-rows 2
    python model_comparison.py --paired-contrast
    python model_comparison.py --visualize-paired-contrast

Figures go to plots/model_comparisons/BP=<nearest|preceding>/. Paired bootstrap CSVs
remain beside their GRU nested-LOPO results. --bp_TR_snap preceding retains
comparison support for the older saved results.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

SEEDS = (42, 123, 456, 789, 2026, 12, 314, 258, 1024, 2048)


def visualize_nested_lopo(bp_TR_snap="nearest", visible_rows=None):
    output_dir = Path("plots/model_comparisons") / f"BP={bp_TR_snap}"

    models = ("GRU", "GRU_time_shuffle", "LR", "motion_classifier")
    labels = ("GRU", "GRU (time\nshuffled)", "LR (temporal\nsummaries)", "LR (motion\nonly)")
    metrics = (
        ("segment_weighted_balanced_accuracy", "Balanced Accuracy", "ci_2.5", "ci_97.5"),
        ("responsive_recall", "Responsive Recall", "responsive_recall_ci_2.5", "responsive_recall_ci_97.5"),
        ("nonresponsive_recall", "Non-Responsive Recall", "nonresponsive_recall_ci_2.5", "nonresponsive_recall_ci_97.5"),
    )
    estimates, intervals = [], []
    for model in models:
        if bp_TR_snap == "nearest":
            label = "motion" if model == "motion_classifier" else "all_rois"
            folder = Path("plots") / model / "nested_LOPO,BP=nearest" / "evaluation" / label
            if model in ("GRU", "GRU_time_shuffle"):
                estimate = pd.read_csv(folder / "seed_mean_std.csv").set_index("statistic").loc["mean"]
            else:
                estimate = pd.read_csv(folder / "model_summary.csv").iloc[0]
        else:
            folder = (Path("plots/GRU_time_shuffle") if model == "GRU_time_shuffle"
                      else Path("plots") / model / "nested_LOPO") / "evaluation"
            estimate = pd.read_csv(folder / "summary.csv").set_index("statistic").loc["mean"]
        estimates.append(estimate)
        intervals.append(pd.read_csv(folder / "bootstrap_ci.csv").iloc[0])

    if visible_rows is not None and not 1 <= visible_rows <= len(models):
        raise ValueError("Invalid number of visible metric rows")
    n_visible = len(models) if visible_rows is None else visible_rows
    fig, axes = plt.subplots(1, 3, figsize=(28, 8), sharex=True,
                             sharey=True, layout="constrained")
    positions = np.arange(len(models))
    colors = ("tab:blue", "tab:green", "#df65a0")
    for ax, (metric, title, lower_column, upper_column), color in zip(axes, metrics, colors):
        points = np.array([row[metric] for row in estimates])
        lower = np.array([row[lower_column] for row in intervals])
        upper = np.array([row[upper_column] for row in intervals])
        # Draw intervals independently so percentile CIs need not contain the estimate.
        ax.hlines(positions[:n_visible], lower[:n_visible], upper[:n_visible], color=color, linewidth=7)
        ax.plot(lower[:n_visible], positions[:n_visible], "|", color=color, markersize=26, markeredgewidth=5)
        ax.plot(upper[:n_visible], positions[:n_visible], "|", color=color, markersize=26, markeredgewidth=5)
        ax.plot(points[:n_visible], positions[:n_visible], "o", color=color, markersize=21)
        if metric == "segment_weighted_balanced_accuracy":
            ax.axvline(0.5, color="0.4", linestyle=":", linewidth=5, zorder=0)
        ax.set_yticks(positions, labels)
        for tick_label in ax.get_yticklabels()[n_visible:]:
            tick_label.set_alpha(0)
        ax.tick_params(axis="y", labelleft=(ax is axes[0]))
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_title(title, fontsize=40, pad=30)
        ax.tick_params(axis="both", labelsize=30, width=3.5, length=13)
        ax.grid(axis="both", alpha=0.3, linewidth=3)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(3.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / ("nested_lopo_metrics.png" if visible_rows is None else
                           f"nested_lopo_metrics_top_{visible_rows}_rows.png")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved nested-LOPO visualization to {output}.")

def save_nearest_balanced_accuracy_bootstraps():
    """Paired participant bootstrap on updated nearest all-ROI outer predictions."""
    from bootstrap import load_predictions, load_or_create_samples, bootstrap_metrics
    models = ("GRU", "GRU_time_shuffle", "LR", "motion_classifier")
    scores, estimates, reference = {}, {}, None
    keys = ["participant", "run", "segment_start_TR", "segment_end_TR_exclusive"]
    expected_participants = [f"AAN_{i:02d}" for i in range(1,22) if i != 13]
    samples, _ = load_or_create_samples(expected_participants, 10000, 0)
    for model in models:
        label = "motion" if model == "motion_classifier" else "all_rois"
        folder = Path("plots") / model / "nested_LOPO,BP=nearest" / "outer_LOPO" / label
        predictions, participants, paths = load_predictions(folder, "lopo")
        if participants != expected_participants:
            raise ValueError(f"Participant mismatch: {folder}")
        for frame, path in zip(predictions, paths):
            if (frame.duplicated(keys).any()
                    or not frame.participant.eq(frame.outer_held_out_participant).all()
                    or not frame[["label_binary", "prediction"]].isin([0,1]).all().all()):
                raise ValueError(f"Invalid outer predictions: {path}")
            identity = frame.set_index(keys).label_binary.sort_index()
            if reference is None: reference = identity
            elif not identity.equals(reference):
                raise ValueError(f"Segment identities or labels differ: {path}")
        scores[model] = bootstrap_metrics(predictions, participants, samples)["spba"]
        estimates[model] = float(np.mean([balanced_accuracy_score(f.label_binary, f.prediction)
                                         for f in predictions]))
    rows = []
    for model in models[1:]:
        differences = scores["GRU"] - scores[model]
        lower, upper = np.percentile(differences, [2.5,97.5])
        rows.append(dict(comparison_model=model, point_estimate=estimates["GRU"]-estimates[model],
                         ci_2_5=lower, ci_97_5=upper))
    results = pd.DataFrame(rows).rename(columns={"ci_2_5":"ci_2.5", "ci_97_5":"ci_97.5"})
    output = Path("plots/GRU/nested_LOPO,BP=nearest/bootstrap_GRU_paired_BA_difference.csv")
    results.to_csv(output,index=False)
    print(results.to_string(index=False))
    print(f"Saved {output}")
    return output


def save_balanced_accuracy_bootstraps(bp_TR_snap="nearest"):
    if bp_TR_snap == "nearest":
        return save_nearest_balanced_accuracy_bootstraps()
    output_dir = Path("plots") / "GRU" / "nested_LOPO"
    sample_path = Path("plots/GRU/bootstrap_samples/participant_samples_B=10000,seed=0.csv")
    samples = pd.read_csv(sample_path, dtype=str).to_numpy()
    if samples.shape != (10000, 20) or pd.isna(samples).any():
        raise ValueError(f"Expected 10,000 bootstrap samples of 20 participants in {sample_path}.")
    participants = sorted(set(samples.ravel()))
    participant_index = {participant: i for i, participant in enumerate(participants)}
    counts = np.zeros((len(samples), len(participants)), dtype=int)
    for row, sample in enumerate(samples):
        counts[row] = np.bincount([participant_index[p] for p in sample], minlength=len(participants))

    scores = {}
    point_estimates = {}
    models = ("GRU", "GRU_time_shuffle", "MLP", "LR", "motion_classifier")
    for model in models:
        folder = (Path("plots") / model / "nested_LOPO" / "evaluation")
        if model == "GRU_time_shuffle" and not folder.exists():
            folder = Path("plots/GRU_time_shuffle/evaluation")
        summary = pd.read_csv(folder / "summary.csv").set_index("statistic")
        point_estimates[model] = float(summary.loc["mean", "segment_weighted_balanced_accuracy"])
        seeds = SEEDS if model in {"GRU", "GRU_time_shuffle", "MLP"} else (None,)
        seed_scores = []
        for seed in seeds:
            path = folder / (f"seed={seed}.csv" if seed is not None else "result.csv")
            if seed is None and not path.exists():
                path = folder / "results.csv"
            frame = pd.read_csv(path, dtype={"participant": str})
            if set(frame["participant"]) != set(participants):
                raise ValueError(f"Participant set does not match the bootstrap samples: {path}")
            if not frame[["label_binary", "prediction"]].isin([0, 1]).all().all():
                raise ValueError(f"Expected binary labels and predictions in {path}.")
            # Sum class totals and correct predictions before applying resampling multiplicities.
            totals = frame.assign(
                n0=frame["label_binary"].eq(0), n1=frame["label_binary"].eq(1),
                correct0=frame["label_binary"].eq(0) & frame["prediction"].eq(0),
                correct1=frame["label_binary"].eq(1) & frame["prediction"].eq(1),
            ).groupby("participant")[["n0", "n1", "correct0", "correct1"]].sum().loc[participants].to_numpy()
            denominators = counts @ totals[:, :2]
            correct = counts @ totals[:, 2:]
            if (denominators == 0).any():
                raise ValueError(f"A bootstrap sample lacks one of the two classes: {path}")
            seed_scores.append((correct / denominators).mean(axis=1))
        scores[model] = np.asarray(seed_scores)

    rows = []
    for model in models[1:]:
        # Seed rows share the explicit SEEDS order; deterministic models broadcast.
        differences = (scores["GRU"] - scores[model]).mean(axis=0)
        lower, upper = np.percentile(differences, [2.5, 97.5])
        rows.append({"comparison_model": model,
                     "point_estimate": point_estimates["GRU"] - point_estimates[model],
                     "ci_2.5": lower, "ci_97.5": upper})
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "bootstrap_GRU_paired_BA_difference.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved paired balanced-accuracy bootstrap intervals to {output}.")

def visualize_balanced_accuracy_paired_contrast(bp_TR_snap="nearest"):
    output_dir = Path("plots/model_comparisons") / f"BP={bp_TR_snap}"
    input_dir = Path("plots/GRU") / ("nested_LOPO,BP=nearest" if bp_TR_snap == "nearest" else "nested_LOPO")
    results = pd.read_csv(input_dir / "bootstrap_GRU_paired_BA_difference.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if bp_TR_snap == "nearest":
        models = ("GRU_time_shuffle", "LR", "motion_classifier")
        labels = ("GRU (time shuffled)", "LR (temporal summaries)", "LR (motion-only)")
    else:
        models = ("GRU_time_shuffle", "LR", "motion_classifier")
        labels = ("GRU (time shuffled)", "LR (temporal summaries)", "LR (motion-only)")
    results = results.set_index("comparison_model").loc[list(models)]
    positions = np.arange(len(models))
    color = "#6baed6"
    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    ax.hlines(positions, results["ci_2.5"], results["ci_97.5"], color=color, linewidth=5)
    ax.plot(results["ci_2.5"], positions, "|", color=color, markersize=20, markeredgewidth=3.5)
    ax.plot(results["ci_97.5"], positions, "|", color=color, markersize=20, markeredgewidth=3.5)
    ax.plot(results["point_estimate"], positions, "o", color=color, markersize=16)
    ax.axvline(0, color="0.4", linewidth=3.5, zorder=0)
    ax.set_yticks(positions, labels)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.margins(x=0.15)
    ax.set_title("Balanced Accuracy Difference\nGRU − Other Models", fontsize=32, pad=26)
    ax.set_xlabel("Balanced Accuracy Difference" + (" (pp)" if bp_TR_snap == "nearest" else ""), fontsize=24, labelpad=18)
    if bp_TR_snap == "nearest":
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:g}"))
    ax.tick_params(axis="both", labelsize=24, width=2.5, length=10)
    ax.grid(axis="both", alpha=0.25, linewidth=2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    output = output_dir / "balanced_accuracy_paired_contrast.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paired balanced-accuracy visualization to {output}.")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--visualize", action="store_true", help="Plot BA and class recalls with bootstrap CIs")
    mode.add_argument("--paired-contrast", action="store_true", help="Compute paired participant-bootstrap BA differences")
    mode.add_argument("--visualize-paired-contrast", action="store_true", help="Plot saved paired BA differences")
    parser.add_argument("--bp_TR_snap", choices=("nearest", "preceding"), default="nearest")
    parser.add_argument("--visible-rows", type=int, choices=(1,2,3,4), help="Progressively reveal nearest-cohort metric rows")
    args = parser.parse_args(argv)
    if args.visible_rows is not None and not args.visualize:
        parser.error("--visible-rows requires --visualize")
    if args.visualize:
        return visualize_nested_lopo(args.bp_TR_snap, args.visible_rows)
    if args.paired_contrast:
        return save_balanced_accuracy_bootstraps(args.bp_TR_snap)
    return visualize_balanced_accuracy_paired_contrast(args.bp_TR_snap)


if __name__ == "__main__":
    main()
