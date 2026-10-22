from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def _run_inventory(raw_data_dir):
    rows = [{"participant": participant.name, "run": run.name}
            for participant in Path(raw_data_dir).iterdir() if participant.is_dir()
            for run in participant.iterdir() if run.is_dir()]
    inventory = pd.DataFrame(rows, columns=["participant", "run"])
    if inventory.empty:
        return inventory
    inventory["_participant_num"] = pd.to_numeric(inventory["participant"].str.extract(r"(\d+)")[0], errors="coerce")
    inventory["_run_num"] = pd.to_numeric(inventory["run"].str.extract(r"(\d+)")[0], errors="coerce")
    return inventory.sort_values(["_participant_num", "_run_num"]).reset_index(drop=True)


def _scatter_by_run(series, raw_data_dir, output_path, ylabel, title, marker_specs, show_legend=False):
    """Plot one or more interval/segment series at each participant-run position."""
    inventory = _run_inventory(raw_data_dir)
    if inventory.empty:
        print("No participant runs to plot.")
        return None
    participants = inventory["participant"].drop_duplicates().tolist()
    colors = dict(zip(participants, plt.cm.tab20(np.linspace(0, 1, len(participants)))))
    fig, ax = plt.subplots(figsize=(27, 9))
    centers, labels, x = [], [], 0
    for participant_index, (participant, participant_runs) in enumerate(inventory.groupby("participant", sort=False)):
        participant_xs = []
        for run_index, run_row in enumerate(participant_runs.itertuples(index=False)):
            for data, offset, marker, filled, label in marker_specs:
                run_data = data[(data["participant"] == participant) & (data["run"] == run_row.run)]
                run_data = run_data[run_data[series] != 0]
                if run_data.empty:
                    continue
                ax.scatter(np.full(len(run_data), x + offset), run_data[series], marker=marker, s=180,
                           facecolors=colors[participant] if filled else "none", edgecolors=colors[participant],
                           alpha=0.75, linewidths=2, zorder=3, label=label)
            participant_xs.append(x)
            if run_index < len(participant_runs) - 1:
                ax.axvline(x + 0.5, color="lightgray", linewidth=2)
            x += 1
        centers.append(np.mean(participant_xs))
        labels.append(int(participant_runs["_participant_num"].iloc[0]))
        if participant_index < len(participants) - 1:
            ax.axvline(x - 0.5, color="gray", linewidth=3, alpha=0.6)
    ax.set_xticks(centers, labels, fontsize=22)
    ax.tick_params(axis="x", width=2, length=8)
    ax.tick_params(axis="y", which="both", length=0, labelsize=22)
    ax.set_xlabel("Participant", fontsize=32, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=30, labelpad=15)
    ax.set_title(title, fontsize=34, fontweight="bold")
    if show_legend:
        handles = [Line2D([0], [0], marker=marker, linestyle="none", markerfacecolor="0.4" if filled else "none",
                          markeredgecolor="0.4", markeredgewidth=2, markersize=13, label=label)
                   for _, _, marker, filled, label in marker_specs]
        ax.legend(handles=handles, fontsize=18, loc="upper left",
                  ncol=1, frameon=True, facecolor="white", edgecolor="lightgray", framealpha=1,
                  handletextpad=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(2)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def scatter_responsive_intervals(intervals, raw_data_dir, output_dir):
    """Save responsive interval durations by participant and run."""
    return _scatter_by_run("duration_TRs", raw_data_dir, Path(output_dir) / "responsive_intervals.png",
                           "Responsive Interval Duration (TRs)",
                           "Responsive Interval Durations by Participant and Run",
                           ((intervals, 0, "o", True, "Responsive interval"),))


def scatter_non_responsive_intervals(bounded, whole_runs, end_intervals, raw_data_dir, output_dir):
    """Save non-responsive interval durations by participant and run."""
    specs = ((bounded, 0, "o", True, "Bounded interval"),
             (whole_runs, 0, "*", True, "Whole run"),
             (end_intervals, 0, "x", True, "End interval"))
    return _scatter_by_run("duration_TRs", raw_data_dir, Path(output_dir) / "nonresponsive_intervals.png",
                           "Non-responsive Interval Duration (TRs)",
                           "Non-responsive Interval Durations by Participant and Run", specs)


def scatter_segement_counts(segments, raw_data_dir, output_dir):
    """Save motion-free responsive and non-responsive segment counts by run."""
    counts = segments.groupby(["participant", "run", "label"]).size().unstack(fill_value=0).reset_index()
    for label in ("responsive", "non-responsive"):
        if label not in counts:
            counts[label] = 0
    specs = ((counts.rename(columns={"responsive": "count"}), -0.2, "o", True, "R"),
             (counts.rename(columns={"non-responsive": "count"}), 0.2, "o", False, "NR"))
    output_path = _scatter_by_run("count", raw_data_dir, Path(output_dir) / "segment_counts.png",
                                  "Number of motion-free segments",
                                  "Responsive and Non-responsive Segments by Participant and Run", specs,
                                  show_legend=True)
    return output_path


def visualize_leave_one_out(analysis, result_dir):
    """Save fold-level responsive and non-responsive recall for LORO or LOPO."""
    analysis = analysis.strip().lower()
    if analysis not in {"loro", "lopo"}:
        raise ValueError('analysis must be "loro" or "lopo"')

    result_dir = Path(result_dir)
    results_path = result_dir / f"{analysis}_fold_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Fold results not found: {results_path}")
    results = pd.read_csv(results_path)
    required = {"participant", "n_responsive", "n_nonresponsive", "responsive_recall", "nonresponsive_recall"}
    if analysis == "loro":
        required.add("run")
    if missing := required.difference(results.columns):
        raise ValueError(f"Fold results are missing columns: {', '.join(sorted(missing))}")

    results["_participant_num"] = pd.to_numeric(
        results["participant"].astype(str).str.extract(r"(\d+)$")[0], errors="coerce")
    sort_columns = ["_participant_num"]
    if analysis == "loro":
        results["_run_num"] = pd.to_numeric(results["run"].astype(str).str.extract(r"(\d+)$")[0], errors="coerce")
        sort_columns.append("_run_num")
    results = results.sort_values(sort_columns).reset_index(drop=True)

    participants = results["participant"].drop_duplicates().tolist()
    colors = dict(zip(participants, plt.cm.tab20(np.linspace(0, 1, len(participants)))))
    fig, ax = plt.subplots(figsize=(27, 9) if analysis == "loro" else (18, 9))
    ax.axhline(0.5, color="lightgray", linestyle=":", linewidth=2, zorder=0)
    metrics = (("responsive_recall", "n_responsive", -0.2, True, "Responsive recall"),
               ("nonresponsive_recall", "n_nonresponsive", 0.2, False, "Non-responsive recall"))
    counts = pd.concat([results["n_responsive"], results["n_nonresponsive"]], ignore_index=True)
    counts = counts[counts > 0]
    size_cap = max(float(counts.quantile(0.95)), 1.0)

    def marker_area(count):
        return 15 + 345 * np.log1p(min(float(count), size_cap)) / np.log1p(size_cap)

    centers, labels, x = [], [], 0
    for participant_index, (participant, participant_df) in enumerate(results.groupby("participant", sort=False)):
        participant_xs = []
        for row_index, (_, row) in enumerate(participant_df.iterrows()):
            for metric, count_column, offset, filled, _ in metrics:
                if row[count_column] == 0 or pd.isna(row[metric]):
                    continue
                ax.scatter(x + offset, row[metric], s=marker_area(row[count_column]), marker="o",
                           facecolors=colors[participant] if filled else "none", edgecolors=colors[participant],
                           alpha=0.85, linewidths=2, zorder=3)
            participant_xs.append(x)
            if analysis == "loro" and row_index < len(participant_df) - 1:
                ax.axvline(x + 0.5, color="lightgray", linewidth=2, zorder=0)
            x += 1
        centers.append(np.mean(participant_xs))
        participant_number = participant_df["_participant_num"].iloc[0]
        labels.append(int(participant_number) if pd.notna(participant_number) else participant)
        if participant_index < len(participants) - 1:
            ax.axvline(x - 0.5, color="gray", linewidth=3, alpha=0.6, zorder=0)

    ax.set_xticks(centers, labels, fontsize=22)
    ax.tick_params(axis="x", width=2, length=8)
    ax.tick_params(axis="y", which="both", length=0, labelsize=22)
    ax.set(xlabel="Participant", ylabel="Recall", ylim=(-0.05, 1.05), title=f"{analysis.upper()} Fold Performance")
    ax.xaxis.label.set_size(32)
    ax.yaxis.label.set_size(30)
    ax.title.set_fontsize(34)
    ax.title.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(2)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.4" if filled else "none",
                      markeredgecolor="0.4", markeredgewidth=2, markersize=13, label=label)
               for _, _, _, filled, label in metrics]
    ax.legend(handles=handles, fontsize=18, frameon=True, facecolor="white", edgecolor="0.85",
              framealpha=1, loc="lower left")
    fig.tight_layout()
    output_path = result_dir / f"{analysis}_fold_performance.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def pool_segment_history(history, metric):
    """Pool segment counts and moments across folds for each epoch and split."""
    columns = [f"{metric}_count", f"{metric}_sum", f"{metric}_sum_sq"]
    if not set(columns).issubset(history.columns):
        raise ValueError(
            "Training history lacks segment-level statistics. "
            "Rerun training to generate counts, sums, and squared sums; "
            "segment-level loss variance cannot be recovered from fold averages."
        )
    pooled = history.groupby(["split", "epoch"])[columns].sum()
    count, total, total_sq = (pooled[column] for column in columns)
    mean = total / count.replace(0, np.nan)
    variance = ((total_sq - total.pow(2) / count.replace(0, np.nan))
                / (count - 1).where(count > 1)).clip(lower=0)
    return pd.DataFrame({"mean": mean, "std": np.sqrt(variance), "count": count,
                         "ci95": 1.96 * np.sqrt(variance / count.replace(0, np.nan))})


def plot_training_history(analysis, result_dir):
    """Plot pooled segment means and 95% confidence intervals for loss and recall.

    Training pools repeated segment evaluations across folds. Confidence bands
    describe segment observations, without adjusting for dependence across folds.
    """
    analysis, result_dir = analysis.lower(), Path(result_dir)
    history = pd.read_csv(result_dir / f"{analysis}_training_history.csv")
    panels = (("r_loss", "Responsive loss"), ("nr_loss", "Non-responsive loss"),
              ("r_recall", "Responsive recall"), ("nr_recall", "Non-responsive recall"))
    summaries = {metric: pool_segment_history(history, metric) for metric, _ in panels}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    colors = {"train": "tab:blue", "validation": "tab:orange"}
    for ax, (metric, title) in zip(axes.flat, panels):
        for split, color in colors.items():
            if split not in summaries[metric].index.get_level_values("split"):
                continue
            summary = summaries[metric].xs(split, level="split")
            epochs = summary.index.to_numpy()
            average, ci = summary["mean"].to_numpy(), summary["ci95"].to_numpy()
            ax.plot(epochs, average, color=color, linewidth=2, label=split.title())
            ax.fill_between(epochs, average - ci, average + ci, color=color, alpha=0.2)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    fig.suptitle("Segment-weighted means and 95% confidence intervals")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].set_ylabel("Binary cross-entropy loss")
    axes[1, 0].set_ylabel("Recall")
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
        ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    output_path = result_dir / f"{analysis}_training_history.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
