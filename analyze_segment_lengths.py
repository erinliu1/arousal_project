"""Collect the all-ROI LOPO segment-length sweep and bootstrap intervals."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bootstrap import load_predictions, return_config_folder


OUTPUT = Path("plots/segment_lengths/all_rois_segment_sweep.csv")
FIGURE_OUTPUT = OUTPUT.with_suffix(".png")
LENGTHS = range(10, 61, 10)
MODELS = {
    "GRU": {"configuration": "H64, WD=0.01, dropout=0.2", "hidden": 64, "WD": 1e-2,
            "LR": 1e-3, "drop": 0.2, "epochs": 30, "batch": 32, "pool": "final"},
    "MLP": {"configuration": "H16, WD=0.01, dropout=0.2", "hidden": 16, "WD": 1e-2,
            "LR": 1e-3, "drop": 0.2, "epochs": 10, "batch": 32},
    "LR": {"configuration": "C=0.01", "C": 0.01},
    "motion_LR": {"configuration": "C=0.01, FD=0.25", "C": 0.01, "FD": 0.25},
}
COLORS = {"GRU": "tab:green", "MLP": "tab:blue", "LR": "tab:purple", "motion_LR": "tab:orange"}
LABELS = {"GRU": "GRU", "MLP": "MLP", "LR": "LR", "motion_LR": "Motion-only LR"}
METRICS = {
    "segment_pooled_balanced_accuracy": "Segment-Pooled Balanced Accuracy",
    "responsive_recall": "Responsive Recall",
    "nonresponsive_recall": "Nonresponsive Recall",
}


def prediction_metrics(frame):
    """Return segment-pooled class recalls and balanced accuracy."""
    label, prediction = frame["label_binary"].to_numpy(), frame["prediction"].to_numpy()
    recalls = [np.mean(prediction[label == cls] == cls) for cls in (0, 1)]
    if not np.isfinite(recalls).all():
        raise ValueError("Predictions must contain both binary classes.")
    return {"segment_pooled_balanced_accuracy": np.mean(recalls),
            "responsive_recall": recalls[0], "nonresponsive_recall": recalls[1]}


def collect():
    rows = []
    for model, settings in MODELS.items():
        configuration = settings["configuration"]
        options = {key: value for key, value in settings.items() if key != "configuration"}
        for length in LENGTHS:
            folder = return_config_folder(model=model, TR=length, SW=length, **options)
            metadata_folder = return_config_folder(model="GRU", TR=length, SW=length, **{
                key: value for key, value in MODELS["GRU"].items() if key != "configuration"})
            metadata = pd.read_csv(metadata_folder.parents[2] / "all_segment_metadata.csv")
            segment_counts = metadata["label"].value_counts()
            predictions, _, paths = load_predictions(folder, "lopo")
            run_metrics = pd.DataFrame(prediction_metrics(frame) for frame in predictions)
            bootstrap_path = folder / "bootstrap_ci.csv"
            if not bootstrap_path.exists():
                raise FileNotFoundError(f"Missing bootstrap results: {bootstrap_path}")
            bootstrap = pd.read_csv(bootstrap_path)
            if len(bootstrap) != 1 or int(bootstrap.iloc[0]["n_model_seeds"]) != len(paths):
                raise ValueError(f"Bootstrap run count does not match predictions: {bootstrap_path}")
            ci = bootstrap.iloc[0]
            rows.append({
                "model": model, "configuration": configuration, "analysis": "lopo",
                "rois": "all_rois", "segment_length": length, "segment_stride": length,
                "n_model_runs": len(paths),
                "n_responsive_segments": int(segment_counts.get("responsive", 0)),
                "n_nonresponsive_segments": int(segment_counts.get("non-responsive", 0)),
                **run_metrics.mean().to_dict(),
                "segment_pooled_balanced_accuracy_ci_2.5": ci["ci_2.5"],
                "segment_pooled_balanced_accuracy_ci_97.5": ci["ci_97.5"],
                "responsive_recall_ci_2.5": ci["responsive_recall_ci_2.5"],
                "responsive_recall_ci_97.5": ci["responsive_recall_ci_97.5"],
                "nonresponsive_recall_ci_2.5": ci["nonresponsive_recall_ci_2.5"],
                "nonresponsive_recall_ci_97.5": ci["nonresponsive_recall_ci_97.5"],
                "n_bootstrap": int(ci["n_bootstrap"]), "bootstrap_seed": int(ci["bootstrap_seed"]),
            })
    return pd.DataFrame(rows)


def plot_results(results):
    """Plot LOPO means and participant-clustered 95% intervals by segment length."""
    with plt.rc_context({"font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 16,
                         "ytick.labelsize": 16, "axes.linewidth": 1.4,
                         "xtick.major.width": 1.4, "ytick.major.width": 1.4}):
        fig, axes = plt.subplots(4, 1, figsize=(10.5, 11.2), sharex=True)
        fig.subplots_adjust(left=0.12, right=0.68, bottom=0.08, top=0.96, hspace=0.18)
        for row, (metric, title) in enumerate(METRICS.items()):
            ax = axes[row]
            for model in MODELS:
                selected = results.loc[results["model"].eq(model)].sort_values("segment_length")
                lengths, means = selected["segment_length"].to_numpy(), selected[metric].to_numpy()
                lower = selected[f"{metric}_ci_2.5"].to_numpy()
                upper = selected[f"{metric}_ci_97.5"].to_numpy()
                ax.plot(lengths, means, color=COLORS[model], linewidth=4,
                        marker="o", markersize=6, label=LABELS[model])
                ax.fill_between(lengths, lower, upper, color=COLORS[model], alpha=0.1, linewidth=0)
            panel_title = "Non-responsive Recall" if metric == "nonresponsive_recall" else title
            ax.set_title(panel_title, fontsize=18, pad=7)
            ax.set_ylabel("Balanced Accuracy" if row == 0 else "Recall")
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            if row == 0:
                ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1.8, zorder=0)
            ax.grid(which="both", alpha=0.22, linewidth=1)
            ax.spines[["top", "right"]].set_visible(False)

        count_ax = axes[3]
        counts = results.drop_duplicates("segment_length").sort_values("segment_length")
        lengths = counts["segment_length"].to_numpy()
        width = 3.5
        count_ax.bar(lengths - width / 2, counts["n_responsive_segments"], width,
                     color="#9ADF8C", edgecolor="black", linewidth=1, label="Responsive")
        count_ax.bar(lengths + width / 2, counts["n_nonresponsive_segments"], width,
                     color="pink", edgecolor="black", linewidth=1, label="Non-responsive")
        count_ax.set_title("Number of Segments", fontsize=18, pad=7)
        count_ax.set_ylabel("Segments")
        count_ax.grid(axis="y", alpha=0.22, linewidth=1)
        count_ax.spines[["top", "right"]].set_visible(False)
        count_ax.legend(loc="upper right", fontsize=13, frameon=True, edgecolor="0.8")
        axes[-1].set_xticks(list(LENGTHS))
        axes[-1].set_xlabel("Segment length (TRs)")
        for ax in axes[:-1]:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        handles = [Line2D([], [], color=COLORS[model], linewidth=4, marker="o", markersize=6,
                          label=LABELS[model]) for model in MODELS]
        handles.append(Patch(facecolor="0.55", alpha=0.3, linewidth=0,
                             label="95% participant-\nclustered CI"))
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.71, 0.5),
                   fontsize=16, handlelength=1.8, labelspacing=1.0, frameon=True,
                   facecolor="white", edgecolor="0.8", framealpha=1, borderpad=0.8)
    return fig, axes


def main():
    results = collect()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)
    figure, _ = plot_results(results)
    figure.savefig(FIGURE_OUTPUT, dpi=250)
    plt.close(figure)
    print(f"Saved {len(results)} configurations to {OUTPUT}")
    print(f"Saved {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
