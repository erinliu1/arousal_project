"""Compare completed trim-off GRU configurations across segment lengths."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path("plots/GRU")
LENGTHS = np.arange(10, 61, 10)
METRICS = {
    "balanced_test_accuracy": "Balanced Test Accuracy",
    "responsive_recall": "Responsive Recall",
    "nonresponsive_recall": "Nonresponsive Recall",
}
CONFIGS = {
    "H=64, WD=1e-3": (64, 1e-3),
    "H=64, WD=1e-2": (64, 1e-2),
    "H=32, WD=1e-3": (32, 1e-3),
    "H=32, WD=1e-2": (32, 1e-2),
}
COLORS = {
    "H=64, WD=1e-3": "tab:blue",
    "H=64, WD=1e-2": "tab:cyan",
    "H=32, WD=1e-3": "tab:green",
    "H=32, WD=1e-2": "tab:orange",
}
SEGMENT_PATTERN = re.compile(
    r"R\(W=30,S=1,N_min=8,G_max=4\):NR\(T_min=20\):"
    r"Segment\(TR=(\d+),SW=(\d+)(,trim)?\)"
)


def load_results(root=ROOT / "segments"):
    """Load completed matching summaries; silently omit unfinished conditions."""
    rows = []
    paths = Path(root).glob("*/*/model=GRU,*/*/SEED=*/*_summary.csv")
    for path in sorted(paths):
        segment_dir, roi, model_dir, analysis_dir, seed_dir, _ = path.relative_to(root).parts
        match = SEGMENT_PATTERN.fullmatch(segment_dir)
        if not match or match.group(3) or roi != "all_rois":
            continue
        length, stride = map(int, match.groups()[:2])
        if length != stride or length not in LENGTHS:
            continue
        settings = dict(item.split("=", 1) for item in model_dir.split(","))
        analysis, *options = analysis_dir.split(",")
        options = dict(item.split("=", 1) for item in options)
        if (settings.get("model") != "GRU" or settings.get("pool", "final") != "final"
                or float(settings.get("drop", -1)) != 0.2 or int(settings.get("epochs", 0)) != 30
                or int(settings.get("batch", 0)) != 32 or float(settings.get("LR", 0)) != 1e-3
                or int(options.get("innerCV", 0)) != 0 or analysis.lower() not in {"lopo", "loro"}):
            continue
        config = (int(settings.get("hidden", 0)), float(settings.get("WD", 0)))
        label = next((name for name, values in CONFIGS.items() if values == config), None)
        if label is None or path.name != f"{analysis.lower()}_summary.csv":
            continue
        try:
            summary = pd.read_csv(path).iloc[0]
        except (OSError, pd.errors.EmptyDataError, IndexError):
            continue
        if any(metric not in summary or pd.isna(summary[metric]) for metric in METRICS):
            continue
        rows.append({"config": label, "analysis": analysis.lower(), "length": length,
                     "seed": int(seed_dir.removeprefix("SEED=")),
                     **{metric: float(summary[metric]) for metric in METRICS}})
    results = pd.DataFrame(rows)
    if results.empty:
        return results
    if results.duplicated(["config", "analysis", "length", "seed"]).any():
        raise ValueError("Conflicting summaries found for a GRU configuration/analysis/length/seed.")
    return results


def visualize_gru(results=None):
    """Plot means and 95% seed-level confidence intervals in a 3×2 panel."""
    results = load_results() if results is None else results
    available = [config for config in CONFIGS if not results.empty and config in set(results["config"])]
    if not available:
        raise ValueError("No completed matching GRU summaries were found.")
    grouped = results.groupby(["config", "analysis", "length", "seed"])[list(METRICS)].mean()
    with plt.rc_context({"font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 16,
                         "ytick.labelsize": 16, "axes.linewidth": 1.4,
                         "xtick.major.width": 1.4, "ytick.major.width": 1.4}):
        fig, axes = plt.subplots(3, 2, figsize=(15.5, 8.8), sharex=True, sharey="row")
        fig.subplots_adjust(left=0.07, right=0.73, bottom=0.09, top=0.92, hspace=0.13, wspace=0.08)
        for row, (metric, title) in enumerate(METRICS.items()):
            for column, analysis in enumerate(("lopo", "loro")):
                ax = axes[row, column]
                for config in available:
                    means = np.full(len(LENGTHS), np.nan)
                    widths = np.full(len(LENGTHS), np.nan)
                    for i, length in enumerate(LENGTHS):
                        key = (config, analysis, length)
                        if key not in grouped.index.droplevel("seed"):
                            continue
                        values = grouped.loc[key, metric].dropna().to_numpy()
                        means[i] = values.mean()
                        if len(values) > 1:
                            widths[i] = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
                    ax.plot(LENGTHS, means, color=COLORS[config], linewidth=3,
                            marker="o", markersize=6, label=config)
                    ax.fill_between(LENGTHS, means - widths, means + widths,
                                    color=COLORS[config], alpha=0.18, linewidth=0)
                    isolated = np.isfinite(widths) & ~np.r_[False, np.isfinite(widths[:-1])] & ~np.r_[np.isfinite(widths[1:]), False]
                    for i in np.flatnonzero(isolated):
                        ax.fill_betweenx([means[i] - widths[i], means[i] + widths[i]],
                                         LENGTHS[i] - 0.6, LENGTHS[i] + 0.6,
                                         color=COLORS[config], alpha=0.18, linewidth=0)
                if column == 0:
                    ax.set_ylabel("Accuracy" if row == 0 else "Recall")
                ax.text(0.025, 0.96, title, transform=ax.transAxes, ha="left", va="top", fontsize=16,
                        bbox=dict(facecolor="white", edgecolor="0.8", linewidth=0.8,
                                  boxstyle="round,pad=0.3"))
                ax.set_ylim(0.5, 1)
                ax.set_yticks(np.arange(0.5, 1.01, 0.1))
                ax.grid(which="both", alpha=0.22, linewidth=1)
                ax.spines[["top", "right"]].set_visible(False)
        for column, analysis in enumerate(("LOPO", "LORO")):
            axes[0, column].set_title(analysis, fontsize=22, pad=10)
            axes[-1, column].set_xticks(LENGTHS)
            axes[-1, column].set_xlabel("Segment length (TRs)")
        for ax in axes[:-1].flat:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        handles = [Line2D([], [], color=COLORS[config], linewidth=3, marker="o",
                          markersize=6, label=config) for config in available]
        handles.append(Patch(facecolor="0.55", alpha=0.3, linewidth=0,
                             label="95% CI across available seeds"))
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.75, 0.5),
                   fontsize=17, handlelength=1.8, labelspacing=1.0, frameon=True,
                   facecolor="white", edgecolor="0.8", framealpha=1, borderpad=0.8)
    return fig, axes


