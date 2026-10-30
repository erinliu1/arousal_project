"""Compare GRU, MLP, ROI-feature LR, and motion-only LR (trim disabled)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import t

CLASSIFIER_DIR = Path("plots/fMRI_classifier_exploration2")
RESULTS_DIR = CLASSIFIER_DIR / "results"
LR_DIR = Path("plots/LR")
MOTION_DIR = Path("plots/motion_classifier")
MLP_DIR = Path("plots/MLP")
MLP_PARAMETERS = {
    "model": "MLP", "hidden_size": 16, "dropout": 0.2, "epochs": 10,
    "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-2,
    "roi_label": "all_rois", "activation": "ReLU", "hidden_layers": 1,
}
LENGTHS = np.arange(10, 61, 10)
METRICS = {
    "balanced_test_accuracy": "Balanced Test Accuracy",
    "responsive_recall": "Responsive Recall",
    "nonresponsive_recall": "Nonresponsive Recall",
}
SEGMENT_PARAMETERS = {
    "responsive_window_size": 30, "responsive_window_stride": 1,
    "responsive_min_bp_trs": 8, "responsive_max_bp_gap": 4,
    "nonresponsive_min_gap": 20,
}
GRU_PARAMETERS = {
    "model": "GRU", "pool": "final", "hidden": 32, "drop": 0.2,
    "epochs": 30, "batch": 32, "lr": 1e-3, "wd": 1e-3,
}
COLORS = {"LR": "tab:purple", "GRU": "tab:green", "Motion-only": "tab:orange", "MLP": "tab:blue"}
STYLES = {"lopo": "-", "loro": ":"}


def select_results(df, parameters):
    """Select matching settings with trim off and length=stride in 10–60 TRs."""
    required = {*parameters, "analysis", "segment_length", "segment_stride", "trim", *METRICS}
    if missing := required.difference(df.columns):
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    mask = pd.Series(True, index=df.index)
    for column, value in parameters.items():
        mask &= df[column].eq(value)
    trim = df["trim"].map({True: True, False: False, "True": True, "False": False,
                           "true": True, "false": False})
    mask &= trim.eq(False)
    mask &= df["segment_length"].eq(df["segment_stride"])
    mask &= df["segment_length"].isin(LENGTHS)
    selected = df.loc[mask].copy()
    selected["analysis"] = selected["analysis"].str.lower()
    return selected.loc[selected["analysis"].isin(STYLES)]


def load_lr_results(root=LR_DIR, c=0.01):
    """Read fixed-C LR outputs; identical deterministic repeats count once."""
    rows = []
    for path in sorted(Path(root).rglob("*_summary.csv")):
        settings_path = path.parent / "settings.json"
        if not settings_path.exists():
            continue
        settings = json.loads(settings_path.read_text())
        if (settings.get("c") != c or settings.get("roi_label") != "all_rois"
                or settings.get("features") != ["mean", "sd", "slope_per_TR"]):
            continue
        if path.name != f"{settings['analysis'].lower()}_summary.csv":
            continue
        for summary in pd.read_csv(path).to_dict("records"):
            rows.append({
                **{key: settings[key] for key in SEGMENT_PARAMETERS},
                "segment_length": settings["segment_length"],
                "segment_stride": settings["segment_stride"],
                "trim": settings["trim_responsive_intervals"],
                "analysis": settings["analysis"],
                **{metric: summary[metric] for metric in METRICS},
            })
    if not rows:
        raise ValueError("No fixed-C=0.01, all-ROI LR results found.")
    selected = select_results(pd.DataFrame(rows), SEGMENT_PARAMETERS).drop_duplicates()
    if selected.duplicated(["analysis", "segment_length"]).any():
        raise ValueError("Conflicting deterministic LR results for the same analysis/length.")
    return selected


def load_motion_results(root=MOTION_DIR, c=0.01, fd_threshold=0.25):
    """Read the new motion-only runs with matching segmentation and fixed C."""
    rows = []
    expected_features = ["mean_fd", "max_fd", "median_fd", "sd_fd",
                         f"fraction_fd_above_{fd_threshold:g}"]
    for path in sorted(Path(root).rglob("*_summary.csv")):
        settings_path = path.parent / "settings.json"
        if not settings_path.exists():
            continue
        settings = json.loads(settings_path.read_text())
        if (settings.get("c") != c or settings.get("fd_threshold") != fd_threshold
                or settings.get("features") != expected_features):
            continue
        if path.name != f"{settings['analysis'].lower()}_summary.csv":
            continue
        for summary in pd.read_csv(path).to_dict("records"):
            rows.append({
                **{key: settings[key] for key in SEGMENT_PARAMETERS},
                "segment_length": settings["segment_length"],
                "segment_stride": settings["segment_stride"],
                "trim": settings["trim_responsive_intervals"],
                "analysis": settings["analysis"],
                **{metric: summary[metric] for metric in METRICS},
            })
    if not rows:
        raise ValueError("No matching fixed-C motion-only results found.")
    selected = select_results(pd.DataFrame(rows), SEGMENT_PARAMETERS).drop_duplicates()
    if selected.duplicated(["analysis", "segment_length"]).any():
        raise ValueError("Conflicting motion-only results for the same analysis/length.")
    return selected


def load_mlp_results(root=MLP_DIR):
    """Read only the 16-unit, WD=0.01, 10-epoch MLP, one result per seed."""
    rows = []
    for path in sorted(Path(root).rglob("*_summary.csv")):
        settings_path = path.parent / "settings.json"
        if not settings_path.exists():
            continue
        settings = json.loads(settings_path.read_text())
        if (any(settings.get(key) != value for key, value in MLP_PARAMETERS.items())
                or settings.get("features") != ["mean", "sd", "slope_per_TR"]):
            continue
        if path.name != f"{settings['analysis'].lower()}_summary.csv":
            continue
        for summary in pd.read_csv(path).to_dict("records"):
            rows.append({
                **{key: settings[key] for key in SEGMENT_PARAMETERS},
                "segment_length": settings["segment_length"],
                "segment_stride": settings["segment_stride"],
                "trim": settings["trim_responsive_intervals"],
                "analysis": settings["analysis"], "seed": settings["seed"],
                **{metric: summary[metric] for metric in METRICS},
            })
    if not rows:
        return pd.DataFrame()
    selected = select_results(pd.DataFrame(rows), SEGMENT_PARAMETERS).drop_duplicates()
    if selected.duplicated(["analysis", "segment_length", "seed"]).any():
        raise ValueError("Conflicting MLP results for the same analysis/length/seed.")
    return selected


def visualize_length_and_trim(df, lr_results=None, motion_results=None, mlp_results=None):
    """Return one three-panel figure with ROI LR, GRU, MLP, and motion-only LOPO/LORO.

    Color encodes model; solid/dotted lines encode LOPO/LORO. GRU/MLP shading is
    a 95% Student-t interval for the mean across seeds, not across participants.
    Deterministic LR and motion-only fits have no seed interval; unavailable lengths remain missing.
    """
    gru = select_results(df, {**SEGMENT_PARAMETERS, **GRU_PARAMETERS})
    lr = load_lr_results() if lr_results is None else select_results(lr_results, SEGMENT_PARAMETERS)
    motion = (load_motion_results() if motion_results is None
              else select_results(motion_results, SEGMENT_PARAMETERS))
    if gru.empty or lr.empty or motion.empty:
        raise ValueError("Matching GRU, LR, and motion-only results are required.")
    deterministic = {"LR": lr, "Motion-only": motion}
    mlp = load_mlp_results() if mlp_results is None else mlp_results
    seeded_results = {"GRU": gru}
    if mlp.empty:
        print("No 16-unit, WD=0.01, 10-epoch MLP results found under plots/MLP; old MLP runs are excluded.")
    else:
        seeded_results["MLP"] = select_results(mlp, SEGMENT_PARAMETERS)
    seeded = {}
    count_labels = {}
    for model, results in seeded_results.items():
        if "seed" not in results or results["seed"].isna().any():
            raise ValueError(f"{model} results must have valid seed values.")
        if results.empty:
            continue
        seeded[model] = results.groupby(["analysis", "segment_length", "seed"])[list(METRICS)].mean()
        counts = set(seeded[model].groupby(level=["analysis", "segment_length"]).size())
        count_labels[model] = str(min(counts)) if len(counts) == 1 else f"{min(counts)}–{max(counts)}"
    models = ["LR", *seeded, "Motion-only"]
    with plt.rc_context({"font.size": 18, "axes.labelsize": 20,
                         "xtick.labelsize": 16, "ytick.labelsize": 16,
                         "axes.linewidth": 1.4, "xtick.major.width": 1.4,
                         "ytick.major.width": 1.4}):
        fig, axes = plt.subplots(3, 1, figsize=(7.3, 8.4), sharex=True)
        fig.subplots_adjust(left=0.12, right=0.61, bottom=0.09, top=0.98, hspace=0.13)
        for ax, (metric, ylabel) in zip(axes, METRICS.items()):
            for model in models:
                for analysis, linestyle in STYLES.items():
                    means = np.full(len(LENGTHS), np.nan)
                    widths = means.copy()
                    for i, length in enumerate(LENGTHS):
                        if model in seeded:
                            per_seed = seeded[model]
                            if (analysis, length) not in per_seed.index.droplevel("seed"):
                                continue
                            values = per_seed.loc[(analysis, length), metric].dropna().to_numpy()
                        else:
                            results = deterministic[model]
                            values = results.loc[results.analysis.eq(analysis) & results.segment_length.eq(length), metric].dropna().to_numpy()
                            if len(values) > 1:
                                raise ValueError(f"{model} must have one deterministic result per analysis/length.")
                        if not len(values):
                            continue
                        means[i] = values.mean()
                        if model in seeded and len(values) > 1:
                            widths[i] = (t.ppf(0.975, df=len(values) - 1)
                                         * values.std(ddof=1) / np.sqrt(len(values)))
                    ax.plot(LENGTHS, means, color=COLORS[model], linestyle=linestyle,
                            linewidth=3, marker="o", markersize=6,
                            label=f"{model} {analysis.upper()}")
                    if model in seeded:
                        ax.fill_between(LENGTHS, means - widths, means + widths,
                                        color=COLORS[model], alpha=0.18, linewidth=0)
            ax.set_ylabel("Accuracy" if metric == "balanced_test_accuracy" else "Recall")
            ax.text(0.025, 0.96, ylabel, transform=ax.transAxes,
                    ha="left", va="top", fontsize=16,
                    bbox=dict(facecolor="white", edgecolor="0.8",
                              linewidth=0.8, boxstyle="round,pad=0.3"))
            lower = 0.2 if metric == "nonresponsive_recall" else 0.5
            # Leave room for the label above motion-only responsive recall (~0.97).
            upper = 1.08 if metric == "responsive_recall" else 1
            ax.set_ylim(lower, upper)
            if metric == "nonresponsive_recall":
                ax.set_yticks(np.arange(0.2, 1.01, 0.2))
                ax.set_yticks(np.arange(0.3, 1.0, 0.2), minor=True)
            else:
                ax.set_yticks(np.arange(lower, 1.01, 0.1))
            ax.grid(which="both", alpha=0.22, linewidth=1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for ax in axes[:-1]:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        axes[-1].set_xticks(LENGTHS)
        axes[-1].set_xlabel("Segment length (TRs)")
        handles = [
            Line2D([], [], color=COLORS["Motion-only"], linestyle="-", linewidth=3, label="Motion-only"),
            Line2D([], [], color=COLORS["LR"], linestyle="-", linewidth=3, label="LR"),
        ]
        for model in seeded:
            handles.extend([
                Line2D([], [], color=COLORS[model], linestyle="-", marker="o", markersize=6, linewidth=3,
                       label=f"{model} mean\nacross {count_labels[model]} seeds"),
                Patch(facecolor=COLORS[model], alpha=0.18, linewidth=0,
                      label=f"{model} 95% CI\nacross {count_labels[model]} seeds"),
            ])
        handles.extend([
            Line2D([], [], color="black", linestyle=STYLES["lopo"], linewidth=3, label="LOPO"),
            Line2D([], [], color="black", linestyle=STYLES["loro"], linewidth=3, label="LORO"),
        ])
        fig.legend(handles=handles, ncol=1, loc="center left", bbox_to_anchor=(0.63, 0.5),
                   fontsize=14, handlelength=1.8, handletextpad=0.6,
                   labelspacing=1.0, frameon=True, facecolor="white",
                   edgecolor="0.8", framealpha=1, borderpad=0.8)
    return fig, axes


if __name__ == "__main__":
    all_results = pd.read_csv(RESULTS_DIR / "all_rois_summary.csv")
    figure, _ = visualize_length_and_trim(all_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "results.png"
    figure.savefig(output, dpi=250)
    plt.close(figure)
    print(f"Saved {output}")
