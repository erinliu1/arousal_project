from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSIFIER_DIR = Path("plots/fMRI_classifier_exploration2")
RESULTS_DIR = CLASSIFIER_DIR / "results"


def _plot_length_and_trim(df, analysis):
    """Return three stacked metric plots with seed means and ±1 SD error bars.

    Select G_max=4 and segment length=stride in 10, 20, ..., 60 TRs.
    Repeated rows for a seed are averaged first so every seed has equal weight.
    Sample standard deviation bars are omitted for points with only one seed. Filter other
    experiment settings in ``df`` before calling to compare a specific model.
    """
    metrics = {
        "balanced_test_accuracy": "Balanced test accuracy",
        "responsive_recall": "Responsive recall",
        "nonresponsive_recall": "Nonresponsive recall",
    }
    required = {"responsive_max_bp_gap", "segment_length", "segment_stride",
                "trim", "seed", *metrics}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    lengths = np.arange(10, 61, 10)
    selected = df.loc[
        df["responsive_max_bp_gap"].eq(4)
        & df["segment_length"].eq(df["segment_stride"])
        & df["segment_length"].isin(lengths)
    ].copy()
    if selected.empty:
        raise ValueError("No rows match G_max=4 and segment length=stride in 10–60 TRs.")
    # Handle both Boolean values and their CSV string representations.
    selected["trim"] = selected["trim"].map(
        {True: True, False: False, "True": True, "False": False,
         "true": True, "false": False}
    )
    if selected["trim"].isna().any() or selected["seed"].isna().any():
        raise ValueError("Matching rows must have valid trim flags and seed values.")
    per_seed = selected.groupby(["trim", "segment_length", "seed"])[list(metrics)].mean()

    with plt.rc_context({"font.size": 16, "axes.titlesize": 21,
                         "axes.labelsize": 18, "legend.fontsize": 15}):
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True,
                                 constrained_layout=True)
        for ax, (metric, title) in zip(axes, metrics.items()):
            for trim, color, label in [(False, "tab:blue", "Trim disabled"),
                                       (True, "tab:orange", "Trim enabled")]:
                means = np.full(len(lengths), np.nan)
                stds = means.copy()
                for i, length in enumerate(lengths):
                    if (trim, length) not in per_seed.index.droplevel("seed"):
                        continue
                    values = per_seed.loc[(trim, length), metric].dropna().to_numpy()
                    if not len(values):
                        continue
                    means[i] = values.mean()
                    if len(values) > 1:
                        stds[i] = values.std(ddof=1)
                ax.plot(lengths, means, color=color, label=label,
                        linewidth=3, marker="o", markersize=8)
                has_std = np.isfinite(stds)
                ax.errorbar(lengths[has_std], means[has_std], yerr=stds[has_std],
                            fmt="none", ecolor=color, elinewidth=2,
                            capsize=5, capthick=2)
            ax.set_title(title)
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            ax.set_xticks(lengths)
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_xlabel("Segment length (TRs)")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(
            f"{analysis.upper()}\n"
            "R(W=30,S=1,N_min=8,G_max=4):NR(T_min=20)\n"
            "model=GRU,pool=final,hidden=32,drop=0.2,epochs=30,batch=32,LR=1e-3,WD=1e-3",
            fontsize=12,
        )
    return fig, axes


def visualize_length_and_trim(df):
    """Return {analysis: (figure, axes)} for separate LOPO and LORO plots.

    Fix the interval and model parameters below, vary length=stride from 10 to
    60 TRs, and compare trim settings using seed means and ±1 sample SD bars.
    Analyses with no matching rows are reported and skipped.
    """
    parameters = {
        "responsive_window_size": 30,
        "responsive_window_stride": 1,
        "responsive_min_bp_trs": 8,
        "responsive_max_bp_gap": 4,
        "nonresponsive_min_gap": 20,
        "model": "GRU",
        "pool": "final",
        "hidden": 32,
        "drop": 0.2,
        "epochs": 30,
        "batch": 32,
        "lr": 1e-3,
        "wd": 1e-3,
    }
    required = {*parameters, "analysis", "segment_length", "segment_stride",
                "trim", "seed", "balanced_test_accuracy", "responsive_recall",
                "nonresponsive_recall"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    mask = pd.Series(True, index=df.index)
    for column, value in parameters.items():
        mask &= df[column].eq(value)
    mask &= df["segment_length"].eq(df["segment_stride"])
    mask &= df["segment_length"].isin(range(10, 61, 10))
    selected = df.loc[mask]
    if selected.empty:
        raise ValueError("No rows match the requested interval and model parameters.")

    figures = {}
    for analysis in ("lopo", "loro"):
        subset = selected.loc[selected["analysis"].str.lower().eq(analysis)]
        if subset.empty:
            print(f"No matching {analysis.upper()} results; skipping plot.")
            continue
        figures[analysis] = _plot_length_and_trim(subset, analysis)
    return figures


if __name__ == "__main__":
    all_results = pd.read_csv(RESULTS_DIR / "all_rois_summary.csv")
    figures = visualize_length_and_trim(all_results)
    for analysis, (figure, _) in figures.items():
        figure.savefig(RESULTS_DIR / f"length_and_trim_{analysis}.png", dpi=200)
    plt.show()
