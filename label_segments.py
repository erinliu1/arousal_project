from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


SAVE_PATH = Path('plots/label_segments')
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def plot_non_responsive_segment_counts(threshold=20):
    participant_counts = pd.read_csv(Path('plots/analyze_non_responsiveness') / 'threshold_sensitivity_participant_counts.csv')
    participants = participant_counts['participant'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(participants)))
    participant_colors = dict(zip(participants, colors))

    col = f"T{threshold}"

    df = (
        participant_counts[["participant", col]]
        .rename(columns={col: "n_events"})
        .sort_values("n_events", ascending=False)
        .copy()
    )
    
    df['participant_num'] = df['participant'].str.split('_').str[-1].astype(int)

    fig, ax = plt.subplots(figsize=(8, 4))
    
    bar_colors = [
        participant_colors[p]
        for p in df["participant"]
    ]

    bars = ax.bar(
        df["participant_num"].astype(str),
        df["n_events"]
    , color=bar_colors)

    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Participant")
    ax.set_ylabel(f"Number of BP intervals > {threshold} TRs")
    ax.set_title(f"Non-responsive intervals per participant at {threshold}-TR threshold")

    ax.tick_params(axis="x")

    plt.tight_layout()
    plt.savefig(SAVE_PATH / f"non_responsive_intervals_per_participant_{threshold}.png", dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__":
    plot_non_responsive_segment_counts(threshold=20)