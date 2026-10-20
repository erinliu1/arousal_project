from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json

RAW_DATA = Path("raw_data")

motion_TR_cache = {}

def load_motion_TRs(participant, run):
    key = (participant, run)

    if key not in motion_TR_cache:
        path = RAW_DATA / participant / run / "motion_TRs.npy"
        motion_TR_cache[key] = np.asarray(
            np.load(path),
            dtype=int
        )

    return motion_TR_cache[key]

SAVE_PATH = Path('plots/label_segments')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

def get_centered_segments(start_TR, end_TR_exclusive, segment_length=10, left_buffer=5,right_buffer=5):
    """
    Split an interval into non-overlapping fixed-length segments.

    Interval convention:
        [start_TR, end_TR_exclusive)

    distribute unused TRs approximately equally between the two sides
    """
    usable_start = int(start_TR) + left_buffer
    usable_end = int(end_TR_exclusive) - right_buffer
    usable_length = usable_end - usable_start
    n_segments = usable_length // segment_length
    if n_segments <= 0:
        return []
    used_length = n_segments * segment_length
    leftover = usable_length - used_length
    offset = leftover // 2
    first_start = usable_start + offset
    segments = []
    for i in range(n_segments):
        seg_start = first_start + i * segment_length
        seg_end = seg_start + segment_length
        segments.append((seg_start, seg_end))
    return segments

def label_non_responsive_segments(segment_length=10, buffer=10):
    non_responsiveness_dir = Path('plots/analyze_non_responsiveness')
    intervals = pd.read_csv(non_responsiveness_dir / 'non_responsive_intervals.csv')
    end_intervals = pd.read_csv(non_responsiveness_dir / 'non_responsive_end_intervals.csv')
    whole_runs = pd.read_csv(non_responsiveness_dir / 'non_responsive_whole_runs.csv')

    labeled_segments = []

    # bounded non-responsive interval between two button presses
    for _, row in intervals.iterrows():
        interval_start = int(row["start_TR"])
        interval_end = int(row["end_TR"])

        segments = get_centered_segments(
            start_TR=interval_start,
            end_TR_exclusive=interval_end,
            segment_length=segment_length,
            left_buffer=buffer,
            right_buffer=buffer,
        )
        participant = row["participant"]
        run = row["run"]
        motion_TRs = load_motion_TRs(participant, run)

        for segment_start, segment_end in segments:
            contains_motion = np.any((motion_TRs >= segment_start) & (motion_TRs < segment_end))

            labeled_segments.append({
                "participant": row["participant"],
                "run": row["run"],
                "label": "non-responsive",
                "segment_start_TR": segment_start,
                "segment_end_TR_exclusive": segment_end,
                "segment_length": segment_length,
                "contains_motion": contains_motion,
                "source_type": "bounded_interval",
            })

    # last button press -> end of run
    for _, row in end_intervals.iterrows():
        interval_start = int(row["start_TR"])
        interval_end = int(row["end_TR"])

        segments = get_centered_segments(
            start_TR=interval_start,
            end_TR_exclusive=interval_end,
            segment_length=segment_length,
            left_buffer=buffer,
            right_buffer=buffer,
        )
        participant = row["participant"]
        run = row["run"]
        motion_TRs = load_motion_TRs(participant, run)

        for segment_start, segment_end in segments:
            contains_motion = np.any((motion_TRs >= segment_start) & (motion_TRs < segment_end))

            labeled_segments.append({
                "participant": row["participant"],
                "run": row["run"],
                "label": "non-responsive",
                "segment_start_TR": segment_start,
                "segment_end_TR_exclusive": segment_end,
                "segment_length": segment_length,
                "contains_motion": contains_motion,
                "source_type": "end_interval",
            })

    # whole runs with no usable button presses
    for _, row in whole_runs.iterrows():
        interval_start = int(row["start_TR"])
        interval_end = int(row["end_TR"])

        segments = get_centered_segments(
            start_TR=interval_start,
            end_TR_exclusive=interval_end,
            segment_length=segment_length,
            left_buffer=buffer,
            right_buffer=buffer,
        )
        participant = row["participant"]
        run = row["run"]
        motion_TRs = load_motion_TRs(participant, run)

        for segment_start, segment_end in segments:
            contains_motion = np.any((motion_TRs >= segment_start) & (motion_TRs < segment_end))

            labeled_segments.append({
                "participant": row["participant"],
                "run": row["run"],
                "label": "non-responsive",
                "segment_start_TR": segment_start,
                "segment_end_TR_exclusive": segment_end,
                "segment_length": segment_length,
                "contains_motion": contains_motion,
                "source_type": "whole_run",
            })

    segments_df = pd.DataFrame(labeled_segments)

    if len(segments_df) == 0:
        return segments_df

    # Sort
    segments_df["participant_num"] = segments_df["participant"].str.extract(r"(\d+)")[0].astype(int)
    segments_df["run_num"] = segments_df["run"].str.extract(r"(\d+)")[0].astype(int)

    segments_df = (
        segments_df
        .sort_values([
            "participant_num",
            "run_num",
            "segment_start_TR",
        ])
        .reset_index(drop=True)
    )
    segments_df = segments_df.drop(columns=["participant_num", "run_num"])

    n_motion = int(segments_df["contains_motion"].sum())
    print(f"Non-responsive {segment_length}-TR segments: {len(segments_df)} total, {n_motion} contain motion, {len(segments_df) - n_motion} motion-free")
    print(segments_df["source_type"].value_counts())

    segments_df.to_csv(SAVE_PATH / "non_responsive_10TR_segments.csv",index=False)
    return segments_df
    
def label_responsive_segments(segment_length=10, buffer=10):
    responsive_dir = Path("plots/analyze_responsiveness")
    responsive_intervals = pd.read_csv(responsive_dir / 'responsive_intervals' / "responsive_intervals.csv")

    labeled_segments = []
    for _, row in responsive_intervals.iterrows():
        interval_start = int(row["start_TR"])
        interval_end = int(row["end_TR"])   # exclusive

        segments = get_centered_segments(
            start_TR=interval_start,
            end_TR_exclusive=interval_end,
            segment_length=segment_length,
            left_buffer=buffer,
            right_buffer=buffer,
        )
        participant = row["participant"]
        run = row["run"]
        motion_TRs = load_motion_TRs(participant, run)

        for segment_start, segment_end in segments:
            contains_motion = np.any(
                (motion_TRs >= segment_start) &
                (motion_TRs < segment_end)
            )

            labeled_segments.append({
                "participant": participant,
                "run": run,
                "label": "responsive",
                "segment_start_TR": segment_start,
                "segment_end_TR_exclusive": segment_end,
                "segment_length": segment_length,
                "contains_motion": contains_motion,
                "source_type": "responsive_interval",
            })

    segments_df = pd.DataFrame(labeled_segments)

    if len(segments_df) == 0:
        return segments_df

    # ------------------------------------------------------------
    # Sort by participant, run, and segment
    # ------------------------------------------------------------

    segments_df["participant_num"] = segments_df["participant"].str.extract(r"(\d+)")[0].astype(int)
    segments_df["run_num"] = segments_df["run"].str.extract(r"(\d+)")[0].astype(int)

    segments_df = (
        segments_df
        .sort_values([
            "participant_num",
            "run_num",
            "segment_start_TR",
        ])
        .reset_index(drop=True)
    )
    segments_df = segments_df.drop(columns=["participant_num", "run_num"])
    output_path = (SAVE_PATH / "responsive_10TR_segments.csv")
    segments_df.to_csv(output_path, index=False)

    n_motion = int(segments_df["contains_motion"].sum())
    print(f"Responsive {segment_length}-TR segments: {len(segments_df)} total, {n_motion} contain motion, {len(segments_df) - n_motion} motion-free")
    print(f"{segments_df['participant'].nunique()} participants")
    print(f"{segments_df[['participant', 'run']].drop_duplicates().shape[0]} runs")

    return segments_df

def plot_segment_counts_by_participant():
    responsive = pd.read_csv(SAVE_PATH / "responsive_10TR_segments.csv")
    non_responsive = pd.read_csv(SAVE_PATH / "non_responsive_10TR_segments.csv")
    responsive = responsive[~responsive["contains_motion"].astype(str).str.lower().eq("true")]
    non_responsive = non_responsive[~non_responsive["contains_motion"].astype(str).str.lower().eq("true")]
    participant_numbers = pd.Index(range(1, 22), name="participant_num")

    def counts_by_participant(segments):
        participant_num = segments["participant"].str.extract(r"(\d+)")[0].astype(int)
        return participant_num.value_counts().reindex(participant_numbers, fill_value=0).sort_index()

    responsive_counts = counts_by_participant(responsive)
    non_responsive_counts = counts_by_participant(non_responsive)
    x = np.arange(1, 22)
    width = 0.4

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="lightgray", linewidth=0.8, alpha=0.6)
    for separator in np.arange(1.5, 21, 1):
        ax.axvline(separator, color="lightgray", linewidth=0.8, zorder=0)
    ax.bar(x - width / 2, responsive_counts, width, color="#9ADF8C", edgecolor='black', linewidth=1, label="Responsive")
    ax.bar(x + width / 2, non_responsive_counts, width, color="pink", edgecolor='black', linewidth=1, label="Non-responsive")
    ax.set_xlabel("Participant", fontsize=22)
    ax.set_ylabel("Number of segments", fontsize=22)
    ax.set_title("Motion-free Responsive and Non-responsive Segments per Participant", fontsize=28)
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(SAVE_PATH / "segment_counts_by_participant.png", dpi=300)
    plt.close()

def plot_segment_counts_by_participant_and_run():
    responsive = pd.read_csv(SAVE_PATH / "responsive_10TR_segments.csv")
    non_responsive = pd.read_csv(SAVE_PATH / "non_responsive_10TR_segments.csv")
    responsive = responsive[~responsive["contains_motion"].astype(str).str.lower().eq("true")]
    non_responsive = non_responsive[~non_responsive["contains_motion"].astype(str).str.lower().eq("true")]

    responsive_counts = responsive.groupby(["participant", "run"]).size().rename("responsive")
    non_responsive_counts = non_responsive.groupby(["participant", "run"]).size().rename("non_responsive")
    counts = (
        pd.concat([responsive_counts, non_responsive_counts], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    counts["_participant_num"] = pd.to_numeric(
        counts["participant"].astype(str).str.extract(r"(\d+)$")[0],
        errors="coerce",
    )
    counts["_run_num"] = pd.to_numeric(
        counts["run"].astype(str).str.extract(r"(\d+)$")[0],
        errors="coerce",
    )
    counts = counts.sort_values(["_participant_num", "_run_num"]).reset_index(drop=True)

    participants = counts["participant"].drop_duplicates().tolist()
    participant_colors = dict(
        zip(participants, plt.cm.tab20(np.linspace(0, 1, len(participants))))
    )
    metric_specs = (
        ("responsive", -0.20, True, "Responsive segments"),
        ("non_responsive", 0.20, False, "Non-responsive segments"),
    )

    fig, ax = plt.subplots(figsize=(27, 9))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="lightgray", linewidth=1, alpha=0.6)
    participant_centers = []
    participant_labels = []
    x = 0

    for participant_index, (participant, participant_df) in enumerate(
        counts.groupby("participant", sort=False)
    ):
        participant_xs = []
        color = participant_colors[participant]

        for row_index, (_, row) in enumerate(participant_df.iterrows()):
            for metric, offset, filled, _ in metric_specs:
                if row[metric] == 0:
                    continue
                ax.scatter(
                    x + offset,
                    row[metric],
                    marker="o",
                    s=170,
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    alpha=0.85,
                    linewidths=2,
                    zorder=3,
                )

            participant_xs.append(x)
            if row_index < len(participant_df) - 1:
                ax.axvline(x + 0.5, color="lightgray", linewidth=2, zorder=0)
            x += 1

        participant_centers.append(np.mean(participant_xs))
        participant_number = participant_df["_participant_num"].iloc[0]
        participant_labels.append(
            int(participant_number)
            if pd.notna(participant_number)
            else participant
        )
        if participant_index < len(participants) - 1:
            ax.axvline(x - 0.5, color="gray", linewidth=3, alpha=0.6, zorder=0)

    ax.set_xticks(participant_centers)
    ax.set_xticklabels(participant_labels, fontsize=22)
    ax.tick_params(axis="x", width=2, length=8)
    ax.tick_params(axis="y", which="both", length=0, labelsize=22)
    ax.set_xlabel("Participant", fontsize=32, labelpad=15)
    ax.set_ylabel("Number of motion-free segments", fontsize=30, labelpad=15)
    ax.set_ylim(bottom=-1)
    ax.set_title(
        "Responsive and Non-responsive Segments by Participant and Run",
        fontsize=34,
        fontweight="bold",
    )
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="0.4" if filled else "none",
            markeredgecolor="0.4",
            markeredgewidth=2,
            markersize=13,
            label=label,
        )
        for _, _, filled, label in metric_specs
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=18,
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        framealpha=1,
        loc="upper right",
    )

    fig.tight_layout()
    output_path = SAVE_PATH / "segment_counts_by_participant_and_run.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved segment count visualization to {output_path}")
    return output_path

if __name__ == "__main__":
    # label_non_responsive_segments()
    # label_responsive_segments()
    plot_segment_counts_by_participant()
