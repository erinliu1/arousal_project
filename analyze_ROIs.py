"""Compare mean ROI z-scores in responsive and non-responsive segments."""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from all_rois import aan_rois, thalamic_rois, cortical_rois, all_rois

METADATA_PATH = Path('plots/GRU/segments/R(W=30,S=1,N_min=8,G_max=4):NR(T_min=20):Segment(TR=40,SW=40)/all_segment_metadata.csv')
RAW_DATA_DIR = Path('raw_data')


def summarize_roi_segments():
    metadata = pd.read_csv(METADATA_PATH)
    data = []

    for (participant, run), segments in metadata.groupby(['participant', 'run']):
        responsive_segments = segments[segments['label'] == 'responsive']
        nonresponsive_segments = segments[segments['label'] == 'non-responsive']

        for roi in all_rois:
            roi_path = RAW_DATA_DIR / participant / run / 'z_scored' / f'{roi}.npy'
            roi_data = np.load(roi_path)

            # Average each segment, then average the segment means within each class.
            responsive_means = []
            for segment in responsive_segments.itertuples():
                start = int(segment.segment_start_TR)
                end = int(segment.segment_end_TR_exclusive)
                responsive_means.append(np.mean(roi_data[start:end]))

            nonresponsive_means = []
            for segment in nonresponsive_segments.itertuples():
                start = int(segment.segment_start_TR)
                end = int(segment.segment_end_TR_exclusive)
                nonresponsive_means.append(np.mean(roi_data[start:end]))

            mean_R = np.mean(responsive_means) if responsive_means else np.nan
            mean_NR = np.mean(nonresponsive_means) if nonresponsive_means else np.nan

            data.append({
                'participant': participant,
                'run': run,
                'roi': roi,
                'n_R': len(responsive_segments),
                'n_NR': len(nonresponsive_segments),
                'mean_R': mean_R,
                'mean_NR': mean_NR,
                'delta_R_minus_NR': mean_R - mean_NR,
            })

    summary = pd.DataFrame(data)
    save_path = Path('plots') / 'z_score' / 'roi_run_means.csv'
    summary.to_csv(save_path, index=False)
    print(f'Saved ROI summaries to {save_path}')
    return summary

def summarize_roi_segments_by_participant(min_n_segments=None):
    summary = summarize_roi_segments()
    eligible_runs = summary[(summary['n_R'] > 0) & (summary['n_NR'] > 0)]
    if min_n_segments is not None:
        eligible_runs = eligible_runs[
            (eligible_runs['n_R'] >= min_n_segments)
            & (eligible_runs['n_NR'] >= min_n_segments)
        ]

    participant_summary = eligible_runs.groupby(['participant', 'roi']).agg(
        n_eligible_runs=('run', 'nunique'),
        mean_delta_R_minus_NR=('delta_R_minus_NR', 'mean'),
    ).reset_index()

    name = f'roi_participant_means'
    if min_n_segments:
        name += f'_minN={min_n_segments}'
    
    save_path = Path('plots') / 'z_score' / f'{name}.csv'
    participant_summary.to_csv(save_path, index=False)
    print(f'Saved ROI summaries by participant to {save_path}')
    return participant_summary

def visualize_participant_roi_heatmap():
    df = summarize_roi_segments_by_participant()
    # columns should be the 42 ROIs, rows should be the participants, and the values should be the mean delta_R_minus_NR

    heatmap_data = df.pivot(index='participant', columns='roi', values='mean_delta_R_minus_NR')
    
    # Count segments from the same eligible runs used in the participant means.
    metadata = pd.read_csv(METADATA_PATH)
    run_counts = metadata.groupby(['participant', 'run', 'label']).size().unstack(fill_value=0)
    run_counts = run_counts.reindex(columns=['responsive', 'non-responsive'], fill_value=0)
    eligible_counts = run_counts[
        (run_counts['responsive'] > 0) & (run_counts['non-responsive'] > 0)
    ]
    participant_counts = eligible_counts.groupby('participant').sum()
    average_counts = participant_counts[['responsive', 'non-responsive']].mean(axis=1)
    has_five_each = (participant_counts['responsive'] >= 5) & (participant_counts['non-responsive'] >= 5)

    # Thalamus, cortex, then AAN; sort using only participants with >= 5 of each class.
    participants_for_roi_means = has_five_each.reindex(heatmap_data.index)
    roi_means = heatmap_data.loc[participants_for_roi_means].mean()
    roi_order = []
    group_sizes = []
    for roi_group in [thalamic_rois, cortical_rois, aan_rois]:
        group_means = roi_means[[roi for roi in roi_group if roi in roi_means.index]]
        roi_order.extend(group_means.sort_values(ascending=False).index)
        group_sizes.append(len(group_means))
    heatmap_data = heatmap_data[roi_order]


    participant_sort = pd.DataFrame({
        'has_five_each': has_five_each,
        'average_count': average_counts,
    }).reindex(heatmap_data.index)
    participant_order = participant_sort.sort_values(
        ['has_five_each', 'average_count'], ascending=[False, False],
    ).index
    heatmap_data = heatmap_data.loc[participant_order]

    participant_labels = []
    for participant in heatmap_data.index:
        participant_id = int(participant.removeprefix('AAN_'))
        n_R = int(participant_counts.loc[participant, 'responsive'])
        n_NR = int(participant_counts.loc[participant, 'non-responsive'])
        participant_labels.append(f'#{participant_id:2} (R={n_R:2}, NR={n_NR:3})')

    plt.figure(figsize=(18, 6))

    ax = sns.heatmap(heatmap_data, cmap='coolwarm', center=0, annot=True, fmt=".1f", annot_kws={'fontsize': 6}, vmin=-2, vmax=2, cbar_kws={'label': 'Mean ΔR-NR (zBOLD)'})
    ax.set_yticklabels(participant_labels, rotation=0, fontfamily='monospace')

    group_start = 0
    for group_name, group_size in zip(['Thalamus', 'Cortex', 'AAN'], group_sizes):
        if group_size == 0:
            continue
        if group_start > 0:
            ax.axvline(group_start, color='black', linewidth=2.5)
        ax.text(group_start + group_size / 2, 1.02, group_name,
                transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=12, fontweight='bold')
        group_start += group_size
    plt.xlabel("ROI")
    plt.ylabel("Participant")
    plt.title("Mean ΔR-NR (BOLD z-scored within runs) by Participant and ROI", pad=35)
    plt.xticks(rotation=90)
    plt.tight_layout()
    save_path = Path('plots') / 'z_score' / 'roi_participant_heatmap.png'
    plt.savefig(save_path)
    print(f'Saved heatmap to {save_path}')

def visualize_group_average_delta():
    df = summarize_roi_segments_by_participant()
    participant_data = df.pivot(index='participant', columns='roi', values='mean_delta_R_minus_NR')

    # Use the same participant counts and ROI ordering as the heatmap.
    metadata = pd.read_csv(METADATA_PATH)
    run_counts = metadata.groupby(['participant', 'run', 'label']).size().unstack(fill_value=0)
    run_counts = run_counts.reindex(columns=['responsive', 'non-responsive'], fill_value=0)
    eligible_counts = run_counts[
        (run_counts['responsive'] > 0) & (run_counts['non-responsive'] > 0)
    ]
    participant_counts = eligible_counts.groupby('participant').sum()
    has_five_each = (participant_counts['responsive'] >= 5) & (participant_counts['non-responsive'] >= 5)
    sort_means = participant_data.loc[has_five_each.reindex(participant_data.index)].mean()
    roi_order = []
    group_sizes = []
    for roi_group in [thalamic_rois, cortical_rois, aan_rois]:
        group_means = sort_means[[roi for roi in roi_group if roi in sort_means.index]]
        roi_order.extend(group_means.sort_values(ascending=False).index)
        group_sizes.append(len(group_means))
    participant_data = participant_data[roi_order]

    fig, ax = plt.subplots(figsize=(18, 6))
    positions = np.arange(len(roi_order))
    participants_with_five = participant_data.loc[has_five_each.reindex(participant_data.index)]
    rng = np.random.default_rng(42)
    for group_data, label, color, offset in [
        (participant_data, 'All eligible participants', 'tab:blue', -0.15),
        (participants_with_five, 'Participants with R ≥ 5 and NR ≥ 5', 'tab:orange', 0.15),
    ]:
        # Resample participants within each group, keeping their ROI values together.
        values = group_data.to_numpy()
        group_means = group_data.mean().to_numpy()
        bootstrap_means = []
        for _ in range(10000):
            sample_indices = rng.integers(0, len(values), size=len(values))
            bootstrap_means.append(np.nanmean(values[sample_indices], axis=0))
        lower, upper = np.nanpercentile(bootstrap_means, [2.5, 97.5], axis=0)

        ax.vlines(positions + offset, lower, upper, color=color, linewidth=1.5)
        ax.scatter(positions + offset, group_means, color=color, s=25, zorder=3,
                   label=f'{label} (N={len(group_data)})')
    ax.legend()
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(roi_order, rotation=90)
    ax.set_xlim(-0.5, len(roi_order) - 0.5)

    group_start = 0
    for group_name, group_size in zip(['Thalamus', 'Cortex', 'AAN'], group_sizes):
        if group_size == 0:
            continue
        if group_start > 0:
            ax.axvline(group_start - 0.5, color='black', linewidth=2.5)
        ax.text(group_start + (group_size - 1) / 2, 1.02, group_name,
                transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=12, fontweight='bold')
        group_start += group_size

    ax.set_xlabel('ROI')
    ax.set_ylabel('Mean ΔR-NR (BOLD_z)')
    ax.set_title('Group Mean ΔR-NR with 95% Participant-Bootstrap CIs', pad=35)
    fig.tight_layout()
    save_path = Path('plots') / 'z_score' / 'roi_group_average_delta.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f'Saved group average delta plot to {save_path}')

    
if __name__ == '__main__':
    # visualize_participant_roi_heatmap()
    visualize_group_average_delta()
