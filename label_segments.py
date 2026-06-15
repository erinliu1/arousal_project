import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint as pprint
import joblib

from all_rois import get_all_rois_with_cortical_and_thalamic, all_cortical_rois, all_thalamic_rois

all_rois = get_all_rois_with_cortical_and_thalamic()

my_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/data'
training_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/training_data'
save_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/fyi'
os.makedirs(training_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# def get_prearousal_segments(arousal_TRs, bp_TRs, motion_TRs, TR=0.98, window_s=10.0, no_bps=20.0):
#     """
#     Return a list of tuples (start_idx, end_idx) for valid pre-arousal segments for this run.
#     A pre-arousal segment is the `window_s` segment ending at the arousal TR (inclusive).
#     Exclude the segment if it contains any motion TRs, if there are any other arousals within the segment, or if there are any button presses within no_bps seconds before the segment.
#     """
#     N_inside = int(window_s / TR) + 1 # the time between two consecutive TRs = 0.98s, so across N_10 = 11 TRs, there are 10 intervals of 0.98s, which gets approximately 10s of data (9.8s to be exact)

#     N_no_bps = int(no_bps / TR)
#     arousal_TRs = np.asarray(arousal_TRs, dtype=int)
#     bp_TRs = np.asarray(bp_TRs, dtype=int)
#     motion_TRs = np.asarray(motion_TRs, dtype=int)

#     prearousal_segments = []
    
#     for arousal_idx in arousal_TRs:
            
#         end_idx = arousal_idx # the segment ends exactly at the arousal time, but the arousal time is actually the TR time preceding the real arousal time, so the segment still does not capture the actual moment of arousal.
        
#         start_idx = end_idx - N_inside + 1 
            
#         if start_idx < 0:
#             continue
            
#         segment_TRs = np.arange(start_idx, end_idx + 1)

#         other_arousal_TRs = arousal_TRs[arousal_TRs != arousal_idx]
        
#         segment_has_other_arousal = np.isin(segment_TRs, other_arousal_TRs).any()
#         segment_has_motion = np.isin(segment_TRs, motion_TRs).any()

#         if segment_has_other_arousal:
#             continue
        
#         if segment_has_motion:
#             continue

#         preceding_start_idx = max(0, start_idx - N_no_bps)
#         preceding_end_idx = start_idx - 1

#         has_preceding_bp = np.any(
#             (bp_TRs >= preceding_start_idx) & (bp_TRs <= preceding_end_idx)
#         )

#         if has_preceding_bp:
#             continue

#         prearousal_segments.append((start_idx, end_idx))
    
#     return prearousal_segments

def get_prearousal_segments(
    arousal_TRs,
    bp_TRs,
    motion_TRs,
    num_TRs,
    TR=0.98,
    before_s=10.0,
    after_s=10.0,
    no_bps_before_s=10.0
):
    """
    Return valid sleep-to-wake transition segments.

    Segment definition:
      [10 s before arousal, 10 s after arousal]

    The arousal TR is treated as the transition event.
    """

    N_before = int(before_s / TR)
    N_after = int(after_s / TR)
    N_no_bps_before = int(no_bps_before_s / TR)

    arousal_TRs = np.asarray(arousal_TRs, dtype=int)
    bp_TRs = np.asarray(bp_TRs, dtype=int)
    motion_TRs = np.asarray(motion_TRs, dtype=int)

    prearousal_segments = []

    for arousal_idx in arousal_TRs:
        start_idx = arousal_idx - N_before
        end_idx = arousal_idx + N_after

        if start_idx < 0:
            continue

        if end_idx >= num_TRs:
            continue

        segment_TRs = np.arange(start_idx, end_idx + 1)

        other_arousal_TRs = arousal_TRs[arousal_TRs != arousal_idx]

        segment_has_other_arousal = np.isin(segment_TRs, other_arousal_TRs).any()
        segment_has_motion = np.isin(segment_TRs, motion_TRs).any()

        if segment_has_other_arousal:
            continue

        if segment_has_motion:
            continue

        preceding_start_idx = max(0, start_idx - N_no_bps_before)
        preceding_end_idx = start_idx - 1

        has_preceding_bp = np.any(
            (bp_TRs >= preceding_start_idx) & (bp_TRs <= preceding_end_idx)
        )

        if has_preceding_bp:
            continue

        prearousal_segments.append((start_idx, end_idx))

    return prearousal_segments

# def get_sleep_onset_segments(sleep_onset_TRs, motion_TRs, num_TRs, TR=0.98, window_s=10.0):
#     """
#     Return a list of tuples (start_idx, end_idx) for valid sleep onset segments for this run.
#     A sleep onset segment is the `window_s` segment ending at the sleep onset TR (inclusive).
#     Exclude the segment if it contains any motion TRs.
#     """
#     N_inside = int(window_s / TR) + 1

#     sleep_onset_TRs = np.asarray(sleep_onset_TRs, dtype=int)
#     motion_TRs = np.asarray(motion_TRs, dtype=int)

#     sleep_onset_segments = []

#     for sleep_onset_idx in sleep_onset_TRs:
#         end_idx = sleep_onset_idx
#         start_idx = end_idx - N_inside + 1

#         if start_idx < 0:
#             continue

#         if end_idx >= num_TRs:
#             continue

#         segment_has_motion = np.any(
#             (motion_TRs >= start_idx) & (motion_TRs <= end_idx)
#         )

#         if segment_has_motion:
#             continue

#         sleep_onset_segments.append((start_idx, end_idx))

#     return sleep_onset_segments

def get_sleep_onset_segments(
    sleep_onset_TRs,
    motion_TRs,
    num_TRs,
    TR=0.98,
    before_s=10.0,
    after_s=10.0
):
    """
    Return valid wake-to-sleep transition segments.

    Segment definition:
      [10 s before sleep onset, 10 s after sleep onset]

    The sleep onset TR is treated as the transition event.
    """

    N_before = int(before_s / TR)
    N_after = int(after_s / TR)

    sleep_onset_TRs = np.asarray(sleep_onset_TRs, dtype=int)
    motion_TRs = np.asarray(motion_TRs, dtype=int)

    sleep_onset_segments = []

    for sleep_onset_idx in sleep_onset_TRs:
        start_idx = sleep_onset_idx - N_before
        end_idx = sleep_onset_idx + N_after

        if start_idx < 0:
            continue

        if end_idx >= num_TRs:
            continue

        segment_has_motion = np.any(
            (motion_TRs >= start_idx) & (motion_TRs <= end_idx)
        )

        if segment_has_motion:
            continue

        sleep_onset_segments.append((start_idx, end_idx))

    return sleep_onset_segments

def get_stable_wakes(bp_TRs, motion_TRs, num_TRs, TR=0.98, window_s=10.0, surrounding_window=10.0, min_bps_inside_main=5, min_bps_inside_surrounding=5):
    """
    Stable wake = non-overlapping `window_s` segments where:
    1. there is at least `min_bps_inside_surrounding` bps in the `surrounding_window` before the segment
    2. there is at least `min_bps_inside_main` bps inside the segment
    3. there is at least `min_bps_inside_surrounding` bps in the `surrounding_window` after the segment
    4. there is no motion TR inside the `window_s` segment
    Return an array of tuples (start_idx, end_idx) for the start and end TR indices of each stable wake segment.
    """
    N_inside = int(window_s / TR) + 1
    N_surrounding = int(surrounding_window / TR)
    bp_TRs = np.asarray(bp_TRs, dtype=int)
    motion_TRs = np.asarray(motion_TRs, dtype=int)

    wake_segments = []
    last_selected_end = -1

    for start_idx in range(0, num_TRs - N_inside + 1):
        end_idx = start_idx + N_inside - 1

        if start_idx <= last_selected_end:
            continue

        before_start = start_idx - N_surrounding
        before_end = start_idx - 1
        after_start = end_idx + 1
        after_end = end_idx + N_surrounding

        # require full 10s before and after
        if before_start < 0 or after_end >= num_TRs:
            continue

        n_bps_before = np.sum((bp_TRs >= before_start) & (bp_TRs <= before_end))
        n_bps_inside = np.sum((bp_TRs >= start_idx) & (bp_TRs <= end_idx))
        n_bps_after = np.sum((bp_TRs >= after_start) & (bp_TRs <= after_end))

        has_motion = np.any((motion_TRs >= start_idx) & (motion_TRs <= end_idx))

        if has_motion:
            continue

        if n_bps_before < min_bps_inside_surrounding:
            continue

        if n_bps_after < min_bps_inside_surrounding:
            continue
        
        if n_bps_inside < min_bps_inside_main:
            continue

        wake_segments.append((start_idx, end_idx))
        last_selected_end = end_idx

    return wake_segments

def get_stable_sleeps(bp_TRs, motion_TRs, num_TRs, TR=0.98, window_s=10.0, no_bps=10.0):
    """
    Stable sleep = non-overlapping `window_s` segments with:
    1. no motion TRs inside the `window_s` segment
    2. no bps within `no_bps` seconds before or after the segment

    Return an array of tuples (start_idx, end_idx) for the start and end TR indices of each stable sleep segment.
    """
    N_inside = int(window_s / TR) + 1
    N_buffer = int(no_bps / TR)

    bp_TRs = np.asarray(bp_TRs, dtype=int)
    motion_TRs = np.asarray(motion_TRs, dtype=int)

    sleep_segments = []
    last_selected_end = -1

    for start_idx in range(0, num_TRs - N_inside + 1):
        end_idx = start_idx + N_inside - 1

        if start_idx <= last_selected_end:
            continue

        check_start = start_idx - N_buffer
        check_end = end_idx + N_buffer

        # require full no-bp buffer before and after
        if check_start < 0 or check_end >= num_TRs:
            continue

        has_motion = np.any((motion_TRs >= start_idx) & (motion_TRs <= end_idx))
        if has_motion:
            continue

        has_bp_nearby = np.any((bp_TRs >= check_start) & (bp_TRs <= check_end))
        if has_bp_nearby:
            continue

        sleep_segments.append((start_idx, end_idx))
        last_selected_end = end_idx

    return sleep_segments

from collections import defaultdict

def count_overlap(prearousal_segments, sleep_segments, wake_segments, sleep_onset_segments):
    segment_sets = {
        'prearousal': set(prearousal_segments),
        'sleep': set(sleep_segments),
        'wake': set(wake_segments),
        'sleep_onset': set(sleep_onset_segments),
    }

    tuple_to_groups = defaultdict(list)

    for name, segments in segment_sets.items():
        for seg in segments:
            tuple_to_groups[seg].append(name)

    overlap_counts = defaultdict(int)

    for seg, groups in tuple_to_groups.items():
        if len(groups) > 1:
            overlap_counts[tuple(sorted(groups))] += 1

    return dict(overlap_counts)

def get_all_segments(my_dir):
    """
    Returns a dataframe of gap, participant, run, list of tuples (start_idx, end_idx) for valid prearousal segments, stable wake segments, and stable sleep segments.
    """
    all_segments = []
    for participant in sorted(os.listdir(my_dir)):
        for run in sorted(os.listdir(os.path.join(my_dir, participant))):
            arousal_TRs = np.load(os.path.join(my_dir, participant, run, 'arousal_TRs.npy'))
            sleep_onset_TRs = np.load(os.path.join(my_dir, participant, run, 'sleep_onset_TRs.npy'))
            motion_TRs = np.load(os.path.join(my_dir, participant, run, 'motion_TRs.npy'))
            bp_TRs = np.load(os.path.join(my_dir, participant, run, 'bp_TRs.npy'))
            time_vector = np.load(os.path.join(my_dir, participant, run, 'time_vector.npy'))
            nTRs = len(time_vector)
            # prearousal_segments = get_prearousal_segments(arousal_TRs, bp_TRs, motion_TRs, window_s=20.0, no_bps=10.0)
            prearousal_segments = get_prearousal_segments(
                arousal_TRs,
                bp_TRs,
                motion_TRs,
                nTRs,
                TR=0.98,
                before_s=10.0,
                after_s=10.0,
                no_bps_before_s=10.0
            )
            sleep_segments = get_stable_sleeps(bp_TRs, motion_TRs, nTRs, window_s=20.0, no_bps=5.0)
            wake_segments = get_stable_wakes(bp_TRs, motion_TRs, nTRs, window_s=20.0, surrounding_window=5.0, min_bps_inside_main=10, min_bps_inside_surrounding=3)
            # sleep_onset_segments = get_sleep_onset_segments(sleep_onset_TRs, motion_TRs, nTRs, window_s=20.0)
            sleep_onset_segments = get_sleep_onset_segments(
                sleep_onset_TRs,
                motion_TRs,
                nTRs,
                TR=0.98,
                before_s=10.0,
                after_s=10.0
            )
            counts = count_overlap(prearousal_segments, sleep_segments, wake_segments, sleep_onset_segments)
            if counts:
                print(f'{participant} {run} has overlapping segments: {counts}')
            all_segments.append({
                    'participant': participant,
                    'run': run,
                    'prearousal_segments': prearousal_segments,
                    'sleep_segments': sleep_segments,
                    'wake_segments': wake_segments,
                    'sleep_onset_segments': sleep_onset_segments
            })
    return pd.DataFrame(all_segments)

def save_segment_counts(all_segments, save_dir):
    all_segments['prearousal_segments'] = all_segments['prearousal_segments'].apply(len)
    all_segments['sleep_segments'] = all_segments['sleep_segments'].apply(len)
    all_segments['wake_segments'] = all_segments['wake_segments'].apply(len)
    all_segments['sleep_onset_segments'] = all_segments['sleep_onset_segments'].apply(len)
    all_segments.to_csv(os.path.join(save_dir, 'all_segments.csv'), index=False)

def plot_segments_by_participant(save_dir):
    all_segments = pd.read_csv(os.path.join(save_dir, 'all_segments.csv'))

    all_segments['participant'] = (all_segments['participant'].str.replace('_b$', '', regex=True))

    segment_cols = [
        'prearousal_segments',
        'sleep_segments',
        'wake_segments',
        'sleep_onset_segments'
    ]

    # Aggregate across runs
    participant_df = (
        all_segments
        .groupby('participant', as_index=False)[segment_cols]
        .sum()
    )

    # Order participants by total pre-arousal segments
    participant_df = participant_df.sort_values(
        'prearousal_segments',
        ascending=False
    ).reset_index(drop=True)

    x = np.arange(len(participant_df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(28, 8))

    colors = [
        'red',      # prearousal
        'green',    # sleep
        'blue',     # wake
        'purple'    # sleep onset
    ]

    bars1 = ax.bar(
        x - 1.5 * width,
        participant_df['prearousal_segments'],
        width,
        label='Pre-arousal',
        color=colors[0]
    )

    bars2 = ax.bar(
        x - 0.5 * width,
        participant_df['sleep_segments'],
        width,
        label='Stable sleep',
        color=colors[1]
    )

    bars3 = ax.bar(
        x + 0.5 * width,
        participant_df['wake_segments'],
        width,
        label='Stable wake',
        color=colors[2]
    )

    bars4 = ax.bar(
        x + 1.5 * width,
        participant_df['sleep_onset_segments'],
        width,
        label='Sleep onset',
        color=colors[3]
    )

    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{int(height)}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
                


    labels = [
        p.replace('AAN_', '')
        for p in participant_df['participant']
    ]
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)

    ax.set_xlabel(
        'Participant ID',
        fontsize=20
    )

    ax.set_ylabel(
        '# Segments',
        fontsize=20
    )

    ax.set_title(
        'Segment Counts by Participant',
        fontsize=24,
        pad=20
    )

    ax.tick_params(axis='y', labelsize=16)

    ax.legend(
        fontsize=16,
        title='Segment Type',
        title_fontsize=18,
        loc='upper right'
    )

    ax.grid(axis='y', alpha=0.3)

    ax.set_axisbelow(True)

    ax.grid(
        axis='y',
        linestyle='--',
        linewidth=0.8,
        color='lightgray',
        alpha=0.8
    )

    for xline in np.arange(-0.5, len(participant_df) + 0.5, 1):
        ax.axvline(
            xline,
            color='lightgray',
            linestyle='--',
            linewidth=1.0,
            alpha=0.8,
            zorder=0
        )
    for i in range(len(participant_df)):
        if i % 2 == 0:
            ax.axvspan(
                i - 0.5,
                i + 0.5,
                color='lightgray',
                alpha=0.15,
                zorder=0
            )
    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, 'segment_counts_by_participant.png'),
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


def collect_all_roi_segments(segment_df, my_dir, save_dir):
    segment_cols = {
        'prearousal': 'prearousal_segments',
        'sleep': 'sleep_segments',
        'wake': 'wake_segments',
        'sleep_onset': 'sleep_onset_segments'
    }

    data = {}

    # track max existing run number for each non-_b participant
    max_run_num = {}

    for participant in sorted(segment_df['participant'].unique()):
        participant_base = participant.removesuffix('_b')
        participant_segments = segment_df[segment_df['participant'] == participant]

        if participant_base not in data:
            data[participant_base] = {}

        if participant_base not in max_run_num:
            existing_runs = (
                segment_df[
                    segment_df['participant'] == participant_base
                ]['run']
                .astype(str)
                .str.extract(r'(\d+)')[0]
                .dropna()
                .astype(int)
            )
            max_run_num[participant_base] = existing_runs.max() if len(existing_runs) > 0 else 0

        for run in sorted(participant_segments['run'].unique()):
            run_str = str(run)

            if participant.endswith('_b'):
                max_run_num[participant_base] += 1
                save_run = f'run{max_run_num[participant_base]:02d}'
            else:
                save_run = run_str

            data[participant_base][save_run] = {}

            participant_run_segments = participant_segments[
                participant_segments['run'] == run
            ]

            for roi in all_rois:
                data[participant_base][save_run][roi] = {
                    'prearousal': [],
                    'sleep': [],
                    'wake': [],
                    'sleep_onset': []
                }

                if roi in all_cortical_rois:
                    roi_data = np.load(
                        os.path.join(my_dir, participant, run, 'cortex', f'{roi}.npy')
                    )
                elif roi in all_thalamic_rois:
                    roi_data = np.load(
                        os.path.join(my_dir, participant, run, 'thalamus', f'{roi}.npy')
                    )
                else:
                    roi_data = np.load(
                        os.path.join(my_dir, participant, run, f'{roi}.npy')
                    )

                for save_name, col in segment_cols.items():
                    for _, row in participant_run_segments.iterrows():
                        for start_idx, end_idx in row[col]:
                            segment_data = roi_data[start_idx:end_idx + 1]
                            data[participant_base][save_run][roi][save_name].append(
                                segment_data
                            )

                for save_name in segment_cols.keys():
                    segments = data[participant_base][save_run][roi][save_name]

                    if len(segments) > 0:
                        data[participant_base][save_run][roi][save_name] = np.stack(
                            segments
                        )
                    else:
                        data[participant_base][save_run][roi][save_name] = np.array([])

    return data                    

def save_training_data(my_dir, save_dir):
    segment_df = get_all_segments(my_dir)
    data = collect_all_roi_segments(segment_df, my_dir, save_dir)
    save_path = os.path.join(save_dir, 'all_training_data3.joblib')
    joblib.dump(data, save_path)
    print(f'saved to {save_path}')

# segment_df = get_all_segments(my_dir)
# print(segment_df['sleep_onset_segments'].values)
# save_segment_counts(segment_df, save_dir)
# plot_segments_by_participant(save_dir)
save_training_data(my_dir, save_dir)