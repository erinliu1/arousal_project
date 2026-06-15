from scipy.io import loadmat
import numpy as np
import os
import re

nicks_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/nicks_data'
my_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/data'

def read_mat_file(file_path):
    data = loadmat(file_path)
    return data

def load_behav_data(data):
    """
    Extract behavioral data from the loaded .mat file.
    Returns:
        event_codes: numpy array of event codes
        timestamps: numpy array of corresponding timestamps
    """
    behav_data = data['clicks'] # shape should be (2, N) or (N, 2) --> we want (2, N)
    if behav_data.shape[1] == 2:
        behav_data = behav_data.T
    event_codes = behav_data[0]     # 22 for TRs, 0 ignore, other numbers for button presses
    timestamps = behav_data[1]      # the corresponding timestamps in seconds
    return event_codes, timestamps

def load_bps_mat(participant, run, data_dir=nicks_dir):
    """
    Return a list of times (in seconds, relative to the first TR timestamp) when button presses occurred.
    """
    if type(run) == str:
        run = int(re.match(r'run(\d+)', run).group(1))
    file_path = f'{data_dir}/{participant}/run{run:02d}/bp.mat'
    data = read_mat_file(file_path)
    event_codes, timestamps = load_behav_data(data)

    t0 = timestamps[0]                              # the very first timestamp
    tr_times = timestamps[event_codes == 22] - t0   # timestamps of all TR's relative to t0
    MRstart = tr_times[0]                           # timestamp of first TR relative to t0

    # times of button presses relative to MRstart 
    press_times = timestamps[(event_codes != 22) & (event_codes != 0)] - t0
    bps = press_times[press_times >= MRstart] - MRstart
    return bps

def read_txt_file(file_path):
    with open(file_path, "r") as f:
        raw = f.read()
    raw_no_whitespace = ''.join(raw.split())
    tokens = np.array(list(raw_no_whitespace))
    return tokens

def assign_nearest_bp_timestamps(trs_indices, bp_indices, TR):
    """
    Helper function for load_bps_txt. 
    - For each bp, count the # of '=' before it. This represents which TR number you were at when the bp happened
    - e.g. if the bp occured after the 37th TR (but before the 38th TR), then pretend it happened exactly on the 37th TR
    - Let the 1st TR time be at timestamp 0
    - Assign the timestamp of the i'th TR as (i-1) * TR seconds.
    """
    bp_timestamps = []
    for bp_idx in bp_indices:
        num_trs_before = np.sum(trs_indices < bp_idx)
        bp_timestamps.append((num_trs_before - 1) * TR)
    return np.array(bp_timestamps)

def space_out_bp_timestamps(trs_indices, bp_indices, TR):
    """
    Helper function for load_bps_txt.
    - For each TR interval (i.e. between consecutive TRs), find all bp's that occur within that interval
    - then space them out evenly within the timestamps of that interval
    """
    bp_timestamps = []
    for i in range(len(trs_indices) - 1):
        tr_idx_start = trs_indices[i]
        tr_idx_end = trs_indices[i + 1]
        
        # find all bp's that occur between these two TRs (inclusive of start, exclusive of end)
        bps_in_interval = bp_indices[(bp_indices >= tr_idx_start) & (bp_indices < tr_idx_end)]

        tr_time_start = i * TR # timestamp of the start TR
        tr_time_end = (i+1) * TR # timestamp of the end TR

        num_bps = len(bps_in_interval)

        if num_bps == 1:
            # If there's only one bp, place it in the middle timestamp of the interval
            midpoint_time = (tr_time_start + tr_time_end) / 2
            bp_timestamps.append(midpoint_time)

        elif num_bps > 1:
            # If there are multiple bp's, space them evenly within the interval
            spaced_times = np.linspace(tr_time_start, tr_time_end, num_bps + 2)[1:-1] # exclude the start and end points
            bp_timestamps.extend(spaced_times.tolist())

    return np.array(bp_timestamps)

def load_bps_txt(participant, run, data_dir=nicks_dir, method="space_out", TR=0.98):
    """
    Return a list of timestamps (in seconds, relative to the first TR timestamp) when button presses occurred.
    """
    if type(run) == str:
        run = int(re.match(r'run(\d+)', run).group(1))
    
    file_path = f'{data_dir}/{participant}/run{run:02d}/bp'
    tokens = read_txt_file(file_path) # array of characters, e.g. ['=', '=', '1', '=', '2', '3', '=', '=']

    trs_indices = np.where(tokens == '=')[0] # indices where TRs occur
    bp_indices = np.where(np.char.isdigit(tokens))[0] # indices where button presses occur

    if method == "assign_nearest":
        bp_timestamps = assign_nearest_bp_timestamps(trs_indices, bp_indices, TR)
    elif method == "space_out":
        bp_timestamps = space_out_bp_timestamps(trs_indices, bp_indices, TR)
    else:
        raise ValueError("Invalid method. Choose either 'assign_nearest' or 'space_out'.")
        
    # remove bp timestamps that occur before the first TR or after the last TR
    first_tr_time = 0
    last_tr_time = (len(trs_indices) - 1) * TR
    bp_timestamps = bp_timestamps[(bp_timestamps >= first_tr_time) & (bp_timestamps <= last_tr_time)]

    return bp_timestamps

def load_bps(participant, run, data_dir=nicks_dir):
    """
    Wrapper function to load button press timestamps. Use .mat for all subjects except AAN_08/run02 and AAN_20/run01, which uses .txt
    """
    if participant == 'AAN_08' and run == 'run02':
        bps = load_bps_txt(participant, run, data_dir=data_dir, method="space_out", TR=0.98)
    elif participant == 'AAN_20' and run == 'run01':
        bps = load_bps_txt(participant, run, data_dir=data_dir, method="space_out", TR=0.98)
    else:
        bps = load_bps_mat(participant, run, data_dir=data_dir)
    return bps

def load_arousals(bps, seconds_between_bps=30):
    """
    Return a list of times (in seconds, relative to the first TR timestamp) when arousals occured
    Arousals are defined as the timestamp of the 1st button press after at least `seconds_between_bps` of no button presses.
    The first button press is not considered arousal; we only consider a button press to be an arousal if it is preceded by a gap of at least `seconds_between_bps` since the previous button press.
    """
    interclick_intervals = np.diff(bps)
    arousal_indices = np.where(interclick_intervals >= seconds_between_bps)[0] + 1 # +1 to get the index of the button press after the long gap
    arousal_times = bps[arousal_indices]
    return arousal_times

def load_sleep_onsets(bps, seconds_between_bps=30):
    """
    Return a list of times (in seconds, relative to the first TR timestamp) when sleep onset events occurred.

    Sleep onsets are defined as the timestamp of the last button press before at least `seconds_between_bps` seconds of no button presses.

    For a gap between bps[i] and bps[i+1] that is >= seconds_between_bps, the sleep onset is taken to be bps[i].
    """
    interclick_intervals = np.diff(bps)
    sleep_onset_indices = np.where(interclick_intervals >= seconds_between_bps)[0]
    sleep_onset_times = bps[sleep_onset_indices]
    return sleep_onset_times

def load_nearest_preceding_TR(list_of_timestamps, TR=0.98):
    """
    Return a list of TR indices corresponding to the nearest TR timestamp preceding each timestamp in the input list.
    """
    TRs = np.floor(list_of_timestamps / TR).astype(int)
    return TRs

def load_nearest_following_TR(list_of_timestamps, TR=0.98):
    """
    Return a list of TR indices corresponding to the nearest TR timestamp
    following each timestamp in the input list.
    """
    TRs = np.ceil(list_of_timestamps / TR).astype(int)
    return TRs

def load_no_bp_intervals(bps, seconds_between_bps=30):
    """
    Return intervals of no button presses lasting at least `seconds_between_bps`.

    Sleep onset = timestamp of last BP before long no-BP gap
    Arousal = timestamp of first BP after long no-BP gap

    Returns:
    - intervals: list of tuples (sleep_onset_time, arousal_time)
    """
    bps = np.asarray(bps)

    interclick_intervals = np.diff(bps)

    long_gap_indices = np.where(interclick_intervals >= seconds_between_bps)[0]

    sleep_onset_times = bps[long_gap_indices]
    arousal_times = bps[long_gap_indices + 1]

    intervals = list(zip(sleep_onset_times, arousal_times))

    return intervals

# def main(nicks_dir=nicks_dir, my_dir=my_dir):
#     num_sleeps = 0
#     for participant in sorted(os.listdir(nicks_dir)):
#         for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
#             bps = load_bps(participant, run, data_dir=nicks_dir)
#             sleep_intervals = load_no_bp_intervals(bps, seconds_between_bps=30)
#             length_of_intervals = [arousal - sleep_onset for sleep_onset, arousal in sleep_intervals]
#             num_sleeps += len(sleep_intervals)
#     print(f"Total number of sleeps across all participants and runs: {num_sleeps}")
            # arousals = load_arousals(bps, seconds_between_bps=30) # update May 27: using 30 seconds as the threshold for defining arousals, which is more conservative than 20 seconds which was used previously
            # sleep_onsets = load_sleep_onsets(bps, seconds_between_bps=30)
            # sleep_onset_TRs = load_nearest_following_TR(sleep_onsets)
            # print(len(bps), len(arousals), len(sleep_onsets))
            # arousal_TRs = load_nearest_preceding_TR(arousals)
            # bp_TRs = load_nearest_preceding_TR(bps)
            
            # Save bps and arousals to .npy files
            # np.save(os.path.join(my_dir, participant, run, 'bps.npy'), bps)
            # np.save(os.path.join(my_dir, participant, run, 'arousals.npy'), arousals)
            # np.save(os.path.join(my_dir, participant, run, 'arousal_TRs.npy'), arousal_TRs)
            # np.save(os.path.join(my_dir, participant, run, 'bp_TRs.npy'), bp_TRs)
            # np.save(os.path.join(my_dir, participant, run, 'sleep_onsets.npy'), sleep_onsets)
            # np.save(os.path.join(my_dir, participant, run, 'sleep_onset_TRs.npy'), sleep_onset_TRs)

import matplotlib.pyplot as plt
from collections import defaultdict

def base_participant_name(participant):
    return re.sub(r'_b$', '', participant)

def run_sort_key(run):
    nums = re.findall(r'\d+', run)
    return int(nums[-1]) if nums else run

def participant_number(participant):
    return str(int(re.search(r"(\d+)", participant).group(1)))

def plot_sleep_interval_lengths_by_participant(nicks_dir):
    data = defaultdict(list)

    # collect all sleep interval lengths
    for participant in sorted(os.listdir(nicks_dir)):
        participant_dir = os.path.join(nicks_dir, participant)

        if not os.path.isdir(participant_dir):
            continue

        base_participant = base_participant_name(participant)
        repeat_order = 1 if participant.endswith('_b') else 0

        for run in sorted(os.listdir(participant_dir), key=run_sort_key):
            run_dir = os.path.join(participant_dir, run)

            if not os.path.isdir(run_dir):
                continue

            bps = load_bps(participant, run, data_dir=nicks_dir)
            sleep_intervals = load_no_bp_intervals(
                bps,
                seconds_between_bps=30
            )

            if len(sleep_intervals) == 0:
                continue

            lengths = [
                arousal - sleep_onset
                for sleep_onset, arousal in sleep_intervals
            ]

            data[base_participant].append({
                "participant_raw": participant,
                "repeat_order": repeat_order,
                "run": run,
                "lengths": lengths,
                "count": len(lengths)
            })

    # remove participants with no sleep segments
    data = {
        participant: runs
        for participant, runs in data.items()
        if len(runs) > 0
    }

    # sort runs: original first, then _b, runs in order
    for participant in data:
        data[participant] = sorted(
            data[participant],
            key=lambda x: (
                x["repeat_order"],
                run_sort_key(x["run"])
            )
        )

    # same binning/order idea: participants sorted by total number of segments
    participant_order = sorted(
        data.keys(),
        key=lambda p: sum(r["count"] for r in data[p]),
        reverse=True
    )

    cmap = plt.get_cmap("tab20")
    color_map = {
        participant: cmap(i % 20)
        for i, participant in enumerate(participant_order)
    }

    x_values = []
    y_values = []
    colors = []

    participant_centers = []
    separator_positions = []

    x = 0
    gap = 1.0

    rng = np.random.default_rng(42)

    for participant in participant_order:
        start_x = x

        for run_info in data[participant]:
            for length in run_info["lengths"]:
                x_values.append(x + rng.uniform(-0.25, 0.25))
                y_values.append(length)
                colors.append(color_map[participant])

            x += 1

        end_x = x - 1
        participant_centers.append((start_x + end_x) / 2)

        separator_positions.append(
            x - 0.5 + gap / 2
        )

        x += gap

    fig, ax = plt.subplots(figsize=(15, 7.5))

    ax.scatter(
        x_values,
        y_values,
        c=colors,
        s=35,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.3
    )

    for sep in separator_positions[:-1]:
        ax.axvline(
            sep,
            color="lightgray",
            linewidth=2.5,
            zorder=0
        )

    max_y = max(y_values)

    ax.set_xticks([])

    for center, participant in zip(participant_centers, participant_order):
        ax.text(
            center,
            -max_y * 0.08,
            participant_number(participant),
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            clip_on=False
        )

    ax.set_ylabel(
        "Sleep interval length (seconds)",
        fontsize=16
    )

    ax.set_xlabel(
        "Participant",
        fontsize=16
    )

    ax.set_title(
        "Lengths of Sleep Intervals by Participant and Run",
        fontsize=20,
        fontweight="bold"
    )

    ax.tick_params(axis="y", labelsize=13)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(
        0,
        max_y * 1.12
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.18
    )
    plt.savefig('sleep_interval_lengths_by_participant.png', dpi=300)

def plot_sleep_segments_by_participant_run(nicks_dir):
    counts = defaultdict(list)
    for participant in sorted(os.listdir(nicks_dir)):
        participant_dir = os.path.join(nicks_dir, participant)
        if not os.path.isdir(participant_dir):
            continue
        base_participant = base_participant_name(participant)
        repeat_order = 1 if participant.endswith('_b') else 0
        for run in sorted(os.listdir(participant_dir), key=run_sort_key):
            run_dir = os.path.join(participant_dir, run)
            if not os.path.isdir(run_dir):
                continue
            bps = load_bps(participant, run, data_dir=nicks_dir)
            sleep_intervals = load_no_bp_intervals(bps, seconds_between_bps=30)
            n_sleep = len(sleep_intervals)
            if n_sleep == 0:
                continue
            counts[base_participant].append({
                "participant_raw": participant,
                "repeat_order": repeat_order,
                "run": run,
                "count": n_sleep
            })
    counts = {participant: runs for participant, runs in counts.items() if len(runs) > 0}
    for participant in counts:
        counts[participant] = sorted(counts[participant], key=lambda x: (x["repeat_order"], run_sort_key(x["run"])))
    participant_order = sorted(counts.keys(), key=lambda p: sum(r["count"] for r in counts[p]), reverse=True)
    x_positions = []
    heights = []
    colors = []
    participant_centers = []
    separator_positions = []
    cmap = plt.get_cmap("tab20")
    color_map = {participant: cmap(i % 20) for i, participant in enumerate(participant_order)}
    x = 0
    gap = 1.0
    for participant in participant_order:
        start_x = x
        for run_info in counts[participant]:
            x_positions.append(x)
            heights.append(run_info["count"])
            colors.append(color_map[participant])
            x += 1
        end_x = x - 1
        participant_centers.append((start_x + end_x) / 2)
        separator_positions.append(x - 0.5 + gap / 2)
        x += gap
    fig, ax = plt.subplots(figsize=(15, 7.5))
    bars = ax.bar(x_positions, heights, color=colors, edgecolor="black", linewidth=0.6)
    max_height = max(heights)
    for bar, count in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_height * 0.015, str(count), ha="center", va="bottom", fontsize=8)
    for sep in separator_positions[:-1]:
        ax.axvline(sep, color="lightgray", linewidth=2.5, zorder=0)
    ax.set_xticks([])
    for center, participant in zip(participant_centers, participant_order):
        participant_num = int(re.search(r"(\d+)", participant).group(1))
        ax.text(center, -max_height * 0.08, str(participant_num), ha="center", va="top", fontsize=13, fontweight="bold", clip_on=False)
    ax.set_ylabel("Number of Sleep Segments", fontsize=16)
    ax.set_xlabel("Participant", fontsize=16)
    ax.set_title("Sleep Segments by Participant and Run", fontsize=20, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max_height * 1.15)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.18)
    plt.savefig('sleep_segments_by_participant_run.png', dpi=300)

# usage
plot_sleep_interval_lengths_by_participant(nicks_dir)