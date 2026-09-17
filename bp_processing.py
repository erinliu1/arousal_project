from scipy.io import loadmat
import numpy as np
import os
import re

nicks_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/nicks_data'
my_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/raw_data'

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
            # If there are multiple bp's, space them evenly within the timestamps of that interval
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

def load_nearest_preceding_TR(list_of_timestamps, TR=0.98):
    """
    Return a list of TR indices corresponding to the nearest TR timestamp preceding each timestamp in the input list.
    """
    TRs = np.floor(list_of_timestamps / TR).astype(int)
    return TRs

def main(nicks_dir=nicks_dir, my_dir=my_dir):
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            bps = load_bps(participant, run, data_dir=nicks_dir)
            bp_TRs = load_nearest_preceding_TR(bps)

            save_dir = os.path.join(my_dir, participant, run)
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, 'bps.npy'), bps)
            np.save(os.path.join(save_dir, 'bp_TRs.npy'), bp_TRs)
            print(f'Processed {participant} {run}')

if __name__ == '__main__':
    main()
