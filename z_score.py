import os
import numpy as np
import matplotlib.pyplot as plt
from framewise_displacement import framewise_displacement, get_motion_TRs
from all_rois import all_rois
from roi_processing import filepath_lookup, load_txt_boi, roi_dict_lookup

"""
Need to compute the mean and std for each ROI timeseries to z-score.
- Exclude motion TRs and +/- k TRs around them
- Exclude first 20 TRs due to start-up noise
- Compute mean and std of the remaining TRs
- For choosing k, compute percent change in STD as k increases from 0 to 6
- Plot average median percent change in STD across ROIs vs k transition (see plots/std_v_k.png)
- Use that value of k to z-score the time-series of each ROI
"""

NICKS_DIR = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/nicks_data'
RAW_DATA_DIR = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/raw_data'

def print_high_motion_percentages(nicks_dir=NICKS_DIR):
    # Print the percentage of high-motion TRs for each run, sorted in descending order
    motion_TRs = []
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            fd = framewise_displacement(participant, run)
            excluded_TRs_motion = get_motion_TRs(fd, threshold=0.5, surrounding_TRs=0)
            percent_motion = len(excluded_TRs_motion) / (len(fd) + 1)
            motion_TRs.append((percent_motion, f'{participant}_{run}'))
    for percent, run_id in sorted(motion_TRs, reverse=True):
        print(f'{run_id}: {percent:.2%} high-motion TRs')

# print_high_motion_percentages()
# top 4 runs with the highest percentage of high-motion TRs:
# AAN_06_run01: 12.77% high-motion TRs
# AAN_18_run04: 12.46% high-motion TRs
# AAN_20_run01: 11.08% high-motion TRs
# AAN_10_run02: 11.03% high-motion TRs

def compute_mean_stds(participant, run, k, roi_type):
    # roi_type = 'all', 'cortex', or 'thalamus'

    fd = framewise_displacement(participant, run)
    excluded_TRs_motion = get_motion_TRs(fd, threshold=0.5, surrounding_TRs=k)
    excluded_TRs_start = np.arange(20)
    excluded_TRs = np.unique(np.concatenate((excluded_TRs_motion, excluded_TRs_start))) 

    mean_stds = {}
    for roi in roi_dict_lookup(roi_type):
        roi_path = os.path.join(filepath_lookup(nicks_dir, participant, run, roi_type), f'{roi}.txt')
        roi_data = load_txt_boi(roi_path)

        T0 = len(roi_data) # original number of TRs
        original_TRs = np.arange(T0)
        remaining_TRs = np.setdiff1d(original_TRs, excluded_TRs)

        mean_roi = np.mean(roi_data[remaining_TRs])
        std_roi = np.std(roi_data[remaining_TRs])

        mean_stds[roi] = (mean_roi, std_roi)
    
    return mean_stds

def compute_percent_change_in_std(participant, run, ks=[0,1,2,3,4,5,6]):
    delta_ks = {f'{ks[i]} → {ks[i+1]}': {roi: None for roi in all_rois} for i in range(len(ks)-1)}
    for k in ks:
        if k == 0:
            continue
        else:
            prev_stds = compute_mean_stds(participant, run, k-1)
            mean_stds = compute_mean_stds(participant, run, k)
            for roi in all_rois:
                curr_std = mean_stds[roi][1]
                prev_std = prev_stds[roi][1]
                percent_change = (curr_std - prev_std) / prev_std * 100
                delta_ks[f'{k-1} → {k}'][roi] = abs(float(percent_change))
    
    delta_ks_summary = {}
    for transition, roi_changes in delta_ks.items():
        changes = list(roi_changes.values())
        median_change = np.median(changes)
        percentile_90 = np.percentile(changes, 90)
        delta_ks_summary[transition] = (median_change, percentile_90)
    return delta_ks_summary

def plot_stds_vs_k(trials, ks=[0,1,2,3,4,5,6]):
    # trials is a list of (participant, run) tuples
    transitions = [f'{ks[i]} → {ks[i+1]}' for i in range(len(ks)-1)]
    all_changes = {transition: [] for transition in transitions}
    for participant, run in trials:
        delta_ks = compute_percent_change_in_std(participant, run, ks)
        for transition in transitions:
            all_changes[transition].append(delta_ks[transition][0])  # median change
    average_median_changes = []
    for transition, changes in all_changes.items():
        average_median_changes.append(np.mean(changes))
        print(f'k transition {transition}: {np.mean(changes):.2f}% average median percent change across ROIs and trials')
    plt.figure()
    plt.plot(transitions, average_median_changes, marker='o')
    plt.xlabel('k transition')
    plt.ylabel('Average Median |Δσ| (%)')
    plt.title('Average Median % STD Change vs k Transition')
    plt.savefig('plots/std_v_k.png')
    plt.close()

# plot_stds_vs_k([('AAN_06', 'run01'), ('AAN_18', 'run04'), ('AAN_20', 'run01'), ('AAN_10', 'run02')])
# choose k = 4

def zscore_and_save(participant, run, k=4, nicks_dir=NICKS_DIR, my_dir=RAW_DATA_DIR):
    for roi_type in ['cortex','thalamus']:
        mean_stds = compute_mean_stds(participant, run, k, roi_type)
        for roi in roi_dict_lookup(roi_type):
            roi_path = os.path.join(filepath_lookup(nicks_dir, participant, run, roi_type), f'{roi}.txt')
            roi_data = load_txt_boi(roi_path)
            mean_roi, std_roi = mean_stds[roi]
            zscored_data = (roi_data - mean_roi) / std_roi
            save_folder = filepath_lookup(my_dir, participant, run, roi_type)
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f'{roi}.npy')
            np.save(save_path, zscored_data)
    
def save_all(nicks_dir=NICKS_DIR, my_dir=RAW_DATA_DIR, k=4):
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            zscore_and_save(participant, run, k, nicks_dir, my_dir)

# if __name__ == "__main__":
#     save_all()
    
