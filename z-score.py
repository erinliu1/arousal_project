import os
import re
import numpy as np
import matplotlib.pyplot as plt
import shutil
from framewise_displacement import framewise_displacement, get_motion_TRs
from all_rois import all_rois, all_cortical_rois, all_thalamic_rois

"""
Preprocessing requires getting the mean and std time series for each ROI:
- Exclude TRs with framewise displacement > 0.5 mm and +/- k TRs around them
- Exclude first 20 TRs due to start-up noise
- Compute mean and STD
- For choosing k, compute percent change in STD as k increases from 0 to 6
- Plot average median percent change in STD across ROIs vs k transition (see plots/std_v_k.png)
- Use that value of k to z-score the time-series of each ROI
- Save the TRs excluded due to motion in seconds (will be used to exclude epochs from training later)
"""

nicks_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/nicks_data'
my_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/data'
os.makedirs(os.path.join(my_dir), exist_ok=True)

def print_high_motion_percentages(nicks_dir='/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/nicks_data'):
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
# AAN_06_run01: 12.77% high-motion TRs
# AAN_18_run04: 12.46% high-motion TRs
# AAN_20_run01: 11.08% high-motion TRs
# AAN_10_run02: 11.03% high-motion TRs

def what_are_you(x):
    try:
        return float(x)
    except:
        try:
            return float.fromhex(x)
        except:
            print(f'what are you??? {x}')
    
def get_valid_cortical_ROIs(nicks_dir):
    all_cortex_rois = set()
    never_add_me_again = {'unknown'}
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            cortex_folder = os.path.join(nicks_dir, participant, run, 'cortex')
            for roi_txt_filename in os.listdir(cortex_folder):
                roi = roi_txt_filename.removesuffix('.txt')
                if roi in never_add_me_again:
                    continue
                all_cortex_rois.add(roi)
                filepath = os.path.join(cortex_folder, roi_txt_filename)
                roi_data = np.loadtxt(filepath, converters=what_are_you)
                if np.all(roi_data == 0):
                    print(roi)
                    never_add_me_again.add(roi)
                    all_cortex_rois.remove(roi)
    all_cortex_rois = sorted(list(all_cortex_rois))
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            cortex_folder = os.path.join(nicks_dir, participant, run, 'cortex')
            rois_found = set()
            for roi_txt_filename in os.listdir(cortex_folder):
                roi = roi_txt_filename.removesuffix('.txt')
                if roi in all_cortex_rois:
                    rois_found.add(roi)
            missing_rois = set(all_cortex_rois) - rois_found
            if missing_rois:
                print(f'Participant: {participant}, Run: {run}, Missing Cortex Parcelations: {sorted(list(missing_rois))}')
    return all_cortex_rois

def get_valid_thalamic_ROIs(nicks_dir):
    all_thalamus_rois = set()
    never_add_me_again = set()
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            thalamus_folder = os.path.join(nicks_dir, participant, run, 'thalamus')
            for roi_txt_filename in os.listdir(thalamus_folder):
                roi = roi_txt_filename.removesuffix('.txt')
                if roi in never_add_me_again:
                    continue
                all_thalamus_rois.add(roi)
                filepath = os.path.join(thalamus_folder, roi_txt_filename)
                roi_data = np.loadtxt(filepath, converters=what_are_you)
                if np.all(roi_data == 0):
                    never_add_me_again.add(roi)
                    all_thalamus_rois.remove(roi)
    all_thalamus_rois = sorted(list(all_thalamus_rois))
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            thalamus_folder = os.path.join(nicks_dir, participant, run, 'thalamus')
            rois_found = set()
            for roi_txt_filename in os.listdir(thalamus_folder):
                roi = roi_txt_filename.removesuffix('.txt')
                if roi in all_thalamus_rois:
                    rois_found.add(roi)
            missing_rois = set(all_thalamus_rois) - rois_found
            if missing_rois:
                print(f'Participant: {participant}, Run: {run}, Missing Thalamus Parcelations: {sorted(list(missing_rois))}')
            all_thalamus_rois = set(all_thalamus_rois) - missing_rois
    return sorted(list(all_thalamus_rois))

def load_txt_boi(filepath):
    print(filepath)
    if 'cortex' in filepath or 'thalamus' in filepath:
        data = np.loadtxt(filepath, converters=what_are_you)
    else:
        data = np.loadtxt(filepath)
    return data

def roi_dict_lookup(roi_type):
    if roi_type == 'all':
        return all_rois
    elif roi_type == 'cortex':
        return all_cortical_rois
    elif roi_type == 'thalamus':
        return all_thalamic_rois
    else:
        raise ValueError(f'Invalid roi_type: {roi_type}')
def filepath_lookup(base_dir, participant, run, roi_type):
    # roi_type = 'all', 'cortex', or 'thalamus'
    if roi_type == 'all':
        return os.path.join(base_dir, participant, run)
    elif roi_type == 'cortex':
        return os.path.join(base_dir, participant, run, 'cortex')
    elif roi_type == 'thalamus':
        return os.path.join(base_dir, participant, run, 'thalamus')
    else:
        raise ValueError(f'Invalid roi_type: {roi_type}')

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

def save_motion_TRs(participant, run, my_dir=my_dir):
    fd = framewise_displacement(participant, run)
    excluded_TRs_motion = get_motion_TRs(fd, threshold=0.5, surrounding_TRs=0)
    excluded_TRs_start = np.arange(20)
    excluded_TRs = np.unique(np.concatenate((excluded_TRs_motion, excluded_TRs_start))) 

    save_path = os.path.join(my_dir, participant, run, 'motion_TRs.npy')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, excluded_TRs)

def save_all_motion_TRs(nicks_dir=nicks_dir, my_dir=my_dir):
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            save_motion_TRs(participant, run, my_dir)

def zscore_and_save(participant, run, k=4, nicks_dir=nicks_dir, my_dir=my_dir):
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
    
def save_all(nicks_dir=nicks_dir, my_dir=my_dir, k=4):
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            zscore_and_save(participant, run, k, nicks_dir, my_dir)

def save_time_vector(participant, run, TR=0.98, my_dir=my_dir):
    save_folder = os.path.join(my_dir, participant, run)
    all_roi_TR_lengths = []
    for roi in all_rois:
        roi_path = os.path.join(save_folder, f'{roi}.npy')
        if not os.path.exists(roi_path):
            print(f'ROI file {roi_path} does not exist for {participant} {run}.')
            continue
        roi_data = np.load(roi_path)
        all_roi_TR_lengths.append(len(roi_data))
    if len(set(all_roi_TR_lengths)) > 1:
        print(f'Warning: ROIs have different lengths for {participant} {run}:\n {dict(zip(all_rois, all_roi_TR_lengths))}')
        return 
    T = all_roi_TR_lengths[0]
    time_vector = np.arange(T) * TR
    save_path = os.path.join(save_folder, 'time_vector.npy')
    np.save(save_path, time_vector)
    
def save_time_vectors_for_all_runs(nicks_dir=nicks_dir, my_dir=my_dir, TR=0.98):
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            save_time_vector(participant, run, TR, my_dir)

if __name__ == "__main__":
    # save_all_motion_TRs()
    # save_all()
    # save_time_vectors_for_all_runs()
    
