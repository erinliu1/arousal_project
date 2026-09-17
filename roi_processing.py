"""Helpers for discovering, loading, and locating ROI time series."""

import os
import numpy as np
from all_rois import (
    all_rois,
    all_cortical_rois,
    all_thalamic_rois,
    get_all_rois_with_cortical_and_thalamic,
)

NICKS_DIR = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/nicks_data'
RAW_DATA_DIR = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/raw_data'

def what_are_you(x):
    if isinstance(x, bytes):
        x = x.decode()
    try:
        return float(x)
    except:
        try: 
            return float.fromhex(x)
        except:
            print(f'what are you??? {x}')
            return None

def load_roi_text(filepath):
    return np.loadtxt(filepath, converters={0: what_are_you})

def load_txt_boi(filepath):
    # some of the ROI text files under cortex or thalamus folders in /nicks_data have hex values instead of normal numbers; convert them to floats when loading
    if 'cortex' in filepath or 'thalamus' in filepath:
        data = load_roi_text(filepath)
    else:
        data = np.loadtxt(filepath)
    return data

def get_valid_cortical_ROIs(nicks_dir=NICKS_DIR):
    # Find all valid cortical rois, excluding any roi's called unknown and rois for which 1 or more runs have all-zero time series.
    all_cortex_rois = set()
    never_add_me_again = {'unknown'}
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            cortex_folder = os.path.join(nicks_dir, participant, run, 'cortex')
            for roi_txt_filename in os.listdir(cortex_folder):
                roi = os.path.splitext(roi_txt_filename)[0]
                if roi in never_add_me_again:
                    continue
                all_cortex_rois.add(roi)
                filepath = os.path.join(cortex_folder, roi_txt_filename)
                roi_data = load_roi_text(filepath)
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
                roi = os.path.splitext(roi_txt_filename)[0]
                if roi in all_cortex_rois:
                    rois_found.add(roi)
            missing_rois = set(all_cortex_rois) - rois_found
            if missing_rois:
                print(
                    f'Participant: {participant}, Run: {run}, '
                    f'Missing Cortex Parcelations: {sorted(list(missing_rois))}'
                )
    return all_cortex_rois

def get_valid_thalamic_ROIs(nicks_dir=NICKS_DIR):
    # Find all valid thalamic rois, excluding any roi's called unknown and rois for which 1 or more runs have all-zero time series.
    all_thalamus_rois = set()
    never_add_me_again = set()
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            thalamus_folder = os.path.join(nicks_dir, participant, run, 'thalamus')
            for roi_txt_filename in os.listdir(thalamus_folder):
                roi = os.path.splitext(roi_txt_filename)[0]
                if roi in never_add_me_again:
                    continue
                all_thalamus_rois.add(roi)
                filepath = os.path.join(thalamus_folder, roi_txt_filename)
                roi_data = load_roi_text(filepath)
                if np.all(roi_data == 0):
                    never_add_me_again.add(roi)
                    all_thalamus_rois.remove(roi)
    all_thalamus_rois = sorted(list(all_thalamus_rois))
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            thalamus_folder = os.path.join(nicks_dir, participant, run, 'thalamus')
            rois_found = set()
            for roi_txt_filename in os.listdir(thalamus_folder):
                roi = os.path.splitext(roi_txt_filename)[0]
                if roi in all_thalamus_rois:
                    rois_found.add(roi)
            missing_rois = set(all_thalamus_rois) - rois_found
            if missing_rois:
                print(
                    f'Participant: {participant}, Run: {run}, '
                    f'Missing Thalamus Parcelations: {sorted(list(missing_rois))}'
                )
            all_thalamus_rois = set(all_thalamus_rois) - missing_rois
    return sorted(list(all_thalamus_rois))

def roi_dict_lookup(roi_type):
    if roi_type == 'all':
        return all_rois
    if roi_type == 'cortex':
        return all_cortical_rois
    if roi_type == 'thalamus':
        return all_thalamic_rois
    raise ValueError(f'Invalid roi_type: {roi_type}')

def filepath_lookup(nicks_dir, participant, run, roi_type):
    # Return the folder path under /nicks_data containing the specified category of roi time series.
    if roi_type == 'all':
        return os.path.join(nicks_dir, participant, run)
    if roi_type == 'cortex':
        return os.path.join(nicks_dir, participant, run, 'cortex')
    if roi_type == 'thalamus':
        return os.path.join(nicks_dir, participant, run, 'thalamus')
    raise ValueError(f'Invalid roi_type: {roi_type}')

def save_rois(participant, run, nicks_dir=NICKS_DIR, raw_data_dir=RAW_DATA_DIR):
    save_folder = os.path.join(raw_data_dir, participant, run, 'rois')
    os.makedirs(save_folder, exist_ok=True)

    for roi_type in ['all', 'cortex', 'thalamus']:
        for roi in roi_dict_lookup(roi_type):
            source_roi = 'ostralmiddlefrontal' if roi == 'rostralmiddlefrontal' else roi
            roi_path = os.path.join(
                filepath_lookup(nicks_dir, participant, run, roi_type),
                f'{source_roi}.txt',
            )
            roi_data = load_txt_boi(roi_path)
            np.save(os.path.join(save_folder, f'{roi}.npy'), roi_data)

def save_time_vector(participant, run, TR=0.98, raw_data_dir=RAW_DATA_DIR):
    save_folder = os.path.join(raw_data_dir, participant, run)
    roi_folder = os.path.join(save_folder, 'rois')
    all_roi_names = get_all_rois_with_cortical_and_thalamic()
    all_roi_TR_lengths = []
    for roi in all_roi_names:
        roi_path = os.path.join(roi_folder, f'{roi}.npy')
        if not os.path.exists(roi_path):
            print(f'ROI file {roi_path} does not exist for {participant} {run}.')
            continue
        roi_data = np.load(roi_path)
        all_roi_TR_lengths.append(len(roi_data))
    if len(set(all_roi_TR_lengths)) > 1:
        print(
            f'Warning: ROIs have different lengths for {participant} {run}:\n '
            f'{dict(zip(all_roi_names, all_roi_TR_lengths))}'
        )
        return
    T = all_roi_TR_lengths[0]
    time_vector = np.arange(T) * TR
    save_path = os.path.join(save_folder, 'time_vector.npy')
    np.save(save_path, time_vector)

def save_all_rois(nicks_dir=NICKS_DIR, raw_data_dir=RAW_DATA_DIR):
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            save_rois(participant, run, nicks_dir, raw_data_dir)

def save_time_vectors_for_all_runs(nicks_dir=NICKS_DIR, raw_data_dir=RAW_DATA_DIR, TR=0.98):
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            save_time_vector(participant, run, TR, raw_data_dir)

def save_all_rois_and_time_vectors(nicks_dir=NICKS_DIR, raw_data_dir=RAW_DATA_DIR, TR=0.98):
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            save_rois(participant, run, nicks_dir, raw_data_dir)
            save_time_vector(participant, run, TR, raw_data_dir)

# if __name__ == "__main__":
    # get_valid_cortical_ROIs()
    # get_valid_thalamic_ROIs()
    # save_all_rois_and_time_vectors()