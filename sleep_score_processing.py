"""
Stores the 3 EEG channels from the 5-7-25 MAT files (in nicks_data) in raw_data/.../sleep_scoring folder, along with the runN_hypno2.txt file converted to JSON.
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

NICKS_DIR = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/nicks_data'
RAW_DATA_DIR = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/raw_data'

def load_eeg(participant, run, nicks_dir=NICKS_DIR):
    directory = Path(nicks_dir) / participant / run / 'sleep_scoring'
    path = next(directory.glob("*_9chan_toscore_filt_5-7-25_NEW_ref.mat"))
    mat = loadmat(path, variable_names=['data_filt', 'eeg_t'])
    data_filt = np.asarray(mat['data_filt'], dtype=np.float32)
    eeg_t = np.asarray(mat['eeg_t'], dtype=np.float32).reshape(-1) # flatten
    if data_filt.ndim != 2 or data_filt.shape[0] < 3:
        raise ValueError(f'Expected at least three EEG rows in {path}, got shape {data_filt.shape}')
    if data_filt.shape[1] != eeg_t.size:
        raise ValueError(f'EEG and eeg_t lengths differ in {path}: {data_filt.shape[1]} vs {eeg_t.size}')
    return data_filt, eeg_t

def save_eeg(participant, run, data_filt, eeg_t, raw_data_dir=RAW_DATA_DIR):
    output_dir = Path(raw_data_dir) / participant / run / 'sleep_scoring'
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'frontal_eeg.npy', data_filt[0])
    np.save(output_dir / 'central_eeg.npy', data_filt[1])
    np.save(output_dir / 'occipital_eeg.npy', data_filt[2])
    np.save(output_dir / 'eeg_t.npy', eeg_t)

def is_hypnogram_in_seconds(hypnogram):
    timestamps = [timestamp for intervals in hypnogram.values() for interval in intervals for timestamp in interval]
    final_timestamp = max(timestamps)
    return all(float(t).is_integer() for t in timestamps if t != final_timestamp)

def convert_hypnogram_to_seconds(hypnogram):
    # some hypnogram times are in minutes; convert to seconds
    hypnogram_seconds = {}
    for k, v in hypnogram.items():
        hypnogram_seconds[k] = []
        for start_timestamp, end_timestamp in v:
            hypnogram_seconds[k].append((start_timestamp * 60, end_timestamp * 60))
    return hypnogram_seconds

def load_hypnogram(participant, run, nicks_dir=NICKS_DIR):
    """Read runN_hypno2.txt and group its intervals by sleep-stage label."""
    directory = Path(nicks_dir) / participant / run / 'sleep_scoring'
    path = next(directory.glob("*_hypno2.txt"))
    with open(path) as file:
        lines = [line.strip() for line in file if line.strip()]
    stages = {}
    for i, line in enumerate(lines):
        start, stop, stage = line.split(",")
        start, stop = float(start), float(stop)
        stages.setdefault(stage.strip(), []).append([start, stop])    
    return stages


def save_hypnogram(participant, run, stages, raw_data_dir=RAW_DATA_DIR):
    output_dir = Path(raw_data_dir) / participant / run / 'sleep_scoring'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'hypno.json', 'w') as file:
        json.dump(stages, file, indent=2)

def sleep_scoring_runs(nicks_dir=NICKS_DIR):
    """Yield runs that have copied sleep-scoring MAT files in nicks_data."""
    for participant in sorted(os.listdir(nicks_dir)):
        participant_dir = os.path.join(nicks_dir, participant)
        if not os.path.isdir(participant_dir):
            continue
        for run in sorted(os.listdir(participant_dir)):
            folder = os.path.join(participant_dir, run, 'sleep_scoring')
            if os.path.isdir(folder):
                yield participant, run

def save_all_sleep_scoring(nicks_dir=NICKS_DIR, raw_data_dir=RAW_DATA_DIR):
    for participant, run in sleep_scoring_runs(nicks_dir):
        print(f'Processing sleep scoring for {participant} {run}')
        data_filt, eeg_t = load_eeg(participant, run, nicks_dir)
        stages = load_hypnogram(participant, run, nicks_dir)

        if not is_hypnogram_in_seconds(stages):
            print('\tHypnogram appears to be in minutes; converting to seconds.')
            stages = convert_hypnogram_to_seconds(stages)
        
        eeg_duration = eeg_t[-1] - eeg_t[0]
        hypno_end = max(end for intervals in stages.values() for start, end in intervals)
        if hypno_end < eeg_duration / 10:
            print('\tHypnogram appears to be in minutes; converting to seconds.')
            stages = convert_hypnogram_to_seconds(stages)

        save_eeg(participant, run, data_filt, eeg_t, raw_data_dir)
        save_hypnogram(participant, run, stages, raw_data_dir)

if __name__ == '__main__':
    save_all_sleep_scoring()