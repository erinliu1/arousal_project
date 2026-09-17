"""
Stores the 3 EEG channels from the 5-7-25 MAT files (in nicks_data) in raw_data/.../sleep_scoring folder, along with the hypnogram json which came from the source runN_hypno2.txt files.
"""

import argparse
import json
import os
import re

import numpy as np
from scipy.io import loadmat

AAN_DIR = '/orcd/data/ldlewis/001/om2/ncicero/AAN'
NICKS_DIR = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/nicks_data'
SLEEP_SCORES_DIR = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/sleep_scores'
RAW_DATA_DIR = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/raw_data'

MULTICHANNEL_MAT_SUFFIX = '2-11-26_NEW_ref.mat'
SELECTED_CHANNEL_MAT_SUFFIX = '5-7-25_NEW_ref.mat'

def run_number(run):
    if isinstance(run, int):
        return run
    match = re.fullmatch(r'run0*(\d+)', run)
    if not match:
        raise ValueError(f'Invalid run name: {run}')
    return int(match.group(1))


def run_name(run):
    return f'run{run_number(run):02d}'


def find_mat_variant(participant, run, suffix, nicks_dir=NICKS_DIR):
    number = run_number(run)
    folder = os.path.join(
        nicks_dir, participant, run_name(run), 'sleep_scoring'
    )
    filename = f'r{number:02d}_9chan_toscore_filt_{suffix}'
    path = os.path.join(folder, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def load_required_mat_variables(path, variable_names):
    mat = loadmat(path, variable_names=variable_names)
    missing = [name for name in variable_names if name not in mat]
    if missing:
        raise KeyError(f'{path} is missing MATLAB variables: {missing}')
    return mat


def save_multichannel_sleep_score(participant, run, nicks_dir=NICKS_DIR,
                                  sleep_scores_dir=SLEEP_SCORES_DIR):
    """Convert one 2-11-26 MAT file into the top-level sleep_scores format."""
    mat_path = find_mat_variant(
        participant, run, MULTICHANNEL_MAT_SUFFIX, nicks_dir
    )
    mat = load_required_mat_variables(
        mat_path, ['data_filt', 'eeg_t', 'chanlocs', 'goodChans']
    )

    data_filt = np.asarray(mat['data_filt'], dtype=np.float32)
    eeg_t = np.asarray(mat['eeg_t'], dtype=np.float32)
    chanlocs = np.asarray(mat['chanlocs'], dtype=object)
    good_chans_float = np.asarray(mat['goodChans'])

    if data_filt.ndim != 2 or eeg_t.ndim != 2:
        raise ValueError(f'Unexpected EEG array shapes in {mat_path}')
    if data_filt.shape[1] != eeg_t.size:
        raise ValueError(f'EEG and eeg_t lengths differ in {mat_path}')
    if not np.all(good_chans_float == good_chans_float.astype(np.uint8)):
        raise ValueError(f'goodChans cannot be represented as uint8 in {mat_path}')
    good_chans = good_chans_float.astype(np.uint8)

    output_dir = os.path.join(sleep_scores_dir, participant, run_name(run))
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, 'data_filt.npz'), arr=data_filt)
    np.savez_compressed(os.path.join(output_dir, 'eeg_t.npz'), arr=eeg_t)
    np.save(os.path.join(output_dir, 'chanlocs.npy'), chanlocs)
    np.save(os.path.join(output_dir, 'goodChans.npy'), good_chans)


def load_hypnogram(participant, run, aan_dir=AAN_DIR):
    """Read runN_hypno2.txt and group its intervals by sleep-stage label."""
    path = os.path.join(
        aan_dir,
        participant,
        'eeg',
        'gac',
        'sleep_scoring',
        f'run{run_number(run)}_hypno2.txt',
    )
    stages = {}
    with open(path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(',')]
            if len(fields) != 3:
                raise ValueError(f'Malformed hypnogram line {line_number} in {path}')
            start, stop, stage = fields
            stages.setdefault(stage, []).append([float(start), float(stop)])
    return stages


def save_raw_sleep_scoring(participant, run, nicks_dir=NICKS_DIR,
                           aan_dir=AAN_DIR, raw_data_dir=RAW_DATA_DIR):
    """Save the three selected EEG rows, time, and hypnogram for one run."""
    mat_path = find_mat_variant(
        participant, run, SELECTED_CHANNEL_MAT_SUFFIX, nicks_dir
    )
    mat = load_required_mat_variables(mat_path, ['data_filt', 'eeg_t'])
    data_filt = np.asarray(mat['data_filt'], dtype=np.float32)
    eeg_t = np.asarray(mat['eeg_t'], dtype=np.float32).reshape(-1)

    if data_filt.ndim != 2 or data_filt.shape[0] < 3:
        raise ValueError(f'Expected at least three EEG rows in {mat_path}')
    if data_filt.shape[1] != eeg_t.size:
        raise ValueError(f'EEG and eeg_t lengths differ in {mat_path}')

    output_dir = os.path.join(
        raw_data_dir, participant, run_name(run), 'sleep_scoring'
    )
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'frontal_eeg.npy'), data_filt[0])
    np.save(os.path.join(output_dir, 'central_eeg.npy'), data_filt[1])
    np.save(os.path.join(output_dir, 'occipital_eeg.npy'), data_filt[2])
    np.save(os.path.join(output_dir, 'eeg_t.npy'), eeg_t)

    hypnogram = load_hypnogram(participant, run, aan_dir)
    with open(os.path.join(output_dir, 'hypno.json'), 'w') as file:
        json.dump(hypnogram, file, indent=2)


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


def process_sleep_scoring(participant, run, make_sleep_scores=True,
                          make_raw_data=True, nicks_dir=NICKS_DIR,
                          aan_dir=AAN_DIR,
                          sleep_scores_dir=SLEEP_SCORES_DIR,
                          raw_data_dir=RAW_DATA_DIR):
    if make_sleep_scores:
        save_multichannel_sleep_score(
            participant, run, nicks_dir, sleep_scores_dir
        )
    if make_raw_data:
        save_raw_sleep_scoring(
            participant, run, nicks_dir, aan_dir, raw_data_dir
        )


def process_all_sleep_scoring(make_sleep_scores=True, make_raw_data=True,
                              nicks_dir=NICKS_DIR, aan_dir=AAN_DIR,
                              sleep_scores_dir=SLEEP_SCORES_DIR,
                              raw_data_dir=RAW_DATA_DIR):
    for participant, run in sleep_scoring_runs(nicks_dir):
        process_sleep_scoring(
            participant,
            run,
            make_sleep_scores,
            make_raw_data,
            nicks_dir,
            aan_dir,
            sleep_scores_dir,
            raw_data_dir,
        )
        print(f'Processed sleep scoring for {participant} {run}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participant', help='Process one participant, e.g. AAN_01')
    parser.add_argument('--run', help='Process one run, e.g. run01')
    parser.add_argument(
        '--output',
        choices=['all', 'sleep_scores', 'raw_data'],
        default='all',
        help='Which derived dataset to create (default: all)',
    )
    args = parser.parse_args()
    if (args.participant is None) != (args.run is None):
        parser.error('--participant and --run must be supplied together')
    return args


if __name__ == '__main__':
    args = parse_args()
    make_sleep_scores = args.output in ('all', 'sleep_scores')
    make_raw_data = args.output in ('all', 'raw_data')
    if args.participant:
        process_sleep_scoring(
            args.participant, args.run, make_sleep_scores, make_raw_data
        )
    else:
        process_all_sleep_scoring(make_sleep_scores, make_raw_data)
