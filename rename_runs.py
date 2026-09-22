from pathlib import Path
import re

RAW_DATA_DIR = Path('/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/raw_data')

# in this directory, each participant gets a folder, with some participant names ending with '_b'
# For participants ending with '_b', we want to rename their runs to be continuous with the runs of the base participant (without '_b'). e.g. if participant 'AAN_07' has runs 1-4, and 'AAN_07_b' has runs 1-3, we want to rename the runs of 'AAN_07_b' to be 5-7 and rename the participant to 'AAN_07' for those runs.

def get_run_number(run):
    # extract the run number from a run name like 'run01' or 'run02'
    return int(re.match(r'run(\d+)', run).group(1))

def rename_runs(raw_data_dir=RAW_DATA_DIR):
    for participant_b_dir in sorted(raw_data_dir.iterdir()):
        if not participant_b_dir.is_dir():
            continue
        participant_b = participant_b_dir.name
        if not participant_b.endswith('_b'):
            continue

        base_participant = participant_b[:-2]
        base_participant_dir = raw_data_dir / base_participant

        base_run_paths = list(base_participant_dir.iterdir())
        base_run_numbers = sorted([get_run_number(run.name) for run in base_run_paths])
        last_run_number = base_run_numbers[-1] if base_run_numbers else 0

        participant_b_run_paths = sorted(participant_b_dir.iterdir(), key=lambda x: get_run_number(x.name))
        for i, participant_b_run_path in enumerate(participant_b_run_paths, start=1):
            new_run_number = last_run_number + i
            new_run_name = f'run{new_run_number:02d}'
            new_run_path = base_participant_dir / new_run_name
            participant_b_run_path.rename(new_run_path)

        participant_b_dir.rmdir()

# if __name__ == '__main__':
#     rename_runs()
