import os
import shutil

def get_sleepscore_directory(participant):
    directory = f'/orcd/data/ldlewis/001/om2/ncicero/AAN/{participant}/eeg/gac/sleep_scoring'
    return directory

nicks_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/nicks_data'

for participant in sorted(os.listdir(nicks_dir)):
    sleepscore_dir = get_sleepscore_directory(participant)
    if not os.path.isdir(os.path.join(nicks_dir, participant, 'sleep_scoring')):
        continue
    runs = {}
    for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
        if run.startswith('run'):
            runs[run] = []
    for file in os.listdir(os.path.join(nicks_dir, participant, 'sleep_scoring')):
        run_id = file[1:3]
        if f'run{run_id}' in runs:
            runs[f'run{run_id}'].append(file)
    for run, files in runs.items():
        for file in files:
            source_dir = os.path.join(nicks_dir, participant, 'sleep_scoring', file)
            destination_dir = os.path.join(nicks_dir, participant, run, 'sleep_scoring')
            os.makedirs(destination_dir, exist_ok=True)
            destination_path = os.path.join(destination_dir, file)
            shutil.copy(source_dir, destination_path)
    shutil.rmtree(os.path.join(nicks_dir, participant, 'sleep_scoring'))