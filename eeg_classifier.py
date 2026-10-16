from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


RAW_DATA_DIR = Path('raw_data')
SEGMENTS_DIR = Path('plots/label_segments')
CLASSIFIER_DIR = Path('plots/EEG_classifier')
CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)

def get_eeg_participant_runs():
    if (CLASSIFIER_DIR / 'participant_runs.csv').exists():
        df = pd.read_csv(CLASSIFIER_DIR / 'participant_runs.csv')
        return df
    participant_runs = []
    for participant_dir in RAW_DATA_DIR.iterdir():
        if participant_dir.is_dir():
            participant = participant_dir.name
            for run_dir in participant_dir.iterdir():
                if run_dir.is_dir() and (run_dir / 'sleep_scoring').exists():
                    run = run_dir.name
                    participant_runs.append({
                        'participant': participant,
                        'run': run
                    })
    df = pd.DataFrame(participant_runs)
    df = df.sort_values(by=['participant', 'run']).reset_index(drop=True)
    df.to_csv(CLASSIFIER_DIR / 'participant_runs.csv', index=False)
    return df

def load_all_segment_metadata():
    if (CLASSIFIER_DIR / 'all_segment_metadata.csv').exists():
        df = pd.read_csv(CLASSIFIER_DIR / 'all_segment_metadata.csv')
        return df

    segment_data = []
    for segment_file in SEGMENTS_DIR.glob('*.csv'):
        df = pd.read_csv(segment_file)
        segment_data.append(df)
    segment_data = pd.concat(segment_data, ignore_index=True)
    segment_data = segment_data[segment_data['contains_motion'] == False][['participant', 'run', 'label', 'segment_start_TR', 'segment_end_TR_exclusive']]
    segment_data = segment_data.sort_values(['participant', 'run', 'segment_start_TR'])
    participant_runs = get_eeg_participant_runs()
    segment_data = segment_data.merge(participant_runs, on=['participant', 'run'], how='inner')
    segment_data.to_csv(CLASSIFIER_DIR / 'all_segment_metadata.csv', index=False)
    return segment_data

def plot_segment_counts_by_participant():
    participant_runs = get_eeg_participant_runs()
    participant_numbers = pd.Index(
        sorted(participant_runs['participant'].str.extract(r"(\d+)")[0].astype(int).unique()),
        name="participant_num",
    )

    segment_data = pd.read_csv(CLASSIFIER_DIR / 'all_segment_metadata.csv')

    def counts_by_participant(segments):
        participant_num = segments["participant"].str.extract(r"(\d+)")[0].astype(int)
        return participant_num.value_counts().reindex(participant_numbers, fill_value=0).sort_index()

    responsive_counts = counts_by_participant(segment_data[segment_data['label'] == 'responsive'])
    non_responsive_counts = counts_by_participant(segment_data[segment_data['label'] == 'non-responsive'])
    x = np.arange(len(participant_numbers))
    width = 0.4

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="lightgray", linewidth=0.8, alpha=0.6)
    for separator in np.arange(0.5, len(participant_numbers) - 0.5, 1):
        ax.axvline(separator, color="lightgray", linewidth=0.8, zorder=0)
    ax.bar(x - width / 2, responsive_counts, width, color="#9ADF8C", edgecolor='black', linewidth=1, label="Responsive")
    ax.bar(x + width / 2, non_responsive_counts, width, color="pink", edgecolor='black', linewidth=1, label="Non-responsive")
    ax.set_xlabel("Participant", fontsize=22)
    ax.set_ylabel("Number of segments", fontsize=22)
    ax.set_title("Motion-free Responsive and Non-responsive Segments per Participant with EEG", fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(participant_numbers, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(CLASSIFIER_DIR / "segment_counts_by_participant.png", dpi=300)
    plt.close()



if __name__ == "__main__":
    participant_runs = get_eeg_participant_runs()
    for _, row in participant_runs.iterrows():
        participant = row['participant']
        run = row['run']
        # to be continued