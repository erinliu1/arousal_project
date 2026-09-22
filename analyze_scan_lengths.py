from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW_DATA = Path('/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/raw_data')

def get_scan_lengths(raw_data_dir=RAW_DATA):
    scan_lengths = []
    for participant_dir in sorted(raw_data_dir.iterdir()):
        if not participant_dir.is_dir():
            continue
        participant = participant_dir.name
        for run_dir in sorted(participant_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run = run_dir.name
            time_vector = np.load(run_dir / 'time_vector.npy')
            bps = np.load(run_dir / 'bps.npy')
            if len(bps) == 0:
                bps = [None]
            try:
                eeg_t = np.load(run_dir / 'sleep_scoring' / 'eeg_t.npy')
            except FileNotFoundError:
                eeg_t = [None]

            scan_lengths.append({
                'participant': participant,
                'run': run,
                'fMRI_length': time_vector[-1],
                'bps_length': bps[-1],
                'eeg_length': eeg_t[-1]
            })

    return scan_lengths

def plot_scan_lengths():
    scan_lengths = get_scan_lengths()
    scan_lengths_df = pd.DataFrame(scan_lengths)

    fig, ax = plt.subplots(figsize=(27, 7))
    bar_width = 0.25
    group_spacing = 0.25

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['fMRI', 'Button presses', 'EEG']

    x_positions = []
    participant_centers = []
    participant_labels = []
    separator_positions = []

    x = 0
    for participant, participant_df in scan_lengths_df.groupby('participant', sort=False):
        participant_x = []

        for _, row in participant_df.iterrows():
            # Center of this participant-run group
            x_positions.append(x)
            participant_x.append(x)
            x += 1

        # Center participant label under all of their runs
        participant_centers.append(np.mean(participant_x))
        participant_labels.append(int(participant[-2:]))

        # Separator after this participant
        separator_positions.append(x - 0.5 + group_spacing)

        # Add extra space before next participant
        x += group_spacing
    x_positions = np.array(x_positions)

    # Three bars per participant-run
    ax.bar(
        x_positions - bar_width,
        scan_lengths_df['fMRI_length'],
        width=bar_width,
        label='fMRI',
        color=colors[0]
    )

    ax.bar(
        x_positions,
        scan_lengths_df['bps_length'],
        width=bar_width,
        label='Button presses',
        color=colors[1]
    )

    ax.bar(
        x_positions + bar_width,
        scan_lengths_df['eeg_length'],
        width=bar_width,
        label='EEG',
        color=colors[2]
    )

    # Participant separators
    for sep in separator_positions[:-1]:
        ax.axvline(
            sep,
            color='lightgray',
            linewidth=2
        )

    # Participant labels centered beneath their runs
    ax.set_xticks(participant_centers)
    ax.set_xticklabels(participant_labels, fontsize=11)

    ax.set_xlabel('Participant', fontsize=14)
    ax.set_ylabel('Scan Length (seconds)', fontsize=14)
    ax.set_title('Scan Lengths by Participant and Run', fontsize=18, weight='bold')

    ax.legend(frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(Path('plots') / 'scan_lengths.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # which participant-runs where EEG length is less than fMRI length?
    # scan_lengths = get_scan_lengths()
    # scan_lengths_df = pd.DataFrame(scan_lengths)
    # print(scan_lengths_df[scan_lengths_df['eeg_length'] >= scan_lengths_df['fMRI_length']])
    
    # which participant-runs are missing any of the three lengths?
    # length_cols = ['fMRI_length', 'bps_length', 'eeg_length']
    # n_complete = scan_lengths_df[length_cols].notna().all(axis=1).sum()
    # n_missing = scan_lengths_df[length_cols].isna().any(axis=1).sum()
    # print(f"Runs with all 3: {n_complete}")
    # print(f"Runs missing at least one: {n_missing}")
    # print(f"Total runs: {len(scan_lengths_df)}")

    # plot_scan_lengths()