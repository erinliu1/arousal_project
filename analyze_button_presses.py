from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RAW_DATA = Path('/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/raw_data')

def get_button_press_intervals(raw_data_dir=RAW_DATA, snap_to_TRs=True, save_path=Path('plots')):
    button_press_intervals = []
    no_bp_intervals = []
    end_bp_intervals = []
    for participant_dir in sorted(raw_data_dir.iterdir()):
        if not participant_dir.is_dir():
            continue
        participant = participant_dir.name
        for run_dir in sorted(participant_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run = run_dir.name
            time_vector = np.load(run_dir / 'time_vector.npy')
            if snap_to_TRs:
                # intervals lengths will be computed in units of TR indices, e.g. if a button press occurs at TR = 10, and the next on occurs at TR = 15, the interval length will be 15 - 10 = 5 TRs
                bps = np.load(run_dir / 'bp_TRs.npy')
                bps = bps[bps < len(time_vector)] 
                bps = np.unique(bps)        # collapses multiple presses within the same TR; removes sub-second presses
            else:
                bps = np.load(run_dir / 'bps.npy')
                bps = bps[bps <= time_vector[-1]]

            if len(bps) < 2:
                no_bp_intervals.append({
                    'participant': participant,
                    'run': run,
                    'duration': time_vector[-1]
                })
            else:
                # get all button press intervals (time between consecutive button presses)
                intervals = np.diff(bps)
                for i in intervals:
                    button_press_intervals.append({
                        'participant': participant,
                        'run': run,
                        'interval': i
                    })
                if snap_to_TRs:
                    last_TR = len(time_vector) - 1
                    end_interval = last_TR - bps[-1]
                    if end_interval > 0:
                        end_bp_intervals.append({
                            'participant': participant,
                            'run': run,
                            'interval': end_interval
                        })
                else:
                    end_interval = time_vector[-1] - bps[-1]
                    if end_interval > 0:
                        end_bp_intervals.append({
                            'participant': participant,
                            'run': run,
                            'interval': end_interval
                        })

    if save_path:
        button_press_intervals_df = pd.DataFrame(button_press_intervals)
        button_press_intervals_df.to_csv(save_path / 'button_press_intervals.csv', index=False)

        no_bp_intervals_df = pd.DataFrame(no_bp_intervals)
        no_bp_intervals_df.to_csv(save_path / 'no_button_press_intervals.csv', index=False)

        end_bp_intervals_df = pd.DataFrame(end_bp_intervals)
        end_bp_intervals_df.to_csv(save_path / 'end_button_press_intervals.csv', index=False)

    return button_press_intervals, no_bp_intervals, end_bp_intervals

def scatter_bp_intervals(log_scale=False, snap_to_TRs=True):
    fig, ax = plt.subplots(figsize=(27, 9))

    button_press_intervals, no_bp_intervals, end_bp_intervals = get_button_press_intervals(snap_to_TRs=snap_to_TRs)
    df = pd.DataFrame(button_press_intervals)
    df = df.groupby(['participant', 'run'], as_index=False).agg(intervals=('interval', list))

    empty_intervals = []
    for element in no_bp_intervals:
        empty_intervals.append({
            'participant': element['participant'],
            'run': element['run'],
            'intervals': []
        })
    no_bp_intervals_df = pd.DataFrame(no_bp_intervals)
    empty_intervals_df = pd.DataFrame(empty_intervals)    
    end_bp_intervals_df = pd.DataFrame(end_bp_intervals)

    df = pd.concat([df, empty_intervals_df], ignore_index=True)

    df['participant_num'] = df['participant'].str.split('_').str[-1].astype(int)
    df['run_num'] = df['run'].str.replace('run', '', regex=False).astype(int)
    df = df.sort_values(['participant_num', 'run_num'])

    x = 0
    participant_centers, participant_labels = [], []

    participants = df['participant'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(participants)))
    participant_colors = dict(zip(participants, colors))

    for p_idx, (participant, participant_df) in enumerate(df.groupby('participant', sort=False)):
        participant_xs = []
        color = participant_colors[participant]

        for r_idx, (_, row) in enumerate(participant_df.iterrows()):
            intervals = row['intervals']
            if len(intervals) == 0:
                # lookup the row with participant and run in no_bp_intervals_df
                no_bp_row = no_bp_intervals_df[(no_bp_intervals_df['participant'] == participant) & (no_bp_intervals_df['run'] == row['run'])]
                duration = no_bp_row['duration'].values[0]
                ax.scatter(x, duration, marker='*', alpha=0.7, s=180, color=color, linewidths=2)
            else:
                ax.scatter(np.full(len(intervals), x), intervals, alpha=0.7, s=100, color=color)
            
            # lookup the row with participant and run in end_bp_intervals_df
            end_bp_row = end_bp_intervals_df[(end_bp_intervals_df['participant'] == participant) & (end_bp_intervals_df['run'] == row['run'])]
            if not end_bp_row.empty:
                end_interval = end_bp_row['interval'].values[0]
                ax.scatter(x, end_interval, marker='x', alpha=0.7, s=140, color=color)

            participant_xs.append(x)
            if r_idx < len(participant_df) - 1:
                ax.axvline(x+0.5, color='lightgray', linewidth=2)
            x += 1
        
        participant_centers.append(np.mean(participant_xs))
        participant_labels.append(participant_df['participant_num'].unique()[0])
        if p_idx < df['participant'].nunique() - 1:
            ax.axvline(x-0.5, color='gray', linewidth=3, alpha=0.6)

    ax.set_xticks(participant_centers)
    ax.set_xticklabels(participant_labels, fontsize=22)

    ax.tick_params(axis='x', width=2, length=8)
    ax.tick_params(axis='y', which='both', length=0, labelsize=22)
    
    metric = 's' if not snap_to_TRs else 'TRs'
    ax.set_xlabel('Participant', fontsize=32, labelpad=15)
    ax.set_ylabel(f'Button Press Interval ({metric})', fontsize=32, labelpad=15)
    ax.set_title('Button Press Intervals by Participant and Run', fontsize=34, fontweight='bold')

    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f'{y:,.0f}')
        )

    save_name = 'button_press_intervals_log.png' if log_scale else 'button_press_intervals.png'
    plt.tight_layout()
    plt.savefig(Path('plots') / save_name, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    scatter_bp_intervals(log_scale=True, snap_to_TRs=True)
