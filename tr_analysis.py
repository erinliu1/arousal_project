import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast 

pts = pd.read_csv('/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/fyi/pts.csv')
is_b = pts['participant'].str.endswith('_b')
pts.loc[is_b, 'run'] = pts.loc[is_b, 'run'] + '_b'
pts.loc[is_b, 'participant'] = (
    pts.loc[is_b, 'participant']
    .str.removesuffix('_b')
)
pts_gap0 = pts[pts['gap'] == 0]
pts_gap0['num_segments'] = pts_gap0['pts'] * 2
pts_gap0 = pts_gap0[pts_gap0['num_segments'] > 0]

model_type = 'snapshot_45'
loocv = 'lorocv'
hyperparam_results_dir = f'/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/{model_type}/{loocv}/hyperparam_tuning_results'
all_results = []
for job_file in os.listdir(hyperparam_results_dir):
    results = pd.read_csv(os.path.join(hyperparam_results_dir, job_file))
    all_results.append(results)

all_results_df = pd.concat(all_results, ignore_index=True)

for col in ['y_pred', 'y_prob', 'y_true', 'tr_positions']:
    all_results_df[col] = all_results_df[col].apply(ast.literal_eval)

hyperparams = [(0.1, 'liblinear'), (1, 'liblinear'), (0.1, 'lbfgs'), (1, 'lbfgs'), (0.1, 'saga'), (1, 'saga')]
tr_df = []
for _, row in all_results_df.iterrows():
    C = row['C']
    l1_ratio = row['l1_ratio']
    solver = row['solver']
    
    if (C, solver) not in hyperparams:
        continue

    y_true = np.array(row['y_true'])
    y_pred = np.array(row['y_pred'])
    y_prob = np.array(row['y_prob'])
    tr_positions = np.array(row['tr_positions'])
    n_trs = tr_positions.max() + 1

    for idx, (y_true_i, y_pred_i, y_prob_i, tr_pos_i) in enumerate(zip(y_true, y_pred, y_prob, tr_positions)):

        tr_df.append({
            'bootstrap_seed': row['bootstrap_seed'],
            'participant': row['participant'],
            'run_id': row['run_id'],
            'y_true': y_true_i,
            'y_pred': y_pred_i,
            'y_prob': y_prob_i,
            'tr_position': tr_pos_i,
            'segment_id': idx // n_trs,
            'correct': int(y_true_i == y_pred_i)
        })

tr_df = pd.DataFrame(tr_df)

def get_snapshot_accuracy_by_tr(tr_df):
    tr_df_within_bootstraps = tr_df.groupby(['bootstrap_seed', 'tr_position'], as_index=False).agg(
        accuracy=('correct', 'mean'),
    )
    tr_df_across_bootstraps = tr_df_within_bootstraps.groupby('tr_position', as_index=False).agg(
        mean_accuracy=('accuracy', 'mean'),
        std_accuracy=('accuracy', 'std') 
    )
    n_bootstraps = tr_df_within_bootstraps['bootstrap_seed'].nunique()
    tr_df_across_bootstraps['sem_accuracy'] = tr_df_across_bootstraps['std_accuracy'] / np.sqrt(n_bootstraps)
    return tr_df_across_bootstraps

def get_run_accuracy_by_tr(tr_df):
    tr_acc_within_run = tr_df.groupby(['bootstrap_seed', 'participant', 'run_id', 'tr_position'], as_index=False).agg(
        accuracy=('correct', 'mean'),
    )
    tr_acc_within_bootstraps = tr_acc_within_run.groupby(['bootstrap_seed', 'tr_position'], as_index=False).agg(
        accuracy=('accuracy', 'mean'),
    )
    tr_acc_across_bootstraps = tr_acc_within_bootstraps.groupby('tr_position', as_index=False).agg(
        mean_accuracy=('accuracy', 'mean'),
        std_accuracy=('accuracy', 'std')
    )
    n_bootstraps = tr_acc_within_bootstraps['bootstrap_seed'].nunique()
    tr_acc_across_bootstraps['sem_accuracy'] = tr_acc_across_bootstraps['std_accuracy'] / np.sqrt(n_bootstraps)
    return tr_acc_across_bootstraps

def get_participant_accuracy_by_tr(tr_df):
    tr_acc_within_run = tr_df.groupby(['bootstrap_seed', 'participant', 'run_id', 'tr_position'], as_index=False).agg(
        accuracy=('correct', 'mean'),
    )
    tr_acc_within_participant = tr_acc_within_run.groupby(['bootstrap_seed', 'participant', 'tr_position'], as_index=False).agg(
        accuracy=('accuracy', 'mean'),
    )
    tr_acc_across_participants = tr_acc_within_participant.groupby(['bootstrap_seed', 'tr_position'], as_index=False).agg(
        accuracy=('accuracy', 'mean'),
    )
    tr_acc_across_bootstraps = tr_acc_across_participants.groupby('tr_position', as_index=False).agg(
        mean_accuracy=('accuracy', 'mean'),
        std_accuracy=('accuracy', 'std')
    )
    n_bootstraps = tr_acc_across_participants['bootstrap_seed'].nunique()
    tr_acc_across_bootstraps['sem_accuracy'] = tr_acc_across_bootstraps['std_accuracy'] / np.sqrt(n_bootstraps)
    return tr_acc_across_bootstraps

def plot_snapshot_accuracy_by_tr(tr_df, save_dir):
    snapshot_df = get_snapshot_accuracy_by_tr(tr_df)
    run_df = get_run_accuracy_by_tr(tr_df)
    participant_df = get_participant_accuracy_by_tr(tr_df)

    plt.figure(figsize=(10, 6))
    snapshot_line = plt.plot(snapshot_df['tr_position'], snapshot_df['mean_accuracy'], marker='o', label='Snapshot-Weighted')[0]
    plt.fill_between(snapshot_df['tr_position'], snapshot_df['mean_accuracy'] - 1.96 * snapshot_df['sem_accuracy'], snapshot_df['mean_accuracy'] + 1.96 * snapshot_df['sem_accuracy'],alpha=0.2, color=snapshot_line.get_color())
    run_line = plt.plot(run_df['tr_position'], run_df['mean_accuracy'], marker='s', label='Run-Weighted')[0]
    plt.fill_between(run_df['tr_position'], run_df['mean_accuracy'] - 1.96 * run_df['sem_accuracy'], run_df['mean_accuracy'] + 1.96 * run_df['sem_accuracy'],alpha=0.2, color=run_line.get_color())
    participant_line = plt.plot(participant_df['tr_position'], participant_df['mean_accuracy'], marker='^', label='Participant-Weighted')[0]
    plt.fill_between(participant_df['tr_position'], participant_df['mean_accuracy'] - 1.96 * participant_df['sem_accuracy'], participant_df['mean_accuracy'] + 1.96 * participant_df['sem_accuracy'],alpha=0.2, color=participant_line.get_color())
    plt.xlabel('TR Position within Segment')
    plt.ylabel('Accuracy')
    plt.title('Accuracy (Mean ± 95% CI) by TR Position Across Bootstraps and Hyperparameter Combos')
    plt.legend()
    plt.ylim(0.49, 0.61)
    plt.yticks([0.5, 0.52, 0.54, 0.56, 0.58, 0.6])
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'snapshot_accuracy_by_tr.png'))
    plt.close()

save_dir = f'/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/{model_type}/{loocv}/plots'
os.makedirs(save_dir, exist_ok=True)

plot_snapshot_accuracy_by_tr(tr_df, save_dir)