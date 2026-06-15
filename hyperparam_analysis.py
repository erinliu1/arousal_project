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
loocv = 'lopocv'
hyperparam_results_dir = f'/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/{model_type}/{loocv}/hyperparam_tuning_results'
all_results = []
for job_file in os.listdir(hyperparam_results_dir):
    results = pd.read_csv(os.path.join(hyperparam_results_dir, job_file))
    all_results.append(results)

all_results_df = pd.concat(all_results, ignore_index=True)

def get_segment_weighted_accuracy(row):
    # each row contains one run; each run contains segments; each segment contains multiple TRs
    # average the probabilities of the TRs for each segment, then classify the segment as positive if the average probability >= 0.5, negative otherwise
    # accuracy is then the proportion of segments correctly classified
        y_prob = np.array(row['y_prob'])
        y_true = np.array(row['y_true'])
        tr_positions = np.array(row['tr_positions'])
        n_trs = tr_positions.max() + 1
        n_segments = len(tr_positions) // n_trs
        segment_probs = y_prob.reshape(n_segments, n_trs)
        segment_probs = segment_probs.mean(axis=1)
        segment_true = y_true.reshape(n_segments, n_trs)[:, 0]
        segment_pred = (segment_probs >= 0.5).astype(int)
        segment_accuracy = np.mean(segment_pred == segment_true)
        return segment_accuracy


if 'snapshot' in model_type:
    for col in ['y_pred', 'y_prob', 'y_true', 'tr_positions']:
        all_results_df[col] = all_results_df[col].apply(ast.literal_eval)
    all_results_df['accuracy'] = all_results_df.apply(get_segment_weighted_accuracy, axis=1)
    all_results_df.drop(columns=['y_pred', 'y_prob', 'y_true', 'tr_positions'], inplace=True)


participant_acc_df = (
    all_results_df
    .groupby(['bootstrap_seed', 'C', 'l1_ratio', 'solver', 'participant'], as_index=False)
    .agg(participant_accuracy=('accuracy', 'mean'))  # average over runs within participant
)

participant_acc_df = (
    participant_acc_df
    .groupby(['bootstrap_seed', 'C', 'l1_ratio', 'solver'], as_index=False)
    .agg(participant_weighted_accuracy=('participant_accuracy', 'mean'))  # average over participants
)

all_results_df = all_results_df.merge(
    pts_gap0[['participant', 'run', 'num_segments']],
    left_on=['participant', 'run_id'],
    right_on=['participant', 'run'],
    how='left'
)
all_results_df['weighted_accuracy'] = all_results_df['accuracy'] * all_results_df['num_segments']

# aggregate across runs
all_results_df = all_results_df.groupby(['bootstrap_seed', 'C', 'l1_ratio', 'solver']).agg(
    run_weighted_accuracy=('accuracy', 'mean'),
    segment_weighted_accuracy=('weighted_accuracy', 'sum'),
    total_segments=('num_segments', 'sum'),
    n_runs=('run_id', 'nunique')
).reset_index()

all_results_df['segment_weighted_accuracy'] = all_results_df['segment_weighted_accuracy'] / all_results_df['total_segments']


# add participant-weighted accuracy
all_results_df = all_results_df.merge(
    participant_acc_df,
    on=['bootstrap_seed', 'C', 'l1_ratio', 'solver'],
    how='left'
)
# aggregate across bootstraps
all_results_df = all_results_df.groupby(['C', 'l1_ratio', 'solver']).agg(
    segment_mean=('segment_weighted_accuracy', 'mean'),
    segment_std=('segment_weighted_accuracy', 'std'),
    run_mean=('run_weighted_accuracy', 'mean'),
    run_std=('run_weighted_accuracy', 'std'),
    participant_mean=('participant_weighted_accuracy', 'mean'),
    participant_std=('participant_weighted_accuracy', 'std'),
    n_seeds=('bootstrap_seed', 'nunique')
).reset_index()

# # sort
# top_10_participant_weighted = all_results_df.sort_values('participant_mean', ascending=False).reset_index(drop=True).head(10)
# top_10_segment_weighted = all_results_df.sort_values('segment_mean', ascending=False).reset_index(drop=True).head(10)
# top_10_run_weighted = all_results_df.sort_values('run_mean', ascending=False).reset_index(drop=True).head(10)

# # combine top 10s
# top_10 = pd.concat([top_10_segment_weighted, top_10_run_weighted, top_10_participant_weighted]).drop_duplicates().reset_index(drop=True)
# print(top_10)

plot_df = all_results_df[
    (all_results_df['solver'] != 'lbfgs') & (all_results_df['C'] != 0.0001) & (all_results_df['C'] != 100) & (all_results_df['C'] != 10)
    ].copy()

# make readable hyperparameter labels
superscript = str.maketrans(
    "-0123456789",
    "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"
)

def format_C(C):
    exp = int(np.log10(C))
    return f"10{str(exp).translate(superscript)}"


plot_df['combo_label'] = plot_df.apply(
    lambda row: (
        f"{row['solver']}\nC={format_C(row['C'])}\nl1={row['l1_ratio']:g}"
        if row['solver'] == 'saga'
        else f"{row['solver']}\nC={format_C(row['C'])}"
    ),
    axis=1
)

# sort in a sensible order
plot_df = plot_df.sort_values(['solver', 'C', 'l1_ratio']).reset_index(drop=True)

n_seeds = plot_df['n_seeds']

plot_df['segment_ci'] = 1.96 * plot_df['segment_std'] / np.sqrt(n_seeds)
plot_df['run_ci'] = 1.96 * plot_df['run_std'] / np.sqrt(n_seeds)
plot_df['participant_ci'] = 1.96 * plot_df['participant_std'] / np.sqrt(n_seeds)

fig, ax = plt.subplots(figsize=(18, 6))

x = np.arange(len(plot_df))

for mean_col, ci_col, label in [
    ('segment_mean', 'segment_ci', 'Segment-weighted'),
    ('run_mean', 'run_ci', 'Run-weighted'),
    ('participant_mean', 'participant_ci', 'Participant-weighted'),
]:
    line = ax.plot(
        x,
        plot_df[mean_col],
        marker='o',
        label=label
    )[0]

    ax.fill_between(
        x,
        plot_df[mean_col] - plot_df[ci_col],
        plot_df[mean_col] + plot_df[ci_col],
        color=line.get_color(),
        alpha=0.2
    )

for i in range(len(plot_df) - 1):

    curr = plot_df.iloc[i]
    nxt = plot_df.iloc[i + 1]

    if (
        curr['solver'] != nxt['solver']
        or (
            curr['solver'] == 'saga'
            and curr['C'] != nxt['C']
        )
    ):
        ax.axvline(
            i + 0.5,
            color='gray',
            linestyle='--',
            linewidth=1,
            alpha=0.4
        )
    
ax.set_xticks(x)
ax.set_xticklabels(plot_df['combo_label'], fontsize=8)

ax.set_xlabel('Hyperparameter combo')
ax.set_ylabel('Accuracy')
ax.set_title('Hyperparameter Tuning Accuracy (Mean ± 95% CI) Across Bootstraps')
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0.49, 0.61)
ax.set_yticks([0.5, 0.52, 0.54, 0.56, 0.58, 0.6])
plt.tight_layout()

save_dir = f'/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/{model_type}/{loocv}/plots'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'all_hyperparameter_tuning.png'), dpi=300)
plt.close()