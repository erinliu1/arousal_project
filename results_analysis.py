import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
# import ast 

def correct_modes(df):
    df['mode'] = df['mode'].fillna('null')
    df['mode'] = df['mode'].replace(True, 'true')
    return df

def correct_participants(df):
    df['participant'] = df['participant'].str.replace('_b$', '', regex=True)
    return df

def merge_segment_count(df):
    pts = pd.read_csv('/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/fyi/pts.csv')
    pts = pts.copy()

    is_b = pts['participant'].str.endswith('_b')

    pts.loc[is_b, 'run'] = pts.loc[is_b, 'run'] + '_b'
    pts.loc[is_b, 'participant'] = (
        pts.loc[is_b, 'participant']
        .str.removesuffix('_b')
    )
    return df.merge(
        pts,
        left_on=['participant', 'run_id', 'gap'],
        right_on=['participant', 'run', 'gap'],
        how='left'
    ).drop(columns=['run'])

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

from tqdm import tqdm 

def get_all_results_df(model_type, loocv):
    results_dir = f'/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/{model_type}/{loocv}/training_results'
    all_results = []
    for bootstrap_idx in tqdm(range(len(os.listdir(results_dir))), desc="Processing bootstrap iterations"):
        results_df = pd.read_csv(os.path.join(results_dir, f'{bootstrap_idx}.csv'))
        results_df.drop(columns=['y_pred', 'segment_ids'], inplace=True)
        all_results.append(results_df)
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    if 'snapshot' in model_type:
        for col in ['y_true', 'y_prob', 'tr_positions']:
            tqdm.pandas(desc=f"Parsing {col}")
            all_results_df[col] = all_results_df[col].progress_apply(
                lambda x: np.fromstring(
                    x.strip("[]").replace(",", " "),
                    sep=" "
                )
            )
        all_results_df['y_true'] = all_results_df['y_true'].apply(lambda x: x.astype(int))
        all_results_df['tr_positions'] = all_results_df['tr_positions'].apply(lambda x: x.astype(int))
        tqdm.pandas(desc="Computing segment-weighted accuracy")
        all_results_df['accuracy'] = all_results_df.progress_apply(
            get_segment_weighted_accuracy,
            axis=1
        )
        all_results_df.drop(columns=['y_prob', 'y_true', 'tr_positions'], inplace=True)
    all_results_df = merge_segment_count(all_results_df)
    all_results_df = correct_modes(all_results_df)
    all_results_df = correct_participants(all_results_df)
    return all_results_df

def get_participant_df(df, participant):
    return df[df['participant'] == participant]
    
def compute_participant_weighted_accuracy(df):
    df = df.groupby(['bootstrap_seed', 'gap', 'mode', 'participant'], as_index=False).agg(
        accuracy=('accuracy', 'mean')
    )
    df = df.groupby(['bootstrap_seed', 'gap', 'mode'], as_index=False).agg(
        accuracy=('accuracy', 'mean')
    )
    return df

def compute_run_weighted_accuracy(df):
    # average over runs
    df = df.groupby(['bootstrap_seed', 'gap', 'mode'], as_index=False).agg(
        accuracy=('accuracy', 'mean')
    )
    return df

# segment-weighted accuracy
def compute_segment_weighted_accuracy(df):
    df['weighted_accuracy'] = df['accuracy'] * df['pts']
    df = df.groupby(['bootstrap_seed', 'gap', 'mode'], as_index=False).agg(
        sum_weighted_accuracy=('weighted_accuracy', 'sum'),
        total_segments=('pts', 'sum')
    )
    df['accuracy'] = df['sum_weighted_accuracy'] / df['total_segments']
    df.drop(columns=['sum_weighted_accuracy', 'total_segments'], inplace=True)
    return df

def get_distribution_separation(df):
    gaps = sorted(df['gap'].unique())
    results = []
    for gap in gaps:
        true_values = df[(df['mode'] == 'true') & (df['gap'] == gap)]['accuracy'].values
        null_values = df[(df['mode'] == 'null') & (df['gap'] == gap)]['accuracy'].values
        pairwise_comparisons = true_values[:, None] > null_values[None, :]
        separation = pairwise_comparisons.mean()
        results.append({
            'gap': gap,
            'P(True > Null)': separation
        })
    return pd.DataFrame(results)

def plot_distribution_separation(participant_df, run_df, segment_df, save_dir):
    participant_sep_df = get_distribution_separation(participant_df)
    run_sep_df = get_distribution_separation(run_df)
    segment_sep_df = get_distribution_separation(segment_df)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(segment_sep_df['gap'], segment_sep_df['P(True > Null)'], marker='o', linewidth=2, label='Segment-Weighted', color='C3')
    ax.plot(run_sep_df['gap'], run_sep_df['P(True > Null)'], marker='o', linewidth=2, label='Run-Weighted', color='C2')
    ax.plot(participant_sep_df['gap'], participant_sep_df['P(True > Null)'], marker='o', linewidth=2, label='Participant-Weighted', color='C4')
    ax.axhline(0.5, linestyle='--', linewidth=1, label='Chance (P=0.5)')
    ax.legend(fontsize=12)
    ax.set_xlabel('Gap', fontsize=12)
    ax.set_ylabel('P(True > Null)', fontsize=12)
    ax.tick_params(labelsize=12)
    ax.set_title(f'P(True > Null)', fontsize=14)
    ax.set_ylim(0.45, 1)
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'distribution_separation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_and_CI(df, metric, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    gaps = sorted(df['gap'].unique())
    fig, ax = plt.subplots(figsize=(16, 6))
    summary = df.groupby(['gap', 'mode'])['accuracy'].agg(mean='mean', ci_lower=lambda x: np.percentile(x, 2.5), ci_upper=lambda x: np.percentile(x, 97.5)).reset_index()
    null_thresholds = (
        df[df['mode'] == 'null']
        .groupby('gap')['accuracy']
        .apply(lambda x: np.percentile(x, 95))
        .reset_index(name='null_95th')
    )
    colors = {'null': 'C0', 'true': 'C1'}

    for mode in ['true', 'null']:
        mode_df = summary[summary['mode'] == mode].sort_values('gap')

        x = mode_df['gap'].values
        mean = mode_df['mean'].values
        ci_lower = mode_df['ci_lower'].values
        ci_upper = mode_df['ci_upper'].values

        ax.plot(x, mean, marker='o', linewidth=2, label=f'{mode.title()} Mean Accuracy', color=colors[mode])
        ax.fill_between(x, ci_lower, ci_upper, alpha=0.2, color=colors[mode])

    ax.plot(null_thresholds['gap'], null_thresholds['null_95th'], '--k', alpha=0.7, linewidth=1, label='Null 95th Percentile')
    ax.set_xlabel('Gap', fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.tick_params(labelsize=12)
    ax.set_ylim(0.50-0.15, 0.50+0.15)

    ax.set_title(
        f'{metric.replace("_", " ").title()} (Mean ± 95% CI) Across Bootstraps', fontsize=14
    )
    
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.savefig(
        os.path.join(save_dir, f'{metric}.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

def get_count_df(all_results_df):
    count_df = all_results_df[(all_results_df['mode'] == 'true') & (all_results_df['gap'] == 0) & (all_results_df['bootstrap_seed'] == 0)]
    count_df = count_df.groupby('participant')['pts'].sum().reset_index()
    return count_df

def get_accuracy_diff_df(df):
    pivot_df = df.pivot_table(index=['bootstrap_seed', 'gap'], columns='mode', values='accuracy').reset_index()
    accuracy_diff_df = pivot_df.copy()
    accuracy_diff_df['accuracy_diff'] = pivot_df['true'] - pivot_df['null']
    accuracy_diff_df = accuracy_diff_df.groupby('gap').agg(
        mean_diff=('accuracy_diff', 'mean'),
    ).reset_index()
    return accuracy_diff_df

def get_heatmap_matrices(all_results_df):
    gaps = sorted(all_results_df['gap'].unique())
    count_df = get_count_df(all_results_df)
    participant_order = count_df.sort_values('pts', ascending=False)['participant'].tolist()
    df_row_names = []
    segment_accuracy_diff_heatmap, segment_sep_heatmap = [], []

    for participant in participant_order:
        print(f'Processing participant {participant}...')
        n = count_df[count_df['participant'] == participant]['pts'].values[0]
        row_name = f"{participant} {f'(n={n})':>7}"
        df_row_names.append(row_name)

        participant_df = get_participant_df(all_results_df, participant)
        segment_accuracy_df = compute_segment_weighted_accuracy(participant_df)
        segment_sep_df = get_distribution_separation(segment_accuracy_df)
        segment_accuracy_diff_df = get_accuracy_diff_df(segment_accuracy_df)

        segment_accuracy_diff_heatmap.append(segment_accuracy_diff_df['mean_diff'].values)
        segment_sep_heatmap.append(segment_sep_df['P(True > Null)'].values)
    
    segment_accuracy_diff_heatmap_df = pd.DataFrame(segment_accuracy_diff_heatmap, index=df_row_names, columns=gaps)
    segment_sep_heatmap_df = pd.DataFrame(segment_sep_heatmap, index=df_row_names, columns=gaps)

    return segment_accuracy_diff_heatmap_df, segment_sep_heatmap_df

def plot_participant_heatmaps(all_results_df, save_dir):
    gaps = sorted(all_results_df['gap'].unique())
    segment_accuracy_diff_heatmap_df, segment_sep_heatmap_df = get_heatmap_matrices(all_results_df)

    max_diff = np.max(np.abs(segment_accuracy_diff_heatmap_df.values))
    gap_tick_step = 5

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for i, ax in enumerate(axes):
        if i == 0:
            sns.heatmap(segment_accuracy_diff_heatmap_df, ax=axes[0], cmap='coolwarm', center=0, vmin=-max_diff, vmax=max_diff, cbar=True)
            title = 'Mean Accuracy Difference (True - Null)'
        else:
            sns.heatmap(segment_sep_heatmap_df, ax=axes[1], cmap='viridis', center=None, vmin=0, vmax=1, cbar=True)
            title = 'P(True > Null)'
            ax.set_yticklabels([])

        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Gap', fontsize=12)
        ax.set_xticks(np.arange(0, len(gaps), gap_tick_step) + 0.5)
        ax.set_xticklabels(gaps[::gap_tick_step], rotation=0, fontsize=12)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'participant_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close() 

def plot_all_empirical_and_null_distributions(df, count, aggregate_type='',save_dir=''):
    count_df = get_count_df(all_results_df)
    os.makedirs(save_dir, exist_ok=True)
    x_min = df['accuracy'].min()
    x_max = df['accuracy'].max()

    gaps = sorted(df['gap'].unique())
    ncols = 8
    nrows = int(np.ceil(len(gaps) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4*ncols, 3.5*nrows),
        constrained_layout=True,
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()
    for i, gap in enumerate(gaps):
        ax = axes[i]
        gap_df = df[df['gap'] == gap]
        plot_one_empirical_and_null_distributions(gap_df, gap, ax)
        # ax.set_xlim(x_min, x_max)
        # if i % ncols == 0:
        #     ax.set_ylabel('Density', fontsize=20)
        # else:
        #     ax.set_ylabel('')

    for j in range(len(gaps), len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel('Bootstrap Accuracy', fontsize=24)
    fig.supylabel('Density', fontsize=24)
    # handles = [
    #     Patch(facecolor='C0', alpha=0.7, label='Null Distribution'),
    #     Patch(facecolor='C1', alpha=0.7, label='Empirical Distribution'),
    #     plt.Line2D([0], [0], color='green', linestyle='--', label='95% Null Threshold'),
    #     plt.Line2D([0], [0], color='red', linestyle='--', label='Empirical Mean'),
    #     plt.Line2D([0], [0], color='blue', linestyle='--', label='Null Mean')
    # ]

    # fig.legend(
    #     handles=handles,
    #     loc='center left',
    #     bbox_to_anchor=(1.05, 0.5),
    #     borderaxespad=0.
    # )    
    # fig.suptitle(
    #     f'{aggregate_type} Distributions (n={count})',
    #     fontsize=24
    # )
    plt.savefig(
        os.path.join(save_dir, f'{aggregate_type}.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

def plot_one_empirical_and_null_distributions(gap_df, gap, ax):
    true_df = gap_df[gap_df['mode'] == 'true']
    null_df = gap_df[gap_df['mode'] == 'null']
    null_threshold = np.percentile(null_df['accuracy'], 95)

    ax.hist(null_df['accuracy'], bins=30, alpha=0.7, density=True, label='Null Distribution', color='C0')
    ax.axvline(null_threshold, color='green', linestyle='--', label='95th Percentile Threshold')
    ax.hist(true_df['accuracy'], bins=30, alpha=0.7, density=True, label='Empirical Distribution', color='C1')
    ax.axvline(true_df['accuracy'].mean(), color='red', linestyle='--', label='Empirical Mean')
    ax.axvline(null_df['accuracy'].mean(), color='blue', linestyle='--', label='Null Mean')
    ax.set_title(f'Gap {gap}', fontsize=20)
    ax.set_xlabel(None)
    # ax.set_ylabel('Density', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.set_ylim(0, 20)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlim(0.15, 0.85)
    ax.set_xticks([0.25, 0.50, 0.75])

model_type = 'snapshot'
loocv = 'lopocv'
all_results_df = get_all_results_df(model_type, loocv)

print('Computing participant-weighted accuracy...')
participant_weighted_df = compute_participant_weighted_accuracy(all_results_df)

print('Computing run-weighted accuracy...')
run_weighted_df = compute_run_weighted_accuracy(all_results_df)

print('Computing segment-weighted accuracy...')
segment_weighted_df = compute_segment_weighted_accuracy(all_results_df)

save_dir = f'/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/{model_type}/{loocv}/plots'
os.makedirs(save_dir, exist_ok=True)

print('Plotting participant heatmaps...')
plot_participant_heatmaps(all_results_df, save_dir=save_dir)

print('Plotting accuracy and confidence intervals...')
plot_accuracy_and_CI(
    participant_weighted_df,
    metric='participant_weighted_accuracy',
    save_dir=save_dir
)
plot_accuracy_and_CI(
    run_weighted_df,
    metric='run_weighted_accuracy',
    save_dir=save_dir
)
plot_accuracy_and_CI(
    segment_weighted_df,
    metric='segment_weighted_accuracy',
    save_dir=save_dir
)
plot_distribution_separation(participant_weighted_df, run_weighted_df, segment_weighted_df, save_dir=save_dir)

print('Plotting individual participant distributions...')
for participant in sorted(all_results_df['participant'].unique()):
    participant_df = get_participant_df(all_results_df, participant)
    count_df = get_count_df(all_results_df)
    count = int(count_df[count_df['participant'] == participant]['pts'].values[0])

    segment_weighted_df = compute_segment_weighted_accuracy(participant_df)
    plot_all_empirical_and_null_distributions(
        segment_weighted_df,
        count,
        aggregate_type=participant,
        save_dir=os.path.join(save_dir, 'participants_distributions')
    )