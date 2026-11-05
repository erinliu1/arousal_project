from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import welch


RAW_DATA_DIR = Path('raw_data')
CLASSIFIER_DIR = Path('plots/EEG')
SEGMENT_CONFIG = 'R(W=30,S=1,N_min=8,G_max=4):NR(T_min=20):Segment(TR=40,SW=40)'
SEGMENT_METADATA_PATH = Path('plots/GRU/segments') / SEGMENT_CONFIG / 'all_segment_metadata.csv'
EEG_CHANNELS = ('frontal', 'central', 'occipital')
FREQUENCY_BANDS = {
    'delta_power': (1, 4),
    'theta_power': (5, 7),
    'alpha_power': (8, 11.5),
    'sigma_power': (12, 16),
    'beta_power': (18, 29.5),
}
SLEEP_STAGES = ('Wake', 'N1', 'N2', 'N3', 'REM', 'Art', 'Unscored')
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
    """Load the selected GRU labels, retain EEG runs, and resave EEG metadata."""
    required = ['participant', 'run', 'label', 'segment_start_TR', 'segment_end_TR_exclusive']
    segment_data = pd.read_csv(SEGMENT_METADATA_PATH, usecols=required)
    lengths = segment_data['segment_end_TR_exclusive'] - segment_data['segment_start_TR']
    if not lengths.eq(40).all() or not set(segment_data['label']).issubset({'responsive', 'non-responsive'}):
        raise ValueError(f'Unexpected segment metadata in {SEGMENT_METADATA_PATH}')

    segment_data = segment_data.merge(get_eeg_participant_runs(), on=['participant', 'run'], how='inner')
    segment_data = segment_data.sort_values(['participant', 'run', 'segment_start_TR']).reset_index(drop=True)
    segment_data.to_csv(CLASSIFIER_DIR / 'all_segment_metadata.csv', index=False)
    return segment_data

def load_run_eeg(participant, run):
    """Load three aligned EEG channels and infer their sampling frequency."""
    directory = RAW_DATA_DIR / participant / run / 'sleep_scoring'
    eeg_t = np.load(directory / 'eeg_t.npy')
    eeg = np.stack([np.load(directory / f'{channel}_eeg.npy') for channel in EEG_CHANNELS])
    if eeg.ndim != 2 or eeg.shape[1] != len(eeg_t) or not np.isfinite(eeg).all():
        raise ValueError(f'Invalid EEG arrays for {participant} {run}')
    fs = round(1 / np.median(np.diff(eeg_t)))
    return eeg, eeg_t, fs

def calculate_band_power(eeg, fs):
    """Calculate log-absolute and relative band power for every EEG channel."""
    nperseg = min(4 * fs, eeg.shape[1])
    frequencies, psd = welch(eeg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=1)
    total_mask = (frequencies >= 1) & (frequencies <= 30)
    total_power = np.trapezoid(psd[:, total_mask], frequencies[total_mask], axis=1)
    features = {}
    for channel_index, channel in enumerate(EEG_CHANNELS):
        for name, (low, high) in FREQUENCY_BANDS.items():
            band_mask = (frequencies >= low) & (frequencies <= high)
            absolute_power = np.trapezoid(psd[channel_index, band_mask], frequencies[band_mask])
            prefix = f'{channel}_{name.removesuffix("_power")}'
            features[f'{prefix}_log_absolute_power'] = np.log10(max(absolute_power, np.finfo(float).tiny))
            features[f'{prefix}_relative_power'] = absolute_power / total_power[channel_index]
    return features

def save_segment_eeg_features(metadata=None):
    """Calculate log-absolute and relative band power per channel and segment."""
    metadata = load_all_segment_metadata() if metadata is None else metadata.copy()
    feature_rows = []
    for (participant, run), segments in metadata.groupby(['participant', 'run'], sort=False):
        eeg, eeg_t, fs = load_run_eeg(participant, run)
        time_vector = np.load(RAW_DATA_DIR / participant / run / 'time_vector.npy')
        tr = float(np.median(np.diff(time_vector)))
        for segment in segments.itertuples(index=False):
            start_time = float(time_vector[segment.segment_start_TR])
            end_tr = segment.segment_end_TR_exclusive
            end_time = float(time_vector[end_tr]) if end_tr < len(time_vector) else float(time_vector[-1] + tr)
            start = int(np.searchsorted(eeg_t, start_time, side='left'))
            end = int(np.searchsorted(eeg_t, end_time, side='left'))
            if start_time < eeg_t[0] or end_time > eeg_t[-1] + 1 / fs or end - start < 4 * fs:
                raise ValueError(f'EEG does not cover segment {participant} {run} TRs {segment.segment_start_TR}:{end_tr}')
            feature_rows.append(calculate_band_power(eeg[:, start:end], fs))

    features = pd.concat([metadata.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)
    features.to_csv(CLASSIFIER_DIR / 'segment_eeg_features.csv', index=False)
    return features

def add_within_participant_standardized_power(features):
    """Z-score each log-power feature within participant across all eligible segments."""
    log_columns = [f'{channel}_{band.removesuffix("_power")}_log_absolute_power'
                   for channel in EEG_CHANNELS for band in FREQUENCY_BANDS]
    grouped = features.groupby('participant')[log_columns]
    means = grouped.transform('mean')
    standard_deviations = grouped.transform(lambda values: values.std(ddof=0))
    if standard_deviations.eq(0).any().any():
        raise ValueError('Cannot standardize a participant/channel/band with zero variance')
    standardized = (features[log_columns] - means) / standard_deviations
    standardized.columns = [column.replace('_log_absolute_power', '_within_participant_z')
                            for column in log_columns]
    features = pd.concat([features, standardized], axis=1)
    features.to_csv(CLASSIFIER_DIR / 'segment_eeg_features.csv', index=False)
    return features

def save_within_participant_r_minus_nr_differences(features):
    """Save participant-level mean responsive minus non-responsive z-scores."""
    z_columns = [column for column in features if column.endswith('_within_participant_z')]
    participants = pd.Index(sorted(features['participant'].unique()), name='participant')
    means = features.groupby(['participant', 'label'])[z_columns].mean()
    responsive = means.xs('responsive', level='label').reindex(participants)
    nonresponsive = means.xs('non-responsive', level='label').reindex(participants)
    differences = responsive - nonresponsive
    differences.columns = [column.replace('_within_participant_z', '_r_minus_nr_delta')
                           for column in z_columns]
    counts = features.groupby(['participant', 'label']).size().unstack(fill_value=0).reindex(participants)
    counts = counts.reindex(columns=['responsive', 'non-responsive'], fill_value=0)
    counts.columns = ['n_responsive', 'n_nonresponsive']
    results = pd.concat([counts, differences], axis=1).reset_index()
    results.to_csv(CLASSIFIER_DIR / 'within_participant_r_minus_nr_differences.csv', index=False)
    return results

def save_segment_sleep_stage_proportions(metadata=None):
    """Save the proportion of every segment overlapping each hypnogram stage."""
    metadata = load_all_segment_metadata() if metadata is None else metadata.copy()
    proportion_rows = []
    for (participant, run), segments in metadata.groupby(['participant', 'run'], sort=False):
        run_dir = RAW_DATA_DIR / participant / run
        time_vector = np.load(run_dir / 'time_vector.npy')
        tr = float(np.median(np.diff(time_vector)))
        with open(run_dir / 'sleep_scoring' / 'hypno.json') as file:
            hypnogram = json.load(file)
        unknown = set(hypnogram) - set(SLEEP_STAGES)
        if unknown:
            raise ValueError(f'Unknown sleep stages for {participant} {run}: {sorted(unknown)}')
        for segment in segments.itertuples(index=False):
            start_time = float(time_vector[segment.segment_start_TR])
            end_tr = segment.segment_end_TR_exclusive
            end_time = float(time_vector[end_tr]) if end_tr < len(time_vector) else float(time_vector[-1] + tr)
            duration = end_time - start_time
            proportions = {}
            for stage in SLEEP_STAGES[:-1]:
                overlap = sum(max(0, min(end_time, end) - max(start_time, start))
                              for start, end in hypnogram.get(stage, []))
                proportions[f'{stage}_proportion'] = overlap / duration
            covered = sum(proportions.values())
            if covered > 1 + 1e-6:
                raise ValueError(f'Overlapping hypnogram stages for {participant} {run}')
            proportions['Unscored_proportion'] = max(0, 1 - covered)
            proportion_rows.append(proportions)

    results = pd.concat([metadata.reset_index(drop=True), pd.DataFrame(proportion_rows)], axis=1)
    results.to_csv(CLASSIFIER_DIR / 'segment_sleep_stage_proportions.csv', index=False)
    return results

def plot_sleep_stage_proportions(stage_proportions=None):
    """Plot mean segment-level sleep-stage proportions for R and NR segments."""
    import matplotlib.pyplot as plt

    if stage_proportions is None:
        stage_proportions = pd.read_csv(CLASSIFIER_DIR / 'segment_sleep_stage_proportions.csv')
    columns = [f'{stage}_proportion' for stage in SLEEP_STAGES]
    means = stage_proportions.groupby('label')[columns].mean().rename(columns=lambda name: name.removesuffix('_proportion'))
    means.to_csv(CLASSIFIER_DIR / 'mean_sleep_stage_proportions_by_label.csv')
    stages = [stage for stage in SLEEP_STAGES if means[stage].max() > 1e-8]
    colors = {'Wake': '#69B85A', 'N1': '#8EC5E8', 'N2': '#4C78A8', 'N3': '#263B80',
              'REM': '#8B6BB1', 'Art': '#D16D6A', 'Unscored': '#B8B8B8'}
    groups = (('responsive', 'Responsive (R)'), ('non-responsive', 'Non-responsive (NR)'))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (label, title) in zip(axes, groups):
        values = means.loc[label, stages].to_numpy()
        wedges, _, percentage_texts = ax.pie(
            values, colors=[colors[stage] for stage in stages], startangle=90, counterclock=False,
            autopct=lambda pct: f'{pct:.1f}%' if pct >= 0.05 else '', pctdistance=0.65,
            wedgeprops={'edgecolor': 'none', 'linewidth': 0},
            textprops={'fontsize': 15, 'fontweight': 'bold'},
        )
        small_labels = [(wedge, text) for value, wedge, text in zip(values, wedges, percentage_texts)
                        if text.get_text() and value < 0.10]
        small_labels.sort(key=lambda item: np.cos(np.deg2rad((item[0].theta1 + item[0].theta2) / 2)))
        for text_x, (wedge, percentage_text) in zip(np.linspace(-0.32, 0.32, len(small_labels)), small_labels):
            angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
            percentage_text.set_position((text_x, 1.055))
            percentage_text.set_horizontalalignment('center')
            percentage_text.set_verticalalignment('bottom')
            ax.plot([0.97 * np.cos(angle), text_x], [0.97 * np.sin(angle), 1.04],
                    color=wedge.get_facecolor(), linewidth=2.5)
        count = int((stage_proportions['label'] == label).sum())
        ax.set_title(f'{title}\n{count:,} segments', fontsize=17)
    fig.legend(wedges, stages, loc='lower center', ncol=len(stages), frameon=False, fontsize=17,
               handlelength=2.2, handleheight=1.3, columnspacing=2.2)
    fig.suptitle('Mean Sleep-stage Proportions Across 40-TR Segments', fontsize=21)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    output_path = CLASSIFIER_DIR / 'sleep_stage_proportions_by_behavioral_class.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path

def save_participant_sleep_stage_agreement(stage_proportions=None):
    """Save R Wake and NR sleep agreement over total and valid-scored duration."""
    if stage_proportions is None:
        stage_proportions = pd.read_csv(CLASSIFIER_DIR / 'segment_sleep_stage_proportions.csv')
    participants = pd.Index(sorted(stage_proportions['participant'].unique()), name='participant')
    scored_columns = ['Wake_proportion', 'N1_proportion', 'N2_proportion', 'N3_proportion', 'REM_proportion']
    stage_proportions = stage_proportions.copy()
    stage_proportions['valid_scored_proportion'] = stage_proportions[scored_columns].sum(axis=1)
    responsive = stage_proportions[stage_proportions['label'] == 'responsive'].copy()
    nonresponsive = stage_proportions[stage_proportions['label'] == 'non-responsive'].copy()
    sleep_columns = ['N1_proportion', 'N2_proportion', 'N3_proportion', 'REM_proportion']
    nonresponsive['sleep_proportion'] = nonresponsive[sleep_columns].sum(axis=1)
    responsive['wake_fraction_valid_scored'] = (
        responsive['Wake_proportion'] / responsive['valid_scored_proportion'].replace(0, np.nan)
    )
    nonresponsive['sleep_fraction_valid_scored'] = (
        nonresponsive['sleep_proportion'] / nonresponsive['valid_scored_proportion'].replace(0, np.nan)
    )
    results = pd.DataFrame(index=participants)
    results['mean_r_wake_fraction_total_duration'] = responsive.groupby('participant')['Wake_proportion'].mean()
    results['mean_r_wake_fraction_valid_scored_duration'] = responsive.groupby('participant')['wake_fraction_valid_scored'].mean()
    results['n_r_total_segments'] = responsive.groupby('participant').size()
    results['n_r_valid_scored_segments'] = responsive.groupby('participant')['wake_fraction_valid_scored'].count()
    results['mean_nr_sleep_fraction_total_duration'] = nonresponsive.groupby('participant')['sleep_proportion'].mean()
    results['mean_nr_sleep_fraction_valid_scored_duration'] = nonresponsive.groupby('participant')['sleep_fraction_valid_scored'].mean()
    results['n_nr_total_segments'] = nonresponsive.groupby('participant').size()
    results['n_nr_valid_scored_segments'] = nonresponsive.groupby('participant')['sleep_fraction_valid_scored'].count()
    count_columns = [column for column in results if column.startswith('n_')]
    results[count_columns] = results[count_columns].fillna(0).astype(int)
    results = results.reset_index()
    results.to_csv(CLASSIFIER_DIR / 'participant_sleep_stage_agreement.csv', index=False)
    return results

def plot_power_distributions(features, feature_suffix, y_label, title, filename):
    """Plot responsive versus non-responsive power distributions in a 3x5 grid."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    groups = (('responsive', 'R', '#69B85A'), ('non-responsive', 'NR', '#E78AA8'))
    if set(features['label']) != {group[0] for group in groups}:
        raise ValueError('Expected both responsive and non-responsive EEG segments')

    fig, axes = plt.subplots(len(EEG_CHANNELS), len(FREQUENCY_BANDS), figsize=(22, 12), sharey=True)
    for row, channel in enumerate(EEG_CHANNELS):
        for column, (band_key, limits) in enumerate(FREQUENCY_BANDS.items()):
            band = band_key.removesuffix('_power')
            feature = f'{channel}_{band}_{feature_suffix}'
            data = [features.loc[features['label'] == label, feature].to_numpy() for label, _, _ in groups]
            ax = axes[row, column]
            violin = ax.violinplot(data, positions=[1, 2], widths=0.8, showmeans=False,
                                   showmedians=True, showextrema=False)
            for body, (_, _, color) in zip(violin['bodies'], groups):
                body.set_facecolor(color)
                body.set_edgecolor('black')
                body.set_alpha(0.8)
            violin['cmedians'].set_color('black')
            violin['cmedians'].set_linewidth(1.5)
            ax.set_xticks([1, 2], [group[1] for group in groups])
            ax.grid(axis='y', color='0.88', linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            if row == 0:
                ax.set_title(f'{band.title()} ({limits[0]}–{limits[1]} Hz)', fontsize=15)
            if column == 0:
                ax.set_ylabel(f'{channel.title()}\n{y_label}', fontsize=14)
            ax.tick_params(labelsize=12)

    legend = [Patch(facecolor=color, edgecolor='black', label=name, alpha=0.8)
              for _, name, color in groups]
    fig.legend(handles=legend, loc='upper center', ncol=2, frameon=False, fontsize=14,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(title, fontsize=22, y=0.99)
    fig.supxlabel('Behavioral label', fontsize=16)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    output_path = CLASSIFIER_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path

def plot_log_absolute_power_distributions(features=None):
    features = pd.read_csv(CLASSIFIER_DIR / 'segment_eeg_features.csv') if features is None else features
    return plot_power_distributions(features, 'log_absolute_power', 'Log$_{10}$ absolute power',
                                    'Responsive vs. Non-responsive EEG Log Absolute Power',
                                    'log_absolute_power_distributions.png')

def plot_standardized_power_distributions(features=None):
    features = pd.read_csv(CLASSIFIER_DIR / 'segment_eeg_features.csv') if features is None else features
    return plot_power_distributions(features, 'within_participant_z', 'Within-participant z-score',
                                    'Responsive vs. Non-responsive Within-participant Standardized EEG Power',
                                    'standardized_power_distributions.png')

def plot_segment_counts_by_participant():
    import matplotlib.pyplot as plt

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
    metadata = load_all_segment_metadata()
    features = save_segment_eeg_features(metadata)
    features = add_within_participant_standardized_power(features)
    differences = save_within_participant_r_minus_nr_differences(features)
    stage_proportions = save_segment_sleep_stage_proportions(metadata)
    agreement = save_participant_sleep_stage_agreement(stage_proportions)
    stage_plot_path = plot_sleep_stage_proportions(stage_proportions)
    log_plot_path = plot_log_absolute_power_distributions(features)
    standardized_plot_path = plot_standardized_power_distributions(features)
    print(f'Saved 45 EEG features for {len(features)} segments')
    print(f'Saved R-NR differences for {len(differences)} participants')
    print(f'Saved sleep-stage proportions for {len(stage_proportions)} segments to {stage_plot_path}')
    print(f'Saved sleep-stage agreement for {len(agreement)} participants')
    print(f'Saved distribution plots to {log_plot_path} and {standardized_plot_path}')
