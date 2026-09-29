from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyze_non_responsiveness import get_participant_run_bp_intervals, get_participant_bp_intervals

SAVE_PATH = Path('plots/analyze_responsiveness')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

RAW_DATA = Path('raw_data')

INTERVAL_FREQ_PATH = SAVE_PATH / 'interval_freq'
INTERVAL_FREQ_PATH.mkdir(parents=True, exist_ok=True)

def get_interval_frequency_distribution(nonresponsive_threshold=20):
    # for each participant, calculate the fraction of intervals that are equal to 1, 2, ..., nonresponsive_threshold-1, computed after removing sustained non-responsive intervals. 
    # Return a dataframe with columns = 'participant', 'interval', 'fraction'

    df = pd.read_csv(Path('plots/analyze_button_presses') / 'bp_intervals.csv')
    participant_bp_intervals = get_participant_bp_intervals(df)

    max_interval = nonresponsive_threshold - 1
    interval_lengths = np.arange(1, max_interval + 1)

    rows = []
    for _, row in participant_bp_intervals.iterrows():
        participant = row["participant"]
        intervals = np.asarray(row["intervals"])
        intervals = intervals[intervals < nonresponsive_threshold] # remove sustained non-responsive intervals
        if len(intervals) == 0:
            continue
        for k in interval_lengths:
            fraction = np.mean(intervals == k)
            rows.append({
                "participant": participant,
                "interval": k,
                "fraction": fraction
            })
    df = pd.DataFrame(rows)
    df.to_csv(INTERVAL_FREQ_PATH / 'interval_frequency_distribution.csv', index=False)
    return df

def plot_interval_frequency_distribution(nonresponsive_threshold=20):
    df = get_interval_frequency_distribution(nonresponsive_threshold=nonresponsive_threshold)

    participants = df['participant'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(participants)))
    participant_colors = dict(zip(participants, colors))

    fig, ax = plt.subplots(figsize=(8, 4))
    for participant, group in df.groupby("participant"):
        ax.plot(
            group["interval"],
            group["fraction"],
            marker="o",
            alpha=0.7,
            color=participant_colors[participant],
            label=participant
        )

    max_interval = nonresponsive_threshold - 1
    max_y = df.loc[df["interval"] == max_interval, "fraction"].max()
    ax.axhline(max_y, linestyle="--", color="grey", linewidth=1, alpha=0.7)
    ax.text(
        0.82, 0.1,
        f"max fraction @ {max_interval} TRs = {max_y * 100:.2f}%",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="grey"
    )
    ax.set_xlabel("Button press interval (TRs)")
    ax.set_ylabel("Fraction of intervals of this length")
    ax.set_xticks(np.arange(1, max_interval + 1))
    ax.set_title(f"Distribution of BP intervals < {nonresponsive_threshold} TRs")
    ax.legend(
        loc="upper right",
        ncol=2,
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(INTERVAL_FREQ_PATH / 'interval_frequency_distribution.png', dpi=300)
    plt.show()
    plt.close()

# plot_interval_frequency_distribution(nonresponsive_threshold=20)

INTERVAL_QUANTILE_PATH = SAVE_PATH / 'interval_quantiles'
INTERVAL_QUANTILE_PATH.mkdir(parents=True, exist_ok=True)

def summarize_interval_quantiles(nonresponsive_threshold=20):
    """After removing intervals at or above the non-response threshold,
    compute the median and selected upper quantiles.
    """
    df = pd.read_csv(Path('plots/analyze_button_presses') / 'bp_intervals.csv')
    participant_df = get_participant_bp_intervals(df)
    rows = []
    for _, row in participant_df.iterrows():
        participant = row["participant"]
        intervals = np.asarray(row["intervals"])
        intervals = intervals[intervals < nonresponsive_threshold]
        if len(intervals) == 0:
            continue
        rows.append({
            "participant": participant,
            "n_intervals": len(intervals),
            "median": np.quantile(intervals, 0.50),
            "q75": np.quantile(intervals, 0.75),
            "q90": np.quantile(intervals, 0.90),
            "q95": np.quantile(intervals, 0.95)
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(INTERVAL_QUANTILE_PATH / "interval_quantiles_by_participant.csv", index=False)
    return out_df

def plot_interval_quantiles(nonresponsive_threshold=20):
    df = summarize_interval_quantiles(nonresponsive_threshold=nonresponsive_threshold)
    df['participant_num'] = (df['participant'].str.split('_').str[-1].astype(int))
    df = df.sort_values('participant_num')
    fig, ax = plt.subplots(figsize=(8, 4))
    # ax.bar(df["participant_num"], df["q95"], label="95th percentile", alpha=0.5, color='lightblue')
    ax.bar(df["participant_num"], df["q90"], label="90th percentile", alpha=0.5, color='lightblue')
    # ax.bar(df["participant_num"], df["q75"], label="75th percentile")
    ax.bar(df["participant_num"], df["median"], label="Median")
    ax.set_xlabel("Participant")
    ax.set_ylabel("BP interval length (TRs)")
    ax.set_xticks(df["participant_num"])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(INTERVAL_QUANTILE_PATH / "interval_quantiles_by_participant.png", dpi=300)
    plt.close()

def get_participants_with_high_quartile(quartile='q95', responsive_threshold=5, nonresponsive_threshold=20):
    df = summarize_interval_quantiles(nonresponsive_threshold=nonresponsive_threshold)
    if quartile not in ['median', 'q75', 'q90', 'q95']:
        raise ValueError("quartile must be one of 'median', 'q75', 'q90', or 'q95'")
    high_quartile_df = df[df[quartile] > responsive_threshold]
    return high_quartile_df['participant'].tolist()

# plot_interval_quantiles() 

TIMELINE_SAVE_PATH = SAVE_PATH / 'interval_timelines'
TIMELINE_SAVE_PATH.mkdir(parents=True, exist_ok=True)

def get_run_bp_interval_timeline(participant, run):
    """Represent consecutive unique button presses as a run-level timeline.
    Each returned row identifies a gap's endpoints, midpoint,
    and duration; runs with fewer than two presses yield an empty table.
    """
    run_dir = RAW_DATA / participant / run
    time_vector = np.load(run_dir / 'time_vector.npy')
    bp_TRs = np.load(run_dir / 'bp_TRs.npy')
    # keep only TRs within fMRI run
    bp_TRs = bp_TRs[bp_TRs < len(time_vector)]
    # collapse multiple presses in the same TR
    bp_TRs = np.unique(bp_TRs)
    if len(bp_TRs) < 2:
        return pd.DataFrame(columns=['participant', 'run', 'start_TR', 'end_TR', 'midpoint_TR', 'interval'])
    intervals = np.diff(bp_TRs)
    df = pd.DataFrame({
        'participant': participant,
        'run': run,
        'start_TR': bp_TRs[:-1],
        'end_TR': bp_TRs[1:],
        'midpoint_TR': (bp_TRs[:-1] + bp_TRs[1:]) / 2,
        'interval': intervals
    })
    return df

def plot_bp_interval_timeline(participant, run, nonresponsive_threshold=20):
    """Plot gaps occur within one run. Gaps are positioned at their temporal midpoint."""
    df = get_run_bp_interval_timeline(participant, run)
    if len(df) == 0:
        print(f'{participant} {run}: fewer than 2 unique button presses')
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df['midpoint_TR'], df['interval'], s=18, alpha=0.7)
    ax.axhline(
        nonresponsive_threshold,
        linestyle='--',
        linewidth=1.5,
        label=f'Non-responsive threshold = {nonresponsive_threshold} TRs'
    )
    ax.set_xlabel('TR')
    ax.set_ylabel('BP interval length (TRs)')
    ax.set_title(f'{participant}, {run}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(TIMELINE_SAVE_PATH / f'{participant}_{run}_bp_interval_timeline.png', dpi=300)
    plt.close()

def plot_all_bp_interval_timelines(participants, nonresponsive_threshold=20):
    for participant in participants:
        participant_dir = RAW_DATA / participant
        for run_dir in participant_dir.iterdir():
            if run_dir.is_dir():
                run = run_dir.name
                plot_bp_interval_timeline(participant=participant, run=run, nonresponsive_threshold=nonresponsive_threshold)

# plot_all_bp_interval_timelines(participants=get_participants_with_high_quartile(quartile='q95', responsive_threshold=5, nonresponsive_threshold=20), nonresponsive_threshold=20)

RUN_SUMMARY_SAVE_PATH = SAVE_PATH / 'run_summaries'
RUN_SUMMARY_SAVE_PATH.mkdir(parents=True, exist_ok=True)

def get_run_interval_summary(nonresponsive_threshold=20):
    """Create run-level diagnostics of responsive and non-responsive gaps.
    For every run, separate the subthreshold gaps from long gaps,
    summarizes the short-gap distribution and fast-response fractions, and
    records the fraction of all gaps exceeding the threshold. The sorted table
    is saved and returned, including explicit empty-run rows.
    """
    rows = []

    for participant_dir in RAW_DATA.iterdir():
        if not participant_dir.is_dir():
            continue
        participant = participant_dir.name
        for run_dir in participant_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run = run_dir.name
            df = get_run_bp_interval_timeline(participant, run)

            n_intervals, n_below_20, median, q75, q90, q95, fraction_le_2, fraction_le_3, fraction_le_4, fraction_le_5, fraction_ge_20 = (None, None, None, None, None, None, None, None, None, None, None)

            if len(df) == 0:
                n_intervals = 0
                n_below_20 = 0
            else:
                intervals = df['interval'].to_numpy()
                subthreshold = intervals[intervals < nonresponsive_threshold]

                n_intervals = len(intervals)
                n_below_20 = len(subthreshold)
                fraction_ge_20 = np.mean(intervals >= nonresponsive_threshold)

                if len(subthreshold) > 0:
                    median = np.quantile(subthreshold, 0.50)
                    q75 = np.quantile(subthreshold, 0.75)
                    q90 = np.quantile(subthreshold, 0.90)
                    q95 = np.quantile(subthreshold, 0.95)
                    fraction_le_2 = np.mean(subthreshold <= 2)
                    fraction_le_3 = np.mean(subthreshold <= 3)
                    fraction_le_4 = np.mean(subthreshold <= 4)
                    fraction_le_5 = np.mean(subthreshold <= 5)

            rows.append({
                'participant': participant,
                'run': run,
                'n_intervals': n_intervals,
                'n_below_20': n_below_20,
                'median': median,
                'q75': q75,
                'q90': q90,
                'q95': q95,
                'fraction_le_2': fraction_le_2,
                'fraction_le_3': fraction_le_3,
                'fraction_le_4': fraction_le_4,
                'fraction_le_5': fraction_le_5,
                'fraction_ge_20': fraction_ge_20
            })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(RUN_SUMMARY_SAVE_PATH /'run_interval_summary.csv', index=False)
    return out_df

def plot_run_interval_quantiles(participant):
    """Show how a participant's subthreshold-gap quantiles change across runs.
    Runs without subthreshold intervals are omitted; the remaining median,
    75th, and 90th percentiles are plotted.
    """
    df = get_run_interval_summary()
    participant_df = (df[df['participant'] == participant]).copy()
    participant_df['run_num'] = (participant_df['run'].str.extract(r'(\d+)')[0].astype(int))
    participant_df.sort_values('run_num', inplace=True)
    participant_df = participant_df[participant_df['n_below_20'] > 0]

    if len(participant_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(participant_df['run_num'], participant_df['median'], marker='o', linewidth=2, label='Median')
    ax.plot(participant_df['run_num'], participant_df['q75'], marker='o', linewidth=1.5, label='75th percentile')
    ax.plot(participant_df['run_num'], participant_df['q90'], marker='o', linewidth=1.5, label='90th percentile')
    ax.set_xlabel('Run')
    ax.set_ylabel('BP interval length (TRs)')
    ax.set_title(participant)
    ax.set_xticks(participant_df['run_num'])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUN_SUMMARY_SAVE_PATH / f'{participant}_run_interval_quantiles.png', dpi=300)
    plt.close()

def plot_run_fast_response_fraction(participant):
    """Compare the prevalence of very fast responses across a participant's runs.
    For each run, the plot overlays the fractions of responsive gaps no
    longer than 2, 3, 4, and 5 TRs.
    """
    df = get_run_interval_summary()
    participant_df = (df[df['participant'] == participant]).copy()
    participant_df['run_num'] = (participant_df['run'].str.extract(r'(\d+)')[0].astype(int))
    participant_df.sort_values('run_num', inplace=True)
    participant_df = participant_df[participant_df['n_below_20'] > 0]

    if len(participant_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for k in [2, 3, 4, 5]:
        ax.plot(participant_df['run_num'], participant_df[f'fraction_le_{k}'], marker='o', label=f'≤ {k} TRs')
    ax.set_xlabel('Run')
    ax.set_ylabel('Fraction of intervals')
    ax.set_title(participant)
    ax.set_xticks(participant_df['run_num'])
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUN_SUMMARY_SAVE_PATH / f'{participant}_fast_response_fraction_by_run.png', dpi=300)
    plt.close()

# for participant in get_participants_with_high_quartile(quartile='q95', responsive_threshold=5, nonresponsive_threshold=20):
    # plot_run_interval_quantiles(participant)
    # plot_run_fast_response_fraction(participant)

RESPONSE_DENSITY_PATH = SAVE_PATH / 'response_density'
RESPONSE_DENSITY_PATH.mkdir(parents=True, exist_ok=True)

def get_response_density(window_size=30):
    """
    Every participant/run is divided into full, non-overlapping ``window_size``-TR blocks.
    Response density per block = the number of TRs containing a press (multiple presses in the same TR are collapsed) divided by window length. Partial trailing blocks are excluded.
    """
    rows = []
    for participant_dir in RAW_DATA.iterdir():
        if not participant_dir.is_dir():
            continue
        participant = participant_dir.name
        for run_dir in participant_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run = run_dir.name
            time_vector = np.load(run_dir / 'time_vector.npy')
            bp_TRs = np.load(run_dir / 'bp_TRs.npy')

            bp_TRs = np.unique(bp_TRs[
                (bp_TRs >= 20) & (bp_TRs < len(time_vector))])

            # only full windows
            for start_TR in range(20, len(time_vector) - window_size + 1, window_size):
                end_TR = start_TR + window_size
                n_presses = np.sum((bp_TRs >= start_TR) & (bp_TRs < end_TR))
                density = n_presses / window_size
                rows.append({
                    'participant': participant,
                    'run': run,
                    'start_TR': start_TR,
                    'end_TR': end_TR,
                    'n_presses': n_presses,
                    'response_density': density
                })
    df = pd.DataFrame(rows)
    df['participant_num'] = df['participant'].str.split('_').str[-1].astype(int)
    df['run_num'] = df['run'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values(['participant_num', 'run_num', 'start_TR']).reset_index(drop=True)
    df.drop(columns=['participant_num', 'run_num'], inplace=True)
    df.to_csv(RESPONSE_DENSITY_PATH / 'response_density.csv', index=False)
    return df

def summarize_response_density(window_size=30):
    df = get_response_density(window_size=window_size)
    summary = (
        df.groupby('participant')['response_density']
        .agg(
            median='median',
            q75=lambda x: np.quantile(x, 0.75),
            q80=lambda x: np.quantile(x, 0.80),
            q90=lambda x: np.quantile(x, 0.90),
            q95=lambda x: np.quantile(x, 0.95),
            maximum='max',
            n_windows='count'
        )
        .reset_index()
    )
    summary['participant_num'] = summary['participant'].str.split('_').str[-1].astype(int)
    summary = summary.sort_values('participant_num')
    summary.to_csv(RESPONSE_DENSITY_PATH / f'response_density_summary.csv', index=False)
    return summary

def get_response_density_overlapping(window_size=30, step_size=10):
    rows = []
    for participant_dir in RAW_DATA.iterdir():
        if not participant_dir.is_dir():
            continue
        participant = participant_dir.name
        for run_dir in participant_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run = run_dir.name
            time_vector = np.load(run_dir / 'time_vector.npy')
            bp_TRs = np.load(run_dir / 'bp_TRs.npy')

            bp_TRs = np.unique(bp_TRs[
                (bp_TRs >= 20) & (bp_TRs < len(time_vector))])

            for start_TR in range(20, len(time_vector) - window_size + 1, step_size):
                end_TR = start_TR + window_size
                n_presses = np.sum((bp_TRs >= start_TR) & (bp_TRs < end_TR))
                rows.append({
                    'participant': participant,
                    'run': run,
                    'start_TR': start_TR,
                    'end_TR': end_TR,
                    'midpoint_TR': (start_TR + end_TR) / 2,
                    'n_presses': n_presses,
                    'response_density': n_presses / window_size
                })
    df = pd.DataFrame(rows)
    df['participant_num'] = df['participant'].str.split('_').str[-1].astype(int)
    df['run_num'] = df['run'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values(['participant_num', 'run_num', 'start_TR']).reset_index(drop=True)
    df.drop(columns=['participant_num', 'run_num'], inplace=True)
    df.to_csv(RESPONSE_DENSITY_PATH / 'response_density_overlapping.csv', index=False)
    return df

# summarize_response_density()

RESPONSIVE_WINDOWS_PATH = SAVE_PATH / 'responsive_windows'
RESPONSIVE_WINDOWS_PATH.mkdir(parents=True, exist_ok=True)

def get_responsive_windows(window_size=30, step_size=10, min_density=0.25, max_gap=5):
    """
    Slide a window of size 'window_size' in TRs across the run, with step size of 10 TRs.
    A window is considered responsive iff:
    - press density (i.e. # of TRs with at least one press / window size) >= min_density
    - the largest gap between presses (including gaps to the window edges) <= max_gap
    """
    df = get_response_density_overlapping(window_size=window_size, step_size=step_size)
    rows = []
    for _, row in df.iterrows():
        participant = row["participant"]
        run = row["run"]
        start_TR = int(row["start_TR"])
        end_TR = int(row["end_TR"])

        run_dir = RAW_DATA / participant / run
        time_vector = np.load(run_dir / "time_vector.npy")
        bp_TRs = np.load(run_dir / "bp_TRs.npy")

        bp_TRs = np.unique(bp_TRs[
                (bp_TRs >= 20) & (bp_TRs < len(time_vector))])

        # count button presses inside this window
        window_bps = bp_TRs[(bp_TRs >= start_TR) &(bp_TRs < end_TR)]
        density = len(window_bps) / window_size

        # check maximum gap between presses in this window (including gaps to the window edges)
        if len(window_bps) > 0:
            gaps = np.diff(
                np.concatenate((
                    [start_TR],
                    window_bps,
                    [end_TR]
                ))
            )
            maximum_gap = gaps.max()
        else:
            maximum_gap = window_size

        responsive = (
            density >= min_density
            and maximum_gap <= max_gap
        )

        rows.append({
            **row.to_dict(),
            "max_within_window_gap": maximum_gap,
            "responsive": responsive
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESPONSIVE_WINDOWS_PATH / "responsive_windows.csv", index=False)
    return out

def summarize_responsive_windows(window_size=30, step_size=10, min_density=0.25, max_gap=5):
    """
    For each run, report:
    -  the total number of windows
    - the number of windows classified as responsive
    - the fraction of responsive windows. 
    """
    df = get_responsive_windows(window_size=window_size, step_size=step_size, min_density=min_density, max_gap=max_gap)
    summary = (
        df.groupby(["participant", "run"])
        .agg(
            n_windows=("responsive", "size"),
            n_responsive_windows=("responsive", "sum"),
            fraction_responsive=("responsive", "mean")
        )
        .reset_index()
    )
    summary['participant_num'] = summary['participant'].str.split('_').str[-1].astype(int)
    summary['run_num'] = summary['run'].str.extract(r'(\d+)').astype(int)
    summary = summary.sort_values(['participant_num', 'run_num']).reset_index(drop=True)
    summary.drop(columns=['participant_num', 'run_num'], inplace=True)
    summary.to_csv(RESPONSIVE_WINDOWS_PATH / "responsive_window_summary.csv", index=False)
    return summary

def plot_responsive_windows(window_size=30, step_size=10, min_density=0.25, max_gap=5):
    summary_df = summarize_responsive_windows(window_size=window_size, step_size=step_size, min_density=min_density, max_gap=max_gap)
    participants = summary_df['participant'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(participants)))
    participant_colors = dict(zip(participants, colors))

    fig, ax = plt.subplots(figsize=(27, 9))

    x = 0
    participant_centers = []
    participant_labels = []

    for participant_idx, (participant, participant_df) in enumerate(
        summary_df.groupby('participant', sort=False)
    ):
        participant_xs = []
        color = participant_colors[participant]

        for _, row in participant_df.iterrows():
            fraction_responsive = row['fraction_responsive']

            if fraction_responsive == 0:
                ax.scatter(
                    x,
                    0,
                    marker='*',
                    alpha=0.7,
                    s=180,
                    color=color,
                    linewidths=2,
                    zorder=3
                )
            else:
                ax.bar(
                    x,
                    fraction_responsive,
                    alpha=0.7,
                    color=color,
                    width=0.8,
                    zorder=2
                )

            participant_xs.append(x)
            x += 1

        participant_centers.append(np.mean(participant_xs))
        participant_labels.append(int(participant.split('_')[-1]))

    ax.set_xticks(participant_centers)
    ax.set_xticklabels(participant_labels, fontsize=22)
    ax.tick_params(axis='x', width=2, length=8)
    ax.tick_params(axis='y', which='both', length=0, labelsize=22)

    ax.set_xlabel('Participant', fontsize=32, labelpad=15)
    ax.set_ylabel('Fraction of Responsive Windows', fontsize=32, labelpad=15)
    ax.set_title(
        'Responsive Windows by Participant and Run',
        fontsize=34,
        fontweight='bold'
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.3)

    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(RESPONSIVE_WINDOWS_PATH / 'responsive_windows.png', dpi=300, bbox_inches='tight')
    plt.close()

RESPONSIVE_INTERVALS_PATH = SAVE_PATH / 'responsive_intervals'
RESPONSIVE_INTERVALS_PATH.mkdir(parents=True, exist_ok=True)

def get_responsive_intervals(window_size=30, step_size=10, min_density=0.25, max_gap=5):
    # merge overlapping or touching windows geometrically
    df = get_responsive_windows(window_size=window_size, step_size=step_size, min_density=min_density, max_gap=max_gap)

    intervals = []
    for (participant, run), run_df in df.groupby(["participant", "run"]):
        responsive_df = run_df[run_df["responsive"]].sort_values("start_TR").reset_index(drop=True)
        if len(responsive_df) == 0:
            continue
        # Start first responsive segment
        current_start = int(responsive_df.iloc[0]["start_TR"])
        current_end = int(responsive_df.iloc[0]["end_TR"])
        current_windows = [responsive_df.iloc[0]]
        epoch_num = 1
        for i in range(1, len(responsive_df)):
            row = responsive_df.iloc[i]
            start_TR = int(row["start_TR"])
            end_TR = int(row["end_TR"])

            # Merge if windows overlap or touch
            if start_TR <= current_end:
                current_end = max(current_end, end_TR)
                current_windows.append(row)

            else:
                # Save previous responsive segment
                epoch_windows = pd.DataFrame(current_windows)

                intervals.append({
                    "participant": participant,
                    "run": run,
                    "epoch": epoch_num,
                    "start_TR": current_start,
                    "end_TR": current_end,
                    "duration_TRs": current_end - current_start,
                    "n_responsive_windows": len(epoch_windows),
                    "mean_response_density": epoch_windows["response_density"].mean(),
                    "min_response_density": epoch_windows["response_density"].min(),
                    "max_within_window_gap": epoch_windows["max_within_window_gap"].max(),
                })

                epoch_num += 1

                # Start new segment
                current_start = start_TR
                current_end = end_TR
                current_windows = [row]

        # Save final segment
        epoch_windows = pd.DataFrame(current_windows)

        intervals.append({
            "participant": participant,
            "run": run,
            "epoch": epoch_num,
            "start_TR": current_start,
            "end_TR": current_end,
            "duration_TRs": current_end - current_start,
            "n_responsive_windows": len(epoch_windows),
            "mean_response_density": epoch_windows["response_density"].mean(),
            "min_response_density": epoch_windows["response_density"].min(),
            "max_within_window_gap": epoch_windows["max_within_window_gap"].max(),
        })
    intervals_df = pd.DataFrame(intervals)
    intervals_df.to_csv(RESPONSIVE_INTERVALS_PATH / "responsive_intervals.csv", index=False)
    return intervals_df

def summarize_responsive_intervals(window_size=30, step_size=10, min_density=0.25, max_gap=5):
    df = get_responsive_intervals(window_size=window_size, step_size=step_size, min_density=min_density, max_gap=max_gap)

    if len(df) == 0:
        return df

    summary = (
        df
        .groupby(["participant", "run"])
        .agg(
            n_responsive_epochs=("epoch", "size"),
            total_responsive_TRs=("duration_TRs", "sum"),
            longest_responsive_epoch=("duration_TRs", "max"),
            median_responsive_epoch=("duration_TRs", "median"),
        )
        .reset_index()
    )
    summary.to_csv(RESPONSIVE_INTERVALS_PATH / "responsive_intervals_summary.csv", index=False)
    return summary

def scatter_responsive_intervals(window_size=30, step_size=10, min_density=0.25, max_gap=5):
    """Scatter responsive-segment durations for each participant and run.

    All intervals from a run share one x position, participant labels are
    centered beneath their runs, and colors distinguish participants. The
    completed figure is saved in ``RESPONSIVE_INTERVALS_PATH``.
    """
    df = get_responsive_intervals(
        window_size=window_size,
        step_size=step_size,
        min_density=min_density,
        max_gap=max_gap
    )

    # Build the x-axis from every run on disk, not just runs represented in the
    # segment table. This preserves an empty slot when a run has no intervals.
    run_rows = []
    for participant_dir in RAW_DATA.iterdir():
        if not participant_dir.is_dir():
            continue
        for run_dir in participant_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run_rows.append({
                'participant': participant_dir.name,
                'run': run_dir.name
            })

    run_inventory = pd.DataFrame(run_rows)
    if run_inventory.empty:
        print('No participant runs to plot.')
        return

    run_inventory['participant_num'] = (
        run_inventory['participant'].str.split('_').str[-1].astype(int)
    )
    run_inventory['run_num'] = (
        run_inventory['run'].str.extract(r'(\d+)')[0].astype(int)
    )
    run_inventory = run_inventory.sort_values(
        ['participant_num', 'run_num']
    ).reset_index(drop=True)

    participants = run_inventory['participant'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(participants)))
    participant_colors = dict(zip(participants, colors))

    fig, ax = plt.subplots(figsize=(27, 9))

    x = 0
    participant_centers = []
    participant_labels = []

    for participant_idx, (participant, participant_df) in enumerate(
        run_inventory.groupby('participant', sort=False)
    ):
        participant_xs = []
        color = participant_colors[participant]

        for run_idx, (_, run_row) in enumerate(participant_df.iterrows()):
            if df.empty:
                run_intervals = df
            else:
                run_intervals = df[
                    (df['participant'] == participant) &
                    (df['run'] == run_row['run'])
                ]

            positive_intervals = run_intervals[
                run_intervals['duration_TRs'] > 0
            ] if not run_intervals.empty else run_intervals
            zero_intervals = run_intervals[
                run_intervals['duration_TRs'] == 0
            ] if not run_intervals.empty else run_intervals

            if not positive_intervals.empty:
                ax.scatter(
                    np.full(len(positive_intervals), x),
                    positive_intervals['duration_TRs'],
                    alpha=0.7,
                    s=180,
                    color=color
                )

            if not zero_intervals.empty:
                ax.scatter(
                    np.full(len(zero_intervals), x),
                    zero_intervals['duration_TRs'],
                    marker='*',
                    alpha=0.7,
                    s=220,
                    color=color,
                    linewidths=2,
                    zorder=3
                )

            participant_xs.append(x)
            if run_idx < len(participant_df) - 1:
                ax.axvline(x + 0.5, color='lightgray', linewidth=2)
            x += 1

        participant_centers.append(np.mean(participant_xs))
        participant_labels.append(
            participant_df['participant_num'].iloc[0]
        )

        if participant_idx < len(participants) - 1:
            ax.axvline(x - 0.5, color='gray', linewidth=3, alpha=0.6)

    ax.set_xticks(participant_centers)
    ax.set_xticklabels(participant_labels, fontsize=22)

    ax.tick_params(axis='x', width=2, length=8)
    ax.tick_params(axis='y', which='both', length=0, labelsize=22)

    ax.set_xlabel('Participant', fontsize=32, labelpad=15)
    ax.set_ylabel('Responsive Interval Duration (TRs)', fontsize=32, labelpad=15)
    ax.set_title(
        'Responsive Interval Durations by Participant and Run',
        fontsize=34,
        fontweight='bold'
    )

    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        RESPONSIVE_INTERVALS_PATH / 'responsive_intervals.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

if __name__ == "__main__":
    plot_responsive_windows()
    summarize_responsive_intervals()
    scatter_responsive_intervals()
