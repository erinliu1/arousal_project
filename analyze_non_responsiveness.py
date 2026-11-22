from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

SAVE_PATH = Path('plots/analyze_non_responsiveness')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

RAW_DATA = Path('raw_data')

def get_participant_bp_intervals(button_press_intervals_df):
    # pool across all participants, so each row represents one participant and a list of all their button press intervals across all runs in a list (columns = 'participant', 'intervals')
    return button_press_intervals_df.groupby('participant')['interval'].apply(list).reset_index().rename(columns={'interval': 'intervals'})

def get_participant_run_bp_intervals(button_press_intervals_df):
    # pool across all participants, so each row represents one participant and a list of all their button press intervals across all runs in a list (columns = 'participant', 'intervals')
    return button_press_intervals_df.groupby(['participant', 'run'])['interval'].apply(list).reset_index().rename(columns={'interval': 'intervals'})


def plot_nonresponsive_intervals_pooled(bp_TR_snap="nearest", threshold=20):
    """Pool bounded nonresponsive intervals across runs; circles only."""
    from GRU import save_nonresponsive_intervals
    suffix = ",BP=nearest" if bp_TR_snap == "nearest" else ""
    source = Path("plots/GRU/non_responsiveness") / f"T_min={threshold}{suffix}" / "non_responsive_intervals.csv"
    if not source.exists():
        save_nonresponsive_intervals(threshold, bp_TR_snap=bp_TR_snap)
    intervals = pd.read_csv(source)
    participants = sorted([p.name for p in RAW_DATA.iterdir() if p.is_dir()],
                          key=lambda name: int(name.split("_")[-1]))
    colors = dict(zip(participants, plt.cm.tab20(np.linspace(0, 1, len(participants)))))
    fig, ax = plt.subplots(figsize=(20, 6), layout="constrained")
    rng = np.random.default_rng(0)
    for i, participant in enumerate(participants):
        values = intervals.loc[intervals.participant.eq(participant), "duration_TRs"].to_numpy()
        ax.scatter(i + rng.uniform(-.18, .18, len(values)), values,
                   marker="o", s=110, color=colors[participant], alpha=.65, zorder=3)
    ax.set_xticks(range(len(participants)), [str(int(p.split("_")[-1])) for p in participants])
    ax.set_xlim(-.5, len(participants) - .5)
    for boundary in np.arange(.5, len(participants) - .5):
        ax.axvline(boundary, color=".75", linewidth=2, zorder=0)
    ax.set_xlabel("Participant", fontsize=30, labelpad=15)
    ax.set_ylabel("Interval duration (TRs)", fontsize=30, labelpad=15)
    ax.set_title(f"Non-responsive intervals = button press intervals ≥ {threshold} TRs", fontsize=34, pad=18)
    ax.tick_params(axis="both", labelsize=26, width=2.5, length=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.4, linewidth=1.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(2.5)
    destination = Path(f"plots/button_press_analysis,{bp_TR_snap}")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "nonresponsive_intervals_by_participant.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output} ({len(intervals)} bounded intervals)")
    return output


def plot_hazard_and_persistence(bp_TR_snap="nearest"):
    """Regenerate all four summaries from unfiltered within-run BP intervals."""
    global SAVE_PATH
    destination = Path(f"plots/button_press_analysis,{bp_TR_snap}")
    if not (destination / "bp_intervals.csv").exists():
        plot_pooled_bp_intervals(bp_TR_snap)
    intervals = pd.read_csv(destination / "bp_intervals.csv")
    pooled = get_participant_bp_intervals(intervals)
    previous = SAVE_PATH
    try:
        SAVE_PATH = destination
        plot_all_participant_hazards(pooled, max_interval=50, min_at_risk=20)
        plot_participant_contributions(pooled, max_interval=50, min_at_risk=20)
        plot_hazard_summary(pooled, max_interval=50, min_at_risk=20)
        plot_persistence_summary(pooled, plot_future_TRs=[10], min_at_risk=20)
    finally:
        SAVE_PATH = previous
    print(f"Saved hazard and persistence plots to {destination}")


#### PARTICIPANT-WISE INTERVAL FREQUENCY DISTRIBUTION: COUNT FRACTION OF INTERVALS OF LENGTH 1, 2, ..., K TRs #####

def get_interval_frequency_distribution(participant_bp_intervals, max_interval=15, output_dir=None):
    # for each participant, calculate the fraction of intervals that are equal to 1, 2, ..., max_interval. Return a dataframe with columns = 'participant', 'interval', 'fraction'
    
    interval_lengths = np.arange(1, max_interval + 1)

    rows = []

    for _, row in participant_bp_intervals.iterrows():
        participant = row["participant"]
        intervals = np.asarray(row["intervals"])

        for k in interval_lengths:
            fraction = np.mean(intervals == k)
            rows.append({
                "participant": participant,
                "interval": k,
                "fraction": fraction
            })
    df = pd.DataFrame(rows)
    destination = SAVE_PATH if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination / 'interval_frequency_distribution.csv', index=False)
    return df

def plot_interval_frequency_distribution(participant_bp_intervals, max_interval=15, output_dir=None, title=None, show=True):
    df = get_interval_frequency_distribution(participant_bp_intervals, max_interval=max_interval, output_dir=output_dir)

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
            label=str(int(str(participant).removeprefix("AAN_")))
        )

    ax.set_xlabel("Button press interval length (TRs)")
    ax.set_ylabel("Fraction of intervals of this length")
    ax.set_xticks(np.arange(1, max_interval + 1))

    ax.legend(
        loc="upper right",
        ncol=2,
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if title:
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
    destination = SAVE_PATH if output_dir is None else Path(output_dir)
    output = destination / 'interval_frequency_distribution.png'
    plt.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close()
    return output

def plot_pooled_bp_intervals(bp_TR_snap="nearest", output_dir=None, max_interval=20):
    """Plot within-run inter-press gaps pooled across runs for each participant.

    Match GRU snapping: deduplicate occupied TRs, discard out-of-run indices.
    End-of-run gaps and runs without two presses are not inter-press intervals.
    Neither responsiveness nor motion filters are applied to this distribution.
    """
    from GRU import load_bp_TRs

    destination = (Path(f"plots/button_press_analysis,{bp_TR_snap}")
                   if output_dir is None else Path(output_dir))
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(RAW_DATA.glob("*/run*")):
        if not run_dir.is_dir() or not (run_dir / "time_vector.npy").exists():
            continue
        n_trs = len(np.load(run_dir / "time_vector.npy"))
        presses = np.unique(load_bp_TRs(run_dir, bp_TR_snap))
        presses = presses[(presses >= 0) & (presses < n_trs)]
        for start, end in zip(presses[:-1], presses[1:]):
            rows.append(dict(participant=run_dir.parent.name, run=run_dir.name,
                             start_TR=int(start), end_TR=int(end), interval=int(end - start)))
    intervals = pd.DataFrame(rows)
    if intervals.empty:
        raise ValueError("No inter-press intervals found")
    intervals.to_csv(destination / "bp_intervals.csv", index=False)
    pooled = get_participant_bp_intervals(intervals)
    pooled["order"] = pooled.participant.str.extract(r"(\d+)$", expand=False).astype(int)
    pooled = pooled.sort_values("order").drop(columns="order").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(18, 10), layout="constrained")
    rng = np.random.default_rng(0)
    colors = plt.cm.tab20(np.linspace(0, 1, len(pooled)))
    for i, row in pooled.iterrows():
        values = np.asarray(row.intervals)
        ax.scatter(i + rng.uniform(-.22, .22, len(values)), values,
                   s=18, alpha=.3, color=colors[i], rasterized=True, zorder=2)
    ax.axhline(20, linestyle="--", color=".4", linewidth=2, label="20 TRs", zorder=1)
    ax.set_yscale("log")
    maximum = max(max(values) for values in pooled.intervals)
    ticks = [value for value in (1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
             if value <= maximum]
    ax.set_yticks(ticks, [str(value) for value in ticks])
    ax.tick_params(axis="y", which="minor", labelleft=False)
    participant_numbers = pooled.participant.str.extract(r"(\d+)$", expand=False).astype(int)
    ax.set_xticks(range(len(pooled)), participant_numbers.astype(str))
    ax.set_xlim(-.5, len(pooled) - .5)
    ax.set_xlabel("Participant", fontsize=26, labelpad=16)
    ax.set_ylabel("Button press interval length (TRs)", fontsize=26, labelpad=16)
    ax.set_title(f"Button-press intervals — {bp_TR_snap} TR snapping", fontsize=28, pad=20)
    ax.tick_params(axis="both", which="major", labelsize=22, width=2, length=8)
    ax.tick_params(axis="y", which="minor", width=1.5, length=4)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=".8", linewidth=1.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(2)
    ax.legend(fontsize=20)
    scatter_path = destination / "bp_intervals_by_participant.png"
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)
    frequency_path = plot_interval_frequency_distribution(
        pooled, max_interval=max_interval, output_dir=destination,
        title=f"Inter-press interval frequency — {bp_TR_snap} TR snapping", show=False)
    print(f"Saved {scatter_path} and {frequency_path}")
    return intervals, scatter_path, frequency_path


#### HAZARD: PROBABILITY THAT AN INTERVAL ENDS AT EXACTLY k TRs GIVEN THAT IT HAS ALREADY LASTED ≥ k TRs #####

def get_press_hazard(intervals, max_interval=50):
    intervals = np.asarray(intervals)

    rows = []

    for k in range(1, max_interval + 1):

        # Intervals that have lasted at least k TRs
        n_at_risk = np.sum(intervals >= k)

        if n_at_risk == 0:
            break

        # Intervals that end exactly at k TRs
        n_end = np.sum(intervals == k)

        hazard = n_end / n_at_risk

        rows.append({
            "TR": k,
            "n_at_risk": n_at_risk,
            "n_end": n_end,
            "hazard": hazard
        })

    return pd.DataFrame(rows)

def plot_hazard(participant_bp_intervals, participant, max_interval=50):
    row = participant_bp_intervals[
        participant_bp_intervals["participant"] == participant
    ].iloc[0]

    hazard_df = get_press_hazard(row["intervals"], max_interval=max_interval)

    hazard_df = hazard_df[hazard_df["n_at_risk"] >= 20]

    plt.figure(figsize=(8, 5))

    plt.plot(
        hazard_df["TR"],
        hazard_df["hazard"],
        marker="o"
    )
    plt.title(participant)
    plt.xlabel("TRs since previous button press")
    plt.ylabel("Probability of button press at this TR")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()
    plt.savefig(SAVE_PATH / f'hazard_{participant}.png', dpi=300)
    plt.close()

def plot_all_participant_hazards(participant_bp_intervals, max_interval=50, min_at_risk=20):
    fig, ax = plt.subplots(figsize=(8, 4))
    participants = sorted(participant_bp_intervals["participant"].unique())
    participant_colors = dict(zip(participants, plt.cm.tab20(np.linspace(0, 1, len(participants)))))

    for _, row in participant_bp_intervals.iterrows():
        participant = row["participant"]

        hazard_df = get_press_hazard(row["intervals"], max_interval=max_interval)

        hazard_df = hazard_df[hazard_df["n_at_risk"] >= min_at_risk]

        ax.plot(
            hazard_df["TR"],
            hazard_df["hazard"],
            marker=None,
            alpha=0.5,
            label=str(int(str(participant).removeprefix("AAN_"))),
            color=participant_colors[participant],
            linewidth=2
        )

    ax.set_xlabel("k = # of TRs")
    ax.set_ylabel("hazard")
    ax.set_title("hazard = P( length of interval is exactly k TRs | length of interval ≥ k TRs )",
                 fontsize=9)
    ax.set_ylim(0, 1)
    
    ax.legend(
        loc="upper right",
        ncol=2
    )
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig(SAVE_PATH / 'hazard_all_participants.png', dpi=300)
    plt.close()

def get_all_participant_hazards(participant_bp_intervals, max_interval=50, min_at_risk=20):
    dfs = []

    for _, row in participant_bp_intervals.iterrows():
        hazard_df = get_press_hazard(row["intervals"], max_interval=max_interval)

        hazard_df["participant"] = row["participant"]

        hazard_df = hazard_df[hazard_df["n_at_risk"] >= min_at_risk] # only include TRs where at least min_at_risk intervals are available

        dfs.append(hazard_df)

    return pd.concat(dfs, ignore_index=True)

def summarize_hazards(participant_bp_intervals, max_interval=50, min_at_risk=20):
    hazard_all = get_all_participant_hazards(participant_bp_intervals, max_interval=max_interval, min_at_risk=min_at_risk)
    hazard_summary = (
        hazard_all
        .groupby("TR")["hazard"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            n_participants="count"
        )
        .reset_index()
    )
    hazard_summary.to_csv(SAVE_PATH / 'hazard_summary.csv', index=False)
    return hazard_summary

def plot_hazard_summary(participant_bp_intervals, max_interval=50, min_at_risk=20):
    hazard_summary = summarize_hazards(participant_bp_intervals, max_interval=max_interval, min_at_risk=min_at_risk)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        hazard_summary["TR"],
        hazard_summary["median"],
        marker="o",
        label="Median across participants",
        color="red"
    )

    ax.fill_between(
        hazard_summary["TR"],
        hazard_summary["q25"],
        hazard_summary["q75"],
        alpha=0.2,
        label="25th–75th percentile",
        color="red"
    )

    ax.set_xlabel("k = # of TRs")
    ax.set_ylabel("hazard")
    ax.set_title("hazard = P( length of interval is exactly k TRs | length of interval ≥ k TRs )",
                 fontsize=9)
    ax.set_ylim(0, 1)

    ax.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_PATH / 'hazard_summary.png', dpi=300)
    plt.close()

def plot_participant_contributions(participant_bp_intervals, max_interval=50, min_at_risk=20):
    hazard_summary = summarize_hazards(participant_bp_intervals, max_interval=max_interval, min_at_risk=min_at_risk)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        hazard_summary["TR"],
        hazard_summary["n_participants"],
        marker="o"
    )

    ax.set_xlabel("k = # of TRs")
    ax.set_ylabel("# of participants with at least\n20 intervals of length ≥ k TRs")
    ax.set_axisbelow(True)
    ax.grid(axis="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_PATH / 'hazard_participant_contributions.png', dpi=300)
    plt.close()

#### PERSISTENCE: PROBABILITY THAT AN INTERVAL CONTINUES FOR ≥ L TRs GIVEN THAT IT HAS ALREADY LASTED ≥ k TRs #####
def get_gap_persistence(intervals, future_TRs=5):
    intervals = np.asarray(intervals)
    thresholds=np.arange(1, 41)
    
    rows = []
    for T in thresholds:
        n_at_risk = np.sum(intervals >= T)

        if n_at_risk == 0:
            continue

        n_persist = np.sum(intervals >= T + future_TRs)

        rows.append({
            "threshold": T,
            "n_at_risk": n_at_risk,
            "p_persist": n_persist / n_at_risk
        })

    return pd.DataFrame(rows)

def get_all_participant_persistence(participant_bp_intervals, future_TRs=5, min_at_risk=20):
    dfs = []
    for _, row in participant_bp_intervals.iterrows():
        df = get_gap_persistence(row["intervals"], future_TRs=future_TRs)

        df["participant"] = row["participant"]

        df = df[df["n_at_risk"] >= min_at_risk] # only include thresholds where at least min_at_risk intervals are available

        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def get_participant_balanced_persistence_summary(participant_bp_intervals, future_TRs=5, min_at_risk=20):
    persistence_all = get_all_participant_persistence(participant_bp_intervals, future_TRs=future_TRs, min_at_risk=min_at_risk)
    persistence_summary = (
        persistence_all
        .groupby("threshold")["p_persist"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            n_participants="count"
        )
        .reset_index()
    )
    persistence_summary.to_csv(SAVE_PATH / f'persistence_summary_L{future_TRs}.csv', index=False)
    return persistence_summary

def plot_persistence_summary(participant_bp_intervals, plot_future_TRs=(10,), min_at_risk=20):
    colors = ['green', 'orange'] # add more colors here
    fig, ax = plt.subplots(figsize=(8, 4))
    for future_TRs in plot_future_TRs:
        persistence_summary = get_participant_balanced_persistence_summary(participant_bp_intervals, future_TRs=future_TRs, min_at_risk=min_at_risk)
        ax.plot(
            persistence_summary["threshold"],
            persistence_summary["median"],
            marker="o",
            label="Median across participants" if len(plot_future_TRs) == 1 else f"Median across participants (L={future_TRs})",
            color="orange" if future_TRs == 10 else colors[plot_future_TRs.index(future_TRs) % len(colors)]
        )

        ax.fill_between(
            persistence_summary["threshold"],
            persistence_summary["q25"],
            persistence_summary["q75"],
            alpha=0.2,
            label="25th–75th percentile",
            color="orange" if future_TRs == 10 else colors[plot_future_TRs.index(future_TRs) % len(colors)]
        )
    ax.axhline(0.7, linestyle='--', color='grey', linewidth=1, alpha=0.7, label='70% persistence threshold')
    ax.legend(loc='lower right')
    ax.set_xlabel("k = # of TRs")
    ax.set_ylabel("persistence")
    horizon = str(plot_future_TRs[0]) if len(plot_future_TRs) == 1 else "L"
    ax.set_title(f"persistence = P( interval continues for ≥ {horizon} TRs | length of interval ≥ k TRs )",
                 fontsize=9)
    ax.set_ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_PATH / f'persistence_summary.png', dpi=300)
    plt.close()

##### THRESHOLD SENSITIVITY: COUNT SURVIVING INTERVALS AND PARTICIPANTS/RUNS WITH ≥1 SURVIVING INTERVAL #####

def get_threshold_sensitivity_summary(participant_run_bp_intervals, thresholds=[15, 20, 25, 30]):
    rows = []

    for threshold in thresholds:

        n_events = 0
        participants_with_events = set()
        runs_with_events = set()

        events_per_run = []
        events_per_participant = {}

        for _, row in participant_run_bp_intervals.iterrows():

            participant = row["participant"]
            run = row["run"]
            intervals = np.asarray(row["intervals"])

            # Number of gaps in this run that meet the threshold
            n_qualifying = np.sum(intervals >= threshold)

            events_per_run.append(n_qualifying)

            if participant not in events_per_participant:
                events_per_participant[participant] = 0

            events_per_participant[participant] += n_qualifying

            if n_qualifying > 0:
                n_events += n_qualifying
                participants_with_events.add(participant)
                runs_with_events.add((participant, run))

        rows.append({
            "threshold": threshold,

            # overall event count
            "n_events": n_events,

            # sample retained
            "n_participants": len(participants_with_events),
            "n_runs": len(runs_with_events),

            # event counts across ALL participants/runs
            "median_events_per_participant": np.median(list(events_per_participant.values())),

            "median_events_per_run": np.median(events_per_run),

            # among only those contributing ≥1 event
            "median_events_per_contributing_participant":
                np.median([
                    n for n in events_per_participant.values()
                    if n > 0
                ]),

            "median_events_per_contributing_run":
                np.median([
                    n for n in events_per_run
                    if n > 0
                ])
        })
    df = pd.DataFrame(rows)
    df.to_csv(SAVE_PATH / 'threshold_sensitivity_summary.csv', index=False)
    return df
    
def get_participant_threshold_counts(participant_run_bp_intervals, thresholds=[15, 20, 25, 30]):
    rows = []
    for participant, group in participant_run_bp_intervals.groupby("participant"):
        all_intervals = np.concatenate(group["intervals"].apply(np.asarray).to_list())
        row = {
            "participant": participant
        }
        for threshold in thresholds:
            row[f"T{threshold}"] = np.sum(all_intervals >= threshold)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(SAVE_PATH / 'threshold_sensitivity_participant_counts.csv', index=False)
    return df

def get_run_threshold_counts(participant_run_bp_intervals, thresholds=[15, 20, 25, 30]):
    rows = []
    for _, row in participant_run_bp_intervals.iterrows():
        intervals = np.asarray(row["intervals"])
        result = {
            "participant": row["participant"],
            "run": row["run"]
        }
        for threshold in thresholds:
            result[f"T{threshold}"] = np.sum(intervals >= threshold)
        rows.append(result)
    df = pd.DataFrame(rows)
    df.to_csv(SAVE_PATH / 'threshold_sensitivity_run_counts.csv', index=False)
    return df

def get_run_threshold_fractions(participant_run_bp_intervals, thresholds=[15, 20, 25, 30]):
    rows = []
    for _, row in participant_run_bp_intervals.iterrows():
        intervals = np.asarray(row["intervals"])
        result = {
            "participant": row["participant"],
            "run": row["run"],
            "n_intervals": len(intervals)
        }
        for threshold in thresholds:
            result[f"T{threshold}_count"] = np.sum(intervals >= threshold)
            result[f"T{threshold}_fraction"] = np.mean(intervals >= threshold)
        rows.append(result)
    df = pd.DataFrame(rows)
    df.to_csv(SAVE_PATH / 'threshold_sensitivity_run_fractions.csv', index=False)
    return df

def plot_total_events_vs_threshold(sensitivity_summary):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(sensitivity_summary["threshold"], sensitivity_summary["n_events"], marker="o", color="blue")
    for x, y in zip(sensitivity_summary["threshold"], sensitivity_summary["n_events"]):
        ax.annotate(
            f"{y}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center"
        )

    ax.set_xlabel("Non-response threshold (TRs)")
    ax.set_ylabel("Number of qualifying intervals")
    ax.set_xticks([15, 20, 25, 30])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(SAVE_PATH / f'threshold_sensitivity_total_events.png', dpi=300)
    plt.show()

def plot_participant_counts_vs_threshold(sensitivity_summary):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(sensitivity_summary["threshold"], sensitivity_summary["n_participants"], marker="o", color="orange")
    for x, y in zip(sensitivity_summary["threshold"], sensitivity_summary["n_participants"]):
        ax.annotate(
            f"{y}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center"
        )
    ax.set_xlabel("Non-response threshold (TRs)")
    ax.set_ylabel("Participants with ≥1 event")
    ax.set_xticks([15, 20, 25, 30])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(SAVE_PATH / f'threshold_sensitivity_participant_counts.png', dpi=300)
    plt.show()
    plt.close()

def plot_run_counts_vs_threshold(sensitivity_summary):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sensitivity_summary["threshold"], sensitivity_summary["n_runs"], marker="o")
    for x, y in zip(sensitivity_summary["threshold"], sensitivity_summary["n_runs"]):
        ax.annotate(
            f"{y}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center"
        )
    ax.set_xlabel("Non-response threshold (TRs)")
    ax.set_ylabel("Runs with ≥1 event")
    ax.set_xticks([15, 20, 25, 30])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(SAVE_PATH / f'threshold_sensitivity_run_counts.png', dpi=300)
    plt.show()
    plt.close()

def get_non_responsive_intervals(threshold=20):
    """
    Extract all forms of run-level non-responsive intervals.

    Behavioral non-responsiveness is defined using the original button-presses. The first 20 TRs are then excluded from the usable fMRI domain.

    Consecutive valid, unique button-press TRs define bounded candidate
    intervals. Runs with zero button presses contribute a whole-run interval,
    while the gap from the final press to the end of the run contributes an
    end interval.

    Only behavioral gaps meeting `threshold` whose usable post-startup
    portion also meets `threshold` are retained.
    """    
    intervals = []
    whole_runs = []
    end_intervals = []

    participant_dirs = sorted(
        (path for path in RAW_DATA.iterdir() if path.is_dir()),
        key=lambda path: int(path.name.split('_')[-1])
    )

    for participant_dir in participant_dirs:
        run_dirs = sorted(
            (path for path in participant_dir.iterdir() if path.is_dir()),
            key=lambda path: int(''.join(filter(str.isdigit, path.name)))
        )

        for run_dir in run_dirs:
            time_vector = np.load(run_dir / 'time_vector.npy')
            bp_TRs = np.load(run_dir / 'bp_TRs.npy')

            bp_TRs = np.unique(bp_TRs[bp_TRs < len(time_vector)])
            
            if len(bp_TRs) == 0: # no button presses, whole run is non-responsive
                run_duration_TRs = len(time_vector) - 20
                if run_duration_TRs >= threshold:
                    whole_runs.append({
                        'participant': participant_dir.name,
                        'run': run_dir.name,
                        'start_TR': 20,
                        'end_TR': len(time_vector), # exclusive
                        'duration_TRs': run_duration_TRs
                    })
                
            if len(bp_TRs) >= 2: # at least two button presses, so there are intervals between them
                interval_starts = bp_TRs[:-1]
                interval_ends = bp_TRs[1:]
                durations = np.diff(bp_TRs)
                qualifying = durations >= threshold

                for start_TR, end_TR, duration_TRs in zip(
                    interval_starts[qualifying], 
                    interval_ends[qualifying], 
                    durations[qualifying]
                ):
                    clipped_start = max(int(start_TR), 20)
                    clipped_end = end_TR
                    usable_duration = clipped_end - clipped_start

                    if usable_duration < threshold:
                        continue

                    intervals.append({
                        'participant': participant_dir.name,
                        'run': run_dir.name,
                        'start_TR': clipped_start,
                        'end_TR': clipped_end, # exclusive
                        'duration_TRs': usable_duration
                    })

            if len(bp_TRs) >= 1: # end intervals from last button press to end of run
                last_bp_TR = bp_TRs[-1]
                behavioral_duration = len(time_vector) - last_bp_TR
                usable_start_TR = max(last_bp_TR, 20)
                usable_duration = len(time_vector) - usable_start_TR
                if behavioral_duration >= threshold and usable_duration >= threshold:
                    end_intervals.append({
                        'participant': participant_dir.name,
                        'run': run_dir.name,
                        'start_TR': usable_start_TR,
                        'end_TR': len(time_vector), # exclusive
                        'duration_TRs': usable_duration
                    })

    columns = ['participant', 'run', 'start_TR', 'end_TR', 'duration_TRs']
    intervals_df = pd.DataFrame(intervals, columns=columns)
    intervals_df.to_csv(SAVE_PATH / 'non_responsive_intervals.csv', index=False)
    
    whole_runs_df = pd.DataFrame(whole_runs, columns=columns)
    whole_runs_df.to_csv(SAVE_PATH / 'non_responsive_whole_runs.csv', index=False)

    end_intervals_df = pd.DataFrame(end_intervals, columns=columns)
    end_intervals_df.to_csv(SAVE_PATH / 'non_responsive_end_intervals.csv', index=False)

    return intervals_df, whole_runs_df, end_intervals_df


def get_all_participants():
    return  pd.DataFrame({
        "participant": [
            path.name
            for path in RAW_DATA.iterdir()
            if path.is_dir()
        ]
    })

def get_participant_colors():
    participants = sorted(
        [
            path.name
            for path in RAW_DATA.iterdir()
            if path.is_dir()
        ],
        key=lambda participant: int(participant.split("_")[-1])
    )

    colors = plt.cm.tab20(np.linspace(0, 1, len(participants)))
    return dict(zip(participants, colors))

def scatter_non_responsive_intervals(threshold=20):
    """Scatter long button-press intervals for every participant and run.

    Each interval at or above ``threshold`` is treated as a non-responsive
    interval and plotted at its run's x position. The x-axis is built from all
    run directories so runs without qualifying intervals retain empty slots.
    """
    non_responsive_df, whole_runs_df, end_intervals_df = get_non_responsive_intervals(threshold=threshold)

    # Inventory runs independently of qualifying intervals so empty runs are
    # represented by an x-axis position with no marker.
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

    participants = get_all_participants()['participant'].unique()
    participant_colors = get_participant_colors()

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
            run_intervals = non_responsive_df[
                (non_responsive_df['participant'] == participant) &
                (non_responsive_df['run'] == run_row['run'])
            ]
            run_whole_runs = whole_runs_df[
                (whole_runs_df['participant'] == participant) &
                (whole_runs_df['run'] == run_row['run'])
            ]
            run_end_intervals = end_intervals_df[
                (end_intervals_df['participant'] == participant) &
                (end_intervals_df['run'] == run_row['run'])
            ]

            if not run_intervals.empty:
                ax.scatter(
                    np.full(len(run_intervals), x),
                    run_intervals['duration_TRs'],
                    alpha=0.7,
                    s=180,
                    color=color
                )

            if not run_whole_runs.empty:
                ax.scatter(
                    np.full(len(run_whole_runs), x),
                    run_whole_runs['duration_TRs'],
                    marker='*',
                    alpha=0.7,
                    s=220,
                    color=color,
                    linewidths=2
                )

            if not run_end_intervals.empty:
                ax.scatter(
                    np.full(len(run_end_intervals), x),
                    run_end_intervals['duration_TRs'],
                    marker='x',
                    alpha=0.7,
                    s=180,
                    color=color,
                    linewidths=2
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
    ax.set_ylabel('Non-responsive Interval Duration (TRs)', fontsize=30, labelpad=15)
    ax.set_title(
        'Non-responsive Interval Durations by Participant and Run',
        fontsize=34,
        fontweight='bold'
    )

    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(SAVE_PATH / f'non_responsive_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_non_responsive_interval_counts(threshold=20):
    non_responsive_df, _, _ = get_non_responsive_intervals(threshold=threshold)
    all_participants = get_all_participants()
    participant_colors = get_participant_colors()
    participants = get_all_participants()['participant'].unique()

    observed_counts = (
        non_responsive_df
        .groupby("participant")
        .size()
        .reset_index(name="n_events")
    )
    participant_counts = (
        all_participants
        .merge(observed_counts, on="participant", how="left")
        .fillna({"n_events": 0})
    )
    participant_counts["n_events"] = participant_counts["n_events"].astype(int)

    participant_counts = participant_counts.sort_values("n_events", ascending=False).copy()
    participant_counts['participant_num'] = participant_counts['participant'].str.split('_').str[-1].astype(int)

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = [participant_colors[p] for p in participant_counts["participant"]]

    bars = ax.bar(participant_counts["participant_num"].astype(str), participant_counts["n_events"] , color=bar_colors)

    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Participant")
    ax.set_ylabel(f"Number of BP intervals ≥ {threshold} TRs")
    ax.set_title(f"Non-responsive BP intervals per participant ≥ {threshold}-TR threshold")

    ax.tick_params(axis="x")

    plt.tight_layout()
    plt.savefig(SAVE_PATH / f"non_responsive_intervals_per_participant.png", dpi=300)
    plt.show()
    plt.close()

def print_non_responsive_summary(threshold=20):
    non_responsive_df, whole_runs_df, end_intervals_df = get_non_responsive_intervals(threshold=threshold)

    total_intervals = len(non_responsive_df)
    total_whole_runs = len(whole_runs_df)
    total_end_intervals = len(end_intervals_df)

    print(f"Total non-responsive intervals ≥ {threshold} TRs: {total_intervals}")
    print(f"Total whole-run non-responsive intervals ≥ {threshold} TRs: {total_whole_runs}")
    print(f"Total end non-responsive intervals ≥ {threshold} TRs: {total_end_intervals}")

if __name__ == '__main__' and not any(flag in sys.argv for flag in ('--plot-bp-intervals', '--plot-hazard-persistence', '--plot-nonresponsive-pooled')):
    scatter_non_responsive_intervals(threshold=20)
    plot_non_responsive_interval_counts(threshold=20)
    print_non_responsive_summary(threshold=20)
    pass
    # button_press_intervals_df = pd.read_csv(Path('plots/analyze_button_presses') / 'bp_intervals.csv')
    # participant_run_bp_intervals = get_participant_run_bp_intervals(button_press_intervals_df)
    # participant_bp_intervals = get_participant_bp_intervals(button_press_intervals_df)
    
    # plot_interval_frequency_distribution(participant_bp_intervals, max_interval=15)
    # summarize_hazards(participant_bp_intervals, max_interval=50, min_at_risk=20)
    # plot_participant_contributions(participant_bp_intervals, max_interval=50, min_at_risk=20)
    # plot_hazard(participant_bp_intervals, participant='AAN_01', max_interval=50)
    # plot_hazard(participant_bp_intervals, participant='AAN_03', max_interval=50)
    # plot_hazard(participant_bp_intervals, participant='AAN_20', max_interval=50)
    # plot_hazard(participant_bp_intervals, participant='AAN_12', max_interval=50)

    # plot_all_participant_hazards(participant_bp_intervals, max_interval=50, min_at_risk=20)

    # plot_hazard_summary(participant_bp_intervals, max_interval=50, min_at_risk=20)
    # plot_participant_contributions(participant_bp_intervals, max_interval=50, min_at_risk=20)
    
    # plot_persistence_summary(participant_bp_intervals, plot_future_TRs=[5, 10], min_at_risk=20)
    # get_participant_balanced_persistence_summary(participant_bp_intervals, future_TRs=5, min_at_risk=20)
    # get_participant_balanced_persistence_summary(participant_bp_intervals, future_TRs=10, min_at_risk=20)

    # sensitivity_summary = get_threshold_sensitivity_summary(participant_run_bp_intervals, thresholds=[15, 20, 25, 30])

    # get_participant_threshold_counts(participant_run_bp_intervals, thresholds=[15, 20, 25, 30])
    # get_run_threshold_counts(participant_run_bp_intervals, thresholds=[15, 20, 25, 30])
    # get_run_threshold_fractions(participant_run_bp_intervals, thresholds=[15, 20, 25, 30])

    # plot_total_events_vs_threshold(sensitivity_summary)
    # plot_participant_counts_vs_threshold(sensitivity_summary)
    # plot_run_counts_vs_threshold(sensitivity_summary)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Participant-pooled button-press interval plots")
    parser.add_argument("--plot-bp-intervals", action="store_true")
    parser.add_argument("--plot-hazard-persistence", action="store_true")
    parser.add_argument("--plot-nonresponsive-pooled", action="store_true")
    parser.add_argument("--bp_TR_snap", choices=("nearest", "preceding"), default="preceding")
    parser.add_argument("--max-interval", type=int, default=20)
    args = parser.parse_args()
    if args.plot_bp_intervals:
        if args.max_interval < 1:
            parser.error("--max-interval must be positive")
        plot_pooled_bp_intervals(args.bp_TR_snap, max_interval=args.max_interval)

    if args.plot_hazard_persistence:
        plot_hazard_and_persistence(args.bp_TR_snap)

    if args.plot_nonresponsive_pooled:
        plot_nonresponsive_intervals_pooled(args.bp_TR_snap)
