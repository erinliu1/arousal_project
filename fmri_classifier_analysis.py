"""Three within-run correlation analyses of fMRI classifier predictions.

Only true responsive segments are analyzed. Each analysis correlates a segment
predictor with probability_nonresponsive, the classifier's continuous P(NR)
score. Each participant/run is analyzed separately; runs are never pooled.
These are separate Pearson correlations, without adjustment for other predictors.

The numbered blocks in main() attach predictors, calculate correlations, and
save these files under OUTPUT_DIR:
1. motion_feature_correlations_by_run.csv: five FD/P(NR) correlations per run.
2. normalized_time_correlations_by_run.csv: one time/P(NR) correlation per run.
3. bp_rate_correlations_by_run.csv: one BP-rate/P(NR) correlation per run.

Each correlation CSV includes participant, run, and n_responsive_segments.
Undefined correlations are saved as blank cells (NaN); no p-values are computed.
responsive_segments_with_motion_features.csv contains the original predictions
and all predictors (FD summaries, normalized time, BP count, and BP rate).
"""

from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_PATH = Path(
    "plots/fMRI_classifier/LORO_participant_class_balanced/loro_predictions.csv"
)
RAW_DATA_DIR = Path("raw_data")
OUTPUT_DIR = Path("plots/fMRI_classifier/LORO_participant_class_balanced/analysis")

FEATURE_NAMES = ("mean_fd", "max_fd", "median_fd", "sd_fd", "fraction_fd_above_0.25",)

def calculate_fd_features(segment_fd):
    """Calculate the same five features used by motion_classifier.py."""
    return {
        "mean_fd": np.mean(segment_fd),
        "max_fd": np.max(segment_fd),
        "median_fd": np.median(segment_fd),
        "sd_fd": np.std(segment_fd, ddof=0),
        "fraction_fd_above_0.25": np.mean(segment_fd > 0.25),
    }


def load_responsive_predictions():
    predictions = pd.read_csv(PREDICTIONS_PATH)
    responsive = predictions.loc[predictions["label"].eq("responsive")].copy()
    return responsive.reset_index(drop=True)

def attach_fd_features(responsive_predictions):
    """Attach five FD summaries over each segment's [start, end) TR interval."""
    enriched = responsive_predictions.copy()
    for feature in FEATURE_NAMES:
        enriched[feature] = np.nan

    for (participant, run), run_df in enriched.groupby(["participant", "run"], sort=False):
        fd_path = RAW_DATA_DIR / participant / run / "fd.npy"
        fd = np.asarray(np.load(fd_path), dtype=np.float64)

        # fd[i] is movement from TR i to TR i+1, so it is assigned to TR i+1.
        # TR 0 has no preceding displacement and is assigned zero.
        fd_by_tr = np.concatenate(([0.0], fd))

        for row_index, segment in run_df.iterrows():
            start = int(segment["segment_start_TR"])
            end = int(segment["segment_end_TR_exclusive"])
            segment_fd = fd_by_tr[start:end]

            features = calculate_fd_features(segment_fd)
            for feature, value in features.items():
                enriched.at[row_index, feature] = value

    return enriched


def attach_within_run_time(responsive_predictions):
    """Attach normalized_time = segment midpoint / full run length in TRs.

    Preserve the interval-midpoint convention: (start + end_exclusive) / 2.
    fd.npy has one displacement between adjacent TRs, so len(fd) + 1 gives
    the run length, using the same TR coordinate system as FD attachment.
    """
    enriched = responsive_predictions.copy()
    enriched["normalized_time"] = np.nan
    for (participant, run), run_df in enriched.groupby(["participant", "run"], sort=False):
        fd_path = RAW_DATA_DIR / participant / run / "fd.npy"
        n_run_trs = len(np.load(fd_path, mmap_mode="r")) + 1
        segment_midpoints = (
            run_df["segment_start_TR"] + run_df["segment_end_TR_exclusive"]
        ) / 2
        enriched.loc[run_df.index, "normalized_time"] = segment_midpoints / n_run_trs

    return enriched


def pearson_r(x, y):
    """Return Pearson's r, or NaN when it is undefined."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def attach_segment_bp_rates(responsive_predictions):
    """Count presses in each [start, end) segment and calculate rate in Hz."""
    enriched = responsive_predictions.copy()
    enriched["bp_count"] = 0
    enriched["bp_rate_hz"] = np.nan

    for (participant, run), run_df in enriched.groupby(
        ["participant", "run"], sort=False
    ):
        run_dir = RAW_DATA_DIR / participant / run
        press_times = np.asarray(np.load(run_dir / "bps.npy"), dtype=np.float64)
        time_vector = np.asarray(
            np.load(run_dir / "time_vector.npy"), dtype=np.float64
        )
        if press_times.ndim != 1 or not np.all(np.isfinite(press_times)):
            raise ValueError(f"Expected finite one-dimensional BP data in {run_dir}")
        if (
            time_vector.ndim != 1
            or len(time_vector) < 2
            or not np.all(np.isfinite(time_vector))
            or not np.all(np.diff(time_vector) > 0)
        ):
            raise ValueError(f"Expected increasing finite TR timestamps in {run_dir}")
        tr_seconds = float(np.median(np.diff(time_vector)))
        press_times = np.sort(press_times)

        for row_index, segment in run_df.iterrows():
            start = int(segment["segment_start_TR"])
            end = int(segment["segment_end_TR_exclusive"])
            if not 0 <= start < end <= len(time_vector):
                raise ValueError(
                    f"Invalid segment for {participant} {run}: TRs [{start}, {end})"
                )
            start_seconds = time_vector[start]
            end_seconds = (
                time_vector[end] if end < len(time_vector)
                else time_vector[-1] + tr_seconds
            )
            count = int(
                np.searchsorted(press_times, end_seconds, side="left")
                - np.searchsorted(press_times, start_seconds, side="left")
            )
            enriched.at[row_index, "bp_count"] = count
            enriched.at[row_index, "bp_rate_hz"] = count / (
                end_seconds - start_seconds
            )

    return enriched


def calculate_bp_rate_correlations_by_run(enriched_segments):
    """Analysis 3: correlate button-press rate (Hz) with P(NR) per run.

    Uses true responsive segments only. Positive r means higher press rates
    accompany higher P(NR). Returns one row per participant/run with
    pearson_r_bp_rate_hz for bp_rate_correlations_by_run.csv.
    """
    responsive = enriched_segments.loc[enriched_segments["label"].eq("responsive")]
    rows = []
    for (participant, run), run_df in responsive.groupby(["participant", "run"], sort=False):
        rows.append(
            {
                "participant": participant,
                "run": run,
                "n_responsive_segments": len(run_df),
                "pearson_r_bp_rate_hz": pearson_r(
                    run_df["bp_rate_hz"].to_numpy(),
                    run_df["probability_nonresponsive"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(rows)


def calculate_motion_correlations_by_run(enriched_segments):
    """Analysis 1: correlate each FD summary with P(NR) within each run.

    Input contains true responsive segments only. Positive r means higher
    values of that motion feature accompany higher P(NR). Each feature is
    analyzed separately. Returns one row per participant/run with five
    pearson_r_<feature> columns for motion_feature_correlations_by_run.csv.
    """
    rows = []
    for (participant, run), run_df in enriched_segments.groupby(
        ["participant", "run"], sort=False
    ):
        probability = run_df["probability_nonresponsive"].to_numpy()
        row = {
            "participant": participant,
            "run": run,
            "n_responsive_segments": len(run_df),
        }
        for feature in FEATURE_NAMES:
            row[f"pearson_r_{feature}"] = pearson_r(
                run_df[feature].to_numpy(), probability
            )
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_time_correlations_by_run(enriched_segments):
    """Analysis 2: correlate normalized segment midpoint with P(NR) per run.

    Input contains true responsive segments only. Positive r means later
    responsive segments tend to have higher P(NR), a linear time association.
    Returns one row per participant/run with pearson_r_normalized_time for
    normalized_time_correlations_by_run.csv.
    """
    rows = []
    for (participant, run), run_df in enriched_segments.groupby(
        ["participant", "run"], sort=False
    ):
        rows.append(
            {
                "participant": participant,
                "run": run,
                "n_responsive_segments": len(run_df),
                "pearson_r_normalized_time": pearson_r(
                    run_df["normalized_time"].to_numpy(),
                    run_df["probability_nonresponsive"].to_numpy(),
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    # Shared population: true responsive segments from LORO predictions.
    # Every analysis uses the continuous P(NR) score as its outcome.
    responsive_predictions = load_responsive_predictions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Analysis 1 — motion: summarize FD in each segment, then correlate each
    # of the five summaries with P(NR) across responsive segments of each run.
    enriched_segments = attach_fd_features(responsive_predictions)
    motion_correlations = calculate_motion_correlations_by_run(enriched_segments)
    motion_path = OUTPUT_DIR / "motion_feature_correlations_by_run.csv"
    motion_correlations.to_csv(motion_path, index=False)
    print(f"Saved motion correlations for {len(motion_correlations)} runs to {motion_path}")

    # Analysis 2 — within-run time: attach normalized segment midpoints, then
    # test whether P(NR) tends to increase or decrease over each run.
    enriched_segments = attach_within_run_time(enriched_segments)
    time_correlations = calculate_time_correlations_by_run(enriched_segments)
    time_path = OUTPUT_DIR / "normalized_time_correlations_by_run.csv"
    time_correlations.to_csv(time_path, index=False)
    print(f"Saved time correlations for {len(time_correlations)} runs to {time_path}")

    # Analysis 3 — button presses: count presses in each [start, end) segment
    # interval, divide by duration in seconds, then correlate rate with P(NR).
    enriched_segments = attach_segment_bp_rates(enriched_segments)
    bp_correlations = calculate_bp_rate_correlations_by_run(enriched_segments)
    bp_path = OUTPUT_DIR / "bp_rate_correlations_by_run.csv"
    bp_correlations.to_csv(bp_path, index=False)
    print(f"Saved BP rate correlations for {len(bp_correlations)} runs to {bp_path}")

    # Save the segment-level inputs to all three analyses, retaining the
    # existing filename even though the table also includes time and BP data.
    enriched_path = OUTPUT_DIR / "responsive_segments_with_appended_features.csv"
    enriched_segments.to_csv(enriched_path, index=False)
    print(f"Saved {len(enriched_segments)} responsive segments to {enriched_path}")


if __name__ == "__main__":
    main()
