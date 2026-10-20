from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_PATH = Path(
    "plots/fMRI_classifier/LORO_participant_class_balanced/loro_predictions.csv"
)
RAW_DATA_DIR = Path("raw_data")
OUTPUT_DIR = Path("plots/fMRI_classifier/analysis")
FD_THRESHOLD = 0.25

FEATURE_NAMES = (
    "mean_fd",
    "max_fd",
    "median_fd",
    "sd_fd",
    "fraction_fd_above_0.25",
)


def calculate_fd_features(segment_fd):
    """Calculate the same five features used by motion_classifier.py."""
    return {
        "mean_fd": np.mean(segment_fd),
        "max_fd": np.max(segment_fd),
        "median_fd": np.median(segment_fd),
        "sd_fd": np.std(segment_fd, ddof=0),
        "fraction_fd_above_0.25": np.mean(segment_fd > FD_THRESHOLD),
    }


def load_responsive_predictions():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Predictions not found: {PREDICTIONS_PATH}")

    predictions = pd.read_csv(PREDICTIONS_PATH)
    required_columns = {
        "participant",
        "run",
        "label",
        "segment_start_TR",
        "segment_end_TR_exclusive",
        "probability_nonresponsive",
    }
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Predictions CSV is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    responsive = predictions.loc[
        predictions["label"].eq("responsive")
    ].copy()
    if responsive.empty:
        raise ValueError("No responsive segments found in the predictions CSV")

    return responsive.reset_index(drop=True)


def attach_segment_features(responsive_predictions):
    """Attach FD features and normalized within-run time to each segment."""
    enriched = responsive_predictions.copy()
    for feature in FEATURE_NAMES:
        enriched[feature] = np.nan
    enriched["normalized_time"] = np.nan

    for (participant, run), run_df in enriched.groupby(
        ["participant", "run"], sort=False
    ):
        fd_path = RAW_DATA_DIR / participant / run / "fd.npy"
        if not fd_path.exists():
            raise FileNotFoundError(f"FD data not found: {fd_path}")

        fd = np.asarray(np.load(fd_path), dtype=np.float64)
        if fd.ndim != 1:
            raise ValueError(f"Expected one-dimensional FD data in {fd_path}")

        # fd[i] is movement from TR i to TR i+1, so it is assigned to TR i+1.
        # TR 0 has no preceding displacement and is assigned zero.
        fd_by_tr = np.concatenate(([0.0], fd))
        n_run_trs = len(fd_by_tr)

        for row_index, segment in run_df.iterrows():
            start = int(segment["segment_start_TR"])
            end = int(segment["segment_end_TR_exclusive"])
            segment_fd = fd_by_tr[start:end]
            if len(segment_fd) != 10:
                raise ValueError(
                    f"Expected 10 FD values for {participant} {run} "
                    f"TRs [{start}, {end}), found {len(segment_fd)}"
                )

            features = calculate_fd_features(segment_fd)
            for feature, value in features.items():
                enriched.at[row_index, feature] = value

            segment_midpoint = (start + end) / 2
            enriched.at[row_index, "normalized_time"] = (
                segment_midpoint / n_run_trs
            )

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


def calculate_motion_correlations_by_run(enriched_segments):
    """Calculate motion-feature/P(non-responsive) correlations by run."""
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
    """Calculate normalized-time/P(non-responsive) correlations by run."""
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
    responsive_predictions = load_responsive_predictions()
    enriched_segments = attach_segment_features(responsive_predictions)
    motion_correlations = calculate_motion_correlations_by_run(enriched_segments)
    time_correlations = calculate_time_correlations_by_run(enriched_segments)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    enriched_path = OUTPUT_DIR / "responsive_segments_with_motion_features.csv"
    motion_path = OUTPUT_DIR / "motion_feature_correlations_by_run.csv"
    time_path = OUTPUT_DIR / "normalized_time_correlations_by_run.csv"
    enriched_segments.to_csv(enriched_path, index=False)
    motion_correlations.to_csv(motion_path, index=False)
    time_correlations.to_csv(time_path, index=False)

    print(f"Saved {len(enriched_segments)} responsive segments to {enriched_path}")
    print(
        f"Saved motion correlations for {len(motion_correlations)} runs to "
        f"{motion_path}"
    )
    print(
        f"Saved time correlations for {len(time_correlations)} runs to "
        f"{time_path}"
    )


if __name__ == "__main__":
    main()
