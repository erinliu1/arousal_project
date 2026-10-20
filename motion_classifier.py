from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


# Change these globals, then run this file. ANALYSIS can be "loro" or "lopo".
ANALYSIS = "loro"
RUN_TRAINING = True
REBUILD_SEGMENT_ARRAYS = False
PARTICIPANT_CLASS_BALANCED_SAMPLING = True

SEED = 42
FD_THRESHOLD = 0.25

RAW_DATA_DIR = Path("raw_data")
SEGMENTS_DIR = Path("plots/label_segments")
CLASSIFIER_DIR = Path("plots/motion_classifier")
FEATURE_NAMES = (
    "mean_fd",
    "max_fd",
    "median_fd",
    "sd_fd",
    "fraction_fd_above_0.25",
)


def load_segment_metadata():
    """Load the same motion-free responsive/non-responsive segments as fmri_classifier."""
    segment_paths = sorted(SEGMENTS_DIR.glob("*.csv"))
    if not segment_paths:
        raise FileNotFoundError(f"No segment CSV files found in {SEGMENTS_DIR}")

    metadata = pd.concat(
        [pd.read_csv(path) for path in segment_paths], ignore_index=True
    )
    motion_free = ~metadata["contains_motion"].astype(str).str.lower().eq("true")
    metadata = metadata.loc[
        motion_free,
        [
            "participant",
            "run",
            "label",
            "segment_start_TR",
            "segment_end_TR_exclusive",
        ],
    ]
    metadata = metadata.sort_values(
        ["participant", "run", "segment_start_TR"]
    ).reset_index(drop=True)

    segment_lengths = (
        metadata["segment_end_TR_exclusive"] - metadata["segment_start_TR"]
    )
    if metadata.empty or not segment_lengths.eq(10).all():
        raise ValueError("All selected segments must be 10 TRs long")

    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(CLASSIFIER_DIR / "all_segment_metadata.csv", index=False)
    return metadata


def calculate_fd_features(segment_fd):
    """Return the five motion features for one 10-TR segment."""
    return np.array(
        [
            np.mean(segment_fd),
            np.max(segment_fd),
            np.median(segment_fd),
            np.std(segment_fd, ddof=0),
            np.mean(segment_fd > FD_THRESHOLD),
        ],
        dtype=np.float32,
    )


def save_segment_arrays():
    """Build and save motion features and labels for all selected segments."""
    metadata = load_segment_metadata()
    X = np.empty((len(metadata), len(FEATURE_NAMES)), dtype=np.float32)
    mapped_labels = metadata["label"].map(
        {"responsive": 0, "non-responsive": 1}
    )
    if mapped_labels.isna().any():
        raise ValueError("Unexpected label in segment metadata")
    y = mapped_labels.to_numpy(dtype=np.int64)
    participants = metadata["participant"].to_numpy(dtype=str)

    for (participant, run), run_metadata in metadata.groupby(
        ["participant", "run"], sort=False
    ):
        fd = np.asarray(
            np.load(RAW_DATA_DIR / participant / run / "fd.npy"), dtype=np.float64
        )
        # fd[i] measures movement from TR i to TR i+1 and therefore belongs to
        # TR i+1. TR 0 has no preceding displacement, so assign it zero.
        fd_by_tr = np.concatenate(([0.0], fd))

        for segment_index, segment in run_metadata.iterrows():
            start = int(segment["segment_start_TR"])
            end = int(segment["segment_end_TR_exclusive"])
            segment_fd = fd_by_tr[start:end]
            if len(segment_fd) != 10:
                raise ValueError(
                    f"Expected 10 FD values for {participant} {run} "
                    f"TRs [{start}, {end}), found {len(segment_fd)}"
                )
            X[segment_index] = calculate_fd_features(segment_fd)

    np.save(CLASSIFIER_DIR / "X.npy", X)
    np.save(CLASSIFIER_DIR / "y.npy", y)
    np.save(CLASSIFIER_DIR / "participants.npy", participants)
    return X, y, participants, metadata


def load_segments():
    array_paths = [
        CLASSIFIER_DIR / "X.npy",
        CLASSIFIER_DIR / "y.npy",
        CLASSIFIER_DIR / "participants.npy",
    ]
    metadata_path = CLASSIFIER_DIR / "all_segment_metadata.csv"
    if (
        REBUILD_SEGMENT_ARRAYS
        or not all(path.exists() for path in array_paths)
        or not metadata_path.exists()
    ):
        return save_segment_arrays()

    X = np.load(array_paths[0])
    y = np.load(array_paths[1])
    participants = np.load(array_paths[2])
    metadata = pd.read_csv(metadata_path).reset_index(drop=True)
    if X.shape != (len(metadata), len(FEATURE_NAMES)):
        raise ValueError(
            "Cached X.npy does not match the current feature definition; set "
            "REBUILD_SEGMENT_ARRAYS = True"
        )
    return X, y, participants, metadata


def get_folds(X, y, participants, metadata):
    if ANALYSIS == "lopo":
        return [
            (
                np.flatnonzero(participants != participant),
                np.flatnonzero(participants == participant),
            )
            for participant in np.unique(participants)
        ]
    if ANALYSIS == "loro":
        run_groups = (
            metadata["participant"].astype(str)
            + "::"
            + metadata["run"].astype(str)
        ).to_numpy()
        return list(LeaveOneGroupOut().split(X, y, groups=run_groups))
    raise ValueError('ANALYSIS must be "loro" or "lopo"')


def participant_class_balanced_weights(participants, labels):
    """Give equal weight to participants and to classes within participants."""
    weights = np.empty(len(labels), dtype=np.float64)
    unique_participants = np.unique(participants)

    for participant in unique_participants:
        participant_mask = participants == participant
        participant_labels = np.unique(labels[participant_mask])
        for label in participant_labels:
            cell_mask = participant_mask & (labels == label)
            weights[cell_mask] = 1.0 / (
                len(unique_participants)
                * len(participant_labels)
                * cell_mask.sum()
            )

    # Scaling all weights by the same constant does not change their relative
    # contribution, but a mean of one keeps regularization on a familiar scale.
    return weights / weights.mean()


def run_fold(X, y, participants, train_indices, test_indices):
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    if len(np.unique(y_train)) < 2:
        raise ValueError("A training fold contains only one class")

    sample_weight = None
    if PARTICIPANT_CLASS_BALANCED_SAMPLING:
        sample_weight = participant_class_balanced_weights(
            participants[train_indices], y_train
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        solver="liblinear",
        random_state=SEED,
        max_iter=1000,
    )
    model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

    train_predictions = model.predict(X_train_scaled)
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    return {
        "labels": y_test,
        "predictions": predictions,
        "probabilities": probabilities,
        "train_accuracy": np.mean(train_predictions == y_train),
        "test_accuracy": np.mean(predictions == y_test),
        "n_train_segments": len(train_indices),
        "n_train_correct": int(np.sum(train_predictions == y_train)),
    }


def output_settings():
    suffix = (
        "_participant_class_balanced"
        if PARTICIPANT_CLASS_BALANCED_SAMPLING
        else ""
    )
    return CLASSIFIER_DIR / f"{ANALYSIS.upper()}{suffix}", ANALYSIS


def run_analysis():
    np.random.seed(SEED)
    X, y, participants, metadata = load_segments()
    folds = get_folds(X, y, participants, metadata)
    save_dir, prefix = output_settings()
    save_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    prediction_rows = []
    fold_results = []

    for fold, (train_indices, test_indices) in enumerate(folds, start=1):
        print(f"\n{'=' * 50}\nFold {fold}/{len(folds)}\n{'=' * 50}")
        results = run_fold(X, y, participants, train_indices, test_indices)
        fold_results.append(results)
        print(f"Train accuracy: {results['train_accuracy']:.4f}")
        print(f"Test accuracy: {results['test_accuracy']:.4f}")

        held_out_participant = metadata.iloc[test_indices[0]]["participant"]
        held_out_run = metadata.iloc[test_indices[0]]["run"]
        responsive = results["labels"] == 0
        nonresponsive = results["labels"] == 1
        fold_rows.append(
            {
                "fold": fold,
                "participant": held_out_participant,
                **({"run": held_out_run} if ANALYSIS == "loro" else {}),
                "train_accuracy": results["train_accuracy"],
                "test_accuracy": results["test_accuracy"],
                "n_responsive": int(responsive.sum()),
                "n_nonresponsive": int(nonresponsive.sum()),
                "responsive_recall": (
                    np.mean(results["predictions"][responsive] == 0)
                    if responsive.any()
                    else np.nan
                ),
                "nonresponsive_recall": (
                    np.mean(results["predictions"][nonresponsive] == 1)
                    if nonresponsive.any()
                    else np.nan
                ),
            }
        )

        train_participants = set(participants[train_indices])
        for segment_index, label, prediction, probability in zip(
            test_indices,
            results["labels"],
            results["predictions"],
            results["probabilities"],
        ):
            segment = metadata.iloc[segment_index]
            row = {
                "segment_index": int(segment_index),
                "fold": fold,
                "participant": participants[segment_index],
                "run": segment["run"],
                "label": segment["label"],
                "label_binary": int(label),
                "segment_start_TR": int(segment["segment_start_TR"]),
                "segment_end_TR_exclusive": int(
                    segment["segment_end_TR_exclusive"]
                ),
                "prediction": int(prediction),
                "probability_nonresponsive": float(probability),
            }
            if ANALYSIS == "loro":
                row["participant_seen"] = int(
                    participants[segment_index] in train_participants
                )
            prediction_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    fold_df.to_csv(save_dir / f"{prefix}_fold_results.csv", index=False)
    prediction_df.to_csv(save_dir / f"{prefix}_predictions.csv", index=False)

    responsive = prediction_df["label_binary"] == 0
    nonresponsive = prediction_df["label_binary"] == 1
    summary = {
        "pooled_train_accuracy": sum(
            result["n_train_correct"] for result in fold_results
        )
        / sum(result["n_train_segments"] for result in fold_results),
        "pooled_test_accuracy": np.mean(
            prediction_df["prediction"] == prediction_df["label_binary"]
        ),
        "responsive_recall": np.mean(
            prediction_df.loc[responsive, "prediction"] == 0
        ),
        "nonresponsive_recall": np.mean(
            prediction_df.loc[nonresponsive, "prediction"] == 1
        ),
        "balanced_test_accuracy": balanced_accuracy_score(
            prediction_df["label_binary"], prediction_df["prediction"]
        ),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(save_dir / f"{prefix}_summary.csv", index=False)
    print("\n" + summary_df.to_string(index=False))


def visualize_leave_one_out(analysis):
    """Plot responsive and non-responsive recall for each held-out fold."""
    analysis = analysis.strip().lower()
    if analysis not in {"loro", "lopo"}:
        raise ValueError('analysis must be "loro" or "lopo"')

    sampling_suffix = (
        "_participant_class_balanced"
        if PARTICIPANT_CLASS_BALANCED_SAMPLING
        else ""
    )
    result_dir = CLASSIFIER_DIR / f"{analysis.upper()}{sampling_suffix}"
    results_path = result_dir / f"{analysis}_fold_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Fold results not found: {results_path}")

    results = pd.read_csv(results_path)
    required_columns = {
        "participant",
        "n_responsive",
        "n_nonresponsive",
        "responsive_recall",
        "nonresponsive_recall",
    }
    if analysis == "loro":
        required_columns.add("run")
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(
            "Fold results are missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    results["_participant_num"] = pd.to_numeric(
        results["participant"].astype(str).str.extract(r"(\d+)$")[0],
        errors="coerce",
    )
    sort_columns = ["_participant_num"]
    if analysis == "loro":
        results["_run_num"] = pd.to_numeric(
            results["run"].astype(str).str.extract(r"(\d+)$")[0],
            errors="coerce",
        )
        sort_columns.append("_run_num")
    results = results.sort_values(sort_columns).reset_index(drop=True)

    participants = results["participant"].drop_duplicates().tolist()
    participant_colors = dict(
        zip(participants, plt.cm.tab20(np.linspace(0, 1, len(participants))))
    )
    figsize = (27, 9) if analysis == "loro" else (18, 9)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0.5, color="lightgray", linestyle=":", linewidth=2, zorder=0)

    metric_specs = (
        ("responsive_recall", "n_responsive", -0.20, True, "Responsive recall"),
        ("nonresponsive_recall", "n_nonresponsive", 0.20, False, "Non-responsive recall"),
    )
    positive_counts = pd.concat(
        [results["n_responsive"], results["n_nonresponsive"]],
        ignore_index=True,
    )
    positive_counts = positive_counts[positive_counts > 0]
    size_cap = max(float(positive_counts.quantile(0.95)), 1.0)
    min_marker_area = 15
    max_marker_area = 360

    def marker_area(count):
        capped_count = min(float(count), size_cap)
        log_fraction = np.log1p(capped_count) / np.log1p(size_cap)
        return min_marker_area + (max_marker_area - min_marker_area) * log_fraction

    participant_centers = []
    participant_labels = []
    x = 0

    for participant_index, (participant, participant_df) in enumerate(
        results.groupby("participant", sort=False)
    ):
        participant_xs = []
        color = participant_colors[participant]

        for row_index, (_, row) in enumerate(participant_df.iterrows()):
            for metric, count_column, offset, filled, _ in metric_specs:
                missing_class = row[count_column] == 0 or pd.isna(row[metric])
                if missing_class:
                    continue
                ax.scatter(
                    x + offset,
                    row[metric],
                    marker="o",
                    s=marker_area(row[count_column]),
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    alpha=0.85,
                    linewidths=2,
                    zorder=3,
                )

            participant_xs.append(x)
            if analysis == "loro" and row_index < len(participant_df) - 1:
                ax.axvline(x + 0.5, color="lightgray", linewidth=2, zorder=0)
            x += 1

        participant_centers.append(np.mean(participant_xs))
        participant_number = participant_df["_participant_num"].iloc[0]
        participant_labels.append(
            int(participant_number)
            if pd.notna(participant_number)
            else participant
        )
        if participant_index < len(participants) - 1:
            ax.axvline(x - 0.5, color="gray", linewidth=3, alpha=0.6, zorder=0)

    ax.set_xticks(participant_centers)
    ax.set_xticklabels(participant_labels, fontsize=22)
    ax.tick_params(axis="x", width=2, length=8)
    ax.tick_params(axis="y", which="both", length=0, labelsize=22)
    ax.set_xlabel("Participant", fontsize=32, labelpad=15)
    ax.set_ylabel("Recall", fontsize=30, labelpad=15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"{analysis.upper()} Fold Performance",
        fontsize=34,
        fontweight="bold",
    )
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="0.4" if filled else "none",
            markeredgecolor="0.4",
            markeredgewidth=2,
            markersize=13,
            label=label,
        )
        for _, _, _, filled, label in metric_specs
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=18,
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        framealpha=1,
        loc="lower left",
    )

    fig.tight_layout()
    output_path = result_dir / f"{analysis}_fold_performance.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {analysis.upper()} visualization to {output_path}")
    return output_path


def main():
    print(f"Analysis: {ANALYSIS.upper()}")
    print(
        "Participant/class-balanced sampling: "
        f"{PARTICIPANT_CLASS_BALANCED_SAMPLING}"
    )
    if RUN_TRAINING:
        run_analysis()


if __name__ == "__main__":
    # main()
    visualize_leave_one_out("loro")
