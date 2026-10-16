from pathlib import Path
import random

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from all_rois import all_rois


# Change these globals, then run this file. ANALYSIS can be:
# "segment_cv", "loro", "lopo", "buffer", or "blocked_buffer".
ANALYSIS = "buffer"
BUFFER_TR = 20
RUN_TRAINING = True
MAKE_PROBABILITY_PLOTS = False
REBUILD_SEGMENT_ARRAYS = False

SEED = 42
N_SPLITS = 10
N_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

RAW_DATA_DIR = Path("raw_data")
SEGMENTS_DIR = Path("plots/label_segments")
CLASSIFIER_DIR = Path("plots/fMRI_classifier")
N_ROIS = len(all_rois)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_segment_metadata():
    # save a single CSV with all segment metadata per row (participant, run, label, start TR, end TR)
    segment_data = [pd.read_csv(path) for path in SEGMENTS_DIR.glob("*.csv")]
    metadata = pd.concat(segment_data, ignore_index=True)
    metadata = metadata[metadata["contains_motion"] == False][["participant", "run", "label", "segment_start_TR", "segment_end_TR_exclusive"]]
    metadata = metadata.sort_values(["participant", "run", "segment_start_TR"]).reset_index(drop=True) 
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(CLASSIFIER_DIR / "all_segment_metadata.csv", index=False)
    return metadata


def save_segment_arrays():
    # save a single X, y, and participant array for all segments
    # X: (n_segments, segment_length, n_rois)
    # y: (n_segments,) 0=responsive, 1=non-responsive
    # participants: (n_segments,) participant ID for each segment
    metadata = load_segment_metadata()
    segment_length = int((metadata["segment_end_TR_exclusive"] - metadata["segment_start_TR"]).iloc[0])
    X = np.empty((len(metadata), segment_length, N_ROIS), dtype=np.float32)
    y = metadata["label"].map({"responsive": 0.0, "non-responsive": 1.0}).to_numpy(dtype=np.float32)
    participants = metadata["participant"].to_numpy(dtype=str)

    for (participant, run), run_metadata in metadata.groupby(["participant", "run"], sort=False):
        run_data = np.column_stack([
            np.load(RAW_DATA_DIR / participant / run / "z_scored" / f"{roi}.npy")
            for roi in all_rois
        ]).astype(np.float32, copy=False)
        for segment_index, segment in run_metadata.iterrows():
            start = int(segment["segment_start_TR"])
            end = int(segment["segment_end_TR_exclusive"])
            X[segment_index] = run_data[start:end]

    np.save(CLASSIFIER_DIR / "X.npy", X)
    np.save(CLASSIFIER_DIR / "y.npy", y)
    np.save(CLASSIFIER_DIR / "participants.npy", participants)
    return X, y, participants, metadata


def load_segments():
    # load or rebuild the segment arrays and metadata; returns X, y, participants, metadata
    array_paths = [
        CLASSIFIER_DIR / "X.npy",
        CLASSIFIER_DIR / "y.npy",
        CLASSIFIER_DIR / "participants.npy",
    ]
    metadata_path = CLASSIFIER_DIR / "all_segment_metadata.csv"
    if REBUILD_SEGMENT_ARRAYS or not all(path.exists() for path in array_paths) or not metadata_path.exists():
        return save_segment_arrays()
    return (
        np.load(array_paths[0]),
        np.load(array_paths[1]),
        np.load(array_paths[2]),
        pd.read_csv(metadata_path).reset_index(drop=True),
    )


class ArousalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class ArousalGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(N_ROIS, 32, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.fc(self.dropout(hidden[-1])).squeeze(-1)


def evaluate_model(model, loader):
    model.eval()
    labels, predictions, probabilities = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probability = torch.sigmoid(model(X_batch.to(DEVICE)))
            labels.append(y_batch.numpy())
            probabilities.append(probability.cpu().numpy())
            predictions.append((probability >= 0.5).float().cpu().numpy())
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    probabilities = np.concatenate(probabilities)
    return {
        "accuracy": np.mean(predictions == labels),
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def run_fold(dataset, train_indices, test_indices):
    set_seed()
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=BATCH_SIZE, shuffle=False)
    model = ArousalGRU().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    progress = tqdm(range(N_EPOCHS), desc="Training")
    for _ in progress:
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        progress.set_postfix(train_loss=f"{total_loss / len(train_loader.dataset):.4f}")

    train_results = evaluate_model(model, train_loader)
    test_results = evaluate_model(model, test_loader)
    test_results["train_accuracy"] = train_results["accuracy"]
    test_results["n_train_segments"] = len(train_indices)
    test_results["n_train_correct"] = int((train_results["predictions"] == train_results["labels"]).sum())
    return test_results


def blocked_within_run_folds(metadata):
    # split segments of each run into N_SPLITS blocks; assign block i to fold i's test set
    # indices refer to segment index in metadata
    test_indices_by_fold = [[] for _ in range(N_SPLITS)]
    for _, run_metadata in metadata.groupby(["participant", "run"], sort=False):
        run_indices = run_metadata.sort_values("segment_start_TR").index.to_numpy()
        for fold, test_chunk in enumerate(np.array_split(run_indices, N_SPLITS)):
            test_indices_by_fold[fold].extend(test_chunk)

    all_indices = np.arange(len(metadata))
    for test_indices in test_indices_by_fold:
        test_indices = np.sort(np.asarray(test_indices, dtype=int))
        train_indices = np.setdiff1d(all_indices, test_indices)
        yield train_indices, test_indices


def exclude_buffer_segments(train_indices, test_indices, metadata):
    # return only train segment indices that are at least BUFFER_TR away from any test segment in the same run
    if BUFFER_TR == 0:
        return train_indices

    train_metadata = metadata.iloc[train_indices]
    test_metadata = metadata.iloc[test_indices]
    keep = np.ones(len(train_indices), dtype=bool)
    for (participant, run), run_test in test_metadata.groupby(["participant", "run"], sort=False):
        same_run_positions = np.flatnonzero(
            (train_metadata["participant"].to_numpy() == participant)
            & (train_metadata["run"].to_numpy() == run)
        )
        run_train = train_metadata.iloc[same_run_positions]
        train_starts = run_train["segment_start_TR"].to_numpy()[:, None]
        train_ends = run_train["segment_end_TR_exclusive"].to_numpy()[:, None]
        test_starts = run_test["segment_start_TR"].to_numpy()[None, :]
        test_ends = run_test["segment_end_TR_exclusive"].to_numpy()[None, :]
        distances = np.maximum(np.maximum(test_starts - train_ends, train_starts - test_ends), 0)
        keep[same_run_positions[np.any(distances < BUFFER_TR, axis=1)]] = False
    return train_indices[keep]


def get_folds(X, y, participants, metadata):
    if ANALYSIS in {"segment_cv", "buffer"}:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        return list(cv.split(X, y))
    if ANALYSIS == "blocked_buffer":
        return list(blocked_within_run_folds(metadata))
    if ANALYSIS == "lopo":
        return [
            (np.where(participants != participant)[0], np.where(participants == participant)[0])
            for participant in np.unique(participants)
        ]
    run_groups = (metadata["participant"].astype(str) + "::" + metadata["run"].astype(str)).to_numpy()
    return list(LeaveOneGroupOut().split(X, y, groups=run_groups))


def output_settings():
    # return the save directory and prefix for output files based on ANALYSIS
    if ANALYSIS == "segment_cv":
        return CLASSIFIER_DIR / "segment_CV", "segment"
    if ANALYSIS == "loro":
        return CLASSIFIER_DIR / "LORO", "loro"
    if ANALYSIS == "lopo":
        return CLASSIFIER_DIR / "LOPO", "lopo"
    folder_prefix = "blocked_buffer" if ANALYSIS == "blocked_buffer" else "buffer"
    return CLASSIFIER_DIR / "buffer" / f"{folder_prefix}_{BUFFER_TR}TR", "segment"


def run_analysis():
    set_seed()
    X, y, participants, metadata = load_segments()
    dataset = ArousalDataset(X, y)
    folds = get_folds(X, y, participants, metadata)
    save_dir, prefix = output_settings()
    save_dir.mkdir(parents=True, exist_ok=True)

    fold_rows, prediction_rows, fold_metrics = [], [], []
    for fold, (train_indices, test_indices) in enumerate(folds, start=1):
        print(f"\n{'=' * 50}\nFold {fold}/{len(folds)}\n{'=' * 50}")
        original_n_train = len(train_indices)
        held_out_participant = metadata.iloc[test_indices[0]]["participant"]
        held_out_run = metadata.iloc[test_indices[0]]["run"]

        if ANALYSIS in {"buffer", "blocked_buffer"}:
            train_indices = exclude_buffer_segments(train_indices, test_indices, metadata)
            print(
                f"Training segments excluded by {BUFFER_TR}-TR buffer: "
                f"{original_n_train - len(train_indices)}"
            )

        results = run_fold(dataset, train_indices, test_indices)
        fold_metrics.append(results)
        print(f"Train accuracy: {results['train_accuracy']:.4f}")
        print(f"Test accuracy: {results['accuracy']:.4f}")

        row = {
            "fold": fold,
            "train_accuracy": results["train_accuracy"],
            "test_accuracy": results["accuracy"],
        }
        if ANALYSIS in {"lopo", "loro"}:
            responsive = results["labels"] == 0
            nonresponsive = results["labels"] == 1
            row = {
                "fold": fold,
                "participant": held_out_participant,
                **({"run": held_out_run} if ANALYSIS == "loro" else {}),
                "train_accuracy": results["train_accuracy"],
                "test_accuracy": results["accuracy"],
                "n_responsive": int(responsive.sum()),
                "n_nonresponsive": int(nonresponsive.sum()),
                "responsive_accuracy_pct": 100 * (results["predictions"][responsive] == 0).mean() if responsive.any() else np.nan,
                "nonresponsive_accuracy_pct": 100 * (results["predictions"][nonresponsive] == 1).mean() if nonresponsive.any() else np.nan,
            }
        elif ANALYSIS in {"buffer", "blocked_buffer"}:
            row.update({
                "n_train_segments": len(train_indices),
                "n_excluded_segments": original_n_train - len(train_indices),
            })
            row = {key: row[key] for key in ["fold", "n_train_segments", "n_excluded_segments", "train_accuracy", "test_accuracy"]}
        fold_rows.append(row)

        train_participants = set(participants[train_indices])
        for segment_index, label_binary, prediction, probability in zip(
            test_indices,
            results["labels"],
            results["predictions"],
            results["probabilities"],
        ):
            segment = metadata.iloc[segment_index]
            prediction_row = {
                "segment_index": int(segment_index),
                "fold": fold,
                "participant": participants[segment_index],
                "run": segment["run"],
                "label": segment["label"],
                "label_binary": int(label_binary),
                "segment_start_TR": int(segment["segment_start_TR"]),
                "segment_end_TR_exclusive": int(segment["segment_end_TR_exclusive"]),
                "prediction": int(prediction),
                "probability_nonresponsive": float(probability),
            }
            if ANALYSIS == "loro":
                prediction_row.update({
                    "participant_seen": int(participants[segment_index] in train_participants),
                })
            prediction_rows.append(prediction_row)

    results_df = pd.DataFrame(fold_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    results_df.to_csv(save_dir / f"{prefix}_fold_results.csv", index=False)
    predictions_df.to_csv(save_dir / f"{prefix}_predictions.csv", index=False)

    responsive = predictions_df["label_binary"] == 0
    nonresponsive = predictions_df["label_binary"] == 1
    summary = {
        "mean_train_accuracy": sum(result["n_train_correct"] for result in fold_metrics) / sum(result["n_train_segments"] for result in fold_metrics),
        "mean_test_accuracy": (predictions_df["prediction"] == predictions_df["label_binary"]).mean(),
        "responsive_test_accuracy_pct": 100 * (predictions_df.loc[responsive, "prediction"] == 0).mean(),
        "nonresponsive_test_accuracy_pct": 100 * (predictions_df.loc[nonresponsive, "prediction"] == 1).mean(),
    }
    if ANALYSIS in {"buffer", "blocked_buffer"}:
        summary["mean_n_training_segments"] = results_df["n_train_segments"].mean()
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(save_dir / f"{prefix}_summary.csv", index=False)

    print(summary_df.to_string(index=False))


def plot_probability_by_run():
    save_dir, prefix = output_settings()
    predictions = pd.read_csv(save_dir / f"{prefix}_predictions.csv")
    plot_dir = save_dir / "probability_by_run"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"responsive": "#2b8cbe", "non-responsive": "#d7301f"}
    label_y = {"responsive": 0, "non-responsive": 1}
    color_map = LinearSegmentedColormap.from_list(
        "probability", [colors["responsive"], colors["non-responsive"]]
    )

    for (participant, run), run_df in predictions.groupby(
        ["participant", "run"], sort=False
    ):
        run_df = run_df.sort_values("segment_start_TR")
        midpoint = (
            run_df["segment_start_TR"] + run_df["segment_end_TR_exclusive"]
        ) / 2
        fig, (probability_ax, label_ax) = plt.subplots(
            2, 1, figsize=(12, 5), sharex=True,
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
        )
        fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.13)
        probability_ax.plot(
            midpoint, run_df["probability_nonresponsive"], color="0.65", linewidth=1
        )
        probability_ax.scatter(
            midpoint, run_df["probability_nonresponsive"],
            c=run_df["label"].map(colors), s=24, edgecolor="white", linewidth=0.35,
        )
        probability_ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        probability_ax.set(ylim=(-0.03, 1.03), ylabel="P(non-responsive)", title=f"{participant} | {run}")
        probability_ax.grid(axis="y", alpha=0.2)
        fig.legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors[label], markeredgecolor="none", label=f"Actual: {label}")
                for label in ("non-responsive", "responsive")
            ],
            loc="upper right", bbox_to_anchor=(0.98, 0.98),
        )
        for row in run_df.itertuples(index=False):
            label_ax.hlines(
                label_y[row.label], row.segment_start_TR,
                row.segment_end_TR_exclusive, color=colors[row.label], linewidth=7,
            )
        label_ax.set_yticks([0, 1], labels=["Responsive", "Non-responsive"])
        label_ax.set(ylim=(-0.5, 1.5), xlabel="Time (TR)", ylabel="Actual")
        top_position = probability_ax.get_position()
        colorbar_ax = fig.add_axes([
            top_position.x0 - 0.075, top_position.y0, 0.015, top_position.height
        ])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=color_map), cax=colorbar_ax
        )
        colorbar.set_ticks([0, 1])
        colorbar.ax.yaxis.set_ticks_position("left")
        fig.savefig(
            plot_dir / f"{participant}_{run}_probability.png",
            dpi=160, bbox_inches="tight",
        )
        plt.close(fig)


def main():
    print(f"Using device: {DEVICE}")
    if RUN_TRAINING:
        run_analysis()
    if MAKE_PROBABILITY_PLOTS:
        plot_probability_by_run()


if __name__ == "__main__":
    main()
