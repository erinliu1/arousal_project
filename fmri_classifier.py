from pathlib import Path
import random

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from tqdm import tqdm

from all_rois import all_rois


# Change these globals, then run this file. ANALYSIS can be:
# "segment_cv", "loro", "lopo", "buffer", or "blocked_buffer".
ANALYSIS = "lopo"
BUFFER_TR = None # only used for "buffer" and "blocked_buffer" analyses
RUN_TRAINING = True
MAKE_PROBABILITY_PLOTS = True
REBUILD_SEGMENT_ARRAYS = True
# Turn on one sampling option at a time. Class-balanced takes precedence if both are True.
PARTICIPANT_CLASS_BALANCED_SAMPLING = True
PARTICIPANT_BALANCED_SAMPLING = False

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


class ParticipantBalancedSampler(Sampler):
    def __init__(self, train_indices, participants, labels, class_balanced=False):
        train_participants = participants[train_indices]
        train_labels = labels[train_indices].astype(int)
        self.participants = np.unique(train_participants)
        self.segment_positions_by_participant = {}
        self.labels_by_participant = {}
        self.segment_positions = {}
        self.class_balanced = class_balanced
        self.n_samples = len(train_indices)
        self.rng = np.random.default_rng(SEED)

        for participant in self.participants:
            self.segment_positions_by_participant[participant] = np.flatnonzero(
                train_participants == participant
            )
            participant_labels = np.unique(train_labels[train_participants == participant])
            self.labels_by_participant[participant] = participant_labels
            for label in participant_labels:
                self.segment_positions[(participant, label)] = np.flatnonzero(
                    (train_participants == participant) & (train_labels == label)
                )

    def __iter__(self):
        for _ in range(self.n_samples):
            participant = self.rng.choice(self.participants)
            if self.class_balanced:
                label = self.rng.choice(self.labels_by_participant[participant])
                positions = self.segment_positions[(participant, label)]
            else:
                positions = self.segment_positions_by_participant[participant]
            yield int(self.rng.choice(positions))

    def __len__(self):
        return self.n_samples


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


def run_fold(dataset, train_indices, test_indices, participants):
    set_seed()
    train_dataset = Subset(dataset, train_indices)
    if PARTICIPANT_CLASS_BALANCED_SAMPLING or PARTICIPANT_BALANCED_SAMPLING:
        sampler = ParticipantBalancedSampler(
            train_indices,
            participants,
            dataset.y.numpy(),
            class_balanced=PARTICIPANT_CLASS_BALANCED_SAMPLING,
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_evaluation_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
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

    train_results = evaluate_model(model, train_evaluation_loader)
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
    if PARTICIPANT_CLASS_BALANCED_SAMPLING:
        sampling_suffix = "_participant_class_balanced"
    elif PARTICIPANT_BALANCED_SAMPLING:
        sampling_suffix = "_participant_balanced"
    else:
        sampling_suffix = ""
    if ANALYSIS == "segment_cv":
        return CLASSIFIER_DIR / f"segment_CV{sampling_suffix}", "segment"
    if ANALYSIS == "loro":
        return CLASSIFIER_DIR / f"LORO{sampling_suffix}", "loro"
    if ANALYSIS == "lopo":
        return CLASSIFIER_DIR / f"LOPO{sampling_suffix}", "lopo"
    folder_prefix = "blocked_buffer" if ANALYSIS == "blocked_buffer" else "buffer"
    buffer_dir = f"buffer{sampling_suffix}"
    return CLASSIFIER_DIR / buffer_dir / f"{folder_prefix}_{BUFFER_TR}TR", "segment"


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

        results = run_fold(dataset, train_indices, test_indices, participants)
        fold_metrics.append(results)
        print(f"Train accuracy: {results['train_accuracy']:.4f}")
        print(f"Test accuracy: {results['accuracy']:.4f}")

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
                "responsive_recall": (results["predictions"][responsive] == 0).mean() if responsive.any() else np.nan,
                "nonresponsive_recall": (results["predictions"][nonresponsive] == 1).mean() if nonresponsive.any() else np.nan,
            }
        elif ANALYSIS in {"buffer", "blocked_buffer"}:
            row = {
                "fold": fold,
                "n_train_segments": len(train_indices),
                "n_excluded_segments": original_n_train - len(train_indices),
                "train_accuracy": results["train_accuracy"],
                "test_accuracy": results["accuracy"],
            }
        else: # segment_cv
            row = {
                "fold": fold,
                "train_accuracy": results["train_accuracy"],
                "test_accuracy": results["accuracy"],
            }
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
    results_df.to_csv(save_dir / f"{prefix}_fold_results.csv", index=False)
    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(save_dir / f"{prefix}_predictions.csv", index=False)

    responsive = predictions_df["label_binary"] == 0
    nonresponsive = predictions_df["label_binary"] == 1
    pooled_labels = predictions_df["label_binary"]
    pooled_predictions = predictions_df["prediction"]
    summary = {
        "pooled_train_accuracy": sum(result["n_train_correct"] for result in fold_metrics) / sum(result["n_train_segments"] for result in fold_metrics),
        "pooled_test_accuracy": (pooled_predictions == pooled_labels).mean(),
        "responsive_recall": (predictions_df.loc[responsive, "prediction"] == 0).mean(),
        "nonresponsive_recall": (predictions_df.loc[nonresponsive, "prediction"] == 1).mean(),
    }
    if ANALYSIS in {"buffer", "blocked_buffer"}:
        summary["mean_n_training_segments"] = results_df["n_train_segments"].mean()
    summary["balanced_test_accuracy"] = balanced_accuracy_score(pooled_labels, pooled_predictions)
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
    color_map = LinearSegmentedColormap.from_list("probability", [colors["responsive"], colors["non-responsive"]])

    for (participant, run), run_df in predictions.groupby(["participant", "run"], sort=False):
        run_df = run_df.sort_values("segment_start_TR")
        motion_TRs = np.load(RAW_DATA_DIR / participant / run / "motion_TRs.npy")
        midpoint = (run_df["segment_start_TR"] + run_df["segment_end_TR_exclusive"]) / 2
        fig, (probability_ax, label_ax) = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})
        fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.13)
        probability_ax.plot(midpoint, run_df["probability_nonresponsive"], color="0.65", linewidth=1)
        probability_ax.scatter(midpoint, run_df["probability_nonresponsive"], c=run_df["label"].map(colors), s=24, edgecolor="white", linewidth=0.35)
        probability_ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        probability_ax.set(ylim=(-0.03, 1.03), ylabel="P(non-responsive)", title=f"{participant} | {run}")
        probability_ax.grid(axis="y", alpha=0.2)
        for motion_TR in motion_TRs:
            for ax in (probability_ax, label_ax):
                ax.axvline(motion_TR, color="0.35", linestyle=":", linewidth=1, alpha=0.7, zorder=3)
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors[label], markeredgecolor="none", label=f"Actual: {label}")
            for label in ("non-responsive", "responsive")
        ]
        legend_handles.append(Line2D([0], [0], color="0.35", linestyle=":", linewidth=1, label="Motion TR"))
        fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))
        for row in run_df.itertuples(index=False):
            label_ax.hlines(label_y[row.label], row.segment_start_TR, row.segment_end_TR_exclusive, color=colors[row.label], linewidth=7)
        label_ax.set_yticks([0, 1], labels=["Responsive", "Non-responsive"])
        label_ax.set(ylim=(-0.5, 1.5), xlabel="Time (TR)", ylabel="Actual")
        top_position = probability_ax.get_position()
        colorbar_ax = fig.add_axes([top_position.x0 - 0.075, top_position.y0, 0.015, top_position.height])
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=color_map), cax=colorbar_ax)
        colorbar.set_ticks([0, 1])
        colorbar.ax.yaxis.set_ticks_position("left")
        fig.savefig(plot_dir / f"{participant}_{run}_probability.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

def summarize_leave_one_out():
    save_dir = CLASSIFIER_DIR
    all_results = []
    for folder_dir in save_dir.iterdir():
        if folder_dir.is_dir() and folder_dir.name.startswith("LORO"):
            analysis = "LORO"
        elif folder_dir.is_dir() and folder_dir.name.startswith("LOPO"):
            analysis = "LOPO"
        else:
            continue
        folder_name = folder_dir.name
        if 'participant_class_balanced' in folder_name:
            class_balanced_sampling = True
            participant_balanced_sampling = True
        elif 'participant_balanced' in folder_name:
            class_balanced_sampling = False
            participant_balanced_sampling = True
        else:
            class_balanced_sampling = False
            participant_balanced_sampling = False
        summary_path = folder_dir / f"{analysis.lower()}_summary.csv"
        summary_df = pd.read_csv(summary_path)
        summary_df["leave_out_type"] = "run" if analysis == "LORO" else "participant"
        summary_df["class_balanced"] = class_balanced_sampling
        summary_df["participant_balanced"] = participant_balanced_sampling
        all_results.append(summary_df)
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        column_order = ["leave_out_type", "class_balanced", "participant_balanced"] + [col for col in combined_results.columns if col not in ["leave_out_type", "class_balanced", "participant_balanced"]]
        combined_results = combined_results[column_order]
        combined_results = combined_results.sort_values(by=["leave_out_type", "participant_balanced", "class_balanced"]).reset_index(drop=True)
        combined_results.to_csv(save_dir / "leave_one_out_summary.csv", index=False)


def summarize_buffer_exclusion():
    buffer_dir = CLASSIFIER_DIR / "buffer"
    all_results = []
    for folder_dir in buffer_dir.iterdir():
        if folder_dir.is_dir() and "buffer" in folder_dir.name:
            summary_path = folder_dir / f"segment_summary.csv"
            summary_df = pd.read_csv(summary_path)
            summary_df["exclusion_type"] = "block" if "blocked" in folder_dir.name else "segment"
            summary_df["buffer_TR"] = int(folder_dir.name.split("_")[-1].replace("TR", ""))
            all_results.append(summary_df)
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        column_order = ["exclusion_type", "buffer_TR"] + [col for col in combined_results.columns if col not in ["exclusion_type", "buffer_TR"]]
        combined_results = combined_results[column_order]
        combined_results = combined_results.sort_values(by=["exclusion_type", "buffer_TR"]).reset_index(drop=True)
        combined_results.to_csv(CLASSIFIER_DIR / "buffer_exclusion_summary.csv", index=False)

def visualize_buffer_exclusion():
    df = pd.read_csv(CLASSIFIER_DIR / "buffer_exclusion_summary.csv")
    df = df[df["mean_n_training_segments"] > 1000]

    line_width = 4
    spine_width = 2.5
    tick_width = 2.5
    label_fontsize = 24
    title_fontsize = 28
    tick_fontsize = 20
    legend_fontsize = 19

    # Sharing the y-axis keeps the scale and ticks identical across both panels.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    # Plot for segment exclusion
    segment_df = df[df["exclusion_type"] == "segment"]
    ax1.plot(segment_df["buffer_TR"], segment_df["responsive_recall"], label="Responsive", linewidth=line_width)
    ax1.plot(segment_df["buffer_TR"], segment_df["nonresponsive_recall"], label="Non-Responsive", linewidth=line_width)
    ax1.set_xlabel("Buffer TR", fontsize=label_fontsize)
    ax1.set_ylabel("Recall", fontsize=label_fontsize)
    ax1.set_title("Segment Exclusion", fontsize=title_fontsize, fontweight="bold")

    # Plot for block exclusion
    block_df = df[df["exclusion_type"] == "block"]
    ax2.plot(block_df["buffer_TR"], block_df["responsive_recall"], label="Responsive", linewidth=line_width)
    ax2.plot(block_df["buffer_TR"], block_df["nonresponsive_recall"], label="Non-Responsive", linewidth=line_width)
    ax2.set_xlabel("Buffer TR", fontsize=label_fontsize)
    ax2.set_ylabel("Recall", fontsize=label_fontsize)
    ax2.set_title("Block Exclusion", fontsize=title_fontsize, fontweight="bold")

    for ax in (ax1, ax2):
        ax.tick_params(axis="both", labelsize=tick_fontsize, width=tick_width, length=8)
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)

    # plot horizontal line at y=responsive recall and non-responsive recall for LORO 
    loro_summary_path = CLASSIFIER_DIR / "LORO" / "loro_summary.csv"
    loro_summary = pd.read_csv(loro_summary_path)
    responsive_recall_loro = loro_summary["responsive_recall"].mean()
    nonresponsive_recall_loro = loro_summary["nonresponsive_recall"].mean()
    ax1.axhline(responsive_recall_loro, color="C0", linestyle="--", linewidth=line_width)
    ax1.axhline(nonresponsive_recall_loro, color="C1", linestyle="--", linewidth=line_width)
    ax2.axhline(responsive_recall_loro, color="C0", linestyle="--", linewidth=line_width)
    ax2.axhline(nonresponsive_recall_loro, color="C1", linestyle="--", linewidth=line_width)

    loro_legend_line = Line2D([0], [0], color="0.5", linestyle="--", linewidth=line_width, label="LORO")
    for ax in (ax1, ax2):
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles=handles + [loro_legend_line], labels=labels + ["LORO"], fontsize=legend_fontsize, frameon=True, loc="upper right")
        legend.get_frame().set_linewidth(spine_width)

    plt.tight_layout()
    plt.savefig(CLASSIFIER_DIR / "buffer_exclusion_summary.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def visualize_leave_one_out(analysis):
    """Plot responsive and non-responsive recall for each held-out fold.

    LORO folds are arranged by participant and run, while LOPO folds have one
    position per participant. Recalls are omitted when the held-out fold does
    not contain any examples of that class.
    """
    analysis = analysis.strip().lower()
    if analysis not in {"loro", "lopo"}:
        raise ValueError('analysis must be "loro" or "lopo"')

    if PARTICIPANT_CLASS_BALANCED_SAMPLING:
        sampling_suffix = "_participant_class_balanced"
    elif PARTICIPANT_BALANCED_SAMPLING:
        sampling_suffix = "_participant_balanced"
    else:
        sampling_suffix = ""

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
                missing_class = (
                    count_column is not None
                    and (row[count_column] == 0 or pd.isna(row[metric]))
                )
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
    print(f"Using device: {DEVICE}")
    print(f"Participant/class-balanced sampling: {PARTICIPANT_CLASS_BALANCED_SAMPLING}")
    print(f"Participant-balanced sampling: {PARTICIPANT_BALANCED_SAMPLING}")
    if RUN_TRAINING:
        run_analysis()
    if MAKE_PROBABILITY_PLOTS:
        plot_probability_by_run()


if __name__ == "__main__":
    # main()
    # summarize_leave_one_out()
    # visualize_leave_one_out("loro")
    # summarize_buffer_exclusion()
    # visualize_buffer_exclusion()
    pass
