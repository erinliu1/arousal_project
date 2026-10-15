from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from tqdm import tqdm
import random

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader

from all_rois import all_rois

SEED = 42
BUFFER_TR = 0

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

RAW_DATA_DIR = Path('raw_data')
SEGMENTS_DIR = Path('plots/label_segments')
CLASSIFIER_DIR = Path('plots/classifier')

SAVE_DIR = CLASSIFIER_DIR / f"buffers" / f"blocked_buffer_{BUFFER_TR}TR"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PROBABILITY_PLOT_DIR = SAVE_DIR / "probability_by_run"

N_ROIS = len(all_rois)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {DEVICE}")

def load_all_segment_metadata():
    # load all non-motion segments for both responsive and non-responsive labels
    segment_data = []
    for segment_file in SEGMENTS_DIR.glob('*.csv'):
        df = pd.read_csv(segment_file)
        segment_data.append(df)
    segment_data = pd.concat(segment_data, ignore_index=True)
    segment_data = segment_data[segment_data['contains_motion'] == False][['participant', 'run', 'label', 'segment_start_TR', 'segment_end_TR_exclusive']]
    segment_data = segment_data.sort_values(['participant', 'run', 'segment_start_TR'])
    segment_data.to_csv(CLASSIFIER_DIR / 'all_segment_metadata.csv', index=False)
    return segment_data

def save_all_segments():
    metadata = load_all_segment_metadata().reset_index(drop=True)
    
    n_segments = len(metadata)
    segment_length = (metadata['segment_end_TR_exclusive'] - metadata['segment_start_TR']).unique()[0]
    label_values = {'responsive': 0.0, 'non-responsive': 1.0}
    
    X = np.empty((n_segments, segment_length, N_ROIS), dtype=np.float32)
    y = metadata['label'].map(label_values).to_numpy(dtype=np.float32)
    participants = metadata['participant'].to_numpy(dtype=str)

    # run_metadata is a dataframe containing all segments for a given participant and run
    for (participant, run), run_metadata in metadata.groupby(['participant', 'run'], sort=False):
        # load all ROI data (full run timeseries) for this participant/run into a single array
        # run_data: (n_TRs, N_ROIS)

        all_roi_data = []
        for roi in all_rois:
            roi_path = RAW_DATA_DIR / participant / run / 'z_scored' / f'{roi}.npy'
            all_roi_data.append(np.load(roi_path))
        run_data = np.column_stack(all_roi_data).astype(np.float32, copy=False)
        
        # iterate through all segments
        for row_index, segment in run_metadata.iterrows():
            # row_index is a unique index for segment_metadata; keep same for X, y, and participants
            start = segment['segment_start_TR']
            end = segment['segment_end_TR_exclusive']
            X[row_index] = run_data[start:end] # (10 TRs, N_ROIS)

    np.save(CLASSIFIER_DIR / "X.npy", X)
    np.save(CLASSIFIER_DIR / "y.npy", y)
    np.save(CLASSIFIER_DIR / "participants.npy", participants)

    return X, y, participants

def load_all_segments():
    if not ((CLASSIFIER_DIR / "X.npy").exists() and (CLASSIFIER_DIR / "y.npy").exists() and (CLASSIFIER_DIR / "participants.npy").exists()):
        return save_all_segments()

    X = np.load(CLASSIFIER_DIR / "X.npy")
    y = np.load(CLASSIFIER_DIR / "y.npy")
    participants = np.load(CLASSIFIER_DIR / "participants.npy")
    return X, y, participants

def get_fold_datasets(dataset, train_indices, test_indices):
    # dataset should be an instance of ArousalDataset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset

def get_blocked_within_run_folds(metadata, n_splits=10):
    test_indices_by_fold = [[] for _ in range(n_splits)]

    for _, run_metadata in metadata.groupby(['participant', 'run'], sort=False):
        sorted_run_indices = (
            run_metadata.sort_values('segment_start_TR').index.to_numpy(dtype=int)
        )
        for fold_index, test_chunk in enumerate(np.array_split(sorted_run_indices, n_splits)):
            test_indices_by_fold[fold_index].extend(test_chunk.tolist())

    all_indices = np.arange(len(metadata))
    for test_indices in test_indices_by_fold:
        test_indices = np.sort(np.asarray(test_indices, dtype=int))
        train_mask = np.ones(len(metadata), dtype=bool)
        train_mask[test_indices] = False
        yield all_indices[train_mask], test_indices

def exclude_nearby_training_segments(train_indices, test_indices, metadata, buffer_tr=BUFFER_TR):
    if buffer_tr < 0:
        raise ValueError("BUFFER_TR must be nonnegative")
    if buffer_tr == 0:
        return train_indices

    train_metadata = metadata.iloc[train_indices]
    test_metadata = metadata.iloc[test_indices]
    keep = np.ones(len(train_indices), dtype=bool)

    for (participant, run), run_test_metadata in test_metadata.groupby(['participant', 'run'], sort=False):
        same_run = (
            (train_metadata['participant'].to_numpy() == participant)
            & (train_metadata['run'].to_numpy() == run)
        )
        same_run_positions = np.flatnonzero(same_run)
        if len(same_run_positions) == 0:
            continue

        run_train_metadata = train_metadata.iloc[same_run_positions]
        train_starts = run_train_metadata['segment_start_TR'].to_numpy()
        train_ends = run_train_metadata['segment_end_TR_exclusive'].to_numpy()
        test_block_start = run_test_metadata['segment_start_TR'].min()
        test_block_end = run_test_metadata['segment_end_TR_exclusive'].max()

        temporal_distances = np.maximum(
            np.maximum(test_block_start - train_ends, train_starts - test_block_end),
            0,
        )
        keep[same_run_positions[temporal_distances < buffer_tr]] = False

    return train_indices[keep]

class ArousalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class ArousalGRU(nn.Module):
    def __init__(self, input_size=N_ROIS, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, 10 TRs, 44 ROIs)
        _, hidden = self.gru(x)
        # hidden: (1, batch_size, hidden_size)
        x = hidden[-1]
        x = self.dropout(x)
        logits = self.fc(x).squeeze(-1)
        return logits

def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(train_loader.dataset)

def train_model(model, train_loader, n_epochs=30, learning_rate=1e-3, log=False):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    progress_bar = tqdm(range(n_epochs), desc="Training")
    for epoch in progress_bar:
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        progress_bar.set_postfix({"train_loss": f"{train_loss:.4f}"})
    return model

def evaluate_model(model, test_loader):
    model.eval()
    all_labels, all_predictions, all_probabilities = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()
            all_labels.append(y_batch.numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    accuracy = np.mean(predictions == labels)
    return {
        'accuracy': accuracy,
        'labels': labels,
        'predictions': predictions,
        'probabilities': probabilities
    }

def run_cv_fold(dataset, train_indices, test_indices, batch_size=32, n_epochs=30):
    set_seed()
    
    train_dataset, test_dataset = get_fold_datasets(dataset, train_indices, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = ArousalGRU().to(DEVICE)
    model = train_model(model, train_loader, n_epochs=n_epochs)

    train_results = evaluate_model(model, train_loader)
    test_results = evaluate_model(model, test_loader)
    test_results['train_accuracy'] = train_results['accuracy']
    return test_results

def summarize_participant_predictions(predictions_df):
    summaries = []
    for participant, participant_df in predictions_df.groupby('participant', sort=False):
        responsive = participant_df['label'] == 0
        nonresponsive = participant_df['label'] == 1

        summaries.append({
            'participant': participant,
            'n_responsive': int(responsive.sum()),
            'n_nonresponsive': int(nonresponsive.sum()),
            'pct_predicted_responsive': 100 * (participant_df['prediction'] == 0).mean(),
            'pct_predicted_nonresponsive': 100 * (participant_df['prediction'] == 1).mean(),
            'responsive_accuracy_pct': 100 * (participant_df.loc[responsive, 'prediction'] == 0).mean(),
            'nonresponsive_accuracy_pct': 100 * (participant_df.loc[nonresponsive, 'prediction'] == 1).mean(),
        })

    return pd.DataFrame(summaries)

def main():
    set_seed()

    X, y, participant_indices = load_all_segments()
    dataset = ArousalDataset(X, y)
    metadata = pd.read_csv(CLASSIFIER_DIR / 'all_segment_metadata.csv').reset_index(drop=True)
    if len(metadata) != len(dataset):
        raise ValueError("Segment metadata and saved segment arrays are not aligned")

    all_results, all_predictions = [], []
    for fold, (train_indices, test_indices) in enumerate(get_blocked_within_run_folds(metadata), start=1):
        original_n_train = len(train_indices)
        train_indices = exclude_nearby_training_segments(train_indices, test_indices, metadata)
        print(f"\n{'=' * 50}")
        print(f"Fold {fold}/10")
        print(f"Training segments excluded by {BUFFER_TR}-TR buffer: {original_n_train - len(train_indices)}")
        print(f"{'=' * 50}")

        results = run_cv_fold(dataset, train_indices, test_indices, batch_size=32, n_epochs=30)
        print(f"Fold {fold} train accuracy: {results['train_accuracy']:.4f}")
        print(f"Fold {fold} test accuracy: {results['accuracy']:.4f}")

        all_results.append({
            'fold': fold,
            'n_train_segments': len(train_indices),
            'n_excluded_segments': original_n_train - len(train_indices),
            'train_accuracy': results['train_accuracy'],
            'test_accuracy': results['accuracy'],
        })
        
        for segment_index, label, prediction, probability in zip(test_indices, results['labels'], results['predictions'], results['probabilities']):
            all_predictions.append({
                'segment_index': int(segment_index),
                'fold': fold,
                'participant': participant_indices[segment_index],
                'label': int(label),
                'prediction': int(prediction),
                'probability_nonresponsive': float(probability)
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(SAVE_DIR / f"segment_fold_results_buffer.csv", index=False)

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(SAVE_DIR / f"segment_fold_predictions_buffer.csv", index=False)

    participant_summary_df = summarize_participant_predictions(predictions_df)
    participant_summary_df.to_csv(SAVE_DIR / f"segment_fold_participant_summary_buffer.csv", index=False, float_format='%.2f', na_rep='N/A')
    join_predictions_to_segments()

    print(f"Mean fold train accuracy: {results_df['train_accuracy'].mean():.4f}")
    print(f"Mean fold test accuracy: {results_df['test_accuracy'].mean():.4f}")
    print(f"Mean segment accuracy: {(predictions_df['label'] == predictions_df['prediction']).mean():.4f}")

def join_predictions_to_segments():
    metadata = pd.read_csv(CLASSIFIER_DIR / 'all_segment_metadata.csv')
    predictions = pd.read_csv(SAVE_DIR / f"segment_fold_predictions_buffer.csv").sort_values('segment_index')

    joined = pd.concat(
        [
            metadata.reset_index(drop=True),
            predictions[["prediction", "probability_nonresponsive"]].reset_index(drop=True),
        ],
        axis=1,
    )
    joined.to_csv(SAVE_DIR / f"segment_results_buffer.csv", index=False)
    return joined

def plot_probability_by_run():
    segment_results = pd.read_csv(SAVE_DIR / "segment_results_buffer.csv")
    PROBABILITY_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"responsive": "#2b8cbe", "non-responsive": "#d7301f"}
    label_y = {"responsive": 0, "non-responsive": 1}
    probability_cmap = LinearSegmentedColormap.from_list("probability", [colors["responsive"], colors["non-responsive"]])

    for (participant, run), run_df in segment_results.groupby(["participant", "run"], sort=False):
        run_df = run_df.sort_values("segment_start_TR")
        midpoint = (run_df["segment_start_TR"] + run_df["segment_end_TR_exclusive"]) / 2
        fig, (probability_ax, label_ax) = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})
        fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.13)

        probability_ax.plot(midpoint, run_df["probability_nonresponsive"], color="0.65", linewidth=1)
        probability_ax.scatter(midpoint, run_df["probability_nonresponsive"], c=run_df["label"].map(colors), s=24, edgecolor="white", linewidth=0.35, zorder=2)
        probability_ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        probability_ax.set_ylim(-0.03, 1.03)
        probability_ax.set_ylabel("P(non-responsive)")
        probability_ax.yaxis.set_label_coords(-0.05, 0.5)
        probability_ax.set_title(f"{participant} | {run}")
        probability_ax.grid(axis="y", alpha=0.2)

        legend_handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors[label], markeredgecolor="none", label=f"Actual: {label}") for label in ("non-responsive", "responsive")]
        fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, facecolor="white", edgecolor="0.85")

        for row in run_df.itertuples(index=False):
            label_ax.hlines(label_y[row.label], row.segment_start_TR, row.segment_end_TR_exclusive, color=colors[row.label], linewidth=7)
        label_ax.set_yticks([0, 1], labels=["Responsive", "Non-responsive"])
        label_ax.set_ylim(-0.5, 1.5)
        label_ax.set_xlabel("Time (TR)")
        label_ax.set_ylabel("Actual")
        label_ax.grid(axis="x", alpha=0.15)

        top_position = probability_ax.get_position()
        gradient_ax = fig.add_axes([top_position.x0 - 0.075, top_position.y0, 0.015, top_position.height])
        probability_scale = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=probability_cmap)
        colorbar = fig.colorbar(probability_scale, cax=gradient_ax)
        colorbar.set_ticks([0, 1])
        colorbar.ax.yaxis.set_ticks_position("left")

        output_path = PROBABILITY_PLOT_DIR / f"{participant}_{run}_probability.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved run plots to {PROBABILITY_PLOT_DIR}")


if __name__ == "__main__":
    main()
    
