import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from tqdm import tqdm

from all_rois import aan_rois, cortical_rois, thalamic_rois, all_rois


RAW_DATA_DIR = Path("raw_data")
CLASSIFIER_DIR = Path("plots/fMRI_classifier_exploration2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MOTION_TR_CACHE = {}
ROI_GROUPS = {
    "all_rois": all_rois,
    "aan_rois": aan_rois,
    "thalamic_rois": thalamic_rois,
    "cortical_rois": cortical_rois,
}

ARGUMENT_SPECS = {    
    "--run-training": {
        "action": "store_true", 
        "help": "Train and save predictions; otherwise only show the settings.",
    },

    "--rois": {"type": str, "nargs": "+", "default": ["all_rois"],
               "help": "list ROI groups i.e. all_rois, aan_rois, thalamic_rois, or cortical_rois, and/or individual ROI names in quotes separated by spaces"},

    "--analysis": {
        "type": str, "choices": ("lopo", "loro"), "default": "loro",
        "help": "Cross-validation unit: leave out one participant or one run.",
    },

    "--seed": {"type": int, "default": 42, "help": "Random seed for reproducible training."},
    "--epochs": {"type": int, "default": 30, "help": "Training passes through each fold."},
    "--batch-size": {"type": int, "default": 32, "help": "Segments per training batch."},
    "--learning-rate": {"type": float, "default": 1e-3, "help": "Adam optimizer learning rate."},

    "--rebuild-segment-arrays": {
        "action": "store_true", 
        "help": "Recreate cached segment arrays from ROI files before training.",
    },
    "--segment-length": {"type": int, "default": 20, "help": "Consecutive TRs in each segment."},
    "--segment-stride": {"type": int, "default": 1, "help": "TR step between segment starts; use segment length for non-overlapping segments."},

    "--responsive-window-size": {"type": int, "default": 30, "help": "Responsive sliding-window width in TRs."},
    "--responsive-window-stride": {"type": int, "default": 1, "help": "TR step between responsive windows."},
    "--responsive-min-bp-trs": {"type": int, "default": 8, "help": "Minimum BP-containing TRs per responsive window."},
    "--responsive-max-bp-gap": {"type": int, "default": 4, "help": "Maximum BP-free gap within a responsive window."},

    "--nonresponsive-min-gap": {"type": int, "default": 20, "help": "Minimum BP-free interval duration in TRs."},
}

def parse_args():
    """Parse classifier and training settings from the command line."""
    parser = argparse.ArgumentParser(
        description="Train an fMRI classifier for behavioral responsiveness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for flag, options in ARGUMENT_SPECS.items():
        parser.add_argument(flag, **options)
    args = parser.parse_args()
    try:
        args.rois, args.roi_label = resolve_rois(args.rois)
        print(f"Selected ({len(args.rois)}) ROIs: {args.rois}")
        print(f"ROI label: {args.roi_label}")
    except ValueError as error:
        parser.error(str(error))
    return args

def resolve_rois(values):
    """
    From argument parser, takes in a list of strings, which could be ROI groups, i.e. "all_rois", "aan_rois", "thalamic_rois", or "cortical_rois", and/or individual ROI names.

    Expand ROI groups into individual ROI names, remove duplicates, and return a folder label for the selected ROIs. Raise ValueError if any unknown ROI names are provided.
    """ 
    invalid = [value for value in values if value not in ROI_GROUPS and value not in all_rois]
    if invalid:
        raise ValueError(f"Unknown ROI selection(s): {', '.join(invalid)}")
    
    rois_from_groups = set()
    rois_from_individual_names = set()
    selected_groups = set()

    for value in values:
        if value in ROI_GROUPS:
            selected_groups.add(value)
            rois_from_groups.update(ROI_GROUPS[value])
        elif value in all_rois:
            rois_from_individual_names.add(value)

    all_selected_rois = rois_from_groups.union(rois_from_individual_names)
    all_selected_rois_ordered = [roi for roi in all_rois if roi in all_selected_rois]
    selected_groups_ordered = [group for group in ROI_GROUPS if group in selected_groups]
    rois_from_individual_names_ordered = [roi for roi in all_rois if roi in rois_from_individual_names and roi not in rois_from_groups]
    
    label = "+".join(selected_groups_ordered + rois_from_individual_names_ordered)
        
    return all_selected_rois_ordered, label

def set_seed(seed):
    """Seed all random number generators for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_run_inventory():
    """Return a DataFrame with columns for participant, run, and participant-run-directories in sorted order."""
    rows = [{"participant": participant_dir.name, "run": run_dir.name, "run_dir": run_dir}
            for participant_dir in RAW_DATA_DIR.iterdir() if participant_dir.is_dir()
            for run_dir in participant_dir.iterdir() if run_dir.is_dir()]
    inventory = pd.DataFrame(rows)
    inventory["_participant_num"] = pd.to_numeric(inventory["participant"].str.extract(r"(\d+)")[0], errors="coerce")
    inventory["_run_num"] = pd.to_numeric(inventory["run"].str.extract(r"(\d+)")[0], errors="coerce")
    return inventory.sort_values(["_participant_num", "_run_num"]).drop(columns=["_participant_num", "_run_num"]).reset_index(drop=True)

def load_motion_TRs(participant, run):
    """Load and cache the motion-contaminated TR indices for one run."""
    key = (participant, run)
    if key not in MOTION_TR_CACHE:
        motion_file = RAW_DATA_DIR / participant / run / "motion_TRs.npy"
        MOTION_TR_CACHE[key] = np.asarray(np.load(motion_file), dtype=int)
    return MOTION_TR_CACHE[key]

def sort_segments(segments):
    """Sort segments numerically by participant, run, and starting TR."""
    if segments.empty:
        return segments
    segments = segments.copy()
    segments["_participant_num"] = pd.to_numeric(segments["participant"].str.extract(r"(\d+)")[0], errors="coerce")
    segments["_run_num"] = pd.to_numeric(segments["run"].str.extract(r"(\d+)")[0], errors="coerce")
    return segments.sort_values(["_participant_num", "_run_num", "segment_start_TR"]).drop(columns=["_participant_num", "_run_num"]).reset_index(drop=True)

def save_responsive_windows(window_size=30, stride=1, min_bp_TRs=8, max_gap=4):
    """Classify and save all sliding windows using BP-count and maximum-gap criteria."""
    if min(window_size, stride, min_bp_TRs, max_gap) <= 0:
        raise ValueError("Responsive window criteria must all be positive")
    if min_bp_TRs > window_size or max_gap > window_size:
        raise ValueError("Responsive BP count and gap criteria cannot exceed the window size")
    output_dir = CLASSIFIER_DIR / "responsiveness" / (f"W={window_size},S={stride},N_min={min_bp_TRs},G_max={max_gap}")
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = get_run_inventory()
    window_rows = []
    for row in inventory.itertuples(index=False):
        n_TRs = len(np.load(row.run_dir / "time_vector.npy"))
        bp_TRs = np.unique(np.load(row.run_dir / "bp_TRs.npy"))
        bp_TRs = bp_TRs[(bp_TRs >= 20) & (bp_TRs < n_TRs)]
        for start in range(20, n_TRs - window_size + 1, stride):
            end = start + window_size
            window_bps = bp_TRs[(bp_TRs >= start) & (bp_TRs < end)]
            # Including both boundaries makes the edge-to-nearest-BP gaps part of G_max.
            edge_and_inter_bp_gaps = np.diff(np.r_[start, window_bps, end])
            maximum_gap = int(edge_and_inter_bp_gaps.max())
            window_rows.append({
                "participant": row.participant, "run": row.run, "start_TR": start, "end_TR": end,
                "n_presses": len(window_bps), "max_within_window_gap": maximum_gap,
                "responsive": len(window_bps) >= min_bp_TRs and maximum_gap <= max_gap,
            })
    columns = ["participant", "run", "start_TR", "end_TR", "n_presses", "max_within_window_gap", "responsive"]
    windows = pd.DataFrame(window_rows, columns=columns)
    windows.to_csv(output_dir / "responsive_windows.csv", index=False)
    return windows, output_dir

def save_responsive_intervals(windows, output_dir, max_gap=4):
    """Merge responsive windows, split long BP gaps, trim boundaries, and save intervals."""
    merged_intervals = []
    for (participant, run), run_windows in windows[windows["responsive"]].groupby(["participant", "run"]):
        run_windows = run_windows.sort_values("start_TR").reset_index(drop=True)
        current_start = int(run_windows.iloc[0]["start_TR"])
        current_end = int(run_windows.iloc[0]["end_TR"])
        for _, window in run_windows.iloc[1:].iterrows():
            if int(window["start_TR"]) <= current_end:
                current_end = max(current_end, int(window["end_TR"]))
                continue
            merged_intervals.append((participant, run, current_start, current_end))
            current_start, current_end = int(window["start_TR"]), int(window["end_TR"])
        merged_intervals.append((participant, run, current_start, current_end))

    # Split merged intervals into smaller intervals if the BP-free gap exceeds max_gap.
    interval_rows = []
    for participant, run, merged_start, merged_end in merged_intervals:
        bp_TRs = np.unique(np.load(RAW_DATA_DIR / participant / run / "bp_TRs.npy"))
        bp_TRs = bp_TRs[(bp_TRs >= merged_start) & (bp_TRs < merged_end)]
        for piece in np.split(bp_TRs, np.flatnonzero(np.diff(bp_TRs) > max_gap) + 1):
            if not len(piece):
                continue
            start, end = int(piece[0]), int(piece[-1]) + 1
            interval_rows.append({
                "participant": participant, "run": run, "start_TR": start, "end_TR": end,
                "duration_TRs": end - start,
                "max_within_window_gap": int(np.diff(piece).max()) if len(piece) > 1 else 0,
            })
    columns = ["participant", "run", "start_TR", "end_TR", "duration_TRs", "max_within_window_gap"]
    intervals = pd.DataFrame(interval_rows, columns=columns)
    intervals.to_csv(output_dir / "responsive_intervals.csv", index=False)
    return intervals

def save_responsive_segments(intervals, output_dir, segment_length=20, stride=1):
    """Extract fixed-length sliding segments from responsive intervals and save them."""
    if segment_length <= 0 or stride <= 0:
        raise ValueError("Segment length and stride must be positive")
    segment_dir = output_dir / f"TR={segment_length},SW={stride}"
    segment_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for interval in intervals.itertuples(index=False):
        participant, run = str(interval.participant), str(interval.run)
        motion_TRs = load_motion_TRs(participant, run)
        for start in range(int(interval.start_TR), int(interval.end_TR) - segment_length + 1, stride):
            end = start + segment_length
            rows.append({
                "participant": participant, "run": run, "label": "responsive",
                "segment_start_TR": start, "segment_end_TR_exclusive": end,
                "contains_motion": bool(np.any((motion_TRs >= start) & (motion_TRs < end))),
            })
    columns = ["participant", "run", "label", "segment_start_TR", "segment_end_TR_exclusive", "contains_motion"]
    segments = sort_segments(pd.DataFrame(rows, columns=columns))
    segments.to_csv(segment_dir / "responsive_segments.csv", index=False)
    return segments

def save_nonresponsive_intervals(threshold=20):
    """Generate and save bounded, end-of-run, and whole-run non-responsive intervals."""
    if threshold <= 0:
        raise ValueError("The non-responsive minimum gap must be positive")
    output_dir = CLASSIFIER_DIR / "non_responsiveness" / f"T_min={threshold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = get_run_inventory()
    bounded, end_intervals, whole_runs = [], [], []
    for row in inventory.itertuples(index=False):
        n_TRs = len(np.load(row.run_dir / "time_vector.npy"))
        bp_TRs = np.unique(np.load(row.run_dir / "bp_TRs.npy"))
        bp_TRs = bp_TRs[(bp_TRs >= 0) & (bp_TRs < n_TRs)]
        base = {"participant": row.participant, "run": row.run}
        if len(bp_TRs) == 0 and n_TRs - 20 >= threshold:
            whole_runs.append({**base, "start_TR": 20, "end_TR": n_TRs, "duration_TRs": n_TRs - 20})
        for start, end in zip(bp_TRs[:-1], bp_TRs[1:]):
            clipped_start = max(int(start), 20)
            if int(end) - int(start) >= threshold and int(end) - clipped_start >= threshold:
                bounded.append({**base, "start_TR": clipped_start, "end_TR": int(end),
                                "duration_TRs": int(end) - clipped_start})
        if len(bp_TRs):
            start = max(int(bp_TRs[-1]), 20)
            if n_TRs - int(bp_TRs[-1]) >= threshold and n_TRs - start >= threshold:
                end_intervals.append({**base, "start_TR": start, "end_TR": n_TRs,
                                      "duration_TRs": n_TRs - start})
    columns = ["participant", "run", "start_TR", "end_TR", "duration_TRs"]
    bounded = pd.DataFrame(bounded, columns=columns)
    end_intervals = pd.DataFrame(end_intervals, columns=columns)
    whole_runs = pd.DataFrame(whole_runs, columns=columns)
    bounded.to_csv(output_dir / "non_responsive_intervals.csv", index=False)
    end_intervals.to_csv(output_dir / "non_responsive_end_intervals.csv", index=False)
    whole_runs.to_csv(output_dir / "non_responsive_whole_runs.csv", index=False)
    return bounded, end_intervals, whole_runs, output_dir

def save_nonresponsive_segments(bounded, end_intervals, whole_runs, output_dir, segment_length=20, stride=1):
    """Extract and save sliding segments from each non-responsive interval category."""
    if segment_length <= 0 or stride <= 0:
        raise ValueError("Segment length and stride must be positive")
    segment_dir = output_dir / f"TR={segment_length},SW={stride}"
    segment_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    interval_specs = (
        (bounded, "bounded_interval"),
        (end_intervals, "end_interval"),
        (whole_runs, "whole_run"),
    )
    for intervals, source_type in interval_specs:
        for interval in intervals.itertuples(index=False):
            participant, run = str(interval.participant), str(interval.run)
            motion_TRs = load_motion_TRs(participant, run)
            for start in range(int(interval.start_TR), int(interval.end_TR) - segment_length + 1, stride):
                end = start + segment_length
                rows.append({
                    "participant": participant, "run": run, "label": "non-responsive",
                    "segment_start_TR": start, "segment_end_TR_exclusive": end,
                    "contains_motion": bool(np.any((motion_TRs >= start) & (motion_TRs < end))),
                    "source_type": source_type,
                })
    columns = ["participant", "run", "label", "segment_start_TR", "segment_end_TR_exclusive",
               "contains_motion", "source_type"]
    segments = sort_segments(pd.DataFrame(rows, columns=columns))
    segments.to_csv(segment_dir / "nonresponsive_segments.csv", index=False)
    return segments

def get_responsive_segments_path(args):
    W, S, N_min, G_max = args.responsive_window_size, args.responsive_window_stride, args.responsive_min_bp_trs, args.responsive_max_bp_gap
    TR, SW = args.segment_length, args.segment_stride
    return CLASSIFIER_DIR / "responsiveness" / f"W={W},S={S},N_min={N_min},G_max={G_max}" / f"TR={TR},SW={SW}" / "responsive_segments.csv"

def get_nonresponsive_segments_path(args):
    T_min = args.nonresponsive_min_gap
    TR, SW = args.segment_length, args.segment_stride
    return CLASSIFIER_DIR / "non_responsiveness" / f"T_min={T_min}" / f"TR={TR},SW={SW}" / "nonresponsive_segments.csv"

def get_responsive_segments(args):
    try:
        return pd.read_csv(get_responsive_segments_path(args))
    except FileNotFoundError:
        responsive_windows, responsive_dir = save_responsive_windows(
            args.responsive_window_size, args.responsive_window_stride,
            args.responsive_min_bp_trs, args.responsive_max_bp_gap,
        )
        responsive_intervals = save_responsive_intervals(
            responsive_windows, responsive_dir, args.responsive_max_bp_gap
        )
        return save_responsive_segments(
            responsive_intervals, responsive_dir, args.segment_length, args.segment_stride
        )

def get_nonresponsive_segments(args):
    try:
        return pd.read_csv(get_nonresponsive_segments_path(args))
    except FileNotFoundError:
        bounded, end_intervals, whole_runs, nonresponsive_dir = save_nonresponsive_intervals(
            args.nonresponsive_min_gap
        )
        return save_nonresponsive_segments(
            bounded, end_intervals, whole_runs, nonresponsive_dir, args.segment_length, args.segment_stride
        )
    
def get_segment_dir(args):
    W, S, N_min, G_max = args.responsive_window_size, args.responsive_window_stride, args.responsive_min_bp_trs, args.responsive_max_bp_gap
    T_min = args.nonresponsive_min_gap
    TR, SW = args.segment_length, args.segment_stride
    segment_dir = CLASSIFIER_DIR / "segments" /f"R(W={W},S={S},N_min={N_min},G_max={G_max}):NR(T_min={T_min}):Segment(TR={TR},SW={SW}):ROIs={args.roi_label}"
    segment_dir.mkdir(parents=True, exist_ok=True)
    return segment_dir

def save_segment_metadata(args):
    """Combine responsive and non-responsive segments, filter motion, and save metadata."""
    responsive_segments = get_responsive_segments(args)
    nonresponsive_segments = get_nonresponsive_segments(args)

    metadata = pd.concat([responsive_segments, nonresponsive_segments], ignore_index=True)
    metadata = metadata.loc[
        ~metadata["contains_motion"].astype(bool),
        ["participant", "run", "label", "segment_start_TR", "segment_end_TR_exclusive"],
    ]
    metadata = metadata.sort_values(["participant", "run", "segment_start_TR"]).reset_index(drop=True)

    segment_dir = get_segment_dir(args)
    metadata.to_csv(segment_dir / "all_segment_metadata.csv", index=False)
    return metadata

def load_segment_metadata(args):
    """Return all motion-free segments from both behavioral classes."""
    try:
        return pd.read_csv(get_segment_dir(args) / "all_segment_metadata.csv")
    except FileNotFoundError:
        return save_segment_metadata(args)

def save_segment_arrays(args):
    """Build and store model inputs, labels, and their segment metadata."""
    metadata = load_segment_metadata(args)
    segment_dir = get_segment_dir(args)
    lengths = metadata["segment_end_TR_exclusive"] - metadata["segment_start_TR"]
    if not (lengths == lengths.iloc[0]).all():
        raise ValueError("All segments must have the same length")
    X = np.empty((len(metadata), int(lengths.iloc[0]), len(args.rois)), dtype=np.float32)
    y = metadata["label"].map({"responsive": 0.0, "non-responsive": 1.0})
    y = y.to_numpy(dtype=np.float32)
    for (participant, run), run_metadata in metadata.groupby(["participant", "run"], sort=False):
        run_data = np.column_stack([
            np.load(RAW_DATA_DIR / participant / run / "z_scored" / f"{roi}.npy") for roi in args.rois
        ]).astype(np.float32, copy=False)
        for segment_index, segment in run_metadata.iterrows():
            start, end = int(segment["segment_start_TR"]), int(segment["segment_end_TR_exclusive"])
            X[segment_index] = run_data[start:end]
    np.save(segment_dir / "X.npy", X)
    np.save(segment_dir / "y.npy", y)
    pd.DataFrame({"roi": args.rois}).to_csv(segment_dir / "rois.csv", index=False)
    return X, y, metadata

def load_segments(args):
    """Load cached segment data, rebuilding it when requested or missing."""
    segment_dir = get_segment_dir(args)
    array_paths = [segment_dir / name for name in ("X.npy", "y.npy")]
    metadata_path = segment_dir / "all_segment_metadata.csv"
    if args.rebuild_segment_arrays or not all(path.exists() for path in array_paths) or not metadata_path.exists():
        print("Building segment arrays in ", segment_dir)
        return save_segment_arrays(args)
    return np.load(array_paths[0]), np.load(array_paths[1]), pd.read_csv(metadata_path).reset_index(drop=True)

class ArousalDataset(Dataset):
    """Store segment arrays and binary labels as PyTorch tensors."""

    def __init__(self, X, y):
        """Convert NumPy inputs and labels to float tensors."""
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        """Return the number of segments."""
        return len(self.X)

    def __getitem__(self, index):
        """Return one segment and its label."""
        return self.X[index], self.y[index]

class ParticipantClassBalancedSampler(Sampler):
    """Sample participants equally, then sample their available classes equally."""

    def __init__(self, train_indices, participants, labels, seed):
        """Index training segments by participant and class."""
        train_participants = participants[train_indices]
        train_labels = labels[train_indices].astype(int)
        self.participants = np.unique(train_participants)
        self.labels_by_participant, self.positions = {}, {}
        self.n_samples = len(train_indices)
        self.rng = np.random.default_rng(seed)
        for participant in self.participants:
            labels = np.unique(train_labels[train_participants == participant])
            self.labels_by_participant[participant] = labels
            for label in labels:
                self.positions[(participant, label)] = np.flatnonzero(
                    (train_participants == participant) & (train_labels == label)
                )

    def __iter__(self):
        """Yield balanced positions within the training subset."""
        for _ in range(self.n_samples):
            participant = self.rng.choice(self.participants)
            label = self.rng.choice(self.labels_by_participant[participant])
            yield int(self.rng.choice(self.positions[(participant, label)]))

    def __len__(self):
        """Return the number of samples drawn per epoch."""
        return self.n_samples

class ArousalGRU(nn.Module):
    """Classify an ROI time series using its final GRU hidden state."""

    def __init__(self, n_rois):
        """Create the recurrent, dropout, and output layers."""
        super().__init__()
        self.gru = nn.GRU(n_rois, 32, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        """Return one non-responsiveness logit per segment."""
        _, hidden = self.gru(x)
        return self.fc(self.dropout(hidden[-1])).squeeze(-1)

def evaluate_model(model, loader):
    """Return accuracy, labels, predictions, and probabilities for a loader."""
    model.eval()
    labels, predictions, probabilities = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probability = torch.sigmoid(model(X_batch.to(DEVICE)))
            labels.append(y_batch.numpy())
            probabilities.append(probability.cpu().numpy())
            predictions.append((probability >= 0.5).float().cpu().numpy())
    labels, predictions, probabilities = map(np.concatenate, (labels, predictions, probabilities))
    return {"accuracy": np.mean(predictions == labels), "labels": labels,
            "predictions": predictions, "probabilities": probabilities}

def run_fold(dataset, train_indices, test_indices, participants, args):
    """Train and evaluate one cross-validation fold."""
    set_seed(args.seed)
    train_dataset = Subset(dataset, train_indices)
    sampler = ParticipantClassBalancedSampler(train_indices, participants, dataset.y.numpy(), args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    train_evaluation_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=args.batch_size, shuffle=False)
    model = ArousalGRU(dataset.X.shape[-1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    progress = tqdm(range(args.epochs), desc="Training")
    for _ in progress:
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        progress.set_postfix(train_loss=f"{total_loss / len(train_dataset):.4f}")

    train_results = evaluate_model(model, train_evaluation_loader)
    test_results = evaluate_model(model, test_loader)
    test_results["train_accuracy"] = train_results["accuracy"]
    return test_results


def get_folds(X, y, participants, metadata, analysis):
    """Create leave-one-participant-out or leave-one-run-out folds."""
    if analysis == "lopo":
        return [(np.flatnonzero(participants != participant), np.flatnonzero(participants == participant))
                for participant in np.unique(participants)]
    run_groups = (metadata["participant"].astype(str) + "::" + metadata["run"].astype(str)).to_numpy()
    return list(LeaveOneGroupOut().split(X, y, groups=run_groups))

def format_learning_rate(learning_rate):
    """Format a learning rate compactly for a model-results folder name."""
    return f"{learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")

def output_settings(args):
    """Return the hierarchical model-results directory and filename prefix."""
    model_dir = f"model=GRU,epochs={args.epochs},batch={args.batch_size},LR={format_learning_rate(args.learning_rate)}"
    save_dir = CLASSIFIER_DIR / "model_results" / args.roi_label / model_dir / args.analysis.upper()
    return save_dir, args.analysis


def run_analysis(args):
    """Run all folds and save fold-level results and segment predictions."""
    set_seed(args.seed)
    X, y, metadata = load_segments(args)
    participants = metadata["participant"].astype(str).to_numpy()
    dataset = ArousalDataset(X, y)
    folds = get_folds(X, y, participants, metadata, args.analysis)
    save_dir, prefix = output_settings(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    fold_rows, prediction_rows = [], []
    for fold, (train_indices, test_indices) in enumerate(folds, start=1):
        print(f"\n{'=' * 50}\nFold {fold}/{len(folds)}\n{'=' * 50}")
        held_out = metadata.iloc[test_indices[0]]
        results = run_fold(dataset, train_indices, test_indices, participants, args)
        print(f"Train accuracy: {results['train_accuracy']:.4f}")
        print(f"Test accuracy: {results['accuracy']:.4f}")

        responsive, nonresponsive = results["labels"] == 0, results["labels"] == 1
        fold_rows.append({
            "fold": fold, "participant": held_out["participant"],
            **({"run": held_out["run"]} if args.analysis == "loro" else {}),
            "train_accuracy": results["train_accuracy"], "test_accuracy": results["accuracy"],
            "n_responsive": int(responsive.sum()), "n_nonresponsive": int(nonresponsive.sum()),
            "responsive_recall": (results["predictions"][responsive] == 0).mean() if responsive.any() else np.nan,
            "nonresponsive_recall": (results["predictions"][nonresponsive] == 1).mean() if nonresponsive.any() else np.nan,
        })

        train_participants = set(participants[train_indices])
        for segment_index, label, prediction, probability in zip(
            test_indices, results["labels"], results["predictions"], results["probabilities"]
        ):
            segment = metadata.iloc[segment_index]
            row = {
                "segment_index": int(segment_index), "fold": fold,
                "participant": participants[segment_index], "run": segment["run"],
                "label": segment["label"], "label_binary": int(label),
                "segment_start_TR": int(segment["segment_start_TR"]),
                "segment_end_TR_exclusive": int(segment["segment_end_TR_exclusive"]),
                "prediction": int(prediction), "probability_nonresponsive": float(probability),
            }
            if args.analysis == "loro":
                row["participant_seen"] = int(participants[segment_index] in train_participants)
            prediction_rows.append(row)

    pd.DataFrame(fold_rows).to_csv(save_dir / f"{prefix}_fold_results.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(save_dir / f"{prefix}_predictions.csv", index=False)
    print(f"Saved fold results and predictions to {save_dir}")


def main():
    """Parse settings and optionally launch classifier training."""
    args = parse_args()
    if args.run_training:
        print(f"Using device: {DEVICE}")
        print(f"Analysis: {args.analysis}")
        print(f"Selected ROIs: {args.rois}")
        print(f"Running training with settings: {args}")
        run_analysis(args)


if __name__ == "__main__":
    main()
