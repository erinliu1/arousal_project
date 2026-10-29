"""Nonlinear classifier on the exact ROI summary features used by LR.py.

    python MLP.py --analysis lopo --segment-length 40 --segment-stride 40 --seed 42
    python MLP.py --analysis loro --segment-length 20 --segment-stride 20 --seed 123
    python MLP.py --summarize

Default: all ROIs, trim disabled, LOPO, 40-TR length/stride, one 32-unit ReLU
hidden layer, dropout=0.2, Adam LR=1e-3, weight decay=1e-3, 30 epochs.
CPU training; use the same Python environment and data as the GRU and LR.
Each outer fold uses its final epoch. Test curves are diagnostic only: no
inner tuning, early stopping, threshold selection, or test-based checkpointing.
Outputs: plots/MLP/<segment settings>/<ROIs>/<model settings>/<analysis>/SEED=*.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from LR import extract_features, macro_group_balanced_accuracy, participant_class_weights
from fmri_classifier_exploration2 import (
    ARGUMENT_SPECS, ParticipantClassBalancedSampler,
    get_folds, get_segment_dir_name, load_segments, resolve_rois, set_seed,
)
from scatterplots import visualize_leave_one_out

CLASSIFIER_DIR = Path("plots/MLP")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for flag in (
        "--rois", "--analysis", "--seed", "--segment-length", "--segment-stride",
        "--responsive-window-size", "--responsive-window-stride",
        "--responsive-min-bp-trs", "--responsive-max-bp-gap",
        "--trim-responsive-intervals", "--nonresponsive-min-gap",
        "--rebuild-segment-arrays", "--epochs", "--batch-size",
        "--learning-rate", "--weight-decay", "--dropout", "--hidden-size", "--summarize",
    ):
        options = ARGUMENT_SPECS[flag].copy()
        if flag == "--hidden-size":
            options["help"] = "Number of units in the single ReLU hidden layer."
        parser.add_argument(flag, **options)
    parser.set_defaults(analysis="lopo")
    parser.add_argument("--plot", action="store_true", help="Also save a fold-recall plot.")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for PyTorch.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_size, args.threads, args.segment_stride) < 1 or args.segment_length < 2:
        parser.error("Epochs, batch size, hidden size, threads, stride must be positive; length >= 2.")
    if not np.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        parser.error("Dropout must be finite and in [0, 1).")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Weight decay must be finite and nonnegative.")
    try:
        args.rois, args.roi_label = resolve_rois(args.rois)
    except ValueError as error:
        parser.error(str(error))
    return args


class SummaryMLP(nn.Module):
    def __init__(self, n_features, hidden_size=32, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, 1),
        )

    def forward(self, X):
        return self.layers(X).squeeze(-1)


def classification_metrics(y, predictions):
    """Balanced accuracy averages recalls for classes present in a fold."""
    recalls = {}
    for label, name in ((0, "responsive_recall"), (1, "nonresponsive_recall")):
        mask = y == label
        recalls[name] = float((predictions[mask] == label).mean()) if mask.any() else np.nan
    return {"test_accuracy": float((predictions == y).mean()),
            "balanced_test_accuracy": float(np.nanmean(list(recalls.values()))), **recalls}


def epoch_class_metrics(model, loader):
    """CPU evaluation with the same per-class history columns as the GRU."""
    model.eval()
    labels, predictions, losses = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            labels.append(y_batch.numpy())
            predictions.append((logits >= 0).numpy())
            losses.append(nn.functional.binary_cross_entropy_with_logits(
                logits, y_batch, reduction="none").numpy())
    labels, predictions, losses = map(np.concatenate, (labels, predictions, losses))
    metrics = {}
    for prefix, label in (("r", 0), ("nr", 1)):
        mask = labels == label
        for name, values in (("loss", losses[mask]),
                             ("recall", predictions[mask] == label)):
            values = values.astype(np.float64)
            key = f"{prefix}_{name}"
            metrics.update({key: values.mean() if len(values) else np.nan,
                            f"{key}_count": len(values), f"{key}_sum": values.sum(),
                            f"{key}_sum_sq": np.square(values).sum()})
    return metrics


def run_fold(features, y, participants, train, test, args):
    """Train only on training participants/runs; evaluate the fixed final epoch."""
    if len(np.unique(y[train])) != 2:
        raise ValueError("A training fold contains only one class.")
    set_seed(args.seed)
    weights = participant_class_weights(participants[train], y[train])
    scaler = StandardScaler().fit(features[train], sample_weight=weights)
    X_train = torch.tensor(scaler.transform(features[train]), dtype=torch.float32)
    X_test = torch.tensor(scaler.transform(features[test]), dtype=torch.float32)
    train_data = TensorDataset(X_train, torch.tensor(y[train], dtype=torch.float32))
    test_data = TensorDataset(X_test, torch.tensor(y[test], dtype=torch.float32))
    sampler = ParticipantClassBalancedSampler(train, participants, y, args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler)
    # Sampling balances participants/classes; applying additional loss weights
    # would double-count that correction, so BCE below is unweighted.
    train_eval = DataLoader(train_data, batch_size=512, shuffle=False)
    test_eval = DataLoader(test_data, batch_size=512, shuffle=False)
    model = SummaryMLP(features.shape[1], args.hidden_size, args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
        for split, loader in (("train", train_eval), ("test", test_eval)):
            history.append({"epoch": epoch, "split": split, **epoch_class_metrics(model, loader)})
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_test)).numpy()
        train_predictions = (model(X_train).numpy() >= 0).astype(int)
    return {
        "predictions": (probabilities >= 0.5).astype(int),
        "probabilities": probabilities,
        "train_accuracy": float((train_predictions == y[train]).mean()),
        "history": pd.DataFrame(history), "scaler": scaler,
    }


def output_settings(args):
    model_dir = (
        f"model=MLP,hidden={args.hidden_size},drop={args.dropout:g},epochs={args.epochs},"
        f"batch={args.batch_size},LR={args.learning_rate:g},WD={args.weight_decay:g}"
    )
    return (CLASSIFIER_DIR / get_segment_dir_name(args) / args.roi_label
            / model_dir / args.analysis.upper() / f"SEED={args.seed}")


def run_analysis(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    X, y, metadata = load_segments(args)
    metadata = metadata.reset_index(drop=True)
    y = np.asarray(y, dtype=np.int64)
    if len(X) != len(y) or len(y) != len(metadata) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected aligned segment arrays/metadata with both binary classes.")
    labels = metadata["label"].map({"responsive": 0, "non-responsive": 1}).to_numpy()
    if not np.array_equal(labels, y):
        raise ValueError("Cached labels do not match segment metadata.")
    features, names = extract_features(X, args.rois)
    participants = metadata["participant"].astype(str).to_numpy()
    groups = participants if args.analysis == "lopo" else (
        metadata["participant"].astype(str) + "::" + metadata["run"].astype(str)
    ).to_numpy()
    folds = get_folds(X, y, participants, metadata, args.analysis)
    output = output_settings(args)
    output.mkdir(parents=True, exist_ok=True)
    prefix = args.analysis
    settings = vars(args).copy()
    settings.update(model="MLP", activation="ReLU", hidden_layers=1,
                    features=["mean", "sd", "slope_per_TR"], feature_names=names,
                    optimizer="Adam", device="cpu", threshold=0.5,
                    weighting="participant/class-balanced sampling; weighted train-only scaling",
                    checkpoint_selection="fixed final epoch; no test-based tuning",
                    history_test_split="outer test fold; diagnostic only")
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    pd.DataFrame(features, columns=names).to_csv(output / "features.csv", index=False)
    metadata.to_csv(output / "segment_metadata.csv", index=False)
    predictions = metadata.copy()
    predictions.insert(0, "segment_index", np.arange(len(y)))
    predictions["label_binary"] = y
    predictions["fold"] = -1
    predictions["prediction"] = -1
    predictions["probability_nonresponsive"] = np.nan
    if args.analysis == "loro":
        predictions["participant_seen"] = 0
    rows, scaler_rows, train_sizes = [], [], []
    for fold, (train, test) in enumerate(folds, start=1):
        print(f"Fold {fold}/{len(folds)}: held out {groups[test[0]]}", flush=True)
        result = run_fold(features, y, participants, train, test, args)
        predictions.loc[test, "fold"] = fold
        predictions.loc[test, "prediction"] = result["predictions"]
        predictions.loc[test, "probability_nonresponsive"] = result["probabilities"]
        if args.analysis == "loro":
            predictions.loc[test, "participant_seen"] = np.isin(participants[test], participants[train]).astype(int)
        rows.append({
            "fold": fold, "participant": participants[test[0]],
            **({"run": metadata.iloc[test[0]]["run"]} if args.analysis == "loro" else {}),
            "train_accuracy": result["train_accuracy"],
            "n_responsive": int((y[test] == 0).sum()),
            "n_nonresponsive": int((y[test] == 1).sum()),
            **classification_metrics(y[test], result["predictions"]),
        })
        train_sizes.append(len(train))
        history = result["history"]
        history.insert(0, "fold", fold)
        history.to_csv(output / f"{prefix}_training_history.csv",
                       mode="w" if fold == 1 else "a", header=fold == 1, index=False)
        scaler_rows.extend({"fold": fold, "feature": name, "scaler_mean": float(mean),
                            "scaler_scale": float(scale)}
                           for name, mean, scale in zip(names, result["scaler"].mean_, result["scaler"].scale_))
        pd.DataFrame(rows).to_csv(output / f"{prefix}_fold_results.csv", index=False)
        print(f"Train accuracy={result['train_accuracy']:.4f}, test accuracy={rows[-1]['test_accuracy']:.4f}", flush=True)
    if predictions["fold"].lt(0).any():
        raise RuntimeError("Some segments have no held-out prediction.")
    pred = predictions["prediction"].to_numpy()
    metrics = classification_metrics(y, pred)
    metrics["pooled_test_accuracy"] = metrics.pop("test_accuracy")
    metrics.update(
        pooled_train_accuracy=float(np.average([row["train_accuracy"] for row in rows], weights=train_sizes)),
        fold_averaged_accuracy=float(np.mean([row["test_accuracy"] for row in rows])),
        group_averaged_balanced_accuracy=macro_group_balanced_accuracy(y, pred, groups),
        n_segments=len(y), n_folds=len(folds), n_features=len(names),
    )
    predictions.to_csv(output / f"{prefix}_predictions.csv", index=False)
    pd.DataFrame(scaler_rows).to_csv(output / f"{prefix}_scalers.csv", index=False)
    summary = pd.DataFrame([metrics])
    summary.to_csv(output / f"{prefix}_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved results to {output}")
    if args.plot:
        visualize_leave_one_out(args.analysis, output)
    return output


def summarize():
    rows = []
    for path in sorted(CLASSIFIER_DIR.rglob("settings.json")):
        settings = json.loads(path.read_text())
        summary_path = path.parent / f"{settings['analysis']}_summary.csv"
        if not summary_path.exists():
            continue
        parameters = {key: value for key, value in settings.items()
                      if not isinstance(value, (list, dict)) and key not in {"summarize", "plot", "rebuild_segment_arrays"}}
        for metrics in pd.read_csv(summary_path).to_dict("records"):
            rows.append({**parameters, **metrics, "result_dir": str(path.parent)})
    if not rows:
        print("No completed MLP runs to summarize.")
        return
    output = CLASSIFIER_DIR / "results" / "all_rois_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {len(rows)} results to {output}")


def main():
    args = parse_args()
    if args.summarize:
        summarize()
    else:
        run_analysis(args)


if __name__ == "__main__":
    main()
