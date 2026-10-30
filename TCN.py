"""Small temporal CNN on the same ROI sequences and folds as the GRU.

    python TCN.py --analysis lopo --seed 42
    python TCN.py --analysis loro --segment-length 50 --segment-stride 50
    python TCN.py --summarize
    python TCN.py --analysis lopo --plot-history

Default: untrimmed length=stride=40, all 42 ROIs, fixed 30 epochs, Adam
LR=1e-3, weight decay=1e-2, dropout=0.2. There is no inner CV or ensembling.
A 1x1 projection produces 16 channels, followed by four residual blocks with
2 convolutions each, kernel 3, dilations 1/2/4/8, and temporal mean readout.
The receptive field is 61 TRs. Symmetric padding is appropriate for offline
segment classification: context is used only within each input segment.
Inputs use the existing run-wise z-scoring, exactly as in the GRU.
Outputs: plots/TCN/<segment settings>/<ROI selection>/<model settings>/
         <LOPO or LORO>/SEED=*/ (CSVs and fold/history PNGs).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from LR import macro_group_balanced_accuracy
from fmri_classifier_exploration2 import (
    ARGUMENT_SPECS, DEVICE, ArousalDataset, ParticipantClassBalancedSampler,
    epoch_class_metrics, evaluate_model, format_learning_rate, get_folds,
    get_segment_dir_name, load_segments, resolve_rois, set_seed,
)
from scatterplots import plot_training_history, visualize_leave_one_out

CLASSIFIER_DIR = Path("plots/TCN")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in (
        "--rois", "--analysis", "--seed", "--segment-length", "--segment-stride",
        "--responsive-window-size", "--responsive-window-stride", "--responsive-min-bp-trs",
        "--responsive-max-bp-gap", "--trim-responsive-intervals", "--nonresponsive-min-gap",
        "--rebuild-segment-arrays", "--epochs", "--batch-size", "--learning-rate",
        "--weight-decay", "--dropout", "--summarize",
    ):
        options = ARGUMENT_SPECS[flag].copy()
        if flag == "--epochs":
            options["help"] = "Fixed training epochs per outer fold; no inner CV."
        elif flag == "--dropout":
            options["help"] = "Elementwise dropout within the residual temporal blocks."
        parser.add_argument(flag, **options)
    parser.set_defaults(analysis="lopo", epochs=30, weight_decay=1e-2)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--plot-history", action="store_true",
                        help="Plot existing history for matching settings without training.")
    args = parser.parse_args()
    if min(args.channels, args.epochs, args.batch_size, args.segment_stride, args.threads) < 1 or args.segment_length < 2:
        parser.error("Channels, epochs, batch size, stride, threads must be positive; length >= 2.")
    if args.kernel_size < 1 or args.kernel_size % 2 == 0 or min(args.dilations) < 1:
        parser.error("Kernel size must be positive and odd; dilations must be positive.")
    if not np.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        parser.error("Dropout must be in [0, 1).")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Weight decay must be finite and nonnegative.")
    try:
        args.rois, args.roi_label = resolve_rois(args.rois)
    except ValueError as error:
        parser.error(str(error))
    return args


class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, X):
        return torch.relu(X + self.layers(X))


class ArousalTCN(nn.Module):
    def __init__(self, n_rois, channels=16, kernel_size=3, dilations=(1, 2, 4, 8), dropout=0.2):
        super().__init__()
        self.projection = nn.Conv1d(n_rois, channels, kernel_size=1)
        self.blocks = nn.Sequential(*[
            TemporalBlock(channels, kernel_size, dilation, dropout) for dilation in dilations
        ])
        self.fc = nn.Linear(channels, 1)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)

    def forward(self, X):
        # The shared loader returns (batch, TRs, ROIs); Conv1d takes channels first.
        hidden = self.blocks(self.projection(X.transpose(1, 2)))
        return self.fc(hidden.mean(dim=-1)).squeeze(-1)


def classification_metrics(labels, predictions):
    recalls = {}
    for label, name in ((0, "responsive_recall"), (1, "nonresponsive_recall")):
        mask = labels == label
        recalls[name] = float((predictions[mask] == label).mean()) if mask.any() else np.nan
    return {"test_accuracy": float((labels == predictions).mean()),
            "balanced_test_accuracy": float(np.nanmean(list(recalls.values()))), **recalls}


def run_fold(dataset, train, test, participants, args):
    """Use the GRU's exact participant/class-balanced sampling and objective."""
    if len(np.unique(dataset.y[train].numpy())) != 2:
        raise ValueError("A training fold contains only one class.")
    set_seed(args.seed)
    train_data = Subset(dataset, train)
    sampler = ParticipantClassBalancedSampler(train, participants, dataset.y.numpy(), args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler)
    train_eval = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    test_eval = DataLoader(Subset(dataset, test), batch_size=args.batch_size, shuffle=False)
    model = ArousalTCN(dataset.X.shape[-1], args.channels, args.kernel_size, args.dilations, args.dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch.to(DEVICE)), y_batch.to(DEVICE))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
        for split, loader in (("train", train_eval), ("test", test_eval)):
            history.append({"epoch": epoch, "split": split, **epoch_class_metrics(model, loader)})
    results = evaluate_model(model, test_eval)
    results["train_accuracy"] = evaluate_model(model, train_eval)["accuracy"]
    return results, pd.DataFrame(history)


def output_settings(args):
    model = (f"model=TCN,channels={args.channels},kernel={args.kernel_size},"
             f"dilations={'-'.join(map(str, args.dilations))},drop={args.dropout:g},"
             f"epochs={args.epochs},batch={args.batch_size},"
             f"LR={format_learning_rate(args.learning_rate)},WD={format_learning_rate(args.weight_decay)}")
    return CLASSIFIER_DIR / get_segment_dir_name(args) / args.roi_label / model / args.analysis.upper() / f"SEED={args.seed}"


def run_analysis(args):
    torch.set_num_threads(args.threads)
    X, y, metadata = load_segments(args)
    metadata = metadata.reset_index(drop=True)
    y = np.asarray(y, dtype=np.float32)
    if len(X) != len(y) or len(y) != len(metadata) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected aligned segment arrays/metadata and both binary classes.")
    if not np.isfinite(X).all():
        raise ValueError("Segment arrays contain nonfinite values.")
    if not np.array_equal(y, metadata.label.map({"responsive": 0, "non-responsive": 1}).to_numpy()):
        raise ValueError("Cached labels disagree with segment metadata.")
    participants = metadata.participant.astype(str).to_numpy()
    groups = participants if args.analysis == "lopo" else (metadata.participant.astype(str) + "::" + metadata.run.astype(str)).to_numpy()
    dataset = ArousalDataset(X, y)
    folds = get_folds(X, y, participants, metadata, args.analysis)
    output = output_settings(args)
    output.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    reference_model = ArousalTCN(X.shape[-1], args.channels, args.kernel_size, args.dilations, args.dropout)
    settings.update(model="TCN", convolutions_per_block=2, activation="ReLU", pooling="mean",
                    padding="symmetric; offline segment classification", optimizer="Adam",
                    device=DEVICE, threshold=0.5, innerCV=0,
                    receptive_field=reference_model.receptive_field,
                    n_parameters=sum(p.numel() for p in reference_model.parameters()),
                    inputs="same run-wise z-scored ROI sequences as GRU",
                    weighting="participant-first, then available classes equally",
                    checkpoint_selection="fixed final epoch; outer test is diagnostic only")
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    metadata.to_csv(output / "segment_metadata.csv", index=False)
    predictions = metadata.copy()
    predictions.insert(0, "segment_index", np.arange(len(y)))
    predictions["label_binary"] = y.astype(int)
    predictions["fold"] = -1
    predictions["prediction"] = -1
    predictions["probability_nonresponsive"] = np.nan
    if args.analysis == "loro":
        predictions["participant_seen"] = 0
    rows, train_sizes = [], []
    for fold, (train, test) in enumerate(folds, start=1):
        print(f"Fold {fold}/{len(folds)}: held out {groups[test[0]]}", flush=True)
        results, history = run_fold(dataset, train, test, participants, args)
        predictions.loc[test, "fold"] = fold
        predictions.loc[test, "prediction"] = results["predictions"].astype(int)
        predictions.loc[test, "probability_nonresponsive"] = results["probabilities"]
        if args.analysis == "loro":
            predictions.loc[test, "participant_seen"] = np.isin(participants[test], participants[train]).astype(int)
        rows.append({"fold": fold, "participant": participants[test[0]],
                     **({"run": metadata.iloc[test[0]]["run"]} if args.analysis == "loro" else {}),
                     "selected_epoch": args.epochs, "train_accuracy": results["train_accuracy"],
                     "n_responsive": int((y[test] == 0).sum()), "n_nonresponsive": int((y[test] == 1).sum()),
                     **classification_metrics(y[test], results["predictions"])})
        train_sizes.append(len(train))
        history.insert(0, "fold", fold)
        history.to_csv(output / f"{args.analysis}_training_history.csv", mode="w" if fold == 1 else "a",
                       header=fold == 1, index=False)
        pd.DataFrame(rows).to_csv(output / f"{args.analysis}_fold_results.csv", index=False)
        print(f"Train accuracy={results['train_accuracy']:.4f}, test accuracy={results['accuracy']:.4f}", flush=True)
    if predictions.fold.lt(0).any():
        raise RuntimeError("Some segments have no outer-fold prediction.")
    pred = predictions.prediction.to_numpy()
    metrics = classification_metrics(y, pred)
    metrics["pooled_test_accuracy"] = metrics.pop("test_accuracy")
    metrics.update(pooled_train_accuracy=float(np.average([r["train_accuracy"] for r in rows], weights=train_sizes)),
                   fold_averaged_accuracy=float(np.mean([r["test_accuracy"] for r in rows])),
                   group_averaged_balanced_accuracy=macro_group_balanced_accuracy(y, pred, groups),
                   n_segments=len(y), n_folds=len(folds), n_rois=X.shape[-1])
    predictions.to_csv(output / f"{args.analysis}_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output / f"{args.analysis}_summary.csv", index=False)
    plot_training_history(args.analysis, output)
    visualize_leave_one_out(args.analysis, output)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Saved results to {output}")
    return output


def summarize():
    rows_by_roi = {}
    for path in sorted(CLASSIFIER_DIR.rglob("settings.json")):
        settings = json.loads(path.read_text())
        summary_path = path.parent / f"{settings['analysis']}_summary.csv"
        if not summary_path.exists():
            continue
        parameters = {key: value for key, value in settings.items()
                      if not isinstance(value, (list, dict)) and key not in {"summarize", "plot_history", "rebuild_segment_arrays"}}
        parameters["dilations"] = "-".join(map(str, settings["dilations"]))
        parameters["meanEpoch"] = settings["epochs"]
        rows = rows_by_roi.setdefault(settings["roi_label"], [])
        rows.extend({**parameters, **metrics, "result_dir": str(path.parent)}
                    for metrics in pd.read_csv(summary_path).to_dict("records"))
    for roi, rows in rows_by_roi.items():
        output = CLASSIFIER_DIR / "results" / f"{roi}_summary.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"Saved {len(rows)} results to {output}")
    if not rows_by_roi:
        print("No completed TCN runs to summarize.")


def main():
    args = parse_args()
    if args.summarize:
        summarize()
    elif args.plot_history:
        output = plot_training_history(args.analysis, output_settings(args))
        print(f"Saved {output}")
    else:
        run_analysis(args)


if __name__ == "__main__":
    main()
