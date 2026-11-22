"""Nonlinear classifier on the exact ROI summary features used by LR.py.

    python MLP.py --analysis lopo --segment-length 40 --segment-stride 40 --seed 42
    python MLP.py --analysis loro --segment-length 20 --segment-stride 20 --seed 123
    python MLP.py --run_inner --outer-held-out-participant AAN_01 --seed 42
    python MLP.py --get_best_starting_config
    python MLP.py --eval --seed 42
    python MLP.py --summarize
    python MLP.py --summarize-legacy
    python MLP.py --analysis lopo --plot-history

Default: all ROIs, trim disabled, LOPO, 40-TR length/stride, one 16-unit ReLU
hidden layer, dropout=0.2, Adam LR=1e-3, weight decay=1e-2, 10 epochs.
CPU training; use the same Python environment and data as the GRU and LR.
Each fold uses its final epoch. Test curves are diagnostic only.
Nested modes use canonical BP=nearest all-ROI segments, HD=16/32/64 x
WD=1e-2/3e-3/1e-3, ten canonical seeds, and 10 epochs. Inner predictions
are pooled over 19 participants; configurations rank by ten-seed mean BA.
Nested output: plots/MLP/nested_LOPO,BP=nearest/{inner_LOPO,outer_LOPO,evaluation}.
--summarize writes nested seed summaries and participant-bootstrap CIs;
--summarize-legacy retains the original fixed-configuration summary.
Outputs: plots/MLP/segments/<segment settings>/<ROIs>/<model settings>/<analysis>/SEED=*.
Training automatically saves a four-panel loss/recall history PNG. Use
--plot-history with matching run settings to plot an existing CSV without training.
"""

import argparse
import json
import fcntl
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from LR import extract_features, participant_class_weights
from GRU import (
    ARGUMENT_SPECS, ParticipantClassBalancedSampler,
    get_folds, get_segment_dir_name, load_segments, resolve_rois, set_seed,
)
from nested_LOPO_GRU import SEEDS, scientific, training_args as gru_training_args, fold_metrics
from nested_LOPO_GRU_new import load_data as load_nearest_data, PARTICIPANTS
from scatterplots import visualize_leave_one_out, plot_training_history

CLASSIFIER_DIR = Path("plots/MLP")
SEGMENT_DIR = CLASSIFIER_DIR / "segments"
ROOT = CLASSIFIER_DIR / "nested_LOPO,BP=nearest"
HIDDEN_SIZES = (16, 32, 64)
WEIGHT_DECAYS = (1e-2, 3e-3, 1e-3)
COLUMNS = ["outer_held_out_participant", "roi", "hidden", "WD", "seed",
           "n_validation_segments", "segment_pooled_balanced_accuracy"]


def nested_training_args(args, hidden, wd, seed):
    config = gru_training_args(args, hidden, wd, seed)
    config.epochs = 10  # Preserve the established MLP training duration.
    return config



def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for flag in (
        "--rois", "--analysis", "--seed", "--segment-length", "--segment-stride",
        "--responsive-window-size", "--responsive-window-stride",
        "--responsive-min-bp-trs", "--responsive-max-bp-gap",
        "--trim-responsive-intervals", "--nonresponsive-min-gap",
        "--bp_TR_snap", "--rebuild-segment-arrays", "--epochs", "--batch-size",
        "--learning-rate", "--weight-decay", "--dropout", "--hidden-size", "--summarize",
    ):
        options = ARGUMENT_SPECS[flag].copy()
        if flag == "--hidden-size":
            options["help"] = "Number of units in the single ReLU hidden layer."
        if flag == "--summarize":
            options["help"] = "Summarize ten nested-evaluation seeds, including bootstrap CIs."
        if flag == "--epochs":
            options["help"] = "Epochs for original fixed-configuration runs; nested modes use 10."
        parser.add_argument(flag, **options)
    parser.set_defaults(analysis="lopo", hidden_size=16, weight_decay=1e-2, epochs=10, seed=None)
    parser.add_argument("--plot", action="store_true", help="Also save a fold-recall plot.")
    parser.add_argument("--plot-history", action="store_true",
                        help="Plot the existing training-history CSV for these settings without training.")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for PyTorch.")
    parser.add_argument("--run_inner", action="store_true")
    parser.add_argument("--outer-held-out-participant", choices=PARTICIPANTS)
    parser.add_argument("--get_best_starting_config", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--summarize-legacy", action="store_true", help="Summarize original fixed-configuration runs")
    args = parser.parse_args()
    if args.run_inner and (args.eval or args.get_best_starting_config or args.summarize):
        parser.error("--run_inner cannot be combined with other nested modes")
    if args.summarize and (args.eval or args.get_best_starting_config):
        parser.error("--summarize is standalone")
    if args.run_inner and args.outer_held_out_participant is None:
        parser.error("--run_inner requires --outer-held-out-participant")
    if not args.run_inner and args.outer_held_out_participant is not None:
        parser.error("--outer-held-out-participant requires --run_inner")
    if (args.run_inner or args.eval) and args.seed is None:
        parser.error("Nested training/eval requires --seed")
    if (args.run_inner or args.eval) and args.seed not in SEEDS:
        parser.error("Nested modes require a canonical seed")

    if args.seed is None:
        args.seed = 42
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
    def __init__(self, n_features, hidden_size=16, dropout=0.2):
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


def macro_group_balanced_accuracy(y, predictions, groups):
    return float(np.mean([
        classification_metrics(y[groups == group], predictions[groups == group])["balanced_test_accuracy"]
        for group in np.unique(groups)
    ]))


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
    return (SEGMENT_DIR / get_segment_dir_name(args) / args.roi_label
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
    plot_training_history(args.analysis, output)
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
                      if not isinstance(value, (list, dict)) and key not in {"summarize", "plot", "plot_history", "rebuild_segment_arrays"}}
        for metrics in pd.read_csv(summary_path).to_dict("records"):
            rows.append({**parameters, **metrics, "result_dir": str(path.parent)})
    if not rows:
        print("No completed MLP runs to summarize.")
        return
    output = CLASSIFIER_DIR / "results" / "all_rois_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {len(rows)} results to {output}")


def run_outer_participant_seed(outer_held_out_participant, seed):
    """Evaluate 9 configurations x 19 inner folds for one outer participant/seed."""
    outer = str(outer_held_out_participant)
    if outer not in PARTICIPANTS or seed not in SEEDS:
        raise ValueError("Use a canonical outer participant and seed")
    X, y, metadata = load_nearest_data()
    y = np.asarray(y, dtype=int)
    metadata = metadata.reset_index(drop=True)
    participants = metadata.participant.astype(str).to_numpy()
    if (X.ndim != 3 or X.shape[1:] != (40, 42) or len(X) != len(y)
            or len(y) != len(metadata) or set(participants) != set(PARTICIPANTS)):
        raise ValueError("Expected aligned nearest-snapped canonical 40-TR, all-42-ROI data")
    from all_rois import all_rois
    features, _ = extract_features(X, all_rois)
    inner_participants = [participant for participant in PARTICIPANTS if participant != outer]
    outputs = []
    for hidden in HIDDEN_SIZES:
        for wd in WEIGHT_DECAYS:
            output = (ROOT / "inner_LOPO" / outer / "all_rois"
                      / f"hidden={hidden},WD={scientific(wd)}" / f"seed={seed}.csv")
            if output.is_file() and output.stat().st_size:
                saved = pd.read_csv(output)
                if (list(saved.columns) != COLUMNS or len(saved) != 1
                        or saved.iloc[0].outer_held_out_participant != outer
                        or saved.iloc[0].roi != "all_rois" or saved.iloc[0].hidden != hidden
                        or saved.iloc[0].WD != wd or saved.iloc[0].seed != seed
                        or saved.iloc[0].n_validation_segments != int((participants != outer).sum())
                        or not 0 <= saved.iloc[0].segment_pooled_balanced_accuracy <= 1):
                    raise ValueError(f"Invalid saved inner result: {output}")
                print(f"Skipping completed configuration: {output}", flush=True)
                outputs.append(output)
                continue
            from all_rois import all_rois
            args = nested_training_args(argparse.Namespace(rois=list(all_rois), roi_label="all_rois",
                                                       bp_TR_snap="nearest"), hidden, wd, seed)
            labels, predictions, seen = [], [], []
            for fold, validation_participant in enumerate(inner_participants, start=1):
                train = np.flatnonzero((participants != outer) & (participants != validation_participant))
                validation = np.flatnonzero(participants == validation_participant)
                print(f"Outer={outer} | seed={seed} | HD={hidden} | WD={scientific(wd)} | "
                      f"inner {fold}/19: {validation_participant}", flush=True)
                result = run_fold(features, y, participants, train, validation, args)
                pred = np.asarray(result["predictions"])
                if pred.shape != y[validation].shape or not np.isin(pred, [0, 1]).all():
                    raise ValueError("Invalid inner validation predictions")
                labels.append(y[validation]); predictions.append(pred); seen.extend(validation.tolist())
            if sorted(seen) != np.flatnonzero(participants != outer).tolist():
                raise RuntimeError("Expected exactly one validation prediction per non-outer segment")
            labels, predictions = np.concatenate(labels), np.concatenate(predictions)
            if set(np.unique(labels)) != {0, 1}:
                raise ValueError("Pooled validation requires both classes")
            row = dict(outer_held_out_participant=outer, roi="all_rois", hidden=hidden, WD=wd,
                       seed=seed, n_validation_segments=len(labels),
                       segment_pooled_balanced_accuracy=balanced_accuracy_score(labels, predictions))
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".csv.tmp")
            pd.DataFrame([row], columns=COLUMNS).to_csv(temporary, index=False)
            temporary.replace(output)
            outputs.append(output)
            print(f"Saved {output}", flush=True)
    return outputs


def get_best_starting_config():
    """Rank nine configurations using ten-seed inner means, separately per outer fold."""
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / ".best_configurations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rankings = {}
        for outer in PARTICIPANTS:
            rows, all_counts = [], set()
            for hidden in HIDDEN_SIZES:
                for wd in WEIGHT_DECAYS:
                    scores, counts = [], []
                    for seed in SEEDS:
                        path = (ROOT / "inner_LOPO" / outer / "all_rois"
                                / f"hidden={hidden},WD={scientific(wd)}" / f"seed={seed}.csv")
                        frame = pd.read_csv(path)
                        if list(frame.columns) != COLUMNS or len(frame) != 1:
                            raise ValueError(f"Invalid inner result: {path}")
                        row = frame.iloc[0]
                        score, count = float(row.segment_pooled_balanced_accuracy), float(row.n_validation_segments)
                        if (row.outer_held_out_participant != outer or row.roi != "all_rois"
                                or row.hidden != hidden or row.WD != wd or row.seed != seed
                                or not np.isfinite(score) or not 0 <= score <= 1
                                or not np.isfinite(count) or count <= 0 or not count.is_integer()):
                            raise ValueError(f"Invalid inner result: {path}")
                        scores.append(score)
                        counts.append(int(count))
                    all_counts.update(counts)
                    rows.append(dict(outer_held_out_participant=outer, roi="all_rois", hidden=hidden,
                                     WD=wd, n_seeds=len(SEEDS), n_validation_segments=counts[0],
                                     mean_segment_pooled_balanced_accuracy=float(np.mean(scores))))
            if len(all_counts) != 1:
                raise ValueError(f"Inconsistent validation counts for {outer}")
            ranking = pd.DataFrame(rows).sort_values(
                ["mean_segment_pooled_balanced_accuracy", "hidden", "WD"],
                ascending=[False, True, True]).reset_index(drop=True)
            ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
            rankings[outer] = ranking
        for outer, ranking in rankings.items():
            path = ROOT / "inner_LOPO" / outer / "best_configurations.csv"
            temporary = path.with_suffix(".csv.tmp")
            ranking.to_csv(temporary, index=False)
            temporary.replace(path)
            best = ranking.iloc[0]
            print(f"{outer}: HD={best.hidden}, WD={scientific(best.WD)}, mean BA={best.mean_segment_pooled_balanced_accuracy:.6f}", flush=True)
        combined = pd.concat([frame.iloc[[0]] for frame in rankings.values()], ignore_index=True)
        path = ROOT / "best_configurations.csv"
        temporary = path.with_suffix(".csv.tmp")
        combined.to_csv(temporary, index=False)
        temporary.replace(path)
        return path


def run_evaluation(seed):
    """Evaluate all 20 outer folds with inner-selected HD/WD on nearest-snapped summary features."""
    if seed not in SEEDS:
        raise ValueError("Use a canonical evaluation seed")
    roi_label = "all_rois"
    path = ROOT / "best_configurations.csv"
    selected_frame = pd.read_csv(path)
    if (len(selected_frame) != len(PARTICIPANTS)
            or selected_frame.outer_held_out_participant.duplicated().any()
            or set(selected_frame.outer_held_out_participant) != set(PARTICIPANTS)
            or not selected_frame.roi.eq("all_rois").all()
            or not selected_frame.hidden.isin(HIDDEN_SIZES).all()
            or not selected_frame.WD.isin(WEIGHT_DECAYS).all()
            or not selected_frame.n_seeds.eq(len(SEEDS)).all()
            or not selected_frame.mean_segment_pooled_balanced_accuracy.between(0, 1).all()):
        raise ValueError(f"Invalid best configurations: {path}")
    choices = [row for _, row in selected_frame.iterrows()]
    selected = pd.DataFrame(choices).set_index("outer_held_out_participant")
    X, y, metadata = load_nearest_data()
    metadata = metadata.reset_index(drop=True)
    y = np.asarray(y, dtype=int)
    from all_rois import all_rois
    features, _ = extract_features(X, all_rois)
    participants = metadata.participant.astype(str).to_numpy()
    if set(participants) != set(PARTICIPANTS):
        raise ValueError("Evaluation data must contain all canonical outer participants")
    rows, fold_rows = [], []
    for fold, outer in enumerate(PARTICIPANTS, start=1):
        choice = selected.loc[outer]
        from all_rois import all_rois
        args = nested_training_args(argparse.Namespace(rois=list(all_rois), roi_label="all_rois",
                                                   bp_TR_snap="nearest"), int(choice.hidden), float(choice.WD), seed)
        train = np.flatnonzero(participants != outer)
        test = np.flatnonzero(participants == outer)
        print(f"Outer seed={seed} | fold {fold}/{len(PARTICIPANTS)} | held out={outer} | "
              f"ROI={args.roi_label} | WD={scientific(args.weight_decay)}", flush=True)
        result = run_fold(features, y, participants, train, test, args)
        predicted = np.asarray(result["predictions"])
        probabilities = np.asarray(result["probabilities"])
        if (predicted.shape != y[test].shape or probabilities.shape != y[test].shape
                or not np.isin(predicted, [0, 1]).all()
                or not np.isfinite(probabilities).all()
                or not ((probabilities >= 0) & (probabilities <= 1)).all()):
            raise ValueError(f"Invalid held-out predictions for {outer}")
        fold_rows.append({
            "fold": fold, "outer_held_out_participant": outer,
            "roi": args.roi_label, "hidden": int(choice.hidden), "WD": args.weight_decay,
            "best_mean_segment_pooled_balanced_accuracy": float(choice.mean_segment_pooled_balanced_accuracy),
            "seed": seed, "n_train_segments": len(train), "n_test_segments": len(test),
            "train_accuracy": float(result["train_accuracy"]),
            **fold_metrics(y[test], predicted),
        })
        frame = metadata.iloc[test].copy()
        frame["segment_index"] = test
        frame["fold"] = fold
        frame["outer_held_out_participant"] = outer
        frame["roi"] = args.roi_label
        frame["hidden"] = int(choice.hidden)
        frame["WD"] = args.weight_decay
        frame["seed"] = seed
        frame["best_mean_segment_pooled_balanced_accuracy"] = float(choice.mean_segment_pooled_balanced_accuracy)
        frame["label_binary"] = y[test]
        frame["prediction"] = predicted.astype(int)
        frame["probability_nonresponsive"] = probabilities
        rows.append(frame)
    predictions = pd.concat(rows, ignore_index=True).sort_values("segment_index").reset_index(drop=True)
    if sorted(predictions.segment_index) != list(range(len(y))):
        raise RuntimeError("Outer LOPO must predict every segment exactly once")
    output_dir = ROOT / "outer_LOPO" / roi_label / f"seed={seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(fold_rows)
    labels, predicted = predictions.label_binary, predictions.prediction
    participant_accuracy = predictions.assign(correct=predicted.eq(labels)).groupby("participant").correct.mean()
    summary = pd.DataFrame([{
        "seed": seed,
        "pooled_train_accuracy": float(np.average(folds.train_accuracy, weights=folds.n_train_segments)),
        "pooled_test_accuracy": float(predicted.eq(labels).mean()),
        "participant_averaged_accuracy": float(participant_accuracy.mean()),
        "responsive_recall": float(predicted[labels.eq(0)].eq(0).mean()),
        "nonresponsive_recall": float(predicted[labels.eq(1)].eq(1).mean()),
        "segment_weighted_balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }])
    for name, frame in (("predictions.csv", predictions), ("fold_results.csv", folds), ("summary.csv", summary)):
        output = output_dir / name
        temporary = output.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(output)
    print(f"Saved outer LOPO seed results to {output_dir}", flush=True)
    print(summary.to_string(index=False), flush=True)
    return output_dir




def summarize_outer():
    """Write per-seed, mean/SD, and participant-bootstrap summaries per ROI."""
    import subprocess
    import sys

    selections = ("all_rois",)
    outputs = []
    for roi in selections:
        rows, predictions = [], []
        reference = None
        keys = ["participant", "run", "segment_start_TR", "segment_end_TR_exclusive"]
        for seed in SEEDS:
            folder = ROOT / "outer_LOPO" / roi / f"seed={seed}"
            summary = pd.read_csv(folder / "summary.csv")
            if len(summary) != 1 or summary.iloc[0]["seed"] != seed:
                raise ValueError(f"Expected one summary row for seed {seed}: {folder}")
            if not np.isfinite(summary.select_dtypes(include="number").to_numpy()).all():
                raise ValueError(f"Nonfinite summary metrics: {folder}")
            frame = pd.read_csv(folder / "predictions.csv")
            if (not frame.seed.eq(seed).all() or not frame.roi.eq(roi).all()
                    or frame.duplicated(keys).any()
                    or not frame.participant.eq(frame.outer_held_out_participant).all()
                    or not frame.label_binary.isin([0, 1]).all()
                    or not frame.prediction.isin([0, 1]).all()):
                raise ValueError(f"Invalid held-out predictions: {folder}")
            identity = frame.set_index(keys).label_binary.sort_index()
            if reference is None:
                reference = identity
            elif not identity.equals(reference):
                raise ValueError(f"Segment identities/labels differ across seeds: {folder}")
            rows.append(summary.iloc[0].to_dict())
            predictions.append(frame)
        seed_summary = pd.DataFrame(rows)
        seed_summary["seed"] = seed_summary.seed.astype(int)
        metrics = [column for column in seed_summary if column != "seed"]
        mean_std = seed_summary[metrics].agg(["mean", "std"]).rename_axis("statistic").reset_index()
        participants = sorted(predictions[0].participant.astype(str).unique())
        if set(participants) != set(PARTICIPANTS):
            raise ValueError(f"Expected all canonical participants for {roi}")
        evaluation = ROOT / "evaluation" / roi
        evaluation.mkdir(parents=True, exist_ok=True)
        for name, frame in (("seed_summary.csv", seed_summary), ("seed_mean_std.csv", mean_std)):
            output = evaluation / name
            temporary = output.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(output)
            outputs.append(output)
            print(f"Saved {output}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve().with_name("bootstrap.py")),
                        "--predictions-folder", str(ROOT / "outer_LOPO" / roi)], check=True)
        outputs.append(evaluation / "bootstrap_ci.csv")
    return outputs




def main():
    args = parse_args()
    if args.run_inner or args.eval:
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
    if args.summarize:
        summarize_outer()
    elif args.summarize_legacy:
        summarize()
    elif args.run_inner:
        run_outer_participant_seed(args.outer_held_out_participant, args.seed)
    elif args.get_best_starting_config or args.eval:
        if args.get_best_starting_config:
            get_best_starting_config()
        if args.eval:
            run_evaluation(args.seed)
    elif args.plot_history:
        output = plot_training_history(args.analysis, output_settings(args))
        print(f"Saved {output}")
    else:
        run_analysis(args)


if __name__ == "__main__":
    main()
