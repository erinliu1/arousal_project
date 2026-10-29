"""Motion-only LR baseline using the fMRI classifier's segments and folds.

Examples (run in the same environment as fmri_classifier_exploration2.py):
    python motion_classifier.py --run-training
    python motion_classifier.py --run-training --analysis loro --segment-length 50 --segment-stride 50
    python motion_classifier.py --summarize

Default: trim disabled, 40-TR segments/stride, LOPO, fixed C=0.01.
Only segment metadata and fd.npy are loaded; ROI arrays are not needed.
Results: plots/motion_classifier/<segment settings>/FD=<threshold>/
         model=LR,C=<C>/<LOPO or LORO>/
Fits are deterministic, so there is no seed parameter or seed folder.
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fmri_classifier_exploration2 import (
    ARGUMENT_SPECS, get_folds, get_segment_dir_name, load_segment_metadata,
)
from LR import macro_group_balanced_accuracy, participant_class_weights
from scatterplots import visualize_leave_one_out

RAW_DATA_DIR = Path("raw_data")
CLASSIFIER_DIR = Path("plots/motion_classifier")
FD_THRESHOLD = 0.25
FEATURE_NAMES = (
    "mean_fd", "max_fd", "median_fd", "sd_fd", "fraction_fd_above_0.25",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for flag in (
        "--run-training", "--analysis", "--segment-length", "--segment-stride",
        "--responsive-window-size", "--responsive-window-stride",
        "--responsive-min-bp-trs", "--responsive-max-bp-gap",
        "--trim-responsive-intervals", "--nonresponsive-min-gap",
        "--rebuild-segment-arrays", "--summarize",
    ):
        options = ARGUMENT_SPECS[flag].copy()
        if flag == "--rebuild-segment-arrays":
            options["help"] = "Rebuild shared segment metadata before extracting motion features."
        parser.add_argument(flag, **options)
    parser.set_defaults(analysis="lopo")
    parser.add_argument("--C", "--c", dest="c", type=float, default=0.01,
                        help="Fixed inverse L2 strength for all folds (default: 0.01).")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--fd-threshold", type=float, default=FD_THRESHOLD,
                        help="Threshold for the fraction-above-threshold feature; not motion exclusion.")
    parser.add_argument("--plot", action="store_true",
                        help="Save the fold-recall plot after training, or plot existing results.")
    args = parser.parse_args()
    if args.segment_length < 1 or args.segment_stride < 1 or args.max_iter < 1:
        parser.error("Segment length, stride, and max-iter must be positive.")
    if not np.isfinite(args.c) or args.c <= 0:
        parser.error("C must be finite and positive.")
    if not np.isfinite(args.fd_threshold) or args.fd_threshold < 0:
        parser.error("FD threshold must be finite and nonnegative.")
    return args


def feature_names(threshold):
    return [*FEATURE_NAMES[:4], f"fraction_fd_above_{threshold:g}"]


def calculate_fd_features(segment_fd, threshold=FD_THRESHOLD):
    """Return five FD summaries for a segment of any positive length."""
    values = np.asarray(segment_fd, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("FD segments must be nonempty, finite one-dimensional arrays.")
    return np.array([
        values.mean(), values.max(), np.median(values), values.std(ddof=0),
        np.mean(values > threshold),
    ], dtype=np.float64)


def load_segments(args):
    """Use exactly the GRU metadata, with FD assigned to the arriving TR."""
    metadata = load_segment_metadata(args).reset_index(drop=True)
    lengths = metadata["segment_end_TR_exclusive"] - metadata["segment_start_TR"]
    if metadata.empty or not lengths.eq(args.segment_length).all():
        raise ValueError("Segment metadata is empty or does not match --segment-length.")
    labels = metadata["label"].map({"responsive": 0, "non-responsive": 1})
    if labels.isna().any() or labels.nunique() != 2:
        raise ValueError("Expected both responsive and non-responsive labels.")
    X = np.empty((len(metadata), len(FEATURE_NAMES)), dtype=np.float64)
    for (participant, run), segments in metadata.groupby(["participant", "run"], sort=False):
        fd = np.asarray(np.load(RAW_DATA_DIR / str(participant) / str(run) / "fd.npy"),
                        dtype=np.float64)
        if fd.ndim != 1 or not np.isfinite(fd).all():
            raise ValueError(f"Invalid FD array for {participant} {run}.")
        # fd[i] is motion from TR i to i+1, assigned to TR i+1; TR 0 is zero.
        fd_by_tr = np.concatenate(([0.0], fd))
        for index, segment in segments.iterrows():
            start = int(segment["segment_start_TR"])
            end = int(segment["segment_end_TR_exclusive"])
            if start < 0 or end > len(fd_by_tr) or end - start != args.segment_length:
                raise ValueError(f"FD bounds mismatch for {participant} {run}: [{start}, {end}).")
            X[index] = calculate_fd_features(fd_by_tr[start:end], args.fd_threshold)
    return X, labels.to_numpy(dtype=np.int64), metadata


def run_fold(X, y, participants, train_indices, test_indices, args):
    """Fit train-only scaling and participant/class-weighted fixed-C LR."""
    X_train, y_train = X[train_indices], y[train_indices]
    if len(np.unique(y_train)) != 2:
        raise ValueError("A training fold contains only one class.")
    weights = participant_class_weights(participants[train_indices], y_train)
    scaler = StandardScaler().fit(X_train, sample_weight=weights)
    model = LogisticRegression(C=args.c, solver="lbfgs", max_iter=args.max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(X_train), y_train, sample_weight=weights)
    probabilities = model.predict_proba(scaler.transform(X[test_indices]))[:, 1]
    return {
        "predictions": (probabilities >= 0.5).astype(int),
        "probabilities": probabilities,
        "train_accuracy": float((model.predict(scaler.transform(X_train)) == y_train).mean()),
        "scaler": scaler, "model": model,
    }


def output_settings(args):
    return (CLASSIFIER_DIR / get_segment_dir_name(args) / f"FD={args.fd_threshold:g}"
            / f"model=LR,C={args.c:g}" / args.analysis.upper()), args.analysis


def classification_metrics(y, predictions):
    """Average available class recalls, including folds with only one class."""
    recalls = {}
    for label, name in ((0, "responsive_recall"), (1, "nonresponsive_recall")):
        mask = y == label
        recalls[name] = float((predictions[mask] == label).mean()) if mask.any() else np.nan
    return {
        "test_accuracy": float((predictions == y).mean()),
        "balanced_test_accuracy": float(np.nanmean(list(recalls.values()))),
        **recalls,
    }


def run_analysis(args):
    X, y, metadata = load_segments(args)
    participants = metadata["participant"].astype(str).to_numpy()
    groups = participants if args.analysis == "lopo" else (
        metadata["participant"].astype(str) + "::" + metadata["run"].astype(str)
    ).to_numpy()
    folds = get_folds(X, y, participants, metadata, args.analysis)
    save_dir, prefix = output_settings(args)
    save_dir.mkdir(parents=True, exist_ok=True)
    names = feature_names(args.fd_threshold)
    settings = vars(args).copy()
    settings.update(model="LR", features=names, regularization="L2", solver="lbfgs",
                    threshold=0.5, weighting="equal participants, then available classes",
                    regularization_selection="fixed C; no inner cross-validation",
                    fd_alignment="fd[i] assigned to TR i+1; TR 0 assigned zero")
    (save_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    pd.DataFrame(X, columns=names).to_csv(save_dir / "features.csv", index=False)
    metadata.to_csv(save_dir / "segment_metadata.csv", index=False)
    predictions = metadata.copy()
    predictions.insert(0, "segment_index", np.arange(len(y)))
    predictions["label_binary"] = y
    predictions["fold"] = -1
    predictions["prediction"] = -1
    predictions["probability_nonresponsive"] = np.nan
    if args.analysis == "loro":
        predictions["participant_seen"] = 0
    fold_rows, coefficient_rows, train_sizes = [], [], []
    for fold, (train, test) in enumerate(folds, start=1):
        print(f"Fold {fold}/{len(folds)}: held out {groups[test[0]]}", flush=True)
        result = run_fold(X, y, participants, train, test, args)
        predictions.loc[test, "fold"] = fold
        predictions.loc[test, "prediction"] = result["predictions"]
        predictions.loc[test, "probability_nonresponsive"] = result["probabilities"]
        if args.analysis == "loro":
            predictions.loc[test, "participant_seen"] = np.isin(participants[test], participants[train]).astype(int)
        fold_rows.append({
            "fold": fold, "participant": participants[test[0]],
            **({"run": metadata.iloc[test[0]]["run"]} if args.analysis == "loro" else {}),
            "C": args.c, "train_accuracy": result["train_accuracy"],
            "n_responsive": int((y[test] == 0).sum()),
            "n_nonresponsive": int((y[test] == 1).sum()),
            **classification_metrics(y[test], result["predictions"]),
        })
        train_sizes.append(len(train))
        model, scaler = result["model"], result["scaler"]
        coefficient_rows.extend({
            "fold": fold, "feature": name, "coefficient_standardized": float(coef),
            "scaler_mean": float(mean), "scaler_scale": float(scale),
            "intercept": float(model.intercept_[0]),
        } for name, coef, mean, scale in zip(names, model.coef_[0], scaler.mean_, scaler.scale_))
        pd.DataFrame(fold_rows).to_csv(save_dir / f"{prefix}_fold_results.csv", index=False)
    if (predictions["fold"] < 0).any():
        raise RuntimeError("Some segments have no held-out prediction.")
    folds_df = pd.DataFrame(fold_rows)
    pred = predictions["prediction"].to_numpy()
    summary = classification_metrics(y, pred)
    summary["pooled_test_accuracy"] = summary.pop("test_accuracy")
    summary.update(
        pooled_train_accuracy=float(np.average(folds_df["train_accuracy"], weights=train_sizes)),
        fold_averaged_accuracy=float(folds_df["test_accuracy"].mean()),
        group_averaged_balanced_accuracy=macro_group_balanced_accuracy(y, pred, groups),
        n_segments=len(y), n_folds=len(folds), n_features=len(names),
    )
    predictions.to_csv(save_dir / f"{prefix}_predictions.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(save_dir / f"{prefix}_coefficients.csv", index=False)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(save_dir / f"{prefix}_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved results to {save_dir}")
    if args.plot:
        visualize_leave_one_out(args.analysis, save_dir)
    return save_dir


def summarize():
    """Collect completed parameterized runs; leave legacy results untouched."""
    rows = []
    for settings_path in sorted(CLASSIFIER_DIR.glob("*/FD=*/model=LR,C=*/*/settings.json")):
        settings = json.loads(settings_path.read_text())
        summary_path = settings_path.parent / f"{settings['analysis']}_summary.csv"
        if not summary_path.exists():
            continue
        parameters = {key: value for key, value in settings.items()
                      if key not in {"run_training", "summarize", "plot", "rebuild_segment_arrays"}
                      and not isinstance(value, (list, dict))}
        for metrics in pd.read_csv(summary_path).to_dict("records"):
            rows.append({**parameters, **metrics, "result_dir": str(settings_path.parent)})
    if not rows:
        print("No completed parameterized motion runs to summarize.")
        return
    output = CLASSIFIER_DIR / "results" / "motion_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {len(rows)} results to {output}")


def main():
    args = parse_args()
    if args.summarize:
        summarize()
    elif args.run_training:
        run_analysis(args)
    elif args.rebuild_segment_arrays:
        X, _, _ = load_segments(args)
        print(f"Rebuilt segment metadata; extracted {len(X)} motion feature rows.")
    elif args.plot:
        save_dir, _ = output_settings(args)
        visualize_leave_one_out(args.analysis, save_dir)
    else:
        save_dir, _ = output_settings(args)
        print(f"Analysis: {args.analysis.upper()} | C={args.c:g} | trim={args.trim_responsive_intervals}")
        print(f"Segments: {get_segment_dir_name(args)}")
        print(f"Output: {save_dir}\nUse --run-training to fit and save results.")


if __name__ == "__main__":
    main()
