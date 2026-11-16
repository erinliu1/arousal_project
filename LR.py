"""Logistic-regression baseline for the fMRI GRU's exact segments and folds.

Run from the project directory in the GRU training environment:
    python LR.py
    python LR.py --segment-length 50 --segment-stride 50
    python LR.py --analysis loro
    python LR.py --C 0.01

Uses a fixed L2 regularization C (default 0.01) in every outer fold.
Outputs: plots/LR/segments/<segment configuration>/<ROI selection>/<analysis>/C=*.
Missing segment caches are built by the existing GRU data loader, requiring the
original raw_data/<participant>/<run>/z_scored/<ROI>.npy files.
No GPU training is needed, but the shared loader imports the GRU dependencies.
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from GRU import (
    ARGUMENT_SPECS, get_folds, get_segment_dir_name, load_segments, resolve_rois,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    shared = (
        "--rois", "--analysis", "--segment-length", "--segment-stride",
        "--responsive-window-size", "--responsive-window-stride",
        "--responsive-min-bp-trs", "--responsive-max-bp-gap",
        "--trim-responsive-intervals", "--nonresponsive-min-gap",
        "--rebuild-segment-arrays",
    )
    for flag in shared:
        parser.add_argument(flag, **ARGUMENT_SPECS[flag])
    parser.set_defaults(analysis="lopo")
    parser.add_argument("--C", "--c", dest="c", type=float, default=0.01,
                        help="Fixed inverse L2 strength used in every fold (default: 0.01).")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--features", choices=("mean", "mean_sd_slope"),
                        default="mean_sd_slope", help="Features per ROI (default: mean, SD, slope).")
    parser.add_argument("--individual-rois", action="store_true",
                        help="Run each selected ROI separately and write plots/LR/results/roi_summary.csv.")
    args = parser.parse_args(argv)
    if args.max_iter < 1 or args.segment_length < 2 or args.segment_stride < 1:
        parser.error("Require max-iter >= 1, length >= 2, stride >= 1.")
    if not np.isfinite(args.c) or args.c <= 0:
        parser.error("C must be finite and positive.")
    try:
        args.rois, args.roi_label = resolve_rois(args.rois)
    except ValueError as error:
        parser.error(str(error))
    return args


def extract_features(X, rois, feature_set="mean_sd_slope"):
    """Per ROI: temporal mean, population SD, and OLS slope per TR."""
    if X.ndim != 3 or X.shape[1] < 2 or X.shape[2] != len(rois):
        raise ValueError("Expected X shaped (segments, >=2 TRs, selected ROIs).")
    if not np.isfinite(X).all():
        raise ValueError("Segment arrays contain NaN or infinite values.")
    X = np.asarray(X, dtype=np.float64)
    if feature_set == "mean":
        return X.mean(axis=1), [f"{roi}__mean" for roi in rois]
    time = np.arange(X.shape[1], dtype=np.float64)
    time -= time.mean()
    slopes = np.einsum("str,t->sr", X, time) / np.dot(time, time)
    features = np.concatenate([X.mean(axis=1), X.std(axis=1), slopes], axis=1)
    names = [f"{roi}__{stat}" for stat in ("mean", "sd", "slope") for roi in rois]
    return features, names


def participant_class_weights(participants, y):
    """Match expected GRU sampling: equal participants, then available classes.

    Use deterministic weights rather than random resampling. Normalize to a
    mean of one so C retains a comparable scale across training subsets.
    """
    weights = np.empty(len(y), dtype=float)
    for participant in np.unique(participants):
        member = participants == participant
        classes = np.unique(y[member])
        for label in classes:
            mask = member & (y == label)
            weights[mask] = 1.0 / (len(classes) * mask.sum())
    return weights / weights.mean()


def fit_model(X, y, participants, c, args):
    if len(np.unique(y)) != 2:
        raise ValueError("A training fold has only one class; cannot fit binary LR.")
    weights = participant_class_weights(participants, y)
    scaler = StandardScaler().fit(X, sample_weight=weights)
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=args.max_iter)
    # Do not silently report results from a fit that failed to converge.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(X), y, sample_weight=weights)
    return scaler, model


def roc_auc_metrics(predictions):
    """Pooled AUC and equally weighted participant AUCs from held-out probabilities.

    Participant AUC is undefined for participants with only one observed class.
    """
    rows = []
    for participant, frame in predictions.groupby("participant", sort=True):
        y = frame["label_binary"]
        rows.append({
            "participant": participant, "n_segments": len(frame),
            "n_responsive": int((y == 0).sum()),
            "n_nonresponsive": int((y == 1).sum()),
            "roc_auc": (float(roc_auc_score(y, frame["probability_nonresponsive"]))
                        if y.nunique() == 2 else np.nan),
        })
    participants = pd.DataFrame(rows)
    metrics = {
        "pooled_roc_auc": float(roc_auc_score(
            predictions["label_binary"], predictions["probability_nonresponsive"])),
        "participant_mean_roc_auc": participants["roc_auc"].mean(),
        "n_participants": len(participants),
        "n_participants_with_auc": int(participants["roc_auc"].notna().sum()),
    }
    return metrics, participants


def classification_metrics(y, predictions):
    responsive, nonresponsive = y == 0, y == 1
    return {
        "test_accuracy": float((predictions == y).mean()),
        "balanced_test_accuracy": float(balanced_accuracy_score(y, predictions)),
        "responsive_recall": float((predictions[responsive] == 0).mean()) if responsive.any() else np.nan,
        "nonresponsive_recall": float((predictions[nonresponsive] == 1).mean()) if nonresponsive.any() else np.nan,
    }


def run_analysis(args, data=None, participant_rows=None):
    X, y, metadata = load_segments(args) if data is None else data
    y = np.asarray(y, dtype=int)
    if not (len(X) == len(y) == len(metadata)) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected aligned segment arrays/metadata and binary labels 0/1.")
    expected = metadata["label"].map({"responsive": 0, "non-responsive": 1})
    if not np.array_equal(y, expected.to_numpy()):
        raise ValueError("Cached y does not agree with segment metadata labels.")
    features, feature_names = extract_features(X, args.rois, args.features)
    participants = metadata["participant"].astype(str).to_numpy()
    groups = participants if args.analysis == "lopo" else (
        metadata["participant"].astype(str) + "::" + metadata["run"].astype(str)
    ).to_numpy()
    folds = get_folds(X, y, participants, metadata, args.analysis)
    output = (Path("plots/LR/segments") / get_segment_dir_name(args) / args.roi_label
              / args.analysis.upper() / f"C={args.c:g}")
    # Preserve existing multi-feature results when fitting a mean-only model.
    if args.features == "mean":
        output = output / "features=mean"
    output.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings.update(features=["mean"] if args.features == "mean" else ["mean", "sd", "slope_per_TR"],
                    regularization="L2", solver="lbfgs", threshold=0.5,
                    weighting="equal participants, then available classes",
                    regularization_selection="fixed C; no inner cross-validation",
                    note="Deterministic fit; no seed parameter.")
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    prefix = args.analysis
    pd.DataFrame(features, columns=feature_names).to_csv(output / "features.csv", index=False)
    metadata.to_csv(output / "segment_metadata.csv", index=False)
    predictions = metadata.copy()
    predictions.insert(0, "segment_index", np.arange(len(y)))
    predictions["label_binary"] = y
    predictions["fold"] = -1
    predictions["prediction"] = -1
    predictions["probability_nonresponsive"] = np.nan
    fold_rows, coefficient_rows, training_sizes = [], [], []
    for fold, (train, test) in enumerate(folds, start=1):
        print(f"Fold {fold}/{len(folds)}: held out {groups[test[0]]}", flush=True)
        scaler, model = fit_model(features[train], y[train], participants[train], args.c, args)
        probability = model.predict_proba(scaler.transform(features[test]))[:, 1]
        pred = (probability >= 0.5).astype(int)
        predictions.loc[test, "fold"] = fold
        predictions.loc[test, "prediction"] = pred
        predictions.loc[test, "probability_nonresponsive"] = probability
        train_accuracy = float((model.predict(scaler.transform(features[train])) == y[train]).mean())
        fold_rows.append({
            "fold": fold, "participant": participants[test[0]],
            **({"run": metadata.iloc[test[0]]["run"]} if args.analysis == "loro" else {}),
            "C": args.c, "train_accuracy": train_accuracy,
            "n_responsive": int((y[test] == 0).sum()),
            "n_nonresponsive": int((y[test] == 1).sum()),
            **classification_metrics(y[test], pred),
        })
        training_sizes.append(len(train))
        coefficient_rows.extend({
            "fold": fold, "feature": name, "coefficient_standardized": float(coef),
            "scaler_mean": float(mean), "scaler_scale": float(scale),
            "intercept": float(model.intercept_[0]),
        } for name, coef, mean, scale in zip(feature_names, model.coef_[0], scaler.mean_, scaler.scale_))
        pd.DataFrame(fold_rows).to_csv(output / f"{prefix}_fold_results.csv", index=False)

    fold_df = pd.DataFrame(fold_rows)
    if (predictions["fold"] < 0).any():
        raise RuntimeError("Some segments were not assigned an outer-fold prediction.")
    if args.analysis == "loro":
        predictions["participant_seen"] = 0
        for train, test in folds:
            predictions.loc[test, "participant_seen"] = np.isin(participants[test], participants[train]).astype(int)
    pred = predictions["prediction"].to_numpy()
    metrics = classification_metrics(y, pred)
    metrics["pooled_test_accuracy"] = metrics.pop("test_accuracy")
    metrics.update(
        pooled_train_accuracy=float(np.average(fold_df["train_accuracy"], weights=training_sizes)),
        fold_averaged_accuracy=float(fold_df["test_accuracy"].mean()),
        n_segments=len(y), n_folds=len(folds), n_features=features.shape[1],
    )
    auc_metrics, participant_auc = roc_auc_metrics(predictions)
    metrics.update(auc_metrics)
    if participant_rows is not None:
        participant_rows.append(participant_auc.assign(roi=args.roi_label))
    participant_auc.to_csv(output / f"{prefix}_participant_roc_auc.csv", index=False)
    predictions.to_csv(output / f"{prefix}_predictions.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output / f"{prefix}_coefficients.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output / f"{prefix}_summary.csv", index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Saved results to {output}")
    return metrics


def main(argv=None):
    args = parse_args(argv)
    if not args.individual_rois:
        return run_analysis(args)
    # Load the shared cache once; fit each ROI sequentially with the same folds.
    X, y, metadata = load_segments(args)
    rows, participant_rows = [], []
    output = Path("plots/LR/results/roi_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, roi in enumerate(args.rois):
        print(f"ROI {index + 1}/{len(args.rois)}: {roi}", flush=True)
        roi_args = argparse.Namespace(**{**vars(args), "rois": [roi], "roi_label": roi})
        metrics = run_analysis(roi_args, (X[:, :, index:index + 1], y, metadata), participant_rows)
        rows.append({"roi": roi, "C": args.c, "features": args.features,
                     "analysis": args.analysis, "segment_configuration": get_segment_dir_name(args),
                     **metrics})
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.concat(participant_rows, ignore_index=True).to_csv(
        output.parent / "roi_participant_roc_auc.csv", index=False)
    print(f"Saved combined ROI summary to {output}")
    return output


if __name__ == "__main__":
    main()
