"""Logistic-regression baseline for the fMRI GRU's exact segments and folds.

Run from the project directory in the GRU training environment:
    python LR.py
    python LR.py --segment-length 50 --segment-stride 50
    python LR.py --analysis loro --trim-responsive-intervals

Outputs go to plots/LR/<segment configuration>/<ROI selection>/<analysis>/SEED=*.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from fmri_classifier_exploration2 import (
    ARGUMENT_SPECS, get_folds, get_segment_dir_name, load_segments, resolve_rois,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    shared = (
        "--rois", "--analysis", "--seed", "--segment-length", "--segment-stride",
        "--responsive-window-size", "--responsive-window-stride",
        "--responsive-min-bp-trs", "--responsive-max-bp-gap",
        "--trim-responsive-intervals", "--nonresponsive-min-gap",
        "--rebuild-segment-arrays",
    )
    for flag in shared:
        parser.add_argument(flag, **ARGUMENT_SPECS[flag])
    parser.set_defaults(analysis="lopo")
    parser.add_argument("--c-grid", type=float, nargs="+",
                        default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                        help="Inverse L2 strengths, selected by grouped inner CV.")
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=5000)
    args = parser.parse_args()
    if (args.inner_splits < 2 or args.max_iter < 1 or args.segment_length < 2
            or args.segment_stride < 1):
        parser.error("Require inner-splits >= 2, max-iter >= 1, length >= 2, stride >= 1.")
    if not all(np.isfinite(c) and c > 0 for c in args.c_grid):
        parser.error("All C values must be finite and positive.")
    args.c_grid = sorted(set(args.c_grid))
    try:
        args.rois, args.roi_label = resolve_rois(args.rois)
    except ValueError as error:
        parser.error(str(error))
    return args


def extract_features(X, rois):
    """Per ROI: temporal mean, population SD, and OLS slope per TR."""
    if X.ndim != 3 or X.shape[1] < 2 or X.shape[2] != len(rois):
        raise ValueError("Expected X shaped (segments, >=2 TRs, selected ROIs).")
    if not np.isfinite(X).all():
        raise ValueError("Segment arrays contain NaN or infinite values.")
    X = np.asarray(X, dtype=np.float64)
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
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=args.max_iter,
                               random_state=args.seed)
    # Do not silently report results from a fit that failed to converge.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(X), y, sample_weight=weights)
    return scaler, model


def macro_group_balanced_accuracy(y, predictions, groups):
    """Equal group weight; average recalls only over classes present per group."""
    scores = []
    for group in np.unique(groups):
        mask = groups == group
        scores.append(np.mean([
            (predictions[mask & (y == label)] == label).mean()
            for label in np.unique(y[mask])
        ]))
    return float(np.mean(scores))


def select_c(X, y, participants, groups, args):
    """Tune only on outer-training data, grouping by participant or run."""
    n_splits = min(args.inner_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Need at least two groups in each outer training fold.")
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    rows = []
    for c in args.c_grid:
        predictions = np.full(len(y), -1, dtype=int)
        for train, validation in splits:
            scaler, model = fit_model(X[train], y[train], participants[train], c, args)
            predictions[validation] = model.predict(scaler.transform(X[validation]))
        rows.append({"C": c, "inner_group_balanced_accuracy":
                     macro_group_balanced_accuracy(y, predictions, groups)})
    # The ascending C grid makes ties favor stronger regularization.
    best = max(rows, key=lambda row: row["inner_group_balanced_accuracy"])
    return best["C"], rows


def classification_metrics(y, predictions):
    responsive, nonresponsive = y == 0, y == 1
    return {
        "test_accuracy": float((predictions == y).mean()),
        "balanced_test_accuracy": float(balanced_accuracy_score(y, predictions)),
        "responsive_recall": float((predictions[responsive] == 0).mean()) if responsive.any() else np.nan,
        "nonresponsive_recall": float((predictions[nonresponsive] == 1).mean()) if nonresponsive.any() else np.nan,
    }


def main():
    args = parse_args()
    X, y, metadata = load_segments(args)
    y = np.asarray(y, dtype=int)
    if not (len(X) == len(y) == len(metadata)) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected aligned segment arrays/metadata and binary labels 0/1.")
    expected = metadata["label"].map({"responsive": 0, "non-responsive": 1})
    if not np.array_equal(y, expected.to_numpy()):
        raise ValueError("Cached y does not agree with segment metadata labels.")
    features, feature_names = extract_features(X, args.rois)
    participants = metadata["participant"].astype(str).to_numpy()
    groups = participants if args.analysis == "lopo" else (
        metadata["participant"].astype(str) + "::" + metadata["run"].astype(str)
    ).to_numpy()
    folds = get_folds(X, y, participants, metadata, args.analysis)
    output = (Path("plots/LR") / get_segment_dir_name(args) / args.roi_label
              / args.analysis.upper() / f"SEED={args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings.update(features=["mean", "sd", "slope_per_TR"],
                    regularization="L2", solver="lbfgs", threshold=0.5,
                    weighting="equal participants, then available classes",
                    inner_selection="group-averaged balanced accuracy",
                    note="Deterministic fits; seeds do not represent independent repetitions.")
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
    fold_rows, tuning_rows, coefficient_rows, training_sizes = [], [], [], []
    for fold, (train, test) in enumerate(folds, start=1):
        print(f"Fold {fold}/{len(folds)}: held out {groups[test[0]]}", flush=True)
        c, tuning = select_c(features[train], y[train], participants[train], groups[train], args)
        tuning_rows.extend({"fold": fold, **row} for row in tuning)
        scaler, model = fit_model(features[train], y[train], participants[train], c, args)
        probability = model.predict_proba(scaler.transform(features[test]))[:, 1]
        pred = (probability >= 0.5).astype(int)
        predictions.loc[test, "fold"] = fold
        predictions.loc[test, "prediction"] = pred
        predictions.loc[test, "probability_nonresponsive"] = probability
        train_accuracy = float((model.predict(scaler.transform(features[train])) == y[train]).mean())
        fold_rows.append({
            "fold": fold, "participant": participants[test[0]],
            **({"run": metadata.iloc[test[0]]["run"]} if args.analysis == "loro" else {}),
            "selected_C": c, "train_accuracy": train_accuracy,
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
        pd.DataFrame(tuning_rows).to_csv(output / f"{prefix}_inner_cv.csv", index=False)

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
        group_averaged_balanced_accuracy=macro_group_balanced_accuracy(y, pred, groups),
        n_segments=len(y), n_folds=len(folds), n_features=features.shape[1],
    )
    predictions.to_csv(output / f"{prefix}_predictions.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output / f"{prefix}_coefficients.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output / f"{prefix}_summary.csv", index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
