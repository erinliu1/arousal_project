"""Sequential ROI pruning using inner LOPO only, nearest-snapped canonical data.

Run one outer participant/seed/round with --run_pruning. Average a completed
round with --average_across_seeds --round N (all participants by default).
WD is fixed from the FIRST ROW of each participant's initial ranking, even
when that row's ROI selection differs from the 31-ROI starting selection.
All scores pool segment predictions across 19 inner validation participants.
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from GRU import ArousalDataset, resolve_rois, run_fold
from nested_LOPO_GRU_old import scientific, training_args
from nested_LOPO_GRU_new import load_data, SEEDS, PARTICIPANTS, WEIGHT_DECAYS

SOURCE = Path("plots/GRU/nested_LOPO,BP=nearest")
ROOT = Path("plots/GRU/nested_LOPO,BP=nearest,pruning")
BASE = "thalamic_rois+cortical_rois"
BASE_ROIS, _ = resolve_rois(["thalamic_rois", "cortical_rois"])
COLUMNS = ["outer_held_out_participant", "baseline_roi", "excluded_roi", "hidden", "WD",
           "responsive_recall", "nonresponsive_recall", "segment_weighted_balanced_accuracy"]


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent seeds can recompute the same previous-round averages.
    # Unique temporary files let each identical result be replaced atomically.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv.tmp", dir=path.parent,
                                     delete=False) as handle:
        temporary = Path(handle.name)
        try:
            frame.to_csv(handle, index=False)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def round_dir(outer, number):
    return ROOT / "inner_LOPO" / outer / f"round{number}"


def baseline_label(removed):
    return BASE + ("__exclude=" + "+".join(r for r in BASE_ROIS if r in removed) if removed else "")


def selected_wd(outer):
    if outer not in PARTICIPANTS:
        raise ValueError(f"Unknown outer participant: {outer}")
    path = SOURCE / "inner_LOPO" / outer / "best_configurations.csv"
    ranking = pd.read_csv(path)
    if ranking.empty:
        raise ValueError(f"Empty initial ranking: {path}")
    top = ranking.iloc[0]
    if top.outer_held_out_participant != outer or top.hidden != 64 or top.WD not in WEIGHT_DECAYS:
        raise ValueError(f"Invalid first-row configuration: {path}")
    return float(top.WD)


def initial_baseline(outer, wd):
    scores = {}
    for seed in SEEDS:
        path = SOURCE / "inner_LOPO" / outer / BASE / f"hidden=64,WD={scientific(wd)}" / f"seed={seed}.csv"
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(f"Expected one baseline result: {path}")
        row = frame.iloc[0]
        score = float(row.segment_pooled_balanced_accuracy)
        if (row.outer_held_out_participant != outer or row.roi != BASE
                or row.hidden != 64 or row.WD != wd or row.seed != seed
                or not np.isfinite(score) or not 0 <= score <= 1):
            raise ValueError(f"Invalid baseline identifiers/score: {path}")
        scores[seed] = score
    return scores


def read_round(path, outer, removed, wd):
    frame = pd.read_csv(path)
    candidates = set(BASE_ROIS) - set(removed)
    if (list(frame.columns) != COLUMNS or len(frame) != len(candidates)
            or frame.excluded_roi.duplicated().any() or set(frame.excluded_roi) != candidates
            or not frame.outer_held_out_participant.eq(outer).all()
            or not frame.baseline_roi.eq(baseline_label(removed)).all()
            or not frame.hidden.eq(64).all() or not frame.WD.eq(wd).all()):
        raise ValueError(f"Invalid round schema/configuration: {path}")
    metrics = frame[["responsive_recall", "nonresponsive_recall", "segment_weighted_balanced_accuracy"]]
    if (not np.isfinite(metrics.to_numpy()).all() or not ((metrics >= 0) & (metrics <= 1)).all().all()
            or not np.allclose(frame.segment_weighted_balanced_accuracy,
                               (frame.responsive_recall + frame.nonresponsive_recall) / 2)):
        raise ValueError(f"Invalid pooled metrics: {path}")
    return frame.set_index("excluded_roi")


def average_round(outer, number, removed, wd, baseline):
    """Require ten complete seed CSVs and compute paired differences per candidate."""
    frames = {seed: read_round(round_dir(outer, number) / f"seed={seed}.csv", outer, removed, wd)
              for seed in SEEDS}
    rows = []
    base = np.array([baseline[seed] for seed in SEEDS])
    for roi in BASE_ROIS:
        if roi in removed:
            continue
        scores = np.array([frames[seed].loc[roi, "segment_weighted_balanced_accuracy"] for seed in SEEDS])
        change = scores - base
        rows.append(dict(outer_held_out_participant=outer, baseline_roi=baseline_label(removed),
                         excluded_roi=roi, hidden=64, WD=wd, n_seed_pairs=len(SEEDS),
                         mean_BA=float(scores.mean()), baseline_mean_BA=float(base.mean()),
                         mean_BA_change=float(change.mean()), std_BA_change=float(change.std(ddof=1)),
                         n_improved_seeds=int((change > 0).sum())))
    averages = pd.DataFrame(rows).sort_values(["mean_BA_change", "excluded_roi"],
                                            ascending=[False, True]).reset_index(drop=True)
    atomic_csv(averages, round_dir(outer, number) / "seed_averages.csv")
    return averages, frames


def current_state(outer, number):
    if not 1 <= number < len(BASE_ROIS):
        raise ValueError("Round must be between 1 and 30 (at least one ROI must remain)")
    wd = selected_wd(outer)
    removed, baseline = [], initial_baseline(outer, wd)
    for previous in range(1, number):
        averages, frames = average_round(outer, previous, removed, wd, baseline)
        winner = averages.iloc[0].excluded_roi
        baseline = {seed: float(frames[seed].loc[winner, "segment_weighted_balanced_accuracy"])
                    for seed in SEEDS}
        removed.append(winner)
    return removed, wd, baseline


def run_pruning_round(outer_held_out_participant, seed, round_number):
    """One outer/seed/round: ablate each remaining ROI using 19 inner folds."""
    outer = str(outer_held_out_participant)
    if seed not in SEEDS:
        raise ValueError("Use a canonical model seed")
    removed, wd, _ = current_state(outer, round_number)
    output = round_dir(outer, round_number) / f"seed={seed}.csv"
    if output.is_file() and output.stat().st_size:
        read_round(output, outer, removed, wd)
        print(f"Skipping completed round: {output}", flush=True)
        return output
    X, y, metadata = load_data()
    participants = metadata.participant.astype(str).to_numpy()
    if set(participants) != set(PARTICIPANTS):
        raise ValueError("Expected all 20 canonical participants")
    remaining = [r for r in BASE_ROIS if r not in removed]
    from all_rois import all_rois
    rows = []
    for excluded in remaining:
        rois = [r for r in remaining if r != excluded]
        args = training_args(argparse.Namespace(rois=rois, roi_label=baseline_label(removed + [excluded]),
                                               bp_TR_snap="nearest"), 64, wd, seed)
        dataset = ArousalDataset(X[:, :, [all_rois.index(r) for r in rois]], y)
        labels, predictions, seen = [], [], []
        for validation_participant in PARTICIPANTS:
            if validation_participant == outer:
                continue
            train = np.flatnonzero((participants != outer) & (participants != validation_participant))
            validation = np.flatnonzero(participants == validation_participant)
            print(f"Outer={outer} | round={round_number} | exclude={excluded} | seed={seed} | "
                  f"validation={validation_participant}", flush=True)
            result, _ = run_fold(dataset, train, validation, participants, args, evaluation_split="validation")
            pred = np.asarray(result["predictions"])
            if pred.shape != y[validation].shape or not np.isin(pred, [0, 1]).all():
                raise ValueError("Invalid inner validation predictions")
            labels.append(y[validation]); predictions.append(pred); seen.extend(validation.tolist())
        if sorted(seen) != np.flatnonzero(participants != outer).tolist():
            raise ValueError("Inner validation must cover each non-outer segment exactly once")
        labels, predictions = np.concatenate(labels), np.concatenate(predictions)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Both classes must be present in pooled inner validation")
        rows.append(dict(outer_held_out_participant=outer, baseline_roi=baseline_label(removed),
                         excluded_roi=excluded, hidden=64, WD=wd,
                         responsive_recall=float((predictions[labels == 0] == 0).mean()),
                         nonresponsive_recall=float((predictions[labels == 1] == 1).mean()),
                         segment_weighted_balanced_accuracy=float(balanced_accuracy_score(labels, predictions))))
    atomic_csv(pd.DataFrame(rows, columns=COLUMNS), output)
    print(f"Saved {len(rows)} ablations to {output}", flush=True)
    return output


def average_across_seeds(round_number, outer_held_out_participant=None):
    participants = PARTICIPANTS if outer_held_out_participant is None else [outer_held_out_participant]
    outputs = []
    for outer in participants:
        removed, wd, baseline = current_state(outer, round_number)
        averages, _ = average_round(outer, round_number, removed, wd, baseline)
        top = averages.iloc[0]
        print(f"{outer} round {round_number}: top exclusion={top.excluded_roi}; "
              f"mean BA change={top.mean_BA_change:+.6f}", flush=True)
        outputs.append(round_dir(outer, round_number) / "seed_averages.csv")
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run_pruning", action="store_true")
    mode.add_argument("--average_across_seeds", action="store_true")
    parser.add_argument("--round", type=int, required=True, dest="round_number", choices=range(1, 31))
    parser.add_argument("--outer-held-out-participant", choices=PARTICIPANTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args(argv)
    if args.average_across_seeds:
        if args.seed is not None:
            parser.error("Averaging uses all ten seeds; omit --seed")
        return average_across_seeds(args.round_number, args.outer_held_out_participant)
    if args.outer_held_out_participant is None or args.seed is None:
        parser.error("--run_pruning requires --outer-held-out-participant and --seed")
    return run_pruning_round(args.outer_held_out_participant, args.seed, args.round_number)


if __name__ == "__main__":
    main()
