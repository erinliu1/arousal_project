"""Summarize seed-paired balanced-accuracy changes after single-ROI pruning.

Run: python roi_pruning.py
     python roi_pruning.py --rois thalamic_rois cortical_rois
Uses the canonical 40-TR setup, HD=64, LR=0.001, 30 epochs.
Default WD is 0.003 for all_rois and 0.001 for thalamus + cortex.
Change = pruned balanced accuracy - baseline balanced accuracy (fraction units).
Positive changes indicate improvement after pruning. SD uses ddof=1.
Uses available completed seed pairs for each ROI; n_seed_pairs reports the count.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from all_rois import all_rois, thalamic_rois, cortical_rois

SEEDS = (42, 123, 456, 789, 2026, 12, 314, 258, 1024, 2048)
SEGMENT_CONFIG = "R(W=30,S=1,N_min=8,G_max=4):NR(T_min=20):Segment(TR=40,SW=40)"
MODEL_CONFIG = "model=GRU,pool=final,hidden=64,drop=0.2,epochs=30,batch=32,LR=1e-3,WD=3e-3"


def read_balanced_accuracy(folder):
    """Read a completed run, rejecting incomplete or invalid results."""
    for name in ("lopo_summary.csv", "lopo_fold_results.csv", "lopo_predictions.csv"):
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing completed-run output: {path}")
    summary = pd.read_csv(folder / "lopo_summary.csv")
    if len(summary) != 1:
        raise ValueError(f"Expected one summary row: {folder}")
    value = float(summary.iloc[0]["balanced_test_accuracy"])
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid balanced accuracy in {folder}: {value}")
    return value


def calculate_averages(segments_root, base_label="all_rois", rois=None, model_config=MODEL_CONFIG):
    """Pair each exclusion with the same seed's unpruned baseline."""
    root = Path(segments_root) / SEGMENT_CONFIG
    rois = all_rois if rois is None else rois
    baseline = {}
    for seed in SEEDS:
        try:
            baseline[seed] = read_balanced_accuracy(
                root / base_label / model_config / "LOPO" / f"SEED={seed}")
        except FileNotFoundError:
            continue
    rows = []
    for roi in rois:
        paired_seeds, changes = [], []
        for seed in SEEDS:
            if seed not in baseline:
                continue
            try:
                pruned = read_balanced_accuracy(
                    root / f"{base_label}__exclude={roi}" / model_config / "LOPO" / f"SEED={seed}")
            except FileNotFoundError:
                continue
            paired_seeds.append(seed)
            changes.append(pruned - baseline[seed])
        changes = np.asarray(changes, dtype=float)
        rows.append({
            "excluded_roi": roi,
            "n_seed_pairs": len(paired_seeds),
            "mean_balanced_accuracy_change": float(changes.mean()) if len(changes) else np.nan,
            "std_balanced_accuracy_change": float(changes.std(ddof=1)) if len(changes) >= 2 else np.nan,
            "improved_seeds": json.dumps([seed for seed, change in zip(paired_seeds, changes) if change > 0]),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments-root", type=Path, default=Path("plots/GRU/segments"))
    parser.add_argument("--rois", nargs="+", default=["all_rois"],
                        choices=["all_rois", "thalamic_rois", "cortical_rois"],
                        help="Use all_rois, or thalamic_rois cortical_rois together.")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override the baseline and pruned models' WD.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: plots/GRU/results/<base selection>_exclude1_averages.csv")
    args = parser.parse_args()
    selection = set(args.rois)
    if selection == {"all_rois"}:
        base_label, rois, default_wd = "all_rois", all_rois, .003
    elif selection == {"thalamic_rois", "cortical_rois"}:
        base_label = "thalamic_rois+cortical_rois"
        rois = [roi for roi in all_rois if roi in thalamic_rois or roi in cortical_rois]
        default_wd = .001
    else:
        parser.error("Use --rois all_rois or --rois thalamic_rois cortical_rois.")
    wd = default_wd if args.weight_decay is None else args.weight_decay
    if not np.isfinite(wd) or wd < 0:
        parser.error("--weight-decay must be finite and nonnegative.")
    wd_label = f"{wd:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    model_config = MODEL_CONFIG.rsplit("WD=", 1)[0] + "WD=" + wd_label
    if args.output is None:
        args.output = Path("plots/GRU/results") / f"{base_label}_exclude1_averages.csv"
    # Validate available results before writing; missing or incomplete pairs are skipped.
    try:
        result = calculate_averages(args.segments_root, base_label, rois, model_config)
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.exit(1, f"Cannot calculate the pruning summary: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} ROI summaries to {args.output}")
    print("Change = pruned - baseline; positive means pruning improved balanced accuracy.")


if __name__ == "__main__":
    main()
