"""Summarize seed-paired balanced-accuracy changes after ROI pruning.

Run: python roi_pruning.py
     python roi_pruning.py --rois thalamic_rois cortical_rois
Uses the canonical 40-TR setup, HD=64, LR=0.001, 30 epochs.
Default WD is 0.003 for all_rois and 0.001 for thalamus + cortex.
Change = pruned balanced accuracy - baseline balanced accuracy (fraction units).
Positive changes indicate improvement after pruning. SD uses ddof=1.
Uses available completed seed pairs for each ROI; n_seed_pairs reports the count.
For thalamus + cortex exclude4, the baseline excludes fusiform, AV, CM.
For thalamus + cortex exclude5, the baseline excludes fusiform, AV, CM,
and entorhinal; changes measure the effect of the fifth removal only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from all_rois import all_rois, thalamic_rois, cortical_rois

SEEDS = (42, 123, 456, 789, 2026, 12, 314, 258, 1024, 2048)
SEGMENT_CONFIG = "R(W=30,S=1,N_min=8,G_max=4):NR(T_min=20):Segment(TR=40,SW=40)"
FIXED_EXCLUSIONS = ("fusiform", "AV", "CM", "entorhinal")
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


def calculate_averages(segments_root, base_label="all_rois", rois=None, model_config=MODEL_CONFIG,
                       fixed_exclusions=()):
    """Pair each exclusion with the same seed's baseline, optionally already pruned."""
    root = Path(segments_root) / SEGMENT_CONFIG
    rois = all_rois if rois is None else rois
    baseline_label = base_label
    baseline_display = base_label
    if fixed_exclusions:
        canonical_fixed = [roi for roi in all_rois if roi in fixed_exclusions]
        baseline_label += "__exclude=" + "+".join(canonical_fixed)
        baseline_display += "_exclude_" + "+".join(canonical_fixed)
    baseline = {}
    for seed in SEEDS:
        try:
            baseline[seed] = read_balanced_accuracy(
                root / baseline_label / model_config / "LOPO" / f"SEED={seed}")
        except FileNotFoundError:
            continue
    rows = []
    for roi in rois:
        additional = set(roi.split("+")) - set(fixed_exclusions)
        if fixed_exclusions and (not set(fixed_exclusions).issubset(roi.split("+")) or len(additional) != 1):
            continue
        paired_seeds, changes, pruned_scores, baseline_scores = [], [], [], []
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
            pruned_scores.append(pruned)
            baseline_scores.append(baseline[seed])
        changes = np.asarray(changes, dtype=float)
        rows.append({
            "baseline": baseline_display,
            **({"additional_excluded_roi": next(iter(additional))}
               if fixed_exclusions else {"excluded_roi": roi}),
            "n_seed_pairs": len(paired_seeds),
            "mean_BA": float(np.mean(pruned_scores)) if paired_seeds else np.nan,
            "baseline_mean_BA": float(np.mean(baseline_scores)) if paired_seeds else np.nan,
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
    parser.add_argument("--exclude-count", type=int, default=1,
                        help="Number of excluded ROIs; discover existing multi-ROI folders when >1.")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override the baseline and pruned models' WD.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: plots/GRU/results/<base selection>_excludeN_averages.csv")
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
    if args.exclude_count < 1:
        parser.error("--exclude-count must be at least 1.")
    fixed_exclusions = ()
    if base_label == "thalamic_rois+cortical_rois":
        if args.exclude_count == 4:
            fixed_exclusions = FIXED_EXCLUSIONS[:3]
        elif args.exclude_count == 5:
            fixed_exclusions = FIXED_EXCLUSIONS
    if args.exclude_count > 1:
        exclusions = []
        root = args.segments_root / SEGMENT_CONFIG
        for folder in sorted(root.glob(f"{base_label}__exclude=*")):
            excluded = folder.name.split("__exclude=", 1)[1]
            names = excluded.split("+")
            if folder.is_dir() and len(set(names)) == args.exclude_count and set(names).issubset(rois):
                if not fixed_exclusions or set(fixed_exclusions).issubset(names):
                    exclusions.append(excluded)
        if not exclusions:
            parser.error(f"No matching exclude{args.exclude_count} folders found under {root}.")
        rois = exclusions
    if args.output is None:
        args.output = Path("plots/GRU/results") / f"{base_label}_exclude{args.exclude_count}_averages.csv"
    # Validate available results before writing; missing or incomplete pairs are skipped.
    try:
        result = calculate_averages(args.segments_root, base_label, rois, model_config, fixed_exclusions)
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.exit(1, f"Cannot calculate the pruning summary: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} ROI summaries to {args.output}")
    print("Change = pruned - baseline; positive means pruning improved balanced accuracy.")


if __name__ == "__main__":
    main()
