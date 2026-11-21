"""Paired participant bootstrap for the fixed GRU sequential-pruning models.

Run from the project directory: python roi_pruning_paired_bootstraps.py
Differences are smaller/pruned model minus larger/baseline model.
Uses 10,000 identical participant resamples (bootstrap seed 0), averages metrics
across the ten matched model seeds, and reports percentile 95% intervals.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap import (
    B, NESTED_GRU_SEEDS, PROJECT_DIR, bootstrap_metrics,
    load_or_create_samples, load_predictions, return_config_folder,
)

EXCLUSIONS = {
    31: "",
    28: "fusiform+AV+CM",
    27: "entorhinal+fusiform+AV+CM",
    26: "entorhinal+fusiform+rostralmiddlefrontal+AV+CM",
}
COMPARISONS = ((31, 28), (31, 27), (31, 26), (28, 27), (27, 26))
KEYS = ["participant", "run", "segment_start_TR", "segment_end_TR_exclusive"]
METRICS = {"BA": "spba", "responsive_recall": "responsive_recall",
           "nonresponsive_recall": "nonresponsive_recall"}
OUTPUT = PROJECT_DIR / "plots/GRU/results/roi_pruning_paired_bootstraps.csv"


def load_models():
    """Require exactly ten matching seeds and identical labeled segments."""
    models, reference = {}, None
    for n_rois, excluded in EXCLUSIONS.items():
        label = "thalamic_rois+cortical_rois" + ("__exclude=" + excluded if excluded else "")
        folder = return_config_folder(rois=label, TR=40, SW=40, hidden=64,
                                      WD=.001, LR=.001, epochs=30, analysis="lopo")
        frames, participants, paths = load_predictions(folder, "lopo")
        by_seed = {}
        for frame, path in zip(frames, paths):
            if not path.parent.name.startswith("SEED="):
                raise ValueError(f"Expected a seed subfolder: {path}")
            seed = int(path.parent.name.removeprefix("SEED="))
            if seed in by_seed:
                raise ValueError(f"Duplicate seed {seed}: {folder}")
            missing = set(KEYS + ["label_binary", "prediction"]) - set(frame.columns)
            if missing:
                raise ValueError(f"Missing columns {missing}: {path}")
            if frame[KEYS + ["label_binary", "prediction"]].isna().any().any():
                raise ValueError(f"Missing segment identities, labels, or predictions: {path}")
            if frame.duplicated(KEYS).any():
                raise ValueError(f"Duplicate segments: {path}")
            if not frame.label_binary.isin([0, 1]).all() or not frame.prediction.isin([0, 1]).all():
                raise ValueError(f"Non-binary labels or predictions: {path}")
            frame = frame.copy()
            frame["participant"] = frame["participant"].astype(str)
            identity = frame.set_index(KEYS)["label_binary"].sort_index()
            if reference is None:
                reference = identity
            elif not identity.equals(reference):
                raise ValueError(f"Segments or true labels differ between models/seeds: {path}")
            by_seed[seed] = frame
        if set(by_seed) != set(NESTED_GRU_SEEDS):
            raise ValueError(f"Expected exactly the ten model seeds in {folder}; found {sorted(by_seed)}")
        if set(participants) != set(reference.index.get_level_values("participant")):
            raise ValueError(f"Participant mismatch in {folder}")
        models[n_rois] = [by_seed[seed] for seed in NESTED_GRU_SEEDS]
    return models, sorted(reference.index.get_level_values("participant").unique())


def paired_interval(baseline, pruned):
    """Subtract matching replicate metrics before computing mean and percentiles."""
    difference = np.asarray(pruned) - np.asarray(baseline)
    if difference.ndim != 1 or np.shape(pruned) != np.shape(baseline) or not np.isfinite(difference).all():
        raise ValueError("Paired bootstrap metrics must be aligned, finite vectors")
    lower, upper = np.percentile(difference, [2.5, 97.5])
    return float(difference.mean()), float(lower), float(upper)


def main():
    models, participants = load_models()
    samples, sample_path = load_or_create_samples(participants, B, 0)
    bootstraps = {n: bootstrap_metrics(frames, participants, samples)
                  for n, frames in models.items()}
    # Original-data point estimates, using each participant once, averaged over seeds.
    original = np.asarray([participants])
    observed = {n: bootstrap_metrics(frames, participants, original)
                for n, frames in models.items()}
    rows = []
    for baseline, pruned in COMPARISONS:
        row = {"comparison": f"{pruned}_minus_{baseline}",
               "baseline_n_rois": baseline, "pruned_n_rois": pruned,
               "n_participants": len(participants), "n_model_seeds": len(NESTED_GRU_SEEDS),
               "n_bootstrap": B, "bootstrap_seed": 0}
        for name, metric in METRICS.items():
            mean, lower, upper = paired_interval(bootstraps[baseline][metric], bootstraps[pruned][metric])
            row.update({f"{name}_observed_diff": float(observed[pruned][metric][0] - observed[baseline][metric][0]),
                        f"{name}_mean_diff": mean, f"{name}_ci_lower": lower,
                        f"{name}_ci_upper": upper})
        rows.append(row)
    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(f"Differences are pruned minus baseline, in fraction units.\nSamples: {sample_path}\nSaved: {OUTPUT}")
    return result


if __name__ == "__main__":
    main()
