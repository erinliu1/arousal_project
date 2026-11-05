"""Participant-clustered bootstrap confidence intervals for saved predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
B = 10_000


def format_rate(value):
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def return_config_folder(rois="all_rois", W=30, S=1, N_min=8, G_max=4, T_min=20,
                         TR=10, SW=10, trim=False, model="GRU", pool="final", hidden=64,
                         drop=0.2, epochs=30, batch=32, LR=1e-3, WD=1e-2,
                         analysis="lopo", balance="participant", innerCV=0, C=0.01, FD=0.25):
    """Return the results folder matching a training configuration."""
    trim_suffix = ",trim" if trim else ""
    segment = f"R(W={W},S={S},N_min={N_min},G_max={G_max}):NR(T_min={T_min}):Segment(TR={TR},SW={SW}{trim_suffix})"
    analysis_dir = analysis.upper()
    if model == "GRU":
        model_dir = (f"model=GRU,pool={pool},hidden={hidden},drop={drop},epochs={epochs},"
                     f"batch={batch},LR={format_rate(LR)},WD={format_rate(WD)}")
        if innerCV > 0:
            analysis_dir += f",innerCV={innerCV}"
        if balance == "class":
            analysis_dir += ",balance=class"
        return PROJECT_DIR / "plots" / "GRU" / "segments" / segment / rois / model_dir / analysis_dir
    if model == "MLP":
        model_dir = (f"model=MLP,hidden={hidden},drop={drop:g},epochs={epochs},batch={batch},"
                     f"LR={LR:g},WD={WD:g}")
        return PROJECT_DIR / "plots" / "MLP" / segment / rois / model_dir / analysis_dir
    if model == "LR":
        return PROJECT_DIR / "plots" / "LR" / segment / rois / analysis_dir / f"C={C:g}"
    if model == "motion_LR":
        return PROJECT_DIR / "plots" / "motion_classifier" / segment / f"FD={FD:g}" / f"model=LR,C={C:g}" / analysis_dir
    raise ValueError(f"Unsupported model: {model}")


def load_predictions(config_folder, analysis):
    paths = sorted(config_folder.glob(f"SEED=*/{analysis.lower()}_predictions.csv"))
    direct_path = config_folder / f"{analysis.lower()}_predictions.csv"
    if direct_path.exists():
        paths.append(direct_path)
    results_path = config_folder / "results.csv"
    if results_path.exists():
        paths.append(results_path)
    if not paths:
        paths = sorted(config_folder.glob("seed=*.csv"))
    if not paths:
        raise FileNotFoundError(f"No prediction files found under {config_folder}")
    required = {"participant", "label_binary", "prediction"}
    predictions = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        predictions.append(frame)
    participants = sorted(set.intersection(*(set(frame["participant"].astype(str)) for frame in predictions)))
    if not participants:
        raise ValueError("Prediction files have no participants in common.")
    return predictions, participants, paths


def load_or_create_samples(participants, n_bootstrap, bootstrap_seed):
    sample_dir = PROJECT_DIR / "plots" / "GRU" / "bootstrap_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"participant_samples_B={n_bootstrap},seed={bootstrap_seed}.csv"
    if sample_path.exists():
        samples = pd.read_csv(sample_path, dtype=str).to_numpy()
        if samples.shape != (n_bootstrap, len(participants)) or not set(samples.ravel()).issubset(participants):
            raise ValueError(f"Existing bootstrap samples do not match this participant set: {sample_path}")
        return samples, sample_path
    rng = np.random.default_rng(bootstrap_seed)
    samples = rng.choice(participants, size=(n_bootstrap, len(participants)), replace=True)
    pd.DataFrame(samples, columns=[f"draw_{i + 1}" for i in range(len(participants))]).to_csv(sample_path, index=False)
    return samples, sample_path


def bootstrap_metrics(predictions, participants, samples):
    """Calculate seed-averaged SPBA and class recalls for every bootstrap replicate."""
    participant_index = {participant: i for i, participant in enumerate(participants)}
    counts = np.zeros((len(samples), len(participants)), dtype=np.int16)
    for row, sample in enumerate(samples):
        indices = np.fromiter((participant_index[p] for p in sample), dtype=int)
        counts[row] = np.bincount(indices, minlength=len(participants))

    seed_recalls = []
    for frame in predictions:
        frame = frame.assign(participant=frame["participant"].astype(str))
        frame = frame[frame["participant"].isin(participants)]
        by_participant = []
        for participant in participants:
            rows = frame[frame["participant"].eq(participant)]
            label, prediction = rows["label_binary"].to_numpy(), rows["prediction"].to_numpy()
            by_participant.append([(label == cls).sum() for cls in (0, 1)]
                                  + [((label == cls) & (prediction == cls)).sum() for cls in (0, 1)])
        totals = np.asarray(by_participant)
        denominators, correct = counts @ totals[:, :2], counts @ totals[:, 2:]
        if (denominators == 0).any():
            raise ValueError("A bootstrap replicate contains no segments for one of the two classes.")
        seed_recalls.append(correct / denominators)
    recalls = np.mean(seed_recalls, axis=0)
    return {"responsive_recall": recalls[:, 0], "nonresponsive_recall": recalls[:, 1],
            "spba": recalls.mean(axis=1)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    specs = {
        "rois": (str, "all_rois"), "W": (int, 30), "S": (int, 1), "N_min": (int, 8),
        "G_max": (int, 4), "T_min": (int, 20), "TR": (int, 10), "SW": (int, 10),
        "model": (str, "GRU"), "pool": (str, "final"), "hidden": (int, 64),
        "drop": (float, 0.2), "epochs": (int, 30), "batch": (int, 32),
        "LR": (float, 1e-3), "WD": (float, 1e-2), "analysis": (str, "lopo"),
        "balance": (str, "participant"), "innerCV": (int, 0),
        "C": (float, 0.01), "FD": (float, 0.25),
    }
    for name, (kind, default) in specs.items():
        parser.add_argument(f"--{name}", type=kind, default=default)
    parser.add_argument("--trim", action="store_true")
    parser.add_argument("--predictions-folder", type=Path,
                        help="Folder containing prediction CSVs; overrides the model configuration path")
    parser.add_argument("--n-bootstrap", type=int, default=B)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be at least 1")
    return args


def main():
    args = parse_args()
    config_names = ["rois", "W", "S", "N_min", "G_max", "T_min", "TR", "SW", "trim", "model",
                    "pool", "hidden", "drop", "epochs", "batch", "LR", "WD", "analysis", "balance",
                    "innerCV", "C", "FD"]
    config_folder = args.predictions_folder or return_config_folder(
        **{name: getattr(args, name) for name in config_names})
    predictions, participants, paths = load_predictions(config_folder, args.analysis)
    samples, sample_path = load_or_create_samples(participants, args.n_bootstrap, args.bootstrap_seed)
    metrics = bootstrap_metrics(predictions, participants, samples)
    intervals = {name: np.percentile(values, [2.5, 97.5]) for name, values in metrics.items()}
    result = pd.DataFrame([{
        "n_bootstrap": args.n_bootstrap, "bootstrap_seed": args.bootstrap_seed,
        "n_participants": len(participants), "n_model_seeds": len(predictions),
        "bootstrap_mean_spba": metrics["spba"].mean(),
        "ci_2.5": intervals["spba"][0], "ci_97.5": intervals["spba"][1],
        "bootstrap_mean_responsive_recall": metrics["responsive_recall"].mean(),
        "responsive_recall_ci_2.5": intervals["responsive_recall"][0],
        "responsive_recall_ci_97.5": intervals["responsive_recall"][1],
        "bootstrap_mean_nonresponsive_recall": metrics["nonresponsive_recall"].mean(),
        "nonresponsive_recall_ci_2.5": intervals["nonresponsive_recall"][0],
        "nonresponsive_recall_ci_97.5": intervals["nonresponsive_recall"][1],
    }])
    output_path = config_folder / "bootstrap_ci.csv"
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False))
    print(f"Predictions: {len(paths)} seeds\nSamples: {sample_path}\nSaved: {output_path}")


if __name__ == "__main__":
    main()
