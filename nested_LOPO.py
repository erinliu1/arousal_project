"""Run one model/seed/configuration of inner LOPO for one held-out participant."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import torch

from GRU import ArousalDataset, load_segments, resolve_rois, run_fold as run_gru_fold
from LR import extract_features, fit_model
from MLP import run_fold as run_mlp_fold
from motion_classifier import load_segments as load_motion_segments, run_fold as run_motion_fold


SEEDS = (42, 123, 456, 789, 2026, 12, 314, 258, 1024, 2048)
CONFIGS = {
    "GRU": {"H32_WD0.01": (32, 0.01), "H64_WD0.01": (64, 0.01), "H64_WD0.001": (64, 0.001)},
    "LR": {"C0.001": (0.001,), "C0.01": (0.01,), "C0.1": (0.1,), "C1": (1.0,)},
    "MLP": {"H16_WD0.001": (16, 0.001), "H16_WD0.01": (16, 0.01),
            "H32_WD0.001": (32, 0.001), "H32_WD0.01": (32, 0.01)},
    "motion_classifier": {"C0.001": (0.001,), "C0.01": (0.01,), "C0.1": (0.1,), "C1": (1.0,)},
}
MODELS = (*CONFIGS, "GRU_time_shuffle")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--paired-contrast", action="store_true")
    parser.add_argument("--visualize-paired-contrast", action="store_true")
    parser.add_argument("--model", choices=MODELS, default="GRU")
    parser.add_argument("--held-out-participant")
    parser.add_argument("--run", choices=["inner_lopo", "eval"], default="inner_lopo")
    parser.add_argument("--configuration", choices=sorted({name for configs in CONFIGS.values() for name in configs}))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.visualize or args.paired_contrast or args.visualize_paired_contrast:
        return args
    if args.model == "GRU_time_shuffle":
        if args.run != "eval" or args.summarize:
            parser.error("GRU_time_shuffle supports only --run eval, which summarizes automatically.")
        if any(value is not None for value in (args.held_out_participant, args.configuration, args.seed)):
            parser.error("GRU_time_shuffle evaluation does not accept participant, configuration, or seed arguments.")
        return args
    if args.configuration and args.configuration not in CONFIGS[args.model]:
        parser.error(f"Configuration {args.configuration} is not valid for {args.model}.")
    seeded = args.model in {"GRU", "MLP"}
    if args.seed is not None and not seeded:
        parser.error(f"{args.model} is deterministic and does not use --seed.")
    if args.run == "eval":
        if args.summarize:
            parser.error("--run eval automatically summarizes; do not use --summarize.")
        if any(value is not None for value in (args.held_out_participant, args.configuration, args.seed)):
            parser.error("--run eval does not accept participant, configuration, or seed arguments.")
    elif not args.summarize:
        required = [args.held_out_participant, args.configuration]
        if seeded:
            required.append(args.seed)
        if any(value is None for value in required):
            parser.error("Inner LOPO requires a held-out participant and configuration" +
                         (", and seed." if seeded else "."))
    return args


def training_args(model, configuration, seed):
    rois, roi_label = resolve_rois(["all_rois"])
    values = CONFIGS[model][configuration]
    hidden_size = values[0] if model in {"GRU", "MLP"} else 16
    weight_decay = values[1] if model in {"GRU", "MLP"} else 0.01
    c = values[0] if model in {"LR", "motion_classifier"} else 0.01
    return argparse.Namespace(
        seed=seed, hidden_size=hidden_size, weight_decay=weight_decay,
        epochs=30 if model == "GRU" else 10, batch_size=32, learning_rate=1e-3, dropout=0.2,
        pooling="final", balance="participant_then_class", rois=rois, roi_label=roi_label,
        segment_length=40, segment_stride=40, responsive_window_size=30,
        responsive_window_stride=1, responsive_min_bp_trs=8, responsive_max_bp_gap=4,
        trim_responsive_intervals=False, nonresponsive_min_gap=20,
        rebuild_segment_arrays=False, c=c, max_iter=5000, fd_threshold=0.25, threads=4,
    )


def output_dir(model):
    if model == "GRU_time_shuffle":
        return Path("plots/GRU_time_shuffle")
    return Path("plots") / model / "nested_LOPO"


def configuration_fields(model, configuration, prefix=""):
    values = CONFIGS[model][configuration]
    if model in {"GRU", "MLP"}:
        return {f"{prefix}hidden_size": values[0], f"{prefix}weight_decay": values[1]}
    return {f"{prefix}C": values[0]}


def selected_configuration(model, row):
    if "best_configuration" in row.index:
        return row["best_configuration"]
    if model in {"GRU", "MLP"}:
        values = (int(row["best_hidden_size"]), float(row["best_weight_decay"]))
    else:
        values = (float(row["best_C"]),)
    for configuration, parameters in CONFIGS[model].items():
        if len(values) == len(parameters) and all(np.isclose(left, right) for left, right in zip(values, parameters)):
            return configuration
    raise ValueError(f"Selected hyperparameters {values} do not match a {model} configuration.")


def load_model_data(model, model_args):
    if model == "GRU_time_shuffle":
        X, y, metadata = load_model_data("GRU", model_args)
        path = Path("plots/GRU_time_shuffle/segments") / f"seed={model_args.seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing time permutations at {path}. Run GRU_time_shuffle.py first.")
        shuffled = pd.read_csv(path)
        expected = metadata[["participant", "run"]].reset_index().rename(columns={"index": "segment_index"})
        if len(shuffled) != len(metadata) or not shuffled[["segment_index", "participant", "run"]].equals(
                expected[["segment_index", "participant", "run"]]):
            raise ValueError(f"Permutation metadata does not match the loaded segments: {path}")
        permutations = np.asarray([json.loads(value) for value in shuffled["permuted_TR_indices"]], dtype=int)
        if permutations.shape != (len(X), 40) or not np.all(np.sort(permutations, axis=1) == np.arange(40)):
            raise ValueError(f"Invalid 40-TR permutations in {path}.")
        return np.take_along_axis(X, permutations[:, :, None], axis=1), y, metadata
    if model == "motion_classifier":
        return load_motion_segments(model_args)
    X, y, metadata = load_segments(model_args)
    if model in {"LR", "MLP"}:
        X, _ = extract_features(X, model_args.rois)
    return X, np.asarray(y, dtype=int), metadata.reset_index(drop=True)


def run_model_fold(model, X, y, metadata, train, test, model_args):
    participants = metadata["participant"].astype(str).to_numpy()
    if model in {"GRU", "GRU_time_shuffle"}:
        result, _ = run_gru_fold(ArousalDataset(X, y), train, test, participants, model_args)
    elif model == "LR":
        scaler, fitted = fit_model(X[train], y[train], participants[train], model_args.c, model_args)
        probabilities = fitted.predict_proba(scaler.transform(X[test]))[:, 1]
        train_predictions = fitted.predict(scaler.transform(X[train]))
        result = {"predictions": (probabilities >= 0.5).astype(int), "probabilities": probabilities,
                  "train_accuracy": float((train_predictions == y[train]).mean())}
    elif model == "MLP":
        torch.set_num_threads(model_args.threads)
        torch.use_deterministic_algorithms(True)
        result = run_mlp_fold(X, y, participants, train, test, model_args)
    else:
        result = run_motion_fold(X, y, participants, train, test, model_args)
    return result


def run_nested_lopo(model, held_out_participant, configuration, seed):
    model_args = training_args(model, configuration, seed)
    X, y, metadata = load_model_data(model, model_args)
    participants = metadata["participant"].astype(str).to_numpy()
    held_out = str(held_out_participant)
    unique_participants = np.unique(participants)
    if held_out not in unique_participants:
        raise ValueError(f"Unknown held-out participant {held_out!r}.")
    inner_participants = unique_participants[unique_participants != held_out]
    labels, predictions = [], []
    for fold, validation_participant in enumerate(inner_participants, start=1):
        train = np.flatnonzero((participants != held_out) & (participants != validation_participant))
        validation = np.flatnonzero(participants == validation_participant)
        print(f"Inner fold {fold}/{len(inner_participants)}: {validation_participant}", flush=True)
        result = run_model_fold(model, X, y, metadata, train, validation, model_args)
        labels.append(y[validation].astype(int))
        predictions.append(result["predictions"].astype(int))
    labels, predictions = np.concatenate(labels), np.concatenate(predictions)
    score = balanced_accuracy_score(labels, predictions)
    row = {"held_out_participant": held_out, "configuration": configuration,
           **configuration_fields(model, configuration),
           "seed": seed, "inner_folds": len(inner_participants), "n_segments": len(labels),
           "segment_pooled_balanced_accuracy": score}
    filename = f"seed={seed}.csv" if seed is not None else "result.csv"
    output = output_dir(model) / held_out / configuration / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output, index=False)
    print(f"Segment-pooled balanced accuracy: {score:.6f}\nSaved {output}")


def summarize_nested_lopo(model):
    input_dir = output_dir(model)
    pattern = "*/*/seed=*.csv" if model in {"GRU", "MLP"} else "*/*/result.csv"
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No nested-LOPO results found under {input_dir}.")
    results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    keys = ["held_out_participant", "configuration"] + (["seed"] if model in {"GRU", "MLP"} else [])
    if results.duplicated(keys).any():
        raise ValueError("Duplicate held-out/configuration/seed results found.")
    comparison = {}
    for (held_out, configuration), group in results.groupby(keys[:2]):
        if model in {"GRU", "MLP"} and set(group["seed"]) != set(SEEDS):
            missing = sorted(set(SEEDS) - set(group["seed"]))
            raise ValueError(f"{held_out}/{configuration} is missing seeds: {missing}")
        comparison.setdefault(held_out, []).append(
            (configuration, float(group["segment_pooled_balanced_accuracy"].mean())))
    rows = []
    for held_out, scores in comparison.items():
        completed = {configuration for configuration, _ in scores}
        if completed != set(CONFIGS[model]):
            raise ValueError(f"{held_out} is missing configurations: {sorted(set(CONFIGS[model]) - completed)}")
        configuration, score = max(scores, key=lambda item: item[1])
        rows.append({"held_out_participant": held_out, "best_configuration": configuration,
                     **configuration_fields(model, configuration, "best_"),
                     "best_mean_segment_pooled_balanced_accuracy": score})
    summary = pd.DataFrame(rows)
    summary.to_csv(input_dir / "best_configurations.csv", index=False)
    return summary


def evaluate_best_configuration(model, held_out_participant, seed):
    selection_model = "GRU" if model == "GRU_time_shuffle" else model
    best_path = output_dir(selection_model) / "best_configurations.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"Best configurations file not found at {best_path}. Please summarize first.")
    best = pd.read_csv(best_path)
    selected = best[best["held_out_participant"].astype(str) == str(held_out_participant)]
    if selected.empty:
        raise ValueError(f"No best configuration found for held-out participant {held_out_participant}.")
    configuration = selected_configuration(selection_model, selected.iloc[0])
    model_args = training_args(selection_model, configuration, seed)
    X, y, metadata = load_model_data(model, model_args)
    participants = metadata["participant"].astype(str).to_numpy()
    held_out = str(held_out_participant)
    unique_participants = np.unique(participants)
    if held_out not in unique_participants:
        raise ValueError(f"Unknown held-out participant {held_out!r}.")
    train, test = np.flatnonzero(participants != held_out), np.flatnonzero(participants == held_out)
    result = run_model_fold(model, X, y, metadata, train, test, model_args)
    evaluation = metadata.iloc[test].copy()
    evaluation.insert(0, "segment_index", metadata.index[test].to_numpy())
    evaluation.insert(1, "fold", unique_participants.tolist().index(held_out) + 1)
    evaluation["label_binary"] = y[test]
    evaluation["prediction"] = result["predictions"].astype(int)
    evaluation["probability_nonresponsive"] = result["probabilities"]
    evaluation["train_accuracy"] = result["train_accuracy"]
    evaluation["n_train_segments"] = len(train)
    output = output_dir(model) / "evaluation" / held_out
    output.mkdir(parents=True, exist_ok=True)
    filename = f"seed={seed}.csv" if seed is not None else "result.csv"
    evaluation.to_csv(output / filename, index=False)
    print(f"Saved evaluation results to {output / filename}.")


def evaluate_all_best_configurations(model):
    selection_model = "GRU" if model == "GRU_time_shuffle" else model
    best_path = output_dir(selection_model) / "best_configurations.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"Best configurations file not found at {best_path}. Please summarize first.")
    best = pd.read_csv(best_path)
    seeds = SEEDS if model in {"GRU", "MLP", "GRU_time_shuffle"} else (None,)
    total = len(best) * len(seeds)
    run = 0
    for held_out_participant in best["held_out_participant"].astype(str):
        for seed in seeds:
            run += 1
            print(f"Evaluation {run}/{total}: held out={held_out_participant}" +
                  (f" | seed={seed}" if seed is not None else ""), flush=True)
            evaluate_best_configuration(model, held_out_participant, seed)
    summarize_eval(model)


def summarize_eval(model):
    input_dir = output_dir(model)
    pattern = "evaluation/*/seed=*.csv" if model in {"GRU", "MLP", "GRU_time_shuffle"} else "evaluation/*/result.csv"
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No evaluation results found under {input_dir}.")
    output = input_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for path in paths:
        key = path.stem.removeprefix("seed=") if path.stem != "result" else "result"
        groups.setdefault(key, []).append(path)
    rows = []
    for key, group_paths in groups.items():
        results = pd.concat([pd.read_csv(path) for path in group_paths], ignore_index=True)
        results.to_csv(output / (f"seed={key}.csv" if key != "result" else "results.csv"), index=False)
        labels, predictions = results["label_binary"], results["prediction"]
        participant_accuracy = results.assign(correct=predictions.eq(labels)).groupby("participant")["correct"].mean()
        training = results[["fold", "participant", "train_accuracy", "n_train_segments"]].drop_duplicates()
        rows.append({"seed": int(key) if key != "result" else np.nan,
                     "pooled_train_accuracy": np.average(training["train_accuracy"],
                                                          weights=training["n_train_segments"]),
                     "pooled_test_accuracy": predictions.eq(labels).mean(),
                     "participant_averaged_accuracy": participant_accuracy.mean(),
                     "responsive_recall": predictions[labels.eq(0)].eq(0).mean(),
                     "nonresponsive_recall": predictions[labels.eq(1)].eq(1).mean(),
                     "segment_weighted_balanced_accuracy": balanced_accuracy_score(labels, predictions)})
    seed_summary = pd.DataFrame(rows)
    metrics = [column for column in seed_summary if column != "seed"]
    seed_summary.to_csv(output / "seed_summary.csv", index=False)
    seed_summary[metrics].agg(["mean", "std"]).rename_axis("statistic").reset_index().to_csv(
        output / "summary.csv", index=False)
    print(f"Saved evaluation summaries to {output}.")

def visualize_nested_lopo():
    output_dir = Path("plots") / "GRU" / "nested_LOPO"

    models = ("GRU", "GRU_time_shuffle", "LR", "MLP", "motion_classifier")
    labels = ("GRU", "GRU (time\nshuffled)", "LR", "MLP", "Motion")
    metrics = (
        ("segment_weighted_balanced_accuracy", "Balanced Accuracy", "ci_2.5", "ci_97.5"),
        ("responsive_recall", "Responsive Recall", "responsive_recall_ci_2.5", "responsive_recall_ci_97.5"),
        ("nonresponsive_recall", "Non-Responsive Recall", "nonresponsive_recall_ci_2.5", "nonresponsive_recall_ci_97.5"),
    )
    estimates, intervals = [], []
    for model in models:
        folder = (Path("plots/GRU_time_shuffle") if model == "GRU_time_shuffle"
                  else Path("plots") / model / "nested_LOPO") / "evaluation"
        summary = pd.read_csv(folder / "summary.csv")
        estimates.append(summary.set_index("statistic").loc["mean"])
        intervals.append(pd.read_csv(folder / "bootstrap_ci.csv").iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharex=True, layout="constrained")
    positions = np.arange(len(models))
    colors = ("tab:blue", "tab:green", "#df65a0")
    for ax, (metric, title, lower_column, upper_column), color in zip(axes, metrics, colors):
        points = np.array([row[metric] for row in estimates])
        lower = np.array([row[lower_column] for row in intervals])
        upper = np.array([row[upper_column] for row in intervals])
        # Draw intervals independently so percentile CIs need not contain the estimate.
        ax.hlines(positions, lower, upper, color=color, linewidth=5)
        ax.plot(lower, positions, "|", color=color, markersize=20, markeredgewidth=3.5)
        ax.plot(upper, positions, "|", color=color, markersize=20, markeredgewidth=3.5)
        ax.plot(points, positions, "o", color=color, markersize=16)
        if metric == "segment_weighted_balanced_accuracy":
            ax.axvline(0.5, color="0.4", linestyle=":", linewidth=3.5, zorder=0)
        ax.set_yticks(positions, labels)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_title(title, fontsize=32, pad=26)
        ax.tick_params(axis="both", labelsize=24, width=2.5, length=10)
        ax.grid(axis="both", alpha=0.25, linewidth=2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "nested_lopo_metrics.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved nested-LOPO visualization to {output}.")

def save_balanced_accuracy_bootstraps():
    output_dir = Path("plots") / "GRU" / "nested_LOPO"
    sample_path = Path("plots/GRU/bootstrap_samples/participant_samples_B=10000,seed=0.csv")
    samples = pd.read_csv(sample_path, dtype=str).to_numpy()
    if samples.shape != (10000, 20) or pd.isna(samples).any():
        raise ValueError(f"Expected 10,000 bootstrap samples of 20 participants in {sample_path}.")
    participants = sorted(set(samples.ravel()))
    participant_index = {participant: i for i, participant in enumerate(participants)}
    counts = np.zeros((len(samples), len(participants)), dtype=int)
    for row, sample in enumerate(samples):
        counts[row] = np.bincount([participant_index[p] for p in sample], minlength=len(participants))

    scores = {}
    point_estimates = {}
    models = ("GRU", "GRU_time_shuffle", "MLP", "LR", "motion_classifier")
    for model in models:
        folder = (Path("plots") / model / "nested_LOPO" / "evaluation")
        if model == "GRU_time_shuffle" and not folder.exists():
            folder = Path("plots/GRU_time_shuffle/evaluation")
        summary = pd.read_csv(folder / "summary.csv").set_index("statistic")
        point_estimates[model] = float(summary.loc["mean", "segment_weighted_balanced_accuracy"])
        seeds = SEEDS if model in {"GRU", "GRU_time_shuffle", "MLP"} else (None,)
        seed_scores = []
        for seed in seeds:
            path = folder / (f"seed={seed}.csv" if seed is not None else "result.csv")
            if seed is None and not path.exists():
                path = folder / "results.csv"
            frame = pd.read_csv(path, dtype={"participant": str})
            if set(frame["participant"]) != set(participants):
                raise ValueError(f"Participant set does not match the bootstrap samples: {path}")
            if not frame[["label_binary", "prediction"]].isin([0, 1]).all().all():
                raise ValueError(f"Expected binary labels and predictions in {path}.")
            # Sum class totals and correct predictions before applying resampling multiplicities.
            totals = frame.assign(
                n0=frame["label_binary"].eq(0), n1=frame["label_binary"].eq(1),
                correct0=frame["label_binary"].eq(0) & frame["prediction"].eq(0),
                correct1=frame["label_binary"].eq(1) & frame["prediction"].eq(1),
            ).groupby("participant")[["n0", "n1", "correct0", "correct1"]].sum().loc[participants].to_numpy()
            denominators = counts @ totals[:, :2]
            correct = counts @ totals[:, 2:]
            if (denominators == 0).any():
                raise ValueError(f"A bootstrap sample lacks one of the two classes: {path}")
            seed_scores.append((correct / denominators).mean(axis=1))
        scores[model] = np.asarray(seed_scores)

    rows = []
    for model in models[1:]:
        # Seed rows share the explicit SEEDS order; deterministic models broadcast.
        differences = (scores["GRU"] - scores[model]).mean(axis=0)
        lower, upper = np.percentile(differences, [2.5, 97.5])
        rows.append({"comparison_model": model,
                     "point_estimate": point_estimates["GRU"] - point_estimates[model],
                     "ci_2.5": lower, "ci_97.5": upper})
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "bootstrap_GRU_paired_BA_difference.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved paired balanced-accuracy bootstrap intervals to {output}.")

def visualize_balanced_accuracy_paired_contrast():
    output_dir = Path("plots") / "GRU" / "nested_LOPO"
    results = pd.read_csv(output_dir / "bootstrap_GRU_paired_BA_difference.csv")
    
    models = ("GRU_time_shuffle", "LR", "MLP", "motion_classifier")
    labels = ("GRU (time\nshuffled)", "LR", "MLP", "Motion")
    results = results.set_index("comparison_model").loc[list(models)]
    positions = np.arange(len(models))
    color = "#6baed6"
    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    ax.hlines(positions, results["ci_2.5"], results["ci_97.5"], color=color, linewidth=5)
    ax.plot(results["ci_2.5"], positions, "|", color=color, markersize=20, markeredgewidth=3.5)
    ax.plot(results["ci_97.5"], positions, "|", color=color, markersize=20, markeredgewidth=3.5)
    ax.plot(results["point_estimate"], positions, "o", color=color, markersize=16)
    ax.axvline(0, color="0.4", linewidth=3.5, zorder=0)
    ax.set_yticks(positions, labels)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.margins(x=0.15)
    ax.set_title("Balanced Accuracy Difference\nGRU − Other Models", fontsize=32, pad=26)
    ax.set_xlabel("Balanced Accuracy Difference", fontsize=24, labelpad=18)
    ax.tick_params(axis="both", labelsize=24, width=2.5, length=10)
    ax.grid(axis="both", alpha=0.25, linewidth=2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    output = output_dir / "balanced_accuracy_paired_contrast.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paired balanced-accuracy visualization to {output}.")

if __name__ == "__main__":
    args = parse_args()
    if args.visualize:
        visualize_nested_lopo()
    elif args.visualize_paired_contrast:
        visualize_balanced_accuracy_paired_contrast()
    elif args.paired_contrast:
        save_balanced_accuracy_bootstraps()
    elif args.summarize:
        summarize_nested_lopo(args.model)
    elif args.run == "inner_lopo":
        run_nested_lopo(args.model, args.held_out_participant, args.configuration, args.seed)
    else:
        evaluate_all_best_configurations(args.model)
