"""Seed-free nested LOPO shared by the ROI and motion logistic baselines."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

PARTICIPANTS = tuple(f"AAN_{i:02d}" for i in range(1, 22) if i != 13)
DEFAULT_C_GRID = (0.001, 0.01, 0.1, 1., 10.)
KEYS = ["participant", "run", "segment_start_TR", "segment_end_TR_exclusive"]


def add_arguments(parser):
    parser.add_argument("--run_inner", action="store_true")
    parser.add_argument("--outer-held-out-participant", choices=PARTICIPANTS)
    parser.add_argument("--get_best_starting_config", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--c-grid", nargs="+", type=float, default=DEFAULT_C_GRID)
    parser.add_argument("--summarize-legacy", action="store_true")


def validate_arguments(parser, args):
    if len(set(args.c_grid)) != len(args.c_grid) or not all(np.isfinite(c) and c > 0 for c in args.c_grid):
        parser.error("--c-grid must contain unique finite positive values")
    if args.run_inner and (args.eval or args.get_best_starting_config or args.summarize):
        parser.error("--run_inner cannot be combined with selection, eval, or summary")
    if args.run_inner != (args.outer_held_out_participant is not None):
        parser.error("--run_inner requires --outer-held-out-participant (and vice versa)")
    if args.summarize and (args.eval or args.get_best_starting_config):
        parser.error("--summarize is standalone")


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


class NestedLOPO:
    def __init__(self, root, label, args, loader, fit_fold):
        self.root, self.label, self.args = Path(root), label, args
        self.loader, self.fit_fold = loader, fit_fold
        self.settings = dict(roi=label, BP="nearest", features=getattr(args, "features", "fd_summaries"),
                             fd_threshold=getattr(args, "fd_threshold", None), max_iter=args.max_iter,
                             c_grid=list(args.c_grid))
        # Build canonical arguments independently of legacy CLI defaults.
        self.data_args = argparse.Namespace(**{**vars(args),
            "bp_TR_snap": "nearest", "segment_length": 40, "segment_stride": 40,
            "responsive_window_size": 30, "responsive_window_stride": 1,
            "responsive_min_bp_trs": 8, "responsive_max_bp_gap": 4,
            "trim_responsive_intervals": False, "nonresponsive_min_gap": 20,
            "rebuild_segment_arrays": False})

    def data(self):
        X, y, metadata = self.loader(self.data_args)
        metadata = metadata.reset_index(drop=True)
        y = np.asarray(y, dtype=int)
        participants = metadata.participant.astype(str).to_numpy()
        if (len(X) != len(y) or len(y) != len(metadata) or set(participants) != set(PARTICIPANTS)
                or set(np.unique(y)) != {0, 1} or metadata.duplicated(KEYS).any()
                or not np.array_equal(metadata.label.map({"responsive":0,"non-responsive":1}), y)):
            raise ValueError("Expected aligned canonical nearest-snapped data with all 20 participants")
        return X, y, metadata, participants

    def result(self, X, y, participants, train, test, c):
        args = argparse.Namespace(**{**vars(self.data_args), "c": c})
        result = self.fit_fold(X, y, participants, train, test, args)
        pred, prob = np.asarray(result["predictions"]), np.asarray(result["probabilities"])
        if (pred.shape != y[test].shape or prob.shape != y[test].shape
                or not np.isin(pred, [0, 1]).all() or not np.isfinite(prob).all()
                or not ((prob >= 0) & (prob <= 1)).all()):
            raise ValueError("Invalid held-out predictions")
        return result

    def run_inner(self, outer):
        X, y, metadata, participants = self.data()
        rows = []
        for c in self.args.c_grid:
            indices, predicted = [], []
            for validation in PARTICIPANTS:
                if validation == outer: continue
                train = np.flatnonzero((participants != outer) & (participants != validation))
                test = np.flatnonzero(participants == validation)
                print(f"Outer={outer} | C={c:g} | inner validation={validation}", flush=True)
                result = self.result(X, y, participants, train, test, c)
                indices.extend(test); predicted.extend(result["predictions"])
            if sorted(indices) != np.flatnonzero(participants != outer).tolist():
                raise ValueError("Inner validation coverage mismatch")
            rows.append(dict(outer_held_out_participant=outer, roi=self.label, C=c,
                             n_validation_segments=len(indices),
                             segment_pooled_balanced_accuracy=balanced_accuracy_score(y[indices], predicted)))
        folder = self.root / "inner_LOPO" / outer / self.label
        import json
        atomic_csv(pd.DataFrame(rows), folder / "configurations.csv")
        (folder / "settings.json").write_text(json.dumps(self.settings, sort_keys=True, indent=2)+"\n")
        return folder / "configurations.csv"

    def select(self):
        import json
        rankings = {}
        for outer in PARTICIPANTS:
            folder = self.root / "inner_LOPO" / outer / self.label
            if json.loads((folder / "settings.json").read_text()) != self.settings:
                raise ValueError(f"Inner settings mismatch: {folder}")
            frame = pd.read_csv(folder / "configurations.csv")
            if (len(frame) != len(self.args.c_grid) or frame.C.duplicated().any()
                    or set(frame.C) != set(self.args.c_grid)
                    or not frame.outer_held_out_participant.eq(outer).all()
                    or not frame.roi.eq(self.label).all()
                    or not frame.segment_pooled_balanced_accuracy.between(0, 1).all()
                    or frame.n_validation_segments.nunique() != 1 or (frame.n_validation_segments <= 0).any()):
                raise ValueError(f"Invalid inner results: {folder}")
            ranking = frame.sort_values(["segment_pooled_balanced_accuracy", "C"], ascending=[False, True]).reset_index(drop=True)
            ranking.insert(0, "rank", np.arange(1, len(ranking)+1))
            rankings[outer] = ranking
        for outer, ranking in rankings.items():
            atomic_csv(ranking, self.root / "inner_LOPO" / outer / self.label / "best_configurations.csv")
            print(f"{outer}: best C={ranking.iloc[0].C:g}, pooled inner BA={ranking.iloc[0].segment_pooled_balanced_accuracy:.6f}", flush=True)
        path = (self.root / "best_configurations.csv" if self.label in ("all_rois", "motion") else
                self.root / "best_configurations" / self.label / "best_configurations.csv")
        atomic_csv(pd.concat([f.iloc[[0]] for f in rankings.values()], ignore_index=True), path)
        return path

    def evaluate(self):
        # Validate all inner inputs, and write the selected configurations before fitting.
        selected = pd.read_csv(self.select()).set_index("outer_held_out_participant")
        X, y, metadata, participants = self.data()
        frames, folds = [], []
        for fold, outer in enumerate(PARTICIPANTS, 1):
            c = float(selected.loc[outer, "C"])
            train, test = np.flatnonzero(participants != outer), np.flatnonzero(participants == outer)
            result = self.result(X, y, participants, train, test, c)
            metrics = self.metrics(y[test], np.asarray(result["predictions"]))
            folds.append(dict(fold=fold, outer_held_out_participant=outer, roi=self.label, C=c,
                              n_train_segments=len(train), n_test_segments=len(test),
                              train_accuracy=result["train_accuracy"], **metrics))
            frame = metadata.iloc[test].copy()
            frame["segment_index"], frame["fold"] = test, fold
            frame["outer_held_out_participant"], frame["roi"], frame["C"] = outer, self.label, c
            frame["label_binary"], frame["prediction"] = y[test], result["predictions"]
            frame["probability_nonresponsive"] = result["probabilities"]
            frames.append(frame)
        predictions = pd.concat(frames).sort_values("segment_index").reset_index(drop=True)
        if predictions.segment_index.tolist() != list(range(len(y))):
            raise ValueError("Outer evaluation coverage mismatch")
        folds = pd.DataFrame(folds)
        summary = self.metrics(predictions.label_binary.to_numpy(), predictions.prediction.to_numpy())
        summary["pooled_train_accuracy"] = np.average(folds.train_accuracy, weights=folds.n_train_segments)
        folder = self.root / "outer_LOPO" / self.label
        for name, frame in (("predictions.csv",predictions),("fold_results.csv",folds),("summary.csv",pd.DataFrame([summary]))):
            atomic_csv(frame, folder/name)
        print(f"Saved deterministic outer LOPO: {folder}", flush=True)
        return folder

    @staticmethod
    def metrics(y, pred):
        r, nr = y == 0, y == 1
        return dict(pooled_test_accuracy=float((y == pred).mean()),
                    segment_weighted_balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                    responsive_recall=float((pred[r] == 0).mean()) if r.any() else np.nan,
                    nonresponsive_recall=float((pred[nr] == 1).mean()) if nr.any() else np.nan)

    def summarize(self):
        folder = self.root / "outer_LOPO" / self.label
        frame = pd.read_csv(folder / "predictions.csv")
        if (set(frame.participant) != set(PARTICIPANTS) or frame.duplicated(KEYS).any()
                or not frame.participant.eq(frame.outer_held_out_participant).all()
                or not frame.roi.eq(self.label).all()
                or not frame.label_binary.isin([0,1]).all() or not frame.prediction.isin([0,1]).all()):
            raise ValueError("Invalid deterministic outer predictions")
        summary = pd.read_csv(folder / "summary.csv")
        expected = self.metrics(frame.label_binary.to_numpy(), frame.prediction.to_numpy())
        if len(summary) != 1 or any(not np.isclose(summary.iloc[0][k],v) for k,v in expected.items()):
            raise ValueError("Summary does not match saved predictions")
        output = self.root / "evaluation" / self.label / "model_summary.csv"
        atomic_csv(summary, output)
        subprocess.run([sys.executable, str(Path(__file__).with_name("bootstrap.py")),
                        "--predictions-folder", str(folder)], check=True)
        return output
