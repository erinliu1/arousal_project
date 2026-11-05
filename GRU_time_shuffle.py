"""Create seeded, independently time-permuted 40-TR segment definitions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nested_LOPO import SEEDS, load_model_data, training_args


OUTPUT_DIR = Path("plots/GRU_time_shuffle/segments")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS,
                        help="Generate one seed only; default generates all ten seeds.")
    return parser.parse_args()


def generate(seed):
    model_args = training_args("GRU", "H32_WD0.01", seed)
    X, _, metadata = load_model_data("GRU", model_args)
    if X.shape[1] != 40:
        raise ValueError(f"Expected 40 TRs per segment, found {X.shape[1]}.")
    participants = metadata["participant"].astype(str)
    unique_participants = np.unique(participants)
    folds = {participant: fold for fold, participant in enumerate(unique_participants, start=1)}
    rng = np.random.default_rng(seed)
    permutations = np.argsort(rng.random((len(metadata), X.shape[1])), axis=1)
    output = pd.DataFrame({
        "segment_index": metadata.index.to_numpy(),
        "fold": participants.map(folds).to_numpy(),
        "participant": metadata["participant"].to_numpy(),
        "run": metadata["run"].to_numpy(),
        "permuted_TR_indices": [json.dumps(row.tolist()) for row in permutations],
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"seed={seed}.csv"
    output.to_csv(path, index=False)
    print(f"Saved {len(output)} segment permutations to {path}.")


if __name__ == "__main__":
    args = parse_args()
    for selected_seed in (SEEDS if args.seed is None else (args.seed,)):
        generate(selected_seed)
