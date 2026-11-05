"""Run all MLP nested-LOPO configurations and seeds for one participant."""

import argparse
import subprocess
import sys
from pathlib import Path


PARTICIPANTS = (
    "AAN_01", "AAN_02", "AAN_03", "AAN_04", "AAN_05", "AAN_06", "AAN_07",
    "AAN_08", "AAN_09", "AAN_10", "AAN_11", "AAN_12", "AAN_14", "AAN_15",
    "AAN_16", "AAN_17", "AAN_18", "AAN_19", "AAN_20", "AAN_21",
)
CONFIGURATIONS = ("H16_WD0.001", "H16_WD0.01", "H32_WD0.001", "H32_WD0.01")
SEEDS = (42, 123, 456, 789, 2026, 12, 314, 258, 1024, 2048)
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "plots/MLP/nested_LOPO"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True, choices=PARTICIPANTS)
    return parser.parse_args()


def complete(path):
    return path.is_file() and path.stat().st_size > 0 and "segment_pooled_balanced_accuracy" in path.read_text()


def main():
    participant = parse_args().participant
    failed = []
    for configuration in CONFIGURATIONS:
        for seed in SEEDS:
            output = OUTPUT_DIR / participant / configuration / f"seed={seed}.csv"
            if complete(output):
                print(f"Skipping completed {participant} {configuration} seed={seed}", flush=True)
                continue
            print(f"Running held out={participant} | configuration={configuration} | seed={seed}", flush=True)
            command = [sys.executable, "-u", "nested_LOPO.py", "--model", "MLP",
                       "--held-out-participant", participant, "--configuration", configuration,
                       "--seed", str(seed)]
            if subprocess.run(command, cwd=PROJECT_DIR, check=False).returncode:
                failed.append(f"{configuration}/seed={seed}")
    if failed:
        raise RuntimeError(f"Failed runs for {participant}: {', '.join(failed)}")


if __name__ == "__main__":
    main()
