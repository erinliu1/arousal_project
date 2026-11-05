"""Run the complete motion-classifier nested-LOPO workflow sequentially."""

import subprocess
import sys
from pathlib import Path


PARTICIPANTS = (
    "AAN_01", "AAN_02", "AAN_03", "AAN_04", "AAN_05", "AAN_06", "AAN_07",
    "AAN_08", "AAN_09", "AAN_10", "AAN_11", "AAN_12", "AAN_14", "AAN_15",
    "AAN_16", "AAN_17", "AAN_18", "AAN_19", "AAN_20", "AAN_21",
)
CONFIGURATIONS = ("C0.001", "C0.01", "C0.1", "C1")
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "plots/motion_classifier/nested_LOPO"


def complete(path):
    return path.is_file() and path.stat().st_size > 0 and "segment_pooled_balanced_accuracy" in path.read_text()


def run(*arguments):
    subprocess.run([sys.executable, "-u", *arguments], cwd=PROJECT_DIR, check=True)


def main():
    failed = []
    for participant in PARTICIPANTS:
        for configuration in CONFIGURATIONS:
            output = OUTPUT_DIR / participant / configuration / "result.csv"
            if complete(output):
                print(f"Skipping completed {participant} {configuration}", flush=True)
                continue
            print(f"Running held out={participant} | configuration={configuration}", flush=True)
            command = [sys.executable, "-u", "nested_LOPO.py", "--model", "motion_classifier",
                       "--held-out-participant", participant, "--configuration", configuration]
            if subprocess.run(command, cwd=PROJECT_DIR, check=False).returncode:
                failed.append(f"{participant}/{configuration}")
    if failed:
        raise RuntimeError(f"Failed runs: {', '.join(failed)}")

    print("Selecting the best configuration for each participant", flush=True)
    run("nested_LOPO.py", "--model", "motion_classifier", "--summarize")
    print("Running and summarizing outer evaluations", flush=True)
    run("nested_LOPO.py", "--model", "motion_classifier", "--run", "eval")
    print("Calculating participant-clustered bootstrap confidence intervals", flush=True)
    run("bootstrap.py", "--predictions-folder", "plots/motion_classifier/nested_LOPO/evaluation")


if __name__ == "__main__":
    main()
