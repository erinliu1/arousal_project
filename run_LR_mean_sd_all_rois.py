#!/usr/bin/env python3
"""Run mean+SD nested LOPO for the six remaining canonical ROI groups (All ROIs already completed), without seeds.

Sequential complete workflow: python run_LR_mean_sd_all_rois.py
Single Slurm task: --stage inner --task-id 0 (0..119), or --stage eval --task-id 0 (0..5).
Eval tasks also generate participant-bootstrap summaries.
"""
import argparse
from pathlib import Path
import subprocess
import sys

PARTICIPANTS = tuple(f"AAN_{i:02d}" for i in range(1, 22) if i != 13)
ROI_GROUPS = (
    ("aan_rois",), ("cortical_rois",), ("thalamic_rois",),
    ("aan_rois", "cortical_rois"), ("aan_rois", "thalamic_rois"),
    ("thalamic_rois", "cortical_rois"),
)


def run(group, *arguments):
    command = [sys.executable, "-u", str(Path(__file__).with_name("LR.py")),
               "--features", "mean_sd", "--rois", *group, *arguments]
    print("Running: " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "inner", "eval", "summary"), default="all")
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()
    limit = 120 if args.stage == "inner" else 7
    if args.task_id is not None and (args.stage == "all" or not 0 <= args.task_id < limit):
        parser.error("--task-id requires inner (0..119), eval or summary (0..5)")
    if args.stage in ("all", "inner"):
        tasks = range(120) if args.task_id is None else (args.task_id,)
        for task in tasks:
            group, outer = divmod(task, len(PARTICIPANTS))
            run(ROI_GROUPS[group], "--run_inner", "--outer-held-out-participant", PARTICIPANTS[outer])
    if args.stage in ("all", "eval", "summary"):
        groups = range(6) if args.task_id is None else (args.task_id,)
        for index in groups:
            if args.stage != "summary":
                run(ROI_GROUPS[index], "--eval")
            run(ROI_GROUPS[index], "--summarize")


if __name__ == "__main__":
    main()
