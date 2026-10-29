#!/usr/bin/env python3
#SBATCH --job-name=motion_classifier
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --array=0-11
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=logs/motion_%A_%a.out
#SBATCH --error=logs/motion_%A_%a.err
"""Submit from the cluster project directory: mkdir -p logs; sbatch sbatch_motion.py

12 CPU jobs: LOPO and LORO, each with length=stride in 10,20,30,40,50,60.
Trim is disabled and C is fixed at 0.01 unless explicitly overridden.
Preview all commands locally: python sbatch_motion.py --dry-run
"""

import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess

PROJECT_DIR = Path("/orcd/data/ldlewis/001/om/erinliu/Arousal_Project")
CONDA = Path("/orcd/data/ldlewis/001/om/erinliu/miniconda3/bin/conda")
SEGMENT_LENGTHS = (10, 20, 30, 40, 50, 60)
TASKS = [(analysis, length) for analysis in ("lopo", "loro") for length in SEGMENT_LENGTHS]


def build_command(task_id, args):
    analysis, length = TASKS[task_id]
    command = [
        str(args.conda), "run", "--no-capture-output", "-n", args.conda_env,
        "python", "-u", str(args.project_dir / "motion_classifier.py"),
        "--run-training", "--analysis", analysis,
        "--segment-length", str(length), "--segment-stride", str(length),
        "--C", str(args.c), "--responsive-max-bp-gap", "4",
    ]
    if args.trim_responsive_intervals:
        command.append("--trim-responsive-intervals")
    if args.plot:
        command.append("--plot")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--conda", type=Path, default=CONDA)
    parser.add_argument("--conda-env", default="pytorch_env")
    parser.add_argument("--C", "--c", dest="c", type=float, default=0.01)
    parser.add_argument("--trim-responsive-intervals", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-id", type=int, choices=range(len(TASKS)),
                        help="Override SLURM_ARRAY_TASK_ID for a local single-task run.")
    args = parser.parse_args()
    if not math.isfinite(args.c) or args.c <= 0:
        parser.error("C must be finite and positive.")
    task_id = args.task_id
    if task_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        try:
            task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        except ValueError:
            parser.error("SLURM_ARRAY_TASK_ID must be an integer.")
    if task_id is not None and task_id not in range(len(TASKS)):
        parser.error("SLURM_ARRAY_TASK_ID must be 0–11.")
    if task_id is None and not args.dry_run:
        parser.error("Submit with sbatch, supply --task-id, or preview with --dry-run.")
    task_ids = range(len(TASKS)) if task_id is None else [task_id]
    for index in task_ids:
        command = build_command(index, args)
        print(f"Task {index}: {shlex.join(command)}", flush=True)
        if not args.dry_run:
            env = os.environ.copy()
            threads = env.get("SLURM_CPUS_PER_TASK", "4")
            env.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                       MKL_NUM_THREADS=threads, MPLBACKEND="Agg")
            subprocess.run(command, cwd=args.project_dir, env=env, check=True)


if __name__ == "__main__":
    main()
