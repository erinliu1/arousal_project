#!/bin/bash
#SBATCH --job-name=MODEL_BOOTSTRAP
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --array=0-23
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=logs/MODEL_BOOTSTRAP_%A_%a.out
#SBATCH --error=logs/MODEL_BOOTSTRAP_%A_%a.err

# Submit with: mkdir -p logs && sbatch sbatch_bootstrap.sh
# Four models x six lengths: GRU H64, compact MLP H16, LR, and motion-only LR.

set -euo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
MODELS=(GRU MLP LR motion_LR)
LENGTHS=(10 20 30 40 50 60)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit with sbatch sbatch_bootstrap.sh}

if (( TASK_ID < 0 || TASK_ID > 23 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 23" >&2
    exit 1
fi

MODEL=${MODELS[$((TASK_ID / 6))]}
LENGTH=${LENGTHS[$((TASK_ID % 6))]}

# Initialize Conda (the non-interactive equivalent of conda_init) and activate the project environment.
source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"

COMMON=(python -u bootstrap.py --model "$MODEL" --TR "$LENGTH" --SW "$LENGTH"
        --analysis lopo --n-bootstrap 10000 --bootstrap-seed 0)
case "$MODEL" in
    GRU) EXTRA=(--hidden 64 --WD 1e-2 --LR 1e-3 --drop 0.2 --epochs 30 --batch 32 --pool final) ;;
    MLP) EXTRA=(--hidden 16 --WD 1e-2 --LR 1e-3 --drop 0.2 --epochs 10 --batch 32) ;;
    LR|motion_LR) EXTRA=(--C 0.01) ;;
esac

echo "Task $TASK_ID | model=$MODEL | LOPO | length=stride=$LENGTH"
"${COMMON[@]}" "${EXTRA[@]}"
