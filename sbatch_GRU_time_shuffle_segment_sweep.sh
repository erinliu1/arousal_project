#!/bin/bash
#SBATCH --job-name=GRU_shuffle_segments
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-59
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=logs/GRU_shuffle_segments_%A_%a.out
#SBATCH --error=logs/GRU_shuffle_segments_%A_%a.err

# 60 CPU jobs: six original-cohort segment lengths x ten canonical seeds.
# Fixed HD32, WD0.01, 30 epochs, LR0.001, batch32, dropout0.2.
# Submit: mkdir -p logs && sbatch sbatch_GRU_time_shuffle_segment_sweep.sh
set -euo pipefail
PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
LENGTHS=(10 20 30 40 50 60)
SEEDS=(42 123 456 789 2026 12 314 258 1024 2048)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit with sbatch}
if (( TASK_ID < 0 || TASK_ID > 59 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 59" >&2
    exit 1
fi
LENGTH=${LENGTHS[$((TASK_ID / 10))]}
SEED=${SEEDS[$((TASK_ID % 10))]}
source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
python -u GRU_time_shuffle.py --run_fixed --segment-length "$LENGTH" --seed "$SEED"
