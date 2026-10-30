#!/bin/bash
#SBATCH --job-name=roi_TCN
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --array=0-4
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=logs/TCN_%A_%a.out
#SBATCH --error=logs/TCN_%A_%a.err

# Submit from the cluster project directory: mkdir -p logs; sbatch sbatch_TCN.sh
# Matched GRU comparison: five seeds, LOPO, untrimmed length=stride=40.
set -euo pipefail
PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
SEEDS=(42 123 456 789 2026)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit with sbatch sbatch_TCN.sh}
if [[ ! "$TASK_ID" =~ ^[0-4]$ ]]; then
    echo "SLURM_ARRAY_TASK_ID must be 0–4" >&2
    exit 1
fi
SEED=${SEEDS[$TASK_ID]}
COMMAND=(python -u TCN.py --analysis lopo --seed "$SEED"
    --segment-length 40 --segment-stride 40 --rois all_rois
    --channels 16 --kernel-size 3 --dilations 1 2 4 8
    --dropout 0.2 --epochs 30 --batch-size 32
    --learning-rate 1e-3 --weight-decay 1e-2
    --responsive-window-size 30 --responsive-window-stride 1
    --responsive-min-bp-trs 8 --responsive-max-bp-gap 4 --nonresponsive-min-gap 20
    --threads "${SLURM_CPUS_PER_TASK:-4}")

echo "Task $TASK_ID | LOPO | seed=$SEED | length=stride=40 | channels=16 | WD=1e-2 | epochs=30 | trim disabled"
# Preview: SLURM_ARRAY_TASK_ID=0 TCN_DRY_RUN=1 bash sbatch_TCN.sh
if [[ ${TCN_DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi
source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MPLBACKEND=Agg
"${COMMAND[@]}"
