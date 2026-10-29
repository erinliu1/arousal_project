#!/bin/bash
#SBATCH --job-name=roi_summary_MLP
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-59
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=logs/MLP_%A_%a.out
#SBATCH --error=logs/MLP_%A_%a.err

# From the cluster project directory: mkdir -p logs; sbatch sbatch_MLP.sh
# 2 analyses x 6 lengths x 5 seeds. Trim disabled; length equals stride.
set -euo pipefail

PROJECT_DIR=${MLP_PROJECT_DIR:-/orcd/data/ldlewis/001/om/erinliu/Arousal_Project}
CONDA_INIT=${MLP_CONDA_INIT:-/orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh}
ANALYSES=(lopo loro)
SEGMENT_LENGTHS=(10 20 30 40 50 60)
SEEDS=(42 123 456 789 2026)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit with sbatch sbatch_MLP.sh}
if [[ ! "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID > 59 )); then
    echo "SLURM_ARRAY_TASK_ID must be 0–59" >&2
    exit 1
fi
ANALYSIS=${ANALYSES[$((TASK_ID / 30))]}
TR=${SEGMENT_LENGTHS[$(((TASK_ID / 5) % 6))]}
SEED=${SEEDS[$((TASK_ID % 5))]}

echo "Task $TASK_ID | Analysis: $ANALYSIS | length=stride: $TR | seed: $SEED | trim disabled"
# Preview the entire array without loading conda or training:
# for i in {0..59}; do SLURM_ARRAY_TASK_ID=$i MLP_DRY_RUN=1 bash sbatch_MLP.sh; done
if [[ ${MLP_DRY_RUN:-0} == 1 ]]; then
    printf 'python -u MLP.py --analysis %s --segment-length %s --segment-stride %s --seed %s\n' "$ANALYSIS" "$TR" "$TR" "$SEED"
    exit 0
fi

source "$CONDA_INIT"
conda activate pytorch_env
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MPLBACKEND=Agg

python -u MLP.py \
    --analysis "$ANALYSIS" \
    --segment-length "$TR" \
    --segment-stride "$TR" \
    --seed "$SEED" \
    --responsive-max-bp-gap 4 \
    --hidden-size 32 \
    --dropout 0.2 \
    --epochs 30 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --weight-decay 1e-3 \
    --threads "$OMP_NUM_THREADS"
