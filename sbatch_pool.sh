#!/bin/bash
#SBATCH --job-name=fmri_classifier_pool
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/logs/classifier_pool_%A_%a.out
#SBATCH --error=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/logs/classifier_pool_%A_%a.err

set -euo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project

conda_init() {
    source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
}
conda_init
conda activate pytorch_env
cd "$PROJECT_DIR"

SEEDS=(4 123 456 789 2026)
TASK_ID=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$TASK_ID]}

echo "Node: $SLURM_NODELIST"
echo "Task: $TASK_ID | Analysis: lopo | Pooling: mean | Seed: $SEED | Trim: enabled"

python -u fmri_classifier_exploration2.py \
    --run-training \
    --pooling mean \
    --trim-responsive-intervals \
    --analysis lopo \
    --seed "$SEED"
