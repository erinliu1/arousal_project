#!/bin/bash
#SBATCH --job-name=fmri_classifier_trim
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --array=0-19
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/logs/classifier_trim_%A_%a.out
#SBATCH --error=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/logs/classifier_trim_%A_%a.err

set -euo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project

conda_init() {
    source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
}
conda_init
conda activate pytorch_env
cd "$PROJECT_DIR"

SEGMENT_LENGTHS=(10 20 30 50 60)
SEEDS=(123 456 789 2026)

TASK_ID=$SLURM_ARRAY_TASK_ID

SEGMENT_ID=$((TASK_ID / 4))
SEED_ID=$((TASK_ID % 4))

TR=${SEGMENT_LENGTHS[$SEGMENT_ID]}
SEED=${SEEDS[$SEED_ID]}

echo "Node: $SLURM_NODELIST"
echo "Task: $TASK_ID | Analysis: loro | TR: $TR | SW: $TR | Seed: $SEED | Trim: disabled"

python -u fmri_classifier_exploration2.py \
    --run-training \
    --analysis loro \
    --segment-length "$TR" \
    --segment-stride "$TR" \
    --seed "$SEED" \