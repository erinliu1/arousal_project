#!/bin/bash
#SBATCH --job-name=fmri_classifier_trim
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --array=0-7
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

ANALYSES=(lopo loro)
SEEDS=(123 456 789 2026)

TASK_ID=$SLURM_ARRAY_TASK_ID
ANALYSIS=${ANALYSES[$((TASK_ID / 4))]}
SEED=${SEEDS[$((TASK_ID % 4))]}

echo "Node: $SLURM_NODELIST"
echo "Task: $TASK_ID | Analysis: $ANALYSIS | Dropout: 0.4 | Seed: $SEED | Trim: enabled"

python -u fmri_classifier_exploration2.py \
    --run-training \
    --analysis "$ANALYSIS" \
    --dropout 0.4 \
    --trim-responsive-intervals \
    --seed "$SEED"
