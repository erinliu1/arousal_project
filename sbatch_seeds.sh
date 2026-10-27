#!/bin/bash
#SBATCH --job-name=fmri_classifier_seeds
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-17
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/logs/classifier_seeds_%A_%a.out
#SBATCH --error=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/logs/classifier_seeds_%A_%a.err

set -euo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project

conda_init() {
    source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
}
conda_init
conda activate pytorch_env
cd "$PROJECT_DIR"

# Only combinations not already recovered from the overwritten outputs.
ANALYSES=(lopo lopo lopo  lopo lopo lopo  lopo lopo lopo  lopo lopo lopo  loro loro loro  loro loro loro)
G_MAXS=(   4    4    4     4    4    4     5    5    5     4    4    4     4    4    4     5    5    5)
WEIGHT_DECAYS=(1e-3 1e-3 1e-3  1e-2 1e-2 1e-2  1e-3 1e-3 1e-3  1e-4 1e-4 1e-4  1e-3 1e-3 1e-3  1e-4 1e-4 1e-4)
SEEDS=(         123  456 2026    123  456 2026    123  456  789    123  789 2026    123  456  789    123  456  789)

TASK_ID=$SLURM_ARRAY_TASK_ID
ANALYSIS=${ANALYSES[$TASK_ID]}
G_MAX=${G_MAXS[$TASK_ID]}
WD=${WEIGHT_DECAYS[$TASK_ID]}
SEED=${SEEDS[$TASK_ID]}
TR=40

echo "Node: $SLURM_NODELIST"
echo "Task: $TASK_ID | Analysis: $ANALYSIS | TR: $TR | SW: $TR | G_MAX: $G_MAX | WD: $WD | Seed: $SEED"

python -u fmri_classifier_exploration2.py \
    --run-training \
    --analysis "$ANALYSIS" \
    --segment-length "$TR" \
    --segment-stride "$TR" \
    --responsive-window-stride 1 \
    --trim-responsive-intervals \
    --responsive-max-bp-gap "$G_MAX" \
    --weight-decay "$WD" \
    --seed "$SEED"
