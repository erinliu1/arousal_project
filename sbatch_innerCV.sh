#!/bin/bash
#SBATCH --job-name=fmri_innerCV
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --array=0-14
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=logs/innerCV_%A_%a.out
#SBATCH --error=logs/innerCV_%A_%a.err

# Submit from the cluster project directory: mkdir -p logs; sbatch sbatch_innerCV.sh
# Three weight decays x five seeds; LOPO; untrimmed length=stride=40;
# five inner participant groups.
set -euo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
SEEDS=(42 123 456 789 2026)
WEIGHT_DECAYS=(1e-3 3e-3 1e-2)
INNER_CV_FOLDS=5
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit with sbatch sbatch_innerCV.sh}
if [[ ! "$TASK_ID" =~ ^([0-9]|1[0-4])$ ]]; then
    echo "SLURM_ARRAY_TASK_ID must be 0–14" >&2
    exit 1
fi
SEED=${SEEDS[$((TASK_ID % 5))]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$((TASK_ID / 5))]}
COMMAND=(python -u fmri_classifier_exploration2.py
    --run-training --analysis lopo
    --segment-length 40 --segment-stride 40
    --inner_cv_folds "$INNER_CV_FOLDS"
    --seed "$SEED" --weight-decay "$WEIGHT_DECAY"
    --responsive-max-bp-gap 4
    --hidden-size 32 --dropout 0.2 --pooling final
    --batch-size 32 --learning-rate 1e-3)

echo "Task $TASK_ID | LOPO | TR=SW=40 | seed=$SEED | innerCV=$INNER_CV_FOLDS | WD=$WEIGHT_DECAY | trim disabled"
# Preview without conda or training: SLURM_ARRAY_TASK_ID=0 INNER_CV_DRY_RUN=1 bash sbatch_innerCV.sh
if [[ ${INNER_CV_DRY_RUN:-0} == 1 ]]; then
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
