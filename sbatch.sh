#!/bin/bash
#SBATCH --job-name=GRU_ROI_CPU
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --array=0-239
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=logs/GRU_ROI_CPU_%A_%a.out
#SBATCH --error=logs/GRU_ROI_CPU_%A_%a.err

# Submit with: mkdir -p logs && sbatch sbatch.sh
#
# The 240 array tasks are:
#   8 remaining hidden-size/weight-decay configurations
#   x 3 ROI groups
#   x 10 model seeds
# Each task runs exactly one configuration/ROI/seed combination on one CPU.
# Hidden=64 with WD=1e-2 is omitted because it was previously run.
# Completed runs are skipped; incomplete runs are retried.
set -euo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
SEEDS=(42 123 456 789 2026 12 314 258 1024 2048)
ROI_GROUPS=(aan_rois cortical_rois thalamic_rois)
CONFIG_HIDDEN=(16 16 16 32 32 32 64 64)
CONFIG_WD=(1e-2 3e-3 1e-3 1e-2 3e-3 1e-3 3e-3 1e-3)
DROPOUT=0.2

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this file with sbatch}
if (( TASK_ID < 0 || TASK_ID > 239 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 239" >&2
    exit 1
fi

# Seed varies fastest, then ROI group, then hidden-size/weight-decay configuration.
SEED_INDEX=$((TASK_ID % 10))
ROI_INDEX=$((TASK_ID / 10 % 3))
CONFIG_INDEX=$((TASK_ID / 30))
SEED=${SEEDS[$SEED_INDEX]}
ROI_GROUP=${ROI_GROUPS[$ROI_INDEX]}
HIDDEN=${CONFIG_HIDDEN[$CONFIG_INDEX]}
WD=${CONFIG_WD[$CONFIG_INDEX]}

source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"

# Prevent each process from creating extra CPU worker threads.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

RESULTS_ROOT="plots/GRU/segments/R(W=30,S=1,N_min=8,G_max=4):NR(T_min=20):Segment(TR=40,SW=40)"
MODEL_DIR="model=GRU,pool=final,hidden=$HIDDEN,drop=$DROPOUT,epochs=30,batch=32,LR=1e-3,WD=$WD"
RUN_DIR="$RESULTS_ROOT/$ROI_GROUP/$MODEL_DIR/LOPO/SEED=$SEED"

# These three final outputs identify a completed run. A partial run is retried.
if [[ -s "$RUN_DIR/lopo_summary.csv" \
      && -s "$RUN_DIR/lopo_fold_results.csv" \
      && -s "$RUN_DIR/lopo_predictions.csv" ]]; then
    echo "Skipping completed run: $RUN_DIR"
    exit 0
fi

echo "Task $TASK_ID | CPU | seed=$SEED | rois=$ROI_GROUP | hidden=$HIDDEN | drop=$DROPOUT | WD=$WD | LOPO"
python -u GRU.py --run-training --analysis lopo --rois "$ROI_GROUP" \
    --responsive-window-size 30 --responsive-window-stride 1 \
    --responsive-min-bp-trs 8 --responsive-max-bp-gap 4 \
    --nonresponsive-min-gap 20 \
    --segment-length 40 --segment-stride 40 \
    --pooling final --hidden-size "$HIDDEN" \
    --weight-decay "$WD" --seed "$SEED" \
    --dropout "$DROPOUT" --learning-rate 0.001 \
    --epochs 30 --batch-size 32 \
    --inner_cv_folds 0 --balance participant_then_class
