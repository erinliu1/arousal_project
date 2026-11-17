#!/bin/bash
#SBATCH --job-name=GRU_TC_exclude3_CPU
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-9
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=logs/GRU_TC_exclude3_CPU_%A_%a.out
#SBATCH --error=logs/GRU_TC_exclude3_CPU_%A_%a.err

# Submit from the cluster project directory:
#   mkdir -p logs && sbatch sbatch_GRU_exclude_each_ROI.sh
# 10 CPU tasks: fixed exclusions fusiform, AV, CM; one task per seed.
# Base selection is thalamus + cortex (31 ROIs); exclusions leave 28 ROIs.
# Each task runs full outer LOPO, HD=64, WD=0.001, 30 fixed epochs.
set -euo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
SEEDS=(42 123 456 789 2026 12 314 258 1024 2048)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this file with sbatch}
if (( TASK_ID < 0 || TASK_ID > 9 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 9" >&2
    exit 1
fi

# One full outer LOPO run per seed.
SEED=${SEEDS[$TASK_ID]}

source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

EXCLUDED_ROIS=(fusiform AV CM)
# resolve_rois uses canonical ROI order for individual names.
ROI_LABEL="thalamic_rois+cortical_rois__exclude=fusiform+AV+CM"
RESULTS_ROOT="plots/GRU/segments/R(W=30,S=1,N_min=8,G_max=4):NR(T_min=20):Segment(TR=40,SW=40)"
MODEL_DIR="model=GRU,pool=final,hidden=64,drop=0.2,epochs=30,batch=32,LR=1e-3,WD=1e-3"
RUN_DIR="$RESULTS_ROOT/$ROI_LABEL/$MODEL_DIR/LOPO/SEED=$SEED"

if [[ -s "$RUN_DIR/lopo_summary.csv" \
      && -s "$RUN_DIR/lopo_fold_results.csv" \
      && -s "$RUN_DIR/lopo_predictions.csv" ]]; then
    echo "Skipping completed run: $RUN_DIR"
    exit 0
fi

echo "Task $TASK_ID | CPU | thalamus + cortex excluding ${EXCLUDED_ROIS[*]} | seed=$SEED | HD=64 | WD=0.001 | LOPO"
python -u GRU.py --run-training --analysis lopo \
    --rois thalamic_rois cortical_rois --rois-exclude "${EXCLUDED_ROIS[@]}" \
    --responsive-window-size 30 --responsive-window-stride 1 \
    --responsive-min-bp-trs 8 --responsive-max-bp-gap 4 \
    --nonresponsive-min-gap 20 --segment-length 40 --segment-stride 40 \
    --pooling final --hidden-size 64 --weight-decay 0.001 --seed "$SEED" \
    --dropout 0.2 --learning-rate 0.001 --epochs 30 --batch-size 32 \
    --inner_cv_folds 0 --balance participant_then_class
