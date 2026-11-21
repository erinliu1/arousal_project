#!/bin/bash
#SBATCH --job-name=nLOPO_GRU_HD32_nearest
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --array=0-59
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=logs/nLOPO_GRU_HD32_nearest_%A_%a.out
#SBATCH --error=logs/nLOPO_GRU_HD32_nearest_%A_%a.err

# All 42 ROIs, nearest-TR snapping.
# 60 CPU tasks: 20 held-out participants x 3 weight decays; no concurrency cap.
# Each task runs all ten seeds for hidden size 32.
set -euo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
PARTICIPANTS=(AAN_01 AAN_02 AAN_03 AAN_04 AAN_05 AAN_06 AAN_07 AAN_08 AAN_09 AAN_10 AAN_11 AAN_12 AAN_14 AAN_15 AAN_16 AAN_17 AAN_18 AAN_19 AAN_20 AAN_21)
WEIGHT_DECAYS=(1e-2 3e-3 1e-3)
HIDDEN=32

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this file with sbatch}
if (( TASK_ID < 0 || TASK_ID > 59 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 59" >&2
    exit 1
fi

WD_INDEX=$((TASK_ID % 3))
PARTICIPANT_INDEX=$((TASK_ID / 3))
WD=${WEIGHT_DECAYS[$WD_INDEX]}
HELD_OUT=${PARTICIPANTS[$PARTICIPANT_INDEX]}

source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

echo "Task $TASK_ID | rois=all_rois | BP=nearest | held_out=$HELD_OUT | hidden=$HIDDEN | WD=$WD | all seeds"
python -u nested_LOPO_GRU.py --run inner_lopo \
    --rois all_rois --bp_TR_snap nearest \
    --held-out-participant "$HELD_OUT" \
    --hidden-size "$HIDDEN" \
    --weight-decay "$WD"
