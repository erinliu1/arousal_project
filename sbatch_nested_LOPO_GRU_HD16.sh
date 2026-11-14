#!/bin/bash
#SBATCH --job-name=nLOPO_GRU_HD16
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --array=0-179
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=logs/nLOPO_GRU_HD16_%A_%a.out
#SBATCH --error=logs/nLOPO_GRU_HD16_%A_%a.err

# One task per ROI pair x held-out participant x weight decay.
# Each task runs all ten seeds for hidden size 16.
set -euo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
ROI_PAIR_A=(cortical_rois aan_rois aan_rois)
ROI_PAIR_B=(thalamic_rois cortical_rois thalamic_rois)
PARTICIPANTS=(AAN_01 AAN_02 AAN_03 AAN_04 AAN_05 AAN_06 AAN_07 AAN_08 AAN_09 AAN_10 AAN_11 AAN_12 AAN_14 AAN_15 AAN_16 AAN_17 AAN_18 AAN_19 AAN_20 AAN_21)
WEIGHT_DECAYS=(1e-2 3e-3 1e-3)
HIDDEN=16

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this file with sbatch}
if (( TASK_ID < 0 || TASK_ID > 179 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 179" >&2
    exit 1
fi

WD_INDEX=$((TASK_ID % 3))
PARTICIPANT_INDEX=$((TASK_ID / 3 % 20))
PAIR_INDEX=$((TASK_ID / 60))
WD=${WEIGHT_DECAYS[$WD_INDEX]}
HELD_OUT=${PARTICIPANTS[$PARTICIPANT_INDEX]}
ROI_A=${ROI_PAIR_A[$PAIR_INDEX]}
ROI_B=${ROI_PAIR_B[$PAIR_INDEX]}

source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

echo "Task $TASK_ID | rois=$ROI_A+$ROI_B | held_out=$HELD_OUT | hidden=$HIDDEN | WD=$WD | all seeds"
python -u nested_LOPO_GRU.py --run inner_lopo \
    --rois "$ROI_A" "$ROI_B" \
    --held-out-participant "$HELD_OUT" \
    --hidden-size "$HIDDEN" \
    --weight-decay "$WD"
