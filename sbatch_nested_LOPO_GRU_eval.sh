#!/bin/bash
#SBATCH --job-name=nLOPO_GRU_eval_nearest
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --array=0-69
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=logs/nLOPO_GRU_eval_nearest_%A_%a.out
#SBATCH --error=logs/nLOPO_GRU_eval_nearest_%A_%a.err

# First generate rankings: python nested_LOPO_GRU_new.py --get_best_starting_config
# Submit: mkdir -p logs && sbatch sbatch_nested_LOPO_GRU_eval.sh
# 70 CPU tasks: seven ROI selections x ten seeds; no concurrency cap.
# Each runs full outer LOPO using the saved best WD for its ROI/participant.
set -euo pipefail
PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
ROI_SELECTIONS=(
    "all_rois"
    "aan_rois"
    "cortical_rois"
    "thalamic_rois"
    "aan_rois cortical_rois"
    "aan_rois thalamic_rois"
    "thalamic_rois cortical_rois"
)
SEEDS=(42 123 456 789 2026 12 314 258 1024 2048)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit with sbatch}
if (( TASK_ID < 0 || TASK_ID > 69 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 69" >&2
    exit 1
fi
ROI_INDEX=$((TASK_ID / 10))
SEED=${SEEDS[$((TASK_ID % 10))]}
read -r -a ROIS <<< "${ROI_SELECTIONS[$ROI_INDEX]}"
source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

echo "Task $TASK_ID | CPU outer evaluation | rois=${ROIS[*]} | seed=$SEED"
python -u nested_LOPO_GRU_new.py --eval --rois "${ROIS[@]}" --seed "$SEED"
