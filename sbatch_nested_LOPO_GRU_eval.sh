#!/bin/bash
#SBATCH --job-name=nLOPO_GRU_eval
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=logs/nLOPO_GRU_eval_%A_%a.out
#SBATCH --error=logs/nLOPO_GRU_eval_%A_%a.err

# One task per ROI group. Each task selects participant-specific configurations
# and runs the complete outer LOPO for all ten seeds.
set -euo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
ROI_GROUPS=(all_rois aan_rois cortical_rois thalamic_rois)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this file with sbatch}
if (( TASK_ID < 0 || TASK_ID > 3 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 3" >&2
    exit 1
fi
ROI_GROUP=${ROI_GROUPS[$TASK_ID]}

source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

echo "Task $TASK_ID | outer evaluation | rois=$ROI_GROUP | all seeds"
python -u nested_LOPO_GRU.py --run eval --rois "$ROI_GROUP"
