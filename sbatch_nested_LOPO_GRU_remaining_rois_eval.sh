#!/bin/bash
#SBATCH --job-name=nLOPO_GRU_remaining_eval
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --array=0-49
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=logs/nLOPO_GRU_remaining_eval_%A_%a.out
#SBATCH --error=logs/nLOPO_GRU_remaining_eval_%A_%a.err

# Submit after all 200 remaining-ROI inner jobs finish.
# 50 CPU jobs: five ROI groups x ten seeds; select across all nine HD/WD configurations.
set -euo pipefail
PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
ROI_SELECTIONS=("aan_rois" "cortical_rois" "thalamic_rois" "aan_rois cortical_rois" "aan_rois thalamic_rois")
SEEDS=(42 123 456 789 2026 12 314 258 1024 2048)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit with sbatch}
if (( TASK_ID < 0 || TASK_ID > 49 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 49" >&2
    exit 1
fi
SEED=${SEEDS[$((TASK_ID % 10))]}
read -r -a ROIS <<< "${ROI_SELECTIONS[$((TASK_ID / 10))]}"
ROI_KEY=${ROI_SELECTIONS[$((TASK_ID / 10))]// /+}
source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

# Serialize ranking writes across array tasks. Global/pruning rankings remain untouched.
flock "plots/GRU/nested_LOPO,BP=nearest/.${ROI_KEY}_ranking.lock" \
    python -u nested_LOPO_GRU_new.py --get_best_starting_config --rois "${ROIS[@]}"
python -u nested_LOPO_GRU_new.py --eval --rois "${ROIS[@]}" --seed "$SEED"
