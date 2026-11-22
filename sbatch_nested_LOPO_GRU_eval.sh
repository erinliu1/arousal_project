#!/bin/bash
#SBATCH --job-name=nLOPO_GRU_eval_nearest
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --array=0-9
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=logs/nLOPO_GRU_eval_nearest_%A_%a.out
#SBATCH --error=logs/nLOPO_GRU_eval_nearest_%A_%a.err

# Submit after all 200 new inner jobs complete.
# mkdir -p logs && sbatch sbatch_nested_LOPO_GRU_eval.sh
# Ten CPU jobs: all_rois only, one seed each, selecting across all nine HD/WD combinations.
set -euo pipefail
PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
SEEDS=(42 123 456 789 2026 12 314 258 1024 2048)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit with sbatch}
if (( TASK_ID < 0 || TASK_ID > 9 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 9" >&2
    exit 1
fi
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

# Serialize ranking writes across array tasks. Global/pruning rankings remain untouched.
flock "plots/GRU/nested_LOPO,BP=nearest/.all_rois_ranking.lock" \
    python -u nested_LOPO_GRU_new.py --get_best_starting_config --rois all_rois
python -u nested_LOPO_GRU_new.py --eval --rois all_rois --seed "$SEED"
