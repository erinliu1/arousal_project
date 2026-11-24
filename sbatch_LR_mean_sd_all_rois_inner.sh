#!/bin/bash
#SBATCH --job-name=LR_mean_sd_all_inner
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-119
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=logs/LR_mean_sd_all_inner_%A_%a.out
#SBATCH --error=logs/LR_mean_sd_all_inner_%A_%a.err

# Six remaining ROI combinations (excluding completed All ROIs); mean + population SD; BP=nearest; no seeds.
set -euo pipefail
PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
python -u run_LR_mean_sd_all_rois.py --stage inner --task-id "$SLURM_ARRAY_TASK_ID"
