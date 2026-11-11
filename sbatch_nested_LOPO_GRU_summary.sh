#!/bin/bash
#SBATCH --job-name=nLOPO_GRU_summary
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:15:00
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=logs/nLOPO_GRU_summary_%j.out
#SBATCH --error=logs/nLOPO_GRU_summary_%j.err

# One lightweight job that summarizes all four ROI groups sequentially.
set -euo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
ROI_GROUPS=(all_rois aan_rois cortical_rois thalamic_rois)

source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for ROI_GROUP in "${ROI_GROUPS[@]}"; do
    echo "Summarizing rois=$ROI_GROUP"
    python -u nested_LOPO_GRU.py --summarize --rois "$ROI_GROUP"
done
