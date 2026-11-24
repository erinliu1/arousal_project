#!/bin/bash
#SBATCH --job-name=nLOPO_GRU_remaining_summary
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=logs/nLOPO_GRU_remaining_summary_%A_%a.out
#SBATCH --error=logs/nLOPO_GRU_remaining_summary_%A_%a.err

# Submit after the 50-task remaining-ROI evaluation array completes.
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

for ROI_SELECTION in "aan_rois" "cortical_rois" "thalamic_rois" "aan_rois cortical_rois" "aan_rois thalamic_rois"; do
    read -r -a ROIS <<< "$ROI_SELECTION"
    python -u nested_LOPO_GRU_new.py --summarize --rois "${ROIS[@]}"
done
