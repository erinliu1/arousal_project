#!/bin/bash
#SBATCH --job-name=nLOPO_GRU_bootstrap
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:30:00
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=logs/nLOPO_GRU_bootstrap_%j.out
#SBATCH --error=logs/nLOPO_GRU_bootstrap_%j.err

# One lightweight job that bootstraps all three ROI pairs sequentially.
set -euo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
ROI_PAIR_A=(cortical_rois aan_rois aan_rois)
ROI_PAIR_B=(thalamic_rois cortical_rois thalamic_rois)

source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for PAIR_INDEX in "${!ROI_PAIR_A[@]}"; do
    ROI_A=${ROI_PAIR_A[$PAIR_INDEX]}
    ROI_B=${ROI_PAIR_B[$PAIR_INDEX]}
    echo "Bootstrapping rois=$ROI_A+$ROI_B"
    python -u nested_LOPO_GRU.py --bootstrap \
        --rois "$ROI_A" "$ROI_B" \
        --n-bootstrap 10000 \
        --bootstrap-seed 0
done
