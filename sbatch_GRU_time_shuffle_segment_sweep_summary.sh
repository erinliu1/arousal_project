#!/bin/bash
#SBATCH --job-name=GRU_shuffle_sweep_plot
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=logs/GRU_shuffle_sweep_plot_%A_%a.out
#SBATCH --error=logs/GRU_shuffle_sweep_plot_%A_%a.err

# Submit with --dependency=afterok:<training_array_jobid> after the 60-task array.
# Validates the original cohort, bootstraps all six lengths, and updates the sweep PNG.
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
export MPLBACKEND=Agg
python -u analyze_segment_lengths.py --add-time-shuffled
