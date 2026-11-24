#!/bin/bash
#SBATCH --job-name=nLOPO_pruning_round3
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-199
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=logs/nLOPO_pruning_round3_%A_%a.out
#SBATCH --error=logs/nLOPO_pruning_round3_%A_%a.err

# mkdir -p logs && sbatch sbatch_nested_LOPO_pruning_round3.sh
# 200 tasks: 20 outer participants x 10 seeds. Each writes 29 rows (551 fold fits).
# Each outer fold starts from its own sequential best round-1 and round-2 exclusions (29-ROI baseline).
# Requires all ten seeds from rounds 1 and 2 per outer fold.
# Fixed HD64, canonical nearest-snapped segments, CPU only, no concurrency cap.
set -euo pipefail
PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
PARTICIPANTS=(AAN_01 AAN_02 AAN_03 AAN_04 AAN_05 AAN_06 AAN_07 AAN_08 AAN_09 AAN_10 AAN_11 AAN_12 AAN_14 AAN_15 AAN_16 AAN_17 AAN_18 AAN_19 AAN_20 AAN_21)
SEEDS=(42 123 456 789 2026 12 314 258 1024 2048)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit with sbatch}
if (( TASK_ID < 0 || TASK_ID > 199 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 199" >&2
    exit 1
fi
OUTER=${PARTICIPANTS[$((TASK_ID / 10))]}
SEED=${SEEDS[$((TASK_ID % 10))]}
source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
python -u nested_LOPO_pruning.py --run_pruning --round 3 --outer-held-out-participant "$OUTER" --seed "$SEED"
