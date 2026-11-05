#!/bin/bash
#SBATCH --job-name=MLP_NESTED_LOPO
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=logs/MLP_NESTED_LOPO_%A_%a.out
#SBATCH --error=logs/MLP_NESTED_LOPO_%A_%a.err

# Run one participant per task; run_MLP.py handles its four configurations and ten seeds sequentially:
#   mkdir -p logs
#   sbatch sbatch_nested_LOPO.sh
# After the array finishes: python nested_LOPO.py --model MLP --summarize

set -eo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
PARTICIPANTS=(AAN_01 AAN_02 AAN_03 AAN_04 AAN_05 AAN_06 AAN_07 AAN_08 AAN_09 AAN_10 AAN_11 AAN_12 AAN_14 AAN_15 AAN_16 AAN_17 AAN_18 AAN_19 AAN_20 AAN_21)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit with sbatch sbatch_nested_LOPO.sh}

PARTICIPANT=${PARTICIPANTS[$TASK_ID]}

conda_init() { source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh; }
conda_init
conda activate pytorch_env
set -u
cd "$PROJECT_DIR"

echo "Task $TASK_ID | held out=$PARTICIPANT | 40 sequential MLP runs"
python -u run_MLP.py --participant "$PARTICIPANT"
