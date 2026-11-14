#!/bin/bash
#SBATCH --job-name=nLOPO_GRU_eval
#SBATCH --partition=ou_bcs_normal
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=logs/nLOPO_GRU_eval_%A_%a.out
#SBATCH --error=logs/nLOPO_GRU_eval_%A_%a.err

# One GPU task per ROI pair. Each task selects participant-specific configurations
# and runs the complete outer LOPO for all ten seeds on its allocated GPU.
set -euo pipefail

PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
ROI_PAIR_A=(cortical_rois aan_rois aan_rois)
ROI_PAIR_B=(thalamic_rois cortical_rois thalamic_rois)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this file with sbatch}
if (( TASK_ID < 0 || TASK_ID > 2 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 2" >&2
    exit 1
fi
ROI_A=${ROI_PAIR_A[$TASK_ID]}
ROI_B=${ROI_PAIR_B[$TASK_ID]}

source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
cd "$PROJECT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_DYNAMIC=FALSE

python -u - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in pytorch_env; GPU evaluation cannot start.")
device = torch.device("cuda:0")
torch.zeros(1, device=device)
print(f"Using GPU: {torch.cuda.get_device_name(device)}", flush=True)
PY

echo "Task $TASK_ID | GPU outer evaluation | rois=$ROI_A+$ROI_B | all seeds"
python -u nested_LOPO_GRU.py --run eval --rois "$ROI_A" "$ROI_B"
