#!/bin/bash
# Submit inner CV, then outer evaluation + summaries after all inner jobs succeed.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
INNER_JOB=$(sbatch --parsable sbatch_LR_mean_sd_all_rois_inner.sh)
INNER_JOB=${INNER_JOB%%;*}
EVAL_JOB=$(sbatch --parsable --dependency="afterok:$INNER_JOB" sbatch_LR_mean_sd_all_rois_eval.sh)
printf 'Inner array: %s\nEval + summary array: %s\n' "$INNER_JOB" "$EVAL_JOB"
