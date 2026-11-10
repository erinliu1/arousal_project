#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --array=0-7
#SBATCH --mail-user=erinliu@mit.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=logs/GRU_LOPO_%A_%a.out
#SBATCH --error=logs/GRU_LOPO_%A_%a.err

# Submit: mkdir -p logs && sbatch sbatch.sh
# Eight jobs: one remaining hidden-size/weight-decay configuration per job.
# Each job runs all 3 ROI groups x all 10 seeds, skipping completed runs.
# The previously run hidden=64, WD=1e-2 configuration is omitted.
# All other training settings stay fixed.
# Results: plots/GRU/segments/R(W=30,S=1,N_min=8,G_max=4):NR(T_min=20):Segment(TR=40,SW=40)/
#   <roi subgroup>/model=GRU,pool=final,hidden=<hidden>,drop=<drop>,epochs=30,batch=32,LR=1e-3,WD=<WD>/LOPO/SEED=<seed>/
PROJECT_DIR=/orcd/data/ldlewis/001/om/erinliu/Arousal_Project
SEEDS=(42 123 456 789 2026 12 314 258 1024 2048)
ROI_GROUPS=(aan_rois cortical_rois thalamic_rois)
DROPOUT=0.2
CONFIG_HIDDEN=(16 16 16 32 32 32 64 64)
CONFIG_WD=(1e-2 3e-3 1e-3 1e-2 3e-3 1e-3 3e-3 1e-3)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit with sbatch sbatch.sh}
if (( TASK_ID < 0 || TASK_ID > 7 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 7" >&2
    exit 1
fi
HIDDEN=${CONFIG_HIDDEN[$TASK_ID]}
WD=${CONFIG_WD[$TASK_ID]}

source /orcd/data/ldlewis/001/om/erinliu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
PY

RESULTS_ROOT="plots/GRU/segments/R(W=30,S=1,N_min=8,G_max=4):NR(T_min=20):Segment(TR=40,SW=40)"
MODEL_DIR="model=GRU,pool=final,hidden=$HIDDEN,drop=$DROPOUT,epochs=30,batch=32,LR=1e-3,WD=$WD"
for ROI_GROUP in "${ROI_GROUPS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_DIR="$RESULTS_ROOT/$ROI_GROUP/$MODEL_DIR/LOPO/SEED=$SEED"
        # These files are written after all folds finish; a folder alone is insufficient.
        if [[ -s "$RUN_DIR/lopo_summary.csv" && -s "$RUN_DIR/lopo_fold_results.csv" && -s "$RUN_DIR/lopo_predictions.csv" ]]; then
            echo "Skipping completed run: $RUN_DIR"
            continue
        fi
        echo "Task $TASK_ID | seed=$SEED | rois=$ROI_GROUP | hidden=$HIDDEN | drop=$DROPOUT | WD=$WD | LOPO"
        python -u GRU.py --run-training --analysis lopo --rois "$ROI_GROUP" \
            --responsive-window-size 30 --responsive-window-stride 1 \
            --responsive-min-bp-trs 8 --responsive-max-bp-gap 4 \
            --nonresponsive-min-gap 20 --segment-length 40 --segment-stride 40 \
            --pooling final --hidden-size "$HIDDEN" --weight-decay "$WD" --seed "$SEED" \
            --dropout "$DROPOUT" --learning-rate 0.001 --epochs 30 --batch-size 32 \
            --inner_cv_folds 0 --balance participant_then_class
    done
done