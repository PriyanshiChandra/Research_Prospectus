#!/bin/bash
# =============================================================================
# exp6.sh — Statistical power comparison
#
# Two settings:
#   A: d_X=1, d_Y=1 — II vs dCor, HSIC, Pearson, Spearman
#   B: d_X=5, d_Y=3 — II vs dCor, HSIC (multivariate)
#
# Grid: 10 distributions x 6 n_values x 4 noise_levels x 2 settings
#       = 480 tasks (0-479)
#
# Indexing:
#   setting_idx = task_id / 240          (2 settings: 0=A, 1=B)
#   noise_idx   = (task_id % 240) / 60   (4 noise levels)
#   n_idx       = (task_id % 60)  / 10   (6 sample sizes)
#   rel_idx     = task_id % 10           (10 distributions)
#
# Adaptive B and P based on n to keep wall time < 2 h:
#   n <= 200  : B=200, P=199
#   n <= 1000 : B=100, P=99
#   n =  2000 : B=50,  P=99
#
# Submit from experiments/exp6_power/:
#   sbatch exp6.sh
# =============================================================================

#SBATCH --job-name=ii_power
#SBATCH --output=logs/ii_power_%A_%a.out
#SBATCH --error=logs/ii_power_%A_%a.err
#SBATCH --array=0-479
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
unset PYTHONPATH
module purge
module load python/3.11.7-gcc-8.5.0-xhbe4xp

mkdir -p logs results plots

echo "========================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Array task  : $SLURM_ARRAY_TASK_ID"
echo "Node        : $(hostname)"
echo "Started at  : $(date)"
echo "========================================"

source ~/measure_comparison_env/bin/activate

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
relationship_types=(
    "independent"
    "linear"
    "quadratic"
    "cubic"
    "sine"
    "cosine"
    "exponential"
    "logarithmic"
    "step"
    "parabolic"
)
n_values=(50 100 200 500 1000 2000)
noise_values=(0.1 0.5 1.0 2.0)
settings=("A" "B")

# Decode task ID
setting_idx=$(( SLURM_ARRAY_TASK_ID / 240 ))
remainder=$(( SLURM_ARRAY_TASK_ID % 240 ))
noise_idx=$(( remainder / 60 ))
remainder2=$(( remainder % 60 ))
n_idx=$(( remainder2 / 10 ))
rel_idx=$(( remainder2 % 10 ))

setting=${settings[$setting_idx]}
relationship=${relationship_types[$rel_idx]}
n_samples=${n_values[$n_idx]}
noise=${noise_values[$noise_idx]}

# Adaptive B and P — fewer reps at large n to keep wall time < 2 h
if [ "$n_samples" -le 200 ]; then
    B=200
    P=199
elif [ "$n_samples" -le 1000 ]; then
    B=100
    P=99
else
    B=50
    P=99
fi

noise_label=$(echo "$noise" | tr '.' '-')
output_file="results/${relationship}_n${n_samples}_noise${noise_label}_setting${setting}_task${SLURM_ARRAY_TASK_ID}.pkl"

echo "Setting      : $setting"
echo "Distribution : $relationship"
echo "n_samples    : $n_samples"
echo "noise_level  : $noise"
echo "B            : $B"
echo "P            : $P"
echo "Output       : $output_file"
echo "----------------------------------------"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python exp6_power.py \
    --mode              run             \
    --setting           "$setting"      \
    --relationship_type "$relationship" \
    --n_samples         "$n_samples"    \
    --noise_level       "$noise"        \
    --B                 "$B"            \
    --P                 "$P"            \
    --output_file       "$output_file"  \
    --random_seed       $(( 9999 + SLURM_ARRAY_TASK_ID ))

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Completed successfully at $(date)"
else
    echo "FAILED (exit code $EXIT_CODE) at $(date)"
    exit $EXIT_CODE
fi

# ---------------------------------------------------------------------------
# Plotting — triggered by the last array task (task 479)
# ---------------------------------------------------------------------------
if [ "$SLURM_ARRAY_TASK_ID" -eq 479 ]; then
    echo ""
    echo "Last task — waiting 120 s for stragglers, then plotting..."
    sleep 120

    python exp6_power.py \
        --mode        plot      \
        --results_dir results/  \
        --plots_dir   plots/

    if [ $? -eq 0 ]; then
        echo "Plots written at $(date)"
    else
        echo "Plotting failed. Re-run manually:"
        echo "  python exp6_power.py --mode plot"
    fi
fi

echo "========================================"
echo "Job finished at $(date)"
echo "========================================"
