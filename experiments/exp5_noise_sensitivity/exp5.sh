#!/bin/bash
# =============================================================================
# exp5.sh — II estimator vs noise level
#
# Fixes n=5000, d_Y=3 and sweeps noise_level from 0 to 5 across
# 10 distributions and 3 d_X values.
#
# Grid: 10 distributions x 12 noise levels x 3 d_X values = 360 tasks (0-359)
#
# Indexing (outermost to innermost):
#   dx_idx    = task_id / 120          (3 d_X values: 1, 2, 5)
#   noise_idx = (task_id % 120) / 10   (12 noise levels)
#   rel_idx   = task_id % 10           (10 distributions)
#
# Submit from experiments/exp5_noise_sensitivity/:
#   sbatch exp5.sh
# =============================================================================

#SBATCH --job-name=ii_noise_sensitivity
#SBATCH --output=logs/ii_noise_%A_%a.out
#SBATCH --error=logs/ii_noise_%A_%a.err
#SBATCH --array=0-359
#SBATCH --time=00:30:00
#SBATCH --mem=4G
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

# 12 noise levels — dense near 0 to capture the transition,
# then sparser at high noise where the curve is flat
noise_values=(0.0 0.05 0.1 0.2 0.3 0.5 0.75 1.0 1.5 2.0 3.0 5.0)

dx_values=(1 2 5)

# Decode task ID
dx_idx=$(( SLURM_ARRAY_TASK_ID / 120 ))
remainder=$(( SLURM_ARRAY_TASK_ID % 120 ))
noise_idx=$(( remainder / 10 ))
rel_idx=$(( remainder % 10 ))

relationship=${relationship_types[$rel_idx]}
noise=${noise_values[$noise_idx]}
dim_x=${dx_values[$dx_idx]}

noise_label=$(echo "$noise" | tr '.' '-')
output_file="results/${relationship}_noise${noise_label}_dx${dim_x}_task${SLURM_ARRAY_TASK_ID}.pkl"

echo "Distribution : $relationship"
echo "noise_level  : $noise"
echo "dim_x        : $dim_x"
echo "n_samples    : 5000"
echo "B            : 200"
echo "Output       : $output_file"
echo "----------------------------------------"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python exp5_noise_sensitivity.py \
    --mode              run             \
    --relationship_type "$relationship" \
    --noise_level       "$noise"        \
    --n_samples         5000            \
    --n_simulations     200             \
    --dim_x             "$dim_x"        \
    --dim_y             3               \
    --output_file       "$output_file"  \
    --random_seed       $(( 1234 + SLURM_ARRAY_TASK_ID ))

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Completed successfully at $(date)"
else
    echo "FAILED (exit code $EXIT_CODE) at $(date)"
    exit $EXIT_CODE
fi

# ---------------------------------------------------------------------------
# Plotting — triggered by the last array task (task 359)
# ---------------------------------------------------------------------------
if [ "$SLURM_ARRAY_TASK_ID" -eq 359 ]; then
    echo ""
    echo "Last task — waiting 60 s for stragglers, then plotting..."
    sleep 60

    python exp5_noise_sensitivity.py \
        --mode        plot      \
        --results_dir results/  \
        --plots_dir   plots/

    if [ $? -eq 0 ]; then
        echo "Plots written at $(date)"
    else
        echo "Plotting failed. Re-run manually:"
        echo "  python exp5_noise_sensitivity.py --mode plot"
    fi
fi

echo "========================================"
echo "Job finished at $(date)"
echo "========================================"
