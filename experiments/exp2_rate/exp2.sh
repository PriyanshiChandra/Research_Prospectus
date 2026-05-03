#!/bin/bash
# =============================================================================
# exp2.sh — Rate-of-convergence experiment for the II estimator
#
# Runs Setup.md Experiment 2: confirms Var(II_hat_n) ~ sigma^2 / n.
#
# Grid: 10 distributions × 4 sample sizes × 3 d_X values × 2 noise levels
#       = 240 tasks (0–239)
#
# Indexing (outermost to innermost, matching exp2_rate.py exactly):
#   noise_idx = task_id / 120          (2 noise levels)
#   dx_idx    = (task_id % 120) / 40   (3 d_X values)
#   n_idx     = (task_id % 40)  / 10   (4 sample sizes)
#   rel_idx   = task_id % 10           (10 distributions)
#
# Submit from experiments/exp2_rate/:
#   sbatch exp2.sh
# =============================================================================

#SBATCH --job-name=ii_rate
#SBATCH --output=logs/ii_rate_%A_%a.out
#SBATCH --error=logs/ii_rate_%A_%a.err
#SBATCH --array=0-239
#SBATCH --time=03:00:00
#SBATCH --mem=16G
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
n_values=(100 500 1000 5000)
dx_values=(1 2 5)
noise_values=(0.1 0.5)

n_rel=${#relationship_types[@]}   # 10
n_n=${#n_values[@]}               # 4
n_dx=${#dx_values[@]}             # 3
n_noise=${#noise_values[@]}       # 2

# Decode task ID
noise_idx=$(( SLURM_ARRAY_TASK_ID / 120 ))
remainder=$(( SLURM_ARRAY_TASK_ID % 120 ))
dx_idx=$(( remainder / 40 ))
remainder2=$(( remainder % 40 ))
n_idx=$(( remainder2 / 10 ))
rel_idx=$(( remainder2 % 10 ))

relationship=${relationship_types[$rel_idx]}
n_samples=${n_values[$n_idx]}
dim_x=${dx_values[$dx_idx]}
noise=${noise_values[$noise_idx]}

# B = 1000 for all sizes (rate experiment needs stable variance estimates)
n_simulations=1000

# Noise label for filenames (0.1 -> 0-1, 0.5 -> 0-5)
noise_label=$(echo "$noise" | tr '.' '-')
output_file="results/${relationship}_n${n_samples}_dx${dim_x}_noise${noise_label}_task${SLURM_ARRAY_TASK_ID}.pkl"

echo "Distribution : $relationship"
echo "n_samples    : $n_samples"
echo "dim_x        : $dim_x"
echo "noise_level  : $noise"
echo "B            : $n_simulations"
echo "Output       : $output_file"
echo "----------------------------------------"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python exp2_rate.py \
    --mode              run             \
    --relationship_type "$relationship" \
    --n_samples         "$n_samples"    \
    --n_simulations     "$n_simulations" \
    --output_file       "$output_file"  \
    --dim_x             "$dim_x"        \
    --dim_y             3               \
    --noise_level       "$noise"        \
    --random_seed       $(( 42 + SLURM_ARRAY_TASK_ID ))

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed successfully at $(date)"
else
    echo "✗ Failed (exit code $EXIT_CODE) at $(date)"
    exit $EXIT_CODE
fi

# ---------------------------------------------------------------------------
# Plotting — triggered by the last array task (task 239)
# If you need a hard guarantee, submit a separate plotting job with:
#   sbatch --dependency=afterok:$SLURM_ARRAY_JOB_ID plot_only.slurm
# ---------------------------------------------------------------------------
if [ "$SLURM_ARRAY_TASK_ID" -eq 239 ]; then
    echo ""
    echo "Last array task — waiting 120 s for stragglers, then plotting..."
    sleep 120

    python exp2_rate.py \
        --mode        plot      \
        --results_dir results/  \
        --plots_dir   plots/

    if [ $? -eq 0 ]; then
        echo "✓ Plots written at $(date)"
    else
        echo "✗ Plotting failed — re-run manually: python exp2_rate.py --mode plot"
    fi
fi

echo "========================================"
echo "Job finished at $(date)"
echo "========================================"
