#!/bin/bash
# =============================================================================
# exp3.sh — Convergence of II_hat_n to population quantity II*
#
# Runs Setup.md Experiment 3: verifies E[II_hat_n] -> II* and measures rate.
# Theoretical rate: n^{-min(1/2, 1/d_X)} * log(n)^{d_X+1+beta}
#
# Grid: 10 distributions × 6 sample sizes × 4 d_X values × 2 noise levels
#       = 480 tasks (0–479)
#
# Indexing (outermost to innermost):
#   noise_idx = task_id / 240          (2 noise levels)
#   dx_idx    = (task_id % 240) / 60   (4 d_X values)
#   n_idx     = (task_id % 60)  / 10   (6 sample sizes)
#   rel_idx   = task_id % 10           (10 distributions)
#
# Submit from experiments/exp3_convergence/:
#   sbatch exp3.sh
# =============================================================================

#SBATCH --job-name=ii_convergence
#SBATCH --output=logs/ii_convergence_%A_%a.out
#SBATCH --error=logs/ii_convergence_%A_%a.err
#SBATCH --array=0-479
#SBATCH --time=06:00:00
#SBATCH --mem=24G
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
n_values=(100 500 1000 5000 10000 30000)
dx_values=(1 2 5 10)
noise_values=(0.1 0.5)

# Decode task ID
noise_idx=$(( SLURM_ARRAY_TASK_ID / 240 ))
remainder=$(( SLURM_ARRAY_TASK_ID % 240 ))
dx_idx=$(( remainder / 60 ))
remainder2=$(( remainder % 60 ))
n_idx=$(( remainder2 / 10 ))
rel_idx=$(( remainder2 % 10 ))

relationship=${relationship_types[$rel_idx]}
n_samples=${n_values[$n_idx]}
dim_x=${dx_values[$dx_idx]}
noise=${noise_values[$noise_idx]}

# Adaptive B: fewer reps at large n to keep wall time < 6 h
if [ "$n_samples" -le 1000 ]; then
    n_simulations=500
elif [ "$n_samples" -le 5000 ]; then
    n_simulations=200
elif [ "$n_samples" -le 10000 ]; then
    n_simulations=100
else
    n_simulations=50
fi

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
python exp3_convergence.py \
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
# Plotting — triggered by the last array task (task 479)
# ---------------------------------------------------------------------------
if [ "$SLURM_ARRAY_TASK_ID" -eq 479 ]; then
    echo ""
    echo "Last array task — waiting 120 s for stragglers, then plotting..."
    sleep 120

    python exp3_convergence.py \
        --mode        plot      \
        --results_dir results/  \
        --plots_dir   plots/

    if [ $? -eq 0 ]; then
        echo "✓ Plots written at $(date)"
    else
        echo "✗ Plotting failed — re-run manually: python exp3_convergence.py --mode plot"
    fi
fi

echo "========================================"
echo "Job finished at $(date)"
echo "========================================"
