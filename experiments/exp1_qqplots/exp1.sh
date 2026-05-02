#!/bin/bash
# =============================================================================
# exp1.sh — Asymptotic normality experiments for the II estimator
#
# Runs Setup.md Experiments 3 (QQ plots), 4 (histograms), 5 (KS test).
#
# Array layout:  10 distributions × 4 sample sizes = 40 tasks (0–39)
#   rel_idx  = TASK_ID % 10   →  which distribution
#   n_idx    = TASK_ID / 10   →  which sample size (100, 500, 1000, 5000)
#
# Submit from experiments/exp1_qqplots/:
#   sbatch exp1.sh
# =============================================================================

#SBATCH --job-name=ii_normality
#SBATCH --output=logs/ii_normality_%A_%a.out
#SBATCH --error=logs/ii_normality_%A_%a.err
#SBATCH --array=0-39
#SBATCH --time=02:00:00
#SBATCH --mem=12G
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

n_rel=${#relationship_types[@]}   # 10

rel_idx=$(( SLURM_ARRAY_TASK_ID % n_rel ))
n_idx=$(( SLURM_ARRAY_TASK_ID / n_rel ))

relationship=${relationship_types[$rel_idx]}
n_samples=${n_values[$n_idx]}

# B = 1000 for n ≤ 1000; 500 for n = 5000 (keeps wall time < 2 h)
if [ "$n_samples" -le 1000 ]; then
    n_simulations=1000
else
    n_simulations=500
fi

output_file="results/${relationship}_n${n_samples}_task${SLURM_ARRAY_TASK_ID}.pkl"

echo "Distribution : $relationship"
echo "n_samples    : $n_samples"
echo "B            : $n_simulations"
echo "Output       : $output_file"
echo "----------------------------------------"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python exp1_qqplots.py \
    --mode             run            \
    --relationship_type "$relationship" \
    --n_samples        "$n_samples"   \
    --n_simulations    "$n_simulations" \
    --output_file      "$output_file" \
    --dim_x            5            \
    --dim_y            3            \
    --noise_level      0.5            \
    --random_seed      $(( 42 + SLURM_ARRAY_TASK_ID ))

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed successfully at $(date)"
else
    echo "✗ Failed (exit code $EXIT_CODE) at $(date)"
    exit $EXIT_CODE
fi

# ---------------------------------------------------------------------------
# Plotting — triggered by the last array task (task 39 = parabolic, n=5000)
# This is the largest job, most likely to finish last.
# If you need a hard guarantee, submit a separate plotting job with:
#   sbatch --dependency=afterok:$SLURM_ARRAY_JOB_ID plot_only.slurm
# ---------------------------------------------------------------------------
if [ "$SLURM_ARRAY_TASK_ID" -eq 39 ]; then
    echo ""
    echo "Last array task — waiting 90 s for stragglers, then plotting..."
    sleep 90

    python exp1_qqplots.py \
        --mode        plot      \
        --results_dir results/  \
        --plots_dir   plots/

    if [ $? -eq 0 ]; then
        echo "✓ Plots written at $(date)"
    else
        echo "✗ Plotting failed — re-run manually: python exp1_qqplots.py --mode plot"
    fi
fi

echo "========================================"
echo "Job finished at $(date)"
echo "========================================"
