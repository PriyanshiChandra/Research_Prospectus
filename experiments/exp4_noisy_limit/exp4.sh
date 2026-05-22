#!/bin/bash
# =============================================================================
# exp4.sh — Estimate empirical noisy limit II*_noisy for each
#           (distribution, d_X, noise) combination.
#
# Motivation: Exp 3 uses II* = 0 for functional distributions (noiseless
# theoretical limit).  With finite noise the true population quantity
# II*_noisy > 0.  This script estimates II*_noisy by running the II
# estimator at large_n = 50000 with B_limit = 10 replications.
#
# Grid: 10 distributions x 4 d_X x 2 noise = 80 tasks (0-79)
#
# Indexing (outermost to innermost):
#   noise_idx = task_id / 40          (2 noise levels)
#   dx_idx    = (task_id % 40) / 10   (4 d_X values)
#   rel_idx   = task_id % 10          (10 distributions)
#
# After all tasks finish, run plot mode:
#   source ~/measure_comparison_env/bin/activate
#   python exp4_noisy_limit.py --mode plot \
#       --limit_dir results_limit/ \
#       --exp3_dir ../exp3_convergence/results/ \
#       --plots_dir plots_corrected/
#
# Submit from experiments/exp4_noisy_limit/:
#   sbatch exp4.sh
# =============================================================================

#SBATCH --job-name=ii_noisy_limit
#SBATCH --output=logs/ii_noisy_limit_%A_%a.out
#SBATCH --error=logs/ii_noisy_limit_%A_%a.err
#SBATCH --array=0-79
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
unset PYTHONPATH
module purge
module load python/3.11.7-gcc-8.5.0-xhbe4xp

mkdir -p logs results_limit plots_corrected

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
dx_values=(1 2 5 10)
noise_values=(0.1 0.5)

# Decode task ID
noise_idx=$(( SLURM_ARRAY_TASK_ID / 40 ))
remainder=$(( SLURM_ARRAY_TASK_ID % 40 ))
dx_idx=$(( remainder / 10 ))
rel_idx=$(( remainder % 10 ))

relationship=${relationship_types[$rel_idx]}
dim_x=${dx_values[$dx_idx]}
noise=${noise_values[$noise_idx]}

noise_label=$(echo "$noise" | tr '.' '-')
output_file="results_limit/limit_${relationship}_dx${dim_x}_noise${noise_label}.pkl"

echo "Distribution : $relationship"
echo "dim_x        : $dim_x"
echo "noise_level  : $noise"
echo "large_n      : 50000"
echo "B_limit      : 10"
echo "Output       : $output_file"
echo "----------------------------------------"

# ---------------------------------------------------------------------------
# Run limit estimation
# ---------------------------------------------------------------------------
python exp4_noisy_limit.py \
    --mode           run_limit       \
    --relationship_type "$relationship" \
    --dim_x          "$dim_x"        \
    --dim_y          3               \
    --noise_level    "$noise"        \
    --large_n        50000           \
    --B_limit        10              \
    --chunk_size     2000            \
    --large_n_threshold 8000        \
    --output_file    "$output_file"  \
    --random_seed    $(( 7777 + SLURM_ARRAY_TASK_ID ))

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Completed successfully at $(date)"
else
    echo "FAILED (exit code $EXIT_CODE) at $(date)"
    exit $EXIT_CODE
fi

# ---------------------------------------------------------------------------
# Plotting — triggered by the last array task (task 79)
# ---------------------------------------------------------------------------
if [ "$SLURM_ARRAY_TASK_ID" -eq 79 ]; then
    echo ""
    echo "Last array task — waiting 60 s for stragglers, then plotting..."
    sleep 60

    python exp4_noisy_limit.py \
        --mode      plot                              \
        --limit_dir results_limit/                    \
        --exp3_dir  ../exp3_convergence/results/      \
        --plots_dir plots_corrected/

    if [ $? -eq 0 ]; then
        echo "Plots written at $(date)"
    else
        echo "Plotting failed. Re-run manually:"
        echo "  python exp4_noisy_limit.py --mode plot \\"
        echo "    --limit_dir results_limit/ \\"
        echo "    --exp3_dir ../exp3_convergence/results/ \\"
        echo "    --plots_dir plots_corrected/"
    fi
fi

echo "========================================"
echo "Job finished at $(date)"
echo "========================================"
