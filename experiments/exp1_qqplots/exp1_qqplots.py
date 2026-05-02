"""
experiments/exp1_qqplots/exp1_qqplots.py
-----------------------------------------
Asymptotic normality experiments for the II estimator.

Covers Setup.md Experiments 3, 4, 5:
  Exp 3 — QQ plots of Z_n vs N(0,1)
  Exp 4 — Histograms of Z_n overlaid with N(0,1) density
  Exp 5 — Kolmogorov-Smirnov test for normality

Standardised statistic (Setup.md §4):
    Z_n^(b) = (II_hat^(b) - II_bar_n) / sigma_hat_B
where II_bar_n and sigma_hat_B are the empirical mean and std across B reps.

Usage
-----
# Run one (distribution, n) pair  — called by SLURM array task:
python exp1_qqplots.py --mode run \\
    --relationship_type linear --n_samples 1000 \\
    --n_simulations 1000 --output_file results/linear_n1000.pkl \\
    --dim_x 2 --dim_y 2 --noise_level 0.1 --random_seed 42

# After all array tasks finish, generate figures + KS table:
python exp1_qqplots.py --mode plot \\
    --results_dir results/ --plots_dir plots/
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import  List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest, norm as norm_dist

# ---------------------------------------------------------------------------
# Path setup — import canonical estimator from utils/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from utils.ii_estimator import compute_ii_vectorized  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISTRIBUTIONS = [
    "independent",
    "linear",
    "quadratic",
    "cubic",
    "sine",
    "cosine",
    "exponential",
    "logarithmic",
    "step",
    "parabolic",
]

# Distributions shown in QQ / histogram figures (Setup.md §7 Exp 3 & 4)
PLOT_DISTRIBUTIONS = ["independent", "linear", "logarithmic"]

DIST_LABELS = {
    "independent": "D0 Independent",
    "linear":      "D1 Linear",
    "quadratic":   "D2 Quadratic",
    "cubic":       "D3 Cubic",
    "sine":        "D4 Sine",
    "cosine":      "D5 Cosine",
    "exponential": "D6 Exponential",
    "logarithmic": "D7 Logarithmic",
    "step":        "D8 Step",
    "parabolic":   "D9 Parabolic",
}


# ---------------------------------------------------------------------------
# Data generation  (Setup.md §6)
# ---------------------------------------------------------------------------

def _x_to_y_elementwise(f_X: np.ndarray, dim_y: int) -> np.ndarray:
    """
    Map a transformed X matrix (n, dim_x) to shape (n, dim_y).
    Uses the first min(dim_x, dim_y) columns; tiles the last column if
    dim_x < dim_y so we always return (n, dim_y).
    """
    n, k = f_X.shape
    k_use = min(k, dim_y)
    out = f_X[:, :k_use]
    if k_use < dim_y:
        out = np.concatenate(
            [out, np.tile(out[:, -1:], (1, dim_y - k_use))], axis=1
        )
    return out


def generate_sample(
    relationship_type: str,
    n: int,
    dim_x: int,
    dim_y: int,
    noise_level: float,
    rng: np.random.Generator,
    linear_A=None,  # type: Optional[np.ndarray]
):
    """
    Generate one (X, Y) sample for the given dependency structure.

    X ~ N(0, I_{d_X}) always.
    linear_A must be pre-generated and passed in so it stays fixed
    across all Monte Carlo replications of the linear case.

    Returns
    -------
    X : (n, dim_x)
    Y : (n, dim_y)
    """
    X = rng.standard_normal((n, dim_x))
    eps = noise_level * rng.standard_normal((n, dim_y))

    if relationship_type == "independent":
        # Y independent of X — true II = 1
        Y = rng.standard_normal((n, dim_y))

    elif relationship_type == "linear":
        # Y = AX + eps,  A fixed random (dim_x, dim_y) matrix
        if linear_A is None:
            raise ValueError("Pass linear_A to keep it fixed across replications.")
        Y = X @ linear_A + eps

    elif relationship_type == "quadratic":
        Y = _x_to_y_elementwise(X ** 2, dim_y) + eps

    elif relationship_type == "cubic":
        Y = _x_to_y_elementwise(X ** 3, dim_y) + eps

    elif relationship_type == "sine":
        Y = _x_to_y_elementwise(np.sin(X), dim_y) + eps

    elif relationship_type == "cosine":
        Y = _x_to_y_elementwise(np.cos(X), dim_y) + eps

    elif relationship_type == "exponential":
        Y = _x_to_y_elementwise(np.exp(X / 2), dim_y) + eps

    elif relationship_type == "logarithmic":
        # Scalar function of ||X||, broadcast to dim_y  (Setup.md D7)
        norm_X = np.linalg.norm(X, axis=1, keepdims=True)   # (n, 1)
        Y = np.log(norm_X + 1.0) * np.ones((1, dim_y)) + eps

    elif relationship_type == "step":
        Y = _x_to_y_elementwise(np.sign(X), dim_y) + eps

    elif relationship_type == "parabolic":
        # Y_1 = X_1^2 + X_2^2;  Y_k = X_{k+1}^2 for k > 1 (wraps around)
        comps = []
        if dim_x >= 2:
            comps.append(X[:, 0] ** 2 + X[:, 1] ** 2)
        else:
            comps.append(X[:, 0] ** 2)
        for k in range(1, dim_y):
            xi = (k + 1) % dim_x
            comps.append(X[:, xi] ** 2)
        Y = np.stack(comps, axis=1) + eps

    else:
        raise ValueError(f"Unknown relationship type: {relationship_type!r}")

    return X, Y


def _make_linear_A(dim_x: int, dim_y: int) -> np.ndarray:
    """Fixed random projection matrix for the linear case, seeded deterministically."""
    rng_A = np.random.default_rng(999 + dim_x * 100 + dim_y)
    return rng_A.standard_normal((dim_x, dim_y)) / np.sqrt(dim_x)


# ---------------------------------------------------------------------------
# Run mode — one SLURM array task
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> None:
    """
    Run B Monte Carlo replications of II estimation for one (dist, n) pair.

    Saves a .pkl file with:
      - ii_values   : (B,) raw II estimates
      - z_values    : (B,) standardised statistics Z_n^(b)
      - ks_stat     : KS statistic vs N(0,1)
      - ks_pval     : p-value
      - metadata    : all run parameters
    """
    rng = np.random.default_rng(args.random_seed)
    B   = args.n_simulations
    n   = args.n_samples

    linear_A = _make_linear_A(args.dim_x, args.dim_y) \
        if args.relationship_type == "linear" else None

    ii_values = np.empty(B)
    for b in range(B):
        X, Y = generate_sample(
            args.relationship_type, n,
            args.dim_x, args.dim_y,
            args.noise_level, rng,
            linear_A=linear_A,
        )
        ii_values[b] = compute_ii_vectorized(X, Y)
        if (b + 1) % 200 == 0:
            print(f"  rep {b+1}/{B}  II={ii_values[b]:.4f}", flush=True)

    ii_mean = float(np.mean(ii_values))
    ii_std  = float(np.std(ii_values, ddof=1))

    # Standardise  (Setup.md §4)
    z_values = (ii_values - ii_mean) / ii_std   # Z_n^(b)

    # KS test against N(0,1)
    ks_stat, ks_pval = kstest(z_values, "norm")

    results = {
        "relationship_type": args.relationship_type,
        "n_samples":         n,
        "n_simulations":     B,
        "dim_x":             args.dim_x,
        "dim_y":             args.dim_y,
        "noise_level":       args.noise_level,
        "random_seed":       args.random_seed,
        "ii_values":         ii_values,
        "ii_mean":           ii_mean,
        "ii_std":            ii_std,
        "z_values":          z_values,
        "ks_stat":           float(ks_stat),
        "ks_pval":           float(ks_pval),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "wb") as fh:
        pickle.dump(results, fh)

    print(
        f"[done] {args.relationship_type} n={n} B={B} "
        f"| II_mean={ii_mean:.4f}  II_std={ii_std:.6f} "
        f"| KS={ks_stat:.4f}  p={ks_pval:.4f}"
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _qq_axis(ax: plt.Axes, z: np.ndarray, title: str = "") -> None:
    """Normal QQ plot on ax (Exp 3)."""
    n = len(z)
    p = (np.arange(1, n + 1) - 0.5) / n
    theoretical = norm_dist.ppf(p)
    empirical   = np.sort(z)

    ax.scatter(theoretical, empirical, s=4, alpha=0.55, color="steelblue", zorder=3)

    lim = max(3.5, np.abs(empirical).max() * 1.05)
    ax.plot([-lim, lim], [-lim, lim], color="crimson", lw=1.4, ls="--", zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Theoretical $\\mathcal{N}(0,1)$ quantiles", fontsize=8)
    ax.set_ylabel("Sample quantiles", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)


def _hist_axis(ax: plt.Axes, z: np.ndarray, title: str = "") -> None:
    """Histogram + N(0,1) density overlay on ax (Exp 4)."""
    x_ref = np.linspace(-4.5, 4.5, 300)
    ax.hist(z, bins=40, density=True, alpha=0.55, color="steelblue",
            label="$Z_n$", zorder=2)
    ax.plot(x_ref, norm_dist.pdf(x_ref), color="crimson", lw=1.6,
            label="$\\mathcal{N}(0,1)$", zorder=3)
    ax.set_xlim(-4.5, 4.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("$Z_n$", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)


# ---------------------------------------------------------------------------
# Plot mode — aggregate results and produce figures
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> dict:
    """Load all .pkl files from results_dir; key = (relationship_type, n_samples)."""
    data = {}
    for pkl in sorted(Path(results_dir).glob("*.pkl")):
        with open(pkl, "rb") as fh:
            r = pickle.load(fh)
        data[(r["relationship_type"], r["n_samples"])] = r
    return data


def plot_qq_grid(
    data: dict,
    plots_dir: str,
    n_values,  # type: List[int]
    distributions=PLOT_DISTRIBUTIONS,  # type: List[str]
) -> None:
    """Experiment 3 — QQ plot grid: rows = distributions, cols = n."""
    nrows, ncols = len(distributions), len(n_values)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.8 * ncols, 3.6 * nrows),
        squeeze=False,
    )

    for i, dist in enumerate(distributions):
        for j, n in enumerate(n_values):
            ax = axes[i][j]
            key = (dist, n)
            if key not in data:
                ax.set_visible(False)
                continue
            label = f"{DIST_LABELS[dist]},  n = {n}"
            _qq_axis(ax, data[key]["z_values"], title=label)

    fig.suptitle(
        "Exp 3 — QQ plots of $Z_n$ vs $\\mathcal{N}(0,1)$\n"
        "(diagonal alignment improves with $n$)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(plots_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(plots_dir, f"exp3_qq_plots.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"Saved: {out}")
    plt.close(fig)


def plot_histogram_grid(
    data: dict,
    plots_dir: str,
    n_values,  # type: List[int]
    distributions=PLOT_DISTRIBUTIONS,  # type: List[str]
) -> None:
    """Experiment 4 — histogram grid: rows = distributions, cols = n."""
    nrows, ncols = len(distributions), len(n_values)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.8 * ncols, 3.4 * nrows),
        squeeze=False,
    )

    for i, dist in enumerate(distributions):
        for j, n in enumerate(n_values):
            ax = axes[i][j]
            key = (dist, n)
            if key not in data:
                ax.set_visible(False)
                continue
            label = f"{DIST_LABELS[dist]},  n = {n}"
            _hist_axis(ax, data[key]["z_values"], title=label)

    # Single legend from first non-empty axis
    for ax_row in axes:
        for ax in ax_row:
            if ax.get_visible():
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper right",
                           fontsize=9, framealpha=0.9)
                break
        else:
            continue
        break

    fig.suptitle(
        "Exp 4 — Histograms of $Z_n$ vs $\\mathcal{N}(0,1)$ density",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 0.93, 0.96])

    os.makedirs(plots_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(plots_dir, f"exp4_histograms.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"Saved: {out}")
    plt.close(fig)


def export_ks_table(
    data: dict,
    plots_dir: str,
    n_values,  # type: List[int]
) -> None:
    """Experiment 5 — KS test results as CSV + printed table."""
    rows = []
    for dist in DISTRIBUTIONS:
        for n in n_values:
            key = (dist, n)
            if key not in data:
                continue
            r = data[key]
            rows.append({
                "Distribution": DIST_LABELS[dist],
                "n":            n,
                "B":            r["n_simulations"],
                "II_mean":      round(r["ii_mean"], 5),
                "II_std":       round(r["ii_std"],  6),
                "KS_stat":      round(r["ks_stat"], 4),
                "p_value":      round(r["ks_pval"], 4),
            })

    df = pd.DataFrame(rows)
    os.makedirs(plots_dir, exist_ok=True)
    out_csv = os.path.join(plots_dir, "exp5_ks_table.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df.to_string(index=False))

    # Annotated LaTeX table (p-values ≥ 0.05 bold)
    out_tex = os.path.join(plots_dir, "exp5_ks_table.tex")
    with open(out_tex, "w") as fh:
        fh.write(
            "% Exp 5 — KS test for normality of Z_n\n"
            "% p-value ≥ 0.05 indicates failure to reject H0: Z_n ~ N(0,1)\n\n"
        )
        fh.write(df.to_latex(index=False, float_format="%.4f"))
    print(f"Saved: {out_tex}")


def plot_mode(args: argparse.Namespace) -> None:
    data = load_results(args.results_dir)
    if not data:
        print(f"No .pkl files found in {args.results_dir!r}")
        return

    n_values = sorted({k[1] for k in data})
    print(f"Loaded {len(data)} result files.  n values found: {n_values}")

    plot_qq_grid(data, args.plots_dir, n_values)
    plot_histogram_grid(data, args.plots_dir, n_values)
    export_ks_table(data, args.plots_dir, n_values)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asymptotic normality experiments for the II estimator."
    )
    parser.add_argument("--mode", choices=["run", "plot"], required=True,
                        help="'run': one MC task; 'plot': aggregate and plot.")

    # --- run-mode arguments -------------------------------------------------
    parser.add_argument("--relationship_type", choices=DISTRIBUTIONS)
    parser.add_argument("--n_samples",    type=int)
    parser.add_argument("--n_simulations", type=int, default=1000,
                        help="Number of Monte Carlo replications B (default: 1000).")
    parser.add_argument("--output_file",  type=str)
    parser.add_argument("--dim_x",        type=int,   default=2)
    parser.add_argument("--dim_y",        type=int,   default=2)
    parser.add_argument("--noise_level",  type=float, default=0.1)
    parser.add_argument("--random_seed",  type=int,   default=42)

    # --- plot-mode arguments ------------------------------------------------
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--plots_dir",   type=str, default="plots/")

    args = parser.parse_args()

    if args.mode == "run":
        for req in ("relationship_type", "n_samples", "output_file"):
            if getattr(args, req) is None:
                parser.error(f"--{req} is required in --mode run")
        run_experiment(args)
    else:
        plot_mode(args)


if __name__ == "__main__":
    main()
