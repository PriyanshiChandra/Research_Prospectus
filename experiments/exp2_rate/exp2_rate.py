"""
experiments/exp2_rate/exp2_rate.py
-----------------------------------
Experiment 2 — Rate of convergence of the II estimator.

Goal: confirm that Var(II_hat_n) decays at rate 1/n, i.e. the estimator
converges at the parametric sqrt(n) rate.

Diagnostic plots
----------------
  Plot A — n * Var_hat vs n  (should be flat if rate is sqrt(n))
  Plot B — Var_hat vs 1/n   (should be linear; slope = sigma^2)
  Table  — estimated sigma^2 per (distribution, d_X, noise_level)

Grid
----
  10 distributions x 4 sample sizes x 3 d_X values x 2 noise levels
  = 240 SLURM array tasks

Usage
-----
# Run one task (called by SLURM array):
python exp2_rate.py --mode run \
    --relationship_type linear --n_samples 1000 \
    --dim_x 2 --dim_y 3 --noise_level 0.1 \
    --n_simulations 1000 --output_file results/linear_n1000_dx2_noise0.1.pkl \
    --random_seed 42

# After all tasks finish, generate plots + table:
python exp2_rate.py --mode plot \
    --results_dir results/ --plots_dir plots/
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

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

DIST_LABELS = {
    "independent": "D0 Indep.",
    "linear":      "D1 Linear",
    "quadratic":   "D2 Quad.",
    "cubic":       "D3 Cubic",
    "sine":        "D4 Sine",
    "cosine":      "D5 Cosine",
    "exponential": "D6 Exp.",
    "logarithmic": "D7 Log.",
    "step":        "D8 Step",
    "parabolic":   "D9 Parab.",
}

# Colour cycle — one colour per distribution
COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# Data generation  (reused from exp1 — keep consistent)
# ---------------------------------------------------------------------------

def _x_to_y_elementwise(f_X,  # type: np.ndarray
                         dim_y  # type: int
                         ):
    # type: (...) -> np.ndarray
    n, k = f_X.shape
    k_use = min(k, dim_y)
    out = f_X[:, :k_use]
    if k_use < dim_y:
        out = np.concatenate(
            [out, np.tile(out[:, -1:], (1, dim_y - k_use))], axis=1
        )
    return out


def generate_sample(relationship_type,  # type: str
                    n,                   # type: int
                    dim_x,               # type: int
                    dim_y,               # type: int
                    noise_level,         # type: float
                    rng,
                    linear_A=None,       # type: Optional[np.ndarray]
                    ):
    X = rng.standard_normal((n, dim_x))
    eps = noise_level * rng.standard_normal((n, dim_y))

    if relationship_type == "independent":
        Y = rng.standard_normal((n, dim_y))

    elif relationship_type == "linear":
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
        norm_X = np.linalg.norm(X, axis=1, keepdims=True)
        Y = np.log(norm_X + 1.0) * np.ones((1, dim_y)) + eps

    elif relationship_type == "step":
        Y = _x_to_y_elementwise(np.sign(X), dim_y) + eps

    elif relationship_type == "parabolic":
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


def _make_linear_A(dim_x, dim_y):
    # type: (int, int) -> np.ndarray
    rng_A = np.random.default_rng(999 + dim_x * 100 + dim_y)
    return rng_A.standard_normal((dim_x, dim_y)) / np.sqrt(dim_x)


# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------

def run_experiment(args):
    # type: (argparse.Namespace) -> None
    """
    Run B replications for one (dist, n, d_X, noise) combination.
    Saves ii_values, empirical variance, and n*variance.
    """
    rng = np.random.default_rng(args.random_seed)
    B = args.n_simulations
    n = args.n_samples

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
    emp_var = float(np.var(ii_values, ddof=1))   # Var_hat(II_hat_n)
    n_times_var = n * emp_var                     # should be ~constant in n

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
        "emp_var":           emp_var,
        "n_times_var":       n_times_var,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "wb") as fh:
        pickle.dump(results, fh)

    print(
        f"[done] {args.relationship_type} n={n} d_X={args.dim_x} "
        f"noise={args.noise_level} | II_mean={ii_mean:.4f} "
        f"Var={emp_var:.2e}  n*Var={n_times_var:.4f}"
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _load_results(results_dir):
    # type: (str) -> list
    """Load all pkl files; return list of result dicts."""
    rows = []
    for pkl in sorted(Path(results_dir).glob("*.pkl")):
        with open(pkl, "rb") as fh:
            rows.append(pickle.load(fh))
    return rows


def _pivot(rows, noise_level):
    # type: (list, float) -> dict
    """
    Filter rows by noise_level and organise into:
      data[(dist, dim_x)] = {n: n_times_var, ...}
    Also collect sigma2 estimates (slope of Var vs 1/n).
    """
    subset = [r for r in rows if abs(r["noise_level"] - noise_level) < 1e-9]
    data = {}  # type: Dict
    for r in subset:
        key = (r["relationship_type"], r["dim_x"])
        if key not in data:
            data[key] = {}
        data[key][r["n_samples"]] = r
    return data


def plot_n_times_var(rows,         # type: list
                     plots_dir,    # type: str
                     noise_level,  # type: float
                     dx_values,    # type: List[int]
                     n_values,     # type: List[int]
                     ):
    # type: (...) -> None
    """
    Plot A: n * Var_hat(II_hat_n) vs n.
    One panel per d_X. Flat line confirms sqrt(n) rate.
    Saved as exp2_n_times_var_noise{noise_label}.pdf/png
    """
    data = _pivot(rows, noise_level)
    noise_label = str(noise_level).replace(".", "-")

    ncols = len(dx_values)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5), sharey=False)
    if ncols == 1:
        axes = [axes]

    for col, dx in enumerate(dx_values):
        ax = axes[col]
        for ci, dist in enumerate(DISTRIBUTIONS):
            key = (dist, dx)
            if key not in data:
                continue
            ns = sorted(data[key].keys())
            nv = [data[key][n]["n_times_var"] for n in ns]
            ax.plot(ns, nv, marker="o", ms=5, lw=1.5,
                    color=COLOURS[ci], label=DIST_LABELS[dist])

        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.set_title(f"$d_X = {dx}$", fontsize=11)
        ax.set_xlabel("$n$", fontsize=10)
        ax.set_ylabel("$n \\cdot \\widehat{{\\mathrm{{Var}}}}(\\widehat{{\\mathrm{{II}}}}_n)$",
                      fontsize=10)
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        if col == ncols - 1:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

    fig.suptitle(
        f"Exp 2 — $n \\cdot \\widehat{{\\mathrm{{Var}}}}(\\widehat{{\\mathrm{{II}}}}_n)$ vs $n$"
        f"  ($\\sigma_\\varepsilon = {noise_level}$)\n"
        "Flat line confirms $\\sqrt{{n}}$ convergence rate",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(plots_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(plots_dir, f"exp2_n_times_var_noise{noise_label}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"Saved: {out}")
    plt.close(fig)


def plot_var_vs_inv_n(rows,         # type: list
                      plots_dir,    # type: str
                      noise_level,  # type: float
                      dx_values,    # type: List[int]
                      n_values,     # type: List[int]
                      ):
    # type: (...) -> None
    """
    Plot B: Var_hat(II_hat_n) vs 1/n.
    Linear relationship confirms sqrt(n) rate.
    Slope of fitted line = sigma^2 estimate.
    """
    data = _pivot(rows, noise_level)
    noise_label = str(noise_level).replace(".", "-")

    ncols = len(dx_values)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5), sharey=False)
    if ncols == 1:
        axes = [axes]

    for col, dx in enumerate(dx_values):
        ax = axes[col]
        for ci, dist in enumerate(DISTRIBUTIONS):
            key = (dist, dx)
            if key not in data:
                continue
            ns   = sorted(data[key].keys())
            invn = [1.0 / n for n in ns]
            vrs  = [data[key][n]["emp_var"] for n in ns]
            ax.plot(invn, vrs, marker="o", ms=5, lw=1.5,
                    color=COLOURS[ci], label=DIST_LABELS[dist])
            # fitted line
            if len(invn) >= 2:
                slope, intercept, *_ = linregress(invn, vrs)
                x_fit = np.linspace(0, max(invn) * 1.05, 100)
                ax.plot(x_fit, slope * x_fit + intercept,
                        color=COLOURS[ci], lw=0.8, ls="--", alpha=0.6)

        ax.set_title(f"$d_X = {dx}$", fontsize=11)
        ax.set_xlabel("$1/n$", fontsize=10)
        ax.set_ylabel("$\\widehat{{\\mathrm{{Var}}}}(\\widehat{{\\mathrm{{II}}}}_n)$",
                      fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        if col == ncols - 1:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.85)

    fig.suptitle(
        f"Exp 2 — $\\widehat{{\\mathrm{{Var}}}}(\\widehat{{\\mathrm{{II}}}}_n)$ vs $1/n$"
        f"  ($\\sigma_\\varepsilon = {noise_level}$)\n"
        "Linear fit (dashed) — slope estimates $\\sigma^2$",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(plots_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(plots_dir, f"exp2_var_vs_invn_noise{noise_label}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"Saved: {out}")
    plt.close(fig)


def export_sigma2_table(rows,       # type: list
                         plots_dir,  # type: str
                         dx_values,  # type: List[int]
                         n_values,   # type: List[int]
                         ):
    # type: (...) -> None
    """
    Estimate sigma^2 = slope of Var_hat vs 1/n for each
    (distribution, d_X, noise_level) triple.
    Export as CSV and LaTeX.
    """
    noise_levels = sorted({r["noise_level"] for r in rows})
    table_rows = []

    for noise in noise_levels:
        data = _pivot(rows, noise)
        for dist in DISTRIBUTIONS:
            for dx in dx_values:
                key = (dist, dx)
                if key not in data:
                    continue
                ns   = sorted(data[key].keys())
                if len(ns) < 2:
                    continue
                invn = np.array([1.0 / n for n in ns])
                vrs  = np.array([data[key][n]["emp_var"] for n in ns])
                slope, intercept, r_val, *_ = linregress(invn, vrs)
                table_rows.append({
                    "Distribution":  DIST_LABELS[dist],
                    "d_X":           dx,
                    "noise":         noise,
                    "sigma2_est":    round(slope, 6),
                    "intercept":     round(intercept, 6),
                    "R2":            round(r_val ** 2, 4),
                    "n_times_var_mean": round(
                        float(np.mean([data[key][n]["n_times_var"] for n in ns])), 4
                    ),
                })

    df = pd.DataFrame(table_rows)
    os.makedirs(plots_dir, exist_ok=True)

    out_csv = os.path.join(plots_dir, "exp2_sigma2_table.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df.to_string(index=False))

    out_tex = os.path.join(plots_dir, "exp2_sigma2_table.tex")
    with open(out_tex, "w") as fh:
        fh.write("% Exp 2 — Estimated asymptotic variance sigma^2\n")
        fh.write("% sigma2_est = slope of Var_hat vs 1/n linear fit\n\n")
        fh.write(df.to_latex(index=False, float_format="%.6f"))
    print(f"Saved: {out_tex}")


def plot_mode(args):
    # type: (argparse.Namespace) -> None
    rows = _load_results(args.results_dir)
    if not rows:
        print(f"No .pkl files found in {args.results_dir!r}")
        return

    n_values  = sorted({r["n_samples"]  for r in rows})
    dx_values = sorted({r["dim_x"]      for r in rows})
    noises    = sorted({r["noise_level"] for r in rows})
    print(f"Loaded {len(rows)} result files.")
    print(f"  n values : {n_values}")
    print(f"  d_X      : {dx_values}")
    print(f"  noise    : {noises}")

    for noise in noises:
        plot_n_times_var(rows, args.plots_dir, noise, dx_values, n_values)
        plot_var_vs_inv_n(rows, args.plots_dir, noise, dx_values, n_values)

    export_sigma2_table(rows, args.plots_dir, dx_values, n_values)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp 2 — Rate of convergence of the II estimator."
    )
    parser.add_argument("--mode", choices=["run", "plot"], required=True)

    # run-mode
    parser.add_argument("--relationship_type", choices=DISTRIBUTIONS)
    parser.add_argument("--n_samples",     type=int)
    parser.add_argument("--n_simulations", type=int,   default=1000)
    parser.add_argument("--output_file",   type=str)
    parser.add_argument("--dim_x",         type=int,   default=2)
    parser.add_argument("--dim_y",         type=int,   default=3)
    parser.add_argument("--noise_level",   type=float, default=0.1)
    parser.add_argument("--random_seed",   type=int,   default=42)

    # plot-mode
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
