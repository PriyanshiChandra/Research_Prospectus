"""
experiments/exp3_convergence/exp3_convergence.py
-------------------------------------------------
Experiment 3 — Convergence of II_hat_n to the population quantity II*.

Goal: verify empirically that E[II_hat_n] -> II* and measure the rate.

Theoretical rate (from draft, Section 3):
    |E[II_hat_n] - II*| = O(n^{-min(1/2, 1/d_X)} * log(n)^{d_X+1+beta})

True population values:
    II* = 1  for independent (D0)
    II* = 0  for all functional relationships (D1-D9)
    (exact: 0 is only achieved in the noiseless limit; with noise the limit
     is slightly above 0, but we use 0 as the theoretical benchmark following
     the draft's treatment.)

Diagnostic plots
----------------
  Plot A — Error |mean(II_hat_n) - II*| vs n  (log-log)
            One panel per d_X; reference slope = -min(1/2, 1/d_X)
  Plot B — Convergence: mean(II_hat_n) vs n with II* as horizontal line
            One panel per distribution, lines coloured by d_X
  Table  — Empirical slope of log(error) vs log(n) per (dist, d_X, noise)

Grid
----
  10 distributions x 6 sample sizes x 4 d_X values x 2 noise levels
  = 480 SLURM array tasks

Usage
-----
# Run one task (called by SLURM array):
python exp3_convergence.py --mode run \
    --relationship_type linear --n_samples 1000 \
    --dim_x 2 --dim_y 3 --noise_level 0.1 \
    --n_simulations 500 --output_file results/linear_n1000_dx2_noise0.1.pkl \
    --random_seed 42

# After all tasks finish, generate plots + table:
python exp3_convergence.py --mode plot \
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

# True population II*: 1 for independent, 0 for all functional relationships
TRUE_II = {
    "independent": 1.0,
    "linear":      0.0,
    "quadratic":   0.0,
    "cubic":       0.0,
    "sine":        0.0,
    "cosine":      0.0,
    "exponential": 0.0,
    "logarithmic": 0.0,
    "step":        0.0,
    "parabolic":   0.0,
}

# Theoretical convergence rate exponent: -min(1/2, 1/d_X)
def _ref_slope(dim_x):
    # type: (int) -> float
    return -min(0.5, 1.0 / dim_x)

COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

DX_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # one per d_X value


# ---------------------------------------------------------------------------
# Data generation  (identical to exp1 and exp2 — keep consistent)
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
            raise ValueError("Pass linear_A for the linear relationship.")
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
        raise ValueError("Unknown relationship type: {!r}".format(relationship_type))

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
    Saves ii_values, mean, std, true II*, and error = |mean - II*|.
    """
    rng = np.random.default_rng(args.random_seed)
    B = args.n_simulations
    n = args.n_samples
    true_ii = TRUE_II[args.relationship_type]

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
        if (b + 1) % 100 == 0:
            print("  rep {}/{} II={:.4f}".format(b + 1, B, ii_values[b]),
                  flush=True)

    ii_mean = float(np.mean(ii_values))
    ii_std  = float(np.std(ii_values, ddof=1))
    error   = abs(ii_mean - true_ii)

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
        "true_ii":           true_ii,
        "error":             error,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "wb") as fh:
        pickle.dump(results, fh)

    print("[done] {} n={} d_X={} noise={} | "
          "II_mean={:.4f} II*={:.1f} error={:.4f}".format(
              args.relationship_type, n, args.dim_x,
              args.noise_level, ii_mean, true_ii, error))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _load_results(results_dir):
    # type: (str) -> list
    rows = []
    for pkl in sorted(Path(results_dir).glob("*.pkl")):
        with open(pkl, "rb") as fh:
            rows.append(pickle.load(fh))
    return rows


def _pivot(rows, noise_level):
    # type: (list, float) -> dict
    """Organise into data[(dist, dim_x)][n_samples] = result dict."""
    subset = [r for r in rows if abs(r["noise_level"] - noise_level) < 1e-9]
    data = {}  # type: Dict
    for r in subset:
        key = (r["relationship_type"], r["dim_x"])
        if key not in data:
            data[key] = {}
        data[key][r["n_samples"]] = r
    return data


# ---------------------------------------------------------------------------
# Combined plot — one figure per d_X, 2×2 panels
# ---------------------------------------------------------------------------

def plot_by_dx(rows,       # type: list
               plots_dir,  # type: str
               dx_values,  # type: List[int]
               noises,     # type: List[float]
               ):
    # type: (...) -> None
    """
    One figure per d_X value (4 figures total), saved as exp3_dx{dx}.pdf/png.

    Layout (2 rows × 2 cols):
      [0,0] Convergence  noise=noises[0]   [0,1] Convergence  noise=noises[1]
      [1,0] Error        noise=noises[0]   [1,1] Error        noise=noises[1]

    Colour = distribution (all 10 on every panel).
    d_X is fixed per figure so no linestyle variation needed.
    Reference lines:
      Convergence panels : dashed at II*=0, dotted at II*=1
      Error panels       : one dotted black reference line, slope=-min(1/2,1/d_X),
                           annotated as a text box
    """
    os.makedirs(plots_dir, exist_ok=True)

    for dx in dx_values:
        slope = _ref_slope(dx)

        fig, axes = plt.subplots(2, 2, figsize=(14.0, 12.0))
        fig.suptitle(
            "Exp 3 — $d_X = {}$\n"
            "Top: convergence of $\\bar{{\\mathrm{{II}}}}_n$ to $\\mathrm{{II}}^*$  "
            "   Bottom: error $|\\bar{{\\mathrm{{II}}}}_n - \\mathrm{{II}}^*|$  (log-log)".format(dx),
            fontsize=15, fontweight="bold",
        )

        handles, labels = [], []
        ref_conv_built  = False   # add II*=0/1 lines to legend once
        ref_err_built   = False   # add error reference to legend once

        for col, noise in enumerate(noises):
            data = _pivot(rows, noise)

            ax_conv = axes[0][col]
            ax_err  = axes[1][col]

            all_ns_err, all_errs = [], []

            for di, dist in enumerate(DISTRIBUTIONS):
                key = (dist, dx)
                if key not in data:
                    continue
                ns    = sorted(data[key].keys())
                means = [data[key][n]["ii_mean"] for n in ns]
                errs  = [data[key][n]["error"]   for n in ns]

                # — Convergence panel —
                line, = ax_conv.plot(ns, means,
                                     marker="o", ms=6, lw=2.0,
                                     color=COLOURS[di])
                # — Error panel (drop zeros for log scale) —
                ns_nz   = [n for n, e in zip(ns, errs) if e > 0]
                errs_nz = [e for e in errs if e > 0]
                if ns_nz:
                    ax_err.plot(ns_nz, errs_nz,
                                marker="o", ms=6, lw=2.0,
                                color=COLOURS[di])
                    all_ns_err.extend(ns_nz)
                    all_errs.extend(errs_nz)

                # Build legend once (from first column)
                if col == 0:
                    handles.append(plt.Line2D([0], [0], color=COLOURS[di], lw=2.0))
                    labels.append(DIST_LABELS[dist])

            # — Convergence panel formatting —
            r0 = ax_conv.axhline(0.0, color="black", lw=2.0, ls="--", alpha=0.75)
            r1 = ax_conv.axhline(1.0, color="black", lw=2.0, ls=":",  alpha=0.75)
            if not ref_conv_built:
                handles += [r0, r1]
                labels  += ["$\\mathrm{II}^* = 0$ (functional)",
                            "$\\mathrm{II}^* = 1$ (independent)"]
                ref_conv_built = True
            ax_conv.set_xscale("log")
            ax_conv.set_ylim(-0.05, 1.10)
            ax_conv.set_title("Convergence  $\\sigma_\\varepsilon = {}$".format(noise),
                              fontsize=13, fontweight="bold")
            ax_conv.set_xlabel("$n$  (log scale)", fontsize=12, fontweight="bold")
            ax_conv.set_ylabel("$\\bar{\\mathrm{II}}_n$",
                               fontsize=12, fontweight="bold")
            ax_conv.grid(alpha=0.25)
            ax_conv.tick_params(labelsize=11)
            for spine in ax_conv.spines.values():
                spine.set_linewidth(1.2)

            # — Error panel formatting —
            if all_ns_err:
                log_ns   = np.log(all_ns_err)
                log_errs = np.log(all_errs)
                log_C    = float(np.mean(log_errs) - slope * np.mean(log_ns))
                x_ref    = np.geomspace(min(all_ns_err) * 0.8,
                                        max(all_ns_err) * 1.25, 200)
                ref_err, = ax_err.plot(x_ref, np.exp(log_C) * x_ref ** slope,
                                       color="black", lw=2.5, ls=":",
                                       alpha=0.70)
                ax_err.text(0.97, 0.05,
                            "theory slope = {:.2f}".format(slope),
                            transform=ax_err.transAxes,
                            fontsize=11, fontweight="bold",
                            ha="right", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      fc="white", ec="grey", alpha=0.85))
                if not ref_err_built:
                    handles.append(plt.Line2D([0], [0], color="black",
                                              lw=2.5, ls=":"))
                    labels.append("Ref. slope $= {:.2f}$".format(slope))
                    ref_err_built = True

            ax_err.set_xscale("log")
            ax_err.set_yscale("log")
            ax_err.set_title("Error  $\\sigma_\\varepsilon = {}$".format(noise),
                             fontsize=13, fontweight="bold")
            ax_err.set_xlabel("$n$  (log scale)", fontsize=12, fontweight="bold")
            ax_err.set_ylabel(
                "$|\\bar{\\mathrm{II}}_n - \\mathrm{II}^*|$  (log scale)",
                fontsize=12, fontweight="bold")
            ax_err.grid(alpha=0.25, which="both")
            ax_err.tick_params(labelsize=11)
            for spine in ax_err.spines.values():
                spine.set_linewidth(1.2)

        fig.legend(handles, labels,
                   loc="center left",
                   bbox_to_anchor=(1.0, 0.5),
                   fontsize=11,
                   framealpha=0.95,
                   edgecolor="grey",
                   title="Distribution",
                   title_fontsize=12)
        plt.tight_layout(rect=[0, 0, 0.80, 0.94])

        for ext in ("pdf", "png"):
            out = os.path.join(plots_dir, "exp3_dx{}.{}".format(dx, ext))
            fig.savefig(out, bbox_inches="tight", dpi=300)
            print("Saved: {}".format(out))
        plt.close(fig)


# ---------------------------------------------------------------------------
# Table — empirical slope of log(error) vs log(n)
# ---------------------------------------------------------------------------

def export_rate_table(rows,      # type: list
                      plots_dir, # type: str
                      dx_values, # type: List[int]
                      ):
    # type: (...) -> None
    """
    For each (dist, d_X, noise): fit log(error) ~ slope * log(n) + intercept.
    Compare empirical slope to theoretical -min(1/2, 1/d_X).
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
                errs = [data[key][n]["error"] for n in ns]
                # drop zeros
                pairs = [(n, e) for n, e in zip(ns, errs) if e > 0]
                if len(pairs) < 2:
                    continue
                log_ns   = np.log([p[0] for p in pairs])
                log_errs = np.log([p[1] for p in pairs])
                slope, intercept, r_val, *_ = linregress(log_ns, log_errs)
                theory_slope = _ref_slope(dx)
                table_rows.append({
                    "Distribution":   DIST_LABELS[dist],
                    "d_X":            dx,
                    "noise":          noise,
                    "emp_slope":      round(slope, 4),
                    "theory_slope":   round(theory_slope, 4),
                    "slope_diff":     round(slope - theory_slope, 4),
                    "R2":             round(r_val ** 2, 4),
                })

    df = pd.DataFrame(table_rows)
    os.makedirs(plots_dir, exist_ok=True)

    out_csv = os.path.join(plots_dir, "exp3_rate_table.csv")
    df.to_csv(out_csv, index=False)
    print("\nSaved: {}".format(out_csv))
    print(df.to_string(index=False))

    out_tex = os.path.join(plots_dir, "exp3_rate_table.tex")
    with open(out_tex, "w") as fh:
        fh.write("% Exp 3 — Empirical vs theoretical convergence rate slopes\n\n")
        fh.write(df.to_latex(index=False, float_format="%.4f"))
    print("Saved: {}".format(out_tex))


# ---------------------------------------------------------------------------
# Plot mode dispatcher
# ---------------------------------------------------------------------------

def plot_mode(args):
    # type: (argparse.Namespace) -> None
    rows = _load_results(args.results_dir)
    if not rows:
        print("No .pkl files found in {!r}".format(args.results_dir))
        return

    n_values  = sorted({r["n_samples"] for r in rows})
    dx_values = sorted({r["dim_x"]     for r in rows})
    noises    = sorted({r["noise_level"] for r in rows})
    print("Loaded {} result files.".format(len(rows)))
    print("  n values : {}".format(n_values))
    print("  d_X      : {}".format(dx_values))
    print("  noise    : {}".format(noises))

    plot_by_dx(rows, args.plots_dir, dx_values, noises)
    export_rate_table(rows, args.plots_dir, dx_values)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp 3 — Convergence of II_hat_n to population quantity II*."
    )
    parser.add_argument("--mode", choices=["run", "plot"], required=True)

    # run-mode
    parser.add_argument("--relationship_type", choices=DISTRIBUTIONS)
    parser.add_argument("--n_samples",     type=int)
    parser.add_argument("--n_simulations", type=int,   default=500)
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
                parser.error("--{} is required in --mode run".format(req))
        run_experiment(args)
    else:
        plot_mode(args)


if __name__ == "__main__":
    main()
