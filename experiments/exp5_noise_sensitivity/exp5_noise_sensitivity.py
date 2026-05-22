"""
experiments/exp5_noise_sensitivity/exp5_noise_sensitivity.py
-------------------------------------------------------------
Experiment 5 — II as a function of noise level.

Goal
----
Fix d_X, d_Y, n and sweep noise_level sigma_eps from 0 to 5.
At sigma_eps = 0: Y = f(X) exactly -> II* = 0 for all functional distributions.
At sigma_eps -> inf: Y is swamped by noise -> II* -> 1 (independence).
D0 (independent) stays at ~1 for all noise levels.

This experiment visualises how the II estimator interpolates between the
functional and independent regimes as noise grows, and how fast that
transition happens for different dependence structures and dimensions.

Grid
----
  10 distributions x 12 noise levels x 3 d_X values
  = 360 SLURM tasks (0-359)

  Indexing:
    dx_idx    = task_id // 120         (3 d_X values: 1, 2, 5)
    noise_idx = (task_id % 120) // 10  (12 noise levels)
    rel_idx   = task_id % 10           (10 distributions)

Fixed parameters
----------------
  n_samples    = 5000
  n_simulations (B) = 200
  d_Y          = 3

Usage
-----
# One task (called by SLURM):
python exp5_noise_sensitivity.py --mode run \\
    --relationship_type linear \\
    --noise_level 0.5 \\
    --dim_x 2 --dim_y 3 --n_samples 5000 --n_simulations 200 \\
    --output_file results/linear_noise0-5_dx2_task0.pkl \\
    --random_seed 42

# After all tasks finish:
python exp5_noise_sensitivity.py --mode plot \\
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from utils.ii_estimator import compute_ii_vectorized  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISTRIBUTIONS = [
    "independent", "linear", "quadratic", "cubic", "sine",
    "cosine", "exponential", "logarithmic", "step", "parabolic",
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

COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Noise levels swept — dense near 0 to capture the fast rise, then sparser
NOISE_VALUES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

DX_VALUES    = [1, 2, 5]


# ---------------------------------------------------------------------------
# Data generating process  (identical to exp3 / exp4)
# ---------------------------------------------------------------------------

def _x_to_y_elementwise(f_X, dim_y):
    # type: (np.ndarray, int) -> np.ndarray
    n, k = f_X.shape
    k_use = min(k, dim_y)
    out = f_X[:, :k_use]
    if k_use < dim_y:
        out = np.concatenate(
            [out, np.tile(out[:, -1:], (1, dim_y - k_use))], axis=1)
    return out


def _make_linear_A(dim_x, dim_y):
    # type: (int, int) -> np.ndarray
    rng_A = np.random.default_rng(999 + dim_x * 100 + dim_y)
    return rng_A.standard_normal((dim_x, dim_y)) / np.sqrt(dim_x)


def generate_sample(relationship_type,  # type: str
                    n,                   # type: int
                    dim_x,               # type: int
                    dim_y,               # type: int
                    noise_level,         # type: float
                    rng,
                    linear_A=None,       # type: Optional[np.ndarray]
                    ):
    # type: (...) -> tuple
    X   = rng.standard_normal((n, dim_x))
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
        Y = _x_to_y_elementwise(np.exp(X / 2.0), dim_y) + eps

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
        raise ValueError("Unknown relationship_type: {!r}".format(relationship_type))

    return X, Y


# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------

def run_experiment(args):
    # type: (argparse.Namespace) -> None
    rng      = np.random.default_rng(args.random_seed)
    B        = args.n_simulations
    n        = args.n_samples
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
        if (b + 1) % 50 == 0:
            print("  rep {}/{} II={:.4f}".format(b + 1, B, ii_values[b]),
                  flush=True)

    ii_mean = float(np.mean(ii_values))
    ii_std  = float(np.std(ii_values, ddof=1))

    result = {
        "relationship_type": args.relationship_type,
        "noise_level":       args.noise_level,
        "n_samples":         n,
        "n_simulations":     B,
        "dim_x":             args.dim_x,
        "dim_y":             args.dim_y,
        "random_seed":       args.random_seed,
        "ii_values":         ii_values,
        "ii_mean":           ii_mean,
        "ii_std":            ii_std,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)),
                exist_ok=True)
    with open(args.output_file, "wb") as fh:
        pickle.dump(result, fh)

    print("[done] {} noise={} d_X={} | II_mean={:.4f} II_std={:.4f}".format(
        args.relationship_type, args.noise_level, args.dim_x,
        ii_mean, ii_std))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _load_results(results_dir):
    # type: (str) -> list
    rows = []
    for pkl in sorted(Path(results_dir).glob("*.pkl")):
        try:
            with open(pkl, "rb") as fh:
                rows.append(pickle.load(fh))
        except Exception as exc:
            print("WARNING: could not load {} — {}".format(pkl, exc))
    return rows


def _pivot(rows):
    # type: (list) -> Dict
    """
    data[(dist, dx)][noise] = {"ii_mean": float, "ii_std": float, "B": int}
    """
    data = {}  # type: Dict
    for r in rows:
        key = (r.get("relationship_type"), r.get("dim_x"))
        if None in key:
            continue
        noise = r.get("noise_level")
        if key not in data:
            data[key] = {}
        data[key][noise] = {
            "ii_mean": r["ii_mean"],
            "ii_std":  r["ii_std"],
            "B":       r["n_simulations"],
        }
    return data


# ---------------------------------------------------------------------------
# Plot mode
# ---------------------------------------------------------------------------

def plot_noise_sensitivity(rows, plots_dir, dx_values):
    # type: (list, str, List[int]) -> None
    """
    One figure with len(dx_values) panels side by side.
    Each panel: II_mean vs noise_level for all 10 distributions.
    Shaded band = mean +/- 1 std / sqrt(B)  (standard error of the mean).

    Interpretation guide on each panel:
      - Dashed line at II = 0: noiseless functional limit
      - Dotted line at II = 1: independence limit
      - Functional distributions rise from ~0 at sigma=0 toward 1 at large sigma
      - D0 independent stays flat at ~1 across all noise levels
    """
    os.makedirs(plots_dir, exist_ok=True)
    data = _pivot(rows)

    ncols = len(dx_values)
    fig, axes = plt.subplots(1, ncols, figsize=(7.0 * ncols, 6.5),
                             sharey=True)
    if ncols == 1:
        axes = [axes]

    fig.suptitle(
        "Exp 5 — $\\bar{{\\rm II}}_n$ vs noise level $\\sigma_\\varepsilon$\n"
        "$\\sigma_\\varepsilon = 0$: $Y = f(X)$ exactly ($\\rm II^* = 0$)     "
        "$\\sigma_\\varepsilon \\to \\infty$: independence ($\\rm II^* = 1$)     "
        "Shaded band: $\\pm 1$ std across $B$ replications",
        fontsize=14, fontweight="bold",
    )

    handles, labels = [], []
    refs_built = False

    for col, dx in enumerate(dx_values):
        ax = axes[col]

        for di, dist in enumerate(DISTRIBUTIONS):
            key = (dist, dx)
            if key not in data:
                continue

            noises = sorted(data[key].keys())
            means  = [data[key][s]["ii_mean"] for s in noises]
            stds   = [data[key][s]["ii_std"]  for s in noises]

            line, = ax.plot(noises, means,
                            marker="o", ms=6, lw=2.0,
                            color=COLOURS[di])
            # Shaded band = mean +/- 1 std across B replications.
            # This shows how much a single experiment would vary,
            # not just the uncertainty in the mean.
            ax.fill_between(noises,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=COLOURS[di], alpha=0.15)

            if col == 0:
                handles.append(plt.Line2D([0], [0], color=COLOURS[di], lw=2.0))
                labels.append(DIST_LABELS[dist])

        # Reference lines
        r0 = ax.axhline(0.0, color="black", lw=1.8, ls="--", alpha=0.6)
        r1 = ax.axhline(1.0, color="black", lw=1.8, ls=":",  alpha=0.6)
        if not refs_built:
            handles += [r0, r1]
            labels  += ["$\\rm II^* = 0$ (functional, noiseless)",
                        "$\\rm II^* = 1$ (independent)"]
            refs_built = True

        ax.set_xlim(left=-0.05)
        ax.set_ylim(-0.03, 1.08)
        ax.set_title("$d_X = {}$".format(dx), fontsize=14, fontweight="bold")
        ax.set_xlabel("Noise level $\\sigma_\\varepsilon$",
                      fontsize=13, fontweight="bold")
        if col == 0:
            ax.set_ylabel("$\\bar{{\\rm II}}_n$",
                          fontsize=13, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=11)
        for sp in ax.spines.values():
            sp.set_linewidth(1.2)

    fig.legend(handles, labels,
               loc="center left",
               bbox_to_anchor=(1.0, 0.5),
               fontsize=11,
               framealpha=0.95,
               edgecolor="grey",
               title="Distribution",
               title_fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.82, 0.93])

    for ext in ("pdf", "png"):
        out = os.path.join(plots_dir, "exp5_noise_sensitivity.{}".format(ext))
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print("Saved: {}".format(out))
    plt.close(fig)


def plot_mode(args):
    # type: (argparse.Namespace) -> None
    rows = _load_results(args.results_dir)
    if not rows:
        print("ERROR: no .pkl files found in {!r}".format(args.results_dir))
        sys.exit(1)

    dx_values = sorted({r["dim_x"] for r in rows})
    noises    = sorted({r["noise_level"] for r in rows})
    dists     = sorted({r["relationship_type"] for r in rows})
    print("Loaded {} result files.".format(len(rows)))
    print("  d_X values : {}".format(dx_values))
    print("  noise vals : {}".format(noises))
    print("  dists      : {}".format(dists))

    plot_noise_sensitivity(rows, args.plots_dir, dx_values)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp 5 — II estimator vs noise level.")
    parser.add_argument("--mode", choices=["run", "plot"], required=True)

    # run args
    parser.add_argument("--relationship_type", choices=DISTRIBUTIONS,
                        default="linear")
    parser.add_argument("--noise_level",   type=float, default=0.5)
    parser.add_argument("--n_samples",     type=int,   default=5000)
    parser.add_argument("--n_simulations", type=int,   default=200)
    parser.add_argument("--dim_x",         type=int,   default=2)
    parser.add_argument("--dim_y",         type=int,   default=3)
    parser.add_argument("--output_file",   type=str,   default="results/out.pkl")
    parser.add_argument("--random_seed",   type=int,   default=42)

    # plot args
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--plots_dir",   type=str, default="plots/")

    args = parser.parse_args()

    if args.mode == "run":
        run_experiment(args)
    else:
        plot_mode(args)


if __name__ == "__main__":
    main()
