"""
experiments/exp4_noisy_limit/exp4_noisy_limit.py
-------------------------------------------------
Experiment 4 — Convergence to the empirical noisy limit II*_noisy.

Motivation
----------
Experiment 3 uses II* = 0 for all functional distributions (the noiseless
theoretical limit).  With finite noise sigma_eps > 0, the true population
quantity is II*_noisy > 0.  At d_X = 1 the estimator converges so fast
(rate n^{-1/2}) that it has already reached II*_noisy before n = 100, so
the Exp-3 error plot is flat — not because convergence failed, but because
the estimator hit the WRONG reference target.

Fix: run the II estimator at a large reference sample size (large_n = 50000)
with B_limit = 10 replications to estimate II*_noisy empirically.  Then
re-compute errors from the Exp-3 results against this corrected reference.

Modes
-----
  run_limit   Estimate II*_noisy for one (dist, d_X, noise) combination.
              Called by one SLURM array task; saves results_limit/*.pkl.

  plot        Load all limit pkl files + existing Exp-3 result pkl files.
              Re-compute corrected errors.  Produce one 2x2 figure per d_X.
              Saved as exp4_dx{dx}.pdf/png in --plots_dir.

Grid (run_limit mode)
---------------------
  10 distributions x 4 d_X x 2 noise = 80 SLURM tasks (0-79)
  Indexing:
    noise_idx = task_id // 40         (2 noise levels)
    dx_idx    = (task_id % 40) // 10  (4 d_X values: 1,2,5,10)
    rel_idx   = task_id % 10          (10 distributions)

Usage
-----
# One limit-estimation task (called by SLURM):
python exp4_noisy_limit.py --mode run_limit \\
    --relationship_type linear \\
    --dim_x 1 --dim_y 3 --noise_level 0.1 \\
    --large_n 50000 --B_limit 10 \\
    --output_file results_limit/linear_dx1_noise0-1.pkl \\
    --random_seed 999

# After all 80 tasks finish:
python exp4_noisy_limit.py --mode plot \\
    --limit_dir      results_limit/ \\
    --exp3_dir       ../exp3_convergence/results/ \\
    --plots_dir      plots_corrected/
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
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Repo-root import for shared II estimator
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from utils.ii_estimator import compute_ii_vectorized  # noqa: E402


# ---------------------------------------------------------------------------
# Constants  (kept in sync with exp3_convergence.py)
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


def _ref_slope(dim_x):
    # type: (int) -> float
    return -min(0.5, 1.0 / float(dim_x))


# ---------------------------------------------------------------------------
# Data generating process  (identical to exp3_convergence.py)
# ---------------------------------------------------------------------------

def _x_to_y_elementwise(f_X,   # type: np.ndarray
                         dim_y  # type: int
                         ):
    # type: (...) -> np.ndarray
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
# Chunked II estimator — memory-safe for large n
# ---------------------------------------------------------------------------

def compute_ii_chunked(X,           # type: np.ndarray
                       Y,           # type: np.ndarray
                       chunk_size,  # type: int
                       ):
    # type: (...) -> float
    """
    Memory-efficient II estimator for large n.

    The vectorized version builds an (n x n) Y-distance matrix which costs
    n^2 * 8 bytes (20 GB at n=50,000).  This version processes rows of the
    Y-distance matrix in chunks of `chunk_size`, keeping peak memory to
    chunk_size * n * 8 bytes (0.8 GB at chunk_size=2000, n=50,000).

    The X nearest-neighbour search uses cKDTree — O(n log n) time and
    O(n) memory, same as the vectorized version.

    Result is identical to compute_ii_vectorized up to floating-point order.
    """
    n = X.shape[0]

    # Step 1 — 1-NN in X-space via KDTree (memory-efficient)
    tree_X = cKDTree(X)
    _, indices = tree_X.query(X, k=2)   # k=2: self + 1 neighbour
    nn_idx = indices[:, 1]              # (n,) nearest neighbour index

    # Step 2 — distance from each point to its X-NN in Y-space
    d_nn = np.linalg.norm(Y[nn_idx] - Y, axis=1)  # (n,)

    # Step 3 — chunked rank computation
    # For each point i: rank = #{j : ||Y_j - Y_i|| <= d_nn[i]}
    # We chunk over the "source" dimension (rows of the distance matrix).
    total_rank = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_Y  = Y[start:end]                        # (chunk, d_Y)
        D_chunk  = cdist(chunk_Y, Y, metric="euclidean")  # (chunk, n)
        # Count columns <= d_nn for each row
        ranks_chunk = np.sum(
            D_chunk <= d_nn[start:end, np.newaxis], axis=1)  # (chunk,)
        total_rank += float(np.sum(ranks_chunk))

    return 2.0 / (n * n) * total_rank


def compute_ii_auto(X, Y, large_n_threshold=8000, chunk_size=2000):
    # type: (np.ndarray, np.ndarray, int, int) -> float
    """
    Use vectorized for small n (fast, uses BLAS fully),
    chunked for large n (memory-safe).
    """
    if X.shape[0] <= large_n_threshold:
        return compute_ii_vectorized(X, Y)
    return compute_ii_chunked(X, Y, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# Mode: run_limit
# ---------------------------------------------------------------------------

def run_limit_mode(args):
    # type: (argparse.Namespace) -> None
    """
    Estimate II*_noisy for one (dist, d_X, noise) combination by running
    the II estimator at large_n with B_limit replications.
    For D0 (independent), II*_noisy = 1.0 exactly so we skip computation.
    """
    rng = np.random.default_rng(args.random_seed)

    # D0 independent: limit is always exactly 1.0, skip expensive run
    if args.relationship_type == "independent":
        result = {
            "relationship_type": "independent",
            "dim_x":             args.dim_x,
            "dim_y":             args.dim_y,
            "noise_level":       args.noise_level,
            "large_n":           args.large_n,
            "B_limit":           args.B_limit,
            "ii_values":         [1.0],
            "ii_star_noisy":     1.0,
            "ii_star_std":       0.0,
            "note":              "Exact; independent of noise by construction.",
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)),
                    exist_ok=True)
        with open(args.output_file, "wb") as fh:
            pickle.dump(result, fh)
        print("D0 independent: II*_noisy = 1.0 (exact). Saved: {}".format(
            args.output_file))
        return

    linear_A = _make_linear_A(args.dim_x, args.dim_y) \
        if args.relationship_type == "linear" else None

    print("Estimating II*_noisy:")
    print("  dist={} dx={} noise={} large_n={} B_limit={}".format(
        args.relationship_type, args.dim_x, args.noise_level,
        args.large_n, args.B_limit))

    ii_values = []
    for b in range(args.B_limit):
        X, Y = generate_sample(
            args.relationship_type, args.large_n,
            args.dim_x, args.dim_y, args.noise_level,
            rng, linear_A=linear_A,
        )
        val = compute_ii_auto(X, Y,
                              large_n_threshold=args.large_n_threshold,
                              chunk_size=args.chunk_size)
        ii_values.append(val)
        print("  rep {}/{} II = {:.6f}".format(b + 1, args.B_limit, val),
              flush=True)

    ii_star_noisy = float(np.mean(ii_values))
    ii_star_std   = float(np.std(ii_values))

    result = {
        "relationship_type": args.relationship_type,
        "dim_x":             args.dim_x,
        "dim_y":             args.dim_y,
        "noise_level":       args.noise_level,
        "large_n":           args.large_n,
        "B_limit":           args.B_limit,
        "ii_values":         ii_values,
        "ii_star_noisy":     ii_star_noisy,
        "ii_star_std":       ii_star_std,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)),
                exist_ok=True)
    with open(args.output_file, "wb") as fh:
        pickle.dump(result, fh)

    print("Saved {} — II*_noisy = {:.6f} +/- {:.6f}".format(
        args.output_file, ii_star_noisy, ii_star_std))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _load_limits(limit_dir, dx_values, noises):
    # type: (str, List[int], List[float]) -> Dict
    """
    Returns {(dist, dx, noise): ii_star_noisy}.
    Prints a warning for any missing file.
    """
    limits = {}
    for dist in DISTRIBUTIONS:
        for dx in dx_values:
            for noise in noises:
                key = (dist, dx, noise)
                noise_label = "{:.1f}".format(noise).replace(".", "-")
                fname = "limit_{}_dx{}_noise{}.pkl".format(dist, dx, noise_label)
                fpath = os.path.join(limit_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "rb") as fh:
                        res = pickle.load(fh)
                    limits[key] = res["ii_star_noisy"]
                else:
                    print("WARNING: missing limit file — {}".format(fpath))
                    limits[key] = None
    return limits


def _load_exp3(exp3_dir):
    # type: (str) -> list
    rows = []
    for pkl in sorted(Path(exp3_dir).glob("*.pkl")):
        try:
            with open(pkl, "rb") as fh:
                rows.append(pickle.load(fh))
        except Exception as exc:
            print("WARNING: could not load {} — {}".format(pkl, exc))
    return rows


def _pivot_exp3(rows, noise_level):
    # type: (list, float) -> Dict
    """data[(dist, dx)][n] = result dict (same layout as exp3 _pivot)."""
    subset = [r for r in rows
              if abs(r.get("noise_level", -1) - noise_level) < 1e-9]
    data = {}  # type: Dict
    for r in subset:
        key = (r.get("relationship_type"), r.get("dim_x"))
        if None in key:
            continue
        if key not in data:
            data[key] = {}
        data[key][r["n_samples"]] = r
    return data


def _export_corrected_table(rows, limits, plots_dir, dx_values, noises):
    # type: (list, Dict, str, List[int], List[float]) -> None
    """Fit log(corrected_error) ~ slope*log(n); compare to theory."""
    import csv
    table = []
    for noise in noises:
        data = _pivot_exp3(rows, noise)
        for dist in DISTRIBUTIONS:
            for dx in dx_values:
                key_data  = (dist, dx)
                key_limit = (dist, dx, noise)
                if key_data not in data or limits.get(key_limit) is None:
                    continue
                limit = limits[key_limit]
                ns    = sorted(data[key_data].keys())
                errs  = [abs(data[key_data][n]["ii_mean"] - limit) for n in ns]
                pairs = [(n, e) for n, e in zip(ns, errs) if e > 1e-12]
                if len(pairs) < 2:
                    continue
                log_ns   = np.log([p[0] for p in pairs])
                log_errs = np.log([p[1] for p in pairs])
                slope, _, r_val, *_ = linregress(log_ns, log_errs)
                table.append({
                    "Distribution": DIST_LABELS[dist],
                    "d_X":          dx,
                    "noise":        noise,
                    "II*_noisy":    round(limit, 5),
                    "emp_slope":    round(slope, 4),
                    "theory_slope": round(_ref_slope(dx), 4),
                    "slope_diff":   round(slope - _ref_slope(dx), 4),
                    "R2":           round(r_val ** 2, 4),
                })

    os.makedirs(plots_dir, exist_ok=True)
    out_csv = os.path.join(plots_dir, "exp4_corrected_rate_table.csv")
    if table:
        fields = list(table[0].keys())
        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
        print("Saved: {}".format(out_csv))
        for row in table:
            print("  {Distribution} dx={d_X} noise={noise}  "
                  "II*_noisy={II*_noisy:.4f}  "
                  "emp={emp_slope:.3f}  theory={theory_slope:.3f}  "
                  "R2={R2:.4f}".format(**row))


# ---------------------------------------------------------------------------
# Plot mode — one 2x2 figure per d_X
# ---------------------------------------------------------------------------

def plot_corrected(rows, limits, plots_dir, dx_values, noises):
    # type: (list, Dict, str, List[int], List[float]) -> None
    """
    Layout per figure (fixed d_X):
      [0,0] Convergence   noise=0.1   [0,1] Convergence   noise=0.5
      [1,0] Corrected err noise=0.1   [1,1] Corrected err noise=0.5

    Corrected error = |II_mean(n) - II*_noisy|  (not |II_mean(n) - 0|).
    Colour = distribution.  Dashed horizontal lines on convergence panels
    show the estimated II*_noisy for each functional distribution.
    """
    os.makedirs(plots_dir, exist_ok=True)

    for dx in dx_values:
        slope = _ref_slope(dx)
        fig, axes = plt.subplots(2, 2, figsize=(14.0, 12.0))
        fig.suptitle(
            "Exp 4 — $d_X = {}$   Corrected convergence to $II^*_{{\\rm noisy}}$\n"
            "Top: $\\bar{{\\rm II}}_n$ vs $n$     "
            "Bottom: $|\\bar{{\\rm II}}_n - II^*_{{\\rm noisy}}|$ vs $n$ (log-log)".format(dx),
            fontsize=15, fontweight="bold",
        )

        handles, labels = [], []
        ref_conv_built = False
        ref_err_built  = False

        for col, noise in enumerate(noises):
            data = _pivot_exp3(rows, noise)
            ax_conv = axes[0][col]
            ax_err  = axes[1][col]

            all_ns_err  = []
            all_errs    = []

            for di, dist in enumerate(DISTRIBUTIONS):
                key_data  = (dist, dx)
                key_limit = (dist, dx, noise)

                if key_data not in data:
                    continue

                limit = limits.get(key_limit)
                ns    = sorted(data[key_data].keys())
                means = [data[key_data][n]["ii_mean"] for n in ns]

                # --- Convergence panel ---
                ax_conv.plot(ns, means,
                             marker="o", ms=6, lw=2.0, color=COLOURS[di])

                # Mark estimated II*_noisy as a faint dashed horizontal line
                if limit is not None and dist != "independent":
                    ax_conv.axhline(limit,
                                    color=COLOURS[di], lw=1.2,
                                    ls="--", alpha=0.45)

                # --- Corrected error panel ---
                if limit is not None:
                    errs    = [abs(m - limit) for m in means]
                    ns_nz   = [n for n, e in zip(ns, errs) if e > 1e-12]
                    errs_nz = [e for e in errs if e > 1e-12]
                    if ns_nz:
                        ax_err.plot(ns_nz, errs_nz,
                                    marker="o", ms=6, lw=2.0,
                                    color=COLOURS[di])
                        all_ns_err.extend(ns_nz)
                        all_errs.extend(errs_nz)

                # Build legend once (first noise column)
                if col == 0:
                    handles.append(
                        plt.Line2D([0], [0], color=COLOURS[di], lw=2.0))
                    if limit is not None and dist != "independent":
                        labels.append(
                            "{} ($II^*_{{\\rm noisy}}={:.3f}$)".format(
                                DIST_LABELS[dist], limit))
                    else:
                        labels.append(DIST_LABELS[dist])

            # --- Convergence panel formatting ---
            r0 = ax_conv.axhline(0.0, color="black", lw=2.0, ls="--", alpha=0.7)
            r1 = ax_conv.axhline(1.0, color="black", lw=2.0, ls=":",  alpha=0.7)
            if not ref_conv_built:
                handles += [r0, r1]
                labels  += ["$II^* = 0$ (noiseless limit)",
                            "$II^* = 1$ (independent)"]
                ref_conv_built = True

            ax_conv.set_xscale("log")
            ax_conv.set_ylim(-0.05, 1.10)
            ax_conv.set_title(
                "Convergence  $\\sigma_\\varepsilon = {}$".format(noise),
                fontsize=13, fontweight="bold")
            ax_conv.set_xlabel("$n$  (log scale)", fontsize=12,
                               fontweight="bold")
            ax_conv.set_ylabel("$\\bar{\\rm II}_n$",
                               fontsize=12, fontweight="bold")
            ax_conv.grid(alpha=0.25)
            ax_conv.tick_params(labelsize=11)
            for sp in ax_conv.spines.values():
                sp.set_linewidth(1.2)

            # --- Error panel formatting ---
            if all_ns_err:
                log_ns   = np.log(all_ns_err)
                log_errs = np.log(all_errs)
                log_C    = float(np.mean(log_errs) - slope * np.mean(log_ns))
                x_ref    = np.geomspace(min(all_ns_err) * 0.8,
                                        max(all_ns_err) * 1.25, 200)
                ref_line, = ax_err.plot(
                    x_ref, np.exp(log_C) * x_ref ** slope,
                    color="black", lw=2.5, ls=":", alpha=0.70)
                ax_err.text(
                    0.97, 0.05,
                    "theory slope = {:.2f}".format(slope),
                    transform=ax_err.transAxes,
                    fontsize=11, fontweight="bold",
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="grey", alpha=0.85))
                if not ref_err_built:
                    handles.append(
                        plt.Line2D([0], [0], color="black", lw=2.5, ls=":"))
                    labels.append(
                        "Ref. slope $= {:.2f}$".format(slope))
                    ref_err_built = True

            ax_err.set_xscale("log")
            ax_err.set_yscale("log")
            ax_err.set_title(
                "Corrected error  $\\sigma_\\varepsilon = {}$".format(noise),
                fontsize=13, fontweight="bold")
            ax_err.set_xlabel("$n$  (log scale)", fontsize=12,
                              fontweight="bold")
            ax_err.set_ylabel(
                "$|\\bar{\\rm II}_n - II^*_{{\\rm noisy}}|$  (log scale)",
                fontsize=12, fontweight="bold")
            ax_err.grid(alpha=0.25, which="both")
            ax_err.tick_params(labelsize=11)
            for sp in ax_err.spines.values():
                sp.set_linewidth(1.2)

        fig.legend(handles, labels,
                   loc="center left",
                   bbox_to_anchor=(1.0, 0.5),
                   fontsize=10,
                   framealpha=0.95,
                   edgecolor="grey",
                   title="Distribution",
                   title_fontsize=11)
        plt.tight_layout(rect=[0, 0, 0.78, 0.94])

        for ext in ("pdf", "png"):
            out = os.path.join(plots_dir, "exp4_dx{}.{}".format(dx, ext))
            fig.savefig(out, bbox_inches="tight", dpi=300)
            print("Saved: {}".format(out))
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot mode dispatcher
# ---------------------------------------------------------------------------

def plot_mode(args):
    # type: (argparse.Namespace) -> None
    dx_values = [1, 2, 5, 10]
    noises    = [0.1, 0.5]

    print("Loading Exp-3 results from: {}".format(args.exp3_dir))
    rows = _load_exp3(args.exp3_dir)
    if not rows:
        print("ERROR: no .pkl files found in {!r}".format(args.exp3_dir))
        sys.exit(1)
    print("  Loaded {} result files.".format(len(rows)))

    print("Loading limit estimates from: {}".format(args.limit_dir))
    limits = _load_limits(args.limit_dir, dx_values, noises)
    n_found = sum(1 for v in limits.values() if v is not None)
    print("  Found {}/{} limit estimates.".format(n_found, len(limits)))

    plot_corrected(rows, limits, args.plots_dir, dx_values, noises)
    _export_corrected_table(rows, limits, args.plots_dir, dx_values, noises)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4 — Corrected convergence to empirical II*_noisy.")
    parser.add_argument("--mode", choices=["run_limit", "plot"], required=True)

    # run_limit args
    parser.add_argument("--relationship_type", choices=DISTRIBUTIONS,
                        default="linear")
    parser.add_argument("--dim_x",       type=int,   default=1)
    parser.add_argument("--dim_y",       type=int,   default=3)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--large_n",     type=int,   default=50000,
                        help="Sample size for limit estimation (default 50000).")
    parser.add_argument("--B_limit",     type=int,   default=10,
                        help="Replications for limit estimation (default 10).")
    parser.add_argument("--chunk_size",  type=int,   default=2000,
                        help="Row-chunk size for memory-safe II computation.")
    parser.add_argument("--large_n_threshold", type=int, default=8000,
                        help="Use chunked II above this n (default 8000).")
    parser.add_argument("--output_file", type=str,
                        default="results_limit/out.pkl")
    parser.add_argument("--random_seed", type=int,   default=42)

    # plot args
    parser.add_argument("--limit_dir",  type=str, default="results_limit/")
    parser.add_argument("--exp3_dir",   type=str,
                        default="../exp3_convergence/results/",
                        help="Directory containing Exp-3 .pkl result files.")
    parser.add_argument("--plots_dir",  type=str, default="plots_corrected/")

    args = parser.parse_args()

    if args.mode == "run_limit":
        run_limit_mode(args)
    else:
        plot_mode(args)


if __name__ == "__main__":
    main()
