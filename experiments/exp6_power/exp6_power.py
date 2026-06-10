"""
experiments/exp6_power/exp6_power.py
--------------------------------------
Experiment 6 — Statistical power comparison.

Goal
----
Determine whether II is a powerful test for detecting association,
and how it compares to established competitors across different
relationship types, sample sizes, noise levels, and dimensions.

For each dataset, every test produces a p-value via a permutation
test (same procedure for all tests — fair comparison).  Power is
the fraction of datasets where p-value < alpha.

Two experimental settings
--------------------------
  Setting A  d_X=1, d_Y=1   Tests: II, dCor, HSIC, Pearson, Spearman
  Setting B  d_X=5, d_Y=3   Tests: II, dCor, HSIC

Tests
-----
  II       Information Imbalance (this work). Small value = dependent.
  dCor     Distance Correlation (Székely et al. 2007). Large = dependent.
  HSIC     Hilbert-Schmidt Independence Criterion (Gretton et al. 2005).
           Uses RBF kernel with median heuristic bandwidth. Large = dependent.
  Pearson  |r|, standard linear correlation. Large = dependent.
  Spearman |rho|, rank correlation. Large = dependent.

Permutation test (same for all)
--------------------------------
  For each observed statistic T_obs, generate P permuted statistics
  T_1*, ..., T_P* by shuffling rows of Y.
  p-value = fraction of {T_1*, ..., T_P*} more extreme than T_obs.
  "More extreme" = <= for II (small = dependent), >= for all others.

Storage format
--------------
  One pkl per SLURM task, containing:
    - Raw p-values:   shape (B,) per test — recompute power at any alpha
    - Raw statistics: shape (B,) per test — enables ROC curves
    - Null summary:   mean/std of permutation nulls, shape (B,) per test
    - Power:          pre-computed at alpha in {0.01, 0.05, 0.10}
    - Full metadata

Grid (480 tasks, 0-479)
------------------------
  Indexing:
    setting_idx = task_id // 240      (2 settings: A=0, B=1)
    noise_idx   = (task_id%240)//60   (4 noise levels)
    n_idx       = (task_id%60)//10    (6 sample sizes)
    rel_idx     = task_id % 10        (10 distributions)

Usage
-----
# One SLURM task:
python exp6_power.py --mode run \\
    --setting A --relationship_type linear \\
    --n_samples 500 --noise_level 0.5 \\
    --B 100 --P 99 \\
    --output_file results/linear_n500_noise0-5_A_task0.pkl \\
    --random_seed 42

# After all tasks finish:
python exp6_power.py --mode plot \\
    --results_dir results/ --plots_dir plots/
"""

import argparse
import datetime
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


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

# Colour and label per test
TEST_COLOURS = {
    "ii":       "#1f77b4",
    "dcor":     "#d62728",
    "hsic":     "#2ca02c",
    "pearson":  "#ff7f0e",
    "spearman": "#9467bd",
}
TEST_LABELS = {
    "ii":       "II (this work)",
    "dcor":     "dCor",
    "hsic":     "HSIC",
    "pearson":  "Pearson |r|",
    "spearman": "Spearman |ρ|",
}

# Tests run per setting
SETTING_TESTS = {
    "A": ["ii", "dcor", "hsic", "pearson", "spearman"],
    "B": ["ii", "dcor", "hsic"],
}
SETTING_DX = {"A": 1, "B": 5}
SETTING_DY = {"A": 1, "B": 3}

ALPHAS = [0.01, 0.05, 0.10]   # pre-compute power at all three


# ---------------------------------------------------------------------------
# Data generating process
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


def generate_sample(relationship_type, n, dim_x, dim_y,
                    noise_level, rng, linear_A=None):
    # type: (str, int, int, int, float, np.random.Generator, Optional[np.ndarray]) -> Tuple
    X   = rng.standard_normal((n, dim_x))
    eps = noise_level * rng.standard_normal((n, dim_y))

    if relationship_type == "independent":
        Y = rng.standard_normal((n, dim_y))
    elif relationship_type == "linear":
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
# Test statistics
# ---------------------------------------------------------------------------

def _double_center(D):
    # type: (np.ndarray) -> np.ndarray
    """Double-center a distance matrix for dCor computation."""
    row_mean   = D.mean(axis=1, keepdims=True)
    col_mean   = D.mean(axis=0, keepdims=True)
    grand_mean = D.mean()
    return D - row_mean - col_mean + grand_mean


def _center_kernel(K):
    # type: (np.ndarray) -> np.ndarray
    """Centre a kernel matrix: H @ K @ H where H = I - (1/n)*11^T."""
    n          = K.shape[0]
    row_mean   = K.mean(axis=1, keepdims=True)
    col_mean   = K.mean(axis=0, keepdims=True)
    grand_mean = K.mean()
    return K - row_mean - col_mean + grand_mean


def _rbf_kernel(X, sigma):
    # type: (np.ndarray, float) -> np.ndarray
    D2 = cdist(X, X, metric="sqeuclidean")
    return np.exp(-D2 / (2.0 * sigma ** 2))


def ii_stat_from_precomputed(nn_idx, D_Y):
    # type: (np.ndarray, np.ndarray) -> float
    """Compute II from precomputed nn_idx and full Y-distance matrix."""
    n      = len(nn_idx)
    d_nn   = D_Y[np.arange(n), nn_idx]          # (n,) distance to X-NN in Y
    ranks  = np.sum(D_Y <= d_nn[:, np.newaxis], axis=1)  # (n,)
    return 2.0 * float(np.sum(ranks)) / (n * n)


def dcor_stat_from_precomputed(AX, D_Y):
    # type: (np.ndarray, np.ndarray) -> float
    """Compute dCor given pre-double-centred AX and raw D_Y."""
    AY     = _double_center(D_Y)
    n      = AX.shape[0]
    dCov2  = float(np.sum(AX * AY)) / n ** 2
    dVar_X = float(np.sum(AX * AX)) / n ** 2
    dVar_Y = float(np.sum(AY * AY)) / n ** 2
    denom  = dVar_X * dVar_Y
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(dCov2 / np.sqrt(denom), 0.0)))


def hsic_stat_from_precomputed(KX_c, K_Y):
    # type: (np.ndarray, np.ndarray) -> float
    """Compute HSIC given pre-centred KX_c and raw (uncentred) KY."""
    KY_c = _center_kernel(K_Y)
    n    = KX_c.shape[0]
    return float(np.sum(KX_c * KY_c)) / (n - 1) ** 2


# ---------------------------------------------------------------------------
# Precomputed structures (built once per dataset, reused across P permutations)
# ---------------------------------------------------------------------------

class PrecomputedDataset(object):
    """Precompute all expensive structures for one (X, Y) dataset."""

    def __init__(self, X, Y, tests):
        # type: (np.ndarray, np.ndarray, List[str]) -> None
        n = X.shape[0]

        # --- II ---
        if "ii" in tests:
            tree_X    = cKDTree(X)
            _, idx    = tree_X.query(X, k=2)
            self.nn_idx = idx[:, 1]           # (n,) nearest neighbour in X
            self.D_Y    = cdist(Y, Y)         # (n, n) all Y distances
            np.fill_diagonal(self.D_Y, np.inf)

        # --- dCor ---
        if "dcor" in tests:
            self.D_X    = cdist(X, X)
            self.AX     = _double_center(self.D_X)
            if "ii" not in tests:
                self.D_Y = cdist(Y, Y)

        # --- HSIC ---
        if "hsic" in tests:
            D_X_sq  = cdist(X, X, metric="sqeuclidean")
            D_Y_sq  = cdist(Y, Y, metric="sqeuclidean")
            # Median heuristic (computed once, fixed across permutations)
            nonzero_X = D_X_sq[D_X_sq > 0]
            nonzero_Y = D_Y_sq[D_Y_sq > 0]
            self.sigma_X = float(np.sqrt(np.median(nonzero_X) / 2.0))
            self.sigma_Y = float(np.sqrt(np.median(nonzero_Y) / 2.0))
            self.KX_raw  = np.exp(-D_X_sq / (2.0 * self.sigma_X ** 2))
            self.KX_c    = _center_kernel(self.KX_raw)   # fixed
            self.KY_raw  = np.exp(-D_Y_sq / (2.0 * self.sigma_Y ** 2))
            if not hasattr(self, "D_Y"):
                self.D_Y = cdist(Y, Y)

        # --- Pearson / Spearman (univariate only) ---
        if "pearson" in tests or "spearman" in tests:
            self.x1d = X[:, 0]
            self.y1d = Y[:, 0]

        self.n     = n
        self.tests = tests

    def observed_stats(self):
        # type: () -> Dict[str, float]
        """Compute all test statistics on the original (unpermuted) data."""
        stats = {}
        if "ii" in self.tests:
            D_Y_obs = self.D_Y.copy()
            np.fill_diagonal(D_Y_obs, np.inf)
            stats["ii"] = ii_stat_from_precomputed(self.nn_idx, D_Y_obs)
        if "dcor" in self.tests:
            stats["dcor"] = dcor_stat_from_precomputed(self.AX, self.D_Y)
        if "hsic" in self.tests:
            stats["hsic"] = hsic_stat_from_precomputed(self.KX_c, self.KY_raw)
        if "pearson" in self.tests:
            r, _ = pearsonr(self.x1d, self.y1d)
            stats["pearson"] = abs(r)
        if "spearman" in self.tests:
            rho, _ = spearmanr(self.x1d, self.y1d)
            stats["spearman"] = abs(rho)
        return stats

    def permuted_stats(self, perm):
        # type: (np.ndarray) -> Dict[str, float]
        """
        Compute all test statistics after permuting rows of Y by perm.
        Reuses precomputed X structures — only Y-side matrices are reindexed.
        """
        stats = {}
        n = self.n

        if "ii" in self.tests:
            # D_Y_perm[i,j] = D_Y[perm[i], perm[j]]
            D_Y_perm = self.D_Y[np.ix_(perm, perm)]
            np.fill_diagonal(D_Y_perm, np.inf)
            stats["ii"] = ii_stat_from_precomputed(self.nn_idx, D_Y_perm)

        if "dcor" in self.tests:
            D_Y_perm = self.D_Y[np.ix_(perm, perm)] \
                if "ii" not in self.tests \
                else self.D_Y[np.ix_(perm, perm)]
            stats["dcor"] = dcor_stat_from_precomputed(self.AX, D_Y_perm)

        if "hsic" in self.tests:
            KY_perm = self.KY_raw[np.ix_(perm, perm)]
            stats["hsic"] = hsic_stat_from_precomputed(self.KX_c, KY_perm)

        if "pearson" in self.tests:
            r, _ = pearsonr(self.x1d, self.y1d[perm])
            stats["pearson"] = abs(r)

        if "spearman" in self.tests:
            rho, _ = spearmanr(self.x1d, self.y1d[perm])
            stats["spearman"] = abs(rho)

        return stats


# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------

def run_experiment(args):
    # type: (argparse.Namespace) -> None
    rng     = np.random.default_rng(args.random_seed)
    tests   = SETTING_TESTS[args.setting]
    dim_x   = SETTING_DX[args.setting]
    dim_y   = SETTING_DY[args.setting]
    B, P    = args.B, args.P
    n       = args.n_samples

    linear_A = _make_linear_A(dim_x, dim_y) \
               if args.relationship_type == "linear" else None

    # Storage arrays
    pvalues    = {t: np.zeros(B) for t in tests}
    statistics = {t: np.zeros(B) for t in tests}
    null_mean  = {t: np.zeros(B) for t in tests}
    null_std   = {t: np.zeros(B) for t in tests}

    print("Exp 6 | setting={} dist={} n={} noise={} B={} P={}".format(
        args.setting, args.relationship_type, n,
        args.noise_level, B, P), flush=True)

    for b in range(B):
        X, Y = generate_sample(
            args.relationship_type, n, dim_x, dim_y,
            args.noise_level, rng, linear_A=linear_A)

        pc = PrecomputedDataset(X, Y, tests)

        # Observed statistics
        obs = pc.observed_stats()

        # Permutation null
        perm_stats = {t: np.zeros(P) for t in tests}
        for p in range(P):
            perm = rng.permutation(n)
            ps   = pc.permuted_stats(perm)
            for t in tests:
                perm_stats[t][p] = ps[t]

        # P-values and null summaries
        for t in tests:
            statistics[t][b] = obs[t]
            null_mean[t][b]  = float(np.mean(perm_stats[t]))
            null_std[t][b]   = float(np.std(perm_stats[t]))
            if t == "ii":
                # Small II -> dependent; reject when II_obs < null
                pvalues[t][b] = float(np.mean(perm_stats[t] <= obs[t]))
            else:
                # Large stat -> dependent; reject when stat_obs > null
                pvalues[t][b] = float(np.mean(perm_stats[t] >= obs[t]))

        if (b + 1) % 20 == 0:
            # Quick progress report
            pw = {t: float(np.mean(pvalues[t][:b+1] < 0.05))
                  for t in tests}
            print("  b={}/{} power@0.05={}".format(
                b + 1, B,
                " ".join("{}:{:.2f}".format(t, pw[t]) for t in tests)),
                flush=True)

    # Pre-compute power at all alphas
    power = {}
    for t in tests:
        power[t] = {a: float(np.mean(pvalues[t] < a)) for a in ALPHAS}

    result = {
        # --- Metadata ---
        "relationship_type": args.relationship_type,
        "setting":           args.setting,
        "n_samples":         n,
        "noise_level":       args.noise_level,
        "dim_x":             dim_x,
        "dim_y":             dim_y,
        "B":                 B,
        "P":                 P,
        "alphas":            ALPHAS,
        "random_seed":       args.random_seed,
        "timestamp":         datetime.datetime.now().isoformat(),
        "tests_run":         tests,
        # --- Raw results ---
        "statistics":        statistics,   # {test: ndarray(B)}
        "pvalues":           pvalues,      # {test: ndarray(B)}
        "null_summary":      {"mean": null_mean, "std": null_std},
        # --- Derived ---
        "power":             power,        # {test: {alpha: float}}
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)),
                exist_ok=True)
    with open(args.output_file, "wb") as fh:
        pickle.dump(result, fh)

    print("[done] Saved: {}".format(args.output_file))
    for t in tests:
        print("  {}  power@0.05={:.3f}  power@0.10={:.3f}".format(
            t, power[t][0.05], power[t][0.10]))


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


def _pivot(rows, setting, alpha=0.05):
    # type: (list, str, float) -> Dict
    """
    data[(dist, n, noise)][test] = power
    """
    subset = [r for r in rows if r.get("setting") == setting]
    data   = {}  # type: Dict
    for r in subset:
        key = (r["relationship_type"], r["n_samples"], r["noise_level"])
        data[key] = {}
        for t in r["tests_run"]:
            pw = r["power"].get(t, {})
            data[key][t] = pw.get(alpha, float("nan"))
    return data


# ---------------------------------------------------------------------------
# Plot 1 — Power vs n curves
# ---------------------------------------------------------------------------

def plot_power_vs_n(rows, plots_dir, setting, alpha=0.05):
    # type: (list, str, str, float) -> None
    """
    One figure per (setting, noise level).
    Panels = relationship types (D1-D9, excluding D0 independent).
    Lines  = tests.
    """
    os.makedirs(plots_dir, exist_ok=True)
    tests  = SETTING_TESTS[setting]
    subset = [r for r in rows if r.get("setting") == setting]
    if not subset:
        return

    noises   = sorted({r["noise_level"] for r in subset})
    n_values = sorted({r["n_samples"]   for r in subset})
    func_dists = [d for d in DISTRIBUTIONS if d != "independent"]

    for noise in noises:
        nrows, ncols = 3, 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.0 * nrows),
                                 sharey=True)
        noise_label = str(noise).replace(".", "-")
        fig.suptitle(
            "Exp 6 — Power vs $n$   (Setting {}, $\\sigma_\\varepsilon={}$, "
            "$\\alpha={}$)".format(setting, noise, alpha),
            fontsize=15, fontweight="bold",
        )

        handles, labels = [], []
        built = False

        for idx, dist in enumerate(func_dists):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            for t in tests:
                powers = []
                for n in n_values:
                    key = (dist, n, noise)
                    found = next(
                        (r for r in subset
                         if r["relationship_type"] == dist
                         and r["n_samples"] == n
                         and abs(r["noise_level"] - noise) < 1e-9),
                        None)
                    if found is not None:
                        powers.append(found["power"].get(t, {}).get(alpha, float("nan")))
                    else:
                        powers.append(float("nan"))

                line, = ax.plot(n_values, powers,
                                marker="o", ms=5, lw=2.0,
                                color=TEST_COLOURS[t],
                                label=TEST_LABELS[t])
                if not built:
                    handles.append(line)
                    labels.append(TEST_LABELS[t])

            built = True
            ax.axhline(alpha, color="grey", lw=1.2, ls="--", alpha=0.6)
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.05)
            ax.set_title(DIST_LABELS[dist], fontsize=12, fontweight="bold")
            ax.set_xlabel("$n$", fontsize=11)
            if col == 0:
                ax.set_ylabel("Power", fontsize=11)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=10)

        fig.legend(handles, labels,
                   loc="lower center",
                   bbox_to_anchor=(0.5, -0.03),
                   ncol=len(tests),
                   fontsize=11,
                   framealpha=0.95)
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])

        fname = "exp6_power_vs_n_setting{}_noise{}.".format(setting, noise_label)
        for ext in ("pdf", "png"):
            out = os.path.join(plots_dir, fname + ext)
            fig.savefig(out, bbox_inches="tight", dpi=300)
            print("Saved: {}".format(out))
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — Power heatmap (tests x relationships, fixed n and noise)
# ---------------------------------------------------------------------------

def plot_power_heatmap(rows, plots_dir, setting,
                       ref_n=500, ref_noise=0.5, alpha=0.05):
    # type: (list, str, str, int, float, float) -> None
    """
    Rows = tests, columns = relationships (D1-D9).
    Fixed at ref_n and ref_noise.
    """
    os.makedirs(plots_dir, exist_ok=True)
    tests      = SETTING_TESTS[setting]
    func_dists = [d for d in DISTRIBUTIONS if d != "independent"]
    subset     = [r for r in rows if r.get("setting") == setting]

    # Find closest available n and noise to reference
    avail_n     = sorted({r["n_samples"]   for r in subset})
    avail_noise = sorted({r["noise_level"] for r in subset})
    use_n     = min(avail_n,     key=lambda x: abs(x - ref_n))
    use_noise = min(avail_noise, key=lambda x: abs(x - ref_noise))

    matrix = np.full((len(tests), len(func_dists)), float("nan"))
    for j, dist in enumerate(func_dists):
        rec = next(
            (r for r in subset
             if r["relationship_type"] == dist
             and r["n_samples"] == use_n
             and abs(r["noise_level"] - use_noise) < 1e-9),
            None)
        if rec is None:
            continue
        for i, t in enumerate(tests):
            matrix[i, j] = rec["power"].get(t, {}).get(alpha, float("nan"))

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(func_dists)),
                                   max(3, 0.9 * len(tests))))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Power ($\\alpha={}$)".format(alpha))

    ax.set_xticks(range(len(func_dists)))
    ax.set_xticklabels([DIST_LABELS[d] for d in func_dists],
                       rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(len(tests)))
    ax.set_yticklabels([TEST_LABELS[t] for t in tests], fontsize=11)

    for i in range(len(tests)):
        for j in range(len(func_dists)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, "{:.2f}".format(v),
                        ha="center", va="center",
                        fontsize=9,
                        color="black" if v < 0.6 else "white")

    ax.set_title(
        "Exp 6 — Power heatmap (Setting {}, $n={}$, "
        "$\\sigma_\\varepsilon={}$, $\\alpha={}$)".format(
            setting, use_n, use_noise, alpha),
        fontsize=13, fontweight="bold")
    plt.tight_layout()

    fname = "exp6_heatmap_setting{}_n{}_noise{}.".format(
        setting, use_n, str(use_noise).replace(".", "-"))
    for ext in ("pdf", "png"):
        out = os.path.join(plots_dir, fname + ext)
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print("Saved: {}".format(out))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3 — Power vs noise (fixed n)
# ---------------------------------------------------------------------------

def plot_power_vs_noise(rows, plots_dir, setting, ref_n=500, alpha=0.05):
    # type: (list, str, str, int, float) -> None
    """
    One figure per setting, panels = relationships (D1-D9), lines = tests.
    Fixed at ref_n.
    """
    os.makedirs(plots_dir, exist_ok=True)
    tests      = SETTING_TESTS[setting]
    func_dists = [d for d in DISTRIBUTIONS if d != "independent"]
    subset     = [r for r in rows if r.get("setting") == setting]

    avail_n = sorted({r["n_samples"] for r in subset})
    use_n   = min(avail_n, key=lambda x: abs(x - ref_n))
    noises  = sorted({r["noise_level"] for r in subset})

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.0 * nrows),
                             sharey=True)
    fig.suptitle(
        "Exp 6 — Power vs $\\sigma_\\varepsilon$   "
        "(Setting {}, $n={}$, $\\alpha={}$)".format(setting, use_n, alpha),
        fontsize=15, fontweight="bold",
    )

    handles, labels = [], []
    built = False

    for idx, dist in enumerate(func_dists):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        for t in tests:
            powers = []
            for noise in noises:
                rec = next(
                    (r for r in subset
                     if r["relationship_type"] == dist
                     and r["n_samples"] == use_n
                     and abs(r["noise_level"] - noise) < 1e-9),
                    None)
                if rec is not None:
                    powers.append(rec["power"].get(t, {}).get(alpha, float("nan")))
                else:
                    powers.append(float("nan"))

            line, = ax.plot(noises, powers,
                            marker="o", ms=5, lw=2.0,
                            color=TEST_COLOURS[t])
            if not built:
                handles.append(line)
                labels.append(TEST_LABELS[t])

        built = True
        ax.axhline(alpha, color="grey", lw=1.2, ls="--", alpha=0.6)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(DIST_LABELS[dist], fontsize=12, fontweight="bold")
        ax.set_xlabel("$\\sigma_\\varepsilon$", fontsize=11)
        if col == 0:
            ax.set_ylabel("Power", fontsize=11)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=10)

    fig.legend(handles, labels,
               loc="lower center",
               bbox_to_anchor=(0.5, -0.03),
               ncol=len(tests),
               fontsize=11,
               framealpha=0.95)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    fname = "exp6_power_vs_noise_setting{}_n{}.".format(setting, use_n)
    for ext in ("pdf", "png"):
        out = os.path.join(plots_dir, fname + ext)
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print("Saved: {}".format(out))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot mode dispatcher
# ---------------------------------------------------------------------------

def plot_mode(args):
    # type: (argparse.Namespace) -> None
    rows = _load_results(args.results_dir)
    if not rows:
        print("ERROR: no .pkl files found in {!r}".format(args.results_dir))
        sys.exit(1)
    print("Loaded {} result files.".format(len(rows)))

    for setting in ("A", "B"):
        has = [r for r in rows if r.get("setting") == setting]
        if not has:
            print("No results for setting {}. Skipping.".format(setting))
            continue
        print("Plotting setting {} ({} files)...".format(setting, len(has)))
        plot_power_vs_n(rows, args.plots_dir, setting)
        plot_power_heatmap(rows, args.plots_dir, setting)
        plot_power_vs_noise(rows, args.plots_dir, setting)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp 6 — Power comparison: II vs dCor, HSIC, Pearson, Spearman.")
    parser.add_argument("--mode", choices=["run", "plot"], required=True)

    # run args
    parser.add_argument("--setting",          choices=["A", "B"], default="A")
    parser.add_argument("--relationship_type", choices=DISTRIBUTIONS,
                        default="linear")
    parser.add_argument("--n_samples",        type=int,   default=500)
    parser.add_argument("--noise_level",      type=float, default=0.5)
    parser.add_argument("--B",                type=int,   default=100,
                        help="Number of datasets per cell.")
    parser.add_argument("--P",                type=int,   default=99,
                        help="Number of permutations per dataset.")
    parser.add_argument("--output_file",      type=str,
                        default="results/out.pkl")
    parser.add_argument("--random_seed",      type=int,   default=42)

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
