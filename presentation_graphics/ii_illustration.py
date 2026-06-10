"""
presentation_graphics/ii_illustration.py
-----------------------------------------
Generates TWO II illustration figures:

  Fig 1 — ii_relationships.pdf/png
      2×2 grid: noise, linear, quadratic, spiral scatter plots

  Fig 2 — ii_scatter.pdf/png
      II(X→Y) vs II(Y→X) scatter showing asymmetry / feature selection

Run from repo root:
    python presentation_graphics/ii_illustration.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.weight":          "bold",
    "axes.labelweight":     "bold",
    "axes.titleweight":     "bold",
    "axes.labelsize":       14,
    "axes.titlesize":       15,
    "xtick.labelsize":      12,
    "ytick.labelsize":      12,
    "legend.fontsize":      12,
    "legend.title_fontsize":13,
    "figure.dpi":           150,
    "axes.linewidth":       1.4,
    "lines.linewidth":      2.2,
})

# ---------------------------------------------------------------------------
# Colours (darker than defaults)
# ---------------------------------------------------------------------------
C_NOISE   = "#1a5fa8"   # deep blue
C_LINEAR  = "#b85c00"   # dark orange
C_QUAD    = "#6b0000"   # dark red
C_SPIRAL  = "#4a0080"   # deep purple
C_GREEN   = "#145214"   # dark green  (fit line)

# ---------------------------------------------------------------------------
# II estimator
# ---------------------------------------------------------------------------

def compute_ii(X, Y):
    """II(X → Y). Accepts 1-D or 2-D numpy arrays. Returns scalar."""
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    n = X.shape[0]
    tree   = cKDTree(X)
    _, idx = tree.query(X, k=2)
    nn_idx = idx[:, 1]
    D_Y    = cdist(Y, Y)
    np.fill_diagonal(D_Y, np.inf)
    d_nn   = D_Y[np.arange(n), nn_idx]
    ranks  = np.sum(D_Y <= d_nn[:, np.newaxis], axis=1)
    return 2.0 * float(np.sum(ranks)) / (n * n)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
n   = 600

# Independent noise
X_noise  = rng.standard_normal(n)
Y_noise  = rng.standard_normal(n)

# Linear (noiseless — II ≈ 0 both ways)
X_lin    = rng.standard_normal(n)
Y_lin    = X_lin + 0.02 * rng.standard_normal(n)

# Linear noisy (moderate symmetric II)
X_lnoise = rng.standard_normal(n)
Y_lnoise = X_lnoise + 0.3 * rng.standard_normal(n)

# Quadratic (moderate noise — X→Y good, Y→X bad: non-invertible)
X_quad   = np.linspace(-2.5, 2.5, n)
Y_quad   = X_quad ** 2 + 0.4 * rng.standard_normal(n)

# Spiral: X = [t·cos t, t·sin t],  Y = t (normalised to [0,1])
# Inverted: t goes from large to small so spiral is wide at bottom, narrow at top
# Small noise added to X to make it look more realistic
t        = np.linspace(4.5 * np.pi, 0.5, n)   # reversed: large radius first
noise_sp = 0.3
x1s      = t * np.cos(t) + noise_sp * rng.standard_normal(n)
x2s      = t * np.sin(t) + noise_sp * rng.standard_normal(n)
X_sp2    = np.column_stack([x1s, x2s])
X_sp1    = x1s[:, np.newaxis]
Y_sp     = ((t - t.min()) / (t.max() - t.min()))[:, np.newaxis]  # still [0,1]


# ---------------------------------------------------------------------------
# Compute II
# ---------------------------------------------------------------------------

ii = {}
ii["noise"]   = (compute_ii(X_noise,  Y_noise),  compute_ii(Y_noise,  X_noise))
ii["linear"]  = (compute_ii(X_lin,    Y_lin),    compute_ii(Y_lin,    X_lin))
ii["lnoisy"]  = (compute_ii(X_lnoise, Y_lnoise), compute_ii(Y_lnoise, X_lnoise))
ii["quad"]    = (compute_ii(X_quad,   Y_quad),   compute_ii(Y_quad,   X_quad))
ii["spiral2"] = (compute_ii(X_sp2,    Y_sp),     compute_ii(Y_sp,     X_sp2))
ii["spiral1"] = (compute_ii(X_sp1,    Y_sp),     compute_ii(Y_sp,     X_sp1))

print("II values:")
for k, (xy, yx) in ii.items():
    print(f"  {k:10s}  II(X→Y)={xy:.3f}  II(Y→X)={yx:.3f}")


# ===========================================================================
# FIGURE 1 — Relationship scatter plots
# ===========================================================================

fig1, axes = plt.subplots(2, 2, figsize=(11, 9))
fig1.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.09,
                     hspace=0.45, wspace=0.35)

# ── noise ──────────────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.scatter(X_noise, Y_noise, s=8, alpha=0.6, color=C_NOISE, rasterized=True)
ax.set_title("Noise (independent)")
ax.set_xlabel("$X$")
ax.set_ylabel("$Y$")
ax.grid(alpha=0.2)

# ── linear noisy ────────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.scatter(X_lnoise, Y_lnoise, s=8, alpha=0.6, color=C_LINEAR, rasterized=True)
xr = np.array([X_lnoise.min(), X_lnoise.max()])
m  = np.polyfit(X_lnoise, Y_lnoise, 1)
ax.plot(xr, np.polyval(m, xr), color=C_GREEN, lw=2.5)
ax.set_title("Linear (noisy)")
ax.set_xlabel("$X$")
ax.set_ylabel("$Y$")
ax.grid(alpha=0.2)

# ── quadratic ───────────────────────────────────────────────────────────────
ax = axes[1, 0]
xsort = np.argsort(X_quad)
ax.scatter(X_quad, Y_quad, s=8, alpha=0.6, color=C_QUAD, rasterized=True)
xr_q = np.linspace(-2.5, 2.5, 300)
ax.plot(xr_q, xr_q ** 2, color=C_QUAD, lw=2.5, alpha=0.5)  # reference curve
ax.set_title("Quadratic ($Y = X^2$)")
ax.set_xlabel("$X$")
ax.set_ylabel("$Y$")
ax.grid(alpha=0.2)

# ── spiral (3-D) ─────────────────────────────────────────────────────────────
ax4 = fig1.add_subplot(2, 2, 4, projection="3d")
# colour by Y value: wide (Y≈0) → narrow (Y≈1)
colors_sp = plt.cm.plasma(Y_sp[:, 0])
ax4.scatter(x2s, x1s, Y_sp[:, 0], c=colors_sp, s=6, rasterized=True)
ax4.set_title("Spiral")
ax4.set_xlabel("$X_2$", labelpad=1)
ax4.set_ylabel("$X_1$", labelpad=1)
ax4.set_zlabel("$Y$",   labelpad=1)
ax4.tick_params(labelsize=10)
ax4.set_box_aspect([1, 1, 0.85])
ax4.set_zlim(0, 1)
ax4.xaxis.pane.fill = False
ax4.yaxis.pane.fill = False
ax4.zaxis.pane.fill = False
ax4.view_init(elev=20, azim=-60)   # angle: wide base visible at bottom

# remove the blank 2,2 cell that was replaced by 3-D axes
axes[1, 1].set_visible(False)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"ii_relationships.{ext}"
    fig1.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved: {out}")
plt.close(fig1)


# ===========================================================================
# FIGURE 2 — II(X→Y) vs II(Y→X) scatter
# ===========================================================================

fig2, ax = plt.subplots(figsize=(7, 7))
fig2.subplots_adjust(left=0.13, right=0.95, top=0.93, bottom=0.11)

# reference lines
ax.plot([0, 1], [0, 1], ls="--", color="#555555", lw=1.6, alpha=0.7,
        zorder=1, label="_nolegend_")
ax.axhline(1.0, color="#888888", lw=1.0, ls=":", alpha=0.5, zorder=1)
ax.axvline(1.0, color="#888888", lw=1.0, ls=":", alpha=0.5, zorder=1)

# ── fixed markers ──────────────────────────────────────────────────────────
STYLE = {
    "noise":  dict(color=C_NOISE,         marker="o", s=220, zorder=5),
    "lnoisy": dict(color=C_LINEAR,        marker="v", s=220, zorder=5),
    "linear": dict(color="#145214",       marker="s", s=220, zorder=5),
    "quad":   dict(color=C_QUAD,          marker="p", s=260, zorder=5),
}
LABELS = {
    "noise":  "Noise",
    "lnoisy": "Linear noisy",
    "linear": "Linear",
    "quad":   "Quadratic",
}

for key, style in STYLE.items():
    x, y = ii[key]
    ax.scatter(x, y, **style, label=LABELS[key])

# ── spiral: two stars joined by a line ─────────────────────────────────────
xs2, ys2 = ii["spiral2"]
xs1, ys1 = ii["spiral1"]
ax.plot([xs2, xs1], [ys2, ys1], color=C_SPIRAL, lw=2.0, zorder=4)
ax.scatter(xs2, ys2, marker="*", color=C_SPIRAL, s=280, zorder=5,
           label="Spiral")
ax.scatter(xs1, ys1, marker="*", color=C_SPIRAL, s=280, zorder=5,
           label="_nolegend_")

# annotations
ax.annotate(r"$X = [x_1, x_2]$",
            xy=(xs2, ys2), xytext=(xs2 - 0.06, ys2 + 0.06),
            color=C_SPIRAL, fontsize=11, fontweight="bold", ha="right",
            arrowprops=dict(arrowstyle="-", color=C_SPIRAL, lw=1.2))
ax.annotate(r"$X = x_1$",
            xy=(xs1, ys1), xytext=(xs1 + 0.05, ys1 + 0.06),
            color=C_SPIRAL, fontsize=11, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-", color=C_SPIRAL, lw=1.2))

# ── legend & axes ──────────────────────────────────────────────────────────
ax.legend(loc="upper left", framealpha=0.93, edgecolor="#aaaaaa",
          markerscale=1.1)

ax.set_xlabel(r"$II(X \to Y)$", fontsize=15, fontweight="bold")
ax.set_ylabel(r"$II(Y \to X)$", fontsize=15, fontweight="bold")
ax.set_xlim(-0.06, 1.12)
ax.set_ylim(-0.06, 1.55)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
ax.grid(alpha=0.25)
for sp in ax.spines.values():
    sp.set_linewidth(1.5)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"ii_scatter.{ext}"
    fig2.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved: {out}")
plt.close(fig2)
