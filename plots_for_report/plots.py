"""
Illustrations of the Information Imbalance coefficient.

Figure 1 (univariate X, Y):
  - Panel A: nearest neighbour found in X-space (look at x-axis only).
  - Panel B: rank computed in Y-space (horizontal band of width 2r
    around Y_i, where r = |Y_i - Y_{N_X(i)}|).

Figure 2: asymmetry of II using Y = X^2.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "mathtext.fontset": "cm",
})

POINT_COLOR  = "#3a3a3a"
FOCAL_COLOR  = "#c0392b"   # red — focal point i
NN_COLOR     = "#2c7fb8"   # blue — nearest neighbour
INSIDE_COLOR = "#f39c12"   # orange — points inside the Y-band


# =============================================================================
# Figure 1: rank computation (univariate X and Y)
# =============================================================================
def figure_rank_computation(savepath="fig1_rank.png"):
    # Manually chosen data for a clean didactic figure:
    # - X-values reasonably spread out (no clumps)
    # - Y-values such that NN-in-X is NOT NN-in-Y, and rank > 1
    rng = np.random.default_rng(0)
    n = 10
    X = np.array([0.6, 1.7, 2.8, 3.6, 4.5, 5.2, 6.1, 7.2, 8.4, 9.3])
    # Underlying trend + small structured noise
    Y = 0.4 * X + 0.7 * np.sin(0.9 * X) + np.array(
        [0.1, -0.3, 0.4, -0.2, 0.5, -0.4, 0.2, -0.1, 0.3, -0.2]
    )

    def rank_for(focal):
        dx = np.abs(X - X[focal]); dx[focal] = np.inf
        nn = int(np.argmin(dx))
        r = abs(Y[focal] - Y[nn])
        inside = (np.abs(Y - Y[focal]) <= r) & (np.arange(n) != focal)
        return int(inside.sum()), nn, r

    # Pick a focal index whose rank is roughly 4 (excluding endpoints)
    candidates = [(rank_for(i)[0], i) for i in range(1, n - 1)]
    focal = sorted(candidates, key=lambda t: (abs(t[0] - 4), -t[0]))[0][1]
    rank, nn, radius = rank_for(focal)

    inside_mask = (np.abs(Y - Y[focal]) <= radius) & (np.arange(n) != focal)
    inside_no_nn = [k for k in range(n) if inside_mask[k] and k != nn]
    outside = [k for k in range(n) if k not in (focal, nn) and not inside_mask[k]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)

    # ---------- Panel A: nearest neighbour in X ----------
    ax = axes[0]

    # Faint vertical guide lines — emphasises "X-axis only"
    for k in range(n):
        ax.axvline(X[k], color="lightgray", lw=0.6, zorder=0)

    ax.scatter(X[outside + inside_no_nn], Y[outside + inside_no_nn],
               c=POINT_COLOR, s=60, zorder=2)
    ax.scatter(X[focal], Y[focal], c=FOCAL_COLOR, s=160, zorder=4,
               edgecolor="black", linewidth=1.2, label=r"$(X_i, Y_i)$")
    ax.scatter(X[nn], Y[nn], c=NN_COLOR, s=160, zorder=4,
               edgecolor="black", linewidth=1.2,
               label=r"$(X_{N_X(i)}, Y_{N_X(i)})$")

    # Horizontal double-headed arrow showing X-distance, near bottom
    y_arrow = Y.min() - 0.7
    arr = FancyArrowPatch((X[focal], y_arrow), (X[nn], y_arrow),
                          arrowstyle="<->", mutation_scale=14,
                          color=NN_COLOR, lw=1.8)
    ax.add_patch(arr)
    ax.text((X[focal] + X[nn]) / 2, y_arrow - 0.25,
            r"closest in $\mathbf{X}$:  $\mathbf{|X_i - X_{N_X(i)}|}$ is smallest",
            ha="center", va="top", color=NN_COLOR, fontsize=10,
            fontweight="bold")

    # Labels on the focal and NN dots only
    ax.annotate(r"$\mathbf{i}$", (X[focal], Y[focal]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=14, color=FOCAL_COLOR)
    ax.annotate(r"$\mathbf{N_X(i)}$", (X[nn], Y[nn]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=14, color=NN_COLOR)

    ax.set_title(r"Panel A: nearest neighbour of $X_i$ in $X$-space")
    ax.set_xlabel(r"$X$"); ax.set_ylabel(r"$Y$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, frameon=False)
    ax.grid(alpha=0.25)
    ax.set_ylim(y_arrow - 1.0, Y.max() + 0.8)

    # ---------- Panel B: rank in Y ----------
    ax = axes[1]

    # Horizontal band of half-width = radius around Y_i.
    # No separate radius annotation — the band itself, plus the fact that
    # the blue NN point sits exactly on the band's edge, makes the radius
    # visually obvious.
    ax.axhspan(Y[focal] - radius, Y[focal] + radius,
               facecolor=INSIDE_COLOR, alpha=0.18, zorder=0)
    ax.axhline(Y[focal] - radius, color=INSIDE_COLOR, lw=1, linestyle=":")
    ax.axhline(Y[focal] + radius, color=INSIDE_COLOR, lw=1, linestyle=":")
    ax.axhline(Y[focal], color=FOCAL_COLOR, lw=0.8, linestyle=":", alpha=0.6)

    ax.scatter(X[outside], Y[outside], c=POINT_COLOR, s=60, zorder=2)
    if inside_no_nn:
        ax.scatter(X[inside_no_nn], Y[inside_no_nn], c=INSIDE_COLOR,
                   s=95, zorder=3, edgecolor="black", linewidth=0.8,
                   label=f"inside band ({len(inside_no_nn)} pts)")
    ax.scatter(X[focal], Y[focal], c=FOCAL_COLOR, s=160, zorder=4,
               edgecolor="black", linewidth=1.2, label=r"$(X_i, Y_i)$")
    ax.scatter(X[nn], Y[nn], c=NN_COLOR, s=160, zorder=4,
               edgecolor="black", linewidth=1.2,
               label=r"$(X_{N_X(i)}, Y_{N_X(i)})$")

    # Labels:
    #   - focal dot: "i"
    #   - NN dot:    "N_X(i)"
    #   - other points inside the band: their rank from Y_i
    #   - points outside the band: no label
    dY = np.abs(Y - Y[focal])
    others = np.array([k for k in range(n) if k != focal])
    order = others[np.argsort(dY[others])]
    y_rank = {k: r for r, k in enumerate(order, start=1)}

    ax.annotate(r"$\mathbf{i}$", (X[focal], Y[focal]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=14, color=FOCAL_COLOR)
    # NN dot: identifier and rank are combined into one formula label,
    # coloured blue to associate it with the NN
    ax.annotate(rf"$\mathbf{{R^Y_i(N_X(i)) = {y_rank[nn]}}}$", (X[nn], Y[nn]),
                xytext=(0, 16), textcoords="offset points",
                ha="center", fontsize=13, color=NN_COLOR)

    for k in inside_no_nn:
        if y_rank[k] == 2:
            offset, va = (0, -16), "top"
        else:
            offset, va = (0, 14), "bottom"
        ax.annotate(rf"$\mathbf{{rank\ {y_rank[k]}}}$", (X[k], Y[k]),
                    xytext=offset, textcoords="offset points",
                    ha="center", va=va, fontsize=12, color="#7a4e00")

    ax.set_title(r"Panel B: rank of $Y_{N_X(i)}$ among $Y$-neighbours of $Y_i$")
    ax.set_xlabel(r"$X$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=3, frameon=False)
    ax.grid(alpha=0.25)
    ax.set_xlim(X.min() - 1.5, X.max() + 0.8)

    fig.suptitle(
        r"Information Imbalance: $\widehat{II}_n(X\to Y)$"
        r"$= \frac{2}{n^2}\sum_{i=1}^n R^Y_i(N_X(i))$",
        y=1.00, fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {savepath}  (focal i={focal}, NN={nn}, rank={rank})")


# =============================================================================
# Figure 2: asymmetry — Y = X^2 example
# =============================================================================
def ii_estimator(A, B):
    n = len(A)
    A = A.reshape(-1, 1) if A.ndim == 1 else A
    B = B.reshape(-1, 1) if B.ndim == 1 else B
    total = 0.0
    for i in range(n):
        dA = np.linalg.norm(A - A[i], axis=1); dA[i] = np.inf
        nn = int(np.argmin(dA))
        rB = np.linalg.norm(B[nn] - B[i])
        dB = np.linalg.norm(B - B[i], axis=1)
        rank = int(((dB <= rB) & (np.arange(n) != i)).sum())
        total += rank
    return 2 * total / (n ** 2)


def figure_asymmetry(seed=3, n=60, savepath="fig2_asymmetry.png"):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=n)
    Y = X ** 2 + rng.normal(0, 0.05, n)

    ii_xy = ii_estimator(X, Y)
    ii_yx = ii_estimator(Y, X)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.scatter(X, Y, c=POINT_COLOR, s=35, zorder=2)
    for i in range(n):
        dx = np.abs(X - X[i]); dx[i] = np.inf
        nn = int(np.argmin(dx))
        ax.plot([X[i], X[nn]], [Y[i], Y[nn]],
                color=NN_COLOR, alpha=0.35, lw=0.9, zorder=1)
    ax.set_title(
        r"$X \to Y$:  NN in $X$ stay close in $Y$" + "\n"
        + r"$\widehat{II}_n(X\to Y) \approx $" + f"{ii_xy:.3f}  (small)"
    )
    ax.set_xlabel(r"$X$"); ax.set_ylabel(r"$Y$")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter(X, Y, c=POINT_COLOR, s=35, zorder=2)
    for i in range(n):
        dy = np.abs(Y - Y[i]); dy[i] = np.inf
        nn = int(np.argmin(dy))
        ax.plot([X[i], X[nn]], [Y[i], Y[nn]],
                color=FOCAL_COLOR, alpha=0.4, lw=0.9, zorder=1)
    ax.set_title(
        r"$Y \to X$:  NN in $Y$ can be far in $X$" + "\n"
        + r"$\widehat{II}_n(Y\to X) \approx $" + f"{ii_yx:.3f}  (large)"
    )
    ax.set_xlabel(r"$X$"); ax.set_ylabel(r"$Y$")
    ax.grid(alpha=0.25)

    fig.suptitle(
        r"Asymmetry of II:  $Y = X^2$  —  $X$ determines $Y$, but not vice versa",
        y=1.02, fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {savepath}  (II(X→Y) = {ii_xy:.3f}, II(Y→X) = {ii_yx:.3f})")


if __name__ == "__main__":
    figure_rank_computation(savepath="/Users/priyanshichandra/Downloads/Research_Prospectus/plots_for_report/fig11_rank.pdf")
    figure_asymmetry(savepath="/Users/priyanshichandra/Downloads/Research_Prospectus/plots_for_report/fig22_asymmetry.pdf")