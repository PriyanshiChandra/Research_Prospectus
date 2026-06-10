#!/usr/bin/env python3
"""
Separate plots: Data distributions (4 panels) + Bidirectional II analysis (single large plot)

Outputs (saved to same directory as this script):
    ii_distributions.pdf / .png
    ii_bidirectional.pdf  / .png
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Global style — bolder, larger fonts throughout
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.weight":          "bold",
    "axes.labelweight":     "bold",
    "axes.titleweight":     "bold",
    "axes.labelsize":       22,
    "axes.titlesize":       22,
    "xtick.labelsize":      16,
    "ytick.labelsize":      16,
    "legend.fontsize":      17,
    "axes.linewidth":       1.6,
    "lines.linewidth":      2.5,
    "figure.dpi":           150,
})


# ---------------------------------------------------------------------------
class SeparateIIVisualization:

    # ── data generation ─────────────────────────────────────────────────────

    def generate_data(self, n, relationship_type):
        if relationship_type == "noise":
            X = np.random.randn(n).reshape(-1, 1)
            Y = np.random.randn(n).reshape(-1, 1)
            return X, Y

        elif relationship_type == "linear":
            x = np.random.randn(n)
            X = x.reshape(-1, 1)
            Y = (2 * x + 1).reshape(-1, 1)
            return X, Y

        elif relationship_type == "linear_noisy":
            x = np.random.randn(n)
            X = x.reshape(-1, 1)
            Y = (2 * x + 1 + np.random.randn(n) * 0.8).reshape(-1, 1)
            return X, Y

        elif relationship_type == "quadratic":
            x = np.random.uniform(-2, 2, n)
            X = x.reshape(-1, 1)
            Y = (x ** 2 - 1).reshape(-1, 1)
            return X, Y

        elif relationship_type == "spiral":
            t = np.linspace(0, 4 * np.pi, n)
            y_spiral = t / (4 * np.pi)
            r = (1 - y_spiral) * 2 + 0.3
            x1 = r * np.cos(t) + np.random.randn(n) * 0.1
            x2 = r * np.sin(t) + np.random.randn(n) * 0.1
            y_spiral += np.random.randn(n) * 0.02
            X1  = x1.reshape(-1, 1)
            X12 = np.column_stack([x1, x2])
            Y   = y_spiral.reshape(-1, 1)
            return X1, x2.reshape(-1, 1), X12, Y

    # ── II calculation ───────────────────────────────────────────────────────

    def calculate_II(self, X, Y):
        n   = X.shape[0]
        D_X = cdist(X, X)
        np.fill_diagonal(D_X, np.inf)
        nn  = np.argmin(D_X, axis=1)           # nearest neighbour in X

        D_Y = cdist(Y, Y)
        np.fill_diagonal(D_Y, np.inf)
        d_nn = D_Y[np.arange(n), nn]           # dist to X-NN in Y-space
        ranks = np.sum(D_Y <= d_nn[:, np.newaxis], axis=1)
        return 2.0 * float(np.sum(ranks)) / (n * n)

    # ── run analysis ─────────────────────────────────────────────────────────

    def run_analysis(self, n_samples=500, n_simulations=50):
        results = {}

        for key, rel in [("noise",        "noise"),
                         ("linear",       "linear"),
                         ("linear_noisy", "linear_noisy"),
                         ("quadratic",    "quadratic")]:
            print(f"Analyzing: {key}")
            xy_vals, yx_vals = [], []
            for _ in range(n_simulations):
                X, Y = self.generate_data(n_samples, rel)
                xy_vals.append(self.calculate_II(X, Y))
                yx_vals.append(self.calculate_II(Y, X))
            results[key] = {"II_XY": np.mean(xy_vals), "II_YX": np.mean(yx_vals)}

        print("Analyzing: spiral")
        x1y, yx1, x12y, yx12 = [], [], [], []
        for _ in range(n_simulations):
            X1, X2, X12, Y = self.generate_data(n_samples, "spiral")
            x1y.append(self.calculate_II(X1,  Y));  yx1.append(self.calculate_II(Y, X1))
            x12y.append(self.calculate_II(X12, Y)); yx12.append(self.calculate_II(Y, X12))

        results["spiral_X1"]  = {"II_XY": np.mean(x1y),  "II_YX": np.mean(yx1)}
        results["spiral_X12"] = {"II_XY": np.mean(x12y), "II_YX": np.mean(yx12)}

        print("\nII values:")
        for k, v in results.items():
            print(f"  {k:15s}  II(X→Y)={v['II_XY']:.3f}  II(Y→X)={v['II_YX']:.3f}")
        return results

    # ── Figure 1: data distributions ─────────────────────────────────────────

    def plot_data_distributions(self, n_samples=500):
        fig = plt.figure(figsize=(12, 10))
        np.random.seed(42)

        # ── noise
        ax1 = plt.subplot(2, 2, 1)
        X, Y = self.generate_data(n_samples, "noise")
        ax1.scatter(X, Y, alpha=0.6, s=20, color="#4A86C8")
        ax1.set_xlabel("$X$"); ax1.set_ylabel("$Y$")
        ax1.set_title("noise")
        ax1.set_xlim(-3, 3); ax1.set_ylim(-3, 3)
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.grid(alpha=0.3)

        # ── linear (noisy scatter + clean line)
        ax2 = plt.subplot(2, 2, 2)
        X_n, Y_n = self.generate_data(n_samples, "linear_noisy")
        ax2.scatter(X_n, Y_n, alpha=0.6, s=20, color="#D4803A")
        xr = np.linspace(X_n.min(), X_n.max(), 200)
        ax2.plot(xr, 2 * xr + 1, color="#3A8A3A", lw=3.0)
        ax2.set_xlabel("$X$"); ax2.set_ylabel("$Y$")
        ax2.set_title("linear")
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.grid(alpha=0.3)

        # ── quadratic
        ax3 = plt.subplot(2, 2, 3)
        X_q, Y_q = self.generate_data(n_samples, "quadratic")
        ax3.scatter(X_q, Y_q, alpha=0.6, s=20, color="#B03535")
        xr = np.linspace(-2, 2, 300)
        ax3.plot(xr, xr ** 2 - 1, color="#B03535", lw=3.0)
        ax3.set_xlabel("$X$"); ax3.set_ylabel("$Y$")
        ax3.set_title("quadratic")
        ax3.set_xlim(-2.5, 2.5); ax3.set_ylim(-1.5, 3.5)
        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.grid(alpha=0.3)

        # ── spiral (3-D)
        ax4 = plt.subplot(2, 2, 4, projection="3d")
        X1, X2, X12, Y_sp = self.generate_data(n_samples, "spiral")
        ax4.scatter(X1.flatten(), X2.flatten(), Y_sp.flatten(),
                    c=Y_sp.flatten(), cmap="plasma",
                    alpha=0.7, s=20, edgecolors="k", linewidth=0.2)
        ax4.set_xlabel("$X_1$", labelpad=5)
        ax4.set_ylabel("$X_2$", labelpad=5)
        ax4.set_zlabel("$Y$",   labelpad=5)
        ax4.set_title("spiral")
        ax4.view_init(elev=25, azim=45)
        ax4.set_xticks([]); ax4.set_yticks([])
        ax4.grid(alpha=0.3)

        plt.tight_layout()

        for ext in ("pdf", "png"):
            out = OUT_DIR / f"ii_distributions.{ext}"
            fig.savefig(out, bbox_inches="tight", dpi=300)
            print(f"Saved: {out}")
        plt.close(fig)

    # ── Figure 2: bidirectional II ────────────────────────────────────────────

    def plot_ii_analysis(self, results):
        # Extra right margin so the external legend fits without clipping
        fig, ax = plt.subplots(figsize=(9, 8))
        fig.subplots_adjust(left=0.12, right=0.72, top=0.95, bottom=0.11)

        # ── reference diagonal
        ax.plot([0, 1], [0, 1], "k--", lw=2.5, alpha=0.5, zorder=1)

        # ── marker styles (medium saturation — readable but not harsh)
        STYLES = {
            "noise":        dict(marker="o", color="#4A86C8", s=500),
            "linear_noisy": dict(marker="v", color="#D4803A", s=500),
            "linear":       dict(marker="s", color="#3A8A3A", s=500),
            "quadratic":    dict(marker="p", color="#B03535", s=550),
        }
        for key, style in STYLES.items():
            if key not in results:
                continue
            ax.scatter(results[key]["II_XY"], results[key]["II_YX"],
                       edgecolors="black", linewidth=1.8,
                       alpha=0.9, zorder=3, **style)

        # ── spiral: two stars connected by a line
        r12 = results.get("spiral_X12")
        r1  = results.get("spiral_X1")
        if r12 and r1:
            ax.plot([r12["II_XY"], r1["II_XY"]],
                    [r12["II_YX"], r1["II_YX"]],
                    color="#7B3FBE", lw=3.0, alpha=0.8, zorder=2)
            ax.scatter(r12["II_XY"], r12["II_YX"],
                       marker="*", color="#7B3FBE", s=650,
                       edgecolors="black", linewidth=1.8, zorder=4)
            ax.scatter(r1["II_XY"], r1["II_YX"],
                       marker="*", color="#7B3FBE", s=650,
                       edgecolors="black", linewidth=1.8, zorder=4)

            # ── annotations: placed to avoid the legend area (right of axes)
            ax.annotate(
                r"$X = [x_1, x_2]$",
                xy=(r12["II_XY"], r12["II_YX"]),
                xytext=(55, 0), textcoords="offset points",
                fontsize=15, fontweight="bold", color="#7B3FBE", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="#7B3FBE", lw=1.8, alpha=0.95),
            )
            ax.annotate(
                r"$X = x_1$",
                xy=(r1["II_XY"], r1["II_YX"]),
                xytext=(20, 0), textcoords="offset points",
                fontsize=15, fontweight="bold", color="#7B3FBE", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="#7B3FBE", lw=1.8, alpha=0.95),
            )

        # ── axes formatting
        ax.set_xlabel(r"$II(X \rightarrow Y)$", fontsize=24, fontweight="bold")
        ax.set_ylabel(r"$II(Y \rightarrow X)$", fontsize=24, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3, linewidth=1)
        ax.tick_params(labelsize=18)
        for sp in ax.spines.values():
            sp.set_linewidth(1.6)

        # ── legend placed OUTSIDE the axes to the right
        #    bbox_to_anchor is in axes-fraction coordinates; (1.03, 0.5) = just
        #    right of the right edge, vertically centred.
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#4A86C8",
                   markersize=16, markeredgecolor="black", markeredgewidth=1.5,
                   label="noise"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#D4803A",
                   markersize=16, markeredgecolor="black", markeredgewidth=1.5,
                   label="linear noisy"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#3A8A3A",
                   markersize=16, markeredgecolor="black", markeredgewidth=1.5,
                   label="linear"),
            Line2D([0], [0], marker="p", color="w", markerfacecolor="#B03535",
                   markersize=16, markeredgecolor="black", markeredgewidth=1.5,
                   label="quadratic"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#7B3FBE",
                   markersize=20, markeredgecolor="black", markeredgewidth=1.5,
                   label="spiral"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),   # just outside right edge, centred
            framealpha=0.95,
            edgecolor="grey",
            fancybox=True,
            borderpad=0.9,
            labelspacing=0.8,
        )

        for ext in ("pdf", "png"):
            out = OUT_DIR / f"ii_bidirectional.{ext}"
            fig.savefig(out, bbox_inches="tight", dpi=300)
            print(f"Saved: {out}")
        plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    np.random.seed(42)
    analyzer = SeparateIIVisualization()

    print("Creating data distributions plot...")
    analyzer.plot_data_distributions(n_samples=1000)

    print("\nRunning II analysis (this takes a minute)...")
    results = analyzer.run_analysis(n_samples=1000, n_simulations=100)

    print("\nCreating II bidirectional plot...")
    analyzer.plot_ii_analysis(results)

    print("\nDone. Files written to:", OUT_DIR)


if __name__ == "__main__":
    main()
