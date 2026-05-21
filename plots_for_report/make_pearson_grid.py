"""
Generate a 2x2 scatterplot grid for the prospectus opening slide.
Top row: relationships classical measures detect.
Bottom row: relationships classical measures miss (rho ~ 0).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

rng = np.random.default_rng(42)
n = 300

# ---- Generate the four datasets ----

# Top-left: linear
x1 = rng.uniform(-3, 3, n)
y1 = 0.8 * x1 + rng.normal(0, 0.4, n)

# Top-right: monotonic nonlinear (cubic-ish)
x2 = rng.uniform(-2, 2, n)
y2 = np.sign(x2) * np.abs(x2) ** 1.5 + rng.normal(0, 0.3, n)

# Bottom-left: parabolic (U-shape)
x3 = rng.uniform(-3, 3, n)
y3 = x3 ** 2 + rng.normal(0, 0.5, n)

# Bottom-right: circular
theta = rng.uniform(0, 2 * np.pi, n)
x4 = np.cos(theta) + rng.normal(0, 0.05, n)
y4 = np.sin(theta) + rng.normal(0, 0.05, n)

datasets = [
    (x1, y1, "Linear"),
    (x2, y2, "Monotonic nonlinear"),
    (x3, y3, "Parabolic"),
    (x4, y4, "Circular"),
]

# ---- Plot ----

fig, axes = plt.subplots(2, 2, figsize=(7, 6.5))
axes = axes.flatten()

for ax, (x, y, label) in zip(axes, datasets):
    rho, _ = pearsonr(x, y)
    ax.scatter(x, y, s=10, alpha=0.6, color="#1f6f6b", edgecolor="none")
    ax.set_title(label, fontsize=13)
    # Put rho in upper-right corner of each panel
    ax.text(
        0.97, 0.95,
        rf"$\rho = {rho:.2f}$",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, pad=3),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("gray")

# Subtle row labels
fig.text(0.02, 0.75, "Pearson\nworks", ha="center", va="center",
         fontsize=11, rotation=90, color="gray", style="italic")
fig.text(0.02, 0.28, "Pearson\nfails", ha="center", va="center",
         fontsize=11, rotation=90, color="gray", style="italic")

plt.tight_layout(rect=[0.05, 0, 1, 1])
plt.savefig("/Users/priyanshichandra/Downloads/Research_Prospectus/plots_for_report/pearson_grid.pdf", bbox_inches="tight")
plt.savefig("/Users/priyanshichandra/Downloads/Research_Prospectus/plots_for_report/pearson_grid.png", dpi=200, bbox_inches="tight")
print("Saved pearson_grid.pdf and pearson_grid.png")
print("\nCorrelations:")
for x, y, label in datasets:
    rho, _ = pearsonr(x, y)
    print(f"  {label:25s}  rho = {rho:+.3f}")
