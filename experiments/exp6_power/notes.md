# Experiment 6 — Statistical Power Comparison

## What This Experiment Measures

Experiment 6 asks: **Is II a powerful test for detecting statistical dependence?**

A test has high *statistical power* if it reliably rejects the null hypothesis of independence when a true dependence exists. We compare the empirical power of the II permutation test against four established competitors:

| Test | Handles multivariate X, Y? | Type |
|---|---|---|
| **II** (Information Imbalance) | Yes | Rank-based, non-parametric |
| **dCor** (Distance Correlation) | Yes | Distance-based, non-parametric |
| **HSIC** (Hilbert-Schmidt Independence Criterion) | Yes | Kernel-based, non-parametric |
| **Pearson** correlation | No — univariate only | Linear, parametric |
| **Spearman** correlation | No — univariate only | Monotone, non-parametric |

All tests use **identical permutation procedures**: for each of B independent datasets, the observed test statistic is compared against a null distribution built from P random permutations of the rows of Y. Power is the fraction of the B datasets for which the permutation p-value falls below a significance level α.

---

## The Ten Relationship Types (D0–D9)

Each relationship defines how Y depends on X (plus additive Gaussian noise ε ~ N(0, σ²·I)).

### D0 — Independent
```
X ~ N(0, I_{d_X}),   Y ~ N(0, I_{d_Y})   (drawn independently)
```
X and Y share **no information**. The population II* = 1 (maximal imbalance). All tests should fail to reject H0 at rate α (the false-positive rate). This serves as a **sanity check**: power should equal α.

---

### D1 — Linear
```
Y = X · A + ε,   A ∈ R^{d_X × d_Y} fixed random matrix
```
A scaled random Gaussian matrix A (entries i.i.d. N(0, 1/d_X)) mixes all dimensions of X into Y. This is the canonical parametric setting where **Pearson/Spearman have maximum power** (the relationship is exactly what they are designed to detect). A good benchmark for II and dCor to match.

---

### D2 — Quadratic
```
Y_j = X_j²  + ε_j   (elementwise, padded if d_Y > d_X)
```
A symmetric, non-monotone relationship. **Pearson correlation is near zero** (positive and negative contributions cancel), so Pearson has low power here. II, dCor, and HSIC should detect this.

---

### D3 — Cubic
```
Y_j = X_j³ + ε_j
```
Odd-powered, monotone but strongly non-linear. Spearman (rank-based) handles monotone relationships well. This tests how close II and dCor track Spearman's power for monotone alternatives.

---

### D4 — Sine
```
Y_j = sin(X_j) + ε_j
```
Oscillatory relationship with multiple local maxima and minima. **Pearson correlation is near zero** (positive/negative half-cycles cancel). The frequency of oscillation interacts with the noise level — at high noise the signal is washed out faster than for monotone relationships.

---

### D5 — Cosine
```
Y_j = cos(X_j) + ε_j
```
Same oscillatory structure as sine but phase-shifted. Numerically very similar to D4 in terms of power; included to verify the tests are invariant to phase shifts as expected.

---

### D6 — Exponential
```
Y_j = exp(X_j / 2) + ε_j
```
Monotone and strongly non-linear — grows rapidly for large X. The division by 2 prevents numerical overflow. At low noise, the non-linearity is mild and Pearson/Spearman can detect it; at high noise the tail behaviour matters.

---

### D7 — Logarithmic
```
Y = log(||X||_2 + 1) · 1_{d_Y}^T + ε
```
A single scalar summary (the log-norm of X) broadcast across all d_Y dimensions. This creates a **rank-deficient** Y–X relationship — all variation in Y comes from a single 1-D projection of X. In higher d_X this is a **low-signal, high-dimension** setting that is hard for all tests.

---

### D8 — Step
```
Y_j = sign(X_j) + ε_j
```
Y is piecewise constant: −1 for negative X, +1 for positive X. The discontinuity is not captured by correlation, but rank-based and kernel methods detect it. Particularly interesting at the boundary between low and high noise — noise blurs the step function and power drops abruptly.

---

### D9 — Parabolic
```
Y_1 = X_1² + X_2²  (or X_1² if d_X = 1),   Y_j = X_{(j mod d_X)}²  for j > 1
```
A radially symmetric relationship: Y depends on the squared norm of subsets of X dimensions. **Pearson and Spearman have zero power** (symmetric around zero means correlations vanish), while distance- and kernel-based methods should detect it. This is the canonical adversarial case for linear and monotone tests.

---

## Experimental Design

### Two Settings

| | Setting A | Setting B |
|---|---|---|
| **d_X** | 1 | 5 |
| **d_Y** | 1 | 3 |
| **Tests compared** | II, dCor, HSIC, Pearson, Spearman | II, dCor, HSIC |
| **Purpose** | Full comparison, univariate baseline | Multivariate regime |

Pearson and Spearman are included only in Setting A because they are defined for scalar (X, Y) only.

### Parameter Grid

```
10 distributions × 6 sample sizes × 4 noise levels × 2 settings = 480 SLURM tasks
```

| Axis | Values |
|---|---|
| n_samples | 50, 100, 200, 500, 1000, 2000 |
| noise_level (σ_ε) | 0.1, 0.5, 1.0, 2.0 |
| significance levels | α = 0.01, 0.05, 0.10 (post-hoc) |

### Adaptive B and P (permutation repetitions)

To keep each array task under 2 hours wall time:

| n | B (datasets) | P (permutations) |
|---|---|---|
| ≤ 200 | 200 | 199 |
| ≤ 1000 | 100 | 99 |
| 2000 | 50 | 99 |

### Permutation Test Procedure

For each of the B independent datasets:

1. Generate (X, Y) from the target relationship
2. Compute observed statistic T_obs for each test
3. For p = 1 … P: permute rows of Y, recompute T_perm
4. **P-value for II**: fraction of null statistics ≤ T_obs (small II = dependent)
5. **P-value for others**: fraction of null statistics ≥ T_obs (large stat = dependent)
6. Reject H0 if p-value < α

Power = fraction of B datasets where H0 is rejected.

### What Is Precomputed for Efficiency

The `PrecomputedDataset` class avoids redundant computation across the P permutations:

| Test | Precomputed once | Reused per permutation |
|---|---|---|
| II | X nearest-neighbour indices, full D_Y matrix | D_Y[perm, :][:, perm] reindexing |
| dCor | D_X, doubly-centred A_X, D_Y | D_Y[perm, :][:, perm] |
| HSIC | Centred K_X, raw K_Y (median σ) | K_Y[perm, :][:, perm] |
| Pearson | x_1d = X[:,0], y_1d = Y[:,0] | y_1d[perm] |
| Spearman | x_1d, y_1d | y_1d[perm] |

---

## Result Storage Format

Each task writes one `.pkl` file to `results/`:

```
results/{relationship}_n{n}_noise{noise_label}_setting{A|B}_task{id}.pkl
```

Each file contains:

```python
{
    # Metadata
    "relationship_type": str,
    "n_samples":         int,
    "noise_level":       float,
    "setting":           "A" or "B",
    "n_datasets":        int,          # B
    "n_permutations":    int,          # P
    "random_seed":       int,
    "dim_x":             int,
    "dim_y":             int,
    "tests":             list[str],

    # Raw results — shape (B,) per test
    "statistics":   {test_name: ndarray},   # observed statistic
    "pvalues":      {test_name: ndarray},   # permutation p-values

    # Null distribution summary — shape (B,) per test, each entry avg over P
    "null_summary": {test_name: {"mean": ndarray, "std": ndarray}},

    # Pre-computed power
    "power":        {test_name: {0.01: float, 0.05: float, 0.10: float}},
}
```

Storing raw p-values means you can:
- Recompute power at any α without rerunning
- Draw ROC curves (power vs α)
- Compute confidence intervals on power estimates
- Aggregate across tasks in any grouping

---

## Output Plots

### 1. `power_vs_n_setting{A|B}.pdf`

A 3×3 panel grid (rows = 3 noise levels 0.1/0.5/1.0, columns = 3 relationship types). Each panel plots **power vs n** for all tests at α=0.05. Shows:
- How quickly each test detects dependence as n grows
- Which tests are competitive at small n (sample efficiency)
- Whether II matches or exceeds dCor/HSIC at various noise levels

### 2. `power_heatmap_setting{A|B}_alpha{α}.pdf`

An imshow heatmap with rows = tests, columns = (relationship × n) combinations. Shows at a glance which (test, n, distribution) combinations achieve high power. Useful for spotting **where II fails** compared to competitors.

### 3. `power_vs_noise_setting{A|B}.pdf`

Plots **power vs noise level** at fixed n=500, α=0.05, one panel per relationship type. Shows how noise degrades power for each test — whether II degrades faster or slower than dCor/HSIC.

---

## How to Run

```bash
# From experiments/exp6_power/
mkdir -p logs results plots

# Submit array job (480 tasks)
sbatch exp6.sh

# After all tasks complete, plot manually if needed:
source ~/measure_comparison_env/bin/activate
python exp6_power.py --mode plot \
    --results_dir results/ \
    --plots_dir   plots/
```

### Checking Progress

```bash
# Count completed result files
ls results/*.pkl | wc -l   # should reach 480

# Check for failures
grep -l "FAILED" logs/ii_power_*.err 2>/dev/null | head

# Tail a specific task log
tail -20 logs/ii_power_{JOB_ID}_{TASK_ID}.out
```

---

## Key Hypotheses

Based on the structure of II:

1. **II should match dCor and HSIC** on symmetric/radial relationships (D2 quadratic, D9 parabolic) where linear and monotone tests fail
2. **II may lag Pearson/Spearman on D1 linear** at very small n, but should catch up as n grows
3. **In Setting B (d_X=5, d_Y=3)**, all tests should show reduced power vs Setting A; the relative ordering may change as the curse of dimensionality affects NN-based (II) vs distance-based (dCor) vs kernel-based (HSIC) methods differently
4. **D0 independent** should show power ≈ α for all tests at all n (false positive calibration check)
5. **High noise (σ=2.0)** should substantially reduce power for all tests on non-linear relationships, but the *relative* ranking across tests reveals which is most robust

---

## Dependencies

```
numpy, scipy, matplotlib, scikit-learn (optional), pickle
utils/ii_estimator.py  (compute_ii_vectorized)
```

The experiment is self-contained — no imports from other exp directories.
