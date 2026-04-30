# Experiment Setup: Asymptotic Theory of the Information Imbalance Coefficient
# Covers: (1) Consistency experiments, (2) Asymptotic normality experiments

---

## 1. Notation

| Symbol | Description |
|--------|-------------|
| $X \in \mathbb{R}^{d_X}$ | Input random variable, dimension $d_X$ (parameter) |
| $Y \in \mathbb{R}^{d_Y}$ | Target random variable, dimension $d_Y$ (parameter) |
| $(x_i, y_i)_{i=1}^n$ | i.i.d. sample from joint distribution $P_{X,Y}$ |
| $n$ | Sample size — benchmarked at $n \in \{100, 500, 1000, 5000\}$ |
| $B$ | Number of Monte Carlo replications (default: $B = 1000$) |
| $N_X(i)$ | Index of the 1-nearest neighbor of $x_i$ in $X$-space (Euclidean, ties: first minimum index) |
| $R^Y_i(N_X(i))$ | Rank of $y_{N_X(i)}$ among $\{y_k : k \neq i\}$ w.r.t. distance from $y_i$ |
| $\widehat{\mathrm{II}}_n$ | Empirical II estimator (see Section 2) |
| $\mathrm{II}$ | Population-level (true) II — distribution-dependent (see Section 3) |
| $\sigma^2 = n \cdot \mathrm{Var}(\widehat{\mathrm{II}}_n)$ | Asymptotic variance — **currently open, estimated empirically** |
| $\widehat{\sigma}^2_B$ | Empirical variance of $\widehat{\mathrm{II}}_n$ across $B$ replications (used in place of $\sigma^2$) |
| $Z_n$ | Standardized statistic for CLT verification (see Section 4) |
| $\varepsilon \sim \mathcal{N}(0, \sigma^2_\varepsilon)$ | Additive noise in functional relationships |
| $\mathrm{SEED}$ | Global random seed for reproducibility (default: 42) |

---

## 2. Estimator

Given a sample $(x_i, y_i)_{i=1}^n$:

**Step 1 — Nearest neighbor in $X$-space:**
$$N_X(i) = \arg\min_{k \neq i} \|x_k - x_i\|_2$$

**Step 2 — Rank in $Y$-space:**
$$R^Y_i(N_X(i)) = \sum_{j \neq i} \mathbf{1}\left\{\|y_{N_X(i)} - y_i\|_2 \geq \|y_i - y_j\|_2\right\}$$

That is: $R^Y_i(N_X(i))$ counts how many $y_j$ are at least as close to $y_i$ as $y_{N_X(i)}$ is.

**Step 3 — II estimator:**
$$\widehat{\mathrm{II}}_n(X \to Y) = \frac{2}{n^2} \sum_{i=1}^n R^Y_i(N_X(i))$$

Equivalently, writing ranks via the empirical CDF $\hat{F}_{Y_i}(t) = \frac{1}{n}\sum_{j=1}^n \mathbf{1}\{\|y_j - y_i\| \leq t\}$:

$$\widehat{\mathrm{II}}_n = \frac{2}{n} \sum_{i=1}^n \hat{F}_{Y_i}\left(\|y_{N_X(i)} - y_i\|\right) - \frac{2}{n}$$

> **Note on the $-2/n$ term:** This correction arises from excluding self-distances in the rank computation. It vanishes as $n \to \infty$ and does not affect the asymptotic results, but is present in every finite-sample computation.

> **Asymmetry:** $\widehat{\mathrm{II}}_n(X \to Y) \neq \widehat{\mathrm{II}}_n(Y \to X)$ in general. This directional asymmetry is the defining feature of II.

---

## 3. Population Quantity and Asymptotic Limit

The true population-level II is the deterministic quantity:

$$\mathrm{II}(X \to Y) = 2\, P_{\tilde{Y}, Y_1, Y'_1, X_1}\left(\|Y_1 - Y'_1\| \geq \|Y_1 - \tilde{Y}\|\right)$$

where:
- $\tilde{Y} \sim f_Y$ — marginal distribution of $Y$, independent of $X$
- $Y_1, Y_1' \overset{\mathrm{iid}}{\sim} f_{Y|X=X_1}$ — two independent copies from the conditional, tied to the same $X_1$
- $X_1 \sim f_X$ — marginal of $X$

**Known limiting values:**

| Dependency structure | $\mathrm{II}(X \to Y)$ |
|----------------------|------------------------|
| $Y \perp X$ (independence) | $1$ |
| $Y = f(X)$ a.s., $f$ measurable, $Y$ continuous | $0$ |
| Intermediate dependence | $(0, 1)$ |

**Rate of convergence:**

_to be calculated_

---

## 4. CLT and Standardized Statistic

**Theorem (Asymptotic Normality):** For any fixed continuous $F_{X,Y}$ such that $Y$ is not a measurable function of $X$ almost surely:

$$\frac{\widehat{\mathrm{II}}_n - \mathbb{E}[\widehat{\mathrm{II}}_n]}{\sqrt{\mathrm{Var}(\widehat{\mathrm{II}}_n)}} \xrightarrow{d} \mathcal{N}(0,1)$$

**Key point:** The asymptotic variance $\sigma^2 = n \cdot \mathrm{Var}(\widehat{\mathrm{II}}_n)$ has **no closed-form expression** yet. In all experiments it is replaced by the empirical estimate across $B$ replications:

$$\widehat{\sigma}^2_B = \frac{1}{B-1} \sum_{b=1}^B \left(\widehat{\mathrm{II}}^{(b)}_n - \bar{\mathrm{II}}_n\right)^2, \qquad \bar{\mathrm{II}}_n = \frac{1}{B}\sum_{b=1}^B \widehat{\mathrm{II}}^{(b)}_n$$

**Standardized statistic used in all normality experiments:**

$$Z_n^{(b)} = \frac{\widehat{\mathrm{II}}^{(b)}_n - \bar{\mathrm{II}}_n}{\widehat{\sigma}_B}$$

The collection $\{Z_n^{(b)}\}_{b=1}^B$ should look like draws from $\mathcal{N}(0,1)$ if the CLT holds at sample size $n$.

---

## 5. Parameters

```python
# Dimensions (vary across experiments)
d_X = ...           # dimension of X: test over {1, 2, 5, 10}
d_Y = ...           # dimension of Y: test over {1, 2, 5}

# Sample sizes (fixed grid for all experiments)
n_values = [100, 500, 1000, 5000]

# Pilot sample for true II estimation
n_pilot  = 50000
B_pilot  = 10       # averaged to reduce noise

# Monte Carlo replications
B = 1000            # per (n, distribution) pair

# Noise level in functional relationships
sigma_eps = 0.1     # std of additive Gaussian noise

# Random seed
SEED = 42
```

---

## 6. Distributions / Dependency Structures

All $X \sim \mathcal{N}(0, I_{d_X})$. Noise $\varepsilon \sim \mathcal{N}(0, \sigma^2_\varepsilon)$, independent of $X$.

| ID | Name | $Y$ given $X$ | True $\mathrm{II}$ | Notes |
|----|------|----------------|---------------------|-------|
| D0 | Independent | $Y \sim \mathcal{N}(0, I_{d_Y})$ | $1$ | Sanity check |
| D1 | Linear | $AX + \varepsilon$, $A$ fixed random matrix | $\approx 0$ | Basic functional |
| D2 | Quadratic | $X^2 + \varepsilon$ | $\approx 0$ | Nonlinear |
| D3 | Cubic | $X^3 + \varepsilon$ | $\approx 0$ | Nonlinear |
| D4 | Sine | $\sin(X) + \varepsilon$ | $\approx 0$ | Oscillatory |
| D5 | Cosine | $\cos(X) + \varepsilon$ | $\approx 0$ | Oscillatory |
| D6 | Exponential | $\exp(X/2) + \varepsilon$ | $\approx 0$ | Heavy tail |
| D7 | Logarithmic | $\log(\|X\| + 1) + \varepsilon$ | $\approx 0$ | **Slowest convergence** |
| D8 | Step | $\mathrm{sign}(X) + \varepsilon$ | $\approx 0$ | Discontinuous |
| D9 | Parabolic | $(X_1^2 + X_2^2,\, X_3^2, \ldots) + \varepsilon$ | $\approx 0$ | Multivariate nonlinear |

> **True II for functional cases:** Theoretically $0$ when $\sigma_\varepsilon \to 0$. For finite $\sigma_\varepsilon = 0.1$, it is slightly positive. Estimate via pilot sample at $n_{\mathrm{pilot}} = 50000$ and store in `pilot_estimates.csv`.

---

## 7. Experiments

### Experiment 1 — Consistency: Convergence to Limit

**Goal:** Confirm $\widehat{\mathrm{II}}_n \xrightarrow{a.s.} \mathrm{II}$.

**Protocol:**
1. For each distribution D0--D9 and each $n \in \{100, 500, 1000, 5000\}$:
   - Run $B = 1000$ replications; record $\bar{\mathrm{II}}_n$ and $\widehat{\sigma}_B$
   - Compute pilot estimate $\widehat{\mathrm{II}}_\infty$
   - Record empirical bias $|\bar{\mathrm{II}}_n - \widehat{\mathrm{II}}_\infty|$

**Outputs:** Convergence plot, error plot, full results table.
**Status:** Already computed — Figures 1--3 and results table exist in your limit draft.

---

### Experiment 2 — Consistency: Rate of Convergence

**Goal:** Verify the theoretical rate; examine effect of $d_X$.

**Protocol:**
1. Fix distribution D1 (linear). Vary $d_X \in \{1, 2, 5\}$
2. For each $n$: compute $\widehat{\mathrm{Var}}(\widehat{\mathrm{II}}_n)$ across $B$ replications
3. Plot $n \cdot \widehat{\mathrm{Var}}(\widehat{\mathrm{II}}_n)$ vs $n$ — flat line confirms $\sqrt{n}$ rate for $d_X = 1$

**Output:** Multi-panel line plot (one panel per $d_X$).

---

### Experiment 3 — Normality: QQ Plots

**Goal:** Visual confirmation of the CLT.

**Protocol:**
1. For distributions D0 and D1 (independence and linear — contrasting cases)
2. For each $n \in \{100, 500, 1000, 5000\}$:
   - Compute $\{Z_n^{(b)}\}_{b=1}^{1000}$ using $\widehat{\sigma}_B$
   - Plot QQ-plot against $\mathcal{N}(0,1)$ quantiles
3. Repeat for D7 (logarithmic) as a stress test — slowest convergence

**Output:** QQ-plot grid: rows = distributions, columns = $n$ values. Diagonal alignment should improve with $n$.

---

### Experiment 4 — Normality: Histogram vs $\mathcal{N}(0,1)$

**Goal:** Visually compelling figure for the presentation.

**Protocol:**
1. Same setup as Experiment 3
2. For each $n$: plot histogram of $\{Z_n^{(b)}\}$ overlaid with $\mathcal{N}(0,1)$ density curve

**Output:** $2 \times 4$ grid (2 distributions $\times$ 4 sample sizes). Best figure for slides — immediately readable by a non-specialist audience.

---

### Experiment 5 — Normality: Kolmogorov-Smirnov Test

**Goal:** Quantitative confirmation beyond visual inspection.

**Protocol:**
1. For each $(n, \text{distribution})$ pair: compute $\{Z_n^{(b)}\}$
2. Run two-sided KS test against $\mathcal{N}(0,1)$
3. Record KS statistic and $p$-value

**Output:** Table of KS statistics and $p$-values. Expect $p$-values increasing toward $1$ as $n$ grows, confirming normality.

---

### Experiment 6 — Asymmetry of II

**Goal:** Demonstrate directional asymmetry — best intuition-builder for the presentation.

**Protocol:**
1. For each distribution, compute both $\widehat{\mathrm{II}}_n(X \to Y)$ and $\widehat{\mathrm{II}}_n(Y \to X)$
2. Fix $n = 1000$, $B = 500$; report empirical means side by side

**Output:** Bar chart or paired table. Put this early in the presentation before any formulas.

---

## 8. Output File Structure

```
results/
├── pilot_estimates.csv            # True II at n=50000 for each distribution
├── exp1_consistency/
│   ├── convergence_plot.pdf
│   ├── error_plot.pdf
│   └── results_table.csv
├── exp2_rate/
│   └── rate_by_dim.pdf
├── exp3_qqplots/
│   └── qq_{dist}_{n}.pdf
├── exp4_histograms/
│   └── hist_{dist}.pdf
├── exp5_ks/
│   └── ks_table.csv
└── exp6_asymmetry/
    └── asymmetry.pdf
```

---

## 9. Open Questions and Notes

| Item | Status |
|------|--------|
| Asymptotic variance $\sigma^2 = n\,\mathrm{Var}(\widehat{\mathrm{II}}_n)$ | **Open** — no closed form; all experiments use empirical $\widehat{\sigma}^2_B$ |
| Bootstrap variance estimation | Planned future work — will replace $\widehat{\sigma}^2_B$ with a single-sample estimate |
| Logarithmic case (D7) | Slowest convergence in Exp 1 — worth one sentence of explanation in the document |
| Tie-breaking in $N_X(i)$ | First minimum index throughout — sensitivity check advisable for large $d_X$ |
| Dimension scaling | Run all experiments for $(d_X, d_Y) \in \{(1,1), (2,2), (5,5)\}$ at minimum |
| CLT under independence ($\mathrm{II} = 1$) | Theorem requires $Y$ not a measurable function of $X$ — independence satisfies this; include D0 in normality experiments |
s
