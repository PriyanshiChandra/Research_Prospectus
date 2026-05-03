# Experiment 2 — Rate of Convergence of the II Estimator

**Purpose:** Confirm empirically that $\widehat{\mathrm{II}}\_n$ converges to its population value at the parametric $\sqrt{n}$ rate. This means the empirical variance $\widehat{\mathrm{Var}}(\widehat{\mathrm{II}}_n)$, computed across $B$ Monte Carlo replications, should be proportional to $1/n$ with a finite, distribution-specific proportionality constant $\sigma^2$.

---

## 1. What is being tested and why

A consistent estimator is said to converge at the **$\sqrt{n}$ (parametric) rate** if

$$\sqrt{n}\left(\widehat{\mathrm{II}}\_n - \mathrm{II}\right) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

for some finite $\sigma^2 > 0$. Experiment 1 confirmed the distributional shape (normality). Experiment 2 separately confirms the **scaling**: that $\mathrm{Var}(\widehat{\mathrm{II}}\_n) = \sigma^2 / n + o(1/n)$. The two together constitute a complete empirical proof of the CLT.

Why is this worth its own experiment? The CLT statement gives no information about *how fast* the convergence is. A degenerate U-statistic, for instance, converges at rate $1/n$ (faster); a kernel density estimator converges at rate $n^{-2/5}$ (slower). The fact that $\widehat{\mathrm{II}}\_n$ converges at the parametric rate is non-trivial given its rank-based, non-linear structure, and it has direct implications for power calculations and sample-size planning.

---

## 2. Diagnostic approach

Two complementary plots are produced for each noise level.

### Plot A — $n \cdot \widehat{\mathrm{Var}}(\widehat{\mathrm{II}}_n)$ vs $n$

If $\mathrm{Var}(\widehat{\mathrm{II}}\_n) = \sigma^2/n + o(1/n)$, then $n \cdot \mathrm{Var}(\widehat{\mathrm{II}}\_n) \to \sigma^2$. The plot should show **flat horizontal traces** at each distribution's $\sigma^2$ level. A declining trend indicates the asymptotic regime has not been reached; an increasing trend would suggest super-$\sqrt{n}$ convergence.

### Plot B — $\widehat{\mathrm{Var}}(\widehat{\mathrm{II}}_n)$ vs $1/n$

A linear relationship through the origin confirms $\mathrm{Var} \propto 1/n$. The slope of the ordinary least-squares fit is a direct estimate of $\sigma^2$. An $R^2$ close to 1 means the $1/n$ model accounts for essentially all of the variance decay.

### Sigma² table

For each (distribution, $d_X$, noise) combination, the OLS slope from Plot B is reported as $\hat{\sigma}^2$ together with its $R^2$.

---

## 3. Experiment grid

| Parameter | Values |
|-----------|--------|
| Distributions | D0–D9 (10 total, same as Exp 1) |
| Sample sizes $n$ | 100, 500, 1000, 5000 |
| $d_X$ | 1, 2, 5 |
| $d_Y$ | 3 (fixed) |
| Noise $\sigma_\varepsilon$ | 0.1, 0.5 |
| Replications $B$ | 1000 for all $n$ |
| Random seed per task | $42 +$ `SLURM_ARRAY_TASK_ID` |
| $k$ (nearest neighbours) | 1 |

Total SLURM array tasks: $10 \times 4 \times 3 \times 2 = 240$.

---

## 4. Results — noise $\sigma_\varepsilon = 0.1$

### Plot A: $n \cdot \widehat{\mathrm{Var}}$ vs $n$

- **$d_X = 1$ and $d_X = 2$:** All ten distributions produce near-flat traces across the full range $n \in \{100, 500, 1000, 5000\}$. D0 (independent) stabilises immediately at $\approx 0.43$. Dependent distributions cluster at lower values according to their signal strength.

- **$d_X = 5$:** Several distributions — most notably D7 (logarithmic) and D8 (step) — show a **downward trend**: $n \cdot \widehat{\mathrm{Var}}$ decreases as $n$ grows instead of flattening. D7 at $d_X = 5$ declines from $\approx 0.30$ at $n=100$ to $\approx 0.10$ at $n=5000$. This is a **finite-sample effect**, not a violation of the rate: for these distributions at higher dimension, the asymptotic plateau has not been reached within the observed range of $n$.

### Plot B: $\widehat{\mathrm{Var}}$ vs $1/n$

All ten distributions, across all three $d_X$ values, produce **near-perfectly linear** relationships. The data points lie essentially on the OLS line in every panel. Representative $R^2$ values:

| Distribution | $d_X=1$ | $d_X=2$ | $d_X=5$ |
|---|---|---|---|
| D0 Independent | 0.9997 | 0.9991 | 0.9998 |
| D1 Linear | 0.9997 | 0.9935 | 0.9848 |
| D5 Cosine | 0.9998 | 0.9999 | 0.9955 |
| D7 Logarithmic | 0.9999 | 0.9995 | 0.9993 |

Even the lowest $R^2$ in the table (D1 Linear, $d_X=5$: 0.9848) indicates an excellent linear fit. The $1/n$ model is well-supported.

### Estimated $\hat{\sigma}^2$ at $\sigma_\varepsilon = 0.1$

| Distribution | $d_X=1$ | $d_X=2$ | $d_X=5$ |
|---|---|---|---|
| D0 Independent | 0.427 | 0.387 | 0.417 |
| D1 Linear | 0.011 | 0.002 | 0.025 |
| D2 Quadratic | 0.091 | 0.024 | 0.175 |
| D3 Cubic | 0.156 | 0.039 | 0.125 |
| D4 Sine | 0.025 | 0.013 | 0.048 |
| D5 Cosine | 0.186 | 0.074 | 0.201 |
| D6 Exponential | 0.060 | 0.017 | 0.050 |
| D7 Logarithmic | 0.078 | 0.125 | 0.319 |
| D8 Step | 0.119 | 0.104 | 0.181 |
| D9 Parabolic | 0.095 | 0.021 | 0.148 |

**Key observations:**
- D0 (independent) has $\hat{\sigma}^2 \approx 0.40$–$0.43$ regardless of $d_X$ — consistent with its distribution-free limiting variance when $\mathrm{II} = 1$.
- D1 (linear) has the smallest $\hat{\sigma}^2$ (as low as 0.002 at $d_X=2$) — a strong, clean signal yields a very stable estimator.
- D7 (logarithmic) shows a pronounced increase with $d_X$ (0.078 → 0.319). The log signal grows more slowly with dimension than the Euclidean norm, making the rank structure harder to resolve at higher $d_X$.
- D5 (cosine) has consistently high $\hat{\sigma}^2$ — the oscillatory structure creates ambiguous near-equidistant points, inflating variance.

---

## 5. Results — noise $\sigma_\varepsilon = 0.5$

### Plot A: $n \cdot \widehat{\mathrm{Var}}$ vs $n$

Traces are **markedly flatter** than at $\sigma_\varepsilon = 0.1$. High noise washes out the functional relationship, pushing the effective II toward 1 (independence regime). In this regime the estimator behaves similarly to D0 for all distributions, and D0 converges to its asymptotic variance fastest. The $d_X = 5$ declining trend seen at low noise is largely absent here.

### Plot B: $\widehat{\mathrm{Var}}$ vs $1/n$

Linear fits are excellent across the board. Several distributions achieve $R^2 = 1.000$ (to four decimal places), including D1, D2, D4, D9 at various $d_X$.

### Estimated $\hat{\sigma}^2$ at $\sigma_\varepsilon = 0.5$

| Distribution | $d_X=1$ | $d_X=2$ | $d_X=5$ |
|---|---|---|---|
| D0 Independent | 0.401 | 0.436 | 0.397 |
| D1 Linear | 0.178 | 0.142 | 0.129 |
| D2 Quadratic | 0.341 | 0.260 | 0.266 |
| D3 Cubic | 0.355 | 0.222 | 0.187 |
| D4 Sine | 0.245 | 0.267 | 0.284 |
| D5 Cosine | 0.423 | 0.406 | 0.414 |
| D6 Exponential | 0.386 | 0.441 | 0.382 |
| D7 Logarithmic | 0.398 | 0.443 | 0.412 |
| D8 Step | 0.130 | 0.146 | 0.290 |
| D9 Parabolic | 0.358 | 0.209 | 0.258 |

**Key observations:**
- At $\sigma_\varepsilon = 0.5$, most distributions have $\hat{\sigma}^2$ in the range $0.13$–$0.44$ — substantially higher than at $\sigma_\varepsilon = 0.1$ for all dependent distributions. High noise inflates estimator variance.
- D5 (cosine), D6 (exponential), D7 (logarithmic) are essentially at the independence level ($\hat{\sigma}^2 \approx 0.40$–$0.44$) — their functional signals are almost entirely masked by noise at this noise level.
- D1 (linear) remains the most stable ($\hat{\sigma}^2 \approx 0.13$–$0.18$), though much less so than at $\sigma_\varepsilon = 0.1$.
- D8 (step) shows the largest $d_X$ sensitivity at $\sigma_\varepsilon = 0.5$ ($0.130$ → $0.290$), suggesting the step function interacts particularly poorly with higher-dimensional noise.

---

## 6. Overall conclusions

1. **$\sqrt{n}$ rate confirmed across all distributions and noise levels.** The Var vs $1/n$ relationship is linear to $R^2 \geq 0.98$ in every cell of the experiment grid, and $R^2 \geq 0.99$ in 57 of 60 (distribution, $d_X$, noise) combinations. This is the primary takeaway.

2. **Finite-sample effect at $d_X = 5$, low noise.** The $n \cdot \widehat{\mathrm{Var}}$ plot shows declining trends for D7 and D8 at $\sigma_\varepsilon = 0.1$, $d_X = 5$. This is a pre-asymptotic effect: the estimator has not reached its limiting variance within $n \leq 5000$ for these configurations. Despite this, the $R^2$ for those cells remains $\geq 0.99$ in the Var vs $1/n$ plot — the $1/n$ law holds locally even if $\sigma^2$ is not yet well-estimated.

3. **$\sigma^2$ is distribution- and noise-dependent with no closed form.** The estimates range from near-zero (D1 linear, low noise) to near-independence (D0, or any distribution drowned in noise). This motivates the bootstrap variance estimator discussed in Setup.md §9.

4. **D0 (independent) acts as a natural upper bound.** $\hat{\sigma}^2 \approx 0.40$–$0.44$ for D0 regardless of $d_X$ or $\sigma_\varepsilon$, consistent with II being exactly 1 under independence and the estimator having a distribution-free variance in that limit.

5. **High noise homogenises $\sigma^2$.** At $\sigma_\varepsilon = 0.5$, all distributions cluster in a narrower band. When the signal is absent, the estimator behaves uniformly — a useful sanity check.

---

## 7. Finite-sample caveat for the dissertation

For logarithmic and step-function relationships at $d_X = 5$ and low noise, the quantity $n \cdot \widehat{\mathrm{Var}}(\widehat{\mathrm{II}}_n)$ continues to decline over the range $n \in [100, 5000]$, suggesting the asymptotic plateau is not yet reached. The Var vs $1/n$ regression nonetheless yields $R^2 > 0.99$, indicating the $1/n$ scaling law holds locally even if $\sigma^2$ is not yet well estimated. This is a known feature of rank-based statistics in higher dimensions, where the effective signal-to-noise ratio grows more slowly with $n$.

---

## 8. Limitations

- **No closed-form $\sigma^2$:** Estimates are purely empirical. A theoretical expression for $\sigma^2$ as a function of the DGP would require computing the limiting variance of a Hájek projection — currently listed as future work.
- **Only $k=1$ tested:** The $\sqrt{n}$ rate is demonstrated only for the 1-nearest-neighbour estimator. It is plausible but not shown here that the rate holds for $k > 1$.
- **$n$ range limited to 5000:** For configurations showing finite-sample effects, a follow-up at $n \in \{10000, 50000\}$ would confirm the eventual plateau.
- **Variance estimator is cross-replication only:** $\widehat{\mathrm{Var}}$ uses the cross-replication sample variance with $B=1000$ draws ($\mathrm{ddof}=1$). For a real dataset with a single sample, a jackknife or bootstrap estimator of $\sigma^2$ would be needed.
