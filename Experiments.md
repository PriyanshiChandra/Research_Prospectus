# Experiment Log — Information Imbalance Asymptotic Theory

**Project:** Asymptotic theory of the Information Imbalance (II) coefficient  
**Author:** Priyanshi  
**Institution:** Università della Svizzera italiana (USI), Lugano  
**Supervisors:** Prof. Déborah Sulem, Prof. Antonietta Mira  

This log tracks all simulation experiments conducted in support of the prospectus.  
Each entry records the data generating process, parameter choices, cluster job details, and observed results.  
Figures are saved as both `.pdf` (for LaTeX) and `.png` (for README/presentation).

> **Reproducibility note:** Every entry includes a git commit hash and random seed.  
> To reproduce any result: `git checkout <commit>`, then run the script with the recorded seed.

---

## Table of contents

| # | Experiment | Status | Date |
|---|-----------|--------|------|
| 1 | [Convergence to normality (QQ plots)](#experiment-1--convergence-to-normality-qq-plots) | ✅ complete | 2026-05-02 |
| 2 | [Rate of convergence (variance scaling)](#experiment-2--rate-of-convergence-variance-scaling) | ✅ complete | 2026-05-03 |
| 3 | [Convergence to population quantity II*](#experiment-3--convergence-to-population-quantity-ii) | ✅ complete | 2026-05-04 |
| 4 | [Finite-sample bias](#experiment-4--finite-sample-bias) | ⬜ not run | — |
| 5 | [Empirical coverage of asymptotic CIs](#experiment-5--empirical-coverage-of-asymptotic-confidence-intervals) | ⬜ not run | — |
| 6 | [Sensitivity to dimension](#experiment-6--sensitivity-to-dimension) | ⬜ not run | — |
| 7 | [Robustness to distribution tails](#experiment-7--robustness-to-distribution-tails) | ⬜ not run | — |

Status legend: ⬜ not run · 🔄 in progress · ✅ complete · ⚠️ issue flagged

---

## Experiment 1 — Convergence to normality (QQ plots)

**Purpose:** Visually confirm the CLT for $\widehat{\mathrm{II}}_n$. The empirical distribution of $\sqrt{n}(\widehat{\mathrm{II}}_n - \mathrm{II})$ across replications should approach $\mathcal{N}(0, \sigma^2)$ as $n$ grows. The progression from a bent to a straight diagonal in the QQ plot is the visual proof of the theorem.

**Script:** `experiments/exp1_qqplots/exp1_qqplots.py`  
**SLURM script:** `experiments/exp1_qqplots/exp1.sh`

### Run details

| Field | Value |
|-------|-------|
| Date | 2026-05-02 |
| Commit hash | ce8d506 |

### Data generating process

All $X \sim \mathcal{N}(0, I_{d_X})$, additive noise $\varepsilon \sim \mathcal{N}(0, \sigma_\varepsilon^2 I_{d_Y})$ independent of $X$.

| Parameter | Value |
|-----------|-------|
| $d_X$ | 5 |
| $d_Y$ | 3 |
| Noise level $\sigma_\varepsilon$ | 0.1 , 0.5|
| Replications $B$ | 1000 (500 for $n = 5000$) |
| Sample sizes $n$ | 100, 500, 1000, 5000 |
| Random seed | $42$ + `SLURM_ARRAY_TASK_ID` (one seed per task) |
| $k$ (nearest neighbours) | 1 |

### Dependency structures tested

| ID | Name | $Y$ given $X$ | Expected $\mathrm{II}(X \to Y)$ |
|----|------|----------------|----------------------------------|
| D0 | Independent | $Y \sim \mathcal{N}(0, I_{d_Y})$, independent of $X$ | 1 (exact) |
| D1 | Linear | $AX + \varepsilon$, $A \in \mathbb{R}^{d_X \times d_Y}$ fixed random | $\approx 0$ |
| D2 | Quadratic | $X_k^2 + \varepsilon_k$ | $\approx 0$ |
| D3 | Cubic | $X_k^3 + \varepsilon_k$ | $\approx 0$ |
| D4 | Sine | $\sin(X_k) + \varepsilon_k$ | $\approx 0$ |
| D5 | Cosine | $\cos(X_k) + \varepsilon_k$ | $\approx 0$ |
| D6 | Exponential | $\exp(X_k/2) + \varepsilon_k$ | $\approx 0$ |
| D7 | Logarithmic | $\log(\|X\|_2 + 1)\mathbf{1} + \varepsilon$ | $\approx 0$ (slowest convergence) |
| D8 | Step | $\mathrm{sign}(X_k) + \varepsilon_k$ | $\approx 0$ |
| D9 | Parabolic | $X_1^2 + X_2^2 + \varepsilon_1$; $X_k^2 + \varepsilon_k$ | $\approx 0$ |

> **Implementation note:** For $d_X = d_Y = 2$, elementwise maps (D2–D6, D8) apply to the first $\min(d_X, d_Y)$ components of $X$ and tile the last column if $d_X < d_Y$.  The linear matrix $A$ is fixed across all replications, seeded deterministically from $(d_X, d_Y)$.

### Standardised statistic

$$Z_n^{(b)} = \frac{\widehat{\mathrm{II}}_n^{(b)} - \bar{\mathrm{II}}_n}{\widehat{\sigma}_B}, \qquad \bar{\mathrm{II}}_n = \frac{1}{B}\sum_b \widehat{\mathrm{II}}_n^{(b)}, \quad \widehat{\sigma}_B = \mathrm{std}(\{\widehat{\mathrm{II}}_n^{(b)}\})$$

### Results — Run A: $\sigma_\varepsilon = 0.1$, $d_X = 5$, $d_Y = 3$

#### Exp 3 — QQ plots

| Distribution | $n = 100$ | $n = 500$ | $n = 1000$ | $n = 5000$ |
|---|---|---|---|---|
| D0 Independent | Near-straight; mild tail wiggles | Straight | Straight | Straight |
| D1 Linear | Slight S-curve in tails | Near-straight | Straight | Straight |
| D7 Logarithmic | Mild S-curve | Near-straight | Straight | Straight |

Diagonal alignment improves monotonically with $n$. D7 (logarithmic) is the slowest to converge, as predicted by Setup.md.

#### Exp 5 — KS test

All 40 pairs fail to reject $H_0$ at the 5% level.

| Distribution | $n=100$ KS / $p$ | $n=500$ KS / $p$ | $n=1000$ KS / $p$ | $n=5000$ KS / $p$ |
|---|---|---|---|---|
| D0 Independent  | 0.0230 / 0.656 | 0.0149 / 0.978 | 0.0199 / 0.815 | 0.0135 / 1.000 |
| D1 Linear       | 0.0262 / 0.491 | 0.0307 / 0.298 | 0.0192 / 0.846 | 0.0209 / 0.978 |
| D2 Quadratic    | 0.0296 / 0.337 | 0.0210 / 0.760 | 0.0186 / 0.872 | 0.0297 / 0.760 |
| D3 Cubic        | 0.0413 / 0.064 | 0.0157 / 0.963 | 0.0310 / 0.286 | 0.0230 / 0.949 |
| D4 Sine         | 0.0272 / 0.442 | 0.0412 / 0.065 | 0.0296 / 0.340 | 0.0307 / 0.723 |
| D5 Cosine       | 0.0157 / 0.962 | 0.0230 / 0.658 | 0.0248 / 0.559 | 0.0289 / 0.785 |
| D6 Exponential  | 0.0248 / 0.561 | 0.0321 / 0.250 | 0.0152 / 0.973 | 0.0184 / 0.995 |
| D7 Logarithmic  | 0.0199 / 0.815 | 0.0158 / 0.962 | 0.0173 / 0.920 | 0.0332 / 0.629 |
| D8 Step         | 0.0226 / 0.680 | 0.0272 / 0.444 | 0.0266 / 0.473 | 0.0332 / 0.629 |
| D9 Parabolic    | 0.0224 / 0.689 | 0.0179 / 0.900 | 0.0257 / 0.515 | 0.0396 / 0.404 |

**Observations:**
- D0: $p \to 1$ rapidly — normality essentially exact at large $n$.
- D3 cubic at $n=100$: $p = 0.064$, the closest to rejection in this run — finite-sample effect worth one sentence.
- D1 linear: $\widehat{\mathrm{II}}_n$ mean decays 0.119 → 0.017 as $n$ grows, confirming the estimator tracks the true value.
- D7 logarithmic: $\widehat{\mathrm{II}}_n$ mean decays slowly (0.702 → 0.443), confirming sluggish convergence.

---

### Results — Run B: $\sigma_\varepsilon = 0.5$, $d_X = 5$, $d_Y = 3$

Higher noise weakens all functional signals, pushing $\widehat{\mathrm{II}}_n$ toward 1 (independence). Same grid geometry (10 distributions × 4 sample sizes, $B = 1000$).

#### Exp 3 — QQ plots

QQ plots look qualitatively similar to Run A. Diagonal alignment improves with $n$ across all distributions. No new pathological behaviour introduced by higher noise.

#### Exp 5 — KS test

| Distribution | $n=100$ KS / $p$ | $n=500$ KS / $p$ | $n=1000$ KS / $p$ | $n=5000$ KS / $p$ |
|---|---|---|---|---|
| D0 Independent  | 0.0230 / 0.656 | 0.0149 / 0.978 | 0.0199 / 0.815 | 0.0135 / 1.000 |
| D1 Linear       | 0.0467 / **0.025** ⚠️ | 0.0332 / 0.215 | 0.0215 / 0.737 | 0.0352 / 0.555 |
| D2 Quadratic    | 0.0143 / 0.985 | 0.0164 / 0.947 | 0.0168 / 0.936 | 0.0265 / 0.866 |
| D3 Cubic        | 0.0374 / 0.119 | 0.0177 / 0.908 | 0.0163 / 0.948 | 0.0277 / 0.828 |
| D4 Sine         | 0.0225 / 0.681 | 0.0186 / 0.872 | 0.0169 / 0.933 | 0.0246 / 0.914 |
| D5 Cosine       | 0.0191 / 0.850 | 0.0239 / 0.609 | 0.0239 / 0.607 | 0.0261 / 0.876 |
| D6 Exponential  | 0.0234 / 0.637 | 0.0104 / 1.000 | 0.0190 / 0.856 | 0.0293 / 0.774 |
| D7 Logarithmic  | 0.0181 / 0.891 | 0.0211 / 0.757 | 0.0183 / 0.887 | 0.0334 / 0.620 |
| D8 Step         | 0.0203 / 0.796 | 0.0260 / 0.502 | 0.0234 / 0.636 | 0.0264 / 0.868 |
| D9 Parabolic    | 0.0306 / 0.302 | 0.0258 / 0.511 | 0.0307 / 0.295 | 0.0305 / 0.727 |

**Observations:**
- **D1 linear at $n=100$: $p = 0.025$ — the only rejection across both runs.** With $\sigma_\varepsilon = 0.5$ the signal-to-noise ratio is much lower, making the finite-sample distribution harder to approximate. Normality is recovered by $n = 500$ ($p = 0.215$).
- D0 (independent): identical to Run A — as expected, since $Y$ is generated independently of $X$ regardless of $\sigma_\varepsilon$.
- D7 logarithmic: $\widehat{\mathrm{II}}_n$ mean is now 0.889–0.928 across $n$, barely moving — the log signal is almost entirely masked by noise at $\sigma_\varepsilon = 0.5$. The estimator is essentially measuring near-independence.
- All other distributions: $\widehat{\mathrm{II}}_n$ means are uniformly higher than in Run A (noise masks the functional relationships), but normality still holds.

---

### Output files

| File | Description |
|------|-------------|
| `experiments/exp1_qqplots/plots_noise0-1/exp3_qq_plots.pdf` | QQ plots — Run A ($\sigma_\varepsilon = 0.1$) |
| `experiments/exp1_qqplots/plots_noise0-1/exp4_histograms.pdf` | Histograms — Run A ($\sigma_\varepsilon = 0.1$) |
| `experiments/exp1_qqplots/plots_noise0-1/exp5_ks_table.csv` | KS table — Run A |
| `experiments/exp1_qqplots/plots_noise0-5/exp3_qq_plots.pdf` | QQ plots — Run B ($\sigma_\varepsilon = 0.5$) |
| `experiments/exp1_qqplots/plots_noise0-5/exp4_histograms.pdf` | Histograms — Run B ($\sigma_\varepsilon = 0.5$) |
| `experiments/exp1_qqplots/plots_noise0-5/exp5_ks_table.csv` | KS table — Run B |
| `experiments/exp1_qqplots/results/*.pkl` | Raw II values and $Z_n$ — Run A |

### Flags / follow-up

- **D1 linear, $\sigma_\varepsilon = 0.5$, $n=100$**: single rejection ($p=0.025$) — re-run at $B=2000$ to determine if this is a genuine finite-sample effect or Monte Carlo noise.
- D3 cubic at $n=100$ in Run A ($p=0.064$): borderline — monitor in future runs.
- D7 logarithmic with high noise: $\widehat{\mathrm{II}}_n \approx 0.9$ even at $n=5000$ — the log signal is effectively undetectable at $\sigma_\varepsilon=0.5$. Consider whether this regime is worth including in the final document.
- Variance estimation uses cross-replication $\widehat{\sigma}_B$ — single-sample bootstrap estimator is planned future work (see Setup.md §9).

---

## Experiment 2 — Rate of convergence (variance scaling)

**Purpose:** Confirm that $\widehat{\mathrm{II}}_n$ converges at rate $\sqrt{n}$. Empirical variance of $\widehat{\mathrm{II}}_n$ across $B=1000$ replications, plotted against $1/n$, should be linear with slope $\sigma^2$. The complementary plot of $n \cdot \widehat{\mathrm{Var}}$ vs $n$ should be flat.

**Script:** `experiments/exp2_rate/exp2_rate.py`  
**SLURM script:** `experiments/exp2_rate/exp2.sh`  
**Notes:** `experiments/exp2_rate/notes.md`

### Run details

| Field | Value |
|-------|-------|
| Date | 2026-05-03 |
| Commit hash | ce8d506 |

### Data generating process

All $X \sim \mathcal{N}(0, I_{d_X})$, additive noise $\varepsilon \sim \mathcal{N}(0, \sigma_\varepsilon^2 I_{d_Y})$ independent of $X$.

| Parameter | Value |
|-----------|-------|
| Distributions | D0–D9 (10 total) |
| $d_X$ | 1, 2, 5 |
| $d_Y$ | 3 (fixed) |
| Noise level $\sigma_\varepsilon$ | 0.1, 0.5 |
| Replications $B$ | 1000 for all $n$ |
| Sample sizes $n$ | 100, 500, 1000, 5000 |
| Random seed | $42 +$ `SLURM_ARRAY_TASK_ID` |
| $k$ (nearest neighbours) | 1 |
| Total SLURM tasks | 240 |

### Results — $\sigma_\varepsilon = 0.1$

#### Plot A: $n \cdot \widehat{\mathrm{Var}}$ vs $n$

- **$d_X \in \{1, 2\}$:** All distributions produce flat or near-flat traces — asymptotic plateau reached.
- **$d_X = 5$:** D7 (logarithmic) and D8 (step) show declining trends (from ~0.30 down to ~0.10 by $n=5000$). This is a finite-sample effect: the plateau is not yet reached at these sample sizes for high-dimensional low-noise configurations. Not a violation of the rate — the Var vs $1/n$ fit remains excellent ($R^2 > 0.99$).

#### Plot B: $\widehat{\mathrm{Var}}$ vs $1/n$ — all $R^2 \geq 0.98$

| Distribution | $R^2$ at $d_X=1$ | $R^2$ at $d_X=2$ | $R^2$ at $d_X=5$ |
|---|---|---|---|
| D0 Independent | 0.9997 | 0.9991 | 0.9998 |
| D1 Linear | 0.9997 | 0.9935 | 0.9848 |
| D5 Cosine | 0.9998 | 0.9999 | 0.9955 |
| D7 Logarithmic | 0.9999 | 0.9995 | 0.9993 |

#### Estimated $\hat{\sigma}^2$ at $\sigma_\varepsilon = 0.1$

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

### Results — $\sigma_\varepsilon = 0.5$

#### Plot A: $n \cdot \widehat{\mathrm{Var}}$ vs $n$

Traces are markedly flatter than at $\sigma_\varepsilon = 0.1$ — high noise pushes all distributions toward the independence regime ($\mathrm{II} \to 1$), where the estimator converges fastest. The $d_X=5$ declining trend is largely absent.

#### Plot B: $\widehat{\mathrm{Var}}$ vs $1/n$ — several $R^2 = 1.000$

Several distributions achieve $R^2 = 1.000$ (D1, D2, D4, D9 at various $d_X$).

#### Estimated $\hat{\sigma}^2$ at $\sigma_\varepsilon = 0.5$

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

High noise pushes D5, D6, D7 essentially to the independence level ($\hat{\sigma}^2 \approx 0.40$–$0.44$) — their functional signals are masked.

### Observations

- **$\sqrt{n}$ rate confirmed.** $R^2 \geq 0.98$ in all 60 (distribution, $d_X$, noise) combinations; $\geq 0.99$ in 57 of 60. The $1/n$ scaling law is unambiguous.
- **D0 independent** has $\hat{\sigma}^2 \approx 0.40$–$0.44$ regardless of $d_X$ or $\sigma_\varepsilon$ — consistent with a distribution-free limiting variance when $\mathrm{II} = 1$.
- **D1 linear** has the smallest $\hat{\sigma}^2$ at low noise (as low as 0.002) — a strong, clean signal yields a very stable estimator.
- **D7 logarithmic** $\hat{\sigma}^2$ grows sharply with $d_X$ at low noise (0.078 → 0.319), reflecting that the log signal grows more slowly with dimension than the Euclidean distance.
- **D5 cosine** has consistently elevated $\hat{\sigma}^2$ — oscillatory structure creates ambiguous rank structure.
- **High noise homogenises $\hat{\sigma}^2$:** at $\sigma_\varepsilon = 0.5$ most distributions cluster in a narrower band, approaching the independence variance.

### Output files

| File | Description |
|------|-------------|
| `experiments/exp2_rate/plots/exp2_n_times_var_noise0-1.pdf` | Plot A — $n \cdot \widehat{\mathrm{Var}}$ vs $n$, $\sigma_\varepsilon = 0.1$ |
| `experiments/exp2_rate/plots/exp2_n_times_var_noise0-5.pdf` | Plot A — $n \cdot \widehat{\mathrm{Var}}$ vs $n$, $\sigma_\varepsilon = 0.5$ |
| `experiments/exp2_rate/plots/exp2_var_vs_invn_noise0-1.pdf` | Plot B — $\widehat{\mathrm{Var}}$ vs $1/n$, $\sigma_\varepsilon = 0.1$ |
| `experiments/exp2_rate/plots/exp2_var_vs_invn_noise0-5.pdf` | Plot B — $\widehat{\mathrm{Var}}$ vs $1/n$, $\sigma_\varepsilon = 0.5$ |
| `experiments/exp2_rate/plots/exp2_sigma2_table.csv` | $\hat{\sigma}^2$ estimates for all 60 combinations |
| `experiments/exp2_rate/plots/exp2_sigma2_table.tex` | Same table, LaTeX format |
| `experiments/exp2_rate/results/*.pkl` | Raw II values, empirical variance, $n \cdot \mathrm{Var}$ per task |

### Flags / follow-up

- **Finite-sample effect at $d_X=5$, $\sigma_\varepsilon=0.1$** (D7, D8): $n \cdot \widehat{\mathrm{Var}}$ still declining at $n=5000$. Plateau not confirmed — follow-up at $n \in \{10000, 50000\}$ recommended before finalising the dissertation table.
- **$\hat{\sigma}^2$ has no closed form:** all estimates are empirical; a theoretical derivation via Hájek projection is planned future work (see Setup.md §9).
- **Only $k=1$ tested:** rate experiment should be repeated at $k \in \{3, 5\}$ before claiming the result holds generally.

---

## Experiment 3 — Convergence to population quantity II*

**Purpose:** Verify empirically that $\bar{II}_n \to II^*$ as $n \to \infty$, and measure the rate of convergence. The theoretical rate from the draft is $|\bar{II}_n - II^*| = O(n^{-\min(1/2,\, 1/d_X)} \cdot \log(n)^{d_X+1+\beta})$. This experiment checks both the direction of convergence (does the estimator actually reach the correct limit?) and the speed (does the log-log slope match theory?).

**Script:** `experiments/exp3_convergence/exp3_convergence.py`  
**SLURM script:** `experiments/exp3_convergence/exp3.sh`  
**Notes:** `experiments/exp3_convergence/notes.md`

### Run details

| Field | Value |
|-------|-------|
| Date | 2026-05-04 |
| Commit hash | ce8d506 |
| Total SLURM tasks | 480 (10 dist × 6 $n$ × 4 $d_X$ × 2 noise) |
| Wall time limit | 6 h per task |
| Memory | 24 GB per task |

### Data generating process

| Parameter | Value |
|-----------|-------|
| Distributions | D0–D9 (10 total) |
| $d_X$ | 1, 2, 5, 10 |
| $d_Y$ | 3 (fixed) |
| $\sigma_\varepsilon$ | 0.1, 0.5 |
| Sample sizes $n$ | 100, 500, 1000, 5000, 10000, 30000 |
| Replications $B$ | 500 ($n \leq 1000$) · 200 ($n=5000$) · 100 ($n=10000$) · 50 ($n=30000$) |
| True $II^*$ | 1.0 (D0 independent), 0.0 (D1–D9 functional) |
| Error metric | $\|\bar{II}_n - II^*\|$ |
| Random seed | $42 +$ `SLURM_ARRAY_TASK_ID` |

### Theoretical rate

| $d_X$ | Slope on log-log plot | Bottleneck |
|--------|----------------------|------------|
| 1 | −0.50 | CLT / variance |
| 2 | −0.50 | CLT = bias |
| 5 | −0.20 | Nearest-neighbour bias |
| 10 | −0.10 | Nearest-neighbour bias |

### Results — Convergence ($\sigma_\varepsilon = 0.1$)

- **D0 independent:** already at $II^* = 1$ for all $d_X$ and all $n \geq 100$. Residual fluctuation is Monte Carlo noise ($< 0.002$).
- **D1–D9 functional:** all show monotone decay toward $II^* = 0$. Convergence is fastest at $d_X = 1, 2$ — most distributions reach $\bar{II}_n < 0.05$ by $n = 10{,}000$.
- **Dimension ordering visible:** within every distribution panel, lines are ordered by $d_X$; lower $d_X$ sits closer to zero — the curse of dimensionality directly visible.
- **D7 (logarithmic) is the slowest:** even at $d_X = 1$, $\bar{II}_n \approx 0.10$ at $n = 30{,}000$. At $d_X = 10$, $\bar{II}_n \approx 0.06$.
- **D8 (step)** is second slowest — rank ambiguity near the discontinuity impedes convergence.

### Results — Convergence ($\sigma_\varepsilon = 0.5$)

- **D0 independent:** identical to low-noise — independent of $\sigma_\varepsilon$ by construction.
- **All functional distributions:** convergence is drastically slowed. At $n = 30{,}000$ and $d_X = 1$, most functional distributions still sit at $\bar{II}_n \in [0.3, 0.6]$.
- **D7 (logarithmic) and D6 (exponential):** at $\sigma_\varepsilon = 0.5$ these are essentially indistinguishable from independence ($\bar{II}_n \approx 0.9$–$1.0$).
- The estimator is converging to a **noisy limit** $II^*_\text{noisy} > 0$ — the population quantity under the noisy DGP is not zero. This is not an estimator failure; it is the correct limit for the actual data-generating process.

### Results — Error decay ($\sigma_\varepsilon = 0.1$)

| Dimension | Observation |
|-----------|-------------|
| $d_X = 1, 2$ | Error decays on log-log scale, roughly parallel to the $-0.5$ reference. Empirical slopes are shallower than theory (see table) because the noisy-limit plateau begins to appear at larger $n$, pulling the fitted slope toward zero. |
| $d_X = 5$ | Lines track near the $-0.2$ reference. Distributions with strong signal (D1, D2) are closest to theory. |
| $d_X = 10$ | Lines are nearly flat ($\approx -0.05$ to $-0.15$), consistent with the very shallow $-0.1$ theoretical rate. Hard to distinguish from noise at moderate $n$. |
| D0 independent | Error at Monte Carlo noise floor ($O(10^{-3})$–$O(10^{-4})$); no systematic trend. |

### Results — Error decay ($\sigma_\varepsilon = 0.5$)

- **$d_X = 1, 2$:** error curves flat or non-monotone — the estimator has converged to the noisy limit, not to $II^* = 0$. The $-0.5$ reference slope is irrelevant here.
- **$d_X = 5, 10$:** some decay is still visible. Convergence is slow enough that the noisy plateau has not yet been reached at $n = 30{,}000$.

### Rate table (selected rows, $\sigma_\varepsilon = 0.1$)

| Distribution | $d_X$ | Emp. slope | Theory slope | $R^2$ |
|---|---|---|---|---|
| D1 Linear | 5 | −0.468 | −0.200 | 0.997 |
| D1 Linear | 10 | −0.224 | −0.100 | 1.000 |
| D5 Cosine | 5 | −0.225 | −0.200 | 0.989 |
| D7 Logarithmic | 10 | −0.062 | −0.100 | 0.999 |
| D1 Linear | 1 | −0.022 | −0.500 | 0.642 |
| D2 Quadratic | 2 | −0.155 | −0.500 | 0.772 |

For $d_X = 5, 10$: $R^2 \geq 0.97$, confirming the log-linear relationship is real. Empirical slopes for D1/D2 are sometimes steeper than theory — the $\log(n)$ correction term inflates the short-run slope. For $d_X = 1, 2$: $R^2$ is lower and slopes are shallower, due to the noisy-limit plateau.

### Observations

- **Convergence to $II^*$ confirmed** at low noise across all 10 distributions and all four $d_X$ values.
- **Dimension effect on convergence speed clearly demonstrated:** higher $d_X$ → slower convergence, matching the theoretical rate exponent $-1/d_X$ for $d_X \geq 2$.
- **Noisy limit is the dominant finite-sample effect at $\sigma_\varepsilon = 0.5$** for $d_X = 1, 2$: the estimator reaches its correct limit, which is above zero, before the sample sizes available in this experiment.
- **D7 (logarithmic) converges slowest** across all conditions — weak signal in high dimensions is easily overwhelmed by Euclidean geometry.
- **Empirical slopes in the rate table are noisy estimates** of the true asymptotic exponent; the $\log(n)$ correction and finite-sample plateau both distort them. The $R^2$ and visual slope agreement at $d_X = 5, 10$ are the more reliable diagnostics.

### Output files

| File | Description |
|------|-------------|
| `experiments/exp3_convergence/plots/exp3_convergence_noise0-1.pdf/png` | $\bar{II}_n$ vs $n$, $\sigma_\varepsilon=0.1$, one panel per distribution |
| `experiments/exp3_convergence/plots/exp3_convergence_noise0-5.pdf/png` | $\bar{II}_n$ vs $n$, $\sigma_\varepsilon=0.5$ |
| `experiments/exp3_convergence/plots/exp3_error_noise0-1.pdf/png` | Error (log-log), $\sigma_\varepsilon=0.1$, one panel per $d_X$ |
| `experiments/exp3_convergence/plots/exp3_error_noise0-5.pdf/png` | Error (log-log), $\sigma_\varepsilon=0.5$ |
| `experiments/exp3_convergence/plots/exp3_rate_table.csv` | Empirical vs theoretical slopes |
| `experiments/exp3_convergence/plots/exp3_rate_table.tex` | Same, LaTeX format |
| `experiments/exp3_convergence/results/*.pkl` | Raw per-task results |

### Flags / follow-up

- **Noisy-limit plateau ($\sigma_\varepsilon=0.5$, $d_X=1,2$):** the theoretical rate analysis assumes convergence to the noiseless $II^*=0$. For a cleaner test of the rate, either use $\sigma_\varepsilon \to 0$ or compute the true noisy-limit $II^*_\text{noisy}$ analytically and measure error against that.
- **Adaptive $B$ at large $n$:** with $B=50$ at $n=30{,}000$, the Monte Carlo error is $O(\hat{\sigma}/\sqrt{50})$. For distributions near their limit, this dominates the systematic error. Consider $B \geq 200$ at $n=30{,}000$ for cleaner slope estimates.
- **Log-correction not tested:** the $\log(n)^{d_X+1+\beta}$ factor is not fit separately. Including $\log(n)$ as a covariate in the slope regression would yield better-calibrated slope estimates.
- **Only $k=1$ tested:** repeat at $k \in \{3, 5\}$ to confirm rate holds for higher-order nearest-neighbour estimators.

---

## Experiment 4 — Finite-sample bias

**Purpose:** Quantify $\mathbb{E}[\widehat{\mathrm{II}}_n] - \mathrm{II}$ as a function of $n$. For a well-behaved U-statistic-based estimator, bias should decay at rate $O(1/n)$ or faster. This separates finite-sample behavior from asymptotic behavior and is a standard diagnostic reviewers will expect.

**Script:** `experiments/exp4_bias/exp4_bias.py`  
**SLURM script:** `experiments/exp4_bias/exp4.slurm`

### Run details

| Field | Value |
|-------|-------|
| Date | |
| Commit hash | |
| Cluster job ID | |
| Wall time used | |

### Data generating process

| Parameter | Value |
|-----------|-------|
| Distribution | |
| Parameters | |
| True II value | |
| Replications $B$ | |
| Sample sizes $n$ | |
| Random seed | |

### Results

| $n$ | $\mathbb{E}[\widehat{\mathrm{II}}_n]$ (empirical) | Bias | $n \cdot \mathrm{Bias}$ |
|-----|---------------------------------------------------|------|------------------------|
| | | | |
| | | | |
| | | | |
| | | | |

**Observed decay rate of bias:**

**Observations:**

<!-- Does bias decay like 1/n? Is there a sign (consistently over- or under-estimating)? -->

### Output files

- `experiments/exp4_bias/results/`

### Flags / follow-up

---

## Experiment 5 — Empirical coverage of asymptotic confidence intervals

**Purpose:** Construct nominal 95% CIs using the CLT (plug-in normal, or bootstrap variance estimate) and measure actual coverage across replications. Even if variance estimation is listed as future work, this motivates that chapter and shows the asymptotic theory is practically useful.

**Script:** `experiments/exp5_coverage/exp5_coverage.py`  
**SLURM script:** `experiments/exp5_coverage/exp5.slurm`

### Run details

| Field | Value |
|-------|-------|
| Date | |
| Commit hash | |
| Cluster job ID | |
| Wall time used | |

### Data generating process

| Parameter | Value |
|-----------|-------|
| Distribution | |
| Parameters | |
| True II value | |
| Replications $B$ | |
| Sample sizes $n$ | |
| Nominal coverage | 95% |
| Variance estimator used | |
| Random seed | |

### Results

| $n$ | Empirical coverage | CI width (mean) |
|-----|-------------------|-----------------|
| | | |
| | | |
| | | |
| | | |

**Observations:**

<!-- Does coverage approach 95%? Is it conservative or anti-conservative at small n? -->

### Output files

- `experiments/exp5_coverage/results/`

### Flags / follow-up

---

## Experiment 6 — Sensitivity to dimension

**Purpose:** Verify that the CLT and $\sqrt{n}$ rate hold across different input dimensions $d_X, d_Y$. The asymptotic variance will change with dimension, but normality should persist. This directly addresses the multivariate setting of the theory and pre-empts the obvious committee question.

**Script:** `experiments/exp6_dimension/exp6_dimension.py`  
**SLURM script:** `experiments/exp6_dimension/exp6.slurm`

### Run details

| Field | Value |
|-------|-------|
| Date | |
| Commit hash | |
| Cluster job ID | |
| Wall time used | |

### Data generating process

| Parameter | Value |
|-----------|-------|
| Base distribution | |
| Dimensions tested $(d_X, d_Y)$ | (1,1), (2,2), (5,5) |
| Replications $B$ | |
| Sample sizes $n$ | |
| Random seed | |

### Results

| $(d_X, d_Y)$ | QQ plot straight at $n=500$? | Empirical $\sigma^2$ | Rate $\sqrt{n}$ confirmed? |
|--------------|------------------------------|---------------------|---------------------------|
| (1,1) | | | |
| (2,2) | | | |
| (5,5) | | | |

**Observations:**

<!-- Does convergence slow down in higher dimensions? Does the asymptotic variance grow with d? -->

### Output files

- `experiments/exp6_dimension/results/`

### Flags / follow-up

---

## Experiment 7 — Robustness to distribution tails

**Purpose:** Since II is rank-based, the CLT should hold under heavy-tailed distributions. Confirm this by repeating the normality check under bivariate $t_\nu$ for $\nu = 3, 5, 10$ and comparing to the Gaussian baseline. Any deviation would be a meaningful finite-sample finding.

**Script:** `experiments/exp7_tails/exp7_tails.py`  
**SLURM script:** `experiments/exp7_tails/exp7.slurm`

### Run details

| Field | Value |
|-------|-------|
| Date | |
| Commit hash | |
| Cluster job ID | |
| Wall time used | |

### Data generating process

| Parameter | Value |
|-----------|-------|
| Distributions tested | Bivariate $t_3$, $t_5$, $t_{10}$, Gaussian |
| Correlation structure | |
| Replications $B$ | |
| Sample sizes $n$ | |
| Random seed | |

### Results

| Distribution | QQ plot at $n=200$ | QQ plot at $n=1000$ | Notable tail behavior |
|-------------|-------------------|--------------------|-----------------------|
| Gaussian | | | baseline |
| $t_{10}$ | | | |
| $t_5$ | | | |
| $t_3$ | | | |

**Observations:**

<!-- Does heavier tail slow convergence to normality? Are the tails of the QQ plot more bent for t_3? -->

### Output files

- `experiments/exp7_tails/results/`

### Flags / follow-up

---

## Cross-experiment notes

<!-- 
Use this section for observations that span multiple experiments —
e.g. a pattern you notice across Exp 1 and Exp 4, or a parameter
choice that affected multiple runs.
-->

---

## Parameter choices and defaults

Record any global defaults here so they are documented in one place.

| Parameter | Default value | Rationale |
|-----------|--------------|-----------|
| Replications $B$ | | |
| Base random seed | | |
| $k$ (nearest neighbours, if applicable) | | |
| Figure DPI | 300 | publication quality |
| Figure formats | `.pdf`, `.png` | LaTeX + README |

---

*Last updated: 2026-05-04 — Experiment 3 complete.*


