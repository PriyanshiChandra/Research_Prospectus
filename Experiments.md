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
| 1 | [Convergence to normality (QQ plots)](#experiment-1--convergence-to-normality-qq-plots) | ⬜ not run | — |
| 2 | [Rate of convergence (variance scaling)](#experiment-2--rate-of-convergence-variance-scaling) | ⬜ not run | — |
| 3 | [Directional asymmetry across dependence structures](#experiment-3--directional-asymmetry-across-dependence-structures) | ⬜ not run | — |
| 4 | [Finite-sample bias](#experiment-4--finite-sample-bias) | ⬜ not run | — |
| 5 | [Empirical coverage of asymptotic CIs](#experiment-5--empirical-coverage-of-asymptotic-confidence-intervals) | ⬜ not run | — |
| 6 | [Sensitivity to dimension](#experiment-6--sensitivity-to-dimension) | ⬜ not run | — |
| 7 | [Robustness to distribution tails](#experiment-7--robustness-to-distribution-tails) | ⬜ not run | — |

Status legend: ⬜ not run · 🔄 in progress · ✅ complete · ⚠️ issue flagged

---

## Experiment 1 — Convergence to normality (QQ plots)

**Purpose:** Visually confirm the CLT for $\widehat{\mathrm{II}}_n$. The empirical distribution of $\sqrt{n}(\widehat{\mathrm{II}}_n - \mathrm{II})$ across replications should approach $\mathcal{N}(0, \sigma^2)$ as $n$ grows. The progression from a bent to a straight diagonal in the QQ plot is the visual proof of the theorem.

**Script:** `experiments/exp1_qqplots/exp1_qqplots.py`  
**SLURM script:** `experiments/exp1_qqplots/exp1.slurm`

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
| Parameters (e.g. ρ, μ, Σ) | |
| True II value (if known) | |
| Replications $B$ | |
| Sample sizes $n$ | 50, 200, 500, 1000 |
| Random seed | |

### Results

| $n$ | QQ plot qualitative description |
|-----|--------------------------------|
| 50 | |
| 200 | |
| 500 | |
| 1000 | |

**Observations:**

<!-- 
What does the progression look like? 
At what n does the plot look convincingly straight?
Any asymmetry in the tails? Any outliers?
-->

### Output files

- `experiments/exp1_qqplots/results/`

### Flags / follow-up

<!-- Anything unexpected? Anything to revisit in another experiment? -->

---

## Experiment 2 — Rate of convergence (variance scaling)

**Purpose:** Confirm that $\widehat{\mathrm{II}}_n$ converges at rate $\sqrt{n}$. Empirical variance of $\widehat{\mathrm{II}}_n$ across replications, plotted against $1/n$, should be linear. The slope of that line estimates the asymptotic variance $\sigma^2$.

**Script:** `experiments/exp2_rate/exp2_rate.py`  
**SLURM script:** `experiments/exp2_rate/exp2.slurm`

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
| True II value (if known) | |
| Replications $B$ | |
| Sample sizes $n$ | |
| Random seed | |

### Results

| $n$ | $\mathrm{Var}(\widehat{\mathrm{II}}_n)$ (empirical) | $1/n$ | Ratio $n \cdot \mathrm{Var}$ |
|-----|------------------------------------------------------|-------|------------------------------|
| | | | |
| | | | |
| | | | |
| | | | |

**Estimated asymptotic variance $\hat{\sigma}^2$ (slope of linear fit):**

**$R^2$ of linear fit:**

**Observations:**

<!-- Is the plot linear? Does the ratio n·Var stabilize? Any curvature at small n? -->

### Output files

- `experiments/exp2_rate/results/`

### Flags / follow-up

---

## Experiment 3 — Directional asymmetry across dependence structures

**Purpose:** Demonstrate that $\widehat{\mathrm{II}}$ genuinely captures *directional* dependence — i.e. $\widehat{\mathrm{II}}(X \to Y) \neq \widehat{\mathrm{II}}(Y \to X)$ in asymmetric settings, and that both converge to the correct true values. This is the intuition-building experiment for a broad committee.

**Script:** `experiments/exp3_asymmetry/exp3_asymmetry.py`  
**SLURM script:** `experiments/exp3_asymmetry/exp3.slurm`

### Run details

| Field | Value |
|-------|-------|
| Date | |
| Commit hash | |
| Cluster job ID | |
| Wall time used | |

### Distributions tested

| # | Distribution | True $\mathrm{II}(X \to Y)$ | True $\mathrm{II}(Y \to X)$ | Expected behavior |
|---|-------------|----------------------------|----------------------------|-------------------|
| A | Bivariate Gaussian, $\rho = 0$ | | | Both directions ~ 0 |
| B | Bivariate Gaussian, $\rho = 0.4$ | | | Symmetric, moderate |
| C | Bivariate Gaussian, $\rho = 0.8$ | | | Symmetric, strong |
| D | $Y = f(X) + \varepsilon$, small noise | | | Strong $X \to Y$, weak reverse |
| E | $X \perp Y$ | | | Both ~ 0 |

### Results

| Distribution | $\widehat{\mathrm{II}}(X \to Y)$ (mean) | $\widehat{\mathrm{II}}(Y \to X)$ (mean) | Asymmetry visible? |
|-------------|----------------------------------------|----------------------------------------|-------------------|
| A | | | |
| B | | | |
| C | | | |
| D | | | |
| E | | | |

**Observations:**

<!-- Does the estimator recover the direction correctly? Is the asymmetry sharp and clear in plots? -->

### Output files

- `experiments/exp3_asymmetry/results/`

### Flags / follow-up

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

*Last updated:*


