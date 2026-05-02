# Experiment 1 — Normality of the II Estimator: Detailed Notes

**Author:** Priyanshi Chandra  
**Institution:** Università della Svizzera italiana (USI), Lugano  
**Supervisors:** Prof. Déborah Sulem, Prof. Antonietta Mira  
**Script:** `exp1_qqplots.py` | **SLURM:** `exp1.sh`

---

## 1. What We Are Testing and Why

The central asymptotic result we wish to validate is the **Central Limit Theorem (CLT) for the Information Imbalance (II) estimator**. Informally, the theorem states that after appropriate centring and scaling, the empirical II estimator converges in distribution to a standard normal as the sample size $n$ grows.

More precisely, for any fixed continuous joint distribution $F_{X,Y}$ such that $Y$ is not a measurable function of $X$ almost surely:

$$\frac{\widehat{\mathrm{II}}_n - \mathbb{E}[\widehat{\mathrm{II}}_n]}{\sqrt{\mathrm{Var}(\widehat{\mathrm{II}}_n)}} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty$$

This result is foundational: it underpins hypothesis testing, confidence interval construction, and model comparison using II. Without it, we cannot justify any inferential procedure built on top of the estimator.

The purpose of this experiment is to provide **empirical evidence** that this convergence is real, quantitatively meaningful, and holds across a wide range of dependency structures and noise levels — not just the toy cases for which an exact proof might be easiest to construct.

---

## 2. The Estimator

Given a sample $(x_i, y_i)\_{i=1}^n$ from a joint distribution $P_\{X,Y}$, the empirical II estimator is computed in three steps:

**Step 1 — Nearest neighbour in $X$-space:**

$$N_X(i) = \arg\min_{k \neq i} \lVert x_k - x_i \rVert$$

For each observation $i$, find the index of the point closest to it in $X$-space (Euclidean distance, ties broken by first minimum index).

**Step 2 — Rank in $Y$-space:**

$$R^Y_i(N_X(i)) = \sum_{j \neq i} \mathbf{1}_\left\lbrace \lVert y_{N_X(i)} - y_i \rVert \geq \lVert y_j - y_i \rVert \right\rbrace$$

For each $i$, find the rank of $y_{N_X(i)}$ — the $Y$-value of $i$'s nearest $X$-neighbour — when all other $y_j$ are sorted by distance from $y_i$. A rank of 1 means the $X$-neighbour is also the $Y$-neighbour: $X$ is highly predictive of $Y$. A rank near $n$ means the $X$-neighbour lands far away in $Y$-space: $X$ carries little information about $Y$.

**Step 3 — The estimator:**

$$\widehat{\mathrm{II}}_n(X \to Y) = \frac{2}{n^2} \sum_{i=1}^n R^Y_i(N_X(i))$$

The factor $2/n^2$ normalises so that $\widehat{\mathrm{II}}_n \in (0, 1)$ approximately. The estimator is **directional**: $\widehat{\mathrm{II}}_n(X \to Y) \neq \widehat{\mathrm{II}}_n(Y \to X)$ in general, which is its defining feature.

**Interpretation of the population limit:**
- $\mathrm{II}(X \to Y) \approx 1$: $X$ and $Y$ are nearly independent — knowing the nearest neighbour in $X$ tells you nothing about proximity in $Y$.
- $\mathrm{II}(X \to Y) \approx 0$: $Y$ is nearly a deterministic function of $X$ — nearest neighbours in $X$ are also nearest neighbours in $Y$.
- $\mathrm{II}(X \to Y) \in (0, 1)$: intermediate dependence.

---

## 3. The Standardised Statistic

A key practical challenge is that the asymptotic variance $\sigma^2 = n \cdot \mathrm{Var}(\widehat{\mathrm{II}}_n)$ has **no known closed form**. It depends on the joint distribution $F_{X,Y}$ in a complex way. Deriving it analytically is listed as open work in Setup.md §9.

In the experiments, we bypass this by using the **empirical variance across Monte Carlo replications** as a proxy. Concretely, for each fixed $(n, \text{distribution})$ pair we run $B$ independent replications, each producing a fresh sample of size $n$ and a corresponding estimate $\widehat{\mathrm{II}}_n^{(b)}$. We then form:

$$\bar{\mathrm{II}}_n = \frac{1}{B} \sum_{b=1}^B \widehat{\mathrm{II}}_n^{(b)}, \qquad \widehat{\sigma}_B = \sqrt{\frac{1}{B-1} \sum_{b=1}^B \left(\widehat{\mathrm{II}}_n^{(b)} - \bar{\mathrm{II}}_n\right)^2}$$

and standardise each replication:

$$Z_n^{(b)} = \frac{\widehat{\mathrm{II}}_n^{(b)} - \bar{\mathrm{II}}_n}{\widehat{\sigma}_B}$$

The collection $\{Z_n^{(b)}\}_{b=1}^B$ is what we study. If the CLT holds at sample size $n$, these should look like draws from $\mathcal{N}(0, 1)$.

**Important caveat:** This standardisation uses the cross-replication mean and standard deviation, not the true $\mathrm{II}$ and $\sigma$. We are therefore testing a slightly weaker statement than the theorem — we are checking that the empirical distribution of $B$ estimates, after self-standardisation, is approximately normal. This is a necessary but not sufficient condition for the full CLT. It is the right first check in the absence of a closed-form variance expression.

---

## 4. Data Generating Processes

All experiments use $X \sim \mathcal{N}(0, I_{d_X})$. The dependency structures cover a wide range of functional relationships and a null (independence) case.

### 4.1 Rationale for the choice of distributions

The 10 distributions were chosen to stress-test the CLT along several axes:

- **D0 (independent):** The baseline null case where $\mathrm{II} = 1$ exactly. The CLT theorem requires $Y$ to not be a measurable function of $X$ a.s. — independence trivially satisfies this. This is included as a sanity check: if normality fails here, something is fundamentally wrong.

- **D1 (linear):** The simplest functional relationship. $Y = AX + \varepsilon$ with $A$ a fixed random matrix, seeded deterministically from $(d_X, d_Y)$ so it is identical across all replications. This is the easiest case for the estimator and the first one to converge.

- **D2–D5 (quadratic, cubic, sine, cosine):** Smooth nonlinear maps. These test whether normality holds when the functional relationship is nonlinear. The CLT should still apply since these functions are sufficiently regular.

- **D6 (exponential):** Heavy right tail in $Y$. Tests robustness to distributional asymmetry induced by the nonlinear transform.

- **D7 (logarithmic):** $Y_k = \log(\|X\| + 1) + \varepsilon_k$. A scalar function of $\|X\|$ broadcast across all $Y$-dimensions. This is the hardest case: convergence is slowest here because the log function compresses large values of $\|X\|$, reducing the effective signal, and the II values are far from both 0 and 1 even at large $n$ (they stay near 0.5–0.7 for noise 0.1). Setup.md explicitly flags this as the slowest case.

- **D8 (step):** $Y_k = \mathrm{sign}(X_k) + \varepsilon_k$. A discontinuous function. Tests whether the CLT holds under non-smooth dependence.

- **D9 (parabolic):** $Y_1 = X_1^2 + X_2^2$, $Y_k = X_{k+1}^2$ for $k > 1$. A multivariate nonlinear map. Tests the higher-dimensional case where the relationship involves interactions between $X$-components.

### 4.2 How data is generated for each distribution

For each replication $b$ and each distribution:

1. Draw $X^{(b)} \sim \mathcal{N}(0, I_{d_X})$ of size $n$.
2. Draw $\varepsilon^{(b)} \sim \mathcal{N}(0, \sigma_\varepsilon^2 I_{d_Y})$ independently.
3. Apply the relationship to get $Y^{(b)}$.
4. Compute $\widehat{\mathrm{II}}_n^{(b)}$ from $(X^{(b)}, Y^{(b)})$.

For the linear case, the matrix $A$ is fixed across all replications (seeded from $(d_X, d_Y)$) so that the underlying population distribution $F_{X,Y}$ is the same in every replication. This is essential: the CLT is about repeated sampling from the same distribution, not about varying the distribution itself.

### 4.3 Parameter choices

| Parameter | Run A | Run B | Rationale |
|---|---|---|---|
| $d_X$ | 5 | 5 | Multivariate input — more realistic than $d_X = 1$ |
| $d_Y$ | 3 | 3 | Multivariate output |
| $\sigma_\varepsilon$ | 0.1 | 0.5 | Low vs. high noise — tests robustness of CLT to SNR |
| $n$ | 100, 500, 1000, 5000 | same | Covers small through large sample regimes |
| $B$ | 1000 (500 for $n=5000$) | same | Large enough to estimate the distribution of $Z_n$ well |
| Seed | $42 + \text{task\_id}$ | same | One independent seed per (dist, $n$) pair |

The choice of two noise levels is deliberate: with $\sigma_\varepsilon = 0.1$ the signal is strong and II values are close to 0 for functional cases, whereas with $\sigma_\varepsilon = 0.5$ the signal is weaker and II values are pushed toward 1 (the independence value). This tests whether the CLT holds throughout the parameter space, not just in the favourable low-noise regime.

---

## 5. The Three Checks

### 5.1 Experiment 3 — QQ Plots

**What it is:** A quantile-quantile plot compares the empirical quantiles of $\{Z_n^{(b)}\}_{b=1}^B$ against the theoretical quantiles of $\mathcal{N}(0, 1)$. Points lying along the 45° line indicate the empirical distribution matches $\mathcal{N}(0,1)$.

**How to read it:**
- **Perfect diagonal:** The sample distribution is consistent with $\mathcal{N}(0,1)$.
- **S-curve (tails bend away):** The distribution has heavier or lighter tails than normal.
- **Systematic upward/downward shift:** The distribution is not centred at 0 — would indicate a bias in the standardisation.
- **Convergence across $n$:** As $n$ increases along the columns of the grid, the plot should straighten. This progression is the visual proof of the CLT — what we are showing is the theorem happening in practice.

**Why QQ plots are the right tool here:** Histograms smooth over the tails, which is where non-normality first appears. QQ plots are sensitive to tail behaviour and provide a much more diagnostic view. Committees and reviewers familiar with the CLT literature will expect QQ plots.

**What we observed:**
- D0 (independent): Nearly straight even at $n = 100$. Convergence is effectively complete by $n = 500$.
- D1 (linear): Slight S-curve at $n = 100$ (mild heavy tails), straight by $n = 500$.
- D7 (logarithmic): The most bent at $n = 100$, still noticeably curved at $n = 500$, straight by $n = 1000$. This is consistent with it being the hardest case — the log compression of $\|X\|$ slows down the approach to normality.

### 5.2 Experiment 4 — Histograms

**What it is:** A histogram of $\{Z_n^{(b)}\}_{b=1}^B$ overlaid with the $\mathcal{N}(0,1)$ density curve. Arranged in a grid: rows are distributions, columns are sample sizes.

**How to read it:**
- The histogram should increasingly hug the red $\mathcal{N}(0,1)$ curve as $n$ grows.
- Jaggedness at $n = 100$ is expected — with $B = 1000$ bins and 40 histogram bins the resolution is limited.
- Any persistent skewness, multi-modality, or fat shoulders across $n$ values would be a warning sign.

**Why include it alongside QQ plots:** Histograms are more immediately readable to a non-specialist audience — a supervisor, committee member, or referee who is not a statistician will understand a histogram immediately. For a presentation or dissertation, include both: QQ for diagnostic rigour, histograms for visual impact.

**What we observed:** All panels show the histogram converging to the $\mathcal{N}(0,1)$ bell curve as $n$ increases. The convergence is smooth and monotone across all distributions and both noise levels.

### 5.3 Experiment 5 — Kolmogorov-Smirnov Test

**What it is:** The two-sided KS test compares the empirical CDF of $\{Z_n^{(b)}\}$ against the $\mathcal{N}(0,1)$ CDF. The test statistic is:

$$D_n = \sup_z \left| F_n(z) - \Phi(z) \right|$$

where $F_n$ is the empirical CDF of the $B$ standardised statistics and $\Phi$ is the standard normal CDF. Under $H_0: Z_n \sim \mathcal{N}(0,1)$, $D_n \sqrt{B} \xrightarrow{d} $ Kolmogorov distribution. A large $p$-value means we fail to reject normality.

**Why include a formal test:** QQ plots and histograms are visual and subject to interpretation. The KS test gives a single number — the $p$-value — that is objective, reproducible, and directly citable. In the dissertation you can write: "At every $(n, \text{distribution})$ combination tested, the KS test failed to reject normality at the 5% level" and point to Table X.

**What we observed — Run A ($\sigma_\varepsilon = 0.1$):**  
All 40 pairs failed to reject $H_0$. The closest to rejection was D3 cubic at $n = 100$ ($p = 0.064$). $p$-values increase toward 1 as $n$ grows for all distributions, confirming the CLT.

**What we observed — Run B ($\sigma_\varepsilon = 0.5$):**  
39 out of 40 pairs failed to reject $H_0$. The one rejection was D1 linear at $n = 100$ ($p = 0.025$). With higher noise, the signal-to-noise ratio for the linear relationship drops substantially, and the finite-sample distribution of $\widehat{\mathrm{II}}_n$ is harder to approximate at small $n$. By $n = 500$ ($p = 0.215$) normality is recovered. This is a finite-sample effect, not a failure of the CLT.

---

## 6. What the Results Collectively Establish

Taking the three checks together, the experiments provide strong empirical evidence for the following claims:

1. **The CLT holds across dependency structures.** From the trivial (independence) to the highly nonlinear (parabolic, logarithmic), the standardised statistic $Z_n$ converges to $\mathcal{N}(0,1)$ as $n$ grows. No pathological cases were found.

2. **The CLT holds across noise levels.** With both $\sigma_\varepsilon = 0.1$ and $\sigma_\varepsilon = 0.5$, normality is established by $n = 500$ in all but one case. The single rejection (D1 linear, $\sigma_\varepsilon = 0.5$, $n = 100$) is a finite-sample effect that disappears at larger $n$.

3. **The rate of convergence is distribution-dependent.** D0 (independence) and D1 (linear) converge fastest. D7 (logarithmic) converges slowest, consistent with Setup.md's prediction. This is visible in the QQ plots and in the $p$-values being slightly lower for D7 at small $n$.

4. **The empirical variance $\widehat{\sigma}_B^2$ is a workable substitute for the true $\sigma^2$.** The fact that $Z_n$ constructed with $\widehat{\sigma}_B$ looks normal suggests that the cross-replication standard deviation is a consistent estimator of the true asymptotic standard deviation. This motivates the planned bootstrap variance estimator (Setup.md §9), which would provide a single-sample analogue.

5. **The estimator is well-behaved at the independence boundary.** D0 achieves $\mathrm{II} = 1$ exactly in the limit, and the CLT holds at this boundary. This is non-trivial: the theorem requires $Y$ not to be a measurable function of $X$, and independence is the extreme case in the other direction.

---

## 7. Limitations and What These Results Do Not Establish

- **The variance $\sigma^2$ is not characterised.** We establish normality of the standardised statistic, but we do not determine how $\sigma^2$ depends on $(F_{X,Y}, d_X, d_Y)$. This is the most significant open question.

- **The cross-replication standardisation is not usable in practice.** Running $B = 1000$ replications to estimate $\sigma$ is not a deployment strategy. A practitioner with one dataset cannot do this. The bootstrap estimator is the planned solution.

- **Dimensions are fixed.** All runs used $d_X = 5$, $d_Y = 3$. Experiment 6 (sensitivity to dimension) will explore how convergence changes with $(d_X, d_Y)$.

- **The D1 linear rejection at $\sigma_\varepsilon = 0.5$, $n = 100$ should be re-examined.** Re-running at $B = 2000$ would determine whether this is a genuine finite-sample effect or a Monte Carlo artefact (low $B$ leading to a noisy estimate of $\widehat{\sigma}_B$).

- **Tie-breaking in $N_X(i)$ is deterministic (first minimum index).** Under continuous distributions ties occur with probability zero, but in finite samples with floating-point arithmetic they can occur. A sensitivity check for large $d_X$ is advisable.

---

## 8. Files

| File | Contents |
|------|----------|
| `exp1_qqplots.py` | Full experiment script (data generation, estimation, plots, KS test) |
| `exp1.sh` | SLURM array job script (10 distributions × 4 sample sizes = 40 tasks) |
| `plots_noise0-1/` | QQ plots, histograms, KS table — $\sigma_\varepsilon = 0.1$, $d_X=5$, $d_Y=3$ |
| `plots_noise0-5/` | QQ plots, histograms, KS table — $\sigma_\varepsilon = 0.5$, $d_X=5$, $d_Y=3$ |
| `results/*.pkl` | Raw $\widehat{\mathrm{II}}_n^{(b)}$ and $Z_n^{(b)}$ arrays for all tasks |

---

## 9. Open Questions

| Question | Status |
|---|---|
| Closed-form expression for $\sigma^2 = n\,\mathrm{Var}(\widehat{\mathrm{II}}_n)$ | Open |
| Bootstrap variance estimator (single-sample $\sigma$ estimate) | Planned |
| D1 linear rejection at $\sigma_\varepsilon = 0.5$, $n=100$: genuine or MC artefact? | Re-run at $B=2000$ |
| Dimension sensitivity: does convergence slow for large $d_X$? | Experiment 6 |
| Rate of convergence: is it $\sqrt{n}$ uniformly across distributions? | Experiment 2 |
