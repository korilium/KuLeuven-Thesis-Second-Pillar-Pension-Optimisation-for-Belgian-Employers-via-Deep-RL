# Model walkthrough: from the OLO data to the objective, and one career by hand

This note follows the model in two ways:
- one interest-rate scenario, end to end, from the NBB data to the joint value $J$ that the DP oracle and the RL agent are scored on (Sections 1–2);
- one plan member, by hand, for three years under constant rates and under Hull-White rates (Section 3).

It then uses those numbers to show:
- why the reduced state $(t, F, \rho)$ is sufficient at constant rates (Section 4);
- why it stops being sufficient once the guarantee is booked horizontally at moving rates (Section 5).

The numbers come from the code. The blocks between `gen` markers are written by `python experiments/walkthrough.py --write`, and `--check` fails if any of them no longer matches the code.

---

## Notation

| Symbol | Meaning | Code |
|---|---|---|
| $R_t$, $L_t$, $S_t$ | Reserve, guaranteed (WAP) reserve, salary | `State.R`, `State.L`, `State.S` |
| $F = R/L$, $\rho = S/L$ | Funding ratio, salary-to-liability ratio | `State.F`, `State.rho` |
| $a_t \in [0,1]$ | Funding decision; contribution $c_t = a_t \Gamma S_t$ | `step(state, a, ...)` |
| $G_t$ | WAP guarantee rate fixed for year $t$ | `Exogenous.G` |
| $\mu_t$ | Credited return (constant $\mu$, or the book yield) | `Exogenous.mu` |
| $z_R, z_L$ | Asset and guarantee shocks, $N(0,1)$ | `Exogenous.zR`, `.zL` |
| $\delta_f$, $\delta_e$ | Employer and employee discount rates | `DISC_ER`, `DISC_EMP` |
| $\lambda$ | Employee weight; $(w_{emp}, w_{er}) = (\lambda, 1-\lambda)$ | `LAMBDA` |
| $\ddot a$ | Annuity factor | `ANNUITY` |

All parameters are in `pension/params.py` (`DEFAULT`). The trace uses `DEFAULT.replace(RATE_MODEL="hull_white")`.

---

## 1. The pipeline at a glance

```mermaid
flowchart TD
    CSV["NBB OLO yields (CSV cache)<br/><code>rates.data.load_olo</code>"]
    VAS["Vasicek OLS on the 10Y history: κ, σ<br/><code>calibrateVasicek</code>"]
    NSS["NSS fit of today's curve: f(0,T)<br/><code>bootstrapForwardCurve</code>"]
    HW["Monthly short-rate paths<br/><code>simulateHullWhite</code>"]
    Y10["Future 10Y per path<br/><code>reconstructFutureYield</code>"]
    WAP["WAP rate G_t: 0.85 × 24-month mean, 25 bp grid<br/><code>wap.computeWAPRate</code>"]
    MU["Book yield μ_t: 8-year rolling mean<br/><code>economy.draw_rate_scenarios</code>"]
    EXO["All randomness of n careers<br/><code>dynamics.Exogenous</code>"]
    STEP["One year: contribute, credit, book vintage, grow, churn<br/><code>dynamics.step</code> × T"]
    SET["Payout max(R,L), shortfall (L−R)+<br/><code>dynamics.settle</code>"]
    OBJ["Value: employee leg − employer leg<br/><code>objective.Objective</code>"]
    J["simulate(...)['joint'] / episode return<br/><code>dp.simulate</code>, <code>PensionEnv</code>"]
    CE["Moments of the scenarios<br/><code>dp.certainty_equivalent</code>"]
    SOLVE["Backward induction on (t, F, ρ)<br/><code>dp.solve</code> → policy a*(t,F,ρ)"]

    CSV --> VAS --> HW
    CSV --> NSS --> HW
    HW --> Y10
    CSV -- observed history --> WAP
    CSV -- observed history --> MU
    Y10 --> WAP
    Y10 --> MU
    WAP --> EXO
    MU --> EXO
    EXO --> STEP --> SET --> OBJ --> J
    WAP --> CE
    MU --> CE
    CE --> SOLVE -- policy --> STEP
```

There are two routes to a policy.
- **DP oracle.** It solves on the reduced state $(t, F, \rho)$ with rates summarised by their moments. See `certainty_equivalent` in Section 2.6.
- **RL agent.** It acts on the full path in `PensionEnv`.

Both are scored by the same `step()` and the same `Objective` (Section 2.8).

---

## 2. Data-flow trace of one scenario

### 2.1 Data
`load_olo()` reads the cached NBB series. The 10Y history drives the calibration, the WAP window and the book yield. The latest cross-section is the curve the model starts on.

<!-- gen:data -->
- 10Y series: 321 monthly observations, Jan 2000 to Sep 2026; the last one, 4.12%, is month 0 of the model.
- Latest cross-section (Sep 2026), in %:

| maturity | 1Y | 2Y | 5Y | 10Y | 20Y | 30Y |
|---|---|---|---|---|---|---|
| yield | 3.06 | 3.20 | 3.60 | 4.12 | 4.66 | 4.83 |
<!-- /gen -->

### 2.2 Vasicek: speed and volatility
OLS of $\Delta r = a + b\,r + \varepsilon$ on the monthly 10Y gives $\kappa = -b/\Delta t$, $\theta = -a/b$ and $\sigma = \mathrm{sd}(\varepsilon)/\sqrt{\Delta t}$. Hull-White keeps only $\kappa$ and $\sigma$.

<!-- gen:vasicek -->
| kappa | theta | sigma | t-stat of b | p-value | half-life |
|---|---|---|---|---|---|
| 0.1104 | 2.23% | 0.61% | -1.61 | 0.109 | 6.3 y |
<!-- /gen -->

The mean reversion is weak. $b$ is not significant at 10%, so $\kappa$ is imprecise, and it is estimated on the 10Y rather than on a short rate. These are known caveats of this step.

### 2.3 Nelson-Siegel-Svensson: today's curve
The NSS fit gives the forward curve $f(0,T)$ and its slope analytically. The Hull-White target that reprices today's curve is

$$
\theta(t) = f(0,t) + \frac{1}{\kappa}\frac{\partial f(0,t)}{\partial t} + \frac{\sigma^2}{2\kappa^2}\left(1-e^{-2\kappa t}\right).
$$

<!-- gen:nss -->
| b0 | b1 | b2 | b3 | tau1 | tau2 |
|---|---|---|---|---|---|
| 3.635% | -0.728% | -2.528% | 7.032% | 9.05 y | 13.73 y |

- Fit to the 30 observed maturities: RMSE 1.29 bp.
- Short rate today r(0) = f(0,0) = b0 + b1 = 2.907%; f(0,10) = 4.941%; f(0,30) = 5.032%.
- Hull-White target at 0: theta(0) = f(0,0) + f'(0,0)/kappa = 5.744%.
<!-- /gen -->

### 2.4 One rate path
`simulateHullWhite` steps $r = \alpha(t) + x(t)$ exactly, month by month, where
- $x$ is a zero-mean Ornstein-Uhlenbeck process;
- $\alpha(t) = f(0,t) + \frac{\sigma^2}{2\kappa^2}(1-e^{-\kappa t})^2$.

`reconstructFutureYield` then prices the 10-year bond at every month and path, $y_{10}(t) = -\ln P(t, t+10)/10$. Below is scenario 0 of a block of 200, drawn with `RATE_SEED`. It is the same draw `economy.draw_rate_scenarios` makes, rebuilt step by step and checked against it.

<!-- gen:paths -->
| month | date | short rate r | 10Y yield (repriced) |
|---|---|---|---|
| 0 | Sep 2026 | 2.907% | 4.102% |
| 12 | Sep 2027 | 4.088% | 4.832% |
| 24 | Sep 2028 | 4.562% | 5.128% |
| 36 | Sep 2029 | 4.560% | 5.128% |
<!-- /gen -->

At month 0 the repriced 10Y is within a few basis points of the observed 4.12%. It is not exactly equal: the model reprices the NSS curve, and the NSS fit has a 1–3 bp error.

### 2.5 The contract rates: WAP fixings and book yield
The observed 10Y history is placed in front of the simulated months.

**WAP fixing.** Model year $k$ starts $12k$ months after the last observation. Its WAP rate averages the 24 months ending 8 months earlier (the 1 June → 1 January lag), times 0.85. The result is rounded to 25 bp and clipped to $[1.75\%, 3.75\%]$.

<!-- gen:wap -->
| year k | 24-month window | data | average 10Y | x 0.85 | G_k (25 bp grid, [1.75%, 3.75%]) |
|---|---|---|---|---|---|
| 0 | Feb 2024 - Jan 2026 | observed | 3.073% | 2.612% | 2.50% |
| 1 | Feb 2025 - Jan 2027 | mixed | 3.481% | 2.959% | 3.00% |
| 2 | Feb 2026 - Jan 2028 | mixed | 4.212% | 3.581% | 3.50% |
<!-- /gen -->

Year 0 uses observed data only, so $G_0 = 2.50\%$, which is the rate in force today.

**Book yield.** The credited return $\mu_k$ is the mean of the 10Y over the 8 years up to the start of year $k$. It reflects a Branch-21 portfolio that rolls over slowly.

<!-- gen:book -->
| year k | 8-year window | mu_k |
|---|---|---|
| 0 | Oct 2018 - Sep 2026 | 1.736% |
| 1 | Oct 2019 - Sep 2027 | 2.224% |
| 2 | Oct 2020 - Sep 2028 | 2.864% |
<!-- /gen -->

Both rates come from the same OLO path, which is what correlates the guarantee with the reserve.

### 2.6 The DP's view of the same scenarios
The DP cannot carry a rate path in its state. `certainty_equivalent` replaces the 5000 scenarios by constant moments. `solve` then optimises on $(t, F, \rho)$ with these moments:

<!-- gen:ce -->
G = 3.619%, MU = 4.749%, SIGMA_L = 0.292%, SIGMA_R = 5.100% (from 5000 scenarios)
<!-- /gen -->

### 2.7 All randomness of the careers, and one year by hand
`Exogenous.draw(p, n, seed, rates)` holds everything random about $n$ careers: the shocks, the churn uniforms, and the path's $G_t$, $\mu_t$. For career 0:

<!-- gen:exo -->
| year t | zR | zL | u (churn) | hazard h(t) | G_t | mu_t |
|---|---|---|---|---|---|---|
| 0 | +0.0012 | -1.2466 | 0.8037 | 0.1200 | 2.50% | 1.736% |
| 1 | +1.0709 | -0.9194 | 0.1886 | 0.1074 | 3.00% | 2.224% |
| 2 | +0.8968 | -2.5227 | 0.7211 | 0.0964 | 3.50% | 2.864% |
<!-- /gen -->

Year 0 of career 0 under the solved policy, each line one operation of `dynamics.step`:

<!-- gen:step -->
- Entry: R = 1.0001, L = 1.0000, S = 17.5412, so F = 1.0001, rho = 17.541.
- Policy: a = a*(0, F, rho) = 0.9992, so c = a * GAMMA * S = 0.9992 x 0.15 x 17.541 = 2.6292.
- Reserve: R' = (R + c) e^(mu_0 + 0.05 zR) = (1.0001 + 2.6292) e^(1.736% + 0.05 x +0.0012) = 3.6930.
- Ledger: vintage 0 (opening L, locked at 2.50%) and vintage 1 (this contribution, locked at G_0 = 2.50%); L' = 3.7210.
- Salary: S' = S (1 + W) = 17.9798; churn: u = 0.8037 vs h(0) = 0.1200 -> stays.
- New state: F' = 0.9925, rho' = 4.832.
- Reward of year 0: -w_er x a GAMMA (1+W)^-(T-0) x e^0 = -0.5 x 0.9992 x 0.15 x 0.3292 = -0.024669 (final-salary units; S_T = 53.29).
<!-- /gen -->

### 2.8 From careers to the objective
`settle` pays $\max(R_T, L_T)$, and the employer covers $(L_T - R_T)^+$. The `Objective` values the outcome per unit of final salary $S_T$:

$$
J = \lambda\, e^{-\delta_e T}\, \text{employee}(\text{RR}, \text{target}) \;-\; (1-\lambda)\Big[\sum_t e^{-\delta_f t}\,\frac{c_t}{S_T} + e^{-\delta_f T}\,\frac{(L_T - R_T)^+}{S_T}\Big].
$$

`PensionEnv` pays the first term at retirement, the contribution terms year by year, and the shortfall at retirement. So an episode's return is exactly one path's contribution to $J$, and the mean over the careers is `simulate(...)["joint"]`:

<!-- gen:joint -->
- Career 0: leaves before T; final total replacement rate 0.708.
- Its return: contributions -0.160215 plus the terminal reward +0.281377 = +0.121162.
- Mean return over the 200 careers: +0.04030808.
- simulate(...)["joint"] on the same paths: +0.04030808 (difference 3.5e-17).
- Of which benefit 0.550075 (employee leg, weight 0.5) and cost 0.469458 (employer leg, weight 0.5).
<!-- /gen -->

---

## 3. One career by hand: constant rates against Hull-White

**Setup.**
- Member: $R_0 = 1$, $L_0 = 1$, $S_0 = 20$ (so $F = 1$, $\rho = 20$).
- Action: $a = 0.5$ every year, i.e. a contribution of 7.5% of salary.
- Churn: none.
- Shocks: the same $z_R, z_L$ in both regimes. The Hull-White case uses the rates of scenario 0 from Section 2.

<!-- gen:shocks -->
| year t | zR | zL | G_t (HW) | mu_t (HW) |
|---|---|---|---|---|
| 0 | +0.3456 | +0.8216 | 2.50% | 1.736% |
| 1 | -1.3032 | +0.9054 | 3.00% | 2.224% |
| 2 | -0.5370 | +0.5811 | 3.50% | 2.864% |
<!-- /gen -->

**Constant rates.** $R$ is credited at $\mu = 3\%$ with noise $\sigma_R z_R$. $L$ is one pot growing at $G = 3\%$ with noise $\sigma_L z_L$.

<!-- gen:career_constant -->
| after year | S | c = a Γ S | R growth | R | ledger (amount@locked G) | L | F = R/L | ρ = S/L | reward of the year |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 20.000 |  |  | 1.0000 | one pot | 1.0000 | 1.0000 | 20.000 |  |
| 1 | 20.500 | 1.5000 | e^(3.00% +0.0173) | 2.6210 | one pot | 2.6188 | 1.0008 | 7.828 | -0.012344 |
| 2 | 21.012 | 1.5375 | e^(3.00% -0.0652) | 4.0149 | one pot | 4.3612 | 0.9206 | 4.818 | -0.012036 |
| 3 | 21.538 | 1.5759 | e^(3.00% -0.0268) | 5.6085 | one pot | 6.1894 | 0.9061 | 3.480 | -0.011735 |
<!-- /gen -->

**Hull-White.** $R$ is credited at the book yield $\mu_t$ (noise $\sigma_R z_R$). Each contribution is a separate vintage of the guarantee, locked at the $G_t$ of its year.

<!-- gen:career_hw -->
| after year | S | c = a Γ S | R growth | R | ledger (amount@locked G) | L | F = R/L | ρ = S/L | reward of the year |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 20.000 |  |  | 1.0000 | 1.0000@2.50% | 1.0000 | 1.0000 | 20.000 |  |
| 1 | 20.500 | 1.5000 | e^(1.74% +0.0173) | 2.5881 | 1.0253@2.50%; 1.5380@2.50% | 2.5633 | 1.0097 | 7.998 | -0.012344 |
| 2 | 21.012 | 1.5375 | e^(2.22% -0.0652) | 3.9523 | 1.0513@2.50%; 1.5769@2.50%; 1.5843@3.00% | 4.2125 | 0.9382 | 4.988 | -0.012036 |
| 3 | 21.538 | 1.5759 | e^(2.86% -0.0268) | 5.5381 | 1.0779@2.50%; 1.6168@2.50%; 1.6326@3.00%; 1.6321@3.50% | 5.9594 | 0.9293 | 3.614 | -0.011735 |
<!-- /gen -->

What the tables show:
- **Contributions and rewards are identical in the two regimes.** They depend only on $a$, $t$ and salary. The regimes differ only in how $R$ and $L$ grow.
- **The rates diverge from the constant case within three years.**
  - Credited return: the Hull-White book yield starts at 1.74%, below the constant 3%, so $R$ grows more slowly.
  - Guarantee: the first two vintages lock at 2.50%, the third at 3.00% and the fourth at 3.50%.
  - Net effect: early on, the lower guarantee outweighs the lower return, and $F$ ends above the constant-rate path.
- **In the constant case, $F$ and $\rho$ match the reduced form exactly.** The script recomputes them with `dp.F_next` / `dp.rho_next` from $(F, \rho, a, z)$ alone, and they equal the level computation (asserted to $10^{-12}$).

---

## 4. Why $(t, F, \rho)$ is a sufficient state at constant rates

One year in levels is

$$
R' = (R + c)\,e^{\mu + \sigma_R z_R},\qquad L' = (L + c)\,e^{G + \sigma_L z_L},\qquad S' = (1+W)\,S,\qquad c = a\Gamma S.
$$

Divide by $L$ and write $l = c/L = a\Gamma\rho$:

$$
F' = \frac{F + l}{1 + l}\, e^{(\mu - G) + \sigma_R z_R - \sigma_L z_L},
\qquad
\rho' = \frac{(1+W)\,\rho}{(1+l)\, e^{G + \sigma_L z_L}}.
$$

These are `dp.F_next` and `dp.rho_next`.
- **Transitions.** The next $(F, \rho)$ depends only on the current $(F, \rho)$, the action and the shocks. The level of $L$ (or $R$, $S$) does not enter.
- **Contribution cost.** It is $c_t / S_T = a\Gamma(1+W)^{-(T-t)}$, which depends only on $(a, t)$.
- **Terminal value.** It depends only on $F_T$ and $\rho_T$:
$$
\text{RR} = \text{RR}_{\text{legal}} + \frac{\max(F_T, 1)}{\ddot a\,\rho_T},
\qquad
\frac{(L_T - R_T)^+}{S_T} = \frac{\max(1 - F_T, 0)}{\rho_T}.
$$
- **Leavers.** After leaving, nothing random remains: $F$ grows at $\mu$ and $\rho$ at $1+W$. The leaver's value is a deterministic function of $(\tau, F, \rho)$, which is `paidup_service`.

So the model is homogeneous of degree zero in $(R, L, S)$, and $(t, F, \rho)$ is a Markov state for the whole objective. The scale-invariance check confirms it to $10^{-15}$ (`checks.scale_invariance`). The constant-rate table in Section 3 is one instance.

---

## 5. Where it stops being sufficient: horizontal vintages at moving rates

Under the horizontal method each contribution keeps the rate it was booked at:

$$
L' = \sum_{v} L_v\, e^{G_v} + c\, e^{G_t}
\quad\Longrightarrow\quad
\frac{L'}{L} = \sum_v w_v\, e^{G_v} + l\, e^{G_t},
\qquad w_v = \frac{L_v}{L}.
$$

The growth of $L$ now depends on the **composition** of the ledger, i.e. the weights $w_v$ and their locked rates. $F$ and $\rho$ cannot see this.

**A counterexample.** Two members have the same $R$, $L$ and $S$, hence the same $F$ and $\rho$, and take the same action at the same rates. They differ only in when their guarantee was booked.

<!-- gen:not_markov -->
| member | vintages (amount@locked G) | F now | ρ now | F next year | L at T if nothing more is paid (43 years) |
|---|---|---|---|---|---|
| early money (locked low) | 1.0@1.75% + 0.5@2.50% | 0.8000 | 12.000 | 0.8994 | 8.715 |
| late money (locked high) | 0.5@2.50% + 1.0@3.75% | 0.8000 | 12.000 | 0.8932 | 11.763 |
<!-- /gen -->

Same $(F, \rho)$ today, different $F$ next year, and a liability at retirement that differs by about a third. So $(F, \rho)$ is not a Markov state here.

The rates themselves add more state.
- $G_{t+1}$ depends on the last 24 months of the 10Y, ending 8 months back.
- $\mu_{t+1}$ depends on the last 8 years of it.
- Under Hull-White the 10Y is a function of $(t, r_t)$, so a sufficient state is the short rate plus enough of its recent history to form those windows.

**What a minimal extended state looks like.**
- **Ledger.** $G$ only takes values on a 25 bp grid in $[1.75\%, 3.75\%]$, so the ledger compresses exactly into 9 buckets of "liability locked at rate $g$". Vintages with the same rate grow identically. `tests/test_dynamics.py` pins this compression against the full vintage ledger.
- **Rates.** $r_t$, plus the running sums behind the next WAP fixings and the book yield.

**Consequences.**
- **The DP uses the certainty-equivalent model (Section 2.6).** It keeps the tractable $(t, F, \rho)$ state and summarises the rate regime by moments. `simulate` then scores the resulting policy on the true path-wise rates. This is an approximation: the policy cannot react to the vintage mix or the rate history.
- **The RL observation is partially informative.** It is $(t/T, F, \log\rho, G_t, \mu_t)$: it carries today's rates but neither the bucket shares nor the rate history. The natural additions are
  - the 9 bucket shares $L_g/L$ (or their weighted mean rate),
  - the short rate $r_t$ or the running WAP average.

  This is also where an RL agent can in principle beat the certainty-equivalent DP.

---

## 6. Reproduce

```bash
python experiments/walkthrough.py            # print all blocks
python experiments/walkthrough.py --write    # refresh this document
python experiments/walkthrough.py --check    # fail if this document is out of date
```

The scenario, the career noise and the entry cohort are fixed by seeds (`RATE_SEED`, 7). The OLO data is the cached CSV, so the numbers change only when the code, the parameters or the data vintage change.
