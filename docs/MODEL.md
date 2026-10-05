# The model

This document describes the model as the code implements it in the canonical configuration: `EMPLOYER_NUMERAIRE = "retirement"`, `RATE_MODEL = "vasicek"`, and constant rates as the DP-verifiable rung. Where code and intention differ, it documents the code and refers to `docs/REVIEW.md`.

## 1. Units and symbols

All money is expressed **per unit of final salary** $S_T$, and in **retirement-date money**, i.e. valued at $T$. A premium paid at $t$ is carried to $T$ at the short rate. The terminal payments are not discounted.

| symbol | meaning | unit | code |
|---|---|---|---|
| $t = 0,\dots,T-1$ | model year (year 0 starts at t0, the last OLO observation) | years | loop index |
| $R_t$, $L_t$, $S_t$ | reserve, guaranteed reserve (WAP liability), salary | money | `State.R`, `State.L`, `State.S` |
| $F_t = R_t/L_t$, $\rho_t = S_t/L_t$ | funding ratio, salary-to-liability ratio | – | `State.F`, `State.rho` |
| $a_t\in[0,1]$ | funding decision, share of capacity | – | action |
| $c_t = a_t\,\Gamma\,S_t$ | premium | money | `step` |
| $\mu_t$, $G_t$ | credited log-return, WAP guarantee rate | 1/yr | `Exogenous.mu`, `.G` (constant: `MU`, `G`) |
| $r_t$, $s$ | short rate, financing spread | 1/yr | `Exogenous.r` (constant: `SHORT_RATE`), `FINANCING_SPREAD` |
| $z_R$, $z_L$ | asset and guarantee shocks, $N(0,1)$ | – | `Exogenous.zR`, `.zL` |
| $\tau$ | year of leaving ($T$ if the member stays) | years | `State.leave_t` |
| $h(t)$ | churn hazard | 1/yr | `dp.tenure_hazard` |
| $\lambda$ | employee weight; $(w_{emp}, w_{er}) = (\lambda, 1-\lambda)$ | – | `LAMBDA` |
| $u$ | normalised CRRA utility, $u(x)=\frac{x^{1-\eta}-1}{1-\eta}$ ($\log$ at $\eta=1$) | – | `objective.u` |
| $\ddot a$ | annuity factor | years | `ANNUITY` |

Defaults:

<!-- gen:defaults -->
| symbol | Params field | default |
|---|---|---|
| $T$ | `T` | 45 |
| $\Gamma$ | `GAMMA` | 0.15 |
| $W$ | `W` | 0.025 |
| $\lambda$ | `LAMBDA` | 0.5 |
| $\eta$ | `ETA` | 2.0 |
| $\ddot a$ | `ANNUITY` | 15.0 |
| $\text{RR}_{legal}$ | `RR_LEGAL` | 0.43 |
| $\text{RR}^*$ | `RR_TARGET` | 0.7 |
| $\mu$, $G$ (constant) | `MU`, `G` | 0.03, 0.03 |
| $r$ (constant) | `SHORT_RATE` | 0.03 |
| $s$ | `FINANCING_SPREAD` | 0.0 |
| $\sigma_R$, $\sigma_L$ | `SIGMA_R`, `SIGMA_L` | 0.05, 0.02 |
| $\sigma_R$ (rate mode) | `SIGMA_R_RATES` | 0.05 |
<!-- /gen -->

## 2. The objective

With $\text{RR}_\text{tot} = \text{RR}_\text{legal} + \dfrac{\max(R_T, L_\tau)}{\ddot a\, S_T}$, where $L_\tau$ is the liability frozen when the member leaves, and the service-pro-rated target
$$\text{target}_\tau = \text{RR}_\text{legal} + \frac{\tau}{T}\left(\text{RR}^\* - \text{RR}_\text{legal}\right),$$
the `baseline` objective that `dp.solve` maximises and `dp.simulate` scores is

$$
J \;=\; \lambda\,\mathbb E^P\!\left[\text{target}_\tau\,\ddot a\;u\!\left(\frac{\text{RR}_\text{tot}}{\text{target}_\tau}\right)\right]
\;-\;(1-\lambda)\,\mathbb E^P\!\left[\sum_{t<\tau} a_t\,\Gamma\,(1+W)^{-(T-t)}\,A(t,T)
\;+\;\frac{(L_\tau - R_T)^+}{S_T}\right],
$$

$$
A(t,T) = \mathbb E^P_t\!\left[\exp\!\int_t^T (r_u + s)\,du\right].
$$

- $(1+W)^{-(T-t)}$ converts a year-$t$ premium into final-salary units. It is a change of units, not a discount.
- $A(t,T)$ is the conditional expectation given $r_t$, so each reward is known when it is paid.
- The terminal legs are paid at $T$ and are not discounted.

The other registered objectives (`docs/REFERENCE.md`) replace the employee curve, the shortfall cost, the contribution cost or the weights. The valuation-date factors (`pension/numeraire.py`) are applied around them.

**Closed forms of $A(t,T)$** (`rates/accrual.py`, $\tau = T-t$, $B(\tau) = (1-e^{-\kappa\tau})/\kappa$):
- **constant rates:** $A = e^{(r+s)\tau}$.
- **vasicek:** the modelled rate is the 10Y $y$ (κ, θ, σ under $P$), and $r = y - \text{spread}$:
$$
\text{mean}(I_y) = \theta\tau + (y_t-\theta)B(\tau),\quad
\text{var}(I_y) = \frac{\sigma^2}{\kappa^2}\left(\tau - B(\tau)\right) - \frac{\sigma^2}{2\kappa}B(\tau)^2,
$$
$$
A(t,T) = \exp\!\left(\text{mean} + \tfrac12\text{var} + (s - \text{spread})\,\tau\right).
$$

<!-- gen:vasicek_params -->
Canonical vasicek: $\kappa$ = 0.1104, $\theta$ = 2.23% (OLS; `LONG_RATE_P` = None), $\sigma$ = 0.61%, $y_0$ = 4.12%, spread = 128.4 bp.
<!-- /gen -->

`LONG_RATE_P` replaces θ, the asymptotic mean of the **10Y**, so the implied long-run short rate is θ − spread. In the other real-world models `LONG_RATE_P` is a short rate (REVIEW M1).

## 3. Why the retirement date

Both terminal payments, the pension $\max(R_T, L_\tau)$ and the employer's top-up $(L_\tau - R_T)^+$, fall at $T$, leavers included: the WAP shortfall is settled at retirement. So retirement-date money is the natural unit.

Under constant rates the legacy inception-date objective ("discounted": premiums at $e^{-\delta_f t}$, the employee leg at $e^{-\delta_e T}$, the shortfall at $e^{-\delta_f T}$) with $\delta_f = r$ equals the retirement-date objective times the positive constant $e^{-rT}(1-\lambda)/(1-\lambda')$, at the remapped weight
$$
\lambda' = \frac{\lambda e^{-\delta_e T}}{\lambda e^{-\delta_e T} + (1-\lambda) e^{-rT}}.
$$
The argmax is unchanged. This is checked to $5\times10^{-10}$ in `tests/test_numeraire.py`.

<!-- gen:lambda_prime -->
With the legacy defaults ($\lambda$ = 0.5, $\delta_f$ = 0.05, $\delta_e$ = 0.03, $T$ = 45) the inception-date objective equals the retirement-date one at $\lambda'$ = 0.7109 with $r = \delta_f$.
<!-- /gen -->

## 4. One year of a career (`dynamics.step`)

For every path, with $a_t = 0$ once the member has left:

| | in force | paid-up (after leaving) |
|---|---|---|
| premium | $c_t = a_t\Gamma S_t$ | 0 |
| reserve | $R_{t+1} = (R_t + c_t)\,e^{\mu_t + \sigma z_R}$ | $R_{t+1} = R_t\,e^{\mu_t}$ (no shock) |
| liability | see the ledgers | frozen: $L_{t+1}=L_t$ |
| salary | $S_{t+1} = (1+W)S_t$ | same |
| churn | leaves with probability $h(t)$: $u_t < h(t)$ | – |

$\sigma$ is `SIGMA_R` at constant rates and `SIGMA_R_RATES` under a rate model. In force, the mean growth of $R$ is $\mu + \sigma^2/2$; paid-up it is $\mu$ (REVIEW M6).

**Ledgers.**
- **Vertical** (constant rates): $L_{t+1} = (L_t + c_t)\,e^{G + \sigma_L z_L}$.
- **Horizontal** (rate models, the Branch-21 method): each premium is a vintage locked at the $G_t$ of its year. The opening liability is locked at $G_0$.
$$
L_{t+1} = \sum_{v\le t} L^{(v)}_t e^{G_v} + c_t e^{G_t},\qquad L = \sum_v L^{(v)}.
$$
- **At retirement** (`dynamics.settle`): payout $\max(R_T, L_T)$, shortfall $(L_T-R_T)^+$, service $\min(\tau/T, 1)$.

## 5. The reduced state $(t, F, \rho)$

**When it suffices.** Dividing the vertical, constant-rate year by $L$, with $l = a\Gamma\rho$:
$$
F' = \frac{F + l}{1+l}\,e^{(\mu-G) + \sigma_R z_R - \sigma_L z_L},
\qquad
\rho' = \frac{(1+W)\rho}{(1+l)\,e^{G+\sigma_L z_L}}
$$
(`dp.F_next`, `dp.rho_next`; contract C2).
- The premium cost $a\Gamma(1+W)^{-(T-t)}A(t,T)$ depends only on $(a,t)$.
- The terminal value depends only on $(F_T,\rho_T)$, because $\max(R,L)/S_T = \max(F,1)/\rho$ and $(L-R)^+/S_T = \max(1-F,0)/\rho$.

So the problem is homogeneous of degree 0 in $(R,L,S)$, and $(t,F,\rho)$ is a Markov state (`checks.scale_invariance`).

**Where it stops.** Under the horizontal ledger, $L'/L = \sum_v w_v e^{G_v} + l\,e^{G_t}$ with $w_v = L^{(v)}/L$. The growth of $L$ depends on the vintage mix, which $(F,\rho)$ cannot see. A counterexample is in `paper/explanations/model-walkthrough.md` §5.
- **Rate memory.** $G_{t+1}$ needs the trailing 24 months of the 10Y, $\mu_{t+1}$ the trailing 8 years, and $A$ needs $r_t$.
- **The ledger compresses exactly.** $G$ lives on a 25 bp grid in $[1.75\%, 3.75\%]$, so 9 buckets "liability locked at $g$" suffice (`tests/test_dynamics.py`).

## 6. The RL environment (`envs/pension_env.py`)

**Observation** (float64, 6): $(t/T,\ F,\ \log\rho,\ G_t,\ \mu_t,\ r_t)$, where $r_t = y_t - \text{spread}$ under vasicek and `SHORT_RATE` at constant rates. `DPPolicyAgent` reads components 0–2.

**Action:** $a\in[0,1]$, clipped to `band`.

**Reward per step:**
- each year in force: $-(1-\lambda)\,a\Gamma(1+W)^{-(T-t)}A(t,T;r_t)$;
- at $T$, or immediately when the member leaves (the remaining paid-up years are rolled forward inside the step): $\lambda\,\text{employee}(\cdot) - (1-\lambda)\,\text{shortfall}(\cdot)$.

$\gamma = 1$. The mean episode return equals `simulate(...)["joint"]` (contract C1). The observation does not contain the vintage mix or the rate history, so the problem is partially observed under rates.

## 7. The DP (`dp.solve`)

**Grids:** $F\in[0,3]$ linear (`make_F_grid`; $F=1$ is a node), $\rho\in[0.01,35]$ log-spaced, $a$ on a grid (or a band). **Backward induction from** $V_T = $ `terminal`:
$$
V_t(F,\rho) = \max_a\Big\{-w_{er}\,a\Gamma(1+W)^{-(T-t)}\,\pi_t
\;+\;\sum_q \omega_q\,\bar V_{t+1}\big(F'(z_q),\rho'(z_q)\big)\Big\},
\qquad
\bar V_{t+1} = (1-h(t))\,V_{t+1} + h(t)\,\Phi_{t+1}.
$$
- **Premium factor:** $\pi_t$ comes from `numeraire.premium_schedule`: $e^{(r+s)(T-t)}$ at constant rates.
- **Churn blend:** a leaver still pays year $t$; the freeze applies from $t+1$.
- **Paid-up value:** $\Phi_\tau(F,\rho)$ (`paidup_service`) is closed-form, since nothing random remains: $F\to Fe^{\mu m}$, $\rho\to\rho(1+W)^m$ with $m = T-\tau$, valued against $\text{target}_\tau$.
- **Quadrature:** a tensor-product Gauss-Hermite rule over $(z_R,z_L)$ (`gauss_hermite_2d`, default 5×5; converged, REVIEW m11).
- **Interpolation:** `bilinear` over $(F,\log\rho)$; off-grid points are clipped to the edge (REVIEW m10).
- **Policy extraction:** the argmax, or with `BETA > 0` a softmax over actions with temperature relative to the local Q-range (`_soft_readout`). $V$ is $\max_a Q$ either way.

**Rate mode: certainty equivalent.** The state cannot carry $r_t$, the WAP window or the vintage mix. `solve` draws `RATE_CE_PATHS` scenarios and solves the constant-rate DP with:
- $G, \mu$ = the scenario means;
- $\sigma_L$ = the std of $G_t$, and $\sigma_R = \sqrt{\sigma_{R,rates}^2 + \text{var}(\mu_t)}$ (REVIEW M10);
- a vertical ledger;
- $\pi_t$ = the scenario mean of $A(t,T;r_t)$.

The resulting policy is scored by `simulate` on the true paths.
