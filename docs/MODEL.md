# The model

The baseline as `pension/baseline.py` implements it: one member, a fixed premium, and each year the agent chooses the share of that premium that goes to Branch 21. It is a fully observed MDP. Every equation below names the function that implements it.

Other rungs: the tabular rung (`pension/envs/tabular.py`), the constant-rate single-vehicle DP (`pension/dp.py`, the first-principles reference), and the pre-October-2026 model frozen in `pension/legacy_v1/` (its model document is `pension/legacy_v1/MODEL.md`).

## 1. Units and timing

- Model year $t = 0,\dots,T-1$ is a calendar year. Year 0 starts in the first January after the last OLO observation. The 10Y is simulated monthly. A year's state is read in January, with that month's yield observed.
- Money is in units of the **current salary** $S_t$, and $S_{t+1} = (1+W)S_t$. The reward is in units of $S_T$, in retirement-date money under $P$. Terminal legs are undiscounted, and no market price of risk enters anywhere.
- Premium $P_t = c\,S_t$, split as $P^{21}_t = \alpha_t P_t$ and $P^{23}_t = (1-\alpha_t)P_t$.
- Within year $t$: the premium is paid in January; $L$ and $R^{21}$ accrue at the rates fixed in that January; twelve months of rates follow (February to next January); then the member leaves with probability $h(t)$, at service $\tau = t+1$.
- After exit, premiums stop and $L$ is fixed at $L_\tau$, but the reserves accrue until $T$: $R^{21}$ at $i_t$, and $R^{23}$ by holding its units to $T$. This asymmetry is intended. The guarantee is frozen when the member leaves, but the assets are not liquidated.
- New member: $L_0 = R^{21}_0 = N_0 = 0$.

## 2. State and action

| variable | meaning | why it is needed |
|---|---|---|
| $t$ | year | horizon; $P(t,T)$, $h(t)$ and the target depend on $t$ |
| $y_t$ | 10Y OLO, January | Vasicek is Markov in $y$; gives $r_t$ and $P(t,T)$ |
| $G_t$ | WAP rate of year $t$ | accrual of $L$ this year |
| $H_t$ | sum of the 20 observed months (Jun $t{-}2$ … Jan $t$) of $G_{t+1}$'s window | $G_{t+1}$ |
| $K_t$ | sum of the 8 observed months (Jun $t{-}1$ … Jan $t$) of $G_{t+2}$'s window | $H_{t+1}$ |
| $\mu_t$ | book yield (exponential filter) | $i_t$, and $\mu_{t+1}$ |
| $L_t$, $R^{21}_t$, $N_t$ | liability, Branch 21 reserve, Branch 23 units | balances |

Nine variables (`State.obs`, which divides $t$, $H$ and $K$ by $T$, 20 and 8). Every transition and the reward are functions of $(s_t,\alpha_t,\varepsilon_t)$, where $\varepsilon_t$ is the year's 12 monthly normals and one exit uniform. The in-force flag and $\tau$ are not needed: exit ends the episode, so every decision state is in force with $\tau = t$. `State` carries both only for the roll-forward to $T$.

**Action:** $\alpha_t \in [0,1]$.

## 3. Equations

| # | equation | function |
|---|---|---|
| (1) | $y_{m+1} = y_m e^{-\kappa/12} + \theta(1-e^{-\kappa/12}) + \sigma\sqrt{\tfrac{1-e^{-\kappa/6}}{2\kappa}}\,\varepsilon_{m+1}$ (monthly, exact) | `vasicek_month` |
| (2) | $r_t = y_t - s_{\text{term}}$ | `short_rate` |
| (3) | $G_{t+1} = \mathrm{clip}\big(\mathrm{round}_{25\text{bp}}(0.85\cdot\tfrac{1}{24}\textstyle\sum_{\text{Jun }t-2}^{\text{May }t} y),\ 1.75\%,\ 3.75\%\big)$, i.e. $\tfrac{1}{24}(H_t + \sum_{\text{Feb}}^{\text{May}} y)$; then $H_{t+1} = K_t + \sum_{\text{Feb }t}^{\text{Jan }t+1} y$, $K_{t+1} = \sum_{\text{Jun }t}^{\text{Jan }t+1} y$ | `wap_rate`, `step` |
| (4) | $\mu \leftarrow \mu + w\,(y - \mu)$ every month | `book_yield` |
| (5) | $i_t = \max(i^{\text{tariff}},\ \mu_t - m)$ | `credited_rate` |
| (6) | $L_{t+1} = (L_t + P_t)\,e^{G_t}$, $R^{21}_{t+1} = (R^{21}_t + P^{21}_t)\,e^{i_t}$: one recursion, two rates | `accrue` |
| (7) | $P(t,T) = e^{-y_t (T-t)}$ (flat at the 10Y), $P(T,T) = 1$ | `bond_price` |
| (8) | $N_{t+1} = N_t + P^{23}_t / P(t,T)$, $R^{23}_t = N_t P(t,T)$, so $R^{23}_T = N_T$ | `buy_units` |
| (9) | $h(t) = h_\infty + (h_0 - h_\infty)\,e^{-t/\tau_h}$ | `hazard` |
| (10) | reward at $T$ (Sec. 4) | `reward_legs` |
| (11) | $A(t,T) = \mathbb E^P_t[e^{\int_t^T r_u du}] = \exp\!\big(\theta\tau + (y_t-\theta)B + \tfrac12 V - s_{\text{term}}\tau\big)$, with $\tau = T-t$, $B = \tfrac{1-e^{-\kappa\tau}}{\kappa}$, $V = \tfrac{\sigma^2}{\kappa^2}(\tau - B) - \tfrac{\sigma^2}{2\kappa}B^2$ | `premium_accrual` |

In salary units, (6) and (8) are divided by $1+W$. Out of force, $P_t = 0$, $L$ accrues at rate 0, and $R^{21}$ still accrues at $i_t$. The year (premium, accrual, 12 months, exit) is `step`; January of year 0 comes from the observed history plus the pre-roll months (`initial_state`).

## 4. Reward

With $R_T = R^{21}_T + R^{23}_T$,

$$
\text{target}_\tau = \text{RR}_{\text{LEGAL}} + \tfrac{\tau}{T}\left(\text{RR}_{\text{TARGET}} - \text{RR}_{\text{LEGAL}}\right),\qquad
\text{RR}_{\text{tot}} = \text{RR}_{\text{LEGAL}} + \frac{\max(R_T, L_\tau)}{\ddot a\,S_T},
$$

$$
J = \lambda\,\mathbb E\!\left[\text{target}_\tau\,\ddot a\;u\!\left(\frac{\text{RR}_{\text{tot}}}{\text{target}_\tau}\right)\right] - (1-\lambda)\,\mathbb E\!\left[\frac{(L_\tau - R_T)^+}{S_T}\right],
\qquad u(x) = \frac{x^{1-\eta}-1}{1-\eta}.
$$

- The shortfall $(L_\tau - R_T)^+$ is borne by the employer and settled at $T$.
- The reward is paid once, at $T$; it is zero before. $\gamma = 1$.
- The premium cost $\sum_{t<\tau} c\,(1+W)^{-(T-t)}A(t,T)$ does not depend on $\alpha$. It is left out of the reward and reported by `simulate()` as `premium_cost`.
- The WAP minimum, the 80% rule and non-discrimination are not part of the optimisation.

## 5. Environment and evaluation

- `PensionEnv`: the observation is `State.obs` (9 values) and the action is $\alpha_t$.
  - When the member leaves, the remaining years are rolled forward on the episode's own noise, and the reward at $T$ is paid in that step.
  - Training episodes draw their noise from a pool seeded by `TRAIN_SEED`. `reset(options={"noise": ..., "state": ...})` replays given noise, optionally from a snapshot.
- `simulate(policy)`: the same `step` over many careers on noise seeded by `EVAL_SEED`. It returns `joint` $= J$, the two legs, the mean shortfall and its frequency, `premium_cost`, the per-path `reward`, and per-year rate diagnostics.

## 6. Parameters

All parameters are in `baseline.Params`; the default is `BASELINE`. The statutory WAP constants live in `pension/rates/wap.py`.

<!-- gen:defaults -->
| symbol | field | default |
|---|---|---|
| $T$ | `T` | 45 |
| $W$ | `W` | 0.025 |
| $c$ | `C` | 0.05 |
| $i^{\text{tariff}}$ | `I_TARIFF` | 0.01 |
| $m$ | `M` | 0.0 |
| $w$ (per month) | `BOOK_WEIGHT` | 0.02083 (1/48) |
| $\lambda$ | `LAMBDA` | 0.5 |
| $\eta$ | `ETA` | 2.0 |
| $\ddot a$ | `ANNUITY` | 15.0 |
| $\text{RR}_{\text{LEGAL}}$ | `RR_LEGAL` | 0.43 |
| $\text{RR}_{\text{TARGET}}$ | `RR_TARGET` | 0.7 |
| $h_0$, $h_\infty$, $\tau_h$ | `H0`, `HINF`, `HTAU` | 0.12, 0.025, 7.0 |
| long-run short rate | `LONG_RATE_P` | None |
| $s_{\text{term}}$ | `SPREAD_10Y_SHORT` | None |
| seeds | `TRAIN_SEED`, `EVAL_SEED` | 2026, 7 |
<!-- /gen -->

<!-- gen:vasicek_params -->
Calibration (OLS on 321 monthly 10Y observations up to Sep 2026): $\kappa$ = 0.1104, $\theta$ = 2.23% (`LONG_RATE_P` = None: the OLS estimate; long-run short rate 0.95%), $\sigma$ = 0.61%, $y_0$ = 4.12%; $s_{\text{term}}$ = 128.4 bp (mean 10Y - 1Y over the same months).
<!-- /gen -->
