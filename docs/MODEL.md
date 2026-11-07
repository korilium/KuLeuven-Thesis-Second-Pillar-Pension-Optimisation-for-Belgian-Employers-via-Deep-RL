# The model

The baseline as `pension/baseline.py` implements it: one member, a fixed premium, and each year the agent chooses the share of that premium that goes to Branch 21. The rest buys a zero-coupon OLO maturing at retirement $T$ (a duration-matched ladder held in a Branch 23 fund). The liability follows the **horizontal** WAP method: each premium keeps the rate $G$ of the year it was paid in. It is a fully observed MDP. Every equation below names the function that implements it.

The baseline's DP reference is `pension/baseline_dp.py` (Sec. 7). Also kept: the tabular rung (`pension/envs/tabular.py`) and the pre-October-2026 model frozen in `pension/legacy_v1/` (its model document is `pension/legacy_v1/MODEL.md`).

## 1. Units and timing

- Model year $t = 0,\dots,T-1$ is a calendar year. Year 0 starts in the first January after the last OLO observation. The 10Y is simulated monthly. A year's state is read in January, with that month's yield observed.
- Money is in units of the **current salary** $S_t$, and $S_{t+1} = (1+W)S_t$. The reward is in units of $S_T$, in retirement-date money under $P$. Terminal legs are undiscounted. A market price of risk $\lambda_r$ enters only the price of the ladder bonds (Eq. 7); its default is 0.
- Premium $P_t = c\,S_t$, split as $P^{21}_t = \alpha_t P_t$ and $P^{23}_t = (1-\alpha_t)P_t$.
- Within year $t$: $\alpha_t$ is chosen in January, on the January state; twelve months of rates follow (February to next January); the premium is paid **mid-year (July)**. Each liability bucket and $R^{21}$ accrue at the rates fixed in that January ($g$, $i_t$): a full year on the January balance, half a year on the July premium. The ladder units are bought at the July price $P(t+\tfrac12,T)$. Then the member leaves with probability $h(t)$, at service $\tau = t+1$.
- After exit, premiums stop and every liability bucket is fixed in money at its value at $\tau$ (0% after exit), but the reserves accrue until $T$: $R^{21}$ at $i_t$, and $R^{23}$ by holding its units to $T$, net of the fund fee. This asymmetry is intended. The guarantee is frozen when the member leaves, but the assets are not liquidated.
- New member: $L^{(g)}_0 = R^{21}_0 = N_0 = 0$.

## 2. State and action

| variable | meaning | why it is needed |
|---|---|---|
| $t$ | year | horizon; $P(t,T)$, $h(t)$ and the target depend on $t$ |
| $y_t$ | 10Y OLO, January | Vasicek is Markov in $y$; gives $r_t$ and $P(t,T)$ |
| $G_t$ | WAP rate of year $t$ | the bucket that receives this year's premium |
| $H_t$ | sum of the 20 observed months (Jun $t{-}2$ … Jan $t$) of $G_{t+1}$'s window | $G_{t+1}$ |
| $K_t$ | sum of the 8 observed months (Jun $t{-}1$ … Jan $t$) of $G_{t+2}$'s window | $H_{t+1}$ |
| $\mu_t$ | book yield (exponential filter) | $i_t$, and $\mu_{t+1}$ |
| $R^{21}_t$, $N_t$ | Branch 21 reserve, Branch 23 (ladder) units | balances |
| $L^{(g)}_t$, $g \in \mathcal G$ | liability bucket: the premiums paid while $G = g$, accrued at $g$ | balances; $L_t = \sum_g L^{(g)}_t$ |

$\mathcal G = \{1.75\%, 2.00\%, \dots, 3.75\%\}$ is the statutory grid (9 rates, `wap_grid`, from the constants in `rates/wap.py`). The observation `State.obs` has 17 values, $(t/T,\,y,\,G,\,H/20,\,K/8,\,\mu,\,R^{21},\,N,\,L^{(g)}\times 9)$, with the buckets in grid order. Every transition and the reward are functions of $(s_t,\alpha_t,\varepsilon_t)$, where $\varepsilon_t$ is the year's 12 monthly normals and one exit uniform. The in-force flag and $\tau$ are not needed: exit ends the episode, so every decision state is in force with $\tau = t$. `State` carries both only for the roll-forward to $T$.

**Action:** $\alpha_t \in [0,1]$.

## 3. Equations

| # | equation | function |
|---|---|---|
| (1) | $y_{m+1} = y_m e^{-\kappa/12} + \theta(1-e^{-\kappa/12}) + \sigma\sqrt{\tfrac{1-e^{-\kappa/6}}{2\kappa}}\,\varepsilon_{m+1}$ (monthly, exact) | `vasicek_month` |
| (2) | $r_t = y_t - s_{\text{term}}$ | `short_rate` |
| (3) | $G_{t+1} = \mathrm{clip}\big(\mathrm{round}_{25\text{bp}}(0.85\cdot\tfrac{1}{24}\textstyle\sum_{\text{Jun }t-2}^{\text{May }t} y),\ 1.75\%,\ 3.75\%\big)$, i.e. $\tfrac{1}{24}(H_t + \sum_{\text{Feb}}^{\text{May}} y)$; then $H_{t+1} = K_t + \sum_{\text{Feb }t}^{\text{Jan }t+1} y$, $K_{t+1} = \sum_{\text{Jun }t}^{\text{Jan }t+1} y$. Hence $G_t \in \mathcal G$ | `wap_rate`, `wap_grid`, `step` |
| (4) | $\mu \leftarrow \mu + w\,(y - \mu)$ every month | `book_yield` |
| (5) | $i_t = \max(i^{\text{tariff}},\ \mu_t - m)$ | `credited_rate` |
| (6a) | $R^{21}_{t+1} = R^{21}_t\,e^{i_t} + P^{21}_t\,e^{i_t/2}$ (premium in July) | `accrue` |
| (6b) | $L^{(g)}_{t+1} = L^{(g)}_t\,e^{g} + P_t\,\mathbf 1\{G_t = g\}\,e^{g/2}$ for each $g \in \mathcal G$; $L_t = \sum_g L^{(g)}_t$. Each bucket is (6a) at its own rate $g$ | `accrue_liability`, `State.L` |
| (7) | $P(t,T) = A(\tau)\,e^{-B(\tau)\,r_t}$, $\tau = T-t$, $B = \tfrac{1-e^{-\kappa\tau}}{\kappa}$, $\ln A = \big(\theta_Q - \tfrac{\sigma^2}{2\kappa^2}\big)(B-\tau) - \tfrac{\sigma^2 B^2}{4\kappa}$, $\theta_Q = \theta - s_{\text{term}} - \lambda_r\sigma/\kappa$; $P(T,T) = 1$. Locked yield $y^{\text{long}}_t = -\ln P(t,T)/(T-t)$ | `zero_price`, `bond_price`, `locked_yield` |
| (7a) | $\lambda_r$ = `MPR`, or (default) the value at which the model's 10Y zero yield at $r = \theta_r$ is $\theta$: $\theta_Q = \tfrac{\sigma^2}{2\kappa^2} + \dfrac{B_{10}\theta_r + \sigma^2 B_{10}^2/(4\kappa) - 10\,\theta}{B_{10} - 10}$, $\lambda_r = (\theta_r - \theta_Q)\,\kappa/\sigma$ | `market_price_of_risk` |
| (8) | $N_{t+1} = (1-c_{\text{fee}})\,N_t + (1-c_{\text{fee}})^{1/2}\,P^{23}_t / P(t+\tfrac12,T)$, with $r$ at the July 10Y; $R^{23}_t = N_t P(t,T)$, so $R^{23}_T = N_T$ | `buy_units` |
| (9) | $h(t) = h_\infty + (h_0 - h_\infty)\,e^{-t/\tau_h}$ | `hazard` |
| (10) | reward at $T$ (Sec. 4) | `reward_legs` |
| (11) | $A(t,T) = \mathbb E^P_t[e^{\int_t^T r_u du}] = \exp\!\big(\theta\tau + (y_t-\theta)B + \tfrac12 V - s_{\text{term}}\tau\big)$, with $\tau = T-t$, $B = \tfrac{1-e^{-\kappa\tau}}{\kappa}$, $V = \tfrac{\sigma^2}{\kappa^2}(\tau - B) - \tfrac{\sigma^2}{2\kappa}B^2$ | `premium_accrual` |
| (12) | DP: $Q_t(s,\alpha_k) = \mathbb E\big[(1-h(t))\,V_{t+1}(s') + h(t)\,\Phi_{t+1}(s')\big]$, $V_t = \max_k Q_t$, $V_T$ = reward (Sec. 7) | `baseline_dp.solve` |

In salary units, (6a), (6b) and (8) are divided by $1+W$. Out of force, $P_t = 0$, every bucket accrues at rate 0, $R^{21}$ still accrues at $i_t$, and the units still pay the fee. The year (12 months, July premium, accrual, exit) is `step`; January of year 0 comes from the observed history plus the pre-roll months (`initial_state`).

**Ladder pricing (7).** The short rate $r = y - s_{\text{term}}$ inherits the 10Y dynamics (1): $dr = \kappa(\theta_r - r)\,dt + \sigma\,dW^P$, $\theta_r = \theta - s_{\text{term}}$. Sign convention: $dW^Q = dW^P + \lambda_r\,dt$, so $\theta_Q = \theta_r - \lambda_r\sigma/\kappa$, and under $P$ a zero-coupon bond earns $r - \lambda_r\sigma B$. A negative $\lambda_r$ therefore gives a positive term premium.

**Why $\lambda_r$ is calibrated (7a).** With $\lambda_r = 0$, the model's long yield tends to $\theta - s_{\text{term}} - \sigma^2/(2\kappa^2)$, below the long-run mean $\theta$ of the simulated 10Y by the term spread plus convexity. Branch 21 credits a smoothed 10Y with mean near $\theta$. So $\lambda_r = 0$ would make the ladder lose almost always, an artefact of the calibration that also contradicts the observed curve, where the 30Y trades above the 10Y. The default fixes $\lambda_r$ by one condition: the model's 10Y zero yield at $r = \theta_r$ equals $\theta$. With the calibration of Sec. 6 this gives $\lambda_r = -0.606$ and $\theta_Q = 4.28\%$. *Validation, not a fitting target:* at $r = \theta_r$ the model's 20Y−10Y, 25Y−10Y and 30Y−10Y spreads are 0.64%, 0.84% and 0.99%. The 2000–2026 OLO means are 0.50%, 0.61% and 0.64% (30Y on the 304 valid months). The model's long end is therefore somewhat too steep, which overstates the ladder's long locked yields by about 0.15–0.35 pp. The fee $c_{\text{fee}}$ runs over the time a unit is held: a full year on the units held in January, half a year on those bought in July. A unit bought at $t+\tfrac12$ is therefore worth $(1-c_{\text{fee}})^{T-t-1/2}$ in money at $T$.

**Branch 21 (4)–(6a)** is unchanged: one scalar reserve credited at the book-yield rate $i_t$, with $i^{\text{tariff}} = 0$. Every euro in it, old or new, earns the current $i_t$, so it behaves as a floating-rate asset. The ladder locks $y^{\text{long}}$ per premium. A locked per-premium tariff for Branch 21 is future work and is not implemented.

**Limitation.** The 10Y is the simulated factor (1), and the short rate is derived from it by a constant spread (2). The ladder is priced as if that short rate were the Vasicek factor (7). The 10Y yield implied by (7) is therefore not the simulated $y_t$: the curve is not internally consistent. The baseline accepts this, keeping the calibrated 10Y for the WAP rate and the book yield. The calibrated $\lambda_r$ (7a) closes the gap on average at the 10Y point, not path by path.

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
- The premium cost $\sum_{t<\tau} c\,(1+W)^{-(T-t)}A(t,T)$ does not depend on $\alpha$. It is left out of the reward and reported by `simulate()` as `premium_cost`. Eq. (11) still compounds each premium from January, while premiums are paid in July (Sec. 1). The reported cost is therefore overstated by about half a year of accrual per premium. This is reporting only: neither the reward nor any policy uses it.
- The WAP minimum, the 80% rule and non-discrimination are not part of the optimisation.

## 5. Environment and evaluation

- `PensionEnv`: the observation is `State.obs` (17 values) and the action is $\alpha_t$.
  - When the member leaves, the remaining years are rolled forward on the episode's own noise, and the reward at $T$ is paid in that step.
  - Training episodes draw their noise from a pool seeded by `TRAIN_SEED`. `reset(options={"noise": ..., "state": ...})` replays given noise, optionally from a snapshot.
- `simulate(policy)`: the same `step` over many careers on noise seeded by `EVAL_SEED`. It returns `joint` $= J$, the two legs, the mean shortfall and its frequency, `premium_cost`, the per-path `reward`, and per-year rate diagnostics (including the locked yield $y^{\text{long}}_t$).
- Benchmark policies (`baseline.py`): `all_branch21` ($\alpha = 1$), `all_ladder` ($\alpha = 0$), and `threshold_rule`, $\alpha_t = \mathbf 1\{i_t \ge y^{\text{long}}_t - c_{\text{fee}}\}$. The rule reads $y^{\text{long}}$ in January as a proxy for the July purchase price. It is optimal only in the known-answer configuration (`BOOK_WEIGHT` = 0: constant crediting, so Branch 21 is locked as well), up to that January/July proxy. By default Branch 21 credits a floating rate, so a gap between the threshold rule and the DP is the value of optimising, not an error.

## 6. Parameters

All parameters are in `baseline.Params`; the default is `BASELINE`. The statutory WAP constants live in `pension/rates/wap.py`.

| symbol | field | default |
|---|---|---|
| $T$ | `T` | 45 |
| $W$ | `W` | 0.025 |
| $c$ | `C` | 0.05 |
| $i^{\text{tariff}}$ | `I_TARIFF` | 0.0 |
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
| $\lambda_r$ (market price of rate risk, ladder price) | `MPR` | None (Eq. 7a: $-0.606$) |
| $c_{\text{fee}}$ (Branch 23 fund fee, per year) | `FUND_FEE` | 0.0 |
| seeds | `TRAIN_SEED`, `EVAL_SEED` | 2026, 7 |

Calibration (OLS on 321 monthly 10Y observations up to Sep 2026): $\kappa$ = 0.1104, $\theta$ = 2.23% (`LONG_RATE_P` = None: the OLS estimate; long-run short rate 0.95%), $\sigma$ = 0.61%, $y_0$ = 4.12%; $s_{\text{term}}$ = 128.4 bp (mean 10Y - 1Y over the same months).

## 7. DP reference (`pension/baseline_dp.py`)

Backward induction on the **exact** state (eight variables and nine liability buckets). A grid cannot run, so $Q$ is fitted by regression on simulated states (regression Monte Carlo, Longstaff–Schwartz style):

$$
Q_t(s,\alpha_k) = \mathbb E\big[(1-h(t))\,V_{t+1}(s') + h(t)\,\Phi_{t+1}(s') \,\big|\, s, \alpha_k\big],\qquad V_t(s) = Q_t(s,\pi_t(s)),\quad \pi_t = \arg\max_k \hat Q_t,\qquad V_T = \text{reward at } T. \tag{12}
$$

- **Next state.** $s'$ is the state after year $t$, from `baseline.step` itself.
- **Exit branch.** $\Phi_j$ is the value of leaving at $\tau = j$ (`leaver_value`). Nothing is decided after exit, so it is rolled forward exactly: every bucket is fixed in money, the units decay to $N_j(1-c_{\text{fee}})^{T-j}$, and $R^{21}$ grows by $\prod_{s\ge j} e^{i_s}$ (`paidup_growth`).
- **Targets (`rollout_target`).** The target of $Q_t(\cdot,\alpha_k)$ is not the fitted $\hat V_{t+1}$ but what the path actually earns. From $s'$, the already-fitted greedy policies $\pi_{t+1},\dots,\pi_{T-1}$ are followed on the training noise, in force. Exit enters as exact weights along that realised rate path, never sampled (Eq. 12 unrolled):
  $$h(t)\,\Phi_{t+1} + \sum_{j=t+1}^{T-1}\prod_{m=t}^{j-1}(1-h(m))\,h(j)\,\Phi_{j+1} + \prod_{m=t}^{T-1}(1-h(m))\cdot\text{reward}_T.$$
  *Why:* regressing on $\hat V_{t+1} = \max_k \hat Q_{t+1}$ selects the upward errors, since $\mathbb E[\max_k \hat Q_k] \ge \max_k \mathbb E[Q_k]$, and backward induction passes the overshoot back year by year. Under this baseline that blew up, with in-sample $V_0 = +12.5$ against $-0.42$ out of sample. Clipping bounds the values but keeps the selection bias. With realised targets the regression only ranks actions, so the estimate is a lower bound (the value of a possibly suboptimal policy), not an overshoot. The cost is about $K\,T^2/2$ steps.
- **Training states.** Paths are simulated in force with exploratory actions on the action grid (11 points). The exit hazard depends on $t$ only, so the in-force state distribution is the same with or without exit.
- **Fit.** For each year, every action's target is realised on the same noise, at $t$ and in the roll-forward. $Q_t(\cdot,\alpha_k)$ is its least-squares fit on `features(s)`.
- **Features.** The nine buckets enter only through summaries, which keeps the quadratic basis small:
  - the total $L_t$ and its weighted rate $\bar G_t = \sum_g g L^{(g)}_t / L_t$ (0 if $L_t = 0$);
  - the projected terminal liability $L^p_t = \sum_g L^{(g)}_t\big(\tfrac{e^g}{1+W}\big)^{T-t} + c\sum_{k=1}^{T-t}\tfrac{e^{G_t(k-1/2)}}{(1+W)^k}$ (future premiums paid in July at today's $G_t$);
  - the projected reserve $R^p_t = R^{21}_t\big(\tfrac{e^{i_t}}{1+W}\big)^{T-t} + N_t(1-c_{\text{fee}})^{T-t}(1+W)^{-(T-t)}$;
  - the spreads $y^{\text{long}}_t - c_{\text{fee}} - i_t$ and $y^{\text{long}}_t - G_t$, with their positive parts.
  
  The base list is $(y, G, H/20, K/8, \mu, L, R^{21}, N, \bar G, L^p, R^p$, the two spreads and their two positive parts$)$. The basis is a constant, that list, the reward legs at $(L^p, R^p)$, and all pairwise products of the list.
- **Policy.** `policy(sol)` is the greedy $\alpha$, a function of the observation (the exact state) only. `interior_share(sol)` reports the share of in-force evaluation decisions with $0 < \alpha < 1$.

This is the MDP itself: no state variable is dropped. The only error is the fit of $Q$, through the actions it ranks. The policy is scored out of sample by `simulate()`, which gives a lower bound on the optimum. The in-sample $V_0$ (the realised target of the greedy action at $t = 0$) is reported next to it.

## 8. Changelog

- **2026-11 — horizontal WAP liability, duration-matched OLO ladder** (tag `pre-horizontal` holds the previous baseline).
  - (3) $G$ lives on the 9-point statutory grid $\mathcal G$ (`wap_grid`).
  - (6) is split: (6a) $R^{21}$ as before; (6b) the liability is held in 9 buckets $L^{(g)}$, each accruing at its own $g$ (horizontal method), replacing the single liability at $G_t$ (vertical).
  - Premiums are paid in July: the year-$t$ premium earns half a year.
  - (7) The flat price $e^{-y_t(T-t)}$ is replaced by the Vasicek zero-coupon price of the short rate $r = y - s_{\text{term}}$, with $\theta_Q$ from a market price of risk $\lambda_r$ (`MPR`). (7a) calibrates $\lambda_r$ by default, so the model 10Y matches $\theta$ at $r = \theta_r$.
  - (8) The ladder is bought at the July price, and the units pay the fund fee $c_{\text{fee}}$ (`FUND_FEE`).
  - The observation has 17 values, and the benchmark policies are added.
  - (12) The DP uses realised (Longstaff–Schwartz) targets instead of the fitted $\hat V_{t+1}$, and bucket summaries as features.
  - The `I_TARIFF` default in Sec. 6 is corrected to 0.0, matching the code since commit 9080c3b.

