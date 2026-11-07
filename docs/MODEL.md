# The model

The baseline as `pension/baseline.py` implements it: one member, a fixed premium, and each year the agent chooses the share of that premium that goes to Branch 21. The rest buys a Branch 23 government bond fund of constant duration. The liability follows the **horizontal** WAP method: each premium keeps the rate $G$ of the year it was paid in. Branch 21 is horizontal too: each premium gets a tariff locked to retirement, plus profit-sharing on top. It is a fully observed MDP. Every equation below names the function that implements it.

The baseline's DP reference is `pension/baseline_dp.py` (Sec. 7). Also kept: the tabular rung (`pension/envs/tabular.py`) and the pre-October-2026 model frozen in `pension/legacy_v1/` (its model document is `pension/legacy_v1/MODEL.md`).

## 1. Units and timing

- Model year $t = 0,\dots,T-1$ is a calendar year. Year 0 starts in the first January after the last OLO observation. The 10Y is simulated monthly. A year's state is read in January, with that month's yield observed.
- Money is in units of the **current salary** $S_t$, and $S_{t+1} = (1+W)S_t$. The reward is in units of $S_T$, in retirement-date money under $P$. Terminal legs are undiscounted. A market price of risk $\lambda_r$ enters only the bond prices behind the fund's return (Eqs. 7–8); by default it is calibrated (Eq. 7a).
- Premium $P_t = c\,S_t$, split as $P^{21}_t = \alpha_t P_t$ and $P^{23}_t = (1-\alpha_t)P_t$.
- Within year $t$: $\alpha_t$ is chosen in January, on the January state; twelve months of rates follow (February to next January); the premium is paid **mid-year (July)**. Each liability bucket and each Branch 21 bucket accrue at their rates for the year: $g$, and $r_{k,t}$ from the tariff and from $\mu_t$ read in January. That is a full year on the January balance and half a year on the July premium. The July premium's Branch 21 share gets the tariff of the July book yield $\mu_{\text{Jul}}$. The rest is added to the bond fund after the fund's July return; the fund earns its return month by month all year. Then the member leaves with probability $h(t)$, at service $\tau = t+1$.
- After exit, premiums stop and every liability bucket is fixed in money at its value at $\tau$ (0% after exit), but the reserves accrue until $T$: every Branch 21 bucket at its tariff plus profit-sharing, and $R^{23}$ by staying in the fund, with its monthly returns net of the fee. This asymmetry is intended. The guarantee is frozen when the member leaves, but the assets are not liquidated.
- New member: $L^{(g)}_0 = R^{21,(k)}_0 = N_0 = 0$.

## 2. State and action

| variable | meaning | why it is needed |
|---|---|---|
| $t$ | year | horizon; $h(t)$ and the target depend on $t$ |
| $y_t$ | 10Y OLO, January | Vasicek is Markov in $y$; gives $r_t$, the tariff proxy and the fund's first monthly return |
| $G_t$ | WAP rate of year $t$ | the bucket that receives this year's premium |
| $H_t$ | sum of the 20 observed months (Jun $t{-}2$ … Jan $t$) of $G_{t+1}$'s window | $G_{t+1}$ |
| $K_t$ | sum of the 8 observed months (Jun $t{-}1$ … Jan $t$) of $G_{t+2}$'s window | $H_{t+1}$ |
| $\mu_t$ | book yield (exponential filter) | profit-sharing this year, and $\mu_{t+1}$ |
| $R^{23}_t$ | Branch 23 bond-fund value | balance |
| $L^{(g)}_t$, $g \in \mathcal G$ | liability bucket: the premiums paid while $G = g$, accrued at $g$ | balances; $L_t = \sum_g L^{(g)}_t$ |
| $R^{21,(k)}_t$, $i_k \in \mathcal I$ | Branch 21 bucket: the premiums whose tariff is $i_k$, accrued at tariff plus profit-sharing | balances; $R^{21}_t = \sum_k R^{21,(k)}_t$ |

$\mathcal G = \{1.75\%, 2.00\%, \dots, 3.75\%\}$ is the statutory WAP grid (9 rates, `wap_grid`, from the constants in `rates/wap.py`). $\mathcal I = \{0, 0.25\%, \dots, 3.75\%\}$ is the tariff grid (16 levels, `tariff_grid`, step `TARIFF_STEP`, cap 3.75%). The observation `State.obs` has 32 values, $(t/T,\,y,\,G,\,H/20,\,K/8,\,\mu,\,R^{23},\,L^{(g)}\times 9,\,R^{21,(k)}\times 16)$, with the buckets in grid order. Every transition and the reward are functions of $(s_t,\alpha_t,\varepsilon_t)$, where $\varepsilon_t$ is the year's 12 monthly normals and one exit uniform. The in-force flag and $\tau$ are not needed: exit ends the episode, so every decision state is in force with $\tau = t$. `State` carries both only for the roll-forward to $T$.

**Action:** $\alpha_t \in [0,1]$.

## 3. Equations

| # | equation | function |
|---|---|---|
| (1) | $y_{m+1} = y_m e^{-\kappa/12} + \theta(1-e^{-\kappa/12}) + \sigma\sqrt{\tfrac{1-e^{-\kappa/6}}{2\kappa}}\,\varepsilon_{m+1}$ (monthly, exact) | `vasicek_month` |
| (2) | $r_t = y_t - s_{\text{term}}$ | `short_rate` |
| (3) | $G_{t+1} = \mathrm{clip}\big(\mathrm{round}_{25\text{bp}}(0.85\cdot\tfrac{1}{24}\textstyle\sum_{\text{Jun }t-2}^{\text{May }t} y),\ 1.75\%,\ 3.75\%\big)$, i.e. $\tfrac{1}{24}(H_t + \sum_{\text{Feb}}^{\text{May}} y)$; then $H_{t+1} = K_t + \sum_{\text{Feb }t}^{\text{Jan }t+1} y$, $K_{t+1} = \sum_{\text{Jun }t}^{\text{Jan }t+1} y$. Hence $G_t \in \mathcal G$ | `wap_rate`, `wap_grid`, `step` |
| (4) | $\mu \leftarrow \mu + w\,(y - \mu)$ every month | `book_yield` |
| (5a) | Tariff of a July premium: $i = \mathrm{floor}_{\mathcal I}\big(\min(3.75\%,\ \max(0,\ \mu_{\text{Jul}} - m_{\text{tar}}))\big)$, floored to the grid step; $\mu_{\text{Jul}}$ is the book yield (4) after the July month | `tariff_grid`, `tariff` |
| (5b) | Bucket rate: $r_{k,t} = \max(i_k,\ \mu_t - m_{\text{ps}})$; profit-sharing $r_{k,t} - i_k \ge 0$ | `bucket_rate` |
| (6a) | $X_{t+1} = X_t\,e^{r} + P_t\,e^{r/2}$: a balance at rate $r$ with a July premium | `accrue` |
| (6c) | $R^{21,(k)}_{t+1} = R^{21,(k)}_t\,e^{r_{k,t}} + P^{21}_t\,\mathbf 1\{i = i_k\}\,e^{r_{k,t}/2}$ for each $i_k \in \mathcal I$; $R^{21}_t = \sum_k R^{21,(k)}_t$. Each bucket is (6a) at its rate | `accrue_branch21`, `State.R21` |
| (6b) | $L^{(g)}_{t+1} = L^{(g)}_t\,e^{g} + P_t\,\mathbf 1\{G_t = g\}\,e^{g/2}$ for each $g \in \mathcal G$; $L_t = \sum_g L^{(g)}_t$. Each bucket is (6a) at its own rate $g$ | `accrue_liability`, `State.L` |
| (7) | $P(\tau; r) = A(\tau)\,e^{-B(\tau)\,r}$, the zero-coupon price of maturity $\tau$: $B = \tfrac{1-e^{-\kappa\tau}}{\kappa}$, $\ln A = \big(\theta_Q - \tfrac{\sigma^2}{2\kappa^2}\big)(B-\tau) - \tfrac{\sigma^2 B^2}{4\kappa}$; $P(0; r) = 1$ | `zero_price` |
| (7a) | $\theta_Q = \theta - s_{\text{term}} - \lambda_r\sigma/\kappa$, with $\lambda_r$ = `MPR`, or (default) the value at which the model's 10Y zero yield at $r = \theta_r$ is $\theta$: $\theta_Q = \tfrac{\sigma^2}{2\kappa^2} + \dfrac{B_{10}\theta_r + \sigma^2 B_{10}^2/(4\kappa) - 10\,\theta}{B_{10} - 10}$, $\lambda_r = (\theta_r - \theta_Q)\,\kappa/\sigma$ | `theta_q`, `market_price_of_risk` |
| (8) | Bond fund: a zero-coupon of maturity $D$ rolled every month, with monthly gross return $\dfrac{P(D - \tfrac1{12};\, r_{m+1})}{P(D;\, r_m)}\,(1-c_{\text{fee}})^{1/12}$, $r_m = y_m - s_{\text{term}}$ (from the January 10Y to the next January's). Then $R^{23} \leftarrow R^{23}\cdot$ return each month, $+\,P^{23}_t$ right after the July return | `fund_return`, `accrue_fund` |
| (9) | $h(t) = h_\infty + (h_0 - h_\infty)\,e^{-t/\tau_h}$ | `hazard` |
| (10) | reward at $T$ (Sec. 4) | `reward_legs` |
| (11) | $A(t,T) = \mathbb E^P_t[e^{\int_t^T r_u du}] = \exp\!\big(\theta\tau + (y_t-\theta)B + \tfrac12 V - s_{\text{term}}\tau\big)$, with $\tau = T-t$, $B = \tfrac{1-e^{-\kappa\tau}}{\kappa}$, $V = \tfrac{\sigma^2}{\kappa^2}(\tau - B) - \tfrac{\sigma^2}{2\kappa}B^2$. A July premium accrues $A(t,T)\,e^{-r_t/2}$ (an approximation, Sec. 4) | `premium_accrual`, `july_premium_accrual` |
| (12) | DP: $Q_t(s,\alpha_k) = \mathbb E\big[(1-h(t))\,V_{t+1}(s') + h(t)\,\Phi_{t+1}(s')\big]$, $V_t = \max_k Q_t$, $V_T$ = reward (Sec. 7) | `baseline_dp.solve` |

In salary units, (6a)–(6c) and (8) are divided by $1+W$. Out of force, $P_t = 0$, every liability bucket accrues at rate 0, every Branch 21 bucket still accrues at $r_{k,t}$, and the fund keeps its monthly returns, net of the fee. The year (12 months, July premium, accrual, exit) is `step`; January of year 0 comes from the observed history plus the pre-roll months (`initial_state`).

**Bond pricing (7).** The short rate $r = y - s_{\text{term}}$ inherits the 10Y dynamics (1): $dr = \kappa(\theta_r - r)\,dt + \sigma\,dW^P$, $\theta_r = \theta - s_{\text{term}}$. Sign convention: $dW^Q = dW^P + \lambda_r\,dt$, so $\theta_Q = \theta_r - \lambda_r\sigma/\kappa$, and under $P$ a zero-coupon bond earns $r - \lambda_r\sigma B$. A negative $\lambda_r$ therefore gives a positive term premium. The fund holds a bond of maturity $D$, so it earns about $r + (-\lambda_r)\sigma B(D)$ a year under $P$, plus the gains and losses as $r$ moves. With $\sigma \to 0$ and $\lambda_r = 0$ it earns exactly $e^{\int r}$, whatever $D$.

**Why $\lambda_r$ is calibrated (7a).** With $\lambda_r = 0$, the model's long yield tends to $\theta - s_{\text{term}} - \sigma^2/(2\kappa^2)$, below the long-run mean $\theta$ of the simulated 10Y by the term spread plus convexity. Branch 21 credits a smoothed 10Y with mean near $\theta$. So $\lambda_r = 0$ would make bonds lose to Branch 21 almost always, an artefact of the calibration that also contradicts the observed curve, where the 30Y trades above the 10Y. The default fixes $\lambda_r$ by one condition: the model's 10Y zero yield at $r = \theta_r$ equals $\theta$. With the calibration of Sec. 6 this gives $\lambda_r = -0.606$ and $\theta_Q = 4.28\%$. *Validation, not a fitting target:* at $r = \theta_r$ the model's 20Y−10Y, 25Y−10Y and 30Y−10Y spreads are 0.64%, 0.84% and 0.99%. The 2000–2026 OLO means are 0.50%, 0.61% and 0.64% (30Y on the 304 valid months). The model's long end is therefore somewhat too steep, by about 0.15–0.35 pp at 20–30 years. At the fund's $D = 7$ the term premium matters less than at the long end. The fee is charged monthly: a premium added in July is worth $(1-c_{\text{fee}})^{(12(T-t)-6)/12}$ times its fee-free value at $T$.

**Branch 21 (5a)–(6c)** is horizontal. Each premium gets a tariff $i$ from the insurer's book yield in July, locked to retirement (the guaranteed floor).

*Why the book yield, not the 10Y.* A tariff of the July 10Y minus a margin, locked for up to 45 years with profit-sharing on top, made Branch 21 far more generous than any insurer. Observed group tariffs lag the market instead. AG Insurance's tariffs on new group premiums, against the 10Y OLO annual average, were 2.25% vs 2.41% (2013), 1.50% vs 0.84% (2015), 1.25% vs 3.09% (2023) and 1.75% vs 2.93% (2024). A tariff based on the smoothed book yield $\mu$ reproduces that lag.

The same filter run on the observed 10Y (from $\mu = y_{\text{Jan 2000}}$), with the default margin, gives July tariffs of 2.50%, 1.75%, 0.25% and 0.50% for those years. Those values are printed by `tests/test_baseline.py`; the comparison is not a fitting target. The model gets the 2013–2015 direction right: in 2015 the tariff sits above the 10Y. In 2023–2024 it is about 1 pp below AG, because the filter's July $\mu$ is only 1.26–1.67% there. `TARIFF_MARGIN` and `BOOK_WEIGHT` are left for calibration. On top, every bucket shares in the insurer's book yield: profit-sharing $\max(0,\ \mu_t - m_{\text{ps}} - i_k)$. A premium is therefore locked at its tariff and floating above it. The previous scalar crediting $i_t = \max(i^{\text{tariff}}, \mu_t - m)$ maps onto this: `I_TARIFF` becomes the per-premium tariff (5a), with its margin `TARIFF_MARGIN`, and `M` becomes `PS_MARGIN` (5b). Both old parameters are removed. The tariff is set in July, so a January decision only sees its proxy `new_premium_rate` $= r(\text{tariff}(\mu_{\text{Jan}}), \mu_t)$. That proxy is used by the $i - G$ diagnostic. The market-practice rule uses the tariff proxy $\text{tariff}(\mu_{\text{Jan}})$ alone. `simulate()` reports per year the reserve-weighted profit-sharing rate (`ps`) and the share of the reserve earning profit-sharing (`ps_share`).

**Limitation.** The 10Y is the simulated factor (1), and the short rate is derived from it by a constant spread (2). The fund's bonds are priced as if that short rate were the Vasicek factor (7). The 10Y yield implied by (7) is therefore not the simulated $y_t$: the curve is not internally consistent. The baseline accepts this, keeping the calibrated 10Y for the WAP rate and the book yield. The calibrated $\lambda_r$ (7a) closes the gap on average at the 10Y point, not path by path.

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
- The premium cost $\sum_{t<\tau} c\,(1+W)^{-(T-t)}A(t,T)\,e^{-r_t/2}$ does not depend on $\alpha$. It is left out of the reward and reported by `simulate()` as `premium_cost`. Premiums are paid in July, so each accrues from July. $A(t,T)\,e^{-r_t/2}$ is an *approximation*: it removes half a year at the January short rate and ignores the drift of $r$ over that half year. With $\sigma \to 0$ it differs from the exact $e^{\int_{t+1/2}^T r}$ by a relative $-6.5\cdot10^{-4}$ to $+3.0\cdot10^{-4}$ for $y_t$ between 0% and 7%. This is reporting only: neither the reward nor any policy uses it.
- The WAP minimum, the 80% rule and non-discrimination are not part of the optimisation.

## 5. Environment and evaluation

- `PensionEnv`: the observation is `State.obs` (32 values) and the action is $\alpha_t$.
  - When the member leaves, the remaining years are rolled forward on the episode's own noise, and the reward at $T$ is paid in that step.
  - Training episodes draw their noise from a pool seeded by `TRAIN_SEED`. `reset(options={"noise": ..., "state": ...})` replays given noise, optionally from a snapshot.
- `simulate(policy)`: the same `step` over many careers on noise seeded by `EVAL_SEED`. It returns `joint` $= J$, the two legs, the mean shortfall and its frequency, `premium_cost`, the per-path `reward`, and per-year rate diagnostics (including the locked yield $y^{\text{long}}_t$).
- Benchmark policies (`baseline.py`): `all_branch21` ($\alpha = 1$), `all_fund` ($\alpha = 0$), `fifty_fifty` ($\alpha = \tfrac12$), and `market_practice_rule`, $\alpha_t = \mathbf 1\{i^{\text{new}}_t \ge G_t\}$. Here $i^{\text{new}}_t = \text{tariff}(\mu_{\text{Jan}})$ is the tariff a premium would get at the January book yield, a proxy, since the real tariff is set at the July book yield. These are benchmarks, not the optimum: a gap between them and the DP is the value of optimising, not an error.

## 6. Parameters

All parameters are in `baseline.Params`; the default is `BASELINE`. The statutory WAP constants live in `pension/rates/wap.py`.

| symbol | field | default |
|---|---|---|
| $T$ | `T` | 45 |
| $W$ | `W` | 0.025 |
| $c$ | `C` | 0.05 |
| tariff grid step | `TARIFF_STEP` | 0.0025 |
| $m_{\text{tar}}$ (tariff margin below the July book yield) | `TARIFF_MARGIN` | 0.010 (PLACEHOLDER) |
| $m_{\text{ps}}$ (profit-sharing margin below $\mu_t$) | `PS_MARGIN` | 0.005 (PLACEHOLDER) |
| $w$ (per month) | `BOOK_WEIGHT` | 0.02083 (1/48) |
| $\lambda$ | `LAMBDA` | 0.5 |
| $\eta$ | `ETA` | 2.0 |
| $\ddot a$ | `ANNUITY` | 15.0 |
| $\text{RR}_{\text{LEGAL}}$ | `RR_LEGAL` | 0.43 |
| $\text{RR}_{\text{TARGET}}$ | `RR_TARGET` | 0.7 |
| $h_0$, $h_\infty$, $\tau_h$ | `H0`, `HINF`, `HTAU` | 0.12, 0.025, 7.0 |
| long-run short rate | `LONG_RATE_P` | None |
| $s_{\text{term}}$ | `SPREAD_10Y_SHORT` | None |
| $\lambda_r$ (market price of rate risk, bond prices) | `MPR` | None (Eq. 7a: $-0.606$) |
| $D$ (maturity the bond fund rolls) | `FUND_DURATION` | 7.0 (PLACEHOLDER) |
| $c_{\text{fee}}$ (Branch 23 fund fee, per year, charged monthly) | `FUND_FEE` | 0.0 (PLACEHOLDER) |
| seeds | `TRAIN_SEED`, `EVAL_SEED` | 2026, 7 |

Calibration (OLS on 321 monthly 10Y observations up to Sep 2026): $\kappa$ = 0.1104, $\theta$ = 2.23% (`LONG_RATE_P` = None: the OLS estimate; long-run short rate 0.95%), $\sigma$ = 0.61%, $y_0$ = 4.12%; $s_{\text{term}}$ = 128.4 bp (mean 10Y - 1Y over the same months).

## 7. DP reference (`pension/baseline_dp.py`)

Backward induction on the **exact** state (seven variables, nine liability buckets and 16 Branch 21 buckets). A grid cannot run, so $Q$ is fitted by regression on simulated states (regression Monte Carlo, Longstaff–Schwartz style):

$$
Q_t(s,\alpha_k) = \mathbb E\big[(1-h(t))\,V_{t+1}(s') + h(t)\,\Phi_{t+1}(s') \,\big|\, s, \alpha_k\big],\qquad V_t(s) = Q_t(s,\pi_t(s)),\quad \pi_t = \arg\max_k \hat Q_t,\qquad V_T = \text{reward at } T. \tag{12}
$$

- **Next state.** $s'$ is the state after year $t$, from `baseline.step` itself.
- **Exit branch.** $\Phi_j$ is the value of leaving at $\tau = j$ (`leaver_value`). Nothing is decided after exit, so it is rolled forward exactly along the path's own training rates. Every liability bucket is fixed in money, and each Branch 21 bucket grows by $\prod_{s\ge j} e^{r_{k,s}}$ (`paidup_growth`, per bucket). The fund grows by the product of its monthly returns from $j$ to $T$ (`paidup_fund_growth`). That product has no closed form, but the fund is linear in $R^{23}$ and its returns do not depend on the member. Replaying each training path's months once therefore gives the paid-up factor for every $j$ exactly.
- **Targets (`rollout_target`).** The target of $Q_t(\cdot,\alpha_k)$ is not the fitted $\hat V_{t+1}$ but what the path actually earns. From $s'$, the already-fitted greedy policies $\pi_{t+1},\dots,\pi_{T-1}$ are followed on the training noise, in force. Exit enters as exact weights along that realised rate path, never sampled (Eq. 12 unrolled):
  $$h(t)\,\Phi_{t+1} + \sum_{j=t+1}^{T-1}\prod_{m=t}^{j-1}(1-h(m))\,h(j)\,\Phi_{j+1} + \prod_{m=t}^{T-1}(1-h(m))\cdot\text{reward}_T.$$
  *Why:* regressing on $\hat V_{t+1} = \max_k \hat Q_{t+1}$ selects the upward errors, since $\mathbb E[\max_k \hat Q_k] \ge \max_k \mathbb E[Q_k]$, and backward induction passes the overshoot back year by year. Under this baseline that blew up, with in-sample $V_0 = +12.5$ against $-0.42$ out of sample. Clipping bounds the values but keeps the selection bias. With realised targets the regression only ranks actions, so the estimate is a lower bound (the value of a possibly suboptimal policy), not an overshoot. The cost is about $K\,T^2/2$ steps.
- **Training states.** Paths are simulated in force with exploratory actions on the action grid (11 points). The exit hazard depends on $t$ only, so the in-force state distribution is the same with or without exit.
- **Fit.** For each year, every action's target is realised on the same noise, at $t$ and in the roll-forward. $Q_t(\cdot,\alpha_k)$ is its least-squares fit on `features(s)`.
- **Clipping guard.** Each fitted $Q_t(\cdot,\alpha_k)$ is clipped to $[\min, \max]$ of the targets it was fitted on (`q_values`; `raw_q_values` gives the unclipped fit). An extrapolating fit in a tail state therefore cannot win the argmax. `clip_stats(sol)` reports, per year on the in-force evaluation states, how often the guard binds for some action, for the chosen action, and how often it changes the chosen action.
- **Features.** The buckets enter only through summaries, which keeps the quadratic basis small. No column is a derived affine function of $y$ at fixed $t$; such a column would only duplicate $y$ and its products.
  - rate state: $y_t$, $G_t$, $H_t/20$, $K_t/8$, $\mu_t$;
  - liability:
    - the total $L_t$;
    - the projected terminal liability $L^p_t = \sum_g L^{(g)}_t\big(\tfrac{e^g}{1+W}\big)^{T-t} + c\sum_{k=1}^{T-t}\tfrac{e^{G_t(k-1/2)}}{(1+W)^k}$, with future premiums paid in July at today's $G_t$;
    - the weighted rate $\bar G_t = \sum_g g L^{(g)}_t / L_t$ (0 if $L_t = 0$);
  - Branch 21:
    - the total $R^{21}_t$;
    - the guaranteed floor at $T$, $\sum_k R^{21,(k)}_t\big(\tfrac{e^{i_k}}{1+W}\big)^{T-t}$;
    - the projection at the current crediting, $\sum_k R^{21,(k)}_t\big(\tfrac{e^{\max(i_k,\,\mu_t - m_{\text{ps}})}}{1+W}\big)^{T-t}$;
    - the reserve-weighted tariff $\bar\imath_t$ (0 if $R^{21}_t = 0$);
  - Branch 23: $R^{23}_t$.

  The basis is a constant, that list of 13, the reward legs at ($L^p$, the Branch 21 projection $+\,R^{23}_t(1+W)^{-(T-t)}$; the fund has no fixed rate to project at), and all pairwise products of the list: 107 columns.
- **Policy.** `policy(sol)` is the greedy $\alpha$, a function of the observation (the exact state) only. `interior_share(sol)` reports the share of in-force evaluation decisions with $0 < \alpha < 1$.

This is the MDP itself: no state variable is dropped. The only error is the fit of $Q$, through the actions it ranks. The policy is scored out of sample by `simulate()`, which gives a lower bound on the optimum. The in-sample $V_0$ (the realised target of the greedy action at $t = 0$) is reported next to it. `solve` also returns `timings` (forward simulation, rollouts, fits).

**Known-answer check** (`tests/test_baseline_dp.py`). With $\sigma \to 0$ and the monthly normals set to 0, the rate path is deterministic and the same on every path; only exits differ. Each premium's choice is then separate: the reward rises with $R_T$, and both assets keep accruing after exit. So the optimal $\alpha_t$ puts the year-$t$ premium where it is worth more at $T$. That is Branch 21 at $\tfrac12 r_t + \sum_{s>t} r_s$ in logs, with $r_s = \max(\text{tariff}(\mu_{\text{Jul},t}), \mu_s - m_{\text{ps}})$, against the fund's log monthly returns after its July return. The DP must match that on at least 95% of in-force evaluation states. The test sets `TARIFF_MARGIN` = 0 for itself only: with the default margin, Branch 21 never wins on the deterministic path, and an answer that never switches would not test the DP.

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
- **2026-11 — Stage 0 of the bond-fund model** (tag `pre-bondfund` holds the horizontal-WAP baseline above).
  - `rates/calibration.py`: the `lru_cache` on `calibration()` is restored. Commit ff86184 ("check calibration") had commented it out without a stated reason. Every caller only reads the returned dict and its arrays. Without the cache, each `step()` reread the CSV (about 10 ms per call), which dominated the DP's run time.
  - `baseline_dp.solve` reports `timings` (forward simulation, rollouts, fits).
- **2026-11 — Stage 1: horizontal Branch 21 with profit-sharing.**
  - (5) is replaced by (5a), the tariff of a July premium on the 16-level grid $\mathcal I$, and (5b), the bucket rate $\max(i_k, \mu_t - m_{\text{ps}})$.
  - (6c): Branch 21 is held in 16 tariff buckets, in force and after exit.
  - `I_TARIFF` and `M` are removed; `TARIFF_STEP`, `TARIFF_MARGIN` (PLACEHOLDER) and `PS_MARGIN` (PLACEHOLDER) are added.
  - The observation has 32 values: $N$ sits where $R^{23}$ will go in Stage 2.
  - January proxy `new_premium_rate` for the threshold rule, the diagnostics and the DP features.
  - DP: per-bucket paid-up growth and projection.
  - The ladder known-answer test is removed: its premise, a locked Branch 21 under `BOOK_WEIGHT` = 0, no longer holds.
- **2026-11 — Stage 2: Branch 23 bond fund** (the zero-coupon ladder is removed; git history keeps it at tag `pre-bondfund`).
  - (7) is now the price of maturity $\tau$, $P(\tau; r)$; (7a) adds `theta_q`.
  - (8): a constant-duration fund rolls a zero-coupon of maturity $D$ monthly, with the July premium added after the July return. It keeps earning after exit.
  - The state holds the fund value $R^{23}$ in place of the units $N$.
  - Removed: `bond_price`, `locked_yield`, `buy_units`, `all_ladder`, `threshold_rule`, and the $y^{\text{long}}$ diagnostic.
  - Benchmarks: `all_branch21`, `all_fund`, `fifty_fifty`, `market_practice_rule`.
  - New parameter `FUND_DURATION` (PLACEHOLDER); `FUND_FEE` is now charged monthly (PLACEHOLDER).
  - DP: `paidup_fund_growth`. The two ladder spreads leave the features (both used $y^{\text{long}}$), and the January proxy $i_t$ is added; Stage 3 finalises the basis.
- **2026-11 — Stage 3: the DP.**
  - Features: the rate state plus the summaries of Sec. 7. That adds the Branch 21 floor at $T$, the projection at the current crediting and the reserve-weighted tariff. The January proxy $i_t$ is dropped, and no derived column is affine in $y$ at fixed $t$.
  - Clipping guard on every fitted $Q_t(\cdot,\alpha_k)$, with `clip_stats`.
  - DP tests run at $n = 2000$. They add the clipping-guard test and the $\sigma \to 0$ known-answer test (per-premium comparison).
- **2026-11 — book-yield tariff** (tag `pre-booktariff` holds the 10Y-based tariff).
  - (5a): the tariff of a July premium is floored from the July book yield $\mu_{\text{Jul}}$, no longer from the July 10Y, with `TARIFF_MARGIN` = 0.010 (PLACEHOLDER). Reason: observed group tariffs lag the market (AG Insurance, Sec. 3).
  - The market-practice rule and the January proxy `new_premium_rate` use $\text{tariff}(\mu_{\text{Jan}})$.
  - (11): the reported premium cost accrues from July, $A(t,T)\,e^{-r_t/2}$ (an approximation; Sec. 4).
  - Tests:
    - New: the tariff follows $\mu_{\text{Jul}}$, the printed historical comparison, and the July premium accrual in closed form.
    - The tariff-grid and vintage tests now take the tariff from $\mu_{\text{Jul}}$.
    - The known-answer test uses `TARIFF_MARGIN` = 0 (see Sec. 7).
    - `test_policy_beats_every_constant_policy` now fails at $n = 2000$ (DP −0.30740 vs best constant −0.30718) and is left failing on purpose, pending a decision.

## 9. Verify by hand (one path)

These are `EVAL_SEED` path 4 under $\alpha = \tfrac12$ every year, at year $t = 10$ (January 2037, in force), with default parameters. Each line can be recomputed on paper from the numbers given, and matches the code to all printed digits.

- **State in January:** $y_t = 0.0260657571$, $G_t = 2.50\%$, $\mu_t = 0.0265879593$. Premium $P_t = 0.05$, with $P^{21} = P^{23} = 0.025$ (units of $S_t$), and $W = 0.025$.
- **(1) Liability bucket $g = G_t = 2.50\%$** (Eq. 6b): $L_t^{(g)} = 0.049470044753$, so
  $L_{t+1}^{(g)} = (0.049470044753\cdot e^{0.025} + 0.05\cdot e^{0.0125})/1.025 = (0.049470044753\cdot 1.025315120524 + 0.05\cdot 1.012578451541)/1.025 = 0.098879324366$.
  A bucket that gets no premium, $g = 2.00\%$: $0.049040073002\cdot e^{0.02}/1.025 = 0.048810486041$.
- **(2) Branch 21 bucket with profit-sharing** (Eqs. 4, 5a–6c):
  - The book yield is filtered from February to July ($\mu \leftarrow \mu + w(y-\mu)$, $w = 1/48$), giving $\mu_{\text{Jul}} = 0.0266423859$. The July 10Y is $0.0270406985$ and is not used.
  - The tariff is $\mathrm{floor}_{25\text{bp}}(0.0266423859 - 0.010) = 1.50\%$.
  - The bucket $i_k = 1.50\%$ earns $r_k = \max(0.015,\ 0.0265879593 - 0.005) = 0.0215879593$, of which profit-sharing is $0.0065879593$.
  - $R^{21,(k)}_{t+1} = (0.097964703662\cdot e^{r_k} + 0.025\cdot e^{r_k/2})/1.025 = (0.097964703662\cdot 1.021822665219 + 0.025\cdot 1.010852444830)/1.025 = 0.122315966550$.
- **(3) One month of the fund** (Eqs. 7–8), April:
  - The 10Y goes from $0.0287225959$ to $0.0272843256$. With $s_{\text{term}} = 0.0128385047$, that gives $r_{\text{now}} = 0.0158840912$ and $r_{\text{next}} = 0.0144458209$.
  - Inputs: $\kappa = 0.1104185098$, $\sigma = 0.0060601606$, $\theta_Q = 0.0427692751$, $D = 7$.
  - $B(7) = (1-e^{-7\kappa})/\kappa = 4.8754631475$, so $\ln P(7; r_{\text{now}}) = (\theta_Q - \tfrac{\sigma^2}{2\kappa^2})(B - 7) - \tfrac{\sigma^2 B^2}{4\kappa} - B\,r_{\text{now}} = -0.1670839385$.
  - $B(7 - \tfrac1{12}) = 4.8368140525$, so $\ln P(7-\tfrac1{12}; r_{\text{next}}) = -0.1576383607$.
  - Gross return: $e^{-0.1576383607 + 0.1670839385} = 1.009490327998$ (no fee).
- **(4) Calibrated $\lambda_r$** (Eq. 7a):
  - Inputs: $\theta = 0.0223352927$, $s_{\text{term}} = 0.0128385047$, so $\theta_r = 0.0094967880$; and $B(10) = 6.0544115711$.
  - $\theta_Q = \tfrac{\sigma^2}{2\kappa^2} + \dfrac{B\theta_r + \sigma^2B^2/(4\kappa) - 10\,\theta}{B - 10} = 0.0427692751$.
  - $\lambda_r = (\theta_r - \theta_Q)\,\kappa/\sigma = -0.6062378058$.
  - Check: the model 10Y at $r = \theta_r$ is $-\ln P(10;\theta_r)/10 = 2.2335\% = \theta$.

