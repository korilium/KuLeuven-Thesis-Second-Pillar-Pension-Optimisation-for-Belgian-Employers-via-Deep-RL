# Debugging guide

The tables between `gen` markers are generated from the code (`python experiments/walkthrough.py --write debugging`). That includes every line number. If a marked code line moves or disappears, `--check debugging` fails.

## 1. Which entry point for which question

| question | entry point | time |
|---|---|---|
| Is anything broken? | `pytest -m "not slow"` (then `pytest` for the slow checks) | ~15 s / ~25 s |
| Do the old (discounted) numbers still come out? | `pytest tests/test_regression.py` | ~5 s |
| Are the RL environment and the DP still on one scale? | `pytest tests/test_env.py tests/test_numeraire.py::test_env_contract_both_numeraires` | ~6 s |
| Does a known-answer property hold, and by how much? | the measuring function in `pension/checks.py`, called from a REPL (`docs/REFERENCE.md`, table "Checks") | seconds to a minute |
| What does one scenario / one career look like, number by number? | `python experiments/walkthrough.py walkthrough` (prints Part A and Part B) | ~10 s |
| Do the documents still match the code? | `python experiments/walkthrough.py --check` | ~80 s (the `review` producer re-runs the evidence) |
| A full figure suite | `python experiments/dynpro/<suite>.py <section>` | minutes; rate-mode headline sections wait for `LONG_RATE_P` (REVIEW M7) |
| The rate engine on its own | `python experiments/report_numeraire.py stage1b` | ~10 s |

The VS Code launch configurations (`.vscode/launch.json`) start the report, a suite section, the current file, or pytest.

## 2. Small drivers to step through

Run these under the debugger (VS Code: "Python: current file") and set the breakpoints from the tables below.

**(a) One career, one year, constant rates**

```python
import numpy as np, pension.dp as dp
from pension.params import DEFAULT as p
Fg, rg, ag = dp.grids(nF=31, nR=25, na=7)
pol = dp.const_policy(1.0, p.T, len(Fg), len(rg))           # a = 1 every year
r = dp.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=20.0, n_paths=1, seed=7, p=p)
```

**(b) The same under vasicek** (horizontal ledger, r_t, A(t,T), the 6th observation)

```python
import numpy as np, pension.dp as dp
from pension.params import DEFAULT
from pension.envs.pension_env import PensionEnv
p = DEFAULT.replace(RATE_MODEL="vasicek")
Fg, rg, ag = dp.grids(nF=31, nR=25, na=7)
pol = dp.const_policy(1.0, p.T, len(Fg), len(rg))
r = dp.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=20.0, n_paths=1, seed=7, p=p)
env = PensionEnv(p=p, rate_pool=50); obs, _ = env.reset(seed=0); env.step(np.array([1.0]))
```

**(c) One scenario from the NBB CSV to the joint value**

```python
import numpy as np, pension.dp as dp
from pension.params import DEFAULT
p = DEFAULT.replace(RATE_MODEL="vasicek")
Fg, rg, ag = dp.grids(nF=31, nR=25, na=7)
pol = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=3, p=p)["policy"]    # CE solve: draws 5000 scenarios
R0, L0, S0 = dp.new_plan_init(200, np.random.default_rng(7))
r = dp.simulate(pol, Fg, rg, R0=R0, L0=L0, S0=S0, n_paths=200, seed=7, p=p)
```

`economy.rate_calibration()` runs only once per process. To step through the calibration (stops 1–4 of map c), set `pension.economy._CALIBRATION = None` first, or put the breakpoints before the first call.

## 3. Breakpoint maps (in execution order)

### (a) One career, one year, constant rates

<!-- gen:bp_constant -->
| # | where | inspect | expect |
|---|---|---|---|
| 1 | `pension/dp.py:507` | `exo.zR[0]`, `exo.zL[0]`, `exo.u[0]`, `exo.r[0]` | `exo.r` = `SHORT_RATE` everywhere; `exo.G is None` (vertical ledger) |
| 2 | `pension/dp.py:517` | `t`, `F`, `rho`, `a` | `F = R/L`, `rho = S/L`; a in the band |
| 3 | `pension/dp.py:522` | `numeraire.premium_factor(t, r_t, p)`, `obj.contribution(a, t, p)` | factor e^{(r+s)(T-t)} (t = 0: table below); reward of a = 1 at t = 0: table below |
| 4 | `pension/dynamics.py:172` | `c`, `state.S` | c = a Γ S; 0 for paths that have left |
| 5 | `pension/dynamics.py:182` | `state.R` before/after, `mu`, `sig`, `drift` | in force: (R + c) e^{μ - σ_R²/2 + σ_R z_R} (mean e^μ); paid-up: R e^{μ} |
| 6 | `pension/dynamics.py:102` | `self.L`, `g` | (L + c) e^{G - σ_L²/2 + σ_L z_L} (mean e^G); frozen when absent |
| 7 | `pension/dynamics.py:185` | `exo.u[t]`, `hazard(t)`, `lv` | leaves iff u < h(t); `leave_t` becomes t + 1 |
| 8 | `pension/dp.py:524` | `state.F`, `state.rho`, `end` | after one step: F' = dp.F_next(F, aΓρ, zR, zL, p), ρ' = dp.rho_next(...) (identity, table below); paid-up roll-forward F e^{μm}, ρ(1+W)^m |
<!-- /gen -->

### (b) One career, one year, vasicek

<!-- gen:bp_vasicek -->
| # | where | inspect | expect |
|---|---|---|---|
| 1 | `pension/economy.py:265` | `vp`, `y10_m[0]`, `r_m[0]` | `y10_m[0]` = last observed 10Y; `r_m = y10_m - spread`; `pre` = months to the first 1 January |
| 2 | `pension/economy.py:281` | `G_[0]`, `G_[1]` | G_0 = 2.75% (1 January 2027, observed window Jun 2024 - May 2026); later years on the 25 bp grid in [1.75%, 3.75%] |
| 3 | `pension/economy.py:285` | `acc_[0]` | exp(Σ of the 12 monthly r·dt) |
| 4 | `pension/dynamics.py:86` | `r[0]`, `acc[0]`, `G[0]`, `mu[0]` | the scenario's annual r, acc, G, μ for these paths |
| 5 | `pension/dynamics.py:126` | `self.Lv[:t+2]`, `self.lock[:t+2]` | a new vintage locked at G_t; every in-force vintage grows at its own lock |
| 6 | `pension/rates/accrual.py:67` | `r_t`, `y_t`, `mean`, `tau` | A(0,T; r_0) (table below); = numeraire.premium_factor(0, r_0, p) |
| 7 | `pension/envs/pension_env.py:106` | the 6 components | r_t = y_t - spread; at reset: obs[5] = r_0 (table below) |
| 8 | `pension/envs/pension_env.py:122` | `reward`, `r_t` | -(1-λ) a Γ (1+W)^{-(T-t)} A(t,T; r_t) |
<!-- /gen -->

### (c) One scenario from the NBB CSV to `simulate()["joint"]`

<!-- gen:bp_flow -->
| # | where | inspect | expect |
|---|---|---|---|
| 1 | `pension/rates/data.py:77` | `dfYield`, `cache` | the cached NBB vintage (block `calibration` of ARCHITECTURE.md) |
| 2 | `pension/rates/calibration.py:42` | `a`, `b`, `dt` | kappa = -b/dt, theta = -a/b |
| 3 | `pension/economy.py:73` | `df10Y`, `short_label` | computed once per process |
| 4 | `pension/economy.py:172` | `spread`, `p.LONG_RATE_P` | theta = LONG_RATE_P + spread if set; spread = historical mean (10Y - 1Y) if SPREAD_10Y_SHORT is None |
| 5 | `pension/rates/simulation.py:58` | `paths[t]`, `eps[t]` | exact OU step |
| 6 | `pension/rates/wap.py:64` | `k`, `lo`, `end` | window ends WAP_LAG = 8 months before the year's 1 January: May of the year before |
| 7 | `pension/dp.py:335` | `ce`, `prem[:3]` | CE moments and the scenario-mean premium factor (walkthrough block `ce`) |
| 8 | `pension/dp.py:534` | `benefit_paths`, `cost` | joint = mean of the per-path env returns (contract C1) |
<!-- /gen -->

## 4. Values and identities to expect

Computed at the defaults. Under vasicek, r_0 = the last observed 10Y minus the spread.

<!-- gen:expect -->
| quantity | expression | value |
|---|---|---|
| year-0 premium factor, constant | $A(0,T)=e^{(r+s)T}$ | 3.857426 |
| year-0 premium reward at a = 1, constant | $-(1-\lambda)\,\Gamma(1+W)^{-T}A(0,T)$ | -0.095232 |
| short rate at t0, vasicek | $y_0-$spread | 2.8361% |
| year-0 premium factor, vasicek | closed form at $r_0$ | 1.905071 |
| year-0 premium reward at a = 1, vasicek | same with $A(0,T;r_0)$ | -0.047033 |
| $l$ at F=1, ρ=20, a=0.5 | $a\Gamma\rho$ | 1.5000 |
| $F'$ at zR=0.3, zL=−0.2 (constant) | $\frac{F+l}{1+l}e^{(\mu-\sigma_R^2/2)-(G-\sigma_L^2/2)+\sigma_R z_R-\sigma_L z_L}$ | 1.018112 |
| $\rho'$ (same) | $\frac{(1+W)\rho}{(1+l)e^{G-\sigma_L^2/2+\sigma_L z_L}}$ | 7.991146 |
| paid-up roll-forward of F over 40 years | $F\,e^{\mu m}$ | 3.320117 |
| paid-up roll-forward of ρ over 40 years | $\rho\,(1+W)^m$ | 2.685064 |
<!-- /gen -->

Identities that must hold at any state (exact up to floating point):

- **one year, constant rates:** $F' = \frac{F+l}{1+l}e^{(\mu-\sigma_R^2/2)-(G-\sigma_L^2/2)+\sigma_R z_R - \sigma_L z_L}$ and $\rho' = \frac{(1+W)\rho}{(1+l)e^{G-\sigma_L^2/2+\sigma_L z_L}}$ (`DRIFT_CORRECTION`; drop the $\sigma^2/2$ terms for the legacy convention) with $l = a\Gamma\rho$ (`dp.F_next`, `dp.rho_next`);
- **horizontal ledger:** $L = \sum_v L^{(v)}$ and the 9-bucket sum (`tests/test_dynamics.py`);
- **valuation date:** `numeraire.terminal_factors(p) == (1.0, 1.0)` under "retirement";
- **accrual:** `numeraire.premium_factor(t, r_t, p) == rates.accrual.closed_form_accrual(t, r_t, p, s=p.FINANCING_SPREAD)` (vasicek), and `== exp((SHORT_RATE + s)(T - t))` (constant);
- **career returns:** the sum of a career's rewards in `PensionEnv` equals its contribution to `simulate()["joint"]`;
- **leavers:** a leaver at year τ has $F_T = F_\tau e^{\mu(T-\tau)}$ at constant rates, and `dp.paidup_service(...)[T] == dp.terminal(...)`.

A worked example with every intermediate number: `paper/explanations/model-walkthrough.md` (Part A, block `step`; Part B, the career tables).

## 5. What to step over (slow)

| call | why it is slow | typical time |
|---|---|---|
| `rates.data.load_olo()` with no cache | network fetch from the NBB | seconds |
| `economy.rate_calibration()` (first call) | `bootstrapForwardCurve` → `_fit_nss`: differential evolution + Nelder-Mead | ~2 s |
| `economy.draw_rate_scenarios(n)` for `hull_white`, `hull_white_p` | `reconstructFutureYield` prices a 10Y bond at every month and path | seconds per 10,000 paths |
| `dp.solve` (any mode) | T × |actions| Bellman evaluations over the grid with quadrature | 1–6 s at the protocol grid |
| `dp.solve` under a rate model | additionally draws `RATE_CE_PATHS` (5000) scenarios | + ~1 s (memoised afterwards) |
| `dp.simulate` with 15,000+ paths | T steps over all paths | ~1 s |
| `checks.*` (`lambda_monotonicity`, `cross_scores`, `numeraire_equivalence`, ...) | several solves each | 10 s – 1 min |
| `PensionEnv` loops over many episodes | one Python step per year per episode | ~0.5 s per 100 episodes |

Under the debugger, step **over** these calls and put breakpoints **inside** the loop of interest instead (e.g. `dynamics.step`, with a condition such as `t == 0`).
