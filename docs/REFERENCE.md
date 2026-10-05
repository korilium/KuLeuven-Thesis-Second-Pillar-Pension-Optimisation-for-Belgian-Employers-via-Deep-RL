# Reference

Every table between `gen` markers is generated from the code by `python experiments/walkthrough.py --write reference`, and `--check reference` fails if any of them is out of date. "Used in" lists the files that call or import a name. It is found by text search, so it is indicative, not a call graph. For the model behind these functions see `docs/MODEL.md`; for known problems see `docs/REVIEW.md`.

## Side effects and state, by module

| module | side effects / state |
|---|---|
| `pension/params.py` | none (immutable `Params`) |
| `pension/objective.py` | none; `OBJECTIVES` is a module-level registry (a dict, mutable by design: add entries to register an objective) |
| `pension/numeraire.py` | none; lazily imports `rates.accrual` |
| `pension/dynamics.py` | `step` mutates and returns the `State` it is given (arrays are rebound; `HorizontalLedger` vintages are written in place) |
| `pension/economy.py` | **process-level caches**: `_CALIBRATION` (keyed on the data vintage) and `_SCENARIOS` (keyed by `Params` inputs and the vintage, at most `_SCENARIO_CACHE_SIZE` entries, arrays read-only). Module constants `T, G, …, N_EVAL` copied from `DEFAULT` for the tabular rung |
| `pension/dp.py` | none (pure functions of their arguments, apart from reading the economy caches) |
| `pension/checks.py` | none (measures and returns) |
| `pension/envs/pension_env.py` | an env instance holds its state, its RNG and a lazily drawn scenario pool |
| `pension/envs/tabular.py` | module globals rebound by `experiments/tabular/run_tabular.apply_config`; writes figures to `results/tabular/` |
| `pension/rates/data.py` | `load_olo` writes `pension/rates/data/olo_yields.csv` on the first call (network fetch from the NBB) or on `refresh=True` |
| `pension/rates/calibration.py` | prints the fit when `verbose=True` (the economy calls it silently) |
| `pension/rates/simulation.py` | without `rng`: calls `np.random.seed(seed)` (global state; legacy, plots only) |
| `pension/rates/pricing.py`, `wap.py`, `accrual.py` | none |

## Modules and their public functions

<!-- gen:api -->
### `pension.params` (pension/params.py)

every parameter of the model, as ONE immutable object.

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `Params(T: int = 45, G: float = 0.03, MU: float = 0.03, W: float = 0.025, S0: float = 1.0, DIS...)` | dataclass (fields: see the signature and the tables in this document) | dp.py, exp/walkthrough.py, objective.py |

### `pension.objective` (pension/objective.py)

the value function of the funding problem, as swappable parts.

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `Objective(name: str, employee: Callable, shortfall: Callable, contribution: Callable, weights: C...)` | dataclass (fields: see the signature and the tables in this document) | dp.py, envs/pension_env.py, numeraire.py |
| `convex_contribution(k=2.0)` | Cash-flow strain: each euro costs (1 + k * aGAMMA), where aGAMMA is the contribution as a share of the CURRENT payroll. k=2: paying 15% of salary costs 30% extra per euro; paying 2% costs 4% extra. |  |
| `convex_shortfall(k=1.0)` | short + k*short^2: large deficits hurt more than proportionally (balance- sheet stress, a lump-sum settlement at T). k=1: a shortfall of half a final salary costs 1.5x its face value. |  |
| `crra(eta=None)` | target * ANNUITY * u(RR/target): the committed leg. u is in RELATIVE shortfall units; times the target it is in annual-RR units, times ANNUITY in capital (final-salary-years), the employer's numeraire. eta=None reads ... | exp/walkthrough.py |
| `employer_only()` | Both legs at weight 1: meant for an employee leg that is ONLY a penalty (floor_penalty with base=None), i.e. minimise cost subject to a floor. |  |
| `floor_penalty(k=1.0, rr_min=None, base=None)` | base minus k * ANNUITY * (rr_min - RR)+ : a replacement-rate floor as a Lagrangian penalty, in capital units. k=1 prices a missing point of RR at its actuarial cost. rr_min=None uses the target being judged against (p... |  |
| `lambda_weights()` | The committed negotiation weight: lambda on the employee, 1-lambda on the employer. |  |
| `linear_contribution()` | The committed leg: a contribution a*GAMMA*S_t, expressed per unit of final salary, i.e. a*GAMMA*(1+W)^-(T-t). |  |
| `linear_shortfall()` | The committed leg: the employer pays the WAP shortfall, euro for euro. |  |
| `loss_averse(k=2.5, base=None)` | Shortfalls below the target weigh k times more than the base curve says (a kink at the target, Kahneman-Tversky style); gains are unchanged. |  |
| `resolve(obj, default)` | None -> OBJECTIVES[default]; a name -> its registry entry; an Objective as is. | dp.py |
| `satiated(base=None)` | No value above the target: the employee is indifferent to overshoot. |  |
| `threshold_shortfall(s0=0.1)` | Only the part of the shortfall above s0 final salaries costs anything (a deficit buffer the employer can absorb). |  |
| `u(x, eta)` | Normalized (Box-Cox) CRRA utility, eta != 1:     u(x) = (x^(1-eta) - 1) / (1 - eta), which satisfies u(1) = 0 and u'(1) = 1 for every eta. The zero at the target makes "on target" the natural origin; the unit slope at... | checks.py, dp.py, exp/dynpro/benchmark_suite.py, exp/dynpro/lambda_dial_suite.py, exp/dynpro/objective_suite.py, exp/dynpro/scenario_suite.py ... |

### `pension.numeraire` (pension/numeraire.py)

when the employer's money is valued: the timing around the objective.

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `premium_factor(t, r_t, p)` | The factor on a premium paid in year t: A(t, T) given the short rate r_t (scalar or one per path) under "retirement", e^{-DISC_ER t} under "discounted". | dp.py, envs/pension_env.py, exp/walkthrough.py, tests/test_mechanisms.py |
| `premium_schedule(p, rates=None)` | One deterministic factor per year t = 0..T-1, for the DP's Bellman flow. Without a scenario: the constant-rate (or discounted) factor. With a scenario (the certainty-equivalent solve): the scenario MEAN of A(t, T; r_t... | checks.py, dp.py, exp/walkthrough.py, tests/test_mechanisms.py |
| `retirement(p)` |  | dynamics.py |
| `terminal_factors(p)` | (employee, employer) factors on the terminal legs. | checks.py, dp.py, envs/pension_env.py |

### `pension.dynamics` (pension/dynamics.py)

THE career dynamics of a second-pillar plan, for n paths at once.

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `Exogenous(zR: numpy.ndarray, zL: numpy.ndarray, u: numpy.ndarray, G: Optional[numpy.ndarray] = N...)` | Everything random about n careers over T years, drawn once. | checks.py, dp.py, envs/pension_env.py, exp/walkthrough.py, tests/test_dynamics.py, tests/test_env.py |
| `HorizontalLedger(Lv: numpy.ndarray, lock: numpy.ndarray, _total: numpy.ndarray = None) -> None` | Rate model, Branch 21 horizontal method: slot 0 holds the opening liability (locked at G_0), slot t+1 the year-t contribution locked at G_t. Each slot keeps its own rate until retirement; L is their sum. | exp/walkthrough.py |
| `State(t: int, R: numpy.ndarray, S: numpy.ndarray, ledger: object, present: numpy.ndarray, le...)` | n careers at the start of year t. | checks.py, dp.py, envs/pension_env.py, exp/walkthrough.py, tests/test_dynamics.py |
| `VerticalLedger(L: numpy.ndarray) -> None` | Constant-rate model: the whole guaranteed reserve grows at G (+ SIGMA_L shock). |  |
| `settle(state, p)` | Retirement: payout max(R, L), employer shortfall (L - R)+, and the service fraction tau/T that sets the leaver's pro-rated target. | dp.py, envs/pension_env.py, exp/walkthrough.py |
| `step(state, a, exo, p, hazard)` | One year: contribute a (capacity fraction, already 0 for absent paths), credit, accrue the guarantee, grow salary, then draw churn. Mutates and returns `state` (arrays are replaced, never written in place, except the ... | checks.py, dp.py, envs/pension_env.py, envs/tabular.py, exp/dynpro/benchmark_suite.py, exp/walkthrough.py ... |

### `pension.economy` (pension/economy.py)

the economic scenario (exogenous world).

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `check_scenario(rates, p)` | A scenario dict must come from p.RATE_MODEL (rates["model"]); otherwise the dynamics would follow one model's G_t, mu_t while the premiums accrue under another's short rate. | dp.py |
| `data_vintage()` | Identity of the OLO data the calibration was built from: the cache file's path, size and modification time. Part of every scenario key, so a refreshed CSV can never be served stale scenarios (and it invalidates the ca... |  |
| `draw_rate_scenarios(n_paths=2000, seed=None, model=None, horizon=None, p=None)` | Annual rate scenarios for the reserve/liability simulation. | checks.py, dp.py, envs/pension_env.py, exp/dynpro/rates_suite.py, exp/report_numeraire.py, exp/walkthrough.py ... |
| `hull_white_p_params(p=None, legacy_sigma=False, phi=None)` | The parameters hull_white_p simulates with. | checks.py, exp/report_numeraire.py, rates/accrual.py |
| `hull_white_paths(hp, H, n, dt, rng)` | Monthly short-rate paths and the repriced 10Y for Hull-White parameters hp (shared by hull_white and hull_white_p: only kappa, sigma and phi differ). | checks.py |
| `plan_age(rate0=0.03, step=0.01, band=10)` |  | envs/tabular.py, exp/tabular/run_tabular.py |
| `plan_fixed(rate=0.05)` |  | envs/tabular.py, exp/tabular/run_tabular.py |
| `plan_step(rate_low=0.04, rate_high=0.1, ceiling=1.5)` |  | envs/tabular.py, exp/tabular/run_tabular.py |
| `premonths(p=None)` | Months between the last OLO observation and the start of model year 0. YEAR_START = "january" (default): model years are calendar years, year 0 starts on the first 1 January after the last observation (0 if that obser... | checks.py, exp/walkthrough.py, tests/test_rates.py |
| `rate_calibration()` | Calibrate once per process, from the cached NBB OLO data: Vasicek (kappa, sigma, theta, r0) on the monthly 10Y history and the NSS curve on the latest cross-section. The rate engine (pension.rates) is imported HERE, n... | checks.py, exp/walkthrough.py, tests/test_rates.py |
| `vasicek_params(p=None)` | The parameters of the canonical "vasicek" model: kappa, sigma of the 10Y (OLS), y0 = the last observed 10Y, the spread to the short rate, r = y10 - spread (p.SPREAD_10Y_SHORT, or the historical mean 10Y - 1Y if None),... | exp/report_numeraire.py, exp/walkthrough.py, rates/accrual.py, tests/test_mechanisms.py, tests/test_rates.py |
| `vasicek_short_params(p=None)` | The parameters vasicek_short simulates with: kappa, sigma, theta_Q and the t0 offset as calibrated; theta_P = p.LONG_RATE_P if set (else the OLS estimate), with phi = (theta_P - theta_Q) kappa / sigma recomputed from ... | checks.py, exp/report_numeraire.py, rates/accrual.py, tests/test_rates.py |
| `year_offset(p=None)` | Calendar time (years) from the curve date (the last observation) to the start of model year 0 -- the shift between model year t and the time axis of a curve-fitted model (hull_white_p's alpha(t)). | checks.py, rates/accrual.py |

### `pension.dp` (pension/dp.py)

Rung-2 benchmark: DP oracle over (t, F, rho).

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `F_next(F, l, zR, zL, p=None)` |  | checks.py, exp/walkthrough.py, tests/test_dynamics.py |
| `bilinear(Fg, lrg, V, Fq, lrq)` |  | envs/pension_env.py, exp/walkthrough.py |
| `certainty_equivalent(rates, p=None)` | Constant-rate parameters that summarise a rate scenario for the DP. |  |
| `const_policy(a, n_years, nF, nR)` |  | checks.py, exp/dynpro/contribution_schedule_suite.py, exp/dynpro/lambda_dial_suite.py, exp/dynpro/objective_suite.py, exp/dynpro/rates_suite.py, exp/dynpro/report.py ... |
| `gauss_hermite_2d(n=15)` |  | exp/dynpro/report.py |
| `grids(nF=145, nR=91, na=31)` |  | exp/dynpro/benchmark_suite.py, exp/dynpro/common.py, exp/dynpro/contribution_schedule_suite.py, exp/dynpro/lambda_dial_suite.py, exp/dynpro/objective_suite.py, exp/dynpro/rates_suite.py ... |
| `make_F_grid(F_max=3.0, n=241)` |  | exp/dynpro/report.py, exp/walkthrough.py, tests/conftest.py, tests/regression/make_golden.py, tests/test_dynamics.py, tests/test_env.py ... |
| `make_a_grid(n=41)` |  | exp/dynpro/benchmark_suite.py, exp/dynpro/contribution_schedule_suite.py, exp/dynpro/lambda_dial_suite.py, exp/dynpro/report.py, tests/conftest.py, tests/regression/make_golden.py ... |
| `make_rho_grid(lo=0.01, hi=35.0, n=91)` | Log-spaced grid for rho = S/L. | exp/dynpro/report.py, exp/walkthrough.py, tests/conftest.py, tests/regression/make_golden.py, tests/test_dynamics.py, tests/test_env.py ... |
| `new_plan_init(n, rng, rho_lo=15.0, rho_hi=34.0, F_sd=0.08)` | Fresh plans: F0 ~ 1, high rho (liability small relative to salary). | envs/pension_env.py, exp/dynpro/common.py, exp/dynpro/contribution_schedule_suite.py, exp/dynpro/lambda_dial_suite.py, exp/dynpro/report.py, exp/dynpro/sensitivity_suite.py ... |
| `paidup_service(Fg, rg, obj=None, p=None)` | Per-leave-cohort paid-up value Phi[tau](F,rho), in closed form. | checks.py, exp/dynpro/report.py, objective.py, tests/regression/make_golden.py |
| `rho_next(rho, l, zL, p=None)` |  | checks.py, exp/walkthrough.py, tests/test_dynamics.py |
| `sample_entry(rng, n, F_mu=1.0, F_sd=0.15, F_clip=(0.4, 2.5), rho_lo=3.0, rho_hi=30.0)` | Placeholder plan-entry distribution (NOT calibrated -- swap for DB2P aggregate stats when available). L0=1; the model is scale-free. | exp/dynpro/lambda_dial_suite.py, exp/dynpro/sensitivity_suite.py |
| `schedule_policy(a_of_t, nF, nR, n_years=None, p=None)` | Lift a STATE-INDEPENDENT schedule a(t) to a (T, nF, nR) policy array. | exp/dynpro/benchmark_suite.py, exp/dynpro/common.py, exp/dynpro/rates_suite.py |
| `simulate(policy, Fg, rg, R0=1.0, L0=1.0, S0=None, band=None, n_paths=30000, seed=7, hazard=<fun...)` | Canonical forward Monte-Carlo of a reduced policy a*(t,F,rho) under the committed model: Belgian churn (immediate-vesting, paid-up leavers), the employer's numeraire (pension/numeraire.py), and the service-pro-rated a... | checks.py, dynamics.py, economy.py, envs/pension_env.py, exp/dynpro/benchmark_suite.py, exp/dynpro/common.py ... |
| `solve(mode='optimize', plan_rule=None, Fg=None, rg=None, ag=None, n_quad=5, hazard=<function...)` | Backward induction for the committed (churn-aware) objective. | checks.py, exp/dynpro/benchmark_suite.py, exp/dynpro/common.py, exp/dynpro/contribution_schedule_suite.py, exp/dynpro/lambda_dial_suite.py, exp/dynpro/objective_suite.py ... |
| `survival(hazard, p=None)` |  | exp/dynpro/benchmark_suite.py, exp/dynpro/contribution_schedule_suite.py, exp/dynpro/report.py |
| `tenure_hazard(t, h0=0.12, hinf=0.025, tau=7.0)` | Belgian tenure hazard: ~12%/yr early -> ~2.5%/yr long-tenure floor, giving ~45% staying 10+ years and ~15% a full career (avg tenure ~11y, ~50% at 10+y; Goulart & Oesch 2024, OECD/Eurostat). | envs/pension_env.py, exp/walkthrough.py |
| `terminal(Fg, rg, obj=None, p=None)` | Full-career stayer at T: the objective's employee leg at RR_TARGET minus its shortfall cost, combined by its weights (see objective.py). | checks.py, exp/dynpro/objective_suite.py, exp/dynpro/report.py, exp/dynpro/scenario_suite.py, objective.py, tests/regression/make_golden.py ... |
| `u(x, p=None)` | The normalized CRRA utility at p.ETA (see objective.u). Kept for callers that read dp.u directly; the objective itself lives in objective.py. | checks.py, exp/dynpro/benchmark_suite.py, exp/dynpro/lambda_dial_suite.py, exp/dynpro/objective_suite.py, exp/dynpro/scenario_suite.py, exp/dynpro/sensitivity_suite.py ... |

### `pension.checks` (pension/checks.py)

the known-answer checks of the model, as functions that MEASURE.

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `accrual_accuracy(p, n=4000, years=(0, 5, 10, 20, 30, 40, 44))` | (e) Per payment year t: the Monte Carlo mean of the realised accrual to T, prod_{u >= t} acc_u (monthly r * dt sums), against the mean of the closed-form conditional A(t, T; r_t) over the same paths (tower property). ... | exp/report_numeraire.py, exp/walkthrough.py, tests/test_numeraire.py, tests/test_rates.py |
| `cross_scores(p, policies, Fg, rg, entry, n_paths, seed, band, floor_a)` | J[i, j]: the policy optimised under objective i (policies: {name: policy}), scored under objective j; and floor[j], the score of the constant floor_a policy. An objective's own policy must be best in its column. | exp/dynpro/objective_suite.py, tests/test_objective.py |
| `degenerate_rates_gap(p, Fg, rg, entry, n_paths, seed, a=0.4)` | A rate scenario with G_t = G, mu_t = MU (SIGMA_L = 0) against the constant model: max gap in joint value and RR -- must be ~0. | exp/dynpro/rates_suite.py, tests/test_rates.py |
| `diagonal_margin(J, floor, relative_to='gain')` | Per column j: how far the best OTHER policy is above j's own optimum (<= 0 means the diagonal wins), as a fraction of j's gain over the floor (relative_to="gain") or of /J_jj/ ("value"). The gain normalisation is ill-... | exp/dynpro/objective_suite.py, tests/test_objective.py |
| `env_contract(p, Fg, rg, ag, nq, entry, n_paths, seed, objective=None)` | (a) /mean PensionEnv episode return - simulate()["joint"]/ on the same paths, with the solved policy (0 up to floating point). | exp/report_numeraire.py, tests/test_numeraire.py |
| `eta_log_limit(p, x=(0.4, 1.0, 2.5), eps=0.0001)` | (/u - log/ at eta=1, /u(eta=1) - u(eta=1+eps)/): u is log at eta=1 and continuous there. | exp/dynpro/scenario_suite.py, tests/test_invariants.py |
| `face_value_premiums(p, Fg, rg, entry, n_paths, seed, a=0.4)` | (c) SHORT_RATE = s = 0: every premium factor of the retirement numeraire is exactly 1, i.e. premiums count at face value (per unit of final salary). The same face value is the discounted objective at DISC_ER = DISC_EM... | exp/report_numeraire.py, tests/test_numeraire.py |
| `headline(p, Fg, rg, nq, entry, n_paths, seed, band_pct=(0.02, 0.15), na=20, score_p=None)` | (f) The banded optimum under p, scored under score_p (default p): employer cost, joint value, median stayer / leaver total RR, mean contribution (% of salary) by decade. | exp/report_numeraire.py, tests/test_numeraire.py |
| `horizontal_closed_form(p, G_before=0.03, G_after=0.0175, switch=20, a=0.4, S0=20.0)` | One career, no churn, no shocks, with G stepping G_before -> G_after at year `switch`. Returns (relative error of the horizontal ledger against L_T = L0 e^{G_0 T} + sum_s c_s e^{G_s (T-s)}, and how far the VERTICAL me... | exp/dynpro/rates_suite.py, tests/test_dynamics.py |
| `hw_p_drift(p, n=4000, years=(1, 5, 10, 20, 30, 44))` | (c) Per year: the scenario mean of r_t (with its standard error) against E^P[r_t] = alpha(t + d) + m (1 - e^{-kappa (t + d)})  (E^Q[r_t] = alpha(t + d)), d = economy.year_offset(p) the calendar time from the curve dat... | exp/report_numeraire.py, tests/test_rates.py |
| `hw_p_link(n=200)` | (a) With phi = 0 and the legacy sigma, hull_white_p's simulation IS hull_white's: max /difference/ of the monthly short rate and 10Y (must be 0), and of the annual r against the hull_white scenarios (same seed). | exp/report_numeraire.py, tests/test_rates.py |
| `hw_p_t0_fit(p)` | (b) The model 10Y at t0 against the NSS 10Y (equal: exact fit to today's curve) and against the last observed 10Y (the NSS fitting error). | exp/report_numeraire.py, tests/test_rates.py |
| `lambda_equivalent(p, de_new, lam=None, de_ref=None)` | The LAMBDA that reproduces (lam, de_new) at the reference DISC_EMP:     lambda'' = A / (A + B*exp(-de_ref*T)),  A = lam*exp(-de_new*T), B = 1-lam. delta_e only multiplies the employee leg by exp(-delta_e*T), so it is ... | exp/dynpro/common.py, exp/dynpro/contribution_schedule_suite.py |
| `lambda_monotonicity(p, Fg, rg, ag, nq, entry, n_paths, seed, lambdas=(0.2, 0.4, 0.6, 0.8))` | (employer cost, median stayer RR) at each LAMBDA; both must increase. | exp/dynpro/scenario_suite.py, tests/test_invariants.py |
| `lambda_prime(lam, r, de, T)` | The retirement-numeraire weight equivalent to the discounted objective at (lam, DISC_ER = r, DISC_EMP = de):     lambda' = lam e^{-de T} / (lam e^{-de T} + (1 - lam) e^{-r T}). | exp/walkthrough.py |
| `lambda_reparam(p, Fg, rg, ag, nq, lam=0.5, de_new=0.02, objective=None)` | Max /policy difference/ between (lam, DISC_EMP=de_new) and its LAMBDA equivalent at the reference DISC_EMP -- must be ~0. Returns (gap, lam_eq). A statement about the DISCOUNTED objective (DISC_EMP does not enter the ... | exp/dynpro/objective_suite.py, exp/dynpro/scenario_suite.py, tests/test_invariants.py |
| `leaver_terminal_gap(p, Fg, rg, objective=None)` | Max /Phi[T] - terminal/: a leaver with full service IS a stayer. Must be 0. | exp/dynpro/objective_suite.py, exp/dynpro/scenario_suite.py, tests/test_invariants.py |
| `no_capacity_gap(p, Fg, rg, nq, entry, n_paths, seed, gamma=0.001, band_pct=(0.02, 0.15), na=20)` | GAMMA -> 0: median stayer RR of the banded optimum minus that of paying nothing -- must be ~0 (contributions cannot add anything). | tests/test_invariants.py |
| `numeraire_equivalence(p, Fg, rg, ag, nq, lam, r, de)` | (b) Constant rates. The retirement objective with SHORT_RATE = r, s = 0 at lambda' and the discounted objective with DISC_ER = r, DISC_EMP = de at lambda differ by the positive factor e^{-rT} (1 - lam) / (1 - lambda')... | exp/report_numeraire.py, tests/test_numeraire.py |
| `reduced_form_gap(p, n=200, seed=0)` | (a) Max /F, rho from dynamics.step - dp.F_next / rho_next/ over one year, constant rates (0 up to floating point). | exp/report_numeraire.py, tests/test_numeraire.py |
| `scale_invariance(p, Fg, rg, ag, nq, n_paths, seed, k=5.0)` | Max relative change in RR when (R, L, S) are all scaled by k. The model is homogeneous of degree 0, so this must be ~0. | exp/dynpro/scenario_suite.py, tests/test_invariants.py |
| `timing_neutrality(p, Fg, rg, nq, entry, n_paths, seed, band_pct=(0.02, 0.15), na=20)` | (early, late) mean contribution (% of salary, years 0-9 and 35-44) when a contribution's cost and value grow alike: SHORT_RATE = MU = G under the retirement numeraire (premiums accrue at the rate R and L earn), DISC_E... | exp/dynpro/scenario_suite.py, exp/report_numeraire.py, tests/test_invariants.py, tests/test_numeraire.py |
| `vasicek_short_fit(p=None)` | Calibration and fit of the P-measure Vasicek short rate: the maturity used as short-rate proxy, kappa, theta_P (OLS, and the one simulated: p.LONG_RATE_P if set), sigma, the significance of the mean reversion, theta_Q... | exp/report_numeraire.py, tests/test_rates.py |
| `vol_target(p, n=500, years=10)` | (d) Std of the model's monthly 10Y changes (first `years` years, pooled) against the historical std of monthly 10Y changes; for hull_white_p and for hull_white (legacy sigma). Also B(10)/10, hull_white's understatemen... | exp/report_numeraire.py, tests/test_rates.py |
| `wap_scenario_stats(p, model, n=2000)` | min, max, distance from the 25 bp grid, and G_0 of the WAP rates of n scenarios of `model`; G0_observed is the statutory formula applied directly to year 0's window of the OLO history (None if that window reaches past... | exp/dynpro/rates_suite.py, exp/walkthrough.py, tests/test_rates.py |
| `wap_vs_fsma(years={2016: 1.75, 2017: 1.75, 2018: 1.75, 2019: 1.75, 2020: 1.75, 2021: 1.75, 2022: 1...)` | {year: (WAP rate from the statutory formula on the cached NBB data, rate published by the FSMA)}, in %. 2027 is left out on purpose: the formula gives 2.75% against 2.50% published (see pension/rates/wap.py). | exp/dynpro/rates_suite.py, tests/test_rates.py |

### `pension.envs.pension_env` (pension/envs/pension_env.py)

the funding problem as a gymnasium environment (the RL rung).

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `DPPolicyAgent(policy, Fg, rg, T)` | Runs a solved DP policy a*(t, F, rho) inside the environment: the oracle as an agent, for benchmarking learned policies on identical episodes. | checks.py, exp/walkthrough.py, tests/test_env.py |
| `PensionEnv(p=Params(T=45, G=0.03, MU=0.03, W=0.025, S0=1.0, DISC_EMP=0.03, DISC_ER=0.05, DISC=0.0...)` | The main Gymnasium class for implementing Reinforcement Learning Agents environments. | checks.py, exp/walkthrough.py, tests/test_env.py |
| `make_env(**kw)` | PensionEnv(**kw) -- convenience for gymnasium.vector / RL libraries. |  |

### `pension.envs.tabular` (pension/envs/tabular.py)

.

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `check_batch_consistency(plan, n_check=5, seed=99)` | run_batch must agree with run_episode path-by-path. | exp/tabular/run_tabular.py |
| `draw_shock_batch(n_paths=2000, seed=12345)` | One frozen batch of noise paths (SAA + common random numbers). None at SIGMA = 0 -> deterministic single-episode evaluation. | exp/tabular/run_tabular.py |
| `evaluate_policy(policy, plan)` | Replay a fixed policy and return unscaled economic metrics. policy: array of 0/1 actions of length T. | exp/tabular/run_tabular.py |
| `gap_benchmark(plan, batch=None)` | Certified optimal policy under linearity, with self-check. Environment-driven: every number comes from run_episode. | exp/tabular/run_tabular.py |
| `local_search_benchmark(plan, batch=None, extra_seeds=4, seed=0)` | Optimal against all 1- and 2-year deviations. Assumes nothing about linearity or monotonicity. Successor benchmark for when gap_benchmark's linearity flag goes false. | exp/tabular/run_tabular.py |
| `mc_control(plan, n_episodes=1000000, seed=0, epsStart=1.0, epsEnd=0.05, decay_frac=0.25, burn_in=...)` |  | exp/tabular/run_tabular.py |
| `numeric_gap(plan, batch=None)` | Per-year marginal value of contributing (CRN-paired at the stochastic rung). NOTE: with SIGMA > 0 the max() terminal breaks linearity, so this is a diagnostic, not a certificate — the benchmark is local search. | exp/tabular/run_tabular.py |
| `plot_results(result, path='/home/korilium/Documents/GitHub/KuLeuven-Thesis-Second-Pillar-Pension-Op...)` |  | exp/tabular/run_tabular.py |
| `policy_value(policy_fn, plan, batch)` |  |  |
| `run_batch(policy, plan, shocks)` | All paths at once for a FIXED policy (length-T 0/1 array). Same recursion as run_episode, vectorized over axis 0. shocks: (n_paths, T). Returns values: (n_paths,). | exp/tabular/run_tabular.py |
| `run_episode(choose_action, plan, shocks=None)` | Play one career. choose_action(t) -> 0 or 1. Returns (actions, reward). |  |
| `sweep_benchmark(plan, batch=None, n_grid=46)` |  |  |

### `pension.rates.data` (pension/rates/data.py)

.

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `cache_path()` | The CSV cache of the NBB OLO yields. | economy.py |
| `extractDataYieldNBB(startPeriod: str = '1993-03', endPeriod: str = '2026-10') -> pandas.core.frame.DataFrame` |  |  |
| `load_olo(startPeriod: str = '2000-01', cache: str = None, refresh: bool = False)` | OLO data the economy needs, fetched from the NBB once and then read from a CSV cache, so environments build offline and every run calibrates on the SAME data. Delete the cache (or pass refresh=True) to pull a newer vi... | checks.py, economy.py, exp/olo/plots.py, exp/olo/validation.py, exp/walkthrough.py |
| `load_olo_short(cache: str = None)` | The monthly history of the SHORTEST OLO maturity in the data -- the short-rate proxy of RATE_MODEL="vasicek_short". Returns (maturity label, DataFrame with DATE and YIELD in %), on the same dates as the 10Y series of ... | economy.py |

### `pension.rates.calibration` (pension/rates/calibration.py)

.

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `bootstrapForwardCurve(maturities: numpy.ndarray, yields: numpy.ndarray, t_grid: numpy.ndarray = None, verbos...)` | Bootstrap instantaneous forward curve f(0,T) using the Nelson-Siegel-Svensson (NSS) parametric model. | economy.py, exp/olo/plots.py, exp/olo/validation.py |
| `calibrateVasicek(df10Y: pandas.core.frame.DataFrame, verbose: bool = True)` | Calibrate Vasicek parameters from a 10Y OLO yield time series using OLS. | economy.py, exp/olo/plots.py, exp/olo/validation.py, exp/walkthrough.py |
| `calibrateVasicekShort(df_short: pandas.core.frame.DataFrame, df10Y: pandas.core.frame.DataFrame, tau: float ...)` | Real-world (P) Vasicek short rate for RATE_MODEL="vasicek_short". | economy.py |
| `computeTheta(t_grid: numpy.ndarray, f: numpy.ndarray, df_dT: numpy.ndarray, kappa: float, sigma: fl...)` | Compute Hull-White time-dependent mean reversion target θ(t). | exp/olo/validation.py |
| `nss_forward(T: numpy.ndarray, beta0, beta1, beta2, beta3, tau1, tau2) -> numpy.ndarray` | Instantaneous forward rate f(0,T) = −d/dT [T · Y(0,T)]. | exp/olo/plots.py, rates/accrual.py, rates/pricing.py, rates/simulation.py, tests/test_pricing.py |
| `nss_forward_deriv(T: numpy.ndarray, beta0, beta1, beta2, beta3, tau1, tau2) -> numpy.ndarray` | df/dT — first derivative of the forward rate. |  |
| `nss_yield(T: numpy.ndarray, beta0, beta1, beta2, beta3, tau1, tau2) -> numpy.ndarray` | Nelson-Siegel-Svensson yield Y(0,T). | checks.py, exp/olo/plots.py, rates/pricing.py, tests/test_pricing.py |

### `pension.rates.simulation` (pension/rates/simulation.py)

.

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `simulateHullWhite(curve: dict, kappa: float, sigma: float, T: int = 40, n_paths: int = 1000, dt: float =...)` | Simulate Hull-White short-rate paths using EXACT step-wise discretisation via the shifted decomposition r(t) = x(t) + alpha(t). | checks.py, economy.py, exp/olo/plots.py |
| `simulateVasicek(kappa: float, theta: float, sigma: float, r0: float, T: int = 40, n_paths: int = 1000,...)` | Simulate Vasicek short-rate paths using exact discretisation. | economy.py, exp/olo/plots.py, exp/walkthrough.py |

### `pension.rates.pricing` (pension/rates/pricing.py)

.

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `computeCumulativeDiscountFactors(paths, dt=0.08333333333333333)` | D(0,t) = exp(-∫₀ᵗ r ds) ≈ exp(-Σ r·dt)  per path  → feeds APV = Σ CF·D. | tests/test_pricing.py |
| `hullWhiteBondPrice(r, t, kappa, sigma, tau, curve)` | Hull-White zero-coupon bond price P(t, t+τ) given the short rate r(t). | tests/test_pricing.py |
| `reconstructFutureYield(paths: numpy.ndarray, kappa: float, sigma: float, curve: dict, tau: float = 10.0, dt: ...)` | Reconstruct the model-implied τ-year yield Y(t, t+τ) at every (time, path), from the simulated short rate via the Hull-White bond price: | checks.py, economy.py, tests/test_pricing.py |
| `vasicekBondPrice(r, kappa, theta, sigma, tau)` | Vasicek zero-coupon bond price P(t, t+τ) = A(τ)·exp(-B(τ)·r(t)). Constant long-run mean θ → a single model-implied curve (does NOT fit market). | economy.py, rates/calibration.py, tests/test_pricing.py, tests/test_rates.py |

### `pension.rates.wap` (pension/rates/wap.py)

the statutory WAP/LPC return guarantee rate G_t (art.

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `computeWAPRate(y10_monthly, start_row, n_years, months_per_year=12)` | Annual WAP guarantee rate per path from a monthly 10Y yield series. | checks.py, economy.py, exp/walkthrough.py |
| `wap_formula(avg10y)` | The statutory map from a 24-month average 10Y yield (decimal) to G (decimal). | checks.py, exp/walkthrough.py, tests/test_rates.py |

### `pension.rates.accrual` (pension/rates/accrual.py)

the real-world accrual factor of a premium paid at t until retirement T.

| function / class | purpose (first docstring paragraph) | used in |
|---|---|---|
| `closed_form_accrual(t, r_t, p, s=0.0)` | A(t, T) for p.RATE_MODEL in {"vasicek", "vasicek_short", "hull_white_p"}; r_t (the SHORT rate at t) scalar or array (one per path). T = p.T; the financing spread s is added per year. | checks.py, exp/walkthrough.py, numeraire.py, tests/test_mechanisms.py, tests/test_rates.py |
| `hull_white_alpha(t, kappa, sigma, curve)` | alpha(t) = f(0,t) + sigma^2/(2 kappa^2) (1 - e^{-kappa t})^2. | checks.py |
| `hull_white_alpha_integral(t, T, kappa, sigma, curve)` | int_t^T alpha(u) du, analytic. |  |

<!-- /gen -->

## `Params`

"Inert (canonical)" asks: does changing this field change any result of the canonical configuration ("retirement", constant or vasicek)? "CE sets it" means that under a rate model the certainty-equivalent solve overwrites the field with scenario moments, and `simulate` uses the path-wise rates instead.

<!-- gen:params -->
| field | default | unit | meaning | used by | inert (canonical) |
|---|---|---|---|---|---|
| `T` | `45` | years | career length | everything | no |
| `G` | `0.03` | 1/yr | WAP guarantee rate (constant model): mean growth of the vertical ledger | dynamics (vertical ledger), dp | vasicek: yes (CE sets it) |
| `MU` | `0.03` | 1/yr | mean credited return, E[growth] = e^MU (constant model; median with DRIFT_CORRECTION off) | dynamics, dp, paidup_service | vasicek: yes (CE sets it) |
| `W` | `0.025` | 1/yr | salary growth | dynamics, objective contribution leg, dp | no |
| `S0` | `1.0` | salary | starting salary (scale-free) | nothing in the DP/env (tabular via economy) | yes |
| `DISC_EMP` | `0.03` | 1/yr | employee discount rate | numeraire (discounted only) | yes |
| `DISC_ER` | `0.05` | 1/yr | employer discount rate | numeraire (discounted only) | yes |
| `DISC` | `0.05` | 1/yr | tabular default, read once at import into economy.DISC | nothing after import (tabular reads economy.DISC) | yes |
| `SIGMA_R` | `0.05` | 1/sqrt(yr) | asset shock (constant model) | dynamics, dp | vasicek: yes (CE sets it) |
| `SIGMA_L` | `0.02` | 1/sqrt(yr) | guarantee shock (vertical ledger) | dynamics, dp | vasicek: yes (CE sets it) |
| `SIGMA` | `0.05` | 1/sqrt(yr) | tabular rung's shock (legacy copy) | nothing (tabular reads economy.SIGMA) | yes |
| `GAMMA` | `0.15` | share of salary | contribution capacity, c = a GAMMA S | dynamics, objective, dp | no |
| `LAMBDA` | `0.5` | - | employee weight | objective weights | no |
| `ETA` | `2.0` | - | CRRA curvature | objective.crra | no |
| `ANNUITY` | `15.0` | years | annuity factor, capital -> annual pension | objective, dp, simulate, env | no |
| `RR_LEGAL` | `0.43` | - | first-pillar replacement rate | dp, simulate, env | no |
| `RR_TARGET` | `0.7` | - | total-adequacy target | dp, simulate, env | no |
| `SATIATE` | `False` | bool | cap RR at the target in crra() | objective.crra | no |
| `OBJECTIVE` | `'baseline'` | name | value function in objective.OBJECTIVES | dp, env | no |
| `BETA` | `0.01` | - | policy extraction temperature (relative) | dp._solve | no |
| `RATE_MODEL` | `'constant'` | name | rate regime | economy, numeraire, dp, env | no |
| `RATE_DT` | `0.08333333333333333` | years | rate time step (must be 1/12: the data are monthly) | economy | no (not free) |
| `RATE_SEED` | `2026` | - | seed of the rate scenarios | economy, dp, env pool | no |
| `BOOK_DURATION` | `8` | years | book-yield averaging window | economy | no |
| `BOOK_SPREAD` | `0.0` | 1/yr | book yield over the 10Y | economy | no |
| `SIGMA_R_RATES` | `0.05` | 1/sqrt(yr) | excess-return noise on the book yield | dynamics, dp.certainty_equivalent | no |
| `LONG_RATE_P` | `None` | 1/yr | long-run real-world short rate (every model; vasicek: theta = this + spread) | economy params helpers, accrual | no |
| `SPREAD_10Y_SHORT` | `None` | 1/yr | vasicek: r = y10 - spread (None: historical mean) | economy.vasicek_params, accrual | no |
| `DRIFT_CORRECTION` | `True` | bool | shocked growth e^{m - s^2/2 + s z} (mean e^m); False: legacy e^{m + s z} (REVIEW M6) | dynamics.step, dp.F_next/rho_next | no |
| `CE_MOMENTS` | `'conditional'` | name | CE shocks: 'conditional' (SIGMA_L = 0, SIGMA_R = SIGMA_R_RATES) | 'levels' (legacy, REVIEW M10) | dp.certainty_equivalent | constant: yes |
| `YEAR_START` | `'january'` | name | 'january': model years are calendar years | 't0' (legacy, REVIEW C1) | economy, accrual (hull_white_p) | constant: yes |
| `RATE_CE_PATHS` | `5000` | - | scenarios behind the CE solve | dp.solve | no |
| `EMPLOYER_NUMERAIRE` | `'retirement'` | name | valuation date: 'retirement' | 'discounted' (REVIEW m9) | numeraire | no |
| `SHORT_RATE` | `0.03` | 1/yr | short rate of the constant model | numeraire, dynamics.Exogenous | vasicek: yes |
| `FINANCING_SPREAD` | `0.0` | 1/yr | s: financing spread over the short rate | numeraire, accrual | no |
| `N_EVAL` | `2000` | - | tabular rung's SAA batch size (legacy copy) | nothing (tabular reads economy.N_EVAL) | yes |
<!-- /gen -->

## Objective registry (`pension.objective.OBJECTIVES`)

Each entry combines four parts: an employee leg, a shortfall cost, a contribution cost and the weights. The valuation-date factors (`pension/numeraire.py`) are applied around them.

<!-- gen:objectives -->
| name | what it is |
|---|---|
| `baseline` | committed model: CRRA(ETA) employee, linear employer, lambda weights |
| `log` | log utility (eta=1, Kelly / ergodicity-canonical) |
| `loss_averse` | shortfalls below the target weigh 2.5x |
| `satiated` | no value above the target |
| `convex_shortfall` | employer: shortfall + shortfall^2 |
| `cashflow_strain` | employer: contributions cost (1 + 2 * share of payroll) per euro |
| `employer_floor` | employer cost only, RR >= target enforced as a penalty |
<!-- /gen -->

## Checks (`pension.checks`)

Each check returns what it measures; the tests assert on it. "Tolerance" is the first `assert` of each test that calls the check. The tolerances were tightened in the REVIEW fix round (M12).

<!-- gen:checks -->
| check | what it measures (known answer) | tolerance (first assert) | asserted in |
|---|---|---|---|
| `accrual_accuracy` | (e) Per payment year t: the Monte Carlo mean of the realised accrual to T, prod_{u >= t} acc_u (monthly r * dt sums), against the mean of the closed-form conditional A(t, T; r_t) over the same paths (tower property). Reported as the relative error of the paired difference and its standard error. | `assert abs(d["rel_err"]) < 4 * d["rel_se"] + 1e-5`<br>`assert abs(d["rel_err"]) < 4 * d["rel_se"] + 1e-5` | `test_numeraire.py::test_accrual_accuracy_vasicek`<br>`test_rates.py::test_closed_form_accrual_matches_monte_carlo` |
| `cross_scores` | J[i, j]: the policy optimised under objective i (policies: {name: policy}), scored under objective j; and floor[j], the score of the constant floor_a policy. An objective's own policy must be best in its column. | `assert np.all(margin <= 0.01)`<br>`assert np.all(margin <= 0.002)` | `test_objective.py::test_each_objective_prefers_its_own_policy`<br>`test_objective.py::test_each_objective_prefers_its_own_policy_protocol_grid` |
| `degenerate_rates_gap` | A rate scenario with G_t = G, mu_t = MU (SIGMA_L = 0) against the constant model: max gap in joint value and RR -- must be ~0. | `assert checks.degenerate_rates_gap(p, grid["Fg"], grid["rg"], entry, 2000, 7) < 1e-10` | `test_rates.py::test_degenerate_scenario_is_the_constant_model` |
| `diagonal_margin` | Per column j: how far the best OTHER policy is above j's own optimum (<= 0 means the diagonal wins), as a fraction of j's gain over the floor (relative_to="gain") or of ∣J_jj∣ ("value"). The gain normalisation is ill-conditioned when an objective barely beats the floor (it amplifies grid error); the value normalisation is not. Returns (margins, best other index). | `assert np.all(margin <= 0.01)`<br>`assert np.all(margin <= 0.002)` | `test_objective.py::test_each_objective_prefers_its_own_policy`<br>`test_objective.py::test_each_objective_prefers_its_own_policy_protocol_grid` |
| `env_contract` | (a) ∣mean PensionEnv episode return - simulate()["joint"]∣ on the same paths, with the solved policy (0 up to floating point). | `assert checks.env_contract(p.replace(EMPLOYER_NUMERAIRE=num), grid["Fg"], grid["rg"], grid["ag"], grid["nq"], small, 200, 7) < 1e-12` | `test_numeraire.py::test_env_contract_both_numeraires` |
| `eta_log_limit` | (∣u - log∣ at eta=1, ∣u(eta=1) - u(eta=1+eps)∣): u is log at eta=1 and continuous there. | `assert to_log < 1e-12 and jump < 1e-3` | `test_invariants.py::test_eta_one_is_log` |
| `face_value_premiums` | (c) SHORT_RATE = s = 0: every premium factor of the retirement numeraire is exactly 1, i.e. premiums count at face value (per unit of final salary). The same face value is the discounted objective at DISC_ER = DISC_EMP = 0, so the two must give the same cost and joint value. Returns (max ∣factor - 1∣, max ∣difference∣ in cost and joint). | `assert f == 0.0 and d < 1e-12` | `test_numeraire.py::test_face_value_premiums` |
| `headline` | (f) The banded optimum under p, scored under score_p (default p): employer cost, joint value, median stayer / leaver total RR, mean contribution (% of salary) by decade. | `assert len(h["by_decade"]) == 5 and h["cost"] > 0` | `test_numeraire.py::test_headline_runs` |
| `horizontal_closed_form` | One career, no churn, no shocks, with G stepping G_before -> G_after at year `switch`. Returns (relative error of the horizontal ledger against L_T = L0 e^{G_0 T} + sum_s c_s e^{G_s (T-s)}, and how far the VERTICAL method -- every euro at the current rate -- would land from it). | `assert rel < 1e-12 and abs(vertical_gap) > 0.05` | `test_dynamics.py::test_horizontal_ledger_closed_form` |
| `hw_p_drift` | (c) Per year: the scenario mean of r_t (with its standard error) against E^P[r_t] = alpha(t + d) + m (1 - e^{-kappa (t + d)})  (E^Q[r_t] = alpha(t + d)), d = economy.year_offset(p) the calendar time from the curve date to model year 0. | `assert abs(d["mc"] - d["analytic"]) < 4 * d["se"]` | `test_rates.py::test_hw_p_real_world_drift` |
| `hw_p_link` | (a) With phi = 0 and the legacy sigma, hull_white_p's simulation IS hull_white's: max ∣difference∣ of the monthly short rate and 10Y (must be 0), and of the annual r against the hull_white scenarios (same seed). | `assert checks.hw_p_link(n=50) == dict(r=0.0, y10=0.0, scenario_r=0.0)` | `test_rates.py::test_hw_p_reproduces_hull_white_at_phi_zero_and_legacy_sigma` |
| `hw_p_t0_fit` | (b) The model 10Y at t0 against the NSS 10Y (equal: exact fit to today's curve) and against the last observed 10Y (the NSS fitting error). | `assert abs(f["model_10y_t0"] - f["nss_10y"]) < 1e-12 and abs(f["r0"] - f["f00"]) < 1e-12` | `test_rates.py::test_hw_p_fits_todays_curve` |
| `lambda_equivalent` | The LAMBDA that reproduces (lam, de_new) at the reference DISC_EMP:     lambda'' = A / (A + B*exp(-de_ref*T)),  A = lam*exp(-de_new*T), B = 1-lam. delta_e only multiplies the employee leg by exp(-delta_e*T), so it is a LAMBDA change up to a positive rescale of the objective. |  | via other checks |
| `lambda_monotonicity` | (employer cost, median stayer RR) at each LAMBDA; both must increase. | `assert np.all(np.diff(cost) > 0) and np.all(np.diff(sty) > 0)` | `test_invariants.py::test_cost_and_adequacy_increase_with_lambda` |
| `lambda_prime` | The retirement-numeraire weight equivalent to the discounted objective at (lam, DISC_ER = r, DISC_EMP = de):     lambda' = lam e^{-de T} / (lam e^{-de T} + (1 - lam) e^{-r T}). |  | via other checks |
| `lambda_reparam` | Max ∣policy difference∣ between (lam, DISC_EMP=de_new) and its LAMBDA equivalent at the reference DISC_EMP -- must be ~0. Returns (gap, lam_eq). A statement about the DISCOUNTED objective (DISC_EMP does not enter the retirement numeraire), so it is evaluated there. | `assert gap < 1e-6` | `test_invariants.py::test_delta_e_is_a_lambda_change` |
| `leaver_terminal_gap` | Max ∣Phi[T] - terminal∣: a leaver with full service IS a stayer. Must be 0. | `assert checks.leaver_terminal_gap(p, grid["Fg"], grid["rg"], objective) < 1e-12` | `test_invariants.py::test_full_service_leaver_is_a_stayer` |
| `no_capacity_gap` | GAMMA -> 0: median stayer RR of the banded optimum minus that of paying nothing -- must be ~0 (contributions cannot add anything). | `assert abs(checks.no_capacity_gap(p, grid["Fg"], grid["rg"], grid["nq"], entry, N, SEED)) < 0.01` | `test_invariants.py::test_no_capacity_adds_nothing` |
| `numeraire_equivalence` | (b) Constant rates. The retirement objective with SHORT_RATE = r, s = 0 at lambda' and the discounted objective with DISC_ER = r, DISC_EMP = de at lambda differ by the positive factor e^{-rT} (1 - lam) / (1 - lambda'), so the policy must coincide. Returns (max ∣policy difference∣, max relative deviation of V_old / V_new from that factor, lambda'). | `assert dpol < 1e-6 and dV < 1e-10` | `test_numeraire.py::test_retirement_equals_discounted_at_lambda_prime` |
| `reduced_form_gap` | (a) Max ∣F, rho from dynamics.step - dp.F_next / rho_next∣ over one year, constant rates (0 up to floating point). | `assert checks.reduced_form_gap(p) < 1e-12` | `test_numeraire.py::test_reduced_form_matches_step` |
| `scale_invariance` | Max relative change in RR when (R, L, S) are all scaled by k. The model is homogeneous of degree 0, so this must be ~0. | `assert checks.scale_invariance(p, grid["Fg"], grid["rg"], grid["ag"], grid["nq"], N, SEED) < 1e-12` | `test_invariants.py::test_scale_invariance` |
| `timing_neutrality` | (early, late) mean contribution (% of salary, years 0-9 and 35-44) when a contribution's cost and value grow alike: SHORT_RATE = MU = G under the retirement numeraire (premiums accrue at the rate R and L earn), DISC_ER = DISC_EMP = MU under the discounted one. The benefit/cost ratio is then flat in t, so the schedule must be roughly level -- provided the reserve's MEAN return is MU, i.e. DRIFT_CORRECTION (REVIEW M6; without it the mean is MU + SIGMA_R^2/2 and early funding pays). Constant rates only. | `assert abs(early - late) / max(early, late) < 0.08`<br>`assert abs(early - late) / max(early, late) < 0.08`<br>`assert (early - late) / early > 0.10` | `test_invariants.py::test_timing_neutral_when_discounts_equal_mu`<br>`test_numeraire.py::test_timing_neutral_at_short_rate_equal_mu`<br>`test_numeraire.py::test_legacy_drift_tilts_towards_early_funding` |
| `vasicek_short_fit` | Calibration and fit of the P-measure Vasicek short rate: the maturity used as short-rate proxy, kappa, theta_P (OLS, and the one simulated: p.LONG_RATE_P if set), sigma, the significance of the mean reversion, theta_Q and phi, the in-sample fit of the reconstructed 10Y from the observed short rate (RMSE, bias -- zero by construction --, correlation, RMSE over the last 24 months), the jump at t0, and the AR(1) decay of the t0 offset (phi_e, its standard error, half-life, and whether the kappa fallback is used). | `assert f["maturity"] == "1Y"`<br>`assert f1["theta_P"] == 0.0225 and f0["theta_P"] == f0["theta_P_ols"]` | `test_rates.py::test_vasicek_short_calibration`<br>`test_rates.py::test_vasicek_short_long_rate_anchor` |
| `vol_target` | (d) Std of the model's monthly 10Y changes (first `years` years, pooled) against the historical std of monthly 10Y changes; for hull_white_p and for hull_white (legacy sigma). Also B(10)/10, hull_white's understatement factor. | `assert abs(v["hull_white_p"] / v["historical"] - 1) < 0.05` | `test_rates.py::test_hw_p_hits_the_10y_volatility` |
| `wap_scenario_stats` | min, max, distance from the 25 bp grid, and G_0 of the WAP rates of n scenarios of `model`; G0_observed is the statutory formula applied directly to year 0's window of the OLO history (None if that window reaches past the last observation, so G_0 is not yet known). | `assert st["min"] >= 0.0175 - 1e-12 and st["max"] <= 0.0375 + 1e-12` | `test_rates.py::test_wap_rates_of_scenarios` |
| `wap_vs_fsma` | {year: (WAP rate from the statutory formula on the cached NBB data, rate published by the FSMA)}, in %. 2027 is left out on purpose: the formula gives 2.75% against 2.50% published (see pension/rates/wap.py). | `assert abs(ours - published) < 1e-9` | `test_rates.py::test_wap_formula_reproduces_fsma_rates` |
<!-- /gen -->
