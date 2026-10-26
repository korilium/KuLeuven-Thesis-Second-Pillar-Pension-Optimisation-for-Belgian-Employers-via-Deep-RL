# Review: baseline v2 (October 2026)


The model is now the bare-bone baseline of `pension/baseline.py`, specified in `docs/MODEL.md`. The pre-October-2026 model is frozen in `pension/legacy_v1/` (tag `v1-contribution-action`) and pinned by `tests/test_legacy_v1.py`. The earlier code review of the DP stack (C1, M1–M13, m1–m14) and that stack itself were removed in the clean-up; both are in tag `v2-before-cleanup`.

**Decisions taken.**
- Book yield: an exponential filter with monthly weight $w = 1/48$ (mean age about 4 years, like the former 8-year window). It replaces the 96-month rolling mean, which would have added 95 monthly variables to the exact state.
- Defaults (confirmed): $c = 5\%$, $m = 0$, $i^{\text{tariff}} = 1\%$, `LONG_RATE_P = None`, and a new member starting with zero balances.
- The premium cost carries no Γ: $c$ is a share of salary.

**C1. WAP formula** (`rates/wap.py`): $0.85\times$ the 24-month mean of the 10Y ending in May of the year before, rounded to the nearest 25 bp, clipped to [1.75%, 3.75%], fixed on 1 January.
- On the NBB series the 85% rule reproduces every FSMA rate from 2016 to 2026. The 24-month averages for 2016–2024 lie between −0.11% and 1.75%, so the floor binds under any share and those years cannot tell 65%, 75% and 85% apart.
- Only 2025 and 2026 identify the share: 2.50/2.50 published; 85% gives 2.50/2.50, 75% gives 2.25/2.25, 65% gives 1.75/2.00.
- 2027: the average is 3.16%, so 0.85 × 3.16% = 2.69% rounds to 2.75%, against 2.50% published. Rounding down would give 2.50% for 2027, but 2.25% for 2025 (published 2.50%), so the rounding rule is not the cause. A different reference series remains the likely explanation.
- Open: the multiplier schedule of art. 24 §3 after 2015 still has to be checked against the coordinated text. `paper/literature/WAP/legaltext.pdf` is the 2003 original.

**C2. Book yield $\mu_t$ in the legacy stack** (`pension/legacy_v1/economy.py`): the rolling mean of the last 96 monthly 10Y yields, ending in January of year $t$, plus `BOOK_SPREAD` = 0, with no floor. It is credited with lognormal noise `SIGMA_R_RATES` = 5%. $m$ corresponds to −`BOOK_SPREAD`. No existing parameter is a tariff floor: `MU` = 3% is the constant rung's credited rate. As a floor, 3% would bind in 61% of path-years.

**A4. Matched zero-coupon bond.** The flat price $e^{-y_t(T-t)}$ earns a mean holding log-return of 4.05%/yr over the career, against a mean $r_t$ of 1.33%/yr (mean $y$ 2.62%, mean $G$ 2.46%). By year:

| year | holding return | $r_t$ |
|---|---|---|
| 0 | 12.5% | 2.8% |
| 5 | 8.2% | 2.0% |
| 20 | 2.6% | 1.1% |
| 44 | 2.2% | 0.9% |

- Held to $T$, each premium locks in $y_t$, which is $s_{\text{term}}$ = 128 bp above $r_t$ by construction.
- Early on, $y$ mean-reverts from 4.1% to $\theta$ = 2.23%, which produces large predictable mark-to-market gains.
- Consequence: $\alpha \equiv 0$ beats $\alpha \equiv 1$ on both legs (see the report). This is reported only; affine pricing is future work.

**Baseline numbers** (`python experiments/baseline_report.py`, 20 000 careers, `EVAL_SEED`):

| year | mean $G_t$ | at 1.75% | mean $\mu_t$ | mean $i_t - G_t$ | mean $r_t$ |
|---|---|---|---|---|---|
| 0 | 2.75% | 0.0% | 2.50% | −0.25% | 2.77% |
| 5 | 2.92% | 7.4% | 3.25% | +0.33% | 1.99% |
| 20 | 2.34% | 41.5% | 2.57% | +0.26% | 1.15% |
| 44 | 2.24% | 48.4% | 2.26% | +0.08% | 0.96% |

| $\alpha$ | $J$ | employee leg | mean shortfall / $S_T$ | P(shortfall > 0) | premium cost |
|---|---|---|---|---|---|
| 1 | −0.3369 | −0.6688 | 0.00497 | 4.9% | 0.5437 |
| 0 | −0.2166 | −0.4319 | 0.00120 | 1.2% | 0.5437 |
| 0.5 | −0.2755 | −0.5497 | 0.00136 | 1.7% | 0.5437 |

**Removing $\sigma_R$:** the same 5% premium, all in Branch 21. (Produced by `experiments/baseline_report.py` as it stood at tag `v2-before-cleanup`; the current report is baseline-only.)

| model | employee leg | mean shortfall / $S_T$ | P(shortfall > 0) |
|---|---|---|---|
| legacy_v1, $\sigma_R$ = 5% (as frozen) | −0.6285 | 0.0457 | 12.0% |
| legacy_v1, $\sigma_R$ = 0 | −0.6510 | 0.0261 | 11.2% |
| baseline, $\alpha$ = 1, no tariff floor | −0.6740 | 0.0082 | 5.5% |

- Within legacy_v1, removing the shock cuts the expected shortfall by 43%.
- The remaining gap to the baseline comes from the vertical liability at the current $G_t$ (against horizontal vintages), the exponential book-yield filter and the zero entry balances.

**DP reference** (`pension/baseline_dp.py`, the exact nine-variable state, value function fitted by regression on 20 000 training paths; scored on 20 000 evaluation careers):

| | value |
|---|---|
| $J$ of the greedy policy (out of sample) | −0.2027 |
| employee leg | −0.4052 |
| mean shortfall / $S_T$ | 0.00016 |
| P(shortfall > 0) | 0.27% |
| in-sample $V_0$ | −0.2150 |

- The policy beats the best constant policy ($\alpha \equiv 0$: −0.2166).
- Mean $\alpha$ in force by year: 0.00 (year 0), 0.17 (5), 0.50 (10), 0.53 (20), 0.47 (30), 0.50 (44). Everything goes to Branch 23 while the 10Y is high early in the career; later the premium is split about evenly, as the yield falls towards $\theta$ and the tariff floor makes Branch 21 the better hedge of $G_t$.
- The fitted $V_0$ moves by about 0.01 with the number of training paths (−0.199 at 4 000, −0.215 at 20 000), while the policy's $J$ is stable (−0.2047, −0.2033). Read the policy's $J$, not $V_0$.

**Future work (not built).**
- horizontal liability ledger;
- per-vintage Branch 21 crediting;
- profit-sharing rules (2/6 with budget rule 3);
- employer cost account in the reward;
- Branch 23 liquidation at exit;
- affine Vasicek bond pricing;
- stochastic salary, a second rate factor, equity Branch 23.

Out of scope, future work only: the full workforce, SAC, Transformers, multi-agent and hierarchical set-ups. The cash-balance design and the analytical limiting cases are removed.
