# legacy_v1 — the pre-October-2026 model (frozen)

The specification at tag `v1-contribution-action`, kept for regression and
comparison only, like the tabular rung. It is not updated, and the October 2026
consistency sweep does not apply to it.

- Action: contribution level, a fraction `a` of capacity `GAMMA * S`.
- Reserve: lognormal shock `SIGMA_R` (constant rates) or `SIGMA_R_RATES` (rate mode)
  on the credited return; book yield = rolling 8-year mean of the 10Y OLO.
- Liability: vertical with shock `SIGMA_L` at constant rates, horizontal ledger
  (one vintage per premium, locked at its `G_t`) in rate mode.
- State: reduced `(t, F, rho)`; the RL environment observes `(t/T, F, log rho, G_t, mu_t, r_t)`.
- Objective: `EMPLOYER_NUMERAIRE` (`"retirement"` / `"discounted"`), premium cost inside the reward.

Self-contained: own parameters, dynamics, objective, numeraire, DP, environment
(`env.py`), a vendored copy of the rate engine (`rates/`) and a snapshot of the OLO
history (`rates/data/olo_yields.csv`, to 2026-09). `MODEL.md` is the model document
as it stood. `tests/test_legacy_v1.py` pins it to the goldens recorded on the tag.
