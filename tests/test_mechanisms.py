"""Known answers for canonical mechanisms that had no test of their own (REVIEW M11):
the financing spread, the vasicek calibration knobs, the CE premium schedule, the
evaluate mode, the soft readout, and the scenario guards (REVIEW M2-M4).

Scope: the DP stack (pension.dp / dynamics / economy) at constant rates, the
single-vehicle rung. Its rate mode is tested on pension.legacy_v1 (tests/test_legacy_v1.py);
the baseline in tests/test_baseline.py."""
import numpy as np
import pytest

import pension.dp as dp
from pension import economy, numeraire


@pytest.mark.parametrize("model", ["constant", "vasicek"])
def test_financing_spread_multiplies_accrual(p, model):
    """s > 0 multiplies A(t,T) by e^{s (T - t)}, in both the constant and the rate model."""
    q0 = p.replace(RATE_MODEL=model)
    q1 = q0.replace(FINANCING_SPREAD=0.01)
    r = np.array([0.0, 0.02, 0.05])
    for t in (0, 10, 44):
        ratio = numeraire.premium_factor(t, r, q1) / numeraire.premium_factor(t, r, q0)
        np.testing.assert_allclose(ratio, np.exp(0.01 * (p.T - t)), rtol=1e-12)


def test_spread_override_sets_the_short_rate(p):
    """SPREAD_10Y_SHORT replaces the historical spread: r = y10 - spread on every path."""
    q = p.replace(RATE_MODEL="vasicek", SPREAD_10Y_SHORT=0.005)
    assert economy.vasicek_params(q)["spread"] == 0.005
    sc = economy.draw_rate_scenarios(100, p=q)
    np.testing.assert_allclose(sc["y10"] - sc["r"], 0.005, atol=1e-15)


def test_evaluate_mode_reproduces_the_optimum(p, grid):
    """Evaluating the argmax policy (beta = 0) as a plan rule gives back the
    optimised value function."""
    kw = dict(Fg=grid["Fg"], rg=grid["rg"], ag=grid["ag"], n_quad=grid["nq"], beta=0.0, p=p)
    opt = dp.solve(**kw)
    ev = dp.solve(mode="evaluate", plan_rule=lambda t, F, r: opt["policy"][t], **kw)
    np.testing.assert_allclose(ev["V"], opt["V"], rtol=1e-10, atol=1e-12)


def test_soft_readout_is_affine_invariant():
    rng = np.random.default_rng(0)
    Q, ag = rng.normal(size=(7, 5, 4)), np.linspace(0, 1, 7)
    for beta in (0.01, 0.1, 1.0):
        np.testing.assert_allclose(dp._soft_readout(3.7 * Q - 12.0, ag, beta),
                                   dp._soft_readout(Q, ag, beta), rtol=1e-12)
    tied = dp._soft_readout(np.ones((7, 2, 2)), ag, 0.1)          # a tie: uniform weights
    np.testing.assert_allclose(tied, ag.mean())


def test_cached_scenarios_are_read_only(p):
    """REVIEW M3: the memoised dict is shared, so its arrays cannot be written."""
    sc = economy.draw_rate_scenarios(10, p=p.replace(RATE_MODEL="vasicek"))
    for k in ("G", "mu", "r", "acc", "y10"):
        with pytest.raises(ValueError):
            sc[k][0, 0] = 0.0


def test_scenario_cache_is_keyed_on_the_data_vintage(p, monkeypatch):
    """REVIEW M4: a different data vintage draws afresh."""
    q = p.replace(RATE_MODEL="vasicek")
    a = economy.draw_rate_scenarios(10, p=q)
    assert economy.draw_rate_scenarios(10, p=q) is a
    monkeypatch.setattr(economy, "data_vintage", lambda: "another vintage")
    assert economy.draw_rate_scenarios(10, p=q) is not a
