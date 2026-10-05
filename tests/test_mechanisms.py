"""Known answers for canonical mechanisms that had no test of their own (REVIEW M11):
the financing spread, the vasicek calibration knobs, the CE premium schedule, the
evaluate mode, the soft readout, and the scenario guards (REVIEW M2-M4)."""
import numpy as np
import pytest

import pension.dp as dp
from pension import economy, numeraire
from pension.rates.accrual import closed_form_accrual


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


def test_ce_premium_is_the_scenario_mean_of_the_accrual(p):
    q = p.replace(RATE_MODEL="vasicek")
    sc = economy.draw_rate_scenarios(500, p=q)
    prem = numeraire.premium_schedule(q, sc)
    for t in (0, 7, 30, 44):
        assert prem[t] == pytest.approx(closed_form_accrual(t, sc["r"][t], q, s=q.FINANCING_SPREAD).mean(),
                                        rel=1e-12)


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


def test_scenario_of_another_model_is_refused(p, grid):
    """REVIEW M2: dynamics and premiums must come from the same model."""
    sc = economy.draw_rate_scenarios(20, p=p.replace(RATE_MODEL="vasicek"))
    pol = dp.const_policy(0.4, p.T, len(grid["Fg"]), len(grid["rg"]))
    with pytest.raises(ValueError):
        dp.simulate(pol, grid["Fg"], grid["rg"], n_paths=20, rates=sc, p=p)
    with pytest.raises(ValueError):
        dp.solve(Fg=grid["Fg"], rg=grid["rg"], ag=grid["ag"], n_quad=grid["nq"], rates=sc, p=p)


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
