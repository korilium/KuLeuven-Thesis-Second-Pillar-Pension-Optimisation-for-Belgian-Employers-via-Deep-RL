"""The rate mode of pension.dp (horizontal ledger, book-yield crediting with
SIGMA_R_RATES, certainty-equivalent solve) is still live code. It is tested two ways:
it equals its frozen copy pension.legacy_v1 at the canonical settings, and it passes
the structural rate-mode checks on pension's own modules."""
import numpy as np
import pytest

import pension.dp as dp
import pension.legacy_v1.dp as ldp
from pension import checks, economy, numeraire
from pension.dynamics import Exogenous, State, step
from pension.params import DEFAULT
from pension.rates.accrual import closed_form_accrual

GRID = dict(Fg=dp.make_F_grid(n=31), rg=dp.make_rho_grid(n=25), ag=dp.make_a_grid(n=7), n_quad=3)


@pytest.mark.parametrize("model", ["vasicek", "hull_white_p"])
def test_rate_mode_equals_legacy_v1(model):
    """Canonical settings (retirement valuation, January years, drift correction,
    conditional CE moments): solve and simulate agree exactly with legacy_v1."""
    p = DEFAULT.replace(RATE_MODEL=model)
    lp = checks.legacy_params(p)
    a, b = dp.solve(**GRID, p=p), ldp.solve(**GRID, p=lp)
    assert np.array_equal(a["policy"], b["policy"])
    np.testing.assert_allclose(a["V"], b["V"], rtol=1e-12)
    R0, L0, S0 = dp.new_plan_init(500, np.random.default_rng(1))
    kw = dict(R0=R0, L0=L0, S0=S0, n_paths=500, seed=7)
    sa = dp.simulate(a["policy"], GRID["Fg"], GRID["rg"], p=p, **kw)
    sb = ldp.simulate(b["policy"], GRID["Fg"], GRID["rg"], p=lp, **kw)
    for k in ("benefit", "cost", "joint", "regime_B"):
        assert sa[k] == pytest.approx(sb[k], rel=1e-12), k
    np.testing.assert_allclose(sa["L_T"], sb["L_T"], rtol=1e-12)


def _exo(p, zR, zL, u, G=None, mu=None):
    tile = lambda x: np.tile(np.asarray(x, float), (p.T, 1))
    return Exogenous(tile(zR), tile(zL), tile(u),
                     None if G is None else tile(G), None if mu is None else tile(mu))


def test_degenerate_rates_equal_constant_rates():
    """G_t = G and mu_t = MU, SIGMA_L = 0: the horizontal ledger with one rate is
    the vertical ledger, so the states coincide."""
    p = DEFAULT.replace(SIGMA_L=0.0, SIGMA_R_RATES=DEFAULT.SIGMA_R)
    rng = np.random.default_rng(0); n = 50
    zR, zL, u, a = rng.standard_normal(n), rng.standard_normal(n), rng.random(n), rng.random(n)
    c_exo, r_exo = _exo(p, zR, zL, u), _exo(p, zR, zL, u, G=np.full(n, p.G), mu=np.full(n, p.MU))
    sc = State.initial(p, c_exo, 1.0, 1.0, 10.0); sr = State.initial(p, r_exo, 1.0, 1.0, 10.0)
    for _ in range(p.T):
        sc, _ = step(sc, np.where(sc.present, a, 0.0), c_exo, p, dp.tenure_hazard)
        sr, _ = step(sr, np.where(sr.present, a, 0.0), r_exo, p, dp.tenure_hazard)
    np.testing.assert_allclose(sr.R, sc.R, rtol=1e-13)
    np.testing.assert_allclose(sr.L, sc.L, rtol=1e-13)
    assert np.array_equal(sr.leave_t, sc.leave_t)


def test_horizontal_ledger_closed_form():
    """G steps 3% -> 1.75% at t = 20: old money keeps its rate; vertical would differ."""
    rel, vertical_gap = checks.horizontal_closed_form(DEFAULT)
    assert rel < 1e-12 and abs(vertical_gap) > 0.05


def test_horizontal_ledger_compresses_into_rate_buckets():
    """G lives on a 25 bp grid in [1.75%, 3.75%]: the horizontal ledger is exactly
    9 buckets "liability locked at rate g"."""
    p = DEFAULT.replace(RATE_MODEL="vasicek")
    n = 50
    exo = Exogenous.draw(p, n, 3, economy.draw_rate_scenarios(n, p=p))
    grid = np.round(np.arange(0.0175, 0.03751, 0.0025), 6)
    s = State.initial(p, exo, 1.0, 1.0, 15.0)
    buckets = np.zeros((len(grid), n))
    buckets[np.searchsorted(grid, np.round(exo.G[0], 6)), np.arange(n)] = 1.0
    rng = np.random.default_rng(0)
    for t in range(p.T):
        a = np.where(s.present, rng.random(n), 0.0)
        present = s.present.copy()
        s, c = step(s, a, exo, p, dp.tenure_hazard)
        g = np.searchsorted(grid, np.round(exo.G[t], 6))
        assert np.allclose(grid[g], exo.G[t], atol=1e-12)
        buckets[g, np.arange(n)] += c
        buckets = np.where(present, buckets * np.exp(grid[:, None]), buckets)
        np.testing.assert_allclose(buckets.sum(axis=0), s.L, rtol=1e-12)


def test_ce_premium_is_the_scenario_mean_of_the_accrual():
    q = DEFAULT.replace(RATE_MODEL="vasicek")
    sc = economy.draw_rate_scenarios(500, p=q)
    prem = numeraire.premium_schedule(q, sc)
    for t in (0, 7, 30, 44):
        assert prem[t] == pytest.approx(closed_form_accrual(t, sc["r"][t], q, s=q.FINANCING_SPREAD).mean(),
                                        rel=1e-12)


def test_scenario_of_another_model_is_refused():
    """REVIEW M2: dynamics and premiums must come from the same model."""
    p = DEFAULT
    sc = economy.draw_rate_scenarios(20, p=p.replace(RATE_MODEL="vasicek"))
    pol = dp.const_policy(0.4, p.T, len(GRID["Fg"]), len(GRID["rg"]))
    with pytest.raises(ValueError):
        dp.simulate(pol, GRID["Fg"], GRID["rg"], n_paths=20, rates=sc, p=p)
    with pytest.raises(ValueError):
        dp.solve(**GRID, rates=sc, p=p)
