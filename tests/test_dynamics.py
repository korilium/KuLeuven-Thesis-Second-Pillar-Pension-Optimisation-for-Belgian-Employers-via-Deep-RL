"""The DP's reduced (F, rho) transitions are the SAME dynamics as pension.dynamics.

The DP works on F = R/L and rho = S/L with closed-form transitions (F_next,
rho_next) and a closed-form paid-up value (paidup_service); simulate() and the RL
environment step the levels (R, L, S) through dynamics.step(). These tests pin the
two to each other, so neither can drift.
"""
import numpy as np

import pension.dp as dp
from pension.dynamics import Exogenous, State, step
from pension.params import DEFAULT

NEVER = lambda t: 0.0          # no churn
ALWAYS = lambda t: 1.0         # leave after the first year


def _exo(p, zR, zL, u, G=None, mu=None):
    T, n = p.T, len(zR)
    tile = lambda x: np.tile(np.asarray(x, float), (T, 1))
    return Exogenous(tile(zR), tile(zL), tile(u),
                     None if G is None else tile(G), None if mu is None else tile(mu))


def test_one_year_in_force_matches_F_next_rho_next():
    p = DEFAULT
    zR = np.array([-1.3, 0.0, 0.7, 2.1]); zL = np.array([0.4, -0.9, 0.0, 1.5])
    R0 = np.array([0.6, 1.0, 1.4, 2.2]); L0 = np.ones(4); S0 = np.array([3.0, 20.0, 0.5, 9.0])
    a = np.array([0.0, 0.3, 0.8, 1.0])
    s = State.initial(p, _exo(p, zR, zL, u=np.ones(4)), R0, L0, S0)
    F, rho = s.F, s.rho
    s, _ = step(s, a, _exo(p, zR, zL, u=np.ones(4)), p, NEVER)
    l = a * p.GAMMA * rho
    np.testing.assert_allclose(s.F, dp.F_next(F, l, zR, zL, p), rtol=1e-13)
    np.testing.assert_allclose(s.rho, dp.rho_next(rho, l, zL, p), rtol=1e-13)


def test_paid_up_roll_forward_matches_paidup_service():
    """A member who leaves at tau, rolled forward by step(), ends where
    paidup_service's closed form says: F grows at MU, rho at (1+W), no shocks."""
    p = DEFAULT
    Fg, rg = dp.make_F_grid(n=31), dp.make_rho_grid(n=25)
    FF, RR = np.meshgrid(Fg[1:], rg, indexing="ij")          # F > 0
    F0, rho0 = FF.ravel(), RR.ravel(); n = F0.size
    exo = _exo(p, np.full(n, 0.8), np.full(n, -0.5), u=np.zeros(n))   # shocks must not matter
    s = State.initial(p, exo, R0=F0, L0=np.ones(n), S0=rho0)
    s.present[:] = False; s.leave_t[:] = 0                    # paid-up from t = 0
    for _ in range(p.T):
        s, _ = step(s, np.zeros(n), exo, p, NEVER)
    np.testing.assert_allclose(s.F, F0 * np.exp(p.MU * p.T), rtol=1e-12)
    np.testing.assert_allclose(s.rho, rho0 * (1 + p.W) ** p.T, rtol=1e-12)


def test_degenerate_rates_equal_constant_rates():
    """A rate scenario with G_t = G and mu_t = MU, SIGMA_L = 0: the horizontal
    ledger with one rate is the vertical ledger, so the states coincide."""
    p = DEFAULT.replace(SIGMA_L=0.0, SIGMA_R_RATES=DEFAULT.SIGMA_R)
    rng = np.random.default_rng(0); n = 50
    zR, zL, u = rng.standard_normal(n), rng.standard_normal(n), rng.random(n)
    a = rng.random(n)
    c_exo = _exo(p, zR, zL, u)
    r_exo = _exo(p, zR, zL, u, G=np.full(n, p.G), mu=np.full(n, p.MU))
    sc = State.initial(p, c_exo, 1.0, 1.0, 10.0); sr = State.initial(p, r_exo, 1.0, 1.0, 10.0)
    for _ in range(p.T):
        sc, _ = step(sc, np.where(sc.present, a, 0.0), c_exo, p, dp.tenure_hazard)
        sr, _ = step(sr, np.where(sr.present, a, 0.0), r_exo, p, dp.tenure_hazard)
    np.testing.assert_allclose(sr.R, sc.R, rtol=1e-13)
    np.testing.assert_allclose(sr.L, sc.L, rtol=1e-13)
    assert np.array_equal(sr.leave_t, sc.leave_t)


def test_horizontal_ledger_closed_form():
    """G steps 3% -> 1.75% at t = 20: old money keeps its rate,
    L_T = L0 e^{G_0 T} + sum_s c_s e^{G_s (T - s)}; vertical would differ."""
    from pension.checks import horizontal_closed_form
    rel, vertical_gap = horizontal_closed_form(DEFAULT)
    assert rel < 1e-12 and abs(vertical_gap) > 0.05


def test_churn_freezes_liability():
    p = DEFAULT; n = 3
    exo = _exo(p, np.zeros(n), np.zeros(n), u=np.zeros(n))
    s = State.initial(p, exo, 1.0, 1.0, 5.0)
    s, _ = step(s, np.full(n, 0.5), exo, p, ALWAYS)          # pays year 0, then leaves
    assert not s.present.any() and np.all(s.leave_t == 1)
    L1 = s.L.copy()
    s, c = step(s, np.zeros(n), exo, p, ALWAYS)
    assert np.all(c == 0) and np.array_equal(s.L, L1)
