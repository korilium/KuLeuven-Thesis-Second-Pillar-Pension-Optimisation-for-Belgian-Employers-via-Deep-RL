"""The DP's reduced (F, rho) transitions are the SAME dynamics as pension.dynamics.

Scope: the DP stack (pension.dp / dynamics / economy) at constant rates, the
single-vehicle rung. Its rate mode is tested in tests/test_dp_rate_mode.py (= legacy_v1);
the baseline in tests/test_baseline.py.

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


def test_churn_freezes_liability():
    p = DEFAULT; n = 3
    exo = _exo(p, np.zeros(n), np.zeros(n), u=np.zeros(n))
    s = State.initial(p, exo, 1.0, 1.0, 5.0)
    s, _ = step(s, np.full(n, 0.5), exo, p, ALWAYS)          # pays year 0, then leaves
    assert not s.present.any() and np.all(s.leave_t == 1)
    L1 = s.L.copy()
    s, c = step(s, np.zeros(n), exo, p, ALWAYS)
    assert np.all(c == 0) and np.array_equal(s.L, L1)
