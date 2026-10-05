"""Every value function: the oracle optimises what simulate() scores."""
import numpy as np

import pension.dp as dp
from pension import checks
from pension.objective import OBJECTIVES


def test_baseline_is_the_committed_formula(p, grid):
    Fg, rg = grid["Fg"], grid["rg"]
    Fc, rc = Fg[:, None], rg[None, :]
    rr = p.RR_LEGAL + np.maximum(Fc, 1.0) / (rc * p.ANNUITY)
    ref = (p.LAMBDA * p.RR_TARGET * p.ANNUITY * dp.u(rr / p.RR_TARGET, p=p) * np.exp(-p.DISC_EMP * p.T)
           - (1 - p.LAMBDA) * np.maximum(1.0 - Fc, 0.0) / rc * np.exp(-p.DISC_ER * p.T))
    got = dp.terminal(Fg, rg, "baseline", p=p)
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-12


def test_each_objective_prefers_its_own_policy(p, grid, entry):
    Fg, rg, nq = grid["Fg"], grid["rg"], grid["nq"]
    lo, hi = 0.02 / p.GAMMA, 0.15 / p.GAMMA
    ag = np.linspace(lo, hi, grid["ag"].size)
    pols = {n: dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=nq, objective=n, p=p)["policy"] for n in OBJECTIVES}
    J, floor = checks.cross_scores(p, pols, Fg, rg, entry, 2000, 7, (lo, hi), lo)
    margin, _ = checks.diagonal_margin(J, floor)
    assert np.all(margin <= 0.02), dict(zip(OBJECTIVES, np.round(margin, 3)))
