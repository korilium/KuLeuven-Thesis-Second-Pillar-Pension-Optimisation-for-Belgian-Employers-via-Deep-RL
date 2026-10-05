"""Record the golden outputs the refactor must reproduce.

Run ONCE, on the code being refactored away from, before any stage starts:
    python tests/regression/make_golden.py
tests/test_regression.py then compares the current code against these files:
policies (and every action-valued output) exactly, values to 1e-12 relative.

Each case is (rate model, objective) on a small grid -- enough to exercise every
branch (constant / Hull-White / Vasicek rates, vertical / horizontal liability,
four value functions, churn, the band, tracking) in seconds.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [("constant", "baseline"), ("hull_white", "baseline"),
         ("vasicek", "employer_floor"), ("constant", "cashflow_strain")]
GRID = dict(nF=31, nR=25, na=7, nq=3)
SIM = dict(n_paths=3000, seed=7)
SIM_KEYS = ["benefit", "cost", "joint", "mean_a", "c_by", "frac", "RR_tot", "a_by_t", "rho_med"]
ACTION_KEYS = {"policy", "mean_a", "c_by", "a_by_t", "frac"}   # compared exactly


def path(model, objective):
    return os.path.join(HERE, "golden", f"{model}__{objective}.npz")


def run(dp, model, objective):
    """The outputs of one case, through the current API. (The files were recorded
    with the pre-refactor API, which set dp.RATE_MODEL as a module global; the
    case definitions -- and therefore the files -- are unchanged.)"""
    p = dp.DEFAULT.replace(RATE_MODEL=model, EMPLOYER_NUMERAIRE="discounted")   # the recorded spec
    Fg, rg = dp.make_F_grid(n=GRID["nF"]), dp.make_rho_grid(n=GRID["nR"])
    ag = dp.make_a_grid(n=GRID["na"])
    out = {}
    sol = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=GRID["nq"], objective=objective, p=p)
    out["V"], out["policy"] = sol["V"], sol["policy"]
    out["terminal"] = dp.terminal(Fg, rg, objective, p=p)
    out["Phi"] = dp.paidup_service(Fg, rg, objective, p=p)[::9]     # every 9th leave cohort
    lo, hi = 0.02 / p.GAMMA, 0.15 / p.GAMMA
    for name, pol, band in [("opt", sol["policy"], None),
                            ("flat", dp.const_policy(0.4, p.T, len(Fg), len(rg)), (lo, hi))]:
        s = dp.simulate(pol, Fg, rg, band=band, track=True, objective=objective, p=p, **SIM)
        for k in SIM_KEYS:
            out[f"{name}_{k}"] = np.asarray(s[k])
    return out


if __name__ == "__main__":
    import pension.dp as dp
    for model, objective in CASES:
        np.savez_compressed(path(model, objective), **run(dp, model, objective))
        print("recorded", model, objective)
