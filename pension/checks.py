"""
checks.py -- the known-answer checks of the model, as functions that MEASURE.

Each check returns the numbers it measures and passes no verdict. Two kinds of
caller share them:
  * tests/ asserts on them, on small grids, in seconds (pytest);
  * experiments/dynpro/*_suite.py prints them on the protocol grid, next to the
    figures they certify.
So a check is written once, and the suites and the test suite cannot drift apart.

`entry` is a dict(R0=..., L0=..., S0=...) of entry states (scalars or arrays),
as produced by experiments/dynpro/common.entry or dp.new_plan_init.
"""

import numpy as np

import pension.dp as dp
from pension.economy import draw_rate_scenarios
from pension.objective import OBJECTIVES


# --- structural invariants of the DP -------------------------------------------------
def scale_invariance(p, Fg, rg, ag, nq, n_paths, seed, k=5.0):
    """Max relative change in RR when (R, L, S) are all scaled by k. The model is
    homogeneous of degree 0, so this must be ~0."""
    pol = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=nq, p=p)["policy"]
    r1 = dp.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=20.0, n_paths=n_paths, seed=seed, p=p)
    rk = dp.simulate(pol, Fg, rg, R0=k, L0=k, S0=k * 20.0, n_paths=n_paths, seed=seed, p=p)
    return float(np.max(np.abs(rk["RR"] - r1["RR"]) / np.abs(r1["RR"])))


def lambda_equivalent(p, de_new, lam=None, de_ref=None):
    """The LAMBDA that reproduces (lam, de_new) at the reference DISC_EMP:
        lambda'' = A / (A + B*exp(-de_ref*T)),  A = lam*exp(-de_new*T), B = 1-lam.
    delta_e only multiplies the employee leg by exp(-delta_e*T), so it is a LAMBDA
    change up to a positive rescale of the objective."""
    lam = p.LAMBDA if lam is None else lam
    de_ref = p.DISC_EMP if de_ref is None else de_ref
    A = lam * np.exp(-de_new * p.T); B = 1.0 - lam
    return float(A / (A + B * np.exp(-de_ref * p.T)))


def lambda_reparam(p, Fg, rg, ag, nq, lam=0.5, de_new=0.02, objective=None):
    """Max |policy difference| between (lam, DISC_EMP=de_new) and its LAMBDA
    equivalent at the reference DISC_EMP -- must be ~0. Returns (gap, lam_eq)."""
    lam_eq = lambda_equivalent(p, de_new, lam=lam)
    pa = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=nq, objective=objective,
                  p=p.replace(LAMBDA=lam, DISC_EMP=de_new))["policy"]
    pb = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=nq, objective=objective,
                  p=p.replace(LAMBDA=lam_eq))["policy"]
    return float(np.abs(pa - pb).max()), lam_eq


def eta_log_limit(p, x=(0.4, 1.0, 2.5), eps=1e-4):
    """(|u - log| at eta=1, |u(eta=1) - u(eta=1+eps)|): u is log at eta=1 and
    continuous there."""
    x = np.asarray(x)
    u1 = dp.u(x, p=p.replace(ETA=1.0)); ue = dp.u(x, p=p.replace(ETA=1.0 + eps))
    return float(np.abs(u1 - np.log(x)).max()), float(np.abs(u1 - ue).max())


def leaver_terminal_gap(p, Fg, rg, objective=None):
    """Max |Phi[T] - terminal|: a leaver with full service IS a stayer. Must be 0."""
    return float(np.abs(dp.paidup_service(Fg, rg, objective, p=p)[p.T]
                        - dp.terminal(Fg, rg, objective, p=p)).max())


def lambda_monotonicity(p, Fg, rg, ag, nq, entry, n_paths, seed, lambdas=(0.2, 0.4, 0.6, 0.8)):
    """(employer cost, median stayer RR) at each LAMBDA; both must increase."""
    cost, sty = [], []
    for lam in lambdas:
        q = p.replace(LAMBDA=lam)
        pol = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=nq, p=q)["policy"]
        r = dp.simulate(pol, Fg, rg, **entry, n_paths=n_paths, seed=seed, p=q)
        cost.append(r["cost"]); sty.append(r["sty"])
    return cost, sty


def timing_neutrality(p, Fg, rg, nq, entry, n_paths, seed, band_pct=(0.02, 0.15), na=20):
    """(early, late) mean contribution (% of salary, years 0-9 and 35-44) with
    DISC_ER = DISC_EMP = MU: the benefit/cost ratio is then flat in t, so the
    schedule must be roughly level. Constant rates only."""
    lo, hi = band_pct[0] / p.GAMMA, band_pct[1] / p.GAMMA
    q = p.replace(DISC_ER=p.MU, DISC_EMP=p.MU)
    pol = dp.solve(Fg=Fg, rg=rg, ag=np.linspace(lo, hi, na), n_quad=nq, p=q)["policy"]
    r = dp.simulate(pol, Fg, rg, **entry, band=(lo, hi), n_paths=n_paths, seed=seed, p=q)
    return float(np.mean(r["c_by"][:10])), float(np.mean(r["c_by"][35:]))


def no_capacity_gap(p, Fg, rg, nq, entry, n_paths, seed, gamma=0.001, band_pct=(0.02, 0.15), na=20):
    """GAMMA -> 0: median stayer RR of the banded optimum minus that of paying
    nothing -- must be ~0 (contributions cannot add anything)."""
    q = p.replace(GAMMA=gamma)
    lo, hi = band_pct[0] / q.GAMMA, min(band_pct[1] / q.GAMMA, 1.0)
    lo = min(lo, hi)
    pol = dp.solve(Fg=Fg, rg=rg, ag=np.linspace(lo, hi, na), n_quad=nq, p=q)["policy"]
    kw = dict(**entry, n_paths=n_paths, seed=seed, p=q)
    sty = dp.simulate(pol, Fg, rg, band=(lo, hi), **kw)["sty"]
    zero = dp.simulate(dp.const_policy(0.0, q.T, len(Fg), len(rg)), Fg, rg, **kw)["sty"]
    return float(sty - zero)


# --- rates -------------------------------------------------------------------------
def degenerate_rates_gap(p, Fg, rg, entry, n_paths, seed, a=0.4):
    """A rate scenario with G_t = G, mu_t = MU (SIGMA_L = 0) against the constant
    model: max gap in joint value and RR -- must be ~0."""
    q = p.replace(RATE_MODEL="constant", SIGMA_L=0.0, SIGMA_R_RATES=p.SIGMA_R)
    pol = dp.const_policy(a, q.T, len(Fg), len(rg))
    kw = dict(**entry, n_paths=n_paths, seed=seed, p=q)
    ref = dp.simulate(pol, Fg, rg, **kw)
    deg = dp.simulate(pol, Fg, rg, rates=draw_rate_scenarios(n_paths, model="constant", p=q), **kw)
    return max(abs(ref["joint"] - deg["joint"]), float(np.abs(ref["RR_tot"] - deg["RR_tot"]).max()))


def horizontal_closed_form(p, G_before=0.03, G_after=0.0175, switch=20, a=0.4, S0=20.0):
    """One career, no churn, no shocks, with G stepping G_before -> G_after at
    year `switch`. Returns (relative error of the horizontal ledger against
    L_T = L0 e^{G_0 T} + sum_s c_s e^{G_s (T-s)}, and how far the VERTICAL method
    -- every euro at the current rate -- would land from it)."""
    from pension.dynamics import Exogenous, State, step
    T = p.T
    Gs = np.where(np.arange(T) < switch, G_before, G_after)
    exo = Exogenous(np.zeros((T, 1)), np.zeros((T, 1)), np.ones((T, 1)),
                    Gs[:, None], np.full((T, 1), 0.02))
    s = State.initial(p, exo, R0=1.0, L0=1.0, S0=S0)
    cs = []
    for _ in range(T):
        s, c = step(s, np.array([a]), exo, p, lambda t: 0.0)
        cs.append(c[0])
    cs = np.array(cs)
    closed = np.exp(Gs[0] * T) + np.sum(cs * np.exp(Gs * (T - np.arange(T))))
    vertical = 1.0
    for t in range(T):
        vertical = (vertical + cs[t]) * np.exp(Gs[t])
    return float(abs(s.L[0] / closed - 1)), float(vertical / closed - 1)


def wap_scenario_stats(p, model, n=2000):
    """min, max, distance from the 25 bp grid, and G_0 of the WAP rates of n
    scenarios of `model`."""
    G = draw_rate_scenarios(n, model=model, p=p)["G"]
    return dict(min=float(G.min()), max=float(G.max()),
                grid_err=float(np.abs(G / 0.0025 - np.round(G / 0.0025)).max()),
                G0=float(G[0, 0]), G0_spread=float(np.ptp(G[0])))


FSMA_PUBLISHED = {2016: 1.75, 2017: 1.75, 2018: 1.75, 2019: 1.75, 2020: 1.75, 2021: 1.75,
                  2022: 1.75, 2023: 1.75, 2024: 1.75, 2025: 2.50, 2026: 2.50}


def wap_vs_fsma(years=FSMA_PUBLISHED):
    """{year: (WAP rate from the statutory formula on the cached NBB data, rate
    published by the FSMA)}, in %. 2027 is left out on purpose: the formula gives
    2.75% against 2.50% published (see pension/rates/wap.py)."""
    import pandas as pd
    from pension.rates.data import load_olo
    from pension.rates.wap import computeWAPRate
    df10Y, _, _ = load_olo()
    y = df10Y["YIELD"].values / 100.0; dates = list(df10Y["DATE"])
    out = {}
    for yr, pub in years.items():
        row = dates.index(pd.Timestamp(f"{yr}-01-01"))
        out[yr] = (float(computeWAPRate(y[:, None], row, 1)[0, 0] * 100), pub)
    return out


# --- objectives ----------------------------------------------------------------------
def cross_scores(p, policies, Fg, rg, entry, n_paths, seed, band, floor_a):
    """J[i, j]: the policy optimised under objective i (policies: {name: policy}),
    scored under objective j; and floor[j], the score of the constant floor_a
    policy. An objective's own policy must be best in its column."""
    names = list(policies)
    kw = dict(**entry, band=band, n_paths=n_paths, seed=seed, p=p)
    J = np.array([[dp.simulate(policies[i], Fg, rg, objective=j, **kw)["joint"] for j in names]
                  for i in names])
    floor_pol = dp.const_policy(floor_a, p.T, len(Fg), len(rg))
    floor = np.array([dp.simulate(floor_pol, Fg, rg, objective=j, **kw)["joint"] for j in names])
    return J, floor


def diagonal_margin(J, floor):
    """Per column j: how far the best OTHER policy is above j's own optimum, as a
    fraction of j's gain over the floor (<= 0 means the diagonal wins). Returns
    (margins, index of that best other policy)."""
    scale = np.abs(np.diag(J) - floor)
    rel = (J - np.diag(J)[None, :]) / np.where(scale > 0, scale, 1.0)[None, :]
    np.fill_diagonal(rel, -np.inf)
    return rel.max(axis=0), rel.argmax(axis=0)


ALL_OBJECTIVES = list(OBJECTIVES)
