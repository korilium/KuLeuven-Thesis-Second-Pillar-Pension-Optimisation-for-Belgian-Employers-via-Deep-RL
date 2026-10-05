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
    equivalent at the reference DISC_EMP -- must be ~0. Returns (gap, lam_eq).
    A statement about the DISCOUNTED objective (DISC_EMP does not enter the
    retirement numeraire), so it is evaluated there."""
    p = p.replace(EMPLOYER_NUMERAIRE="discounted")
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
    """(early, late) mean contribution (% of salary, years 0-9 and 35-44) when a
    contribution's cost and value grow alike: SHORT_RATE = MU = G under the
    retirement numeraire (premiums accrue at the rate R and L earn), DISC_ER =
    DISC_EMP = MU under the discounted one. The benefit/cost ratio is then flat in
    t, so the schedule must be roughly level. Constant rates only."""
    lo, hi = band_pct[0] / p.GAMMA, band_pct[1] / p.GAMMA
    q = p.replace(SHORT_RATE=p.MU, G=p.MU) if p.EMPLOYER_NUMERAIRE == "retirement" \
        else p.replace(DISC_ER=p.MU, DISC_EMP=p.MU)
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


def diagonal_margin(J, floor, relative_to="gain"):
    """Per column j: how far the best OTHER policy is above j's own optimum
    (<= 0 means the diagonal wins), as a fraction of j's gain over the floor
    (relative_to="gain") or of |J_jj| ("value"). The gain normalisation is
    ill-conditioned when an objective barely beats the floor (it amplifies grid
    error); the value normalisation is not. Returns (margins, best other index)."""
    scale = np.abs(np.diag(J) - floor) if relative_to == "gain" else np.abs(np.diag(J))
    rel = (J - np.diag(J)[None, :]) / np.where(scale > 0, scale, 1.0)[None, :]
    np.fill_diagonal(rel, -np.inf)
    return rel.max(axis=0), rel.argmax(axis=0)


ALL_OBJECTIVES = list(OBJECTIVES)


# --- the real-world short rate (RATE_MODEL = "vasicek_short") ---------------------------
def vasicek_short_fit(p=None):
    """Calibration and fit of the P-measure Vasicek short rate: the maturity used
    as short-rate proxy, kappa, theta_P (OLS, and the one simulated: p.LONG_RATE_P
    if set), sigma, the significance of the mean reversion, theta_Q and phi, the
    in-sample fit of the reconstructed 10Y from the observed short rate (RMSE,
    bias -- zero by construction --, correlation, RMSE over the last 24 months),
    the jump at t0, and the AR(1) decay of the t0 offset (phi_e, its standard
    error, half-life, and whether the kappa fallback is used)."""
    from pension.economy import rate_calibration, vasicek_short_params
    from pension.params import DEFAULT
    sh = rate_calibration()["short"]
    vp = vasicek_short_params(DEFAULT if p is None else p)
    d = sh["diagnostics"]
    return dict(maturity=sh["maturity"], kappa=sh["kappa"], theta_P_ols=sh["theta_P"],
                theta_P=vp["theta_P"], sigma=sh["sigma"], t_stat_b=d["t_stat_b"],
                p_val_b=d["p_val_b"], half_life=d["half_life_years"], theta_Q=sh["theta_Q"],
                phi=vp["phi"], r0=sh["r0"], **sh["fit"],
                **{"offset_" + k: v for k, v in sh["offset"].items()})


def scenario_stats(p, model, n=2000):
    """Distribution of the annual short rate r_t, the 10Y, G_t, mu_t and the
    realised accrual of `model`'s scenarios at a few years: dict year -> stats."""
    sc = draw_rate_scenarios(n, model=model, p=p)
    out = {}
    for t in (0, 1, 5, 10, 20, p.T - 1):
        q = lambda x: (float(np.percentile(x, 5)), float(np.median(x)), float(np.percentile(x, 95)))
        out[t] = dict(r=q(sc["r"][t]), y10=q(sc["y10"][t]), G=q(sc["G"][t]), mu=q(sc["mu"][t]),
                      acc=q(sc["acc"][t]), r_neg=float((sc["r"][t] < 0).mean()))
    return out


# --- Hull-White under the real-world measure (RATE_MODEL = "hull_white_p") --------------
def hw_p_link(n=200):
    """(a) With phi = 0 and the legacy sigma, hull_white_p's simulation IS
    hull_white's: max |difference| of the monthly short rate and 10Y (must be 0),
    and of the annual r against the hull_white scenarios (same seed)."""
    from pension.economy import hull_white_p_params, hull_white_paths, rate_calibration
    from pension.params import DEFAULT
    from pension.rates.pricing import reconstructFutureYield
    from pension.rates.simulation import simulateHullWhite
    p = DEFAULT
    vas, curve = rate_calibration()["vasicek"], rate_calibration()["curve"]
    r_q = simulateHullWhite(curve, vas["kappa"], vas["sigma"], T=p.T, n_paths=n, dt=p.RATE_DT,
                            rng=np.random.default_rng(p.RATE_SEED))
    y_q = reconstructFutureYield(r_q, vas["kappa"], vas["sigma"], curve, tau=10.0, dt=p.RATE_DT)
    r_p, y_p = hull_white_paths(hull_white_p_params(p, legacy_sigma=True, phi=0.0), p.T, n,
                                p.RATE_DT, np.random.default_rng(p.RATE_SEED))
    sc = draw_rate_scenarios(n, model="hull_white", p=p)
    yearly = np.arange(p.T + 1) * int(round(1 / p.RATE_DT))
    return dict(r=float(np.abs(r_q - r_p).max()), y10=float(np.abs(y_q - y_p).max()),
                scenario_r=float(np.abs(sc["r"] - r_p[yearly]).max()))


def hw_p_t0_fit(p):
    """(b) The model 10Y at t0 against the NSS 10Y (equal: exact fit to today's
    curve) and against the last observed 10Y (the NSS fitting error)."""
    from pension.economy import hull_white_p_params, rate_calibration
    from pension.rates.calibration import nss_yield
    from pension.rates.data import load_olo
    hp = hull_white_p_params(p)
    sc = draw_rate_scenarios(50, model="hull_white_p", p=p)
    df10Y, mats, ylds = load_olo()
    prm = rate_calibration()["curve"]["params"]
    nss10 = float(nss_yield(10.0, *prm))
    rmse = float(np.sqrt(np.mean((nss_yield(mats.astype(float), *prm) - ylds / 100) ** 2)))
    return dict(model_10y_t0=float(sc["y10"][0, 0]), nss_10y=nss10,
                observed_10y=float(df10Y["YIELD"].iloc[-1] / 100), nss_rmse=rmse,
                r0=float(sc["r"][0, 0]), f00=float(prm[0] + prm[1]), phi=hp["phi"])


def hw_p_drift(p, n=4000, years=(1, 5, 10, 20, 30, 44)):
    """(c) Per year: the scenario mean of r_t (with its standard error) against
    E^P[r_t] = alpha(t) + m (1 - e^{-kappa t})  (E^Q[r_t] = alpha(t))."""
    from pension.economy import hull_white_p_params
    from pension.rates.accrual import hull_white_alpha
    hp = hull_white_p_params(p)
    r = draw_rate_scenarios(n, model="hull_white_p", p=p)["r"]
    out = {}
    for t in years:
        an = float(hull_white_alpha(t, hp["kappa"], hp["sigma"], hp["curve"])
                   + hp["m"] * (1 - np.exp(-hp["kappa"] * t)))
        out[t] = dict(mc=float(r[t].mean()), se=float(r[t].std(ddof=1) / np.sqrt(n)), analytic=an)
    return out


def vol_target(p, n=500, years=10):
    """(d) Std of the model's monthly 10Y changes (first `years` years, pooled)
    against the historical std of monthly 10Y changes; for hull_white_p and for
    hull_white (legacy sigma). Also B(10)/10, hull_white's understatement factor."""
    from pension.economy import hull_white_p_params, hull_white_paths, rate_calibration
    hist = rate_calibration()["hist10Y"]
    out = dict(historical=float(np.std(np.diff(hist), ddof=1)))
    for name, kw in (("hull_white_p", {}), ("hull_white", dict(legacy_sigma=True, phi=0.0))):
        hp = hull_white_p_params(p, **kw)
        _, y = hull_white_paths(hp, years, n, p.RATE_DT, np.random.default_rng(1))
        out[name] = float(np.std(np.diff(y, axis=0), ddof=1))
    out["B10_over_10"] = hull_white_p_params(p)["B10"] / 10.0
    return out


def accrual_accuracy(p, n=4000, years=(0, 5, 10, 20, 30, 40, 44)):
    """(e) Per payment year t: the Monte Carlo mean of the realised accrual to T,
    prod_{u >= t} acc_u (monthly r * dt sums), against the mean of the closed-form
    conditional A(t, T; r_t) over the same paths (tower property). Reported as the
    relative error of the paired difference and its standard error."""
    from pension.rates.accrual import closed_form_accrual
    sc = draw_rate_scenarios(n, p=p)
    out = {}
    for t in years:
        realised = np.prod(sc["acc"][t:p.T], axis=0)
        closed = closed_form_accrual(t, sc["r"][t], p)
        d = realised - closed
        out[t] = dict(closed=float(closed.mean()), realised=float(realised.mean()),
                      rel_err=float(d.mean() / closed.mean()),
                      rel_se=float(d.std(ddof=1) / np.sqrt(n) / closed.mean()))
    return out
