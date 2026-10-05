"""
economy.py -- the economic scenario (exogenous world).
 
Shared by the tabular rung and the DP oracle. Holds:
  * the parameter names of the tabular rung (values from pension/params.py),
  * the contribution plan rules c(t, S),
  * the exogenous credited-return shock process,
  * the interest-rate regime switch and its scenarios (G_t, mu_t).
 
No episode stepping, no reward accounting, no learning, no benchmarks --
those consume these primitives downstream. No module-level internal imports:
the rate engine (pension.rates) is imported lazily, only when a stochastic
rate model is actually drawn.
"""
 
import numpy as np
 
# --- parameters -----------------------------------------------------------
# All parameters live in pension/params.py (Params / DEFAULT). The names below
# are module-level copies for the TABULAR rung (pension/envs/tabular.py), which
# reads them as globals and rebinds them per config; the DP and the rate engine
# take an explicit Params instead.
from pension.params import DEFAULT

T, G, MU, W, S0 = DEFAULT.T, DEFAULT.G, DEFAULT.MU, DEFAULT.W, DEFAULT.S0
LAMBDA, DISC, SIGMA, N_EVAL = DEFAULT.LAMBDA, DEFAULT.DISC, DEFAULT.SIGMA, DEFAULT.N_EVAL

 
# --- plan rules: c(t, S) -> premium --------------------------------------
def plan_fixed(rate=0.05):
    return lambda t, S: rate * S
 
def plan_step(rate_low=0.04, rate_high=0.10, ceiling=1.5):
    # tranche-based, like real plans split around the pension ceiling
    return lambda t, S: rate_low * min(S, ceiling) + rate_high * max(S - ceiling, 0.0)
 
def plan_age(rate0=0.03, step=0.01, band=10):
    # banded age scale; keep step within the WAP non-discrimination bound
    return lambda t, S: (rate0 + step * (t // band)) * S
 
 
# --- exogenous shocks -----------------------------------------------------
def draw_shock_batch(n_paths=N_EVAL, seed=12345):
    """One frozen batch of noise paths (SAA + common random numbers).
    None at SIGMA = 0 -> deterministic single-episode evaluation."""
    if SIGMA == 0.0:
        return None
    return np.random.default_rng(seed).standard_normal((n_paths, T))
 
batch = draw_shock_batch()


# --- interest-rate scenarios ----------------------------------------------
_CALIBRATION = None

def rate_calibration():
    """Calibrate once per process, from the cached NBB OLO data: Vasicek (kappa,
    sigma, theta, r0) on the monthly 10Y history and the NSS curve on the latest
    cross-section. The rate engine (pension.rates) is imported HERE, not at module
    level, so the constant-rate model never loads it."""
    global _CALIBRATION
    if _CALIBRATION is None:
        from pension.rates.data import load_olo, load_olo_short
        from pension.rates.calibration import (calibrateVasicek, bootstrapForwardCurve,
                                               calibrateVasicekShort)
        df10Y, maturities, yields = load_olo()
        short_label, df_short = load_olo_short()
        _CALIBRATION = dict(
            vasicek=calibrateVasicek(df10Y, verbose=False),
            short=dict(calibrateVasicekShort(df_short, df10Y), maturity=short_label),
            curve=bootstrapForwardCurve(maturities, yields, verbose=False),
            hist10Y=df10Y["YIELD"].values / 100.0,      # monthly, oldest first; last = model t0
            t0=df10Y["DATE"].iloc[-1],
        )
    return _CALIBRATION


def draw_rate_scenarios(n_paths=2000, seed=None, model=None, horizon=None, p=None):
    """Annual rate scenarios for the reserve/liability simulation.

    Simulates the rate model monthly from the last observed OLO month (model year
    0), reads off the 10Y OLO per path, and turns it into the two rates the
    pension contract uses:
        G[t]  -- the WAP guarantee rate fixed at the start of year t,
        mu[t] -- the book yield credited during year t.
    Observed history is prepended, so G[0] is today's actual WAP fixing and the
    book yield starts from the real portfolio, not from a model curve.

    model = "hull_white": short rate fitted to today's OLO curve; the 10Y is
            reconstructed per path from the Hull-White bond price.
    model = "vasicek":    the CANONICAL stochastic model. The Vasicek process IS the
            10Y yield, calibrated on its history under P (theta = p.LONG_RATE_P if
            set); the short rate behind the accrual is r = y10 - spread
            (vasicek_params). No curve fit.
    model = "vasicek_short": a real-world (P) Vasicek SHORT rate calibrated on the
            shortest OLO maturity (rates.calibration.calibrateVasicekShort); the
            10Y is the affine Vasicek yield at theta_Q (phi is used only there).
            Its 10Y carries a deterministic decaying offset that starts it on the
            observed 10Y (see calibrateVasicekShort). Robustness model: purely
            historical, no information from today's curve.
    model = "hull_white_p": Hull-White fitted to today's curve, simulated under the
            REAL-WORLD measure with a constant market price of risk set by
            p.LONG_RATE_P (None: phi = 0), and a short-rate sigma consistent with
            the 10Y volatility (hull_white_p_params). Primary real-world model.
    model = "constant":   G and MU broadcast -- the degenerate scenario of the
            original model (used for known-answer checks).

    Returns dict of (horizon, n_paths) arrays G, mu and (horizon+1, n_paths) y10, r
    (annual samples at the start of each year; r is the model's short rate, = y10
    under "vasicek"), plus model, t0. The stochastic models also return acc
    (horizon, n_paths): the realised accrual of each year, exp(sum of monthly r*dt).

    `p` (default DEFAULT) supplies everything not given explicitly: seed=None ->
    p.RATE_SEED, model=None -> p.RATE_MODEL, horizon=None -> p.T, and the book-yield
    and constant-mode parameters.
    """
    p = DEFAULT if p is None else p
    model = p.RATE_MODEL if model is None else model
    seed = p.RATE_SEED if seed is None else seed
    H = p.T if horizon is None else horizon
    # memoised: the suites call this once per simulate(), hundreds of times per run.
    # The key holds every input the scenario depends on (the WAP constants are
    # module constants of pension/rates/wap.py and are not swept). Callers must
    # treat the returned arrays as read-only.
    key = (model, n_paths, seed, H, p.RATE_DT, p.BOOK_DURATION, p.BOOK_SPREAD) + \
          ((p.G, p.MU, p.SHORT_RATE) if model == "constant" else ()) + \
          ((p.LONG_RATE_P,) if model in ("vasicek", "vasicek_short", "hull_white_p") else ()) + \
          ((p.SPREAD_10Y_SHORT,) if model == "vasicek" else ())
    if key not in _SCENARIOS:
        _SCENARIOS[key] = _draw_rate_scenarios(n_paths, seed, model, H, p)
    return _SCENARIOS[key]

_SCENARIOS = {}


def vasicek_params(p=None):
    """The parameters of the canonical "vasicek" model: kappa, sigma of the 10Y
    (OLS), theta = p.LONG_RATE_P if set (else the OLS estimate), y0 = the last
    observed 10Y, and the spread to the short rate, r = y10 - spread
    (p.SPREAD_10Y_SHORT, or the historical mean 10Y - 1Y if None)."""
    p = DEFAULT if p is None else p
    cal = rate_calibration(); vas = cal["vasicek"]
    spread = cal["short"]["fit"]["spread_obs"] if p.SPREAD_10Y_SHORT is None else p.SPREAD_10Y_SHORT
    return dict(kappa=vas["kappa"], sigma=vas["sigma"],
                theta=vas["theta"] if p.LONG_RATE_P is None else p.LONG_RATE_P,
                y0=vas["r0"], spread=spread)


def vasicek_short_params(p=None):
    """The parameters vasicek_short simulates with: kappa, sigma, theta_Q and the
    t0 offset as calibrated; theta_P = p.LONG_RATE_P if set (else the OLS
    estimate), with phi = (theta_P - theta_Q) kappa / sigma recomputed from it --
    theta_Q, which reproduces the historical mean (10Y - short) spread, does not
    depend on theta_P."""
    p = DEFAULT if p is None else p
    sh = rate_calibration()["short"]
    theta_P = sh["theta_P"] if p.LONG_RATE_P is None else p.LONG_RATE_P
    return dict(kappa=sh["kappa"], sigma=sh["sigma"], theta_P=theta_P, theta_Q=sh["theta_Q"],
                phi=(theta_P - sh["theta_Q"]) * sh["kappa"] / sh["sigma"], r0=sh["r0"],
                tau=sh["tau"], e_t0=sh["offset"]["e_t0"],
                monthly_decay=sh["offset"]["monthly_decay"])


def hull_white_p_params(p=None, legacy_sigma=False, phi=None):
    """The parameters hull_white_p simulates with.

    kappa: from the 10Y OLS (in a one-factor affine model the 10Y reverts at the
    same speed). sigma: the INSTANTANEOUS short-rate volatility consistent with the
    10Y's, sigma_10Y * 10 / B(10) -- hull_white uses sigma_10Y itself, which
    understates the model's 10Y volatility by B(10)/10 (legacy_sigma=True gives
    that, for the link check). phi: from p.LONG_RATE_P, the long-run real-world
    short rate, against the long-run Q level beta0 + sigma^2/(2 kappa^2):
        phi = (kappa / sigma) (LONG_RATE_P - beta0 - sigma^2 / (2 kappa^2)),
    0 if LONG_RATE_P is None (expectations hypothesis); m = sigma phi / kappa is
    the level x reverts to under P. An explicit `phi` overrides."""
    p = DEFAULT if p is None else p
    cal = rate_calibration(); vas = cal["vasicek"]; curve = cal["curve"]
    kappa, sigma_10 = vas["kappa"], vas["sigma"]
    B10 = (1 - np.exp(-kappa * 10.0)) / kappa
    sigma = sigma_10 if legacy_sigma else sigma_10 * 10.0 / B10
    beta0 = curve["params"][0]
    if phi is None:
        phi = 0.0 if p.LONG_RATE_P is None else \
            (kappa / sigma) * (p.LONG_RATE_P - (beta0 + sigma**2 / (2 * kappa**2)))
    return dict(kappa=kappa, sigma=sigma, sigma_10Y=sigma_10, B10=B10, phi=phi,
                m=sigma * phi / kappa, beta0=beta0, curve=curve)


def hull_white_paths(hp, H, n, dt, rng):
    """Monthly short-rate paths and the repriced 10Y for Hull-White parameters hp
    (shared by hull_white and hull_white_p: only kappa, sigma and phi differ)."""
    from pension.rates.simulation import simulateHullWhite
    from pension.rates.pricing import reconstructFutureYield
    r_m = simulateHullWhite(hp["curve"], hp["kappa"], hp["sigma"], T=H, n_paths=n, dt=dt,
                            rng=rng, phi=hp["phi"])
    return r_m, reconstructFutureYield(r_m, hp["kappa"], hp["sigma"], hp["curve"], tau=10.0, dt=dt)


def _draw_rate_scenarios(n_paths, seed, model, H, p, chunk=10000):
    """Uncached draw (see draw_rate_scenarios). Paths are simulated in blocks of
    `chunk` so a 40k-path cohort does not hold ~1 GB of monthly arrays at once."""
    if model == "constant":
        G_ = np.full((H, n_paths), p.G); mu_ = np.full((H, n_paths), p.MU)
        return dict(model=model, G=G_, mu=mu_, y10=np.full((H + 1, n_paths), np.nan),
                    r=np.full((H + 1, n_paths), p.SHORT_RATE),
                    acc=np.full((H, n_paths), np.exp(p.SHORT_RATE)), t0=None)

    from pension.rates.simulation import simulateVasicek, simulateHullWhite
    from pension.rates.pricing import reconstructFutureYield, vasicekBondPrice
    from pension.rates.wap import computeWAPRate
    cal = rate_calibration(); vas = cal["vasicek"]
    rng = np.random.default_rng(seed)
    mpy = int(round(1 / p.RATE_DT))
    hist = cal["hist10Y"]
    start = len(hist) - 1                                # row of model year 0
    n_book = p.BOOK_DURATION * mpy
    assert start + 1 >= n_book, "OLO history shorter than BOOK_DURATION"
    rows = start + mpy * np.arange(H)                    # start month of each year
    yearly = np.arange(H + 1) * mpy

    parts = []
    for lo in range(0, n_paths, chunk):
        n = min(chunk, n_paths - lo)
        if model == "hull_white":
            r_m = simulateHullWhite(cal["curve"], vas["kappa"], vas["sigma"], T=H,
                                    n_paths=n, dt=p.RATE_DT, rng=rng)
            y10_m = reconstructFutureYield(r_m, vas["kappa"], vas["sigma"], cal["curve"],
                                           tau=10.0, dt=p.RATE_DT)
        elif model == "vasicek":
            vp = vasicek_params(p)
            y10_m = simulateVasicek(vp["kappa"], vp["theta"], vp["sigma"], vp["y0"], T=H,
                                    n_paths=n, dt=p.RATE_DT, rng=rng)
            r_m = y10_m - vp["spread"]                   # the short rate behind the accrual
        elif model == "vasicek_short":
            vs = vasicek_short_params(p)
            r_m = simulateVasicek(vs["kappa"], vs["theta_P"], vs["sigma"], vs["r0"], T=H,
                                  n_paths=n, dt=p.RATE_DT, rng=rng)
            off = vs["e_t0"] * vs["monthly_decay"] ** np.arange(r_m.shape[0])
            y10_m = -np.log(vasicekBondPrice(r_m, vs["kappa"], vs["theta_Q"], vs["sigma"],
                                             vs["tau"])) / vs["tau"] + off[:, None]
        elif model == "hull_white_p":
            r_m, y10_m = hull_white_paths(hull_white_p_params(p), H, n, p.RATE_DT, rng)
        else:
            raise ValueError(f"unknown RATE_MODEL {model!r}")

        # observed months up to t0, then simulated months 1..12H (row 0 of the
        # simulation is t0 itself, which the history already holds as an observation)
        full = np.vstack([np.repeat(hist[:, None], n, axis=1), y10_m[1:]])
        G_ = computeWAPRate(full, start, H, months_per_year=mpy)
        cs = np.vstack([np.zeros((1, n)), np.cumsum(full, axis=0)])
        mu_ = (cs[rows + 1] - cs[rows + 1 - n_book]) / n_book + p.BOOK_SPREAD
        # realised accrual of each model year: exp(sum of the 12 monthly r * dt)
        acc_ = np.exp(np.add.reduceat(r_m[:-1] * p.RATE_DT, np.arange(0, H * mpy, mpy), axis=0))
        parts.append((G_, mu_, y10_m[yearly], r_m[yearly], acc_))
    G_, mu_, y10_, r_, acc_ = (np.hstack(x) for x in zip(*parts))
    return dict(model=model, G=G_, mu=mu_, y10=y10_, r=r_, acc=acc_, t0=cal["t0"])
