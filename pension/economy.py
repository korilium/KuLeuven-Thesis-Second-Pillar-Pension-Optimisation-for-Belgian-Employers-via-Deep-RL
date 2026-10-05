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
        from pension.rates.data import load_olo
        from pension.rates.calibration import calibrateVasicek, bootstrapForwardCurve
        df10Y, maturities, yields = load_olo()
        _CALIBRATION = dict(
            vasicek=calibrateVasicek(df10Y, verbose=False),
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
    model = "vasicek":    the Vasicek process IS the 10Y yield (that is the series it
            was calibrated on), so it is used directly; no curve fit.
    model = "constant":   G and MU broadcast -- the degenerate scenario of the
            original model (used for known-answer checks).

    Returns dict of (horizon, n_paths) arrays G, mu and (horizon+1, n_paths) y10, r
    (annual samples; r is the short rate, = y10 under Vasicek), plus model, t0.

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
          ((p.G, p.MU) if model == "constant" else ())
    if key not in _SCENARIOS:
        _SCENARIOS[key] = _draw_rate_scenarios(n_paths, seed, model, H, p)
    return _SCENARIOS[key]

_SCENARIOS = {}


def _draw_rate_scenarios(n_paths, seed, model, H, p, chunk=10000):
    """Uncached draw (see draw_rate_scenarios). Paths are simulated in blocks of
    `chunk` so a 40k-path cohort does not hold ~1 GB of monthly arrays at once."""
    if model == "constant":
        G_ = np.full((H, n_paths), p.G); mu_ = np.full((H, n_paths), p.MU)
        return dict(model=model, G=G_, mu=mu_, y10=np.full((H + 1, n_paths), np.nan),
                    r=np.full((H + 1, n_paths), np.nan), t0=None)

    from pension.rates.simulation import simulateVasicek, simulateHullWhite
    from pension.rates.pricing import reconstructFutureYield
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
            r_m = simulateVasicek(vas["kappa"], vas["theta"], vas["sigma"], vas["r0"], T=H,
                                  n_paths=n, dt=p.RATE_DT, rng=rng)
            y10_m = r_m
        else:
            raise ValueError(f"unknown RATE_MODEL {model!r}")

        # observed months up to t0, then simulated months 1..12H (row 0 of the
        # simulation is t0 itself, which the history already holds as an observation)
        full = np.vstack([np.repeat(hist[:, None], n, axis=1), y10_m[1:]])
        G_ = computeWAPRate(full, start, H, months_per_year=mpy)
        cs = np.vstack([np.zeros((1, n)), np.cumsum(full, axis=0)])
        mu_ = (cs[rows + 1] - cs[rows + 1 - n_book]) / n_book + p.BOOK_SPREAD
        parts.append((G_, mu_, y10_m[yearly], r_m[yearly]))
    G_, mu_, y10_, r_ = (np.hstack(x) for x in zip(*parts))
    return dict(model=model, G=G_, mu=mu_, y10=y10_, r=r_, t0=cal["t0"])
