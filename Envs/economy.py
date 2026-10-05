"""
economy.py -- the economic scenario (exogenous world).
 
Single source of truth for the pension economy shared by BOTH the tabular
environment (coreEnv) and the DP oracle (dp_oracle). Holds only:
  * structural / economic parameters,
  * the contribution plan rules c(t, S),
  * the exogenous credited-return shock process,
  * the interest-rate regime switch and its scenarios (G_t, mu_t).
 
No episode stepping, no reward accounting, no learning, no benchmarks --
those consume these primitives downstream. No module-level internal imports:
the rate engine (olo/, liability/) is imported lazily, only when a stochastic
rate model is actually drawn.
"""
 
import numpy as np
 
# --- parameters -----------------------------------------------------------
# THE single source of truth for the economic calibration: both the tabular
# environment (basicEnv) and the DP oracle (DynPro) import from here, so the two
# rungs are guaranteed to run the same economy.
#
# Rung 1 previously carried an older vintage of its own (G=1.75%, the earlier WAP
# floor; MU=1%; a single 1% numeraire). It now inherits the committed calibration
# below, so its drift gap MU-G moves from -0.75% to 0 and its discount from 1% to
# the employer rate.
T = 45                      # career length in years
G, MU, W = 0.03, 0.03, 0.025   # WAP guarantee rate, credited tariff, salary growth

DISC_EMP = 0.03    # EMPLOYEE discount: values future retirement income at the risk-free/OLO rate
DISC_ER  = 0.05     # EMPLOYER discount: firm cost of capital (contributions + shortfall)
DISC = DISC_ER      # single numeraire, used by the tabular rung and as a legacy alias

SIGMA_R, SIGMA_L = 0.05, 0.02   # asset shock, guarantee shock
SIGMA = SIGMA_R     # the tabular rung has ONE shock, on the reserve

GAMMA, LAMBDA, S0 = 0.15, 0.5, 1.0
ETA = 2.0                   # CRRA curvature over the replacement rate (eta != 1)
ANNUITY = 15.0              # actuarial annuity factor: capital -> annual pension
                            # (Belgian life expectancy at 65 ~20y, ~2% technical rate,
                            #  mortality-adjusted). RR is ANNUAL: pension / final salary.
RR_LEGAL = 0.43             # 1st-pillar (legal) gross replacement, Belgian private-sector
                            # average earner (OECD PaaG); the 2nd pillar sits ON TOP.
SATIATE = False             # if True, cap RR at RR_TARGET in the utility (no reward for overshoot)
RR_TARGET = 0.70            # total-adequacy target across all pillars (OECD/EU ~70%);
                            # the employee's lifecycle utility is judged against this.
# GAMMA/ETA/ANNUITY/RR_* and the DISC_EMP/DISC_ER split are Rung-2 concepts; the
# binary-action tabular model simply does not read them.

BETA = 0.01      # policy-EXTRACTION temperature, RELATIVE to the local Q-spread:
                # 0 = hard argmax (the true optimum); > 0 = soft signal readout that blends
                # actions lying within BETA of the state's full value range
                # (max_a Q - min_a Q). Relative, not in Q units, because Q is NOT scale-free
                # in the reward: rescaling the employee leg changes every Q-difference, so a
                # fixed absolute temperature would mean something different per specification.
                # This form is invariant to any affine rescale Q -> alpha*Q + c.
                # Rough ladder: 0.01 barely smooths, 0.1 visible, 0.3 strong, ->inf uniform.
                # Extraction only: the objective and the value function V are IDENTICAL at
                # every BETA, so the value gap to a soft policy stays meaningful.

N_EVAL = 2000   # default number of evaluation paths (SAA batch) -- numerical, not economic

# --- interest-rate regime -------------------------------------------------
# RATE_MODEL is THE switch between the constant-rate economy above and the
# stochastic-rate one. "constant" (default) is the original model, untouched: G,
# MU, SIGMA_R and SIGMA_L as set above. "hull_white" / "vasicek" replace G and MU
# by path-wise rates driven by ONE simulated 10Y OLO path:
#   G_t  = the statutory WAP filter of that path (liability/WAP.py), applied
#          HORIZONTALLY: each contribution keeps the G_t of its payment year;
#   mu_t = the insurer's book yield, a rolling mean of the same 10Y OLO over
#          BOOK_DURATION years plus BOOK_SPREAD (a Branch 21 portfolio rolls over
#          slowly, so its yield lags the market).
# The common driver is what makes the guarantee and the reserve correlated.
# Consumers rebind RATE_MODEL on their own module (e.g. setattr(dp, "RATE_MODEL",
# "hull_white")), like every other parameter here.
RATE_MODEL = "constant"     # "constant" | "hull_white" | "vasicek"
RATE_DT = 1 / 12            # monthly rate step -- matches the OLO calibration
RATE_SEED = 2026            # seed of the rate scenarios (separate from the churn/asset noise)
BOOK_DURATION = 8           # years averaged into the book yield (Branch 21 portfolio duration)
BOOK_SPREAD = 0.0           # book yield over the 10Y OLO (credit/illiquidity pickup, net of costs)
SIGMA_R_RATES = SIGMA_R     # excess-return noise on top of the book yield (profit sharing,
                            # asset-mix risk); kept equal to SIGMA_R so the two regimes carry
                            # the same asset risk. 0 = pure book-yield crediting.

 
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

def _repo_on_path():
    """Make the repo root importable (olo/, liability/) from any working directory."""
    import os, sys
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if os.path.abspath(root) not in map(os.path.abspath, sys.path):
        sys.path.insert(0, root)

def rate_calibration():
    """Calibrate once per process, from the cached NBB OLO data: Vasicek (kappa,
    sigma, theta, r0) on the monthly 10Y history and the NSS curve on the latest
    cross-section. The olo package is imported HERE, not at module level, so
    economy.py still imports nothing and the constant-rate model never needs it."""
    global _CALIBRATION
    if _CALIBRATION is None:
        _repo_on_path()
        from olo.data.extract import load_olo
        from olo.calibration import calibrateVasicek, bootstrapForwardCurve
        df10Y, maturities, yields = load_olo()
        _CALIBRATION = dict(
            vasicek=calibrateVasicek(df10Y),
            curve=bootstrapForwardCurve(maturities, yields),
            hist10Y=df10Y["YIELD"].values / 100.0,      # monthly, oldest first; last = model t0
            t0=df10Y["DATE"].iloc[-1],
        )
    return _CALIBRATION


def draw_rate_scenarios(n_paths=N_EVAL, seed=RATE_SEED, model=None, horizon=None):
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
    """
    model = RATE_MODEL if model is None else model
    H = T if horizon is None else horizon
    if model == "constant":
        G_ = np.full((H, n_paths), G); mu_ = np.full((H, n_paths), MU)
        return dict(model=model, G=G_, mu=mu_, y10=np.full((H + 1, n_paths), np.nan),
                    r=np.full((H + 1, n_paths), np.nan), t0=None)

    _repo_on_path()
    from olo.simulation import simulateVasicek, simulateHullWhite
    from olo.pricing import reconstructFutureYield
    from liability.WAP import computeWAPRate
    cal = rate_calibration(); vas = cal["vasicek"]
    rng = np.random.default_rng(seed)
    mpy = int(round(1 / RATE_DT))
    if model == "hull_white":
        r_m = simulateHullWhite(cal["curve"], vas["kappa"], vas["sigma"], T=H,
                                n_paths=n_paths, dt=RATE_DT, rng=rng)
        y10_m = reconstructFutureYield(r_m, vas["kappa"], vas["sigma"], cal["curve"],
                                       tau=10.0, dt=RATE_DT)
    elif model == "vasicek":
        r_m = simulateVasicek(vas["kappa"], vas["theta"], vas["sigma"], vas["r0"], T=H,
                              n_paths=n_paths, dt=RATE_DT, rng=rng)
        y10_m = r_m
    else:
        raise ValueError(f"unknown RATE_MODEL {model!r}")

    # observed months up to t0, then simulated months 1..12H (row 0 of the simulation
    # is t0 itself, which the history already holds as an observation)
    hist = cal["hist10Y"]
    full = np.vstack([np.repeat(hist[:, None], n_paths, axis=1), y10_m[1:]])
    start = len(hist) - 1                                # row of model year 0
    G_ = computeWAPRate(full, start, H, months_per_year=mpy)
    n_book = BOOK_DURATION * mpy
    cs = np.vstack([np.zeros((1, n_paths)), np.cumsum(full, axis=0)])
    rows = start + mpy * np.arange(H)                    # start month of each year
    assert rows[0] + 1 >= n_book, "OLO history shorter than BOOK_DURATION"
    mu_ = (cs[rows + 1] - cs[rows + 1 - n_book]) / n_book + BOOK_SPREAD
    yearly = np.arange(H + 1) * mpy
    return dict(model=model, G=G_, mu=mu_, y10=y10_m[yearly], r=r_m[yearly], t0=cal["t0"])
 