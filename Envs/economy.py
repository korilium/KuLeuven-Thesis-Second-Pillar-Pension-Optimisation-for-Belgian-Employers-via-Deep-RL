"""
economy.py -- the economic scenario (exogenous world).
 
Single source of truth for the pension economy shared by BOTH the tabular
environment (coreEnv) and the DP oracle (dp_oracle). Holds only:
  * structural / economic parameters,
  * the contribution plan rules c(t, S),
  * the exogenous credited-return shock process.
 
No episode stepping, no reward accounting, no learning, no benchmarks --
those consume these primitives downstream. Zero internal dependencies.
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

DISC_EMP = 0.025    # EMPLOYEE discount: values future retirement income at the risk-free/OLO rate
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

N_EVAL = 2000   # default number of evaluation paths (SAA batch) -- numerical, not economic
 
 
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
 