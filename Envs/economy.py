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
T = 45          # career length in years
G = 0.0175      # frozen WAP guarantee rate (liability leg)
MU = 0.01       # tariff credited on the mathematical reserve
LAMBDA = 0.5    # employee weight in V = λ·V_employee + (1−λ)·V_employer
S0 = 1.0        # starting salary
W = 0.025       # deterministic salary growth
DISC = 0.01     # discount rate (single numeraire, to t=0)
SIGMA = 0.05    # credited-return volatility
N_EVAL = 2000   # default number of evaluation paths (SAA batch)
 
 
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
 