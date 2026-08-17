"""
coreEnv.py -- the decision process (environment dynamics + benchmarks).

Encodes the MDP the agent acts in: the (R, L, S) recursion and the reward,
episode/batch runners, policy evaluation, the consistency self-check, and the
benchmarks that SOLVE the decision problem (sweep / gap / local_search).

Consumes economy.py for all parameters, plans, and shocks; re-exports them so
`import coreEnv as env` exposes the full environment API (this is what the DP
oracle imports for its no-circular-validation bridge). No learner lives here.
"""

import numpy as np
from economy import *          # parameters, plans, draw_shock_batch, batch
from economy import (T, G, MU, LAMBDA, S0, W, DISC, SIGMA)   # explicit for clarity


# ---------------------------------------------------------------------------
# Environment dynamics
# ---------------------------------------------------------------------------
def run_batch(policy, plan, shocks):
    """All paths at once for a FIXED policy (length-T 0/1 array).
    Same recursion as run_episode, vectorized over axis 0.
    shocks: (n_paths, T). Returns values: (n_paths,)."""
    n = shocks.shape[0]
    R = np.zeros(n)
    L = 0.0
    S = S0
    values = np.zeros(n)
    for t in range(T):
        c = plan(t, S) * policy[t]
        values -= (1.0 - LAMBDA) * c * np.exp(-DISC * t)
        R = (R + c) * np.exp(MU + SIGMA * shocks[:, t])
        L = (L + c) * np.exp(G)
        S *= (1.0 + W)
    payout = np.maximum(R, L)
    shortfall = np.maximum(L - R, 0.0)
    values += (LAMBDA * payout - (1.0 - LAMBDA) * shortfall) * np.exp(-DISC * T)
    return values


def run_episode(choose_action, plan, shocks=None):
    """Play one career. choose_action(t) -> 0 or 1. Returns (actions, rewards)."""
    R = L = 0.0
    actions = np.empty(T, dtype=np.int64)
    S = S0
    rewards = np.zeros(T)
    for t in range(T):
        a = choose_action(t)
        actions[t] = a
        c = plan(t, S) * a
        rewards[t] = -(1.0 - LAMBDA) * c * np.exp(-DISC * t)
        z = 0.0 if shocks is None else shocks[t]
        R = (R + c) * np.exp(MU + SIGMA * z)
        L = (L + c) * np.exp(G)
        S *= (1.0 + W)
    payout = max(R, L)
    shortfall = max(L - R, 0.0)
    rewards[-1] += (LAMBDA * payout - (1.0 - LAMBDA) * shortfall) * np.exp(-DISC * T)
    return actions, rewards


# ---------------------------------------------------------------------------
# Policy evaluation & consistency
# ---------------------------------------------------------------------------


def check_batch_consistency(plan, n_check=5, seed=99):
    """run_batch must agree with run_episode path-by-path."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_check, T))
    policy = rng.integers(0, 2, T)
    vec = run_batch(policy, plan, shocks)
    for i in range(n_check):
        _, r = run_episode(lambda t: int(policy[t]), plan, shocks[i])
        assert abs(vec[i] - r.sum()) < 1e-10, f"path {i}: {vec[i]} vs {r.sum()}"


def evaluate_policy(policy, plan):
    """Replay a fixed policy and return unscaled economic metrics."""
    R = L = 0.0
    S = S0
    pv_contrib = 0.0
    t_weighted = 0.0
    for t in range(T):
        c = plan(t, S) * policy[t]
        d = c * np.exp(-DISC * t)
        pv_contrib += d
        t_weighted += t * d
        R = (R + c) * np.exp(MU)
        L = (L + c) * np.exp(G)
        S *= (1.0 + W)
    payout = max(R, L)
    pv_payout = payout * np.exp(-DISC * T)
    return {
        "pv_contrib": pv_contrib,
        "pv_payout": pv_payout,
        "efficiency": pv_payout / pv_contrib if pv_contrib > 0 else np.nan,
        "duration": t_weighted / pv_contrib if pv_contrib > 0 else np.nan,
        "replacement": payout / (S0 * (1.0 + W) ** (T - 1)),
    }

