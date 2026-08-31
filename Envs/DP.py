"""
dp_oracle.py -- Rung 2b benchmark: DP oracle over (t, F, rho), salary-linked action.

Independent ground truth for the PPO learner. Solves the SAME MDP that coreEnv
implements, by backward induction over a discretized (t, F, rho) grid rather
than by sampling -- oracle and learner never share a derivation
(no-circular-validation). The bridge is check_against_env(), asserting this
file's F' AND rho' transitions reproduce a coreEnv rollout path-by-path.

ONE OBJECT, TWO JOBS  (operator switch O over the action):
  * optimize : O = max over a in [0,1]        -> optimal contribution rule a*(t,F,rho)
  * evaluate : O = plug in a given plan_rule   -> value of that plan on the same grid
Both run on the same (F,rho) grid, so their values are directly comparable; the
gap v_optimal - v_plan is the "distance from optimal" a real plan pays.

STATE, ACTION, DYNAMICS  (all economics inherited from coreEnv)
  state    (t, F, rho),  F = R/L (funding ratio / put moneyness),  rho = S/L
  action   a in [0,1] fill-fraction of the salary-linked premium GAMMA*S
           c = a*GAMMA*S   =>   l := c/L = a*GAMMA*rho  (effective liability fraction)
  R' = (R+c) e^{MU+SIGMA z},  L' = (L+c) e^{G},  S' = (1+W) S
    => F'   = (F + l)/(1 + l) * e^{MU-G+SIGMA z}        (carries the shock)
       rho' = (1+W) rho / [(1+l) e^{G}]                 (DETERMINISTIC: W fixed)
       L'/L = (1+l) e^{G}                               (continuation unit-conversion)

VALUE  (per unit L_t; homogeneity gives V_t(R,L,S) = L_t * v_t(F,rho))
  v_t(F,rho) = O_a [ -(1-LAM) a*GAMMA*rho e^{-DISC t}
                     + (1+l) e^{G} E_z[ v_{t+1}(F', rho') ] ]
  v_T(F,rho) = (LAM max(F,1) - (1-LAM) max(1-F,0)) e^{-DISC T}     (rho-independent)

WHY rho enters even with fixed salary: l = a*GAMMA*rho depends on the liability
LEVEL (via S/L), which (t,F) does not pin down. Fixed W removes a SHOCK, not a
state dimension -- so the quadrature stays 1-D (only z), rho evolves shock-free.

INVARIANTS
  * gamma_RL = 1. Discounting lives only in the reward (DISC), never as a decay.
  * F = 1 sits exactly on a grid node (the put kink).
  * v_{t+1} read by BILINEAR interpolation over (F, log rho); monotone in F, no
    overshoot across the kink.
  * horizon T is a TRUE terminal.
  * E_z via Gauss-Hermite over the single (reserve) shock; deterministic.

CAVEAT: GAMMA*S must be the whole premium for the (F,rho) reduction to hold.
plan_fixed (rate*S) and plan_age (time-varying rate*S) fit; plan_step does NOT
(its tranche ceiling is an ABSOLUTE euro threshold -> not scale-free).
"""

import numpy as np
import coreEnv as env

# Parameters from the environment -- single source of truth, no drift.
T, G, MU, LAMBDA = env.T, env.G, env.MU, env.LAMBDA
S0, W, DISC, SIGMA = env.S0, env.W, env.DISC, env.SIGMA
GAMMA = 0.05   # salary-linked premium rate; premium = GAMMA*S (matches plan_fixed)


# --------------------------------------------------------------------------
# Transition & reward (independently derived; checked against env below)
# --------------------------------------------------------------------------
def F_next(F, l, z):
    """F' = (F + l)/(1 + l) * exp(MU - G + SIGMA z).  l is the effective fraction."""
    return (F + l) / (1.0 + l) * np.exp(MU - G + SIGMA * z)

def rho_next(rho, l):
    """rho' = (1+W) rho / [(1+l) e^{G}].  Deterministic (salary growth fixed)."""
    return (1.0 + W) * rho / ((1.0 + l) * np.exp(G))

def growth(l):
    """L_{t+1}/L_t = (1 + l) exp(G) -- the continuation unit-conversion factor."""
    return (1.0 + l) * np.exp(G)

def flow_per_L(t, l):
    """Per-unit-L flow reward, discounted to t=0.  l = a*GAMMA*rho."""
    return -(1.0 - LAMBDA) * l * np.exp(-DISC * t)

def terminal_per_L(F):
    """Per-unit-L maturity value, discounted to t=0. Kinks at F=1. rho-free."""
    return (LAMBDA * np.maximum(F, 1.0)
            - (1.0 - LAMBDA) * np.maximum(1.0 - F, 0.0)) * np.exp(-DISC * T)


# --------------------------------------------------------------------------
# Bilinear interpolation over a grid regular in F and in log(rho).
# --------------------------------------------------------------------------
def bilinear(F_grid, lr_grid, V, Fq, lrq):
    """V has shape (NF, Nrho) on (F_grid, lr_grid=log rho_grid). Query arrays
    Fq, lrq (any common shape) -> interpolated values. Constant-clamped at edges."""
    Fq = np.clip(Fq, F_grid[0], F_grid[-1])
    lrq = np.clip(lrq, lr_grid[0], lr_grid[-1])
    iF = np.clip(np.searchsorted(F_grid, Fq) - 1, 0, len(F_grid) - 2)
    iR = np.clip(np.searchsorted(lr_grid, lrq) - 1, 0, len(lr_grid) - 2)
    F0, F1 = F_grid[iF], F_grid[iF + 1]
    R0, R1 = lr_grid[iR], lr_grid[iR + 1]
    tF = (Fq - F0) / (F1 - F0)
    tR = (lrq - R0) / (R1 - R0)
    v00 = V[iF, iR]; v01 = V[iF, iR + 1]; v10 = V[iF + 1, iR]; v11 = V[iF + 1, iR + 1]
    return (v00 * (1 - tF) * (1 - tR) + v10 * tF * (1 - tR)
            + v01 * (1 - tF) * tR + v11 * tF * tR)


# --------------------------------------------------------------------------
# Grids & quadrature
# --------------------------------------------------------------------------
def make_F_grid(F_max=3.0, n=151):
    grid = np.linspace(0.0, F_max, n)
    assert abs(grid[np.argmin(np.abs(grid - 1.0))] - 1.0) < 1e-12, "F=1 must be on a node"
    return grid

def make_rho_grid(rho_min=0.3, rho_max=35.0, n=81):
    return np.exp(np.linspace(np.log(rho_min), np.log(rho_max), n))   # log-spaced

def make_a_grid(n=101):
    return np.linspace(0.0, 1.0, n)

def gauss_hermite(n=15):
    """E_z[g] for z~N(0,1): nodes sqrt(2) x_i, weights w_i/sqrt(pi) (sum 1)."""
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)


# --------------------------------------------------------------------------
# No-circular-validation bridge: reproduce a coreEnv rollout path-by-path.
# --------------------------------------------------------------------------
def _roll(policy, plan, shocks):
    """Mirror env.run_episode in (R,L,S); emit realized (F, rho, l, z) so
    F_next AND rho_next can be checked against R'/L' and S'/L' directly."""
    R = L = 0.0; S = S0
    rewards = np.zeros(T)
    Fp, rhop, lp, zp = [], [], [], []
    for t in range(T):
        c = plan(t, S) * policy[t]
        l = c / L if L > 0 else 0.0
        F = R / L if L > 0 else np.nan
        rho = S / L if L > 0 else np.nan
        rewards[t] = -(1.0 - LAMBDA) * c * np.exp(-DISC * t)
        z = 0.0 if shocks is None else shocks[t]
        Fp.append(F); rhop.append(rho); lp.append(l); zp.append(z)
        R = (R + c) * np.exp(MU + SIGMA * z)
        L = (L + c) * np.exp(G)
        S *= (1.0 + W)
    payout = max(R, L); shortfall = max(L - R, 0.0)
    rewards[-1] += (LAMBDA * payout - (1.0 - LAMBDA) * shortfall) * np.exp(-DISC * T)
    return rewards, np.array(Fp), np.array(rhop), np.array(lp), np.array(zp)

def check_against_env(n_paths=8, seed=99, tol=1e-11):
    """(1) rewards reproduce run_episode; (2) F_next & rho_next match R'/L', S'/L'.
    Uses plan_fixed(rate=GAMMA) so the env's c = GAMMA*S*a matches c = a*GAMMA*S."""
    rng = np.random.default_rng(seed)
    plan = env.plan_fixed(rate=GAMMA)
    for _ in range(n_paths):
        pol = rng.integers(0, 2, T)
        shk = rng.standard_normal(T) if SIGMA > 0 else None
        r_ours, Fq, rq, lq, zq = _roll(pol, plan, shk)
        _, r_env = env.run_episode(lambda t: int(pol[t]), plan, shk)
        assert np.max(np.abs(r_ours - r_env)) < tol, "reward mismatch"
        for t in range(T - 1):
            if not np.isnan(Fq[t]) and not np.isnan(Fq[t + 1]):
                assert abs(F_next(Fq[t], lq[t], zq[t]) - Fq[t + 1]) < 1e-9, f"F' off t={t}"
                assert abs(rho_next(rq[t], lq[t]) - rq[t + 1]) < 1e-9, f"rho' off t={t}"
    print("check_against_env: rewards reproduce run_episode; F_next & rho_next match  [OK]")


# --------------------------------------------------------------------------
# Backward induction  (mode = 'optimize' | 'evaluate')
# --------------------------------------------------------------------------
def solve(mode="optimize", plan_rule=None, F_grid=None, rho_grid=None,
          a_grid=None, n_quad=15):
    """
    optimize: O = max over a_grid; returns policy a*(t,F,rho).
    evaluate: O = plug in plan_rule(t, F_col, rho_row) -> a in [0,1] on the
              (NF,Nrho) grid; returns that policy's value (no maximisation).
    """
    if F_grid is None:   F_grid = make_F_grid()
    if rho_grid is None: rho_grid = make_rho_grid()
    if a_grid is None:   a_grid = make_a_grid()
    z, om = gauss_hermite(n_quad)
    lr_grid = np.log(rho_grid)
    NF, Nrho = len(F_grid), len(rho_grid)
    eG = np.exp(G); drift = np.exp(MU - G + SIGMA * z)      # (Nq,)

    V = np.empty((T + 1, NF, Nrho))
    policy = np.empty((T, NF, Nrho))
    V[T] = terminal_per_L(F_grid)[:, None] * np.ones((1, Nrho))   # rho-independent

    if mode == "optimize":
        Na = len(a_grid)
        # l(rho, a) = a*GAMMA*rho : (Nrho, Na)
        l = a_grid[None, :] * GAMMA * rho_grid[:, None]
        # rho'(rho, a) : (Nrho, Na)  (deterministic, no z)
        rp = (1.0 + W) * rho_grid[:, None] / ((1.0 + l) * eG)
        lrp = np.log(rp)
        gl = (1.0 + l) * eG                                       # (Nrho, Na)
        # F'(F, rho, a, z) : (NF, Nrho, Na, Nq)
        Fq = ((F_grid[:, None, None, None] + l[None, :, :, None])
              / (1.0 + l[None, :, :, None]) * drift[None, None, None, :])
        lrq = np.broadcast_to(lrp[None, :, :, None], Fq.shape)
        for t in range(T - 1, -1, -1):
            v = bilinear(F_grid, lr_grid, V[t + 1], Fq, lrq)     # (NF,Nrho,Na,Nq)
            cont = gl[None, :, :] * (v * om[None, None, None, :]).sum(axis=3)  # (NF,Nrho,Na)
            flow = flow_per_L(t, l)[None, :, :]                  # (1,Nrho,Na)
            Q = flow + cont
            best = Q.argmax(axis=2)
            V[t] = np.take_along_axis(Q, best[:, :, None], axis=2)[:, :, 0]
            policy[t] = a_grid[best]

    elif mode == "evaluate":
        assert plan_rule is not None, "evaluate mode needs a plan_rule(t, F, rho)"
        Fcol = F_grid[:, None]; rrow = rho_grid[None, :]         # (NF,1),(1,Nrho)
        for t in range(T - 1, -1, -1):
            a = np.clip(plan_rule(t, Fcol, rrow) * np.ones((NF, Nrho)), 0.0, 1.0)
            l = a * GAMMA * rrow                                 # (NF,Nrho)
            rp = (1.0 + W) * rrow / ((1.0 + l) * eG)             # (NF,Nrho)
            lrp = np.log(rp)
            Fq = ((Fcol[..., None] + l[..., None]) / (1.0 + l[..., None])
                  * drift[None, None, :])                        # (NF,Nrho,Nq)
            lrq = np.broadcast_to(lrp[..., None], Fq.shape)
            v = bilinear(F_grid, lr_grid, V[t + 1], Fq, lrq)     # (NF,Nrho,Nq)
            cont = (1.0 + l) * eG * (v * om[None, None, :]).sum(axis=2)
            V[t] = flow_per_L(t, l) + cont
            policy[t] = a
    else:
        raise ValueError("mode must be 'optimize' or 'evaluate'")

    return dict(F_grid=F_grid, rho_grid=rho_grid, a_grid=a_grid, V=V, policy=policy)


# --------------------------------------------------------------------------
# Independent Monte-Carlo value of a fixed action rule (cross-check).
# --------------------------------------------------------------------------
def mc_value(t0, R0, L0, S0v, a_rule, n_paths=200000, seed=1):
    """Value (to t=0 numeraire) from concrete (t0,R,L,S) under a_rule(t,F,rho).
    Sampling, not quadrature -> genuine independent check."""
    rng = np.random.default_rng(seed)
    R = np.full(n_paths, float(R0)); L = np.full(n_paths, float(L0)); S = float(S0v)
    val = np.zeros(n_paths)
    for t in range(t0, T):
        F = R / L; rho = S / L
        a = np.clip(a_rule(t, F, rho) * np.ones(n_paths), 0.0, 1.0)
        c = a * GAMMA * S
        val -= (1.0 - LAMBDA) * c * np.exp(-DISC * t)
        R = (R + c) * np.exp(MU + SIGMA * rng.standard_normal(n_paths))
        L = (L + c) * np.exp(G)
        S *= (1.0 + W)
    val += (LAMBDA * np.maximum(R, L) - (1.0 - LAMBDA) * np.maximum(L - R, 0.0)) * np.exp(-DISC * T)
    return val.mean(), val.std() / np.sqrt(n_paths)


if __name__ == "__main__":
    # Coarse grids for a fast self-test. Production: raise n via the doubling
    # test (solve at n and 2n, watch v_0 move < the PPO tolerance being certified).
    Fg = make_F_grid(n=76)          # step 0.04; F=1 on a node
    rg = make_rho_grid(n=41)
    ag = make_a_grid(n=41)
    NQ = 9

    check_against_env()

    # ---- OPTIMIZE: find a*(t,F,rho) ----
    opt = solve("optimize", F_grid=Fg, rho_grid=rg, a_grid=ag, n_quad=NQ)
    F, rho, pol, V = opt["F_grid"], opt["rho_grid"], opt["policy"], opt["V"]
    mono = all(np.all(np.diff(V[0][:, j]) >= -1e-5) for j in range(len(rho)))
    bang = np.all((pol == 0.0) | (pol == 1.0))
    print(f"\n[optimize] v_0 monotone in F (all rho): {mono}")
    print(f"[optimize] policy bang-bang a in {{0,1}}: {bang}   mean a*={pol.mean():.3f}"
          f"   (a*=1: homogeneous reward -> corner optimum, as derived)")

    # ---- EVALUATE the optimal action (a=1): must reproduce the optimum ----
    ev1 = solve("evaluate", plan_rule=lambda t, F, r: 1.0, F_grid=Fg, rho_grid=rg, n_quad=NQ)
    print(f"[evaluate a=1] == optimize value:       {np.allclose(ev1['V'][0], V[0], atol=1e-6)}")

    # ---- EVALUATE a throttled plan (a=0.5): price it and show the optimality gap ----
    ev2 = solve("evaluate", plan_rule=lambda t, F, r: 0.5, F_grid=Fg, rho_grid=rg, n_quad=NQ)
    gap = V[0] - ev2["V"][0]
    print(f"[evaluate a=.5] optimality gap v*-v_plan at t=0: "
          f"min={gap.min():.4f} max={gap.max():.4f}  (>=0 everywhere: {np.all(gap >= -1e-9)})")

    # ---- independent Monte-Carlo cross-check of the throttled plan's value ----
    t0, R0, L0, Sv = 20, 0.9, 1.0, 3.0
    v_or = float(bilinear(F, np.log(rho), ev2["V"][t0],
                          np.array(R0 / L0), np.array(np.log(Sv / L0))))
    V_mc, se = mc_value(t0, R0, L0, Sv, a_rule=lambda t, F, r: 0.5, n_paths=100000)
    print(f"[MC check]  oracle L*v={L0*v_or:.5f}  vs  MC={V_mc:.5f} +/- {2*se:.5f}  "
          f"match={abs(L0*v_or - V_mc) < 3*se + 1e-3}")