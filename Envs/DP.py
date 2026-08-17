"""
dp_oracle.py -- Rung 2a benchmark: DP oracle over (t, F), derived from basicEnv.

Independent ground truth for the PPO learner. Solves the SAME MDP that
basicEnv.run_episode implements, but by backward induction over a discretized
(t, F) grid rather than by sampling -- so oracle and learner never share a
derivation (no-circular-validation). The bridge to the certified rung-1
environment is check_against_env(), which asserts this file's transition
reproduces run_episode path-by-path.

WHAT CHANGES FROM RUNG 1
  * action: continuous l in [l_min, l_max], the contribution as a FRACTION of
    the liability (c = l * L). Rung 1's binary contribute-the-plan-premium is
    the special case l in {0, plan(t,S)/L}. Salary-linked plans re-enter at
    rung 2b as a rho = S/L state (see NOTE).
  * state: (t, F), F = R/L. Markov-sufficient because the reward is homogeneous
    degree 1 in the level, so value = L_t * v_t(F). See derivation below.

DERIVATION (all economics inherited from basicEnv, nothing invented)
  R' = (R + c) e^{MU + SIGMA z},  L' = (L + c) e^{G},  c = l L
    => F' = R'/L' = (F + l)/(1 + l) * e^{MU - G + SIGMA z}
    => L'/L = (1 + l) e^{G}                              (continuation growth)
  reward (to t=0 numeraire):
    flow_t   = -(1-LAMBDA) c e^{-DISC t} = L_t * [ -(1-LAMBDA) l e^{-DISC t} ]
    terminal = (LAMBDA max(R,L) - (1-LAMBDA) max(L-R,0)) e^{-DISC T}
             = L_T * [ (LAMBDA max(F,1) - (1-LAMBDA) max(1-F,0)) e^{-DISC T} ]
  Guess V_t(F,L) = L v_t(F). Then v closes in F alone:
    v_t(F) = max_l  -(1-LAMBDA) l e^{-DISC t} + (1+l) e^{G} E_z[ v_{t+1}(F') ]
    v_T(F) =        (LAMBDA max(F,1) - (1-LAMBDA) max(1-F,0)) e^{-DISC T}

INVARIANTS
  * gamma = 1. Discounting lives only in the reward (DISC), never as a decay.
  * F = 1 sits exactly on a grid node (the put kink).
  * v_{t+1} interpolated LINEAR (monotone, no kink overshoot).
  * horizon T is a TRUE terminal.
  * E_z via Gauss-Hermite over the lognormal shock (deterministic, no sampling).

NOTE (rung 2b): a salary-linked contribution c = m * plan(t, S) makes
l = m * plan(t,S)/L depend on rho = S/L, which is control-path-dependent and
becomes a second state: rho' = (1+W) rho / [(1 + l) e^{G}], shock-free. The
plans (fixed/step/age) from basicEnv then re-enter as per-plan configs.
"""

import numpy as np
import coreEnv as env

# Parameters come from the environment -- single source of truth, no drift.
T, G, MU, LAMBDA = env.T, env.G, env.MU, env.LAMBDA
S0, W, DISC, SIGMA = env.S0, env.W, env.DISC, env.SIGMA


# --------------------------------------------------------------------------
# Transition & reward (independently derived; checked against env below)
# --------------------------------------------------------------------------
def F_next(F, l, z):
    """F' = (F + l)/(1 + l) * exp(MU - G + SIGMA z)."""
    return (F + l) / (1.0 + l) * np.exp(MU - G + SIGMA * z)

def growth(l):
    """L_{t+1}/L_t = (1 + l) exp(G)."""
    return (1.0 + l) * np.exp(G)

def flow_per_L(t, l):
    """Per-unit-L flow reward, discounted to t=0."""
    return -(1.0 - LAMBDA) * l * np.exp(-DISC * t)

def terminal_per_L(F):
    """Per-unit-L maturity value, discounted to t=0. Kinks at F=1."""
    return (LAMBDA * np.maximum(F, 1.0)
            - (1.0 - LAMBDA) * np.maximum(1.0 - F, 0.0)) * np.exp(-DISC * T)


# --------------------------------------------------------------------------
# No-circular-validation bridge: reproduce run_episode path-by-path.
# --------------------------------------------------------------------------
def _roll(policy, plan, shocks):
    """Mirror env.run_episode in (R, L, S); also emit the realized (F, l) path
    so F_next can be checked against R'/L' directly."""
    R = L = 0.0; S = S0
    rewards = np.zeros(T)
    F_path, l_path, z_path = [], [], []
    for t in range(T):
        c = plan(t, S) * policy[t]
        l = c / L if L > 0 else 0.0
        F = R / L if L > 0 else np.nan
        rewards[t] = -(1.0 - LAMBDA) * c * np.exp(-DISC * t)
        z = 0.0 if shocks is None else shocks[t]
        F_path.append(F); l_path.append(l); z_path.append(z)
        R = (R + c) * np.exp(MU + SIGMA * z)
        L = (L + c) * np.exp(G)
        S *= (1.0 + W)
    payout = max(R, L); shortfall = max(L - R, 0.0)
    rewards[-1] += (LAMBDA * payout - (1.0 - LAMBDA) * shortfall) * np.exp(-DISC * T)
    return rewards, np.array(F_path), np.array(l_path), np.array(z_path)

def check_against_env(n_paths=8, seed=99, tol=1e-11):
    """(1) our roll reproduces run_episode's rewards; (2) F_next matches R'/L'."""
    rng = np.random.default_rng(seed)
    plans = {"fixed": env.plan_fixed(), "step": env.plan_step(), "age": env.plan_age()}
    for name, plan in plans.items():
        for _ in range(n_paths):
            pol = rng.integers(0, 2, T)
            shk = rng.standard_normal(T) if SIGMA > 0 else None
            r_ours, Fp, lp, zp = _roll(pol, plan, shk)
            _, r_env = env.run_episode(lambda t: int(pol[t]), plan, shk)
            assert np.max(np.abs(r_ours - r_env)) < tol, f"{name}: reward mismatch"
            # F-transition algebra vs realized ratio, on years with L>0 and a next year
            for t in range(T - 1):
                if not np.isnan(Fp[t]) and not np.isnan(Fp[t + 1]):
                    f_pred = F_next(Fp[t], lp[t], zp[t])
                    assert abs(f_pred - Fp[t + 1]) < 1e-9, f"{name}: F_next off at t={t}"
    print("check_against_env: rewards reproduce run_episode; F_next matches R'/L'  [OK]")


# --------------------------------------------------------------------------
# Grids & quadrature
# --------------------------------------------------------------------------
def make_F_grid(F_max=3.0, n=301):
    grid = np.linspace(0.0, F_max, n)
    j = int(np.argmin(np.abs(grid - 1.0)))
    assert abs(grid[j] - 1.0) < 1e-12, "F=1 must be exactly on a grid node"
    return grid

def make_l_grid(l_min=0.0, l_max=1.0, n=201):
    return np.linspace(l_min, l_max, n)

def gauss_hermite(n=15):
    """E_z[g] for z~N(0,1): nodes sqrt(2) x_i, weights w_i/sqrt(pi) (sum 1)."""
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)


# --------------------------------------------------------------------------
# Backward induction
# --------------------------------------------------------------------------
def solve(F_grid=None, l_grid=None, n_quad=15):
    if F_grid is None: F_grid = make_F_grid()
    if l_grid is None: l_grid = make_l_grid()
    z, omega = gauss_hermite(n_quad)
    NF, Nl, Nq = len(F_grid), len(l_grid), len(z)

    V = np.empty((T + 1, NF)); policy = np.empty((T, NF))
    V[T] = terminal_per_L(F_grid)

    # F'(F, l, z): (NF, Nl, Nq); growth(l): (Nl,)
    Fp = F_next(F_grid[:, None, None], l_grid[None, :, None], z[None, None, :])
    g_l = growth(l_grid)
    reach_max = Fp.max()

    for t in range(T - 1, -1, -1):
        v_next = np.interp(Fp.ravel(), F_grid, V[t + 1]).reshape(NF, Nl, Nq)
        cont = g_l[None, :] * (v_next * omega[None, None, :]).sum(axis=2)   # (NF, Nl)
        flow = flow_per_L(t, l_grid)[None, :]                              # (1, Nl)
        Q = flow + cont
        best = Q.argmax(axis=1)
        V[t] = Q[np.arange(NF), best]
        policy[t] = l_grid[best]

    return dict(F_grid=F_grid, l_grid=l_grid, V=V, policy=policy, reach_max=reach_max)


def evaluate_open_loop(schedule, plan, F_grid=None, n_quad=15):
    """Policy EVALUATION (not optimization) of a fixed open-loop fill schedule
    a_t in [0,1], c_t = a_t * plan(t, S_t), by backward induction over (t, F).

    Legitimate in (t, F): along a FIXED schedule the contribution does not
    depend on F, so the liability path L_t is deterministic (L has no shock),
    hence rho_t = S_t/L_t is a deterministic function of t -- no rho state.
    (State-feedback OPTIMIZATION over this action is NOT (t,F)-closed; it needs
    rho -- that is rung 2b.)

    Returns E[G_0], directly comparable to run_batch(schedule). Independent of
    run_batch by construction: forward SAA there, backward quadrature here.
    """
    if F_grid is None:
        F_grid = make_F_grid(F_max=4.0, n=401)
    z, omega = gauss_hermite(n_quad)

    # forward deterministic pass: c_t, L_t, ell_t=c_t/L_t, flows (all shock-free)
    S, L = S0, 0.0
    c = np.empty(T); ell = np.full(T, np.inf); flow = np.empty(T)
    for t in range(T):
        c[t] = schedule[t] * plan(t, S)
        flow[t] = -(1.0 - LAMBDA) * c[t] * np.exp(-DISC * t)
        if L > 0.0:
            ell[t] = c[t] / L
        L = (L + c[t]) * np.exp(G)
        S *= (1.0 + W)
    L_final = L                                   # L_45
    t0 = int(np.argmax(c > 0.0))                  # first funded year (L=0 before it)

    # terminal per unit L, WITHOUT discount -- discount is applied once in V
    g = LAMBDA * np.maximum(F_grid, 1.0) - (1.0 - LAMBDA) * np.maximum(1.0 - F_grid, 0.0)

    # backward pass: U_t(F) = E[ g(F_45) | F_t = F ], propagated with finite ell
    U = g                                         # U_45
    for t in range(T - 1, t0, -1):                # t = 44 .. t0+1 (ell finite)
        Fp = F_next(F_grid[:, None], ell[t], z[None, :])          # (NF, Nq)
        U = (np.interp(Fp.ravel(), F_grid, U).reshape(len(F_grid), len(z))
             * omega[None, :]).sum(axis=1)
    # bootstrap off L=0 at the first funded year: F_{t0+1} = exp(MU - G + SIGMA z)
    F_boot = np.exp(MU - G + SIGMA * z)
    E_g = float((np.interp(F_boot, F_grid, U) * omega).sum())

    V = float(flow.sum()) + L_final * np.exp(-DISC * T) * E_g
    return dict(value=V, flows=float(flow.sum()),
                E_g=E_g, L_final=float(L_final), t0=t0)


if __name__ == "__main__":
    check_against_env()
    out = solve()
    F, V, pol = out["F_grid"], out["V"], out["policy"]
    v0, kink = V[0], int(np.argmin(np.abs(out["F_grid"] - 1.0)))

    print(f"v_0(F) monotone non-decreasing in F : {np.all(np.diff(v0) >= -1e-6)}")
    print(f"F=1 exactly on node                 : {1.0 in F}")
    print(f"max reachable F' (grid F_max={F.max():.1f}) : {out['reach_max']:.3f}")
    print("optimal l*(t=0)  at F = 0.7 / 1.0 / 1.3 : "
          f"{np.interp(0.7,F,pol[0]):.3f} / {pol[0][kink]:.3f} / {np.interp(1.3,F,pol[0]):.3f}")
    print("optimal l*(t=44) at F = 0.7 / 1.0 / 1.3 : "
          f"{np.interp(0.7,F,pol[44]):.3f} / {pol[44][kink]:.3f} / {np.interp(1.3,F,pol[44]):.3f}")
    bang = np.all((pol == pol.min()) | (pol == pol.max()))
    print(f"policy hugs a bound everywhere?      : {bang}")
    # where does the policy sit relative to the F=1 kink at mid-career?
    mid = 22
    print(f"l*(t=22) sampled F=[0.5,0.8,1.0,1.2,1.5]: "
          f"{[round(float(np.interp(x,F,pol[mid])),3) for x in (0.5,0.8,1.0,1.2,1.5)]}")