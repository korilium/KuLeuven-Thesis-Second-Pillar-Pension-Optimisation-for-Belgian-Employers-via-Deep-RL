"""
dp_oracle_crra.py -- Rung-2c benchmark: DP oracle over (t, F, rho, L),
euro-CRRA utility on the employee leg, both asset and guarantee shocked.

State (t, F, rho, L):  F=R/L (funding ratio / put moneyness), rho=S/L, L=liability level.
The level L is carried because CRRA on the employee leg (degree 1-eta) and the linear
employer legs (degree 1) have mixed homogeneity degrees, so the value does NOT factor as
L^p v(F,rho). We keep F, rho as coordinates (kink at F=1 is one grid line; dynamics
autonomous) and let L be a coarse, smooth log axis.

Dynamics (z_R, z_L independent N(0,1); c=a*GAMMA*S, l=c/L=a*GAMMA*rho):
    F'   = (F+l)/(1+l) * e^{(MU-G)+SIGMA_R z_R - SIGMA_L z_L}
    rho' = (1+W) rho / [(1+l) e^{G+SIGMA_L z_L}]
    L'   = L (1+l) e^{G+SIGMA_L z_L}

Recursion (absolute euros to t=0; NO growth factor, NO gamma):
    V_t(F,rho,L) = max_a [ -(1-LAM) a GAMMA rho L e^{-DISC t}
                           + E_{z_R,z_L}[ V_{t+1}(F',rho',L') ] ]
    V_T(F,rho,L) = e^{-DISC T}[ LAM*B*u(max(F,1) L) - (1-LAM) L max(1-F,0) ],  u(x)=x^{1-eta}/(1-eta)
"""

import numpy as np

# --- parameters -----------------------------------------------------------
T = 45
G, MU, W, DISC = 0.0175, 0.01, 0.025, 0.01
SIGMA_R, SIGMA_L = 0.05, 0.02        # asset shock, guarantee shock
GAMMA, LAMBDA, S0 = 0.05, 0.5, 1.0
ETA = 2.0                            # CRRA coefficient (eta != 1); higher eta => finer grid needed
B = 10.0                             # employee util-to-euro weight (calibration knob)


def u(x):
    """CRRA utility, eta != 1."""
    return np.power(x, 1.0 - ETA) / (1.0 - ETA)


# --- transitions (the formulas the solver uses; checked in the test suite) --
def F_next(F, l, zR, zL):
    """F' = (F+l)/(1+l) * exp((MU-G) + SIGMA_R z_R - SIGMA_L z_L)."""
    return (F + l) / (1.0 + l) * np.exp((MU - G) + SIGMA_R * zR - SIGMA_L * zL)

def rho_next(rho, l, zL):
    """rho' = (1+W) rho / [(1+l) exp(G + SIGMA_L z_L)]."""
    return (1.0 + W) * rho / ((1.0 + l) * np.exp(G + SIGMA_L * zL))

def L_next(L, l, zL):
    """L' = L (1+l) exp(G + SIGMA_L z_L)."""
    return L * (1.0 + l) * np.exp(G + SIGMA_L * zL)


def gauss_hermite_2d(n=5):
    """2D product rule over (z_R, z_L). Flat nodes zR,zL and weights (sum 1)."""
    x, w = np.polynomial.hermite.hermgauss(n)
    z = np.sqrt(2.0) * x
    om = w / np.sqrt(np.pi)
    ZR, ZL = np.meshgrid(z, z, indexing="ij")
    WW = np.outer(om, om)
    return ZR.ravel(), ZL.ravel(), WW.ravel()


# --- grids ----------------------------------------------------------------
def make_F_grid(F_max=3.0, n=61):
    g = np.linspace(0.0, F_max, n)
    assert abs(g[np.argmin(np.abs(g - 1.0))] - 1.0) < 1e-12, "F=1 must be a node"
    return g

def make_rho_grid(lo=0.3, hi=35.0, n=21):
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))

def make_L_grid(lo=0.03, hi=15.0, n=13):
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))

def make_a_grid(n=21):
    return np.linspace(0.0, 1.0, n)


# --- trilinear interpolation over (F, log rho, log L) ---------------------
def trilinear(Fg, lrg, lLg, V, Fq, lrq, lLq):
    Fq = np.clip(Fq, Fg[0], Fg[-1]); lrq = np.clip(lrq, lrg[0], lrg[-1]); lLq = np.clip(lLq, lLg[0], lLg[-1])
    iF = np.clip(np.searchsorted(Fg, Fq) - 1, 0, len(Fg) - 2)
    iR = np.clip(np.searchsorted(lrg, lrq) - 1, 0, len(lrg) - 2)
    iL = np.clip(np.searchsorted(lLg, lLq) - 1, 0, len(lLg) - 2)
    tF = (Fq - Fg[iF]) / (Fg[iF + 1] - Fg[iF])
    tR = (lrq - lrg[iR]) / (lrg[iR + 1] - lrg[iR])
    tL = (lLq - lLg[iL]) / (lLg[iL + 1] - lLg[iL])
    c000 = V[iF, iR, iL];       c100 = V[iF + 1, iR, iL]
    c010 = V[iF, iR + 1, iL];   c110 = V[iF + 1, iR + 1, iL]
    c001 = V[iF, iR, iL + 1];   c101 = V[iF + 1, iR, iL + 1]
    c011 = V[iF, iR + 1, iL + 1]; c111 = V[iF + 1, iR + 1, iL + 1]
    return (c000*(1-tF)*(1-tR)*(1-tL) + c100*tF*(1-tR)*(1-tL)
          + c010*(1-tF)*tR*(1-tL)     + c110*tF*tR*(1-tL)
          + c001*(1-tF)*(1-tR)*tL     + c101*tF*(1-tR)*tL
          + c011*(1-tF)*tR*tL         + c111*tF*tR*tL)


def terminal(Fg, Lg):
    """V_T on (F, L); rho-independent. Shape (NF, NL)."""
    Fc = Fg[:, None]; Lc = Lg[None, :]
    emp = LAMBDA * B * u(np.maximum(Fc, 1.0) * Lc)          # euro-CRRA employee
    empr = (1.0 - LAMBDA) * Lc * np.maximum(1.0 - Fc, 0.0)  # linear employer put
    return np.exp(-DISC * T) * (emp - empr)


# --- backward induction (mode = 'optimize' | 'evaluate') ------------------
def solve(mode="optimize", plan_rule=None, Fg=None, rg=None, Lg=None, ag=None, n_quad=15):
    if Fg is None: Fg = make_F_grid()
    if rg is None: rg = make_rho_grid()
    if Lg is None: Lg = make_L_grid()
    if ag is None: ag = make_a_grid()
    zR, zL, wq = gauss_hermite_2d(n_quad)
    lrg, lLg = np.log(rg), np.log(Lg)
    NF, NR, NL, Q = len(Fg), len(rg), len(Lg), len(wq)

    V = np.empty((T + 1, NF, NR, NL))
    policy = np.empty((T, NF, NR, NL))
    VT = terminal(Fg, Lg)                                    # (NF, NL)
    V[T] = VT[:, None, :] * np.ones((1, NR, 1))              # broadcast over rho

    def bellman_scalar_a(t, a, Vnext):
        """Q(F,rho,L) for a single scalar action a (used in the optimize sweep)."""
        l = a * GAMMA * rg                                   # (NR,)
        Fp = F_next(Fg[:, None, None], l[None, :, None], zR[None, None, :], zL[None, None, :])  # (NF,NR,Q)
        rp = rho_next(rg[:, None], l[:, None], zL[None, :])                                     # (NR,Q)
        Lp = L_next(Lg[:, None, None], l[None, :, None], zL[None, None, :])                     # (NL,NR,Q)
        Fq = np.broadcast_to(Fp[:, :, None, :], (NF, NR, NL, Q))
        lrq = np.broadcast_to(np.log(rp)[None, :, None, :], (NF, NR, NL, Q))
        lLq = np.broadcast_to(np.log(Lp).transpose(1, 0, 2)[None, :, :, :], (NF, NR, NL, Q))
        vi = trilinear(Fg, lrg, lLg, Vnext, Fq, lrq, lLq)    # (NF,NR,NL,Q)
        cont = (vi * wq[None, None, None, :]).sum(axis=3)     # (NF,NR,NL)
        flow = -(1.0 - LAMBDA) * a * GAMMA * rg[:, None] * Lg[None, :] * np.exp(-DISC * t)      # (NR,NL)
        return flow[None, :, :] + cont

    def bellman_field_a(t, A, Vnext):
        """Q(F,rho,L) for a per-cell action field A of shape (NF,NR,NL) -- the
        general state-dependent rule a(F,rho,L). Used in the evaluate mode so that
        real plans whose contribution depends on the state are priced exactly."""
        l = A * GAMMA * rg[None, :, None]                    # (NF,NR,NL): l=a*GAMMA*rho
        Fb = Fg[:, None, None, None]; lb = l[:, :, :, None]
        zRb = zR[None, None, None, :]; zLb = zL[None, None, None, :]
        Fq = F_next(Fb, lb, zRb, zLb)                        # (NF,NR,NL,Q)
        rob = rg[None, :, None, None]; Lb = Lg[None, None, :, None]
        rp = rho_next(rob, lb, zLb)                          # (NF,NR,NL,Q) (broadcasts)
        Lp = L_next(Lb, lb, zLb)                             # (NF,NR,NL,Q)
        lrq = np.log(np.broadcast_to(rp, Fq.shape))
        lLq = np.log(np.broadcast_to(Lp, Fq.shape))
        vi = trilinear(Fg, lrg, lLg, Vnext, Fq, lrq, lLq)    # (NF,NR,NL,Q)
        cont = (vi * wq[None, None, None, :]).sum(axis=3)     # (NF,NR,NL)
        flow = -(1.0 - LAMBDA) * A * GAMMA * rg[None, :, None] * Lg[None, None, :] * np.exp(-DISC * t)
        return flow + cont

    for t in range(T - 1, -1, -1):
        if mode == "optimize":
            best = np.full((NF, NR, NL), -np.inf); abest = np.zeros((NF, NR, NL))
            for a in ag:
                Q_ = bellman_scalar_a(t, a, V[t + 1])
                upd = Q_ > best
                best = np.where(upd, Q_, best); abest = np.where(upd, a, abest)
            V[t] = best; policy[t] = abest
        else:
            # plan_rule(t, F, rho, L) -> action per cell; broadcast to (NF,NR,NL)
            A = np.clip(plan_rule(t, Fg[:, None, None], rg[None, :, None], Lg[None, None, :])
                        * np.ones((NF, NR, NL)), 0.0, 1.0)
            V[t] = bellman_field_a(t, A, V[t + 1]); policy[t] = A
    return dict(Fg=Fg, rg=rg, Lg=Lg, ag=ag, V=V, policy=policy)


# --- independent Monte-Carlo value of a fixed constant-a rule -------------
def mc_value(t0, R0, L0, S0v, a_const, n_paths=200000, seed=1):
    rng = np.random.default_rng(seed)
    R = np.full(n_paths, float(R0)); L = np.full(n_paths, float(L0)); S = float(S0v)
    val = np.zeros(n_paths)
    for t in range(t0, T):
        c = a_const * GAMMA * S
        val -= (1.0 - LAMBDA) * c * np.exp(-DISC * t)
        R = (R + c) * np.exp(MU + SIGMA_R * rng.standard_normal(n_paths))
        L = (L + c) * np.exp(G + SIGMA_L * rng.standard_normal(n_paths))
        S *= (1.0 + W)
    payout = np.maximum(R, L); short = np.maximum(L - R, 0.0)
    val += np.exp(-DISC * T) * (LAMBDA * B * u(payout) - (1.0 - LAMBDA) * short)
    return val.mean(), val.std() / np.sqrt(n_paths)


if __name__ == "__main__":
    Fg = make_F_grid(n=61); rg = make_rho_grid(n=21); Lg = make_L_grid(n=13); ag = make_a_grid(n=41)
    opt = solve("optimize", Fg=Fg, rg=rg, Lg=Lg, ag=ag, n_quad=3)
    pol = opt["policy"]
    ti = 22; iF = int(np.argmin(np.abs(Fg - 0.9))); jr = int(np.argmin(np.abs(rg - 3.0)))
    print("a*(L) at t=22, F=0.9, rho~3   (fund fully when small; taper as the plan scales up):")
    for Lt in (0.1, 0.3, 1.0, 3.0, 8.0):
        kL = int(np.argmin(np.abs(Lg - Lt)))
        print(f"   L={Lt:>4}:  a*={pol[ti, iF, jr, kL]:.3f}")
    print(f"   bang-bang={np.all((pol == 0) | (pol == 1))}   mean a*={pol.mean():.3f}")