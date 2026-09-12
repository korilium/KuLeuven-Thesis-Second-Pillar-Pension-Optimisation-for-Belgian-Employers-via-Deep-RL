"""
dp_oracle_rr.py -- Rung-2 benchmark: DP oracle over (t, F, rho).

Employee objective (option E reduced to A): mortality-weighted lifecycle CRRA over the
ANNUITISED total replacement rate  RR_total = RR_LEGAL + max(F,1)/(ANNUITY*rho),  judged
against a total-adequacy target RR_TARGET:
        V_employee = ANNUITY * u(RR_total / RR_TARGET)  (lifetime; the front ANNUITY is the
        money-metric that matches the employer's ANNUITY-scaled capital cost -- NOT double counting).
Employer leg: linear WAP shortfall  max(1-F,0)/rho.  Asset (z_R) and guarantee (z_L)
shocks; Gauss-Hermite quadrature; reduced state (t, F, rho).
"""

import numpy as np

# --- parameters -----------------------------------------------------------
T = 45
G, MU, W = 0.03, 0.03, 0.025
DISC_EMP = 0.025     # EMPLOYEE discount: values future retirement income at the risk-free/OLO rate
DISC_ER  = 0.05      # EMPLOYER discount: firm cost of capital (contributions + shortfall)
DISC = DISC_ER       # legacy alias
SIGMA_R, SIGMA_L = 0.05, 0.02        # asset shock, guarantee shock
GAMMA, LAMBDA, S0 = 0.15, 0.5, 1.0
ETA = 2.0                            # CRRA curvature over the replacement rate (eta != 1)
ANNUITY = 15.0                       # actuarial annuity factor: capital -> annual pension
                                     # (Belgian life expectancy at 65 ~20y, ~2% technical rate,
                                     #  mortality-adjusted). RR is ANNUAL: pension / final salary.
RR_LEGAL = 0.43                      # 1st-pillar (legal) gross replacement, Belgian private-sector
                                     # average earner (OECD PaaG); the 2nd pillar sits ON TOP.
SATIATE = False                      # if True, cap RR at RR_TARGET in the utility (no reward for overshoot)
RR_TARGET = 0.70                     # total-adequacy target across all pillars (OECD/EU ~70%);
                                     # the employee's lifecycle utility is judged against this.


def u(x):
    """CRRA utility, eta != 1."""
    return np.power(x, 1.0 - ETA) / (1.0 - ETA)


# --- transitions ----------------------------------------------------------
def F_next(F, l, zR, zL):
    return (F + l) / (1.0 + l) * np.exp((MU - G) + SIGMA_R * zR - SIGMA_L * zL)

def rho_next(rho, l, zL):
    return (1.0 + W) * rho / ((1.0 + l) * np.exp(G + SIGMA_L * zL))


def gauss_hermite_2d(n=15):
    x, w = np.polynomial.hermite.hermgauss(n)
    z = np.sqrt(2.0) * x
    om = w / np.sqrt(np.pi)
    ZR, ZL = np.meshgrid(z, z, indexing="ij")
    return ZR.ravel(), ZL.ravel(), np.outer(om, om).ravel()


# --- grids ----------------------------------------------------------------
def make_F_grid(F_max=3.0, n=241):
    g = np.linspace(0.0, F_max, n)
    assert abs(g[np.argmin(np.abs(g - 1.0))] - 1.0) < 1e-12, "F=1 must be a node"
    return g

def make_rho_grid(lo=0.3, hi=35.0, n=81):
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))

def make_a_grid(n=41):
    return np.linspace(0.0, 1.0, n)


# --- bilinear interpolation over (F, log rho) -----------------------------
def bilinear(Fg, lrg, V, Fq, lrq):
    Fq = np.clip(Fq, Fg[0], Fg[-1]); lrq = np.clip(lrq, lrg[0], lrg[-1])
    iF = np.clip(np.searchsorted(Fg, Fq) - 1, 0, len(Fg) - 2)
    iR = np.clip(np.searchsorted(lrg, lrq) - 1, 0, len(lrg) - 2)
    tF = (Fq - Fg[iF]) / (Fg[iF + 1] - Fg[iF])
    tR = (lrq - lrg[iR]) / (lrg[iR + 1] - lrg[iR])
    return (V[iF, iR] * (1 - tF) * (1 - tR) + V[iF + 1, iR] * tF * (1 - tR)
            + V[iF, iR + 1] * (1 - tF) * tR + V[iF + 1, iR + 1] * tF * tR)


def terminal(Fg, rg):
    Fc = Fg[:, None]; rc = rg[None, :]
    rr = RR_LEGAL + np.maximum(Fc, 1.0) / (rc * ANNUITY)
    if SATIATE: rr = np.minimum(rr, RR_TARGET)                      # satiation: no reward above target
    emp = LAMBDA * ANNUITY * u(rr / RR_TARGET) * np.exp(-DISC_EMP * T)          # employee rate
    empr = (1.0 - LAMBDA) * np.maximum(1.0 - Fc, 0.0) / rc * np.exp(-DISC_ER * T)  # employer rate
    return emp - empr


# --- backward induction (mode = 'optimize' | 'evaluate') ------------------
def solve(mode="optimize", plan_rule=None, Fg=None, rg=None, ag=None, n_quad=15):
    if Fg is None: Fg = make_F_grid()
    if rg is None: rg = make_rho_grid()
    if ag is None: ag = make_a_grid()
    zR, zL, wq = gauss_hermite_2d(n_quad)
    lrg = np.log(rg)
    NF, NR, Q = len(Fg), len(rg), len(wq)

    V = np.empty((T + 1, NF, NR))
    policy = np.empty((T, NF, NR))
    V[T] = terminal(Fg, rg)

    def bellman_scalar_a(t, a, Vnext):
        l = a * GAMMA * rg
        Fp = F_next(Fg[:, None, None], l[None, :, None], zR[None, None, :], zL[None, None, :])
        rp = rho_next(rg[:, None], l[:, None], zL[None, :])
        lrq = np.broadcast_to(np.log(rp)[None, :, :], (NF, NR, Q))
        vi = bilinear(Fg, lrg, Vnext, Fp, lrq)
        cont = (vi * wq[None, None, :]).sum(axis=2)
        flow = -(1.0 - LAMBDA) * a * GAMMA * (1.0 + W) ** (-(T - t)) * np.exp(-DISC_ER * t)
        return flow + cont

    def bellman_field_a(t, A, Vnext):
        l = A * GAMMA * rg[None, :]
        Fp = F_next(Fg[:, None, None], l[:, :, None], zR[None, None, :], zL[None, None, :])
        rp = rho_next(rg[None, :, None], l[:, :, None], zL[None, None, :])
        vi = bilinear(Fg, lrg, Vnext, Fp, np.log(rp))
        cont = (vi * wq[None, None, :]).sum(axis=2)
        flow = -(1.0 - LAMBDA) * A * GAMMA * (1.0 + W) ** (-(T - t)) * np.exp(-DISC * t)
        return flow + cont

    for t in range(T - 1, -1, -1):
        if mode == "optimize":
            best = np.full((NF, NR), -np.inf); abest = np.zeros((NF, NR))
            for a in ag:
                Q_ = bellman_scalar_a(t, a, V[t + 1])
                upd = Q_ > best
                best = np.where(upd, Q_, best); abest = np.where(upd, a, abest)
            V[t] = best; policy[t] = abest
        else:
            A = np.clip(plan_rule(t, Fg[:, None], rg[None, :]) * np.ones((NF, NR)), 0.0, 1.0)
            V[t] = bellman_field_a(t, A, V[t + 1]); policy[t] = A
    return dict(Fg=Fg, rg=rg, ag=ag, V=V, policy=policy)