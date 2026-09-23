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
# Sourced from economy.py, the single source of truth shared with the tabular
# rung. They are imported as module globals on purpose: every function below
# reads them as bare globals, and the sweep harnesses rebind them HERE (e.g.
# setattr(dp, "LAMBDA", x); see tests/dynpro/common.restore) to vary one
# parameter at a time. Rebinding dp.X does not touch economy.X, so patching the
# oracle never silently moves the tabular environment.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from economy import (T, G, MU, W, DISC_EMP, DISC_ER, DISC, SIGMA_R, SIGMA_L,
                     GAMMA, LAMBDA, S0, ETA, ANNUITY, RR_LEGAL, SATIATE, RR_TARGET)


def u(x):
    """CRRA utility, eta != 1."""
    return np.power(x, 1.0 - ETA) / (1.0 - ETA)


# --- transitions ----------------------------------------------------------
def F_next(F, l, zR, zL):
    return (F + l) / (1.0 + l) * np.exp((MU - G) + SIGMA_R * zR - SIGMA_L * zL)

def rho_next(rho, l, zL):
    return (1.0 + W) * rho / ((1.0 + l) * np.exp(G + SIGMA_L * zL))


def gauss_hermite_2d(n=15):
    x, w = np.polynomial.hermite.hermgauss(n) # optimize the approximation of the integral using the gaussian quadrature: https://www.youtube.com/watch?v=Hu6yqs0R7GA
    z = np.sqrt(2.0) * x # the nodes rescaled to the standard-normal scale.
    om = w / np.sqrt(np.pi) # weigts should sum to one 
    ZR, ZL = np.meshgrid(z, z, indexing="ij") # tensor-product grid
    return ZR.ravel(), ZL.ravel(), np.outer(om, om).ravel() # The outer product gives the joint weights.This product structure is valid only because the joint density of two independent standard normals factorises,


# --- grids ----------------------------------------------------------------
def make_F_grid(F_max=3.0, n=241):
    g = np.linspace(0.0, F_max, n)
    assert abs(g[np.argmin(np.abs(g - 1.0))] - 1.0) < 1e-12, "F=1 must be a node"
    return g

def make_rho_grid(lo=0.3, hi=35.0, n=81):
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))

def make_a_grid(n=41):
    return np.linspace(0.0, 1.0, n)

def grids(nF=145, nR=61, na=31):
    return make_F_grid(n=nF), make_rho_grid(n=nR), make_a_grid(n=na)


# --- churn: Belgian tenure hazard -----------------------------------------
def tenure_hazard(t, h0=0.12, hinf=0.025, tau=7.0):
    """Belgian tenure hazard: ~12%/yr early -> ~2.5%/yr long-tenure floor, giving
    ~45% staying 10+ years and ~15% a full career (avg tenure ~11y, ~50% at 10+y;
    Goulart & Oesch 2024, OECD/Eurostat)."""
    return hinf + (h0 - hinf) * np.exp(-t / tau)


def survival(hazard):
    p = np.ones(T + 1)
    for t in range(T):
        p[t + 1] = p[t] * (1.0 - hazard(t))
    return p


# --- policies and plan-entry states ---------------------------------------
def const_policy(a, n_years, nF, nR):
    return np.full((n_years, nF, nR), float(a))


def new_plan_init(n, rng, rho_lo=15.0, rho_hi=34.0, F_sd=0.08):
    """Fresh plans: F0 ~ 1, high rho (liability small relative to salary)."""
    F0 = np.clip(rng.normal(1.0, F_sd, n), 0.5, 1.5)
    rho0 = np.exp(rng.uniform(np.log(rho_lo), np.log(rho_hi), n))
    return F0, np.ones(n), rho0


def sample_entry(rng, n, F_mu=1.0, F_sd=0.15, F_clip=(0.4, 2.5), rho_lo=3.0, rho_hi=30.0):
    """Placeholder plan-entry distribution (NOT calibrated -- swap for DB2P
    aggregate stats when available). L0=1; the model is scale-free."""
    F0 = np.clip(rng.normal(F_mu, F_sd, n), *F_clip)
    rho0 = np.exp(rng.uniform(np.log(rho_lo), np.log(rho_hi), n))
    return F0, np.ones(n), rho0


# --- bilinear interpolation over (F, log rho) -----------------------------
def bilinear(Fg, lrg, V, Fq, lrq):
    Fq = np.clip(Fq, Fg[0], Fg[-1]); lrq = np.clip(lrq, lrg[0], lrg[-1])
    iF = np.clip(np.searchsorted(Fg, Fq) - 1, 0, len(Fg) - 2)
    iR = np.clip(np.searchsorted(lrg, lrq) - 1, 0, len(lrg) - 2)
    tF = (Fq - Fg[iF]) / (Fg[iF + 1] - Fg[iF])
    tR = (lrq - lrg[iR]) / (lrg[iR + 1] - lrg[iR])
    return (V[iF, iR] * (1 - tF) * (1 - tR) + V[iF + 1, iR] * tF * (1 - tR)
            + V[iF, iR + 1] * (1 - tF) * tR + V[iF + 1, iR + 1] * tF * tR)


def paidup_service(Fg, rg):
    """Per-leave-cohort paid-up value Phi[tau](F,rho), in closed form.

    On departure the contract goes paid-up: contributions cease, the reserve goes
    on earning the locked credited return MU, and the liability hard-freezes
    (L'=L) -- the WAP rate no longer applies to someone who has gone. The frozen
    floor does not lapse: the employee still receives max(R_T, L_T) and the
    employer still settles any shortfall against it at T.

    Freezing the contract freezes its risk too. With L fixed, MU locked and
    salary growth deterministic, nothing random happens between the freeze and
    retirement, so Phi carries NO expectation -- it is a single roll-forward over
    the remaining m = T - tau years (no quadrature, no backward pass):
        F   -> F * exp(MU*m)      (R compounds at MU; L frozen, so no -G*m)
        rho -> rho * (1+W)^m      (salary grows; L frozen)

    The leaver keeps their FULL vested pot (immediate vesting), but it is judged
    against a service-pro-rated target on the occupational gap only:
        target_tau = RR_LEGAL + (tau/T)*(RR_TARGET - RR_LEGAL),
    so only a full-career stayer faces the full target and tau=T reproduces
    terminal() exactly. The legal first pillar is not pro-rated -- it is earned
    across the whole career, not with this employer.
    """
    NF, NR = len(Fg), len(rg)
    Phi = np.empty((T + 1, NF, NR))
    for tau in range(0, T + 1):
        m = T - tau; s = tau / T
        target = RR_LEGAL + s * (RR_TARGET - RR_LEGAL)          # pro-rated target (gap only)
        Fp = Fg * np.exp(MU * m)                                # (NF,) deterministic: no asset shock
        rp = rg * (1.0 + W) ** m                                # (NR,)
        rr2 = np.maximum(Fp, 1.0)[:, None] / (ANNUITY * rp[None, :])   # FULL vested pot
        rrtot = RR_LEGAL + rr2
        emp = LAMBDA * ANNUITY * u(rrtot / target) * np.exp(-DISC_EMP * T)
        short = np.maximum(1.0 - Fp, 0.0)[:, None] / rp[None, :]
        empr = (1.0 - LAMBDA) * short * np.exp(-DISC_ER * T)
        Phi[tau] = emp - empr
    return Phi


def terminal(Fg, rg):
    Fc = Fg[:, None]; rc = rg[None, :]
    rr = RR_LEGAL + np.maximum(Fc, 1.0) / (rc * ANNUITY)
    if SATIATE: rr = np.minimum(rr, RR_TARGET)                      # satiation: no reward above target
    emp = LAMBDA * ANNUITY * u(rr / RR_TARGET) * np.exp(-DISC_EMP * T)          # employee rate
    empr = (1.0 - LAMBDA) * np.maximum(1.0 - Fc, 0.0) / rc * np.exp(-DISC_ER * T)  # employer rate
    return emp - empr


# --- backward induction (mode = 'optimize' | 'evaluate') ------------------
def solve(mode="optimize", plan_rule=None, Fg=None, rg=None, ag=None, n_quad=5,
          hazard=tenure_hazard, betas=None):
    """Backward induction for the committed (churn-aware) objective.

    `hazard`: leaving is a hazard on the horizon, not a state variable. At each
    step the continuation is the branch-blend
        Vb = (1 - h(t)) * V[t+1] + h(t) * Phi[t+1],
    where Phi is the paid-up companion value (see paidup_service). Both branches
    pay year t's contribution and land at the same in-force state -- the worker
    earned the year -- so the freeze applies only from t+1. Pass hazard=None for
    the no-churn benchmark, which reduces this to the plain oracle exactly.

    `betas`: optional list of softmax temperatures. The objective, V and the
    hard-optimal `policy` are identical whether or not it is given -- each beta
    only adds a SIGNAL READOUT of the same Q-values the argmax already computes:
        a_soft(t,F,rho) = sum_a a * softmax(Q(t,F,rho,a) / beta),
    i.e. the contribution responds smoothly to how much value the state-action
    actually carries, instead of snapping to the argmax corner. beta -> 0
    recovers the hard policy; larger beta blends near-tied actions. This is a
    presentation/extraction layer, NOT a change of objective: rolling a_soft
    forward is deliberately sub-optimal and the value gap to `policy` is the
    price of that smoothness. Only meaningful in 'optimize' mode.
    """
    if Fg is None: Fg = make_F_grid(n=145)
    if rg is None: rg = make_rho_grid(n=61)
    if ag is None: ag = make_a_grid(n=31)
    ag = np.asarray(ag, float)                 # betas path needs ag[:, None, None]
    betas = list(betas) if betas else []
    zR, zL, wq = gauss_hermite_2d(n_quad)
    lrg = np.log(rg)
    NF, NR, Q = len(Fg), len(rg), len(wq)
    Phi = paidup_service(Fg, rg) if hazard is not None else None

    V = np.empty((T + 1, NF, NR))
    policy = np.empty((T, NF, NR))
    soft = {b: np.empty((T, NF, NR)) for b in betas}
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
        flow = -(1.0 - LAMBDA) * A * GAMMA * (1.0 + W) ** (-(T - t)) * np.exp(-DISC_ER * t)
        return flow + cont

    for t in range(T - 1, -1, -1):
        Vnext = V[t + 1]
        if hazard is not None:
            h = float(hazard(t))
            Vnext = (1.0 - h) * V[t + 1] + h * Phi[t + 1]
        if mode == "optimize":
            best = np.full((NF, NR), -np.inf); abest = np.zeros((NF, NR))
            Qstack = np.empty((len(ag), NF, NR)) if betas else None
            for i, a in enumerate(ag):
                Q_ = bellman_scalar_a(t, a, Vnext)
                if betas: Qstack[i] = Q_
                upd = Q_ > best
                best = np.where(upd, Q_, best); abest = np.where(upd, a, abest)
            V[t] = best; policy[t] = abest
            for b in betas:
                w = np.exp((Qstack - Qstack.max(axis=0, keepdims=True)) / b)
                w /= w.sum(axis=0, keepdims=True)
                soft[b][t] = (w * ag[:, None, None]).sum(axis=0)
        else:
            A = np.clip(plan_rule(t, Fg[:, None], rg[None, :]) * np.ones((NF, NR)), 0.0, 1.0)
            V[t] = bellman_field_a(t, A, Vnext); policy[t] = A
    out = dict(Fg=Fg, rg=rg, ag=ag, V=V, policy=policy)
    if betas: out["policy_soft"] = soft
    return out


# --- forward evaluation of a policy (the committed scoring model) ---------
def simulate(policy, Fg, rg, R0=1.0, L0=1.0, S0=None, band=None, n_paths=30000,
            seed=7, hazard=tenure_hazard, track=False):
    """Canonical forward Monte-Carlo of a reduced policy a*(t,F,rho) under the
    committed model: Belgian churn (immediate-vesting, paid-up leavers), a split
    discount (employee at DISC_EMP, employer at DISC_ER), and the service-pro-rated
    adequacy target
        target_tau = RR_LEGAL + (tau/T)*(RR_TARGET - RR_LEGAL),
    so a full-career stayer is judged against RR_TARGET and a leaver with tau years
    of service against a proportionally lower bar.

    R0/L0/S0 may be scalars (a single anchor state) or arrays (a cohort / sampled
    entry distribution); S0 defaults to the top of the rho-grid. `band=(lo,hi)`
    clips the applied action to a contribution band (banded-DCA); None means the
    unconstrained [0,1] rule.
    """
    lrg = np.log(rg)
    rng = np.random.default_rng(seed)
    n = n_paths
    R = np.full(n, 1.0) * np.asarray(R0); L = np.full(n, 1.0) * np.asarray(L0)
    S = np.full(n, 1.0) * np.asarray(S0 if S0 is not None else rg[-1])
    ST = S * (1.0 + W) ** T
    lo, hi = (0.0, 1.0) if band is None else band
    present = np.ones(n, bool); leave_t = np.full(n, T, float)
    cost = np.zeros(n); a_sum = np.zeros(n)
    a_by_t = np.full(T, np.nan); rho_med = np.zeros(T)
    frac = np.zeros(T); c_by = np.zeros(T)
    for t in range(T):
        F = R / L; rho = S / L
        a = np.clip(bilinear(Fg, lrg, policy[t], F, np.log(rho)), lo, hi)
        a = np.where(present, a, 0.0)
        frac[t] = present.mean()
        c_by[t] = (a[present] * GAMMA).mean() * 100 if present.any() else 0.0
        if track:
            a_by_t[t] = a[present].mean() if present.any() else np.nan
            rho_med[t] = np.median(rho[present]) if present.any() else np.nan
        a_sum += a
        c = a * GAMMA * S
        cost += np.where(present, (c / ST) * np.exp(-DISC_ER * t), 0.0)
        zR = rng.standard_normal(n); zL = rng.standard_normal(n)
        R = np.where(present, (R + c) * np.exp(MU + SIGMA_R * zR), R * np.exp(MU + SIGMA_R * zR))
        L = np.where(present, (L + c) * np.exp(G + SIGMA_L * zL), L)
        S = S * (1.0 + W)
        lv = present & (rng.random(n) < hazard(t))
        leave_t = np.where(lv, t + 1, leave_t); present = present & ~lv
    payout = np.maximum(R, L); short = np.maximum(L - R, 0.0)
    cost += (short / ST) * np.exp(-DISC_ER * T)
    RR2 = payout / (ANNUITY * ST)
    svc = np.minimum(leave_t / T, 1.0)
    target = RR_LEGAL + svc * (RR_TARGET - RR_LEGAL)
    benefit_paths = np.exp(-DISC_EMP * T) * ANNUITY * u((RR_LEGAL + RR2) / target)
    stay = leave_t >= T
    RRtot = RR_LEGAL + RR2
    out = dict(
        benefit=float(benefit_paths.mean()), cost=float(cost.mean()),
        joint=float(LAMBDA * benefit_paths.mean() - (1 - LAMBDA) * cost.mean()),
        mean_a=float((a_sum / T).mean()), c_by=c_by, frac=frac,
        avg=float((c_by * frac).sum() / frac.sum()) if frac.sum() > 0 else np.nan,
        RR=RR2, RR_tot=RRtot, tot=float(np.median(RRtot)), stay=stay,
        sty=float(np.median(RRtot[stay])) if stay.any() else np.nan,
        lea=float(np.median(RRtot[~stay])) if (~stay).any() else np.nan,
    )
    if track:
        out.update(a_by_t=a_by_t, rho_med=rho_med)
    return out



