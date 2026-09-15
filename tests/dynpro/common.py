"""Shared infrastructure for the tests/dynpro suites (sensitivity_suite.py,
contribution_schedule_suite.py, lambda_dial_suite.py): sys.path wiring to
Envs/, the canonical committed rollout, and small grid/plotting helpers.

"Committed" model = the churn-aware oracle (lx.solve_retention: Belgian
immediate-vesting, paid-up leavers) scored with a split discount (employee at
DISC_EMP, employer at DISC_ER) against a service-pro-rated adequacy target
    target_tau = RR_LEGAL + (tau/T)*(RR_TARGET - RR_LEGAL)
so a full-career stayer is judged against RR_TARGET and a leaver with tau
years of service against a proportionally lower bar. This is the model used
by the thesis's headline figures; every suite here scores its policies the
same way so results are mutually comparable.
"""
import os, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Envs"))
import DynPro as dp
import leaveExtension as lx

OUT = "figs"; DPI = 150
mpl.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

PARAMS = ["LAMBDA", "RR_TARGET", "RR_LEGAL", "ANNUITY", "GAMMA", "ETA", "G", "MU",
          "SIGMA_R", "DISC_EMP", "DISC_ER"]
LAB = {"LAMBDA": r"$\lambda$", "RR_TARGET": r"$RR^\star$", "RR_LEGAL": r"$RR_{\rm legal}$",
       "ANNUITY": r"$\ddot a$", "GAMMA": r"$\Gamma$", "ETA": r"$\eta$", "G": r"$G$",
       "MU": r"$\mu$", "SIGMA_R": r"$\sigma_R$", "DISC_EMP": r"$\delta_e$", "DISC_ER": r"$\delta_f$"}

_BASE = {k: getattr(dp, k) for k in PARAMS}


def restore():
    """Reset every dp.* global any suite might sweep back to its baseline value."""
    for k, v in _BASE.items(): setattr(dp, k, v)


def ensure_out():
    os.makedirs(OUT, exist_ok=True)


def grids(nF=145, nR=61, na=31):
    return dp.make_F_grid(n=nF), dp.make_rho_grid(n=nR), dp.make_a_grid(n=na)


def iso_rr(Fg, rg):
    FF, RRr = np.meshgrid(Fg, rg, indexing="ij")
    return np.maximum(FF, 1.0) / RRr


def solve(Fg, rg, ag, nq=5, betas=None):
    """The committed (churn-aware, paid-up service-pro-rated) policy oracle.
    `betas` additionally returns soft (signal) readouts of the same Q-values;
    it leaves V and the hard policy untouched -- see lx.solve_retention."""
    return lx.solve_retention(Fg=Fg, rg=rg, ag=ag, n_quad=nq, betas=betas)


def const_policy(a, T, nF, nR):
    return np.full((T, nF, nR), float(a))


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


def simulate(policy, Fg, rg, R0=1.0, L0=1.0, S0=None, band=None, n_paths=30000,
            seed=7, hazard=lx.tenure_hazard, track=False):
    """Canonical forward Monte-Carlo of a reduced policy a*(t,F,rho) under the
    committed model (see module docstring). R0/L0/S0 may be scalars (a single
    anchor state) or arrays (a cohort / sampled entry distribution); S0
    defaults to the top of the rho-grid. `band=(lo,hi)` clips the applied
    action to a contribution band (banded-DCA); None means the unconstrained
    [0,1] rule.
    """
    lrg = np.log(rg)
    rng = np.random.default_rng(seed)
    n = n_paths
    R = np.full(n, 1.0) * np.asarray(R0); L = np.full(n, 1.0) * np.asarray(L0)
    S = np.full(n, 1.0) * np.asarray(S0 if S0 is not None else rg[-1])
    ST = S * (1.0 + dp.W) ** dp.T
    lo, hi = (0.0, 1.0) if band is None else band
    present = np.ones(n, bool); leave_t = np.full(n, dp.T, float)
    cost = np.zeros(n); a_sum = np.zeros(n)
    a_by_t = np.full(dp.T, np.nan); rho_med = np.zeros(dp.T)
    frac = np.zeros(dp.T); c_by = np.zeros(dp.T)
    for t in range(dp.T):
        F = R / L; rho = S / L
        a = np.clip(dp.bilinear(Fg, lrg, policy[t], F, np.log(rho)), lo, hi)
        a = np.where(present, a, 0.0)
        frac[t] = present.mean()
        c_by[t] = (a[present] * dp.GAMMA).mean() * 100 if present.any() else 0.0
        if track:
            a_by_t[t] = a[present].mean() if present.any() else np.nan
            rho_med[t] = np.median(rho[present]) if present.any() else np.nan
        a_sum += a
        c = a * dp.GAMMA * S
        cost += np.where(present, (c / ST) * np.exp(-dp.DISC_ER * t), 0.0)
        zR = rng.standard_normal(n); zL = rng.standard_normal(n)
        R = np.where(present, (R + c) * np.exp(dp.MU + dp.SIGMA_R * zR), R * np.exp(dp.MU + dp.SIGMA_R * zR))
        L = np.where(present, (L + c) * np.exp(dp.G + dp.SIGMA_L * zL), L)
        S = S * (1.0 + dp.W)
        lv = present & (rng.random(n) < hazard(t))
        leave_t = np.where(lv, t + 1, leave_t); present = present & ~lv
    payout = np.maximum(R, L); short = np.maximum(L - R, 0.0)
    cost += (short / ST) * np.exp(-dp.DISC_ER * dp.T)
    RR2 = payout / (dp.ANNUITY * ST)
    svc = np.minimum(leave_t / dp.T, 1.0)
    target = dp.RR_LEGAL + svc * (dp.RR_TARGET - dp.RR_LEGAL)
    benefit_paths = np.exp(-dp.DISC_EMP * dp.T) * dp.ANNUITY * dp.u((dp.RR_LEGAL + RR2) / target)
    stay = leave_t >= dp.T
    RRtot = dp.RR_LEGAL + RR2
    out = dict(
        benefit=float(benefit_paths.mean()), cost=float(cost.mean()),
        joint=float(dp.LAMBDA * benefit_paths.mean() - (1 - dp.LAMBDA) * cost.mean()),
        mean_a=float((a_sum / dp.T).mean()), c_by=c_by, frac=frac,
        avg=float((c_by * frac).sum() / frac.sum()) if frac.sum() > 0 else np.nan,
        RR=RR2, RR_tot=RRtot, tot=float(np.median(RRtot)), stay=stay,
        sty=float(np.median(RRtot[stay])) if stay.any() else np.nan,
        lea=float(np.median(RRtot[~stay])) if (~stay).any() else np.nan,
    )
    if track:
        out.update(a_by_t=a_by_t, rho_med=rho_med)
    return out
