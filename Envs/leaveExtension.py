"""
leave_extension.py -- "when the worker can leave": the baseline value function
(faithful to Belgian immediate vesting).

Leaving -> paid-up: contributions stop, liability hard-freezes (L'=L), reserve
keeps earning MU, WAP shortfall tested only at T, and the leaver KEEPS their full
accrued reserve (immediate vesting) -- no service scaling. The employee's utility
is the committed lifecycle objective over total replacement,
    V_employee = ANNUITY * u( (RR_LEGAL + max(F,1)/(ANNUITY*rho)) / RR_TARGET ),
credited on whatever pot the reserve supports. Short-tenure leavers get little
because their pot is small (fewer contributions), which is the vesting-faithful
form of proportionality.

The additive legal baseline breaks the closed-form CRRA factorisation, so the
paid-up companion value Phi is computed by a direct actionless backward pass.
Parameters read from dp.* dynamically so sweeps propagate.
"""

import numpy as np
import DynPro as dp


# Churn now lives with the rest of the model in DynPro; re-exported so existing
# lx.tenure_hazard / lx.survival call sites keep working.
tenure_hazard = dp.tenure_hazard
survival = dp.survival


def _terminal_leaver(Fg, rg):
    """Leaver terminal: EMPLOYER shortfall liability ONLY -- the leaver's employee
    benefit is NOT credited to the objective (they have left). The employer still
    owes any WAP guarantee shortfall on the paid-up reserve at maturity."""
    Fc = Fg[:, None]; rc = rg[None, :]
    empr = (1.0 - dp.LAMBDA) * np.maximum(1.0 - Fc, 0.0) / rc
    return np.exp(-dp.DISC * dp.T) * (-empr)


def _gh_1d(n):
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)


def paidup_service(Fg, rg, n_quad=9):
    """Per-leave-cohort paid-up value Phi[tau](F,rho). The leaver KEEPS their full
    vested pot (WAP immediate vesting), but the objective judges it against a
    SERVICE-PRO-RATED TARGET on the second-pillar gap only:
        target_tau = RR_LEGAL + (tau/T)*(RR_TARGET - RR_LEGAL).
    Full-career stayer (tau=T) faces the full target; a short-tenure leaver a low
    one. Employee leg discounted at DISC_EMP, employer shortfall at DISC_ER."""
    z1, w1 = _gh_1d(n_quad); NF, NR = len(Fg), len(rg)
    Phi = np.empty((dp.T + 1, NF, NR))
    for tau in range(0, dp.T + 1):
        m = dp.T - tau; s = tau / dp.T
        target = dp.RR_LEGAL + s * (dp.RR_TARGET - dp.RR_LEGAL)                     # pro-rated target (gap only)
        Fp = Fg[:, None] * np.exp(dp.MU * m + dp.SIGMA_R * np.sqrt(max(m, 1e-9)) * z1[None, :])  # (NF,nq)
        rp = rg * (1.0 + dp.W) ** m                                                 # (NR,)
        rr2 = np.maximum(Fp, 1.0)[:, None, :] / (dp.ANNUITY * rp[None, :, None])    # FULL vested pot
        rrtot = dp.RR_LEGAL + rr2
        emp = dp.LAMBDA * dp.ANNUITY * dp.u(rrtot / target) * np.exp(-dp.DISC_EMP * dp.T)
        short = np.maximum(1.0 - Fp, 0.0)[:, None, :] / rp[None, :, None]
        empr = (1.0 - dp.LAMBDA) * short * np.exp(-dp.DISC_ER * dp.T)
        Phi[tau] = ((emp - empr) * w1[None, None, :]).sum(axis=2)
    return Phi


def solve_retention(hazard=tenure_hazard, Fg=None, rg=None, ag=None, n_quad=5, betas=None):
    """Backward induction for the committed (churn-aware) objective.

    `betas`: optional list of softmax temperatures. The objective, the value
    function V and the hard-optimal `policy` are identical whether or not it is
    given -- each beta only adds a SIGNAL READOUT of the same Q-values the
    argmax already computes:
        a_soft(t,F,rho) = sum_a a * softmax(Q(t,F,rho,a) / beta),
    i.e. the contribution responds smoothly to how much value the state-action
    actually carries, instead of snapping to the argmax corner. beta -> 0
    recovers the hard policy; larger beta blends near-tied actions. This is a
    presentation/extraction layer, NOT a change of objective: rolling a_soft
    forward is deliberately sub-optimal and the value gap to `policy` is the
    price of that smoothness.
    """
    if Fg is None: Fg = dp.make_F_grid(n=145)
    if rg is None: rg = dp.make_rho_grid(n=61)
    if ag is None: ag = dp.make_a_grid(n=31)
    zR, zL, wq = dp.gauss_hermite_2d(n_quad)
    lrg = np.log(rg); NF, NR, Q = len(Fg), len(rg), len(wq)
    Phi = paidup_service(Fg, rg)
    ag = np.asarray(ag, float)
    betas = list(betas) if betas else []

    V = np.empty((dp.T + 1, NF, NR)); V[dp.T] = dp.terminal(Fg, rg)
    policy = np.empty((dp.T, NF, NR))
    soft = {b: np.empty((dp.T, NF, NR)) for b in betas}
    for t in range(dp.T - 1, -1, -1):
        h = float(hazard(t))
        Vb = (1.0 - h) * V[t + 1] + h * Phi[t + 1]
        best = np.full((NF, NR), -np.inf); abest = np.zeros((NF, NR))
        Qstack = np.empty((len(ag), NF, NR)) if betas else None
        for i, a in enumerate(ag):
            l = a * dp.GAMMA * rg
            Fp = dp.F_next(Fg[:, None, None], l[None, :, None],
                           zR[None, None, :], zL[None, None, :])
            rp = dp.rho_next(rg[:, None], l[:, None], zL[None, :])
            lrq = np.broadcast_to(np.log(rp)[None, :, :], (NF, NR, Q))
            vi = dp.bilinear(Fg, lrg, Vb, Fp, lrq)
            cont = (vi * wq[None, None, :]).sum(axis=2)
            flow = -(1.0 - dp.LAMBDA) * a * dp.GAMMA * (1.0 + dp.W) ** (-(dp.T - t)) * np.exp(-dp.DISC_ER * t)
            Qv = flow + cont
            if betas: Qstack[i] = Qv
            upd = Qv > best; best = np.where(upd, Qv, best); abest = np.where(upd, a, abest)
        V[t] = best; policy[t] = abest
        for b in betas:
            w = np.exp((Qstack - Qstack.max(axis=0, keepdims=True)) / b)
            w /= w.sum(axis=0, keepdims=True)
            soft[b][t] = (w * ag[:, None, None]).sum(axis=0)
    out = dict(Fg=Fg, rg=rg, ag=ag, V=V, policy=policy)
    if betas: out["policy_soft"] = soft
    return out