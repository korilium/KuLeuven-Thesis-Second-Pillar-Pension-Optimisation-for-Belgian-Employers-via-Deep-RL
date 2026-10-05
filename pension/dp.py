"""
dp_oracle_rr.py -- Rung-2 benchmark: DP oracle over (t, F, rho).

Employee objective (option E reduced to A): mortality-weighted lifecycle CRRA over the
ANNUITISED total replacement rate  RR_total = RR_LEGAL + max(F,1)/(ANNUITY*rho),  judged
against a total-adequacy target RR_TARGET:
        V_employee = target * ANNUITY * u(RR_total / target)   (lifetime).
        u is normalized (u(1)=0, u'(1)=1), so near the target u ~ (RR - target)/target, a
        dimensionless relative shortfall. Multiplying by `target` puts it in annual
        replacement-rate units and by ANNUITY converts that flow to capital, i.e. the
        employer's units (final-salary-years). The exchange rate is therefore whatever
        denominator defines x: RR_TARGET for a full-career stayer, and the SERVICE-PRO-RATED
        target_tau for a leaver -- using a flat RR_TARGET would over-weight short-tenure
        leavers by RR_TARGET/target_tau (up to ~1.63x at tau=0).
Employer leg: linear WAP shortfall  max(1-F,0)/rho.  (This is the "baseline" objective;
every leg is swappable -- see objective.py and the OBJECTIVE switch.)  Asset (z_R) and guarantee (z_L)
shocks; Gauss-Hermite quadrature; reduced state (t, F, rho).
"""

import numpy as np

# --- parameters -----------------------------------------------------------
# Every function takes the model parameters as an explicit `p`
# (pension.params.Params); p=None means DEFAULT, the committed calibration. There
# is no module state: two calls with different p never interact, so sweeps need no
# restore bookkeeping and independent economies can run side by side.
from pension.params import Params, DEFAULT
from pension.dynamics import Exogenous, State, step, settle
import pension.economy as _economy      # draw_rate_scenarios: the rate engine is loaded only on use
import pension.objective as _objective_mod
from pension.objective import OBJECTIVES, Objective


def _params(p):
    """p if given, else DEFAULT."""
    return DEFAULT if p is None else p


def _objective(obj=None, p=None):
    """The objective in force: obj if given (a name or an Objective), else the
    switch p.OBJECTIVE (default "baseline", the committed model). See
    objective.py for the parts and the registry."""
    p = _params(p)
    return _objective_mod.resolve(obj, p.OBJECTIVE)


def u(x, p=None):
    """The normalized CRRA utility at p.ETA (see objective.u). Kept for callers
    that read dp.u directly; the objective itself lives in objective.py."""
    p = _params(p)
    return _objective_mod.u(x, p.ETA)


# --- transitions ----------------------------------------------------------
def F_next(F, l, zR, zL, p=None):
    p = _params(p)
    return (F + l) / (1.0 + l) * np.exp((p.MU - p.G) + p.SIGMA_R * zR - p.SIGMA_L * zL)

def rho_next(rho, l, zL, p=None):
    p = _params(p)
    return (1.0 + p.W) * rho / ((1.0 + l) * np.exp(p.G + p.SIGMA_L * zL))


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

def make_rho_grid(lo=0.01, hi=35.0, n=91):
    """Log-spaced grid for rho = S/L.

    lo=0.01 (not 0.3) because careers genuinely reach it: under heavy funding the
    median rho falls to ~0.13 by t=44, and bilinear() CLIPS anything below the
    floor. Since RR2 = max(F,1)/(ANNUITY*rho), clipping rho from below caps the
    attainable replacement rate, so a floor of 0.3 made the solver blind to the
    outcomes that heavy contribution actually produces -- it under-valued high-a
    policies by up to 57% and suppressed stayer RR by ~28pp. Headline results are
    converged in n from ~51 upward at this range."""
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))

def make_a_grid(n=41):
    return np.linspace(0.0, 1.0, n)

def grids(nF=145, nR=91, na=31):
    return make_F_grid(n=nF), make_rho_grid(n=nR), make_a_grid(n=na)


# --- churn: Belgian tenure hazard -----------------------------------------
def tenure_hazard(t, h0=0.12, hinf=0.025, tau=7.0):
    """Belgian tenure hazard: ~12%/yr early -> ~2.5%/yr long-tenure floor, giving
    ~45% staying 10+ years and ~15% a full career (avg tenure ~11y, ~50% at 10+y;
    Goulart & Oesch 2024, OECD/Eurostat)."""
    return hinf + (h0 - hinf) * np.exp(-t / tau)


def survival(hazard, p=None):
    p = _params(p)
    surv = np.ones(p.T + 1)
    for t in range(p.T):
        surv[t + 1] = surv[t] * (1.0 - hazard(t))
    return surv


# --- policies and plan-entry states ---------------------------------------
def const_policy(a, n_years, nF, nR):
    return np.full((n_years, nF, nR), float(a))


def schedule_policy(a_of_t, nF, nR, n_years=None, p=None):
    """Lift a STATE-INDEPENDENT schedule a(t) to a (T, nF, nR) policy array.

    The time-varying generalisation of const_policy, for market plan designs whose
    contribution depends on the year alone (a flat percentage of salary, an
    age-related scale) and not on how well funded the plan happens to be. Such a
    design is scale-free, so it needs no extra state: simulate() reads it through
    the same bilinear lookup as an optimised policy, which is what makes the two
    directly comparable.

    `a_of_t` is a callable t -> a or a length-T sequence, in CAPACITY-FRACTION
    units (contribution = a*GAMMA*S), not in percent of salary: convert a
    percentage rate with a = rate / GAMMA.
    """
    n = _params(p).T if n_years is None else n_years
    a = np.array([float(a_of_t(t)) for t in range(n)]) if callable(a_of_t) else \
        np.asarray(a_of_t, float)
    assert a.shape == (n,), f"a_of_t must give {n} values, got {a.shape}"
    return np.broadcast_to(a[:, None, None], (n, nF, nR)).copy()


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


def paidup_service(Fg, rg, obj=None, p=None):
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

    The value of the outcome is the objective's (see objective.py): its employee
    leg at the pro-rated target and its shortfall cost, combined by its weights.
    """
    p = _params(p)
    obj = _objective(obj, p); w_emp, w_er = obj.weights(p)
    NF, NR = len(Fg), len(rg)
    Phi = np.empty((p.T + 1, NF, NR))
    for tau in range(0, p.T + 1):
        m = p.T - tau; s = tau / p.T
        target = p.RR_LEGAL + s * (p.RR_TARGET - p.RR_LEGAL)          # pro-rated target (gap only)
        Fp = Fg * np.exp(p.MU * m)                                # (NF,) deterministic: no asset shock
        rp = rg * (1.0 + p.W) ** m                                # (NR,)
        rr2 = np.maximum(Fp, 1.0)[:, None] / (p.ANNUITY * rp[None, :])   # FULL vested pot
        rrtot = p.RR_LEGAL + rr2
        emp = w_emp * obj.employee(rrtot, target, p) * np.exp(-p.DISC_EMP * p.T)
        short = np.maximum(1.0 - Fp, 0.0)[:, None] / rp[None, :]
        empr = w_er * obj.shortfall(short, p) * np.exp(-p.DISC_ER * p.T)
        Phi[tau] = emp - empr
    return Phi


def terminal(Fg, rg, obj=None, p=None):
    """Full-career stayer at T: the objective's employee leg at RR_TARGET minus its
    shortfall cost, combined by its weights (see objective.py)."""
    p = _params(p)
    obj = _objective(obj, p); w_emp, w_er = obj.weights(p)
    Fc = Fg[:, None]; rc = rg[None, :]
    rr = p.RR_LEGAL + np.maximum(Fc, 1.0) / (rc * p.ANNUITY)
    emp = w_emp * obj.employee(rr, p.RR_TARGET, p) * np.exp(-p.DISC_EMP * p.T)              # employee rate
    empr = w_er * obj.shortfall(np.maximum(1.0 - Fc, 0.0) / rc, p) * np.exp(-p.DISC_ER * p.T)  # employer rate
    return emp - empr


def _soft_readout(Qstack, ag, beta):
    """a_soft = sum_a a * softmax( (Q - max Q) / (beta * spread) ), where
    spread = max_a Q - min_a Q AT THAT STATE.

    beta is therefore RELATIVE, not in Q units: it is the fraction of the local
    Q-range over which actions get blended. This matters because Q is NOT
    scale-free in the reward -- rescaling the employee leg (e.g. introducing the
    RR_TARGET*ANNUITY multiplier) changes every Q-difference, so a fixed absolute
    beta would silently mean something different in each specification. Dividing
    by the local spread makes the readout exactly invariant to any affine
    rescaling Q -> alpha*Q + c: alpha cancels between numerator and spread, and c
    cancels in the difference.

    Read beta as "blend actions lying within this fraction of the state's full
    value range". beta -> 0 recovers the hard argmax; large beta -> uniform.
    Never called with beta == 0 (the argmax branch handles that), so there is no
    1/0. Where the spread is exactly 0 all actions are tied, the numerator is 0,
    and the weights come out uniform -- the right answer for a tie."""
    qmax = Qstack.max(axis=0, keepdims=True)
    spread = qmax - Qstack.min(axis=0, keepdims=True)
    denom = beta * np.where(spread > 0, spread, 1.0)
    w = np.exp((Qstack - qmax) / denom)
    w /= w.sum(axis=0, keepdims=True)
    return (w * ag[:, None, None]).sum(axis=0)


# --- backward induction (mode = 'optimize' | 'evaluate') ------------------
def certainty_equivalent(rates, p=None):
    """Constant-rate parameters that summarise a rate scenario for the DP.

    The backward induction lives on (t, F, rho); following path-wise rates would
    add the short rate AND the 24-month WAP window to the state. Instead the DP
    sees the regime through its moments:
        G, MU    -> scenario means of G_t and mu_t,
        SIGMA_L  -> dispersion of G_t around its mean (guarantee-rate risk),
        SIGMA_R  -> SIGMA_R_RATES and the dispersion of mu_t, added in quadrature.
    A crude moment match, not an equivalence: the forward simulate() then scores
    the resulting policy against the true path-wise rates."""
    p = _params(p)
    Gs, mus = np.asarray(rates["G"]), np.asarray(rates["mu"])
    return dict(G=float(Gs.mean()), MU=float(mus.mean()),
                SIGMA_L=float(Gs.std()),
                SIGMA_R=float(np.sqrt(p.SIGMA_R_RATES ** 2 + mus.var())))


def solve(mode="optimize", plan_rule=None, Fg=None, rg=None, ag=None, n_quad=5,
          hazard=tenure_hazard, beta=None, betas=None, rates=None, objective=None,
          p=None):
    """Backward induction for the committed (churn-aware) objective.

    `p` (pension.params.Params): the model parameters.

    `objective`: the value function -- None follows the OBJECTIVE switch (default
    "baseline"), or pass a registry name / an objective.Objective. The same object
    sets the terminal value, the leaver value and the contribution flow; its name
    is returned under "objective".

    `rates` follows the RATE_MODEL switch, exactly like simulate():
      * None with RATE_MODEL == "constant" (default): the original constant-rate
        oracle with the module's G, MU, SIGMA_R, SIGMA_L;
      * None with a rate model: RATE_CE_PATHS scenarios of that model are drawn
        (RATE_SEED, memoised) and the CERTAINTY-EQUIVALENT model is solved;
      * a scenario dict from economy.draw_rate_scenarios: certainty-equivalent
        to that scenario;
      * "constant": force the constant-rate oracle whatever the switch says.
    The certainty-equivalent overrides (see certainty_equivalent) are applied as
    p.replace(**ce) for this call only, and the moments used are returned under
    "ce". Under a rate model G, MU, SIGMA_L and SIGMA_R
    are therefore SET by the scenario: overriding them has no effect.
"""
    p = _params(p)
    if isinstance(rates, str) and rates == "constant":
        rates = None
    elif rates is None and p.RATE_MODEL != "constant":
        rates = _economy.draw_rate_scenarios(p.RATE_CE_PATHS, seed=p.RATE_SEED,
                                             model=p.RATE_MODEL, horizon=p.T, p=p)
    if rates is not None:
        ce = certainty_equivalent(rates, p)
        out = _solve(mode, plan_rule, Fg, rg, ag, n_quad, hazard, beta, betas, objective,
                     p=p.replace(**ce))
        out["ce"] = ce
        return out
    return _solve(mode, plan_rule, Fg, rg, ag, n_quad, hazard, beta, betas, objective, p=p)


def _solve(mode="optimize", plan_rule=None, Fg=None, rg=None, ag=None, n_quad=5,
           hazard=tenure_hazard, beta=None, betas=None, objective=None, p=None):
    """The constant-rate backward induction (see solve).

    `hazard`: leaving is a hazard on the horizon, not a state variable. At each
    step the continuation is the branch-blend
        Vb = (1 - h(t)) * V[t+1] + h(t) * Phi[t+1],
    where Phi is the paid-up companion value (see paidup_service). Both branches
    pay year t's contribution and land at the same in-force state -- the worker
    earned the year -- so the freeze applies only from t+1. Pass hazard=None for
    the no-churn benchmark, which reduces this to the plain oracle exactly.

    `beta` (defaults to p.BETA): the policy-EXTRACTION temperature, given
    RELATIVE to the local Q-spread -- "blend actions within this fraction of the
    state's full value range". See _soft_readout: this makes the readout invariant
    to any affine rescale of the reward, which a raw Q-unit temperature is not.
        beta == 0  -> `policy` is the hard argmax, the true optimum (default).
        beta  > 0  -> `policy` is the soft readout at that temperature, and the
                      argmax is still returned as `policy_hard`.
    Either way V[t] = max_a Q is untouched, so beta never changes the objective and
    the value gap between a soft policy and V remains the honest price of smoothing.

    `betas`: optional LIST of temperatures for comparing several at once. It leaves
    `policy` alone and returns a {beta: array} dict as `policy_soft`. Each beta only
    adds a SIGNAL READOUT of the same Q-values the argmax already computes:
        a_soft(t,F,rho) = sum_a a * softmax(Q(t,F,rho,a) / beta),
    i.e. the contribution responds smoothly to how much value the state-action
    actually carries, instead of snapping to the argmax corner. beta -> 0
    recovers the hard policy; larger beta blends near-tied actions. This is a
    presentation/extraction layer, NOT a change of objective: rolling a_soft
    forward is deliberately sub-optimal and the value gap to `policy` is the
    price of that smoothness. Only meaningful in 'optimize' mode.
    """
    if Fg is None: Fg = make_F_grid(n=145)
    if rg is None: rg = make_rho_grid(n=91)
    if ag is None: ag = make_a_grid(n=31)
    ag = np.asarray(ag, float)                 # soft path needs ag[:, None, None]
    betas = list(betas) if betas else []
    beta = p.BETA if beta is None else float(beta)   # module default; 0 = hard argmax
    need_Q = bool(betas) or beta > 0
    zR, zL, wq = gauss_hermite_2d(n_quad)
    lrg = np.log(rg)
    NF, NR, Q = len(Fg), len(rg), len(wq)
    p = _params(p)
    obj = _objective(objective, p); w_er = obj.weights(p)[1]
    Phi = paidup_service(Fg, rg, obj, p) if hazard is not None else None

    V = np.empty((p.T + 1, NF, NR))
    policy = np.empty((p.T, NF, NR))
    soft = {b: np.empty((p.T, NF, NR)) for b in betas}
    hard = np.empty((p.T, NF, NR)) if beta > 0 else None
    V[p.T] = terminal(Fg, rg, obj, p)

    def bellman_scalar_a(t, a, Vnext):
        l = a * p.GAMMA * rg
        Fp = F_next(Fg[:, None, None], l[None, :, None], zR[None, None, :], zL[None, None, :], p)
        rp = rho_next(rg[:, None], l[:, None], zL[None, :], p)
        lrq = np.broadcast_to(np.log(rp)[None, :, :], (NF, NR, Q))
        vi = bilinear(Fg, lrg, Vnext, Fp, lrq)
        cont = (vi * wq[None, None, :]).sum(axis=2)
        flow = -w_er * obj.contribution(a, t, p) * np.exp(-p.DISC_ER * t)
        return flow + cont

    def bellman_field_a(t, A, Vnext):
        l = A * p.GAMMA * rg[None, :]
        Fp = F_next(Fg[:, None, None], l[:, :, None], zR[None, None, :], zL[None, None, :], p)
        rp = rho_next(rg[None, :, None], l[:, :, None], zL[None, None, :], p)
        vi = bilinear(Fg, lrg, Vnext, Fp, np.log(rp))
        cont = (vi * wq[None, None, :]).sum(axis=2)
        flow = -w_er * obj.contribution(A, t, p) * np.exp(-p.DISC_ER * t)
        return flow + cont

    for t in range(p.T - 1, -1, -1):
        Vnext = V[t + 1]
        if hazard is not None:
            h = float(hazard(t))
            Vnext = (1.0 - h) * V[t + 1] + h * Phi[t + 1]
        if mode == "optimize":
            best = np.full((NF, NR), -np.inf); abest = np.zeros((NF, NR))
            Qstack = np.empty((len(ag), NF, NR)) if need_Q else None
            for i, a in enumerate(ag):
                Q_ = bellman_scalar_a(t, a, Vnext)
                if need_Q: Qstack[i] = Q_
                upd = Q_ > best
                best = np.where(upd, Q_, best); abest = np.where(upd, a, abest)
            V[t] = best
            if beta > 0:                       # extraction only -- V[t] is still max_a Q
                hard[t] = abest
                policy[t] = _soft_readout(Qstack, ag, beta)
            else:
                policy[t] = abest
            for b in betas:
                soft[b][t] = _soft_readout(Qstack, ag, b)
        else:
            A = np.clip(plan_rule(t, Fg[:, None], rg[None, :]) * np.ones((NF, NR)), 0.0, 1.0)
            V[t] = bellman_field_a(t, A, Vnext); policy[t] = A
    out = dict(Fg=Fg, rg=rg, ag=ag, V=V, policy=policy, objective=obj.name)
    if beta > 0: out["policy_hard"] = hard
    if betas: out["policy_soft"] = soft
    return out


# --- forward evaluation of a policy (the committed scoring model) ---------
def simulate(policy, Fg, rg, R0=1.0, L0=1.0, S0=None, band=None, n_paths=30000,
            seed=7, hazard=tenure_hazard, track=False, visits=False, rates=None,
            objective=None, p=None):
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

    `visits=True` additionally returns out["visits"], a (T, NF, NR) count of how
    many PRESENT paths occupy each (F, rho) cell in each year -- the occupancy of
    the state space. Sum over axis 0 for the pooled density. Cells are assigned by
    nearest grid node (in log-rho, matching the grid), with states off the grid
    clipped to the edge exactly as bilinear() does. This is what distinguishes a
    grid-mean (every cell weighted equally) from a path-weighted average.

    `objective` scores the outcome (None follows the OBJECTIVE switch; see
    objective.py): `benefit` is the discounted employee leg, `cost` the discounted
    employer leg (contributions + terminal shortfall, both in the objective's cost
    units), and `joint` their weighted difference -- the quantity solve() maximises.

    Rates. With RATE_MODEL == "constant" and rates=None this is the original
    constant-rate model, line for line: the reserve earns MU + SIGMA_R*zR, the
    liability grows at G + SIGMA_L*zL (vertical). Otherwise the rates come from
    `rates` (a dict from economy.draw_rate_scenarios, n_paths columns) or, if
    None, are drawn for RATE_MODEL with RATE_SEED, and per path:
      * the reserve earns the book yield:  R <- (R + c) * exp(mu[t] + SIGMA_R_RATES*zR);
      * the liability is HORIZONTAL: the opening L0 is locked at G[0] and each
        contribution at the G[t] of its payment year, until retirement. L is the
        sum of these vintages, so F = R/L and rho = S/L keep their meaning and
        the same (F, rho) policy is read off unchanged. No SIGMA_L shock: the
        guarantee risk is now G_t itself;
      * leavers go paid-up as before: L freezes, R compounds at the path's book
        yield without the excess-return shock.
    The year itself is pension.dynamics.step(); all randomness is one
    dynamics.Exogenous drawn from `seed` (per year: zR, zL, churn), the same in
    both regimes, so a degenerate scenario (G == G, mu == MU) with SIGMA_L = 0
    reproduces the constant branch.
    The output then also carries G_by_t / mu_by_t (path means), regime_B (share
    of paths with L_T > R_T), the terminal R_T / L_T per path, and the scenario
    itself under "rates".
    """
    p = _params(p)
    obj = _objective(objective, p); w_emp, w_er = obj.weights(p)
    if rates is None and p.RATE_MODEL != "constant":
        rates = _economy.draw_rate_scenarios(n_paths, seed=p.RATE_SEED, model=p.RATE_MODEL,
                                             horizon=p.T, p=p)
    exo = Exogenous.draw(p, n_paths, seed, rates)
    state = State.initial(p, exo, R0, L0, S0 if S0 is not None else rg[-1])
    ST = state.S * (1.0 + p.W) ** p.T
    lrg = np.log(rg)
    lo, hi = (0.0, 1.0) if band is None else band
    diag = _Diagnostics(p.T, Fg, lrg, track, visits)
    cost = np.zeros(n_paths); a_sum = np.zeros(n_paths)
    for t in range(p.T):
        present = state.present
        F, rho = state.F, state.rho
        a = np.clip(bilinear(Fg, lrg, policy[t], F, np.log(rho)), lo, hi)
        a = np.where(present, a, 0.0)
        diag.record(t, a, present, F, rho, p)
        a_sum += a
        cost += np.where(present, obj.contribution(a, t, p) * np.exp(-p.DISC_ER * t), 0.0)
        state, _ = step(state, a, exo, p, hazard)
    end = settle(state, p)
    cost += obj.shortfall(end["short"] / ST, p) * np.exp(-p.DISC_ER * p.T)
    RR2 = end["payout"] / (p.ANNUITY * ST)
    target = p.RR_LEGAL + end["svc"] * (p.RR_TARGET - p.RR_LEGAL)
    benefit_paths = np.exp(-p.DISC_EMP * p.T) * obj.employee(p.RR_LEGAL + RR2, target, p)
    stay = end["stay"]
    RRtot = p.RR_LEGAL + RR2
    frac, c_by = diag.frac, diag.c_by
    out = dict(
        benefit=float(benefit_paths.mean()), cost=float(cost.mean()),
        joint=float(w_emp * benefit_paths.mean() - w_er * cost.mean()),
        mean_a=float((a_sum / p.T).mean()), c_by=c_by, frac=frac,
        avg=float((c_by * frac).sum() / frac.sum()) if frac.sum() > 0 else np.nan,
        RR=RR2, RR_tot=RRtot, tot=float(np.median(RRtot)), stay=stay,
        sty=float(np.median(RRtot[stay])) if stay.any() else np.nan,
        lea=float(np.median(RRtot[~stay])) if (~stay).any() else np.nan,
    )
    out.update(diag.extras())
    if exo.rates:
        out.update(G_by_t=exo.G[:p.T].mean(axis=1), mu_by_t=exo.mu[:p.T].mean(axis=1),
                   regime_B=float((state.L > state.R).mean()), R_T=state.R, L_T=state.L,
                   rates=rates)
    return out


class _Diagnostics:
    """Per-year cohort statistics collected by simulate(): participation and mean
    contribution always; the mean action and median rho with track=True; the
    (T, NF, NR) occupancy of the (F, rho) grid with visits=True."""

    def __init__(self, T, Fg, lrg, track, visits):
        self.Fg, self.lrg, self.track, self.visits = Fg, lrg, track, visits
        self.frac = np.zeros(T); self.c_by = np.zeros(T)
        self.a_by_t = np.full(T, np.nan); self.rho_med = np.zeros(T)
        self.visit = np.zeros((T, len(Fg), len(lrg))) if visits else None

    def record(self, t, a, present, F, rho, p):
        any_ = present.any()
        self.frac[t] = present.mean()
        self.c_by[t] = (a[present] * p.GAMMA).mean() * 100 if any_ else 0.0
        if self.visits and any_:
            Fg, lrg = self.Fg, self.lrg
            iF = np.abs(Fg[:, None] - np.clip(F[present], Fg[0], Fg[-1])[None, :]).argmin(axis=0)
            iR = np.abs(lrg[:, None] - np.clip(np.log(rho[present]), lrg[0], lrg[-1])[None, :]).argmin(axis=0)
            np.add.at(self.visit[t], (iF, iR), 1.0)
        if self.track:
            self.a_by_t[t] = a[present].mean() if any_ else np.nan
            self.rho_med[t] = np.median(rho[present]) if any_ else np.nan

    def extras(self):
        out = {}
        if self.track:
            out.update(a_by_t=self.a_by_t, rho_med=self.rho_med)
        if self.visits:
            out["visits"] = self.visit
        return out
