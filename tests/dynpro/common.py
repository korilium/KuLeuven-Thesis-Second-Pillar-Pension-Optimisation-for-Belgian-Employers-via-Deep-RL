"""Presentation/sweep harness for the tests/dynpro suites (sensitivity_suite.py,
contribution_schedule_suite.py, lambda_dial_suite.py): sys.path wiring to Envs/,
figure config and labels, and baseline/restore bookkeeping for parameter sweeps.

The MODEL lives entirely in Envs/DynPro.py -- parameters, transitions, grids,
the churn-aware DP solver, and simulate() (the committed forward Monte-Carlo).
Nothing economic is defined here; the names re-exported below are aliases into
DynPro so the suites can keep calling c.simulate(...), c.grids(...) etc.
"""
import contextlib
import functools
import os, sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Envs"))
import DynPro as dp


# --- the rate regime of this run -----------------------------------------------
# Every suite runs under ONE regime, chosen per run and fixed before anything else:
#     python sensitivity_suite.py tornado                    # constant rates (default)
#     python sensitivity_suite.py tornado --rates=hull_white
#     DYNPRO_RATES=vasicek python scenario_suite.py table
# It becomes the baseline value of dp.RATE_MODEL, so restore() keeps it, and both
# dp.solve (certainty-equivalent) and dp.simulate (path-wise G_t, mu_t) follow it.
# Figures of a rate regime go to their own tree, figs/rates_<model>/..., so they
# never overwrite the constant-rate results they are compared against.
RATE_MODELS = ("constant", "hull_white", "vasicek")

def _pick_rates():
    model = os.environ.get("DYNPRO_RATES", "constant")
    for arg in list(sys.argv[1:]):
        if arg.startswith("--rates="):
            model = arg.split("=", 1)[1]
            sys.argv.remove(arg)          # keep the suites' positional argv parsing intact
    assert model in RATE_MODELS, f"--rates must be one of {RATE_MODELS}, got {model!r}"
    return model

RATES = _pick_rates()
dp.RATE_MODEL = RATES

# Parameters a rate model sets path-wise (simulate) and by moment-matching (solve),
# so overriding them under a rate model does nothing. Sweeps over them are
# constant-rate experiments; under a rate model they are skipped (rate_owned).
# SIGMA_R_RATES is the rate-regime counterpart of SIGMA_R and is swept instead.
RATE_OWNED = ("G", "MU", "SIGMA_L", "SIGMA_R")


def rate_owned(*params):
    """True if the active rate model owns any of `params` -- and says so, so a
    skipped sweep never passes silently. Always False under constant rates."""
    hit = [p for p in params if RATES != "constant" and p in RATE_OWNED]
    if hit:
        print(f"  skipped under RATE_MODEL={RATES}: {', '.join(hit)} "
              f"{'is' if len(hit) == 1 else 'are'} set by the rate scenarios "
              f"(run without --rates for this sweep)")
    return bool(hit)


def rate_overrides(ov):
    """Translate a dict of dp.* overrides to the active regime.

    Under constant rates it is returned unchanged. Under a rate model, SIGMA_R maps
    to SIGMA_R_RATES (the asset noise on top of the book yield) and SIGMA_L is
    dropped (the guarantee risk IS G_t now); an override of G or MU has no
    counterpart, so None is returned and the caller skips that scenario."""
    if RATES == "constant":
        return dict(ov)
    if any(k in ("G", "MU") for k in ov):
        return None
    out = {("SIGMA_R_RATES" if k == "SIGMA_R" else k): v for k, v in ov.items()}
    out.pop("SIGMA_L", None)
    return out


OUT = "figs" if RATES == "constant" else f"figs/rates_{RATES}"; DPI = 150
mpl.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# NOTE: DISC_EMP is deliberately ABSENT. It enters the objective only as
# exp(-DISC_EMP*T), a constant on the employee leg, so it is exactly redundant with
# LAMBDA (verified to ~6e-11; see lambda_equivalent below). LAMBDA is already here,
# so including DISC_EMP too would double-count one lever and mis-rank the tornado.
PARAMS = ["LAMBDA", "RR_TARGET", "RR_LEGAL", "ANNUITY", "GAMMA", "ETA", "G", "MU",
          "SIGMA_R", "DISC_ER"]
_PARAMS_ALL = list(PARAMS)
if RATES != "constant":     # the tornado sweeps what the regime leaves free
    PARAMS = [p for p in PARAMS if p not in RATE_OWNED] + ["SIGMA_R_RATES"]
LAB = {"LAMBDA": r"$\lambda$", "RR_TARGET": r"$RR^\star$", "RR_LEGAL": r"$RR_{\rm legal}$",
       "ANNUITY": r"$\ddot a$", "GAMMA": r"$\Gamma$", "ETA": r"$\eta$", "G": r"$G$",
       "MU": r"$\mu$", "SIGMA_R": r"$\sigma_R$", "SIGMA_R_RATES": r"$\sigma_R$ (excess)", "DISC_EMP": r"$\delta_e$", "DISC_ER": r"$\delta_f$"}

# PARAMS drives tornado(), which perturbs each entry by +/-15%, so it holds only
# continuous ECONOMIC parameters. restore() must cover more than that: anything a
# caller might set on dp. SATIATE is a bool and BETA is extraction-only, so neither
# belongs in a +/-15% sweep, but both must still be reset -- leaving them out let a
# scenario leak SATIATE=True into every later run.
_RESTORE = _PARAMS_ALL + ["SIGMA_R_RATES", "DISC_EMP", "SIGMA_L", "SATIATE", "BETA", "DISC", "T", "W", "S0",
                     "RATE_MODEL", "RATE_SEED"]   # the rate-regime switch

_BASE = {k: getattr(dp, k) for k in _RESTORE}

# model entry points, re-exported for the suites' existing call sites
grids = dp.grids
simulate = dp.simulate
const_policy = dp.const_policy
schedule_policy = dp.schedule_policy
new_plan_init = dp.new_plan_init
sample_entry = dp.sample_entry

# --- THE evaluation protocol (numerical, not economic) -------------------------
# Lives here, not in one suite, so no suite can silently drift onto a different grid
# than the results it is compared against. nR >= 71 is REQUIRED at the rho floor of
# 0.01: the schedule-shape columns are not converged below that (widening the range
# without raising n coarsens the step). Suites that need finer grids for a headline
# figure pass their own, but anything cross-compared uses these.
GRID = dict(nF=73, nR=71, na=20, nq=5)
N_PATHS = 15000
SEED = 7
BAND_PCT = (0.02, 0.15)          # contribution band, as a fraction of salary


def restore():
    """Reset every dp.* global any suite might sweep back to its baseline value."""
    for k, v in _BASE.items(): setattr(dp, k, v)


@contextlib.contextmanager
def overrides(**kw):
    """Apply dp.* overrides for the duration, then ALWAYS restore.

        with c.overrides(G=0.02, MU=0.05):
            ...

    Use this rather than a bare `c.restore(); setattr(...)` pair: on an exception
    the bare form leaks the mutated global into every later figure in the run."""
    try:
        restore()
        for k, v in kw.items(): setattr(dp, k, v)
        yield
    finally:
        restore()


def restores(fn):
    """Decorator: guarantee restore() on the way out, however the call ends.
    Belt-and-braces for functions that sweep globals in a loop."""
    @functools.wraps(fn)
    def _wrapped(*a, **k):
        try:
            return fn(*a, **k)
        finally:
            restore()
    return _wrapped


def lambda_equivalent(de_new, lam=None, de_ref=None):
    """The LAMBDA that reproduces (lam, de_new) at the reference DISC_EMP.

    delta_e multiplies the employee leg by exp(-delta_e*T) and nothing else, so it
    is redundant with LAMBDA up to a positive rescale of the objective (which
    leaves the argmax alone):
        lambda'' = A / (A + B*exp(-de_ref*T)),  A = lam*exp(-de_new*T), B = 1-lam
    Report this alongside any delta_e result so it is not read as an independent
    economic mechanism."""
    lam = _BASE["LAMBDA"] if lam is None else lam
    de_ref = _BASE["DISC_EMP"] if de_ref is None else de_ref
    A = lam * np.exp(-de_new * dp.T); B = 1.0 - lam
    return float(A / (A + B * np.exp(-de_ref * dp.T)))


def ensure_out(path=None):
    """Create the figure directory. Each suite passes its own subfolder."""
    os.makedirs(path or OUT, exist_ok=True)


def entry(n_paths, seed=7):
    """THE STANDARD ENTRY STATE (protocol, not model): a new-plan cohort.

    Splat into simulate():  c.simulate(pol, Fg, rg, **c.entry(n, seed), ...)

    Every figure uses this so results are comparable across the suites. The single
    corner point (F=1, rho=rho_max) is a DIAGNOSTIC only -- it is one arbitrary
    inception state, and conclusions drawn from it were not comparable with
    cohort-based ones. Swap this for the calibrated DB2P/Marsh entry spread when
    it lands; every call site inherits it automatically."""
    rng = np.random.default_rng(seed)
    R0, L0, S0 = dp.new_plan_init(n_paths, rng)
    return dict(R0=R0, L0=L0, S0=S0)


def iso_rr(Fg, rg, total=True):
    """TOTAL annual replacement on the (F, rho) grid:

        RR = RR_LEGAL + max(F, 1) / (ANNUITY * rho).

    This used to return max(F,1)/rho, which policy_map then contoured and labelled
    "RR=%.1f". That quantity is the accrued capital in final-salary-years, i.e.
    ANNUITY times the SECOND-PILLAR rate -- at ANNUITY=15 the contour labelled
    "RR=0.7" actually sat at RR2 = 0.047, a total of 0.477. The labels were off by
    the annuity factor and did not line up with RR_TARGET, which is the one level
    a reader wants to find on that map.

    `total=False` returns the second-pillar rate alone, for a figure that wants to
    separate the employer's contribution from the legal floor."""
    FF, RRr = np.meshgrid(Fg, rg, indexing="ij")
    rr2 = np.maximum(FF, 1.0) / (dp.ANNUITY * RRr)
    return dp.RR_LEGAL + rr2 if total else rr2


def solve(Fg, rg, ag, nq=5, beta=None, betas=None):
    """The committed (churn-aware, paid-up service-pro-rated) policy oracle.
    `betas` additionally returns soft (signal) readouts of the same Q-values;
    it leaves V and the hard policy untouched -- see dp.solve."""
    return dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=nq, beta=beta, betas=betas)
