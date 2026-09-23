"""Presentation/sweep harness for the tests/dynpro suites (sensitivity_suite.py,
contribution_schedule_suite.py, lambda_dial_suite.py): sys.path wiring to Envs/,
figure config and labels, and baseline/restore bookkeeping for parameter sweeps.

The MODEL lives entirely in Envs/DynPro.py -- parameters, transitions, grids,
the churn-aware DP solver, and simulate() (the committed forward Monte-Carlo).
Nothing economic is defined here; the names re-exported below are aliases into
DynPro so the suites can keep calling c.simulate(...), c.grids(...) etc.
"""
import os, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Envs"))
import DynPro as dp

OUT = "figs"; DPI = 150
mpl.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

PARAMS = ["LAMBDA", "RR_TARGET", "RR_LEGAL", "ANNUITY", "GAMMA", "ETA", "G", "MU",
          "SIGMA_R", "DISC_EMP", "DISC_ER"]
LAB = {"LAMBDA": r"$\lambda$", "RR_TARGET": r"$RR^\star$", "RR_LEGAL": r"$RR_{\rm legal}$",
       "ANNUITY": r"$\ddot a$", "GAMMA": r"$\Gamma$", "ETA": r"$\eta$", "G": r"$G$",
       "MU": r"$\mu$", "SIGMA_R": r"$\sigma_R$", "DISC_EMP": r"$\delta_e$", "DISC_ER": r"$\delta_f$"}

_BASE = {k: getattr(dp, k) for k in PARAMS}

# model entry points, re-exported for the suites' existing call sites
grids = dp.grids
simulate = dp.simulate
const_policy = dp.const_policy
new_plan_init = dp.new_plan_init
sample_entry = dp.sample_entry


def restore():
    """Reset every dp.* global any suite might sweep back to its baseline value."""
    for k, v in _BASE.items(): setattr(dp, k, v)


def ensure_out():
    os.makedirs(OUT, exist_ok=True)


def iso_rr(Fg, rg):
    FF, RRr = np.meshgrid(Fg, rg, indexing="ij")
    return np.maximum(FF, 1.0) / RRr


def solve(Fg, rg, ag, nq=5, betas=None):
    """The committed (churn-aware, paid-up service-pro-rated) policy oracle.
    `betas` additionally returns soft (signal) readouts of the same Q-values;
    it leaves V and the hard policy untouched -- see dp.solve."""
    return dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=nq, betas=betas)
