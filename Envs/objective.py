"""
objective.py -- the value function of the funding problem, as swappable parts.

DynPro optimises (solve) and scores (simulate) one objective, per unit of final
salary S_T:

    J = w_emp * e^{-DISC_EMP T} * employee(RR_tot, target)
      - w_er  * [ sum_t e^{-DISC_ER t} * contribution(a_t, t) + e^{-DISC_ER T} * shortfall((L_T-R_T)+/S_T) ]

An Objective names the four parts. DynPro calls the SAME object in terminal(),
paidup_service(), the Bellman flow and simulate(), so the oracle and the
evaluation can never drift apart again.

Every part receives `p`, the parameter namespace (DynPro passes its own module),
and reads parameters from it at call time. Sweeping a parameter the usual way --
setattr(dp, "ETA", 3) -- therefore moves every objective that uses it.

What fits the DP. Backward induction needs J additively separable over time:
  * employee:     any function of the terminal total replacement rate and the
                  target it is judged against (RR_TARGET for a stayer, the
                  service-pro-rated target for a leaver);
  * shortfall:    any function of the terminal shortfall in final-salary units;
  * contribution: any function of (a, t) -- the action and the year;
  * weights:      a LINEAR combination of the two legs. A constraint enters as its
                  Lagrangian penalty (see floor_penalty).
Anything nonlinear in totals or paths -- a variance/CVaR of total cost, a Nash
product of the legs, a penalty on year-to-year changes in a -- needs more state
than (t, F, rho) and belongs to the RL rung.

Add a variant: write a factory below returning the callable, and register an
Objective in OBJECTIVES. Select it with dp.OBJECTIVE = "<name>", or per call
with solve(..., objective=...) / simulate(..., objective=...).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


def u(x, eta):
    """Normalized (Box-Cox) CRRA utility, eta != 1:
        u(x) = (x^(1-eta) - 1) / (1 - eta),
    which satisfies u(1) = 0 and u'(1) = 1 for every eta. The zero at the target
    makes "on target" the natural origin; the unit slope at the target is what
    licenses the money-metric reading of the employee leg (a marginal unit of x
    is worth one unit of the numeraire there). Differs from the bare
    x^(1-eta)/(1-eta) by the additive constant -1/(1-eta) only, so it shifts
    value levels without changing any argmax."""
    if abs(eta - 1.0) < 1e-9:          # removable singularity: the eta->1 limit IS log,
        return np.log(x)               # which is the ergodicity-canonical (Kelly) criterion
    return (np.power(x, 1.0 - eta) - 1.0) / (1.0 - eta)


@dataclass(frozen=True)
class Objective:
    name: str
    employee: Callable       # (rr_tot, target, p) -> value, final-salary-years (undiscounted, unweighted)
    shortfall: Callable      # (short, p)          -> employer cost of short = (L-R)+/S_T at T
    contribution: Callable   # (a, t, p)           -> employer cost of the year-t contribution, final-salary units
    weights: Callable        # (p,)                -> (w_emp, w_er)
    note: str = ""


# --- employee leg -----------------------------------------------------------
def crra(eta=None):
    """target * ANNUITY * u(RR/target): the committed leg. u is in RELATIVE
    shortfall units; times the target it is in annual-RR units, times ANNUITY in
    capital (final-salary-years), the employer's numeraire. eta=None reads p.ETA.
    p.SATIATE caps the replacement rate at the target it is judged against."""
    def employee(rr, target, p):
        if p.SATIATE:
            rr = np.minimum(rr, target)
        return target * p.ANNUITY * u(rr / target, p.ETA if eta is None else eta)
    return employee


def satiated(base=None):
    """No value above the target: the employee is indifferent to overshoot."""
    base = base or crra()
    return lambda rr, target, p: base(np.minimum(rr, target), target, p)


def loss_averse(k=2.5, base=None):
    """Shortfalls below the target weigh k times more than the base curve says
    (a kink at the target, Kahneman-Tversky style); gains are unchanged."""
    base = base or crra()
    def employee(rr, target, p):
        v, v0 = base(rr, target, p), base(target, target, p)
        return np.where(rr < target, v0 + k * (v - v0), v)
    return employee


def floor_penalty(k=1.0, rr_min=None, base=None):
    """base minus k * ANNUITY * (rr_min - RR)+ : a replacement-rate floor as a
    Lagrangian penalty, in capital units. k=1 prices a missing point of RR at its
    actuarial cost. rr_min=None uses the target being judged against (pro-rated
    for leavers). base=None means NO utility beyond the floor."""
    def employee(rr, target, p):
        floor = target if rr_min is None else rr_min
        pen = k * p.ANNUITY * np.maximum(floor - rr, 0.0)
        return (0.0 if base is None else base(rr, target, p)) - pen
    return employee


# --- employer leg: terminal shortfall ---------------------------------------
def linear_shortfall():
    """The committed leg: the employer pays the WAP shortfall, euro for euro."""
    return lambda short, p: short


def convex_shortfall(k=1.0):
    """short + k*short^2: large deficits hurt more than proportionally (balance-
    sheet stress, a lump-sum settlement at T). k=1: a shortfall of half a final
    salary costs 1.5x its face value."""
    return lambda short, p: short + k * short ** 2


def threshold_shortfall(s0=0.1):
    """Only the part of the shortfall above s0 final salaries costs anything (a
    deficit buffer the employer can absorb)."""
    return lambda short, p: np.maximum(short - s0, 0.0)


# --- employer leg: contributions --------------------------------------------
def linear_contribution():
    """The committed leg: a contribution a*GAMMA*S_t, expressed per unit of final
    salary, i.e. a*GAMMA*(1+W)^-(T-t)."""
    return lambda a, t, p: a * p.GAMMA * (1.0 + p.W) ** (-(p.T - t))


def convex_contribution(k=2.0):
    """Cash-flow strain: each euro costs (1 + k * aGAMMA), where aGAMMA is the
    contribution as a share of the CURRENT payroll. k=2: paying 15% of salary
    costs 30% extra per euro; paying 2% costs 4% extra."""
    lin = linear_contribution()
    return lambda a, t, p: lin(a, t, p) * (1.0 + k * a * p.GAMMA)


# --- combination -------------------------------------------------------------
def lambda_weights():
    """The committed negotiation weight: lambda on the employee, 1-lambda on the employer."""
    return lambda p: (p.LAMBDA, 1.0 - p.LAMBDA)


def employer_only():
    """Both legs at weight 1: meant for an employee leg that is ONLY a penalty
    (floor_penalty with base=None), i.e. minimise cost subject to a floor."""
    return lambda p: (1.0, 1.0)


# --- registry -----------------------------------------------------------------
_EMP, _SHORT, _CONTR, _W = crra(), linear_shortfall(), linear_contribution(), lambda_weights()

OBJECTIVES = {o.name: o for o in [
    Objective("baseline", _EMP, _SHORT, _CONTR, _W,
              "committed model: CRRA(ETA) employee, linear employer, lambda weights"),
    Objective("log", crra(eta=1.0), _SHORT, _CONTR, _W,
              "log utility (eta=1, Kelly / ergodicity-canonical)"),
    Objective("loss_averse", loss_averse(k=2.5), _SHORT, _CONTR, _W,
              "shortfalls below the target weigh 2.5x"),
    Objective("satiated", satiated(), _SHORT, _CONTR, _W,
              "no value above the target"),
    Objective("convex_shortfall", _EMP, convex_shortfall(k=1.0), _CONTR, _W,
              "employer: shortfall + shortfall^2"),
    Objective("cashflow_strain", _EMP, _SHORT, convex_contribution(k=2.0), _W,
              "employer: contributions cost (1 + 2 * share of payroll) per euro"),
    Objective("employer_floor", floor_penalty(k=1.0), _SHORT, _CONTR, employer_only(),
              "employer cost only, RR >= target enforced as a penalty"),
]}


def resolve(obj, default):
    """None -> OBJECTIVES[default]; a name -> its registry entry; an Objective as is."""
    if obj is None:
        obj = default
    if isinstance(obj, Objective):
        return obj
    if obj not in OBJECTIVES:
        raise KeyError(f"unknown objective {obj!r}; known: {', '.join(OBJECTIVES)}")
    return OBJECTIVES[obj]
