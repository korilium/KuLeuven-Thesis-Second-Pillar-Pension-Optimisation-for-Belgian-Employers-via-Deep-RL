"""
numeraire.py -- when the employer's money is valued: the timing around the objective.

The Objective (pension/objective.py) says WHAT each leg is worth; this module says
in which money. Two choices, set by p.EMPLOYER_NUMERAIRE:

"retirement" (default) -- both legs in retirement-date money, under P:

    J = lambda * E[ employee(RR_tot, target) ]
      - (1 - lambda) * E[ sum_t contribution(a_t, t) * A(t, T) + shortfall((L - R)^+ / S_T) ]

    A(t, T) = E^P_t[ exp(int_t^T (r_u + s) du) ]: a premium paid at t accrues to T
    at the short rate r plus the financing spread s = FINANCING_SPREAD. It is a
    closed-form function of r_t (pension/rates/accrual.py), so every reward is
    known when it is paid. The terminal legs fall at T and are not discounted.
    contribution(a, t) already carries (1+W)^-(T-t), a change of units to final
    salary, not a discount. Constant rates: A(t, T) = e^{(SHORT_RATE + s)(T - t)}.

"discounted" -- the previous specification: premiums discounted at DISC_ER,
    the employee leg at DISC_EMP and the shortfall at DISC_ER (to t = 0).

Per model under "retirement": vasicek (canonical), hull_white_p and vasicek_short
accrue at their own short rate; hull_white (risk-neutral r) is an error -- use
hull_white_p, whose LONG_RATE_P = None is the phi = 0 case.
"""
import numpy as np

_NUMERAIRES = ("retirement", "discounted")


def _check(p):
    if p.EMPLOYER_NUMERAIRE not in _NUMERAIRES:
        raise ValueError(f"EMPLOYER_NUMERAIRE must be one of {_NUMERAIRES}, got {p.EMPLOYER_NUMERAIRE!r}")


def retirement(p):
    _check(p)
    return p.EMPLOYER_NUMERAIRE == "retirement"


def terminal_factors(p):
    """(employee, employer) factors on the terminal legs."""
    if retirement(p):
        return 1.0, 1.0
    return np.exp(-p.DISC_EMP * p.T), np.exp(-p.DISC_ER * p.T)


def premium_factor(t, r_t, p):
    """The factor on a premium paid in year t: A(t, T) given the short rate r_t
    (scalar or one per path) under "retirement", e^{-DISC_ER t} under "discounted"."""
    if not retirement(p):
        return np.exp(-p.DISC_ER * t)
    if p.RATE_MODEL == "constant":
        return np.exp((p.SHORT_RATE + p.FINANCING_SPREAD) * (p.T - t))
    if p.RATE_MODEL == "hull_white":
        raise ValueError("hull_white's short rate is risk-neutral: it has no place in the "
                         "'retirement' numeraire. Use RATE_MODEL='vasicek' (canonical) or "
                         "'hull_white_p' (LONG_RATE_P=None is hull_white's phi = 0 case), or "
                         "EMPLOYER_NUMERAIRE='discounted'.")
    from pension.rates.accrual import closed_form_accrual
    return closed_form_accrual(t, r_t, p, s=p.FINANCING_SPREAD)


def premium_schedule(p, rates=None):
    """One deterministic factor per year t = 0..T-1, for the DP's Bellman flow.
    Without a scenario: the constant-rate (or discounted) factor. With a scenario
    (the certainty-equivalent solve): the scenario MEAN of A(t, T; r_t) over the
    paths -- the DP does not carry r in its state."""
    if rates is None or not retirement(p):
        q = p if rates is not None or p.RATE_MODEL == "constant" else p.replace(RATE_MODEL="constant")
        return [premium_factor(t, q.SHORT_RATE, q) for t in range(p.T)]
    r = np.asarray(rates["r"])
    return [float(np.mean(premium_factor(t, r[t], p))) for t in range(p.T)]
