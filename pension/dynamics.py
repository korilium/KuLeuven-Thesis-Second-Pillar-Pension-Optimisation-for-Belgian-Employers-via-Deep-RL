"""
dynamics.py -- THE career dynamics of a second-pillar plan, for n paths at once.

One year of a career, for every path:

    contribution   c = a * GAMMA * S                       (0 once the member has left)
    reserve        R <- (R + c) * exp(mu_t + sigma * zR)   in force
                   R <- R * exp(mu_t)                       paid-up: no further asset shock
    liability      L <- the guaranteed reserve, see the ledgers below; frozen once paid-up
    salary         S <- S * (1 + W)
    churn          the member leaves with probability hazard(t); the contract then
                   goes paid-up for the rest of the horizon

and at retirement (settle) the member receives max(R, L); the employer pays the
shortfall (L - R)+.

The constant-rate model credits mu_t = MU with asset noise SIGMA_R and grows the
liability VERTICALLY at G with its own shock SIGMA_L. A rate model credits the
path's book yield mu_t (noise SIGMA_R_RATES) and books every contribution
HORIZONTALLY at the WAP rate G_t of its payment year, until retirement.

Every consumer runs on this module: dp.simulate (the policy evaluator), the RL
environment (pension.envs.pension_env), and the tests that pin the DP's reduced
(F, rho) transitions to it. All randomness of a run sits in one Exogenous object,
drawn up front, so two policies can be compared on the same paths (common random
numbers).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# --- exogenous randomness -----------------------------------------------------------
@dataclass(frozen=True)
class Exogenous:
    """Everything random about n careers over T years, drawn once.

    zR, zL, u: (T, n) asset shock, guarantee shock, churn uniform.
    G, mu:     (>=T, n) path-wise WAP rate and book yield under a rate model;
               None under constant rates (then p.G / p.MU apply).
    r, acc:    (T, n) the short rate at the start of each year and the realised
               accrual of the year, exp(sum of monthly r*dt): the employer's
               numeraire (pension/numeraire.py). Constant rates: r = SHORT_RATE.
    The draw order -- per year zR, zL, u -- is the order dp.simulate always used,
    so pre-drawing reproduces its historical numbers exactly."""
    zR: np.ndarray
    zL: np.ndarray
    u: np.ndarray
    G: Optional[np.ndarray] = None
    mu: Optional[np.ndarray] = None
    r: Optional[np.ndarray] = None
    acc: Optional[np.ndarray] = None

    @property
    def n(self):
        return self.zR.shape[1]

    @property
    def rates(self):
        return self.G is not None

    def paths(self, idx):
        """The sub-batch of paths `idx` (an int gives a batch of one)."""
        sl = slice(idx, idx + 1) if isinstance(idx, (int, np.integer)) else idx
        pick = lambda x: None if x is None else x[:, sl]
        return Exogenous(pick(self.zR), pick(self.zL), pick(self.u), pick(self.G), pick(self.mu),
                         pick(self.r), pick(self.acc))

    @classmethod
    def draw(cls, p, n, seed, rates=None):
        """Shocks from np.random.default_rng(seed); `rates` is a scenario dict
        (economy.draw_rate_scenarios) or None for constant rates."""
        rng = np.random.default_rng(seed)
        zR = np.empty((p.T, n)); zL = np.empty((p.T, n)); u = np.empty((p.T, n))
        for t in range(p.T):
            zR[t] = rng.standard_normal(n); zL[t] = rng.standard_normal(n); u[t] = rng.random(n)
        if rates is None:
            r = np.full((p.T, n), p.SHORT_RATE)
            return cls(zR, zL, u, r=r, acc=np.exp(r))
        G, mu = np.asarray(rates["G"]), np.asarray(rates["mu"])
        assert G.shape[0] >= p.T and G.shape[1] == n, \
            f"rates must cover T={p.T} years x n_paths={n}, got {G.shape}"
        r = np.asarray(rates["r"])[:p.T] if "r" in rates else np.full((p.T, n), p.SHORT_RATE)
        acc = np.asarray(rates["acc"])[:p.T] if "acc" in rates else np.exp(r)
        return cls(zR, zL, u, G, mu, r, acc)


# --- liability ledgers --------------------------------------------------------------
@dataclass
class VerticalLedger:
    """Constant-rate model: the whole guaranteed reserve grows at G (+ SIGMA_L shock)."""
    L: np.ndarray

    @property
    def total(self):
        return self.L

    def book_and_grow(self, c, present, t, exo, p):
        self.L = np.where(present, (self.L + c) * np.exp(p.G + p.SIGMA_L * exo.zL[t]), self.L)


@dataclass
class HorizontalLedger:
    """Rate model, Branch 21 horizontal method: slot 0 holds the opening liability
    (locked at G_0), slot t+1 the year-t contribution locked at G_t. Each slot keeps
    its own rate until retirement; L is their sum."""
    Lv: np.ndarray            # (T+1, n) amounts per vintage
    lock: np.ndarray          # (T+1, n) locked rate per vintage
    _total: np.ndarray = field(default=None)

    @classmethod
    def open(cls, L0, G0, T):
        n = L0.shape[0]
        Lv = np.zeros((T + 1, n)); Lv[0] = L0
        lock = np.zeros((T + 1, n)); lock[0] = G0
        return cls(Lv, lock, L0)

    @property
    def total(self):
        return self._total

    def book_and_grow(self, c, present, t, exo, p):
        self.Lv[t + 1] = c; self.lock[t + 1] = exo.G[t]
        self.Lv[:t + 2] = np.where(present, self.Lv[:t + 2] * np.exp(self.lock[:t + 2]), self.Lv[:t + 2])
        self._total = self.Lv[:t + 2].sum(axis=0)


# --- state ----------------------------------------------------------------------------
@dataclass
class State:
    """n careers at the start of year t."""
    t: int
    R: np.ndarray             # reserve (assets)
    S: np.ndarray             # salary
    ledger: object            # VerticalLedger | HorizontalLedger
    present: np.ndarray       # still employed (in force)
    leave_t: np.ndarray       # year of leaving (T if never)

    @property
    def L(self):
        return self.ledger.total

    @property
    def F(self):
        return self.R / self.L

    @property
    def rho(self):
        return self.S / self.L

    @classmethod
    def initial(cls, p, exo, R0=1.0, L0=1.0, S0=1.0):
        """R0/L0/S0 scalars or (n,) arrays. The ledger follows the regime of `exo`."""
        n = exo.n
        R = np.full(n, 1.0) * np.asarray(R0)
        L = np.full(n, 1.0) * np.asarray(L0)
        S = np.full(n, 1.0) * np.asarray(S0)
        ledger = HorizontalLedger.open(L, exo.G[0], p.T) if exo.rates else VerticalLedger(L)
        return cls(0, R, S, ledger, np.ones(n, bool), np.full(n, p.T, float))


def step(state, a, exo, p, hazard):
    """One year: contribute a (capacity fraction, already 0 for absent paths),
    credit, accrue the guarantee, grow salary, then draw churn. Mutates and returns
    `state` (arrays are replaced, never written in place, except the horizontal
    vintages), together with the contribution paid, c (n,)."""
    t = state.t
    present = state.present
    c = a * p.GAMMA * state.S
    if exo.rates:
        mu, sig = exo.mu[t], p.SIGMA_R_RATES
    else:
        mu, sig = p.MU, p.SIGMA_R
    # in force: contribute and carry the asset shock. Paid-up: the reserve compounds at
    # the credited return with no further shock -- freezing the contract freezes its
    # risk (this is what paidup_service assumes). L freezes once absent.
    state.R = np.where(present, (state.R + c) * np.exp(mu + sig * exo.zR[t]), state.R * np.exp(mu))
    state.ledger.book_and_grow(c, present, t, exo, p)
    state.S = state.S * (1.0 + p.W)
    lv = present & (exo.u[t] < hazard(t))
    state.leave_t = np.where(lv, t + 1, state.leave_t)
    state.present = present & ~lv
    state.t = t + 1
    return state, c


def settle(state, p):
    """Retirement: payout max(R, L), employer shortfall (L - R)+, and the service
    fraction tau/T that sets the leaver's pro-rated target."""
    R, L = state.R, state.L
    return dict(payout=np.maximum(R, L), short=np.maximum(L - R, 0.0),
                svc=np.minimum(state.leave_t / p.T, 1.0), stay=state.leave_t >= p.T)
