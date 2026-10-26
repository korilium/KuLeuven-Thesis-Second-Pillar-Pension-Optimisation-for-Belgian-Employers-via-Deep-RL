"""
baseline.py -- the bare-bone baseline: one member, a fixed premium, and each year
the share of it that goes to Branch 21. A fully observed MDP.

Every equation is numbered as in docs/MODEL.md and implemented by one function
below. Money is in units of the CURRENT salary S_t; one model year is a calendar
year starting in January; the 10Y OLO is simulated monthly.

    state  s_t = (t, y_t, G_t, H_t, K_t, mu_t, L_t, R21_t, N_t)     (docs/MODEL.md, Sec. 2)
    action alpha_t in [0, 1]: the Branch 21 share of the premium c S_t
    reward at T only: lambda target ann u(RR_tot / target) - (1 - lambda) (L_tau - R_T)^+ / S_T

Only the rate engine is used (read-only): the Vasicek calibration of the 10Y
(pension.economy.vasicek_params) and the statutory WAP map (pension.rates.wap).
"""

from dataclasses import dataclass, replace
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pension.economy import rate_calibration, vasicek_params
from pension.rates.wap import WAP_FLOOR, wap_formula


# --- parameters ----------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    T: int = 45                  # career length (years)
    W: float = 0.025             # salary growth
    C: float = 0.05              # contribution rate: premium P_t = C * S_t
    I_TARIFF: float = 0.01       # Branch 21 tariff floor i^tariff
    M: float = 0.0               # insurer margin m below the book yield
    BOOK_WEIGHT: float = 1 / 48  # monthly weight w of the book-yield filter (mean age ~4 years)
    LAMBDA: float = 0.5          # employee weight
    ETA: float = 2.0             # CRRA curvature
    ANNUITY: float = 15.0        # annuity factor (capital -> annual pension)
    RR_LEGAL: float = 0.43       # first-pillar replacement rate
    RR_TARGET: float = 0.70      # total-adequacy target
    H0: float = 0.12             # exit hazard in year 0
    HINF: float = 0.025          # long-tenure exit hazard
    HTAU: float = 7.0            # decay of the exit hazard (years)
    LONG_RATE_P: Optional[float] = None       # long-run short rate; None: OLS theta of the 10Y
    SPREAD_10Y_SHORT: Optional[float] = None  # s_term; None: historical mean (10Y - 1Y)
    TRAIN_SEED: int = 2026       # noise pool of the RL environment
    EVAL_SEED: int = 7           # noise of simulate() and of evaluation episodes


BASELINE = Params()


# --- rates ---------------------------------------------------------------------------
def premonths():
    """Simulated months from the last OLO observation to January of model year 0."""
    return (13 - rate_calibration()["t0"].month) % 12


def vasicek_month(y, eps, vp):
    """Eq. (1), MODEL.md: exact monthly Vasicek step of the 10Y OLO under P."""
    a = np.exp(-vp["kappa"] / 12)
    return y * a + vp["theta"] * (1 - a) + vp["sigma"] * np.sqrt((1 - a ** 2) / (2 * vp["kappa"])) * eps


def short_rate(y, vp):
    """Eq. (2), MODEL.md: r_t = y_t - s_term."""
    return y - vp["spread"]


def wap_rate(sum24):
    """Eq. (3), MODEL.md: G = statutory map of the 24-month mean 10Y (rates/wap.py)."""
    return wap_formula(sum24 / 24)


def book_yield(mu, y, p):
    """Eq. (4), MODEL.md: monthly exponential filter of the 10Y, mu <- mu + w (y - mu)."""
    return mu + p.BOOK_WEIGHT * (y - mu)


def credited_rate(mu, p):
    """Eq. (5), MODEL.md: Branch 21 crediting i_t = max(i^tariff, mu_t - m)."""
    return np.maximum(p.I_TARIFF, mu - p.M)


# --- balances ------------------------------------------------------------------------
def accrue(X, P, rate, p):
    """Eq. (6), MODEL.md: X_{t+1} = (X_t + P_t) e^{rate}, in units of S_{t+1}.
    The ONE recursion of the liability (rate G_t) and the Branch 21 reserve (rate i_t)."""
    return (X + P) * np.exp(rate) / (1 + p.W)


def bond_price(y, t, p):
    """Eq. (7), MODEL.md: P(t, T) = exp(-y_t (T - t)), flat at the 10Y; P(T, T) = 1."""
    return np.exp(-y * (p.T - t))


def buy_units(N, P23, price, p):
    """Eq. (8), MODEL.md: N_{t+1} = N_t + P23_t / P(t, T), in units of S_{t+1}."""
    return (N + P23 / price) / (1 + p.W)


def hazard(t, p):
    """Eq. (9), MODEL.md: exit hazard h(t) = h_inf + (h_0 - h_inf) e^{-t / tau_h}."""
    return p.HINF + (p.H0 - p.HINF) * np.exp(-t / p.HTAU)


# --- reward and reported cost ----------------------------------------------------------
def utility(x, eta):
    """Normalised CRRA, u(1) = 0, u'(1) = 1 (log at eta = 1)."""
    return np.log(x) if abs(eta - 1) < 1e-9 else (x ** (1 - eta) - 1) / (1 - eta)


def reward_legs(L, R, tau, p):
    """Eq. (10), MODEL.md: (employee leg, shortfall) at T, with L = L_tau / S_T and
    R = R_T / S_T: target_tau ann u(RR_tot / target_tau) and (L_tau - R_T)^+ / S_T."""
    target = p.RR_LEGAL + tau / p.T * (p.RR_TARGET - p.RR_LEGAL)
    rr = p.RR_LEGAL + np.maximum(R, L) / p.ANNUITY
    return target * p.ANNUITY * utility(rr / target, p.ETA), np.maximum(L - R, 0.0)


def premium_accrual(y, t, vp, p):
    """Eq. (11), MODEL.md: A(t, T) = E^P_t[exp(int_t^T r_u du)] in closed form, r = y - s_term."""
    k, s, tau = vp["kappa"], vp["sigma"], p.T - t
    B = (1 - np.exp(-k * tau)) / k
    mean = vp["theta"] * tau + (y - vp["theta"]) * B
    var = s ** 2 / k ** 2 * (tau - B) - s ** 2 / (2 * k) * B ** 2
    return np.exp(mean + var / 2 - vp["spread"] * tau)


# --- state and one year ------------------------------------------------------------------
@dataclass
class State:
    """n careers in January of year t. y, G, H, K, mu, L, R21, N are (n,) arrays; H and K
    are the sums of the observed months in the windows of G_{t+1} and G_{t+2}.
    inforce and tau only drive the roll-forward after exit: every decision state is
    in force, with tau = t."""
    t: int
    y: np.ndarray
    G: np.ndarray
    H: np.ndarray
    K: np.ndarray
    mu: np.ndarray
    L: np.ndarray
    R21: np.ndarray
    N: np.ndarray
    inforce: np.ndarray
    tau: np.ndarray

    def obs(self, p):
        """The observation: the state, with t / T and H, K as means of their months."""
        return np.stack([np.full_like(self.y, self.t / p.T), self.y, self.G, self.H / 20,
                         self.K / 8, self.mu, self.L, self.R21, self.N], axis=1)


def initial_state(eps_pre, p):
    """January of year 0 from the observed OLO history and the pre-roll months
    eps_pre (premonths(), n). Zero balances: a new member."""
    vp, hist = vasicek_params(p), rate_calibration()["hist10Y"]
    n = eps_pre.shape[1]
    y = [np.full(n, h) for h in hist]
    for e in eps_pre:
        y.append(vasicek_month(y[-1], e, vp))
    y = np.array(y)
    mu = y[0]
    for row in y[1:]:
        mu = book_yield(mu, row, p)
    z = np.zeros(n)
    return State(0, y[-1], wap_rate(y[-32:-8].sum(0)), y[-20:].sum(0), y[-8:].sum(0), mu,
                 z, z, z, np.ones(n, bool), z)


def step(s, alpha, eps, u, p):
    """One year for every path: premium at the start, accrual over the year, twelve
    months of rates (eps: (12, n)), then exit with probability h(t) (u: (n,) uniforms).
    Out of force: no premium, L fixed (rate 0), units held, R21 still at i_t."""
    vp = vasicek_params(p)
    P = p.C * s.inforce
    L = accrue(s.L, P, s.G * s.inforce, p)
    R21 = accrue(s.R21, alpha * P, credited_rate(s.mu, p), p)
    N = buy_units(s.N, (1 - alpha) * P, bond_price(s.y, s.t, p), p)
    y, mu, months = s.y, s.mu, []
    for e in eps:                                         # February .. next January
        y = vasicek_month(y, e, vp)
        mu = book_yield(mu, y, p)
        months.append(y)
    m = np.array(months)
    leave = s.inforce & (u < hazard(s.t, p))
    return State(s.t + 1, y, wap_rate(s.H + m[:4].sum(0)), s.K + m.sum(0), m[4:].sum(0), mu,
                 L, R21, N, s.inforce & ~leave, np.where(s.inforce, s.t + 1, s.tau))


# --- noise, evaluation and the environment ------------------------------------------------
@dataclass(frozen=True)
class Noise:
    """Everything random about n careers: eps (premonths() + 12 T, n) monthly normals
    of the 10Y, u (T, n) exit uniforms. No look-ahead: year t reads only its own rows."""
    eps: np.ndarray
    u: np.ndarray

    def year(self, t):
        k = premonths() + 12 * t
        return self.eps[k:k + 12], self.u[t]

    def path(self, j):
        return Noise(self.eps[:, j:j + 1], self.u[:, j:j + 1])


def draw_noise(n, seed, p=BASELINE):
    rng = np.random.default_rng(seed)
    return Noise(rng.standard_normal((premonths() + 12 * p.T, n)), rng.random((p.T, n)))


def simulate(policy, n=10000, seed=None, p=BASELINE, noise=None):
    """Monte Carlo of a policy obs (n, 9) -> alpha (n,) over n careers (seed: EVAL_SEED).
    "joint" is the objective J (A5) and "reward" the per-path reward; the premium cost
    sum_{t<tau} C (1+W)^-(T-t) A(t,T), which no policy changes, is reported apart."""
    noise = draw_noise(n, p.EVAL_SEED if seed is None else seed, p) if noise is None else noise
    vp = vasicek_params(p)
    s = initial_state(noise.eps[:premonths()], p)
    cost = np.zeros(s.y.shape)
    diag = {k: np.zeros(p.T) for k in ("G", "G_floor", "mu", "i_minus_G", "r")}
    for t in range(p.T):
        i = credited_rate(s.mu, p)
        for k, v in (("G", s.G), ("G_floor", s.G <= WAP_FLOOR + 1e-12), ("mu", s.mu),
                     ("i_minus_G", i - s.G), ("r", short_rate(s.y, vp))):
            diag[k][t] = v.mean()
        alpha = np.clip(policy(s.obs(p)), 0.0, 1.0)
        cost += s.inforce * p.C * (1 + p.W) ** -(p.T - t) * premium_accrual(s.y, t, vp, p)
        s = step(s, alpha, *noise.year(t), p)
    emp, short = reward_legs(s.L, s.R21 + s.N, s.tau, p)
    reward = p.LAMBDA * emp - (1 - p.LAMBDA) * short
    return dict(joint=float(reward.mean()), employee=float(emp.mean()), shortfall=float(short.mean()),
                shortfall_freq=float((short > 0).mean()), premium_cost=float(cost.mean()),
                reward=reward, L=s.L, R21=s.R21, N=s.N, by_year=diag)


class PensionEnv(gym.Env):
    """One career per episode, gamma = 1. Observation = State.obs (9,), action =
    alpha_t. The reward is 0 until T; when the member leaves, the remaining years are
    rolled forward on the episode's own noise and the reward at T is paid at once.
    reset(options={"noise": Noise of one path}) replays given noise; adding
    "state": State restarts from a snapshot."""
    metadata = {"render_modes": []}

    def __init__(self, p=BASELINE, pool=2000):
        self.p, self.pool = p, pool
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float64)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64)
        self._train = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.noise = options.get("noise")
        if self.noise is None:
            if self._train is None:
                self._train = draw_noise(self.pool, self.p.TRAIN_SEED, self.p)
            self.noise = self._train.path(int(self.np_random.integers(self.pool)))
        self.state = options.get("state") or initial_state(self.noise.eps[:premonths()], self.p)
        return self.state.obs(self.p)[0], {}

    def step(self, action):
        p = self.p
        alpha = np.clip(np.asarray(action, float).reshape(1), 0.0, 1.0)
        s = step(self.state, alpha, *self.noise.year(self.state.t), p)
        while not s.inforce[0] and s.t < p.T:              # left: nothing more to decide
            s = step(s, alpha, *self.noise.year(s.t), p)
        self.state = s
        done = s.t >= p.T
        reward = 0.0
        if done:
            emp, short = reward_legs(s.L, s.R21 + s.N, s.tau, p)
            reward = float(p.LAMBDA * emp[0] - (1 - p.LAMBDA) * short[0])
        return s.obs(p)[0], reward, done, False, {}
