"""
baseline.py -- the bare-bone baseline: one member, a fixed premium, and each year
the share of it that goes to Branch 21; the rest buys a Branch 23 government bond
fund of constant duration. The liability follows the horizontal WAP method: each
premium accrues at the rate G of its own year. Branch 21 is horizontal too: each
premium gets a tariff locked to retirement, plus profit-sharing on top. A fully
observed MDP.

Every equation is numbered as in docs/MODEL.md and implemented by one function
below. Money is in units of the CURRENT salary S_t; one model year is a calendar
year starting in January; the 10Y OLO is simulated monthly; the premium is paid
mid-year (July).

    state  s_t = (t, y_t, G_t, H_t, K_t, mu_t, R23_t, L^(g)_t for the 9 WAP rates g,
                  R21^(k)_t for the 16 tariff levels i_k)       (docs/MODEL.md, Sec. 2)
           the liability is L_t = sum_g L^(g)_t, the Branch 21 reserve R21_t = sum_k R21^(k)_t
    action alpha_t in [0, 1]: the Branch 21 share of the premium c S_t
    reward at T only: lambda target ann u(RR_tot / target) - (1 - lambda) (L_tau - R_T)^+ / S_T

Only the rate engine is used: the Vasicek calibration of the 10Y and the term
spread (pension.rates.calibration) and the statutory WAP map (pension.rates.wap).
"""

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pension.rates.calibration import calibration
from pension.rates.wap import WAP_CAP, WAP_FLOOR, WAP_STEP, wap_formula


# --- parameters ----------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    T: int = 45                  # career length (years)
    W: float = 0.025             # salary growth
    C: float = 0.05              # contribution rate: premium P_t = C * S_t
    TARIFF_STEP: float = 0.0025  # Branch 21 tariffs are floored to this grid step (Eq. 5a)
    TARIFF_MARGIN: float = 0.010  # PLACEHOLDER: tariff = July book yield mu minus this margin (Eq. 5a)
    PS_MARGIN: float = 0.005     # PLACEHOLDER: profit-sharing pays mu_t minus this margin (Eq. 5b)
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
    MPR: Optional[float] = None  # market price of rate risk lambda_r (Eq. 7); None: model 10Y = theta at r = theta_r
    FUND_DURATION: float = 7.0   # PLACEHOLDER: maturity D of the zero-coupon the bond fund rolls monthly (Eq. 8)
    FUND_FEE: float = 0.0        # PLACEHOLDER: Branch 23 fund fee c_fee per year, charged monthly (Eq. 8)
    TRAIN_SEED: int = 2026       # noise pool of the RL environment
    EVAL_SEED: int = 7           # noise of simulate() and of evaluation episodes


BASELINE = Params()


# --- rates ---------------------------------------------------------------------------
def vasicek_params(p):
    """The parameters of Eqs. (1)-(2): kappa, sigma, y0 (last observed 10Y) from the
    OLS calibration; s_term = SPREAD_10Y_SHORT, or the historical mean (10Y - 1Y);
    theta = the OLS estimate, or LONG_RATE_P + s_term (LONG_RATE_P: long-run short rate)."""
    cal = calibration()
    v = cal["vasicek"]
    spread = cal["spread"] if p.SPREAD_10Y_SHORT is None else p.SPREAD_10Y_SHORT
    theta = v["theta"] if p.LONG_RATE_P is None else p.LONG_RATE_P + spread
    return dict(kappa=v["kappa"], sigma=v["sigma"], theta=theta, y0=v["y0"], spread=spread)


def premonths():
    """Simulated months from the last OLO observation to January of model year 0."""
    return (13 - calibration()["t0"].month) % 12


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


def wap_grid():
    """Eq. (3), MODEL.md: the 9 values G can take, 1.75%, 2.00%, ..., 3.75%. Rounded to
    10 decimals exactly as wap_formula rounds, so G == g holds exactly on the grid."""
    n = int(round((WAP_CAP - WAP_FLOOR) / WAP_STEP)) + 1
    return np.round(WAP_FLOOR + WAP_STEP * np.arange(n), 10)


def book_yield(mu, y, p):
    """Eq. (4), MODEL.md: monthly exponential filter of the 10Y, mu <- mu + w (y - mu)."""
    return mu + p.BOOK_WEIGHT * (y - mu)


TARIFF_CAP = 0.0375   # highest Branch 21 tariff (Eq. 5a)


def tariff_grid(p):
    """Eq. (5a), MODEL.md: the 16 tariff levels i_k = 0, 0.25%, ..., 3.75%, derived like
    wap_grid() and rounded to 10 decimals, so i == i_k holds exactly on the grid."""
    n = int(round(TARIFF_CAP / p.TARIFF_STEP)) + 1
    return np.round(p.TARIFF_STEP * np.arange(n), 10)


def tariff(mu, p):
    """Eq. (5a), MODEL.md: the tariff locked to retirement for a premium paid when the
    insurer's book yield is mu (the filter of Eq. 4, read in July):
    i = floor_to_grid( min(3.75%, max(0, mu - TARIFF_MARGIN)) ). Based on the book
    yield, not the 10Y, because observed group tariffs lag the market the way a
    smoothed portfolio yield does. The ratio is rounded to 9 decimals before the
    floor, so a value on a grid point (the cap) is not floored one step down by rounding."""
    x = np.minimum(TARIFF_CAP, np.maximum(0.0, mu - p.TARIFF_MARGIN))
    return np.round(np.floor(np.round(x / p.TARIFF_STEP, 9)) * p.TARIFF_STEP, 10)


def bucket_rate(i_k, mu, p):
    """Eq. (5b), MODEL.md: the year's rate of a Branch 21 bucket with tariff i_k,
    max(i_k, mu_t - PS_MARGIN), mu_t read in January: the tariff plus profit-sharing
    (rate - i_k >= 0 by construction)."""
    return np.maximum(i_k, mu - p.PS_MARGIN)


def new_premium_rate(mu, p):
    """January proxy for what this year's premium would earn in Branch 21:
    bucket_rate(tariff(mu_Jan), mu_t). The real tariff is set at the July book yield."""
    return bucket_rate(tariff(mu, p), mu, p)


# --- balances ------------------------------------------------------------------------
def accrue(X, P, rate, p):
    """Eq. (6a), MODEL.md: X_{t+1} = X_t e^{rate} + P_t e^{rate / 2}, in units of S_{t+1}.
    The premium is paid mid-year, so it earns half of the year's rate. Each Branch 21
    bucket (Eq. 6c) at its bucket rate, and each liability bucket (Eq. 6b) at its g."""
    return (X * np.exp(rate) + P * np.exp(rate / 2)) / (1 + p.W)


def accrue_branch21(R21k, P21, i_new, mu, p):
    """Eq. (6c), MODEL.md: horizontal Branch 21, one bucket per tariff level i_k,
    R21^(k)_{t+1} = R21^(k)_t e^{r_k} + P21_t 1{i_new = i_k} e^{r_k / 2}, r_k = bucket_rate,
    in units of S_{t+1}. R21k is (16, n); i_new is the tariff of this year's (July)
    premium. In force and after exit alike: a paid-up reserve keeps its tariff and
    its profit-sharing."""
    grid = tariff_grid(p)
    # an off-grid tariff would drop its premium from every bucket
    assert np.isin(i_new, grid).all(), "tariff is not on the grid"
    out = np.empty_like(R21k)
    for k, i_k in enumerate(grid):
        out[k] = accrue(R21k[k], P21 * (i_new == i_k), bucket_rate(i_k, mu, p), p)
    return out


def accrue_liability(Lg, P, G, inforce, p):
    """Eq. (6b), MODEL.md: horizontal WAP liability, one bucket per statutory rate g,
    L^(g)_{t+1} = L^(g)_t e^g + P_t 1{G_t = g} e^{g / 2}, in units of S_{t+1}.
    Lg is (9, n). Each premium keeps the rate G of the year it was paid in; after
    exit the buckets are frozen in money (rate 0)."""
    grid = wap_grid()
    # an off-grid G would drop its premium from every bucket
    assert np.isin(G, grid).all(), "G is not on the statutory grid"
    out = np.empty_like(Lg)
    for k, g in enumerate(grid):
        out[k] = accrue(Lg[k], P * (G == g), g * inforce, p)
    return out


def zero_price(r, tau, theta_q, vp):
    """Eq. (7), MODEL.md: Vasicek zero-coupon price of maturity tau at short rate r,
    A exp(-B r), B = (1 - e^{-kappa tau}) / kappa,
    ln A = (theta_Q - sigma^2 / (2 kappa^2)) (B - tau) - sigma^2 B^2 / (4 kappa)."""
    k, s = vp["kappa"], vp["sigma"]
    B = (1 - np.exp(-k * tau)) / k
    lnA = (theta_q - s ** 2 / (2 * k ** 2)) * (B - tau) - s ** 2 * B ** 2 / (4 * k)
    return np.exp(lnA - B * r)


def market_price_of_risk(vp, p):
    """Eq. (7a), MODEL.md: lambda_r = MPR, or (None) the value at which the model's own
    10Y zero yield at r = theta_r equals theta, the long-run mean of the simulated 10Y:
    -ln P(10) / 10 = theta. ln P is linear in theta_Q, so
    theta_Q = sigma^2 / (2 kappa^2) + (B theta_r + sigma^2 B^2 / (4 kappa) - 10 theta) / (B - 10),
    B = B(10), and lambda_r = (theta_r - theta_Q) kappa / sigma. Without it (lambda_r = 0)
    the model's long yields sit below the simulated 10Y by s_term plus convexity."""
    if p.MPR is not None:
        return p.MPR
    k, s, tau = vp["kappa"], vp["sigma"], 10.0
    theta_r = vp["theta"] - vp["spread"]
    B = (1 - np.exp(-k * tau)) / k
    theta_q = s ** 2 / (2 * k ** 2) + (B * theta_r + s ** 2 * B ** 2 / (4 * k) - tau * vp["theta"]) / (B - tau)
    return (theta_r - theta_q) * k / s


def theta_q(vp, p):
    """Eq. (7a), MODEL.md: the pricing-measure mean of the short rate r = y - s_term,
    which has the 10Y's kappa, sigma and theta_r = theta - s_term. Under Q,
    dW^Q = dW^P + lambda_r dt, so theta_Q = theta_r - lambda_r sigma / kappa; a bond
    then earns r - lambda_r sigma B under P, i.e. lambda_r < 0 is a positive term premium."""
    return vp["theta"] - vp["spread"] - market_price_of_risk(vp, p) * vp["sigma"] / vp["kappa"]


def fund_return(y_now, y_next, vp, p):
    """Eq. (8), MODEL.md: one month's gross return of the bond fund, a zero-coupon of
    maturity D rolled every month: bought at P(D; r_now), sold a month later at
    P(D - 1/12; r_next), net of the fee (1 - c_fee)^{1/12}. r = y - s_term (Eq. 2)."""
    D, tq = p.FUND_DURATION, theta_q(vp, p)
    gross = zero_price(short_rate(y_next, vp), D - 1 / 12, tq, vp) / zero_price(short_rate(y_now, vp), D, tq, vp)
    return gross * (1 - p.FUND_FEE) ** (1 / 12)


JULY = 5   # row of July in a year's months, which run February .. next January


def accrue_fund(R23, y_jan, months, P23, vp, p):
    """Eq. (8), MODEL.md: the fund value over one year, in units of S_{t+1}. Each month
    it earns fund_return from the previous month's 10Y to this month's (y_jan, then
    the 12 rows of months, February .. next January); the July premium P23 is added
    after the July return. In force and after exit alike: the value stays invested."""
    y_prev = y_jan
    for k, y in enumerate(months):
        R23 = R23 * fund_return(y_prev, y, vp, p)
        if k == JULY:
            R23 = R23 + P23
        y_prev = y
    return R23 / (1 + p.W)


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


def july_premium_accrual(y, t, vp, p):
    """Eq. (11), MODEL.md: the accrual to T of a premium paid in July of year t,
    approximated as A(t, T) e^{-r_t / 2}: the January factor less half a year at
    the January short rate. An approximation: it ignores the drift of r over the
    half year (relative error of order (theta_r - r_t)(1/2 - B(1/2)), about 1e-4)."""
    return premium_accrual(y, t, vp, p) * np.exp(-short_rate(y, vp) / 2)


# --- state and one year ------------------------------------------------------------------
@dataclass
class State:
    """n careers in January of year t. y, G, H, K, mu, R23 are (n,) arrays; Lg is (9, n),
    the liability buckets L^(g) in the order of wap_grid(); R21k is (16, n), the
    Branch 21 buckets R21^(k) in the order of tariff_grid(). H and K are the sums of
    the observed months in the windows of G_{t+1} and G_{t+2}.
    inforce and tau only drive the roll-forward after exit: every decision state is
    in force, with tau = t."""
    t: int
    y: np.ndarray
    G: np.ndarray
    H: np.ndarray
    K: np.ndarray
    mu: np.ndarray
    Lg: np.ndarray
    R21k: np.ndarray
    R23: np.ndarray
    inforce: np.ndarray
    tau: np.ndarray

    @property
    def L(self):
        """Eq. (6b), MODEL.md: the total liability L_t = sum_g L^(g)_t."""
        return self.Lg.sum(axis=0)

    @property
    def R21(self):
        """Eq. (6c), MODEL.md: the Branch 21 reserve R21_t = sum_k R21^(k)_t."""
        return self.R21k.sum(axis=0)

    def obs(self, p):
        """The observation (32 values): t / T, y, G, H and K as means of their months,
        mu, R23, then the 9 liability buckets and the 16 Branch 21 buckets."""
        return np.column_stack([np.full_like(self.y, self.t / p.T), self.y, self.G, self.H / 20,
                                self.K / 8, self.mu, self.R23, self.Lg.T, self.R21k.T])


def initial_state(eps_pre, p):
    """January of year 0 from the observed OLO history and the pre-roll months
    eps_pre (premonths(), n). Zero balances: a new member."""
    vp, hist = vasicek_params(p), calibration()["hist10Y"]
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
                 np.zeros((len(wap_grid()), n)), np.zeros((len(tariff_grid(p)), n)), z,
                 np.ones(n, bool), z)


def step(s, alpha, eps, u, p):
    """One year for every path: twelve months of rates (eps: (12, n)); the premium is
    paid in July. The liability buckets accrue at their g, the Branch 21 buckets at
    their bucket rates (tariff, and profit-sharing on mu_t read in January); the
    July premium's Branch 21 share gets the tariff of the July book yield, and the
    rest goes into the bond fund after its July return; then exit with probability
    h(t) (u: (n,) uniforms). Out of force: no premium, liability buckets fixed
    (rate 0), the fund and the Branch 21 buckets keep earning."""
    vp = vasicek_params(p)
    y, mu, months, mus = s.y, s.mu, [], []
    for e in eps:                                         # February .. next January
        y = vasicek_month(y, e, vp)
        mu = book_yield(mu, y, p)
        months.append(y)
        mus.append(mu)
    m = np.array(months)
    P = p.C * s.inforce
    Lg = accrue_liability(s.Lg, P, s.G, s.inforce, p)
    R21k = accrue_branch21(s.R21k, alpha * P, tariff(mus[JULY], p), s.mu, p)
    R23 = accrue_fund(s.R23, s.y, m, (1 - alpha) * P, vp, p)
    leave = s.inforce & (u < hazard(s.t, p))
    return State(s.t + 1, y, wap_rate(s.H + m[:4].sum(0)), s.K + m.sum(0), m[4:].sum(0), mu,
                 Lg, R21k, R23, s.inforce & ~leave, np.where(s.inforce, s.t + 1, s.tau))


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
    """Monte Carlo of a policy obs (n, 32) -> alpha (n,) over n careers (seed: EVAL_SEED).
    "joint" is the objective J (A5) and "reward" the per-path reward; the premium cost
    sum_{t<tau} C (1+W)^-(T-t) A(t,T) e^{-r_t/2} (July premiums, Eq. 11), which no
    policy changes, is reported apart.
    by_year: rate diagnostics; "i_minus_G" uses the January proxy new_premium_rate;
    "ps" is the profit-sharing rate mu_t - PS_MARGIN - i_k, floored at 0, weighted by
    the Branch 21 reserve over all buckets and paths; "ps_share" the share of that
    reserve whose profit-sharing is > 0 (both 0 in a year with no reserve yet)."""
    noise = draw_noise(n, p.EVAL_SEED if seed is None else seed, p) if noise is None else noise
    vp = vasicek_params(p)
    s = initial_state(noise.eps[:premonths()], p)
    cost = np.zeros(s.y.shape)
    grid = tariff_grid(p)
    diag = {k: np.zeros(p.T) for k in ("G", "G_floor", "mu", "i_minus_G", "r", "ps", "ps_share")}
    for t in range(p.T):
        i = new_premium_rate(s.mu, p)
        for k, v in (("G", s.G), ("G_floor", s.G <= WAP_FLOOR + 1e-12), ("mu", s.mu),
                     ("i_minus_G", i - s.G), ("r", short_rate(s.y, vp))):
            diag[k][t] = v.mean()
        ps = np.array([bucket_rate(i_k, s.mu, p) - i_k for i_k in grid])      # (16, n)
        reserve = s.R21k.sum()
        if reserve > 0:
            diag["ps"][t] = (ps * s.R21k).sum() / reserve
            diag["ps_share"][t] = s.R21k[ps > 0].sum() / reserve
        alpha = np.clip(policy(s.obs(p)), 0.0, 1.0)
        cost += s.inforce * p.C * (1 + p.W) ** -(p.T - t) * july_premium_accrual(s.y, t, vp, p)
        s = step(s, alpha, *noise.year(t), p)
    emp, short = reward_legs(s.L, s.R21 + s.R23, s.tau, p)
    reward = p.LAMBDA * emp - (1 - p.LAMBDA) * short
    return dict(joint=float(reward.mean()), employee=float(emp.mean()), shortfall=float(short.mean()),
                shortfall_freq=float((short > 0).mean()), premium_cost=float(cost.mean()),
                reward=reward, L=s.L, R21=s.R21, R23=s.R23, by_year=diag)


# --- benchmark policies: obs (n, 32) -> alpha (n,) ----------------------------------------
def all_branch21(obs):
    """alpha = 1: the whole premium to Branch 21."""
    return np.ones(len(obs))


def all_fund(obs):
    """alpha = 0: the whole premium to the bond fund."""
    return np.zeros(len(obs))


def fifty_fifty(obs):
    """alpha = 1/2: half of the premium to each."""
    return np.full(len(obs), 0.5)


def market_practice_rule(p=BASELINE):
    """alpha = 1{ i_new >= G_t }: Branch 21 when the tariff a new premium would get
    covers this year's WAP rate. i_new is the tariff at the January book yield
    mu_Jan, a proxy: the real tariff is set at the July book yield."""
    def act(obs):
        return (tariff(obs[:, 5], p) >= obs[:, 2]).astype(float)
    return act


class PensionEnv(gym.Env):
    """One career per episode, gamma = 1. Observation = State.obs (32,), action =
    alpha_t. The reward is 0 until T; when the member leaves, the remaining years are
    rolled forward on the episode's own noise and the reward at T is paid at once.
    reset(options={"noise": Noise of one path}) replays given noise; adding
    "state": State restarts from a snapshot."""
    metadata = {"render_modes": []}

    def __init__(self, p=BASELINE, pool=2000):
        self.p, self.pool = p, pool
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(32,), dtype=np.float64)
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
            emp, short = reward_legs(s.L, s.R21 + s.R23, s.tau, p)
            reward = float(p.LAMBDA * emp[0] - (1 - p.LAMBDA) * short[0])
        return s.obs(p)[0], reward, done, False, {}
