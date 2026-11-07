"""The baseline (pension/baseline.py): Markov state, one recursion for liability and
Branch 21, cash-flow-matched Branch 23, and the environment = simulate() contract."""
import dataclasses

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from pension import baseline as b
from pension.rates.calibration import calibration
from pension.rates.wap import computeWAPRate, wap_formula

p = b.BASELINE
N = 200


def const(a):
    return lambda obs: np.full(len(obs), a)


def run(noise, alpha, q=p):
    """All states s_0 .. s_T of n careers under a constant alpha (also after exit)."""
    s = b.initial_state(noise.eps[:b.premonths()], q)
    out = [s]
    for t in range(q.T):
        s = b.step(s, np.full(s.y.shape, alpha), *noise.year(t), q)
        out.append(s)
    return out


def finish(env, a=0.7):
    """(observation, reward) of every remaining step of an episode."""
    out, done = [], False
    while not done:
        obs, r, done, _, _ = env.step(np.array([a]))
        out.append((obs, r))
    return out


@pytest.fixture(scope="module")
def noise():
    return b.draw_noise(N, p.EVAL_SEED)


def test_markov_snapshot_restart(noise):
    """Snapshot s_17, rebuild a fresh environment from it alone: same noise, same future."""
    stays = run(noise, 0.3)[17].inforce
    path = noise.path(int(np.argmax(stays)))
    env = b.PensionEnv()
    env.reset(options=dict(noise=path))
    for _ in range(17):
        env.step(np.array([0.3]))
    snap = dataclasses.replace(env.state)
    fresh = b.PensionEnv()
    fresh.reset(options=dict(noise=path, state=snap))
    rest_a, rest_b = finish(env), finish(fresh)
    assert len(rest_a) == len(rest_b)
    for (oa, ra), (ob, rb) in zip(rest_a, rest_b):
        assert np.array_equal(oa, ob) and ra == rb


def test_branch21_is_the_liability_when_credited_at_G(noise, monkeypatch):
    """With G and the tariff both held at 2.5% and no profit-sharing, Branch 21 is
    the liability in force; after exit L is fixed in money while R21 still accrues."""
    q = dataclasses.replace(p, PS_MARGIN=1.0)             # mu - PS_MARGIN < every tariff
    monkeypatch.setattr(b, "wap_rate", lambda sum24: np.full(sum24.shape, 0.025))
    monkeypatch.setattr(b, "tariff", lambda y, q: np.full(y.shape, 0.025))
    s = b.initial_state(noise.eps[:b.premonths()], q)
    for t in range(q.T):
        prev, s = s, b.step(s, np.ones(N), *noise.year(t), q)
        inf = prev.inforce
        assert np.allclose(s.R21[inf], s.L[inf], rtol=1e-13, atol=0)
        out = ~prev.inforce
        assert np.allclose(s.L[out] * (1 + p.W), prev.L[out], rtol=1e-13)    # L fixed in money
        assert np.all(s.R21[out] * (1 + p.W) > prev.R21[out])                # R21 accrues


def test_liability_does_not_depend_on_the_policy(noise):
    runs = [run(noise, a) for a in (0.0, 0.5, 1.0)]
    for states in zip(*runs):
        assert all(np.array_equal(states[0].L, s.L) for s in states[1:])


def test_branch21_never_decreases_and_branch23_pays_its_units(noise):
    """In money, no Branch 21 bucket ever falls (every bucket rate is >= 0), and the
    year's Branch 21 premium lands in exactly one bucket. Checked bucket by bucket in
    units of S_{t+1}, where a rate of 0 gives equality exactly (e^0 = 1)."""
    states = run(noise, 0.5)
    for s0, s1 in zip(states, states[1:]):
        assert np.all(s1.R21k >= s0.R21k / (1 + p.W))
        grew = (s1.R21k > s0.R21k * np.exp(np.array([b.bucket_rate(i_k, s0.mu, p) for i_k in b.tariff_grid(p)]))
                / (1 + p.W) * (1 + 1e-12))                # buckets that received the premium
        assert np.array_equal(grew.sum(axis=0), s0.inforce.astype(int))
    sT = states[-1]
    assert np.all(b.bond_price(sT.y, p.T, b.vasicek_params(p), p) == 1.0)   # R23_T = N_T P(T,T) = N_T


def test_scale_free_in_salary(noise):
    """Money is in salary units: the starting salary appears nowhere, and doubling
    the premium doubles every balance (all three recursions are linear in P)."""
    s1, s2 = run(noise, 0.4)[-1], run(noise, 0.4, dataclasses.replace(p, C=2 * p.C))[-1]
    for k in ("L", "R21", "N"):
        assert np.allclose(getattr(s2, k), 2 * getattr(s1, k), rtol=1e-13)


def test_episode_return_equals_simulate_reward(noise):
    sim = b.simulate(const(0.6), noise=noise)
    env = b.PensionEnv()
    for j in range(N):
        env.reset(options=dict(noise=noise.path(j)))
        total, done = 0.0, False
        while not done:
            _, r, done, _, _ = env.step(np.array([0.6]))
            total += r
        assert abs(total - sim["reward"][j]) <= 1e-12 * max(1.0, abs(total))


def test_wap_calendar_matches_the_statute_window():
    """G_t from the partial window sums H, K equals computeWAPRate (24 months ending
    in May of the year before) on the same monthly path, observed history first."""
    n, pre = 5, b.premonths()
    vp = b.vasicek_params(p)
    eps = np.random.default_rng(1).standard_normal((pre + 12 * p.T, n))
    y = [np.full(n, vp["y0"])]
    for e in eps:
        y.append(b.vasicek_month(y[-1], e, vp))
    states = run(b.Noise(eps, np.ones((p.T, n))), 0.5)
    hist = calibration()["hist10Y"]
    full = np.vstack([np.repeat(hist[:, None], n, 1), np.array(y[1:])])
    assert np.array_equal(np.array([s.G for s in states[:p.T]]),
                          computeWAPRate(full, len(hist) - 1 + pre, p.T))


def test_premium_accrual_matches_monte_carlo():
    """A(t,T) (Eq. 11) against the mean of exp(sum of monthly r / 12) over simulated
    months, from y = y0 at t = 20: within 4 standard errors plus 0.1% discretisation."""
    t, n = 20, 40000
    vp = b.vasicek_params(p)
    rng = np.random.default_rng(3)
    y, I = np.full(n, vp["y0"]), np.zeros(n)
    for _ in range(12 * (p.T - t)):
        I += b.short_rate(y, vp) / 12
        y = b.vasicek_month(y, rng.standard_normal(n), vp)
    mc, A = np.exp(I), b.premium_accrual(vp["y0"], t, vp, p)
    assert abs(mc.mean() / A - 1) < 4 * mc.std() / np.sqrt(n) / A + 1e-3


FSMA_PUBLISHED = {2016: 1.75, 2017: 1.75, 2018: 1.75, 2019: 1.75, 2020: 1.75, 2021: 1.75,
                  2022: 1.75, 2023: 1.75, 2024: 1.75, 2025: 2.50, 2026: 2.50}


def test_wap_formula_reproduces_fsma_rates():
    """The 85% rule on the NBB 10Y reproduces every FSMA rate 2016-2026 (2027 differs:
    formula 2.75%, published 2.50%; docs/REVIEW.md C1)."""
    from pension.rates.data import load_olo
    df = load_olo()[0]
    y = df["YIELD"].values / 100.0
    for yr, pub in FSMA_PUBLISHED.items():
        win = ((df["DATE"] >= f"{yr - 3}-06-01") & (df["DATE"] <= f"{yr - 1}-05-01")).values
        assert win.sum() == 24
        assert float(wap_formula(y[win].mean())) * 100 == pytest.approx(pub), yr


def test_train_and_eval_noise_are_separate():
    assert p.TRAIN_SEED != p.EVAL_SEED
    tr, ev = b.draw_noise(50, p.TRAIN_SEED), b.draw_noise(50, p.EVAL_SEED)
    assert not np.isin(tr.eps[:, 0], ev.eps).any()


def test_liability_buckets_are_the_premiums_at_their_own_rate():
    """Horizontal WAP (Eq. 6b): G is always on the 9-point grid, the buckets are
    non-negative, and each bucket, recomputed premium by premium (paid in July of
    year s at G_s), equals the recursion; their sum is the total L."""
    n = 50
    noise = b.draw_noise(n, 11)
    states = run(b.Noise(noise.eps, np.ones_like(noise.u)), 0.5)       # all stay in force
    grid = b.wap_grid()
    assert len(grid) == 9 and grid[0] == 0.0175 and grid[-1] == 0.0375
    for t, s in enumerate(states):
        assert np.isin(s.G, grid).all()
        assert np.all(s.Lg >= 0)
        for k, g in enumerate(grid):
            # premium of year j in units of S_t: C (1 + W)^-(t - j), accrued t - j - 1/2 years
            by_hand = sum(p.C * (states[j].G == g) * np.exp(g * (t - j - 0.5)) * (1 + p.W) ** -(t - j)
                          for j in range(t))
            np.testing.assert_allclose(s.Lg[k], by_hand, rtol=1e-12, atol=1e-300)
        np.testing.assert_array_equal(s.L, s.Lg.sum(axis=0))


def july_10Y(s, eps, vp):
    """The July 10Y of the year starting in state s (the months run February .. January)."""
    y = s.y
    for e in eps[:b.JULY + 1]:
        y = b.vasicek_month(y, e, vp)
    return y


def test_tariffs_on_grid_buckets_sum_and_profit_sharing_nonnegative(noise):
    """Eq. (5a)-(6c): the tariff grid is 0, 0.25%, ..., 3.75% (16 levels); every
    tariff a July premium gets lies on it; the Branch 21 buckets sum to R21; and no
    bucket's profit-sharing is negative."""
    grid = b.tariff_grid(p)
    assert len(grid) == 16 and grid[0] == 0.0 and grid[-1] == 0.0375
    y = np.linspace(-0.03, 0.10, 2001)
    i = b.tariff(y, p)
    assert np.isin(i, grid).all() and i.min() == 0.0 and i.max() == 0.0375
    assert np.all(i <= np.maximum(y - p.TARIFF_MARGIN, 0) + 1e-15)    # floored, never rounded up
    vp, states = b.vasicek_params(p), run(noise, 0.5)
    for t, s in enumerate(states[:-1]):
        assert np.isin(b.tariff(july_10Y(s, noise.year(t)[0], vp), p), grid).all()
        np.testing.assert_array_equal(s.R21, s.R21k.sum(axis=0))
        for i_k in grid:
            assert np.all(b.bucket_rate(i_k, s.mu, p) >= i_k)


def test_branch21_vintages_without_profit_sharing_are_closed_form():
    """PS_MARGIN very large (no profit-sharing): every bucket is the sum of its
    vintages, each premium of year j worth C (1 + W)^-(t - j) e^{i_j (t - j - 1/2)} in
    units of S_t, with i_j the tariff at year j's July 10Y."""
    q, n = dataclasses.replace(b.BASELINE, PS_MARGIN=1.0), 50
    noise = b.draw_noise(n, 12, q)
    states = run(b.Noise(noise.eps, np.ones_like(noise.u)), 1.0, q)   # all in force, all Branch 21
    vp, grid = b.vasicek_params(q), b.tariff_grid(q)
    i = [b.tariff(july_10Y(s, noise.year(j)[0], vp), q) for j, s in enumerate(states[:-1])]
    for t, s in enumerate(states):
        for k, i_k in enumerate(grid):
            by_hand = sum(q.C * (i[j] == i_k) * np.exp(i_k * (t - j - 0.5)) * (1 + q.W) ** -(t - j)
                          for j in range(t))
            np.testing.assert_allclose(s.R21k[k], by_hand, rtol=1e-12, atol=1e-300)


def test_horizontal_equals_single_rate_when_G_is_constant(noise, monkeypatch):
    """G forced constant over the career: the horizontal liability is the single-rate
    recursion L_{t+1} = (L_t e^G + P_t e^{G/2}) / (1 + W), rate 0 after exit, exactly."""
    G = 0.0275
    monkeypatch.setattr(b, "wap_rate", lambda sum24: np.full(sum24.shape, G))
    L, states = np.zeros(N), run(noise, 0.5)
    for s, s1 in zip(states, states[1:]):
        P = p.C * s.inforce
        g = G * s.inforce
        L = (L * np.exp(g) + P * np.exp(g / 2)) / (1 + p.W)
        np.testing.assert_array_equal(s1.L, L)


def test_bond_price_closed_form():
    """Eq. (7): P(T, T) = 1; sigma -> 0 with lambda_r = 0 is the deterministic-rate
    price; and with lambda_r != 0 the closed form is E^Q[exp(-int r)] by Monte Carlo
    (within 4 standard errors plus 0.1% for the monthly trapezoid)."""
    vp = b.vasicek_params(p)
    y = np.array([0.0, 0.02, 0.0412, 0.07])
    assert np.all(b.bond_price(y, p.T, vp, p) == 1.0)

    t = 20.5
    tau, k = p.T - t, vp["kappa"]
    calm, p0 = dict(vp, sigma=1e-9), dataclasses.replace(p, MPR=0.0)
    r, theta_r = b.short_rate(y, vp), vp["theta"] - vp["spread"]
    B = (1 - np.exp(-k * tau)) / k
    np.testing.assert_allclose(b.bond_price(y, t, calm, p0), np.exp(-(theta_r * tau + (r - theta_r) * B)),
                               rtol=1e-12)

    q = dataclasses.replace(p, MPR=-0.3)
    theta_q = theta_r - q.MPR * vp["sigma"] / k
    vq = dict(vp, theta=theta_q + vp["spread"])           # the 10Y step simulates r + s_term
    n, rng = 40000, np.random.default_rng(4)
    ym, integral = np.full(n, vp["y0"]), np.zeros(n)
    months = int(round(12 * tau))
    for _ in range(months):
        y1 = b.vasicek_month(ym, rng.standard_normal(n), vq)
        integral += (b.short_rate(ym, vp) + b.short_rate(y1, vp)) / 2 / 12
        ym = y1
    mc, P = np.exp(-integral), b.bond_price(vp["y0"], t, vp, q)
    assert abs(mc.mean() / P - 1) < 4 * mc.std() / np.sqrt(n) / P + 1e-3


def test_calibrated_mpr_matches_the_10Y_mean():
    """Eq. (7a): with MPR = None the model's own 10Y zero yield at r = theta_r is theta;
    an explicit MPR is used as given; lambda_r < 0 raises long yields (term premium)."""
    vp = b.vasicek_params(p)
    lam = b.market_price_of_risk(vp, p)
    theta_r = vp["theta"] - vp["spread"]
    theta_q = theta_r - lam * vp["sigma"] / vp["kappa"]
    assert -np.log(b.zero_price(theta_r, 10.0, theta_q, vp)) / 10 == pytest.approx(vp["theta"], abs=1e-14)
    assert lam < 0
    assert b.market_price_of_risk(vp, dataclasses.replace(p, MPR=0.2)) == 0.2
    y = np.array([vp["theta"]])
    assert b.locked_yield(y, 0, vp, p) > b.locked_yield(y, 0, vp, dataclasses.replace(p, MPR=0.0))


def test_zero_fee_is_the_no_fee_unit_path(noise):
    """Eq. (8): with c_fee = 0 the units follow N_{t+1} = (N_t + P23_t / P(t+1/2, T)) / (1 + W)
    exactly; with c_fee > 0 a single July purchase of year 0 holds (1 - c_fee)^{t - 1/2}
    times as many units in January of year t."""
    vp = b.vasicek_params(p)
    states, Nn = run(noise, 0.3), np.zeros(N)
    for t, s in enumerate(states[:-1]):
        y = s.y
        for e in noise.year(t)[0][:b.JULY + 1]:           # February .. July
            y = b.vasicek_month(y, e, vp)
        Nn = (Nn + 0.7 * p.C * s.inforce / b.bond_price(y, t + 0.5, vp, p)) / (1 + p.W)
        np.testing.assert_array_equal(states[t + 1].N, Nn)

    c = 0.01
    def first_year_only(q):
        s, out = b.initial_state(noise.eps[:b.premonths()], q), []
        for t in range(q.T):
            s = b.step(s, np.full(N, 0.0 if t == 0 else 1.0), *noise.year(t), q)
            out.append(s.N)
        return out
    plain, fee = first_year_only(p), first_year_only(dataclasses.replace(p, FUND_FEE=c))
    for t in range(1, p.T + 1):
        np.testing.assert_allclose(fee[t - 1], plain[t - 1] * (1 - c) ** (t - 0.5), rtol=1e-12)


def test_leaver_keeps_frozen_buckets_and_is_settled_at_T(noise):
    """A member who leaves at tau keeps every bucket fixed in money and the units net
    of the fee; the shortfall is (L_tau - R_T)^+ at T, R_T having accrued after exit."""
    q = dataclasses.replace(p, FUND_FEE=0.01)
    states = run(noise, 0.4, q)
    sT = states[-1]
    left = ~sT.inforce
    assert left.sum() > 0
    for j in np.flatnonzero(left):
        tau = int(sT.tau[j])
        d = (1 + q.W) ** -(q.T - tau)
        np.testing.assert_allclose(sT.Lg[:, j], states[tau].Lg[:, j] * d, rtol=1e-12)
        np.testing.assert_allclose(sT.N[j], states[tau].N[j] * (1 - q.FUND_FEE) ** (q.T - tau) * d, rtol=1e-12)
        assert sT.R21[j] >= states[tau].R21[j] * d
    sim = b.simulate(const(0.4), noise=noise, p=q)
    emp, short = b.reward_legs(sT.L, sT.R21 + sT.N, sT.tau, q)
    np.testing.assert_array_equal(sim["reward"], q.LAMBDA * emp - (1 - q.LAMBDA) * short)
    L_tau = np.array([states[int(sT.tau[j])].L[j] for j in range(N)]) * (1 + q.W) ** -(q.T - sT.tau)
    np.testing.assert_allclose(short[left], np.maximum(L_tau - (sT.R21 + sT.N), 0)[left], rtol=1e-12, atol=1e-15)


@pytest.mark.filterwarnings("ignore:.*observation space (minimum|maximum) value is:UserWarning")
def test_gymnasium_api():
    check_env(b.PensionEnv(pool=20), skip_render_check=True)
