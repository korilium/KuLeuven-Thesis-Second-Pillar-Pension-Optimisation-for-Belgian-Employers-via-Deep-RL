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
    G_by_state = {}
    monkeypatch.setattr(b, "credited_rate", lambda mu, q: G_by_state["G"])
    s = b.initial_state(noise.eps[:b.premonths()], p)
    for t in range(p.T):
        G_by_state["G"] = s.G
        prev, s = s, b.step(s, np.ones(N), *noise.year(t), p)
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
    states = run(noise, 0.5)
    money = [s.R21 * (1 + p.W) ** s.t for s in states]
    assert all(np.all(m1 >= m0) for m0, m1 in zip(money, money[1:]))
    sT = states[-1]
    assert np.all(b.bond_price(sT.y, p.T, p) == 1.0)         # R23_T = N_T P(T,T) = N_T


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


@pytest.mark.filterwarnings("ignore:.*observation space (minimum|maximum) value is:UserWarning")
def test_gymnasium_api():
    check_env(b.PensionEnv(pool=20), skip_render_check=True)
