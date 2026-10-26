"""The baseline (pension/baseline.py): Markov state, one recursion for liability and
Branch 21, cash-flow-matched Branch 23, and the environment = simulate() contract."""
import dataclasses

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from pension import baseline as b
from pension.economy import rate_calibration, vasicek_params
from pension.params import DEFAULT
from pension.rates.accrual import closed_form_accrual
from pension.rates.simulation import simulateVasicek
from pension.rates.wap import computeWAPRate

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


def test_rates_match_the_rate_engine():
    """The monthly 10Y equals simulateVasicek on the same normals, G_t equals
    computeWAPRate on that path, and A(t,T) equals the engine's closed form."""
    n, pre = 5, b.premonths()
    vp = vasicek_params(p)
    y_eng = simulateVasicek(vp["kappa"], vp["theta"], vp["sigma"], vp["y0"], T=p.T + 1, n_paths=n,
                            rng=np.random.default_rng(1))
    eps = np.random.default_rng(1).standard_normal((12 * (p.T + 1), n))[:pre + 12 * p.T]
    nz = b.Noise(eps, np.ones((p.T, n)))                   # u = 1: nobody leaves
    states = run(nz, 0.5)
    assert np.allclose([s.y for s in states], y_eng[pre::12][:p.T + 1], rtol=1e-12)
    full = np.vstack([np.repeat(rate_calibration()["hist10Y"][:, None], n, 1), y_eng[1:]])
    G = computeWAPRate(full, len(rate_calibration()["hist10Y"]) - 1 + pre, p.T)
    assert np.array_equal(np.array([s.G for s in states[:p.T]]), G)
    q = DEFAULT.replace(RATE_MODEL="vasicek")
    for t in (0, 20, 44):
        assert np.allclose(b.premium_accrual(states[t].y, t, vp, p),
                           closed_form_accrual(t, b.short_rate(states[t].y, vp), q), rtol=1e-12)


def test_train_and_eval_noise_are_separate():
    assert p.TRAIN_SEED != p.EVAL_SEED
    tr, ev = b.draw_noise(50, p.TRAIN_SEED), b.draw_noise(50, p.EVAL_SEED)
    assert not np.isin(tr.eps[:, 0], ev.eps).any()


@pytest.mark.filterwarnings("ignore:.*observation space (minimum|maximum) value is:UserWarning")
def test_gymnasium_api():
    check_env(b.PensionEnv(pool=20), skip_render_check=True)
