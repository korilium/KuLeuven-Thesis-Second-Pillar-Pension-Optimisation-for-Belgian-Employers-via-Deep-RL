"""The DP of the baseline (pension/baseline_dp.py): its exit branch is the
environment's own roll-forward, and its greedy policy beats every constant policy
out of sample and runs inside PensionEnv."""
import numpy as np
import pytest

from pension import baseline as b
from pension import baseline_dp as bdp

p = b.BASELINE


@pytest.fixture(scope="module")
def sol():
    return bdp.solve(n=3000, actions=np.linspace(0, 1, 6))


def test_paidup_growth_is_the_step_roll_forward():
    """Leave at tau = 12: stepping out of force to T equals L, N fixed in money and
    R21 times paidup_growth."""
    n, tau = 40, 12
    noise = b.draw_noise(n, 5)
    stay = np.ones_like(noise.u)
    states = [b.initial_state(noise.eps[:b.premonths()], p)]
    for t in range(p.T):
        states.append(b.step(states[-1], np.full(n, 0.5), noise.year(t)[0], stay[t], p))
    s = states[tau]
    left = b.State(s.t, s.y, s.G, s.H, s.K, s.mu, s.L, s.R21, s.N, np.zeros(n, bool), np.full(n, tau))
    for t in range(tau, p.T):
        left = b.step(left, np.zeros(n), noise.year(t)[0], stay[t], p)
    d = (1 + p.W) ** -(p.T - tau)
    np.testing.assert_allclose(left.L, s.L * d, rtol=1e-12)
    np.testing.assert_allclose(left.N, s.N * d, rtol=1e-12)
    np.testing.assert_allclose(left.R21, s.R21 * bdp.paidup_growth(states, p)[tau], rtol=1e-12)


def test_policy_beats_every_constant_policy(sol):
    noise = b.draw_noise(4000, p.EVAL_SEED)
    J = b.simulate(bdp.policy(sol), noise=noise)["joint"]
    best = max(b.simulate(lambda o, a=a: np.full(len(o), a), noise=noise)["joint"] for a in sol["actions"])
    assert J > best, (J, best)


def test_policy_runs_in_the_environment(sol):
    noise = b.draw_noise(30, p.EVAL_SEED)
    pol = bdp.policy(sol)
    sim = b.simulate(pol, noise=noise)
    env = b.PensionEnv()
    for j in range(30):
        obs, _ = env.reset(options=dict(noise=noise.path(j)))
        total, done = 0.0, False
        while not done:
            obs, r, done, _, _ = env.step(pol(obs[None, :]))
            total += r
        assert abs(total - sim["reward"][j]) <= 1e-12 * max(1.0, abs(total))
