"""The DP of the baseline (pension/baseline_dp.py): its exit branch is the
environment's own roll-forward, and its greedy policy beats every constant policy
out of sample and runs inside PensionEnv."""
import dataclasses

import numpy as np
import pytest

from pension import baseline as b
from pension import baseline_dp as bdp
from pension.rates.calibration import calibration

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
    left = b.State(s.t, s.y, s.G, s.H, s.K, s.mu, s.Lg, s.R21, s.N, np.zeros(n, bool), np.full(n, tau))
    for t in range(tau, p.T):
        left = b.step(left, np.zeros(n), noise.year(t)[0], stay[t], p)
    d = (1 + p.W) ** -(p.T - tau)
    np.testing.assert_allclose(left.L, s.L * d, rtol=1e-12)
    np.testing.assert_allclose(left.N, s.N * (1 - p.FUND_FEE) ** (p.T - tau) * d, rtol=1e-12)
    np.testing.assert_allclose(left.R21, s.R21 * bdp.paidup_growth(states, p)[tau], rtol=1e-12)
    grow = bdp.paidup_growth(states, p)
    np.testing.assert_allclose(bdp.leaver_value(s, grow, p), bdp._reward(left.L, left.R21 + left.N, tau, p),
                               rtol=1e-12)


def test_policy_beats_every_constant_policy(sol):
    noise = b.draw_noise(4000, p.EVAL_SEED)
    J = b.simulate(bdp.policy(sol), noise=noise)["joint"]
    best = max(b.simulate(lambda o, a=a: np.full(len(o), a), noise=noise)["joint"] for a in sol["actions"])
    assert J > best, (J, best)


def locked_branch21():
    """Known-answer configuration: BOOK_WEIGHT = 0 freezes mu at the first observed 10Y,
    so Branch 21 credits one constant rate i and is locked like the ladder. M puts i
    at the median locked yield of the evaluation states, so the threshold rule
    switches inside the state distribution (ladder early, Branch 21 late)."""
    q = dataclasses.replace(p, BOOK_WEIGHT=0.0)
    vp, mu0 = b.vasicek_params(q), calibration()["hist10Y"][0]
    noise = b.draw_noise(2000, q.EVAL_SEED, q)
    s, y_long = b.initial_state(noise.eps[:b.premonths()], q), []
    for t in range(q.T):
        y_long.append(b.locked_yield(s.y, t, vp, q))
        s = b.step(s, np.ones(2000), *noise.year(t), q)
    return dataclasses.replace(q, M=float(mu0 - np.median(y_long)))


@pytest.mark.slow
def test_dp_recovers_the_threshold_rule_when_branch21_is_locked():
    """With constant crediting both assets are locked, so alpha = 1{i >= y_long - c_fee}
    is optimal (up to the January proxy for the July price). The DP agrees with it on
    >= 95% of the in-force evaluation states, and its out-of-sample J is the rule's
    within 3 standard errors of the paired difference, or better."""
    q = locked_branch21()
    sol = bdp.solve(q, n=4000, actions=np.linspace(0, 1, 6))
    dp, rule = bdp.policy(sol, q), b.threshold_rule(q)
    noise = b.draw_noise(4000, q.EVAL_SEED, q)
    s, agree, decisions = b.initial_state(noise.eps[:b.premonths()], q), 0, 0
    switches = set()
    for t in range(q.T):
        obs = s.obs(q)
        a_dp, a_rule = dp(obs), rule(obs)
        agree += int(((a_dp == a_rule) & s.inforce).sum())
        decisions += int(s.inforce.sum())
        switches.update(np.unique(a_rule[s.inforce]))
        s = b.step(s, a_dp, *noise.year(t), q)
    assert switches == {0.0, 1.0}                         # the rule is not trivial here
    assert agree / decisions >= 0.95, agree / decisions
    diff = b.simulate(dp, noise=noise, p=q)["reward"] - b.simulate(rule, noise=noise, p=q)["reward"]
    assert diff.mean() >= -3 * diff.std() / np.sqrt(len(diff)), (diff.mean(), diff.std() / np.sqrt(len(diff)))


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
