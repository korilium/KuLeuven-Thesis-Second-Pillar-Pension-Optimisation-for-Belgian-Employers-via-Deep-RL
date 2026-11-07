"""The DP of the baseline (pension/baseline_dp.py): its exit branch is the
environment's own roll-forward, its clipping guard holds, its greedy policy beats
every constant policy out of sample and runs inside PensionEnv, and with
deterministic rates it recovers the known per-premium answer. Small n (2000) so
the suite stays fast; the full-size numbers are in the stage reports."""
import numpy as np
import pytest

from pension import baseline as b
from pension import baseline_dp as bdp

p = b.BASELINE
N_DP = 2000


@pytest.fixture(scope="module")
def sol():
    return bdp.solve(n=N_DP, actions=np.linspace(0, 1, 6))


def test_paidup_growth_is_the_step_roll_forward():
    """Leave at tau = 12: stepping out of force to T equals L fixed in money and the
    Branch 21 buckets and the fund times their paid-up growth along the path."""
    n, tau = 40, 12
    noise = b.draw_noise(n, 5)
    stay = np.ones_like(noise.u)
    states = [b.initial_state(noise.eps[:b.premonths()], p)]
    for t in range(p.T):
        states.append(b.step(states[-1], np.full(n, 0.5), noise.year(t)[0], stay[t], p))
    s = states[tau]
    left = b.State(s.t, s.y, s.G, s.H, s.K, s.mu, s.Lg, s.R21k, s.R23, np.zeros(n, bool), np.full(n, tau))
    for t in range(tau, p.T):
        left = b.step(left, np.zeros(n), noise.year(t)[0], stay[t], p)
    d = (1 + p.W) ** -(p.T - tau)
    grow21, grow23 = bdp.paidup_growth(states, p), bdp.paidup_fund_growth(states, noise, p)
    np.testing.assert_allclose(left.L, s.L * d, rtol=1e-12)
    np.testing.assert_allclose(left.R23, s.R23 * grow23[tau], rtol=1e-12)
    np.testing.assert_allclose(left.R21k, s.R21k * grow21[tau], rtol=1e-12)
    np.testing.assert_allclose(bdp.leaver_value(s, grow21, grow23, p),
                               bdp._reward(left.L, left.R21 + left.R23, tau, p), rtol=1e-12)


def test_clipping_guard_holds(sol):
    """Every Q the policy sees lies within the range of its action's training targets,
    and the binding shares clip_stats reports are shares."""
    noise = b.draw_noise(500, p.EVAL_SEED)
    s = b.initial_state(noise.eps[:b.premonths()], p)
    for t in range(p.T):
        fit = sol["fits"][t]
        q = bdp.q_values(fit, s.obs(p), t, p)
        assert np.all(q >= fit["lo"]) and np.all(q <= fit["hi"])
        s = b.step(s, bdp.policy(sol)(s.obs(p)), *noise.year(t), p)
    stats = bdp.clip_stats(sol, n=500)
    for v in stats.values():
        assert v.shape == (p.T,) and np.all((v >= 0) & (v <= 1))


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


def per_premium_answer(p):
    """With deterministic rates every premium's choice is separate: the reward rises
    with R_T, and both assets keep accruing after exit, so the year-t premium should
    go where it is worth more at T. Returns alpha*_t = 1{Branch 21 multiple >= fund
    multiple} for t = 0 .. T-1, both multiples in money from July of year t to T, as logs:
      Branch 21: r_t / 2 + sum_{s > t} r_s, r_s = bucket_rate(tariff(y_Jul,t), mu_s);
      fund:      the sum of the log monthly returns after the July return of year t."""
    vp = b.vasicek_params(p)
    noise = b.draw_noise(1, 0, p)
    s = b.initial_state(noise.eps[:b.premonths()], p)
    years = []                                  # (January state, the 13 month-end 10Ys)
    for t in range(p.T):
        y, ys = s.y, [s.y]
        for e in noise.year(t)[0]:
            y = b.vasicek_month(y, e, vp)
            ys.append(y)
        years.append((s, ys))
        s = b.step(s, np.ones(1), *noise.year(t), p)
    alpha = np.zeros(p.T)
    for t, (s, ys) in enumerate(years):
        i_new = b.tariff(ys[b.JULY + 1], p)     # ys[0] is January, ys[6] July
        r21 = b.bucket_rate(i_new, s.mu, p) / 2 + sum(
            b.bucket_rate(i_new, years[u][0].mu, p) for u in range(t + 1, p.T))
        r23 = sum(np.log(b.fund_return(ys[k], ys[k + 1], vp, p)) for k in range(b.JULY + 1, 12))
        r23 = r23 + sum(np.log(b.fund_return(yu[k], yu[k + 1], vp, p))
                        for _, yu in years[t + 1:] for k in range(12))
        alpha[t] = float((r21 >= r23)[0])
    return alpha


@pytest.mark.slow
def test_dp_matches_the_per_premium_answer_when_rates_are_deterministic(monkeypatch):
    """sigma -> 0 and the monthly normals set to 0: the rate path is deterministic and
    the same on every path (only exits differ). The DP policy matches the per-premium
    comparison on >= 95% of the in-force evaluation states."""
    real_vp, real_noise = b.vasicek_params, b.draw_noise
    monkeypatch.setattr(b, "vasicek_params", lambda q: dict(real_vp(q), sigma=1e-9))
    monkeypatch.setattr(b, "draw_noise", lambda n, seed, q=p: b.Noise(np.zeros_like(real_noise(n, seed, q).eps),
                                                                    real_noise(n, seed, q).u))
    best = per_premium_answer(p)
    assert 0 < best.sum() < p.T                          # the answer switches: not a trivial test
    sol = bdp.solve(n=N_DP, actions=np.linspace(0, 1, 6))
    noise = b.draw_noise(N_DP, p.EVAL_SEED)
    s, agree, decisions = b.initial_state(noise.eps[:b.premonths()], p), 0, 0
    act = bdp.policy(sol)
    for t in range(p.T):
        a = act(s.obs(p))
        agree += int(((a == best[t]) & s.inforce).sum())
        decisions += int(s.inforce.sum())
        s = b.step(s, a, *noise.year(t), p)
    assert agree / decisions >= 0.95, (agree / decisions, best)
