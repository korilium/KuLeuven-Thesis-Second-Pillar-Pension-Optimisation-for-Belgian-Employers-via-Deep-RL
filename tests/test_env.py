"""The RL environment is the DP's model: same dynamics, same objective, same paths.

Contract: for a fixed policy, the mean episode return over the paths simulate()
uses equals simulate(...)["joint"] -- so a learned policy's score and the DP
oracle's value are on one scale.
"""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import pension.dp as dp
from pension.dynamics import Exogenous
from pension.economy import draw_rate_scenarios
from pension.envs.pension_env import DPPolicyAgent, PensionEnv
from pension.params import DEFAULT

N, SEED = 300, 5


def _returns(env, agent, exo, R0, L0, S0):
    rets = []
    for i in range(exo.n):
        obs, _ = env.reset(options=dict(exogenous=exo.paths(i), entry=(R0[i], L0[i], S0[i])))
        total, done = 0.0, False
        while not done:
            obs, r, done, _, _ = env.step(agent(obs))
            total += r
        rets.append(total)
    return np.array(rets)


@pytest.mark.parametrize("numeraire", ["retirement", "discounted"])
@pytest.mark.parametrize("model,objective,band", [
    ("constant", "baseline", None),
    ("constant", "cashflow_strain", (0.2, 0.9)),
    ("vasicek", "baseline", None),
])
def test_episode_returns_equal_simulate_joint(model, objective, band, numeraire):
    p = DEFAULT.replace(RATE_MODEL=model, EMPLOYER_NUMERAIRE=numeraire)
    Fg, rg = dp.make_F_grid(n=31), dp.make_rho_grid(n=25)
    pol = dp.solve(Fg=Fg, rg=rg, ag=dp.make_a_grid(n=7), n_quad=3, objective=objective, p=p)["policy"]
    R0, L0, S0 = dp.new_plan_init(N, np.random.default_rng(1))
    rates = None if model == "constant" else draw_rate_scenarios(N, p=p)
    sim = dp.simulate(pol, Fg, rg, R0=R0, L0=L0, S0=S0, n_paths=N, seed=SEED, band=band,
                      rates=rates, objective=objective, p=p)
    exo = Exogenous.draw(p, N, SEED, rates)
    env = PensionEnv(p=p, objective=objective, band=band)
    rets = _returns(env, DPPolicyAgent(pol, Fg, rg, p.T), exo, R0, L0, S0)
    assert abs(rets.mean() - sim["joint"]) <= 1e-12 * max(1.0, abs(sim["joint"]))


@pytest.mark.filterwarnings("ignore:.*observation space (minimum|maximum) value is:UserWarning")
def test_gymnasium_api():
    # F and log(rho) are unbounded by nature, hence the infinite Box bounds
    check_env(PensionEnv(), skip_render_check=True)


def test_random_episodes_terminate_with_finite_reward():
    env = PensionEnv(p=DEFAULT.replace(RATE_MODEL="vasicek"), rate_pool=50)
    env.action_space.seed(0)
    for ep in range(5):
        obs, _ = env.reset(seed=ep)
        steps, done = 0, False
        while not done:
            obs, r, done, trunc, info = env.step(env.action_space.sample())
            assert np.isfinite(r) and np.all(np.isfinite(obs)) and not trunc
            steps += 1
        assert 1 <= steps <= DEFAULT.T and "RR_tot" in info


def test_observation_has_the_short_rate():
    obs, _ = PensionEnv(p=DEFAULT.replace(SHORT_RATE=0.021)).reset(seed=0)
    assert obs.shape == (6,) and obs[5] == 0.021


def test_q_hull_white_has_no_retirement_numeraire():
    env = PensionEnv(p=DEFAULT.replace(RATE_MODEL="hull_white"), rate_pool=20)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="hull_white_p"):
        env.step(np.array([0.5]))
