"""pension.legacy_v1 reproduces the pre-October-2026 model: the goldens recorded on
tag v1-contribution-action (tests/regression/make_golden_v1.py), and the old
environment's contract with its DP (episode returns = simulate()["joint"])."""
import os
import sys

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression"))
import make_golden_v1 as golden   # noqa: E402

import pension.legacy_v1.dp as dp
from pension.legacy_v1.dynamics import Exogenous
from pension.legacy_v1.economy import draw_rate_scenarios
from pension.legacy_v1.env import DPPolicyAgent, PensionEnv
from pension.legacy_v1.params import DEFAULT

RTOL = 1e-12
N, SEED = 300, 5


@pytest.mark.parametrize("model", golden.MODELS)
def test_matches_golden(model):
    ref = np.load(golden.path(model))
    cur = golden.run("pension.legacy_v1", model)
    assert set(ref.files) == set(cur), set(ref.files) ^ set(cur)
    for k in ref.files:
        a, b = ref[k], np.asarray(cur[k])
        assert a.shape == b.shape, (k, a.shape, b.shape)
        if k.split("_", 1)[-1] in golden.ACTION_KEYS:
            assert np.array_equal(a, b, equal_nan=True), f"{k}: actions differ"
        else:
            rel = np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)) if a.size else 0.0
            assert rel <= RTOL, f"{k}: max relative diff {rel:.2e}"


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
    check_env(PensionEnv(), skip_render_check=True)


def test_q_hull_white_has_no_retirement_numeraire():
    env = PensionEnv(p=DEFAULT.replace(RATE_MODEL="hull_white"), rate_pool=20)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="hull_white_p"):
        env.step(np.array([0.5]))
