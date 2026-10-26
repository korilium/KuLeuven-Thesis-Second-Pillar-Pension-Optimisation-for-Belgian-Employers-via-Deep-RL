"""pension.legacy_v1 reproduces the pre-October-2026 model: the goldens recorded on
tag v1-contribution-action (tests/regression/make_golden_v1.py), the old
environment's contract with its DP (episode returns = simulate()["joint"]), and
the rate mode of the DP stack (horizontal ledger, CE premium, scenario guard)."""
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


# --- the rate mode of the DP stack (moved here from test_dynamics / test_mechanisms) ---
from pension.legacy_v1 import economy, numeraire                  # noqa: E402
from pension.legacy_v1.dynamics import State, step               # noqa: E402
from pension.legacy_v1.rates.accrual import closed_form_accrual  # noqa: E402


def _exo(p, zR, zL, u, G=None, mu=None):
    tile = lambda x: np.tile(np.asarray(x, float), (p.T, 1))
    return Exogenous(tile(zR), tile(zL), tile(u),
                     None if G is None else tile(G), None if mu is None else tile(mu))


def test_degenerate_rates_equal_constant_rates():
    """G_t = G and mu_t = MU, SIGMA_L = 0: the horizontal ledger with one rate is
    the vertical ledger, so the states coincide."""
    p = DEFAULT.replace(SIGMA_L=0.0, SIGMA_R_RATES=DEFAULT.SIGMA_R)
    rng = np.random.default_rng(0); n = 50
    zR, zL, u = rng.standard_normal(n), rng.standard_normal(n), rng.random(n)
    a = rng.random(n)
    c_exo = _exo(p, zR, zL, u)
    r_exo = _exo(p, zR, zL, u, G=np.full(n, p.G), mu=np.full(n, p.MU))
    sc = State.initial(p, c_exo, 1.0, 1.0, 10.0); sr = State.initial(p, r_exo, 1.0, 1.0, 10.0)
    for _ in range(p.T):
        sc, _ = step(sc, np.where(sc.present, a, 0.0), c_exo, p, dp.tenure_hazard)
        sr, _ = step(sr, np.where(sr.present, a, 0.0), r_exo, p, dp.tenure_hazard)
    np.testing.assert_allclose(sr.R, sc.R, rtol=1e-13)
    np.testing.assert_allclose(sr.L, sc.L, rtol=1e-13)
    assert np.array_equal(sr.leave_t, sc.leave_t)


def test_horizontal_ledger_closed_form():
    """G steps 3% -> 1.75% at t = 20: old money keeps its rate,
    L_T = L0 e^{G_0 T} + sum_s c_s e^{G_s (T - s)}; vertical would differ."""
    p, T = DEFAULT, DEFAULT.T
    Gs = np.where(np.arange(T) < 20, 0.03, 0.0175)
    exo = Exogenous(np.zeros((T, 1)), np.zeros((T, 1)), np.ones((T, 1)), Gs[:, None], np.full((T, 1), 0.02))
    s, cs = State.initial(p, exo, R0=1.0, L0=1.0, S0=20.0), []
    for _ in range(T):
        s, c = step(s, np.array([0.4]), exo, p, lambda t: 0.0)
        cs.append(c[0])
    cs = np.array(cs)
    closed = np.exp(Gs[0] * T) + np.sum(cs * np.exp(Gs * (T - np.arange(T))))
    vertical = 1.0
    for t in range(T):
        vertical = (vertical + cs[t]) * np.exp(Gs[t])
    assert abs(s.L[0] / closed - 1) < 1e-12 and abs(vertical / closed - 1) > 0.05


def test_horizontal_ledger_compresses_into_rate_buckets():
    """G lives on a 25 bp grid in [1.75%, 3.75%]: the horizontal ledger is exactly
    9 buckets "liability locked at rate g"."""
    p = DEFAULT.replace(RATE_MODEL="hull_white")
    n = 50
    exo = Exogenous.draw(p, n, 3, draw_rate_scenarios(n, p=p))
    grid = np.round(np.arange(0.0175, 0.03751, 0.0025), 6)
    assert len(grid) == 9
    s = State.initial(p, exo, 1.0, 1.0, 15.0)
    buckets = np.zeros((len(grid), n))
    buckets[np.searchsorted(grid, np.round(exo.G[0], 6)), np.arange(n)] = 1.0
    rng = np.random.default_rng(0)
    for t in range(p.T):
        a = np.where(s.present, rng.random(n), 0.0)
        present = s.present.copy()
        s, c = step(s, a, exo, p, dp.tenure_hazard)
        g = np.searchsorted(grid, np.round(exo.G[t], 6))
        assert np.allclose(grid[g], exo.G[t], atol=1e-12)
        buckets[g, np.arange(n)] += c
        buckets = np.where(present, buckets * np.exp(grid[:, None]), buckets)
        np.testing.assert_allclose(buckets.sum(axis=0), s.L, rtol=1e-12)


def test_ce_premium_is_the_scenario_mean_of_the_accrual():
    q = DEFAULT.replace(RATE_MODEL="vasicek")
    sc = economy.draw_rate_scenarios(500, p=q)
    prem = numeraire.premium_schedule(q, sc)
    for t in (0, 7, 30, 44):
        assert prem[t] == pytest.approx(closed_form_accrual(t, sc["r"][t], q, s=q.FINANCING_SPREAD).mean(),
                                        rel=1e-12)


def test_scenario_of_another_model_is_refused():
    """REVIEW M2: dynamics and premiums must come from the same model."""
    p = DEFAULT
    Fg, rg, ag = dp.make_F_grid(n=31), dp.make_rho_grid(n=25), dp.make_a_grid(n=7)
    sc = economy.draw_rate_scenarios(20, p=p.replace(RATE_MODEL="vasicek"))
    pol = dp.const_policy(0.4, p.T, len(Fg), len(rg))
    with pytest.raises(ValueError):
        dp.simulate(pol, Fg, rg, n_paths=20, rates=sc, p=p)
    with pytest.raises(ValueError):
        dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=3, rates=sc, p=p)
