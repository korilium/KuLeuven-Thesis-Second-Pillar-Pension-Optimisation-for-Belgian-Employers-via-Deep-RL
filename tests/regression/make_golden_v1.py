"""Record the golden outputs of the pre-October-2026 model, frozen as pension.legacy_v1.

Run ONCE, on tag v1-contribution-action, before any later change:
    python tests/regression/make_golden_v1.py
tests/test_legacy_v1.py then checks that pension.legacy_v1 reproduces these files:
actions exactly, values to 1e-12 relative.

Per rate model (constant, vasicek) at the default Params: the DP on a small grid,
simulate() statistics on fixed seeds, and one PensionEnv episode.
"""
import importlib
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["constant", "vasicek"]
GRID = dict(nF=31, nR=25, na=7, nq=3)
SIM = dict(n_paths=3000, seed=7)
SIM_KEYS = ["benefit", "cost", "joint", "mean_a", "c_by", "frac", "RR_tot"]
ACTION_KEYS = {"policy", "mean_a", "c_by", "frac", "actions"}   # compared exactly


def path(model):
    return os.path.join(HERE, "golden", "legacy_v1", f"{model}.npz")


def run(pkg, model):
    """The outputs of one case; pkg = "pension" (recording) or "pension.legacy_v1"."""
    dp = importlib.import_module(f"{pkg}.dp")
    params = importlib.import_module(f"{pkg}.params")
    env_mod = importlib.import_module(f"{pkg}.envs.pension_env" if pkg == "pension" else f"{pkg}.env")
    p = params.DEFAULT.replace(RATE_MODEL=model)
    Fg, rg, ag = dp.make_F_grid(n=GRID["nF"]), dp.make_rho_grid(n=GRID["nR"]), dp.make_a_grid(n=GRID["na"])
    sol = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=GRID["nq"], p=p)
    out = dict(V=sol["V"][::4], policy=sol["policy"])
    R0, L0, S0 = dp.new_plan_init(SIM["n_paths"], np.random.default_rng(3))
    s = dp.simulate(sol["policy"], Fg, rg, R0=R0, L0=L0, S0=S0, p=p, **SIM)
    for k in SIM_KEYS:
        out[f"sim_{k}"] = np.asarray(s[k])
    env = env_mod.PensionEnv(p=p, rate_pool=50)
    obs, _ = env.reset(seed=0)
    obs_l, rew, done = [obs], [], False
    while not done:
        obs, r, done, _, _ = env.step(np.array([0.5]))
        obs_l.append(obs); rew.append(r)
    out["env_obs"], out["env_rew"] = np.array(obs_l), np.array(rew)
    return out


if __name__ == "__main__":
    for model in MODELS:
        np.savez_compressed(path(model), **run("pension", model))
        print("recorded", model)
