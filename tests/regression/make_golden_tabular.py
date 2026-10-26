"""Record the golden outputs of the tabular rung (pension/envs/tabular.py), which this
work leaves untouched. Run once:  python tests/regression/make_golden_tabular.py

At the module's default economy: the frozen-batch values of three fixed policies, and
a short Monte Carlo control run (Q, greedy policy, reward trace) on a fixed seed.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "golden", "tabular.npz")
POLICIES = {"never": 0, "always": 1, "first_20": 20}


def run():
    import pension.envs.tabular as tab
    plan = tab.plan_fixed()
    batch = tab.draw_shock_batch()
    out = {f"batch_{k}": tab.run_batch(np.array([int(t < (tab.T if v == 1 else v)) for t in range(tab.T)]),
                                       plan, batch)
           for k, v in POLICIES.items()}
    mc = tab.mc_control(plan, n_episodes=5000, seed=0, burn_in=500, n0=50)
    out.update(mc_Q=mc["Q"], mc_policy=mc["policy"], mc_trace=mc["reward_trace"])
    return out


if __name__ == "__main__":
    np.savez_compressed(PATH, **run())
    print("recorded", PATH)
