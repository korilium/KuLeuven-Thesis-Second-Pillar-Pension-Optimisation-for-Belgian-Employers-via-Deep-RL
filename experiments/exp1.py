import numpy as np
from pension.baseline import PensionEnv, simulate
from pension import baseline_dp

env = PensionEnv()                       # obs: the 9-variable state; action: Branch 21 share alpha in [0, 1]
simulate(lambda obs: np.full(len(obs), 0.5))     # joint, both legs, shortfall, premium_cost, ...

sol = baseline_dp.solve()                # about 30 s
simulate(baseline_dp.policy(sol))  