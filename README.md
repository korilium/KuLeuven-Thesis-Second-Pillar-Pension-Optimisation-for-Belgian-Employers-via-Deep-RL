# Second-pillar pension funding for Belgian employers

The model behind the thesis: when and how much an employer should contribute to a
Belgian second-pillar (WAP/LPC) plan. It covers employee adequacy, the employer's
cost of the statutory return guarantee, staff turnover, and stochastic interest
rates. It has three rungs, which all share one economy:
- a tabular Monte Carlo agent (rung 1)
- a dynamic-programming oracle (rung 2)
- a reinforcement-learning environment (rung 3)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,rl]"
pytest -m "not slow"          # about 10 s; plain `pytest` also runs the slow checks
```

## Layout

| Path | What it is |
|---|---|
| `pension/params.py` | `Params`: every parameter in one immutable object. `DEFAULT` is the committed calibration; vary it with `DEFAULT.replace(ETA=3)`. |
| `pension/objective.py` | The value function as swappable parts: employee utility, shortfall cost, contribution cost, leg weights. Comes with a registry of named objectives. |
| `pension/dynamics.py` | The career dynamics: contribution, credited return, guarantee ledger (vertical or horizontal), churn, settlement. Used by everything below. |
| `pension/dp.py` | The DP oracle. `solve()` does backward induction over (t, F, ρ); `simulate()` scores a policy forward on the dynamics. |
| `pension/envs/pension_env.py` | The gymnasium environment for RL. An episode's return equals its path's contribution to `simulate`'s joint value. |
| `pension/envs/tabular.py` | Rung 1: tabular Monte Carlo control on a binary contribution decision. |
| `pension/rates/` | The rate engine: NBB OLO data (cached CSV), Vasicek/NSS/Hull-White calibration and simulation, bond pricing, and the statutory WAP rate. |
| `pension/economy.py` | Contribution plan rules and rate scenarios (G_t, μ_t). |
| `pension/checks.py` | Known-answer checks that measure. The tests assert them; the suites print them. |
| `experiments/dynpro/` | Figure suites (`*_suite.py`, with sections as CLI arguments and `--rates=hull_white\|vasicek`) and a printed `report.py`. |
| `experiments/olo/` | Plots and validation of the rate calibration. |
| `experiments/tabular/` | Runner for the rung-1 configurations. |
| `results/` | All figures and run outputs. |
| `tests/` | pytest: golden regression, dynamics, invariants, rates, objectives, pricing, environment. |
| `paper/` | Thesis drafts and derivations. |

## Using the model

```python
import pension.dp as dp
from pension.params import DEFAULT

p = DEFAULT.replace(RATE_MODEL="hull_white", OBJECTIVE="loss_averse")
Fg, rg, ag = dp.grids(nF=73, nR=71, na=20)
policy = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=5, p=p)["policy"]
result = dp.simulate(policy, Fg, rg, n_paths=15000, p=p)       # joint, cost, RR, ...
```

```python
from pension.envs.pension_env import PensionEnv
env = PensionEnv(p=p)          # obs: (t/T, F, log rho, G_t, mu_t); action: a in [0, 1]
```
