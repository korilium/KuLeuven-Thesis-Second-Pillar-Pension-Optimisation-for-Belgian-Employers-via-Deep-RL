# Second-pillar pension funding for Belgian employers

The model behind the thesis: how a Belgian employer should split a fixed
second-pillar (WAP/LPC) premium between Branch 21 and Branch 23. It covers employee
adequacy, the employer-borne WAP shortfall, staff turnover, and stochastic interest
rates. The rungs:
- a tabular Monte Carlo agent (rung 1)
- a dynamic-programming first-principles reference at constant rates, one vehicle (rung 2)
- the baseline RL environment (rung 3, `pension/baseline.py`, specified in `docs/MODEL.md`)
- `pension/legacy_v1/`: the pre-October-2026 model (contribution-level action), frozen for regression and comparison

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,rl]"
pytest -m "not slow"          # about 10 s; plain `pytest` also runs the slow checks
```

## Layout

| Path | What it is |
|---|---|
| `pension/baseline.py` | The baseline: parameters (`BASELINE`), the equations of `docs/MODEL.md`, `simulate()` and `PensionEnv`. |
| `pension/legacy_v1/` | The pre-October-2026 model, self-contained and frozen (see its README). |
| `pension/params.py` | `Params` of the DP and tabular rungs: every parameter in one immutable object. `DEFAULT` is the committed calibration; vary it with `DEFAULT.replace(ETA=3)`. |
| `pension/objective.py` | The reward function of the DP rung as swappable parts: employee utility, shortfall cost, contribution cost, leg weights. Comes with a registry of named objectives. |
| `pension/dynamics.py` | The career dynamics: contribution, credited return, guarantee ledger (vertical; horizontal in the legacy rate mode), churn, settlement. Used by everything below. |
| `pension/dp.py` | The DP first-principles reference (constant-rate rung). `solve()` does backward induction over (t, F, ρ); `simulate()` scores a policy forward on the dynamics. |
| `pension/envs/tabular.py` | Rung 1: tabular Monte Carlo control on a binary contribution decision. |
| `pension/rates/` | The rate engine: NBB OLO data (cached CSV), Vasicek/NSS/Hull-White calibration and simulation, bond pricing, and the statutory WAP rate. |
| `pension/economy.py` | Contribution plan rules and rate scenarios (G_t, μ_t). |
| `pension/checks.py` | Known-answer checks that measure. The tests assert them; the suites print them. |
| `experiments/dynpro/` | Figure suites (`*_suite.py`, with sections as CLI arguments and `--rates=hull_white\|vasicek`) and a printed `report.py`. |
| `experiments/olo/` | Plots and validation of the rate calibration. |
| `experiments/tabular/` | Runner for the rung-1 configurations. |
| `results/` | All figures and run outputs. |
| `tests/` | pytest: baseline, legacy_v1 goldens, golden regression, dynamics, invariants, rates, objectives, pricing. |
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
from pension.baseline import BASELINE, PensionEnv, simulate
env = PensionEnv()             # obs: the 9-variable state; action: Branch 21 share alpha in [0, 1]
simulate(lambda obs: 0.5 + 0 * obs[:, 0])   # joint, both legs, shortfall, premium_cost, ...
```
