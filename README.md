# Second-pillar pension funding for Belgian employers

The model behind the thesis: how a Belgian employer should split a fixed second-pillar (WAP/LPC) premium between Branch 21 and Branch 23. It covers employee adequacy, the employer-borne WAP shortfall, staff turnover, and a stochastic 10-year OLO.

The baseline is a fully observed MDP, specified equation by equation in `docs/MODEL.md`. It comes with:
- a gymnasium environment for RL;
- a DP reference on its exact state.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,rl]"
pytest                        # about 20 s
```

## Layout

| Path | What it is |
|---|---|
| `pension/baseline.py` | The baseline: parameters (`BASELINE`), the equations of `docs/MODEL.md`, `simulate()` and `PensionEnv`. |
| `pension/baseline_dp.py` | The DP reference: backward induction on the exact state with a fitted value function (`docs/MODEL.md` Sec. 7). |
| `pension/rates/` | The NBB OLO data (cached CSV), the Vasicek calibration of the 10Y with the term spread, and the statutory WAP rate. |
| `pension/envs/tabular.py` | The tabular rung: Monte Carlo control on a binary contribution decision. |
| `pension/legacy_v1/` | The pre-October-2026 model, self-contained and frozen for regression and comparison (see its README). |
| `experiments/baseline_report.py` | Every number in `docs/REVIEW.md`. |
| `experiments/tabular/` | Runner for the tabular configurations. |
| `tests/` | Baseline, DP, legacy_v1 and tabular regressions. |
| `docs/` | `MODEL.md` (specification), `REVIEW.md` (decisions, checks, results, future work). |
| `results/`, `paper/` | Figures and run outputs of the earlier rungs; thesis drafts and literature. |

The DP stack of the earlier rungs (constant-rate DP, dynamics, objectives, Hull-White engine, figure suites) is in tag `v2-before-cleanup`.

## Using the model

```python
import numpy as np
from pension.baseline import PensionEnv, simulate
from pension import baseline_dp

env = PensionEnv()                       # obs: the 9-variable state; action: Branch 21 share alpha in [0, 1]
simulate(lambda obs: np.full(len(obs), 0.5))     # joint, both legs, shortfall, premium_cost, ...

sol = baseline_dp.solve()                # about 30 s
simulate(baseline_dp.policy(sol))        # the DP policy, scored out of sample
```
