# The Scenario Battery — What Each Regime Is, and Why It Was Chosen

`tests/dynpro/scenario_suite.py` runs a fixed set of named parameter regimes rather
than a sweep. This note explains the selection: what each group tests, why those
particular configurations, and what the model actually does in each.

All numbers below come from `scenario_suite.py table` on the committed calibration
(`MU = G = DISC_EMP = 0.03`, `DISC_ER = 0.05`, `BETA = 0.01`, `SATIATE = False`)
at a 73×71 grid, `na = 20`, `n_quad = 5`, banded 2–15% of salary, 15 000 paths.
The schedule columns are resolution-sensitive, so those settings matter.

---

## Why a battery, when there is already a sensitivity suite

`sensitivity_suite.py` sweeps eleven parameters ±15% one at a time. That answers
"how much does the result move if this number is a bit wrong". It cannot answer two
other questions.

**First**, some of the model's behaviour is governed not by the *level* of any
parameter but by the **ordering** of several. A ±15% perturbation of `DISC_ER` around
0.05 never crosses `MU = 0.03`, so it never reveals that crossing that point reverses
the contribution schedule. One-at-a-time sweeps are blind to regime boundaries by
construction.

**Second**, a sweep has no notion of a right answer. Every output is "some number",
and nothing in the repository checks that the economics behaves as economics must.
A battery can include cases whose answer is known *before* running the model, and
those function as genuine tests.

---

## The organising principle

Consider one euro of contribution at career year $t$. The employer pays it then, and
discounts at $\delta_f$. It compounds in the reserve at $\mu$ until retirement at $T$.
The employee values the result at $T$, discounting at $\delta_e$. The ratio of what
the employee gains to what the employer gives up is

$$\frac{\text{benefit}}{\text{cost}}
= \frac{e^{\mu(T-t)}\,e^{-\delta_e T}}{e^{-\delta_f t}}
= \exp\big((\mu - \delta_e)\,T + (\delta_f - \mu)\,t\big).$$

Read the two terms separately, because they do different jobs.

The first, $(\mu - \delta_e)T$, is **constant in $t$**. It sets whether funding is
worth anything at all: if the reserve out-grows the employee's own discount rate, a
euro moved into the plan is a positive-NPV transfer before any risk consideration.

The second, $(\delta_f - \mu)\,t$, is **linear in $t$**, and it sets *when*. If the
employer discounts faster than the reserve grows, deferring is cheap and the plan
back-loads; if slower, early funding compounds and it front-loads.

A third comparison, $\mu$ against $G$, does not appear in this ratio because it acts
through the dynamics rather than the discounting: the funding ratio drifts at
$\mu - G$, so its sign decides whether $F$ floats up away from the guarantee or sinks
toward it.

**Groups B and C exist to bracket the sign of each of these.** That is the whole
design: the battery is organised around the comparisons that actually govern
behaviour, not around a list of parameters.

---

## Group A — analytic anchors

A validation battery needs cases where the answer is known without running anything.
These carry a PASS/FAIL, and they are the only entries that do.

| scenario | prediction | measured |
|---|---|---|
| `LAMBDA = 0` | employer alone funds the legal minimum → the band floor | avg **2.0%**, styRR 0.507 — PASS |
| `LAMBDA = 1` | employee alone funds to capacity → the band ceiling | avg **15.0%**, styRR 0.981 — PASS |
| `GAMMA → 0` | no funding capacity → RR collapses to the first pillar | styRR **0.471** ≈ `RR_LEGAL` = 0.43 — PASS |
| `SIGMA_R = SIGMA_L = 0` | deterministic; isolates option value | styRR 0.849 vs 0.875, schedule unchanged |
| `hazard = None` | reduces exactly to the churn-free oracle | verified **bit-identical** to `dp.solve` |

The first three are exact and parameter-free: they do not depend on any calibration
choice, so they remain valid tests whatever else changes. The `σ = 0` case is not a
pass/fail but it is informative — removing volatility *lowers* the replacement rate
slightly, because $\max(R, L)$ is a call on the reserve and loses its option value.

---

## Group B — the drift gap $\mu - G$

This brackets the sign of the funding-ratio drift, which is the guarantee's
moneyness.

| scenario | styRR | avg | early → late |
|---|---|---|---|
| $\mu < G$, underwater (1% / 3%) | 0.765 | 7.3% | 2.0 → 15.0 (defers hard) |
| $\mu = G$, at the money (committed) | 0.875 | 10.0% | 4.4 → 14.9 |
| $\mu > G$, floating (5% / 1.75%) | **1.109** | 12.2% | **14.4 → 12.5 (front-loads)** |

Monotone in the gap, and it flips the schedule: when the reserve out-grows the
guarantee, early money compounds and the plan front-loads; underwater, it waits.

This group also carries the most direct institutional content, because the Belgian
WAP minimum guaranteed return has moved across statutory regimes — the values above
span plausible vintages. **Confirm the current statutory figures before citing
them; they are not established here.**

### The gap alone is not a sufficient statistic

It would be tidy if only $\mu - G$ mattered. It does not. Holding $\mu - G = 0$ and
sliding both levels together:

| $(\mu, G)$ | styRR | early → late |
|---|---|---|
| (0.01, 0.01) | 0.681 | 2.0 → 15.0 |
| (0.03, 0.03) | 0.874 | 4.4 → 14.9 |
| (0.05, 0.05) | 1.103 | 14.8 → 7.4 |

Same gap throughout, and the schedule still flips from deferred to front-loaded.
The level of $\mu$ matters independently, because it compounds the reserve in
absolute terms over 45 years regardless of where the guarantee sits.

---

## Group C — the employer discount $\delta_f$

| scenario | styRR | avg | early → late | shape |
|---|---|---|---|---|
| $\delta_f < \mu$ (0.02) | 0.615 | 6.6% | **10.2 → 2.0** | front-loaded |
| $\delta_f = \mu$ (0.03) | 0.707 | 7.5% | 7.9 → 6.8 | flat |
| $\delta_f > \mu$ (0.05, committed) | 0.875 | 10.0% | **4.4 → 14.9** | back-loaded |

**The sign of $(\delta_f - \mu)$ selects front-load versus back-load.** This is the
cleanest structural result in the battery, and it matters beyond tidiness: it means
the contribution timing in the headline results is driven by the discount wedge, not
by workforce churn. Any claim that churn shapes the schedule has to contend with
this.

---

## Two things the battery ruled *out*

Negative results are worth recording, because both of these look like natural
regimes and neither survives contact with the model.

### $\delta_e$ is exactly redundant with $\lambda$

$\delta_e$ enters the objective **only** as $e^{-\delta_e T}$, a constant multiplying
the employee leg. The argmax is invariant to positive rescaling of the whole
objective, so any $(\lambda, \delta_e')$ has an exactly equivalent
$(\lambda'', \delta_e)$:

$$\lambda'' = \frac{A}{A + B\,e^{-\delta_e T}},
\qquad A = \lambda e^{-\delta_e' T},\quad B = 1 - \lambda.$$

Tested directly: $(\lambda = 0.5,\ \delta_e = 0.02)$ against the predicted equivalent
$(\lambda = 0.61064,\ \delta_e = 0.03)$ gives $\max|\Delta \text{policy}| = 5.9\times10^{-11}$
and an identical grid-mean $a^\star$.

So **$\delta_e$ has no relationship to $\mu$, $G$ or $\delta_f$** — it never interacts
with any of them. It is not an independent degree of freedom, and a "$\delta_e$ panel"
would be a $\lambda$ panel with the wrong label. This is why the `DISC_EMP = 0.02`
scenario reproduced `LAMBDA = 1` exactly, and why it is now labelled as a $\lambda$
reparameterisation in the code.

### $G$ versus $\delta_f$ is not a regime boundary

Their ordering looks like it should matter — a guarantee growing faster than the
employer discounts it ought to behave differently. It does not. Sweeping both with
$\mu$ fixed at 3%:

| $(G, \delta_f)$ | ordering | styRR | early → late |
|---|---|---|---|
| (0.0175, 0.02) | $G < \delta_f$ | 0.612 | 10.4 → 2.2 |
| (0.0300, 0.02) | $G > \delta_f$ | 0.615 | 10.2 → 2.0 |
| (0.0450, 0.02) | $G > \delta_f$ | 0.624 | 6.8 → 2.0 |
| (0.0175, 0.05) | $G < \delta_f$ | 0.850 | 4.5 → 14.9 |
| (0.0300, 0.05) | $G < \delta_f$ | 0.875 | 4.4 → 14.9 |
| (0.0450, 0.05) | $G < \delta_f$ | 0.984 | 4.8 → 14.7 |

The shape tracks $\delta_f$ alone: front-loaded at 0.02 for *every* $G$, back-loaded
at 0.05 for *every* $G$. The ordering tags do not line up with any flip. What $G$
changes is the **level** — stayer RR rises 0.850 → 0.984 at fixed $\delta_f$, because
a higher guarantee raises the floor payout.

So the honest presentation is a $(G \times \delta_f)$ grid separating level from
shape, which is what `scenario_suite.py rates` draws, rather than a regime panel
implying a boundary that is not there.

---

## Group D — specification toggles

| scenario | styRR | avg | reading |
|---|---|---|---|
| `SATIATE = True` | **0.708** | 6.5% | lands essentially on the 0.70 target |
| `ETA = 1` (log) | 0.921 | 11.7% | funds *more* than η=2 |
| `ETA = 2` (committed) | 0.875 | 10.0% | — |
| `ETA = 5` | 0.805 | 8.4% | monotone decreasing in η |
| `BETA = 0` | 0.875 | 10.0% | identical to the committed `BETA = 0.01` |

`SATIATE` answers a live question. With it off, the utility keeps rewarding
replacement without bound, so a plan that pushes a stayer to 0.875 against a 0.70
adequacy target is behaving correctly given the objective — there is simply no notion
of "enough". Turning satiation on restores it.

`ETA = 1` is included because log utility is the canonical criterion for
multiplicative wealth dynamics — it maximises the time-average growth rate rather
than the ensemble average. It funds *more* than η=2, which is the right direction:
marginal utility $1/x$ decays more slowly than $1/x^2$, so overshoot keeps paying.

`BETA = 0` is a control, and it currently shows the smoothing dial is inert: β is
measured relative to the local $Q$-spread, and 0.01 is about 1% of it. A visible
effect needs roughly 0.1–0.3.

---

## What the battery caught on its first run

`common.restore()` reset only the entries in `PARAMS`, which omits `SIGMA_L`,
`SATIATE`, `BETA`, `DISC`, `T`, `W` and `S0`. So `SATIATE = True` leaked into every
scenario that ran after it, and the whole of Group D collapsed to the same value.

It was visible precisely because the battery contains self-consistency checks: the
`MU = G` and `DISC_ER = 0.05` scenarios *are* the committed values, so they must
reproduce the baseline exactly, and they did not. A sweep would have produced eleven
plausible-looking numbers and no signal. That is the argument for known-answer cases
in one line.

---

## The one-line summary

Two rate comparisons govern the model — $(\mu - \delta_e)$ decides whether funding is
worth anything, $(\delta_f - \mu)$ decides whether it happens early or late, and
$(\mu - G)$ decides how exposed the guarantee is; $\delta_e$ is not independent of
$\lambda$, $G$ moves levels rather than shapes, and the cases with known answers are
there to catch the bugs that plausible-looking numbers hide.
