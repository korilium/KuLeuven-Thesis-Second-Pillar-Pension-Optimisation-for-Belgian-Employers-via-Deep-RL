# The Scenario Battery — What Each Regime Is, and Why It Was Chosen

`tests/dynpro/scenario_suite.py` runs a fixed set of named parameter regimes rather
than a sweep. This note explains the selection: what each group tests, why those
particular configurations, and what the model actually does in each.

A final section covers the market-plan benchmark in
`tests/dynpro/benchmark_suite.py`, which runs on the same protocol.

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

The **direction** of the tilt tracks $\delta_f$ alone: front-loaded at 0.02 for *every*
$G$, back-loaded at 0.05 for *every* $G$. The ordering tags do not line up with any
flip, so there is no regime boundary at $G = \delta_f$. And $G$ clearly moves the
**level** — stayer RR rises 0.850 → 0.984 at fixed $\delta_f$, because a higher
guarantee raises the floor payout.

But "$G$ moves levels, not shapes" is too strong, and the table above says so. Read down
the $\delta_f = 0.02$ rows: the early leg falls 10.4 → 10.2 → **6.8** as $G$ rises. A
34% change in early funding is a change of shape, not of level. It is confined to that
one corner, and the marginal analysis says why.

Below the floor the employer's cost of an extra contribution includes its effect on the
terminal shortfall, $d(L_T - R_T) = (e^{Gm} - e^{\mu m})\,dc$. At the committed
$\mu = G$ that term is exactly zero — a contribution lifts reserve and guarantee
equally and leaves the deficit untouched — so the timing condition collapses to
$\mathrm{sign}(\delta_f - \mu)$ and $G$ really is shape-neutral. Away from $\mu = G$
it does not vanish: at $G = 0.045 > \mu$, each contribution is guaranteed at a rate it
does not earn, which *enlarges* the shortfall and penalises early funding. And
$\delta_f = 0.02$ front-loads, which leaves the plan least funded at $T$ and therefore
most often below the floor. The one cell where both conditions hold is $(0.045, 0.02)$
— precisely the cell that breaks the pattern.

So the correct statement is regime-dependent: **above the floor $G$ moves levels only;
below it, $G$ enters the timing condition too**, through $(\delta_f - G)$ and through
the $(\mu - G)$ self-financing term. Our committed calibration sits at $\mu = G$, where
the two regimes agree at the margin, which is why the simpler claim held everywhere we
usually look.

The honest presentation is therefore the $(G \times \delta_f)$ grid that
`scenario_suite.py rates` draws — separating level from shape, and showing the one
corner where the separation fails — rather than a regime panel implying a boundary at
$G = \delta_f$ that is not there.

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

## The market benchmark

`benchmark_suite.py` answers the question the scenario battery cannot: **what does
optimising the schedule actually buy, measured against what employers do?** Everything
runs on the same protocol constants, which now live in `common.py` precisely so the
benchmark cannot drift onto a different grid than the results it is compared with.

Two design families are covered, both *calendar-driven* — the contribution depends on
the year, not on how well funded the plan happens to be:

| design | stayer RR | cost | joint | avg | early → late |
|---|---|---|---|---|---|
| flat 3% of salary | 0.544 | 0.1123 | −0.178 | 3.0% | 3.0 → 3.0 |
| flat 5% of salary | 0.617 | 0.1870 | −0.149 | 5.0% | 5.0 → 5.0 |
| flat 8% of salary | 0.726 | 0.2991 | −0.126 | 8.0% | 8.0 → 8.0 |
| age scale 3% +1%/10y | 0.604 | 0.1429 | −0.154 | 4.2% | 3.0 → 6.5 |
| age scale 2% +1%/5y | 0.644 | 0.1504 | −0.142 | 4.7% | 2.5 → 9.5 |
| **optimised, banded 2–15%** | 0.875 | 0.3201 | **−0.108** | 10.0% | 4.4 → 14.9 |

The optimum dominates every design in joint value, which it must by construction — the
check is that it does, because if it did not, the two evaluation paths would not be
running the same protocol.

The economically meaningful comparisons are the two readings of the same gap, both
taken along the $\lambda$-frontier: the cost the optimum needs to reach a design's
adequacy, and the adequacy it reaches on a design's budget.

| design | cost at matched adequacy | adequacy at matched cost |
|---|---|---|
| flat 3% | −25% | +0.078 |
| flat 5% | −41% | +0.134 |
| flat 8% | −43% | +0.133 |
| age scale 3% +1%/10y | −26% | +0.080 |
| age scale 2% +1%/5y | −19% | +0.052 |

### The gain splits into two parts

The age scale is not merely *better* than flat — it captures a specific part of the
gain. At 3% rising 1% per decade it reaches 0.604 for a cost of 0.1429, against flat
5%'s 0.617 for 0.1870: about a quarter cheaper for essentially the same adequacy. And
the optimum in turn reaches the age scale's adequacy for 0.1050, about another quarter
below it.

So at this one point, roughly half the available saving comes from **back-loading at
all** — which a calendar rule can do — and the other half from making the contribution
respond to the **state**, which it cannot. That decomposition is a single-point
estimate, not a general result, but it is the right shape of claim for the thesis:
the DP is not beating the market by contributing more, it is beating it by contributing
at the right time and in the right states.

### Three honest caveats

**The levels are placeholders.** 3/5/8% and the age steps were chosen to bracket
plausible Belgian practice. They need a DB2P or Assuralia figure before being
described as representative, and any age step must stay inside the WAP
non-discrimination bound. This is the same "confirm before citing" flag the statutory
$G$ carries.

**The matched figures are interpolations, and they understate the gain.** They read
the frontier at the design's cost or adequacy by linear interpolation between swept
$\lambda$ points. The frontier is concave, so a chord lies *below* it: the interpolated
optimum looks worse than it is, and every reported saving is therefore conservative.

Choosing those $\lambda$ points needs care, because the frontier is **doubly censored
by the contribution band**:

| $\lambda$ | cost | stayer RR | |
|---|---|---|---|
| 0.05 – 0.15 | 0.0750 | 0.507 | pinned at the 2% band **floor** |
| 0.20 – 0.50 | 0.081 → 0.320 | 0.532 → 0.875 | the informative range |
| 0.60 – 0.90 | 0.5602 | 0.981 | pinned at the 15% band **ceiling** |

Every market design costs between 0.11 and 0.30, so all of them land in the middle
stretch — and a uniform grid from 0.1 to 0.95 spent 7 of its 10 points on the two flat
ones, leaving four to resolve the part that matters. The default grid is concentrated
on $\lambda \in [0.15, 0.60]$ instead. Retargeting moved every figure by at most one
point of cost and 0.006 of replacement, and always in the conservative direction, which
is what the chord argument predicts. A design falling outside the swept range is
reported as such rather than silently extrapolated.

**There are no error bars.** Every number is one seed and one entry cohort. Monte
Carlo noise at 15 000 paths is visible: the flat-5% cost moves from 0.1870 to 0.1848
on doubling the paths, about 1.2%. Differences of that order should not be read as
real. (The adequacy figures are far more stable — stayer RR is identical to four
decimals across grids, because a calendar-driven design makes no use of the grid
except as an exact lookup.)

### Why the two-tier design is not here

Belgian DC plans commonly split contributions at the social-security wage ceiling: a
low rate below it, a high rate above. That design is **not scale-free**. It compares a
salary *level* against a fixed ceiling, while the whole model is homogeneous of degree
0 in $(R, L, S)$ — which is what lets a three-variable problem collapse to the reduced
state $(F, \rho)$ in the first place. Representing it faithfully would mean carrying
the ceiling as a further ratio in the state.

Little is lost by leaving it out. If salary and ceiling are indexed at the same rate,
then $S_t / C_t$ never moves and the two-tier plan degenerates exactly to a flat rate
whose level depends on where the worker sits relative to the ceiling — already covered
by the flat rows. The design only does something distinctive when career progression
outruns ceiling indexation, and that is a statement about wage dynamics rather than
about pension design.

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
