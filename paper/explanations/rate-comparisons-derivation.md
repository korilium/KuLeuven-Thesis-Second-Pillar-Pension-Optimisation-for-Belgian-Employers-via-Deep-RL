# Rate comparisons in the second-pillar funding model: a derivation from the dynamics

This note derives, from the state dynamics and the objective, the three rate comparisons that govern the optimal funding policy:

- $(\mu - G)$ determines guarantee exposure;
- $(\delta_f - \mu)$, or $(\delta_f - G)$ below the floor, determines the timing of funding;
- $(\mu - \delta_e)$, jointly with $\lambda$, determines whether funding is worthwhile at all.

It also establishes the non-identifiability of $\delta_e$ separately from $\lambda$, and qualifies the claim that $G$ shifts levels rather than shapes.

---

## 1. Setting and assumptions

The derivation is carried out in the **deterministic limit** ($\sigma = 0$), for a member who **stays until retirement** $T$. The stochastic case and the leaver branch are discussed in Section 7.

**Notation.**

| Symbol | Meaning |
|---|---|
| $R_t$ | Accumulated reserve (assets) |
| $L_t$ | Guaranteed reserve (WAP liability) |
| $S_t$ | Salary |
| $F = R/L$ | Funding ratio |
| $\rho = S/L$ | Salary-to-liability ratio |
| $a_t \in [0,1]$ | Funding decision; contribution $c_t = a_t \Gamma S_t$ |
| $\mu$ | Credited (Branch 21) return |
| $G$ | WAP guaranteed rate |
| $W$ | Salary growth |
| $\delta_e,\ \delta_f$ | Employee and employer discount rates |
| $\lambda$ | Employee weight in the joint objective |
| $\ddot a$ | Annuity factor |
| $\text{RR}^\*$ | Target replacement ratio |

**Dynamics.** Contributions are credited to both the reserve and the guaranteed reserve:

$$
\dot R = \mu R + c,\qquad \dot L = G L + c,\qquad \dot S = W S .
$$

**Terminal pension.** The member receives the larger of reserve and guarantee, $P_T = \max(R_T, L_T)$, with total replacement ratio

$$
\text{RR}_T = \text{RR}_{\text{LEGAL}} + \frac{P_T}{\ddot a\, S_T},\qquad x_T = \frac{\text{RR}_T}{\text{RR}^\*}.
$$

**Objective.** Both legs are expressed per unit of final salary. Contributions are assumed to enter the employer leg as a cost discounted at $\delta_f$:

$$
J=\lambda\, e^{-\delta_e T}\,\text{RR}^\*\ddot a\;u(x_T)\;-\;(1-\lambda)\left[\int_0^T e^{-\delta_f t}\,\frac{c_t}{S_T}\,dt\;+\;e^{-\delta_f T}\,\frac{(L_T-R_T)^+}{S_T}\right],
$$

with the normalized CRRA utility $u(x) = \dfrac{x^{1-\eta}-1}{1-\eta}$, so that $u(1)=0$ and $u'(1)=1$.

> If the implementation differs from this specification, in particular regarding how contributions are costed, the steps below carry over but the final expressions must be adapted.

---

## 2. State dynamics: the origin of $(\mu - G)$

Differentiating $F = R/L$ and $\rho = S/L$:

$$
\dot F=(\mu-G)\,F+a\Gamma\rho\,(1-F),
\qquad
\dot\rho=(W-G)\,\rho-a\Gamma\rho^2 .
$$

Two effects follow.

1. **Drift.** Absent contributions, $F_t = F_0\,e^{(\mu-G)t}$. If $\mu < G$ the reserve erodes relative to the guarantee; if $\mu > G$ it outgrows it.
2. **Pull towards par.** A contribution enters $R$ and $L$ in equal amounts, so it carries a funding ratio of exactly one and pulls $F$ towards 1 at rate $a\Gamma\rho$. Contributions cannot lift a plan *above* full funding; only the drift can.

Hence $(\mu - G)$ determines which terminal regime is reached, and how deep into it:

- **Regime A (overfunded):** $F_T > 1$, so $P_T = R_T$;
- **Regime B (underfunded):** $F_T < 1$, so $P_T = L_T$ and the employer settles the shortfall $L_T - R_T$.

This is the precise sense in which $(\mu - G)$ measures the **exposure of the guarantee**.

---

## 3. Marginal value of a contribution

Perturb the contribution at time $t$ by $dc$, and write $m = T - t$ for the remaining horizon. Then

$$
dR_T = e^{\mu m}\,dc,\qquad dL_T = e^{G m}\,dc .
$$

**Effect on the pension.** It depends on the terminal regime:

$$
dP_T=\begin{cases} e^{\mu m}\,dc & \text{regime A,}\\[2pt] e^{G m}\,dc & \text{regime B.}\end{cases}
$$

In regime B the guarantee accrues on the contribution itself, so the member gains even though the pension is determined by the guarantee.

**Employee gain.** By the chain rule, $\partial u(x_T)/\partial P_T = u'(x_T)/(\text{RR}^\*\ddot a\,S_T)$. The multiplier $\text{RR}^\*\ddot a$ therefore cancels, and the gain is

$$
\lambda\, e^{-\delta_e T}\,u'(x_T)\,\frac{dP_T}{S_T}.
$$

The employee leg is thus money-metric at the margin, in units of final salary.

**Employer cost.**

$$
\frac{1-\lambda}{S_T}\Big[e^{-\delta_f t}\,dc\;+\;\mathbb 1_{B}\,e^{-\delta_f T}\big(e^{Gm}-e^{\mu m}\big)\,dc\Big].
$$

The second term is the change in the terminal shortfall, which is present only in regime B.

---

## 4. Regime A (overfunded at $T$)

The ratio of marginal value to marginal cost is

$$
\boxed{\;\frac{\lambda}{1-\lambda}\;u'(x_T)\;e^{(\mu-\delta_e)T}\;e^{(\delta_f-\mu)\,t}\;}
$$

**Timing.** The ratio is increasing in $t$ if and only if $\delta_f > \mu$:

- $\delta_f > \mu$: deferral is cheaper than the return earned by early funding, so the policy **back-loads**;
- $\delta_f < \mu$: the policy **front-loads**;
- $\delta_f = \mu$: timing is **neutral**. This is the frictionless benchmark of pension-funding irrelevance.

**Level.** Whether funding is worthwhile at all is governed by

$$
\ln\frac{\lambda}{1-\lambda}\;+\;(\mu-\delta_e)\,T\;+\;\ln u'(x_T).
$$

The comparison $(\mu - \delta_e)$ therefore operates only jointly with $\lambda$; in isolation it decides nothing.

**Role of $G$.** $G$ does not appear in the timing condition. It affects only whether the plan ends in regime A, and the level of $L$. Within regime A, *$G$ shifts levels, not shapes* holds exactly.

---

## 5. Non-identifiability of $\delta_e$ and $\lambda$

$\delta_e$ enters $J$ only through the factor $\lambda e^{-\delta_e T}$ on the employee leg. For a common and fixed $T$, $J$ is determined, up to a positive multiplicative constant, by the single quantity

$$
\theta=\frac{\lambda\,e^{-\delta_e T}}{1-\lambda}.
$$

A positive rescaling of the objective leaves the argmax unchanged, so **only $\theta$ is identified**. Every pair $(\lambda,\delta_e)$ with the same $\theta$ yields the same optimal policy.

Given a reference pair $(\lambda,\delta_e')$ and a new discount rate $\delta_e$, the equivalent weight is

$$
\lambda''=\frac{A}{A+B\,e^{-\delta_e T}},\qquad A=\lambda\,e^{-\delta_e' T},\quad B=1-\lambda .
$$

This is the relation verified numerically in the scenario suite (to about $6\times10^{-11}$).

**Scope condition.** The equivalence requires that the employee leg is evaluated at a single common date $T$. It breaks down if leavers' entitlements are settled at exit (for example on transfer of reserves), or if retirement timing becomes a decision variable. In either case $\delta_e$ acquires an independent role.

---

## 6. Regime B (underfunded at $T$)

In regime B the member's pension grows at $G$ rather than $\mu$. The relevant quantity is the employer's cost per unit of additional pension:

$$
\frac{\text{cost}}{\text{value}}\;\propto\;e^{-\delta_f t-G m}\;+\;e^{-\delta_f T}\big(1-e^{(\mu-G)m}\big).
$$

Differentiating with respect to $t$:

$$
\frac{d}{dt}\,\frac{\text{cost}}{\text{value}}\;\propto\;(G-\delta_f)\,e^{-\delta_f t-Gm}\;+\;(\mu-G)\,e^{-\delta_f T+(\mu-G)m}.
$$

Late funding is preferred when this derivative is negative. Three consequences follow.

1. **The timing comparison becomes $(\delta_f - G)$ rather than $(\delta_f - \mu)$,** because below the floor the pension grows at the guaranteed rate.
2. **$(\mu - G)$ adds a second timing term.**
   - If $\mu < G$, each contribution *increases* the shortfall: it is guaranteed at $G$ but earns only $\mu$. This favours late funding.
   - If $\mu > G$, each contribution generates a surplus that partly finances the existing deficit. This favours early funding.
3. **$G$ does change the shape of the policy** in regime B. The statement that $G$ shifts levels rather than shapes is therefore exact only in regime A.

**Special case $\mu = G$.** The shortfall term vanishes: a contribution raises $R$ and $L$ equally and leaves the deficit unchanged. The two regimes coincide at the margin, and the timing condition reduces to the sign of $(\delta_f - \mu)$.

---

## 7. Implications and limits of the deterministic analysis

**Independence of $F$ below the floor at the baseline.** At the baseline calibration $\mu = G$, in the deterministic stayer case:

- a contribution below the floor raises the member's pension (by $e^{Gm}$ per unit) but does not reduce the employer's shortfall;
- the marginal condition is therefore **independent of $F$ for $F < 1$**.

Any dependence of the optimal policy on $F$ below the floor, such as the V-shape observed in the policy maps, must originate outside this derivation. Plausible sources are:

- **Return volatility.** With $\sigma_R > 0$ the terminal regime is uncertain, and near $F = 1$ a contribution carries option value.
- **The leaver branch.** In the paid-up treatment $F$ grows at $\mu$ after exit while $\rho$ grows at $W$, i.e. the guaranteed reserve is frozen at exit. For leavers the relevant drift is $\mu$ rather than $\mu - G$, which reintroduces dependence on $F$ through the tenure hazard.
- **Satiation.** It sets $u'$ to zero above the target.

**Proposed known-answer test.** At $\sigma = 0$ with the tenure hazard switched off, the optimal policy should not depend on $F$ below 1. The test isolates whether the V-shape stems from the stochastic or the leaver terms.

**Direction versus magnitude.** The marginal conditions determine the *tendency* of the policy. The realized schedule also depends on:

- curvature, since $u'$ declines as the member approaches the target;
- the tenure hazard;
- the entry state.

This is why a realized schedule can be front-loaded even when $\delta_f > \mu$.

---

## 8. Summary

| Comparison | Derived role | Domain of validity |
|---|---|---|
| $\mu - G$ | Drift of $F$; determines the terminal regime and the guarantee exposure | Always |
| $\delta_f - \mu$ | Timing: late if positive, early if negative, neutral if zero | Regime A (overfunded) |
| $\delta_f - G$, with correction $(\mu - G)$ | Timing, including the self-financing of contributions under the guarantee | Regime B (underfunded) |
| $\ln\frac{\lambda}{1-\lambda} + (\mu-\delta_e)T$ | Level: whether funding is worthwhile at all | Regime A; with $G$ in place of $\mu$ in regime B |
| $\theta = \lambda e^{-\delta_e T}/(1-\lambda)$ | Only identified weight; $\delta_e$ and $\lambda$ are not separately identified | Common, fixed $T$ |

### Known-answer tests implied by the derivation

1. $\delta_f = \mu$, no hazard: timing is neutral.
2. Any change in $\delta_e$ is exactly offset by $\lambda''$ (Section 5), to machine precision, for common $T$.
3. $\sigma = 0$, no hazard, $\mu = G$: the policy is independent of $F$ for $F < 1$.
4. $\mu = G$: the timing conditions of regimes A and B coincide.

### Outlook: stochastic rates

With stochastic rates, $\mu_t$ follows the insurer's book yield and $G_t$ the statutory WAP filter of the 10-year OLO. The three comparisons then become **state variables rather than parameters**. The terminal regime is determined along each path, and the relevant timing comparison switches between $(\delta_f - \mu_t)$ and $(\delta_f - G_t)$ whenever a path crosses the guarantee. The constant-rate analysis above characterizes the current model rung. In the rate-augmented rung, the same logic applies to the *joint dynamics* of the rate gaps.
