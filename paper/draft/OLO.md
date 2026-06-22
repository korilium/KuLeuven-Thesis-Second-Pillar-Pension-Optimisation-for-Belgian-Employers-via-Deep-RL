# Calibrating the Hull-White Model to the Belgian OLO Curve
 
*First draft — methodology section*
 
## 0. Why Hull-White, and what "calibration" means here
 
The Vasicek model is convenient — it is fully analytic and easy to estimate — but it has one fatal flaw for our purposes: it does **not** reprice the bonds we actually observe today. Its constant long-run mean θ produces a single, rigid model-implied yield curve that will generally disagree with the live OLO curve. For a simulation engine that has to value pension reserves against the market, that mismatch is unacceptable: the starting curve must be *exactly* the curve the market is quoting.
 
Hull-White fixes this by promoting the constant long-run mean to a deterministic function of time, θ(t). The short rate follows
 
$$ dr(t) = \kappa\big(\theta(t) - r(t)\big)\,dt + \sigma\,dW(t), $$
 
which is just Vasicek with a time-varying target. The extra freedom in θ(t) is spent on one thing only: forcing the model's initial term structure to coincide with the observed curve. Everything else — the mean-reversion speed κ and the volatility σ — is inherited unchanged from Vasicek.
 
**The intuition (ELI5).** Vasicek says "interest rates are pulled toward one fixed level θ." That one level can't possibly match the whole shape of today's curve. Hull-White says "interest rates are pulled toward a level that *changes over time*, θ(t), and we choose that schedule precisely so the model agrees with today's market prices." We keep Vasicek's idea of *how fast* rates revert (κ) and *how jumpy* they are (σ); we only replace *where* they revert to.
 
So calibration splits cleanly into two independent jobs:
 
| Parameter | Governs | Estimated from | Method |
|-----------|---------|----------------|--------|
| κ, σ | the *dynamics* (speed of reversion, volatility) | historical 10Y OLO time series | Vasicek OLS |
| θ(t) | the *level* (today's whole curve) | the current cross-section of OLO yields | NSS fit + no-arbitrage formula |
 
The first job is a real-world, econometric estimation; the second is a risk-neutral, arbitrage-free fit. This is the standard Hull-White calibration philosophy, and the code follows it in three concrete steps plus a validation pass.
 
---
 
## 1. Step 1 — Estimate κ and σ from history (Vasicek OLS)
 
The mean-reversion speed and volatility are structural features of how Belgian rates *move*, so we estimate them from the long monthly history of the 10Y OLO (2000–present), reusing the Vasicek calibration verbatim.
 
Discretising the Vasicek SDE with an Euler step over Δt = 1/12 (monthly):
 
$$ r(t+\Delta t) - r(t) = \kappa\big(\theta - r(t)\big)\Delta t + \sigma\sqrt{\Delta t}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,1). $$
 
Collecting terms gives a linear regression of the rate change Δr on the current level r(t):
 
$$ \Delta r = a + b\,r(t) + \varepsilon', \qquad a = \kappa\theta\,\Delta t, \quad b = -\kappa\,\Delta t. $$
 
Ordinary least squares on the historical series yields â and b̂, from which we recover
 
$$ \kappa = -\frac{b}{\Delta t}, \qquad \theta = -\frac{a}{b}, \qquad \sigma = \frac{\operatorname{std}(\varepsilon')}{\sqrt{\Delta t}}. $$
 
For Hull-White we **keep only κ and σ**. The Vasicek θ is discarded — it is exactly the quantity Hull-White is about to replace with θ(t). The code also reports diagnostics (t-statistic on b, R², implied half-life log 2 / κ) that justify treating κ as a meaningful reversion speed rather than statistical noise.
 
This is the design decision recorded earlier in the project: Hull-White reuses Vasicek's κ and σ, which keeps the two models structurally continuous and the calibration auditable, while gaining exact fit to the current curve.
 
---
 
## 2. Step 2 — Fit today's curve with Nelson-Siegel-Svensson
 
Hull-White needs the *current* term structure as an input, and specifically it needs the **instantaneous forward curve** f(0,T) and its slope ∂f/∂T. We obtain these by fitting an NSS curve to the most recent cross-section of OLO yields (all maturities on the last available date).
 
The NSS yield function is
 
$$ Y(0,T) = \beta_0 + \beta_1\,\varphi_1(T) + \beta_2\,\varphi_2(T) + \beta_3\,\varphi_3(T), $$
 
with the standard factor loadings
 
$$ \varphi_1(T) = \frac{1 - e^{-T/\tau_1}}{T/\tau_1}, \quad
   \varphi_2(T) = \varphi_1(T) - e^{-T/\tau_1}, \quad
   \varphi_3(T) = \frac{1 - e^{-T/\tau_2}}{T/\tau_2} - e^{-T/\tau_2}. $$
 
The six parameters [β₀, β₁, β₂, β₃, τ₁, τ₂] are fitted by minimising a maturity-weighted squared error (short maturities up-weighted by 1/max(T, 0.5), since the short end drives the Hull-White short-rate consistency). The optimiser is a global **differential evolution** search followed by a **Nelder-Mead** polish — the global stage is needed because the τ₁/τ₂ landscape is non-convex and a pure local solver lands in spurious minima.
 
**Why NSS rather than a spline.** NSS gives us two things a spline cannot:
 
1. **Analytic forward rates.** The instantaneous forward is the exact derivative of the fitted curve,
   $$ f(0,T) = \beta_0 + \beta_1 e^{-T/\tau_1} + \beta_2\frac{T}{\tau_1}e^{-T/\tau_1} + \beta_3\frac{T}{\tau_2}e^{-T/\tau_2}, $$
   with no finite-difference noise.
2. **Analytic slope ∂f/∂T**, available in closed form and fed directly into θ(t) in the next step.
A cubic spline through the published OLO points oscillates badly on the flat long end of the Belgian curve, and that oscillation propagates into ∂f/∂T and then into θ(t). NSS is smooth by construction, extrapolates sensibly toward β₀ beyond 30Y, and matches the parametric family the NBB itself uses. The fit is validated at roughly 1.25 bps RMSE.
 
From the fitted curve we also form the discount factors
 
$$ P(0,T) = e^{-Y(0,T)\,T}, $$
 
which are the model's initial zero-coupon bond prices.
 
---
 
## 3. Step 3 — Derive θ(t) from the no-arbitrage condition
 
This is the step that *is* the Hull-White calibration. Given κ and σ (Step 1) and the forward curve f(0,T) with its slope (Step 2), the time-varying target θ(t) is fixed by the requirement that the model reprice the initial curve exactly. In the κ(θ(t) − r) parametrisation used throughout this project, that target is
 
$$ \theta(t) = f(0,t) + \frac{1}{\kappa}\frac{\partial f(0,t)}{\partial t} + \frac{\sigma^2}{2\kappa^2}\Big(1 - e^{-2\kappa t}\Big). $$
 
Reading the three terms:
 
- **f(0,t)** — the forward rate is the "centre of gravity" the curve says rates should drift toward at time t.
- **(1/κ)·∂f/∂t** — a slope correction: where the forward curve is rising, the target is pushed up so reversion can keep pace with it.
- **σ²/(2κ²)·(1 − e^{−2κt})** — a convexity (volatility) correction. Because rates are random, Jensen's inequality means a naive target would systematically misprice long bonds; this term compensates exactly.
**The intuition (ELI5).** We already know how hard and how fast rates get pulled (κ), and how noisy they are (σ). The forward curve tells us where the market thinks rates are headed. θ(t) is reverse-engineered so that, pulling at speed κ with noise σ toward θ(t), the model's average path and bond prices land precisely on today's curve. The first term is "where the curve points," the second nudges for the curve's slope, the third pays back the bias that randomness would otherwise introduce.
 
> **Notation flag.** The same θ can be written in two equivalent Hull-White parametrisations. The drift `κ(θ(t) − r)` used here gives the θ(t) above. The alternative drift `(θ̃(t) − κr)` gives `θ̃(t) = ∂f/∂t + κf + σ²/(2κ)(1−e^{−2κt})`, with `θ̃ = κθ`. The implemented `computeTheta` returns the first (κ-divided) form, consistent with reusing Vasicek's `κ(θ − r)` drift. One of the docstrings in the NSS block currently writes the `θ̃` form — worth reconciling so the thesis text and the comments agree.
 
---
 
## 4. Step 4 — Quick sanity checks
 
The calibrated objects (f, P, θ) are first passed through four lightweight sanity checks. These confirm nothing is grossly malformed before the more demanding test in Step 5:
 
1. **Forward above yield.** For an upward-sloping curve, f(0,T) must sit above Y(0,T) everywhere (the forward leads the average). Checked as f − Y ≥ 0.
2. **Admissible discount factors.** P(0,T) must start at ≤ 1, be strictly decreasing, and stay positive — the minimal conditions for a legitimate discount curve.
3. **Positive target.** θ(t) > 0 throughout, confirming the calibrated reversion target stays economically sensible.
4. **Curve recovery.** Recompute yields from the bond prices, Y_rec(T) = −log P(0,T) / T, and compare to the fitted Y(0,T).
These are reported alongside θ(t), f(0,t) and P(0,T) read off at the 1Y/5Y/10Y/30Y nodes, and plotted in a four-panel diagnostic.
 
A caveat on check 4: because P(0,T) is constructed *directly* as exp(−Y·T), recomputing −log P(0,T)/T merely inverts that definition and passes to floating-point precision **regardless of whether θ(t) was calibrated correctly**. It verifies the discount-factor construction is internally consistent, but it does not exercise the Hull-White calibration at all. It is a sanity check, not a validation — which is precisely why Step 5 exists.
 
---
 
## 5. Step 5 — Analytic no-arbitrage validation
 
The check that actually exercises the calibration asks: **do θ(t), κ, σ, fed through the Hull-White bond-pricing formula, reproduce the observed curve?** If `computeTheta` inverts the term structure correctly, the model-implied bond prices must coincide with the NSS market prices; any material gap is a calibration error, not noise.
 
Since the integrated short rate ∫₀ᵀ r(s) ds is Gaussian, the time-0 bond price is closed-form:
 
$$ P^{\mathrm{HW}}(0,T) = \exp\!\Big(-M(T) + \tfrac{1}{2}V(T)\Big), $$
 
$$ M(T) = r(0)\,B(0,T) + \int_0^T \theta(u)\big(1 - e^{-\kappa(T-u)}\big)\,du,
\qquad
V(T) = \frac{\sigma^2}{\kappa^2}\Big[\,T - 2B(0,T) + \tfrac{1-e^{-2\kappa T}}{2\kappa}\,\Big], $$
 
with B(0,T) = (1 − e^{−κT})/κ. The θ-integral splits into two cumulative integrals computed once over the grid, so the whole curve prices in O(N):
 
$$ \int_0^T \theta(u)\big(1-e^{-\kappa(T-u)}\big)\,du = I_1(T) - e^{-\kappa T} I_2(T),
\quad I_1=\textstyle\int_0^T \theta\,du,\ \ I_2=\textstyle\int_0^T \theta\,e^{\kappa u}\,du. $$
 
The model prices are then compared to P^NSS(0,T) = exp(−Y(0,T)·T) on a yield basis, with the residual reported in basis points.
 
**Two details that make or break the check.** First, the short rate entering the formula is the **instantaneous rate implied by today's curve**, r(0) = f(0,0) = β₀ + β₁ — *not* the historical 10Y level used to estimate κ and σ in Step 1. Those parameters answer different questions (how rates move, versus where the curve is today), and conflating them throws the repricing off by roughly two hundred basis points. That this mistake produces a ~200 bps error rather than a silent pass is exactly what makes the check worth running. Second, the θ-integrals must be anchored at a true t = 0; the NSS grid starts at T = 0.01, and integrating from there leaves a stub that scales linearly with the start point (~2 bps at t₀ = 0.01). Prepending the exact limits f(0,0) = β₀ + β₁ and θ(0) = f(0,0) + f′(0,0)/κ (the volatility term vanishes at t = 0) removes it.
 
**Result.** On a synthetic Belgian-style curve the calibrated model reprices the entire 0–30Y curve to a maximum error of **0.0017 bps** (≈ 0.001 bps at 1Y, 0.0002 bps at 30Y) — essentially machine precision, confirming `computeTheta` inverts the term structure exactly. This validates the **internal consistency** of the Hull-White calibration, and is independent of the **market-fit quality** of Step 2 (the NSS RMSE of ≈ 1.25 bps against observed OLO points). The two are distinct claims and should be reported separately: the first asks "is the calibration mathematically correct?", the second "how well does the parametric curve match the market?" On the live OLO curve the no-arbitrage residual should remain at the same machine-precision level, since the inversion is exact up to numerical integration — making this the check that certifies the engine is arbitrage-free against the market before it is handed to the PPO training loop.
 
---
 
## 6. Implementation notes to tighten before submission
 
A few honest caveats, since these affect how strongly the section can claim what it claims:
 
- **One leftover from the cubic-spline era.** The final lines of the runner (`spline = curve["spline"]`) will raise a `KeyError` — the NSS `bootstrapForwardCurve` returns `params`, not a `spline` object — and Panel 1 of `checkHullWhiteCalibration` is still titled "Spline fit Y(0,T)" though the curve is now NSS. Drop the `spline` lines (or expose the NSS params under that key) and relabel the panel "NSS fit Y(0,T)".
- **Plot output path.** `plotForwardCurve` calls `os.makedirs("tests", ...)` but then saves to `olo/tests/...`. If `olo/tests/` does not already exist this will throw the same `FileNotFoundError` pattern seen earlier in the project — make the `makedirs` target match the `savefig` path.
- **Negative rates.** The current θ(t) form admits negative short rates (as the Belgian curve has historically gone negative). If that becomes a problem for the reserve simulation, the noted mitigation is the **shifted Hull-White**, r(t) = x(t) + φ(t), where x(t) is a zero-mean OU process and φ(t) absorbs the curve fit. That is a drop-in replacement that preserves closed-form bond prices.