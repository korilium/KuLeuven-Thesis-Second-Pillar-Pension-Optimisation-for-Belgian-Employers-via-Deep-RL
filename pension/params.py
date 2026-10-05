"""
params.py -- every parameter of the model, as ONE immutable object.

    from pension.params import DEFAULT
    p = DEFAULT.replace(ETA=3.0, RATE_MODEL="hull_white")
    dp.solve(..., p=p); dp.simulate(..., p=p)

A Params is passed explicitly to every model function, so a run is fully
described by its Params: nothing is read from mutable module state, a sweep can
never leak into the next run, and two economies can live side by side (one per
RL environment instance, one per worker process). A Params is hashable, so it
can key a cache directly.

Field names are the historical global names, so objective.py (which reads
p.ETA, p.LAMBDA, ...) and the thesis notation carry over unchanged.

The objective (default EMPLOYER_NUMERAIRE = "retirement"), per unit of final salary,
in retirement-date money under the real-world measure P:

    J = LAMBDA * E[ target * ANNUITY * u(RR_tot / target) ]
      - (1 - LAMBDA) * E[ sum_{t<tau} a_t GAMMA (1+W)^-(T-t) A(t,T) + shortfall((L - R_T)^+ / S_T) ]

A(t,T): the premium's expected accrual to T at the short rate plus FINANCING_SPREAD
(pension/numeraire.py). "discounted" restores the previous specification
(premiums at DISC_ER, terminal legs at DISC_EMP / DISC_ER).
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Params:
    # --- career and economy -------------------------------------------------
    T: int = 45                 # career length in years
    G: float = 0.03             # WAP guarantee rate
    MU: float = 0.03            # credited tariff (Branch 21 return)
    W: float = 0.025            # salary growth
    S0: float = 1.0             # starting salary (the model is scale-free in it)

    # --- discounting ----------------------------------------------------------
    DISC_EMP: float = 0.03      # EMPLOYEE discount: values future retirement income at the risk-free/OLO rate
    DISC_ER: float = 0.05       # EMPLOYER discount: firm cost of capital (contributions + shortfall)
    DISC: float = 0.05          # single numeraire of the tabular rung (= DISC_ER), legacy alias.
                                # DISC, SIGMA, N_EVAL are TABULAR DEFAULTS: read once, at import, into
                                # economy's module globals; changing them on a Params does nothing

    # --- shocks ---------------------------------------------------------------
    SIGMA_R: float = 0.05       # asset (reserve) shock
    SIGMA_L: float = 0.02       # guarantee (liability) shock -- constant-rate model only
    SIGMA: float = 0.05         # the tabular rung's ONE shock, on the reserve (= SIGMA_R)

    # --- contribution capacity and preferences (rung 2) -----------------------
    GAMMA: float = 0.15         # contribution capacity: c = a * GAMMA * S, a in [0, 1]
    LAMBDA: float = 0.5         # EMPLOYEE weight in the joint objective
    ETA: float = 2.0            # CRRA curvature over the replacement rate (eta = 1 is log)
    ANNUITY: float = 15.0       # actuarial annuity factor: capital -> annual pension
                                # (Belgian life expectancy at 65 ~20y, ~2% technical rate,
                                #  mortality-adjusted). RR is ANNUAL: pension / final salary.
    RR_LEGAL: float = 0.43      # 1st-pillar (legal) gross replacement, Belgian private-sector
                                # average earner (OECD PaaG); the 2nd pillar sits ON TOP.
    RR_TARGET: float = 0.70     # total-adequacy target across all pillars (OECD/EU ~70%);
                                # the employee's lifecycle utility is judged against this.
    SATIATE: bool = False       # if True, cap RR at the target in the utility (no reward for overshoot)
    OBJECTIVE: str = "baseline" # the value function: a name in pension.objective.OBJECTIVES

    # --- policy extraction ----------------------------------------------------
    BETA: float = 0.01          # policy-EXTRACTION temperature, RELATIVE to the local Q-spread:
                                # 0 = hard argmax (the true optimum); > 0 = soft readout that
                                # blends actions within BETA of the state's value range. Relative,
                                # so invariant to any affine rescale of the reward.
                                # Ladder: 0.01 barely smooths, 0.1 visible, 0.3 strong.
                                # Extraction only: V is identical at every BETA.

    # --- interest-rate regime -------------------------------------------------
    # "constant" is the original model (G, MU, SIGMA_R, SIGMA_L as above).
    # "hull_white" / "vasicek" drive G and MU path-wise from ONE simulated 10Y OLO:
    #   G_t  = the statutory WAP filter (pension/rates/wap.py), applied HORIZONTALLY;
    #   mu_t = the insurer's book yield, a rolling mean of the same OLO over
    #          BOOK_DURATION years plus BOOK_SPREAD.
    RATE_MODEL: str = "constant"   # "constant" | "hull_white" | "vasicek" | "vasicek_short" | "hull_white_p"
    RATE_DT: float = 1 / 12        # monthly rate step -- matches the OLO calibration
    RATE_SEED: int = 2026          # seed of the rate scenarios (separate from churn/asset noise)
    BOOK_DURATION: int = 8         # years averaged into the book yield (Branch 21 portfolio duration)
    BOOK_SPREAD: float = 0.0       # book yield over the 10Y OLO (credit/illiquidity pickup, net of costs)
    SIGMA_R_RATES: float = 0.05    # excess-return noise on top of the book yield (= SIGMA_R, so both
                                   # regimes carry the same asset risk); 0 = pure book-yield crediting
    LONG_RATE_P: Optional[float] = None   # long-run REAL-WORLD SHORT rate, the one anchor of the P-models
                                   # (same meaning in every model, REVIEW M1):
                                   # vasicek: the 10Y reverts to LONG_RATE_P + spread (None: OLS theta);
                                   # hull_white_p: sets its market price of risk (None: phi = 0);
                                   # vasicek_short: theta_P (None: the OLS estimate)
    SPREAD_10Y_SHORT: Optional[float] = None   # vasicek: the short rate is r = y10 - this spread;
                                   # None: the historical mean (10Y - 1Y) of the OLO data
    DRIFT_CORRECTION: bool = True  # True: shocked growth is e^{m - s^2/2 + s z}, so the MEAN growth is
                                   # e^m (reserve: m = MU or mu_t; vertical liability: m = G); False
                                   # (legacy): e^{m + s z}, mean e^{m + s^2/2} (REVIEW M6)
    CE_MOMENTS: str = "conditional"  # the DP's certainty-equivalent shocks in rate mode: "conditional":
                                   # SIGMA_L = 0, SIGMA_R = SIGMA_R_RATES (G_t, mu_t are known at the start
                                   # of year t, so their dispersion is no one-year shock); "levels"
                                   # (legacy): the pooled dispersion of the levels (REVIEW M10)
    YEAR_START: str = "january"    # "january": model years are calendar years from the first 1 January
                                   # after the last OLO observation, so each year holds the WAP rate
                                   # fixed on its 1 January; "t0" (legacy): years start at the last
                                   # observation (REVIEW C1)
    RATE_CE_PATHS: int = 5000      # scenarios behind the certainty-equivalent moments of solve()

    # --- the employer's numeraire ----------------------------------------------
    # "retirement" (default): both legs in retirement-date money under P -- terminal
    # legs undiscounted, each premium accrued to T at the short rate plus
    # FINANCING_SPREAD, A(t,T) = E^P_t[exp(int_t^T (r_u + s) du)] (pension/numeraire.py).
    # "discounted": the previous specification -- premiums discounted at DISC_ER,
    # terminal legs at DISC_EMP / DISC_ER -- reproduced bit for bit.
    EMPLOYER_NUMERAIRE: str = "retirement"   # "retirement" | "discounted"
    SHORT_RATE: float = 0.03       # short rate of the constant-rate model (= MU = G)
    FINANCING_SPREAD: float = 0.0  # s: the employer's financing spread over the short rate

    # --- numerics ---------------------------------------------------------------
    N_EVAL: int = 2000          # default number of evaluation paths (SAA batch) of the tabular rung

    def replace(self, **changes) -> "Params":
        """A copy with some fields changed; unknown names are an error."""
        return dataclasses.replace(self, **changes)


DEFAULT = Params()
