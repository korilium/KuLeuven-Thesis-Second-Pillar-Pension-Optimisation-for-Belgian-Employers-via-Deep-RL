"""
params.py -- every parameter of the model, as ONE immutable object.

    from pension.params import DEFAULT
    p = DEFAULT.replace(ETA=3.0, RATE_MODEL="hull_white")
    dp.solve(..., p=p); dp.simulate(..., p=p)

A Params is passed explicitly to every model function, so a run is fully
described by its Params: nothing is read from mutable module state, a sweep can
never leak into the next run, and two economies can live side by side (one per
RL environment instance, one per worker process). `key()` is a stable hashable
identity used to memoise scenarios and solves.

Field names are the historical global names, so objective.py (which reads
p.ETA, p.LAMBDA, ...) and the thesis notation carry over unchanged.
"""

import dataclasses
from dataclasses import dataclass


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
    DISC: float = 0.05          # single numeraire of the tabular rung (= DISC_ER), legacy alias

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
    RATE_MODEL: str = "constant"   # "constant" | "hull_white" | "vasicek"
    RATE_DT: float = 1 / 12        # monthly rate step -- matches the OLO calibration
    RATE_SEED: int = 2026          # seed of the rate scenarios (separate from churn/asset noise)
    BOOK_DURATION: int = 8         # years averaged into the book yield (Branch 21 portfolio duration)
    BOOK_SPREAD: float = 0.0       # book yield over the 10Y OLO (credit/illiquidity pickup, net of costs)
    SIGMA_R_RATES: float = 0.05    # excess-return noise on top of the book yield (= SIGMA_R, so both
                                   # regimes carry the same asset risk); 0 = pure book-yield crediting
    RATE_CE_PATHS: int = 5000      # scenarios behind the certainty-equivalent moments of solve()

    # --- numerics ---------------------------------------------------------------
    N_EVAL: int = 2000          # default number of evaluation paths (SAA batch) of the tabular rung

    def replace(self, **changes) -> "Params":
        """A copy with some fields changed; unknown names are an error."""
        return dataclasses.replace(self, **changes)

    def key(self) -> tuple:
        """Stable, hashable identity of these parameters (for memoisation)."""
        return dataclasses.astuple(self)

    def diff(self, other: "Params" = None) -> dict:
        """The fields that differ from `other` (default: DEFAULT) -- for labels and logs."""
        other = DEFAULT if other is None else other
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
                if getattr(self, f.name) != getattr(other, f.name)}


FIELDS = tuple(f.name for f in dataclasses.fields(Params))
DEFAULT = Params()
