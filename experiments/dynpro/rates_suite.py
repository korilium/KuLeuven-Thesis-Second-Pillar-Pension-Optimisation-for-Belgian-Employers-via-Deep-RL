"""Stochastic rates: the WAP guarantee G_t and the book yield mu_t driven by the OLO.

The constant-rate model holds G and MU fixed. Here both come from ONE simulated
10Y OLO path per scenario (economy.draw_rate_scenarios), Hull-White or Vasicek:
G_t is the statutory WAP filter of that path (pension/rates/wap.py), applied
HORIZONTALLY to the liability, and mu_t is the insurer's book yield on the
reserve. The switch is Params.RATE_MODEL; "constant" is the original model, so every
other suite is untouched and this one can flip back to it at will.

Checks (asserted; the suite stops on the first failure):
  switch      constant mode is unaffected by having visited a rate model
  known       a degenerate scenario (G_t = G, mu_t = MU, SIGMA_L = 0) reproduces
              the constant branch: horizontal with one rate IS vertical
  horizontal  with G stepping down mid-career, L_T equals the closed form
              L0 e^{G_0 T} + sum_s c_s e^{G_s (T - s)}: old money keeps its rate
  wap         G_t inside [1.75%, 3.75%], on the 25 bp grid, G_0 = today's fixing
  fsma        the statutory formula reproduces the FSMA's published rates 2016-2026

Figures (-> figs/rates/):
  rates_fan.png    10Y OLO, G_t, mu_t and the gap mu_t - G_t, per rate model
  table (printed)  constant-rate DP, certainty-equivalent DP and two market plans,
                   each scored under constant, Hull-White and Vasicek rates

Run:  python rates_suite.py [checks|fan|table]
"""
import numpy as np
import common as c
from pension import checks as known   # (the suite function below is called checks)

# This suite compares the regimes itself, so it always starts from constant rates.
assert c.RATES == "constant", "rates_suite compares the rate regimes itself; run it without --rates"
OUT = f"{c.OUT}/rates"   # this suite writes only here

GRID, N_PATHS, SEED = c.GRID, c.N_PATHS, c.SEED
MODELS = ("hull_white", "vasicek")


def _scen(model, n=N_PATHS):
    return c.dp._economy.draw_rate_scenarios(n, seed=c.P.RATE_SEED, model=model, p=c.P)


def _grids():
    return c.grids(nF=GRID["nF"], nR=GRID["nR"], na=GRID["na"])


# --- checks -----------------------------------------------------------------
@c.restores
def checks(n=4000):
    Fg, rg, _ = _grids()
    pol = c.const_policy(0.4, c.P.T, len(Fg), len(rg))
    kw = dict(**c.entry(n, SEED), n_paths=n, seed=SEED)

    # switch: constant -> hull_white -> constant leaves the constant result unchanged
    base = c.simulate(pol, Fg, rg, **kw)
    c.update(RATE_MODEL="hull_white"); c.simulate(pol, Fg, rg, **kw)
    c.update(RATE_MODEL="constant")
    again = c.simulate(pol, Fg, rg, **kw)
    assert base["joint"] == again["joint"] and "rates" not in again
    print("  switch      constant mode identical after visiting hull_white          OK")

    # known answer: degenerate scenario == constant branch (SIGMA_L = 0 in both)
    gap = known.degenerate_rates_gap(c.P, Fg, rg, c.entry(n, SEED), n, SEED)
    assert gap < 1e-10, gap
    print(f"  known       degenerate scenario vs constant branch: max gap {gap:.1e}       OK")

    # horizontal: G steps 3.00% -> 1.75% at t = 20; deterministic, no churn
    rel, vertical_gap = known.horizontal_closed_form(c.P)
    assert rel < 1e-12, rel
    print(f"  horizontal  L_T = closed form (rel err {rel:.1e}); vertical would give "
          f"{vertical_gap:+.1%}  OK")

    # wap: bounds, grid, today's fixing; and the formula against the FSMA's rates
    for m in MODELS:
        st = known.wap_scenario_stats(c.P, m, 2000)
        assert st["min"] >= 0.0175 - 1e-12 and st["max"] <= 0.0375 + 1e-12 and st["grid_err"] < 1e-9
        assert abs(st["G0"] - 0.025) < 1e-12 and st["G0_spread"] == 0.0, st
        print(f"  wap         {m:10s} G in [{st['min']:.2%}, {st['max']:.2%}], 25 bp grid, "
              f"G_0 = {st['G0']:.2%}  OK")
    fsma = known.wap_vs_fsma()
    bad = {y: v for y, v in fsma.items() if abs(v[0] - v[1]) > 1e-9}
    assert not bad, bad
    print(f"  fsma        statutory formula = published WAP rate, {min(fsma)}-{max(fsma)}          OK")


# --- figures ----------------------------------------------------------------
def fan(n=3000):
    c.ensure_out(OUT)
    plt = c.plt
    rows = [("10Y OLO", lambda s: s["y10"][:-1]), ("WAP guarantee $G_t$", lambda s: s["G"]),
            ("book yield $\\mu_t$", lambda s: s["mu"]),
            ("$\\mu_t - G_t$", lambda s: s["mu"] - s["G"])]
    fig, axes = plt.subplots(len(MODELS), len(rows), figsize=(16, 7), sharex=True)
    for i, m in enumerate(MODELS):
        s = _scen(m, n); t = np.arange(s["G"].shape[0])
        for j, (lab, f) in enumerate(rows):
            ax = axes[i, j]; x = f(s) * 100
            ax.fill_between(t, *np.percentile(x, [5, 95], axis=1), alpha=0.18, color="C0", label="5-95%")
            ax.fill_between(t, *np.percentile(x, [25, 75], axis=1), alpha=0.35, color="C0", label="25-75%")
            ax.plot(t, np.median(x, axis=1), color="C0", lw=1.8, label="median")
            if j == 1:
                for lvl in (1.75, 3.75): ax.axhline(lvl, color="grey", ls=":", lw=1)
            if j == 3:
                ax.axhline(0, color="black", lw=0.8)
            ax.set_title(f"{lab} - {m.replace('_', '-')}", fontsize=10)
            if i == len(MODELS) - 1: ax.set_xlabel("career year t")
            if j == 0: ax.set_ylabel("% per year")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Rate scenarios from t0 = {s['t0']:%b %Y}: one OLO path drives both the guarantee and the reserve",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/rates_fan.png", dpi=c.DPI); plt.close(fig)
    print(f"  wrote {OUT}/rates_fan.png")


@c.restores
def table():
    Fg, rg, ag = _grids()
    designs = {"DP (constant rates)": c.solve(Fg, rg, ag, GRID["nq"])["policy"]}
    for m in MODELS:
        sol = c.dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=GRID["nq"], rates=_scen(m), p=c.P)
        designs[f"DP (CE {m})"] = sol["policy"]       # explicit scenario: CE to that model
        ce = sol["ce"]
        print(f"  CE {m:10s}: G={ce['G']:.2%} MU={ce['MU']:.2%} "
              f"SIGMA_L={ce['SIGMA_L']:.2%} SIGMA_R={ce['SIGMA_R']:.2%}")
    for lab, rate in (("flat 5% of salary", 0.05), ("flat 8% of salary", 0.08)):
        designs[lab] = c.schedule_policy(lambda t, r=rate: min(r / c.P.GAMMA, 1.0), len(Fg), len(rg))

    scen = {"constant": None, **{m: _scen(m) for m in MODELS}}
    print(f"\n  {'policy':24s} {'rates':11s} {'joint':>8s} {'benefit':>8s} {'cost':>7s} "
          f"{'avg c%':>7s} {'RR stay':>8s} {'RR leave':>8s} {'regime B':>8s}")
    for lab, pol in designs.items():
        for sname, sc in scen.items():
            r = c.simulate(pol, Fg, rg, **c.entry(N_PATHS, SEED), n_paths=N_PATHS, seed=SEED,
                           rates=sc)
            rb = f"{r['regime_B']:8.1%}" if "regime_B" in r else f"{'-':>8s}"
            print(f"  {lab:24s} {sname:11s} {r['joint']:8.4f} {r['benefit']:8.4f} {r['cost']:7.4f} "
                  f"{r['avg']:7.2f} {r['sty']:8.3f} {r['lea']:8.3f} {rb}")


_ALL = {"checks": checks, "fan": fan, "table": table}


def main(which=None):
    c.ensure_out(OUT)
    for name, fn in _ALL.items():
        if which in (None, name):
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
            fn()


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
