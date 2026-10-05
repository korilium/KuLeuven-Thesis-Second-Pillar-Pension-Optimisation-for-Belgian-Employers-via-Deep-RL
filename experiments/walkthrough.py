"""Numbers for paper/explanations/model-walkthrough.md, computed from the code.

The document's generated blocks sit between <!-- gen:NAME --> and <!-- /gen -->.

    python experiments/walkthrough.py            # print every block
    python experiments/walkthrough.py --write    # refresh the blocks in the document
    python experiments/walkthrough.py --check    # fail if the document is out of date

Part A follows one Hull-White rate scenario from the NBB CSV to simulate()["joint"].
Part B follows one member by hand for three years, constant vs Hull-White.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

import pension.dp as dp
from pension import economy
from pension.dynamics import Exogenous, State, HorizontalLedger, step
from pension.envs.pension_env import PensionEnv, DPPolicyAgent
from pension.params import DEFAULT
from pension.rates.calibration import (bootstrapForwardCurve, calibrateVasicek, computeTheta,
                                       nss_forward, nss_forward_deriv, nss_yield)
from pension.rates.data import load_olo
from pension.rates.pricing import reconstructFutureYield
from pension.rates.simulation import simulateHullWhite
from pension.rates.wap import WAP_LAG, WAP_SHARE, WAP_WINDOW, computeWAPRate, wap_formula

DOC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "explanations",
                   "model-walkthrough.md")

P_HW = DEFAULT.replace(RATE_MODEL="hull_white")
N = 200          # scenarios / careers in the trace (one block of draw_rate_scenarios)
J = 0            # the traced scenario / career
SEED = 7         # career noise (simulate's default seed)
GRID = dict(nF=73, nR=71, na=20, nq=5)


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# =====================================================================================
# Part A -- one scenario from the CSV to the joint value
# =====================================================================================
def part_a():
    blocks = {}
    df10Y, mats, ylds = load_olo()
    t0 = df10Y["DATE"].iloc[-1]
    hist = df10Y["YIELD"].values / 100.0
    show = [1, 2, 5, 10, 20, 30]
    blocks["data"] = "\n".join([
        f"- 10Y series: {len(df10Y)} monthly observations, {df10Y['DATE'].iloc[0]:%b %Y} to "
        f"{t0:%b %Y}; the last one, {pct(hist[-1])}, is month 0 of the model.",
        f"- Latest cross-section ({t0:%b %Y}), in %:",
        "",
        table(["maturity"] + [f"{m}Y" for m in show],
              [["yield"] + [f"{ylds[list(mats).index(m)]:.2f}" for m in show]]),
    ])

    vas = calibrateVasicek(df10Y, verbose=False)
    d = vas["diagnostics"]
    blocks["vasicek"] = table(
        ["kappa", "theta", "sigma", "t-stat of b", "p-value", "half-life"],
        [[f"{vas['kappa']:.4f}", pct(vas["theta"]), pct(vas["sigma"]), f"{d['t_stat_b']:.2f}",
          f"{d['p_val_b']:.3f}", f"{d['half_life_years']:.1f} y"]])

    curve = bootstrapForwardCurve(mats, ylds, verbose=False)
    prm = curve["params"]
    rmse = np.sqrt(np.mean((nss_yield(mats.astype(float), *prm) - ylds / 100) ** 2)) * 1e4
    f00 = prm[0] + prm[1]
    th0 = computeTheta(np.array([1e-9]), np.array([f00]),
                       np.array([nss_forward_deriv(1e-9, *prm)]), vas["kappa"], vas["sigma"])[0]
    blocks["nss"] = "\n".join([
        table(["b0", "b1", "b2", "b3", "tau1", "tau2"],
              [[pct(prm[0], 3), pct(prm[1], 3), pct(prm[2], 3), pct(prm[3], 3),
                f"{prm[4]:.2f} y", f"{prm[5]:.2f} y"]]),
        "",
        f"- Fit to the {len(mats)} observed maturities: RMSE {rmse:.2f} bp.",
        f"- Short rate today r(0) = f(0,0) = b0 + b1 = {pct(f00, 3)}; "
        f"f(0,10) = {pct(nss_forward(10.0, *prm), 3)}; f(0,30) = {pct(nss_forward(30.0, *prm), 3)}.",
        f"- Hull-White target at 0: theta(0) = f(0,0) + f'(0,0)/kappa = {pct(th0, 3)}.",
    ])

    # --- the scenario block, rebuilt step by step exactly as economy._draw_rate_scenarios ---
    p = P_HW
    mpy = int(round(1 / p.RATE_DT))
    rng = np.random.default_rng(p.RATE_SEED)
    r_m = simulateHullWhite(curve, vas["kappa"], vas["sigma"], T=p.T, n_paths=N, dt=p.RATE_DT, rng=rng)
    y10_m = reconstructFutureYield(r_m, vas["kappa"], vas["sigma"], curve, tau=10.0, dt=p.RATE_DT)
    full = np.vstack([np.repeat(hist[:, None], N, axis=1), y10_m[1:]])
    start = len(hist) - 1
    G = computeWAPRate(full, start, p.T, months_per_year=mpy)
    n_book = p.BOOK_DURATION * mpy
    rows_y = start + mpy * np.arange(p.T)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(full, axis=0)])
    mu = (cs[rows_y + 1] - cs[rows_y + 1 - n_book]) / n_book + p.BOOK_SPREAD
    scen = economy.draw_rate_scenarios(N, p=p)
    assert np.array_equal(G, scen["G"]) and np.array_equal(mu, scen["mu"]), "rebuild != economy"

    month = lambda row: (t0 + pd.DateOffset(months=int(row - start)))
    blocks["paths"] = table(
        ["month", "date", "short rate r", "10Y yield (repriced)"],
        [[m, f"{month(start + m):%b %Y}", pct(r_m[m, J], 3), pct(y10_m[m, J], 3)] for m in (0, 12, 24, 36)])

    wap_rows, book_rows = [], []
    for k in range(3):
        end = start + k * mpy - WAP_LAG; lo = end - WAP_WINDOW + 1
        avg = full[lo:end + 1, J].mean()
        src = "observed" if end <= start else ("mixed" if lo <= start else "simulated")
        wap_rows.append([k, f"{month(lo):%b %Y} - {month(end):%b %Y}", src, pct(avg, 3),
                         pct(WAP_SHARE * avg, 3), pct(G[k, J])])
        assert abs(wap_formula(avg) - G[k, J]) < 1e-12
        b_hi = rows_y[k]; b_lo = b_hi - n_book + 1
        book_rows.append([k, f"{month(b_lo):%b %Y} - {month(b_hi):%b %Y}", pct(mu[k, J], 3)])
    blocks["wap"] = table(["year k", "24-month window", "data", "average 10Y",
                           f"x {WAP_SHARE}", "G_k (25 bp grid, [1.75%, 3.75%])"], wap_rows)
    blocks["book"] = table(["year k", f"{p.BOOK_DURATION}-year window", "mu_k"], book_rows)

    # --- careers: noise, entry, a solved policy ---
    exo = Exogenous.draw(p, N, SEED, scen)
    R0, L0, S0 = dp.new_plan_init(N, np.random.default_rng(SEED))
    blocks["exo"] = table(
        ["year t", "zR", "zL", "u (churn)", "hazard h(t)", "G_t", "mu_t"],
        [[t, f"{exo.zR[t, J]:+.4f}", f"{exo.zL[t, J]:+.4f}", f"{exo.u[t, J]:.4f}",
          f"{dp.tenure_hazard(t):.4f}", pct(exo.G[t, J]), pct(exo.mu[t, J], 3)] for t in range(3)])

    Fg, rg, ag = dp.grids(GRID["nF"], GRID["nR"], GRID["na"])
    sol = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=GRID["nq"], p=p)
    ce = sol["ce"]
    blocks["ce"] = (f"G = {pct(ce['G'], 3)}, MU = {pct(ce['MU'], 3)}, "
                    f"SIGMA_L = {pct(ce['SIGMA_L'], 3)}, SIGMA_R = {pct(ce['SIGMA_R'], 3)} "
                    f"(from {p.RATE_CE_PATHS} scenarios)")
    pol = sol["policy"]

    # --- one step of career J, by hand ---
    obj = dp._objective(None, p); w_emp, w_er = obj.weights(p)
    s = State.initial(p, exo.paths(J), R0[J], L0[J], S0[J])
    F, rho = s.F[0], s.rho[0]
    a = float(np.clip(dp.bilinear(Fg, np.log(rg), pol[0], np.array([F]), np.log([rho]))[0], 0, 1))
    c = a * p.GAMMA * s.S[0]
    R1 = (s.R[0] + c) * np.exp(exo.mu[0, J] + p.SIGMA_R_RATES * exo.zR[0, J])
    L1 = s.L[0] * np.exp(exo.G[0, J]) + c * np.exp(exo.G[0, J])
    ST = s.S[0] * (1 + p.W) ** p.T
    r0 = -w_er * obj.contribution(np.array([a]), 0, p)[0]
    s2, _ = step(s, np.array([a]), exo.paths(J), p, dp.tenure_hazard)
    assert abs(s2.R[0] - R1) < 1e-12 and abs(s2.L[0] - L1) < 1e-12
    blocks["step"] = "\n".join([
        f"- Entry: R = {R0[J]:.4f}, L = {L0[J]:.4f}, S = {S0[J]:.4f}, "
        f"so F = {F:.4f}, rho = {rho:.3f}.",
        f"- Policy: a = a*(0, F, rho) = {a:.4f}, so c = a * GAMMA * S = {a:.4f} x {p.GAMMA} x {S0[J]:.3f} = {c:.4f}.",
        f"- Reserve: R' = (R + c) e^(mu_0 + {p.SIGMA_R_RATES} zR) = ({R0[J]:.4f} + {c:.4f}) "
        f"e^({pct(exo.mu[0, J], 3)} + {p.SIGMA_R_RATES} x {exo.zR[0, J]:+.4f}) = {R1:.4f}.",
        f"- Ledger: vintage 0 (opening L, locked at {pct(exo.G[0, J])}) and vintage 1 "
        f"(this contribution, locked at G_0 = {pct(exo.G[0, J])}); L' = {L1:.4f}.",
        f"- Salary: S' = S (1 + W) = {s2.S[0]:.4f}; churn: u = {exo.u[0, J]:.4f} vs h(0) = "
        f"{dp.tenure_hazard(0):.4f} -> {'leaves' if not s2.present[0] else 'stays'}.",
        f"- New state: F' = {s2.F[0]:.4f}, rho' = {s2.rho[0]:.3f}.",
        f"- Reward of year 0: -w_er x a GAMMA (1+W)^-(T-0) x e^0 = -{w_er} x {a:.4f} x {p.GAMMA} x "
        f"{(1 + p.W) ** -p.T:.4f} = {r0:.6f} (final-salary units; S_T = {ST:.2f}).",
    ])

    # --- whole careers: environment returns vs simulate()["joint"] ---
    sim = dp.simulate(pol, Fg, rg, R0=R0, L0=L0, S0=S0, n_paths=N, seed=SEED, rates=scen, p=p)
    env, agent = PensionEnv(p=p), DPPolicyAgent(pol, Fg, rg, p.T)
    rets, info_j, contrib_j = [], None, 0.0
    for i in range(N):
        obs, _ = env.reset(options=dict(exogenous=exo.paths(i), entry=(R0[i], L0[i], S0[i])))
        total, done = 0.0, False
        while not done:
            obs, r, done, _, info = env.step(agent(obs))
            if i == J and not done:
                contrib_j += r
            total += r
        rets.append(total)
        if i == J:
            info_j = info
    rets = np.array(rets)
    terminal_j = rets[J] - contrib_j
    blocks["joint"] = "\n".join([
        f"- Career {J}: {'stays to retirement' if info_j['stayed'] else 'leaves before T'}; "
        f"final total replacement rate {info_j['RR_tot']:.3f}.",
        f"- Its return: contributions {contrib_j:+.6f} plus the terminal reward "
        f"{terminal_j:+.6f} = {rets[J]:+.6f}.",
        f"- Mean return over the {N} careers: {rets.mean():+.8f}.",
        f"- simulate(...)[\"joint\"] on the same paths: {sim['joint']:+.8f} "
        f"(difference {abs(rets.mean() - sim['joint']):.1e}).",
        f"- Of which benefit {sim['benefit']:.6f} (employee leg, weight {w_emp}) and cost "
        f"{sim['cost']:.6f} (employer leg, weight {w_er}).",
    ])
    assert abs(rets.mean() - sim["joint"]) < 1e-12
    return blocks, exo.paths(J)


# =====================================================================================
# Part B -- one member by hand, constant rates vs Hull-White
# =====================================================================================
def part_b(exo_j):
    blocks = {}
    a, R0, L0, S0, years = 0.5, 1.0, 1.0, 20.0, 3
    never = lambda t: 0.0
    pc, ph = DEFAULT, P_HW
    shocks = Exogenous.draw(pc, 1, 1)                    # the member's own zR, zL
    exo_c = shocks
    exo_h = Exogenous(shocks.zR, shocks.zL, shocks.u, exo_j.G, exo_j.mu)
    obj = dp._objective(None, pc); w_er = obj.weights(pc)[1]

    def career(p, exo, horizontal):
        s = State.initial(p, exo, R0, L0, S0)
        rows = [[0, f"{s.S[0]:.3f}", "", "", f"{s.R[0]:.4f}", ledger_str(s, horizontal),
                 f"{s.L[0]:.4f}", f"{s.F[0]:.4f}", f"{s.rho[0]:.3f}", ""]]
        for t in range(years):
            F, rho = s.F[0], s.rho[0]
            mu, sig = (exo.mu[t, 0], p.SIGMA_R_RATES) if horizontal else (p.MU, p.SIGMA_R)
            s, c = step(s, np.array([a]), exo, p, never)
            reward = -w_er * obj.contribution(np.array([a]), t, p)[0] * np.exp(-p.DISC_ER * t)
            if not horizontal:   # the reduced form the DP uses gives the same state
                l = a * p.GAMMA * rho
                assert abs(dp.F_next(F, l, exo.zR[t, 0], exo.zL[t, 0], p) - s.F[0]) < 1e-12
                assert abs(dp.rho_next(rho, l, exo.zL[t, 0], p) - s.rho[0]) < 1e-12
            rows.append([t + 1, f"{s.S[0]:.3f}", f"{c[0]:.4f}",
                         f"e^({pct(mu, 2)} {sig * exo.zR[t, 0]:+.4f})", f"{s.R[0]:.4f}",
                         ledger_str(s, horizontal), f"{s.L[0]:.4f}", f"{s.F[0]:.4f}",
                         f"{s.rho[0]:.3f}", f"{reward:+.6f}"])
        return rows

    def ledger_str(s, horizontal):
        if not horizontal:
            return "one pot"
        k = s.t + 1
        return "; ".join(f"{s.ledger.Lv[v, 0]:.4f}@{pct(s.ledger.lock[v, 0])}" for v in range(k))

    hdr = ["after year", "S", "c = a Γ S", "R growth", "R", "ledger (amount@locked G)",
           "L", "F = R/L", "ρ = S/L", "reward of the year"]
    blocks["shocks"] = table(["year t", "zR", "zL", "G_t (HW)", "mu_t (HW)"],
                             [[t, f"{exo_c.zR[t, 0]:+.4f}", f"{exo_c.zL[t, 0]:+.4f}",
                               pct(exo_h.G[t, 0]), pct(exo_h.mu[t, 0], 3)] for t in range(years)])
    blocks["career_constant"] = table(hdr, career(pc, exo_c, False))
    blocks["career_hw"] = table(hdr, career(ph, exo_h, True))

    # --- same (F, rho), different vintage mix: (F, rho) is not Markov under rates ---
    T, t = ph.T, 2
    R, S, L = 1.2, 18.0, 1.5
    mixes = {"early money (locked low)": [(1.0, 0.0175), (0.5, 0.0250)],
             "late money (locked high)": [(0.5, 0.0250), (1.0, 0.0375)]}
    G_now, mu_now = 0.03, 0.03
    rows = []
    for name, vint in mixes.items():
        Lv = np.zeros((T + 1, 1)); lock = np.zeros((T + 1, 1))
        for v, (amt, g) in enumerate(vint):
            Lv[v, 0], lock[v, 0] = amt, g
        led = HorizontalLedger(Lv, lock, np.array([L]))
        s = State(t - 1, np.array([R]), np.array([S]), led, np.array([True]), np.array([T], float))
        exo = Exogenous(np.zeros((T, 1)), np.zeros((T, 1)), np.ones((T, 1)),
                        np.full((T, 1), G_now), np.full((T, 1), mu_now))
        F0, rho0 = s.F[0], s.rho[0]
        s, _ = step(s, np.array([a]), exo, ph, never)
        F1 = s.F[0]
        L_T = float(sum(Lv[v, 0] * np.exp(lock[v, 0] * (T - t)) for v in range(T + 1) if Lv[v, 0] > 0))
        rows.append([name, " + ".join(f"{amt:.1f}@{pct(g)}" for amt, g in vint), f"{F0:.4f}",
                     f"{rho0:.3f}", f"{F1:.4f}", f"{L_T:.3f}"])
    blocks["not_markov"] = table(
        ["member", "vintages (amount@locked G)", "F now", "ρ now", "F next year",
         f"L at T if nothing more is paid ({T - t} years)"], rows)
    return blocks


def all_blocks():
    a, exo_j = part_a()
    return {**a, **part_b(exo_j)}


def render(doc, blocks):
    def sub(m):
        name = m.group(1)
        if name not in blocks:
            raise KeyError(f"no generated block '{name}'")
        return f"<!-- gen:{name} -->\n{blocks[name]}\n<!-- /gen -->"
    return re.sub(r"<!-- gen:(\w+) -->.*?<!-- /gen -->", sub, doc, flags=re.S)


if __name__ == "__main__":
    blocks = all_blocks()
    if "--write" in sys.argv or "--check" in sys.argv:
        doc = open(DOC).read()
        new = render(doc, blocks)
        if "--check" in sys.argv:
            if new != doc:
                sys.exit("model-walkthrough.md is out of date: run with --write")
            print("model-walkthrough.md is up to date")
        else:
            open(DOC, "w").write(new)
            print(f"wrote {len(blocks)} blocks into {DOC}")
    else:
        for name, text in blocks.items():
            print(f"\n<!-- gen:{name} -->\n{text}\n<!-- /gen -->")
