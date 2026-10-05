"""Every number stated in the documentation, computed from the code.

Generated blocks sit between <!-- gen:NAME --> and <!-- /gen --> in the documents
listed in DOCS; each document has its own block producers.

    python experiments/walkthrough.py                      # print the blocks of every document
    python experiments/walkthrough.py --write [DOC ...]    # refresh the blocks (default: all documents)
    python experiments/walkthrough.py --check [DOC ...]    # fail if a document is out of date

DOC is a key of DOCS (walkthrough, review, architecture, model, reference, debugging).

Canonical configuration: RATE_MODEL = "vasicek", EMPLOYER_NUMERAIRE = "retirement".
The walkthrough (Part A) follows one vasicek scenario from the NBB CSV to
simulate()["joint"]; Part B follows one member by hand, constant vs vasicek.
"""
import inspect
import os
import re
import sys

import numpy as np
import pandas as pd

import pension.dp as dp
from pension import checks, economy, numeraire
from pension.dynamics import Exogenous, HorizontalLedger, State, step
from pension.envs.pension_env import DPPolicyAgent, PensionEnv
from pension.objective import OBJECTIVES
from pension.params import DEFAULT, Params
from pension.rates import wap
from pension.rates.accrual import closed_form_accrual
from pension.rates.calibration import calibrateVasicek
from pension.rates.data import load_olo
from pension.rates.simulation import simulateVasicek

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOCS = {
    "walkthrough": "paper/explanations/model-walkthrough.md",
    "review": "docs/REVIEW.md",
    "architecture": "docs/ARCHITECTURE.md",
    "model": "docs/MODEL.md",
    "reference": "docs/REFERENCE.md",
    "debugging": "docs/DEBUGGING.md",
}

P_V = DEFAULT.replace(RATE_MODEL="vasicek")          # the canonical stochastic configuration
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


def grids():
    return dp.grids(GRID["nF"], GRID["nR"], GRID["na"])


# =====================================================================================
# walkthrough, Part A -- one vasicek scenario from the CSV to the joint value
# =====================================================================================
def part_a():
    blocks = {}
    p = P_V
    df10Y, mats, ylds = load_olo()
    t0 = df10Y["DATE"].iloc[-1]
    hist = df10Y["YIELD"].values / 100.0
    show = [1, 2, 5, 10, 20, 30]
    blocks["data"] = "\n".join([
        f"- 10Y series: {len(df10Y)} monthly observations, {df10Y['DATE'].iloc[0]:%b %Y} to "
        f"{t0:%b %Y}; the last one, {pct(hist[-1])}, is month 0 of the model.",
        f"- Latest cross-section ({t0:%b %Y}), in % (used by the robustness models only):",
        "",
        table(["maturity"] + [f"{m}Y" for m in show],
              [["yield"] + [f"{ylds[list(mats).index(m)]:.2f}" for m in show]]),
    ])

    vas = calibrateVasicek(df10Y, verbose=False)
    d = vas["diagnostics"]
    vp = economy.vasicek_params(p)
    blocks["vasicek"] = table(
        ["kappa", "theta (OLS)", "theta used", "sigma", "t-stat of b", "p-value", "half-life"],
        [[f"{vas['kappa']:.4f}", pct(vas["theta"]), pct(vp["theta"]), pct(vas["sigma"]),
          f"{d['t_stat_b']:.2f}", f"{d['p_val_b']:.3f}", f"{d['half_life_years']:.1f} y"]])
    blocks["spread"] = (f"SPREAD_10Y_SHORT = {p.SPREAD_10Y_SHORT} -> the historical mean (10Y - "
                        f"{economy.rate_calibration()['short']['maturity']}) = {vp['spread'] * 1e4:.1f} bp; "
                        f"short rate today r_0 = {pct(vp['y0'], 3)} - {vp['spread'] * 1e4:.1f} bp = "
                        f"{pct(vp['y0'] - vp['spread'], 3)}.")

    # --- the scenario block, rebuilt step by step exactly as economy._draw_rate_scenarios ---
    mpy = int(round(1 / p.RATE_DT))
    y10_m = simulateVasicek(vp["kappa"], vp["theta"], vp["sigma"], vp["y0"], T=p.T, n_paths=N,
                            dt=p.RATE_DT, rng=np.random.default_rng(p.RATE_SEED))
    r_m = y10_m - vp["spread"]
    full = np.vstack([np.repeat(hist[:, None], N, axis=1), y10_m[1:]])
    start = len(hist) - 1
    G = wap.computeWAPRate(full, start, p.T, months_per_year=mpy)
    n_book = p.BOOK_DURATION * mpy
    rows_y = start + mpy * np.arange(p.T)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(full, axis=0)])
    mu = (cs[rows_y + 1] - cs[rows_y + 1 - n_book]) / n_book + p.BOOK_SPREAD
    acc = np.exp(np.add.reduceat(r_m[:-1] * p.RATE_DT, np.arange(0, p.T * mpy, mpy), axis=0))
    scen = economy.draw_rate_scenarios(N, p=p)
    assert np.array_equal(G, scen["G"]) and np.array_equal(mu, scen["mu"]), "rebuild != economy"
    assert np.array_equal(acc, scen["acc"]) and np.array_equal(r_m[::mpy], scen["r"]), "rebuild != economy"

    month = lambda row: (t0 + pd.DateOffset(months=int(row - start)))
    blocks["paths"] = table(
        ["month", "date", "10Y y (modelled)", "short rate r = y - spread"],
        [[m, f"{month(start + m):%b %Y}", pct(y10_m[m, J], 3), pct(r_m[m, J], 3)] for m in (0, 12, 24, 36)])

    wap_rows, book_rows = [], []
    for k in range(3):
        end = start + k * mpy - wap.WAP_LAG; lo = end - wap.WAP_WINDOW + 1
        avg = full[lo:end + 1, J].mean()
        src = "observed" if end <= start else ("mixed" if lo <= start else "simulated")
        wap_rows.append([k, f"{month(lo):%b %Y} - {month(end):%b %Y}", src, pct(avg, 3),
                         pct(wap.WAP_SHARE * avg, 3), pct(G[k, J])])
        assert abs(wap.wap_formula(avg) - G[k, J]) < 1e-12
        b_hi = rows_y[k]; b_lo = b_hi - n_book + 1
        book_rows.append([k, f"{month(b_lo):%b %Y} - {month(b_hi):%b %Y}", pct(mu[k, J], 3)])
    blocks["wap"] = table(["year k", "24-month window", "data", "average 10Y",
                           f"x {wap.WAP_SHARE}", "G_k (25 bp grid, [1.75%, 3.75%])"], wap_rows)
    blocks["book"] = table(["year k", f"{p.BOOK_DURATION}-year window", "mu_k"], book_rows)

    # --- careers: noise, entry, the certainty-equivalent policy ---
    exo = Exogenous.draw(p, N, SEED, scen)
    R0, L0, S0 = dp.new_plan_init(N, np.random.default_rng(SEED))
    blocks["exo"] = table(
        ["year t", "zR", "zL", "u (churn)", "hazard h(t)", "G_t", "mu_t", "r_t", "acc_t (realised)"],
        [[t, f"{exo.zR[t, J]:+.4f}", f"{exo.zL[t, J]:+.4f}", f"{exo.u[t, J]:.4f}",
          f"{dp.tenure_hazard(t):.4f}", pct(exo.G[t, J]), pct(exo.mu[t, J], 3), pct(exo.r[t, J], 3),
          f"{exo.acc[t, J]:.5f}"] for t in range(3)])

    Fg, rg, ag = grids()
    sol = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=GRID["nq"], p=p)
    ce = sol["ce"]
    prem_ce = numeraire.premium_schedule(p, economy.draw_rate_scenarios(p.RATE_CE_PATHS, p=p))
    blocks["ce"] = "\n".join([
        f"- Moments ({p.RATE_CE_PATHS} scenarios): G = {pct(ce['G'], 3)}, MU = {pct(ce['MU'], 3)}, "
        f"SIGMA_L = {pct(ce['SIGMA_L'], 3)}, SIGMA_R = {pct(ce['SIGMA_R'], 3)}.",
        "- Premium factor per year (the scenario mean of A(t,T; r_t)): " +
        ", ".join(f"t={t}: {prem_ce[t]:.4f}" for t in (0, 10, 20, 30, 44)) + ".",
    ])
    pol = sol["policy"]

    # --- one year of career J, by hand ---
    obj = dp._objective(None, p); w_emp, w_er = obj.weights(p)
    s = State.initial(p, exo.paths(J), R0[J], L0[J], S0[J])
    F, rho = s.F[0], s.rho[0]
    a = float(np.clip(dp.bilinear(Fg, np.log(rg), pol[0], np.array([F]), np.log([rho]))[0], 0, 1))
    c = a * p.GAMMA * s.S[0]
    R1 = (s.R[0] + c) * np.exp(exo.mu[0, J] + p.SIGMA_R_RATES * exo.zR[0, J])
    L1 = s.L[0] * np.exp(exo.G[0, J]) + c * np.exp(exo.G[0, J])
    # the closed-form accrual by hand (vasicek): y_t = r_t + spread
    tau = p.T
    B = (1 - np.exp(-vp["kappa"] * tau)) / vp["kappa"]
    y_0 = exo.r[0, J] + vp["spread"]
    mean = vp["theta"] * tau + (y_0 - vp["theta"]) * B
    var = (vp["sigma"]**2 / vp["kappa"]**2) * (tau - B) - (vp["sigma"]**2 / (2 * vp["kappa"])) * B**2
    A0 = np.exp(mean + var / 2 + (p.FINANCING_SPREAD - vp["spread"]) * tau)
    assert abs(A0 / closed_form_accrual(0, exo.r[0, J], p, s=p.FINANCING_SPREAD) - 1) < 1e-12
    assert abs(A0 / numeraire.premium_factor(0, exo.r[0, J], p) - 1) < 1e-12
    contrib = a * p.GAMMA * (1 + p.W) ** -p.T
    r0 = -w_er * contrib * A0
    s2, _ = step(s, np.array([a]), exo.paths(J), p, dp.tenure_hazard)
    assert abs(s2.R[0] - R1) < 1e-12 and abs(s2.L[0] - L1) < 1e-12
    blocks["step"] = "\n".join([
        f"- Entry: R = {R0[J]:.4f}, L = {L0[J]:.4f}, S = {S0[J]:.4f}, so F = {F:.4f}, rho = {rho:.3f}.",
        f"- Policy: a = a*(0, F, rho) = {a:.4f}, so c = a GAMMA S = {a:.4f} x {p.GAMMA} x {S0[J]:.3f} = {c:.4f}.",
        f"- Reserve: R' = (R + c) e^(mu_0 + {p.SIGMA_R_RATES} zR) = ({R0[J]:.4f} + {c:.4f}) "
        f"e^({pct(exo.mu[0, J], 3)} + {p.SIGMA_R_RATES} x {exo.zR[0, J]:+.4f}) = {R1:.4f}.",
        f"- Ledger: vintage 0 (opening L) and vintage 1 (this contribution), both locked at "
        f"G_0 = {pct(exo.G[0, J])}; L' = {L1:.4f}.",
        f"- Salary: S' = S (1 + W) = {s2.S[0]:.4f}; churn: u = {exo.u[0, J]:.4f} vs h(0) = "
        f"{dp.tenure_hazard(0):.4f} -> {'leaves' if not s2.present[0] else 'stays'}.",
        f"- New state: F' = {s2.F[0]:.4f}, rho' = {s2.rho[0]:.3f}.",
        f"- Accrual of the year-0 premium to T: r_0 = {pct(exo.r[0, J], 3)}, y_0 = r_0 + spread = "
        f"{pct(y_0, 3)}; mean(I_y) = {mean:.5f}, var(I_y) = {var:.6f}; "
        f"A(0,T) = exp(mean + var/2 + (s - spread) T) = {A0:.4f}.",
        f"- Reward of year 0 (retirement-date money): -w_er x a GAMMA (1+W)^-T x A(0,T) = "
        f"-{w_er} x {a:.4f} x {p.GAMMA} x {(1 + p.W) ** -p.T:.4f} x {A0:.4f} = {r0:.6f}.",
    ])

    # --- whole careers: environment returns vs simulate()["joint"] ---
    sim = dp.simulate(pol, Fg, rg, R0=R0, L0=L0, S0=S0, n_paths=N, seed=SEED, rates=scen, p=p)
    env, agent = PensionEnv(p=p), DPPolicyAgent(pol, Fg, rg, p.T)
    rets, info_j, contrib_j, obs_j = [], None, 0.0, None
    for i in range(N):
        obs, _ = env.reset(options=dict(exogenous=exo.paths(i), entry=(R0[i], L0[i], S0[i])))
        if i == J:
            obs_j = obs.copy()
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
    assert abs(rets.mean() - sim["joint"]) < 1e-12
    blocks["obs"] = table(["t/T", "F", "log rho", "G_t", "mu_t", "r_t"],
                          [[f"{obs_j[0]:.3f}", f"{obs_j[1]:.4f}", f"{obs_j[2]:.4f}", pct(obs_j[3]),
                            pct(obs_j[4], 3), pct(obs_j[5], 3)]])
    blocks["joint"] = "\n".join([
        f"- Career {J}: {'stays to retirement' if info_j['stayed'] else 'leaves before T'}; "
        f"final total replacement rate {info_j['RR_tot']:.3f}.",
        f"- Its return: premiums {contrib_j:+.6f} plus the terminal reward "
        f"{rets[J] - contrib_j:+.6f} = {rets[J]:+.6f} (retirement-date money).",
        f"- Mean return over the {N} careers: {rets.mean():+.8f}.",
        f"- simulate(...)[\"joint\"] on the same paths: {sim['joint']:+.8f} "
        f"(difference {abs(rets.mean() - sim['joint']):.1e}).",
        f"- Of which benefit {sim['benefit']:.6f} (employee leg, weight {w_emp}) and cost "
        f"{sim['cost']:.6f} (employer leg, weight {w_er}).",
    ])
    return blocks, exo.paths(J)


# =====================================================================================
# walkthrough, Part B -- one member by hand, constant rates vs vasicek
# =====================================================================================
def part_b(exo_j):
    blocks = {}
    a, R0, L0, S0, years = 0.5, 1.0, 1.0, 20.0, 3
    never = lambda t: 0.0
    pc, pv = DEFAULT, P_V
    shocks = Exogenous.draw(pc, 1, 1)                    # the member's own zR, zL
    exo_c = shocks
    exo_v = Exogenous(shocks.zR, shocks.zL, shocks.u, exo_j.G, exo_j.mu, exo_j.r, exo_j.acc)
    obj = dp._objective(None, pc); w_er = obj.weights(pc)[1]

    def ledger_str(s, horizontal):
        if not horizontal:
            return "one pot"
        return "; ".join(f"{s.ledger.Lv[v, 0]:.4f}@{pct(s.ledger.lock[v, 0])}" for v in range(s.t + 1))

    def career(p, exo, horizontal):
        s = State.initial(p, exo, R0, L0, S0)
        rows = [[0, f"{s.S[0]:.3f}", "", "", "", f"{s.R[0]:.4f}", ledger_str(s, horizontal),
                 f"{s.L[0]:.4f}", f"{s.F[0]:.4f}", f"{s.rho[0]:.3f}", ""]]
        for t in range(years):
            F, rho = s.F[0], s.rho[0]
            mu, sig = (exo.mu[t, 0], p.SIGMA_R_RATES) if horizontal else (p.MU, p.SIGMA_R)
            A = float(np.asarray(numeraire.premium_factor(t, exo.r[t, 0], p)).ravel()[0])
            s, c = step(s, np.array([a]), exo, p, never)
            reward = -w_er * obj.contribution(np.array([a]), t, p)[0] * A
            if not horizontal:   # the reduced form the DP uses gives the same state
                l = a * p.GAMMA * rho
                assert abs(dp.F_next(F, l, exo.zR[t, 0], exo.zL[t, 0], p) - s.F[0]) < 1e-12
                assert abs(dp.rho_next(rho, l, exo.zL[t, 0], p) - s.rho[0]) < 1e-12
            rows.append([t + 1, f"{s.S[0]:.3f}", f"{c[0]:.4f}", f"{A:.4f}",
                         f"e^({pct(mu, 2)} {sig * exo.zR[t, 0]:+.4f})", f"{s.R[0]:.4f}",
                         ledger_str(s, horizontal), f"{s.L[0]:.4f}", f"{s.F[0]:.4f}",
                         f"{s.rho[0]:.3f}", f"{reward:+.6f}"])
        return rows

    hdr = ["after year", "S", "c = a Γ S", "A(t,T) of c", "R growth", "R", "ledger (amount@locked G)",
           "L", "F = R/L", "ρ = S/L", "reward of the year"]
    blocks["shocks"] = table(["year t", "zR", "zL", "G_t (vasicek)", "mu_t (vasicek)", "r_t (vasicek)"],
                             [[t, f"{exo_c.zR[t, 0]:+.4f}", f"{exo_c.zL[t, 0]:+.4f}",
                               pct(exo_v.G[t, 0]), pct(exo_v.mu[t, 0], 3), pct(exo_v.r[t, 0], 3)]
                              for t in range(years)])
    blocks["career_constant"] = table(hdr, career(pc, exo_c, False))
    blocks["career_hw"] = table(hdr, career(pv, exo_v, True))

    # --- same (F, rho), different vintage mix: (F, rho) is not Markov under rates ---
    T, t = pv.T, 2
    R, S, L = 1.2, 18.0, 1.5
    mixes = {"early money (locked low)": [(1.0, 0.0175), (0.5, 0.0250)],
             "late money (locked high)": [(0.5, 0.0250), (1.0, 0.0375)]}
    rows = []
    for name, vint in mixes.items():
        Lv = np.zeros((T + 1, 1)); lock = np.zeros((T + 1, 1))
        for v, (amt, g) in enumerate(vint):
            Lv[v, 0], lock[v, 0] = amt, g
        s = State(t - 1, np.array([R]), np.array([S]), HorizontalLedger(Lv, lock, np.array([L])),
                  np.array([True]), np.array([T], float))
        exo = Exogenous(np.zeros((T, 1)), np.zeros((T, 1)), np.ones((T, 1)),
                        np.full((T, 1), 0.03), np.full((T, 1), 0.03))
        F0, rho0 = s.F[0], s.rho[0]
        s, _ = step(s, np.array([a]), exo, pv, never)
        L_T = float(sum(Lv[v, 0] * np.exp(lock[v, 0] * (T - t)) for v in range(T + 1) if Lv[v, 0] > 0))
        rows.append([name, " + ".join(f"{amt:.1f}@{pct(g)}" for amt, g in vint), f"{F0:.4f}",
                     f"{rho0:.3f}", f"{s.F[0]:.4f}", f"{L_T:.3f}"])
    blocks["not_markov"] = table(
        ["member", "vintages (amount@locked G)", "F now", "ρ now", "F next year",
         f"L at T if nothing more is paid ({T - t} years)"], rows)
    return blocks


def walkthrough_blocks():
    a, exo_j = part_a()
    return {**a, **part_b(exo_j)}


# =====================================================================================
# reference / model / architecture / debugging -- defaults and computed values
# =====================================================================================
PARAM_DOC = {   # unit, meaning, used by, inert under the canonical configuration
    "T": ("years", "career length", "everything", "no"),
    "G": ("1/yr", "WAP guarantee rate (constant model)", "dynamics (vertical ledger), dp", "vasicek: yes (CE sets it)"),
    "MU": ("1/yr", "credited log-return (constant model)", "dynamics, dp, paidup_service", "vasicek: yes (CE sets it)"),
    "W": ("1/yr", "salary growth", "dynamics, objective contribution leg, dp", "no"),
    "S0": ("salary", "starting salary (scale-free)", "nothing in the DP/env (tabular via economy)", "yes"),
    "DISC_EMP": ("1/yr", "employee discount rate", "numeraire (discounted only)", "yes"),
    "DISC_ER": ("1/yr", "employer discount rate", "numeraire (discounted only)", "yes"),
    "DISC": ("1/yr", "tabular rung's discount (legacy copy)", "nothing (tabular reads economy.DISC)", "yes"),
    "SIGMA_R": ("1/sqrt(yr)", "asset shock (constant model)", "dynamics, dp", "vasicek: yes (CE sets it)"),
    "SIGMA_L": ("1/sqrt(yr)", "guarantee shock (vertical ledger)", "dynamics, dp", "vasicek: yes (CE sets it)"),
    "SIGMA": ("1/sqrt(yr)", "tabular rung's shock (legacy copy)", "nothing (tabular reads economy.SIGMA)", "yes"),
    "GAMMA": ("share of salary", "contribution capacity, c = a GAMMA S", "dynamics, objective, dp", "no"),
    "LAMBDA": ("-", "employee weight", "objective weights", "no"),
    "ETA": ("-", "CRRA curvature", "objective.crra", "no"),
    "ANNUITY": ("years", "annuity factor, capital -> annual pension", "objective, dp, simulate, env", "no"),
    "RR_LEGAL": ("-", "first-pillar replacement rate", "dp, simulate, env", "no"),
    "RR_TARGET": ("-", "total-adequacy target", "dp, simulate, env", "no"),
    "SATIATE": ("bool", "cap RR at the target in crra()", "objective.crra", "no"),
    "OBJECTIVE": ("name", "value function in objective.OBJECTIVES", "dp, env", "no"),
    "BETA": ("-", "policy extraction temperature (relative)", "dp._solve", "no"),
    "RATE_MODEL": ("name", "rate regime", "economy, numeraire, dp, env", "no"),
    "RATE_DT": ("years", "rate time step (must be 1/12: the data are monthly)", "economy", "no (not free)"),
    "RATE_SEED": ("-", "seed of the rate scenarios", "economy, dp, env pool", "no"),
    "BOOK_DURATION": ("years", "book-yield averaging window", "economy", "no"),
    "BOOK_SPREAD": ("1/yr", "book yield over the 10Y", "economy", "no"),
    "SIGMA_R_RATES": ("1/sqrt(yr)", "excess-return noise on the book yield", "dynamics, dp.certainty_equivalent", "no"),
    "LONG_RATE_P": ("1/yr", "long-run P level (meaning differs per model, REVIEW M1)", "economy params helpers, accrual", "no"),
    "SPREAD_10Y_SHORT": ("1/yr", "vasicek: r = y10 - spread (None: historical mean)", "economy.vasicek_params, accrual", "no"),
    "RATE_CE_PATHS": ("-", "scenarios behind the CE solve", "dp.solve", "no"),
    "EMPLOYER_NUMERAIRE": ("name", "valuation date: 'retirement' | 'discounted' (REVIEW m9)", "numeraire", "no"),
    "SHORT_RATE": ("1/yr", "short rate of the constant model", "numeraire, dynamics.Exogenous", "vasicek: yes"),
    "FINANCING_SPREAD": ("1/yr", "s: financing spread over the short rate", "numeraire, accrual", "no"),
    "N_EVAL": ("-", "tabular rung's SAA batch size (legacy copy)", "nothing (tabular reads economy.N_EVAL)", "yes"),
}


def reference_blocks():
    import dataclasses
    rows = []
    for f in dataclasses.fields(Params):
        unit, meaning, used, inert = PARAM_DOC[f.name]
        rows.append([f"`{f.name}`", f"`{getattr(DEFAULT, f.name)!r}`", unit, meaning, used, inert])
    missing = set(PARAM_DOC) ^ {f.name for f in dataclasses.fields(Params)}
    assert not missing, f"PARAM_DOC out of date: {missing}"
    blocks = {"params": table(["field", "default", "unit", "meaning", "used by", "inert (canonical)"], rows)}
    blocks["objectives"] = table(["name", "what it is"], [[f"`{n}`", o.note] for n, o in OBJECTIVES.items()])

    import ast
    test_files = sorted(f for f in os.listdir(os.path.join(ROOT, "tests")) if f.endswith(".py"))
    tests = []                                   # (file, test name, source, [assert sources])
    for f in test_files:
        src = open(os.path.join(ROOT, "tests", f)).read()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                body = ast.get_source_segment(src, node)
                asserts = [" ".join(("assert " + ast.get_source_segment(src, a.test)).split())[:160]
                           for a in ast.walk(node) if isinstance(a, ast.Assert)]
                tests.append((f, node.name, body, [a for a in asserts if a]))
    checks_src = inspect.getsource(checks)
    crow = []
    for name, fn in inspect.getmembers(checks, inspect.isfunction):
        if fn.__module__ != "pension.checks" or name.startswith("_"):
            continue
        doc = (inspect.getdoc(fn) or "").split("\n\n")[0].replace("\n", " ").replace("|", "∣")
        hits = [(f, t, a) for f, t, body, a in tests if re.search(r"\b" + name + r"\b", body)]
        if hits:
            where = "<br>".join(f"`{f}::{t}`" for f, t, _ in hits)
            tol = "<br>".join(f"`{a[0].replace('|', '∣')}`" for _, _, a in hits if a)
        else:
            indirect = len(re.findall(r"\b" + name + r"\(", checks_src)) > 1
            where, tol = ("via other checks" if indirect else "**none**"), ""
        crow.append([f"`{name}`", doc, tol, where])
    blocks["checks"] = table(["check", "what it measures (known answer)", "tolerance (first assert)", "asserted in"], crow)
    blocks["api"] = api_reference()
    return blocks


API_MODULES = ["pension.params", "pension.objective", "pension.numeraire", "pension.dynamics",
               "pension.economy", "pension.dp", "pension.checks", "pension.envs.pension_env",
               "pension.envs.tabular", "pension.rates.data", "pension.rates.calibration",
               "pension.rates.simulation", "pension.rates.pricing", "pension.rates.wap",
               "pension.rates.accrual"]


def _callers(name, own):
    """Files (outside the defining module) that call or import `name`."""
    hits = []
    for base in ("pension", "experiments", "tests"):
        for dirpath, _, files in os.walk(os.path.join(ROOT, base)):
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                rel = os.path.relpath(path, ROOT)
                if rel == own:
                    continue
                src = open(path).read()
                if re.search(r"(?<![\w])" + re.escape(name) + r"\s*\(", src) or \
                        re.search(r"^\s*(from|import)\b[^\n]*\b" + re.escape(name) + r"\b", src, re.M):
                    hits.append(rel.replace("pension/", "").replace("experiments/", "exp/"))
    return hits


def api_reference():
    import importlib
    out = []
    for mod_name in API_MODULES:
        mod = importlib.import_module(mod_name)
        own = os.path.relpath(inspect.getsourcefile(mod), ROOT)
        head = " ".join((inspect.getdoc(mod) or "").split())
        head = head.split(" -- ", 1)[-1].split(". ")[0].rstrip(".") + "."
        out.append(f"### `{mod_name}` ({own})\n\n{head}\n")
        rows = []
        for name, obj in inspect.getmembers(mod, lambda o: inspect.isfunction(o) or inspect.isclass(o)):
            if name.startswith("_") or getattr(obj, "__module__", None) != mod_name:
                continue
            try:
                sig = str(inspect.signature(obj))
            except (TypeError, ValueError):
                sig = "(...)"
            sig = sig if len(sig) < 90 else sig[:87] + "...)"
            doc = (inspect.getdoc(obj) or "").split("\n\n")[0].replace("\n", " ").replace("|", "/")
            if doc.startswith(name + "("):          # a dataclass without its own docstring
                doc = "dataclass (fields: see the signature and the tables in this document)"
            doc = doc if len(doc) < 220 else doc[:217] + "..."
            callers = _callers(name, own)
            rows.append([f"`{name}{sig}`".replace("|", "/"), doc,
                         ", ".join(sorted(set(callers))[:6]) + (" ..." if len(set(callers)) > 6 else "")])
        out.append(table(["function / class", "purpose (first docstring paragraph)", "used in"], rows) if rows
                   else "_no public functions_")
        out.append("")
    return "\n".join(out)


def model_blocks():
    p = P_V
    vp = economy.vasicek_params(p)
    lp = checks.lambda_prime(DEFAULT.LAMBDA, DEFAULT.DISC_ER, DEFAULT.DISC_EMP, DEFAULT.T)
    return {
        "defaults": table(["symbol", "Params field", "default"],
                          [["$T$", "`T`", DEFAULT.T], ["$\\Gamma$", "`GAMMA`", DEFAULT.GAMMA],
                           ["$W$", "`W`", DEFAULT.W], ["$\\lambda$", "`LAMBDA`", DEFAULT.LAMBDA],
                           ["$\\eta$", "`ETA`", DEFAULT.ETA], ["$\\ddot a$", "`ANNUITY`", DEFAULT.ANNUITY],
                           ["$\\text{RR}_{legal}$", "`RR_LEGAL`", DEFAULT.RR_LEGAL],
                           ["$\\text{RR}^*$", "`RR_TARGET`", DEFAULT.RR_TARGET],
                           ["$\\mu$, $G$ (constant)", "`MU`, `G`", f"{DEFAULT.MU}, {DEFAULT.G}"],
                           ["$r$ (constant)", "`SHORT_RATE`", DEFAULT.SHORT_RATE],
                           ["$s$", "`FINANCING_SPREAD`", DEFAULT.FINANCING_SPREAD],
                           ["$\\sigma_R$, $\\sigma_L$", "`SIGMA_R`, `SIGMA_L`", f"{DEFAULT.SIGMA_R}, {DEFAULT.SIGMA_L}"],
                           ["$\\sigma_R$ (rate mode)", "`SIGMA_R_RATES`", DEFAULT.SIGMA_R_RATES]]),
        "lambda_prime": (f"With the legacy defaults ($\\lambda$ = {DEFAULT.LAMBDA}, $\\delta_f$ = {DEFAULT.DISC_ER}, "
                         f"$\\delta_e$ = {DEFAULT.DISC_EMP}, $T$ = {DEFAULT.T}) the inception-date objective "
                         f"equals the retirement-date one at $\\lambda'$ = {lp:.4f} with $r = \\delta_f$."),
        "vasicek_params": (f"Canonical vasicek: $\\kappa$ = {vp['kappa']:.4f}, $\\theta$ = {pct(vp['theta'])} "
                           f"(OLS; `LONG_RATE_P` = {p.LONG_RATE_P}), $\\sigma$ = {pct(vp['sigma'])}, "
                           f"$y_0$ = {pct(vp['y0'])}, spread = {vp['spread'] * 1e4:.1f} bp."),
    }


def architecture_blocks():
    p = P_V
    vp = economy.vasicek_params(p)
    cal = economy.rate_calibration()
    return {"calibration": (f"Data vintage: {len(cal['hist10Y'])} monthly 10Y observations up to "
                            f"{cal['t0']:%b %Y} (t0). Canonical vasicek: kappa {vp['kappa']:.4f}, theta "
                            f"{pct(vp['theta'])}, sigma {pct(vp['sigma'])}, y0 {pct(vp['y0'])}, spread "
                            f"{vp['spread'] * 1e4:.1f} bp.")}


# (file, code snippet that marks the stop, what to inspect, what to expect)
BREAKPOINTS = {
    "bp_constant": [
        ("pension/dp.py", "exo = Exogenous.draw(p, n_paths, seed, rates)",
         "`exo.zR[0]`, `exo.zL[0]`, `exo.u[0]`, `exo.r[0]`", "`exo.r` = `SHORT_RATE` everywhere; `exo.G is None` (vertical ledger)"),
        ("pension/dp.py", "a = np.clip(bilinear(Fg, lrg, policy[t], F, np.log(rho)), lo, hi)",
         "`t`, `F`, `rho`, `a`", "`F = R/L`, `rho = S/L`; a in the band"),
        ("pension/dp.py", "cost += np.where(present, obj.contribution(a, t, p) * numeraire.premium_factor(t, r_t, p), 0.0)",
         "`numeraire.premium_factor(t, r_t, p)`, `obj.contribution(a, t, p)`", "factor e^{(r+s)(T-t)} (t = 0: table below); reward of a = 1 at t = 0: table below"),
        ("pension/dynamics.py", "c = a * p.GAMMA * state.S", "`c`, `state.S`", "c = a Γ S; 0 for paths that have left"),
        ("pension/dynamics.py", "state.R = np.where(present, (state.R + c) * np.exp(mu + sig * exo.zR[t]), state.R * np.exp(mu))",
         "`state.R` before/after, `mu`, `sig`", "in force: (R + c) e^{μ + σ_R z_R}; paid-up: R e^{μ}"),
        ("pension/dynamics.py", "self.L = np.where(present, (self.L + c) * np.exp(p.G + p.SIGMA_L * exo.zL[t]), self.L)",
         "`self.L`", "(L + c) e^{G + σ_L z_L}; frozen when absent"),
        ("pension/dynamics.py", "lv = present & (exo.u[t] < hazard(t))", "`exo.u[t]`, `hazard(t)`, `lv`",
         "leaves iff u < h(t); `leave_t` becomes t + 1"),
        ("pension/dp.py", "end = settle(state, p)", "`state.F`, `state.rho`, `end`",
         "after one step: F' = dp.F_next(F, aΓρ, zR, zL, p), ρ' = dp.rho_next(...) (identity, table below); paid-up roll-forward F e^{μm}, ρ(1+W)^m"),
    ],
    "bp_vasicek": [
        ("pension/economy.py", 'r_m = y10_m - vp["spread"]', "`vp`, `y10_m[0]`, `r_m[0]`",
         "`y10_m[0]` = last observed 10Y; `r_m = y10_m - spread`"),
        ("pension/economy.py", "G_ = computeWAPRate(full, start, H, months_per_year=mpy)", "`G_[0]`, `G_[1]`",
         "G_0 = 2.50% (observed window); later years on the 25 bp grid in [1.75%, 3.75%]"),
        ("pension/economy.py", "acc_ = np.exp(np.add.reduceat(", "`acc_[0]`", "exp(Σ of the 12 monthly r·dt)"),
        ("pension/dynamics.py", "acc = np.asarray(rates[\"acc\"])[:p.T] if \"acc\" in rates else np.exp(r)",
         "`r[0]`, `acc[0]`, `G[0]`, `mu[0]`", "the scenario's annual r, acc, G, μ for these paths"),
        ("pension/dynamics.py", "self.Lv[t + 1] = c; self.lock[t + 1] = exo.G[t]", "`self.Lv[:t+2]`, `self.lock[:t+2]`",
         "a new vintage locked at G_t; every in-force vintage grows at its own lock"),
        ("pension/rates/accrual.py", 'y_t = r_t + vp["spread"]', "`r_t`, `y_t`, `mean`, `tau`",
         "A(0,T; r_0) (table below); = numeraire.premium_factor(0, r_0, p)"),
        ("pension/envs/pension_env.py", "return np.array([s.t / self.p.T, s.F[0], np.log(s.rho[0]), G_t, mu_t, r_t], dtype=np.float64)",
         "the 6 components", "r_t = y_t - spread; at reset: obs[5] = r_0 (table below)"),
        ("pension/envs/pension_env.py", "reward = float(-self.w_er * self.obj.contribution(a, t, p)[0]",
         "`reward`, `r_t`", "-(1-λ) a Γ (1+W)^{-(T-t)} A(t,T; r_t)"),
    ],
    "bp_flow": [
        ("pension/rates/data.py", 'dfYield = pd.read_csv(cache, parse_dates=["DATE"])', "`dfYield`, `cache`",
         "the cached NBB vintage (block `calibration` of ARCHITECTURE.md)"),
        ("pension/rates/calibration.py", "a, b   = coeffs", "`a`, `b`, `dt`", "kappa = -b/dt, theta = -a/b"),
        ("pension/economy.py", "_CALIBRATION = dict(", "`df10Y`, `short_label`", "computed once per process"),
        ("pension/economy.py", "return dict(kappa=vas[\"kappa\"], sigma=vas[\"sigma\"],", "`spread`, `p.LONG_RATE_P`",
         "theta = LONG_RATE_P if set; spread = historical mean (10Y - 1Y) if SPREAD_10Y_SHORT is None"),
        ("pension/rates/simulation.py", "paths[t+1, :] = paths[t, :] * e_kdt + drift + diff * eps[t, :]",
         "`paths[t]`, `eps[t]`", "exact OU step"),
        ("pension/rates/wap.py", "G[k] = wap_formula(y[lo:end + 1].mean(axis=0))", "`k`, `lo`, `end`",
         "window ends WAP_LAG months before the year start (REVIEW C1)"),
        ("pension/dp.py", "prem = numeraire.premium_schedule(p, rates)", "`ce`, `prem[:3]`",
         "CE moments and the scenario-mean premium factor (walkthrough block `ce`)"),
        ("pension/dp.py", "joint=float(w_emp * benefit_paths.mean() - w_er * cost.mean()),", "`benefit_paths`, `cost`",
         "joint = mean of the per-path env returns (contract C1)"),
    ],
}


def _line_of(rel, snippet):
    for i, line in enumerate(open(os.path.join(ROOT, rel)).read().splitlines(), 1):
        if snippet in line:
            return i
    raise AssertionError(f"breakpoint snippet not found in {rel}: {snippet!r} -- update BREAKPOINTS")


def breakpoint_tables():
    out = {}
    for key, rows in BREAKPOINTS.items():
        out[key] = table(["#", "where", "inspect", "expect"],
                         [[i + 1, f"`{f}:{_line_of(f, snip)}`", ins, exp]
                          for i, (f, snip, ins, exp) in enumerate(rows)])
    return out


def debugging_blocks():
    p = DEFAULT
    obj = dp._objective(None, p); w_er = obj.weights(p)[1]
    A0c = float(numeraire.premium_factor(0, p.SHORT_RATE, p))
    rew1 = -w_er * obj.contribution(np.array([1.0]), 0, p)[0] * A0c
    pv = P_V; vp = economy.vasicek_params(pv)
    r0 = vp["y0"] - vp["spread"]
    A0v = float(np.asarray(numeraire.premium_factor(0, r0, pv)).ravel()[0])
    rew1v = -w_er * obj.contribution(np.array([1.0]), 0, pv)[0] * A0v
    F, rho, a, zR, zL = 1.0, 20.0, 0.5, 0.3, -0.2
    l = a * p.GAMMA * rho
    Fn = float(dp.F_next(F, l, zR, zL, p)); rn = float(dp.rho_next(rho, l, zL, p))
    m = 40
    return {**breakpoint_tables(), "expect": table(["quantity", "expression", "value"], [
        ["year-0 premium factor, constant", "$A(0,T)=e^{(r+s)T}$", f"{A0c:.6f}"],
        ["year-0 premium reward at a = 1, constant", "$-(1-\\lambda)\\,\\Gamma(1+W)^{-T}A(0,T)$", f"{rew1:.6f}"],
        ["short rate at t0, vasicek", "$y_0-$spread", pct(r0, 4)],
        ["year-0 premium factor, vasicek", "closed form at $r_0$", f"{A0v:.6f}"],
        ["year-0 premium reward at a = 1, vasicek", "same with $A(0,T;r_0)$", f"{rew1v:.6f}"],
        ["$l$ at F=1, ρ=20, a=0.5", "$a\\Gamma\\rho$", f"{l:.4f}"],
        ["$F'$ at zR=0.3, zL=−0.2 (constant)", "$\\frac{F+l}{1+l}e^{(\\mu-G)+\\sigma_R z_R-\\sigma_L z_L}$", f"{Fn:.6f}"],
        ["$\\rho'$ (same)", "$\\frac{(1+W)\\rho}{(1+l)e^{G+\\sigma_L z_L}}$", f"{rn:.6f}"],
        ["paid-up roll-forward of F over 40 years", "$F\\,e^{\\mu m}$", f"{np.exp(p.MU * m):.6f}"],
        ["paid-up roll-forward of ρ over 40 years", "$\\rho\\,(1+W)^m$", f"{(1 + p.W) ** m:.6f}"],
    ])}


# =====================================================================================
# review -- the evidence behind each finding
# =====================================================================================
def review_blocks():
    import pension.rates.wap as wapmod
    blocks = {}
    p = P_V
    Fg, rg, ag = grids()
    n = 15000
    R0, L0, S0 = dp.new_plan_init(n, np.random.default_rng(7))

    # C1 WAP lag (analysis only: WAP_LAG is patched in this process and restored)
    base = economy._draw_rate_scenarios(2000, p.RATE_SEED, "vasicek", p.T, p)
    saved = wapmod.WAP_LAG
    try:
        wapmod.WAP_LAG = 16
        alt = economy._draw_rate_scenarios(2000, p.RATE_SEED, "vasicek", p.T, p)
    finally:
        wapmod.WAP_LAG = saved
    d = alt["G"] - base["G"]
    t0 = economy.rate_calibration()["t0"]
    blocks["wap_lag"] = "\n".join([
        f"- t0 = {t0:%b %Y}; model year 0 runs {t0:%b %Y} to {(t0 + pd.DateOffset(months=11)):%b %Y}. "
        f"WAP_LAG = {saved}: the window of model year k ends {saved} months before its start "
        f"(year 0: {(t0 - pd.DateOffset(months=saved)):%b %Y}).",
        f"- The rate legally in force at t0 is the calendar-{t0.year} rate, whose window ends in May "
        f"{t0.year - 1}: {12 + t0.month - 5} months before t0.",
        f"- On 2000 vasicek scenarios, the legal alignment (lag {12 + t0.month - 5}) changes G on "
        f"{(d != 0).mean():.1%} of path-years (mean |dG| where different: {np.abs(d[d != 0]).mean() * 1e4:.1f} bp); "
        f"year 1: {(d[1] != 0).mean():.0%} of paths, mean dG {d[1].mean() * 1e4:+.1f} bp; "
        f"year 2: {(d[2] != 0).mean():.0%}, {d[2].mean() * 1e4:+.1f} bp.",
    ])

    vp = economy.vasicek_params(p.replace(LONG_RATE_P=0.0225))
    blocks["long_rate"] = (f"LONG_RATE_P = 2.25% gives a long-run 10Y of 2.25% under vasicek, i.e. a long-run "
                           f"short rate of {pct(0.0225 - vp['spread'])}; under hull_white_p and vasicek_short the "
                           f"same value is the long-run SHORT rate. Within the horizon: e^(-kappa 10) = "
                           f"{np.exp(-vp['kappa'] * 10):.2f}, e^(-kappa 45) = {np.exp(-vp['kappa'] * 45):.3f} "
                           f"(vasicek kappa {vp['kappa']:.4f}).")

    pol = dp.const_policy(0.4, p.T, len(Fg), len(rg))
    r = dp.simulate(pol, Fg, rg, n_paths=200, rates=economy.draw_rate_scenarios(200, p=p), p=DEFAULT)
    blocks["mismatch"] = (f"dp.simulate(p=<constant>, rates=<vasicek scenario>) runs without complaint "
                          f"(joint {r['joint']:+.4f}): the dynamics follow the vasicek G_t, mu_t while the "
                          f"premiums accrue at SHORT_RATE.")

    sc = economy.draw_rate_scenarios(10, p=p)
    blocks["cache"] = (f"economy.draw_rate_scenarios returns the cached dict itself: same object on re-call = "
                       f"{sc is economy.draw_rate_scenarios(10, p=p)}; arrays writeable = {sc['G'].flags.writeable}.")

    rows = []
    Fn, rn = dp.make_F_grid(n=73), dp.make_rho_grid(n=71)
    e8 = dict(R0=R0[:8000], L0=L0[:8000], S0=S0[:8000])
    for lab, q in (("SHORT_RATE = MU", DEFAULT),
                   ("SHORT_RATE = MU + SIGMA_R^2/2", DEFAULT.replace(SHORT_RATE=DEFAULT.MU + DEFAULT.SIGMA_R**2 / 2))):
        # solved and simulated directly: checks.timing_neutrality resets SHORT_RATE to MU itself
        lo, hi = 0.02 / q.GAMMA, 0.15 / q.GAMMA
        pol = dp.solve(Fg=Fn, rg=rn, ag=np.linspace(lo, hi, 20), n_quad=5, p=q)["policy"]
        rr = dp.simulate(pol, Fn, rn, **e8, band=(lo, hi), n_paths=8000, seed=3, p=q)
        rows.append([lab, pct(q.SHORT_RATE, 3), f"{np.mean(rr['c_by'][:10]):.2f}%", f"{np.mean(rr['c_by'][35:]):.2f}%"])
    blocks["convexity"] = table(["anchor", "short rate", "early (years 0-9)", "late (35-44)"], rows)

    sol = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=5, p=p); ce = sol["ce"]
    prem = numeraire.premium_schedule(p, economy.draw_rate_scenarios(p.RATE_CE_PATHS, p=p))
    sol0 = dp._solve(Fg=Fg, rg=rg, ag=ag, n_quad=5, p=p.replace(**dict(ce, SIGMA_L=0.0)), prem=prem)
    scn = economy.draw_rate_scenarios(n, p=p)
    rws = []
    for lab, pp in (("CE as implemented", sol["policy"]), ("CE with SIGMA_L = 0", sol0["policy"])):
        rr = dp.simulate(pp, Fg, rg, R0=R0, L0=L0, S0=S0, n_paths=n, rates=scn, p=p)
        rws.append([lab, f"{rr['joint']:+.5f}", f"{rr['cost']:.4f}", f"{rr['sty']:.3f}", f"{rr['avg']:.2f}%"])
    blocks["ce"] = (f"CE moments: G {pct(ce['G'], 3)}, MU {pct(ce['MU'], 3)}, SIGMA_L {pct(ce['SIGMA_L'], 3)}, "
                    f"SIGMA_R {pct(ce['SIGMA_R'], 3)}.\n\n" +
                    table(["policy (scored on 15000 vasicek paths)", "joint", "cost", "stayer RR", "avg contribution"], rws))

    crow = []
    for lab, q in (("constant", DEFAULT), ("vasicek", p)):
        pol = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=5, p=q)["policy"]
        rates = None if q.RATE_MODEL == "constant" else economy.draw_rate_scenarios(n, p=q)
        exo = Exogenous.draw(q, n, 7, rates); s = State.initial(q, exo, R0, L0, S0)
        cnt = np.zeros(4); lrg = np.log(rg)
        for t in range(q.T):
            pr = s.present
            cnt += [(s.F[pr] > Fg[-1]).sum(), (s.rho[pr] < rg[0]).sum(), (s.rho[pr] > rg[-1]).sum(), pr.sum()]
            a = np.where(pr, np.clip(dp.bilinear(Fg, lrg, pol[t], s.F, np.log(s.rho)), 0, 1), 0.0)
            s, _ = step(s, a, exo, q, dp.tenure_hazard)
        crow.append([lab, f"{cnt[0] / cnt[3]:.4%}", f"{cnt[1] / cnt[3]:.4%}", f"{cnt[2] / cnt[3]:.4%}",
                     f"{np.mean(s.F > Fg[-1]):.1%}", f"{s.F.max():.2f}"])
    blocks["clipping"] = table(["model", "in-force F > 3", "ρ < 0.01", "ρ > 35",
                                "terminal F > 3 (all paths)", "max terminal F"], crow)
    vals = {}
    for nq in (3, 5, 7, 9):
        V = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=nq)["V"][0]
        vals[nq] = dp.bilinear(Fg, np.log(rg), V, R0 / L0, np.log(S0 / L0)).mean()
    blocks["quadrature"] = (", ".join(f"n_quad={k}: {v:.6f}" for k, v in vals.items()) +
                            f"; |V(5) - V(9)| / |V(9)| = {abs(vals[5] - vals[9]) / abs(vals[9]):.1e}.")

    e = checks.accrual_accuracy(p, n=20000, years=(0, 10, 30, 40, 44))
    blocks["accrual"] = table(["t"] + [str(t) for t in e],
                              [["rel. error (se), bp"] +
                               [f"{v['rel_err'] * 1e4:+.1f} ({v['rel_se'] * 1e4:.1f})" for v in e.values()]])
    blocks["lambda_prime"] = (f"lambda' at the legacy defaults (lambda {DEFAULT.LAMBDA}, DISC_ER {DEFAULT.DISC_ER}, "
                              f"DISC_EMP {DEFAULT.DISC_EMP}): "
                              f"{checks.lambda_prime(DEFAULT.LAMBDA, DEFAULT.DISC_ER, DEFAULT.DISC_EMP, DEFAULT.T):.4f}.")
    return blocks


PRODUCERS = {
    "walkthrough": walkthrough_blocks,
    "review": review_blocks,
    "architecture": architecture_blocks,
    "model": model_blocks,
    "reference": reference_blocks,
    "debugging": debugging_blocks,
}


def render(doc, blocks):
    def sub(m):
        name = m.group(1)
        if name not in blocks:
            raise KeyError(f"no generated block '{name}'")
        return f"<!-- gen:{name} -->\n{blocks[name]}\n<!-- /gen -->"
    return re.sub(r"<!-- gen:(\w+) -->.*?<!-- /gen -->", sub, doc, flags=re.S)


def main(argv):
    mode = "--write" if "--write" in argv else ("--check" if "--check" in argv else None)
    names = [a for a in argv if not a.startswith("--")] or list(DOCS)
    bad = []
    for name in names:
        blocks = PRODUCERS[name]()
        if mode is None:
            print(f"\n===== {name} =====")
            for k, text in blocks.items():
                print(f"\n<!-- gen:{k} -->\n{text}\n<!-- /gen -->")
            continue
        path = os.path.join(ROOT, DOCS[name])
        doc = open(path).read()
        new = render(doc, blocks)
        if mode == "--check":
            ok = new == doc
            print(f"{DOCS[name]}: {'up to date' if ok else 'OUT OF DATE'}")
            bad += [] if ok else [name]
        else:
            open(path, "w").write(new)
            print(f"wrote {len(blocks)} blocks into {DOCS[name]}")
    if bad:
        sys.exit(f"out of date: {', '.join(bad)} -- run with --write")


if __name__ == "__main__":
    main(sys.argv[1:])
