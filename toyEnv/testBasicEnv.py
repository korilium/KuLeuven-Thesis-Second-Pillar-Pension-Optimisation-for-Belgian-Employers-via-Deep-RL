"""Regression suite for the tabular MC pension toy.

Each named config is a COMPLETE economy. The suite applies it to the
basicEnv module namespace (where all functions read their globals),
re-draws the frozen shock batch, runs benchmark + agent + validation
per plan, and writes log/plots/arrays to tests/<config>/.

Usage:
    python testBasicEnv.py                  # run all configs
    python testBasicEnv.py rung1_stoch      # run one (or several) by name
"""

import os
import sys

import numpy as np

import basicEnv as env

# ---------------------------------------------------------------------------
# Named environment configurations
# G: frozen per config; becomes OLO-path-dependent at the Hull-White rung
# (enters via the state, not the config).
# ---------------------------------------------------------------------------
CONFIGS = {
    # deterministic rungs (frozen)
    "neg_control_det":   dict(G=0.0175, MU=0.03, DISC=0.03, KAPPA=0.5, SIGMA=0.0),
    "pos_control_det":   dict(G=0.0175, MU=0.03, DISC=0.01, KAPPA=1.5, SIGMA=0.0),
    # stochastic rung 1
    "neg_control_stoch": dict(G=0.0175, MU=0.01, DISC=0.01, KAPPA=1.0, SIGMA=0.05),
    #   ^ KAPPA=1: shortfall cancels PATH-WISE -> objective linear in c even
    #     under noise; validates stochastic machinery on a known-simple economy
    "rung1_stoch":       dict(G=0.0175, MU=0.03, DISC=0.01, KAPPA=1.5, SIGMA=0.05),
    #   ^ live-option configuration: THE rung-1 certificate
    "stress_G_stoch":    dict(G=0.0300, MU=0.03, DISC=0.01, KAPPA=1.5, SIGMA=0.05),
    #   ^ ln(1.03) ~ MU: drift advantage gone, shortfall ~at-the-money.
    #     Prediction: sparser benchmark policy than rung1_stoch; shortfall
    #     P(>0) in the tens of percent vs single digits at rung1.
}

# training budget per regime (deterministic converges much faster)
TRAIN = {
    0.0:  dict(n_episodes=200000, decay_frac=0.25, burn_in=30000, n0=50),
    0.05: dict(n_episodes=1000000, decay_frac=0.25, burn_in=60000, n0=500),
}


def apply_config(cfg):
    """Write the economy into basicEnv's namespace and re-draw the batch.

    All env functions (run_episode, run_batch, mc_control, ...) read
    module-level globals of basicEnv, so the update must land THERE,
    not in this test module."""
    env.__dict__.update(cfg)
    env.batch = env.draw_shock_batch()


def shortfall_diagnostics(bench_policy, plan):
    """How alive is the option? P(shortfall) and mean severity under the
    benchmark policy on the frozen batch. None-safe for SIGMA = 0."""
    if env.batch is None:
        return None
    n_paths = env.batch.shape[0]
    R = np.zeros(n_paths)
    L, S = 0.0, env.S0
    for t in range(env.T):
        c = plan(t, S) * bench_policy[t]
        R = (R + c) * np.exp(env.MU + env.SIGMA * env.batch[:, t])
        L = (L + c) * (1.0 + env.G)
        S *= (1.0 + env.W)
    sf = np.maximum(L - R, 0.0)
    p = float(np.mean(sf > 0))
    sev = float(sf[sf > 0].mean()) if (sf > 0).any() else 0.0
    return p, sev


def validate_agent(result, bench_policy, plan, train_cfg):
    """Decided-years criterion (step-3 logic as a function)."""
    if env.batch is None:
        gaps, ses = env.numeric_gap(plan, env.batch)
        RES = 0.01
    else:
        # decision-relevant gaps: flip each year of the BENCH policy on the
        # shared batch (CRN-paired); gap-vs-none invalid (max() nonlinearity)
        base_vals = env.run_batch(bench_policy, plan, env.batch)
        gaps, ses = np.empty(env.T), np.empty(env.T)
        for k in range(env.T):
            pol_k = bench_policy.copy()
            pol_k[k] ^= 1
            diffs = env.run_batch(pol_k, plan, env.batch) - base_vals
            s = 1 - 2 * bench_policy[k]      # sign: positive = contribute better
            gaps[k] = s * diffs.mean()
            ses[k] = diffs.std(ddof=1) / np.sqrt(len(diffs))
        # agent resolution: return noise / sqrt(non-greedy hold samples)
        sigma_G = np.std(result["reward_trace"][-50000:])
        n_ep = train_cfg["n_episodes"]
        n_hold = n_ep - int(train_cfg["decay_frac"] * n_ep) - train_cfg["burn_in"]
        n_ng = 0.5 * 0.05 * n_hold           # epsEnd/2 * hold episodes
        RES = 3.0 * sigma_G / np.sqrt(n_ng)

    floor = np.maximum(2 * ses, RES)
    decided = np.abs(gaps) > floor
    match = bool(np.array_equal(result["policy"][decided], bench_policy[decided]))
    mism = np.where((result["policy"] != bench_policy) & decided)[0]
    return {"gaps": gaps, "ses": ses, "RES": RES, "decided": decided,
            "match": match, "mismatch_years": mism}


def run_config(cfg_name, out_root="tests"):
    """Full pipeline for one named configuration. Returns True on PASS."""
    apply_config(CONFIGS[cfg_name])

    # tripwire: the config must actually have reached basicEnv
    for k, v in CONFIGS[cfg_name].items():
        assert getattr(env, k) == v, \
            f"{cfg_name}: {k} did not reach basicEnv — check apply_config"

    train_cfg = TRAIN[env.SIGMA]
    out_dir = os.path.join(out_root, cfg_name)
    os.makedirs(out_dir, exist_ok=True)

    lines = [f"config: {cfg_name}  {CONFIGS[cfg_name]}",
             f"train:  {train_cfg}"]

    def log(msg):
        print(msg)
        lines.append(str(msg))

    plans = {"fixed": env.plan_fixed(), "step": env.plan_step(), "age": env.plan_age()}
    all_match, table = True, {}

    for name, plan in plans.items():
        log(f"\n=== {name} plan ===")
        env.check_batch_consistency(plan)

        # 1. benchmark (layered at sigma=0, SAA local search at sigma>0)
        if env.SIGMA == 0.0:
            gb = env.gap_benchmark(plan)
            ls = env.local_search_benchmark(plan)
            v_gap, v_ls = gb["diagnostics"]["value"], ls["diagnostics"]["value"]
            if gb["parameters"]["linear"]:
                assert v_ls <= v_gap + 1e-10, \
                    f"{name}: local search beat certified-linear benchmark"
                bench_policy, bench_value = gb["arrays"]["policy"], v_gap
                log(f"benchmark: gap policy (linearity certified), value {bench_value:.6f}")
            else:
                bench_policy, bench_value = ls["arrays"]["policy"], v_ls
                log(f"benchmark: local search (linearity BROKEN), value {bench_value:.6f}")
        else:
            ls = env.local_search_benchmark(plan, batch=env.batch)
            bench_policy, bench_value = ls["arrays"]["policy"], ls["diagnostics"]["value"]
            log(f"benchmark: local search on SAA ({len(env.batch)} paths), "
                f"value {bench_value:.6f} (certificate: 2-flip optimality on the batch)")

        sf = shortfall_diagnostics(bench_policy, plan)
        if sf is not None:
            log(f"shortfall: P(>0) = {sf[0]:.4f}, E[sf | >0] = {sf[1]:.4f}")

        # 2. train
        result = env.mc_control(plan=plan, **train_cfg)

        # 3. validate
        v = validate_agent(result, bench_policy, plan, train_cfg)
        all_match &= v["match"]
        log(f"agent resolution RES = {v['RES']:.5f}")
        log(f"policy matches benchmark on {v['decided'].sum()}/{env.T} decided years: {v['match']}")
        if not v["match"]:
            log(f"  mismatch at decided years: {v['mismatch_years']}, "
                f"gaps there: {np.round(v['gaps'][v['mismatch_years']], 8)}")

        # 4. plot + persist arrays
        env.plot_results(result, path=os.path.join(out_dir, f"mc_{name}.png"))
        np.savez(os.path.join(out_dir, f"mc_{name}.npz"),
                 Q=result["Q"], policy=result["policy"],
                 bench_policy=bench_policy, gaps=v["gaps"], ses=v["ses"],
                 decided=v["decided"],
                 reward_trace=result["reward_trace"][::100])

        # 5. central-path economics of the benchmark policy
        table[name] = env.evaluate_policy(bench_policy, plan)

    cols = ["pv_contrib", "pv_payout", "efficiency", "duration", "replacement"]
    log("\n[central-path (zero-shock) economics of benchmark policies]")
    log(f"{'plan':<8}" + "".join(f"{c:>14}" for c in cols))
    for name, m in table.items():
        log(f"{name:<8}" + "".join(f"{m[c]:>14.4f}" for c in cols))
    log(f"\nCONFIG RESULT: {'PASS' if all_match else 'FAIL'}")

    with open(os.path.join(out_dir, "log.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return all_match


if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    unknown = [n for n in names if n not in CONFIGS]
    assert not unknown, f"unknown config(s): {unknown} — choose from {list(CONFIGS)}"

    results = {}
    for cfg_name in names:
        print(f"\n{'=' * 70}\nRUNNING: {cfg_name}\n{'=' * 70}")
        results[cfg_name] = run_config(cfg_name)

    print(f"\n{'=' * 70}\nSUITE SUMMARY\n{'=' * 70}")
    for cfg_name, ok in results.items():
        print(f"  {cfg_name:<20} {'PASS' if ok else 'FAIL'}")