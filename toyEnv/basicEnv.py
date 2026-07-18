import os
 
import numpy as np


"""Tabular Monte Carlo control on a deterministic pension contribution toy.

No neural networks, no libraries beyond numpy. One Q-table, one loop.

Environment
-----------
- T = 45 annual steps, binary action: contribute this year's premium or not.
- Premium is salary-linked: c_t = C_RATE * S_t, with S_t = S0 * (1+W)^t.
- Deterministic aggregate recursions (vintage-exact) under a frozen guarantee:
      R <- (R + c) * exp(MU)       reserve credited at the tariff MU
      L <- (L + c) * (1 + G)       WAP liability leg at the guarantee rate
- Rewards are emitted when cash flows occur, already discounted to t=0
  at rate DISC (one numeraire, owned by the environment):
      per year:  -KAPPA * c_t * exp(-DISC * t)
      terminal:  (max(R,L) - KAPPA * max(L-R, 0)) * exp(-DISC * T)
  KAPPA is the employer-cost weight; KAPPA = 1 makes the shortfall a pure
  transfer that cancels from the joint objective.
  Current simplification: DISC = MU (timing-neutral economy).

Agent
-----
- State: the year t only. Sufficient here because salary and rates are
  deterministic functions of time; must grow once anything stochastic
  enters (WAP rate, credited return).
- On-policy every-visit Monte Carlo control, epsilon-greedy with linear
  decay, gamma = 1 (financial discounting lives in the reward; gamma < 1
  would double-discount — do not tune).
- Update target is the return-from-t (reverse cumsum of rewards), so
  sunk costs never pollute later decisions' estimates.

Validation
----------
No closed forms: the benchmark evaluates the environment directly, so it
cannot go stale when run_episode changes.
- sweep_benchmark(): best switch policy by direct evaluation.
- numeric_gap(): per-year marginal value of contributing.
Pass criteria: learned policy matches the sweep's switch; MC Q-gaps track
numeric_gap(); reward trace plateaus at the benchmark value.
Caveats: single-episode evaluation and per-year decomposition are exact
only while the environment is deterministic and the reward is linear in
contributions — both need Monte Carlo averaging at the stochastic rung.
"""
 
# --------------------------------------------------------------------------- 
# Parameters
# ---------------------------------------------------------------------------
T = 45          # career length in years
G = 0.0175      # frozen WAP guarantee rate
MU = 0.01       # tarrif on mathematical reserve 
KAPPA = 0.5       # employer-cost weight (> 1, else the shortfall cancels) sensitivy 
S0 = 1.0        #starting salary
W = 0.025        # deterministic salary growth 
DISC= 0.01
SIGMA = 0.05   # credited-return volatility at zero
N_EVAL = 2000 
ALPHA= 0.02 #WEIGHT DECAY


#stochastic paths

def draw_shock_batch(n_paths=N_EVAL, seed=12345):
    """One frozen batch of noise paths (SAA + common random numbers).
    None at SIGMA = 0 -> deterministic single-episode evaluation."""
    if SIGMA == 0.0:
        return None
    return np.random.default_rng(seed).standard_normal((n_paths, T))
batch = draw_shock_batch()

def run_batch(policy, plan, shocks):
    """All paths at once for a FIXED policy (length-T 0/1 array).
    Same recursion as run_episode, vectorized over axis 0.
    shocks: (n_paths, T). Returns values: (n_paths,)."""
    n = shocks.shape[0]
    R = np.zeros(n)
    L = 0.0                                   # deterministic given the policy
    S = S0
    values = np.zeros(n)
    for t in range(T):
        c = plan(t, S) * policy[t]
        values -= KAPPA * c * np.exp(-DISC * t)
        R = (R + c) * np.exp(MU + SIGMA * shocks[:, t])   # (n,) vector op
        L = (L + c) * (1.0 + G)
        S *= (1.0 + W)
    payout = np.maximum(R, L)
    shortfall = np.maximum(L - R, 0.0)
    values += (payout - KAPPA * shortfall) * np.exp(-DISC * T)
    return values


# --- plan rules: c(t, S) -> premium ---
def plan_fixed(rate=0.05):
    return lambda t, S: rate * S

def plan_step(rate_low=0.04, rate_high=0.10, ceiling=1.5):
    # tranche-based, like real plans split around the pension ceiling
    return lambda t, S: rate_low * min(S, ceiling) + rate_high * max(S - ceiling, 0.0)

def plan_age(rate0=0.03, step=0.01, band=10):
    # banded age scale; keep step within the WAP non-discrimination bound
    return lambda t, S: (rate0 + step * (t // band)) * S


 # ---------------------------------------------------------------------------
# Environment: one function that plays a full episode given an action rule
# ---------------------------------------------------------------------------
 
def run_episode(choose_action, plan, shocks=None):
    """Play one career. choose_action(t) -> 0 or 1. Returns (actions, reward)."""
    R = L = 0.0
    actions = np.empty(T, dtype=np.int64)
    S = S0
    rewards = np.zeros(T) 
    for t in range(T):
        a = choose_action(t)
        actions[t] = a
        c = plan(t, S) * a 
        rewards[t] = -KAPPA * c * np.exp(-DISC * t) 
        z= 0.0 if shocks is None else shocks[t]
        R = (R + c) * np.exp(MU + SIGMA * z)
        L = (L + c) * (1.0 + G)
        S *= (1.0+ W)
    payout = max(R, L)
    shortfall = max(L - R, 0.0)
    rewards[-1] += (payout - KAPPA * shortfall)*np.exp(-DISC*T) 
    return actions, rewards


# ---------------------------------------------------------------------------
# Monte Carlo control
# ---------------------------------------------------------------------------
 
def mc_control(plan, n_episodes=500000, seed=0, epsStart =1.0, epsEnd =0.05, decay_frac=0.25, burn_in=30000, n0=500):
    rng = np.random.default_rng(seed)
    reward_trace = np.empty(n_episodes)
    
    decay_episodes = max(1, int(decay_frac * n_episodes))
 
    Q = np.zeros((T, 2))            # value estimates
    N = np.zeros((T, 2))            # hold-phase visit counts

    for ep in range(n_episodes):
        frac = min(ep / decay_episodes, 1.0)
        eps = epsStart + (epsEnd - epsStart) * frac

        # Restart the averaging window WITHOUT discarding the current Q:
        # N = 0 makes the first post-restart update jump Q to a single
        # episode's return (n=1), which flipped policies at the restart
        # and re-contaminated the averages (the ~80k disturbance in the
        # traces). N = n0 anchors on the existing estimate with prior
        # weight n0; the anchor's influence decays as n0/(n0+n) — under
        # 1% by end of hold — so the estimate is still asymptotically
        # a clean hold-phase average, just without the jump.
        if ep == decay_episodes or ep == decay_episodes + burn_in:
            N[:] = n0
 
        def choose_action(t):
            if rng.random() < eps:
                return int(rng.integers(2))          # explore
            return int(np.argmax(Q[t]))              # exploit
 
        shocks = rng.standard_normal(T) if SIGMA > 0 else None
        actions, rewards = run_episode(choose_action, plan, shocks)
        reward_trace[ep] = rewards.sum()   # G_0, comparable to before
        returns = np.cumsum(rewards[::-1])[::-1]  # G_t = Σ_{s≥t} r_s  (γ = 1) as already discounted with NPV 
        # Every-visit MC update: the terminal reward is the return at every t.
        for t in range(T):
            a = actions[t]
            if frac < 1.0:
                Q[t, a] += ALPHA * (returns[t] - Q[t, a])        # track
            else:
                N[t, a] += 1
                Q[t, a] += (returns[t] - Q[t, a]) / N[t, a]      # estimate
    
    return {
        "plan": plan,
        "Q": Q,
        "policy": np.argmax(Q, axis=1),
        "reward_trace": reward_trace,
    }

# ----------------------------------------------------------------------
# Environment-based benchmark (no learning) 
# ----------------------------------------------------------------------

def policy_value(policy_fn, plan, batch):
    if batch is None:
        _, r = run_episode(policy_fn, plan)
        return r.sum(), 0.0
    policy = np.array([policy_fn(t) for t in range(T)], dtype=np.int64)
    vals = run_batch(policy, plan, batch)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))

def sweep_benchmark(plan, batch=None, n_grid=T + 1):
    best = {"switch": 0, "value": -np.inf}
    for k in range(n_grid):
        value, _ = policy_value(lambda t, k=k: int(t < k), plan, batch)
        if value > best["value"]:
            best = {"switch": k, "value": value}
    return best

def check_batch_consistency(plan, n_check=5, seed=99):
    """run_batch must agree with run_episode path-by-path."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_check, T))
    policy = rng.integers(0, 2, T)
    vec = run_batch(policy, plan, shocks)
    for i in range(n_check):
        _, r = run_episode(lambda t: int(policy[t]), plan, shocks[i])
        assert abs(vec[i] - r.sum()) < 1e-10, f"path {i}: {vec[i]} vs {r.sum()}"

def numeric_gap(plan, batch=None):
    """Per-year marginal value of contributing (CRN-paired at the stochastic
    rung). NOTE: with SIGMA > 0 the max() terminal breaks linearity, so this
    is a diagnostic, not a certificate — the benchmark is local search."""
    v_none, _ = policy_value(lambda t: 0, plan, batch)
    gaps, ses = np.empty(T), np.zeros(T)
    for k in range(T):
        if batch is None:
            v_k, _ = policy_value(lambda t, k=k: int(t == k), plan, batch)
            gaps[k] = v_k - v_none
        else:
            if k == 0:
                vals0 = run_batch(np.zeros(T, dtype=np.int64), plan, batch)
            pol_k = np.zeros(T, dtype=np.int64); pol_k[k] = 1
            diffs = run_batch(pol_k, plan, batch) - vals0
            gaps[k] = diffs.mean()
            ses[k] = diffs.std(ddof=1) / np.sqrt(len(diffs))
    return gaps, ses

def gap_benchmark(plan, batch = None):
    """Certified optimal policy under linearity, with self-check.
    Environment-driven: every number comes from run_episode."""
    gaps, ses = numeric_gap(plan, batch)
    _, r_none = run_episode(lambda t: 0, plan)
    policy = (gaps > 0).astype(np.int64)

    # linearity assert: whole vs sum of parts
    value_parts = r_none.sum() + gaps[gaps > 0].sum()
    _, r_star = run_episode(lambda t: int(policy[t]), plan)
    value_whole = r_star.sum()
    linear = abs(value_whole - value_parts) < 1e-10

    return {
        "parameters": {"linear": linear},
        "diagnostics": {"value": value_whole, "value_parts": value_parts,
                        "value_none": r_none.sum()},
        "arrays": {"policy": policy, "gaps": gaps},
    }

def local_search_benchmark(plan, batch=None, extra_seeds=4, seed=0):
    """Optimal against all 1- and 2-year deviations. Assumes nothing
    about linearity or monotonicity. Successor benchmark for when
    gap_benchmark's linearity flag goes false."""
    rng = np.random.default_rng(seed)

    def value(pol):
        return policy_value(lambda t: int(pol[t]), plan, batch)[0]

    def ascend(pol):
        pol = pol.copy()
        v = value(pol)
        improved = True
        while improved:
            improved = False
            for t in range(T):                        # 1-flip pass
                pol[t] ^= 1
                v_new = value(pol)
                if v_new > v + 1e-12:
                    v, improved = v_new, True
                else:
                    pol[t] ^= 1
            if not improved:                          # 2-flip pass
                for i in range(T):
                    for j in range(i + 1, T):
                        pol[i] ^= 1; pol[j] ^= 1
                        v_new = value(pol)
                        if v_new > v + 1e-12:
                            v, improved = v_new, True
                        else:
                            pol[i] ^= 1; pol[j] ^= 1
        return pol, v
    gaps, _ = numeric_gap(plan, batch)
    seeds = [np.zeros(T, dtype=np.int64),
             np.ones(T, dtype=np.int64),
             (gaps > 0).astype(np.int64)]
    seeds += [rng.integers(0, 2, T) for _ in range(extra_seeds)]

    best_pol, best_v = None, -np.inf
    for s in seeds:
        pol, v = ascend(s)
        if v > best_v:
            best_pol, best_v = pol, v

    return {
        "parameters": {"n_seeds": len(seeds)},
        "diagnostics": {"value": best_v},
        "arrays": {"policy": best_pol},
    }

def evaluate_policy(policy, plan):
    """Replay a fixed policy and return unscaled economic metrics.
    policy: array of 0/1 actions of length T."""
    R = L = 0.0
    S = S0
    pv_contrib = 0.0
    t_weighted = 0.0
    for t in range(T):
        c = plan(t, S) * policy[t]
        d = c * np.exp(-DISC * t)
        pv_contrib += d
        t_weighted += t * d
        R = (R + c) * np.exp(MU)
        L = (L + c) * (1.0 + G)
        S *= (1.0 + W)
    payout = max(R, L)
    pv_payout = payout * np.exp(-DISC * T)
    return {
        "pv_contrib": pv_contrib,
        "pv_payout": pv_payout,
        "efficiency": pv_payout / pv_contrib if pv_contrib > 0 else np.nan,
        "duration": t_weighted / pv_contrib if pv_contrib > 0 else np.nan,
        "replacement": payout / (S0 * (1.0 + W) ** (T - 1)),
    }

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
 
def plot_results(result, path="tests/mc_basic.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plan, Q, policy, trace = result["plan"] , result["Q"], result["policy"], result["reward_trace"]
    ts = np.arange(T)
    gap_numeric, gap_se = numeric_gap(plan=plan, batch=batch)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bench = sweep_benchmark(plan=plan, batch=batch)
    # switch line only if the best policy is interior
    if bench is not None and 0 < bench["switch"] < T:
        for ax in (axes[0], axes[1]):
            ax.axvline(bench["switch"], ls="--", color="grey",
                   label=f"benchmark switch = {bench['switch']}")

    axes[0].plot(ts, gap_numeric, color="black", lw=1, label="numeric marginal gap")
    axes[0].plot(ts, Q[:, 1] - Q[:, 0], marker="o", label="MC estimate")
    axes[0].axhline(0.0, color="grey", lw=0.8)
    axes[0].legend()
    axes[0].fill_between(ts, gap_numeric - 2*gap_se, gap_numeric + 2*gap_se,
                         alpha=0.2, color="black")

    
    axes[1].step(ts, policy, where="mid")
    axes[1].set_yticks([0, 1], ["don't", "contribute"])
    axes[1].set_title("Learned greedy policy")
    axes[1].set_xlabel("year t")
 
    window = 2000
    smoothed = np.convolve(trace, np.ones(window) / window, mode="valid")
    axes[2].plot(smoothed)
    axes[2].set_title(f"Episode reward ({window}-episode moving average)")
    axes[2].set_xlabel("episode")
 
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")
 
 
if __name__ == "__main__":
    plans = {
        "fixed": plan_fixed(),
        "step":  plan_step(),
        "age":   plan_age(),
    }
    table = {}
    for name, plan in plans.items():
        print(f"\n=== {name} plan ===")
        check_batch_consistency(plan)
        # 1. layered benchmark with cross-check
        if SIGMA == 0.0:
            gb = gap_benchmark(plan)
            ls = local_search_benchmark(plan)
            v_gap, v_ls = gb["diagnostics"]["value"], ls["diagnostics"]["value"]
            if gb["parameters"]["linear"]:
                assert v_ls <= v_gap + 1e-10, \
                    f"{name}: local search beat a certified-linear benchmark — bug somewhere"
                bench_policy, bench_value = gb["arrays"]["policy"], v_gap
                print(f"benchmark: gap policy (linearity certified), value {bench_value:.6f}")
            else:
                bench_policy, bench_value = ls["arrays"]["policy"], v_ls
                print(f"benchmark: local search (linearity BROKEN), value {bench_value:.6f}")
        else:
            ls = local_search_benchmark(plan, batch=batch)
            bench_policy, bench_value = ls["arrays"]["policy"], ls["diagnostics"]["value"]
            print(f"benchmark: local search on SAA ({len(batch)} paths), "
                  f"value {bench_value:.6f} (certificate: 2-flip optimality on the batch)")

        # 2. train
        result = mc_control(plan=plan)

        # 3. per-plan validation before trusting metrics
        if batch is None:
            gaps, ses = numeric_gap(plan, batch)          # deterministic: exact, RES floor
            RES = 0.01
        else:
            # decision-relevant gaps at sigma > 0: flip each year of the
            # BENCH policy on the shared batch (CRN-paired). gap-vs-none is
            # invalid here — the max() terminal breaks linearity, so the
            # marginal depends on the rest of the policy.
            base_vals = run_batch(bench_policy, plan, batch)
            gaps, ses = np.empty(T), np.empty(T)
            for k in range(T):
                pol_k = bench_policy.copy(); pol_k[k] ^= 1
                diffs = run_batch(pol_k, plan, batch) - base_vals
                s = 1 - 2 * bench_policy[k]   # sign so positive = contribute better
                gaps[k] = s * diffs.mean()
                ses[k] = s * 0 + diffs.std(ddof=1) / np.sqrt(len(diffs))
            # agent resolution: return noise / sqrt(non-greedy hold samples)
            sigma_G = np.std(result["reward_trace"][-50000:])
            n_hold = 500000 - int(0.25 * 500000) - 30000   # n_episodes - decay - burn_in
            n_ng = 0.5 * 0.05 * n_hold      
            RES = 3.0 * sigma_G / np.sqrt(n_ng)

        floor = np.maximum(2 * ses, RES)
        decided = np.abs(gaps) > floor
        match = np.array_equal(result["policy"][decided], bench_policy[decided])
        print(f"policy matches benchmark on {decided.sum()}/{T} decided years: {match}")
        if not match:
            diff = np.where((result["policy"] != bench_policy) & decided)[0]
            print(f"  mismatch at decided years: {diff}, "
                  f"gaps there: {np.round(gaps[diff], 8)}")

        # 4. per-plan diagnostic plot
        plot_results(result, path=f"tests/mc_{name}.png")

        # 5. unscaled economics of the benchmark policy
        table[name] = evaluate_policy(bench_policy, plan)

    # comparison table
    cols = ["pv_contrib", "pv_payout", "efficiency", "duration", "replacement"]
    print(f"\n{'plan':<8}" + "".join(f"{c:>14}" for c in cols))
    for name, m in table.items():
        print(f"{name:<8}" + "".join(f"{m[c]:>14.4f}" for c in cols))
 


