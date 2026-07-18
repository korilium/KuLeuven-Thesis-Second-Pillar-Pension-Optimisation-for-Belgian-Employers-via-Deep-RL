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
MU = 0.03       # tarrif on mathematical reserve 
KAPPA = 0.5       # employer-cost weight (> 1, else the shortfall cancels) sensitivy 
S0 = 1.0        #starting salary
W = 0.02        # deterministic salary growth 
DISC= MU

ALPHA= 0.02 #WEIGHT DECAY


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
 
def run_episode(choose_action, plan):
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
        R = (R + c) * np.exp(MU)
        L = (L + c) * (1.0 + G)
        S *= (1.0+ W)
    payout = max(R, L)
    shortfall = max(L - R, 0.0)
    rewards[-1] += (payout - KAPPA * shortfall)*np.exp(-DISC*T) 
    return actions, rewards


# ---------------------------------------------------------------------------
# Monte Carlo control
# ---------------------------------------------------------------------------
 
def mc_control(plan, n_episodes=100000, seed=0, epsStart =1.0, epsEnd =0.05, decay_frac=0.5):
    rng = np.random.default_rng(seed)
    Q = np.zeros((T, 2))            # value estimates
    reward_trace = np.empty(n_episodes)
    
    decay_episodes = max(1, int(decay_frac * n_episodes))
 
    for ep in range(n_episodes):
        frac = min(ep / decay_episodes, 1.0)      # reach epsEnd at 50k, then hold
        eps = epsStart + (epsEnd - epsStart) * frac
 
        def choose_action(t):
            if rng.random() < eps:
                return int(rng.integers(2))          # explore
            return int(np.argmax(Q[t]))              # exploit
 
        actions, rewards = run_episode(choose_action, plan)
        reward_trace[ep] = rewards.sum()   # G_0, comparable to before
        returns = np.cumsum(rewards[::-1])[::-1]  # G_t = Σ_{s≥t} r_s  (γ = 1) as already discounted with NPV 
        # Every-visit MC update: the terminal reward is the return at every t.
        for t in range(T):
            a = actions[t]
            Q[t, a] += ALPHA * (returns[t] - Q[t, a]) #apply weight decay 
    
    return {
        "plan": plan,
        "Q": Q,
        "policy": np.argmax(Q, axis=1),
        "reward_trace": reward_trace,
    }

# ----------------------------------------------------------------------
# Environment-based benchmark (no learning) 
# ----------------------------------------------------------------------

def sweep_benchmark(plan,n_grid=T + 1):
    """Best switch policy by direct evaluation — no learning, no formulas.
    Automatically stays correct under any change to run_episode."""
    best = {"switch": 0, "value": -np.inf}
    for k in range(n_grid):                    # contribute years 0..k-1
        actions, rewards = run_episode(lambda t: int(t < k), plan)
        value = rewards.sum()                  # deterministic env: one episode suffices
        if value > best["value"]:
            best = {"switch": k, "value": value}
    return best


def numeric_gap(plan):
    """Marginal value of contributing in each single year, measured directly
    from the environment. No formulas: stays correct under env changes.
    (Exact per-year decomposition only while the env is deterministic and
    the reward linear in contributions — revisit at the stochastic rung.)"""
    _, r_none = run_episode(lambda t: 0, plan)
    base = r_none.sum()
    gaps = np.empty(T)
    for k in range(T):
        _, r = run_episode(lambda t, k=k: int(t == k), plan)
        gaps[k] = r.sum() - base
    return gaps

def gap_benchmark(plan):
    """Certified optimal policy under linearity, with self-check.
    Environment-driven: every number comes from run_episode."""
    gaps = numeric_gap(plan)
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

def local_search_benchmark(plan, extra_seeds=4, seed=0):
    """Optimal against all 1- and 2-year deviations. Assumes nothing
    about linearity or monotonicity. Successor benchmark for when
    gap_benchmark's linearity flag goes false."""
    rng = np.random.default_rng(seed)

    def value(pol):
        _, r = run_episode(lambda t: int(pol[t]), plan)
        return r.sum()

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

    seeds = [np.zeros(T, dtype=np.int64),
             np.ones(T, dtype=np.int64),
             (numeric_gap(plan) > 0).astype(np.int64)]
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
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bench = sweep_benchmark(plan=plan)
    # switch line only if the best policy is interior
    if bench is not None and 0 < bench["switch"] < T:
        for ax in (axes[0], axes[1]):
            ax.axvline(bench["switch"], ls="--", color="grey",
                   label=f"benchmark switch = {bench['switch']}")
    gap_numeric = numeric_gap(plan=plan)

    axes[0].plot(ts, gap_numeric, color="black", lw=1, label="numeric marginal gap")
    axes[0].plot(ts, Q[:, 1] - Q[:, 0], marker="o", label="MC estimate")
    axes[0].axhline(0.0, color="grey", lw=0.8)
    axes[0].legend()

    
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

        # 1. layered benchmark with cross-check
        gb = gap_benchmark(plan)
        ls = local_search_benchmark(plan)
        v_gap, v_ls = gb["diagnostics"]["value"], ls["diagnostics"]["value"]

        if gb["parameters"]["linear"]:
            # linearity certified -> gap policy is globally optimal;
            # local search must not beat it, or one of the two is buggy
            assert v_ls <= v_gap + 1e-10, \
                f"{name}: local search beat a certified-linear benchmark — bug somewhere"
            bench_policy, bench_value = gb["arrays"]["policy"], v_gap
            print(f"benchmark: gap policy (linearity certified), value {bench_value:.6f}")
        else:
            # linearity broken -> fall through to the assumption-free layer
            bench_policy, bench_value = ls["arrays"]["policy"], v_ls
            print(f"benchmark: local search (linearity BROKEN), value {bench_value:.6f}")
            print("  -> gap decomposition invalid at this configuration; "
                  "certificate is 2-flip local optimality only")

        # 2. train
        result = mc_control(plan=plan)

        # 3. per-plan validation before trusting metrics
        match = np.array_equal(result["policy"], bench_policy)
        print(f"learned policy matches benchmark: {match}")
        if not match:
            diff = np.where(result["policy"] != bench_policy)[0]
            print(f"  mismatch at years: {diff}, "
                  f"gaps there: {np.round(gb['arrays']['gaps'][diff], 8)}")

        # 4. per-plan diagnostic plot
        plot_results(result, path=f"tests/mc_{name}.png")

        # 5. unscaled economics of the benchmark policy
        table[name] = evaluate_policy(bench_policy, plan)

    # comparison table
    cols = ["pv_contrib", "pv_payout", "efficiency", "duration", "replacement"]
    print(f"\n{'plan':<8}" + "".join(f"{c:>14}" for c in cols))
    for name, m in table.items():
        print(f"{name:<8}" + "".join(f"{m[c]:>14.4f}" for c in cols))
 


