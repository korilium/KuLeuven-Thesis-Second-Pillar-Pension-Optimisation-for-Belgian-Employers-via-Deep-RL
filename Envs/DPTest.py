"""
test_dp_oracle_crra.py -- test battery for the (t,F,rho,L) euro-CRRA DP oracle.

Three kinds of test:
  EXACT      -- machine-precision algebra (utility, quadrature, interpolation,
                terminal, and the transition formulas the solver actually uses,
                checked against a direct (R,L,S) rollout: the no-circular bridge).
  BEHAVIOUR  -- economic predictions (value monotone in F; CRRA yields an
                interior optimum; the B-limits bracket bang-bang; the optimum
                dominates any fixed plan; a* tapers with scale L).
  NUMERICAL  -- the grid oracle converges to independent Monte Carlo, and the
                error shrinks as the grid refines.

Run:  python test_dp_oracle_crra.py
"""
import time, traceback
import numpy as np
import dp_oracle_crra as dp

PASS, FAIL = [], []
def run(name, fn):
    t = time.time()
    try:
        fn(); print(f"  PASS   {name}   ({time.time()-t:.1f}s)"); PASS.append(name)
    except AssertionError as e:
        print(f"  FAIL   {name}: {e}   ({time.time()-t:.1f}s)"); FAIL.append((name, str(e)))
    except Exception as e:
        print(f"  ERROR  {name}: {e!r}   ({time.time()-t:.1f}s)"); FAIL.append((name, repr(e)))
        traceback.print_exc()

# small helper: coarse grids for fast optimize-based tests (F=1 on a node)
def coarse():
    return (dp.make_F_grid(n=31), dp.make_rho_grid(n=11), dp.make_L_grid(n=7), dp.make_a_grid(n=15))


# ------------------------------------------------------------------ EXACT
def test_crra_utility():
    for eta in (1.5, 2.0, 3.0):
        dp.ETA = eta
        xs = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        assert np.allclose(dp.u(xs), xs**(1-eta)/(1-eta)), "formula"
        assert np.all(np.diff(dp.u(xs)) > 0), "u must be strictly increasing"
        # strict concavity: midpoint utility above chord
        a, b = 1.0, 9.0
        assert dp.u(0.5*(a+b)) > 0.5*(dp.u(a)+dp.u(b)) + 1e-12, "u must be concave"
    dp.ETA = 2.0

def test_gauss_hermite_2d():
    zR, zL, wq = dp.gauss_hermite_2d(10)
    assert abs(wq.sum() - 1.0) < 1e-12, "weights sum to 1"
    assert abs((wq*zR).sum()) < 1e-10 and abs((wq*zL).sum()) < 1e-10, "means 0"
    assert abs((wq*zR**2).sum() - 1) < 1e-10 and abs((wq*zL**2).sum() - 1) < 1e-10, "variances 1"
    assert abs((wq*zR*zL).sum()) < 1e-10, "independence: cov 0"
    # MGF: E[e^z]=e^{1/2}. GH is exact only for polynomials, so a smooth-function
    # tolerance (not machine precision); 10 nodes drive it well below 1e-6.
    assert abs((wq*np.exp(zR)).sum() - np.exp(0.5)) < 1e-6, "E[e^z]=e^{1/2}"

def test_F_grid_node():
    g = dp.make_F_grid(n=61)
    assert 1.0 in g and abs(g[np.argmin(np.abs(g-1.0))]-1.0) < 1e-12, "F=1 on node"
    raised = False
    try: dp.make_F_grid(n=62)          # 61 intervals -> 1.0 not on a node
    except AssertionError: raised = True
    assert raised, "make_F_grid must reject grids without F=1 on a node"

def test_trilinear_at_nodes():
    Fg = np.linspace(0, 3, 7); lr = np.log(np.linspace(0.5, 4, 5)); lL = np.log(np.linspace(0.1, 5, 4))
    rng = np.random.default_rng(0); V = rng.standard_normal((7, 5, 4))
    # exact reproduction at every grid node
    for i in range(7):
        for j in range(5):
            for k in range(4):
                v = dp.trilinear(Fg, lr, lL, V, np.array(Fg[i]), np.array(lr[j]), np.array(lL[k]))
                assert abs(float(v) - V[i, j, k]) < 1e-12, "node reproduction"
    # midpoint along F-edge = average of the two F-neighbours
    Fmid = 0.5*(Fg[2]+Fg[3])
    v = float(dp.trilinear(Fg, lr, lL, V, np.array(Fmid), np.array(lr[1]), np.array(lL[1])))
    assert abs(v - 0.5*(V[2,1,1]+V[3,1,1])) < 1e-12, "edge midpoint linear"

def test_terminal_closed_form():
    dp.ETA, dp.B = 2.0, 10.0
    Fg = dp.make_F_grid(n=31); Lg = dp.make_L_grid(n=7)
    VT = dp.terminal(Fg, Lg)
    dT = np.exp(-dp.DISC*dp.T)
    for i, F in enumerate(Fg):
        for k, L in enumerate(Lg):
            emp = dp.LAMBDA*dp.B*dp.u(max(F,1.0)*L)
            empr = (1-dp.LAMBDA)*L*max(1.0-F, 0.0)
            assert abs(VT[i,k] - dT*(emp-empr)) < 1e-12, "terminal formula"

def test_transitions_match_rollout():
    """The no-circular bridge: F_next/rho_next/L_next reproduce a direct (R,L,S)
    step, F'=R'/L', rho'=S'/L', to machine precision."""
    rng = np.random.default_rng(7)
    for _ in range(2000):
        R, L, S = rng.uniform(0.1, 5), rng.uniform(0.1, 5), rng.uniform(0.1, 5)
        a = rng.uniform(0, 1); zR, zL = rng.standard_normal(2)
        c = a*dp.GAMMA*S
        Rn = (R+c)*np.exp(dp.MU + dp.SIGMA_R*zR)
        Ln = (L+c)*np.exp(dp.G + dp.SIGMA_L*zL)
        Sn = (1+dp.W)*S
        F, rho, l = R/L, S/L, c/L
        assert abs(dp.F_next(F, l, zR, zL)   - Rn/Ln) < 1e-11, "F' vs R'/L'"
        assert abs(dp.rho_next(rho, l, zL)   - Sn/Ln) < 1e-11, "rho' vs S'/L'"
        assert abs(dp.L_next(L, l, zL)       - Ln)    < 1e-11, "L' vs rollout"


# -------------------------------------------------------------- BEHAVIOUR
def test_value_monotone_in_F():
    """Terminal is non-decreasing in F and the dynamics preserve it -> V(.,F,.) up."""
    dp.SIGMA_R, dp.SIGMA_L, dp.ETA, dp.B = 0.05, 0.02, 2.0, 10.0
    Fg, rg, Lg, _ = coarse()
    V = dp.solve("evaluate", plan_rule=lambda t,F,r,L: 0.5, Fg=Fg, rg=rg, Lg=Lg, n_quad=3)["V"]
    d = np.diff(V[0], axis=0)                      # along F
    assert d.min() > -1e-6, f"V should be non-decreasing in F (min diff {d.min():.2e})"

def test_crra_interior_and_B_limits():
    """CRRA breaks the corner; the util-to-euro weight B brackets it:
       B->0 (cost only) => a*~0 ; B->inf (utility dominates) => a*~1 ; mid => interior."""
    dp.SIGMA_R, dp.SIGMA_L, dp.ETA = 0.05, 0.02, 2.0
    Fg, rg, Lg, ag = coarse()
    def mean_a(B):
        dp.B = float(B)
        return dp.solve("optimize", Fg=Fg, rg=rg, Lg=Lg, ag=ag, n_quad=2)["policy"].mean()
    lo, mid, hi = mean_a(1e-4), mean_a(10.0), mean_a(1e6)
    dp.B = 10.0
    # the bracketing itself is the claim: cost-only < interior < utility-dominated
    assert lo < mid < hi, f"B must bracket the optimum: {lo:.3f} < {mid:.3f} < {hi:.3f}"
    assert lo < 0.10, f"B->0 (cost only) should give a*~0, got {lo:.3f}"
    assert hi > 0.80, f"B->inf (utility dominated) should give a* high, got {hi:.3f}"
    assert 0.10 < mid < 0.90, f"moderate B should be interior, got {mid:.3f}"
    # and the moderate case is genuinely not bang-bang
    dp.B = 10.0
    pol = dp.solve("optimize", Fg=Fg, rg=rg, Lg=Lg, ag=ag, n_quad=2)["policy"]
    assert not np.all((pol == 0) | (pol == 1)), "moderate B must be non-degenerate"

def test_optimum_dominates_fixed_plan():
    """Optimize (a-grid includes 0.5) must dominate evaluate(a=0.5) cell-by-cell."""
    dp.SIGMA_R, dp.SIGMA_L, dp.ETA, dp.B = 0.05, 0.02, 2.0, 10.0
    Fg, rg, Lg, ag = coarse()
    assert 0.5 in ag, "test needs 0.5 on the action grid"
    Vopt = dp.solve("optimize", Fg=Fg, rg=rg, Lg=Lg, ag=ag, n_quad=2)["V"]
    Vev  = dp.solve("evaluate", plan_rule=lambda t,F,r,L: 0.5, Fg=Fg, rg=rg, Lg=Lg, n_quad=2)["V"]
    assert (Vopt - Vev).min() > -1e-9, f"optimum must dominate (min gap {(Vopt-Vev).min():.2e})"

def test_policy_tapers_with_scale():
    """a* is (weakly) larger at small scale L than at large L: fund fully when small."""
    dp.SIGMA_R, dp.SIGMA_L, dp.ETA, dp.B = 0.05, 0.02, 2.0, 10.0
    Fg, rg, Lg, ag = coarse()
    pol = dp.solve("optimize", Fg=Fg, rg=rg, Lg=Lg, ag=ag, n_quad=2)["policy"]
    mid = dp.T // 2
    a_small = pol[mid, :, :, 0].mean()             # smallest-L slice
    a_large = pol[mid, :, :, -1].mean()            # largest-L slice
    assert a_small >= a_large - 1e-9, f"a* should not increase with scale ({a_small:.3f} vs {a_large:.3f})"
    assert a_small - a_large > 0.05, "expect a visible taper across the L range"


# --------------------------------------------------------------- NUMERICAL
def test_evaluate_matches_mc_mild_eta():
    """With mild curvature (eta=1.3) discretization is small; evaluate ~ MC."""
    dp.SIGMA_R, dp.SIGMA_L, dp.ETA, dp.B = 0.05, 0.02, 1.3, 10.0
    Fg = dp.make_F_grid(n=151); rg = dp.make_rho_grid(n=41); Lg = dp.make_L_grid(n=25)
    ev = dp.solve("evaluate", plan_rule=lambda t,F,r,L: 0.5, Fg=Fg, rg=rg, Lg=Lg, n_quad=5)
    t0, R0, L0, Sv = 20, 0.9, 1.0, 3.0
    vo = float(dp.trilinear(Fg, np.log(rg), np.log(Lg), ev["V"][t0],
                            np.array(R0/L0), np.array(np.log(Sv/L0)), np.array(np.log(L0))))
    vm, se = dp.mc_value(t0, R0, L0, Sv, a_const=0.5, n_paths=300000)
    rel = abs(vo - vm) / abs(vm)
    dp.ETA = 2.0
    assert rel < 0.06, f"evaluate vs MC rel.err {rel:.3f} exceeds 6% at eta=1.3/151grid"

def test_field_action_exact():
    """EXACT proof that the evaluate mode's per-cell action path (bellman_field_a)
    computes the same MDP as the optimize sweep's scalar path (bellman_scalar_a).
    Optimize over the single-action grid {0.5} isolates the scalar path at a=0.5;
    evaluate with the constant rule 0.5 runs the field path. They must be bit-identical
    -- this validates the state-dependent capability without any grid/MC tolerance,
    since a per-cell field just feeds different values into the same proven expression."""
    dp.SIGMA_R, dp.SIGMA_L, dp.ETA, dp.B = 0.05, 0.02, 2.0, 10.0
    Fg = dp.make_F_grid(n=31); rg = dp.make_rho_grid(n=11); Lg = dp.make_L_grid(n=7)
    Vscalar = dp.solve("optimize", Fg=Fg, rg=rg, Lg=Lg, ag=np.array([0.5]), n_quad=3)["V"]
    Vfield  = dp.solve("evaluate", plan_rule=lambda t, F, r, L: 0.5, Fg=Fg, rg=rg, Lg=Lg, n_quad=3)["V"]
    d = float(np.max(np.abs(Vscalar - Vfield)))
    assert d < 1e-12, f"field-action path diverges from scalar path by {d:.2e} (broadcasting bug)"


def test_grid_convergence():
    """Refining the grid must reduce the evaluate-vs-MC error (eta=2)."""
    dp.SIGMA_R, dp.SIGMA_L, dp.ETA, dp.B = 0.05, 0.02, 2.0, 10.0
    t0, R0, L0, Sv = 20, 0.9, 1.0, 3.0
    vm, se = dp.mc_value(t0, R0, L0, Sv, a_const=0.5, n_paths=400000)
    errs = []
    for (NF, NR, NL) in [(61, 21, 13), (151, 41, 25)]:
        Fg = dp.make_F_grid(n=NF); rg = dp.make_rho_grid(n=NR); Lg = dp.make_L_grid(n=NL)
        ev = dp.solve("evaluate", plan_rule=lambda t,F,r,L: 0.5, Fg=Fg, rg=rg, Lg=Lg, n_quad=5)
        vo = float(dp.trilinear(Fg, np.log(rg), np.log(Lg), ev["V"][t0],
                                np.array(R0/L0), np.array(np.log(Sv/L0)), np.array(np.log(L0))))
        errs.append(abs(vo - vm)/abs(vm))
    assert errs[1] < errs[0], f"error must shrink on refinement ({errs[0]:.3f} -> {errs[1]:.3f})"
    assert errs[1] < 0.5*errs[0] + 0.03, "expect roughly halving per doubling"


if __name__ == "__main__":
    print("EXACT")
    run("crra_utility", test_crra_utility)
    run("gauss_hermite_2d", test_gauss_hermite_2d)
    run("F_grid_node", test_F_grid_node)
    run("trilinear_at_nodes", test_trilinear_at_nodes)
    run("terminal_closed_form", test_terminal_closed_form)
    run("transitions_match_rollout", test_transitions_match_rollout)
    print("BEHAVIOUR")
    run("value_monotone_in_F", test_value_monotone_in_F)
    run("crra_interior_and_B_limits", test_crra_interior_and_B_limits)
    run("optimum_dominates_fixed_plan", test_optimum_dominates_fixed_plan)
    run("policy_tapers_with_scale", test_policy_tapers_with_scale)
    print("NUMERICAL")
    run("evaluate_matches_mc_mild_eta", test_evaluate_matches_mc_mild_eta)
    run("field_action_exact", test_field_action_exact)
    run("grid_convergence", test_grid_convergence)

    print(f"\n{'='*50}\n{len(PASS)} passed, {len(FAIL)} failed")
    for n, e in FAIL: print(f"  FAILED: {n}: {e}")
    raise SystemExit(1 if FAIL else 0)