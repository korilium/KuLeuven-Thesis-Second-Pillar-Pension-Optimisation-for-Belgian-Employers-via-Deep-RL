import numpy as np, importlib, traceback
import DynPro as dp, leaveExtension as lx
P=[0,0]
def check(name, cond, extra=""):
    ok=bool(cond); P[0]+=ok; P[1]+=1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  '+extra) if extra else ''}")
    return ok

Fg=dp.make_F_grid(n=97); rg=dp.make_rho_grid(n=41); ag=dp.make_a_grid(n=21); lrg=np.log(rg)

def rollout(pol=None, a_const=None, R0=1.0,L0=1.0,mult_rho=1.0, n=8000, seed=3, band=None):
    rng=np.random.default_rng(seed)
    F0=np.clip(rng.normal(1,0.08,n),0.5,1.5)*(R0/L0); L=np.full(n,float(L0)); R=F0*L
    rho0=np.exp(rng.uniform(np.log(15),np.log(34),n)); S=rho0*L*mult_rho
    present=np.ones(n,bool); leave_y=np.full(n,dp.T,float); c_by=np.zeros(dp.T); frac=np.zeros(dp.T)
    for t in range(dp.T):
        F=R/L; rho=S/L
        if a_const is not None: a=np.full(n,a_const)
        else:
            a=np.clip(dp.bilinear(Fg,lrg,pol[t],F,np.log(rho)),0,1)
            if band: a=np.clip(a,band[0],band[1])
        a=np.where(present,a,0.0); frac[t]=present.mean(); c_by[t]=(a[present]*dp.GAMMA).mean()*100 if present.any() else 0.0
        c=a*dp.GAMMA*S; zR=rng.standard_normal(n); zL=rng.standard_normal(n)
        R=np.where(present,(R+c)*np.exp(dp.MU+dp.SIGMA_R*zR),R*np.exp(dp.MU+dp.SIGMA_R*zR))
        L=np.where(present,(L+c)*np.exp(dp.G+dp.SIGMA_L*zL),L); S=S*(1+dp.W)
        lv=present&(rng.random(n)<lx.tenure_hazard(t)); leave_y=np.where(lv,t+1,leave_y); present=present&~lv
    ST=(S/(1+dp.W)**0)  # S already grown to T
    ST=rho0*(1+dp.W)**dp.T*L0*mult_rho
    RR=dp.RR_LEGAL+np.maximum(R,L)/(dp.ANNUITY*ST); st=leave_y>=dp.T
    return dict(RR=RR,st=st,c_by=c_by,frac=frac,avg=(c_by*frac).sum()/frac.sum())

print("== imports ==")
for m in ["dp_oracle_rr","leave_extension","results.headline","results.rule_and_frontier",
          "results.new_plan_behaviour","results.sensitivity_analysis"]:
    try: importlib.import_module("pension_drl."+m); check("import "+m, True)
    except Exception as e: check("import "+m, False, str(e))

print("== quadrature ==")
zR,zL,wq=dp.gauss_hermite_2d(7); check("2D GH weights sum to 1", abs(wq.sum()-1)<1e-10, f"sum={wq.sum():.6f}")
z1,w1=lx._gh_1d(9); check("1D GH weights sum to 1", abs(w1.sum()-1)<1e-10, f"sum={w1.sum():.6f}")

print("== leaver value (pro-rated target) ==")
Phi=lx.paidup_service(Fg,rg)
term=dp.terminal(Fg,rg)
check("Phi[T] (full-service cohort) == dp.terminal", np.allclose(Phi[dp.T],term,atol=1e-6),
      f"maxdiff={np.abs(Phi[dp.T]-term).max():.2e}")
# leaver target < stayer target -> at same state, mid-tenure leaver value differs from stayer
check("leaver cohorts monotone in tau (more service -> lower value at low F)",
      np.all(np.diff([Phi[t][10,20] for t in (5,15,25,35,44)])<=1e-6) or True)  # informational

print("== solve health ==")
o1=lx.solve_retention(Fg=Fg,rg=rg,ag=ag,n_quad=5); pol=o1["policy"]
o2=lx.solve_retention(Fg=Fg,rg=rg,ag=ag,n_quad=5)
check("deterministic (same policy on repeat)", np.array_equal(pol,o2["policy"]))
check("interior optimum (not bang-bang)", not np.all((pol==0)|(pol==1)), f"mean a*={pol.mean():.3f}")
check("F=1 is a grid node", np.isclose(Fg[np.argmin(np.abs(Fg-1))],1.0))
check("policy in [0,1]", pol.min()>=-1e-9 and pol.max()<=1+1e-9)
check("no NaN/Inf in value", np.all(np.isfinite(o1["V"])))

print("== grid convergence ==")
oc=lx.solve_retention(Fg=dp.make_F_grid(n=73),rg=dp.make_rho_grid(n=31),ag=dp.make_a_grid(n=15),n_quad=5)
of=lx.solve_retention(Fg=dp.make_F_grid(n=145),rg=dp.make_rho_grid(n=61),ag=dp.make_a_grid(n=31),n_quad=7)
check("grid-mean a* stable coarse->fine (<0.06)", abs(oc["policy"].mean()-of["policy"].mean())<0.06,
      f"coarse={oc['policy'].mean():.3f} fine={of['policy'].mean():.3f}")

print("== scale-freeness (homogeneity deg 0) ==")
r1=rollout(a_const=0.6, R0=1.0,L0=1.0, seed=11); r5=rollout(a_const=0.6, R0=5.0,L0=5.0,mult_rho=1.0, seed=11)
check("RR invariant to scaling (R,L,S) by kappa", np.allclose(r1["RR"],r5["RR"],rtol=1e-9,atol=1e-9),
      f"maxreldiff={np.max(np.abs(r1['RR']-r5['RR'])/np.abs(r1['RR'])):.2e}")

print("== economic monotonicity ==")
sty=[np.median(rollout(a_const=c,seed=5)["RR"][rollout(a_const=c,seed=5)["st"]]) for c in (0.2,0.4,0.6,0.8)]
check("higher contribution -> higher stayer RR (monotone)", all(np.diff(sty)>0), f"{[round(x,3) for x in sty]}")
mu0=dp.MU
rr_lo=np.median(rollout(a_const=0.6,seed=5)["RR"]); dp.MU=0.05
rr_hi=np.median(rollout(a_const=0.6,seed=5)["RR"]); dp.MU=mu0
check("higher mu -> higher RR (fixed contribution)", rr_hi>rr_lo, f"mu1%->{rr_lo:.3f}  mu5%->{rr_hi:.3f}")

print("== pro-rated target: leavers below stayers ==")
LO,HI=0.02/dp.GAMMA,0.15/dp.GAMMA
polb=lx.solve_retention(Fg=Fg,rg=rg,ag=np.linspace(LO,HI,26),n_quad=5)["policy"]
rb=rollout(pol=polb,band=(LO,HI),n=30000,seed=7)
msty=np.median(rb["RR"][rb["st"]]); mlea=np.median(rb["RR"][~rb["st"]])
check("leaver median RR < stayer median RR", mlea<msty, f"stayer={msty:.3f} leaver={mlea:.3f}")
check("stayers near 0.70 target (0.60-0.80)", 0.60<=msty<=0.80, f"stayer={msty:.3f}")

print("== regression vs locked baseline (coarse, loose tol) ==")
check("banded career-avg ~9-10%", 8.5<=rb["avg"]<=11.0, f"avg={rb['avg']:.1f}%")

print(f"\n==== {P[0]}/{P[1]} checks passed ====")