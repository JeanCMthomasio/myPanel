"""Validação das superfícies de comando: rotação das normais em torno da
linha de charneira (Drela), circulação/forças via coefficientsAt, derivadas
por AD via controlDerivatives. Seis verificações, no padrão de study.py.
"""
import matplotlib; matplotlib.use('Agg')
import json, numpy as onp, jax, jax.numpy as np
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from main import Surface, Aircraft

R = {}


# ---------------------------------------------------------------- 1. regressao
# a asa do __main__, com aileron+flap, mas isolada (sem winglets/cauda) para o
# resto do script: os testes de simetria abaixo precisam de uma aeronave onde
# eu conheço a geometria de cabo a rabo.
def makeAircraft():
    wing = Surface(span=[0.,1.,3.5], chord=[1.75,1.,.5], sweep=[10.,20.], dihedral=[1.,2.],
                   position=[0.,0.,0.], discretization=[(5,11),(5,31)], symmetry=True,
                   control_surfaces=[dict(name='aileron', hinge=0.75, span=(0.55,0.95), mode='antisymmetric'),
                                      dict(name='flap', hinge=0.70, span=(0.05,0.50), mode='symmetric')])
    hTail = Surface(span=[0.,1.], chord=[.6,.2], sweep=[15.], dihedral=[0.], position=[-4.,0.,-1.],
                    discretization=[(10,25)], symmetry=True)
    vTail = Surface(span=[0.,.75], chord=[.6,.2], sweep=[5.], dihedral=[90.], position=[-4.,0.,-1.],
                    discretization=[(10,25)], symmetry=False)
    return Aircraft(surfaces=[wing, hTail, vTail], CG=np.array([-1.,0.,0.]))

ac = makeAircraft()
old = [float(v) for v in ac.simulate(1.0, 5.0, 0.0)]
new = [float(v) for v in ac.coefficientsAt(5.0, 0.0, np.zeros(len(ac.control_names)))]
R['regression'] = dict(names=['CD_i','CY','CL','Cl','Cm','Cn'], old=old, new=new,
                        max_abs_diff=max(abs(o-n) for o, n in zip(old, new)))


# ---------------------------------------------------------- 2. aileron sozinho
# The mirror-symmetry argument (not the looser "Cl != 0, everything else ~0"
# guess the plan started from): mirroring the aircraft (Y -> -Y) maps an
# antisymmetric-mode deflection to its own negative, delta -> -delta. Forces
# are true vectors (Y-component flips under the mirror), moments are
# pseudovectors (the components PERPENDICULAR to the mirror axis flip, the
# one ALONG it doesn't). Combined: CY, Cl, Cn are ODD in delta -> free to be
# nonzero, linear leading order, Cl dominant since that's the coupling the
# aileron is built for; CL, CD_i, Cm are EVEN in delta -> exactly zero at
# delta=0 and delta=-delta, so only a small O(delta^2) residual is expected,
# not literally zero at finite delta.
ac = makeAircraft()
i_ail = ac.control_names.index('aileron')
deltas = np.zeros(len(ac.control_names)).at[i_ail].set(5.0)
c = [float(v) for v in ac.coefficientsAt(0.0, 0.0, deltas)]
R['aileron'] = dict(names=['CD_i','CY','CL','Cl','Cm','Cn'], values=c, delta_deg=5.0,
                     even=['CL','CD_i','Cm'], odd=['CY','Cl','Cn'])


# -------------------------------------------------------------------- 3. leme
# A single fin sitting exactly on the Y=0 plane (dihedral=90, symmetry=False)
# is itself mirror-invariant as a SHAPE, but deflecting it swings material to
# one side, so mirror(state(delta)) = state(-delta) here too, same even/odd
# split as the aileron: CY/Cn should dominate (that's the rudder's job), Cl
# is allowed but should be a smaller secondary coupling (the fin's side force
# acts off the roll axis), CL/CD_i/Cm should be near zero.
def rudderAircraft():
    fin = Surface(span=[0.,1.5], chord=[.8,.4], sweep=[10.], dihedral=[90.], position=[-3.,0.,-0.5],
                  discretization=[(8,17)], symmetry=False,
                  control_surfaces=[dict(name='rudder', hinge=0.75, span=(0.1,0.95), mode='symmetric')])
    wing = Surface(span=[0.,4.], chord=[1.,1.], sweep=[0.], dihedral=[0.], position=[0.,0.,0.],
                   discretization=[(6,25)], symmetry=True)
    return Aircraft(surfaces=[wing, fin], CG=np.array([-1.,0.,0.]))

acr = rudderAircraft()
deltas = np.array([10.0])
c = [float(v) for v in acr.coefficientsAt(0.0, 0.0, deltas)]
R['rudder'] = dict(names=['CD_i','CY','CL','Cl','Cm','Cn'], values=c, delta_deg=10.0)


# ------------------------------------------------ 4/6. efetividade de flap
# asa reta, AR alto, flap de envergadura cheia em x/c=0.75 -> teoria de perfil
# fino: dCL/ddelta / dCL/dalpha = (pi - theta_h + sin theta_h)/pi,
# cos(theta_h) = 1 - 2*(x_h/c)
def flapWing(n, m=41):
    return Aircraft(surfaces=[Surface(span=[0.,6.], chord=[1.,1.], sweep=[0.], dihedral=[0.],
                    position=[0.,0.,0.], discretization=[(n,m)], symmetry=True,
                    control_surfaces=[dict(name='flap', hinge=0.75, span=(0.0,1.0), mode='symmetric')])],
                    CG=np.array([0.,0.,0.]))

hinge = 0.75
theta_h = onp.arccos(1 - 2*hinge)
thin_airfoil_ratio = (onp.pi - theta_h + onp.sin(theta_h)) / onp.pi

conv = []
for n in (4, 6, 9, 14, 20):
    a = flapWing(n)
    J = a.controlDerivatives(0.0, 0.0, np.zeros(1))   # (6, 1): CD_i,CY,CL,Cl,Cm,Cn x flap
    dCL_ddelta = float(J[2, 0])                        # per radian (jacfwd w.r.t. degrees below)
    # controlDerivatives differentiates w.r.t. deltas in DEGREES (coefficientsAt
    # takes degrees, converts internally) -> convert to per-radian for the
    # thin-airfoil comparison
    dCL_ddelta_rad = dCL_ddelta * 180.0 / onp.pi

    # dCL/dalpha via central difference on alpha itself (deltas=0 in both)
    eps = 1e-3
    cp = [float(v) for v in a.coefficientsAt( eps, 0.0, np.zeros(1))]
    cm = [float(v) for v in a.coefficientsAt(-eps, 0.0, np.zeros(1))]
    dCL_dalpha = (cp[2]-cm[2])/(2*onp.deg2rad(eps))

    ratio = dCL_ddelta_rad / dCL_dalpha
    conv.append(dict(n=n, N=int(a.collocation.shape[0]), dCL_ddelta=dCL_ddelta_rad,
                      dCL_dalpha=dCL_dalpha, ratio=ratio))

R['flap_effectiveness'] = dict(theory=float(thin_airfoil_ratio), hinge=hinge, conv=conv)


# --------------------------------------------------- 5. AD contra dif. central
# Sweep h instead of trusting one value: float32 central differences trade
# truncation error (large h) against cancellation error (small h), so the
# relative error is expected to trace a V against h with a floor somewhere
# around 1e-3, not vanish as h -> 0. A flat/decreasing error with no floor
# would be the actual red flag (would mean the AD gradient itself is wrong
# and just happens to agree with a biased finite difference at one h).
a = flapWing(9)
J_ad = onp.asarray(a.controlDerivatives(0.0, 0.0, np.zeros(1)))[:, 0]   # (6,)

# CD_i, CY, Cl, Cn are analytically zero here (symmetric flap, alpha=beta=0 on
# a planar wing -- same even/odd argument as the aileron check, mode-reversed:
# a SYMMETRIC-mode control leaves CY/Cl/Cn at zero to leading order). J_ad
# confirms it (~1e-11, float32 noise). A relative-error metric against that
# noise floor is measuring roundoff, not the AD -- restrict the check to the
# components with a real, nonzero derivative (CL, Cm).
meaningful = onp.abs(J_ad) > 1e-4

sweep = []
for h in (1.0, 0.1, 0.01, 0.001, 0.0001):
    c_p = onp.asarray([float(v) for v in a.coefficientsAt(0.0, 0.0, np.array([ h]))])
    c_m = onp.asarray([float(v) for v in a.coefficientsAt(0.0, 0.0, np.array([-h]))])
    J_fd = (c_p - c_m) / (2*h)
    rel_err = onp.abs(J_ad - J_fd) / onp.maximum(onp.abs(J_ad), 1e-8)
    sweep.append(dict(h_deg=h, max_rel_err=float(rel_err[meaningful].max())))

R['ad_vs_fd'] = dict(names=['CD_i','CY','CL','Cl','Cm','Cn'], ad=J_ad.tolist(), sweep=sweep,
                      best=min(sweep, key=lambda d: d['max_rel_err']))


json.dump(R, open(os.path.join(HERE, 'controles.json'), 'w'), indent=1)
print("1. regressao       max|dif| =", R['regression']['max_abs_diff'], '(simulate(deltas=None) vs coefficientsAt(zeros))')
print("2. aileron  5 deg  ", dict(zip(R['aileron']['names'], [round(v,6) for v in R['aileron']['values']])))
print("   par (CL,CD_i,Cm ~ 0):", {k: round(dict(zip(R['aileron']['names'],R['aileron']['values']))[k],6) for k in R['aileron']['even']})
print("   impar, livre (CY,Cl,Cn):", {k: round(dict(zip(R['aileron']['names'],R['aileron']['values']))[k],6) for k in R['aileron']['odd']})
print("3. leme    10 deg  ", dict(zip(R['rudder']['names'], [round(v,6) for v in R['rudder']['values']])))
print("4/6. flap vs teoria de perfil fino, hinge=%.2f, teoria=%.4f" % (R['flap_effectiveness']['hinge'], R['flap_effectiveness']['theory']))
for d in R['flap_effectiveness']['conv']:
    print(f"     n={d['n']:2d} N={d['N']:4d}  ratio={d['ratio']:.4f}  desvio={100*abs(d['ratio']-R['flap_effectiveness']['theory'])/R['flap_effectiveness']['theory']:.1f}%")
print("5. AD vs dif. central, varredura de h:")
for d in R['ad_vs_fd']['sweep']:
    print(f"     h={d['h_deg']:8.4f} deg   max_rel_err={d['max_rel_err']:.2e}")
