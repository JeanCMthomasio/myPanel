import matplotlib; matplotlib.use('Agg')
import json, numpy as onp, jax.numpy as np
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from main import Surface, Aircraft

ALPHA, S, B = 5.0, 8.0, 8.0
AR = B*B/S
R = {}

def rect(n, m):
    w = Surface(span=[0.,4.], chord=[1.,1.], sweep=[0.], dihedral=[0.],
                position=[0.,0.,0.], discretization=[(n,m)], symmetry=True)
    ac = Aircraft(CG=np.array([0.,0.,0.]), surfaces=[w])
    CD, CY, CL, Cl, Cm, Cn = [float(v) for v in ac.coefficientsAt(ALPHA, 0.0, None, V_inf=1.0)]
    return dict(N=int(ac.collocation.shape[0]), nc=n-1, ns=m-1, CL=CL,
                CLa=CL/onp.deg2rad(ALPHA), CD=CD, e=CL**2/(onp.pi*AR*CD),
                xcp=-Cm*float(w.MAC)/CL, cond=float(onp.linalg.cond(onp.asarray(ac.computeAIC(ac.deflect())))))

R['chord'] = [rect(n, 25) for n in (2,3,5,9,17,33)]
R['span']  = [rect(9, m)  for m in (5,9,17,25,41,61)]

def plane(nw, mw, nt, mt):
    return Aircraft(CG=np.array([0.,0.,0.]), surfaces=[
        Surface(span=[0.,1.,3.5], chord=[1.75,1.,.5], sweep=[10.,20.], dihedral=[1.,2.],
                position=[0.,0.,0.], discretization=[(nw,mw),(nw,int(2.5*mw))], symmetry=True),
        Surface(span=[0.,1.], chord=[.6,.2], sweep=[15.], dihedral=[0.], position=[-4.,0.,-1.],
                discretization=[(nt,mt)], symmetry=True),
        Surface(span=[0.,1.], chord=[.6,.2], sweep=[5.], dihedral=[90.], position=[-4.,0.,-1.],
                discretization=[(nt,mt)], symmetry=False)])

def neutral(ac, which):
    c0 = [float(v) for v in ac.coefficientsAt(0.0, 0.0, None, V_inf=1.0)]
    cond0 = float(onp.linalg.cond(onp.asarray(ac.computeAIC(ac.deflect()))))
    c5 = [float(v) for v in ac.coefficientsAt(5.0, 0.0, None, V_inf=1.0)]
    cond5 = float(onp.linalg.cond(onp.asarray(ac.computeAIC(ac.deflect()))))
    mac = float(ac.surfaces[0].MAC)
    npan = int(onp.prod(onp.asarray(ac.surfaces[which].collocation).shape[:2]))
    return dict(N=int(ac.collocation.shape[0]), panels=npan, CL=c5[2], CD=c5[0],
                x_np=-(c5[4]-c0[4])/(c5[2]-c0[2])*mac, cond=max(cond0, cond5))

R['ac_wing'] = [neutral(plane(nw, mw, 10, 15), 0) for nw, mw in [(4,5),(6,7),(8,11),(10,15),(12,21)]]
R['ac_tail'] = [neutral(plane(10, 11, nt, mt), 1) for nt, mt in [(3,4),(5,7),(8,11),(10,15),(14,23)]]

json.dump(R, open(os.path.join(HERE,'study.json'),'w'), indent=1)
for k, v in R.items():
    print(k, len(v), 'casos, N de', v[0]['N'], 'a', v[-1]['N'])
