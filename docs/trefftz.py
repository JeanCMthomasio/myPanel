"""Campo proximo (o que main.py faz) contra o plano de Trefftz (Drela 6.4.4).

O solver calcula forcas no campo proximo: Kutta-Joukowski em cada vortice
ligado, eq. 6.42, com o arrasto induzido saindo da projecao no escoamento
(eq. 6.50). A alternativa do Drela e integrar no plano de Trefftz, longe a
jusante, onde a esteira e um problema 2D -- eq. 6.26 a 6.31. O proprio texto
diz que o campo distante e mais confiavel "especially for the D_i component",
e este estudo mede se isso aparece aqui.

trefftzLoads() fica neste script de proposito: e um caminho de verificacao,
nao uma funcionalidade do solver, e main.py deve seguir simples.
"""
import matplotlib; matplotlib.use('Agg')
import json, numpy as onp, jax.numpy as np
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from main import Surface, Aircraft

R = {}
WAKE = np.array([-1.0, 0.0, 0.0])   # direcao da esteira, a mesma de computeInfluences


def frame(alpha, beta=0.0):
    V_bar = np.array([-onp.cos(onp.deg2rad(alpha))*onp.cos(onp.deg2rad(beta)),
                      -onp.sin(onp.deg2rad(beta)),
                      -onp.sin(onp.deg2rad(alpha))*onp.cos(onp.deg2rad(beta))])
    T_a = np.array([[-onp.cos(onp.deg2rad(alpha)), 0.0, -onp.sin(onp.deg2rad(alpha))],
                    [0.0, 1.0, 0.0],
                    [ onp.sin(onp.deg2rad(alpha)), 0.0, -onp.cos(onp.deg2rad(alpha))]])
    return V_bar, np.zeros(3), T_a


def trefftzLoads(ac, circulation, S_ref):
    """(CD_i, CY, CL) pelo plano de Trefftz, ja em eixos de vento.

    Cada coluna de paineis em corda deixa na esteira um unico par de fios, de
    intensidade igual a soma das circulacoes da coluna: os fios internos de
    ferraduras vizinhas se cancelam. No plano de Trefftz esses fios sao
    vortices pontuais 2D, e a folha entre eles carrega o salto de potencial
    dphi = Gamma da faixa (eq. 6.27/6.28).
    """
    G, A, B = [], [], []
    off = 0
    for s in ac.surfaces:
        nc, ns = s.collocation.shape[0], s.collocation.shape[1]
        G.append(circulation[off:off+nc*ns].reshape(nc, ns).sum(axis=0))
        A.append(s.vortex_a.reshape(nc, ns, 3)[0])   # y,z nao variam ao longo da corda
        B.append(s.vortex_b.reshape(nc, ns, 3)[0])
        off += nc*ns
    G, A, B = np.concatenate(G), np.concatenate(A), np.concatenate(B)

    P  = np.concatenate([A, B])          # posicao dos fios
    gm = np.concatenate([G, -G])         # +G na borda A, -G na borda B

    # a folha vive no plano de Trefftz, entao a tangente e o comprimento saem da
    # PROJECAO de B-A em (y,z). Usar a tangente 3D inclui a componente x do
    # enflechamento da linha de 1/4 de corda, que nao existe naquele plano -- numa asa
    # retangular isso nao aparece, numa asa enflechada e afilada erra o CL em ~10%.
    t     = (B - A).at[:, 0].set(0.0)
    ds    = np.linalg.norm(t, axis=-1)
    t_hat = t / ds[:, None]
    n_hat = np.cross(WAKE, t_hat)
    mid   = 0.5*(A + B)

    # eq. 6.26: campo 2D dos fios, avaliado no meio de cada faixa da folha
    r  = (mid[:, None, :] - P[None, :, :]).at[..., 0].set(0.0)
    r2 = np.sum(r*r, axis=-1)
    r2 = np.where(r2 < 1e-12, np.inf, r2)        # ignora a auto-inducao coincidente
    grad = np.sum((gm[None, :, None]/(2*np.pi)) * np.cross(WAKE, r) / r2[..., None], axis=1)

    # eq. 6.29 (arrasto) e 6.30/6.31 (forcas), adimensionalizados por (1/2) rho V^2 S.
    # Sem T_a: o campo distante ja entrega a forca perpendicular ao escoamento, entao
    # rotacionar de novo projetaria duas vezes -- o erro aparece como um cos^2(alpha)
    # espurio em e, e foi assim que ele foi detectado.
    CD    = (1.0/S_ref) * np.sum(G * np.sum(grad*n_hat, axis=-1) * ds)
    F_bar = -(2.0/S_ref) * np.sum(G[:, None] * np.cross(WAKE, t_hat) * ds[:, None], axis=0)
    return float(CD), float(F_bar[1]), float(-F_bar[2])   # z+ para baixo: sustentacao = -Fz


def bothMethods(ac, S_ref, alpha, beta=0.0):
    V_bar, omega_bar, T_a = frame(alpha, beta)
    normals = ac.deflect(None)
    circ = ac.solveSystem(ac.computeAIC(normals), normals, V_bar, omega_bar)
    near = [float(v) for v in ac.computeInviscousCoefficients(circ, alpha, V_bar, omega_bar)]
    CD_t, CY_t, CL_t = trefftzLoads(ac, circ, S_ref)
    return dict(CD_near=near[0], CY_near=near[1], CL_near=near[2], Cm_near=near[4],
                CD_tref=CD_t,    CY_tref=CY_t,    CL_tref=CL_t)


# ====================================== 1. asa retangular AR=8: varredura em alpha
AR = 8.0
def rectWing(n=9, m=41):
    w = Surface(span=[0.,4.], chord=[1.,1.], sweep=[0.], dihedral=[0.], position=[0.,0.,0.],
                discretization=[(n,m)], symmetry=True)
    return Aircraft(surfaces=[w], CG=np.array([0.,0.,0.])), float(w.S)

ac, S = rectWing()
rows = []
for a in (1., 2., 5., 8., 12.):
    d = bothMethods(ac, S, a)
    d['alpha'] = a
    d['e_near'] = d['CL_near']**2/(onp.pi*AR*d['CD_near'])
    d['e_tref'] = d['CL_tref']**2/(onp.pi*AR*d['CD_tref'])
    rows.append(d)
R['alpha_sweep'] = rows


# ============================================= 2. convergencia de malha, alpha = 5
conv = []
for n, m in [(3,11), (5,21), (9,41), (13,61), (17,81)]:
    ac_k, S_k = rectWing(n, m)
    d = bothMethods(ac_k, S_k, 5.0)
    d.update(nc=n-1, ns=m-1, N=int(ac_k.collocation.shape[0]),
             e_near=d['CL_near']**2/(onp.pi*AR*d['CD_near']),
             e_tref=d['CL_tref']**2/(onp.pi*AR*d['CD_tref']))
    conv.append(d)
R['convergence'] = conv


# ================================================ 3. aeronave completa (a do __main__)
def fullAircraft():
    wing = Surface(span=[0.,1.,3.5], chord=[1.75,1.,.5], sweep=[10.,10.], dihedral=[1.,2.],
                   position=[0.,0.,0.], discretization=[(6,11),(6,31)], symmetry=True)
    wl_r = Surface(span=[0.,0.5], chord=[.5,.5], sweep=[0.], dihedral=[90.],
                   position=[-3.5*onp.sin(onp.deg2rad(10)),  3.4983, -0.1047],
                   discretization=[(5,10)], symmetry=False)
    wl_l = Surface(span=[0.,0.5], chord=[.5,.5], sweep=[0.], dihedral=[90.],
                   position=[-3.5*onp.sin(onp.deg2rad(10)), -3.4983, -0.1047],
                   discretization=[(5,10)], symmetry=False)
    hT = Surface(span=[0.,1.5], chord=[.75,.5], sweep=[25.], dihedral=[0.],
                 position=[-4.,0.,-1.], discretization=[(10,25)], symmetry=True)
    vT = Surface(span=[0.,1.25], chord=[.75,.5], sweep=[5.], dihedral=[90.],
                 position=[-4.,0.,-1.], discretization=[(10,25)], symmetry=False)
    return Aircraft(surfaces=[wing, wl_r, wl_l, hT, vT], CG=np.array([-1.,0.,0.])), float(wing.S)

ac_f, S_f = fullAircraft()
R['full'] = [dict(alpha=a, **bothMethods(ac_f, S_f, a)) for a in (0., 2., 5., 8.)]
R['full_beta'] = [dict(beta=b, **bothMethods(ac_f, S_f, 5.0, b)) for b in (-5., 0., 5.)]


# ===== 4. o caso onde o campo proximo ja divergia de uma referencia independente.
# docs/aerosandbox_compare.md registra CD_i 5,2% abaixo do AeroSandbox na aeronave
# completa; se o campo distante for mesmo mais confiavel, deve encostar mais perto.
def asbAircraft():
    wing = Surface(span=[0.,1.,3.5], chord=[1.75,1.,.5], sweep=[10.,20.], dihedral=[1.,2.],
                   position=[0.,0.,0.], discretization=[(9,41),(9,101)], symmetry=True)
    wl_r = Surface(span=[0.,0.5], chord=[.5,.2], sweep=[0.], dihedral=[90.],
                   position=[-1.0863, 3.4983, -0.1047], discretization=[(9,20)], symmetry=False)
    wl_l = Surface(span=[0.,0.5], chord=[.5,.2], sweep=[0.], dihedral=[90.],
                   position=[-1.0863,-3.4983, -0.1047], discretization=[(9,20)], symmetry=False)
    hT = Surface(span=[0.,1.], chord=[.6,.2], sweep=[15.], dihedral=[0.], position=[-4.,0.,-1.],
                 discretization=[(9,41)], symmetry=True)
    vT = Surface(span=[0.,.75], chord=[.6,.2], sweep=[5.], dihedral=[90.], position=[-4.,0.,-1.],
                 discretization=[(9,41)], symmetry=False)
    return Aircraft(surfaces=[wing, wl_r, wl_l, hT, vT], CG=np.array([-1.,0.,0.])), float(wing.S)

asb_ref = json.load(open(os.path.join(HERE, 'aerosandbox_compare.json')))
row5 = next(r for r in asb_ref['full']['rows'] if r['alpha'] == 5)
ac_a, S_a = asbAircraft()
d = bothMethods(ac_a, S_a, 5.0)
R['vs_aerosandbox'] = dict(alpha=5.0, CD_asb=row5['theirs']['CD_i'], CL_asb=row5['theirs']['CL'],
                            CD_near=d['CD_near'], CD_tref=d['CD_tref'],
                            CL_near=d['CL_near'], CL_tref=d['CL_tref'])


json.dump(R, open(os.path.join(HERE, 'trefftz.json'), 'w'), indent=1)

print("=== 1. asa retangular AR=8, varredura em alpha ===")
print(f"{'alpha':>6} {'CL perto':>10} {'CL tref':>10} {'CD perto':>10} {'CD tref':>10} {'e perto':>9} {'e tref':>8}")
for d in R['alpha_sweep']:
    print(f"{d['alpha']:6.1f} {d['CL_near']:10.5f} {d['CL_tref']:10.5f} "
          f"{d['CD_near']:10.6f} {d['CD_tref']:10.6f} {d['e_near']:9.4f} {d['e_tref']:8.4f}")

print("\n=== 2. convergencia de malha, alpha=5 ===")
print(f"{'nc x ns':>9} {'N':>6} {'CD perto':>10} {'CD tref':>10} {'e perto':>9} {'e tref':>8}")
for d in R['convergence']:
    print(f"{d['nc']:4d} x{d['ns']:3d} {d['N']:6d} {d['CD_near']:10.6f} {d['CD_tref']:10.6f} "
          f"{d['e_near']:9.4f} {d['e_tref']:8.4f}")

print("\n=== 3. aeronave completa ===")
for d in R['full']:
    dif = 100*abs(d['CD_near']-d['CD_tref'])/max(abs(d['CD_tref']), 1e-12)
    print(f"  alpha={d['alpha']:4.1f}  CL {d['CL_near']:8.5f}/{d['CL_tref']:8.5f}   "
          f"CD_i {d['CD_near']:9.6f}/{d['CD_tref']:9.6f}  ({dif:5.1f}%)   Cm {d['Cm_near']:8.5f}/  ---")
for d in R['full_beta']:
    print(f"  beta={d['beta']:+5.1f}  CY {d['CY_near']:+8.5f}/{d['CY_tref']:+8.5f}   "
          f"CD_i {d['CD_near']:9.6f}/{d['CD_tref']:9.6f}")

v = R['vs_aerosandbox']
print("\n=== 4. contra o AeroSandbox (aeronave de aerosandbox_compare.py, alpha=5) ===")
for lbl, cd, cl in [('campo proximo', v['CD_near'], v['CL_near']),
                    ('Trefftz      ', v['CD_tref'], v['CL_tref']),
                    ('AeroSandbox  ', v['CD_asb'],  v['CL_asb'])]:
    print(f"  {lbl}  CD_i={cd:.6f}  CL={cl:.6f}"
          + ("" if 'Aero' in lbl else f"   (desvio de CD_i: {100*abs(cd-v['CD_asb'])/v['CD_asb']:.2f}%)"))
