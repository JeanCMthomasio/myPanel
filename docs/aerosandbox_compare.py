"""Comparação contra o VLM do AeroSandbox (mesmo formalismo de Drela).

main.py usa eixos de estabilidade (x+ frente, z+ baixo). O AeroSandbox usa,
nos seus PRÓPRIOS "geometry axes", x a jusante/ré, z para cima -- fixo na
biblioteca deles, confirmado por inspeção de aerosandbox.performance.
operating_point, e não é algo que se troque. As duas convenções são,
portanto, DIFERENTES uma da outra agora (não eram antes de main.py adotar
eixos de estabilidade) -- surfaceBreakpoints extrai pontos de
Surface.transform (eixos de estabilidade) e precisa convertê-los para o
"geometry axes" do AeroSandbox antes de empacotar em WingXSec; ver o
comentário nela.

Duas frentes:
  1. Asa retangular AR=8 -- o mesmo caso com referência analítica usado em
     docs/study.py (CL_alpha, e, x_cp).
  2. A aeronave completa do __main__ (asa + winglets + cauda h/v), convertida
     superfície a superfície via surfaceBreakpoints/toAsbWing, sem retranscrever
     nenhuma coordenada à mão.

Ambos os códigos rodam com a esteira fixa no eixo do corpo
(align_trailing_vortices_with_wind=False no AeroSandbox; main.py usa
x_hat=[-1,0,0] internamente, aft agora sendo -x).
"""
import matplotlib; matplotlib.use('Agg')
import json, numpy as onp, jax.numpy as np
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from main import Surface, Aircraft
import aerosandbox as asb

R = {}
FLAT = asb.Airfoil("naca0012")   # simetrico: linha de camber plana, casa com o VLM de placa plana daqui


def surfaceBreakpoints(s):
    """LE (x,y,z) e corda em cada estacao de s.span, semi-asa NAO espelhada,
    caminhando pela mesma cadeia de sweep/diedro que Surface.generateMesh usa
    -- exceto que o (x,z) que sai daqui ja vem invertido para o "geometry
    axes" do AeroSandbox (x a jusante, z para cima), oposto aos eixos de
    estabilidade que Surface.transform usa internamente. Sem essa conversao a
    geometria entregue ao AeroSandbox sairia espelhada em x e z."""
    pts, chords = [], []
    root = (0.0, s.span[0], 0.0)
    n_seg = len(s.span) - 1
    for i, (span_root, span_tip, chord_root, chord_tip, sweep, dihedral) in enumerate(
            zip(s.span[:-1], s.span[1:], s.chord[:-1], s.chord[1:], s.sweep, s.dihedral)):
        # LE no INICIO deste segmento = breakpoint i (raiz, na 1a iteracao)
        X, Y, Z, root_next = s.transform(0.0, 0.0, span_tip-span_root, chord_root, chord_tip, sweep, dihedral, root)
        pts.append((-float(X), float(Y), -float(Z))); chords.append(chord_root)
        if i == n_seg - 1:
            # ultimo segmento: acrescenta tambem a ponta (y=1)
            Xt, Yt, Zt, _ = s.transform(0.0, 1.0, span_tip-span_root, chord_root, chord_tip, sweep, dihedral, root)
            pts.append((-float(Xt), float(Yt), -float(Zt))); chords.append(chord_tip)
        root = root_next
    return pts, chords


def toAsbWing(s, name):
    pts, chords = surfaceBreakpoints(s)
    return asb.Wing(
        name=name, symmetric=s.symmetry,
        xsecs=[asb.WingXSec(xyz_le=list(p), chord=c, airfoil=FLAT) for p, c in zip(pts, chords)],
    )


def mine(ac, alpha, beta=0.0):
    c = ac.simulate(V_inf=1.0, alpha=alpha, beta=beta)
    return dict(CD_i=float(c[0]), CY=float(c[1]), CL=float(c[2]),
                Cl=float(c[3]), Cm=float(c[4]), Cn=float(c[5]))


def theirs(airplane, alpha, beta=0.0, spanwise_resolution=20, chordwise_resolution=9):
    # resolucoes de verdade: nada aqui e pre-subdividido por fora (exceto o
    # caso de convergencia casada abaixo, que subdivide explicitamente e entao
    # passa resolution=1 para nao subdividir de novo)
    vlm = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=1.0, alpha=alpha, beta=beta),
        spanwise_resolution=spanwise_resolution, chordwise_resolution=chordwise_resolution,
        align_trailing_vortices_with_wind=False,
    )
    res = vlm.run()
    return dict(CD_i=float(res['CD']), CY=float(res['CY']), CL=float(res['CL']),
                Cl=float(res['Cl']), Cm=float(res['Cm']), Cn=float(res['Cn']))


# ============================================================ 1. asa retangular
AR, S_REF = 8.0, 8.0

def rectSurface(n, m):
    return Surface(span=[0.,4.], chord=[1.,1.], sweep=[0.], dihedral=[0.],
                   position=[0.,0.,0.], discretization=[(n,m)], symmetry=True)

def rectMine(n, m):
    return Aircraft(CG=np.array([0.,0.,0.]), surfaces=[rectSurface(n, m)])

def rectTheirs(spanwise, chordwise):
    # mesma extracao de coordenadas usada na aeronave completa (surfaceBreakpoints),
    # em vez de digitar xyz_le a mao -- evita reintroduzir o mesmo tipo de erro
    ref = rectSurface(9, 5)
    wing = toAsbWing(ref, 'w').subdivide_sections(ratio=spanwise, spacing_function=onp.linspace)
    return asb.Airplane(wings=[wing], xyz_ref=[0.,0.,0.], s_ref=S_REF, c_ref=1.0, b_ref=8.0)

alphas = list(range(-5, 6))
rows = []
n_mine, m_mine = 9, 61          # malha fina, uniforme
n_asb, m_asb   = 60, 60         # coincide com o numero de PAINEIS de cada lado
ac_rect = rectMine(n_mine, m_mine)
ap_rect = rectTheirs(m_asb, n_asb)     # ja pre-subdividido em envergadura
for a in alphas:
    my = mine(ac_rect, float(a))
    # spanwise_resolution=1: nao subdivide de novo por cima do que rectTheirs ja fez
    th = theirs(ap_rect, float(a), spanwise_resolution=1, chordwise_resolution=9)
    rows.append(dict(alpha=a, mine=my, theirs=th))
R['rect'] = dict(N_mine=int(ac_rect.collocation.shape[0]), rows=rows,
                  CL_alpha_theory=None)  # preenchido abaixo


# convergencia conjunta: os dois refinam juntos, olhando a diferenca encolher
conv = []
for k in (2, 4, 8, 16, 32):
    ac_k = rectMine(9, 8*k+1)
    ap_k = rectTheirs(8*k, 9)          # ja pre-subdividido em envergadura, uniforme
    my = mine(ac_k, 5.0)
    th = theirs(ap_k, 5.0, spanwise_resolution=1, chordwise_resolution=8)
    conv.append(dict(paineis_por_semiasa=k*8, N_mine=int(ac_k.collocation.shape[0]),
                      CL_mine=my['CL'], CL_theirs=th['CL'],
                      dCL=abs(my['CL']-th['CL'])))
R['rect_convergence'] = conv


# ============================================================ 2. aeronave real
def buildMine():
    wing = Surface(span=[0.,1.,3.5], chord=[1.75,1.,.5], sweep=[10.,20.], dihedral=[1.,2.],
                   position=[0.,0.,0.], discretization=[(9,41),(9,101)], symmetry=True)
    winglet_right = Surface(span=[0.,0.5], chord=[.5,.2], sweep=[0.], dihedral=[90.],
                          position=[-1.0863,3.4983,-0.1047], discretization=[(9,20)], symmetry=False)
    winglet_left = Surface(span=[0.,0.5], chord=[.5,.2], sweep=[0.], dihedral=[90.],
                          position=[-1.0863,-3.4983,-0.1047], discretization=[(9,20)], symmetry=False)
    hTail = Surface(span=[0.,1.], chord=[.6,.2], sweep=[15.], dihedral=[0.], position=[-4.,0.,-1.],
                    discretization=[(9,41)], symmetry=True)
    vTail = Surface(span=[0.,.75], chord=[.6,.2], sweep=[5.], dihedral=[90.], position=[-4.,0.,-1.],
                    discretization=[(9,41)], symmetry=False)
    return Aircraft(surfaces=[wing, winglet_right, winglet_left, hTail, vTail], CG=np.array([-1.,0.,0.])), wing

ac_full, wing_ref = buildMine()
ac_full.generateMesh()

surfaces_full = ac_full.surfaces
asb_wings = [toAsbWing(s, nm) for s, nm in
             zip(surfaces_full, ('wing', 'winglet_right', 'winglet_left', 'hTail', 'vTail'))]
airplane_full = asb.Airplane(wings=asb_wings, xyz_ref=[1.,0.,0.],
                              s_ref=float(wing_ref.S), c_ref=float(wing_ref.MAC), b_ref=float(wing_ref.b))

rows_full = []
for a in alphas:
    my = mine(ac_full, float(a))
    th = theirs(airplane_full, float(a))
    rows_full.append(dict(alpha=a, mine=my, theirs=th))
R['full'] = dict(N_mine=int(ac_full.collocation.shape[0]), rows=rows_full)

# lateral: mesmo alpha=5, varrendo beta, pra ver CY/Cl/Cn
rows_beta = []
for b in (-5,-2,0,2,5):
    my = mine(ac_full, 5.0, float(b))
    th = theirs(airplane_full, 5.0, float(b))
    rows_beta.append(dict(beta=b, mine=my, theirs=th))
R['full_beta'] = rows_beta


json.dump(R, open(os.path.join(HERE, 'aerosandbox_compare.json'), 'w'), indent=1)

print('=== asa retangular AR=8, alpha=5 ===')
r5 = next(r for r in R['rect']['rows'] if r['alpha']==5)
print('  mine  :', r5['mine'])
print('  theirs:', r5['theirs'])
print()
print('=== convergencia conjunta ===')
for d in conv:
    print(f"  paineis/semiasa={d['paineis_por_semiasa']:3d}  CL_mine={d['CL_mine']:.4f}  CL_theirs={d['CL_theirs']:.4f}  dCL={d['dCL']:.2e}")
print()
print('=== aeronave completa, alpha=5, beta=0 ===')
r5f = next(r for r in R['full']['rows'] if r['alpha']==5)
print('  mine  :', r5f['mine'])
print('  theirs:', r5f['theirs'])
print()
print('=== lateral, alpha=5, beta variavel ===')
for d in rows_beta:
    print(f"  beta={d['beta']:+.0f}  mine CY={d['mine']['CY']:+.4f} Cl={d['mine']['Cl']:+.4f} Cn={d['mine']['Cn']:+.4f}"
          f"   |  theirs CY={d['theirs']['CY']:+.4f} Cl={d['theirs']['Cl']:+.4f} Cn={d['theirs']['Cn']:+.4f}")
