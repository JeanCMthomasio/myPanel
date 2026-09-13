"""Render as duas aeronaves lado a lado -- main.py e a conversao para o
AeroSandbox -- na mesma camera, para conferencia visual direta de que
surfaceBreakpoints/toAsbWing (em aerosandbox_compare.py) reproduz a
geometria certa. Geometria duplicada aqui verbatim da versao ja validada em
aerosandbox_compare.py, para nao reexecutar o estudo inteiro so pra plotar.

main.py usa eixos de estabilidade (x+ frente, z+ baixo); o AeroSandbox usa,
nos seus proprios "geometry axes", x a jusante/re e z para cima, fixo na
biblioteca. surfaceBreakpoints converte de um para o outro (inverte x,z) ao
extrair pontos para o AeroSandbox; os vertices que o VLM deles devolve
tambem sao convertidos de volta para eixos de estabilidade antes de plotar,
para os dois paineis da figura ficarem na mesma camera/orientacao.
"""
import matplotlib; matplotlib.use('Agg')
import os, numpy as onp, jax.numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from main import Surface, Aircraft
import aerosandbox as asb

FLAT = asb.Airfoil("naca0012")


def surfaceBreakpoints(s):
    pts, chords = [], []
    root = (0.0, s.span[0], 0.0)
    n_seg = len(s.span) - 1
    for i, (span_root, span_tip, chord_root, chord_tip, sweep, dihedral) in enumerate(
            zip(s.span[:-1], s.span[1:], s.chord[:-1], s.chord[1:], s.sweep, s.dihedral)):
        X, Y, Z, root_next = s.transform(0.0, 0.0, span_tip-span_root, chord_root, chord_tip, sweep, dihedral, root)
        pts.append((-float(X), float(Y), -float(Z))); chords.append(chord_root)   # -> geometry axes do AeroSandbox
        if i == n_seg - 1:
            Xt, Yt, Zt, _ = s.transform(0.0, 1.0, span_tip-span_root, chord_root, chord_tip, sweep, dihedral, root)
            pts.append((-float(Xt), float(Yt), -float(Zt))); chords.append(chord_tip)
        root = root_next
    return pts, chords

def toAsbWing(s, name):
    pts, chords = surfaceBreakpoints(s)
    return asb.Wing(name=name, symmetric=s.symmetry,
                     xsecs=[asb.WingXSec(xyz_le=list(p), chord=c, airfoil=FLAT) for p, c in zip(pts, chords)])


# ---- a mesma aeronave de aerosandbox_compare.py::buildMine() ----
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
surfaces = [wing, winglet_right, winglet_left, hTail, vTail]
ac = Aircraft(surfaces=surfaces, CG=np.array([-1.,0.,0.]))
ac.generateMesh()

asb_wings = [toAsbWing(s, nm) for s, nm in zip(surfaces, ('wing','winglet_right','winglet_left','hTail','vTail'))]
airplane = asb.Airplane(wings=asb_wings, xyz_ref=[1.,0.,0.],
                         s_ref=float(wing.S), c_ref=float(wing.MAC), b_ref=float(wing.b))
vlm = asb.VortexLatticeMethod(
    airplane=airplane, op_point=asb.OperatingPoint(velocity=1.0, alpha=0.0),
    spanwise_resolution=20, chordwise_resolution=9, align_trailing_vortices_with_wind=False,
)
vlm.run()   # popula front_left_vertices etc.


# ---------------------------------------------------------------- render
INK = '0.35'
fig = plt.figure(figsize=(12, 6.2), dpi=170)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

for surface in surfaces:
    surface.plot(fig_ax=(fig, ax1))

fl, bl, br, fr = (onp.asarray(getattr(vlm, n)) for n in
                   ('front_left_vertices','back_left_vertices','back_right_vertices','front_right_vertices'))
# vlm's vertices are in AeroSandbox's own geometry axes (x aft, z up) --
# convert back to stability axes (x forward, z down) so both panels share
# one consistent frame; otherwise the shared box limits below and the two
# cameras would silently disagree (a mirrored aircraft the right size, in
# the wrong orientation, exactly the bug this comparison exists to catch)
flip = onp.array([-1.0, 1.0, -1.0])
fl, bl, br, fr = (v*flip for v in (fl, bl, br, fr))
segs = onp.concatenate([
    onp.stack([fl, bl], axis=1), onp.stack([bl, br], axis=1),
    onp.stack([br, fr], axis=1), onp.stack([fr, fl], axis=1),
], axis=0)
ax2.add_collection3d(Line3DCollection(segs, colors=INK, linewidths=0.5))

all_pts = onp.concatenate([onp.asarray(ac.nodes), fl, bl, br, fr])
lims = [(all_pts[:,k].min(), all_pts[:,k].max()) for k in range(3)]
for ax, title in [(ax1, 'main.py'), (ax2, 'AeroSandbox (VortexLatticeMethod)')]:
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_box_aspect([hi-lo for lo,hi in lims])
    # stability axes put z+ down, and matplotlib always draws z+ up, so without a
    # correction the camera shows the aircraft belly-up (same fix as Aircraft.plot):
    # roll=180 turns the camera over instead of flipping an axis, which would mirror
    # the scene and swap left/right wing.
    ax.view_init(elev=58, azim=-90, roll=180)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, color='0.1', pad=-10)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92, wspace=0.02)
fig.savefig(os.path.join(HERE, 'asb_geometry_compare.png'))
print('ok')
