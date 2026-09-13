import matplotlib; matplotlib.use('Agg')
import os, sys, json, numpy as onp
HERE = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
INT = FuncFormatter(lambda v, p: f'{int(round(v))}')

R = json.load(open(os.path.join(HERE,'study.json')))
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e2e1dc'
plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':9,
    'axes.facecolor':SURF, 'figure.facecolor':SURF,
    'axes.edgecolor':GRID, 'axes.labelcolor':INK2, 'axes.titlecolor':INK,
    'xtick.color':INK2, 'ytick.color':INK2, 'grid.color':GRID,
    'axes.spines.top':False, 'axes.spines.right':False,
})
def style(ax):
    ax.grid(True, lw=.7, alpha=.9); ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.7)

OUT = HERE

# ---------- 1. CL: chord vs span ----------
fig, ax = plt.subplots(figsize=(6.4,3.9), dpi=160)
c, s = R['chord'], R['span']
conv = s[-1]['CL']
ax.axhline(conv, color=INK2, lw=1, ls=(0,(4,3)), zorder=1)
ax.annotate(f'converged  {conv:.4f}', (1.15, conv), color=INK2, fontsize=8,
            va='bottom', ha='left')
ax.plot([d['nc'] for d in c], [d['CL'] for d in c], color=BLUE, lw=2,
        marker='o', ms=5.5, mec=SURF, mew=1.2, label='chord refinement (span fixed at 24)', zorder=3)
ax.plot([d['ns'] for d in s], [d['CL'] for d in s], color=ORANGE, lw=2,
        marker='s', ms=5.5, mec=SURF, mew=1.2, label='span refinement (chord fixed at 8)', zorder=3)
ax.set_xscale('log', base=2); ax.xaxis.set_major_formatter(INT)
ax.set_xlabel('panels in refined direction'); ax.set_ylabel('$C_L$  (α = 5°)')
ax.set_title('$C_L$ convergence — rectangular wing AR = 8', pad=10, fontsize=10.5, loc='left')
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='upper right')
style(ax); fig.tight_layout(); fig.savefig(f'{OUT}/conv_cl.png'); plt.close(fig)

# ---------- 2. span efficiency ----------
fig, ax = plt.subplots(figsize=(6.4,3.9), dpi=160)
x = [d['ns'] for d in s]; y = [d['e'] for d in s]
ax.axhspan(1.0, 1.13, color=ORANGE, alpha=.09, zorder=0)
ax.axhline(1.0, color=ORANGE, lw=1.4, ls=(0,(4,3)), zorder=2)
ax.annotate('impossible region:\ne > 1 on a planar wing', (62, 1.105), color=ORANGE,
            fontsize=8.5, va='top', ha='right', linespacing=1.4)
ax.plot(x, y, color=AQUA, lw=2, marker='o', ms=5.5, mec=SURF, mew=1.2, zorder=3)
for xi, yi in zip(x, y):
    if yi > 1: ax.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                           xytext=(0,9), ha='center', fontsize=8, color=INK)
ax.annotate(f'{y[-1]:.3f}', (x[-1], y[-1]), textcoords='offset points',
            xytext=(-2,-13), ha='right', fontsize=8, color=INK)
ax.set_xscale('log', base=2); ax.xaxis.set_major_formatter(INT)
ax.set_xlabel('panels in span, per half-wing'); ax.set_ylabel('$e = C_L^2 / (\\pi\\,AR\\,C_{D_i})$')
ax.set_title('Span efficiency — what the coarse mesh invents', pad=10, fontsize=10.5, loc='left')
style(ax); fig.tight_layout(); fig.savefig(f'{OUT}/conv_e.png'); plt.close(fig)

# ---------- 3. neutral point: wing vs tail ----------
fig, ax = plt.subplots(figsize=(6.4,3.9), dpi=160)
w, t = R['ac_wing'], R['ac_tail']
ax.plot([d['panels'] for d in w], [d['x_np'] for d in w], color=BLUE, lw=2,
        marker='o', ms=5.5, mec=SURF, mew=1.2, label='refining the WING (tail fixed)', zorder=3)
ax.plot([d['panels'] for d in t], [d['x_np'] for d in t], color=ORANGE, lw=2,
        marker='s', ms=5.5, mec=SURF, mew=1.2, label='refining the TAIL (wing fixed)', zorder=3)
ax.annotate(f"Δ = {max(d['x_np'] for d in w)-min(d['x_np'] for d in w):.4f}",
            (w[-1]['panels'], w[-1]['x_np']), textcoords='offset points', xytext=(-6,10),
            ha='right', fontsize=8.5, color=BLUE)
ax.annotate(f"Δ = {max(d['x_np'] for d in t)-min(d['x_np'] for d in t):.4f}",
            (t[0]['panels'], t[0]['x_np']), textcoords='offset points', xytext=(10,2),
            ha='left', fontsize=8.5, color=ORANGE)
ax.set_xscale('log', base=2); ax.xaxis.set_major_formatter(INT)
ax.set_xlabel('panels on the refined surface'); ax.set_ylabel('$x_{np}$  (m, CG at origin)')
ax.set_title('Neutral point — governed by the tail, not the wing', pad=10, fontsize=10.5, loc='left')
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='center right')
style(ax); fig.tight_layout(); fig.savefig(f'{OUT}/conv_np.png'); plt.close(fig)
print('ok ->', OUT)
