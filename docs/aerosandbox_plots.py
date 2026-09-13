import matplotlib; matplotlib.use('Agg')
import os, json, numpy as onp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
INT = FuncFormatter(lambda v, p: f'{int(round(v))}')
HERE = os.path.dirname(os.path.abspath(__file__))

R = json.load(open(os.path.join(HERE, 'aerosandbox_compare.json')))
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

# ---------- 1. CL, CD_i, Cm vs alpha on the full aircraft: two codes overlaid ----------
rows = R['full']['rows']
al = [r['alpha'] for r in rows]
fig, axs = plt.subplots(1, 3, figsize=(11,3.6), dpi=160)
for ax, key, ylabel in zip(axs, ('CL','CD_i','Cm'), ('$C_L$', '$C_{D_i}$', '$C_m$')):
    mine_v = [r['mine'][key] for r in rows]
    their_v = [r['theirs'][key] for r in rows]
    ax.plot(al, mine_v, color=BLUE, lw=2, marker='o', ms=5, mec=SURF, mew=1, label='main.py', zorder=3)
    ax.plot(al, their_v, color=ORANGE, lw=1.6, ls=(0,(5,2)), marker='s', ms=4.5, mec=SURF, mew=1,
             label='AeroSandbox', zorder=2)
    ax.set_xlabel('α (°)'); ax.set_ylabel(ylabel)
    style(ax)
axs[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='upper left')
fig.suptitle('Full aircraft — main.py × AeroSandbox VLM', fontsize=11, x=0.02, ha='left', color=INK)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(f'{OUT}/asb_full_alpha.png')
plt.close(fig)

# ---------- 2. joint convergence: dCL -> 0 as mesh refines, log scale ----------
conv = R['rect_convergence']
fig, ax = plt.subplots(figsize=(6.4,3.9), dpi=160)
x = [d['paineis_por_semiasa'] for d in conv]
y = [d['dCL'] for d in conv]
ax.plot(x, y, color=AQUA, lw=2, marker='o', ms=6, mec=SURF, mew=1.2, zorder=3)
for xi, yi in zip(x, y):
    ax.annotate(f'{yi:.1e}', (xi, yi), textcoords='offset points', xytext=(0,9),
                ha='center', fontsize=8, color=INK)
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.xaxis.set_major_formatter(INT)
ax.set_xlabel('panels in span, per half-wing (uniform spacing in both)')
ax.set_ylabel('$|C_L^{main.py} - C_L^{AeroSandbox}|$')
ax.set_title('Joint convergence — rectangular wing AR = 8, α = 5°', pad=10, fontsize=10.5, loc='left')
style(ax)
fig.tight_layout()
fig.savefig(f'{OUT}/asb_convergence.png')
plt.close(fig)

# ---------- 3. lateral coupling: CY, Cl, Cn vs beta ----------
rowsb = R['full_beta']
be = [r['beta'] for r in rowsb]
fig, axs = plt.subplots(1, 3, figsize=(11,3.6), dpi=160)
for ax, key, ylabel in zip(axs, ('CY','Cl','Cn'), ('$C_Y$', '$C_l$', '$C_n$')):
    mine_v = [r['mine'][key] for r in rowsb]
    their_v = [r['theirs'][key] for r in rowsb]
    ax.plot(be, mine_v, color=BLUE, lw=2, marker='o', ms=5, mec=SURF, mew=1, label='main.py', zorder=3)
    ax.plot(be, their_v, color=ORANGE, lw=1.6, ls=(0,(5,2)), marker='s', ms=4.5, mec=SURF, mew=1,
             label='AeroSandbox', zorder=2)
    ax.axhline(0, color=GRID, lw=0.8, zorder=1)
    ax.set_xlabel('β (°)'); ax.set_ylabel(ylabel)
    style(ax)
axs[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='upper right')
fig.suptitle('Lateral coupling, α = 5° — main.py × AeroSandbox VLM', fontsize=11, x=0.02, ha='left', color=INK)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(f'{OUT}/asb_lateral.png')
plt.close(fig)

print('ok ->', OUT)
