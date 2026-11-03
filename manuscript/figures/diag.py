import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5})
C = dict(blue=('#DCEAF7', '#2F5D8A'), green=('#DFF2E1', '#2E7D32'), yellow=('#FFF4CC', '#B8860B'),
         red=('#FBE3E1', '#B03A2E'), purple=('#ECE3F5', '#6A3D9A'), grey=('#EFEFEF', '#555555'),
         teal=('#DDF1EF', '#1F7A70'), orange=('#FDEBD8', '#C25E00'), white=('#FFFFFF', '#555555'))

def fig(w, h):
    f = plt.figure(figsize=(w, h)); ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis('off'); return f, ax

def box(ax, x, y, w, h, text, col='blue', fs=8.5, bold=False, r=0.06, lw=1.0, ls='-', tc=None, ha='center'):
    fs = max(fs, 8.0); fc, ec = C[col]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls))
    tx = x + w / 2 if ha == 'center' else x + 0.08
    ax.text(tx, y + h / 2, text, ha=ha, va='center', fontsize=fs, fontweight='bold' if bold else 'normal', color=tc or '#111111', linespacing=1.15)
    return (x, y, w, h)

def arrow(ax, p, q, col='#333333', lw=1.0, ls='-', both=False, rad=0.0, ms=8):
    a = FancyArrowPatch(p, q, arrowstyle='<|-|>' if both else '-|>', mutation_scale=ms, color=col, lw=lw, ls=ls,
                        connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0)
    ax.add_patch(a)

def save(f, name):
    f.savefig(name, dpi=600, facecolor='white'); plt.close(f)
