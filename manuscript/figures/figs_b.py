from diag import *
# ---------------- Fig 7 BQGS architecture (revised: quantum as optional enhancement with classical fallback)
f, ax = fig(6.3, 4.6)
X0, WL = 0.95, 4.4
def layer(y, h, title, col, items):
    box(ax, X0, y, WL, h, '', col, r=0.08, lw=1.3)
    ax.text(X0 + WL / 2, y + h - 0.17, title, ha='center', va='center', fontsize=8.3, fontweight='bold', color=C[col][1])
    bw = (WL - 0.2 - 3 * 0.1) / 4
    for i, (t, c, ls) in enumerate(items):
        box(ax, X0 + 0.1 + i * (bw + 0.1), y + 0.1, bw, h - 0.42, t, c, fs=8, ls=ls)
layer(3.0, 1.5, 'LAYER 3: CLOUD CORE — "The Strategic Mind"', 'blue',
      [('GAN zero-day\nsimulator vs\ndigital twin', 'teal', '-'), ('Strategic\npolicy agent\n(global min R)', 'green', '-'),
       ('Classical\nkernel / deep\ndetector\n(default)', 'yellow', '-'), ('Quantum\nkernel\n(optional;\nfallback to\nclassical)', 'purple', '--')])
layer(1.55, 1.2, 'LAYER 2: FOG / GATEWAY — "The Tactical Brain"', 'yellow',
      [('Agentic\ncommander\n(SLM/LLM)', 'blue', '-'), ('Generative\nengineer\n(SLM, VAE)', 'green', '-'), ('IACO load\nbalancer', 'yellow', '-'), ('PQC-secured\nchannels', 'purple', '-')])
layer(0.1, 1.2, 'LAYER 1: NANO-EDGE — "The Reflex Layer"', 'red',
      [('ACO/SWSO\nswarm agents', 'red', '-'), ('Physics-\ninformed\nresidual', 'purple', '-'), ('Lightweight\nPQC\n(ML-KEM)', 'grey', '-'), ('Reflex\nisolation', 'blue', '-')])
for y0, y1 in [(1.3, 1.55), (2.75, 3.0)]:
    arrow(ax, (0.8, y0), (0.8, y1), col='#B03A2E', lw=1.1); arrow(ax, (5.5, y1), (5.5, y0), col='#2E7D32', lw=1.1)
ax.text(0.72, 1.43, 'alerts,\ntelemetry', ha='right', va='center', fontsize=7, color='#B03A2E')
ax.text(0.72, 2.87, 'intelligence\npatterns', ha='right', va='center', fontsize=7, color='#B03A2E')
ax.text(5.58, 1.43, 'patches,\nrules', ha='left', va='center', fontsize=7, color='#2E7D32')
ax.text(5.58, 2.87, 'policies,\n"vaccines"', ha='left', va='center', fontsize=7, color='#2E7D32')
save(f, 'fig07.png')

# ---------------- Fig 8 continuous loop
f, ax = fig(6.3, 2.9)
pos = [(0.15, 2.05, '1. Continuous\nmonitoring', 'blue'), (1.4, 2.05, '2. PQC\nencryption', 'purple'), (2.65, 2.05, '3. Swarm\nanalysis', 'yellow'),
       (3.9, 2.05, '4. Reflex\nisolation', 'red'), (5.1, 1.1, '5. IACO\nrouting', 'yellow'), (3.9, 0.15, '6. Agentic\nanalysis', 'blue'),
       (2.65, 0.15, '7. SLM patch\ngeneration', 'green'), (1.4, 0.15, '8. Patch\ndeployment', 'red'), (0.15, 0.15, '9. Strategic\nanalysis\n(classical/QML)', 'purple'),
       (0.15, 1.1, '10. GAN\nsimulation', 'blue')]
W, H = 1.05, 0.62
for x, y, t, c in pos: box(ax, x, y, W, H, t, c, fs=8)
for k in range(10):
    x0, y0, *_ = pos[k]; x1, y1, *_ = pos[(k + 1) % 10]
    p = (x0 + W / 2, y0 + H / 2); q = (x1 + W / 2, y1 + H / 2)
    import numpy as np
    d = np.array(q) - np.array(p); n = np.linalg.norm(d); u = d / n
    sx = W / 2 / max(abs(u[0]), 1e-9); sy = H / 2 / max(abs(u[1]), 1e-9); s_ = min(sx, sy)
    arrow(ax, tuple(np.array(p) + u * s_), tuple(np.array(q) - u * s_), col='#2F5D8A')
ax.text(2.9, 1.42, 'CONTINUOUS\nLEARNING LOOP', ha='center', va='center', fontsize=11, color='#7FB685', fontweight='bold')
save(f, 'fig08.png')

# ---------------- Fig 9 synergy map
f, ax = fig(5.0, 5.0)
cen = {'Generative AI': (2.5, 4.2, 'blue'), 'Agentic AI': (4.2, 2.5, 'green'), 'Nature-inspired\noptimization': (2.5, 0.8, 'yellow'), 'Quantum\ncomputing': (0.8, 2.5, 'purple')}
keys = list(cen)
lab = {(0, 1): 'Red-team\nagents', (1, 2): 'Autonomous\noptimization', (2, 3): 'Quantum\noptimization', (0, 3): 'Quantum\nGANs', (0, 2): 'Synthetic\ndata', (1, 3): 'IACO +\nPQC/QKD'}
for (i, j), t in lab.items():
    (x0, y0, _), (x1, y1, _) = cen[keys[i]], cen[keys[j]]
    ax.plot([x0, x1], [y0, y1], ls='--', color='#999999', lw=0.9, zorder=0)
for t, (x, y, c) in cen.items():
    ax.add_patch(Circle((x, y), 0.62, fc=C[c][0], ec=C[c][1], lw=1.6, zorder=1)); ax.text(x, y, t, ha='center', va='center', fontsize=8.5, fontweight='bold', zorder=2)
ax.add_patch(Circle((2.5, 2.5), 0.5, fc=C['red'][0], ec=C['red'][1], lw=1.4, zorder=1)); ax.text(2.5, 2.5, 'BQGS\nintegrated\nframework', ha='center', va='center', fontsize=8, color='#B03A2E', fontweight='bold', zorder=2)
for (i, j), t in lab.items():
    (x0, y0, _), (x1, y1, _) = cen[keys[i]], cen[keys[j]]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    if (i, j) == (0, 2): my = 3.3
    if (i, j) == (1, 3): mx, my = 3.5, 2.5
    ax.text(mx, my, t, ha='center', va='center', fontsize=8, zorder=3, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#999999', lw=0.6))
save(f, 'fig09.png')

# ---------------- Fig 11 roadmap
f, ax = fig(6.3, 2.5)
hs = [('H1: Simulation\nand validation', 'Docker, OpenPLC,\nQiskit Aer, public\ndatasets (HAI)', 'blue'), ('H2: Fog-layer\nintegration', 'SLM agents,\nML-KEM channels,\nIACO balancing', 'green'),
      ('H3: Quantum–\nclassical\nhybridization', 'Hybrid kernels,\nfederated learning,\nnoise-aware tests', 'yellow'), ('H4: Full BQGS\npilot', 'Testbed pilot,\nreal QPU access,\nsafety certification', 'red'),
      ('H5: Autonomous\necosystems', 'Self-evolving\ndefence, standard\nAPIs, regulation', 'purple')]
for i, (a, b, c) in enumerate(hs):
    x = 0.1 + i * 1.24
    box(ax, x, 1.6, 1.12, 0.7, a, c, fs=8, bold=True)
    box(ax, x, 0.1, 1.12, 0.78, b, c, fs=8)
    ax.plot([x + 0.56, x + 0.56], [0.88, 1.6], ls=':', color='#777777', lw=0.8)
arrow(ax, (0.2, 1.22), (6.1, 1.22), col='#1F3B5A', lw=1.6, ms=11)
ax.text(3.15, 1.33, 'Increasing technology readiness level →', ha='center', fontsize=8, fontweight='bold', color='#1F3B5A')
save(f, 'fig11.png')

# ---------------- Fig 12 PoC pipeline
f, ax = fig(6.3, 2.7)
box(ax, 0.1, 1.0, 1.0, 0.75, 'HAI 21.03\n57 process\nvariables, 1 Hz', 'grey', fs=8)
box(ax, 1.45, 1.95, 1.55, 0.6, 'L1 edge: ACO-selected\nsensors + LTI residual\n+ pheromone smoothing', 'red', fs=8)
box(ax, 1.45, 1.08, 1.55, 0.6, 'L2 fog: VAE score on\n10-s windows', 'green', fs=8)
box(ax, 1.45, 0.1, 1.55, 0.7, 'L3 cloud: 60-s window\nfeatures → 6-qubit ZZ\nkernel OCSVM (or RBF\nfallback; NISQ noise)', 'purple', fs=8)
box(ax, 3.35, 1.0, 1.15, 0.75, 'Calibration to\ntail surprisal on\nheld-out normal', 'blue', fs=8)
box(ax, 4.85, 1.0, 1.35, 0.75, 'PSO-tuned fusion\nS = Σ wᵢ zᵢ ;\nalarm if S > θ', 'yellow', fs=8)
for y in (2.25, 1.38, 0.45):
    arrow(ax, (1.1, 1.38), (1.45, y)); arrow(ax, (3.0, y), (3.35, 1.38))
arrow(ax, (4.5, 1.38), (4.85, 1.38))
ax.text(5.52, 0.75, 'thresholds/weights\ntuned on validation\n(test1–2); reported\non test3–5', ha='center', va='top', fontsize=8, color='#555555')
save(f, 'fig12.png')
print('ok')
