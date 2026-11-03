from diag import *
# ---------------- Fig 1 Purdue
f, ax = fig(6.3, 3.6)
levels = [('Level 4–5: Enterprise IT (ERP, e-mail, web)', 'blue'), ('Level 3: Manufacturing operations (MES, historian)', 'green'),
          ('Level 2: Supervisory control (SCADA, HMI)', 'yellow'), ('Level 1: Basic control (PLC, RTU, DCS)', 'red'),
          ('Level 0: Physical process (sensors, actuators, plant)', 'purple')]
for i, (t, c) in enumerate(levels):
    y = 3.0 - i * 0.66
    box(ax, 1.35, y, 3.6, 0.5, t, c, fs=8.5)
for i in range(4):
    arrow(ax, (3.15, 3.0 - i * 0.66), (3.15, 3.0 - (i + 1) * 0.66 + 0.5), both=True, lw=0.9)
box(ax, 5.1, 2.38, 1.1, 0.62, 'IT/OT\nconvergence\nzone', 'purple', fs=8, ls='--')
attacks = [(3.25, 'APT / phishing', 'L'), (1.92, 'TRITON/TRISIS', 'L'), (0.6, 'False data\ninjection', 'L'),
           (2.6, 'Lateral\nmovement', 'R2'), (1.27, 'Stuxnet', 'R')]
for y, t, s in attacks:
    if s == 'L': ax.text(1.25, y, '⚡ ' + t, ha='right', va='center', fontsize=8, color='#B03A2E', fontweight='bold')
    elif s == 'R': ax.text(5.05, y, '⚡ ' + t, ha='left', va='center', fontsize=8, color='#B03A2E', fontweight='bold')
ax.text(5.65, 3.25, '⚡ Lateral movement', ha='center', va='center', fontsize=8, color='#C25E00', fontweight='bold')
save(f, 'fig01.png')

# ---------------- Fig 2 PRISMA
f, ax = fig(5.2, 5.6)
steps = [('Identification\nRecords identified through database searching\n(n = 1,847)', 'blue'),
         ('Records after duplicates removed\n(n = 1,535)', 'green'),
         ('Screening\nRecords screened by title and abstract\n(n = 1,535)', 'blue'),
         ('Full-text articles assessed for eligibility\n(n = 289)', 'green'),
         ('Eligibility\nStudies assessed for quality\n(n = 162)', 'blue'),
         ('Included\nStudies included in the systematic review\n(n = 127)', 'green')]
excl = {2: 'Excluded\n(n = 1,246)', 3: 'Full-text excluded\n(n = 127)', 4: 'Quality threshold\nnot met (n = 35)'}
for i, (t, c) in enumerate(steps):
    y = 4.85 - i * 0.92
    box(ax, 0.25, y, 3.1, 0.62, t, c, fs=8)
    if i < 5: arrow(ax, (1.8, y), (1.8, y - 0.30))
    if i in excl:
        box(ax, 3.75, y + 0.06, 1.25, 0.5, excl[i], 'red', fs=8)
        arrow(ax, (3.35, y + 0.31), (3.75, y + 0.31), col='#B03A2E', ls=':')
save(f, 'fig02.png')

# ---------------- Fig 3 GAN framework
f, ax = fig(6.3, 2.75)
b1 = box(ax, 0.1, 1.55, 1.15, 0.6, 'Noise z\n~ p_z(z)', 'blue')
b2 = box(ax, 1.65, 1.55, 1.2, 0.6, 'Generator\nG(z)', 'blue')
b3 = box(ax, 3.3, 1.55, 1.3, 0.6, 'Discriminator\nD(x)', 'yellow')
b4 = box(ax, 5.0, 1.55, 1.2, 0.6, 'Real / fake\ndecision', 'green')
b5 = box(ax, 0.1, 0.25, 1.15, 0.7, 'Real CPS\nsensor data\np_data(x)', 'purple')
b6 = box(ax, 1.65, 0.25, 1.2, 0.7, 'Synthetic CPS\nattack data', 'green')
b7 = box(ax, 3.3, 0.25, 1.3, 0.7, 'IDS classifier', 'red')
b8 = box(ax, 5.0, 0.25, 1.2, 0.7, 'Anomaly\nalert', 'red')
arrow(ax, (1.25, 1.85), (1.65, 1.85)); arrow(ax, (2.85, 1.85), (3.3, 1.85)); arrow(ax, (4.6, 1.85), (5.0, 1.85))
arrow(ax, (0.68, 0.95), (0.68, 1.55)); ax.plot([0.68,0.68,3.95],[0.25,0.1,0.1],color='#333333',lw=1.0); arrow(ax, (3.95, 0.1), (3.95, 0.25)); arrow(ax, (2.25, 1.55), (2.25, 0.95))
arrow(ax, (2.85, 0.6), (3.3, 0.6)); arrow(ax, (4.6, 0.6), (5.0, 0.6))
arrow(ax, (3.95, 2.15), (2.25, 2.15), col='#B03A2E', rad=0.45)
ax.text(3.1, 2.62, 'minimax training loop', ha='center', fontsize=8, color='#B03A2E')
save(f, 'fig03.png')

# ---------------- Fig 4 agentic
f, ax = fig(6.3, 3.8)
box(ax, 2.15, 1.25, 2.0, 1.1, 'Agentic Commander\n\nLLM/SLM reasoning · planning\ntool use · memory\nhuman-in-the-loop gate', 'blue', fs=8.5)
ag = [(0.15, 2.55, 'Threat-hunting\nagent', 'green'), (2.5, 2.85, 'Incident-response\nagent', 'yellow'), (4.85, 2.55, 'Patch-generation\nagent (SLM)', 'red'),
      (0.15, 0.25, 'Network-monitor\nagent', 'blue'), (2.5, 0.05, 'Physics-validation\nagent', 'purple'), (4.85, 0.25, 'Forensics\nagent', 'yellow')]
for x, y, t, c in ag:
    box(ax, x, y, 1.3, 0.6, t, c, fs=8)
arrow(ax, (1.45, 2.85), (2.15, 2.15), both=True); arrow(ax, (3.15, 2.85), (3.15, 2.35), both=True); arrow(ax, (4.85, 2.85), (4.15, 2.15), both=True)
arrow(ax, (1.45, 0.55), (2.15, 1.45), both=True); arrow(ax, (3.15, 0.65), (3.15, 1.25), both=True); arrow(ax, (4.85, 0.55), (4.15, 1.45), both=True)
ax.text(3.15, 3.66, 'Authenticated agent messages (A2A-style, PQC-signed)', ha='center', fontsize=8, color='#555555')
save(f, 'fig04.png')

# ---------------- Fig 5 NI taxonomy
f, ax = fig(6.3, 3.0)
box(ax, 1.85, 2.45, 2.6, 0.45, 'Nature-inspired optimization', 'blue', bold=True)
cats = [('Swarm\nintelligence', ['ACO / IACO', 'PSO / BPSO', 'SWSO'], 'blue'), ('Evolutionary\nalgorithms', ['GA', 'GWO', 'DE'], 'green'),
        ('Bio-inspired\nimmune systems', ['AIS', 'Clonal selection', 'Digital antibodies'], 'yellow'), ('Hybrid\nmetaheuristics', ['ACO–PSO', 'GA–PSO', 'IACO–GBM'], 'red')]
for i, (t, items, c) in enumerate(cats):
    x = 0.12 + i * 1.55
    box(ax, x, 1.6, 1.4, 0.55, t, c, fs=8, bold=True)
    arrow(ax, (3.15, 2.45), (x + 0.7, 2.15), lw=0.8)
    for j, it in enumerate(items):
        box(ax, x + 0.15, 1.05 - j * 0.45, 1.1, 0.34, it, 'white', fs=8)
        if j == 0: arrow(ax, (x + 0.7, 1.6), (x + 0.7, 1.39), lw=0.8)
        else: arrow(ax, (x + 0.7, 1.05 - (j - 1) * 0.45), (x + 0.7, 1.39 - j * 0.45), lw=0.8)
save(f, 'fig05.png')

# ---------------- Fig 6 quantum
f, ax = fig(6.3, 2.6)
box(ax, 1.75, 2.05, 2.8, 0.42, 'Quantum technologies for CPS security', 'purple', bold=True, fs=8)
q = [('QKD\nBB84, E91', 'Secure key\nexchange on\nhigh-value links', 'Mature (cost-limited)'),
     ('PQC\nML-KEM, ML-DSA,\nSLH-DSA', 'Protection against\nharvest-now,\ndecrypt-later', 'Standardised 2024'),
     ('QML\nQSVM, VQC', 'Threat classification\n(optional enhancement)', 'NISQ-limited'),
     ('QRNG', 'Entropy for keys,\nnonces, challenges', 'Deployable now')]
for i, (a, b, m) in enumerate(q):
    x = 0.12 + i * 1.55
    box(ax, x, 1.15, 1.4, 0.6, a, 'purple', fs=8)
    arrow(ax, (3.15, 2.05), (x + 0.7, 1.75), col='#6A3D9A', lw=0.8)
    box(ax, x, 0.3, 1.4, 0.6, b, 'green', fs=8)
    arrow(ax, (x + 0.7, 1.15), (x + 0.7, 0.9), col='#2E7D32', ls='--', lw=0.8)
    ax.text(x + 0.7, 0.13, m, ha='center', fontsize=8, style='italic', color='#444444')
save(f, 'fig06.png')
print('ok')
