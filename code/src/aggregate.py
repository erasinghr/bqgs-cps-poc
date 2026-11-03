"""Aggregate 10-seed results: mean +/- SD, Wilcoxon signed-rank tests with Holm correction,
summary JSON/CSV and figures."""
import os, json, glob, numpy as np
from scipy.stats import wilcoxon
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "seed_*.json")), key=lambda s: int(s.split('_')[-1][:-5]))]
names = list(R[0]["variants"].keys()); ref = "BQGS full (noiseless quantum)"
M = ["precision", "recall", "f1", "fpr", "event_dr", "ttd_s", "val_f1"]
summ = {}
for n in names:
    d = {m: np.array([r["variants"][n][m] for r in R], float) for m in M}
    summ[n] = {m: [float(np.nanmean(v)), float(np.nanstd(v, ddof=1))] for m, v in d.items()}
    summ[n]["f1_all"] = d["f1"].tolist()
    summ[n]["weights_mean"] = {k: float(np.mean([r["variants"][n]["weights"][k] for r in R])) for k in R[0]["variants"][n]["weights"]}
# paired two-sided Wilcoxon tests of F1, Holm-adjusted within each family
def wil(x, y):
    x, y = np.array(summ[x]["f1_all"]), np.array(summ[y]["f1_all"])
    return 1.0 if np.allclose(x, y) else float(wilcoxon(x, y, zero_method="wilcox").pvalue)
def holm(pairs):
    raw = {p: wil(*p) for p in pairs}; order = sorted(raw, key=raw.get); m = len(order); adj = {}; prev = 0
    for i, p in enumerate(order):
        v = min(1.0, max(prev, (m - i) * raw[p])); adj[p] = v; prev = v
    return {f"{p[0]} vs {p[1]}": dict(p_raw=raw[p], p_holm=adj[p], diff=summ[p[0]]["f1"][0] - summ[p[1]]["f1"][0]) for p in pairs}
UF = "BQGS full, uniform weights"
fams = {
 "pso_ablation": [(ref, "L1 only"), (ref, "L2 only"), (ref, "BQGS w/o L3 quantum (L1+L2)"), (ref, "BQGS w/o generative L2 (L1+L3)"), (ref, "BQGS w/o edge L1 (L2+L3)")],
 "uniform_ablation": [(UF, "L1 only"), (UF, "L2 only"), (UF, "uniform: w/o L3 (L1+L2)"), (UF, "uniform: w/o L2 (L1+L3)"), (UF, "uniform: w/o L1 (L2+L3)"), (UF, ref)],
 "quantum": [(UF, "uniform: full, classical fallback"), (UF, "uniform: full, NISQ p2=0.01"), (UF, "uniform: full, NISQ p2=0.03"),
             ("L3 only (quantum)", "L3 only (classical)"), ("L3 only (quantum)", "L3 only (quantum, NISQ p2=0.01)"), ("L3 only (quantum)", "L3 only (quantum, NISQ p2=0.03)"),
             (ref, "BQGS full (classical fallback)")],
 "baselines": [(UF, "PCA-SPE"), (UF, "USAD"), (UF, "Isolation Forest"), (ref, "PCA-SPE")],
}
summ["_tests"] = {k: holm(v) for k, v in fams.items()}
for k, v in summ["_tests"].items():
    print(k)
    for n, d in v.items(): print(f"   {n:75s} diff={d['diff']:+.3f} p={d['p_raw']:.4f} holm={d['p_holm']:.4f}")
summ["_meta"] = dict(n_seeds=len(R), aco_subsets=[r["aco_subset"] for r in R], timing_mean={k: float(np.mean([r["timing"][k] for r in R])) for k in R[0]["timing"]})
json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
with open(os.path.join(OUT, "summary.csv"), "w") as f:
    f.write("variant," + ",".join(f"{m}_mean,{m}_sd" for m in M) + "\n")
    for n in names:
        f.write(f'"{n}",' + ",".join(f"{summ[n][m][0]:.4f},{summ[n][m][1]:.4f}" for m in M) + "\n")
for n in names:
    s = summ[n]; print(f"{n:40s} F1 {s['f1'][0]:.3f}±{s['f1'][1]:.3f}  P {s['precision'][0]:.3f} R {s['recall'][0]:.3f} FPR {s['fpr'][0]:.4f}±{s['fpr'][1]:.4f} evDR {s['event_dr'][0]:.3f} TTD {s['ttd_s'][0]:.1f}±{s['ttd_s'][1]:.1f} valF1 {s['val_f1'][0]:.3f}")

# ---------------- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
show = [("L1 only", "L1 only"), ("L2 only", "L2 only"), ("L3 only (classical)", "L3 classical"), ("L3 only (quantum)", "L3 quantum"),
        ("BQGS w/o L3 quantum (L1+L2)", "PSO w/o L3"), ("BQGS w/o generative L2 (L1+L3)", "PSO w/o L2"), ("BQGS w/o edge L1 (L2+L3)", "PSO w/o L1"),
        ("BQGS full (noiseless quantum)", "PSO full"), ("uniform: w/o L3 (L1+L2)", "EW w/o L3"), ("uniform: w/o L2 (L1+L3)", "EW w/o L2"),
        ("uniform: w/o L1 (L2+L3)", "EW w/o L1"), ("BQGS full, uniform weights", "EW full"),
        ("PCA-SPE", "PCA-SPE"), ("USAD", "USAD"), ("Isolation Forest", "Isolation Forest")]
cols = ["#B03A2E", "#2E7D32", "#B8860B", "#6A3D9A"] + ["#7FA7D0"] * 3 + ["#2F5D8A"] + ["#8CC08F"] * 3 + ["#1E6B24"] + ["#7F7F7F"] * 3
f, ax = plt.subplots(figsize=(6.3, 3.4))
x = np.arange(len(show)); mu = [summ[k]["f1"][0] for k, _ in show]; sd = [summ[k]["f1"][1] for k, _ in show]
ax.bar(x, mu, yerr=sd, color=cols, capsize=2.5, width=0.7, error_kw=dict(lw=0.8))
for i, (k, _) in enumerate(show):
    pts = summ[k]["f1_all"]; ax.scatter(np.full(len(pts), i) + np.linspace(-0.18, 0.18, len(pts)), pts, s=3, color="black", zorder=3)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in show], fontsize=8, rotation=40, ha="right", rotation_mode="anchor")
ax.set_ylabel("Test F1-score (mean ± SD, 10 seeds)", fontsize=8.5); ax.set_ylim(0, 0.85)
ax.grid(axis="y", lw=0.4, alpha=0.6); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
f.tight_layout(); f.savefig(os.path.join(FIG, "fig13.png"), dpi=600); plt.close(f)

sel = json.load(open(os.path.join(OUT, "selection.json")))
f, axs = plt.subplots(1, 2, figsize=(6.3, 3.1))
ax = axs[0]
for r in R:
    h = r["variants"]["BQGS full (noiseless quantum)"]["pso_hist"]; ax.plot(range(len(h)), h, color="#2F5D8A", lw=0.8, alpha=0.7)
    a = r["aco_hist"]; ax.plot(range(len(a)), a, color="#B03A2E", lw=0.8, alpha=0.7)
ax.plot([], [], color="#2F5D8A", label="PSO fusion weights (full BQGS)"); ax.plot([], [], color="#B03A2E", label="ACO feature subset (Layer 1)")
ax.set_xlabel("Iteration"); ax.set_ylabel("Best validation F1-score"); ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1); ax.set_title("(a) Convergence (10 seeds)", fontsize=8.5)
ax = axs[1]
cs = sorted(sel["quantum_grid"], key=float); ax.plot([float(c) for c in cs], [sel["quantum_grid"][c] for c in cs], "o-", color="#6A3D9A", label="Quantum kernel vs bandwidth c")
g = sorted(sel["rbf_grid"], key=float); best = max(sel["rbf_grid"].values())
ax.axhline(best, color="#B8860B", ls="--", lw=1, label="Best classical RBF kernel")
ax.set_xscale("log"); ax.set_xlabel("Feature-map bandwidth c"); ax.set_ylabel("Layer-3 validation F1-score"); ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1); ax.set_title("(b) Quantum-kernel bandwidth", fontsize=8.5)
for a in axs:
    for s in ["top", "right"]: a.spines[s].set_visible(False)
f.tight_layout(); f.savefig(os.path.join(FIG, "fig14.png"), dpi=600); plt.close(f)
print("figures written")
