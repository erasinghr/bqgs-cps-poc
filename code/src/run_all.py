"""End-to-end BQGS proof-of-concept experiment on HAI 21.03 (10 seeds).
Writes results/seed_<k>.json and results/selection.json."""
import os, sys, json, time, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from data import load
from common import Surprisal, best_f1, evaluate, pso, hold_to_seconds
from layer1 import Layer1
from layer2 import VAE, USAD
import layer3 as L3

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(10))
N3, SHOTS = 300, 1024
NOISE = {"nisq": (1e-3, 1e-2), "nisq_high": (3e-3, 3e-2)}

D = load()
# ---------------------------------------------------------------- Layer-3 shared preparation
etr, Ftr = L3.win_feats(D["Xtr"], D["str_"]); eca, Fca = L3.win_feats(D["Xcal"], D["scal"])
eva, Fva = L3.win_feats(D["Xva"], D["sva"]); ete, Fte = L3.win_feats(D["Xte"], D["ste"])
emb = L3.Embed().fit(Ftr)
pf = os.path.join(L3.CACHE, "l3_pool.npy")
if not os.path.exists(pf): np.save(pf, np.random.default_rng(0).choice(len(Ftr), 2000, replace=False))
pool = np.load(pf)
U = dict(tr=emb(Ftr[pool]), cal=emb(Fca), va=emb(Fva), te=emb(Fte))
tags = dict(tr="trpool", cal="cal", va="va", te="te")

sel_f = os.path.join(OUT, "selection.json")
if os.path.exists(sel_f):
    SEL = json.load(open(sel_f))
else:  # bandwidth selection on validation only (seed-independent, 3 random subsamples)
    r = np.random.default_rng(123); subs = [r.choice(len(pool), N3, replace=False) for _ in range(3)]
    qs = {}
    for c in [0.05, 0.1, 0.25, 0.5, 1.0]:
        Str = L3.statevectors(U["tr"], c, tags["tr"]); Sva = L3.statevectors(U["va"], c, tags["va"])
        qs[c] = float(np.mean([best_f1(hold_to_seconds(D["sva"], eva, L3.ocsvm_scores(L3.k_sv(Str[i], Str[i]), L3.k_sv(Sva, Str[i]))), D["yva"])[0] for i in subs]))
    gs = {}
    for g in [0.25, 0.5, 1, 2, 4, 8, 16]:
        gs[g] = float(np.mean([best_f1(hold_to_seconds(D["sva"], eva, L3.ocsvm_scores(L3.rbf(U["tr"][i], U["tr"][i], g), L3.rbf(U["va"], U["tr"][i], g))), D["yva"])[0] for i in subs]))
    SEL = dict(c=max(qs, key=qs.get), gamma=max(gs, key=gs.get), quantum_grid=qs, rbf_grid=gs)
    json.dump(SEL, open(sel_f, "w"), indent=1)
c, gam = float(SEL["c"]), float(SEL["gamma"])
SV = {k: L3.statevectors(U[k], c, tags[k]) for k in U}
DM = {n: {k: L3.density_matrices(U[k], c, *NOISE[n], tags[k]) for k in U} for n in NOISE}
print("L3 ready: c=%s gamma=%s" % (c, gam), flush=True)

SPLITS = [("cal", "Xcal", "scal", eca), ("va", "Xva", "sva", eva), ("te", "Xte", "ste", ete)]

def l3_scores(kind, idx, rng):
    """Window scores for cal/va/te -> held to seconds."""
    out = {}
    if kind == "Q":
        Ktr = L3.k_sv(SV["tr"][idx], SV["tr"][idx]); kf = lambda k: L3.k_sv(SV[k], SV["tr"][idx])
    elif kind == "C":
        Ktr = L3.rbf(U["tr"][idx], U["tr"][idx], gam); kf = lambda k: L3.rbf(U[k], U["tr"][idx], gam)
    else:
        dm = DM[kind]
        Ktr = L3.psd(L3.shot(L3.k_dm(dm["tr"][idx], dm["tr"][idx]), SHOTS, rng))
        kf = lambda k: L3.shot(L3.k_dm(dm[k], dm["tr"][idx]), SHOTS, rng)
    from sklearn.svm import OneClassSVM
    m = OneClassSVM(kernel="precomputed", nu=0.05).fit(Ktr)
    for name, _, sg, ends in SPLITS:
        out[name] = hold_to_seconds(D[sg], ends, -m.decision_function(kf(name)))
    return out

def run_seed(seed):
    rng = np.random.default_rng(seed); res = dict(seed=seed, timing={}); S = {}
    t = time.time()
    l1 = Layer1().fit_aco(D["Xtr"], D["str_"], D["Xva"], D["yva"], D["sva"], rng)
    S["L1"] = {n: l1.score(D[x], D[sg]) for n, x, sg, _ in SPLITS}
    res["aco_subset"] = [str(D["cols"][j]) for j in l1.S]; res["aco_hist"] = l1.aco_hist
    res["timing"]["L1_fit_s"] = time.time() - t; t = time.time()
    vae = VAE().fit(D["Xtr"], D["str_"], seed)
    S["L2"] = {n: vae.score(D[x], D[sg]) for n, x, sg, _ in SPLITS}
    res["vae_loss"] = vae.loss_hist; res["timing"]["L2_fit_s"] = time.time() - t; t = time.time()
    idx = rng.choice(len(pool), N3, replace=False)
    for k, kind in [("L3Q", "Q"), ("L3C", "C"), ("L3Qn", "nisq"), ("L3Qn2", "nisq_high")]:
        S[k] = l3_scores(kind, idx, rng)
    res["timing"]["L3_s"] = time.time() - t; t = time.time()
    # baselines
    sub = rng.choice(len(D["Xtr"]), 100000, replace=False)
    iso = IsolationForest(n_estimators=200, max_samples=256, random_state=seed).fit(D["Xtr"][sub])
    S["B_IF"] = {n: -iso.score_samples(np.clip(D[x], -10, 10)) for n, x, _, _ in SPLITS}
    pca = PCA(0.95, random_state=0).fit(D["Xtr"][sub])
    def spe(X):
        X = np.clip(X, -10, 10); return np.sum((X - pca.inverse_transform(pca.transform(X))) ** 2, 1)
    S["B_PCA"] = {n: spe(D[x]) for n, x, _, _ in SPLITS}
    us = USAD().fit(D["Xtr"], D["str_"], seed)
    S["B_USAD"] = {n: us.score(D[x], D[sg]) for n, x, sg, _ in SPLITS}
    res["timing"]["baselines_s"] = time.time() - t; t = time.time()
    # calibration to surprisal
    Z = {k: {n: Surprisal(v["cal"])(v[n]) for n in ("va", "te")} for k, v in S.items()}
    variants = {
        "L1 only": ["L1"], "L2 only": ["L2"], "L3 only (quantum)": ["L3Q"], "L3 only (classical)": ["L3C"],
        "L3 only (quantum, NISQ p2=0.01)": ["L3Qn"], "L3 only (quantum, NISQ p2=0.03)": ["L3Qn2"],
        "BQGS w/o L3 quantum (L1+L2)": ["L1", "L2"], "BQGS w/o generative L2 (L1+L3)": ["L1", "L3Q"],
        "BQGS w/o edge L1 (L2+L3)": ["L2", "L3Q"], "BQGS full (noiseless quantum)": ["L1", "L2", "L3Q"],
        "BQGS full (NISQ noise, p2=0.01)": ["L1", "L2", "L3Qn"], "BQGS full (NISQ noise, p2=0.03)": ["L1", "L2", "L3Qn2"],
        "BQGS full (classical fallback)": ["L1", "L2", "L3C"], "BQGS full, uniform weights": ["L1", "L2", "L3Q"],
        "uniform: w/o L3 (L1+L2)": ["L1", "L2"], "uniform: w/o L2 (L1+L3)": ["L1", "L3Q"], "uniform: w/o L1 (L2+L3)": ["L2", "L3Q"],
        "uniform: full, classical fallback": ["L1", "L2", "L3C"], "uniform: full, NISQ p2=0.01": ["L1", "L2", "L3Qn"],
        "uniform: full, NISQ p2=0.03": ["L1", "L2", "L3Qn2"],
        "Isolation Forest": ["B_IF"], "PCA-SPE": ["B_PCA"], "USAD": ["B_USAD"]}
    res["variants"] = {}
    for name, comps in variants.items():
        Va = np.stack([Z[k]["va"] for k in comps], 1); Te = np.stack([Z[k]["te"] for k in comps], 1)
        if len(comps) == 1 or "uniform" in name:
            w = np.ones(len(comps)) / len(comps); hist = []
        else:
            obj = lambda p: best_f1(Va @ (p / max(p.sum(), 1e-9)), D["yva"])[0]
            w, _, hist = pso(obj, len(comps), [0] * len(comps), [1] * len(comps), np.random.default_rng(seed))
            w = w / w.sum()
        fva, thr = best_f1(Va @ w, D["yva"])
        m = evaluate(Te @ w > thr, D["yte"]); m["val_f1"] = float(fva)
        m["weights"] = dict(zip(comps, map(float, w))); m["threshold"] = float(thr); m["pso_hist"] = list(map(float, hist))
        res["variants"][name] = m
    res["timing"]["fusion_s"] = time.time() - t
    np.savez_compressed(os.path.join(OUT, f"surprisal_seed{seed}.npz"),
                        **{f"{k}__{n}": Z[k][n].astype(np.float32) for k in Z for n in ("va", "te")})
    json.dump(res, open(os.path.join(OUT, f"seed_{seed}.json"), "w"), indent=1)
    v = res["variants"]
    print(f"seed {seed}: " + "; ".join(f"{k}={v[k]['f1']:.3f}" for k in ["L1 only", "L2 only", "BQGS w/o L3 quantum (L1+L2)", "BQGS full (noiseless quantum)", "BQGS full (classical fallback)"]), res["timing"], flush=True)

for s in SEEDS:
    run_seed(s)
