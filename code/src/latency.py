"""Per-decision computational latency and memory footprint of each layer (single CPU thread)."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, platform, subprocess, numpy as np
from data import load
from layer1 import Layer1, Gram
from layer2 import VAE
import layer3 as L3
from common import window_index, make_windows
OUT = os.path.join(os.path.dirname(__file__), "..", "results")

def timeit(fn, n):
    fn(); t = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t) / n * 1e3

D = load(); r0 = json.load(open(os.path.join(OUT, "seed_0.json")))
cols = list(D["cols"]); S = [cols.index(c) for c in r0["aco_subset"]]
g = Gram(D["Xtr"], D["str_"]); Th, sig = g.fit(S)
x_prev = D["Xte"][100, S].astype(np.float64); x = D["Xte"][101, S].astype(np.float64); tau = [0.0]
def l1():
    r = x - x_prev @ Th; d = np.sum((r / sig) ** 2); tau[0] = 0.8 * tau[0] + 0.2 * d
vae = VAE().fit(D["Xtr"], D["str_"], 0)
w = make_windows(D["Xte"], np.array([500]), 10)
def l2(): vae.score_windows(w)
nparams = sum(int(np.prod(a.shape)) + int(np.prod(b.shape)) for part in vae.params.values() for a, b in part)
etr, Ftr = L3.win_feats(D["Xtr"], D["str_"]); emb = L3.Embed().fit(Ftr)
pool = np.load(os.path.join(L3.CACHE, "l3_pool.npy"))  # created by run_all.py; Utr = emb(Ftr[pool])
sel = json.load(open(os.path.join(OUT, "selection.json"))); c, gam = float(sel["c"]), float(sel["gamma"])
Str = L3.statevectors(Utr, c, "trpool"); idx = np.random.default_rng(0).choice(len(pool), 300, replace=False)
from sklearn.svm import OneClassSVM
mq = OneClassSVM(kernel="precomputed", nu=0.05).fit(L3.k_sv(Str[idx], Str[idx]))
mc = OneClassSVM(kernel="precomputed", nu=0.05).fit(L3.rbf(Utr[idx], Utr[idx], gam))
ete, Fte = L3.win_feats(D["Xte"], D["ste"]); u = emb(Fte[:1])
from qiskit.quantum_info import Statevector
fm = L3._fm()
def l3q():
    sv = Statevector(fm.assign_parameters(u[0] * c * np.pi)).data[None]
    mq.decision_function(L3.k_sv(sv, Str[idx]))
def l3c(): mc.decision_function(L3.rbf(u, Utr[idx], gam))
def l3feat(): L3.win_feats(D["Xte"][:60], np.zeros(60, np.int16), stride=60)
from qiskit_aer import AerSimulator
from qiskit import transpile
sim = AerSimulator(method="density_matrix", noise_model=L3.noise_model(1e-3, 1e-2)); base = transpile(fm, sim, optimization_level=0); base.save_density_matrix()
def l3n(): sim.run(base.assign_parameters(u[0] * c * np.pi)).result()
cpu = subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout.strip()
res = dict(cpu=cpu, python=platform.python_version(),
           L1_ms=timeit(l1, 20000), L2_ms=timeit(l2, 2000), L3_features_ms=timeit(l3feat, 500),
           L3_quantum_statevector_ms=timeit(l3q, 300), L3_quantum_noisy_densitymatrix_sim_ms=timeit(l3n, 100), L3_classical_ms=timeit(l3c, 2000),
           L1_bytes_float32=int((len(S) ** 2 + len(S)) * 4), L1_n_vars=len(S), L2_params=nparams, L2_bytes_float32=nparams * 4,
           L3_quantum_sv_bytes=int(300 * 64 * 16), L3_classical_bytes=int(300 * 6 * 8))
print(res); json.dump(res, open(os.path.join(OUT, "latency.json"), "w"), indent=1)
