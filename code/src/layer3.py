"""Layer 3 (cloud, strategic) - quantum-kernel enhancement with classical fallback.

Window features (60-s windows, 30-s stride): per-variable mean and standard deviation ->
PCA (fit on normal training windows) -> n_q = 6 components -> affine map to [0, c*pi]
(c = kernel bandwidth, Shaydulin & Wild 2022) -> ZZ feature map (2 reps, linear entanglement).
Kernel: K(x,x') = |<phi(x)|phi(x')>|^2 (noiseless statevector, Qiskit) or, under a NISQ noise
model, the SWAP-test estimate Tr[rho(x) rho(x')] from Aer density-matrix simulation with
depolarizing gate errors, plus binomial shot noise. Detector: one-class SVM (precomputed kernel).
Classical fallback: the same pipeline with an RBF kernel on the same 6 features.
"""
import os, numpy as np, warnings
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from common import window_index
warnings.filterwarnings("ignore")
NQ, W3, STRIDE3 = 6, 60, 30
CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")

def win_feats(X, seg, stride=STRIDE3):
    ends = window_index(seg, W3, stride)
    F = np.stack([np.r_[X[e - W3 + 1:e + 1].mean(0), X[e - W3 + 1:e + 1].std(0)] for e in ends])
    return ends, np.clip(F, -10, 10)

class Embed:
    def fit(self, Ftr):
        self.pca = PCA(NQ, random_state=0).fit(Ftr); P = self.pca.transform(Ftr)
        self.lo, self.hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0); return self
    def __call__(self, F):
        P = self.pca.transform(F)
        return np.clip((P - self.lo) / (self.hi - self.lo), -0.5, 1.5)   # ~[0,1] for normal data

# ------------------------------------------------------------------ quantum states
def _fm():
    from qiskit.circuit.library import zz_feature_map
    return zz_feature_map(NQ, reps=2, entanglement="linear")

def statevectors(U, c, tag):
    f = os.path.join(CACHE, f"sv_{tag}_c{c}.npy")
    if os.path.exists(f): return np.load(f)
    from qiskit.quantum_info import Statevector
    fm = _fm(); S = np.stack([Statevector(fm.assign_parameters(u * c * np.pi)).data for u in U]).astype(np.complex128)
    np.save(f, S); return S

def noise_model(p1, p2):
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["u", "h", "p", "rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm

def density_matrices(U, c, p1, p2, tag, batch=500):
    f = os.path.join(CACHE, f"dm_{tag}_c{c}_p{p1}_{p2}.npy")
    if os.path.exists(f): return np.load(f)
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    sim = AerSimulator(method="density_matrix", noise_model=noise_model(p1, p2))
    base = transpile(_fm(), sim, optimization_level=0); base.save_density_matrix()
    out = []
    for i in range(0, len(U), batch):
        qcs = [base.assign_parameters(u * c * np.pi) for u in U[i:i + batch]]
        res = sim.run(qcs).result()
        out += [np.asarray(res.data(j)["density_matrix"]).reshape(-1).astype(np.complex64) for j in range(len(qcs))]
    D = np.stack(out); np.save(f, D); return D

def k_sv(A, B): return np.abs(A.conj() @ B.T) ** 2
def k_dm(A, B): return np.real(A.conj() @ B.T).astype(np.float64)       # Tr[rho_a rho_b]
def shot(K, shots, rng): return rng.binomial(shots, np.clip(K, 0, 1)) / shots
def psd(K):
    w, V = np.linalg.eigh((K + K.T) / 2); return (V * np.maximum(w, 0)) @ V.T

def ocsvm_scores(Ktr, Kte, nu=0.05):
    m = OneClassSVM(kernel="precomputed", nu=nu).fit(Ktr)
    return -m.decision_function(Kte)

def rbf(A, B, gamma):
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(d, 0))
