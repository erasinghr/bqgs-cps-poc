"""Layer 1 (edge, 'reflex'): ACO-selected sensor subset + linear (LTI/VAR(1)) one-step
residual detector with pheromone-style evaporative smoothing.

Model (identified by ridge regression on normal data):   x(t+1) = Theta^T x(t) + e(t)
Residual: r(t) = x(t) - Theta^T x(t-1); instantaneous score d(t) = sum_i r_i(t)^2 / sigma_i^2
(chi-square-type statistic, cf. Sec. 2.1). Pheromone trail: tau(t) = (1-rho) tau(t-1) + rho d(t).
"""
import numpy as np
from common import best_f1

class Gram:
    """Sufficient statistics of consecutive pairs within segments, for all variables."""
    def __init__(self, X, seg):
        same = seg[1:] == seg[:-1]
        P = X[:-1][same].astype(np.float64); N = X[1:][same].astype(np.float64)
        self.G = P.T @ P; self.C = P.T @ N; self.A = N.T @ N; self.n = len(P)

    def fit(self, S, lam=1e-3):
        S = np.asarray(S); G = self.G[np.ix_(S, S)]; C = self.C[np.ix_(S, S)]; A = self.A[np.ix_(S, S)]
        Th = np.linalg.solve(G + lam * self.n * np.eye(len(S)), C)
        R = (A - Th.T @ C - C.T @ Th + Th.T @ G @ Th) / self.n     # residual second moment
        return Th, np.sqrt(np.maximum(np.diag(R), 1e-10))

def residual_score(X, seg, S, Th, sig):
    Z = X[:, S].astype(np.float64)
    r = np.zeros_like(Z); r[1:] = Z[1:] - Z[:-1] @ Th
    first = np.r_[True, seg[1:] != seg[:-1]]; r[first] = 0.0
    return np.sum((r / sig) ** 2, axis=1)

def pheromone(d, seg, rho):
    tau = np.empty_like(d); prev = 0.0
    first = np.r_[True, seg[1:] != seg[:-1]]
    for t in range(len(d)):                     # simple loop is fine at 1 Hz scale
        if first[t]: prev = d[t]
        prev = (1 - rho) * prev + rho * d[t]; tau[t] = prev
    return tau

def _pher_fast(d, seg, rho):
    # vectorised EWMA via scipy.signal.lfilter per segment
    from scipy.signal import lfilter
    out = np.empty_like(d)
    for s in np.unique(seg):
        m = seg == s; x = d[m]
        out[m] = lfilter([rho], [1, -(1 - rho)], x, zi=[(1 - rho) * x[0]])[0]
    return out

class Layer1:
    def __init__(self, K=10, rho=0.2, lam=1e-3):
        self.K, self.rho, self.lam = K, rho, lam

    def score(self, X, seg, S=None):
        S = self.S if S is None else S
        Th, sig = self.gram.fit(S, self.lam)
        return _pher_fast(residual_score(X, seg, S, Th, sig), seg, self.rho)

    def fit_aco(self, Xtr, str_, Xva, yva, sva, rng, n_ants=20, iters=30, alpha=1.0, beta=2.0,
                evap=0.1, q_best=5, patience=10):
        """Binary ACO feature selection (Aghdam & Kabiri style) with fixed subset size K.
        Objective = best validation F1 of the Layer-1 detector built on the subset.
        Pheromone update: tau_j <- (1-evap) tau_j + sum_{k in best ants} f_k * 1[j in S_k]."""
        self.gram = Gram(Xtr, str_)
        p = Xtr.shape[1]
        # heuristic desirability eta_j: single-variable validation separability
        eta = np.array([best_f1(self.score(Xva, sva, [j]), yva)[0] for j in range(p)]) + 1e-3
        tau = np.ones(p); best = (-1, None); hist = []; stall = 0
        for it in range(iters):
            sols = []
            for _ in range(n_ants):
                w = tau ** alpha * eta ** beta; S = []
                avail = np.ones(p, bool)
                for _k in range(self.K):
                    pr = w * avail; pr /= pr.sum()
                    j = rng.choice(p, p=pr); S.append(j); avail[j] = False
                S = sorted(S); f = best_f1(self.score(Xva, sva, S), yva)[0]; sols.append((f, S))
            sols.sort(key=lambda z: -z[0])
            tau *= (1 - evap)
            for f, S in sols[:q_best]: tau[S] += f
            if sols[0][0] > best[0] + 1e-4: best = sols[0]; stall = 0
            else: stall += 1
            hist.append(best[0])
            if stall >= patience: break
        self.S = best[1]; self.aco_hist = hist; self.eta = eta
        return self
