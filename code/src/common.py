"""Shared utilities: windowing, score calibration, metrics, PSO."""
import numpy as np

# ---------------------------------------------------------------- windowing
def window_index(seg, W, stride=1):
    """Indices t (window END positions) such that [t-W+1, t] lies in one segment."""
    n = len(seg); idx = []
    start = 0
    for s in np.unique(seg):
        pos = np.where(seg == s)[0]; a, b = pos[0], pos[-1]
        idx.append(np.arange(a + W - 1, b + 1, stride))
    return np.concatenate(idx)

def make_windows(X, ends, W):
    return np.stack([X[e - W + 1:e + 1].reshape(-1) for e in ends]) if len(ends) else np.zeros((0, W * X.shape[1]), X.dtype)

def hold_to_seconds(seg, ends, vals):
    """Causal zero-order hold of window scores onto the 1-s timeline. Seconds before the
    first complete window of a segment take that first window's value."""
    out = np.empty(len(seg), dtype=np.float64)
    for s in np.unique(seg):
        pos = np.where(seg == s)[0]
        m = (ends >= pos[0]) & (ends <= pos[-1]); e = ends[m]; v = vals[m]
        k = np.searchsorted(e, pos, side="right") - 1
        k[k < 0] = 0
        out[pos] = v[k]
    return out

# ---------------------------------------------------------------- calibration
class Surprisal:
    """Maps a raw anomaly score to tail surprisal -log10(1-F_cal(score)), F_cal estimated on
    held-out normal calibration scores. Makes heterogeneous layer scores commensurable."""
    def __init__(self, cal_scores):
        self.c = np.sort(np.asarray(cal_scores)); self.n = len(self.c)
    def __call__(self, s):
        r = np.searchsorted(self.c, s, side="right")
        return -np.log10(1.0 - r / (self.n + 1.0))

# ---------------------------------------------------------------- metrics
def best_f1(score, y):
    """Best point-wise F1 over all thresholds (used only inside tuning on validation)."""
    o = np.argsort(-score); ys = y[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    f1 = 2 * tp / (tp + fp + P)
    i = np.argmax(f1); return f1[i], score[o][i]

def events(y):
    d = np.diff(np.concatenate([[0], y, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))  # [start, end)

def evaluate(alarm, y):
    alarm = alarm.astype(bool); y = y.astype(bool)
    tp = np.sum(alarm & y); fp = np.sum(alarm & ~y); fn = np.sum(~alarm & y); tn = np.sum(~alarm & ~y)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12); fpr = fp / max(fp + tn, 1)
    ev = events(y.astype(int)); det = []; ttd = []
    for a, b in ev:
        hit = np.where(alarm[a:b])[0]
        det.append(len(hit) > 0)
        if len(hit): ttd.append(hit[0])
    return dict(precision=prec, recall=rec, f1=f1, fpr=fpr,
                event_dr=float(np.mean(det)), n_events=len(ev),
                ttd_s=float(np.mean(ttd)) if ttd else float("nan"))

# ---------------------------------------------------------------- PSO
def pso(obj, dim, lo, hi, rng, n_particles=20, iters=60, w_max=0.9, w_min=0.4,
        c1=1.49445, c2=1.49445, vmax_frac=0.2, tol=1e-4, patience=10):
    """Global-best PSO with linearly decreasing inertia (Shi & Eberhart 1998) and the
    Clerc-Kennedy acceleration constants. Maximises obj. Stops when the global best has
    not improved by more than tol for `patience` consecutive iterations."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); vmax = vmax_frac * (hi - lo)
    x = rng.uniform(lo, hi, (n_particles, dim)); v = rng.uniform(-vmax, vmax, (n_particles, dim))
    f = np.array([obj(p) for p in x]); pb = x.copy(); pbf = f.copy()
    g = pb[np.argmax(pbf)].copy(); gf = pbf.max(); hist = [gf]; stall = 0
    for it in range(iters):
        w = w_max - (w_max - w_min) * it / max(iters - 1, 1)
        r1, r2 = rng.random((n_particles, dim)), rng.random((n_particles, dim))
        v = np.clip(w * v + c1 * r1 * (pb - x) + c2 * r2 * (g - x), -vmax, vmax)
        x = np.clip(x + v, lo, hi)
        f = np.array([obj(p) for p in x])
        imp = f > pbf; pb[imp] = x[imp]; pbf[imp] = f[imp]
        if pbf.max() > gf + tol: g = pb[np.argmax(pbf)].copy(); gf = pbf.max(); stall = 0
        else:
            stall += 1
            if pbf.max() > gf: g = pb[np.argmax(pbf)].copy(); gf = pbf.max()
        hist.append(gf)
        if stall >= patience: break
    return g, gf, hist
