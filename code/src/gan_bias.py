"""Distributional-bias analysis of GAN-generated synthetic CPS data and a physics-consistency
filter as mitigation (addresses dataset/augmentation concerns). WGAN-GP (Gulrajani et al. 2017)
trained on PCA codes of 10-s normal windows of HAI 21.03. Metrics vs held-out real normal data:
MMD^2 (RBF, median heuristic; Gretton et al. 2012), mode coverage over k-means clusters (JS
divergence), physics-consistency violation rate (VAR(1) residual statistic), marginal shift."""
import os, json, numpy as np, jax, jax.numpy as jnp, optax
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import ks_2samp
from data import load
from common import window_index, make_windows
from layer1 import Gram
from layer2 import _init_mlp, _mlp
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
W, NC = 10, 32

def mmd2(X, Y, gamma):
    def k(A, B):
        d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(d, 0))
    m, n = len(X), len(Y); Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    return (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1)) + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1)) - 2 * Kxy.mean()

def js(p, q):
    p = p / p.sum(); q = q / q.sum(); m = (p + q) / 2
    f = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)), 0))
    return 0.5 * f(p, m) + 0.5 * f(q, m)

def train_wgan(Z, seed, iters=3000, ncrit=5, lam=10.0, bs=256, zdim=16):
    key = jax.random.PRNGKey(seed); k1, k2, key = jax.random.split(key, 3); d = Z.shape[1]
    G = _init_mlp(k1, (zdim, 128, 128, d)); D = _init_mlp(k2, (d, 128, 128, 1))
    og, od = optax.adam(1e-4, b1=0.5, b2=0.9), optax.adam(1e-4, b1=0.5, b2=0.9); sg, sd = og.init(G), od.init(D)
    def dloss(D, G, x, z, e):
        xf = _mlp(G, z); xh = e * x + (1 - e) * xf
        gp = jax.vmap(jax.grad(lambda u: _mlp(D, u[None])[0, 0]))(xh)
        return jnp.mean(_mlp(D, xf)) - jnp.mean(_mlp(D, x)) + lam * jnp.mean((jnp.linalg.norm(gp, axis=1) - 1) ** 2)
    def gloss(G, D, z): return -jnp.mean(_mlp(D, _mlp(G, z)))
    @jax.jit
    def dstep(D, sd, G, x, z, e):
        l, g = jax.value_and_grad(dloss)(D, G, x, z, e); u, sd = od.update(g, sd, D); return optax.apply_updates(D, u), sd, l
    @jax.jit
    def gstep(G, sg, D, z):
        g = jax.grad(gloss)(G, D, z); u, sg = og.update(g, sg, G); return optax.apply_updates(G, u), sg
    rng = np.random.default_rng(seed)
    for it in range(iters):
        for _ in range(ncrit):
            key, a, b = jax.random.split(key, 3)
            x = jnp.asarray(Z[rng.integers(0, len(Z), bs)]); z = jax.random.normal(a, (bs, zdim)); e = jax.random.uniform(b, (bs, 1))
            D, sd, l = dstep(D, sd, G, x, z, e)
        key, a = jax.random.split(key); G, sg = gstep(G, sg, D, jax.random.normal(a, (bs, zdim)))
    return lambda n, s: np.asarray(_mlp(G, jax.random.normal(jax.random.PRNGKey(s), (n, zdim))))

def main():
    D = load()
    Xall = make_windows(D["Xtr"], window_index(D["str_"], W, 10), W).clip(-10, 10)
    rng = np.random.default_rng(0); perm = rng.permutation(len(Xall)); cut = int(0.8 * len(Xall))
    Xtr, Xho = Xall[perm[:cut]], Xall[perm[cut:]]                       # GAN training / same-period held-out real
    Xlate = make_windows(D["Xcal"], window_index(D["scal"], W, 10), W).clip(-10, 10)   # later period (drift reference)
    pca = PCA(NC, random_state=0).fit(Xtr); ev = float(pca.explained_variance_ratio_.sum())
    Ztr = pca.transform(Xtr); mu, sd = Ztr.mean(0), Ztr.std(0); Zs = (Ztr - mu) / sd
    dec = lambda Zn: pca.inverse_transform(Zn * sd + mu)
    enc = lambda X: ((pca.transform(X) - mu) / sd).astype(np.float64)
    Zs = Zs.astype(np.float64)
    gram = Gram(D["Xtr"], D["str_"]); S = np.arange(D["Xtr"].shape[1]); Th, sig = gram.fit(S)
    def phys(Xw):
        A = Xw.reshape(len(Xw), W, -1); r = A[:, 1:] - A[:, :-1] @ Th
        return np.mean(np.sum((r / sig) ** 2, 2), 1)
    ptr = phys(Xtr[rng.choice(len(Xtr), 20000, replace=False)]); thr = np.percentile(ptr, 99); pmed = np.median(ptr)
    hold = Xho[rng.choice(len(Xho), 3000, replace=False)]
    ref = Xtr[rng.choice(len(Xtr), 3000, replace=False)]; late = Xlate[rng.choice(len(Xlate), 3000, replace=False)]
    sub = np.r_[hold[:400], ref[:400]]; d2 = ((sub[:, None] - sub[None]) ** 2).sum(-1); gamma = 1.0 / np.median(d2[d2 > 0])
    km = KMeans(20, n_init=5, random_state=0).fit(Zs[rng.choice(len(Zs), 20000, replace=False)])
    occ = lambda X: np.bincount(km.predict(enc(X)), minlength=20).astype(float)
    nvar = D["Xtr"].shape[1]
    def ksfrac(Xw):
        A = Xw.reshape(-1, nvar); B = hold.reshape(-1, nvar)
        return float(np.mean(np.array([ks_2samp(A[::7, j], B[::7, j]).statistic for j in range(nvar)]) > 0.1))
    res = dict(pca_explained_var=ev, physics_thr_p99=float(thr), rows=[])
    def row(name, Xw, seed=None, acc=None):
        Xw = Xw[:3000]
        res["rows"].append(dict(name=name, seed=seed, mmd2=float(mmd2(Xw, hold, gamma)), js_modes=float(js(occ(Xw), occ(hold))),
                                empty_modes=int(np.sum(occ(Xw) < 0.002 * len(Xw))), phys_violation=float(np.mean(phys(Xw) > thr)),
                                roughness_ratio=float(np.median(phys(Xw)) / pmed), frac_vars_ks_gt_0_1=ksfrac(Xw), acceptance=acc))
    row("Real, same period (reference)", ref)
    row("Real, later period (drift reference)", late)
    hocc = occ(hold) / occ(hold).sum()
    for seed in range(3):
        gen = train_wgan(Zs, seed); Xg = dec(gen(12000, 100 + seed))
        row("WGAN-GP synthetic", Xg, seed)
        keep = phys(Xg) <= thr; row("WGAN-GP + physics filter", Xg[keep], seed, float(keep.mean()))
        r2 = np.random.default_rng(seed); E = r2.standard_normal(Xg.shape).astype(Xg.dtype) * np.tile(sig, W)
        # measurement-noise re-injection, scale calibrated so that median roughness matches real data
        lo, hi = 0.0, 2.0
        for _ in range(25):
            s_ = (lo + hi) / 2
            if np.median(phys(Xg[:2000] + s_ * E[:2000])) < pmed: lo = s_
            else: hi = s_
        Xn = Xg + s_ * E; res.setdefault("noise_scale", []).append(float(s_))
        row("WGAN-GP + noise re-injection", Xn, seed)
        keep = phys(Xn) <= thr; Xf = Xn[keep]
        row("WGAN-GP + noise + physics filter", Xf, seed, float(keep.mean()))
        lab = km.predict(enc(Xf)); po = np.bincount(lab, minlength=20) / len(lab)
        wts = np.where(po[lab] > 0, hocc[lab] / np.maximum(po[lab], 1e-12), 0); wts /= wts.sum()
        idx = r2.choice(len(Xf), 3000, replace=True, p=wts)                                   # mode-matched resampling
        row("WGAN-GP + noise + filter + mode resampling", Xf[idx], seed)
        print(res["rows"][-5:], flush=True)
    json.dump(res, open(os.path.join(OUT, "gan_bias.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
