"""Layer 2 (fog, generative): variational autoencoder (Kingma & Welling) on sliding
windows; anomaly score A(x) = 0.5||x - dec(mu(x))||^2 + lambda * KL(q(z|x)||N(0,I))  (Sec. 4.2).
Also: USAD baseline (Audibert et al., KDD'20) implemented in the same framework."""
import numpy as np, jax, jax.numpy as jnp, optax
from common import window_index, make_windows

def _init_mlp(key, sizes):
    ps = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a), jnp.zeros(b)))
    return ps

def _mlp(ps, x, final_act=None):
    for i, (W, b) in enumerate(ps):
        x = x @ W + b
        if i < len(ps) - 1: x = jax.nn.relu(x)
    return final_act(x) if final_act else x

CLIP = 10.0  # standardized inputs are clipped to +-10 SD

class VAE:
    def __init__(self, W=10, hidden=(256, 128), z=16, epochs=15, batch=512, lr=1e-3, lam=1.0):
        self.W, self.hidden, self.z, self.epochs, self.batch, self.lr, self.lam = W, hidden, z, epochs, batch, lr, lam

    def fit(self, X, seg, seed, stride=10):
        ends = window_index(seg, self.W, stride)
        Xw = np.clip(make_windows(X, ends, self.W), -CLIP, CLIP); d = Xw.shape[1]
        key = jax.random.PRNGKey(seed); k1, k2, key = jax.random.split(key, 3)
        params = dict(enc=_init_mlp(k1, (d,) + self.hidden + (2 * self.z,)),
                      dec=_init_mlp(k2, (self.z,) + self.hidden[::-1] + (d,)))
        opt = optax.adam(self.lr); st = opt.init(params); zdim = self.z

        def loss(p, x, k):
            h = _mlp(p['enc'], x); mu, lv = h[:, :zdim], jnp.clip(h[:, zdim:], -8, 8)
            zz = mu + jnp.exp(0.5 * lv) * jax.random.normal(k, mu.shape)
            xr = _mlp(p['dec'], zz)
            rec = 0.5 * jnp.sum((x - xr) ** 2, 1); kl = 0.5 * jnp.sum(jnp.exp(lv) + mu ** 2 - 1 - lv, 1)
            return jnp.mean(rec + kl)

        @jax.jit
        def step(p, s, x, k):
            l, g = jax.value_and_grad(loss)(p, x, k); u, s = opt.update(g, s, p)
            return optax.apply_updates(p, u), s, l
        rng = np.random.default_rng(seed); self.loss_hist = []
        for ep in range(self.epochs):
            perm = rng.permutation(len(Xw)); L = []
            for i in range(0, len(Xw) - self.batch + 1, self.batch):
                key, k = jax.random.split(key)
                params, st, l = step(params, st, jnp.asarray(Xw[perm[i:i + self.batch]]), k); L.append(float(l))
            self.loss_hist.append(np.mean(L))
        self.params = params
        zd, lam = self.z, self.lam
        @jax.jit
        def sc(p, x):
            h = _mlp(p['enc'], x); mu, lv = h[:, :zd], jnp.clip(h[:, zd:], -8, 8)
            xr = _mlp(p['dec'], mu)
            return 0.5 * jnp.sum((x - xr) ** 2, 1) + lam * 0.5 * jnp.sum(jnp.exp(lv) + mu ** 2 - 1 - lv, 1)
        self._sc = sc
        return self

    def score_windows(self, Xw):
        out = [np.asarray(self._sc(self.params, jnp.asarray(np.clip(Xw[i:i + 4096], -CLIP, CLIP))))
               for i in range(0, len(Xw), 4096)]
        return np.concatenate(out)

    def score(self, X, seg):
        """Per-second score: window ending at t; first W-1 s of a segment use the first window."""
        ends = window_index(seg, self.W, 1); out = np.empty(len(X))
        for i in range(0, len(ends), 20000):
            e = ends[i:i + 20000]; out[e] = self.score_windows(make_windows(X, e, self.W))
        for s in np.unique(seg):
            pos = np.where(seg == s)[0]; out[pos[:self.W - 1]] = out[pos[self.W - 1]]
        return out

class USAD(VAE):
    """USAD: shared encoder E, decoders D1, D2; two-phase adversarial training;
    score = a*||x-D1(E(x))||^2 + b*||x-D2(E(D1(E(x))))||^2 with a=b=0.5."""
    def __init__(self, W=10, z=40, epochs=15, batch=512, lr=1e-3):
        super().__init__(W=W, z=z, epochs=epochs, batch=batch, lr=lr)

    def _mm(self, Xw):
        return np.clip((Xw - self.mn) / self.rg, -1.0, 2.0).astype(np.float32)

    def score_windows(self, Xw):
        out = [np.asarray(self._sc(self.params, jnp.asarray(self._mm(Xw[i:i + 4096]))))
               for i in range(0, len(Xw), 4096)]
        return np.concatenate(out)

    def fit(self, X, seg, seed, stride=10):
        # as in the original USAD paper: min-max scaling to [0,1] and sigmoid decoder outputs
        ends = window_index(seg, self.W, stride)
        self.mn = np.tile(X.min(0), self.W); self.rg = np.tile(np.maximum(X.max(0) - X.min(0), 1e-6), self.W)
        Xw = self._mm(make_windows(X, ends, self.W)); d = Xw.shape[1]
        key = jax.random.PRNGKey(seed); k1, k2, k3 = jax.random.split(key, 3)
        h1, h2 = d // 2, d // 4
        sig = lambda x: jax.nn.sigmoid(x)
        p = dict(E=_init_mlp(k1, (d, h1, h2, self.z)), D1=_init_mlp(k2, (self.z, h2, h1, d)),
                 D2=_init_mlp(k3, (self.z, h2, h1, d)))
        opt = optax.adam(self.lr); st = opt.init(p)
        def losses(p, x, n):
            z = _mlp(p['E'], x); w1 = _mlp(p['D1'], z, sig); w2 = _mlp(p['D2'], z, sig); w3 = _mlp(p['D2'], _mlp(p['E'], w1), sig)
            m1 = jnp.mean((x - w1) ** 2); m2 = jnp.mean((x - w2) ** 2); m3 = jnp.mean((x - w3) ** 2)
            l1 = m1 / n + (1 - 1 / n) * m3; l2 = m2 / n - (1 - 1 / n) * m3
            return l1, l2
        @jax.jit
        def step(p, s, x, n):
            g1 = jax.grad(lambda q: losses(q, x, n)[0])(p); g2 = jax.grad(lambda q: losses(q, x, n)[1])(p)
            g = dict(E=jax.tree_util.tree_map(lambda a, b: a + b, g1['E'], g2['E']), D1=g1['D1'], D2=g2['D2'])
            u, s = opt.update(g, s, p); return optax.apply_updates(p, u), s
        rng = np.random.default_rng(seed)
        for ep in range(self.epochs):
            perm = rng.permutation(len(Xw))
            for i in range(0, len(Xw) - self.batch + 1, self.batch):
                p, st = step(p, st, jnp.asarray(Xw[perm[i:i + self.batch]]), float(ep + 1))
        self.params = p
        sig = jax.nn.sigmoid
        @jax.jit
        def sc(p, x):
            z = _mlp(p['E'], x); w1 = _mlp(p['D1'], z, sig); w3 = _mlp(p['D2'], _mlp(p['E'], w1), sig)
            return 0.5 * jnp.sum((x - w1) ** 2, 1) + 0.5 * jnp.sum((x - w3) ** 2, 1)
        self._sc = sc
        return self
