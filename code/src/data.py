"""Data loading and preprocessing for the BQGS proof-of-concept on HAI 21.03.

HAI 21.03 (Shin et al., CSET'21) is a public hardware-in-the-loop ICS dataset
released under CC BY-SA 4.0 at https://github.com/icsdataset/hai .
Split used (chronological, no shuffling):
  train      : train1-3 (normal only)  -> model fitting; last 20% of train3 held out for score calibration
  validation : test1, test2            -> threshold / fusion-weight / feature-subset tuning (labels used)
  test       : test3, test4, test5     -> final reporting only
"""
import os, numpy as np, pandas as pd

HAI_DIR = os.environ.get("HAI_DIR", "/home/claude/data/hai/hai-21.03")
CACHE = os.path.join(os.path.dirname(__file__), "..", "cache", "hai2103.npz")

def _read(name):
    d = pd.read_csv(os.path.join(HAI_DIR, f"{name}.csv.gz"))
    y = d["attack"].values.astype(np.int8)
    X = d.drop(columns=[c for c in d.columns if c == "time" or c.startswith("attack")])
    return X, y

def load(cache=True):
    if cache and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    tr = [_read(f"train{i}") for i in (1, 2, 3)]
    va = [_read(f"test{i}") for i in (1, 2)]
    te = [_read(f"test{i}") for i in (3, 4, 5)]
    cols = list(tr[0][0].columns)
    Xtr_all = pd.concat([t[0] for t in tr]).values.astype(np.float64)
    # drop variables that are constant in normal training data (carry no information)
    keep = Xtr_all.std(0) > 1e-8
    mu, sd = Xtr_all[:, keep].mean(0), Xtr_all[:, keep].std(0)
    f = lambda X: ((X.values[:, keep] - mu) / sd).astype(np.float32)
    # segment ids so that windows never straddle file boundaries
    def stack(parts):
        X = np.concatenate([f(p[0]) for p in parts]); y = np.concatenate([p[1] for p in parts])
        seg = np.concatenate([np.full(len(p[1]), i, np.int16) for i, p in enumerate(parts)])
        return X, y, seg
    Xtr, ytr, str_ = stack(tr); Xva, yva, sva = stack(va); Xte, yte, ste = stack(te)
    # calibration hold-out: last 20% of train3
    n3 = len(tr[2][1]); cut = len(Xtr) - int(0.2 * n3)
    out = dict(Xtr=Xtr[:cut], str_=str_[:cut], Xcal=Xtr[cut:], scal=str_[cut:],
               Xva=Xva, yva=yva, sva=sva, Xte=Xte, yte=yte, ste=ste,
               cols=np.array(cols)[keep])
    np.savez_compressed(CACHE, **out)
    return out

if __name__ == "__main__":
    D = load(cache=False)
    for k, v in D.items():
        print(k, v.shape, (v.sum() if k.startswith("y") else ""))
