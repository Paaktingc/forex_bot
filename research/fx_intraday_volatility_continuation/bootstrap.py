"""Bootstrap confidence intervals (iid and moving-block) for mean R."""
from __future__ import annotations

import numpy as np

from . import config as C


def iid_ci(r: np.ndarray, n: int = C.N_BOOT, seed: int = C.SEED,
           alpha: float = 0.05):
    r = np.asarray(r, float)
    if len(r) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = r[rng.integers(0, len(r), size=(n, len(r)))].mean(axis=1)
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def block_ci(r: np.ndarray, block: int = 10, n: int = C.N_BOOT,
             seed: int = C.SEED, alpha: float = 0.05):
    r = np.asarray(r, float)
    m = len(r)
    if m == 0:
        return (float("nan"), float("nan"))
    b = max(1, min(block, m))
    nb = int(np.ceil(m / b))
    rng = np.random.default_rng(seed + 1)
    starts = rng.integers(0, m, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % m
    means = r[idx.reshape(n, -1)[:, :m]].mean(axis=1)
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def summary(r: np.ndarray) -> dict:
    r = np.asarray(r, float)
    if len(r) == 0:
        return dict(n=0)
    g, loss = r[r > 0].sum(), -r[r < 0].sum()
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))
    ddR = float((peak[1:] - cum).max())
    streak = best = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0
        best = max(best, streak)
    ci = iid_ci(r)
    cib = block_ci(r)
    return dict(n=len(r), mean=float(r.mean()), median=float(np.median(r)),
                sd=float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
                wr=float((r > 0).mean()), pf=float(g / loss) if loss > 0 else float("inf"),
                maxdd_r=ddR, longest_loss=best,
                ci_lo=ci[0], ci_hi=ci[1], bci_lo=cib[0], bci_hi=cib[1])
