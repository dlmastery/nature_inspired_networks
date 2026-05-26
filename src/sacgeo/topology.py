"""Lightweight persistent homology + Betti curves on penultimate features.

We sample N points from the test loader, run them through the model, take
each stage's pooled feature representation, and compute β0/β1 from a
Vietoris-Rips diagram (ripser). β-collapse rate = how fast β0 drops from
N (raw) to ~1 (one connected cluster) as we move up the network.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def collect_features(model: nn.Module, loader, device: str = "cuda",
                     n_points: int = 256) -> list[np.ndarray]:
    """Return one feature matrix per stage, each of shape (n_points, dim)."""
    model.eval().to(device)
    have = 0
    bag: list[list[torch.Tensor]] = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        if hasattr(model, "stagewise_features"):
            feats = model.stagewise_features(x)
        else:
            feats = [model(x)]
        if not bag:
            bag = [[] for _ in feats]
        for i, f in enumerate(feats):
            f = F.adaptive_avg_pool2d(f, 1).flatten(1) if f.ndim == 4 else f
            bag[i].append(f.detach().cpu())
        have += x.shape[0]
        if have >= n_points:
            break
    return [torch.cat(b, dim=0)[:n_points].numpy() for b in bag]


def _betti_from_diagram(dgm: np.ndarray, threshold: float = 0.5) -> int:
    """Count persistence-pairs with life > threshold."""
    if dgm.size == 0:
        return 0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if finite.size == 0:
        return 0
    lives = finite[:, 1] - finite[:, 0]
    return int((lives > threshold).sum())


def betti_curve(features: list[np.ndarray]) -> dict[str, list[int]]:
    """Return β0, β1 per stage."""
    try:
        from ripser import ripser  # type: ignore
    except Exception:
        return dict(b0=[len(features[0])] * len(features), b1=[0] * len(features))
    b0s, b1s = [], []
    for f in features:
        f = (f - f.mean(0)) / (f.std(0) + 1e-6)
        res = ripser(f, maxdim=1)
        d = res["dgms"]
        thresh = float(np.median(d[0][:, 1][np.isfinite(d[0][:, 1])])) if len(d[0]) else 0.5
        b0s.append(_betti_from_diagram(d[0], threshold=thresh))
        b1s.append(_betti_from_diagram(d[1], threshold=thresh) if len(d) > 1 else 0)
    return dict(b0=b0s, b1=b1s)


def cka_matrix(features_a: list[np.ndarray], features_b: list[np.ndarray]):
    """Layer-by-layer linear CKA between two models' stage features."""
    from .eval import linear_cka
    n = min(len(features_a), len(features_b))
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A = torch.from_numpy(features_a[i])
            B = torch.from_numpy(features_b[j])
            out[i, j] = linear_cka(A, B)
    return out
