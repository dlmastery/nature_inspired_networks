---
name: autoresearch-topology-metrics
description: Use when computing persistent-homology Betti curves, Centered Kernel Alignment (CKA) similarity, and group-equivariance error on trained networks. These metrics quantify how a network simplifies data topology and how invariant its features are.
---

# Skill — Compute topology / representation metrics

## When to use

After any non-trivial sweep, when you want a *representation-quality*
panel for the dashboard alongside the standard top-1 / params / FLOPs
columns. Useful in particular for any inductive-bias claim
(equivariance, fractality, manifold-preservation).

## Inputs

- A trained model with a `stagewise_features(x) -> list[Tensor]`
  hook (one feature map per stage, pre or post pooling).
- A small test loader (~256 examples is enough).
- Optionally, a second trained model to compute CKA between two
  variants.

## Outputs

| metric | what it measures | typical interpretation |
|---|---|---|
| β₀ per stage | connected components | drops to ~1 as net disentangles |
| β₁ per stage | 1-D holes | drops to ~0 |
| Linear CKA(A, B) | representation similarity in [0, 1] | 1 = identical, 0 = orthogonal |
| Rotation-equivariance error | mean ‖f(Rx) − Rf(x)‖₂ / ‖f(x)‖₂ | 0 = perfectly equivariant, ~1 = not |

## Betti curve recipe

```python
import numpy as np
import torch
import torch.nn.functional as F
from ripser import ripser

@torch.no_grad()
def collect_features(model, loader, n_points=256, device='cuda'):
    model.eval().to(device); bag, have = None, 0
    for x, _ in loader:
        x = x.to(device)
        feats = model.stagewise_features(x)
        if bag is None: bag = [[] for _ in feats]
        for i, f in enumerate(feats):
            if f.ndim == 4:
                f = F.adaptive_avg_pool2d(f, 1).flatten(1)
            bag[i].append(f.detach().cpu())
        have += x.shape[0]
        if have >= n_points: break
    return [torch.cat(b)[:n_points].numpy() for b in bag]

def betti_curve(features, rel_thresh=0.20):
    """rel_thresh: persistence-pair life > 0.20 * max_life counts."""
    dgms, max_life = [], 0.0
    for f in features:
        f = (f - f.mean(0)) / (f.std(0) + 1e-6)
        d = ripser(f, maxdim=1)["dgms"]
        dgms.append(d)
        for k in (0, 1):
            if d[k].size:
                finite = d[k][np.isfinite(d[k][:, 1])]
                if finite.size:
                    max_life = max(max_life,
                                   float((finite[:,1]-finite[:,0]).max()))
    thresh = max(1e-4, rel_thresh * max_life)
    def _b(d): return int(((d[np.isfinite(d[:,1])][:,1]-
                            d[np.isfinite(d[:,1])][:,0]) > thresh).sum())
    return dict(
        b0=[_b(d[0]) for d in dgms],
        b1=[_b(d[1]) if len(d)>1 else 0 for d in dgms],
    )
```

The **relative** threshold is important: an absolute threshold makes
β-curves uninterpretable across stages because feature norms differ
by orders of magnitude as widths change.

## Linear CKA

```python
def linear_cka(X, Y):
    """X: (n, dx), Y: (n, dy). Both numpy or torch."""
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    n = X.shape[0]
    H = torch.eye(n) - torch.ones(n,n)/n
    Kx = H @ (X @ X.t()) @ H
    Ky = H @ (Y @ Y.t()) @ H
    return ((Kx * Ky).sum() / (Kx.norm() * Ky.norm() + 1e-8)).item()
```

CKA matrix between two models: `cka[i, j] = linear_cka(feats_A[i],
feats_B[j])`. Plotted as a heatmap with diagonal close to 1 if the
two models converge to similar internal stages.

## Rotation-equivariance error

```python
@torch.no_grad()
def rot_eq_err(model, loader, max_batches=8, device='cuda'):
    errs = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches: break
        x = x.to(device); y0 = model(x)
        for k in (1, 2, 3):
            yr = model(torch.rot90(x, k=k, dims=(2,3)))
            num = (yr - y0).norm(dim=1)
            den = y0.norm(dim=1) + 1e-8
            errs.append((num / den).mean().item())
    return float(sum(errs) / max(1, len(errs)))
```

For a fully-invariant net under {0°, 90°, 180°, 270°} rotation this
returns 0; for a standard CNN it sits around 0.5–1.0.

## Common pitfalls

1. **Fresh-init vs. trained features.** β-curves on fresh-init features
   discriminate priors weakly because random representations are all
   roughly Gaussian blobs. Compute Betti on *trained* features — your
   sweep runner must save `best.pt`.
2. **Sample size N too small.** With N < 100 the Vietoris-Rips diagram
   has very few persistence pairs and the curves are noisy. Use
   N ≥ 200.
3. **Per-stage normalisation matters.** Late-stage features have much
   larger norms than early-stage. Z-score per stage before VR.
4. **Max-life threshold across stages.** Use the global max-life
   across all stages, not per-stage. Otherwise β-curves are
   incomparable across depth.

## Output for the dashboard

Persist as `<root>/betti.json`:

```json
[
  {"tag": "<tag>", "seed": 0, "b0": [..stage0, ..stage1, ..],
   "b1": [..stage0, ..stage1, ..], "threshold": <float>}
]
```

The dashboard generator then plots one line per tag for β₀ and β₁
across stage index.

## Anti-patterns

- Reporting absolute β-numbers as a headline metric — they are
  scale-dependent. Always report **collapse rate** (β₀ at first vs.
  last stage, or rate of drop per stage).
- Comparing CKA across models with different feature dimensions
  without centering and HSIC normalisation — meaningless.
- Computing rotation-equivariance on the *training* set — only the
  test set matters.
