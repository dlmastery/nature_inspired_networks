# SOTA Comparison — honest mapping to prior art

> One row per claim, with the exact arXiv ID, the headline number the
> paper reports, and the corresponding number from **this** repo's
> CIFAR-10 ablation. We do **not** outperform any of them; the table
> below is a literature check, not a leaderboard.

The canonical YAML is `sota_catalog.yaml`. This file is the human-readable
summary.

## CIFAR-10 — reference checkpoints

| paper | model | top-1 | params | venue |
|---|---|---|---|---|
| He et al. 2015 (arXiv:1512.03385) | ResNet-20 | 91.25 % | 0.27 M | CVPR'16 |
| Larsson et al. 2017 (arXiv:1605.07648) | FractalNet-20 | 93.36 % | ~33 M (full depth) | ICLR'17 |
| Cohen & Welling 2016 (arXiv:1602.07576) | P4-CNN | 91.7 % | ~0.18 M | ICML'16 |
| Tan & Le 2019 (arXiv:1905.11946) | EfficientNet-B0 (CIFAR re-train) | 94.0 % | 5.3 M | ICML'19 |

Our 12-epoch quick runs (RTX 4090 Laptop, bf16) target ≥ 85 % top-1.
**12 epochs is short** — the He et al. recipe uses 164 epochs. Numbers
here read the *shape* of each prior's effect, not its asymptote.

## What we are NOT comparing against (and why)

| benchmark | why we skip | sister project that does it |
|---|---|---|
| ImageNet-1k | single-GPU 4090 Laptop budget; weeks/run | (none yet) |
| WILDS-Camelyon17 | 10 GB pathology data | [`autoresearchimage`](https://github.com/dlmastery/autoresearchimage) |
| IcoMNIST | needs e2cnn (no Py 3.13 wheel yet) | OPEN_AXES |
| ModelNet40 | out of CIFAR-shape scope | OPEN_AXES |
| ogbg-molhiv | GNN, not CNN | OPEN_AXES |

## Where this repo's claim lies on the literature map

```
                ImageNet-1k SOTA
                       │
                       │  (out of scope)
                       │
            FractalNet-20  ◄── 93.4 %
            EfficientNet-B0 ◄── 94.0 %
                       │
                       │  ≈ 7–9 % top-1 above 12-epoch budget
                       │
        ResNet-20  ─── 91.2 %  ◄── He 2015 recipe (164 ep)
                       │
                       │  ≈ 5 % top-1 above 12-epoch budget
                       │
   ┌───────────────────┼───────────────────┐
   │           sacgeometry CIFAR-10        │
   │           (12-epoch quick sweep)      │
   │   sg_full_fib    ────────►  85–87 %   │
   │   sg_chan_fib    ────────►  82–85 %   │
   │   baseline_sg_vanilla ───►  80–83 %   │
   └───────────────────────────────────────┘
```

The honest read: this repo's numbers are **6–9 % below** the literature
asymptote for ResNet-20-shaped models on CIFAR-10. That gap is mostly
the 164→12 epoch shortfall, not the priors. The *ablation deltas*
between rows (1–3 % top-1, multiple % composite) are what carry signal.
The full Tier-2 ImageNet run would be needed to make a SOTA claim.
