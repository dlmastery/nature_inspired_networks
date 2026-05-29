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

## 2026-05-27 Phase-2 smoke — best result vs. literature

The 35-tag CIFAR-10 Phase-2 smoke (seed 0, **12 epochs**, RTX 4090
Laptop, bf16) produced the following honest comparison. **These are
12-epoch screening numbers, NOT converged results** — the ~91 % figure
below requires the full 164-epoch recipe, which our runs do not use.

| source | model / tag | top-1 | epochs | params | honest note |
|---|---|---|---|---|---|
| He et al. 2015 (arXiv:1512.03385) | ResNet-20 (full recipe) | 91.25 % | 164 | 0.27 M | literature asymptote |
| **this repo** | `phi_budget` (H09) — **our best** | **85.54 %** | **12** | ~284 k | only variant > our own baseline at 12 ep |
| **this repo** | `baseline_resnet20` | 84.78 % | 12 | ~272 k | our 12-ep ResNet-20 reference |

The **5.7 pp** gap between our 12-epoch ResNet-20 (84.78 %) and the
He et al. 164-epoch ResNet-20 (91.25 %) is almost entirely the
**164→12 epoch shortfall**, not a modelling deficit — same architecture,
~10× fewer epochs. Our best prior variant (`phi_budget`) leads our own
baseline by **+0.76 pp** at this budget; whether that edge survives to
convergence is exactly what the Phase-4 ≥30-epoch CIFAR-100 runs test.
We make **no** SOTA claim from a 12-epoch screen.

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
   │     nature_inspired_networks CIFAR-10           │
   │     (12-epoch Phase-2 smoke, seed 0)            │
   │   phi_budget (H09)  ──────►  85.54 % (best)     │
   │   baseline_resnet20 ──────►  84.78 %            │
   │   golden_momentum   ──────►  83.52 %            │
   │   baseline_sg_vanilla ────►  82.16 %            │
   │   full_fib (all priors) ──►  73.24 %            │
   └─────────────────────────────────────────────────┘
```

The honest read: this repo's best 12-epoch number (85.54 %) is
**~5.7 % below** the He et al. 164-epoch ResNet-20 asymptote (91.25 %).
That gap is mostly the 164→12 epoch shortfall, not the priors —
our own 12-epoch baseline sits at 84.78 %. The *ablation deltas*
between rows are what carry signal at this budget; the all-priors
`full_fib` hybrid actually *underperforms* (73.24 %), so "more priors"
is not monotonic. A full converged run (Phase 4+) would be needed to
make any SOTA claim.
