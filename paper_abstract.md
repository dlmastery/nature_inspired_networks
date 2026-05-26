# Abstract — SacredGeoBlock: an ablation surface for sacred-geometry priors

We introduce **SacredGeoBlock**, a CIFAR-scale residual block whose six
"sacred-geometry" inductive biases — hexagonal-masked convolution
(HexaConv 2018), C4 group-equivariant convolution as a Platonic proxy
(Cohen & Welling 2016), fractal recursive sub-paths (FractalNet 2017),
toroidal (circular) padding (Pittorino 2022), Chladni-eigenmode
("cymatic") filter initialization, and golden-angle channel
modulation — are each toggleable for principled ablation. Combined with
a Fibonacci-Net 2025-style channel schedule, the block drops into a
ResNet-20-shaped backbone (3 stages × 3 blocks) at **127 k–259 k
parameters**, comparable to a 272 k ResNet-20 baseline.

We embed the block in an autoresearch protocol inherited from
`dlmastery/autoresearchimage`: every ablation row is an experiment with
arXiv-cited motivation, a numeric pre-run prediction, and a
SHA-256-fingerprinted composite metric

$$\textsf{composite} = \textsf{top-1} - 0.05\,\log_{10}(\textsf{params}_M) - 0.05\,\log_{10}(\textsf{latency}_{\text{ms}}).$$

The included 11-row curated sweep on CIFAR-10 reaches **target ≥ 85 %
top-1** in 12 epochs on a single RTX 4090 Laptop (16 GB) while exposing
the **shape** — not the asymptote — of each prior's contribution: a
sortable HTML dashboard surfaces Pareto fronts in
top-1 vs. {params, FLOPs, latency}, an ablation bar chart with
seed-std error bars, training curves, persistent-homology Betti-collapse
curves on stage features, and a linear-CKA matrix between variants.

This work does **not** claim ImageNet-scale SOTA. It claims (1) the
sacred-geometry priors can be cleanly compositionally ablated in one
block; (2) every individual prior runs to convergence on a consumer
GPU; (3) the autoresearch protocol's gates (Citation Rigor, Reasoning
Blob Completeness, Goodhart fingerprint) fire against engineering bugs
before they corrupt published numbers. The open repo,
[`dlmastery/sacgeometry`](https://github.com/dlmastery/sacgeometry),
ships the block, the runner, the dashboard, and a step-by-step
30-minute reproduction recipe.
