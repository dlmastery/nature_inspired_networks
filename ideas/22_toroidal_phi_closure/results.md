# RESULTS — H22

> Auto-generated rollup across every archive sub-directory under
> `experiments/`. Re-generate with `python ../../scripts/build_report.py
> --root .` after each new experiment finishes.

## Top-line table

| archive | tag | seed | top-1 | params | latency_ms | composite | verdict |
|---|---|---|---|---|---|---|---|
| `exp001_tiled_seed0/` (MIGRATED) | sg_only_toroidal | 0 | 78.05 % | 127 k | 9.34 | 0.7768 | mildly negative on upright CIFAR-10 — predicted (data not wrap-aware) |

## Per-experiment narratives

### exp001_tiled_seed0 — single-prior toroidal pad on CIFAR-10 (legacy T1.6)

Migrated from `experiments/cifar10/sg_only_toroidal_seed0/`. The
toroidal closure was tested on upright CIFAR-10, which is the WRONG
benchmark per the hypothesis — CIFAR pixels do not wrap. The result
(top-1 78.05 %, composite 0.7768, Δ -0.0367 vs `sg_chan_fib` reference
0.8135) is the **predicted negative**. The next archived experiment
under this idea will be the wrap-aware tiled-CIFAR / tiled-Tiny-ImageNet
re-test with φ-scaled wrap distance, where the prior is data-aligned.
