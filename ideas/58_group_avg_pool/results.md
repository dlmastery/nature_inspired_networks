# RESULTS — H58

> Auto-generated rollup across every archive sub-directory under
> `experiments/`. Re-generate with `python ../../scripts/build_report.py
> --root .` after each new experiment finishes.

## Top-line table

| archive | tag | seed | top-1 | params | latency_ms | composite | verdict |
|---|---|---|---|---|---|---|---|
| `exp001_sg_only_group_avg_seed0/` (MIGRATED) | sg_only_group_avg | 0 | **65.38 %** | 127 k | 8.82 | **0.6513** | **DISCARDED** (-4.46 pp top-1 vs `sg_only_group` 69.84 %) |

Companion row (full hybrid + avg-pool) — see legacy archive
`experiments/cifar10/sg_full_fib_avg_seed0/`: top-1 **66.86 %**,
composite **0.6432**, Δ -6.38 pp vs `sg_full_fib`.

## Per-experiment narratives

### exp001_sg_only_group_avg_seed0 — DISCARD verdict

Migrated from `experiments/cifar10/sg_only_group_avg_seed0/`. The
H58 prediction was that switching `GroupConv2d` from `reduce='max'`
to `reduce='mean'` would recover +5 to +10 pp on the
`sg_only_group` variant by preserving the full 4-rotation orbit
signal. Observed top-1 65.38 % vs `sg_only_group` 69.84 % yields
Δ -4.46 pp — the WRONG sign at large magnitude.

**Why the intuition was wrong:** the 4 orbit channels are correlated
rotated copies of the same convolution. `amax(dim=1)` is then a soft
argmax over orientations, preserving the strongest evidence at every
spatial location. Mean-pool dilutes that response with three
non-matching orientations, degrading contrast at every layer.

**Trained-feature Betti (first data of the campaign):**

| variant | β₀ per stage (trained) | β₀ per stage (fresh-init) |
|---|---|---|
| `sg_only_group_avg` | [121, 127, 127, 127] | [93, 71, 122, 100] |
| `sg_full_fib_avg` | [123, 124, 127, 127] | [123, 124, 127, 127]* |

Trained models INCREASE β₀ relative to fresh init (they cluster
features by class, producing isolated connected components in the
relative-threshold band). β₁ also rises in `sg_full_fib_avg`,
consistent with class-prototype loop structures. This inverts the
naive "topology simplification" reading and motivates per-epoch Betti
tracking as a follow-up.

**Future direction:** test the equivariance prior on data where it
pays off — rotated-CIFAR-10 with the **max-pool** (better) variant,
plus IcoMNIST / spherical MNIST. The fix is the DATA, not the
reduction operator.
