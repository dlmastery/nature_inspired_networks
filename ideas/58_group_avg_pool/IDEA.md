# Formal claim — H58

## Claim

Replacing the C4 group convolution's max-pool orbit reduction with
mean-pool recovers >= 5 pp top-1 on the `sg_only_group` variant on
CIFAR-10 by preserving the full 4-rotation orbit signal instead of
discarding 75 % of it at every layer.

## Falsifier

If, at 12-epoch CIFAR-10 single seed, switching `reduce` from "max"
to "mean" inside `GroupConv2d` fails to recover at least +1.0 pp top-1
on `sg_only_group` AND composite Δ vs the max-pool row is <= -0.005,
this hypothesis is DISCARDED.

**STATUS: DISCARDED.** Observed `sg_only_group_avg` top-1 = 65.38 %
vs `sg_only_group` 69.84 % yields Δ = **-4.46 pp** (the wrong sign);
composite Δ = -0.0424 (worse than the falsifier threshold). The
full-hybrid analog `sg_full_fib_avg` is even worse: top-1 66.86 %
vs `sg_full_fib` 73.24 % (Δ -6.38 pp).

## Pre-registered prediction (originally — and DISPROVED)

| metric | originally predicted Δ | OBSERVED Δ (T2.1) | verdict |
|---|---|---|---|
| top-1 on `sg_only_group` | [+5.0, +10.0] pp | **-4.46 pp** | DISPROVED |
| top-1 on `sg_full_fib` (composed with avg) | [+3.0, +8.0] pp | **-6.38 pp** | DISPROVED |
| composite on `sg_only_group` | [+0.03, +0.08] | **-0.0424** | DISPROVED |
| composite on `sg_full_fib` | [+0.02, +0.06] | **-0.0534** | DISPROVED |
| latency | no change | +2.5 ms (mean is marginally slower) | NEUTRAL |
| params | 0 | 0 | NEUTRAL |

## Why the prediction was wrong (the lesson)

The intuition "max-pool over 4 orbit channels discards 75 % of the
signal" treated the orbit as 4 independent feature maps. In fact the
4 maps are CORRELATED rotated copies of the SAME convolution applied
to the input. `amax(dim=1)` is then a SOFT ARGMAX OVER ORIENTATIONS:
at each spatial location it picks the rotation whose receptive field
best matches the input. Mean-pool over the orbit DILUTES that
discriminative response by averaging in the three orientations that
DO NOT match — every layer loses contrast.

The actual lesson is that the bottleneck is not the reduction operator
but the **data**: C4 equivariance is useful when the data has C4
variance (rotated CIFAR, IcoMNIST, spherical MNIST). On canonically
oriented CIFAR-10 it is dead weight regardless of the reduction.

## Composite fingerprint at time of registration

`d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`

(matches the SHA-256 stamped on
`experiments/cifar10/sg_only_group_avg_seed0/metrics.json` and
`experiments/cifar10/sg_full_fib_avg_seed0/metrics.json`)

## Signed-off

- 2026-05-27 — original registration + DISCARD verdict (Doc-Agent-A)
- 2026-05-26 — Code-Agent-Y — migrated into the `ideas/` taxonomy
