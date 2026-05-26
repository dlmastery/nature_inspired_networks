# Formal claim — H22

## Claim

Replacing zero padding with circular (toroidal) padding whose effective
wrap distance is φ-scaled raises top-1 on wrap-aware datasets and
reduces boundary-pixel error versus a zero-pad baseline of matched
parameter budget, at no parameter cost.

## Falsifier

If, at 3-seed median on a tiled-texture or wrap-aware synthetic
dataset, the φ-scaled toroidal variant fails to raise top-1 by
>= 1.0 pp AND fails to reduce boundary-pixel error by >= 0.03 versus
the zero-pad baseline (`baseline_sg_vanilla`), this hypothesis is
DISCARDED.

The upright-CIFAR-10 result T1.6 (top-1 78.05 %, composite 0.7768,
Δ -0.0367 vs `sg_chan_fib`) is NOT a falsifier of the wrap-aware
claim — CIFAR images do not wrap, so the prior was tested off-domain.

## Pre-registered prediction

| metric | predicted Δ | rationale |
|---|---|---|
| composite (CIFAR-10) | [-0.020, +0.010] | T1.6 already negative; φ-scaling may recover |
| top-1 (CIFAR-10) | [-2.0 pp, +0.5 pp] | CIFAR is not wrap-aware; near-neutral or slight loss |
| top-1 (tiled-texture / wrap-synth) | [+1.5 pp, +4.0 pp] | the prior's design regime |
| boundary-pixel err | [-0.04, -0.10] | direct geometric prediction (Pittorino 2022) |
| params | [0, 0] | toroidal pad adds no parameters |
| latency | [+1.5x, +2.1x] | known from T1.6 (2.1x) — circular pad is memory-bound |

## Composite fingerprint at time of registration

`d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`

(matches the SHA-256 stamped on `experiments/cifar10/sg_only_toroidal_seed0/metrics.json`)

## Signed-off

- 2026-05-26 — Code-Agent-Y
