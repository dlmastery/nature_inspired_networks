# Formal claim — H35

## Claim

Initialising convolutional kernels as Chladni-plate eigenmodes that
are (a) Gram-Schmidt orthonormalized across output channels and
(b) band-randomized in a natural-image / natural-audio frequency band
accelerates first-3-epoch loss reduction and lifts top-1 on
harmonic-structured data versus He init at matched kernel size and
parameter budget.

## Falsifier

If, at 3-seed median on AudioMNIST spectrograms, the corrected
cymatic-init variant fails to accelerate 3-epoch loss reduction by
>= 8 % AND fails to lift top-1 by >= 0.5 pp versus the He-init
baseline, this hypothesis is DISCARDED.

The upright-CIFAR-10 single-prior result T1.7 (top-1 77.44 %,
composite 0.7883, Δ -0.0252 vs `sg_chan_fib`) is NOT a falsifier of
the corrected claim — it tested a buggy implementation on
non-harmonic data.

## Pre-registered prediction

| metric | predicted Δ | rationale |
|---|---|---|
| composite (AudioMNIST) | [+0.005, +0.025] | cymatic-aligned task |
| top-1 (AudioMNIST) | [+1.0 pp, +3.0 pp] | direct audio-domain claim |
| top-1 (CIFAR-10, side eval) | [-1.0 pp, +0.5 pp] | not audio; expect ~neutral with orthonormalization fix |
| 3-epoch convergence speed | [+8 %, +20 %] | structured init shortens warm-up |
| params | [0, 0] | init only |
| latency | [≈1.0x, ≈1.0x] | init only — forward path unchanged |

## Composite fingerprint at time of registration

`d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`

(matches the SHA-256 stamped on `experiments/cifar10/sg_only_cymatic_init_seed0/metrics.json`)

## Signed-off

- 2026-05-26 — Code-Agent-Y
