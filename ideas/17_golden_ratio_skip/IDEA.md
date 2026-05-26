# Formal claim — H17

## Claim

Multiplying the residual skip path of a ResNet-style block by 1/phi
(= 0.618) instead of the standard 1.0 produces a zero-param, near-
zero-FLOP change that improves convergence speed by >= 10 % on
CIFAR-10 with no statistically-significant top-1 regression, per the
stable-ResNet rescaling argument (Hayou 2021).

## Falsifier

If, at 3-seed median on CIFAR-10 (12 epochs, batch 128), the 1/phi
skip variant LOSES more than 0.3 pp top-1 versus the 1.0-skip baseline
OR fails to reach the baseline's epoch-9 training loss in fewer than
9 epochs, the claim is DISCARDED. The T1.8 single-seed result for
the related `sg_only_golden_modulate` variant (-0.30 pp top-1, +34 %
latency from channel-gate overhead) is the nearest available data
but is NOT a clean H17 test because it confounds skip-scaling with
channel-gating.

## Pre-registered prediction

| metric | predicted delta | rationale |
|---|---|---|
| composite | [-0.005, +0.005] | scalar scaling only; param/latency unchanged |
| top-1 (CIFAR-10) | [-0.3 pp, +0.5 pp] | Hayou 2021 ranges; T1.8 observed -0.30 pp with confound |
| epochs-to-target-loss | [-20 %, -10 %] | stable-ResNet predicts faster convergence |
| params | [0, 0] | scalar only (zero overhead) |
| latency | [-1 %, +1 %] | one extra elementwise multiply per block |
| rot-eq err | [-0.005, +0.005] | scaling-only change |

## Composite fingerprint at time of registration

`d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`
(composite = top1 - 0.05 * log10(params_M) - 0.05 * log10(latency_ms))

## Signed-off

- 2026-05-27 — Code-Agent-X
