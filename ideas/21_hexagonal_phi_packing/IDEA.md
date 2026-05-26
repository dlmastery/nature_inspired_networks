# Formal claim — H21

## Claim

Replacing a standard 3x3 conv with a 7-tap hexagonal stencil whose
six peripheral weights are scaled by 1/phi relative to the centre
(phi-radial energy distribution) lifts top-1 on rotated-CIFAR-10 by
>= +1.0 pp AND lowers rotation-equivariance error by >= 0.02 versus
the priors-off NaturePriorBlock baseline at 3-seed median. The same
operator is predicted to be near-neutral or mildly negative on
upright CIFAR-10 (where the rotational prior is unrewarded).

## Falsifier

If, at 3-seed median on rotated-CIFAR-10 with random 0-90 deg test-time
rotations, the phi-radial hex variant fails to raise top-1 by >= 1.0
pp AND fails to reduce rot-eq error by >= 0.02 relative to the priors-
off baseline, the claim is DISCARDED. T1.3 (`sg_only_hex` with
UNIFORM weights, upright CIFAR-10, single seed) showed -0.79 pp top-1
and -0.022 rot-eq err -- that result does NOT falsify H21 because
T1.3 was the wrong dataset AND the wrong weighting.

## Pre-registered prediction

| metric | predicted delta | rationale |
|---|---|---|
| composite (CIFAR-10 upright) | [-0.005, +0.010] | uniform-weight T1.3 was -0.019; phi-radial may recover |
| top-1 (rotated-CIFAR-10) | [+1.0 pp, +3.5 pp] | the proper-evaluation regime per H21 hypothesis doc |
| top-1 (upright CIFAR-10) | [-1.0 pp, +0.5 pp] | H21 should under-perform when no rotation pressure exists |
| params | [-22 %, -22 %] | 7 of 9 taps active = 22 % weight saving (HexConv2d primitive) |
| FLOPs | [-22 %, -22 %] | matches param drop |
| latency | [+50 %, +100 %] | T1.3 observed +70 % (non-coalesced memory access of masked kernel) |
| rot-eq err | [-0.060, -0.030] | direct geometric prediction; T1.3 already showed -0.022 with UNIFORM weights |

## Composite fingerprint at time of registration

`d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`
(composite = top1 - 0.05 * log10(params_M) - 0.05 * log10(latency_ms))

## Signed-off

- 2026-05-27 — Code-Agent-X
