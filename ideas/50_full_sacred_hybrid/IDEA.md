# Formal claim — H50

## Claim

Activating all six nature-inspired priors simultaneously inside a
single `NaturePriorBlock` (Fib channels, hex conv, C4 group conv,
fractal recursion, toroidal padding, golden-modulate gate, cymatic
init) raises top-1 by >= 5 pp and reduces FLOPs by >= 15 % versus the
`sg_chan_fib` reference at matched epoch budget on CIFAR-10.

## Falsifier

If, at 12-epoch CIFAR-10 single seed, the composite of the
`sg_full_fib` configuration is at most the composite of the
`sg_chan_fib` reference minus 0.005 (i.e., Δ <= -0.005), this
hypothesis is DISCARDED.

**STATUS: FALSIFIED.** Observed composite 0.6966 vs reference 0.8135
yields Δ = **-0.1169**, hitting the falsifier 23× over. The
empirical result is recorded in
`experiments/cifar10/sg_full_fib_seed0/metrics.json` (composite
fingerprint `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`).

## Pre-registered prediction (originally — and DISPROVED)

| metric | originally predicted Δ | OBSERVED Δ (T1.9) | verdict |
|---|---|---|---|
| composite | [+0.020, +0.080] | **-0.1169** | DISPROVED |
| top-1 (CIFAR-10) | [+2.0, +6.0] pp | **-6.87 pp** | DISPROVED |
| params | [-30 %, -10 %] | **+103 %** | DISPROVED |
| FLOPs | [-30 %, -10 %] | larger | DISPROVED |
| GPU latency (b=1) | [-20 %, 0 %] | **+352 % (5.0×)** | DISPROVED |
| rotation-equivariance err | [-0.03, -0.06] | **-0.045** | CONFIRMED — only positive |
| Betti collapse rate | [-0.2, -0.05] | per-stage β₀ INCREASES with training (see FINDINGS.md) | partially disproved |

The rotation-equivariance reduction is the only prediction that held.
It is real but unused — CIFAR-10 test images are not rotated.

## Composite fingerprint at time of registration

`d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`

## Signed-off

- 2026-04-12 — original registration (Doc-Agent-A)
- 2026-04-12 — falsifier hit, status updated to `✗ disproved`
- 2026-05-26 — Code-Agent-Y — migrated into the `ideas/` taxonomy
