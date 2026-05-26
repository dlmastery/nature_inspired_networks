# Formal claim — H05

## Claim

A residual block whose second 3x3 conv is replaced by a FractalNet-style
depth-2 recursive sub-block (per `_FractalPath`) lifts top-1 on
CIFAR-10 by at least +1.5 pp at 3-seed median over the priors-off
`sg_chan_fib` reference, at the cost of at most +110 % params and
+80 % latency. The composite delta may be negative because the
penalty terms exceed the top-1 lift; the explicit 1/phi shrink rule
on sub-block width is the H05.v2 follow-up that should recover the
composite.

## Falsifier

If, at 3-seed median on CIFAR-10, the depth=2 fractal block fails to
lift top-1 by >= +1.0 pp over the priors-off reference, the claim is
DISCARDED. The T1.5 single-seed result (+2.35 pp at top-1 82.46 % vs
80.11 % reference) is treated as suggestive but not decisive.

## Pre-registered prediction

| metric | predicted delta | rationale |
|---|---|---|
| composite | [-0.005, +0.000] | param/latency penalty cancels top-1 lift; T1.5 = -0.0031 |
| top-1 (CIFAR-10) | [+1.0 pp, +3.0 pp] | T1.5 observed +2.35 pp at single seed |
| params | [+90 %, +110 %] | T1.5 observed +104 % (259 k vs 127 k) |
| latency | [+50 %, +80 %] | T1.5 observed +67 % (7.42 vs 4.43 ms) |
| rot-eq err | [-0.05, -0.02] | T1.5 observed -0.036 (unexpected positive) |

## Composite fingerprint at time of registration

`d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`
(composite = top1 - 0.05 * log10(params_M) - 0.05 * log10(latency_ms))

## Signed-off

- 2026-05-27 — Code-Agent-X
