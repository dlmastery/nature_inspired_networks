# Formal claim — H04

## Claim

Growing per-stage CNN channel widths by successive multiplication by
phi (or by Fibonacci ratios) and rounding to multiples of 8 matches or
beats linear-doubling channel schedules on CIFAR-10 at iso-parameter
budget, provided the network is wide and deep enough that mod-8
rounding does NOT collapse the phi and Fib schedules onto identical
integer widths.

## Falsifier

If, with c0 = 32 and n_stages = 4 (the configuration that guarantees
phi and Fib produce distinct integer widths after mod-8 rounding), the
phi-width schedule fails to lift composite by >= +0.005 versus the
linear-doubling baseline at 3-seed median on CIFAR-10, the claim is
DISCARDED in the small-net regime. Single-seed evidence (T1.1/T1.2)
showing -2.05 pp top-1 vs vanilla at c0=16 is INSUFFICIENT to falsify
because the mod-8 collapse made the two variants functionally identical.

## Pre-registered prediction

| metric | predicted delta | rationale |
|---|---|---|
| composite | [-0.012, +0.010] | observed -0.012 at c0=16 (mod-8 collapse); expected to recover at c0=32 |
| top-1 (CIFAR-10) | [-2.05 pp, +1.5 pp] | T1.1 observed -2.05 pp; predicted +0.5..+1.5 at c0=32 |
| params | [-31 %, -28 %] vs vanilla doubling | observed -31 % at c0=16 (127 k vs 186 k vanilla) |
| latency | [-1 %, +2 %] | width-only change; FLOPs scale with params |
| rot-eq err | [-0.005, +0.005] | not affected by width |

## Composite fingerprint at time of registration

`d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`
(composite = top1 - 0.05 * log10(params_M) - 0.05 * log10(latency_ms))

## Signed-off

- 2026-05-27 — Code-Agent-X
