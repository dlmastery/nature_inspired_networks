# RESULTS — H50

> Auto-generated rollup across every archive sub-directory under
> `experiments/`. Re-generate with `python ../../scripts/build_report.py
> --root .` after each new experiment finishes.

## Top-line table

| archive | tag | seed | top-1 | params | latency_ms | composite | verdict |
|---|---|---|---|---|---|---|---|
| `exp001_sg_full_fib_seed0/` (MIGRATED) | sg_full_fib | 0 | **73.24 %** | 259 k | **20.02** | **0.6966** | **FALSIFIED** (-0.1169 vs `sg_chan_fib` 0.8135 — worst single row in sweep) |

## Per-experiment narratives

### exp001_sg_full_fib_seed0 — the headline negative result

Migrated from `experiments/cifar10/sg_full_fib_seed0/`. All six
nature-inspired priors active (hex + group + fractal + toroidal +
cymatic-init + golden-modulate) on a NaturePriorBlock backbone with
Fibonacci channel widths.

**Result:** composite **0.6966** — the **WORST single row** in the
11-run sweep, tied with `sg_only_group` (0.6937) for the bottom.

**Failure decomposition** (from FINDINGS.md):

1. **Multiplicative latency stack:** 5.0× latency (4.43 → 20.02 ms),
   subtracting ~0.05 from composite directly.
2. **C4 max-pool destroys 75 % of orbit signal:** dominant negative
   contributor at -10.27 pp in isolation. The H58 follow-up (avg-pool
   fix) tried to address this and was **DISCARDED** — mean-pool
   was even worse.
3. **Toroidal pad on non-wrap data:** -2.06 pp in isolation; the prior
   is data-misaligned for upright CIFAR-10.
4. **Cymatic init buggy in isolation:** -2.67 pp; root-caused to
   missing orthonormalization + DC-band (see `ideas/35_cymatic_wavelet/`).
5. **Additive vs multiplicative compound:** naive sum of per-prior
   deltas = -13.74 pp predicts top-1 ~67 %; observed 73.24 % is
   slightly BETTER than the additive prediction → some recovery from
   non-orthogonality, but the net is still strongly negative.

The autoresearch protocol delivered exactly what it should: a
falsifiable negative result, clean single-prior decomposition, and a
queue of refined follow-ups (H58 DISCARD, H45 NAS, H60 multi-seed,
H67 LLM-track curated hybrid).
