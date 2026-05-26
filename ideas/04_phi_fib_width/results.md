# RESULTS — H04

> Auto-generated rollup across every archive sub-directory under
> `experiments/`. Re-generate with `python ../../scripts/build_report.py
> --root .` after each new experiment finishes.

## Top-line table

| archive | tag | seed | top-1 / ppl | params | latency_ms | composite | verdict |
|---|---|---|---|---|---|---|---|
| (legacy) T1.1 `sg_chan_fib` (in master log, c0=16 n=3) | sg_chan_fib | 0 | 80.11 % | 127 k | 4.43 | 0.8135 | partial — mod-8 collapse |
| (legacy) T1.2 `sg_chan_phi` (in master log, c0=16 n=3) | sg_chan_phi | 0 | 80.11 % | 127 k | 4.11 | 0.8152 | partial — mod-8 collapse |
| (none yet, c0=32 n=4) | | | | | | | |

## Per-experiment narratives

T1.1 and T1.2 are archived in the legacy 11-row sweep under
`experiments/` at the repo root; they are referenced here because H04
is the hypothesis they tested (and the mod-8 collapse pathology they
exposed). The next experiment to be archived in THIS sub-directory is
`exp001_c0_32_n4` which separates phi and fib at the wrapper's
recommended config; until then this table only references the legacy
archive.
