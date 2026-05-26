# RESULTS — H05

> Auto-generated rollup across every archive sub-directory under
> `experiments/`. Re-generate with `python ../../scripts/build_report.py
> --root .` after each new experiment finishes.

## Top-line table

| archive | tag | seed | top-1 / ppl | params | latency_ms | composite | verdict |
|---|---|---|---|---|---|---|---|
| (legacy) T1.5 `sg_only_fractal` (in master log) | sg_only_fractal | 0 | 82.46 % | 259 k | 7.42 | 0.8104 | partial — +2.35 pp top-1 |
| (none yet, 3-seed) | | | | | | | |

## Per-experiment narratives

T1.5 is the ONLY single prior in the legacy 11-row sweep that lifted
top-1 above the `sg_chan_fib` reference (80.11 %). The lift is +2.35 pp
at the cost of +104 % params and +67 % latency, which the composite
formula penalises enough to give a net composite delta of -0.0031.
The next H05 experiment archived in THIS sub-directory will reproduce
T1.5 at 3 seeds.
