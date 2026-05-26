# RESULTS — H17

> Auto-generated rollup across every archive sub-directory under
> `experiments/`. Re-generate with `python ../../scripts/build_report.py
> --root .` after each new experiment finishes.

## Top-line table

| archive | tag | seed | top-1 / ppl | params | latency_ms | composite | verdict |
|---|---|---|---|---|---|---|---|
| (legacy, related) T1.8 `sg_only_golden_modulate` | sg_only_golden_modulate | 0 | 79.81 % | 127 k | 5.95 | 0.8042 | confound — channel gate, not pure phi-skip |
| (none yet, pure phi-skip) | | | | | | | |

## Per-experiment narratives

T1.8 is the nearest available data point but tests a different
operation (golden-angle channel gate) at a different cost (+34 %
latency). The pure phi-skip variant predicted in IDEA.md has zero
overhead and is expected to land within [-0.3, +0.5] pp top-1 of the
1.0-skip baseline. Until exp001 runs, this table only references the
legacy archive.
