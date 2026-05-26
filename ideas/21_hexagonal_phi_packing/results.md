# RESULTS — H21

> Auto-generated rollup across every archive sub-directory under
> `experiments/`. Re-generate with `python ../../scripts/build_report.py
> --root .` after each new experiment finishes.

## Top-line table

| archive | tag | seed | top-1 / ppl | params | latency_ms | composite | verdict |
|---|---|---|---|---|---|---|---|
| (legacy) T1.3 `sg_only_hex` (UNIFORM weights, upright) | sg_only_hex | 0 | 79.32 % | -- | 7.53 | 0.7941 | partial — wrong dataset, wrong weights |
| (none yet, phi-radial + rotated) | | | | | | | |

## Per-experiment narratives

T1.3 ran the hex prior with UNIFORM 7-tap weights on UPRIGHT CIFAR-10
and showed -0.79 pp top-1 (with the consolation that rot-eq err did
drop by 0.022). This result was used to MOTIVATE H21, not to falsify
it: the phi-radial weighting is the missing prior and rotated-CIFAR-10
is the correct evaluation regime. The next H21 experiment archived in
THIS sub-directory will run on rotated-CIFAR-10 with phi-radial scaling
enabled.
