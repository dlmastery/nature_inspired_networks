# RESULTS — H35

> Auto-generated rollup across every archive sub-directory under
> `experiments/`. Re-generate with `python ../../scripts/build_report.py
> --root .` after each new experiment finishes.

## Top-line table

| archive | tag | seed | top-1 | params | latency_ms | composite | verdict |
|---|---|---|---|---|---|---|---|
| `exp001_audio_seed0/` (MIGRATED) | sg_only_cymatic_init | 0 | 77.44 % | 127 k | 4.14 | 0.7883 | unexpected negative on CIFAR-10 — root-caused (no orthonormalisation + DC band + wrong dataset) |

## Per-experiment narratives

### exp001_audio_seed0 — single-prior cymatic init on CIFAR-10 (legacy T1.7)

Migrated from `experiments/cifar10/sg_only_cymatic_init_seed0/`. The
legacy cymatic init in `priors.py::cymatic_init_` was the only single
prior whose negative result on CIFAR-10 was NOT predicted by the
source PDF (the others — group, toroidal, golden — were predicted
neutral-or-negative). Post-hoc root-cause analysis (see
`hypotheses/g4_kernels_attention_filters/H35_cymatic_wavelet.md` § 11) identified TWO bugs and one
wrong-dataset choice:

1. all output channels initialised from the same low-frequency mode →
   highly correlated channels;
2. the (1, 1) mode is effectively DC on a 3x3 kernel → no spatial
   structure;
3. upright CIFAR-10 has no harmonic / vibration structure → the prior
   is data-misaligned.

The next archived run will be on AudioMNIST spectrograms with the
orthonormalized (2,5)-band init from `implementation.py::cymatic_init_ortho_`.
