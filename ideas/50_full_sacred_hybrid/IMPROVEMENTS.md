# IMPROVEMENTS — H50

> Append-only log of refinements queued or executed in response to the
> H50 falsification.

## Improvement 1 — H58 group avg-pool fix (DISCARDED)

- **Addresses:** AUDIT.md weakness #2 (C4 max-pool destroys 75 % of signal).
- **Change:** ran `sg_only_group_avg` and `sg_full_fib_avg` with
  `group_reduce="mean"`. Implementation in `priors.py::GroupConv2d`
  and surfaced via `NaturePriorFlags.group_reduce`.
- **Test added / updated:**
  - `tests/test_priors.py::test_group_conv_reduce_max_and_mean_h58`
  - `tests/test_blocks.py::test_h58_group_reduce_mean_forward_shape`
  - `tests/test_blocks.py::test_flag_tag_reflects_group_reduce`
- **Outcome:** **DISCARD.** Avg-pool was WORSE than max-pool:
  `sg_only_group_avg` top-1 65.38 % vs `sg_only_group` 69.84 %
  (Δ -4.46 pp); `sg_full_fib_avg` 66.86 % vs `sg_full_fib` 73.24 %
  (Δ -6.38 pp). The intuition "max-pool throws away 75 % of signal"
  was wrong — max-pool over rotated copies is a soft argmax over
  orientations, while mean-pool *dilutes* discriminative features.
  See `ideas/58_group_avg_pool/` for the full DISCARD verdict.
- **Date:** 2026-05-27

## Improvement 2 — leave-one-out probe (QUEUED, T2.8)

- **Addresses:** AUDIT.md weakness #3 (compound interaction).
- **Change:** run 6 configurations, each turning OFF one prior from
  the full hybrid: `sg_loo_no_hex`, `sg_loo_no_group`,
  `sg_loo_no_fractal`, `sg_loo_no_toroidal`, `sg_loo_no_cymatic`,
  `sg_loo_no_golden`. The variant whose `composite` jumps the most
  is the dominant negative contributor in the composition.
- **Test added / updated:** none — covered by existing block tests.
- **Outcome:** PENDING. Expected: `sg_loo_no_group` recovers the most
  composite, confirming C4 max-pool as the dominant negative.
- **Date:** queued 2026-05-27

## Improvement 3 — Sacred NAS (H45) over the 6-flag binary space

- **Addresses:** AUDIT.md weakness #3 + the broader claim that the
  priors are not orthogonal. Rather than reason about pairwise
  interactions, search the 2^6 = 64 binary subsets directly.
- **Change:** queued as H45. Search budget: 64 runs × 1 seed × 12
  epochs ≈ 13 h on RTX 4090 Laptop.
- **Test added / updated:** none in this idea sub-dir.
- **Outcome:** PENDING. Expected outcome: NAS picks 2–3 priors (likely
  fractal + Fib channels + max-pool group) and reaches composite >= 0.82,
  refining the falsified H50.
- **Date:** queued 2026-05-27

## Open items

- 3-seed reproduction of `sg_full_fib` (H60) — the -0.1169 composite
  is single-seed; multi-seed error bars may be ±0.005 or ±0.02. We do
  NOT expect this to overturn the falsification (the threshold is
  ≤ -0.005 and the signal is 23× over), but the error bar is owed to
  the published RESULTS.md.
- LLM-track analog (H67) — the curated full-paradigm hybrid is the
  refined form of H50 on the decoder side.
