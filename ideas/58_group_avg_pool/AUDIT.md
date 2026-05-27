# AUDIT — H58

> Adversarial self-critique of the DISCARDED avg-pool fix. This audit
> doubles as a post-mortem of the wrong intuition that motivated the
> experiment.

## Weaknesses found in v1 (i.e., why the hypothesis was wrong)

1. **"75 % signal loss" treated orbit channels as independent.** The
   pre-registration argued that `amax(dim=1)` over a 4-channel orbit
   discards 3/4 of the information. This is only true if the 4
   channels are independent. In `GroupConv2d._orbit()` the 4 channels
   are CORRELATED rotated copies of the same convolution; max over
   them is a soft argmax over orientations, not an information bottleneck.
   The audit caught this only AFTER the run was launched, by which
   point the falsifier was already on its way to being hit.
2. **No control for the BatchNorm interaction.** Mean-pool over the
   orbit dilutes the magnitude of activations, shifting the running
   mean/var of the downstream `BatchNorm2d`. The training-time BN
   compensates within a few epochs, but the gradient through a smaller
   pre-BN signal is also smaller, slowing learning at the *start* of
   training. The H58 run had only 12 epochs — too few to let BN fully
   compensate. A multi-epoch (50+) re-run might narrow the gap; would
   still not flip the sign.
3. **The hypothesis ignored the DATA dimension.** Even if avg-pool
   recovered the signal that max-pool "discards", the gain only
   materialises when the test set has rotational variance. CIFAR-10's
   test images are canonically oriented; the C4-equivariant feature
   bank is unused, so no choice of reduction can rescue it. The audit
   should have proposed `rotated-CIFAR` as the primary benchmark from
   the start instead of CIFAR-10.

## Bugs caught by tests (good)

- `tests/test_priors.py::test_group_conv_reduce_max_and_mean_h58`
  asserts that `reduce='max'` and `reduce='mean'` yield numerically
  distinct outputs - prevents a silent regression where someone
  collapses one onto the other.
- `tests/test_priors.py::test_group_conv_invalid_reduce_rejected`
  catches typos like `reduce='avg'` at construction.
- `tests/test_blocks.py::test_h58_group_reduce_mean_forward_shape`
  confirms the block-level shape contract is preserved.
- `tests/test_blocks.py::test_flag_tag_reflects_group_reduce` ensures
  the run tag changes when group_reduce changes (audit-trail).

## Bugs NOT caught by tests but suspected

- `scripts/compute_topology.py` iterates `NaturePriorFlags.__dataclass_fields__`
  to reconstruct flags from a config; it must SKIP the new
  `group_reduce` string field. The regression test for THIS bug lives
  in `tests/test_blocks.py::test_flag_field_iteration_distinguishes_string_field`
  and was added at the same time as H58.

## Performance / numerical-stability concerns

- Mean-pool is marginally slower than max-pool on CUDA at the small
  orbit size (G=4): `mean` materialises one extra division, whereas
  `amax` is fused. Negligible in absolute terms (~0.2 ms / forward).

## Mitigations queued for IMPROVEMENTS.md

- Replace the avg-pool follow-up with the **data fix**: train the
  same `sg_only_group` (max-pool) variant on rotated-CIFAR-10. The
  expected outcome is the equivariance prior FINALLY pays off (~+3
  to +6 pp top-1 over a vanilla rotated-CIFAR baseline).
- Add a hypothesis doc `hypotheses/g6_topological_bridging/H58_group_avg_pool.md` capturing
  this post-mortem in the standard committee-grade format.

## Sign-off

- 2026-05-27 — original audit at DISCARD verdict
- 2026-05-26 — Code-Agent-Y — re-signed for the `ideas/` taxonomy
