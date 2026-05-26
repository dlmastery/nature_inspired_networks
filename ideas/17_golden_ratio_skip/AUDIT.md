# AUDIT — H17

> Adversarial self-critique BEFORE first experiment.

## Weaknesses found in v1

1. **PhiSkipBlock requires the wrapped block to expose `forward_branch`,
   and `NaturePriorBlock` does NOT.** Looking at
   `src/nature_inspired_networks/blocks.py:164`, the canonical
   `NaturePriorBlock.forward` returns
   `F.relu(self.conv2(F.relu(self.conv1(x))) + self.skip(x), inplace=True)`.
   There is no public `forward_branch` method and the final `relu` is
   applied to the SUM, so the fallback path `wrapped(x) - skip(x)`
   does NOT recover the pre-activation branch. To actually test H17
   on a real `NaturePriorBlock` we need either (a) a shared-infra
   change exposing `forward_branch`, or (b) a custom drop-in residual
   block that this idea owns. The current wrapper is correct ONLY
   for blocks that expose `forward_branch` and sum without a final
   ReLU -- including the test toy block, but excluding the real
   target.

2. **The final ReLU on `out = branch_scale * branch + skip_scale * skip`
   is missing.** The standard ResNet block applies ReLU after the
   residual sum. Our wrapper does NOT add a ReLU back. This is a
   deliberate decoupling so the wrapper is purely scaling, but it
   means the wrapped block's activation pattern is altered: the
   downstream model now sees pre-activation values where it used to
   see post-activation. This may shift BN statistics during training
   and is a known confound for the H17 test.

3. **Algebraic redundancy of two of the four modes.** `inv_phi` and
   `phi_minus_1` are mathematically identical (phi - 1 = 1/phi by the
   golden-ratio identity). We expose both for documentation clarity
   but a hostile reviewer will note the test suite asserts they agree
   to 1e-12. Keeping both modes in the experiment ablation table is
   redundant -- one should be removed.

4. **No data on convergence speedup.** The H17 prediction claims
   epochs-to-target should drop by 10-20 %, but we have NO test for
   that here (a unit test cannot measure training convergence). The
   falsifier on convergence speed depends entirely on the experiment.
   This means the unit tests can pass while the actual hypothesis
   remains untested.

5. **PHI_INV precomputation may diverge from re-computed 1/PHI.** We
   compute `PHI_INV = 1.0 / PHI` at module import time, but the test
   `test_static_scales_inv_phi_canonical_value` re-imports PHI_INV
   and re-computes 1/PHI; tiny float differences (~ulp) could exist
   in principle. Currently they don't, but if PHI is ever re-defined
   with higher precision (e.g. mpmath) the buffer copy could go
   stale. Tracked but not fixed.

## Bugs caught by tests (good)

- The `inv_phi` mode satisfies the golden-ratio identity
  `1 + 1/phi = phi` (sanity check in `test_inv_phi_skip_math_identity_branch`).
- The `phi_inv2_sum1` mode sums to 1.0 to 1e-12.
- Zero param overhead in all static modes
  (`test_phi_skip_zero_param_overhead_in_static_modes`).

## Bugs NOT caught by tests but suspected

- The learnable mode might collapse during training to skip_scale ~ 0
  (ReZero-style) which would defeat the H17 motivation. No early-stop
  guard in the wrapper.
- The branch_scale=1.0 init in learnable mode may interact poorly with
  the wrapped block's own ReLU(skip + branch) -- not tested.

## Performance / numerical-stability concerns

- One extra elementwise multiply on `skip` per block. At CIFAR scale
  (3-32x32, 9 blocks total) the cost is ~0.01 ms; negligible.
- Static modes use registered buffers, not parameters, so they do not
  add to the optimizer state.

## Mitigations queued for IMPROVEMENTS.md

- Add a custom `PhiSkipResidualBlock` that owns the conv1, conv2, skip,
  and final ReLU explicitly so the wrapper does not depend on the
  `forward_branch` contract.
- Remove `phi_minus_1` from the experiment ablation list (keep the
  mode in code for completeness, but skip in the YAML config).

## Sign-off

- 2026-05-27 — Code-Agent-X
