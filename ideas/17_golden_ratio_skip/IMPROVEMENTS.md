# IMPROVEMENTS — H17

> Append-only log of fixes applied in response to `AUDIT.md`.

## Improvement 1 — Four modes covering the H17 design space

- **Addresses:** AUDIT.md weakness #3 (algebraic redundancy must be
  surfaced, not hidden).
- **Change:** `PhiSkipBlock` exposes four modes
  ({inv_phi, phi_inv2_sum1, phi_minus_1, learnable}) and registers
  them in `VALID_MODES`. The wrapper rejects unknown modes at
  construction time with a clear `ValueError`.
- **Test added / updated:** `test_unknown_mode_rejected`,
  `test_static_scales_inv_phi_canonical_value`.
- **Outcome:** The phi_minus_1 mode is kept as an explicit duplicate
  for documentation; the test suite asserts the algebraic identity
  `1/phi == phi - 1` so future float-precision regressions are visible.
- **Date:** 2026-05-27

## Improvement 2 — Zero-overhead claim locked-in by unit test

- **Addresses:** AUDIT.md weakness #4 (the prediction "params [0, 0]"
  must be testable at unit level).
- **Change:** `test_phi_skip_zero_param_overhead_in_static_modes`
  asserts that wrapping a toy block in any STATIC mode adds zero
  trainable parameters.
- **Test added / updated:** as above.
- **Outcome:** A future refactor that accidentally promotes the
  scale buffers to parameters will fail this test.
- **Date:** 2026-05-27

## Improvement 3 — Distinct from T1.8 golden_modulate

- **Addresses:** AUDIT.md weakness #4 (the related-but-distinct T1.8
  result must not be confused with H17 in the ablation table).
- **Change:** `test_phi_skip_regression_T1_8_distinct_from_modulate`
  asserts the H17 invariant `wrapped(x) = phi * x` under an identity
  branch -- this is a structural property the T1.8 channel-gate
  variant does NOT have.
- **Test added / updated:** as above.
- **Outcome:** Future ablation tables that mis-group the two priors
  will be visible at code-review time.
- **Date:** 2026-05-27

## Open items (audit weaknesses not yet addressed)

- **`forward_branch` contract (AUDIT #1).** The current wrapper only
  works correctly when the wrapped block exposes `forward_branch`.
  The real target `NaturePriorBlock` does not. Until the shared infra
  exposes that hook (filed as a suggestion below), H17's experimental
  validation must use the custom toy block OR a shared-infra patch.
- **Final ReLU semantics (AUDIT #2).** Decoupled deliberately;
  documented but not auto-restored. The experiment.py driver will
  log this as a known config delta.
