# IMPROVEMENTS — H21

> Append-only log of fixes applied in response to `AUDIT.md`.

## Improvement 1 — phi-radial scale buffer in this sub-project

- **Addresses:** AUDIT.md weakness #2 (T1.3 used UNIFORM weights;
  the second sacred prior phi-radial was never tested).
- **Change:** `ideas/21_hexagonal_phi_packing/implementation.py`
  introduces `hex_phi_radial_mask(k)` and `PhiRadialHexConv2d` --
  a thin Conv2d wrapper that multiplies the learnable weight by the
  combined hex x phi-radial mask buffer at every forward.
- **Test added / updated:**
  `test_phi_radial_mask_centre_to_neighbour_ratio_is_phi` locks in
  the ratio; `test_t1_3_regression_uniform_vs_phi_radial_outputs_differ`
  asserts the mask is not silently inert.
- **Outcome:** The full H21 prior is now exposable to the runner; the
  legacy uniform-weight T1.3 result is no longer the only available
  data point.
- **Date:** 2026-05-27

## Improvement 2 — Variance-preserving gain closed-form

- **Addresses:** AUDIT.md weakness #2 (apply_gain default may be
  wrong; we expose the compensation analytically so future runs can
  flip it on).
- **Change:** `variance_preserving_gain(k)` returns
  `sqrt(9 / (1 + 6*(1/phi)^2)) ~= 1.654` for k=3.
  `PhiRadialHexConv2d(apply_gain=True)` multiplies the kaiming-init
  weight by this gain.
- **Test added / updated:** `test_variance_preserving_gain_matches_closed_form`.
- **Outcome:** The init now has a defensible variance argument
  whether the H21 user wants the H21-paper-style raw scale
  (`apply_gain=False`, default) or the variance-preserving variant.
- **Date:** 2026-05-27

## Improvement 3 — Toroidal padding composes orthogonally

- **Addresses:** AUDIT.md weakness #5 (rotation-equivariance test
  missing -- mitigated by exposing toroidal padding so the boundary
  cells see a non-zero neighbourhood, which is a prerequisite for
  meaningful equivariance).
- **Change:** `PhiRadialHexConv2d.__init__(toroidal=True)` uses
  `toroidal_pad` from the shared primitive.
- **Test added / updated:** `test_toroidal_padding_changes_output`
  verifies the flag is not silently ignored.
- **Outcome:** H21 + H22 composition (toroidal hex) is now a one-flag
  flip away.
- **Date:** 2026-05-27

## Open items (audit weaknesses not yet addressed)

- **Mask asymmetry (AUDIT #1).** The shared `hex_kernel_mask(3)` is
  not symmetric to 60-degree rotation; fixing this requires a
  shared-infra change and an upgrade to a true axial-hex stencil
  emulation. Filed as a shared-infra suggestion below.
- **Iso-FLOPs evaluation (AUDIT #4).** Requires a channel-multiplier
  knob in the models layer; out of scope for this idea.
- **60-degree-equivariance unit test (AUDIT #5).** Requires a hex
  rotation transform; deferred until the asymmetric-mask issue is
  fixed.
- **Gradient waste at corner taps (AUDIT, bugs-not-caught).** Will
  not fix; the wasted gradient is bounded above by the corner-tap
  count (2/9 of weights at init time).
