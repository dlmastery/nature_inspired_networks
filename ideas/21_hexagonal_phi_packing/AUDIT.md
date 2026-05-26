# AUDIT — H21

> Adversarial self-critique BEFORE first experiment.

## Weaknesses found in v1

1. **Hex mask is asymmetric.** Inspecting
   `src/nature_inspired_networks/priors.py:hex_kernel_mask` shows the
   k=3 mask is

       [ 1 1 0 ]
       [ 1 1 1 ]
       [ 0 1 1 ]

   This zeros (0,2) and (2,0) corners but keeps (0,0) and (2,2)
   active. That is a hex pattern on offset rows (axial-coordinate
   honeycomb), NOT a symmetric C6 hexagon. The H21 hypothesis
   document § 5.1 promises "centre + six neighbours" symmetric to
   60-degree rotation, but the actual mask is only symmetric to
   180-degree rotation. The 60-degree-equivariance test
   (`test_hex_60deg_eq`) promised in the hypothesis § 9 will FAIL on
   this mask. The bug lives in the shared primitive, not in this
   sub-project, but the hypothesis claim of "6-fold isotropy" is
   currently OVERCLAIMING.

2. **No variance compensation at init by default.** The phi-radial
   scale reduces the active-tap variance to ~3.29 from a He-init
   baseline of 9. The compensation gain
   `variance_preserving_gain(3) ~= 1.654` is exposed but
   `PhiRadialHexConv2d.__init__` defaults `apply_gain=False` to match
   the hypothesis doc. This means early-layer activations will be
   substantially smaller than a vanilla conv's at init, slowing the
   first few epochs of training. Whether this is a feature (implicit
   warmup) or a bug (lost capacity) is empirically open.

3. **Latency overhead from masked-conv pattern.** The forward pass
   multiplies `self.weight * self.mask` every step. T1.3 already
   measured a +70 % latency penalty on the priors-off baseline at
   batch 256 due to non-coalesced memory access. The phi-radial
   variant adds nothing on top of T1.3's hex baseline, but it also
   does NOT fix the issue. The IDEA.md predicts latency overhead
   [+50 %, +100 %].

4. **Iso-FLOPs comparison missing.** The 22 % weight saving means
   H21's nominal param count is lower than a dense baseline, which
   helps the composite formula via the `-0.05 * log10(params_M)`
   term. A fairer evaluation would scale the channel count up by
   sqrt(9/7) ~= 1.13 so the phi-radial-hex network has IDENTICAL
   params to the dense baseline. This sub-project does not implement
   an iso-FLOPs knob.

5. **No 60-degree equivariance unit test.** The hypothesis predicts
   improved rotation-equivariance error -- but at the unit level we
   only test 90-degree (torch.rot90) shape preservation, NOT the
   60-degree equivariance the hex stencil should give. A proper test
   would require a hex-aware rotation transform on the input, which
   is non-trivial to implement here.

## Bugs caught by tests (good)

- `test_phi_radial_mask_centre_to_neighbour_ratio_is_phi` locks in
  the ratio centre/neighbour = phi.
- `test_phi_radial_mask_seven_active_taps` confirms the 7-tap
  invariant.
- `test_effective_weight_zero_at_corners_after_forward` catches any
  future regression that would re-enable the corner taps.
- `test_t1_3_regression_uniform_vs_phi_radial_outputs_differ` catches
  the silently-inert regression (where the radial scale is somehow
  bypassed at forward time).
- `test_toroidal_padding_changes_output` catches a regression where
  the toroidal flag is silently ignored.

## Bugs NOT caught by tests but suspected

- Gradient flow through the masked positions: zero-mask means zero
  gradient for the corner weights. They will remain at their kaiming
  init forever. If `cymatic_init_` is later applied to the same conv,
  those positions will receive non-zero init values that the mask
  immediately suppresses -- silent waste of init energy. Filed but
  not fixed.
- BatchNorm interaction: the phi-radial mask's effective scale ratio
  is 1 : 1/phi : 0; BN will absorb a global scale per channel but
  cannot correct the within-kernel ratio. Probably fine, but not
  empirically validated.

## Performance / numerical-stability concerns

- All static numeric quantities (mask, gain) are float32 buffers; no
  precision drift.
- At bf16 autocast on RTX 4090, the mask is upcast then re-cast
  during the multiplication; potential 1-2 ulp drift but negligible.

## Mitigations queued for IMPROVEMENTS.md

- Add a k=5 (radius-2 hexagon, 19 taps) variant if T1.3-style tests
  show the 7-tap receptive field is too small to see effects.
- Add an `apply_gain=True` config preset for the iso-init experiment.

## Sign-off

- 2026-05-27 — Code-Agent-X
