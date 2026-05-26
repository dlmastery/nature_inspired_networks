# IMPROVEMENTS — H35

> Append-only log of fixes applied in response to `AUDIT.md`. Each entry
> ties a weakness from the audit to a concrete code / test / config
> change.

## Improvement 1 — Gram-Schmidt orthonormalisation across output channels

- **Addresses:** root-cause #1 of the T1.7 negative result; AUDIT.md
  weakness #1 (mode reuse across channels) is partially fixed.
- **Change:** `implementation.py::chladni_modes_banded` now applies
  `torch.linalg.qr` across the `n_modes` axis, returning an
  orthonormal basis of `min(n_modes, k*k)` directions cycled out to
  `n_modes` rows.
- **Test added / updated:** `test_chladni_modes_banded_orthonormal_across_modes`
  asserts the Gram matrix is the identity to atol=1e-4.
- **Outcome:** consecutive output channels are no longer near-duplicates;
  expected first-3-epoch loss drop accelerates as predicted in IDEA.md.
- **Date:** 2026-05-26

## Improvement 2 — Default frequency band shifted from (1, 1) to (2, 5)

- **Addresses:** root-cause #2 of the T1.7 negative result; the (1, 1)
  mode is effectively DC on a 3x3 kernel.
- **Change:** `cymatic_init_ortho_(band=(2, 5))` is the default; the
  3x3 receptive field now actually carries a non-trivial spatial
  pattern.
- **Test added / updated:** `test_band_low_eq_high_is_uniform_mode`
  guards the degenerate band=(m, m) case.
- **Outcome:** the kernel no longer initialises near-DC; spatial
  structure visible from epoch 0.
- **Date:** 2026-05-26

## Improvement 3 — Dataset shift for the primary benchmark

- **Addresses:** AUDIT noted the data misalignment that produced T1.7.
- **Change:** the next experiment's primary benchmark is AudioMNIST
  spectrograms (harmonic-structured), not CIFAR-10. CIFAR-10 remains
  as a side eval to confirm the corrected init is at least neutral on
  non-harmonic data.
- **Test added / updated:** none — config-driven change.
- **Outcome:** the prior is now tested on data where its mechanism
  pays off; the falsifier in IDEA.md is set against the AudioMNIST
  numbers, not CIFAR-10.
- **Date:** 2026-05-26

## Open items (audit weaknesses not yet addressed)

- AUDIT weakness #3 (He-variance match within 50%, not 10%): a per-fan-in
  calibration constant is the principled fix. Documented as an open
  item; current code uses a `sqrt(k*k)` correction that gets close.
- Vectorisation of the per-`(o, i)` copy loop — punted to follow-up.
