# IMPROVEMENTS — H22

> Append-only log of fixes applied in response to `AUDIT.md`. Each entry
> ties a weakness from the audit to a concrete code / test / config
> change.

## Improvement 1 — Document the 3x3 rounding collapse

- **Addresses:** AUDIT.md weakness #1 (phi-scaling rounds to 2 at k=3)
- **Change:** `IDEA.md` § Pre-registered prediction now notes that the
  3x3 result is a rounding-induced "wrap-by-2" baseline; the real
  phi-scaling test requires kernel_size >= 5. Added a 5x5 ablation row
  to the planned experiment matrix.
- **Test added / updated:** `test_phi_toroidal_pad_uses_phi_scaled_distance`
  now asserts the explicit `eff = 2` value at pad=1 so a future change
  in PHI or rounding is caught.
- **Outcome:** the 3x3 baseline is reported transparently; the 5x5
  experiment is the actual phi-scaling test.
- **Date:** 2026-05-26

## Improvement 2 — Surface boundary_err as side metric only

- **Addresses:** AUDIT.md weakness #3 (boundary_err is not in composite)
- **Change:** archived per-experiment README now logs `boundary_err`
  in the metrics table but the composite formula and SHA-256
  fingerprint remain `d65565e9...0893` (gate-protected, see CLAUDE.md
  rule #2).
- **Test added / updated:** none — eval module unchanged.
- **Outcome:** falsifier check uses composite (unchanged) AND
  boundary_err (side-channel) as documented in IDEA.md § Falsifier.
- **Date:** 2026-05-26

## Improvement 3 — Restrict toroidal pad to stage entrances

- **Addresses:** AUDIT.md performance concern (2.1x latency overhead
  in T1.6)
- **Change:** the next-experiment config will enable
  `toroidal_pad` only at the first conv of each stage (3 of 9 conv
  call sites in `NaturePriorNet`), not at every conv. Implemented in
  the upcoming `configs/cifar10_tiled.yaml`.
- **Test added / updated:** none — config-driven change.
- **Outcome:** expected latency reduction from 2.1x to ~1.4x while
  preserving the prior at every stage entrance.
- **Date:** 2026-05-26

## Open items (audit weaknesses not yet addressed)

- AUDIT weakness #2 (output spatial size grows under phi-scaling):
  defer until kernel_size=5 experiment; current `NaturePriorBlock`
  uses 3x3 only.
- Suspected `cymatic_init_` skip path when combining flags: needs an
  integration test in `tests/test_blocks.py` covering H22+H35 at the
  same time. Filed but not blocking H22 in isolation.
