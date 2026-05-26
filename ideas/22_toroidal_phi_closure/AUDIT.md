# AUDIT — H22

> Adversarial self-critique BEFORE first experiment. Treat the
> implementation as if reviewing it for a hostile NeurIPS reviewer.

## Weaknesses found in v1

1. **φ-scaling collapses to an integer at the 3×3 scale.** `round(PHI * 1) = 2`,
   which is the *same* effective wrap as a hand-picked pad=2. The φ-ness of
   the hypothesis only becomes geometrically distinct for kernel_size >= 5
   where `round(PHI * 2) = 3 ≠ 4`. At the canonical 3×3 conv stack used in
   `NaturePriorBlock` this hypothesis is INDISTINGUISHABLE from a "wrap-by-2"
   baseline. We MUST report kernel_size = 5 numbers separately and label
   them as the actual φ-scaling test; the 3×3 numbers are a rounding-induced
   no-op vs integer-2 wrap.
2. **Output spatial size grows under φ-scaling.** `ToroidalConv2d` with
   `phi_scale=True` and `padding=0` on the underlying conv leaves the
   output shape larger than the input (16x16 → 14x14 with kernel=3 after
   wrap=2, but **larger** when we re-do the math: input 16, pad=2 → 20,
   conv k=3 stride=1 → 18, NOT 16). This breaks the "drop-in Conv2d
   replacement" contract — every downstream BN/skip-connection has to
   anticipate the size delta. The fix is to also do a centre-crop after
   the conv, or to require `padding=k//2` with toroidal mode rather than
   over-pad.
3. **Boundary-pixel error metric is not yet defined in the eval module.**
   The IDEA.md pre-registered prediction names `boundary_err` but the
   composite formula in `src/nature_inspired_networks/eval.py` does not
   yet compute it. Adding it before running would change the composite
   SHA-256 fingerprint, which is gate-protected (CompositeFingerprintError).
   The honest choice is to log `boundary_err` as a side metric, keep the
   composite fingerprint stable, and report `boundary_err` in the
   archive README.

## Bugs caught by tests (good)

- `test_phi_toroidal_pad_uses_phi_scaled_distance` would have caught a
  silent regression if someone switched the rounding rule
  (`int()` vs `round()` differ at `PHI * 1 = 1.618 → 1` vs `2`).
- `test_toroidal_pad_is_translation_equivariant_on_torus` would have
  caught a `mode="reflect"` typo (reflect is NOT equivariant under
  cyclic shift).

## Bugs NOT caught by tests but suspected

- The `phi_toroidal_pad` uses `F.pad(mode="circular")` which on Windows
  + PyTorch 2.x has occasionally regressed with non-contiguous inputs;
  the test fixture uses contiguous tensors and so does not exercise
  this path.
- The cymatic-init in `NaturePriorBlock._GenericConv` runs `cymatic_init_`
  on the `nn.Conv2d` inside the H22 wrapper, but ONLY if `flags.hex` is
  false — combining H22 + H35 + H21 may yield a code path where the
  init is silently skipped.

## Performance / numerical-stability concerns

- Circular padding is memory-bound — the legacy T1.6 measured 2.1×
  latency vs zero-pad at the same arch. φ-scaling makes this strictly
  worse because the pad volume grows as `(H + 2·round(PHI·k))² > (H + 2k)²`.
- No fp16 / bf16 issues identified at this scale.

## Mitigations queued for IMPROVEMENTS.md

- Restrict toroidal pad to stage-entry convs only (1/3 the call sites).
- Add `boundary_err` as a side metric in the archive (does NOT enter composite).
- Document the 3x3 collapse explicitly and add a 5x5 kernel ablation.

## Sign-off

- 2026-05-26 — Code-Agent-Y
