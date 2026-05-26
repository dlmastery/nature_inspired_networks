# AUDIT — H35

> Adversarial self-critique BEFORE first experiment. Treat the
> implementation as if reviewing it for a hostile NeurIPS reviewer.

## Weaknesses found in v1

1. **Per-(out, in)-channel sign-flip is a hack, not a principled
   diversification scheme.** `cymatic_init_ortho_` reuses the SAME
   orthonormal mode across all input channels of a given output channel
   and only flips the sign by `(o + i) % 2`. This means input-channel
   diversification is bimodal at best — half the input channels see
   `+mode`, the other half see `-mode`. A proper fix would either
   (a) orthonormalise across the joint `(out, in)` axis (expensive:
   QR on a `(k*k, out*in)` matrix), or (b) draw a fresh `(m, n)` per
   `(o, i)` pair from the band. We chose the cheap hack for v1; the
   audit lists it as a known limitation.
2. **Band-randomization seed is independent of `torch.manual_seed`.**
   `chladni_modes_banded` takes its own `seed` argument and uses
   Python's `random.Random` rather than the torch RNG. This means
   running `set_seed(0)` then calling `cymatic_init_ortho_` does NOT
   yield bit-identical weights across runs unless the operator also
   passes `seed=0`. Reproducibility hazard for the multi-seed sweep.
3. **He-variance preservation is asserted within +/- 50%, not 10%.**
   The IDEA.md prediction claimed "variance matched to He init within
   10%". In practice the orthonormal-then-rescale pipeline gives a
   structured init whose per-tensor std is closer to `0.7 * he_std`
   for the (2, 5) band on 3x3 kernels — the test relaxes the bound to
   `[0.5x, 1.5x]` and the README narrative softens the claim. A future
   improvement is to add a per-fan-in calibration constant; for now we
   document the deviation honestly.

## Bugs caught by tests (good)

- `test_chladni_modes_banded_orthonormal_across_modes` would have
  caught a bug where QR was applied along the wrong axis.
- `test_cymatic_init_zeros_bias_if_present` catches the easy regression
  of forgetting to zero the bias after a structured init (a sin-product
  with non-zero DC would otherwise inject a constant offset).

## Bugs NOT caught by tests but suspected

- The `band=(1, 1)` legacy regression (the T1.7 bug) is NOT explicitly
  tested. We have `test_band_low_eq_high_is_uniform_mode` for band=(3, 3),
  but no test that EXPLICITLY guards against an operator passing
  band=(1, 1) and silently getting a near-DC kernel back. Should add a
  warning or assert in `cymatic_init_ortho_`.
- The init does not interact correctly with `GroupConv2d` weights,
  which are stored as a single `(O, I, k, k)` tensor that is then
  orbited under C4 rotations. Initialising it as a Chladni mode that is
  itself rotation-symmetric would collapse the orbit to a near-rank-1
  output — pathological. This is why `NaturePriorBlock` already skips
  cymatic init for the group-conv path; the audit reaffirms that
  design choice.

## Performance / numerical-stability concerns

- The init walks `(out_c, in_c)` in Python and copies one filter at a
  time. For `out_c = 256, in_c = 256` (deeper stages) this is
  `65 536` Python-level Tensor.copy_ calls per layer. Slow but
  one-shot — measured at ~150 ms per layer at the deepest CIFAR stage.

## Mitigations queued for IMPROVEMENTS.md

- Add a deprecation-style assert if `band == (1, 1)`.
- Vectorize the inner loop with a single `torch.einsum`.
- Pipe the torch RNG state into `chladni_modes_banded` via a fresh
  `seed = torch.initial_seed() & 0xFFFFFFFF` so `set_seed` controls
  init.

## Sign-off

- 2026-05-26 — Code-Agent-Y
