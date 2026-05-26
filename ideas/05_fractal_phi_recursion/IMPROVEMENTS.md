# IMPROVEMENTS — H05

> Append-only log of fixes applied in response to `AUDIT.md`.

## Improvement 1 — Closed-form param-factor prediction

- **Addresses:** AUDIT.md weakness #2 (param-budget cost cancels gain).
- **Change:** `ideas/05_fractal_phi_recursion/implementation.py`
  exports `predicted_param_factor(depth)` returning the closed-form
  blow-up factor 1.0/2.0/3.0 for depth=1/2/3. The experiment runner
  can compare this against measured params and refuse runs that
  diverge by more than 50 % (catches topology bugs).
- **Test added / updated:** `test_predicted_param_factor_matches_T1_5_observation`
  asserts the closed-form values; `test_predicted_param_factor_matches_empirical_depth2`
  checks the empirical block param count vs the prediction within
  50 % tolerance.
- **Outcome:** A future edit that silently widens `_FractalPath`'s
  branches will break the empirical-vs-predicted assertion.
- **Date:** 2026-05-27

## Improvement 2 — Isolated `sg_only_fractal` flag combo

- **Addresses:** AUDIT.md weakness #1 (phi-recursion vs vanilla
  fractal-recursion attribution).
- **Change:** `fractal_only_flags()` returns the exact flag combo
  T1.5 ran on. This guarantees the next H05 run is comparable to
  the legacy 11-row sweep without confounding other priors.
- **Test added / updated:** `test_fractal_only_flags_has_only_fractal_on`.
- **Outcome:** Reproducibility of T1.5 at 3 seeds becomes a
  one-import operation.
- **Date:** 2026-05-27

## Improvement 3 — Forward-pass safety for depth=3

- **Addresses:** AUDIT.md weakness #4 (drop-path regularisation
  absent; deeper recursion may produce NaNs).
- **Change:** `test_block_forward_finite_no_nans` exercises depth=3
  to catch future numerical regressions.
- **Test added / updated:** as above.
- **Outcome:** Any future change to `_FractalPath` that destabilises
  depth=3 is caught at unit-test time, not after 12 epochs of GPU.
- **Date:** 2026-05-27

## Open items (audit weaknesses not yet addressed)

- **Phi-shrink at each recursion level (AUDIT #1).** Requires a
  shared-infra extension to `_FractalPath` to accept a `phi_shrink`
  arg. Filed as a shared-infra suggestion at the bottom of the agent
  report. Until then, H05 in this repo is the FractalNet-style
  uniform-width variant.
- **Weighted branch averaging (AUDIT #3).** Same blocker as above
  (needs `_FractalPath` change).
- **Drop-path regularisation (AUDIT #4).** Tracked under H52;
  deferred until that hypothesis lands.
