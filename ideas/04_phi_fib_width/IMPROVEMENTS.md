# IMPROVEMENTS — H04

> Append-only log of fixes applied in response to `AUDIT.md`.

## Improvement 1 — Surface mod-8 collapse via `schedules_are_distinct`

- **Addresses:** AUDIT.md weakness #1.
- **Change:** `ideas/04_phi_fib_width/implementation.py` now exports
  `schedules_are_distinct(c0, n_stages)` so the runner (and any
  downstream experiment driver) can pre-flight whether a phi-vs-fib
  comparison is functionally meaningful at the chosen (c0, n).
- **Test added / updated:** `test_mod8_collapse_caught_at_c0_16_n3` and
  `test_schedules_distinct_at_recommended_config` -- the first locks
  in the historical T1.1/T1.2 collapse, the second the recommended
  c0=32, n=4 separation.
- **Outcome:** Future runs at (16, 3) can be refused at config-load
  rather than burning 12 epochs to discover the variants are identical.
- **Date:** 2026-05-27

## Improvement 2 — Monotonicity guard in the wrapper

- **Addresses:** AUDIT.md weakness #2 (the `base = fib[1] = 2` quirk
  can produce non-monotone schedules at very small c0).
- **Change:** `phi_fib_widths` raises if the returned schedule is not
  sorted, and rejects c0 < 8.
- **Test added / updated:** `test_rejects_c0_below_8`,
  `test_phi_widths_strictly_increasing_at_c0_32_n4`,
  `test_linear_mode_is_strict_doubling_proxy`.
- **Outcome:** Pathological (c0, n) configurations now error at the
  idea boundary instead of producing a silently-degenerate model.
- **Date:** 2026-05-27

## Improvement 3 — Locked-in golden numbers for the primary config

- **Addresses:** AUDIT.md weakness #4 (the staircase approximation
  loses exactness, so we lock in the actual integer schedule that
  the experiment will use).
- **Change:** `test_schedules_distinct_at_recommended_config` asserts
  `fib == [32, 48, 80, 128]` and `phi == [32, 48, 80, 136]`. Any
  future edit to `fibonacci_channels` that shifts these numbers will
  cause this test to fail loudly.
- **Test added / updated:** as above.
- **Outcome:** The experiment's integer schedule is now an
  invariant of the test suite, not an emergent property of the run.
- **Date:** 2026-05-27

## Open items (audit weaknesses not yet addressed)

- **No iso-FLOPs control.** Adding a `target_flops` knob requires a
  models-level change; out of scope for this idea sub-project. Filed
  as a shared-infra suggestion at the bottom of the agent report.
- **Mod-8 quirk in the primitive's first stage at c0 < 32.** Will not
  fix here; the wrapper's c0 >= 8 guard covers the dangerous range.
