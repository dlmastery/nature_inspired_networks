# AUDIT — H04

> Adversarial self-critique BEFORE first experiment. Treat the
> implementation as if reviewing it for a hostile NeurIPS reviewer.

## Weaknesses found in v1

1. **Mod-8 quantisation collapses phi and fib at the default config.**
   `src/nature_inspired_networks/priors.py::fibonacci_channels` applies
   `_round8` AFTER the geometric expansion. At c0=16, n=3 BOTH modes
   round to the same integer schedule [16, 24, 40], destroying the
   discriminating signal. This is the root cause of T1.1/T1.2's
   identical 80.11 % top-1. Every honest phi-vs-fib comparison MUST
   pre-flight `schedules_are_distinct(c0, n)` and refuse to launch if
   it returns False. The wrapper added in this idea sub-project surfaces
   the issue but does not enforce it at config-load time -- the runner
   does not call the guard.

2. **The 'fib' mode uses `base = fib[1] = 2` (not 1).**
   The primitive's loop is `c = c0 * fib[k+1] / base` with the Fibonacci
   sequence seeded `[1, 2, 3, 5, ...]`. So for c0=32 the unrounded
   sequence is 32 * [1/2, 2/2, 3/2, 5/2] = [16, 32, 48, 80] -- which is
   then bumped to [32, 48, 80, ...] only because `_round8(c=16)` floors
   to 8 then to max(8,..). The k=0 entry of the 'fib' schedule does NOT
   equal c0 in general -- it equals c0/2 mod 8. For c0=32 you happen
   to get back [32, 48, 80, 128] but for c0=8 you get [8, 8, 16, 24],
   which is a silent monotonicity bug. We do not fix it here (out of
   scope: blocks-level edit) but we do test for it.

3. **No iso-FLOPs control.** The H04 prediction interval claims a
   composite lift, but the phi schedule has ~30 % fewer params than
   linear doubling. The composite formula penalises params with
   `-0.05 * log10(params_M)`, which slightly REWARDS the phi variant
   for being smaller. A fair test would compare at iso-FLOPs (scale up
   the phi network's depth or c0 to match linear-doubling FLOPs) but
   neither this sub-project nor the shared infra exposes an iso-FLOPs
   knob.

4. **The phi growth ratio is approximate after mod-8 rounding.**
   The test `test_phi_growth_ratio_close_to_phi` only requires the ratio
   in [PHI*0.75, PHI*1.25]. At c0=32, n=4 the actual ratios are
   48/32=1.5, 80/48=1.667, 136/80=1.7 -- all within tolerance but none
   exactly phi. The Fibonacci-quantised schedule is therefore a STAIRCASE
   approximation to a phi-power curve, not the curve itself.

## Bugs caught by tests (good)

- The mod-8 collapse at c0=16, n=3 (T1.1/T1.2 historical pathology) is
  now locked-in by `test_mod8_collapse_caught_at_c0_16_n3`. Future edits
  to `fibonacci_channels` that "fix" this by post-hoc rerounding will
  be visible as test failures.
- `schedules_are_distinct(16, 3) is False` is asserted, so the runner
  can use this guard with confidence.

## Bugs NOT caught by tests but suspected

- The `_round8` helper uses `max(8, int(round(x/8))*8)`. At very small
  c0 (c0=4..7) the schedule degenerates to a constant 8 across all
  stages, but the wrapper's "c0 >= 8" guard sidesteps this; if a user
  passes c0=10 the second stage may still equal 8 in 'phi' mode for
  certain k.
- The pre-registered prediction in IDEA.md assumes a 3-seed median at
  c0=32, n=4. We have NO data at that config. The interval [-0.012, +0.010]
  is a pre-registration, not a verified bound.

## Performance / numerical-stability concerns

- None at the schedule level. The runtime cost of `fibonacci_channels`
  is microseconds; it is called once at model construction.

## Mitigations queued for IMPROVEMENTS.md

- Add a `validate_separation` helper that the runner can call at
  config-load time to reject pathological (c0, n) combinations.
- Document the `base = fib[1] = 2` quirk in the shared primitive's
  docstring (this requires a `src/` edit -- coordinate with the
  infra-agent).

## Sign-off

- 2026-05-27 — Code-Agent-X
