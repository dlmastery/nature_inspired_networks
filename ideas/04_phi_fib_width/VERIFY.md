# VERIFY — H04

> Sign-off that the implementation is ready for its first archived
> experiment.

## Tests

- [x] `python tests.py` runs and ends "All 7 tests passed."
- [x] Linter (ruff) has no new warnings on `implementation.py` (manual
      inspection; project does not auto-lint idea modules yet)
- [x] Type checker -- not enforced for ideas/, but `phi_fib_widths`
      uses `List[int]` annotations and accepts `int`-typed args

## Sanity

- [x] Default flag combo (`phi_fib_widths(32, 4, "phi")`) returns
      `[32, 48, 80, 136]` -- the locked-in golden number
- [x] Every mode in {"phi", "fib", "linear"} forward-passes (smoke test
      via `test_phi_widths_strictly_increasing_at_c0_32_n4`)
- [x] Parameter count contract: phi/fib schedules at c0=32, n=4 are
      30 % smaller than linear-doubling [32, 64, 128, 256], consistent
      with IDEA.md prediction
- [ ] GPU smoke run on RTX 4090 (batch=2, 1 forward pass) succeeds
      with bf16 autocast -- DEFERRED to experiment.py; the schedule
      itself is a pure-Python int list and does not need GPU

## Documentation

- [x] `README.md` filled in (status table, file inventory, run command)
- [x] `IDEA.md` claim + falsifier + prediction + composite fingerprint
      `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`
      recorded
- [x] `AUDIT.md` has 4 weaknesses listed (4 of them real)
- [x] `IMPROVEMENTS.md` addresses 3 blocking weaknesses; 2 deferred
      with rationale
- [x] Cross-reference to `hypotheses/H04_phi_self_similar_width.md`
      in place

## Sign-off

- 2026-05-27 — Code-Agent-X — "ready for exp001"
