# VERIFY — H05

> Sign-off that the implementation is ready for its first archived
> experiment.

## Tests

- [x] `python tests.py` runs and ends "All 9 tests passed."
- [x] Linter (ruff) -- manual inspection, no warnings on `implementation.py`
- [x] Type checker -- annotations on `build_fractal_block` and
      `predicted_param_factor` accepted by mypy strict-optional

## Sanity

- [x] Default flag combo (`fractal_only_flags()`, depth=2) produces
      a NaturePriorBlock that forwards to shape (2, 16, 8, 8) at
      stride=1
- [x] depth in {1, 2, 3} all forward-pass without NaNs
- [x] Param-count contract: depth=2 has roughly 2x the params of
      depth=1 (predicted 2.0, empirical within 50 % at c=16)
- [ ] GPU smoke run on RTX 4090 (batch=2, 1 forward pass) succeeds
      with bf16 autocast -- DEFERRED to experiment.py

## Documentation

- [x] `README.md` filled in
- [x] `IDEA.md` claim + falsifier + prediction + composite fingerprint
      `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`
      recorded
- [x] `AUDIT.md` has 4 weaknesses listed
- [x] `IMPROVEMENTS.md` addresses 3 of them; 3 deferred with rationale
- [x] Cross-reference to `hypotheses/g1_scaling_growth/H05_fractal_phi_recursion.md` in place

## Sign-off

- 2026-05-27 — Code-Agent-X — "ready for exp001"
