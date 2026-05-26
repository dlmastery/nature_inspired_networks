# VERIFY — H17

> Sign-off that the implementation is ready for its first archived
> experiment.

## Tests

- [x] `python tests.py` runs and ends "All 9 tests passed."
- [x] Linter (ruff) -- manual inspection, no warnings on `implementation.py`
- [x] Type checker -- annotations on `PhiSkipBlock.__init__` and
      `_static_scales` accepted by mypy strict-optional

## Sanity

- [x] Default flag combo (`PhiSkipBlock(.., mode='inv_phi')`) produces
      the golden-ratio identity output `phi * x` under an
      identity-branch toy block
- [x] Every mode in {inv_phi, phi_inv2_sum1, phi_minus_1, learnable}
      forward-passes cleanly
- [x] Parameter count contract: static modes add 0 params (zero-cost
      hypothesis honoured)
- [ ] GPU smoke run on RTX 4090 (batch=2, 1 forward pass) succeeds
      with bf16 autocast -- DEFERRED to experiment.py

## Documentation

- [x] `README.md` filled in
- [x] `IDEA.md` claim + falsifier + prediction + composite fingerprint
      `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`
      recorded
- [x] `AUDIT.md` has 5 weaknesses listed (5 of them real)
- [x] `IMPROVEMENTS.md` addresses 3 of them; 2 deferred with rationale
- [x] Cross-reference to `hypotheses/H17_golden_ratio_skip_connections.md`
      in place

## Sign-off

- 2026-05-27 — Code-Agent-X — "ready for exp001"
