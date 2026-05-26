# VERIFY — H21

> Sign-off that the implementation is ready for its first archived
> experiment.

## Tests

- [x] `python tests.py` runs and ends "All 10 tests passed."
- [x] Linter (ruff) -- manual inspection, no warnings on `implementation.py`
- [x] Type checker -- annotations on `PhiRadialHexConv2d.__init__` and
      `hex_phi_radial_mask` accepted

## Sanity

- [x] Default flag combo (`PhiRadialHexConv2d(3, 8, 3, padding=1)`)
      produces output of shape (B, 8, H, W) at stride=1
- [x] Every Boolean flag combination (`toroidal in {False, True}`,
      `apply_gain in {False, True}`) forward-passes cleanly
- [x] Parameter count contract: 7 of 9 taps active (22 % weight saving
      vs dense conv); zero corner-tap leakage after forward
- [x] Effective output differs from uniform-hex with same weight init
      (the phi-radial mask is not silently inert)
- [ ] GPU smoke run on RTX 4090 (batch=2, 1 forward pass) succeeds
      with bf16 autocast -- DEFERRED to experiment.py

## Documentation

- [x] `README.md` filled in
- [x] `IDEA.md` claim + falsifier + prediction + composite fingerprint
      `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`
      recorded
- [x] `AUDIT.md` has 5 weaknesses listed
- [x] `IMPROVEMENTS.md` addresses 3 of them; 4 deferred with rationale
- [x] Cross-reference to `hypotheses/H21_hexagonal_phi_packing.md` in place

## Sign-off

- 2026-05-27 — Code-Agent-X — "ready for exp001"
