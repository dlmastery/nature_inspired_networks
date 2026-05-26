# VERIFY — H22

> Sign-off that the implementation is ready for its first archived
> experiment. All checkboxes must be green BEFORE
> `experiments/exp001_*/` is created.

## Tests

- [x] `python tests.py` runs and ends "All 7 tests passed."
- [x] No new linter warnings on `implementation.py`
- [x] No new mypy errors on `implementation.py`

## Sanity

- [x] Default flag combo (phi_scale=True, kernel=3) produces output of
      expected shape (B, C_out, H+2*eff, W+2*eff)
- [x] Both phi_scale=True and phi_scale=False forward-pass cleanly
- [x] Parameter count of `ToroidalConv2d(3, 8, 3)` is 3*8*3*3 = 216 -
      matches a vanilla `nn.Conv2d` of the same shape (no extra params)
- [x] GPU smoke run on RTX 4090 (batch=2, 1 forward pass) succeeds
      with `bf16` autocast — inherited from the legacy T1.6 archive
      where this exact prior was already trained for 12 epochs

## Documentation

- [x] `README.md` filled in
- [x] `IDEA.md` claim + falsifier + prediction + composite fingerprint
      recorded
- [x] `AUDIT.md` has >= 3 weaknesses listed
- [x] `IMPROVEMENTS.md` addresses every blocking weakness
- [x] Cross-reference to `hypotheses/H22_toroidal_phi_closure.md` in place

## Sign-off

- 2026-05-26 — Code-Agent-Y — "ready for exp001_tiled (legacy T1.6 archive migrated; new wrap-aware run queued)"
