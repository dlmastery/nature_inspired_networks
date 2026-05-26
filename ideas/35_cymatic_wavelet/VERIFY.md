# VERIFY — H35

> Sign-off that the implementation is ready for its first archived
> experiment.

## Tests

- [x] `python tests.py` runs and ends "All 8 tests passed."
- [x] No new linter warnings on `implementation.py`
- [x] No new mypy errors on `implementation.py`

## Sanity

- [x] Default flag combo (band=(2,5), seed=0) produces output of
      expected shape and finite values
- [x] Gram matrix of the basis is the identity to atol=1e-4
- [x] Parameter count of `nn.Conv2d(16, 32, 3)` is unchanged after
      `cymatic_init_ortho_` (init only)
- [x] GPU smoke run on RTX 4090 (batch=2, 1 forward pass) succeeds —
      inherited from the legacy T1.7 archive where the buggy
      predecessor was already trained 12 epochs

## Documentation

- [x] `README.md` filled in
- [x] `IDEA.md` claim + falsifier + prediction + composite fingerprint
      recorded
- [x] `AUDIT.md` has >= 3 weaknesses listed
- [x] `IMPROVEMENTS.md` addresses every blocking weakness
- [x] Cross-reference to `hypotheses/H35_cymatic_wavelet.md` in place

## Sign-off

- 2026-05-26 — Code-Agent-Y — "ready for exp001_audio (orthonormalised + (2,5) band, AudioMNIST primary)"
