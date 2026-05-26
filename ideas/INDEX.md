# ideas/ — TAXONOMY INDEX

> Every hypothesis from `IDEA_TABLE.md` gets its own sub-directory here,
> following the contract in `ideas/_TEMPLATE/`. Each sub-directory is
> self-contained: anyone reading it cold should be able to (a) understand
> the hypothesis, (b) build and test the implementation, and (c) reproduce
> every archived experiment under its `experiments/` folder.

## Numbering convention

Sub-directories are `NN_<short_name>/` where `NN` is the hypothesis ID from
`IDEA_TABLE.md` (zero-padded to 2 digits). The `_TEMPLATE/` directory
provides the canonical scaffold to copy.

## Status

| NN | Idea | Status | Exemplar? |
|---|---|---|---|
| _TEMPLATE | Canonical scaffold | ✓ done | (template) |
| 00 | Baseline ResNet-20 (literature anchor) | ○ planned | |
| 04 | φ / Fibonacci channel scaling | ◐ exemplar in progress | **YES** |
| 05 | Fractal φ-recursion (sub-block depth=2) | ○ planned | |
| 17 | Golden-ratio skip / channel modulation | ○ planned | |
| 21 | Hex φ-packing (hex-masked conv) | ○ planned | |
| 22 | Toroidal φ-closure (circular padding) | ○ planned | |
| 24 | C4 / D4 group conv (Platonic proxy) | ○ planned | |
| 35 | Cymatic (Chladni-eigenmode) init | ○ planned | |
| 50 | Full hybrid (all 6 priors on) | ○ planned | |
| 58 | **C4 max→avg pool fix** | ◐ exemplar in progress | **YES** |
| 59 | Trained-feature Betti curves | ○ planned | |
| 60 | 3-seed uncertainty bars | ○ planned | |
| 61–71 | Cross-paradigm hybrids (extended-transcript) | ○ planned | |

Full design space + status: see `../IDEA_TABLE.md`.
Master experiment list: see `../EXPERIMENT_LOG.md`.

## How a new idea sub-directory gets made

1. Copy `_TEMPLATE/` to `NN_<short_name>/`.
2. Fill in `README.md`, `IDEA.md` (the hypothesis + falsifier).
3. Implement `implementation.py` (import primitives from
   `nature_inspired_networks`; do not duplicate).
4. Write `tests.py` exercising shape + every branch.
5. Run `python ideas/NN_<short>/tests.py`; must end "All N tests passed."
6. Self-audit in `AUDIT.md`; fix issues in `IMPROVEMENTS.md`; seal in
   `VERIFY.md`.
7. Author a pre-run reasoning entry under
   `experiments/exp001_<short>/reasoning.json`.
8. Run the experiment, archive artifacts under
   `experiments/exp001_<short>/run_seed0/`.
9. Update `results.md` + the per-experiment `README.md`.
10. Add a row to `../EXPERIMENT_LOG.md` and commit + push.
