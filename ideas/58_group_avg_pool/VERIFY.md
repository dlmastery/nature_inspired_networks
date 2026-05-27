# VERIFY — H58

> Sign-off for the DISCARDED avg-pool fix.

## Tests

- [x] `python tests.py` runs and ends "All 8 tests passed."
- [x] No new linter warnings on `implementation.py`
- [x] No new mypy errors on `implementation.py`

## Sanity

- [x] `make_avg_pool_group_conv()` returns a `GroupConv2d` with
      `reduce='mean'`
- [x] Forward pass at stride=1 keeps (H, W); at stride=2 downsamples
- [x] Mean-pool output is bounded above by max-pool output at every
      spatial location (sanity invariant)
- [x] Mean-pool and max-pool outputs are numerically distinct (no
      silent collapse of one onto the other)
- [x] GPU smoke run on RTX 4090 — inherited from the T2.1 archive
      (`experiments/cifar10/sg_only_group_avg_seed0/`)

## Documentation

- [x] `README.md` notes the absence of `hypotheses/g6_topological_bridging/H58_*.md` and
      points to `FINDINGS.md` as the canonical record (per the
      task brief)
- [x] `IDEA.md` records the original prediction AND the empirical
      disproof; falsifier status = DISCARDED
- [x] `AUDIT.md` lists >= 3 weaknesses + doubles as a post-mortem
- [x] `IMPROVEMENTS.md` records the future direction (rotated CIFAR,
      IcoMNIST, spherical MNIST) and the documentation task
- [x] Cross-reference to `FINDINGS.md` § "H58 follow-up - the avg-pool
      fix DISCARDED" is in place

## Sign-off

- 2026-05-27 — original sign-off at DISCARD verdict
- 2026-05-26 — Code-Agent-Y — re-signed for `ideas/` taxonomy
  ("DISCARDED; future work is rotated-CIFAR, not reduction-operator")
