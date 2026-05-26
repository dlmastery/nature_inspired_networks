# IMPROVEMENTS — H58

> Append-only log of refinements queued or executed in response to
> the H58 DISCARD verdict.

## Improvement 1 — The future direction is the DATA, not the operator

- **Addresses:** AUDIT.md weakness #3 — the equivariance prior is
  unused when the test set has no rotational variance.
- **Change:** queued experiment runs the **same** `sg_only_group`
  (with `reduce='max'`, the actually-better choice) on
  **rotated-CIFAR-10** (uniform random rotation in [0°, 360°] per
  test image). Implementation: add a `rotated-cifar10` dataset
  variant to `src/nature_inspired_networks/data.py` and a sweep tag
  `sg_only_group_rotcifar`.
- **Test added / updated:** none in this idea sub-dir; the test will
  land alongside the new dataset loader.
- **Outcome:** PENDING. Expected: `sg_only_group_rotcifar` recovers
  >= +3 pp top-1 over a vanilla rotated-CIFAR-10 baseline, finally
  validating the equivariance prior on data where it pays off. If it
  ALSO fails, the lesson is even stronger: equivariance priors on
  CIFAR-scale data are not worth the latency tax.
- **Date:** queued 2026-05-27

## Improvement 2 — Document the wrong intuition in a hypothesis doc

- **Addresses:** future-Claude / future-operator should not re-derive
  the "75 % signal loss" mistake from first principles.
- **Change:** write `hypotheses/H58_group_avg_pool.md` (currently
  absent — README points to FINDINGS.md as the canonical record)
  with the standard 12-section committee-grade structure. The
  motivation section must explain WHY the intuition was wrong, not
  just present the alternative.
- **Test added / updated:** none.
- **Outcome:** PENDING. Filed as a documentation task in
  `EXPERIMENT_LOG.md`.
- **Date:** queued 2026-05-27

## Improvement 3 — Cross-paradigm test: IcoMNIST / spherical MNIST

- **Addresses:** AUDIT.md weakness #3 + the broader claim that
  equivariance priors need data alignment.
- **Change:** queued as H24 follow-up. Test `sg_only_group` on
  **IcoMNIST** (icosahedral MNIST) and **spherical MNIST** — these
  datasets have explicit rotational symmetry that C4 conv is
  designed for. C4 is only a proxy for the full icosahedral group,
  so this is also a stress test of the proxy approximation.
- **Test added / updated:** none in this idea sub-dir.
- **Outcome:** PENDING.
- **Date:** queued 2026-05-27

## Open items

- The 3-seed reproduction of `sg_only_group_avg` and `sg_full_fib_avg`
  is NOT queued — single-seed already hit the falsifier with a 4-6 pp
  margin in the wrong direction; multi-seed cannot flip that sign.
- The trained-feature Betti curves from H58 (the first `best.pt`
  checkpoints in the project) ARE worth a follow-up: per-epoch β₀
  tracking to see if/when the β-curves cross the random baseline. See
  FINDINGS.md § "Trained-feature Betti (first data)".
