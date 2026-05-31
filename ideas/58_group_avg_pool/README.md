# H58 — C4 max-pool to avg-pool fix (idea sub-project)

> **NOTE:** there is no formal `hypotheses/g6_topological_bridging/H58_group_avg_pool.md`
> design doc in this repo. The H58 hypothesis is documented in
> `FINDINGS.md` § "H58 follow-up — the avg-pool fix DISCARDED",
> reproduced inline in this README and in `IDEA.md`. Once a formal
> `hypotheses/g6_topological_bridging/H58_*.md` is written, replace this paragraph with the
> standard "See the full design document at ..." cross-reference.

## TL;DR

**Pre-registered prediction (H58):** the C4 `GroupConv2d` collapses
its 4-rotation orbit with `amax(dim=1)`, which "throws away 75 % of
the signal at every spatial location." Replacing the max-pool with a
mean-pool over the orbit was expected to recover **+5 to +10 pp**
top-1 on `sg_only_group` (the second-worst legacy row at 69.84 %).

**Verdict: DISCARD.** Mean-pool was **WORSE** than max-pool, in both
the single-prior and full-hybrid configurations. The intuition was
wrong: max-pool over rotated copies is a *soft argmax over
orientations* that preserves the strongest evidence; mean-pool
*dilutes* discriminative features by averaging the response across
orientations that don't match the input.

**Observed (single seed, 12 epochs):**

| variant | reduce | top-1 | params | composite | Δ vs same-arch max |
|---|---|---|---|---|---|
| `sg_only_group` | max | 69.84 % | 127 k | 0.6937 | (ref) |
| `sg_only_group_avg` | mean | **65.38 %** | 127 k | **0.6513** | **-4.46 pp top-1** |
| `sg_full_fib` | max | 73.24 % | 259 k | 0.6966 | (ref) |
| `sg_full_fib_avg` | mean | **66.86 %** | 259 k | **0.6432** | **-6.38 pp top-1** |

The falsifier (the predicted +5 pp lift fails to materialize, AND
composite drops by >= 0.005) is hit decisively.

## Status

| stage | done? |
|---|---|
| `implementation.py` written (re-export of GroupConv2d with reduce='mean') | [x] |
| `tests.py` green (`python tests.py` → all N tests passed) | [x] |
| `AUDIT.md` filed with >= 3 self-found weaknesses | [x] |
| `IMPROVEMENTS.md` records the discarded experiment and the future direction | [x] |
| `VERIFY.md` signed with date | [x] |
| First experiment archived under `experiments/exp001_*/` | [x] (migrated stub from legacy `experiments/cifar10/sg_only_group_avg_seed0/`) |
| Row added to `../../EXPERIMENT_LOG.md` | [x] (T2.1 — DISCARD) |
| Dashboard refreshed | [x] |

## Files in this directory

| file | purpose |
|---|---|
| `README.md` | this file |
| `IDEA.md` | formal claim + falsifier + DISCARD verdict |
| `implementation.py` | re-exports `GroupConv2d` with `reduce='mean'` and notes the DISCARD verdict in its docstring |
| `tests.py` | unit tests for the avg-pool orbit reduction |
| `AUDIT.md` | self-critique: why the intuition failed |
| `IMPROVEMENTS.md` | the future direction (rotated CIFAR-10, IcoMNIST, spherical MNIST) |
| `VERIFY.md` | dated sign-off |
| `experiment.py` | idea-specific experiment driver |
| `configs/` | YAML configs |
| `experiments/exp001_sg_only_group_avg_seed0/` | MIGRATED stub pointing to `experiments/cifar10/sg_only_group_avg_seed0/` |
| `results.md` | rollup |
| `dashboard/` | idea-level dashboard |

## Future direction (the actual fix is the DATA, not the operator)

From `FINDINGS.md`:

> The fix is not the reduction operator but the data. C4-equivariant
> features are useful on data with rotational variance (rotated CIFAR,
> IcoMNIST, spherical MNIST) - not on canonically-oriented CIFAR-10.
> The next experiment should test the *same* `sg_only_group` (max-pool)
> variant on rotated-CIFAR-10 where the equivariance prior is
> data-aligned.

## Cross-references

- Findings: `https://github.com/dlmastery/nature_inspired_networks/blob/main/paper/FINDINGS.md` § "H58 follow-up - the avg-pool fix DISCARDED"
- Related: `ideas/50_full_sacred_hybrid/` (the parent failure mode);
  H24 (icosahedral / Platonic equivariance, the real follow-up).
- Master experiment list: `../../EXPERIMENT_LOG.md` row T2.1
  (`sg_only_group_avg`, DISCARD).
