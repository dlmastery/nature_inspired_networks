# exp001_sg_only_group_avg_seed0 — MIGRATED stub for H58 (avg-pool, DISCARDED)

> **MIGRATED FROM** `experiments/cifar10/sg_only_group_avg_seed0/`
> (legacy single-seed CIFAR-10 H58 follow-up run, T2.1).
> Companion run is at `experiments/cifar10/sg_full_fib_avg_seed0/`.

This archive sub-directory is a **pointer** to the H58 follow-up run
that **DISCARDED** the avg-pool hypothesis. The full per-seed
artifacts (`metrics.json`, `history.json`, `config.yaml`, `best.pt`,
`reasoning.json` if present) live at the legacy path.

H58 was the FIRST run in the project to save `best.pt` checkpoints,
which made trained-feature Betti curves possible for the first time
(see `FINDINGS.md` § "Trained-feature Betti (first data)").

## Headline numbers from the migrated source

### Primary: `sg_only_group_avg` (single-prior, this directory)

| metric | observed | reference (`sg_only_group`, max-pool) | Δ |
|---|---|---|---|
| top-1 | **65.38 %** | 69.84 % | **-4.46 pp** |
| top-5 | 96.98 % | — | — |
| params | 127 146 | 127 146 | 0 |
| FLOPs | 108.3 M | 108.3 M | 0 |
| latency (ms, b=1) | 8.82 | 9.75 | -0.93 (mean is slightly faster here) |
| rot-eq err | 0.7983 | — | larger (worse) |
| **composite** | **0.6513** | 0.6937 | **-0.0424** |
| composite fingerprint | `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893` | (same) | — |
| epochs | 12 | 12 | — |
| seed | 0 | 0 | — |
| train_seconds | 486.9 | — | — |
| generalization_gap | 0.018 | — | — |

### Companion: `sg_full_fib_avg` (full hybrid + avg-pool)

From `experiments/cifar10/sg_full_fib_avg_seed0/metrics.json`:

| metric | observed | reference (`sg_full_fib`, max-pool) | Δ |
|---|---|---|---|
| top-1 | **66.86 %** | 73.24 % | **-6.38 pp** |
| params | 259 443 | 259 443 | 0 |
| composite | **0.6432** | 0.6966 | **-0.0534** |

## Verdict at this scale

**DISCARDED.** The IDEA.md falsifier (need >= +1.0 pp top-1 recovery
AND composite Δ > -0.005) is hit decisively in the WRONG direction.
The pre-registered prediction was +5 to +10 pp; the observed delta is
-4.46 pp on `sg_only_group_avg` and -6.38 pp on `sg_full_fib_avg`.

The intuition "max-pool over the C4 orbit throws away 75 % of the
signal" was wrong: the 4 orbit channels are correlated rotated copies
of the same convolution; `amax` is a soft argmax over orientations
that preserves the strongest evidence at every spatial location.
Mean-pool dilutes that response by averaging in the three orientations
that DO NOT match the input.

## Future direction (the fix is the DATA, not the operator)

Quoting `FINDINGS.md`:

> The fix is not the reduction operator but the data. C4-equivariant
> features are useful on data with rotational variance (rotated CIFAR,
> IcoMNIST, spherical MNIST) — not on canonically-oriented CIFAR-10.
> The next experiment should test the *same* `sg_only_group` (max-pool)
> variant on rotated-CIFAR-10 where the equivariance prior is
> data-aligned.

## Trained-feature Betti curves (first data of the campaign)

This run was the first in the project to save `best.pt` checkpoints,
enabling trained-feature Betti curves for the first time. Two
observations from `FINDINGS.md`:

| variant | β₀ per stage (trained) | β₀ per stage (fresh-init) |
|---|---|---|
| `sg_only_group_avg` | [121, 127, 127, 127] | [93, 71, 122, 100] |
| `sg_full_fib_avg` | [123, 124, 127, 127] | [123, 124, 127, 127]* |

Trained models *increase* β₀ relative to fresh init — they cluster
features by class, producing more isolated connected components.
β₁ also rises (one 1-D hole detected in `sg_full_fib_avg`),
consistent with class-prototype loop structures. This inverts the
naive "topology simplification" reading.

## Reproduce the migrated run

```powershell
$env:SSL_CERT_FILE = "..\..\..\..\.venv\Lib\site-packages\certifi\cacert.pem"
..\..\..\..\.venv\Scripts\python ..\..\..\..\scripts\run_sweep.py `
   --config ..\..\..\..\configs\cifar10_quick.yaml `
   --only sg_only_group_avg --seeds 0 --skip-existing
```

## See also

- Legacy archives:
  - `../../../../experiments/cifar10/sg_only_group_avg_seed0/`
  - `../../../../experiments/cifar10/sg_full_fib_avg_seed0/`
  - `../../../../experiments/cifar10/sg_only_group_seed0/` (max-pool reference)
- Headline campaign verdict: `../../../../FINDINGS.md` § "H58 follow-up - the avg-pool fix DISCARDED"
- Future direction: rotated-CIFAR-10 with max-pool (the actually-better operator) - queued under H24
- Parent failure: `../../50_full_sacred_hybrid/` (H50 - the headline negative)
