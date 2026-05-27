# H50 — Full Sacred-Geometry Hybrid (idea sub-project)

> See the full design document at `hypotheses/g5_optimization_init_reg_nas/H50_full_sacred_hybrid.md`.
> This README is the OPERATOR-FACING summary: implementation status,
> file layout, the **headline negative result**, and what comes next.

## TL;DR

Activate all six nature-inspired priors simultaneously inside a
single `NaturePriorBlock` (Fib channels + hex conv + C4 group conv +
fractal recursion + toroidal padding + golden-modulate gate + cymatic
init) and measure on CIFAR-10. The original PDF source predicted a
+20–50 % efficiency gain via compounding. **The actual result is the
opposite — `sg_full_fib` is the WORST single row in the 11-run sweep.**

**Observed (single seed, 12 epochs):**

| metric | observed | reference (`sg_chan_fib`) | Δ |
|---|---|---|---|
| top-1 | **73.24 %** | 80.11 % | **-6.87 pp** |
| params | 259 k | 127 k | +103 % |
| latency (ms, b=1) | 20.02 | 4.43 | **+352 % (5.0×)** |
| **composite** | **0.6966** | 0.8135 | **-0.1169** |
| rot-eq err | 0.6827 | — | -0.045 (only positive!) |
| composite fingerprint | `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893` | (same) | — |

The falsifier from IDEA.md (Δ <= -0.005) is hit **23× over** in the
negative direction. H50 as originally stated is **DISPROVED at
12-epoch CIFAR-10 scale**. This is the headline negative result of the
entire research program.

## Status

| stage | done? |
|---|---|
| `implementation.py` written | [x] |
| `tests.py` green (`python tests.py` → all N tests passed) | [x] |
| `AUDIT.md` filed with >= 3 self-found weaknesses | [x] |
| `IMPROVEMENTS.md` records the fixes | [x] |
| `VERIFY.md` signed with date | [x] |
| First experiment archived under `experiments/exp001_*/` | [x] (migrated stub from legacy `experiments/cifar10/sg_full_fib_seed0/`) |
| Row added to `../../EXPERIMENT_LOG.md` | [x] (T1.9) |
| Dashboard refreshed | [x] |

## Files in this directory

| file | purpose |
|---|---|
| `README.md` | this file |
| `IDEA.md` | formal claim + falsifier + pre-registered prediction + the disproof |
| `implementation.py` | composes `NaturePriorBlock` with `NaturePriorFlags(hex=True, group=True, fractal=True, toroidal=True, cymatic_init=True, golden_modulate=True)` |
| `tests.py` | unit tests for the full-on flag combination |
| `AUDIT.md` | self-critique of the failure modes |
| `IMPROVEMENTS.md` | what was tried in response (H58 avg-pool, leave-one-out) |
| `VERIFY.md` | dated sign-off |
| `experiment.py` | idea-specific experiment driver |
| `configs/` | YAML configs |
| `experiments/exp001_sg_full_fib_seed0/` | MIGRATED stub pointing to `experiments/cifar10/sg_full_fib_seed0/` |
| `results.md` | rollup |
| `dashboard/` | idea-level dashboard |

## How to run the experiment

```powershell
$env:SSL_CERT_FILE = "..\..\.venv\Lib\site-packages\certifi\cacert.pem"
..\..\.venv\Scripts\python tests.py        # must end 'All N tests passed.'
..\..\.venv\Scripts\python experiment.py --config configs\full_hybrid.yaml --tag exp001_sg_full_fib_seed0 --seed 0
```

## Cross-references

- Full hypothesis design doc: `../../hypotheses/g5_optimization_init_reg_nas/H50_full_sacred_hybrid.md`
- Headline campaign verdict: `../../FINDINGS.md`
- Refined follow-up hypotheses:
  - **H58** — group avg-pool fix (DISCARDED — `ideas/58_group_avg_pool/`)
  - **H45** — Sacred NAS to find the right subset
  - **H67** — curated full-paradigm hybrid (LLM track)
  - **H60** — 3-seed re-sweep for error bars
- Master experiment list: `../../EXPERIMENT_LOG.md` row T1.9
