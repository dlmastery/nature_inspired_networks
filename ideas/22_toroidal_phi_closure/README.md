# H22 — Toroidal φ-Closure (idea sub-project)

> See the full design document at `hypotheses/H22_toroidal_phi_closure.md`.
> This README is the OPERATOR-FACING summary: implementation status,
> file layout, how to test it, how to run an experiment.

## TL;DR

Replace zero / reflect padding with circular (toroidal) padding so the
last two spatial dims wrap around — the network sees the input as a
torus rather than a bounded plane. On upright CIFAR-10 the prior was
mildly negative (top-1 78.05 % vs `sg_chan_fib` 80.11 %, composite
0.7768 vs 0.8135 → Δ = -0.0367) because the wrap is not data-aligned.
The corrected hypothesis adds φ-scaled wrap distance and re-tests on
tiled / wrap-aware data.

## Status

| stage | done? |
|---|---|
| `implementation.py` written | [x] |
| `tests.py` green (`python tests.py` → all N tests passed) | [x] |
| `AUDIT.md` filed with >= 3 self-found weaknesses | [x] |
| `IMPROVEMENTS.md` records the fixes | [x] |
| `VERIFY.md` signed with date | [x] |
| First experiment archived under `experiments/exp001_*/` | [x] (migrated stub from legacy `experiments/cifar10/sg_only_toroidal_seed0/`) |
| Row added to `../../EXPERIMENT_LOG.md` | [ ] |
| Dashboard refreshed | [ ] |

## Files in this directory

| file | purpose |
|---|---|
| `README.md` | this file |
| `IDEA.md` | formal claim + falsifier + pre-registered prediction |
| `implementation.py` | re-exports `toroidal_pad` and a φ-scaled wrapper from `nature_inspired_networks.priors` |
| `tests.py` | unit tests for the toroidal wrap behavior |
| `AUDIT.md` | self-critique: weaknesses found before first run |
| `IMPROVEMENTS.md` | what was fixed in response to the audit |
| `VERIFY.md` | dated sign-off that tests + smoke pass |
| `experiment.py` | idea-specific experiment driver (wraps the project runner with idea-local config) |
| `configs/` | YAML configs |
| `experiments/exp001_tiled_seed0/` | per-experiment archive (currently a MIGRATED stub pointing to `experiments/cifar10/sg_only_toroidal_seed0/`) |
| `results.md` | auto-generated summary across all archived experiments |
| `dashboard/` | idea-level dashboard |

## How to run the experiment

```powershell
$env:SSL_CERT_FILE = "..\..\.venv\Lib\site-packages\certifi\cacert.pem"
..\..\.venv\Scripts\python tests.py        # must end 'All N tests passed.'
..\..\.venv\Scripts\python experiment.py --config configs\toroidal_phi.yaml --tag exp001_tiled_seed0 --seed 0
```

Or via the global sweep:

```powershell
..\..\.venv\Scripts\python ..\..\scripts\run_sweep.py `
   --config configs\cifar10_tiled.yaml --only sg_toroidal_phi --seeds 0
```

## Cross-references

- Full hypothesis design doc: `../../hypotheses/H22_toroidal_phi_closure.md`
- Master experiment list: `../../EXPERIMENT_LOG.md` row T1.6 (legacy single-prior) + queued T2.6 (φ-scaled wrap-aware re-test)
- Manifesto: `../../README.md`
- IDEA table row: `../../IDEA_TABLE.md` H22
- Composes with: H21 (toroidal hex grid), H26 (fractal toroidal), H62 (LLM toroidal KV)
