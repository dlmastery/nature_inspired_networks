# H35 — Cymatic Wavelet Kernels (idea sub-project)

> See the full design document at `hypotheses/g4_kernels_attention_filters/H35_cymatic_wavelet.md`.
> This README is the OPERATOR-FACING summary: implementation status,
> file layout, how to test it, how to run an experiment.

## TL;DR

Initialize convolutional kernels from Chladni-plate eigenmodes (the
standing-wave patterns of the 2-D wave equation on a square boundary).
On upright CIFAR-10 the un-corrected init was **unexpectedly negative**
(top-1 77.44 % vs `sg_chan_fib` 80.11 %, composite 0.7883 vs 0.8135
→ Δ = -0.0252). Root-cause analysis identified two bugs — no
Gram-Schmidt across output channels and DC-dominated (1,1) frequency
band — and one wrong dataset (CIFAR has no harmonic structure). The
corrected hypothesis adds orthonormalization and a (2,5) band, and
re-tests on AudioMNIST spectrograms.

## Status

| stage | done? |
|---|---|
| `implementation.py` written | [x] |
| `tests.py` green (`python tests.py` → all N tests passed) | [x] |
| `AUDIT.md` filed with >= 3 self-found weaknesses | [x] |
| `IMPROVEMENTS.md` records the fixes | [x] |
| `VERIFY.md` signed with date | [x] |
| First experiment archived under `experiments/exp001_*/` | [x] (migrated stub from legacy `experiments/cifar10/sg_only_cymatic_init_seed0/`) |
| Row added to `../../EXPERIMENT_LOG.md` | [ ] |
| Dashboard refreshed | [ ] |

## Files in this directory

| file | purpose |
|---|---|
| `README.md` | this file |
| `IDEA.md` | formal claim + falsifier + pre-registered prediction |
| `implementation.py` | re-exports `chladni_modes` and `cymatic_init_` from `nature_inspired_networks.priors`, plus a Gram-Schmidt-orthonormalised variant for the corrected hypothesis |
| `tests.py` | unit tests for the Chladni-mode basis + init |
| `AUDIT.md` | self-critique: weaknesses found before first run |
| `IMPROVEMENTS.md` | what was fixed in response to the audit |
| `VERIFY.md` | dated sign-off that tests + smoke pass |
| `experiment.py` | idea-specific experiment driver (wraps the project runner with idea-local config) |
| `configs/` | YAML configs |
| `experiments/exp001_audio_seed0/` | per-experiment archive (currently a MIGRATED stub pointing to `experiments/cifar10/sg_only_cymatic_init_seed0/`) |
| `results.md` | auto-generated summary across all archived experiments |
| `dashboard/` | idea-level dashboard |

## How to run the experiment

```powershell
$env:SSL_CERT_FILE = "..\..\.venv\Lib\site-packages\certifi\cacert.pem"
..\..\.venv\Scripts\python tests.py        # must end 'All N tests passed.'
..\..\.venv\Scripts\python experiment.py --config configs\audiomnist.yaml --tag exp001_audio_seed0 --seed 0
```

## Cross-references

- Full hypothesis design doc: `../../hypotheses/g4_kernels_attention_filters/H35_cymatic_wavelet.md`
- Master experiment list: `../../EXPERIMENT_LOG.md` row T1.7 (legacy single-prior negative) + queued T2.7 (orthonormalized + band-corrected re-test)
- IDEA table row: `../../IDEA_TABLE.md` H35
- Composes with: H28 (cymatic hex resonance), H46 (cymatic loss), H56 (cymatic dataset), H66 (cymatic QKV)
