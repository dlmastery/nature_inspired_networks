# SETUP — `nature_inspired_networks`

> Step-by-step environment bring-up for `dlmastery/nature_inspired_networks`
> on a Windows 11 + RTX 4090 Laptop (16 GB) box. Linux / WSL works too
> (skip the SSL workaround in §4). Reading this top-to-bottom takes you
> from a clean machine to a green SOTA-smoke run in roughly **15 minutes
> of wall-clock + 2 minutes of actual training**.

## 0. Hardware contract

Per [`CLAUDE.md`](CLAUDE.md) §2 the canonical hardware is:

- **GPU:** 1× NVIDIA RTX 4090 Laptop, 16 GB VRAM.
- **OS:** Windows 11 (the corporate-SSL workaround in §4 is needed
  here; on Linux it can be skipped).
- **Driver / CUDA:** NVIDIA driver ≥ 555, CUDA 12.x runtime (`nvidia-smi`
  must report `CUDA Version: 12.x`).
- **AMP:** bf16 (an Ada-generation requirement;
  `torch.cuda.is_bf16_supported()` must return `True`).
- **Batch size:** 256 by default; halve in a *new* config file if you
  see `CUDA out of memory` (Rule 1 — never edit an existing config).

Linux equivalent works (Ubuntu 22.04 + CUDA 12.x + driver ≥ 555 tested
informally); A100 / H100 are untested.

## 1. Software prerequisites

- **Python 3.13** (tested on 3.13.13; 3.10 / 3.11 / 3.12 also work).
  Must be on `PATH` (`python --version`).
- **git** (always required).
- **curl.exe** (ships with Windows 10+; used for the corporate-SSL
  CIFAR download in §4).
- **gh** CLI (optional, for repo publish).
- **~3 GB free disk** (CIFAR-10 tarball + extracted pickles + venv).

## 2. Clone

```powershell
git clone https://github.com/dlmastery/nature_inspired_networks.git
cd nature_inspired_networks
```

## 3. Create venv + install PyTorch + editable install

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip wheel setuptools
.\.venv\Scripts\python -m pip install `
   --index-url https://download.pytorch.org/whl/cu124 torch torchvision
.\.venv\Scripts\python -m pip install -e .
```

The `-e .` editable install pulls in the rest of `pyproject.toml`'s
dependencies (`timm`, `einops`, `fvcore`, `numpy`, `pandas`,
`matplotlib`, `seaborn`, `scikit-learn`, `scikit-image`, `rich`,
`pyyaml`, `tqdm`, `tensorboard`, `medmnist`, `persim`, `ripser`).

### 3.1 Verify GPU

```powershell
.\.venv\Scripts\python -c "import torch; print('cuda', torch.cuda.is_available(), `
'device:', torch.cuda.get_device_name(0), 'bf16:', torch.cuda.is_bf16_supported())"
```

Expected:

```
cuda True device: NVIDIA GeForce RTX 4090 Laptop GPU bf16: True
```

If `cuda False`: confirm `nvidia-smi` shows the GPU + a CUDA-12.x
runtime; reinstall PyTorch with the CUDA wheel from §3 (the default
PyPI wheel is CPU-only).

## 4. Download CIFAR-10 (corporate-proxy-safe)

Python 3.13's `urllib` rejects the corporate cert chain shipped on
many Windows boxes (`Basic Constraints of CA cert not marked
critical`). Bypass it with `curl.exe -k` — torchvision still verifies
the tarball's MD5, so the data integrity guarantee is preserved:

```powershell
curl.exe -kL -o data\cifar-10-python.tar.gz `
   https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
.\.venv\Scripts\python -c "from torchvision.datasets import CIFAR10; `
CIFAR10(root='./data', train=True, download=True); `
CIFAR10(root='./data', train=False, download=True)"
```

(Optional) MedMNIST `PathMNIST` for the medical-imaging side study:

```powershell
.\.venv\Scripts\python -c "from medmnist import PathMNIST; `
PathMNIST(split='train', download=True, root='./data/medmnist')"
```

CIFAR-100 is downloaded analogously the first time a CIFAR-100 config
is run; the tarball MD5 is checked by torchvision.

## 5. Mandatory environment variables (Rule 26)

After the 2026-05-28 thread-cap-related crash, the project's hardware
contract is extended to include the following **mandatory** environment
variables before any long-running sweep or multi-agent launch. Without
them, PyTorch + numpy + scipy + MKL each spin up `os.cpu_count()`
threads and oversubscribe the CPU when 5+ agents + a GPU sweep + the
auto-checkpoint loop run concurrently.

```powershell
$env:OMP_NUM_THREADS = 2     # cap OpenMP threads
$env:MKL_NUM_THREADS = 2     # cap MKL threads
$env:KMP_DUPLICATE_LIB_OK = "TRUE"   # tolerate Intel OMP duplicates
```

These leave enough cores for OS / IDE / Claude Code to remain
responsive and prevent the system-wide hang we hit on 2026-05-28.

`num_workers = 0` for `DataLoader` is also mandatory on Windows (see
all configs under `configs/`): spawn-start workers wedge on Windows
and silently hang the sweep. The cap is set in YAML, not at the call
site, so the value is auditable and one-config-change-per-experiment
compliant (Rule 1).

## 6. Verify the test suite (Rule 12)

Before any training run, the unit-test suite must be green
([`CLAUDE.md`](CLAUDE.md) Rule 12).

```powershell
.\.venv\Scripts\python -m pytest tests\ -q
```

Expected tail (the line that confirms suite health):

```
... 780+ passed in ~75s
All N tests passed.
```

If anything fails, **STOP** — do not launch a sweep. A red test is a
defect, not a style issue.

Per-idea test sub-suites live under `ideas/<NN>_<short>/tests.py`; they
are exercised by the same pytest invocation through
`tests/test_idea_suites.py` and also runnable directly with
`pytest ideas/`.

## 7. SOTA-smoke pre-flight (Rule 13)

Every experiment workflow on CIFAR-10 starts with a SOTA-recipe
baseline smoke. The expected band is **top-1 ≥ 80 % @ 12 epochs** for
the ResNet-20 + AdamW + cosine + label-smoothing + RandomCrop/HFlip
recipe. If the smoke falls below the band, STOP and diagnose the
environment — *do not* run any nature-inspired variant.

```powershell
.\.venv\Scripts\python -m nature_inspired_networks.runner `
  --config configs\cifar10_sota_smoke.yaml --tag smoke --seed 0
```

Wall-clock: ≈ 2 min on the canonical hardware. The runner writes
`experiments/cifar10/smoke_seed0/metrics.json`. Verify:

```powershell
.\.venv\Scripts\python -c "import json; d = json.load(open(r'experiments\cifar10\smoke_seed0\metrics.json')); `
print('top1:', d['top1'], 'composite:', d['composite']); `
assert d['top1'] >= 0.80, 'SOTA smoke BELOW expected band — STOP and diagnose'; print('OK')"
```

Expected: `top1: 0.81–0.84 composite: ~0.84 OK`.

## 8. Curated 13-row CIFAR-10 ablation sweep (≈ 90 min)

```powershell
.\.venv\Scripts\python -u scripts\run_sweep.py `
   --config configs\cifar10_quick.yaml --seeds 0 --skip-existing
.\.venv\Scripts\python scripts\compute_topology.py --seeds 0
.\.venv\Scripts\python scripts\build_dashboard.py
.\.venv\Scripts\python scripts\build_report.py
start dashboard\dashboard.html
```

`scripts/build_dashboard.py` regenerates the aggregate
`dashboard/dashboard.html` and mirrors it to `docs/dashboard/` for
GitHub Pages.

## 9. 3-seed Phase-5 re-sweep (≈ 5 hr)

For error bars on the Phase-4 graduates:

```powershell
.\.venv\Scripts\python -u scripts\run_sweep.py `
   --config configs\cifar10_quick.yaml --seeds 0 1 2 --skip-existing
.\.venv\Scripts\python scripts\build_dashboard.py
```

(Per Rule 11, fire an auto-checkpoint loop in a second terminal during
any sweep > 15 min — see
[`skills/autoresearch-auto-checkpoint-loop/`](skills/autoresearch-auto-checkpoint-loop/).)

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `SSL: CERTIFICATE_VERIFY_FAILED — Basic Constraints of CA cert not marked critical` | Corporate proxy injects an MITM cert that Python 3.13's `urllib` rejects | Use `curl.exe -kL` to download the tarball out-of-band (§4); torchvision still verifies MD5 |
| Sweep silently hangs after `[run]  baseline_sg_vanilla` | `num_workers > 0` on Windows spawns workers that wedge | Set `num_workers: 0` in the config (the project-wide default — Rule 26) |
| `CUDA out of memory` at batch 256 | < 16 GB available (other apps holding VRAM) | Drop batch to 128 in a **new** config file (do not edit `cifar10_quick.yaml` — Rule 1) |
| `ImportError: ripser` on Python 3.13 | C extension occasionally needs a rebuild on 3.13 | `pip install --no-binary :all: ripser` (or upgrade pip and retry the editable install) |
| Multi-agent + sweep + checkpoint loop hangs OS | Thread oversubscription (Rule 26) | Set `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `KMP_DUPLICATE_LIB_OK=TRUE` before launch (§5) |
| `CompositeFingerprintError` on launch | Composite formula in `eval.py` was edited (Rule 2) | Revert the edit or open a new branch / repo (new formula ≡ new science) |
| `ValueError: Reasoning entry rejected` | Citation or word-count floors not met (Rule 4/5) | Rewrite the failing field; there is no `--bypass` flag (Rule 7) |
| SOTA smoke top-1 < 80 % | Env drift / torch version / corrupted data | STOP, diagnose; do not run variants (Rule 13) |
| `CUDA Version: 0.0` in `nvidia-smi` | Driver / runtime mismatch | Update NVIDIA driver to ≥ 555; reinstall PyTorch CUDA wheel from §3 |

## 11. Linux / WSL quick-start

The Linux flow is identical except §4 (use `wget` instead of
`curl.exe`) and §5 (the OMP/MKL envvars use shell `export`):

```bash
git clone https://github.com/dlmastery/nature_inspired_networks.git
cd nature_inspired_networks
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -e .
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 KMP_DUPLICATE_LIB_OK=TRUE
pytest tests/ -q
python -m nature_inspired_networks.runner \
  --config configs/cifar10_sota_smoke.yaml --tag smoke --seed 0
```

---

*Last refreshed: 2026-05-29. See [`README.md`](README.md) §11 for
broader reproducibility pointers and
[`NEURIPS_CHECKLIST.md`](../paper/NEURIPS_CHECKLIST.md) for the per-question
NeurIPS Paper Checklist.*
