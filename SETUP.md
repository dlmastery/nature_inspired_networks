# SETUP

> Step-by-step environment bring-up for `dlmastery/sacgeometry` on a
> Windows 11 + RTX 4090 Laptop (16 GB) machine. Linux / WSL works too
> (skip the SSL workaround).

## 0. Prerequisites

- Windows 11 with NVIDIA driver ≥ 555 (`nvidia-smi` must report a
  CUDA 12.x runtime)
- Python 3.10–3.13 in PATH (this repo is tested on 3.13.13)
- ~3 GB free disk (CIFAR-10 tarball + extracted pickles + venv)
- (Optional) `gh` CLI for repo publish; `git` always required

## 1. Clone

```powershell
git clone https://github.com/dlmastery/sacgeometry.git
cd sacgeometry
```

## 2. Create venv + install CUDA torch

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip wheel setuptools
.\.venv\Scripts\python -m pip install `
   --index-url https://download.pytorch.org/whl/cu124 torch torchvision
.\.venv\Scripts\python -m pip install -e .
```

The `-e .` editable install pulls in the rest of `pyproject.toml`'s
dependencies (timm, einops, fvcore, matplotlib, seaborn, pandas,
scikit-learn, rich, pyyaml, tqdm, tensorboard, medmnist, persim, ripser).

## 3. Verify GPU

```powershell
.\.venv\Scripts\python -c "import torch; print('cuda', torch.cuda.is_available(),
'device:', torch.cuda.get_device_name(0), 'bf16:', torch.cuda.is_bf16_supported())"
```

Expected: `cuda True device: NVIDIA GeForce RTX 4090 Laptop GPU bf16: True`.

## 4. Download CIFAR-10 (corp-proxy-safe)

Python 3.13's `urllib` rejects the corporate cert chain shipped on many
Windows boxes (`Basic Constraints of CA cert not marked critical`).
Bypass it with `curl.exe -k` (the tarball's MD5 is checked by
torchvision):

```powershell
curl.exe -kL -o data\cifar-10-python.tar.gz `
   https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

Then let torchvision verify + extract:

```powershell
.\.venv\Scripts\python -c "from torchvision.datasets import CIFAR10;
CIFAR10(root='./data', train=True, download=True);
CIFAR10(root='./data', train=False, download=True)"
```

If you also want MedMNIST PathMNIST:

```powershell
.\.venv\Scripts\python -c "from medmnist import PathMNIST;
PathMNIST(split='train', download=True, root='./data/medmnist')"
```

## 5. Smoke test (≈ 2 min)

```powershell
$env:SSL_CERT_FILE = ".\.venv\Lib\site-packages\certifi\cacert.pem"
.\.venv\Scripts\python -m sacgeo.runner `
   --config configs\cifar10_smoke.yaml --tag smoke --seed 0
```

Expected: `done in ~110s; experiments\cifar10\smoke_seed0\metrics.json`
with `top1 ~ 0.55` (3 epochs).

## 6. Full curated sweep (≈ 90 min)

```powershell
.\.venv\Scripts\python -u scripts\run_sweep.py `
   --config configs\cifar10_quick.yaml --seeds 0 --skip-existing
.\.venv\Scripts\python scripts\build_dashboard.py
start dashboard\dashboard.html
```

## Troubleshooting

| symptom | fix |
|---|---|
| `SSL: CERTIFICATE_VERIFY_FAILED — Basic Constraints of CA cert not marked critical` | Use `curl.exe -kL` to download tarball out-of-band; torchvision MD5 still verifies content |
| Sweep silently hangs after `[run]  baseline_sg_vanilla` line | `num_workers > 0` on Windows can spawn workers that wedge; set `num_workers: 0` in the config |
| `CUDA out of memory` at batch 256 | Drop to 128 in a new config file (don't edit `cifar10_quick.yaml`) |
| `ImportError: ripser` on Python 3.13 | `pip install ripser` (we install it but the C extension occasionally needs a rebuild) |
