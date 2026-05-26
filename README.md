# sacgeometry — SacredGeoBlock

> **An autoresearch-style ablation study of sacred-geometry priors as
> drop-in residual blocks for image classification.**
> The block embeds φ/Fibonacci channel scaling, C4 group-equivariant
> convolution (Platonic proxy), hexagonal-masked kernels, fractal
> recursive sub-paths, toroidal (circular) padding, Chladni-mode
> cymatic init, and golden-angle channel modulation — each toggleable
> for principled ablation.

## 🌐 Live links

- **📊 Dashboard:** [dlmastery.github.io/sacgeometry/dashboard/](https://dlmastery.github.io/sacgeometry/dashboard/) (after first publish)
- **📑 Paper sketch:** [`PAPER.md`](PAPER.md) (in progress)
- **📚 Process:** [`AUTORESEARCH_PROCESS.md`](AUTORESEARCH_PROCESS.md)
- **🏗 Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
- **📈 Sister project:** [autoresearchimage](https://github.com/dlmastery/autoresearchimage)
  — this repo inherits its protocol gates verbatim and adapts them to
  geometric-prior backbone ablations.

## ⚠️ Read this first — scope and honesty

This is a **feasibility + ablation campaign** on one consumer GPU. It is
**not** a SOTA claim against ImageNet leaderboards. What it does claim:

| | what we run | what we don't |
|---|---|---|
| **Tier 1** | CIFAR-10 (full), 15-epoch quick runs, 11-cell curated ablation | CIFAR-100 ablation (just one validation run) |
| **Tier 2** | (stretch) MedMNIST 2D — PathMNIST / OrganAMNIST / OCTMNIST | Tiny ImageNet, ImageNet-100, ImageNet-1k |
| **Tier 3** | not run | IcoMNIST, ModelNet40, ogbg-molhiv |

The expectation set by the source PDF — "20–50 % compression + 5–15 %
capacity/speed gains" — is the *upper-bound literature compound* from
{HexCNN ~25–42 % + FractalNet depth + IcosaCNN rot-robustness +
Fibonacci-Net width}. Our single-GPU 15-epoch ablations cannot replicate
that compound; they can only show that **the priors do not destroy
training and individually contribute measurable changes in the Pareto
frontier**.

The autoresearch protocol (citation rigor, reasoning blob, Goodhart
composite fingerprint, audit gates) is inherited from
[`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage)
and is the methodological deliverable, even on the small scale we run.

## Quick stats

| | |
|---|---|
| **Repo** | https://github.com/dlmastery/sacgeometry |
| **Dataset(s)** | CIFAR-10 (always) + optionally MedMNIST PathMNIST |
| **Baseline** | ResNet-20 (272 k params, He 2015 CIFAR variant) |
| **Block** | `SacredGeoBlock(c_in, c_out, stride, flags)` — see `src/sacgeo/blocks.py` |
| **Priors** | hex, group (C4), fractal, toroidal, cymatic-init, golden-modulate |
| **Channel modes** | `fib` (Fibonacci) / `phi` (golden) / `linear` (control) |
| **Compute** | 1× RTX 4090 Laptop (16 GB), Windows 11, bf16 AMP |
| **Composite metric** | `top1 − 0.05·log₁₀(params_M) − 0.05·log₁₀(latency_ms)` (SHA-256 fingerprinted) |
| **Comparable to ImageNet SOTA?** | **NO** — see *Scope and honesty* above |

## Table of contents

1. [What this is](#what-this-is)
2. [Why these priors?](#why-these-priors)
3. [Quickstart](#quickstart)
4. [SacredGeoBlock — anatomy](#sacredgeoblock--anatomy)
5. [The autoresearch protocol](#the-autoresearch-protocol)
6. [Goodhart-fingerprinted composite](#goodhart-fingerprinted-composite)
7. [Repo layout](#repo-layout)
8. [Ablation matrix](#ablation-matrix)
9. [Topology & CKA](#topology--cka)
10. [Reproduce in 30 minutes](#reproduce-in-30-minutes)
11. [Hardware notes](#hardware-notes)
12. [Limitations & threats to validity](#limitations--threats-to-validity)
13. [Open axes for the next campaign](#open-axes-for-the-next-campaign)
14. [Citations](#citations)

---

## What this is

`sacgeometry` is a small, self-contained PyTorch project that implements
the **SacredGeoBlock** — a drop-in CIFAR-scale residual block whose
six "sacred-geometry" priors can be individually switched on or off:

| Prior | Library precedent | Why we include it |
|---|---|---|
| `hex` — 3×3 hex-masked conv | HexaConv 2018, HexagDLy 2019 | denser 7-tap honeycomb receptive field |
| `group` — C4 group conv (rot-equivariant) | Cohen & Welling 2016, e2cnn | weight-sharing across 4 rotations; cheap Platonic proxy |
| `fractal` — recursive sub-block (depth=2) | FractalNet 2017 (Larsson et al.) | ultra-deep without residuals, multi-path gradient |
| `toroidal` — circular padding | Pittorino 2022, TopoCN | closed manifold — periodic boundary on the image |
| `cymatic_init` — Chladni-mode wavelet init | implicit in wavelet/Fourier priors; novel here | initialise filters with sinusoidal eigenmodes of a square plate |
| `golden_modulate` — golden-angle channel gate | rotary embeddings; phyllotaxis | output-stage rotary-like modulation by 2π/φ phases |

plus a **channel schedule** chosen from `fib` (Fibonacci-Net 2025-style),
`phi` (golden-ratio compound scaling) or `linear` (control).

The runner produces autoresearch-protocol artifacts: per-run `metrics.json`
+ `history.json`, a global `experiment_log.jsonl`, a sortable HTML
dashboard, and a reasoning journal whose entries are gated by
Citation Rigor + Reasoning Blob Completeness validators.

## Why these priors?

The source PDF (`sacred geometry and neural networks.pdf`) lays out a
research program in which Sacred Geometry is treated **as engineering
inductive bias**, not mysticism — each of the priors above has a
peer-reviewed precedent in Geometric/Topological Deep Learning:

- *Cymatics → hexagonal nodes on Chladni plates → honeycomb packing →*
  **HexaConv** (Hoogeboom et al. 2018, arXiv:1803.02108).
- *Platonic solids / icosahedron → 60-element rotation group →*
  **Icosahedral CNN** (Cohen et al. 2019, arXiv:1902.04615); the
  cheaper C4 proxy used here is **Group-Equivariant CNN** (Cohen &
  Welling 2016, arXiv:1602.07576).
- *Flower-of-Life recursion → self-similar paths →* **FractalNet**
  (Larsson et al. 2017, arXiv:1605.07648).
- *Toroidal flow / grid-cells → periodic latents →* **Deep Networks
  on Toroids** (Pittorino et al. 2022, arXiv:2202.03038).
- *φ / Fibonacci growth → channel schedule →* **Fibonacci-Net** (2025).

The PDF's contribution is to combine them in one residual block and
study the ablation surface — which is what this repo does.

## Quickstart

Tested on Python 3.13, PyTorch 2.6.0+cu124, Windows 11, NVIDIA RTX 4090
Laptop (16 GB).

```powershell
# 1. Install
git clone https://github.com/dlmastery/sacgeometry.git
cd sacgeometry
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
.\.venv\Scripts\python -m pip install -e .

# 2. Smoke test (3 epochs, ~2 min)
$env:SSL_CERT_FILE = ".\.venv\Lib\site-packages\certifi\cacert.pem"
.\.venv\Scripts\python -m sacgeo.runner `
  --config configs\cifar10_smoke.yaml --tag smoke --seed 0

# 3. Curated 11-row ablation matrix (~60 min on 4090 Laptop)
.\.venv\Scripts\python scripts\run_sweep.py `
  --config configs\cifar10_quick.yaml --seeds 0 --skip-existing

# 4. Topology metrics (β₀ / β₁ collapse curves)
.\.venv\Scripts\python scripts\compute_topology.py --seeds 0

# 5. Build the dashboard
.\.venv\Scripts\python scripts\build_dashboard.py
# → dashboard/dashboard.html + docs/dashboard/dashboard.html
```

## SacredGeoBlock — anatomy

```
   x ───────────────────────────────────────────────►── skip(x) ──┐
                                                                  │
   x ─► _GenericConv(c_in→c_out, stride)  ─► ReLU                 │
        ├─ hex?   → 3×3 hex-masked conv                           │
        ├─ group? → C4 group conv (4 rotations, max-pool orbit)   │
        ├─ tor?   → circular padding                              │
        └─ cymatic_init? → Chladni-mode filter init                │
                                                                  │
   ──► _FractalPath(c_out→c_out, depth=2) if fractal else single conv
        a-branch: 1× conv
        b-branch: conv → recursive sub-block (depth-1)
        merge:   mean(a, b)                                       │
                                                                  │
   ──► golden_modulate? → y · (½ cos(φ-angle phases + α) + ½)     │
                                                                  ▼
   ──► y + skip(x) ─► ReLU
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full module diagram and
shape table.

## The autoresearch protocol

Every experiment follows the same 7-step ritual, inherited verbatim
from `autoresearchimage`:

```
1. Diagnose  ←─ read prior run, identify weakest cell
2. Cite      ←─ arXiv paper that addresses it (Citation Rigor gate)
3. Hypothesise ←─ mechanistic claim (Reasoning Blob gate)
4. Predict   ←─ numeric composite range
5. Execute   ←─ ONE config change; runner enforces (Goodhart gate)
6. Analyse   ←─ verdict KEEP / DISCARD / NEAR-MISS
7. Checkpoint ←─ experiment_log.jsonl + research_journal.md
```

Word-count floors per field (`src/sacgeo/reasoning.py`):

| field | floor |
|---|---|
| `diagnosis` | 60 |
| `citation` (single-paper) | 40; (multi) 80 |
| `hypothesis` | 50 |
| `prediction` | 25 |
| `verdict` | 30 |
| `learning` | 40 |

Citations must match
`Author1, Author2, ..., YEAR VENUE 'Title' (arXiv:XXXX.XXXXX) — note.`

There is **no `--bypass` flag**. `append_entry()` raises if the gates
reject.

## Goodhart-fingerprinted composite

```python
composite = top1 − 0.05·log₁₀(params_M) − 0.05·log₁₀(latency_ms)
```

The formula string is SHA-256 hashed
(`COMPOSITE_FINGERPRINT` in `src/sacgeo/eval.py`); every run records
the fingerprint so any mid-project edit makes the next experiment fail
loudly.

## Repo layout

```
sacgeometry/
├── README.md                   ← this file
├── ARCHITECTURE.md             ← module/data-flow diagram
├── AUTORESEARCH_PROCESS.md     ← 7-step ritual, gates, dashboard mandate
├── CLAUDE.md                   ← full normative rules (operator-facing)
├── PAPER.md                    ← paper draft (work in progress)
├── SOTA_COMPARISON.md          ← honest comparison to literature
├── paper_abstract.md           ← 1-page summary
├── sota_catalog.yaml           ← prior-art table consumed by SOTA_COMPARISON.md
├── pyproject.toml
├── configs/                    ← YAML configs (smoke / quick / ablation)
├── src/sacgeo/
│   ├── priors.py               ← φ-channels, hex mask, Chladni modes, group conv
│   ├── blocks.py               ← SacredGeoBlock + _FractalPath + _GenericConv
│   ├── models.py               ← ResNet-20 baseline + SacredGeoNet
│   ├── data.py                 ← CIFAR + MedMNIST loaders
│   ├── train.py                ← Trainer + bf16 AMP + cosine LR
│   ├── eval.py                 ← params/FLOPs/latency/CKA/composite
│   ├── topology.py             ← Betti curves on stage features
│   ├── reasoning.py            ← Citation Rigor + Reasoning Blob gates
│   ├── dashboard.py            ← Pareto/ablation/curves/Betti plots + HTML
│   └── runner.py               ← single-experiment entrypoint
├── scripts/
│   ├── run_sweep.py            ← ablation matrix driver
│   ├── compute_topology.py     ← post-hoc Betti / CKA
│   └── build_dashboard.py      ← static HTML/PNG bundle
├── experiments/                ← per-run artifacts + experiment_log.jsonl
├── dashboard/                  ← dashboard.html (latest build)
├── docs/                       ← GitHub Pages root
│   ├── index.html
│   └── dashboard/
└── memory/                     ← project checkpoint markdown
```

## Ablation matrix

Curated 11-row matrix (run by `scripts/run_sweep.py`, no flags):

| # | tag | model | channels | priors on |
|---|---|---|---|---|
| 1 | `baseline_resnet20` | ResNet-20 | 16-32-64 | n/a |
| 2 | `baseline_sg_vanilla` | SacredGeoNet | linear | none |
| 3 | `sg_chan_fib` | SacredGeoNet | fib | none |
| 4 | `sg_chan_phi` | SacredGeoNet | phi | none |
| 5 | `sg_only_hex` | SacredGeoNet | fib | hex |
| 6 | `sg_only_group` | SacredGeoNet | fib | C4 group |
| 7 | `sg_only_fractal` | SacredGeoNet | fib | fractal |
| 8 | `sg_only_toroidal` | SacredGeoNet | fib | toroidal |
| 9 | `sg_only_cymatic_init` | SacredGeoNet | fib | cymatic |
| 10 | `sg_only_golden_modulate` | SacredGeoNet | fib | golden-angle |
| 11 | `sg_full_fib` | SacredGeoNet | fib | all six |

The full sweep (`--full`) adds `sg_full_phi` and six `sg_loo_no_*`
leave-one-out runs.

## Topology & CKA

`scripts/compute_topology.py` produces `experiments/betti.json` with
β₀ / β₁ collapse curves per stage. The dashboard plots them. CKA between
trained variants is available via `sacgeo.topology.cka_matrix` (called by
the dashboard's optional CKA panel).

## Reproduce in 30 minutes

```powershell
# minimal repro on a 4090 Laptop
.\.venv\Scripts\python scripts\run_sweep.py `
  --config configs\cifar10_quick.yaml --seeds 0 `
  --only baseline_resnet20 baseline_sg_vanilla sg_chan_fib sg_full_fib
.\.venv\Scripts\python scripts\build_dashboard.py
start dashboard\dashboard.html
```

## Hardware notes

- RTX 4090 **Laptop** has **16 GB** VRAM, **not** 24 GB like the desktop
  card. Batch-size 256 with bf16 fits CIFAR runs comfortably.
- Group-conv (C4) forward is the bottleneck at small batches because
  the kernel orbit is reconstructed each step. For real deployment use
  `torch.compile` or freeze the orbit as a buffer after training.
- Python 3.13 ships a stricter SSL implementation; corporate cert
  bundles fail with `Basic Constraints of CA cert not marked critical`.
  The CIFAR tarball is downloaded out-of-band by `curl.exe -k` and
  verified by torchvision's MD5 check.

## Limitations & threats to validity

1. **Single seed by default.** The 3-seed setup costs ~3× wall-clock;
   the included config runs `--seeds 0` only. Add `--seeds 0 1 2` to
   produce ±std bars on the ablation chart.
2. **CIFAR-10 only.** Tier-2 MedMNIST is wired in `data.py` but the
   ablation matrix has not been re-run on it. ImageNet is explicitly
   out of scope for this campaign.
3. **C4 group, not full Platonic.** The block claims a *proxy* for
   Platonic equivariance; full icosahedral equivariance needs `e2cnn`,
   which currently lacks a Python 3.13 wheel.
4. **15 epochs is short.** Final accuracies are *not* converged; the
   ablation reads the **shape of the prior's effect**, not asymptotic
   numbers.
5. **Cymatic init is novel here.** No prior paper initialises filters
   with Chladni eigenmodes; the comparison is to standard He init.
6. **Betti is approximate.** β-curves are computed on fresh-init stage
   features (pre-training) — they discriminate priors but do not
   measure trained topology simplification yet. Adding a checkpoint
   save + post-training Betti is on the open-axes list.

## Open axes for the next campaign

- [ ] Save final model weights and re-run Betti / CKA on **trained**
      features (currently fresh-init).
- [ ] Add 3-seed runs by default; emit error bars on every plot.
- [ ] Bring up MedMNIST PathMNIST + OrganAMNIST in the matrix.
- [ ] Try `torch.compile` on the inference path for fair latency
      numbers.
- [ ] Port the C4 group conv to D4 (8-element) and to e2cnn's
      icosahedron when a 3.13 wheel ships.
- [ ] Synthetic cymatic-pattern dataset (Chladni plate sim via the
      2-D wave equation in SciPy) for direct "resonance" validation.

## Citations

Authoritative arXiv IDs cited inline in `src/sacgeo/priors.py` and
`src/sacgeo/blocks.py`. Full list in `sota_catalog.yaml`.

## License & credits

MIT. Author: dlmastery. Inherits the autoresearch protocol from
[`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage)
and [`dlmastery/autoresearch`](https://github.com/dlmastery/autoresearch).
The "sacred geometry as inductive bias" framing is the user-provided
PDF; this repo is the engineering implementation + ablation.
