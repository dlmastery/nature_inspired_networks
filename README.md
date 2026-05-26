# nature_inspired_networks

> **An autoresearch-style ablation study of nature-inspired priors —
> hex / Platonic / fractal / toroidal / φ / cymatic / golden-angle — as
> drop-in residual & attention blocks for image classification and
> decoder-only LLMs, with refusal-to-launch protocol gates, a
> Goodhart-fingerprinted composite metric, and a publicly-pushed
> dashboard refreshed every commit.**

[![public](https://img.shields.io/badge/repo-public-brightgreen)](https://github.com/dlmastery/nature_inspired_networks)
[![dashboard](https://img.shields.io/badge/dashboard-live-blue)](https://dlmastery.github.io/nature_inspired_networks/dashboard/dashboard.html)
[![hypotheses](https://img.shields.io/badge/hypotheses-75-orange)](hypotheses/INDEX.md)
[![tests](https://img.shields.io/badge/tests-29%20core%20%2B%2068%20idea--local-green)](tests/)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

**Quick links:** [MINDMAP](MINDMAP.md) · [Live dashboard](https://dlmastery.github.io/nature_inspired_networks/dashboard/dashboard.html) · [Manifesto](MANIFESTO.md) · [75 hypotheses](hypotheses/INDEX.md) · [Findings](FINDINGS.md) · [Paradigm comparison](PARADIGM_COMPARISON.md) · [Nature-inspired networks reference](NATURE_INSPIRED_NETWORKS.md)

---

## Table of contents

1. [Overview](#1-overview)
2. [Headline finding (honest scope)](#2-headline-finding-honest-scope)
3. [What's in this repo](#3-whats-in-this-repo)
4. [Quick start](#4-quick-start)
5. [Architecture at a glance](#5-architecture-at-a-glance)
6. [The 75-hypothesis design space](#6-the-75-hypothesis-design-space)
7. [Modular `ideas/` taxonomy](#7-modular-ideas-taxonomy)
8. [The autoresearch protocol](#8-the-autoresearch-protocol)
9. [Reproducing the experiments](#9-reproducing-the-experiments)
10. [Extending this work](#10-extending-this-work)
11. [Documentation map](#11-documentation-map)
12. [Sister projects](#12-sister-projects)
13. [License & credits](#13-license--credits)

---

## 1. Overview

`nature_inspired_networks` studies whether the inductive biases that
nature uses — hexagonal lattices (cymatics, honeycomb, grid cells),
Platonic / icosahedral symmetries, fractal self-similarity, toroidal
manifolds (entorhinal grid cells), golden-ratio φ growth (phyllotaxis),
Chladni-eigenmode wave patterns — provide engineering advantage when
**imposed explicitly** rather than waiting for them to emerge from
scale (the Platonic Representation Hypothesis, Huh et al. 2024
[arXiv:2405.07987](https://arxiv.org/abs/2405.07987)).

The core artifact is **`NaturePriorBlock`** — a residual / attention
block whose seven inductive biases are each Boolean-toggleable so a
clean ablation matrix maps each prior's marginal effect. The block
drops into a CIFAR-scale ResNet-20-shaped scaffold *and* a decoder-only
GPT-style stack; the design space totals **75 hypotheses** across **7
thematic groups** (see [§6](#6-the-75-hypothesis-design-space)).

Every experiment runs inside the **autoresearch protocol** inherited
from [`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage):
Citation Rigor + Reasoning Blob Completeness + Goodhart-fingerprinted
composite metric + per-experiment archive subdirectory with a
mandatory detailed README. There is no `--bypass` flag.

## 2. Headline finding (honest scope)

The first 13-run CIFAR-10 campaign (single seed, 12 epochs each, RTX 4090
Laptop, bf16 AMP) produced **a clear falsifiable negative result**:

| rank | tag | top-1 | params | composite |
|---|---|---|---|---|
| 1 | `baseline_resnet20` (15 ep) | 84.78 % | 272 k | **0.8458** |
| 2 | `baseline_sg_vanilla` | 82.16 % | 186 k | 0.8258 |
| 5 | `sg_only_fractal` | 82.46 % | 259 k | 0.8104 (**only single lifter**) |
| 10 | `sg_only_group` (C4, max-pool) | 69.84 % | 127 k | 0.6937 |
| 11 | `sg_full_fib` (all 6 priors on) | 73.24 % | 259 k | **0.6966 (WORST)** |
| 12 | `sg_only_group_avg` (H58 fix attempt) | 65.38 % | 127 k | 0.6597 (**FALSIFIED**) |
| 13 | `sg_full_fib_avg` | 66.86 % | 259 k | 0.6432 |

**Three honest findings:**

1. **The full hybrid is the worst NaturePrior variant.** At 12-epoch
   CIFAR-10 the priors do NOT compound; the popular "20–50 % compound
   efficiency" claim from the source PDF does not survive a single-GPU
   ablation.
2. **`sg_only_fractal` is the only single prior that lifts top-1**
   (+2.35 pp), at 2× parameter cost. (Code-X discovered the
   implementation was uniform-width FractalNet — the "φ" part of
   "φ-recursion" was never wired in, so the gain is FractalNet, not H05.
   v2 of H05 with explicit `phi_shrink` is the next experiment.)
3. **The H58 hypothesis "max-pool over the C4 orbit discards 75 % of
   the signal — replace with mean" is FALSIFIED.** Mean-pool hurts by
   another 4–6 pp. The right fix is the *data*, not the operator:
   group-conv should be tested on rotated-CIFAR / IcoMNIST where the
   equivariance prior is data-aligned. See
   [`FINDINGS.md`](FINDINGS.md#h58-follow-up--the-avg-pool-fix-discarded).

**What this campaign explicitly does NOT claim:** ImageNet SOTA, that
sacred geometry is "true" in any metaphysical sense, that any decoder-
only LLM hypothesis (H01-LLM through H75) has been validated. The LLM
track and the cross-paradigm hybrids (G7, H61–H75) are documented
design but **not yet trained**.

## 3. What's in this repo

```
nature_inspired_networks/
├── README.md ............... operator quick-start (this file)
├── MINDMAP.md .............. one-page link map of every artifact
├── MANIFESTO.md ............ research argument (committee-grade)
├── CLAUDE.md ............... normative rules + 12 always-true assertions
├── ARCHITECTURE.md ......... module + shape tables
├── AUTORESEARCH_PROCESS.md . 7-step ritual + refusal-to-launch gates
├── IDEA_TABLE.md ........... 75-hypothesis status table
├── EXPERIMENT_LOG.md ....... master long-list of every run
├── EXPERIMENT_LEDGER.md .... chunk-by-chunk audit of source documents
├── PARADIGM_COMPARISON.md .. 8-chunk Liquid/JEPA/KAN/Transformer/GNN
├── NATURE_INSPIRED_NETWORKS.md  state-of-the-field reference (May 2026)
├── FINDINGS.md ............. campaign verdicts (H50 + H58 negatives)
├── RESULTS.md .............. auto-generated per-run narratives
├── SOTA_COMPARISON.md ...... honest map to the literature
├── PAPER.md / paper_abstract.md
├── SETUP.md / MEDIUM.md
├── sota_catalog.yaml
├── pyproject.toml
├── hypotheses/ ............. 75 committee-grade design docs in 7 themed dirs
├── ideas/ .................. 8 modular sub-projects + _TEMPLATE
├── src/nature_inspired_networks/  shared infrastructure
├── scripts/ ................ run_sweep / build_dashboard / build_report / compute_topology
├── skills/ ................. 10 content-agnostic reusable auto-research skills
├── tests/ .................. 29 core + 68 idea-local unit tests, all green
├── configs/ ................ shared YAML configs (smoke / quick / ablation)
├── experiments/ ............ legacy CIFAR-10 archive (13 runs + reasoning + dashboards)
├── dashboard/ .............. latest static HTML dashboard
├── docs/ ................... GitHub Pages root
└── memory/ ................. project checkpoint markdown
```

## 4. Quick start

### Prerequisites

- Windows 11 / Linux / macOS
- Python 3.10–3.13
- NVIDIA GPU with CUDA 12.x (tested: RTX 4090 Laptop, 16 GB)
- ~3 GB free disk for CIFAR-10 tarball + venv

### Install

```powershell
git clone https://github.com/dlmastery/nature_inspired_networks.git
cd nature_inspired_networks

python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip wheel setuptools
.\.venv\Scripts\python -m pip install `
   --index-url https://download.pytorch.org/whl/cu124 torch torchvision
.\.venv\Scripts\python -m pip install -e .
```

### Verify the GPU

```powershell
.\.venv\Scripts\python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### Smoke test (≈ 2 min on RTX 4090)

```powershell
$env:SSL_CERT_FILE = ".\.venv\Lib\site-packages\certifi\cacert.pem"
.\.venv\Scripts\python -m nature_inspired_networks.runner `
   --config configs\cifar10_smoke.yaml --tag smoke --seed 0
```

Detailed bring-up steps including the Python 3.13 SSL workaround for
corporate networks: see [`SETUP.md`](SETUP.md).

### Run the 13-row curated ablation (≈ 90 min)

```powershell
.\.venv\Scripts\python scripts\run_sweep.py `
   --config configs\cifar10_quick.yaml --seeds 0 --skip-existing
.\.venv\Scripts\python scripts\compute_topology.py --seeds 0
.\.venv\Scripts\python scripts\build_dashboard.py
.\.venv\Scripts\python scripts\build_report.py
start dashboard\dashboard.html
```

## 5. Architecture at a glance

**`NaturePriorBlock(c_in, c_out, stride, flags)`** is a residual block
whose seven inductive biases are independently togglable via
`NaturePriorFlags`:

```python
@dataclass
class NaturePriorFlags:
    hex: bool = True            # H21: 3×3 hex-masked conv (HexaConv 2018)
    group: bool = True          # H24 proxy: C4 group conv (Cohen 2016)
    fractal: bool = True        # H05: recursive depth=2 path (FractalNet 2017)
    toroidal: bool = True       # H22: circular padding (Pittorino 2022)
    cymatic_init: bool = True   # H35: Chladni-eigenmode kernel init
    golden_modulate: bool = True  # H17/H34: golden-angle channel gate
    group_reduce: str = "max"   # H58 ablation: 'max' vs 'mean'
```

Channel widths follow a Fibonacci, φ-geometric, or linear schedule
(H04). Optional FFN-side priors per the design space include
golden-spiral kernels (H31), Fibottention dilated attention (H32),
Vesica-Piscis multi-path (H33), and Metatron overlap sharing (H40).

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full module + shape
table and [`hypotheses/INDEX.md`](hypotheses/INDEX.md) for the 75
per-hypothesis design documents.

## 6. The 75-hypothesis design space

The complete design space ([`IDEA_TABLE.md`](IDEA_TABLE.md)) is split
into 7 thematic groups, each with its own subdirectory under
`hypotheses/`:

| Group | Count | Theme | Folder |
|---|---|---|---|
| G1 | 10 | Scaling & growth (φ / Fib depth-width-resolution-LR-budget) | [`hypotheses/g1_scaling_growth/`](hypotheses/g1_scaling_growth/) |
| G2 | 10 | Layer / channel / neuron architectures | [`hypotheses/g2_layer_channel_neuron/`](hypotheses/g2_layer_channel_neuron/) |
| G3 | 10 | Topologies & graphs (hex / toroidal / Platonic) | [`hypotheses/g3_topologies_graphs/`](hypotheses/g3_topologies_graphs/) |
| G4 | 10 | Kernels / attention / filters (spiral / Fibottention / Vesica / cymatic) | [`hypotheses/g4_kernels_attention_filters/`](hypotheses/g4_kernels_attention_filters/) |
| G5 | 10 | Optimization / init / regularization / NAS + full hybrid | [`hypotheses/g5_optimization_init_reg_nas/`](hypotheses/g5_optimization_init_reg_nas/) |
| G6 | 10 | Topological / bridging (Betti loss, drop-path, H58, trained-Betti, 3-seed) | [`hypotheses/g6_topological_bridging/`](hypotheses/g6_topological_bridging/) |
| G7 | 15 | Cross-paradigm hybrids (Sacred-Liquid-JEPA-KAN-GNN-Transformer) | [`hypotheses/g7_cross_paradigm_hybrids/`](hypotheses/g7_cross_paradigm_hybrids/) |

Each hypothesis ships a committee-grade design document
(400-800 lines): motivation, formal claim with a numeric falsifier,
multi-paper citations in Citation-Rigor format, mechanism (CNN-track
*and* LLM-track with PyTorch pseudocode), pre-registered predicted Δ
table, three-part experimental protocol, cross-references, ≥ 4
Committee Q&A, verification checklist, and a status journal.

The 4 doc agents that wrote H01-H75 are recorded in
[`MANIFESTO.md`](MANIFESTO.md) and [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md).

## 7. Modular `ideas/` taxonomy

Each implemented hypothesis is a self-contained sub-project at
`ideas/<NN>_<short>/` with:

- `README.md` (operator quick-start) + `IDEA.md` (formal claim)
- `implementation.py` (imports primitives from
  `nature_inspired_networks.priors`; never duplicates code)
- `tests.py` (≥ 4 unit tests + at least one regression test)
- `AUDIT.md` (self-found weaknesses) + `IMPROVEMENTS.md` + `VERIFY.md`
- `experiment.py` (thin runner wrapper)
- `configs/<*.yaml>` + `experiments/exp001_<short>/` archive + `dashboard/`

Current sub-projects (all tests green):

| Sub-project | Tests | Note |
|---|---|---|
| [`ideas/04_phi_fib_width/`](ideas/04_phi_fib_width/) | 7 ✓ | H04 channel scaling (mod-8 collapse documented) |
| [`ideas/05_fractal_phi_recursion/`](ideas/05_fractal_phi_recursion/) | 9 ✓ | H05 — phi-shrink kwarg recommended for v2 |
| [`ideas/17_golden_ratio_skip/`](ideas/17_golden_ratio_skip/) | 9 ✓ | H17 needs `forward_branch(x)` API |
| [`ideas/21_hexagonal_phi_packing/`](ideas/21_hexagonal_phi_packing/) | 10 ✓ | H21 (3×3 mask is only 180° symmetric — k=5 recommended) |
| [`ideas/22_toroidal_phi_closure/`](ideas/22_toroidal_phi_closure/) | 8 ✓ | H22 |
| [`ideas/35_cymatic_wavelet/`](ideas/35_cymatic_wavelet/) | 9 ✓ | H35 with corrected `cymatic_init_ortho_` |
| [`ideas/50_full_sacred_hybrid/`](ideas/50_full_sacred_hybrid/) | 8 ✓ | H50 (the negative-result archive) |
| [`ideas/58_group_avg_pool/`](ideas/58_group_avg_pool/) | 8 ✓ | H58 (DISCARDED) |

The remaining 67 hypotheses are documented but await `ideas/<NN>/`
sub-projects (use `ideas/_TEMPLATE/` as the scaffold).

## 8. The autoresearch protocol

Every experiment passes a refusal-to-launch gate stack inherited
verbatim from [`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage):

1. **Citation Rigor** — author / year / venue / single-quoted title /
   arXiv ID / relevance note. Parenthetical `(He2016)` is rejected.
2. **Reasoning Blob Completeness** — word-count floors per field
   (diagnosis ≥ 60, hypothesis ≥ 50, prediction ≥ 25, verdict ≥ 30,
   learning ≥ 40, citation ≥ 40 single / 80 multi).
3. **Goodhart fingerprint** — the composite formula is SHA-256 hashed;
   any edit breaks the next run.
4. **One config change per experiment** — no silent compounding.
5. **`experiment_log.jsonl` append-only** — corrections add a new row.
6. **Per-experiment archive with mandatory detailed README** — anyone
   reading just that sub-directory should be able to reproduce the
   experiment from cold.

There is **no `--bypass` flag**. See [`AUTORESEARCH_PROCESS.md`](AUTORESEARCH_PROCESS.md)
for the full 7-step ritual.

## 9. Reproducing the experiments

### Smoke (3 epochs, ~2 min)

```powershell
.\.venv\Scripts\python -m nature_inspired_networks.runner `
   --config configs\cifar10_smoke.yaml --tag smoke --seed 0
```

### Curated 13-row CIFAR-10 ablation (~90 min on RTX 4090 Laptop)

```powershell
.\.venv\Scripts\python scripts\run_sweep.py `
   --config configs\cifar10_quick.yaml --seeds 0 --skip-existing
```

### Trained-feature Betti curves

```powershell
.\.venv\Scripts\python scripts\compute_topology.py --seeds 0
```

### Refresh the dashboard

```powershell
.\.venv\Scripts\python scripts\build_dashboard.py
.\.venv\Scripts\python scripts\build_report.py
```

The static HTML lands at `dashboard/dashboard.html` and is mirrored to
`docs/dashboard/` for GitHub Pages.

### 3-seed re-sweep (~5 hr) for error bars

```powershell
.\.venv\Scripts\python scripts\run_sweep.py `
   --config configs\cifar10_quick.yaml --seeds 0 1 2 --skip-existing
```

### Full leave-one-out sweep (~3 hr extra)

```powershell
.\.venv\Scripts\python scripts\run_sweep.py `
   --config configs\cifar10_quick.yaml --seeds 0 --full --skip-existing
```

## 10. Extending this work

### Adding a new experiment

1. Pick or create `ideas/<NN_idea>/` from `ideas/_TEMPLATE/`.
2. Author a pre-run `reasoning.json` entry (Citation Rigor + word-count gates).
3. Write `configs/<config>.yaml`.
4. Run:
   ```powershell
   .\.venv\Scripts\python -m nature_inspired_networks.runner `
     --config ideas\<NN>\configs\<config>.yaml --tag expNNN_<short> --seed 0 `
     --root ideas\<NN>\experiments\expNNN_<short>\run
   ```
5. Append the post-run verdict + learning fields.
6. Regenerate the dashboard and add a row to `EXPERIMENT_LOG.md`.

### Adding a new hypothesis

1. Pick an unused `H<NN>` slot from `IDEA_TABLE.md` (next free is `H76`).
2. Copy `hypotheses/_TEMPLATE.md` to
   `hypotheses/g<N>_<group>/H<NN>_<short>.md`.
3. Fill every section per the template contract.
4. Add a row to `IDEA_TABLE.md` and `EXPERIMENT_LOG.md`.

### Adding a new dataset

See the [`autoresearch-dataset-loader`](skills/autoresearch-dataset-loader/SKILL.md) skill — it covers torchvision / MedMNIST / WILDS / OGB
patterns including the Python 3.13 SSL workaround.

## 11. Documentation map

**Start here:** [`MINDMAP.md`](MINDMAP.md) — single-page link map of
every artifact in this repo.

**Read first:**
- [`README.md`](README.md) — operator quick-start (this file)
- [`MANIFESTO.md`](MANIFESTO.md) — research argument (committee-grade)
- [`FINDINGS.md`](FINDINGS.md) — campaign verdicts (incl. H50 + H58 negatives)
- [`IDEA_TABLE.md`](IDEA_TABLE.md) — 75-hypothesis status table

**Normative:**
- [`CLAUDE.md`](CLAUDE.md) — 12 always-true assertions
- [`AUTORESEARCH_PROCESS.md`](AUTORESEARCH_PROCESS.md) — 7-step ritual + gates

**Reference:**
- [`hypotheses/INDEX.md`](hypotheses/INDEX.md) — 75 per-hypothesis design docs
- [`PARADIGM_COMPARISON.md`](PARADIGM_COMPARISON.md) — Liquid/JEPA/KAN/Transformer/GNN
- [`NATURE_INSPIRED_NETWORKS.md`](NATURE_INSPIRED_NETWORKS.md) — state-of-the-field as of May 2026
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — module + shape tables
- [`SOTA_COMPARISON.md`](SOTA_COMPARISON.md) — honest map to the literature
- [`sota_catalog.yaml`](sota_catalog.yaml) — prior-art YAML

**Per-run:**
- [`RESULTS.md`](RESULTS.md) — auto-generated narratives with verdicts
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) — master long-list, Tiers 0-6
- [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md) — chunk-by-chunk audit of sources

**Reusable infrastructure:**
- [`skills/README.md`](skills/README.md) — 10 content-agnostic skills

## 12. Sister projects

This repo inherits the autoresearch protocol from these sibling
public repos under the same author:

- [`dlmastery/autoresearch`](https://github.com/dlmastery/autoresearch) — FX prediction (the original)
- [`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage) — protocol source-of-truth, OOD pathology
- [`dlmastery/autoresearchtabular`](https://github.com/dlmastery/autoresearchtabular) — tabular ML (Higgs UCI)
- [`dlmastery/autoresearchspy`](https://github.com/dlmastery/autoresearchspy) — SPY ETF prediction
- [`dlmastery/autoresearchindexstock`](https://github.com/dlmastery/autoresearchindexstock) — QQQ index/stock

If you change a gate or composite formula here, **also open an issue
on `autoresearchimage`** explaining why — the gates are inherited and
divergence should be deliberate.

## 13. License & credits

**License:** [MIT](LICENSE).

**Author:** dlmastery (`eranti@gmail.com`).

**Acknowledgments:**

- The original PDF *Sacred Geometry and Neural Networks* (private)
  framed the program; this repo treats every "sacred" prior as an
  engineering hypothesis backed by a peer-reviewed paper.
- The autoresearch protocol is inherited verbatim from
  `dlmastery/autoresearchimage`.
- Doc-team and code-team agents (general-purpose subagents,
  parallel-dispatched in May 2026) wrote the 75 hypothesis design
  documents and 8 idea sub-projects — see commit log for attribution
  by batch.

**How to cite:**

```bibtex
@misc{dlmastery_nature_inspired_networks_2026,
  author = {dlmastery},
  title  = {nature_inspired_networks: NaturePriorBlock and the
            75-hypothesis autoresearch ablation},
  year   = 2026,
  url    = {https://github.com/dlmastery/nature_inspired_networks}
}
```

---

*Last refreshed: 2026-05-27. The dashboard is regenerated on every
commit; the static URL above is always current.*
