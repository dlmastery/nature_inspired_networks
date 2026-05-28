# nature_inspired_networks

> **An autoresearch-style ablation study of nature-inspired priors —
> hex / Platonic / fractal / toroidal / φ / cymatic / golden-angle — as
> drop-in residual & attention blocks for image classification and
> decoder-only LLMs, with refusal-to-launch protocol gates, a
> Goodhart-fingerprinted composite metric, and a publicly-pushed
> dashboard refreshed every commit.**

[![📊 Live Dashboard](https://img.shields.io/badge/📊_Live_Dashboard-online-2ea44f?style=for-the-badge)](https://dlmastery.github.io/nature_inspired_networks/dashboard/dashboard.html)

[![public](https://img.shields.io/badge/repo-public-brightgreen)](https://github.com/dlmastery/nature_inspired_networks)
[![pages](https://img.shields.io/badge/GitHub_Pages-landing-blue)](https://dlmastery.github.io/nature_inspired_networks/)
[![hypotheses](https://img.shields.io/badge/hypotheses-84_docs_(74_impl)-orange)](hypotheses/INDEX.md)
[![tests](https://img.shields.io/badge/unit_tests-780%2B_green-green)](tests/)
[![smoke](https://img.shields.io/badge/CIFAR--10_smoke-35_tags_✓-success)](FINDINGS.md)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](pyproject.toml)

**📊 Live demo:** **[Landing page](https://dlmastery.github.io/nature_inspired_networks/)** · **[Full dashboard](https://dlmastery.github.io/nature_inspired_networks/dashboard/dashboard.html)** (every run links to its own per-experiment drill-down page)

**Quick links:** [MINDMAP](MINDMAP.md) · [Manifesto](MANIFESTO.md) · [84 hypotheses](hypotheses/INDEX.md) · [Findings](FINDINGS.md) · [Results](RESULTS.md) · [SOTA comparison](SOTA_COMPARISON.md) · [Paradigm comparison](PARADIGM_COMPARISON.md) · [Reference](NATURE_INSPIRED_NETWORKS.md)

---

## Table of contents

1. [What this is](#1-what-this-is)
2. [Results at a glance](#2-results-at-a-glance)
3. [Repository map](#3-repository-map)
4. [The 8 hypothesis groups (g1–g8)](#4-the-8-hypothesis-groups-g1g8)
5. [Architecture at a glance](#5-architecture-at-a-glance)
6. [Modular `ideas/` taxonomy](#6-modular-ideas-taxonomy)
7. [Quick start](#7-quick-start)
8. [How the autoresearch protocol works](#8-how-the-autoresearch-protocol-works)
9. [Status](#9-status)
10. [License & citation](#10-license--citation)

---

## 1. What this is

`nature_inspired_networks` is a modular autoresearch framework that asks a
single falsifiable question: do the inductive biases nature uses —
hexagonal lattices, Platonic / icosahedral symmetry, fractal
self-similarity, toroidal closure, golden-ratio (φ) growth, Chladni
cymatic wave modes, golden-angle modulation — provide a measurable
engineering advantage when **imposed explicitly** on a network, rather
than waiting for them to emerge from scale? Each "prior" is mapped to a
peer-reviewed geometric/topological deep-learning paper; the mystical
motivation is acknowledged in prose only, while every artifact carries a
neutral, academic name (Rule 16).

The core artifact is **`NaturePriorBlock`** — a residual / attention
block whose inductive biases are each Boolean-toggleable, so a clean
ablation matrix isolates each prior's marginal effect on a
ResNet-20-shaped CIFAR scaffold (and, in design, a decoder-only GPT
stack). The complete design space is **84 hypotheses across 8 thematic
groups** ([`IDEA_TABLE.md`](IDEA_TABLE.md)), of which **74 are
implemented** with code + tests and **35 have been smoke-trained** on
CIFAR-10. Every run passes a refusal-to-launch protocol (Citation Rigor,
reasoning-completeness floors, a SHA-256-fingerprinted composite metric)
inherited from
[`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage)
— there is no `--bypass` flag.

## 2. Results at a glance

**Phase-2 CIFAR-10 smoke complete:** 35 tags, single seed, 12 epochs
each (baseline at 15 ep), RTX 4090 Laptop, bf16 AMP — **zero failures**.
The number ranked on is the Goodhart-fingerprinted **composite**
(accuracy + efficiency + robustness), not raw top-1.

**Top-5 leaderboard** (full table & every per-run drill-down on the
**[📊 live dashboard](https://dlmastery.github.io/nature_inspired_networks/dashboard/dashboard.html)**):

| rank | tag | top-1 | params | composite | note |
|---|---|---|---|---|---|
| 1 | `baseline_resnet20` (15 ep) | 84.78 % | 272 k | **0.8458** | literature anchor |
| 2 | `sg_only_phi_budget` (**H09**) | **85.54 %** | 284 k | 0.8429 | **only variant beating the baseline on top-1** |
| 3 | `sg_only_golden_momentum` | 83.52 % | 272 k | 0.8307 | φ-scheduled momentum |
| 4 | `sg_only_fib_depth` (**H02**) | 82.18 % | **180 k** | 0.8261 | efficiency story — comp 0.826 at **0.66× params** |
| 5 | `baseline_sg_vanilla` | 82.16 % | 186 k | 0.8258 | priors-off control |

**Headline positive — H09 `phi_budget`.** Allocating the parameter
budget across stages by a φ-geometric rule is the **only** nature prior
that lifts top-1 above the tuned ResNet-20 baseline (85.54 % vs 84.78 %)
within the 12-epoch smoke envelope. It is the single standout of the
campaign.

**Headline efficiency — H02 `fib_depth`.** A Fibonacci depth schedule
reaches composite 0.826 at **0.66× the baseline parameter count**,
making it the best accuracy-per-parameter row in the sweep.

**Clean falsification — H41 `golden_adam`.** Replacing Adam's β-schedule
with a golden-ratio variant collapses to **51.96 %** top-1 (composite
0.514, dead last). A textbook pre-registered negative: the prediction was
disproved decisively, and the negative is logged as a first-class result.

Full verdicts (KEEP / NEAR-MISS / DISCARD), including the earlier H50
full-hybrid and H58 avg-pool negatives, are in
[`FINDINGS.md`](FINDINGS.md); per-run narratives are auto-generated into
[`RESULTS.md`](RESULTS.md).

> **Honest scope.** This is a 12-epoch single-seed *smoke* scan, not a
> SOTA claim. No ImageNet result, no metaphysical claim about "sacred
> geometry," and no decoder-only-LLM hypothesis has been trained. The
> cheap broad scan exists precisely to decide which few hypotheses earn
> the expensive CIFAR-100 deep dive (see [§9](#9-status)).

## 3. Repository map

```
nature_inspired_networks/
├── README.md ............... operator quick-start (this file)
├── MINDMAP.md .............. one-page link map of every artifact
├── MANIFESTO.md ............ research argument (committee-grade)
├── CLAUDE.md ............... normative rules (18 enforced invariants)
├── ARCHITECTURE.md ......... module + shape tables
├── AUTORESEARCH_PROCESS.md . 7-step ritual + refusal-to-launch gates
├── IDEA_TABLE.md ........... 84-hypothesis status table (g1–g8)
├── EXPERIMENT_LOG.md ....... master long-list of every run
├── EXPERIMENT_LEDGER.md .... chunk-by-chunk audit of source documents
├── PARADIGM_COMPARISON.md .. Liquid / JEPA / KAN / Transformer / GNN
├── NATURE_INSPIRED_NETWORKS.md  state-of-the-field reference (May 2026)
├── FINDINGS.md ............. campaign verdicts (incl. negatives)
├── RESULTS.md .............. auto-generated per-run narratives
├── SOTA_COMPARISON.md ...... honest map to the literature
├── hypotheses/ ............. 84 committee-grade design docs in 8 themed dirs
├── ideas/ .................. 8 modular sub-projects + _TEMPLATE
├── src/nature_inspired_networks/  shared infrastructure (80 modules)
├── scripts/ ................ run_sweep / build_dashboard / build_report / compute_topology
├── skills/ ................. content-agnostic reusable auto-research skills
├── tests/ .................. 87 test files, 780+ unit tests (all green)
├── configs/ ................ shared YAML configs (smoke / quick / sota_smoke)
├── experiments/ ............ CIFAR-10 archive (35-tag smoke + reasoning)
├── dashboard/ .............. aggregate HTML dashboard + per-experiment pages
├── docs/ ................... GitHub Pages root (index.html + dashboard mirror)
└── memory/ ................. project checkpoint markdown
```

**Where to go next:**
- [`IDEA_TABLE.md`](IDEA_TABLE.md) — the 84-hypothesis status table (single source of truth for the design space)
- [`MINDMAP.md`](MINDMAP.md) — single-page link map of every artifact
- [`MANIFESTO.md`](MANIFESTO.md) — the research argument, committee-grade
- [`FINDINGS.md`](FINDINGS.md) — campaign verdicts including the H09 positive and the H41/H50/H58 negatives
- [`RESULTS.md`](RESULTS.md) — auto-generated per-run narratives
- [`SOTA_COMPARISON.md`](SOTA_COMPARISON.md) — honest map to the published literature
- [`hypotheses/`](hypotheses/INDEX.md) — 84 per-hypothesis design docs
- [`ideas/`](ideas/INDEX.md) — modular implementation sub-projects
- [`src/nature_inspired_networks/`](src/nature_inspired_networks/) — shared primitives (single import surface)
- [`scripts/`](scripts/) — sweep / dashboard / report / topology drivers
- [`dashboard/`](dashboard/dashboard.html) — local copy of the live dashboard

## 4. The 8 hypothesis groups (g1–g8)

The design space ([`IDEA_TABLE.md`](IDEA_TABLE.md), index at
[`hypotheses/INDEX.md`](hypotheses/INDEX.md)) is split into 8 thematic
groups, each its own subdirectory under `hypotheses/`:

| Group | # | Theme | Folder |
|---|---|---|---|
| G1 | 10 | Scaling & growth — φ / Fibonacci depth, width, resolution, LR, budget (incl. H02 fib_depth, H09 phi_budget) | [`g1_scaling_growth/`](hypotheses/g1_scaling_growth/) |
| G2 | 10 | Layer / channel / neuron architectures — Fib-sized MLPs, channels, neurons | [`g2_layer_channel_neuron/`](hypotheses/g2_layer_channel_neuron/) |
| G3 | 10 | Topologies & graphs — hex lattices, toroidal closure, Platonic adjacency | [`g3_topologies_graphs/`](hypotheses/g3_topologies_graphs/) |
| G4 | 10 | Kernels / attention / filters — golden spirals, Fibottention, Vesica, cymatic wavelets | [`g4_kernels_attention_filters/`](hypotheses/g4_kernels_attention_filters/) |
| G5 | 10 | Optimization / init / regularization / NAS — incl. H41 golden_adam, golden_momentum, full hybrid | [`g5_optimization_init_reg_nas/`](hypotheses/g5_optimization_init_reg_nas/) |
| G6 | 10 | Topological bridging — Betti loss, drop-path, H58 fix, trained-Betti, 3-seed | [`g6_topological_bridging/`](hypotheses/g6_topological_bridging/) |
| G7 | 15 | Cross-paradigm hybrids — nature priors × Liquid / JEPA / KAN / Transformer / GNN | [`g7_cross_paradigm_hybrids/`](hypotheses/g7_cross_paradigm_hybrids/) |
| G8 | 9 | Esoteric extensions — neutral recasts (tetrahedral dual-path, 12-fold radial attn, toroidal latents, morphing polytope, constant-width kernel, sinusoidal act, Voronoi sparse attn, collapse attn, spectral-Hopfield memory) | [`g8_esoteric_extensions/`](hypotheses/g8_esoteric_extensions/) |

Each hypothesis ships a committee-grade design document (motivation,
formal claim with a numeric falsifier, multi-paper Citation-Rigor
citations, CNN-track *and* LLM-track mechanism, pre-registered Δ table,
3-part protocol, cross-references, ≥ 4 Committee Q&A, verification
checklist, status journal).

## 5. Architecture at a glance

**`NaturePriorBlock(c_in, c_out, stride, flags)`** is a residual block
whose inductive biases are independently togglable via
`NaturePriorFlags`:

```python
@dataclass
class NaturePriorFlags:
    hex: bool = True              # H21: 3×3 hex-masked conv (HexaConv 2018)
    group: bool = True            # H24 proxy: C4 group conv (Cohen 2016)
    fractal: bool = True          # H05: recursive depth=2 path (FractalNet 2017)
    toroidal: bool = True         # H22: circular padding (Pittorino 2022)
    cymatic_init: bool = True     # H35: Chladni-eigenmode kernel init
    golden_modulate: bool = True  # H17/H34: golden-angle channel gate
    group_reduce: str = "max"     # H58 ablation: 'max' vs 'mean'
```

Channel widths follow a Fibonacci, φ-geometric, or linear schedule
(H02/H04/H09). Shared primitives live once in
[`src/nature_inspired_networks/`](src/nature_inspired_networks/) (80
modules) and every `ideas/<NN>/implementation.py` imports from there
rather than duplicating code (Rule 14). See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the full module + shape table.

## 6. Modular `ideas/` taxonomy

Each implemented hypothesis is a self-contained sub-project at
`ideas/<NN>_<short>/` carrying `README.md`, `IDEA.md`,
`implementation.py`, `tests.py`, `AUDIT.md`, `IMPROVEMENTS.md`,
`VERIFY.md`, `experiment.py`, configs, and an `experiments/expNNN_*/`
archive. Current sub-projects (all tests green):

| Sub-project | Tests | Note |
|---|---|---|
| [`ideas/04_phi_fib_width/`](ideas/04_phi_fib_width/) | 7 ✓ | H04 channel scaling (mod-8 collapse documented) |
| [`ideas/05_fractal_phi_recursion/`](ideas/05_fractal_phi_recursion/) | 9 ✓ | H05 — `phi_shrink` recommended for v2 |
| [`ideas/17_golden_ratio_skip/`](ideas/17_golden_ratio_skip/) | 9 ✓ | H17 golden-angle skip |
| [`ideas/21_hexagonal_phi_packing/`](ideas/21_hexagonal_phi_packing/) | 10 ✓ | H21 hex-masked conv |
| [`ideas/22_toroidal_phi_closure/`](ideas/22_toroidal_phi_closure/) | 8 ✓ | H22 circular padding |
| [`ideas/35_cymatic_wavelet/`](ideas/35_cymatic_wavelet/) | 9 ✓ | H35 corrected `cymatic_init_ortho_` |
| [`ideas/50_full_sacred_hybrid/`](ideas/50_full_sacred_hybrid/) | 8 ✓ | H50 full-hybrid negative-result archive |
| [`ideas/58_group_avg_pool/`](ideas/58_group_avg_pool/) | 8 ✓ | H58 DISCARD archive |

The full taxonomy & per-sub-project status is in
[`ideas/INDEX.md`](ideas/INDEX.md). The remaining implemented
hypotheses are exercised through the shared sweep harness; new
sub-projects scaffold from [`ideas/_TEMPLATE/`](ideas/_TEMPLATE/).

## 7. Quick start

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

Detailed bring-up, including the Python 3.13 corporate-SSL workaround
for the CIFAR download, is in [`SETUP.md`](SETUP.md).

### Operator commands (per [`CLAUDE.md`](CLAUDE.md) §8)

```powershell
# SOTA smoke (≤ 2 min — Rule 13 pre-flight; must hit ≥ 80% top-1 @ 12 ep)
.\.venv\Scripts\python -m nature_inspired_networks.runner `
  --config configs\cifar10_sota_smoke.yaml --tag smoke --seed 0

# Curated ablation sweep (~90 min)
.\.venv\Scripts\python scripts\run_sweep.py `
  --config configs\cifar10_quick.yaml --seeds 0 --skip-existing

# Trained-feature Betti + dashboard + report
.\.venv\Scripts\python scripts\compute_topology.py --seeds 0
.\.venv\Scripts\python scripts\build_dashboard.py
.\.venv\Scripts\python scripts\build_report.py
start dashboard\dashboard.html

# 3-seed re-sweep for error bars (~5 hr)
.\.venv\Scripts\python scripts\run_sweep.py `
  --config configs\cifar10_quick.yaml --seeds 0 1 2 --skip-existing
```

`scripts/build_dashboard.py` regenerates the aggregate
[`dashboard/dashboard.html`](dashboard/dashboard.html) and mirrors it to
[`docs/dashboard/`](docs/dashboard/) for GitHub Pages. The central
dashboard links to a **per-experiment drill-down page for every run**
under `dashboard/experiments/<tag>_seed<N>.html`, so any row in the
leaderboard opens its own metrics, curves, and reasoning blob.

## 8. How the autoresearch protocol works

Every experiment passes a refusal-to-launch gate stack inherited
verbatim from
[`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage)
(full 7-step ritual in
[`AUTORESEARCH_PROCESS.md`](AUTORESEARCH_PROCESS.md)):

1. **SOTA smoke first** — a known-good ResNet-20 baseline must clear the
   expected band before any nature-inspired variant runs (Rule 13).
2. **Citation Rigor** — author / year / venue / single-quoted title /
   arXiv ID / relevance; parenthetical `(He2016)` is rejected.
3. **Reasoning-completeness floors** — per-field word counts
   (diagnosis ≥ 60, hypothesis ≥ 50, prediction ≥ 25, verdict ≥ 30, …).
4. **Goodhart fingerprint** — the composite formula is SHA-256 hashed;
   any edit refuses the next launch.
5. **One config change per experiment** — no silent compounding.
6. **Append-only `experiment_log.jsonl`** — corrections add a `_v2` row.
7. **Per-experiment archive with a mandatory detailed README** — anyone
   reading just that sub-directory can reproduce the run from cold.

There is **no `--bypass` flag** (Rule 7): if a gate refuses, you fix the
entry, you do not disable the gate.

## 9. Status

- **Implementation campaign complete.** 74 of 75 base hypotheses
  (H01–H75; only H57 audio-cross-modal deferred) **plus** 9 G8
  esoteric-extension hypotheses (H76–H84) are implemented: 80 source
  modules, 87 test files, 780+ unit tests all green, 84 committee-grade
  design docs across groups g1–g8.
- **Phase-2 CIFAR-10 smoke complete.** 35 tags at seed 0 / 12 epochs,
  zero failures. Standout positive H09 `phi_budget` (85.54 %); clean
  falsification H41 `golden_adam` (51.96 %); efficiency story H02
  `fib_depth` (composite 0.826 at 0.66× params).
- **Next — Phase 4 (CIFAR-100 heavy hitters).** The graduating shortlist
  is **phi_budget, fib_depth, fractal, golden_momentum** plus the
  baseline, at ≥ 30 epochs, each with its own experiment archive.
  Phase 5 adds 3-seed error bars before any external claim.

The phased workflow (unit tests → CIFAR-10 smoke all hypotheses →
commit → CIFAR-100 top-K → 3-seed re-run) is normative; no hypothesis
jumps phases.

## 10. License & citation

**License:** MIT (declared in [`pyproject.toml`](pyproject.toml)). **Author:** dlmastery (`eranti@gmail.com`).

The original private PDF *Sacred Geometry and Neural Networks* framed the
program; this repo treats every "sacred" prior as an engineering
hypothesis backed by a peer-reviewed paper, with neutral naming
throughout (Rule 16). The autoresearch protocol is inherited from
[`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage).

```bibtex
@misc{dlmastery_nature_inspired_networks_2026,
  author = {dlmastery},
  title  = {nature_inspired_networks: NaturePriorBlock and the
            84-hypothesis autoresearch ablation},
  year   = 2026,
  url    = {https://github.com/dlmastery/nature_inspired_networks}
}
```

---

*Last refreshed: 2026-05-27. The dashboard is regenerated on every
commit; the live URLs above are always current.*
