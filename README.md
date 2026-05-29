# nature_inspired_networks

[![public](https://img.shields.io/badge/repo-public-brightgreen)](https://github.com/dlmastery/nature_inspired_networks)
[![pages](https://img.shields.io/badge/GitHub_Pages-live-blue)](https://dlmastery.github.io/nature_inspired_networks/)
[![hypotheses](https://img.shields.io/badge/hypotheses-84_docs_(74_impl)-orange)](hypotheses/INDEX.md)
[![tests](https://img.shields.io/badge/unit_tests-780%2B_green-green)](tests/)
[![smoke](https://img.shields.io/badge/CIFAR--10_smoke-35_tags_passed-success)](FINDINGS.md)
[![checkpoint-cadence](https://img.shields.io/badge/checkpoint-≤15min-success)](CLAUDE.md)
[![dual-track-audit](https://img.shields.io/badge/dual--track_audit-pass-brightgreen)](AUDIT_SUMMARY.md)
[![acceptance](https://img.shields.io/badge/status-submission_candidate-yellow)](PAPER.md)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](pyproject.toml)

> **Elevator pitch (for the area chair).** This repository is a fully
> protocol-gated, audit-traced empirical study of *nature-inspired
> inductive biases* — hexagonal lattices, Platonic / icosahedral
> equivariance, fractal self-similarity, toroidal closure, φ /
> Fibonacci scaling, Chladni cymatic initialisation, golden-angle
> modulation — implemented as drop-in residual / attention blocks for
> CIFAR-scale image classification. We pre-register **84 hypotheses
> across 8 thematic groups** with committee-grade design docs, ship
> 780+ unit tests and a SHA-256-fingerprinted composite metric, run a
> 35-tag CIFAR-10 screening sweep, and submit every kept claim to a
> two-stage adversarial audit (implementation-critic + research-critic).
> Negative results are reported at equal prominence to positives. The
> live dashboard regenerates on every commit:
> **https://dlmastery.github.io/nature_inspired_networks/**.

**Quick links:** [PAPER](PAPER.md) · [FINDINGS](FINDINGS.md) · [AUDIT_SUMMARY](AUDIT_SUMMARY.md) · [REVIEWER_CHECKLIST](REVIEWER_CHECKLIST.md) · [NEURIPS_CHECKLIST](NEURIPS_CHECKLIST.md) · [LIMITATIONS](LIMITATIONS.md) · [ETHICS_STATEMENT](ETHICS_STATEMENT.md) · [MINDMAP](MINDMAP.md) · [MANIFESTO](MANIFESTO.md) · [84 hypotheses](hypotheses/INDEX.md) · [SOTA comparison](SOTA_COMPARISON.md)

---

## Table of contents

1. [What this is](#1-what-this-is)
2. [Quick start (clone → SOTA smoke in 4 commands)](#2-quick-start-clone--sota-smoke-in-4-commands)
3. [What's in this repo](#3-whats-in-this-repo)
4. [Headline claims (post-fix)](#4-headline-claims-post-fix)
5. [Negative results (first-class citizens)](#5-negative-results-first-class-citizens)
6. [Methodological notes — screening vs evaluation](#6-methodological-notes--screening-vs-evaluation)
7. [The 8 hypothesis groups (g1–g8)](#7-the-8-hypothesis-groups-g1g8)
8. [Architecture at a glance](#8-architecture-at-a-glance)
9. [Modular `ideas/` taxonomy](#9-modular-ideas-taxonomy)
10. [How the autoresearch protocol works](#10-how-the-autoresearch-protocol-works)
11. [Reproducibility](#11-reproducibility)
12. [Status](#12-status)
13. [License & citation](#13-license--citation)

---

## 1. What this is

`nature_inspired_networks` is a modular autoresearch framework that
asks a single falsifiable question: **do the inductive biases nature
uses — hexagonal lattices, Platonic / icosahedral symmetry, fractal
self-similarity, toroidal closure, golden-ratio (φ) growth, Chladni
cymatic wave modes, golden-angle modulation — provide a measurable
engineering advantage when imposed explicitly on a network, rather
than waiting for them to emerge from scale?** Each prior is mapped to
a peer-reviewed geometric / topological deep-learning paper; the
historical / aesthetic motivation is disclosed honestly in
[`MANIFESTO.md`](MANIFESTO.md) but every artifact carries a neutral,
academic name (CLAUDE.md Rule 16).

The core artifact is **`NaturePriorBlock`** — a residual / attention
block whose inductive biases are each Boolean-toggleable, so a clean
ablation matrix isolates each prior's marginal effect on a
ResNet-20-shaped CIFAR scaffold. The design space is **84
hypotheses across 8 thematic groups** ([`IDEA_TABLE.md`](IDEA_TABLE.md)).
**74 are implemented** with code + tests; **35 have been
smoke-trained** on CIFAR-10; **3 graduate to deeper evaluation**
([§4](#4-headline-claims-post-fix)). Every run passes a
refusal-to-launch protocol (Citation Rigor, reasoning-completeness
floors, a SHA-256-fingerprinted composite metric, append-only logs)
inherited from
[`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage)
— there is no `--bypass` flag.

## 2. Quick start (clone → SOTA smoke in 4 commands)

The four commands below take a clean Windows 11 + RTX 4090 Laptop box
from nothing to a reproduced SOTA-baseline smoke. Detailed bring-up
(including the Python 3.13 corporate-SSL workaround) is in
[`SETUP.md`](SETUP.md).

```powershell
# 1) Clone
git clone https://github.com/dlmastery/nature_inspired_networks.git
cd nature_inspired_networks

# 2) Environment (Python 3.13 + PyTorch 2.x / CUDA 12.x + editable install)
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip wheel setuptools `
  ; .\.venv\Scripts\python -m pip install `
    --index-url https://download.pytorch.org/whl/cu124 torch torchvision `
  ; .\.venv\Scripts\python -m pip install -e .

# 3) Download CIFAR-10 (corp-proxy-safe — torchvision still verifies MD5)
curl.exe -kL -o data\cifar-10-python.tar.gz `
   https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# 4) Reproduce the SOTA baseline smoke (Rule 13 — must hit ≥ 80 % top-1 @ 12 ep)
$env:OMP_NUM_THREADS=2; $env:MKL_NUM_THREADS=2; $env:KMP_DUPLICATE_LIB_OK="TRUE"
.\.venv\Scripts\python -m nature_inspired_networks.runner `
  --config configs\cifar10_sota_smoke.yaml --tag smoke --seed 0
```

Expected wall-clock: < 5 min on a 4090 Laptop. If the baseline misses
the 80 % band, STOP and diagnose your environment per
[`SETUP.md`](SETUP.md) — do not run any nature-inspired variant
(CLAUDE.md Rule 13).

## 3. What's in this repo

| Document | Purpose |
|---|---|
| [`PAPER.md`](PAPER.md) | Draft submission paper (abstract → conclusion). |
| [`paper_abstract.md`](paper_abstract.md) | Stand-alone 1-page abstract for circulation. |
| [`MANIFESTO.md`](MANIFESTO.md) | Committee-grade research argument & honest disclosure of motivation. |
| [`FINDINGS.md`](FINDINGS.md) | Per-tag campaign verdicts: KEEP / NEAR-MISS / DISCARD with full reasoning. |
| [`AUDIT_SUMMARY.md`](AUDIT_SUMMARY.md) | Dual-track audit dashboard (implementation-critic ∩ sci-critic verdicts). |
| [`REVIEWER_CHECKLIST.md`](REVIEWER_CHECKLIST.md) | Per-claim evidence pointers for an adversarial reviewer. |
| [`NEURIPS_CHECKLIST.md`](NEURIPS_CHECKLIST.md) | Filled-in NeurIPS Paper Checklist (17 questions, each with file/line evidence). |
| [`LIMITATIONS.md`](LIMITATIONS.md) | Honest enumeration of scope and statistical caveats. |
| [`ETHICS_STATEMENT.md`](ETHICS_STATEMENT.md) | Data licensing, compute, dual-use, IRB statement. |
| [`IDEA_TABLE.md`](IDEA_TABLE.md) | Single source of truth for the 84-hypothesis design space (G1–G8). |
| [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | Master long-list (Tiers 0–6) of every planned and executed run. |
| [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md) | Chunk-by-chunk audit of source documents (CLAUDE.md Rule 17). |
| [`RESULTS.md`](RESULTS.md) | Auto-generated per-run narratives from `experiment_log.jsonl`. |
| [`SOTA_COMPARISON.md`](SOTA_COMPARISON.md) | Honest map from our numbers to the published literature. |
| [`PARADIGM_COMPARISON.md`](PARADIGM_COMPARISON.md) | Liquid / JEPA / KAN / Transformer / GNN synthesis. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Module + tensor-shape tables. |
| [`AUTORESEARCH_PROCESS.md`](AUTORESEARCH_PROCESS.md) | The 7-step refusal-to-launch ritual + gate stack. |
| [`NATURE_INSPIRED_NETWORKS.md`](NATURE_INSPIRED_NETWORKS.md) | State-of-the-field reference (May 2026). |
| [`MINDMAP.md`](MINDMAP.md) | One-page link map of every artifact in the repo. |
| [`CLAUDE.md`](CLAUDE.md) | 27 normative invariants (refusal table, hardware contract). |
| [`SETUP.md`](SETUP.md) | Detailed environment bring-up (Windows / Linux). |

## 4. Headline claims (post-fix)

After the Fixer campaign (CLAUDE.md Rule 21) and dual-track audit
(CLAUDE.md Rule 22), **three hypotheses graduate** to deeper evaluation
and Phase-5 multi-seed gating. All composites use the SHA-256-
fingerprinted formula
`top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms)`
(fingerprint `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`).

| Rank | Hypothesis (tag) | Top-1 | Params | Composite | Seeds | Phase-5 3-seed gate | Audit verdict |
|---|---|---|---|---|---|---|---|
| 1 | **H09 `sg_only_phi_budget`** | 85.54 % | 284 k | **0.8429** | 3 (post-fix) | PASS | Impl: PASS · Sci: DERIVATIVE+TESTABLE |
| 2 | **H02 `sg_only_fib_depth`** | 82.18 % | 180 k | **0.8261** (at 0.66× params) | 3 (post-fix) | PASS | Impl: PASS · Sci: DERIVATIVE+TESTABLE |
| 3 | **H17 `sg_only_golden_momentum`** | 83.52 % | 272 k | **0.8307** | 3 (post-fix) | PASS | Impl: PASS · Sci: DERIVATIVE+TESTABLE |

(The H41 `golden_adam` β-only requalification — clarified from
−33 pp single-config to −1 pp under Reddi-2018 testing — remains a
disproof, not a winner; see [§5](#5-negative-results-first-class-citizens).)

## 5. Negative results (first-class citizens)

Equal prominence. Three falsifications anchor the campaign's
calibration; every one is reported with the same reasoning blob,
audit, and dashboard page as the winners.

| Hypothesis | Tag | Effect | Status |
|---|---|---|---|
| **H50 full sacred hybrid** | `sg_full_fib` | **−11.54 pp top-1** vs baseline | DISCARD — cautionary tale, motivates Rule 23 (orthogonal-axis stacking only). |
| **H41 golden Adam** | `sg_only_golden_adam` | **−33 pp** in single-config, **REQUALIFIED to −1 pp** when isolated to β-schedule only under Reddi-2018 conditions | DISCARD — original prediction stands disproved; the β-only requalification is logged as a separate, narrower row. |
| **H80 Reuleaux constant-width kernel** | `sg_only_reuleaux` | **−8.83 pp top-1** vs baseline | DISCARD — falsified hypothesis with a publishable mechanism explanation in [`FINDINGS.md`](FINDINGS.md). |

## 6. Methodological notes — screening vs evaluation

Reviewers should treat the bulk of the campaign as **screening data**,
not final evaluation. The cheap broad scan exists to decide which few
hypotheses earn the expensive deep dive (CLAUDE.md Rule 19, Phase 2
→ 4 → 5). Specifically:

- **~80 % of tags** are run at **single seed × single config × 12
  epochs CIFAR-10**. This is a calibration / screening regime and a
  single-seed number is *not* a publication claim.
- **The 3 graduates in [§4](#4-headline-claims-post-fix)** are
  re-evaluated at **3 seeds** (Phase 5) with explicit error bars
  before any external claim. A Phase-9 hill-climb is planned to
  retire the residual single-config dependency.
- **No multiplicity correction** has been applied across the
  84-hypothesis design space — this is acknowledged in
  [`LIMITATIONS.md`](LIMITATIONS.md) and is the headline reason the
  paper frames itself as a *methodology + screening* contribution.
- The composite metric is a project-specific aggregation; raw top-1
  rankings agree with the composite ranking for the three Phase-5
  winners but may diverge elsewhere — see
  [`SOTA_COMPARISON.md`](SOTA_COMPARISON.md).

## 7. The 8 hypothesis groups (g1–g8)

The design space ([`IDEA_TABLE.md`](IDEA_TABLE.md), index at
[`hypotheses/INDEX.md`](hypotheses/INDEX.md)) splits into 8 thematic
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

## 8. Architecture at a glance

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
rather than duplicating code (CLAUDE.md Rule 14). See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the full module + shape table.

## 9. Modular `ideas/` taxonomy

Each implemented hypothesis is a self-contained sub-project at
`ideas/<NN>_<short>/` carrying `README.md`, `IDEA.md`,
`implementation.py`, `tests.py`, `AUDIT.md`, `IMPROVEMENTS.md`,
`VERIFY.md`, `experiment.py`, configs, and an `experiments/expNNN_*/`
archive (CLAUDE.md Rule 8 + 9). Current sub-projects (all tests green):

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
[`ideas/INDEX.md`](ideas/INDEX.md).

## 10. How the autoresearch protocol works

Every experiment passes a refusal-to-launch gate stack
(full 7-step ritual in
[`AUTORESEARCH_PROCESS.md`](AUTORESEARCH_PROCESS.md)):

1. **SOTA smoke first** — a known-good ResNet-20 baseline must clear
   the expected band before any nature-inspired variant runs
   (Rule 13).
2. **Citation Rigor** — author / year / venue / single-quoted title /
   arXiv ID / relevance; parenthetical `(He2016)` is rejected.
3. **Reasoning-completeness floors** — per-field word counts
   (diagnosis ≥ 60, hypothesis ≥ 50, prediction ≥ 25, verdict ≥ 30, …).
4. **Goodhart fingerprint** — the composite formula is SHA-256 hashed;
   any edit refuses the next launch.
5. **One config change per experiment** — no silent compounding.
6. **Append-only `experiment_log.jsonl`** — corrections add a `_v2`
   row.
7. **Per-experiment archive with a mandatory detailed README** —
   anyone reading just that sub-directory can reproduce the run from
   cold.

There is **no `--bypass` flag** (Rule 7): if a gate refuses, you fix
the entry, you do not disable the gate.

## 11. Reproducibility

Reproducibility is treated as a first-class deliverable.

- **NeurIPS-style paper checklist:** filled in line-by-line at
  [`NEURIPS_CHECKLIST.md`](NEURIPS_CHECKLIST.md), with file/line
  evidence per item.
- **Reviewer evidence pointers:** every paper claim is mapped to its
  reproduction command + log file in
  [`REVIEWER_CHECKLIST.md`](REVIEWER_CHECKLIST.md).
- **Limitations & multiplicity:** explicit in
  [`LIMITATIONS.md`](LIMITATIONS.md).
- **Ethics, data licensing, compute budget:** in
  [`ETHICS_STATEMENT.md`](ETHICS_STATEMENT.md).
- **Hardware contract:** 1× RTX 4090 Laptop, 16 GB VRAM, Windows 11.
  bf16 AMP, batch 256, `num_workers=0`, `OMP_NUM_THREADS=2`,
  `MKL_NUM_THREADS=2`, `KMP_DUPLICATE_LIB_OK=TRUE`
  (CLAUDE.md §2 + Rule 26).
- **Append-only run log:** every metric lives in
  `experiments/experiment_log.jsonl` with a journal entry per
  correction (Rule 3).
- **Per-experiment archives:** every run carries its own
  `ideas/<NN>/experiments/expNNN_*/` archive with a detailed README,
  config, reasoning blob, metrics, history, and `best.pt` checkpoint
  (Rule 8 + 9).
- **Live dashboard:** regenerated on every commit at
  https://dlmastery.github.io/nature_inspired_networks/dashboard/dashboard.html.

## 12. Status

- **Implementation campaign complete.** 74 of 75 base hypotheses
  (H01–H75; only H57 audio-cross-modal deferred) **plus** 9 G8
  esoteric-extension hypotheses (H76–H84) are implemented: 80 source
  modules, 87 test files, 780+ unit tests all green, 84
  committee-grade design docs across groups g1–g8.
- **Phase-2 CIFAR-10 screening complete.** 35 tags at seed 0 /
  12 epochs, zero protocol failures.
- **Phase 4 → Phase 5 graduates.** `phi_budget`, `fib_depth`,
  `golden_momentum` cleared the dual-track audit (implementation +
  research-scientist) and the 3-seed Phase-5 gate.
- **Phase 9 (hill-climb)** is the next milestone — moving the three
  winners from screening-grade to evaluation-grade by varying
  config-orthogonal axes around each winner.

The phased workflow (unit tests → CIFAR-10 smoke all hypotheses →
commit → CIFAR-100 top-K → 3-seed re-run) is normative; no hypothesis
jumps phases.

## 13. License & citation

**License:** MIT (declared in [`pyproject.toml`](pyproject.toml)).
**Author (git history):** dlmastery (`eranti@gmail.com`). For
submission the author identification is **anonymised for blind
review** (see [`ETHICS_STATEMENT.md`](ETHICS_STATEMENT.md)).

```bibtex
@misc{dlmastery_nature_inspired_networks_2026,
  author = {dlmastery},
  title  = {nature\_inspired\_networks: NaturePriorBlock and the
            84-hypothesis autoresearch ablation},
  year   = {2026},
  howpublished = {\url{https://github.com/dlmastery/nature_inspired_networks}},
  note   = {Live dashboard: \url{https://dlmastery.github.io/nature_inspired_networks/}}
}
```

---

*Last refreshed: 2026-05-29. The dashboard is regenerated on every
commit; the live URLs above are always current.*
