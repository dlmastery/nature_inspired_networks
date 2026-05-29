# MINDMAP — `dlmastery/nature_inspired_networks`

> Single-page link map of every artifact in this repo, **organised by
> reviewer utility**. Start at the section that matches your role.

## 🌐 Live URLs

- **📊 Live dashboard (regenerated every commit):** https://dlmastery.github.io/nature_inspired_networks/dashboard/dashboard.html
- **Pages landing page:** https://dlmastery.github.io/nature_inspired_networks/
- **Repo (public):** https://github.com/dlmastery/nature_inspired_networks

The aggregate dashboard links to a **per-experiment drill-down page
for every run** (`dashboard/experiments/<tag>_seed<N>.html`) — each
leaderboard row opens its own metrics, training curves, and reasoning
blob.

---

## 🧑‍⚖️ What a reviewer needs first

Read these six files in order; they cover the entire submission.

1. [`README.md`](README.md) — elevator pitch, headline claims, quick-start, repo map.
2. [`PAPER.md`](PAPER.md) — draft submission paper.
3. [`paper_abstract.md`](paper_abstract.md) — stand-alone 1-page abstract.
4. [`FINDINGS.md`](FINDINGS.md) — per-tag verdicts (KEEP / NEAR-MISS / DISCARD) including all negatives.
5. [`AUDIT_SUMMARY.md`](AUDIT_SUMMARY.md) — dual-track audit dashboard (implementation-critic ∩ research-critic).
6. [`REVIEWER_CHECKLIST.md`](REVIEWER_CHECKLIST.md) — per-claim evidence pointers (reproduction command + log file).

**Adversarial-review supplements:**

- [`NEURIPS_CHECKLIST.md`](NEURIPS_CHECKLIST.md) — 17-question NeurIPS Paper Checklist filled with file/line evidence.
- [`LIMITATIONS.md`](LIMITATIONS.md) — honest scope & multiplicity caveats.
- [`ETHICS_STATEMENT.md`](ETHICS_STATEMENT.md) — data licensing, compute, dual-use, IRB.
- [`SOTA_COMPARISON.md`](SOTA_COMPARISON.md) — honest map of our numbers to the published literature.
- [`PARADIGM_COMPARISON.md`](PARADIGM_COMPARISON.md) — Liquid / JEPA / KAN / Transformer / GNN synthesis.

---

## 🧑‍💻 What an implementer needs

For someone reproducing or extending the results.

**Environment & operator commands:**

- [`SETUP.md`](SETUP.md) — Windows / Linux bring-up, corporate-SSL workaround, smoke verification.
- [`CLAUDE.md`](CLAUDE.md) — 27 normative invariants (refusal table, hardware contract).
- [`AUTORESEARCH_PROCESS.md`](AUTORESEARCH_PROCESS.md) — 7-step refusal-to-launch ritual + gate stack.
- [`pyproject.toml`](pyproject.toml) — package metadata (Python ≥ 3.10, MIT licence).

**Architecture & shared primitives:**

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — module + tensor-shape tables.
- [`src/nature_inspired_networks/`](src/nature_inspired_networks/) — 80 source modules (shared import surface).
- [`tests/`](tests/) — 87 test files, 780+ unit tests (all green).

**Sweep / dashboard / report drivers:**

- [`scripts/run_sweep.py`](scripts/run_sweep.py) — ablation matrix driver (SOTA-smoke-gated).
- [`scripts/build_dashboard.py`](scripts/build_dashboard.py) — regenerates the aggregate HTML + per-experiment pages from `experiment_log.jsonl`, mirrors to `docs/dashboard/`.
- [`scripts/build_report.py`](scripts/build_report.py) — emits `RESULTS.md` with KEEP / NEAR-MISS / DISCARD verdicts.
- [`scripts/compute_topology.py`](scripts/compute_topology.py) — Betti curves from `best.pt` checkpoints.

**Shared configs:**

- [`configs/cifar10_sota_smoke.yaml`](configs/cifar10_sota_smoke.yaml) — Rule-13 pre-flight baseline.
- [`configs/cifar10_quick.yaml`](configs/cifar10_quick.yaml) — 12-epoch curated ablation matrix.
- [`configs/cifar10_smoke.yaml`](configs/cifar10_smoke.yaml) — 3-epoch fast smoke.

---

## 🧑‍🎓 What a future contributor needs

Anyone designing a new hypothesis or skill belongs here.

**Design-space single source of truth:**

- [`IDEA_TABLE.md`](IDEA_TABLE.md) — 84-hypothesis status table (G1–G8).
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) — master long-list (Tiers 0–6).
- [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md) — chunk-by-chunk audit of source documents (Rule 17).
- [`NATURE_INSPIRED_NETWORKS.md`](NATURE_INSPIRED_NETWORKS.md) — state-of-the-field reference (May 2026).
- [`MANIFESTO.md`](MANIFESTO.md) — research argument, committee-grade.

**Hypothesis design docs (84 files, 8 thematic groups):**

- [`hypotheses/INDEX.md`](hypotheses/INDEX.md) — file map for the 84 docs.
- [`hypotheses/_TEMPLATE.md`](hypotheses/_TEMPLATE.md) — the contract every hypothesis doc fills.

| Group | Path | Count | Theme |
|---|---|---|---|
| G1 | [`hypotheses/g1_scaling_growth/`](hypotheses/g1_scaling_growth/) | 10 | φ / Fibonacci growth laws (incl. H02 fib_depth, H09 phi_budget) |
| G2 | [`hypotheses/g2_layer_channel_neuron/`](hypotheses/g2_layer_channel_neuron/) | 10 | Fib-sized MLPs, channels, neurons |
| G3 | [`hypotheses/g3_topologies_graphs/`](hypotheses/g3_topologies_graphs/) | 10 | Hex / toroidal / Platonic graphs |
| G4 | [`hypotheses/g4_kernels_attention_filters/`](hypotheses/g4_kernels_attention_filters/) | 10 | Spirals, Fibottention, Vesica, cymatic wavelets |
| G5 | [`hypotheses/g5_optimization_init_reg_nas/`](hypotheses/g5_optimization_init_reg_nas/) | 10 | Optimizer, init, regularization, NAS, full hybrid (incl. H41 golden_adam) |
| G6 | [`hypotheses/g6_topological_bridging/`](hypotheses/g6_topological_bridging/) | 10 | Betti loss, drop-path, H58 fix, trained-Betti, 3-seed |
| G7 | [`hypotheses/g7_cross_paradigm_hybrids/`](hypotheses/g7_cross_paradigm_hybrids/) | 15 | Nature priors × Liquid / JEPA / KAN / Transformer / GNN |
| G8 | [`hypotheses/g8_esoteric_extensions/`](hypotheses/g8_esoteric_extensions/) | 9 | Esoteric extensions — neutral recasts (H76–H84) |

**Modular idea sub-projects:**

- [`ideas/INDEX.md`](ideas/INDEX.md) — taxonomy + status per sub-project.
- [`ideas/_TEMPLATE/`](ideas/_TEMPLATE/) — copy-and-fill scaffold.

| Status | Path |
|---|---|
| ✓ | [`ideas/04_phi_fib_width/`](ideas/04_phi_fib_width/) (7 tests pass) |
| ✓ | [`ideas/05_fractal_phi_recursion/`](ideas/05_fractal_phi_recursion/) (9 tests pass) |
| ✓ | [`ideas/17_golden_ratio_skip/`](ideas/17_golden_ratio_skip/) (9 tests pass) |
| ✓ | [`ideas/21_hexagonal_phi_packing/`](ideas/21_hexagonal_phi_packing/) (10 tests pass) |
| ✓ | [`ideas/22_toroidal_phi_closure/`](ideas/22_toroidal_phi_closure/) (8 tests pass) |
| ✓ | [`ideas/35_cymatic_wavelet/`](ideas/35_cymatic_wavelet/) (9 tests pass) |
| ✓ | [`ideas/50_full_sacred_hybrid/`](ideas/50_full_sacred_hybrid/) (8 tests pass; H50 negative-result archive) |
| ✓ | [`ideas/58_group_avg_pool/`](ideas/58_group_avg_pool/) (8 tests pass; H58 DISCARD archive) |

780+ unit tests across 87 test files, all green. The remaining
implemented hypotheses run through the shared sweep harness.

**Reusable skills (content-agnostic per Rule 10):**

- [`skills/autoresearch-experiment/`](skills/autoresearch-experiment/)
- [`skills/autoresearch-ablation-sweep/`](skills/autoresearch-ablation-sweep/)
- [`skills/autoresearch-dashboard/`](skills/autoresearch-dashboard/)
- [`skills/autoresearch-reasoning-entry/`](skills/autoresearch-reasoning-entry/)
- [`skills/autoresearch-modular-block/`](skills/autoresearch-modular-block/)
- [`skills/autoresearch-dataset-loader/`](skills/autoresearch-dataset-loader/)
- [`skills/autoresearch-topology-metrics/`](skills/autoresearch-topology-metrics/)
- [`skills/autoresearch-experiment-archive/`](skills/autoresearch-experiment-archive/)
- [`skills/autoresearch-idea-scaffold/`](skills/autoresearch-idea-scaffold/)
- [`skills/autoresearch-checkpoint/`](skills/autoresearch-checkpoint/)
- [`skills/autoresearch-multi-agent-dispatch/`](skills/autoresearch-multi-agent-dispatch/)
- [`skills/autoresearch-critic-team/`](skills/autoresearch-critic-team/)
- [`skills/autoresearch-scicritic-team/`](skills/autoresearch-scicritic-team/)
- [`skills/autoresearch-fixer-campaign/`](skills/autoresearch-fixer-campaign/)
- [`skills/autoresearch-combo-ladder/`](skills/autoresearch-combo-ladder/)
- [`skills/autoresearch-per-experiment-page/`](skills/autoresearch-per-experiment-page/)
- [`skills/autoresearch-auto-checkpoint-loop/`](skills/autoresearch-auto-checkpoint-loop/)
- [`skills/autoresearch-per-hypothesis-hillclimb/`](skills/autoresearch-per-hypothesis-hillclimb/)

---

## 🗺 Top-level layout (one-line tour)

```
nature_inspired_networks/
├── README.md                ← operator quick-start + headline claims
├── MINDMAP.md               ← THIS FILE
├── PAPER.md                 ← draft submission paper
├── paper_abstract.md        ← stand-alone 1-page abstract
├── MANIFESTO.md             ← research argument (committee-grade)
├── FINDINGS.md              ← per-tag campaign verdicts (incl. negatives)
├── AUDIT_SUMMARY.md         ← dual-track audit dashboard
├── REVIEWER_CHECKLIST.md    ← per-claim evidence pointers
├── NEURIPS_CHECKLIST.md     ← filled NeurIPS Paper Checklist
├── LIMITATIONS.md           ← scope & multiplicity caveats
├── ETHICS_STATEMENT.md      ← data licensing, compute, dual-use
├── CLAUDE.md                ← 27 normative invariants (refusal table)
├── ARCHITECTURE.md          ← module + shape tables
├── AUTORESEARCH_PROCESS.md  ← 7-step ritual + gates
├── IDEA_TABLE.md            ← 84-hypothesis status table
├── EXPERIMENT_LOG.md        ← master long-list (Tiers 0-6)
├── EXPERIMENT_LEDGER.md     ← chunk-by-chunk audit of source documents
├── PARADIGM_COMPARISON.md   ← Liquid/JEPA/KAN/Transformer/GNN comparison
├── RESULTS.md               ← auto-generated per-run narratives
├── SOTA_COMPARISON.md       ← honest map to the literature
├── NATURE_INSPIRED_NETWORKS.md  ← state-of-the-field reference (May 2026)
├── SETUP.md                 ← bring-up steps
├── MEDIUM.md                ← blog-style narrative
├── sota_catalog.yaml        ← prior-art YAML (single source of truth)
├── pyproject.toml
├── hypotheses/              ← 84 committee-grade docs (8 themed subdirs)
├── ideas/                   ← modular sub-projects + _TEMPLATE
├── src/nature_inspired_networks/  ← shared infrastructure (80 modules)
├── scripts/                 ← run_sweep, build_dashboard, build_report, compute_topology
├── skills/                  ← content-agnostic reusable auto-research skills
├── tests/                   ← 87 test files, 780+ unit tests (all green)
├── configs/                 ← shared YAML configs
├── experiments/             ← CIFAR-10 archive (35-tag smoke + reasoning + dashboards)
├── audits/                  ← per-group implementation-critic audit reports
├── dashboard/               ← aggregate dashboard + per-experiment drill-down pages
├── docs/                    ← GitHub Pages root (index.html + dashboard mirror)
└── memory/                  ← project checkpoint markdown
```

---

## 🔬 Source code

```
src/nature_inspired_networks/   (80 modules)
├── __init__.py
├── priors.py        ← PHI, Fib channels, hex mask, Chladni modes,
│                      GroupConv2d, HexConv2d, toroidal_pad, golden phases
├── blocks.py        ← NaturePriorBlock, NaturePriorFlags, _FractalPath
├── models.py        ← NaturePriorNet, ResNet20, build_model
├── data.py          ← CIFAR-10/100, MedMNIST loaders
├── train.py         ← Trainer (bf16 AMP + cosine LR + checkpoint save)
├── eval.py          ← top-k, FLOPs, latency, rot-eq err, composite (SHA-256 fingerprint)
├── topology.py      ← Betti curves, CKA, persistent homology
├── reasoning.py     ← Citation Rigor + Reasoning Blob Completeness gates
├── dashboard.py     ← rich HTML dashboard (aggregate + per-experiment pages)
├── runner.py        ← single-experiment entrypoint
└── … (g1–g8 prior implementations)
```

---

## 📊 Experiment artifacts & dashboard

```
dashboard/
├── dashboard.html              ← aggregate leaderboard (mirrored to docs/dashboard/)
└── experiments/
    └── <tag>_seed<N>.html      ← per-experiment drill-down page per run

docs/
├── index.html                  ← GitHub Pages landing page
└── dashboard/dashboard.html    ← public mirror of the aggregate dashboard

experiments/
├── experiment_log.jsonl        ← append-only run log (35-tag CIFAR-10 smoke)
├── reasoning_annotations.json  ← citation-gated reasoning per run
├── betti.json                  ← topology curves
└── cifar10/<tag>_seed0/        ← per-run metrics / history / best.pt
```

**Phase-2 CIFAR-10 screening leaderboard (top by composite):**

| tag | top-1 | composite | note |
|---|---|---|---|
| `baseline_resnet20` (15 ep) | 84.78 % | 0.8458 | literature anchor |
| `sg_only_phi_budget` (H09) | **85.54 %** | 0.8429 | only variant beating baseline top-1 |
| `sg_only_golden_momentum` (H17) | 83.52 % | 0.8307 | φ-scheduled momentum |
| `sg_only_fib_depth` (H02) | 82.18 % | 0.8261 | 0.66× params (efficiency) |
| `sg_only_golden_adam` (H41) | 51.96 % | 0.5142 | clean falsification (last) |

Full verdicts → [FINDINGS.md](FINDINGS.md); per-run narratives →
[RESULTS.md](RESULTS.md).

---

## 🧠 Memory entries (cross-session)

Stored under `~/.claude/projects/.../memory/`:

- `MEMORY.md` — index
- `feedback_naming_preference.md` — prefer neutral / academic naming over mystical
- `feedback_test_discipline.md` — every code change ships with a unit test
- `feedback_checkpoint_discipline.md` — commit + push to GitHub frequently (heartbeat)
- `feedback_sota_smoke_first.md` — SOTA baseline smoke before any variant
- `feedback_phased_workflow.md` — unit tests → smoke → CIFAR-100 top-K → 3-seed

---

## 🔗 Sister projects (cross-links, NOT dependencies)

The autoresearch protocol below is fully specified in this repo's
[`CLAUDE.md`](CLAUDE.md); nothing here requires another repo to
implement. Sister projects are listed only for cross-pollination:

- [`dlmastery/autoresearch`](https://github.com/dlmastery/autoresearch) — FX prediction
- [`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage) — earlier image autoresearch project
- [`dlmastery/autoresearchtabular`](https://github.com/dlmastery/autoresearchtabular) — Higgs UCI
- [`dlmastery/autoresearchspy`](https://github.com/dlmastery/autoresearchspy) — SPY ETF
- [`dlmastery/autoresearchindexstock`](https://github.com/dlmastery/autoresearchindexstock) — QQQ
