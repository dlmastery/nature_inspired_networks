# MINDMAP — `dlmastery/nature_inspired_networks`

> Single-page link map of every artifact in this repo. Start here if
> you do not know where to look.
> Want the operator quick-start? → [README.md](README.md).
> Want the research argument? → [MANIFESTO.md](MANIFESTO.md).

## 🌐 Live URLs

- **📊 Dashboard (live, every-commit):** https://dlmastery.github.io/nature_inspired_networks/dashboard/dashboard.html
- **Pages landing page:** https://dlmastery.github.io/nature_inspired_networks/
- **Repo (public):** https://github.com/dlmastery/nature_inspired_networks

The aggregate dashboard links to a **per-experiment drill-down page for
every run** (`dashboard/experiments/<tag>_seed<N>.html`) — each leaderboard
row opens its own metrics, training curves, and reasoning blob.

## 🗺 Top-level layout (one-line tour)

```
nature_inspired_networks/
├── README.md                ← operator quick-start
├── MINDMAP.md               ← THIS FILE
├── MANIFESTO.md             ← research argument (committee-grade)
├── CLAUDE.md                ← normative rules (18 enforced invariants)
├── ARCHITECTURE.md          ← module + shape tables
├── AUTORESEARCH_PROCESS.md  ← 7-step ritual + gates
├── IDEA_TABLE.md            ← 84-hypothesis status table (G1–G8)
├── EXPERIMENT_LOG.md        ← master long-list (Tiers 0-6)
├── EXPERIMENT_LEDGER.md     ← chunk-by-chunk audit of source documents
├── PARADIGM_COMPARISON.md   ← Liquid/JEPA/KAN/Transformer/GNN comparison
├── FINDINGS.md              ← campaign verdicts (H09 positive + H41/H50/H58 negatives)
├── RESULTS.md               ← auto-generated per-run narratives
├── SOTA_COMPARISON.md       ← honest map to the literature
├── PAPER.md                 ← draft paper
├── paper_abstract.md        ← 1-page summary
├── SETUP.md                 ← bring-up steps
├── MEDIUM.md                ← blog-style narrative
├── sota_catalog.yaml        ← prior-art YAML (single source of truth)
├── pyproject.toml
├── hypotheses/              ← 84 committee-grade docs (8 themed subdirs)
├── ideas/                   ← 8 modular sub-projects (full taxonomy)
├── src/nature_inspired_networks/  ← shared infrastructure (80 modules)
├── scripts/                 ← run_sweep, build_dashboard, build_report, compute_topology
├── skills/                  ← content-agnostic reusable auto-research skills
├── tests/                   ← 87 test files, 780+ unit tests (all green)
├── configs/                 ← shared YAML configs
├── experiments/             ← CIFAR-10 archive (35-tag smoke + reasoning + dashboards)
├── dashboard/               ← aggregate dashboard + per-experiment drill-down pages
├── docs/                    ← GitHub Pages root (index.html + dashboard mirror)
└── memory/                  ← project checkpoint markdown
```

## 📚 Documentation map

### Read first

1. **[README.md](README.md)** — orientation + results at a glance
2. **[MANIFESTO.md](MANIFESTO.md)** — what we claim + what we don't
3. **[FINDINGS.md](FINDINGS.md)** — campaign verdicts (H09 positive, H41/H50/H58 negatives)
4. **[IDEA_TABLE.md](IDEA_TABLE.md)** — the 84-hypothesis status table

### Normative rules

- **[CLAUDE.md](CLAUDE.md)** — 18 enforced invariants, hardware contract, refusal table
- **[AUTORESEARCH_PROCESS.md](AUTORESEARCH_PROCESS.md)** — the 7-step ritual + gate stack

### Per-hypothesis design docs (84 files, 8 thematic groups)

- **[hypotheses/INDEX.md](hypotheses/INDEX.md)** — file map for the 84 docs
- **[hypotheses/_TEMPLATE.md](hypotheses/_TEMPLATE.md)** — the contract every hypothesis doc fills

| Group | Path | Count | Theme |
|---|---|---|---|
| G1 | [`hypotheses/g1_scaling_growth/`](hypotheses/g1_scaling_growth/) | 10 | φ / Fibonacci growth laws (incl. H02 fib_depth, H09 phi_budget) |
| G2 | [`hypotheses/g2_layer_channel_neuron/`](hypotheses/g2_layer_channel_neuron/) | 10 | Fib-sized MLPs, channels, neurons |
| G3 | [`hypotheses/g3_topologies_graphs/`](hypotheses/g3_topologies_graphs/) | 10 | Hex / toroidal / Platonic graphs |
| G4 | [`hypotheses/g4_kernels_attention_filters/`](hypotheses/g4_kernels_attention_filters/) | 10 | Spirals, Fibottention, Vesica, cymatic wavelets |
| G5 | [`hypotheses/g5_optimization_init_reg_nas/`](hypotheses/g5_optimization_init_reg_nas/) | 10 | Optimizer, init, regularization, NAS, full hybrid (incl. H41 golden_adam) |
| G6 | [`hypotheses/g6_topological_bridging/`](hypotheses/g6_topological_bridging/) | 10 | Betti loss, drop-path, H58 fix, trained-Betti, 3-seed |
| G7 | [`hypotheses/g7_cross_paradigm_hybrids/`](hypotheses/g7_cross_paradigm_hybrids/) | 15 | Nature priors × Liquid / JEPA / KAN / Transformer / GNN |
| G8 | [`hypotheses/g8_esoteric_extensions/`](hypotheses/g8_esoteric_extensions/) | 9 | Esoteric extensions — neutral recasts (H76–H84: tetrahedral dual-path, 12-fold radial attn, toroidal latents, morphing polytope, constant-width kernel, sinusoidal act, Voronoi sparse attn, collapse attn, spectral-Hopfield memory) |

### Cross-paradigm comparison

- **[PARADIGM_COMPARISON.md](PARADIGM_COMPARISON.md)** — synthesis of Liquid (LFM2) / JEPA family / KAN / decoder-only Transformer / equivariant GNN vs. the NaturePrior synthesis (philosophy / inductive biases / mechanisms / efficiency / training / interpretability / limitations / synthesis)

## 🧪 Modular idea sub-projects

Per-hypothesis self-contained sub-projects with impl + tests + audit +
experiment archives:

- **[ideas/INDEX.md](ideas/INDEX.md)** — taxonomy + status per sub-project
- **[ideas/_TEMPLATE/](ideas/_TEMPLATE/)** — copy-and-fill scaffold

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

## 🛠 Scripts

- [`scripts/run_sweep.py`](scripts/run_sweep.py) — ablation matrix driver (SOTA-smoke-gated)
- [`scripts/build_dashboard.py`](scripts/build_dashboard.py) — regenerates the aggregate HTML dashboard + per-experiment pages from `experiment_log.jsonl`, mirrors to `docs/dashboard/`
- [`scripts/build_report.py`](scripts/build_report.py) — emits `RESULTS.md` with KEEP / NEAR-MISS / DISCARD verdicts
- [`scripts/compute_topology.py`](scripts/compute_topology.py) — Betti curves from `best.pt` checkpoints

## 🎯 Reusable skills (`skills/`)

Content-agnostic auto-research skills any future project can pick up
unchanged (Rule 10):

- [`skills/README.md`](skills/README.md) — how to use the set
- [`skills/autoresearch-experiment/SKILL.md`](skills/autoresearch-experiment/SKILL.md)
- [`skills/autoresearch-ablation-sweep/SKILL.md`](skills/autoresearch-ablation-sweep/SKILL.md)
- [`skills/autoresearch-dashboard/SKILL.md`](skills/autoresearch-dashboard/SKILL.md)
- [`skills/autoresearch-reasoning-entry/SKILL.md`](skills/autoresearch-reasoning-entry/SKILL.md)
- [`skills/autoresearch-modular-block/SKILL.md`](skills/autoresearch-modular-block/SKILL.md)
- [`skills/autoresearch-dataset-loader/SKILL.md`](skills/autoresearch-dataset-loader/SKILL.md)
- [`skills/autoresearch-topology-metrics/SKILL.md`](skills/autoresearch-topology-metrics/SKILL.md)
- [`skills/autoresearch-experiment-archive/SKILL.md`](skills/autoresearch-experiment-archive/SKILL.md)
- [`skills/autoresearch-idea-scaffold/SKILL.md`](skills/autoresearch-idea-scaffold/SKILL.md)
- [`skills/autoresearch-checkpoint/SKILL.md`](skills/autoresearch-checkpoint/SKILL.md)

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

**Phase-2 CIFAR-10 smoke leaderboard (top by composite):**

| tag | top-1 | composite | note |
|---|---|---|---|
| `baseline_resnet20` (15 ep) | 84.78 % | 0.8458 | literature anchor |
| `sg_only_phi_budget` (H09) | **85.54 %** | 0.8429 | only variant beating baseline top-1 |
| `sg_only_golden_momentum` | 83.52 % | 0.8307 | φ-scheduled momentum |
| `sg_only_fib_depth` (H02) | 82.18 % | 0.8261 | 0.66× params (efficiency) |
| `sg_only_golden_adam` (H41) | 51.96 % | 0.5142 | clean falsification (last) |

Full verdicts → [FINDINGS.md](FINDINGS.md); per-run narratives → [RESULTS.md](RESULTS.md).

## 🧠 Memory entries (cross-session)

- `~/.claude/projects/.../memory/MEMORY.md` — index
- `feedback_naming_preference.md` — prefer neutral / academic naming over mystical
- `feedback_test_discipline.md` — every code change ships with a unit test
- `feedback_checkpoint_discipline.md` — commit + push to GitHub frequently (heartbeat)
- `feedback_sota_smoke_first.md` — SOTA baseline smoke before any variant
- `feedback_phased_workflow.md` — unit tests → smoke → CIFAR-100 top-K → 3-seed

## 🔗 Sister projects

- [`dlmastery/autoresearch`](https://github.com/dlmastery/autoresearch) — FX prediction
- [`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage) — protocol source-of-truth
- [`dlmastery/autoresearchtabular`](https://github.com/dlmastery/autoresearchtabular) — Higgs UCI
- [`dlmastery/autoresearchspy`](https://github.com/dlmastery/autoresearchspy) — SPY ETF
- [`dlmastery/autoresearchindexstock`](https://github.com/dlmastery/autoresearchindexstock) — QQQ
```
