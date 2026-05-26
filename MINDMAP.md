# MINDMAP — `dlmastery/nature_inspired_networks`

> Single-page link map of every artifact in this repo. Start here if
> you do not know where to look.
> Want the operator quick-start? → [README.md](README.md).
> Want the research argument? → [MANIFESTO.md](MANIFESTO.md).

## 🌐 Live URLs

- **Repo (public):** https://github.com/dlmastery/nature_inspired_networks
- **Dashboard (live):** https://dlmastery.github.io/nature_inspired_networks/dashboard/dashboard.html
- **Pages root:** https://dlmastery.github.io/nature_inspired_networks/

## 🗺 Top-level layout (one-line tour)

```
nature_inspired_networks/
├── README.md                ← operator quick-start
├── MINDMAP.md               ← THIS FILE
├── MANIFESTO.md             ← research argument (committee-grade)
├── CLAUDE.md                ← normative rules + 12 always-true assertions
├── ARCHITECTURE.md          ← module + shape tables
├── AUTORESEARCH_PROCESS.md  ← 7-step ritual + gates
├── IDEA_TABLE.md            ← 75-hypothesis status table (G1-G7)
├── EXPERIMENT_LOG.md        ← master long-list (Tiers 0-6)
├── EXPERIMENT_LEDGER.md     ← chunk-by-chunk audit of source documents
├── PARADIGM_COMPARISON.md   ← 8-chunk Liquid/JEPA/KAN/Transformer/GNN comparison
├── FINDINGS.md              ← campaign verdicts (incl. H50 negative + H58 DISCARD)
├── RESULTS.md               ← auto-generated per-run narratives
├── SOTA_COMPARISON.md       ← honest map to the literature
├── PAPER.md                 ← draft paper
├── paper_abstract.md        ← 1-page summary
├── SETUP.md                 ← bring-up steps
├── MEDIUM.md                ← blog-style narrative
├── sota_catalog.yaml        ← prior-art YAML (single source of truth)
├── pyproject.toml
├── hypotheses/              ← 75 committee-grade docs (themed subdirs)
├── ideas/                   ← 8 modular sub-projects (full taxonomy)
├── src/nature_inspired_networks/  ← shared infrastructure
├── scripts/                 ← run_sweep, build_dashboard, build_report, compute_topology
├── skills/                  ← 9 content-agnostic reusable auto-research skills
├── tests/                   ← unit tests (29 green + 68 idea-local)
├── configs/                 ← shared YAML configs
├── experiments/             ← legacy CIFAR-10 archive (run-folders + reasoning + dashboard)
├── dashboard/               ← latest aggregated dashboard
├── docs/                    ← GitHub Pages root (index.html + dashboard mirror)
└── memory/                  ← project checkpoint markdown
```

## 📚 Documentation map

### Read first

1. **[README.md](README.md)** — 5-minute orientation
2. **[MANIFESTO.md](MANIFESTO.md)** — what we claim + what we don't
3. **[FINDINGS.md](FINDINGS.md)** — current campaign verdicts (incl. negative results)
4. **[IDEA_TABLE.md](IDEA_TABLE.md)** — the 75-hypothesis status table

### Normative rules

- **[CLAUDE.md](CLAUDE.md)** — 12 always-true assertions, hardware contract, refusal table
- **[AUTORESEARCH_PROCESS.md](AUTORESEARCH_PROCESS.md)** — the 7-step ritual + gate stack

### Per-hypothesis design docs (75 files, 7 thematic groups)

- **[hypotheses/INDEX.md](hypotheses/INDEX.md)** — file map for the 75 docs
- **[hypotheses/_TEMPLATE.md](hypotheses/_TEMPLATE.md)** — the contract every hypothesis doc fills

| Group | Path | Count | Theme |
|---|---|---|---|
| G1 | [`hypotheses/g1_scaling_growth/`](hypotheses/g1_scaling_growth/) | 10 | φ / Fibonacci growth laws |
| G2 | [`hypotheses/g2_layer_channel_neuron/`](hypotheses/g2_layer_channel_neuron/) | 10 | Fib-sized MLPs, channels, neurons |
| G3 | [`hypotheses/g3_topologies_graphs/`](hypotheses/g3_topologies_graphs/) | 10 | Hex / toroidal / Platonic graphs |
| G4 | [`hypotheses/g4_kernels_attention_filters/`](hypotheses/g4_kernels_attention_filters/) | 10 | Spirals, Fibottention, Vesica, cymatic wavelets |
| G5 | [`hypotheses/g5_optimization_init_reg_nas/`](hypotheses/g5_optimization_init_reg_nas/) | 10 | Optimizer, init, regularization, NAS, full hybrid |
| G6 | [`hypotheses/g6_topological_bridging/`](hypotheses/g6_topological_bridging/) | 10 | Betti loss, drop-path, H58 fix, trained-Betti, 3-seed |
| G7 | [`hypotheses/g7_cross_paradigm_hybrids/`](hypotheses/g7_cross_paradigm_hybrids/) | 15 | Sacred + Liquid / JEPA / KAN / Transformer / GNN |

### Cross-paradigm comparison

- **[PARADIGM_COMPARISON.md](PARADIGM_COMPARISON.md)** — 728-line synthesis: Liquid (LFM2) / JEPA family / KAN / decoder-only Transformer / equivariant GNN vs. the SacredGeoBlock-v2 synthesis, in 8 chunks (philosophy / inductive biases / mechanisms / efficiency / training / interpretability / limitations / synthesis)

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
| ✓ | [`ideas/50_full_sacred_hybrid/`](ideas/50_full_sacred_hybrid/) (8 tests pass; H50 negative result archive) |
| ✓ | [`ideas/58_group_avg_pool/`](ideas/58_group_avg_pool/) (8 tests pass; H58 DISCARD archive) |

68 idea-local tests, all green. The remaining 67 hypotheses await
their own `ideas/<NN>_<short>/` sub-projects.

## 🔬 Source code

```
src/nature_inspired_networks/
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
├── dashboard.py     ← rich autoresearchspy-style HTML dashboard
└── runner.py        ← single-experiment entrypoint
```

## 🛠 Scripts

- [`scripts/run_sweep.py`](scripts/run_sweep.py) — ablation matrix driver (curated 11-row + full 20-row)
- [`scripts/build_dashboard.py`](scripts/build_dashboard.py) — regenerates the HTML dashboard from `experiment_log.jsonl`
- [`scripts/build_report.py`](scripts/build_report.py) — emits `RESULTS.md` with KEEP / NEAR-MISS / DISCARD verdicts
- [`scripts/compute_topology.py`](scripts/compute_topology.py) — Betti curves from `best.pt` checkpoints

## 🎯 Reusable skills (`skills/`)

9 content-agnostic auto-research skills any future project can pick up:

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

## 📊 Experiment artifacts

```
experiments/
├── experiment_log.jsonl                ← append-only run log
├── reasoning_annotations.json          ← citation-gated reasoning per run
├── betti.json                          ← topology curves
├── plot_*.png                          ← Pareto / ablation / curves / Betti
├── cifar10/
│   ├── baseline_resnet20_seed0/        ← top-1 84.78%, composite 0.8458 (literature anchor)
│   ├── baseline_sg_vanilla_seed0/      ← 82.16%, 0.8258
│   ├── sg_chan_fib_seed0/              ← 80.11%, 0.8135
│   ├── sg_chan_phi_seed0/              ← 80.11%, 0.8152
│   ├── sg_only_hex_seed0/              ← 79.32%, 0.7941
│   ├── sg_only_group_seed0/            ← 69.84%, 0.6937
│   ├── sg_only_fractal_seed0/          ← 82.46%, 0.8104 (+ only lifter)
│   ├── sg_only_toroidal_seed0/         ← 78.05%, 0.7768
│   ├── sg_only_cymatic_init_seed0/     ← 77.44%, 0.7883
│   ├── sg_only_golden_modulate_seed0/  ← 79.81%, 0.8042
│   ├── sg_full_fib_seed0/              ← 73.24%, 0.6966 (compound FAILS)
│   ├── sg_only_group_avg_seed0/        ← 65.38%, 0.6597 (H58 DISCARD; +best.pt)
│   └── sg_full_fib_avg_seed0/          ← 66.86%, 0.6432 (H58 confirm; +best.pt)
```

13 runs archived. Headline negative result: full hybrid is the worst row.

## 🧠 Memory entries (cross-session)

- `~/.claude/projects/.../memory/MEMORY.md` — index
- `feedback_naming_preference.md` — prefer neutral / academic naming over mystical
- `feedback_test_discipline.md` — every code change ships with a unit test
- `feedback_checkpoint_discipline.md` — commit + push to GitHub frequently (heartbeat)

## 🔗 Sister projects

- [`dlmastery/autoresearch`](https://github.com/dlmastery/autoresearch) — FX prediction
- [`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage) — protocol source-of-truth
- [`dlmastery/autoresearchtabular`](https://github.com/dlmastery/autoresearchtabular) — Higgs UCI
- [`dlmastery/autoresearchspy`](https://github.com/dlmastery/autoresearchspy) — SPY ETF
- [`dlmastery/autoresearchindexstock`](https://github.com/dlmastery/autoresearchindexstock) — QQQ
