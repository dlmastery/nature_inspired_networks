# RESEARCH MANIFESTO — `dlmastery/nature_inspired_networks`

> **One-sentence claim:** A single drop-in residual / attention operator —
> the **NaturePriorBlock** — that bakes seven independently-ablatable
> inductive biases (φ/Fibonacci scaling, hexagonal lattices, Platonic /
> icosahedral equivariance, fractal self-similar recursion, toroidal
> closure, Chladni-eigenmode initialization, golden-angle channel
> modulation) into a single module, with each prior testable in
> isolation under an autoresearch protocol with refusal-to-launch gates
> and a Goodhart-fingerprinted composite metric.

This manifesto is the high-level document that every reviewer, future
contributor, or fake committee should read first. It states what we
claim, what we do NOT claim, why this is a defensible research program
rather than a mystical sermon, and how every artifact in this repo
contributes to a falsifiable scientific argument.

---

## 1. Thesis

Modern deep learning has rediscovered, piece by peer-reviewed piece, the
same inductive biases that nature optimized over billions of years and
that humans called "sacred geometry" for thousands of years:

| Sacred name | Engineering object | Canonical citation |
|---|---|---|
| Honeycomb / Flower-of-Life | hexagonal lattice convolution | HexaConv, Hoogeboom et al. 2018, arXiv:1803.02108 |
| Icosahedron / dodecahedron | spherical / Platonic equivariance | Icosahedral CNN, Cohen et al. 2019, arXiv:1902.04615 |
| Flower-of-Life recursion | fractal self-similar architecture | FractalNet, Larsson et al. 2017, arXiv:1605.07648 |
| Torus / vortex | periodic manifold encoding | Deep Networks on Toroids, Pittorino et al. 2022, arXiv:2202.03038 |
| φ / Fibonacci growth | compound channel scaling | Fibonacci-Net, 2025 |
| Cymatic vibration → geometry | wavelet / Fourier-mode init | (novel here: Chladni-eigenmode init) |
| Phyllotaxis / golden angle | rotary / positional modulation | (golden-angle RoPE variants 2025) |

The **Platonic Representation Hypothesis** (Huh, Cheung, Wang, Isola
2024, arXiv:2405.07987) provides the empirical bridge: sufficiently
large neural networks across modalities and objectives converge to a
shared statistical model of reality. The thesis of this repo is that
**sacred-geometry priors accelerate this convergence by imposing the
exact symmetries nature already chose**.

We treat the framing as engineering hypothesis, not mysticism.

## 2. What we claim — and what we explicitly do NOT claim

### We claim

1. **Each of the seven priors can be cleanly ablated in isolation
   inside a single residual block** (`NaturePriorBlock`), so its
   marginal contribution to a composite metric can be measured.
2. **The autoresearch protocol** (Citation Rigor + Reasoning Blob
   Completeness + Goodhart-fingerprinted composite + append-only
   experiment log) **catches engineering bugs before they corrupt
   published numbers**, with empirical evidence of at least three
   bugs caught in this project alone.
3. **At least one prior** — fractal φ-recursion in the previous CIFAR
   sweep — **measurably lifts top-1** in our 12-epoch ablation
   regime (+2.35 pp vs. the priors-off reference), validating that
   the framework discriminates.
4. **The seemingly-compelling "compound efficiency" claim** sometimes
   asserted in popular synthesis (20–50 % gains when all priors are
   combined) **does not hold** at our 12-epoch CIFAR-10 scale; the
   full hybrid is the WORST single-seed row. This is a falsifiable
   negative result, reported in `FINDINGS.md` and `RESULTS.md`.

### We do NOT claim

- ImageNet-1k SOTA. Out of single-GPU budget.
- That sacred geometry is "true" in any metaphysical sense.
- That a positive result on CIFAR-10 generalises to LLMs without further
  validation. The decoder-only LLM track (H01-LLM through H50-LLM,
  H61–H71) is an **explicitly-untested** design space surfaced from the
  extended Grok transcript.
- That every prior helps. Several priors (cymatic-init, toroidal,
  group-conv-max-pool) **demonstrably hurt** in our negative result.
  These are documented honestly in `FINDINGS.md`.

## 3. Why this is defensible research, not a sermon

Three structural choices ensure scientific rigor:

1. **Refusal-to-launch gates.** Every experiment must clear (a)
   Citation Rigor — author / year / venue / title / arXiv ID / relevance
   note, (b) Reasoning Blob Completeness — word-count floors per field,
   and (c) Goodhart fingerprint — the composite formula is SHA-256
   hashed; edits break the next run. There is no `--bypass` flag.
2. **Append-only experiment log.** `experiment_log.jsonl` is the
   immortal record. Corrections append a `_v2` row with a journal entry,
   never overwrite a past result.
3. **Per-experiment archive sub-directories with mandatory READMEs.**
   Every run lives in `ideas/<NN_idea>/experiments/expNNN_<short>/`
   carrying its config, reasoning, run-folder, and a detailed README
   that stands alone — anyone reading just that sub-directory can
   reproduce the experiment from cold.

## 4. The design-space audit

The full 75-hypothesis design space was assembled by chunk-reading four
source documents (the PDF + three Grok transcripts including a 1224-line
extended transcript) and cross-referencing with `IDEA_TABLE.md`:

- **G1 Scaling & Growth** (H01–H10): φ-compound scaling, Fib depth,
  golden-spiral resolution, golden bottleneck, φ-LR scheduler, etc.
- **G2 Layer / Channel / Neuron** (H11–H20): pure Fib MLP, Fib-channel
  CNN, golden skip, Fib-head diversity, φ activation threshold, etc.
- **G3 Topologies & Graphs** (H21–H30): hex φ-packing, toroidal
  φ-closure, Platonic φ-graph, dodecahedral latent, fractal toroidal,
  φ-small-world, Platonic-Fib hybrid, etc.
- **G4 Kernels / Attention / Filters** (H31–H40): golden-spiral kernel,
  Fibottention (Fib dilation attention), Vesica Piscis filter,
  golden-angle RoPE, cymatic wavelet, Metatron kernel overlap, etc.
- **G5 Optimization / Init / Regularization / NAS** (H41–H50): golden
  optimizer, φ-weight init, Fib-pruning, sacred NAS, cymatic loss,
  full sacred hybrid, etc.
- **G6 Topological & bridging** (H51–H60): Betti loss, drop-path
  anytime, 2D-3D icosa unfold bridge, Platonic Transformers, cymatic
  dataset, **C4 max→avg pool fix** (H58 — the top-priority follow-up
  surfaced by the previous campaign's negative result), trained-feature
  Betti, 3-seed uncertainty.
- **G7 Cross-paradigm hybrids** (H61–H75; 15 entries total): Sacred-Liquid-JEPA decoder
  block, toroidal-KV + hex attention + Fib-pruning, Platonic projection
  auxiliary + cymatic teachers, full Sacred-Liquid-JEPA-KAN-GNN-
  Transformer fusion, on-device world model, KAN edges on Metatron
  graphs, low-data cymatic curriculum, icosahedral RoPE for 3D spatial.

Every hypothesis has its own detailed file at
`hypotheses/H<NN>_<short>.md` with the committee-grade content this
repo's audit demands.

## 5. The two parallel design spaces

The extended transcript introduces a **parallel LLM track**: every
CNN-track hypothesis (H01–H50) has a decoder-only Transformer sibling
with PyTorch pseudocode + WikiText-103 / TinyStories / GSM8K
benchmarks at 124M–1B parameter scale on a single RTX 4090. This
doubles the effective design space without doubling the file count —
each `hypotheses/H<NN>.md` contains BOTH a CNN-track and an LLM-track
section.

## 6. Paradigm comparison (extended-transcript chunks 1–8)

Every paradigm currently in vogue — **Liquid Foundation Models**
(LFM2), the **JEPA family** (I-/V-/seq-JEPA), **KAN** (Kolmogorov-Arnold
Networks), **decoder-only Transformers**, and **equivariant GNNs**
under the **GDL** blueprint — is shown to be a partial rediscovery of
the same sacred priors. The 8-chunk comparison covering philosophical
foundations / inductive biases / computational mechanisms / efficiency
/ training / interpretability / limitations / synthesis lives in
`PARADIGM_COMPARISON.md` (separate document; auto-generated from the
extended transcript).

## 7. Hardware contract

- Target: **1× RTX 4090 Laptop, 16 GB VRAM, Windows 11**.
- bf16 AMP + cosine LR + label smoothing + RandAugment.
- `num_workers: 0` on Windows (worker-spawn wedges).
- Python 3.13 corp-cert SSL workaround: `curl.exe -kL` for CIFAR;
  torchvision verifies MD5.
- Background-task launches **always** preceded by `git push` (per
  checkpoint discipline).

## 8. What an experiment archive looks like

Per `CLAUDE.md` rule 8–9 and the `autoresearch-experiment-archive` skill:

```
ideas/<NN_idea>/experiments/expNNN_<short>/
├── README.md          ← very detailed design + result + verdict
├── config.yaml        ← exact config used
├── reasoning.json     ← pre-run + post-run gated entry
├── run_seed0/{metrics.json, history.json, best.pt}
├── run_seed1/…
├── run_seed2/…
├── dashboard/{dashboard.html, plot_pareto.png, …}
└── verification/      ← committee-grade audit artifacts
    ├── tests.txt      ← unit-test output (timestamp + green/red)
    ├── smoke.txt      ← end-to-end forward-pass smoke result
    ├── gates.txt      ← which gates fired / passed
    └── reproduction.txt ← clean-clone reproduction log
```

The `verification/` directory is the answer to "how do we know you did
it correctly?" — every experiment must carry one.

## 9. Repository layout (canonical)

See `CLAUDE.md` § 3 for the canonical tree. Highlights:

- `src/nature_inspired_networks/` — shared infra (training engine,
  dashboard, gates, topology, loaders, single import surface for all
  priors)
- `ideas/<NN>/` — taxonomy: one self-contained sub-project per
  hypothesis, with `_TEMPLATE/` as the scaffold
- `hypotheses/H<NN>_<short>.md` — committee-grade per-hypothesis design
  document (the long, detailed write-up)
- `skills/` — content-agnostic auto-research skills (checkpoint,
  experiment, dashboard, reasoning-entry, modular-block, dataset-loader,
  topology-metrics, experiment-archive, idea-scaffold)
- `EXPERIMENT_LOG.md` — append-only master long-list of every run
- `EXPERIMENT_LEDGER.md` — chunk-by-chunk audit log of source-document
  reads
- `IDEA_TABLE.md` — 75-hypothesis status table (one row per H)
- `MANIFESTO.md` — this file
- `PAPER.md` / `PAPER_ABSTRACT.md` — auto-fillable paper draft
- `FINDINGS.md` / `RESULTS.md` — campaign verdicts + per-run narratives
- `SOTA_COMPARISON.md` — honest mapping to the literature
- `CLAUDE.md` — normative rules

## 10. What "done" looks like

A fake committee evaluating this work should expect to see:

1. **One README they can read in 5 minutes** that orients them
   (`README.md`).
2. **One manifesto** (this file).
3. **One audit document** (`IDEA_TABLE.md`) showing the full design
   space + status of each.
4. **One master experiment list** (`EXPERIMENT_LOG.md`).
5. **One per-hypothesis design document** (`hypotheses/H<NN>.md`)
   with mechanism, prediction, citations, falsifier, expected outcome,
   committee Q&A.
6. **One per-experiment archive** with README + reasoning + run-folder
   + dashboard + verification.
7. **One ledger** (`EXPERIMENT_LEDGER.md`) showing the chunk-by-chunk
   audit of source documents.
8. **One paradigm comparison** (`PARADIGM_COMPARISON.md`).
9. **One findings document** with the honest negative result
   (`FINDINGS.md`).
10. **Live dashboard** at the Pages URL with Pareto / ablation /
    training-curve / Betti panels.

Every one of those is in this repo or is being generated by the
agentic team building it in parallel.

## 11. License & credits

MIT. Author: dlmastery (eranti@gmail.com).
Inherits the autoresearch protocol from
[`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage)
and [`dlmastery/autoresearch`](https://github.com/dlmastery/autoresearch).

The mystical framing in the source PDF inspired the research program
but is acknowledged only in the prose intros — every artifact name
and class identifier in this repo is academic / neutral
(`NaturePriorBlock`, not `SacredGeoBlock`).
