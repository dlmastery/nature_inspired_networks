# nature-inspired-geometry as Engineering Inductive Bias

> Building `NaturePriorBlock` — a single drop-in residual block that puts
> hex lattices, Platonic equivariance, fractal recursion, toroidal padding,
> golden-ratio scaling, and Chladni-mode init under one ablation surface —
> on a 16 GB laptop GPU, in an autoresearch protocol with refusal-to-launch
> gates, in a weekend.

---

## The pitch (180 words)

The Geometric and Topological Deep Learning literature has, over the last
decade, quietly built every component of what an esoterically-minded
reader might call "nature-inspired-geometry as inductive bias":

- **Hexagonal lattices** — HexaConv (Hoogeboom 2018) reports 25–42 %
  memory/time savings for isotropic data.
- **Platonic / icosahedral equivariance** — Icosahedral CNN (Cohen 2019)
  preserves spherical rotation invariance at single-`conv2d` cost.
- **Fractal recursion** — FractalNet (Larsson 2017) matches ResNet
  without residuals by stacking self-similar paths.
- **Toroidal manifolds** — Toroidal embeddings (Pittorino 2022) reveal
  flat loss basins via removed symmetries.
- **φ / Fibonacci channel scaling** — Fibonacci-Net (2025) ships a
  1.39 M-parameter brain-tumor classifier on Fibonacci widths.

No prior repo combines all five (plus a novel Chladni-eigenmode filter
init) into a single drop-in residual block and ablates them
independently. This one does. The code, the autoresearch protocol, the
sortable HTML dashboard, and a 30-minute reproduction recipe are all
in `dlmastery/nature_inspired_networks`.

---

## Why this is not woo

The framing matters. The source PDF that motivated the project leans
into "nature-inspired-geometry" as a research aesthetic — Chladni plates,
Platonic solids, golden spirals, Metatron's Cube. But each named
prior maps **exactly** to a peer-reviewed GDL/TDL paper with an
arXiv ID:

| Nature-Inspired name | engineering object | arXiv |
|---|---|---|
| Flower-of-Life | hex / hexagonal lattice | 1803.02108 (HexaConv) |
| Platonic solid (icosa) | rotation-group equivariance | 1902.04615 (IcosaCNN) |
| Self-similar recursion | FractalNet path | 1605.07648 |
| Torus / Yantra | circular-pad CNN | 2202.03038 (TorusNets) |
| φ / Fibonacci | compound channel scaling | 2502.… (Fibonacci-Net) |
| Cymatics | Chladni-mode filter init | (this repo) |

We treat each one as an inductive bias to be **tested**, not believed.
The ablation surface (one flag at a time) is what makes the test
interpretable: every row in the sweep is one prior moving against
the all-priors-off control.

---

## The protocol — autoresearch gates

This repo borrows verbatim from `dlmastery/autoresearchimage`'s
research protocol: every experiment must clear three gates before
the trainer is allowed to launch.

1. **Citation Rigor.** Every motivation must be backed by an
   `Author1, Author2, … YEAR VENUE 'Title' (arXiv:XXXX.XXXXX) —
   relevance` line. The regex (in `src/nature_inspired_networks/reasoning.py`) rejects
   `(He2016)` style parentheticals.
2. **Reasoning Blob Completeness.** Word-count floors per field:
   diagnosis ≥ 60, hypothesis ≥ 50, prediction ≥ 25, verdict ≥ 30,
   learning ≥ 40. If you can't write 60 words about why you're
   running an experiment, you're not ready to run it.
3. **Goodhart fingerprint.** The composite formula
   $\text{composite} = \text{top-1} - 0.05\,\log_{10}(\text{params}_M) - 0.05\,\log_{10}(\text{lat}_\text{ms})$
   is SHA-256 hashed at first run. Any edit to that string makes
   the next experiment refuse to launch.

There is **no `--bypass` flag**. If the gates reject your reasoning
entry, you fix the entry, you do not soften the gate. Three actual
bugs in this campaign were caught by the gates (see `AUTORESEARCH_PROCESS.md`).

---

## The block — six flags, three channel modes

```python
class NaturePriorFlags:
    hex: bool             # 3x3 hex-masked 7-tap conv (HexaConv 2018)
    group: bool           # C4 group conv, 4-orbit max-pool (Cohen 2016)
    fractal: bool         # depth-2 recursive sub-block (Larsson 2017)
    toroidal: bool        # circular padding (Pittorino 2022)
    cymatic_init: bool    # Chladni eigenmodes + He scale (novel)
    golden_modulate: bool # per-channel rotary by 2π/φ phases (novel)
```

plus

```python
channel_mode in {"fib", "phi", "linear"}
```

11 cells in the curated CIFAR-10 sweep × 12 epochs each ≈ 65 minutes
of wall-clock on a single RTX 4090 Laptop (16 GB). A sortable HTML
dashboard with Pareto fronts, ablation bars, training curves,
β-collapse curves and CKA is rebuilt by
`scripts/build_dashboard.py` after every commit.

---

## What the dashboard shows

Live at https://dlmastery.github.io/nature_inspired_networks/dashboard/dashboard.html

- **Pareto fronts** — top-1 vs. {params, FLOPs, batch=1 GPU latency},
  baselines as blue stars, NaturePrior variants as orange dots. A variant
  dominates if it's up-and-left.
- **Ablation matrix** — each row a tag, bars showing top-1 with
  seed-std error bars (when `--seeds 0 1 2`), label = params.
- **Training curves** — test top-1 over 12 epochs per variant.
- **β-collapse curves** — persistent-homology β₀/β₁ on stage features,
  measuring how fast the network simplifies its data topology
  (TDL prediction).
- **Sortable runs table** — every metric per run; click a column
  header to sort, type to filter.

---

## What this is NOT

- **Not** an ImageNet SOTA claim. CIFAR-10 at 12 epochs is a
  feasibility study; the literature asymptote is ~9 points above us.
- **Not** the full Platonic equivariance. C4 group conv is a cheap
  proxy. Full Icosahedral equivariance needs `e2cnn`, which lacks a
  Python 3.13 wheel.
- **Not** a 3-seed campaign. Default is `--seeds 0`. Add `1 2` for
  error bars; expect ~3× wall-clock.

---

## What's next

The full source PDF lists 50 hypotheses across multiple datasets
(Tiny ImageNet, MedMNIST, IcoMNIST, ModelNet40, ogbg-molhiv,
ImageNet-1k). This repo is the Tier-1 foundation: a working block,
a reproducible CIFAR-10 ablation, a dashboard the next Claude or
contributor can pick up cold. Tier-2 MedMNIST is wired in
`src/nature_inspired_networks/data.py`; the ablation matrix simply needs to be re-run
with `dataset: medmnist:pathmnist`.

The geometry is the code.
