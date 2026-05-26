# SacredGeoBlock: An Ablation Surface for Sacred-Geometry Priors in Image Classification (DRAFT)

> Draft paper. Numbers updated by `scripts/build_dashboard.py` after every
> sweep. This is the operator-readable version; the final paper PDF
> (LaTeX/NeurIPS template) is generated from this file plus the
> dashboard plots.

## 1. Introduction

Geometric Deep Learning (GDL) and Topological Deep Learning (TDL) have
repeatedly converged on a small set of geometric priors — hexagonal
lattices, Platonic / icosahedral symmetry groups, fractal recursion,
toroidal manifolds, and golden-ratio compound scaling — as cheap,
principled replacements for the standard square-grid CNN scaffold.
HexaConv (Hoogeboom 2018) reports 25–42 % memory / time savings for
isotropic data. Icosahedral CNNs (Cohen 2019) preserve sphere-rotation
equivariance at single-conv cost. FractalNet (Larsson 2017) matches
ResNet without residuals via self-similar paths. Toroidal embeddings
(Pittorino 2022) reveal flat loss basins. Fibonacci-Net (2025) ships a
1.39 M-parameter brain-tumor classifier on Fibonacci-scaled channels.

No prior work synthesises **all** of these into a single
toggle-driven residual block to expose the ablation surface.

The user-provided PDF "*Sacred Geometry and Neural Networks*" makes the
case that these GDL/TDL priors are precisely what ancient and natural
geometry has been pointing at all along — hexagonal Chladni nodes,
Platonic solids as the unique fully-symmetric polyhedra, φ-governed
phyllotaxis. **We treat the framing as engineering hypothesis, not
mysticism.** Each prior has a peer-reviewed citation; the composite
block exists to *test* whether they compound.

This repo's contribution is mechanical, not theoretical:

1. **`SacredGeoBlock`** — a single drop-in residual block whose six
   sacred priors and three channel-scaling modes can be ablated
   independently with one flag flip per experiment.
2. **The autoresearch protocol** (inherited verbatim from
   `dlmastery/autoresearchimage`) wrapped around the block: citation
   rigor + reasoning blob + Goodhart-fingerprinted composite metric.
3. **A reproducible ablation surface on CIFAR-10** running in
   ≲ 90 min on a single RTX 4090 Laptop, with a sortable HTML
   dashboard, Pareto plots, β-collapse curves and CKA.

## 2. SacredGeoBlock — Definition

The block (§ARCHITECTURE) takes input $x \in \mathbb{R}^{B \times C_\text{in} \times H \times W}$
and produces $y \in \mathbb{R}^{B \times C_\text{out} \times H' \times W'}$:

$$
\begin{aligned}
  z_1 &= \text{BN}\!\Big(\,\text{Conv}_{f_1}(x; \text{stride})\Big)\\
  z_2 &= \text{ReLU}(z_1)\\
  z_3 &= \text{Conv}_{f_2}(z_2) \quad \text{or}\quad \text{FractalPath}_{f_2}(z_2)\\
  z_4 &= z_3 \odot g_\phi(\alpha) \quad \text{(if golden-modulate)}\\
   y  &= \text{ReLU}\!\big(z_4 + \text{Skip}(x)\big)
\end{aligned}
$$

where $\text{Conv}_{f}$ is selected from
{plain 3×3, hex-masked 7-tap (HexaConv), C4 group conv with 4-orbit max-pool
(Cohen 2016)} by the flag tuple $f = (f_\text{hex}, f_\text{group})$;
$\text{FractalPath}$ is a depth-2 FractalNet sub-block; padding is
circular if $f_\text{toroidal}$; the convolution is initialised with
Chladni eigenmodes if $f_\text{cymatic}$; and the golden gate is

$$
g_\phi(\alpha)_k = \tfrac{1}{2}\cos\!\big(\tfrac{2\pi k}{\varphi} + \alpha\big) + \tfrac{1}{2},
$$

with $\alpha \in \mathbb{R}$ learnable and $\varphi = (1+\sqrt{5})/2$.

The block composes into SacredGeoNet by stacking three stages of three
blocks each with channel widths from
$c_k = \text{round}_8\!\big(c_0 \cdot \tfrac{F_{k+1}}{F_1}\big)$ (mode `fib`),
$c_0\varphi^k$ (`phi`), or $c_0(k+1)$ (`linear` control).

## 3. Experimental setup

- **Dataset:** CIFAR-10 (50 k / 10 k), augmented with
  `RandomCrop(32, padding=4)` + `RandomHorizontalFlip`.
- **Optimiser:** AdamW, $\eta = 10^{-3}$, weight decay $5 \cdot 10^{-4}$,
  cosine schedule.
- **Loss:** cross-entropy with label smoothing 0.1.
- **Precision:** bfloat16 autocast (RTX 4090 supports bf16 natively).
- **Epochs:** 12 (chosen so 11 runs fit in ~90 min on a 4090 Laptop).
- **Hardware:** 1× RTX 4090 Laptop 16 GB, Windows 11.
- **Compositeformula:**
  $\textsf{composite} = \textsf{top-1} - 0.05\,\log_{10}(\textsf{params}_M) - 0.05\,\log_{10}(\textsf{latency}_\text{ms})$
  (SHA-256 fingerprinted).

## 4. Results

*Filled by `scripts/build_dashboard.py` after every sweep — see
`dashboard/dashboard.html` for the live table. The summary table
below is regenerated.*

<!-- AUTOFILL:summary_table -->

### 4.1 Pareto fronts

See `dashboard/plot_pareto.png`. Three panels — top-1 vs. params,
FLOPs, and batch=1 GPU latency. Baselines are blue stars; SacredGeo
variants are orange dots. A variant dominates if it is up-and-left.

### 4.2 Ablation matrix

See `dashboard/plot_ablation.png`. Each row is one tag (one prior
toggled), sorted by composite. Error bars are seed-std when
`--seeds 0 1 2` is used.

### 4.3 Topology — Betti collapse

See `dashboard/plot_betti.png`. β₀ should drop monotonically across
stages as the net simplifies its data topology (TDL prediction);
β₁ tracks 1-D holes. Sacred priors are hypothesised to *accelerate* the
β₀ collapse without sacrificing top-1.

## 5. Limitations

This is a **feasibility + ablation campaign**, not an ImageNet SOTA
claim. See `README.md → Scope and honesty`.

## 6. Open axes

See `README.md → Open axes for the next campaign`.

## 7. Citations

Inline arXiv IDs in `src/sacgeo/priors.py`, `src/sacgeo/blocks.py`,
and the canonical `sota_catalog.yaml`.
