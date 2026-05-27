## 1. Mathematical Foundations

> The geometric, algebraic, and topological invariants that nature has been quietly solving for billions of years — and the deep-learning primitives that finally let us bake them in.

### 1.1 The Golden Ratio φ & Continued-Fraction Optimality

*Anchors H01-H10 (φ-compound scaling, Fibonacci depth, φ-LR), H17, H34, H39, H41, H42 (golden-ratio optimizer / init / regularization).*

- **Fibottention — official repo** — Non-overlapping Fibonacci-dilated attention across heads built on the Wythoff array; delivers O(N log N) sparse attention with φ-spaced channels. The canonical implementation for H32/H34/H36 (golden-angle & φ-spaced priors). [GitHub] [paper]
- **Golden Ratio in Neural Architecture Search & Optimizers** — Jaeger 2020 (and follow-ups) on φ-scheduling and continued-fraction learning-rate schedules; the literature anchor for H10 (φ-decay LR) and H41 (golden-ratio optimizer). (source: cite-only, no URL given — search arXiv for Jaeger 2020 + "golden ratio learning rate pytorch")
- **Phyllotactic Positional Encodings (2025)** — Golden-angle RoPE variants that swap the standard RoFormer base frequency for φ-spaced phyllotaxis spacing; direct H34 (RoPE-φ) and H36 (φ-spiral PE) implementation. (source: cite-only, "check repo forks" — search "golden-angle RoPE" on GitHub for 2025 implementations)
- **Golden Spiral Kernel Initialization notebook** — Interactive φ-Fibonacci channel explorer ported from the `dlmastery/nature_inspired_networks` reference repo; useful sanity-check tooling for H04/H31. [demo]

### 1.2 The Honeycomb Conjecture & Hexagonal Optimality

*Anchors H21 (hexagonal φ-packing), H58 (group avg-pool fix), and the entire grid-cell bridge in §2.1.*

- **HexaConv — official PyTorch implementation** — Canonical 2018 ICLR paper code for hexagonal convolution with C₆ group equivariance; reported +1.5pp CIFAR-10 gains and is the direct H21/H62 baseline. [paper] [GitHub]
- **HexagDLy — hexagonal convolutions for PyTorch** — Production-ready library used in astroparticle physics; supports arbitrary kernel size/stride on hex grids and is the go-to dependency for compound-eye-inspired vision work. [GitHub]
- **HexCNN framework** — GPU-optimized native hexagonal CNNs (2021); roughly 2.3× faster than naïve masking, important when the hex prior must scale to ImageNet. (source: cite-only, no URL given)
- **Hales' Honeycomb Conjecture (theory anchor)** — The mathematical proof that the regular hexagonal tiling minimises perimeter; cross-referenced through the *Geometric Deep Learning* book companion resources. [paper]

### 1.3 Platonic Solids & Their Rotation Groups (Icosahedral Equivariance)

*Anchors H23-H25 (Platonic / icosa / dodeca graphs and latents), H37 (pentagonal attention), H49 (PRH alignment), H55 (Platonic Transformers).*

- **icoCNN — PyTorch Icosahedral CNNs** — Clean reference implementation of Cohen et al. 2019 ICML "Gauge Equivariant Convolutional Networks and the Icosahedral CNN"; the cleanest entry point for H24. [paper] [GitHub]
- **Awesome Equivariant Networks** — Massive curated meta-list of gauge / SE(3) / Platonic equivariant papers and code, covering Icosahedral CNN, e3nn, EquiFormer, Platonic Transformers, and more. [GitHub]
- **e3nn — Euclidean Neural Networks** — The de-facto irrep-based library powering most 2024-2026 SE(3) / Platonic equivariant work, including icosahedral support; mandatory dependency for H24/H30/H55/H71. [GitHub]
- **Platonic Transformers (2025)** — Discrete Platonic-group attention; the literature anchor for H55 (Islam 2025) and a candidate backbone for the SacredGeoBlock synthesis (H67). (source: cite-only, "code linked in repo — check for updates")
- **Icosahedral group visualizations & irreps demo** — Interactive explorer of icosahedral symmetry and its irreducible representations, part of the broader Neural Geometry awesome list. [demo]

### 1.4 Phyllotaxis and the Golden Angle

*Anchors H03 (golden-spiral resolution), H31 (golden-spiral kernel), H34, H36.*

- **Phyllotactic Positional Encodings (2025 follow-ups)** — Forks of the RoFormer codebase that swap φ-spaced frequencies into RoPE; the most practical drop-in for golden-angle (137.5°) positional priors. (source: cite-only — "search 'golden-angle RoPE' on GitHub for 2025 implementations")
- **Golden Spiral Kernel Initialization notebooks** — Direct interactive code from the reference repo: φ-scaled filters and spirals for H31. [demo]
- **Vogel 1979 sunflower packing + Phyllotaxis Statistics across 12,000 species (2024)** — The original sunflower-disk packing model plus a 2024 large-scale species-level dataset; biological grounding for any φ-divergence positional encoding. [paper]

### 1.5 Fractal Dimension and Natural Mid-Band

*Anchors H05 (fractal φ-recursion), H26 (fractal toroidal), H38 (fractal golden filter), H52 (drop-path anytime).*

- **FractalNet — official repo (Larsson et al. 2017)** — Canonical ultra-deep fractal architecture without residuals, with drop-path regularisation and anytime inference; the literature anchor for H05/H26/H38 and the natural baseline for H52 anytime evaluation. [paper] [GitHub]
- **PyTorch FractalNet (pt.fractalnet)** — Clean, trainable PyTorch port with CIFAR-10/100 and ImageNet training scripts; the ablation baseline for H05/H26/H38. [GitHub]
- **Fractal Generative Models (2025)** — Modern fractal-based generative architecture extending fractal priors to diffusion-style models. [arXiv:2502.17437]
- **Neural Fractal Explorer (complex-valued dynamical systems)** — GPU-accelerated visualiser of fractal behaviour emerging from neural networks; useful intuition pump for D ≈ 1.3-1.5 mid-band claims. [demo]
- **Fractal-Inspired Message Passing GNNs with Fractal Nodes (2025)** — Recent extension of fractal recursion to graph neural networks; bridges H05 with H26/H30. (source: cite-only, no URL given)

### 1.6 Cymatics and Chladni Eigenmodes

*Anchors H28 (cymatic hex resonance), H35 (cymatic wavelet kernels — already a documented negative result in the repo), H46 (cymatic loss), H56 (synthetic cymatic dataset), H66/H70.*

- **Chladni Plate Simulator (Cymatics)** — Interactive eigenmode generator; the starting point for H35/H66 cymatic QKV initialisation experiments and for synthesising the H56 cymatic dataset. [demo]
- **Mishra et al. 2023 — "Cymatic-pattern initialization for transformer attention"** — Early DL paper anchor that flattens Chladni eigenmodes into QKV weight matrices; the literature seed for H35. [arXiv:TBD]
- **"Form Follows Algorithm: Differentiation of Chladni Patterns"** — Generative Art Conference paper with Processing / Cymatify code examples; visual and educational companion to the cymatic-init line of work. [paper] [code]

### 1.7 Persistent Homology and Betti Numbers

*Anchors H51 (topological Betti loss), H54 (PH activation regularization), H59 (trained-feature Betti), H65 (topology-aware losses).*

- **TopologyLayer — Differentiable Persistent Homology for PyTorch** — The foundational 2020 ICLR library; computes filtrations, Betti numbers, and persistence diagrams end-to-end, making H51/H54 directly trainable. [paper] [GitHub]
- **GUDHI TDA tutorials + PyTorch integration** — Comprehensive notebooks covering persistent homology, filtrations, and Betti curves; the standard reference toolkit for any TDL pipeline. [GitHub]
- **torch_topological / pytorch-topological** — Modern, high-performance differentiable PH framework with multiple backends and topological loss functions; pairs with GUDHI / ripser for H54 activation regularisation. [GitHub]
- **torchph** — High-performance PyTorch persistent-homology extensions; the speed-focused alternative when TopologyLayer becomes a bottleneck. [GitHub]
- **Topo-Loss for Image Segmentation** — Reimplementation of topological loss functions using persistent homology; a worked example for porting H51 to medical-imaging Tier-1 datasets. [GitHub]

### 1.8 Information-Theoretic Compounding & Orthogonal Symmetries

*Anchors H49 (PRH alignment loss), H67 (full hybrid synthesis).*

- **Platonic Representation Hypothesis follow-ups** — No single repo but extensively explored across PRH literature; the central claim is that scaling drives multimodal models toward a shared Platonic latent. Cross-reference with the FINDINGS.md negative results on non-orthogonal priors in `dlmastery/nature_inspired_networks`. (source: cite-only, no URL given)
- **CKA similarity tools (equivariant-network benchmarks)** — Standard Centered Kernel Alignment implementations used across the equivariant-networks community to test representational convergence; the measurement backbone for H49. (source: cite-only, no URL given)

### 1.9 Equivariant Group Representations (deep dive)

*Anchors H24 (icosa φ-equivariant), H30 (Platonic-Fib hybrid), H55 (Platonic Transformer), H71 (icosa RoPE).*

- **e3nn — Euclidean Neural Networks** — Gold-standard library for SE(3) / Platonic / irrep-based equivariant layers; powers nearly every 2024-2026 equivariant-attention paper. [GitHub]
- **Awesome Equivariant Networks** — Massive curated list spanning Icosahedral CNN, Gauge CNNs, Platonic Transformers, EquiFormer, GotenNet, and more. [GitHub]
- **icoCNN — Icosahedral CNNs** — Clean implementation of the 2019 Cohen et al. gauge-equivariant icosahedral network; the cleanest reference for H24. [GitHub] [paper]

---

## 2. Neuroscience and Biology Grounding (partial — §2.1-2.3)

> Where the mathematical priors of §1 actually appear in biological wetware — and the datasets / connectomes that let us check our inductive biases against ground truth.

### 2.1 Grid Cells and the Hexagonal Cortex

*Direct biological anchor for H21 (hexagonal φ-packing) and H62 (hex priors); the toroidal-manifold result is the empirical foundation for H22 (toroidal closure) and H68.*

- **GridCellTorus — toroidal topology code (Gardner et al. 2022)** — Official code and data for the persistent-homology proof that grid-cell population activity lives on a torus; the canonical empirical anchor for H21/H22/H62. [paper] [GitHub]
- **Oscillations & Toroidal Topology in Grid Cells (2025)** — Recent follow-up showing the role of theta / eta oscillations in shaping the toroidal manifold; ships with PH analysis scripts. [paper]
- **Unified Theory of Grid Cells via Pattern Formation (Sorscher et al. 2023)** — Theory plus simulation code showing how RNNs spontaneously develop hexagonal firing fields; the H21/H62 mechanistic anchor that links pattern-formation PDEs to learned hex grids. [paper] [code]
- **Emergence of grid-like representations — 2023-2025 RNN training repos** — A cluster of follow-up implementations reproducing hexagonal firing fields in trained RNNs. (source: cite-only — "search 'grid cell RNN pytorch'")

### 2.2 Place Cells, Head-Direction Cells, and Boundary Cells

*Adjacent biological evidence for H22 (toroidal closure) and the hippocampal manifold side of H51/H54.*

- **GridCellTorus repo (place / head-direction ensemble analysis)** — Same Gardner et al. repository as above, which also contains place-cell, head-direction, and boundary-cell ensemble analysis tooling. [GitHub]
- **Hippocampal manifold papers** — A family of follow-ups that reuse the same persistent-homology pipelines (TopologyLayer + GUDHI from §1.7) on hippocampal recordings; the empirical motivation for H54 PH-activation regularisation. (source: cite-only, no URL given)

### 2.3 Cortical Columns and Minicolumns

*Anchors the hex/pentagonal-attention story (H21, H37) at the cortical-microcircuit scale.*

- **Hexagonal minicolumn packing measurements (Casanova 2024)** — Cortical-minicolumn packing statistics visualised in connectome repos; pairs with the HexCNN / HexaConv libraries from §1.2 to give a DL analogue of cortical hex packing and motivates H37 pentagonal/hex attention. (source: cite-only, no URL given)
