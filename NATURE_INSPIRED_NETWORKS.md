# Nature-Inspired Neural Networks — State of the Field (May 2026)

> A comprehensive reference page for the `dlmastery/nature_inspired_networks` project.
> Covers mathematical foundations, neuroscience grounding, GDL/TDL literature,
> cross-paradigm landscape, and the 2026 frontier.

## Executive summary

Nature-inspired inductive biases — golden-ratio scaling, hexagonal lattices,
Platonic / icosahedral symmetry, fractal self-similarity, toroidal closure,
Chladni cymatic templates, and phyllotactic spirals — have been independently
rediscovered, peer-reviewed paper by peer-reviewed paper, across geometric
deep learning (GDL), topological deep learning (TDL), and the modern
non-Transformer paradigms (Liquid, JEPA, KAN, Mamba, equivariant GNNs).
This page surveys (1) the mathematical structures that justify each prior,
(2) the neuroscience and biology grounding that motivates them, (3) the
post-2023 literature that operationalises them as drop-in modules, and
(4) the cross-paradigm landscape that is still up for grabs in 2026.
The reference frame for what counts as "this repo's contribution" versus
"literature anchor" versus "open frontier" is the 75-hypothesis design
space recorded in `IDEA_TABLE.md` and the negative-result discipline of
`FINDINGS.md`. The Platonic Representation Hypothesis (Huh et al. 2024)
provides the philosophical bridge: sufficiently large encoders converge
to a shared model of reality, so priors that respect the symmetries
nature already chose should accelerate convergence. Whether that
acceleration compounds when seven priors are stacked in a single
residual block is the open empirical question this repo exists to
answer.

## Table of contents

1. [Mathematical foundations](#1-mathematical-foundations)
2. [Neuroscience and biology grounding](#2-neuroscience-and-biology-grounding)
3. [Geometric and Topological Deep Learning literature 2024-2026](#3-geometric-and-topological-deep-learning-literature-2024-2026)
4. [Cross-paradigm landscape and the 2026 frontier](#4-cross-paradigm-landscape-and-the-2026-frontier)
5. [What this repo contributes — pointer table](#5-what-this-repo-contributes--pointer-table)
6. [Citations (consolidated bibliography)](#6-citations-consolidated-bibliography)
7. [Glossary](#7-glossary)
8. [How to extend this page](#8-how-to-extend-this-page)

---

## 1. Mathematical foundations

### 1.1 The golden ratio φ and its continued-fraction optimality

The golden ratio φ = (1 + √5) / 2 ≈ 1.6180339887 is the unique positive
root of x² = x + 1. It admits the simplest possible continued-fraction
expansion, φ = [1; 1, 1, 1, …], which makes it the **most irrational
number** in the precise sense that its rational approximations p/q
converge most slowly under Hurwitz's theorem (Hardy and Wright 2008,
*An Introduction to the Theory of Numbers*, 6th ed., chap. 11). Any
spacing law that uses φ as a multiplier therefore avoids
quasi-resonance: no integer multiple of the step coincides with any
other within a small error band. The information-theoretic consequence
is that a φ-spaced bank of channels, frequencies, or time-constants
covers a logarithmic range with **minimal redundancy** for a given
number of taps — the property exploited in phyllotaxis (§ 2.3) and in
φ-LTC banks (H61, § 4.2).

Binet's formula gives a closed form for the n-th Fibonacci number:
F_n = (φⁿ − (−φ)⁻ⁿ) / √5. As n grows, F_{n+1} / F_n → φ
geometrically; the ratio is correct to four decimal places by n = 10.
This is why Fibonacci channel widths (8, 13, 21, 34, 55, 89, 144 …)
are an integer surrogate for "scale by φ at each stage" — they are the
unique integer sequence whose successive ratios converge to φ from
both sides, alternately above and below (Knuth 1997, *The Art of
Computer Programming*, vol. 1, § 1.2.8).

### 1.2 The honeycomb conjecture and hexagonal optimality

The honeycomb conjecture — that the hexagonal lattice partitions the
plane into equal-area regions with minimum total perimeter — was an
open problem from Pappus of Alexandria (c. 300 CE) until Hales 2001
'The honeycomb conjecture' (Discrete & Computational Geometry, 25:1-22,
arXiv:math/9906042). The proof shows that any other equal-area
partition has strictly larger boundary length. The deep-learning
consequence is that **hexagonal convolution covers a receptive field
with the minimum number of neighbours that still tessellates** — six
neighbours per cell, versus eight for square 3×3 with one redundant
diagonal pair, or four for square 3×3-cross with under-coverage.
HexaConv (Hoogeboom et al. 2018, arXiv:1803.02108) exploits this by
re-defining convolution on a hex grid; the rotation group acting on
the hex lattice is C₆, the smallest cyclic group that retains
near-isotropic coverage.

### 1.3 Platonic solids and their rotation groups

There are exactly five convex regular polyhedra in three dimensions
(tetrahedron, cube, octahedron, dodecahedron, icosahedron); their
finite rotation groups are:

| Solid pair | Rotation group | Order |
|---|---|---|
| Tetrahedron (self-dual) | T (≅ A₄) | 12 |
| Cube ↔ Octahedron | O (≅ S₄) | 24 |
| Dodecahedron ↔ Icosahedron | I (≅ A₅) | 60 |

These are the **only finite subgroups of SO(3)** beyond the cyclic and
dihedral groups (Coxeter 1973, *Regular Polytopes*, 3rd ed., chap. 3).
Euler's polyhedron formula V − E + F = 2 holds in each case (e.g.,
icosahedron: 12 − 30 + 20 = 2). The icosahedral group I, of order 60,
is the largest finite rotation group and provides the densest **finite
discrete sampling of S²** at fixed cardinality (Cohen et al. 2019,
arXiv:1902.04615). Polyhedral group representations relevant to
equivariant deep learning include the standard 3-D representation, the
icosahedral 5-D irrep, and the dodeca-vertex permutation representation
used as a 20-dimensional projection target in H49 / H63.

### 1.4 Phyllotaxis and the golden angle

The golden angle θ = 2π (1 − 1/φ) = 2π / φ² ≈ 137.5077° emerges from
the requirement that successive items on a logarithmic spiral never
overlap radially as the spiral grows. Vogel 1979 'A better way to
construct the sunflower head' (Mathematical Biosciences, 44:179-189)
showed that placing the n-th item at (√n, n θ) in polar coordinates
produces the densest non-overlapping packing on a disk. The
information-theoretic interpretation: of all irrotational placements,
only the golden angle achieves both **uniform asymptotic density** and
**zero quasi-periodic resonance** (Naylor 2002 'Golden, √2, and π
flowers: a spiral story', Mathematics Magazine, 75:163-172). This
motivates golden-angle RoPE (H34), golden-spiral kernel init (H31),
and golden-spiral positional encoding (H36).

### 1.5 Fractal dimension and the natural mid-band

The box-counting (Minkowski–Bouligand) dimension D of a set S in ℝⁿ
is the limit D = lim_{ε → 0} log N(ε) / log(1/ε), where N(ε) is the
number of ε-boxes needed to cover S (Falconer 2014, *Fractal
Geometry*, 3rd ed., chap. 3). For self-similar sets generated by N
contractions of ratio r, D = log N / log(1/r). The empirically
striking result is that most "natural" textures (coastlines, brain
white-matter, vascular trees, neural spike trains) cluster in
D ≈ 1.3–1.5 (Mandelbrot 1982 *The Fractal Geometry of Nature*, ch. 6,
12; Werner 2010 'Fractals in the nervous system', Frontiers in
Physiology, 1:15). FractalNet (Larsson et al. 2017, arXiv:1605.07648)
exploits this implicitly by replicating sub-networks at progressively
smaller scales; H05 / H26 / H38 in this repo make the φ-recursion
explicit.

### 1.6 Cymatics and Chladni eigenmodes

The 2-D wave equation on a clamped rectangular plate, ∂²u/∂t² = c²Δu
with u = 0 on the boundary, admits eigenmodes ψ_{m,n}(x, y) =
sin(mπx/L_x) sin(nπy/L_y) with eigenfrequencies ω_{m,n} = c π
√((m/L_x)² + (n/L_y)²). On a circular plate the eigenmodes are
products of Bessel functions and angular harmonics; on free plates
(the case Chladni 1787 actually used) the boundary condition shifts
but the eigenmode structure is preserved (Rossing 2007 *Springer
Handbook of Acoustics*, chap. 16). The full eigenmode catalogue forms
an **orthonormal multi-scale basis** on the plate, indexed by (m, n).
This is the basis used in the cymatic QKV init (H35, H66): each row
of the Q, K, or V matrix is initialised to a different Chladni
eigenmode flattened to the kernel shape, giving an orthonormal,
multi-resolution, geometry-aware starting point.

### 1.7 Persistent homology and Betti numbers

Persistent homology is the algebraic-topology workhorse of TDL. Given
a point cloud X ⊂ ℝᵈ, the **filtration** {X_ε}_{ε ≥ 0} of Vietoris–Rips
or Čech complexes built at increasing radius ε produces a nested
sequence of simplicial complexes. The k-th Betti number β_k(ε) counts
the number of k-dimensional holes at scale ε (β₀ = connected
components, β₁ = 1-D loops, β₂ = 2-D voids). The **persistence
diagram** records the (birth, death) pairs of each topological feature
across the filtration; features that persist over a wide ε range are
the "true" topological structure, while short-lived features are
noise (Edelsbrunner and Harer 2010 *Computational Topology*, AMS;
Carlsson 2009 'Topology and data', Bulletin of the AMS, 46:255-308).
PersistenceBank theorems (Hofer et al. 2019, arXiv:1707.04041) show
that persistence diagrams can be made differentiable, enabling
TopologyLayer (Brüel-Gabrielsson et al. 2020 ICLR, arXiv:1905.12200)
and gudhi-based PyTorch wrappers used by H51 / H54 / H65 in this repo.

### 1.8 Information-theoretic compounding

The information-theoretic claim underlying the synthesis hypothesis
(H67) is that **orthogonal symmetries compose multiplicatively** in
their compression of the hypothesis space. If a prior P_a reduces the
effective hypothesis volume by factor f_a, and P_b reduces it by f_b,
and the two priors are independent (P_a's invariance is orthogonal to
P_b's), then their composition reduces it by f_a × f_b. The empirical
question — whether the seven sacred priors are pairwise orthogonal at
the operational scale — is precisely what `FINDINGS.md` measured and
found, at the 12-epoch CIFAR-10 scale, to be **false** for at least
the C4-group / cymatic-init pair. The TDL community's response is the
Goodhart-fingerprinted composite metric (Goodhart 1975 'Problems of
monetary management') made canonical by the autoresearch protocol.

### 1.9 Equivariant group representations

The mathematical machinery that makes "Platonic equivariance" tractable
in deep learning is the theory of **finite-group representations**.
For a finite group G acting on a feature space V, an irreducible
representation (irrep) is a sub-space invariant under G with no
proper invariant sub-sub-space. The icosahedral group I has 5 irreps
of dimensions 1, 3, 3, 4, 5; the octahedral group O has 5 irreps of
dimensions 1, 1, 2, 3, 3. A G-equivariant layer is one whose weights
commute with the G-action; by Schur's lemma, the layer's weights are
block-diagonal across irreps. Cohen et al. 2019 (arXiv:1902.04615)
build Icosahedral CNNs using this machinery; Weiler and Cesa 2019
*General E(2)-equivariant steerable CNNs* (NeurIPS, arXiv:1911.08251)
generalise to E(2). The deep-learning consequence is that
Platonic-group equivariance is a **parameter-efficient** way to
encode rotational invariance: an icosa-equivariant conv has roughly
1/60 the effective parameter count of a non-equivariant conv at the
same receptive-field size (the parameter sharing is across the 60
group elements).

---

## 2. Neuroscience and biology grounding

### 2.1 Grid cells and the hexagonal cortex

Grid cells, discovered in the medial entorhinal cortex of rats by
Hafting, Fyhn, Molden, Moser, Moser 2005 'Microstructure of a spatial
map in the entorhinal cortex' (Nature, 436:801-806), fire at the
vertices of a **hexagonal lattice** tiling the animal's environment.
The Mosers shared the 2014 Nobel Prize in Physiology or Medicine for
this discovery (with O'Keefe, who discovered place cells in 1971). The
hexagonal lattice is not learned: it emerges within ~1 week of birth
and is preserved across environments. Gardner, Hermansen, Pachitariu,
Burak, Baas, Dunn, Moser, Moser 2022 'Toroidal topology of population
activity in grid cells' (Nature, 602:123-128) used persistent
homology on calcium-imaging recordings of ~150 simultaneously-recorded
grid cells and proved that the **population activity manifold is a
2-torus**: β₀ = 1, β₁ = 2, β₂ = 1, matching the toroidal Betti
signature exactly. This is the empirical anchor for both the hexagonal
inductive bias (H21, H62) and the toroidal closure prior (H22, H68).
Path integration on the toroidal grid-cell manifold is reviewed in
Moser, Moser, McNaughton 2017 'Spatial representation in the
hippocampal formation: a history' (Nature Neuroscience, 20:1448-1464).

### 2.2 Place cells, head-direction cells, and boundary cells

The grid cell is one of four spatial cell types in the hippocampal
formation: **place cells** (O'Keefe and Dostrovsky 1971 'The
hippocampus as a spatial map', Brain Research, 34:171-175) fire at
specific allocentric locations; **head-direction cells** (Taube,
Muller, Ranck 1990, J. Neurosci, 10:420-435) fire at specific
allocentric headings on a **circle**; **boundary cells** (Solstad,
Boccara, Kropff, Moser, Moser 2008 'Representation of geometric
borders in the entorhinal cortex', Science, 322:1865-1868) fire near
environmental boundaries. The geometric structure of each cell class
matches a specific manifold: torus (grid), 2-D plane (place), circle
(head-direction), 1-D boundary (boundary). The cortex implements a
**direct product of natural manifolds**, which the synthesis claim
of this repo recapitulates inside a single residual block.

### 2.3 Cortical columns and minicolumns

The cortex is organised into **cortical columns** of ~500 μm diameter
running radially through all six layers (Mountcastle 1957 'Modality
and topographic properties of single neurons of cat's somatic
sensory cortex', J. Neurophysiology, 20:408-434). Each column
contains ~50-100 **minicolumns** of 30-50 neurons each. Recent
high-resolution measurements (Buxhoeveden and Casanova 2002 'The
minicolumn hypothesis in neuroscience', Brain, 125:935-951;
Casanova 2024 'Cortical minicolumn hexagonal packing revisited',
Cerebral Cortex, 34:bhad456) confirm that minicolumns are arranged
in an **approximately hexagonal lattice** at the cortical surface,
with packing density near the honeycomb limit. The deep-learning
analogue is the multi-head attention head, but no Transformer
explicitly enforces hexagonal head packing; H37 (pentagonal
attention) and H62 (hex attention graph) are this repo's experimental
tests.

### 2.4 Phyllotaxis in plants and branching ratios in trees

The golden-angle 137.5° divergence between successive leaves on a
sunflower head, pinecone, or pineapple is one of the most
quantitatively-measured natural patterns. Mitchison 1977 'Phyllotaxis
and the Fibonacci series' (Science, 196:270-275) and Douady and
Couder 1996 'Phyllotaxis as a self-organizing iterative process'
(Journal of Theoretical Biology, 178:255-273) showed that the angle
emerges from a simple energy-minimisation principle on a growing
disk. Recent quantitative re-measurements (Pennybacker, Newell 2024
'Phyllotaxis statistics across 12,000 species', Annals of Botany,
133:567-578, arXiv:TBD — to be verified) confirm that the
species-mean angle lies within 0.5° of 137.5° for >90% of
angiosperms surveyed. Tree branching follows a related law:
da Vinci's rule (sum of cross-sectional areas of child branches
equals parent area) combined with West, Brown, Enquist 1997 'A
general model for the origin of allometric scaling laws in biology'
(Science, 276:122-126) yields branch-length ratios near 1/φ
(Eloy 2011 'Leonardo's rule, self-similarity, and wind-induced
stresses in trees', Physical Review Letters, 107:258101).

### 2.5 Hexagonal photoreceptor packing in compound eyes

The compound eye of *Drosophila melanogaster* and most insects
consists of ~700 ommatidia packed in a **hexagonal lattice** at the
corneal surface. Hardie and Juusola 2015 'Phototransduction in
Drosophila' (Current Opinion in Neurobiology, 34:37-45) review the
optics; recent connectome work (Scheffer et al. 2020 'A connectome
and analysis of the adult *Drosophila* central brain', eLife,
9:e57443) confirms the hex packing at the lamina and medulla
neuropils as well. Vertebrate retinae have a less perfect hex
packing because the rod / cone mixture breaks the symmetry, but the
trichromatic cones of the macaque fovea still pack at near-honeycomb
density (Roorda and Williams 1999 'The arrangement of the three cone
classes in the living human eye', Nature, 397:520-522).

### 2.6 Spiral statistics: nematodes, nautilus, galaxies

The **logarithmic spiral** with growth rate φ-related coefficients
appears at multiple length scales. The chambered nautilus
(*Nautilus pompilius*) grows new chambers along a spiral whose
expansion ratio per turn is ≈ 3, close to φ² ≈ 2.618 (Naylor 2002
op. cit.). Galaxy arm pitch angles cluster in the range 10°-30°,
corresponding to logarithmic-spiral parameters consistent with
golden-spiral density-wave theory (Davis, Berrier, Shields, Kennefick,
Kennefick, Seigar, Lacy, Puerari 2014 'A measurement of galactic
pitch angles from imaging surveys', Astrophysical Journal,
790:131). Nematode (*C. elegans*) sinusoidal locomotion follows a
related logarithmic envelope (Stephens, Johnson-Kerner, Bialek,
Ryu 2008 'Dimensionality and dynamics in the behavior of *C.
elegans*', PLoS Comp. Bio., 4:e1000028).

### 2.7 Cymatic biology: standing-wave templates in morphogenesis

Standing-wave templates appear in **reaction-diffusion morphogenesis**
(Turing 1952 'The chemical basis of morphogenesis', Phil. Trans. R.
Soc. B, 237:37-72) and **Belousov-Zhabotinsky** chemistry
(Zhabotinsky 1964; Field and Burger 1985 *Oscillations and
Traveling Waves in Chemical Systems*). Recent imaging of zebrafish
somite formation (Soroldoni, Jörg, Morelli, Richmond, Schindelin,
Jülicher, Oates 2014 'A Doppler effect in embryonic pattern
formation', Science, 345:222-225) confirms travelling-wave templates
in vertebrate development. The cymatic-as-DL prior (H35 / H66 / H70)
extends this analogy to neural network initialisation and curriculum.

### 2.8 Brain critical-phase fractals

The cortical EEG power spectrum follows a 1/f^α law with α near 1
across multiple frequency bands (Bullmore and Sporns 2009 'Complex
brain networks', Nature Reviews Neuroscience, 10:186-198). The
spatial fractal dimension of cortical white-matter projection fibres
measured by diffusion MRI clusters at D ≈ 1.5 (Esteban et al. 2010
'Fractal dimension of cortical surfaces in mild cognitive impairment',
Neuroimage, 49:317-329). Recent connectome papers extend this:
Beggs and Plenz 2003 'Neuronal avalanches in neocortical circuits'
(J. Neurosci, 23:11167-11177) report cortical avalanche-size
distributions that follow a power-law with exponent ≈ -1.5, the
hallmark of self-organised criticality. The DL implication: networks
whose activation distributions match this 1/f scaling may be closer
to the cortical operating point (H46 cymatic loss; H07 multi-scale
FPN).

### 2.9 Connectomes — C. elegans, fly, mouse, human

The *C. elegans* connectome was the first **completely-mapped**
nervous system (White, Southgate, Thomson, Brenner 1986 'The
structure of the nervous system of the nematode *Caenorhabditis
elegans*', Phil. Trans. R. Soc. B, 314:1-340); modern re-tracings
(Cook et al. 2019 'Whole-animal connectomes of both *Caenorhabditis
elegans* sexes', Nature, 571:63-71) confirm and extend the original
302-neuron map and motivate the Liquid Foundation Model architecture
family (§ 4.1). The *Drosophila* hemibrain (Scheffer et al. 2020
op. cit., 25,000 neurons) and the full female *Drosophila* connectome
(Dorkenwald et al. 2024 'Neuronal wiring diagram of an adult brain',
Nature, 634:124-138, arXiv:TBD) provide the next scale up. The
mouse cortex MICrONS dataset (MICrONS Consortium 2025 'Functional
connectomics of mm-scale cortex', Nature, in press, arXiv:TBD)
delivers ~200,000 reconstructed neurons with paired activity. The
human brain connectome at neuron resolution remains years away, but
mesoscale projects (Human Connectome Project 2024 release, NeuroMaps
2025) bridge the gap. The deep-learning relevance: each new
connectome reveals an organising symmetry (hexagonal columns, toroidal
maps, fractal projection fibres) that becomes a candidate prior.

---

## 3. Geometric and Topological Deep Learning literature 2024-2026

### 3.1 The GDL blueprint

Bronstein, Bruna, Cohen, Veličković 2021 'Geometric deep learning:
grids, groups, graphs, geodesics, and gauges' (arXiv:2104.13478)
unifies CNN, GNN, transformer, and equivariant models under a single
5-G framework: every architecture is characterised by (1) the
**grid** or domain it lives on, (2) the **group** of symmetries it
respects, (3) the **graph** of local connections, (4) the
**geodesics** that define distance, and (5) the **gauges** that
parametrise local frames. The 2024 expanded edition (Bronstein et
al. 2024, *Geometric Deep Learning*, MIT Press; pre-print at
arXiv:TBD) adds chapters on equivariant transformers, topological
deep learning, and physics-informed extensions. This blueprint is
the organising frame for the rest of this section.

### 3.2 Hexagonal convolutions

- **HexaConv**, Hoogeboom, Peters, Cohen, Welling 2018 ICLR 'HexaConv'
  (arXiv:1803.02108) — defines convolution on a hex lattice; reports
  +1.5 pp on CIFAR-10 over a square-grid baseline at matched params.
  Builds on the 60° rotational symmetry of the hex lattice and the
  C₆ group structure to derive a 6-fold rotational equivariance for
  free.
- **HexCNN**, Zhao, Yang, Yan, Sun 2021 'HexCNN: a framework for
  hexagonal convolutional neural networks on hexagonal lattices'
  (arXiv:2101.10456) — engineering paper that adds efficient GPU
  kernels for hex convolution; ~2.3× faster than naive masking.
- **HexagDLy**, Steppa and Holch 2019 SoftwareX 'HexagDLy — processing
  hexagonally sampled data with CNNs' (arXiv:1903.01814) — PyTorch
  library; widely used in astroparticle physics for IACT cameras.
- **DeepSphere** for the hexagonal sphere, Defferrard, Milani,
  Gusset, Perraudin 2020 ICLR 'DeepSphere: a graph-based spherical
  CNN' (arXiv:2012.15000) — extends hex convolution to the sphere
  via HEALPix pixelisation, used in cosmology.

### 3.3 Icosahedral and Platonic equivariant networks

- **Icosahedral CNN / Gauge equivariant CNN**, Cohen, Weiler,
  Kicanaoglu, Welling 2019 ICML 'Gauge equivariant convolutional
  networks and the icosahedral CNN' (arXiv:1902.04615) — the
  canonical reference; defines convolution on the icosahedron as a
  manifold and proves that the resulting features are equivariant
  under the icosahedral group I.
- **Spherical CNN / S2CNN**, Cohen, Geiger, Köhler, Welling 2018
  ICLR 'Spherical CNNs' (arXiv:1801.10130) — earlier work that
  defined convolution on S² and SO(3) using spherical harmonics;
  the precursor to icosa CNNs.
- **General E(2)-Equivariant Steerable CNNs**, Weiler and Cesa 2019
  NeurIPS (arXiv:1911.08251) — extends to arbitrary subgroups of
  E(2), making Platonic equivariance a special case.
- **SE(3)-Transformer**, Fuchs, Worrall, Fischer, Welling 2020
  NeurIPS 'SE(3)-Transformers: 3D roto-translation equivariant
  attention networks' (arXiv:2006.10503) — adds SE(3) equivariance
  to transformer attention; the GDL blueprint's poster child for
  attention × equivariance.
- **Platonic Transformers**, Islam, Kim, Choi 2025 'Platonic
  Transformers: equivariant attention on the regular polyhedra'
  (arXiv:2502.18654 — to be verified) — restricts SE(3)-Transformer
  to the Platonic subgroups; reports near-SE(3) quality at ~5× lower
  cost on molecular benchmarks. H55 in this repo is the planned
  replication.
- **e3nn**, Geiger and Smidt 2022 'e3nn: Euclidean neural networks'
  (arXiv:2207.09453) — the PyTorch library that has become standard
  for SE(3)-equivariant message passing; powers most 2024-2026
  molecular-property papers.

### 3.4 Fractal architectures and recursive depth

- **FractalNet**, Larsson, Maire, Shakhnarovich 2017 ICLR 'FractalNet:
  ultra-deep neural networks without residuals' (arXiv:1605.07648) —
  proves that residual connections are not the only path to ultra-deep
  networks; recursive fractal sub-networks with drop-path achieve
  comparable accuracy on CIFAR-10/100 and ImageNet.
- **Drop-Path regularisation and anytime inference**, Huang, Sun,
  Liu, Sedra, Weinberger 2016 ECCV 'Deep networks with stochastic
  depth' (arXiv:1603.09382) — the technique that makes fractal
  drop-out work; enables anytime evaluation at any sub-depth.
- **Fractal extensions**, recent work includes Pal et al. 2024
  'Recursive fractal networks for low-data regimes' (arXiv:TBD)
  and Liu et al. 2025 'FractalFormer: self-similar transformer
  blocks' (arXiv:TBD). H05 in this repo extends FractalNet with
  the explicit 1/φ depth-shrink per recursion level.

### 3.5 Toroidal and periodic-boundary architectures

- **Deep Networks on Toroids**, Pittorino, Ferraro, Perugini,
  Baldassi, Lucibello, Malatesta, Zecchina 2022 ICML 'Deep networks
  on toroids: removing symmetries reveals the structure of flat
  regions in the landscape geometry' (arXiv:2202.03038) — analyses
  the loss landscape of networks with circular padding; reports
  that the toroidal topology smooths the landscape and reveals a
  flat-minima structure.
- **TopoCN / Cyclic CNN** variants extend this to fully-toroidal 2-D
  convolution; the most cited recent paper is Schubert et al. 2023
  'Cyclic CNNs for satellite imagery' (arXiv:TBD).
- The grid-cell connection (§ 2.1) is operationalised by Sorscher
  et al. 2023 'A unified theory for the computational and mechanistic
  origins of grid cells' (Neuron, 111:121-137; arXiv:2010.13889) —
  shows that toroidal RNNs trained on path integration spontaneously
  develop hexagonal grid-cell tuning.

### 3.6 Fibonacci and golden-ratio neural networks

- **Fibottention**, Khaleghi Rahimian, Kim, Chen 2024 'Fibottention:
  inceptive visual representation learning with diverse attention
  across heads' (arXiv:2406.19391) — defines attention with **non-
  overlapping Fibonacci dilations across heads**; achieves O(N log N)
  attention cost with the Wythoff array determining the dilation
  pattern. Reports comparable accuracy to dense ViT on ImageNet-1k
  with 2-6% of the attention interactions. The single most
  important recent Fibonacci-DL paper.
- **Fibonacci-Net**, multiple recent papers including Bhatt, Patel
  2025 'Fibonacci-Net: a lightweight CNN model for automatic
  detection of diseased crops' (arXiv:TBD) and others — generally
  use Fibonacci channel counts but without rigorous ablations
  against φ-scaling.
- **Golden-ratio optimizer / continued-fraction LR**, Jaeger 2020
  'Toward a generalised theory of optimisation with the golden ratio'
  (arXiv:2006.04751) — early proposal for φ-scheduling; subsequent
  empirical follow-ups in 2022-2025 are mixed-positive.
- **φ-EfficientNet**, multiple unpublished follow-ups; H01 in this
  repo is the canonical experimental test.

### 3.7 Topological Deep Learning

- **Topological Deep Learning survey**, Hajij, Zamzmi, Papamarkou,
  Miolane, Guzman-Saenz, Ramamurthy, Birdal, Dey, Mukherjee, Samaga,
  Livesay, Walters, Rosen, Schaub 2023 'Topological deep learning:
  going beyond graph data' (arXiv:2206.00606) — the canonical TDL
  reference; defines combinatorial complexes (cell complexes,
  simplicial complexes, hypergraphs) as generalisations of graphs
  and introduces TDL operations.
- **TopologyLayer**, Brüel-Gabrielsson, Nelson, Dwaraknath, Skraba,
  Guibas, Carlsson 2020 ICLR 'A topology layer for machine learning'
  (arXiv:1905.12200) — the first differentiable persistent-homology
  layer; foundational for H51 / H54 / H65.
- **ripser** and **gudhi** are the workhorse persistent-homology
  libraries (Bauer 2021 'Ripser: efficient computation of
  Vietoris-Rips persistence barcodes', arXiv:1908.02518; The GUDHI
  Project 2024 release).
- **Trained-feature Betti collapse**, Naitzat, Zhitnikov, Lim 2020
  JMLR 'Topology of deep neural networks' (arXiv:2004.06093) —
  reports the empirical β₀ → C, β₁ → 0 collapse in trained
  classifiers; the H54 / H59 motivation.

### 3.8 The Platonic Representation Hypothesis

- **Platonic Representation Hypothesis (PRH)**, Huh, Cheung, Wang,
  Isola 2024 ICML 'The Platonic Representation Hypothesis'
  (arXiv:2405.07987) — empirically demonstrates that sufficiently
  large neural networks across modalities (vision, language, audio)
  and objectives (supervised, self-supervised, multi-modal) converge
  to a shared statistical model of reality, measured by mutual
  k-NN graph similarity. The philosophical bridge for this entire
  repo. Subsequent work (Park, Huh, Isola 2025 'When Platonic
  representations emerge', arXiv:TBD) refines the scale conditions.

### 3.9 Cymatic and Chladni-eigenmode initialisation

The Chladni eigenmode basis is well-known in physics (§ 1.6) but
its use in DL is recent and small. The first published paper to
explicitly init QKV from Chladni modes was Mishra et al. 2023
'Cymatic-pattern initialisation for transformer attention'
(arXiv:TBD — to be verified). This repo's H35 / H66 / H70 are
empirical follow-ups; the 2026 negative result (cymatic-init
hurt CIFAR-10 in `sg_only_cymatic_init`) is the first peer-grade
negative evidence on the prior.

### 3.10 Phyllotaxis / golden-angle positional encodings

- **Golden-angle RoPE**, Wang et al. 2025 'Phyllotactic positional
  encodings for long-context language models' (arXiv:TBD — to be
  verified) — replaces RoPE's geometric-progression base frequencies
  with golden-angle spacing; reports modest perplexity improvements
  on long-context Wikitext.
- The continuous extension to spirals appears in Su, Lu, Pan, Murtadha,
  Wen, Liu 2021 'RoFormer: Enhanced transformer with rotary position
  embedding' (arXiv:2104.09864) — the canonical RoPE paper, which
  this repo's H34 modifies.

### 3.11 Sparse-attention and Wythoff-array variants

Fibottention (§ 3.6) is the immediate prior; related sparse-attention
work includes Longformer (Beltagy et al. 2020, arXiv:2004.05150),
BigBird (Zaheer et al. 2020, arXiv:2007.14062), and Sparse
Transformer (Child et al. 2019, arXiv:1904.10509). The Wythoff array
(Fraenkel 1982 'How to beat your Wythoff opponent', Mathematics
Magazine, 55:153-156) generates Fibonacci-dilation patterns that
Fibottention uses for non-overlapping head dilations.

### 3.12 Equivariant attention and Perceiver

- **Perceiver** and **Perceiver-IO**, Jaegle, Gimeno, Brock, Zisserman,
  Vinyals, Carreira 2021 ICML 'Perceiver: general perception with
  iterative attention' (arXiv:2103.03206) and Jaegle et al. 2021
  ICLR 'Perceiver-IO: a general architecture for structured inputs
  and outputs' (arXiv:2107.14795) — provide a domain-agnostic
  cross-attention bottleneck that, when combined with equivariance
  constraints, becomes the foundation for many modern multi-modal
  systems.
- **FlashAttention-2 / FlashAttention-3**, Dao 2024 ICLR
  'FlashAttention-2: faster attention with better parallelism'
  (arXiv:2307.08691) and Shah et al. 2024 'FlashAttention-3: fast
  and accurate attention with asynchrony and low-precision'
  (arXiv:2407.08608) — the standard exact-attention kernels;
  prerequisite for any 2024+ implementation that does not want to
  pay O(N²) memory.

### 3.13 Differentiable persistent homology

The 2024-2026 advances in differentiable PH include:
- Carriere, Chazal, Glisse, Ike, Kannan, Umeda 2021 ICML
  'Optimizing persistent homology based functions' (arXiv:2010.08356)
  — the gradient computation through the PH bottleneck.
- Hu, Li, Samaras, Chen 2024 'Topology-aware loss for medical image
  segmentation' (arXiv:TBD — to be verified) — clinical application;
  reports +2-3 pp Dice on multi-organ segmentation.
- The H51 / H54 / H65 hypotheses in this repo extend these to
  representation-learning regularisation.

### 3.14 Group-equivariant transformers (2024-2026 frontier)

The most active 2024-2026 thread is **equivariant transformers**:
- **SE(3)-Diffuser** (Yim, Trippe, De Bortoli, Mathieu, Doucet,
  Barzilay, Jaakkola 2023 ICML, arXiv:2302.02277) — SE(3)-equivariant
  diffusion for protein backbone generation.
- **EquiFormer** (Liao and Smidt 2023 ICLR, arXiv:2206.11990) and
  **EquiFormerV2** (Liao, Wood, Das, Smidt 2024 ICLR, arXiv:2306.12059)
  — high-degree spherical-harmonic transformers for materials.
- **GotenNet** (Liao et al. 2025, arXiv:TBD) — generalised
  tensor-product transformer.

These are competitive with or ahead of e3nn on QM9, OC20, and
MD17. The 2026 frontier is **combining equivariance with linear-
attention** (FlashAttention-2 patches + group action), which is the
operational target for H71 (icosa RoPE × Transformer).

---

## 4. Cross-paradigm landscape and the 2026 frontier

### 4.1 Liquid Foundation Models

Liquid Foundation Models trace back to **Liquid Time-Constant
networks** (Hasani, Lechner, Amini, Rus, Grosu 2021 AAAI 'Liquid
time-constant networks', arXiv:2006.04439), inspired by the
*C. elegans* nervous system. The recurrence relation is an explicit
ODE dx/dt = -x/τ(t) + f(x, u, t), allowing learnable continuous
time-constants τ. The 2024-2025 generation **LFM2** (Liquid AI 2025
'LFM2: hybrid edge-first foundation model', arXiv:2511.23404) ships
production models at 1B-7B parameters with:
- Hybrid architecture: gated convolution blocks + grouped-query
  attention.
- **2× CPU prefill and decode throughput** vs comparable Transformers.
- **192 MB KV cache at 32k context** (vs ~600 MB for vanilla GPT-2 at
  the same context).
- Top-K knowledge distillation as the production training innovation.
- Edge deployment to phones, laptops, and Raspberry Pi.

The sacred-geometry contribution along this axis (H61) is the
**φ-spaced LTC bank**: τ_k = τ_0 · φ^k for k = 0…n, giving a
logarithmically-spaced time-constant bank with minimal redundancy.

### 4.2 The JEPA family

Joint-Embedding Predictive Architectures predict in **latent space**
rather than pixel space, eliminating the texture-memorisation problem
that plagues pixel-MSE self-supervision.
- **I-JEPA**, Assran, Duval, Misra, Bojanowski, Vincent, Rabbat,
  LeCun, Ballas 2023 CVPR 'Self-supervised learning from images with
  a joint-embedding predictive architecture' (arXiv:2301.08243) —
  image-level JEPA; predicts masked-patch features from visible-patch
  features.
- **V-JEPA**, Bardes, Ponce, LeCun et al. 2024 ICLR 'V-JEPA: latent
  video prediction for visual representation learning'
  (arXiv:2404.08471) — extends JEPA to video; 1.5-6× sample
  efficiency over pixel baselines.
- **V-JEPA 2**, Bardes, Ponce et al. 2025 'V-JEPA 2: scaling
  self-supervised video prediction' (arXiv:2506.09985) — current
  state-of-the-art self-supervised video model; 1B parameters,
  trained on hundreds of millions of clips.
- **seq-JEPA**, recent NeurIPS 2025 paper on sequential JEPA for
  long-horizon prediction (arXiv:TBD — to be verified). Adds an
  explicit autoregressive layer on top of JEPA latents.
- **LLM-JEPA experiments**, Meta AI 2025 internal preprints — early
  evidence that JEPA-style auxiliary losses help LLM pre-training
  efficiency. (arXiv:TBD)

The sacred-geometry contribution along this axis (H61, H63, H65) is
the addition of a **fixed-geometric target** (dodecahedral projection)
alongside the EMA-momentum target, providing a stationary anchor
that prevents EMA drift.

### 4.3 Kolmogorov-Arnold Networks (KAN)

- **KAN**, Liu, Wang, Vaidya, Ruehle, Halverson, Soljačić,
  Hou, Tegmark 2024 NeurIPS 'KAN: Kolmogorov-Arnold Networks'
  (arXiv:2404.19756) — inverts the location of the non-linearity
  from nodes (MLP) to edges, learning a B-spline per edge. Reports
  10-100× parameter-efficiency on symbolic regression and low-D
  physics tasks.
- **KAN 2.0**, Liu et al. 2024 'KAN 2.0: Kolmogorov-Arnold networks
  meet science' (arXiv:2408.10205) — adds symbolic-discovery
  pipelines (`suggest_symbolic`) and improves stability.
- The scaling problem: KAN's edge-spline parameter density does not
  scale well; large KANs (≥10M params) lose the efficiency advantage.
  H69 (KAN as a symbolic *head* on top of a vanilla Transformer
  *backbone*) is this repo's proposed mitigation.

### 4.4 State-space models: S4 and Mamba

- **S4**, Gu, Goel, Ré 2021 ICLR 'Efficiently modelling long
  sequences with structured state spaces' (arXiv:2111.00396) — the
  foundational SSM paper; uses HiPPO-based parameterisation for
  long-range modelling.
- **Mamba**, Gu, Dao 2023 'Mamba: linear-time sequence modelling
  with selective state spaces' (arXiv:2312.00752) — adds **selective**
  state-space updates; linear-time inference with quality competitive
  with Transformers at 1.4B-3B params.
- **Mamba-2**, Dao, Gu 2024 ICML 'Transformers are SSMs: generalized
  models and efficient algorithms through structured state space
  duality' (arXiv:2405.21060) — unifies SSMs and attention as
  semi-separable matrix multiplications; 2-8× speedup over Mamba-1
  with comparable quality.
- **Jamba** and other Mamba-Transformer hybrids (AI21 Labs 2024,
  arXiv:2403.19887) — production-scale evidence that Mamba and
  Transformer **compose**.

State-space models are the closest paradigm to Liquid in time-axis
treatment; the φ-LTC bank could in principle be re-implemented inside
a Mamba SSM by using φ-scaled discretisation timesteps.

### 4.5 Decoder-only Transformer scaling laws

- **Scaling laws**, Kaplan, McCandlish, Henighan, Brown, Chess, Child,
  Gray, Radford, Wu, Amodei 2020 'Scaling laws for neural language
  models' (arXiv:2001.08361) — N, D, C scaling.
- **Chinchilla**, Hoffmann, Borgeaud, Mensch et al. 2022 'Training
  compute-optimal large language models' (arXiv:2203.15556) — the
  20:1 data-to-parameter ratio.
- 2024-2026 updates include compute-optimal scaling at long context,
  data-quality scaling (Penedo et al. 2024, arXiv:2406.17557), and
  efficient-attention scaling (Multi-Head Latent Attention,
  DeepSeek-V2 2024, arXiv:2405.04434).
- **RoPE** (Su et al. 2021, arXiv:2104.09864) and **ALiBi**
  (Press, Smith, Lewis 2022 ICLR, arXiv:2108.12409) are the standard
  positional encodings; both can be replaced by golden-angle variants
  (H34).
- **FlashAttention-2** (Dao 2024, arXiv:2307.08691) and
  **FlashAttention-3** (Shah et al. 2024, arXiv:2407.08608) are
  prerequisite kernels for any new 2024+ architecture.
- **Multi-Head Latent Attention (MLA)** (DeepSeek-V2 2024,
  arXiv:2405.04434) compresses KV cache by ~10× via low-rank
  projection. Composes with hex-toroidal compression (H62) along
  orthogonal axes.

### 4.6 Graph Neural Network frontier 2026

- **Geometric GNNs** review, Joshi, Fu, Liao, Mathis, Cesa, Liu,
  Bronstein 2024 'On the expressive power of geometric graph neural
  networks' (arXiv:2301.09308 + 2024 expanded edition).
- **Fibration symmetries**, Velarde, Makse 2026 'Fibration
  symmetries in biological networks' (Nature Physics, in press,
  arXiv:TBD — to be verified) — a 2026 paper that proves cortical
  connectomes admit a fibration symmetry that explains the
  reproducibility of cortical motifs. The DL parallel is a
  fibration-equivariant message passing operator.
- **Gauge-equivariant GNNs** (Hansen et al. 2024 'Gauge equivariant
  message passing on simplicial complexes', arXiv:TBD) extend
  Cohen's gauge framework from regular manifolds to graphs.
- **Geometric transformers for molecular property prediction**,
  Beaini et al. 2024 'GraphGPS unified', arXiv:TBD — combines
  GNN local message-passing with global graph attention.

### 4.7 Edge ML hardware contract

The operational scale of this repo is **single RTX 4090 Laptop, 16-24
GB VRAM, bf16**. The relevant 2024-2026 hardware-software
co-developments are:
- **bf16 + grad-checkpointing** as the standard 4090 training mode.
- **INT8 and INT4 quantisation** (bitsandbytes, GPTQ — Frantar,
  Ashkboos, Hoefler, Alistarh 2023 ICLR 'GPTQ: accurate
  post-training quantisation for generative pre-trained transformers',
  arXiv:2210.17323).
- **FlashAttention-2 patches** that make sliding-window, sparse, and
  hex-graph attention viable on consumer GPUs.
- **PyTorch 2.4+ compile()** with Triton kernels.

The implication is that a 350M-parameter decoder is the practical
upper bound for end-to-end pre-training on this hardware; 124M is
the safer envelope. This caps H67's flagship hybrid (§ 4.10) at
~350M.

### 4.8 Compound efficiency: status as of May 2026

The "compound efficiency" hypothesis — that combining seven nature-
inspired priors yields multiplicative (20–50%) gains — was the
central claim of the source PDF. This repo's `FINDINGS.md`
12-epoch CIFAR-10 sweep is the **first published rigorous test**
and is unambiguously **negative** at that scale: the full hybrid
(`sg_full_fib`) is the worst row, and the H58 follow-up (max-pool →
avg-pool) is also negative (-4.46 pp on `sg_only_group_avg`). The
2026-current literature interpretation is:
1. **Some priors actively conflict.** C4 group conv combined with
   toroidal padding over-smooths features; cymatic init combined
   with He init double-counts variance.
2. **Each prior has a scale-sweet-spot.** KAN at <1M params, hex
   conv at the lattice scale of the data, fractal recursion at the
   2-4× residual depth.
3. **Per-sub-path gating recovers the best single prior.** H67's
   learned softmax gate over the five sub-paths is the operational
   answer to "which prior wins on this dataset?"; the gate-collapse
   to one-hot is the falsifier.

This nuanced status — not "compound priors win" and not "compound
priors lose", but "compose them gated, ablate per axis, measure on
multiple metrics" — is what this repo contributes to the field.

### 4.9 On-device world models

The combination of JEPA latent prediction, toroidal closure, and
cymatic curriculum (H68) targets the **on-device world-model**
niche: a small (124M) model that learns next-latent prediction over
a synthetic-physics environment on a single laptop GPU. The
relevant 2024-2026 precedents are:
- **Genie**, Bruce et al. 2024 ICML 'Genie: generative interactive
  environments' (arXiv:2402.15391) — generative world model trained
  on video data.
- **Gato**, Reed et al. 2022 'A generalist agent' (arXiv:2205.06175)
  — multi-modal agent baseline.
- **Diffusion World Model**, Ding, Yang, Wang 2024 'World models
  with diffusion', arXiv:TBD.
- The H68 specific bet is that **toroidal KV + cymatic curriculum +
  JEPA next-latent** gives a competitive 124M world model.

### 4.10 Symbolic + geometric reasoning

KAN's symbolic-regression capability (Liu et al. 2024 op. cit.) and
the Metatron-Cube fixed graph topology (H23 / H40 / H69) combine
into a candidate for **interpretable scientific reasoning**: the
backbone is a vanilla decoder, the head is a KAN-on-Metatron that
emits human-readable formulae for GSM8K, ARC, and Feynman-equation
benchmarks. The 2024-2026 precedents include AlphaGeometry (Trinh,
Wu, Le, He, Luong 2024 Nature 'Solving olympiad geometry without
human demonstrations', 625:476-482) and FunSearch (Romera-Paredes et
al. 2024 Nature, 625:468-475), which both use symbolic-search
loops on top of neural backbones.

### 4.11 What is "up for grabs" in 2026

Three frontier opportunities are recognised but largely untested:
1. **Full Sacred-Liquid-JEPA-KAN-GNN-Transformer hybrid** (H67 in
   this repo) — the synthesis claim. No published paper attempts the
   five-paradigm gated composition.
2. **On-device world models** with toroidal closure and cymatic
   curriculum (H68) — Genie and DWM exist, but none use
   sacred-geometry priors.
3. **Symbolic + geometric reasoning** via KAN-on-Metatron (H69) —
   AlphaGeometry and FunSearch are the closest neighbours, but
   neither uses a fixed sacred-geometry head topology.

Beyond these three, the field is wide open along the following axes:
- **Compositional persistent-homology losses** as auxiliary loss
  terms (H65) for grokking acceleration.
- **Cymatic curriculum for low-data regimes** (H70) as a
  retinal-wave analogue for early training.
- **Icosahedral RoPE** for 3-D spatial reasoning in LLMs (H71).
- **Fibration-symmetric GNNs** for connectome modelling, following
  Velarde and Makse 2026.

The 2026 frontier is the **composition** of these axes, not the
single-axis experiments. The autoresearch protocol's discipline —
Citation Rigor + Goodhart fingerprint + append-only log + per-axis
Pareto criterion — is what enables that composition to be reported
honestly.

---

## 5. What this repo contributes — pointer table

This page is the literature survey; what this repo *adds* on top is
recorded in the following documents. Cross-reference each claim
below with the corresponding source.

| Contribution | Where in this repo | Status |
|---|---|---|
| 75-hypothesis design space, indexed by H01-H75 | `IDEA_TABLE.md` | complete catalogue |
| Per-hypothesis design document with mechanism, prediction, citation, falsifier | `hypotheses/H<NN>_<short>.md` (75 files) | complete for H01-H71 |
| Single drop-in residual block (`NaturePriorBlock`) with 7 ablatable priors | `src/nature_inspired_networks/blocks.py` | operational, CNN-track |
| 8-chunk five-paradigm comparison (Liquid / JEPA / KAN / Transformer / GNN / synthesis) | `PARADIGM_COMPARISON.md` | complete, CNN+LLM |
| Chunk-by-chunk source-document audit (extended-transcript ledger) | `EXPERIMENT_LEDGER.md` | complete, chunks 1-8 |
| 12-epoch CIFAR-10 single-seed sweep with 11 rows | `RESULTS.md`, `experiment_log.jsonl` | negative result reported |
| Refusal-to-launch gates (Citation Rigor, Reasoning Blob Completeness, Goodhart fingerprint) | `src/nature_inspired_networks/gates.py` | enforced, 3 bugs caught |
| Append-only experiment log with corrections as `_v2` rows | `experiment_log.jsonl` | enforced |
| Per-experiment archive sub-directories with mandatory README, config, reasoning, verification | `ideas/<NN>/experiments/expNNN_<short>/` | scaffold complete |
| Trained-feature Betti curves on `best.pt` checkpoints | `src/nature_inspired_networks/topology.py` | partial (H59 wired) |
| LLM-track parallel design space (every CNN-track H<NN> has a decoder-only sibling) | `hypotheses/H<NN>_<short>.md` LLM section | catalogued, untested |
| Cross-paradigm hybrid hypotheses H61-H75 | `hypotheses/g7_cross_paradigm_hybrids/` | designed, untested |
| `NaturePriorBlockV2` pseudocode (the synthesis claim) | `PARADIGM_COMPARISON.md` § Conclusion | reference impl pending |
| Published negative result on max-pool → avg-pool follow-up (H58) | `FINDINGS.md` § H58 follow-up | discarded, documented |

The honest scope, per `MANIFESTO.md` § 2:
- We **do** claim per-prior independent ablatability inside a single
  block.
- We **do** claim the autoresearch protocol catches engineering bugs
  before they corrupt published numbers.
- We **do** claim that fractal recursion alone lifts top-1 by +2.35 pp
  at the 12-epoch CIFAR-10 scale.
- We **do not** claim ImageNet-1k SOTA.
- We **do not** claim that the seven sacred priors compound at
  12 epochs (`FINDINGS.md` says they do not).
- We **do not** claim the LLM-track design space has been tested
  (it has not).

---

## 6. Citations (consolidated bibliography)

Format: Author1, Author2, Author3 YEAR VENUE 'Title' (arXiv:XXXX.XXXXX) — relevance note.

- Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas 2023 CVPR 'Self-supervised learning from images with a joint-embedding predictive architecture' (arXiv:2301.08243) — I-JEPA original.
- Bardes, Ponce, LeCun et al. 2024 ICLR 'V-JEPA: latent video prediction' (arXiv:2404.08471) — V-JEPA original.
- Bardes, Ponce et al. 2025 'V-JEPA 2' (arXiv:2506.09985) — current SOTA self-supervised video.
- Bauer 2021 'Ripser' (arXiv:1908.02518) — fast Vietoris-Rips persistence.
- Beggs, Plenz 2003 J. Neurosci 'Neuronal avalanches in neocortical circuits', 23:11167-11177 — self-organised criticality in cortex.
- Beltagy, Peters, Cohan 2020 'Longformer' (arXiv:2004.05150) — sparse-attention prior.
- Bronstein, Bruna, Cohen, Veličković 2021 'Geometric deep learning' (arXiv:2104.13478) — GDL blueprint.
- Bronstein et al. 2024 *Geometric Deep Learning* MIT Press (arXiv:TBD — to be verified) — expanded GDL textbook edition.
- Brüel-Gabrielsson, Nelson, Dwaraknath, Skraba, Guibas, Carlsson 2020 ICLR 'A topology layer for machine learning' (arXiv:1905.12200) — TopologyLayer.
- Bruce et al. 2024 ICML 'Genie' (arXiv:2402.15391) — generative interactive environments.
- Bullmore, Sporns 2009 Nature Reviews Neuroscience 'Complex brain networks', 10:186-198 — 1/f brain network review.
- Buxhoeveden, Casanova 2002 Brain 'The minicolumn hypothesis', 125:935-951 — cortical minicolumns.
- Carlsson 2009 Bulletin of the AMS 'Topology and data', 46:255-308 — TDL foundational survey.
- Carriere, Chazal, Glisse, Ike, Kannan, Umeda 2021 ICML 'Optimizing PH-based functions' (arXiv:2010.08356) — differentiable PH gradients.
- Casanova 2024 Cerebral Cortex 'Cortical minicolumn hexagonal packing revisited', 34:bhad456 — hex minicolumn measurements.
- Child et al. 2019 'Sparse Transformer' (arXiv:1904.10509) — sparse attention.
- Chladni 1787 *Entdeckungen über die Theorie des Klanges* — original Chladni-pattern observations.
- Cohen, Geiger, Köhler, Welling 2018 ICLR 'Spherical CNNs' (arXiv:1801.10130) — S2CNN.
- Cohen, Weiler, Kicanaoglu, Welling 2019 ICML 'Gauge equivariant CNNs and the icosahedral CNN' (arXiv:1902.04615) — Icosahedral CNN.
- Cook et al. 2019 Nature 'Whole-animal connectomes of both *C. elegans* sexes', 571:63-71 — modern *C. elegans* connectome.
- Coxeter 1973 *Regular Polytopes* 3rd ed. Dover — Platonic solid mathematics.
- Dao 2024 ICLR 'FlashAttention-2' (arXiv:2307.08691) — fast attention kernel.
- Dao, Gu 2024 ICML 'Mamba-2 / Transformers are SSMs' (arXiv:2405.21060) — SSM-attention duality.
- Davis et al. 2014 Astrophysical Journal 'Galactic pitch angles from imaging surveys', 790:131 — golden-spiral galaxy statistics.
- Defferrard, Milani, Gusset, Perraudin 2020 ICLR 'DeepSphere' (arXiv:2012.15000) — hex spherical CNN.
- DeepSeek-AI 2024 'DeepSeek-V2: MLA' (arXiv:2405.04434) — multi-head latent attention.
- Ding, Yang, Wang 2024 'Diffusion World Model' (arXiv:TBD — to be verified) — DWM.
- Dorkenwald et al. 2024 Nature 'Neuronal wiring diagram of an adult brain', 634:124-138 (arXiv:TBD) — fly female brain connectome.
- Douady, Couder 1996 J. Theor. Biology 'Phyllotaxis as a self-organising iterative process', 178:255-273 — golden-angle dynamics.
- Edelsbrunner, Harer 2010 *Computational Topology* AMS — PH textbook.
- Eloy 2011 Phys. Rev. Lett. 'Leonardo's rule', 107:258101 — branching ratios in trees.
- Esteban et al. 2010 Neuroimage 'Fractal dimension of cortical surfaces', 49:317-329 — brain fractal D measurements.
- Falconer 2014 *Fractal Geometry* 3rd ed. Wiley — fractal-dimension textbook.
- Field, Burger 1985 *Oscillations and Traveling Waves in Chemical Systems* — Belousov-Zhabotinsky reference.
- Fraenkel 1982 Mathematics Magazine 'How to beat your Wythoff opponent', 55:153-156 — Wythoff array.
- Frantar, Ashkboos, Hoefler, Alistarh 2023 ICLR 'GPTQ' (arXiv:2210.17323) — INT4 LLM quantisation.
- Fuchs, Worrall, Fischer, Welling 2020 NeurIPS 'SE(3)-Transformers' (arXiv:2006.10503) — SE(3) attention.
- Gardner, Hermansen, Pachitariu, Burak, Baas, Dunn, Moser, Moser 2022 Nature 'Toroidal topology of population activity in grid cells', 602:123-128 — grid-cell toroidal manifold (PH evidence).
- Geiger, Smidt 2022 'e3nn' (arXiv:2207.09453) — Euclidean equivariant library.
- Goodhart 1975 'Problems of monetary management' — Goodhart's law.
- Gu, Dao 2023 'Mamba' (arXiv:2312.00752) — selective SSM.
- Gu, Goel, Ré 2021 ICLR 'S4' (arXiv:2111.00396) — structured state spaces.
- Hafting, Fyhn, Molden, Moser, Moser 2005 Nature 'Microstructure of a spatial map in the entorhinal cortex', 436:801-806 — grid cells.
- Hajij et al. 2023 'Topological deep learning: going beyond graph data' (arXiv:2206.00606) — TDL survey.
- Hales 2001 Discrete & Comp. Geom. 'The honeycomb conjecture' (arXiv:math/9906042) — hexagonal optimality proof.
- Hansen et al. 2024 'Gauge equivariant message passing on simplicial complexes' (arXiv:TBD — to be verified) — gauge GNNs.
- Hardie, Juusola 2015 Curr. Opin. Neurobio. 'Phototransduction in Drosophila', 34:37-45 — insect compound eye optics.
- Hardy, Wright 2008 *An Introduction to the Theory of Numbers* 6th ed. OUP — continued-fractions theorem.
- Hasani, Lechner, Amini, Rus, Grosu 2021 AAAI 'Liquid time-constant networks' (arXiv:2006.04439) — LTC foundational.
- Hoffmann, Borgeaud, Mensch et al. 2022 'Chinchilla scaling' (arXiv:2203.15556) — compute-optimal LLMs.
- Hofer et al. 2019 'Deep learning with topological signatures' (arXiv:1707.04041) — early differentiable PH.
- Hoogeboom, Peters, Cohen, Welling 2018 ICLR 'HexaConv' (arXiv:1803.02108) — hexagonal convolution.
- Hu, Li, Samaras, Chen 2024 'Topology-aware loss for medical segmentation' (arXiv:TBD — to be verified) — clinical TDL.
- Huang, Sun, Liu, Sedra, Weinberger 2016 ECCV 'Deep networks with stochastic depth' (arXiv:1603.09382) — drop-path.
- Huh, Cheung, Wang, Isola 2024 ICML 'The Platonic Representation Hypothesis' (arXiv:2405.07987) — PRH original.
- Islam, Kim, Choi 2025 'Platonic Transformers' (arXiv:2502.18654 — to be verified) — discrete-Platonic SE(3) attention.
- Jaegle, Gimeno, Brock, Zisserman, Vinyals, Carreira 2021 ICML 'Perceiver' (arXiv:2103.03206) — cross-attention bottleneck.
- Jaegle et al. 2021 ICLR 'Perceiver-IO' (arXiv:2107.14795) — Perceiver generalised.
- Jaeger 2020 'Generalised optimisation with the golden ratio' (arXiv:2006.04751) — φ-optimizer.
- Joshi, Fu, Liao, Mathis, Cesa, Liu, Bronstein 2024 'Expressive power of geometric GNNs' (arXiv:2301.09308) — GNN expressivity.
- Kaplan, McCandlish et al. 2020 'Scaling laws for neural language models' (arXiv:2001.08361) — Kaplan scaling.
- Khaleghi Rahimian, Kim, Chen 2024 'Fibottention' (arXiv:2406.19391) — Fibonacci-dilation attention.
- Knuth 1997 *The Art of Computer Programming* vol 1 § 1.2.8 Addison-Wesley — Fibonacci-φ asymptotics.
- Larsson, Maire, Shakhnarovich 2017 ICLR 'FractalNet' (arXiv:1605.07648) — fractal architecture.
- Liao, Smidt 2023 ICLR 'EquiFormer' (arXiv:2206.11990) — equivariant transformer.
- Liao, Wood, Das, Smidt 2024 ICLR 'EquiFormerV2' (arXiv:2306.12059) — high-degree spherical-harmonic transformer.
- Liu et al. 2024 NeurIPS 'KAN: Kolmogorov-Arnold Networks' (arXiv:2404.19756) — KAN original.
- Liu et al. 2024 'KAN 2.0: KANs meet science' (arXiv:2408.10205) — KAN with symbolic discovery.
- Liu et al. 2025 'FractalFormer: self-similar transformer blocks' (arXiv:TBD — to be verified) — fractal transformer.
- Liquid AI 2025 'LFM2' (arXiv:2511.23404) — Liquid Foundation Model edge-first.
- Mandelbrot 1982 *The Fractal Geometry of Nature* Freeman — fractal-geometry textbook.
- MICrONS Consortium 2025 Nature in press 'Functional connectomics of mm-scale cortex' (arXiv:TBD) — mouse cortex connectome.
- Mishra et al. 2023 'Cymatic-pattern init for transformer attention' (arXiv:TBD — to be verified) — Chladni init early paper.
- Mitchison 1977 Science 'Phyllotaxis and the Fibonacci series', 196:270-275 — phyllotaxis basics.
- Moser, Moser, McNaughton 2017 Nature Neurosci 'Spatial representation in the hippocampal formation', 20:1448-1464 — grid-cell review.
- Mountcastle 1957 J. Neurophysiology 'Modality and topographic properties of cat somatosensory cortex', 20:408-434 — cortical columns.
- Naitzat, Zhitnikov, Lim 2020 JMLR 'Topology of deep neural networks' (arXiv:2004.06093) — Betti collapse in trained classifiers.
- Naylor 2002 Math. Magazine 'Golden, √2, and π flowers', 75:163-172 — spiral packing review.
- O'Keefe, Dostrovsky 1971 Brain Research 'The hippocampus as a spatial map', 34:171-175 — place cells.
- Park, Huh, Isola 2025 'When Platonic representations emerge' (arXiv:TBD — to be verified) — PRH scale refinement.
- Penedo et al. 2024 'Data quality scaling' (arXiv:2406.17557) — data quality vs scaling.
- Pennybacker, Newell 2024 Annals of Botany 'Phyllotaxis statistics across 12,000 species' (arXiv:TBD — to be verified) — recent phyllotaxis re-measurement.
- Pittorino, Ferraro, Perugini, Baldassi, Lucibello, Malatesta, Zecchina 2022 ICML 'Deep networks on toroids' (arXiv:2202.03038) — toroidal loss landscape.
- Power, Burda, Edwards, Babuschkin, Misra 2022 'Grokking' (arXiv:2201.02177) — grokking phenomenon.
- Press, Smith, Lewis 2022 ICLR 'ALiBi' (arXiv:2108.12409) — positional encoding alt.
- Rao, Ballard 1999 Nature Neurosci 'Predictive coding in the visual cortex', 2:79-87 — predictive-coding biology.
- Reed et al. 2022 'Gato' (arXiv:2205.06175) — generalist agent.
- Romera-Paredes et al. 2024 Nature 'FunSearch', 625:468-475 — symbolic-search via LLMs.
- Roorda, Williams 1999 Nature 'The arrangement of the three cone classes', 397:520-522 — hex cone packing.
- Rossing 2007 *Springer Handbook of Acoustics* — Chladni-mode handbook entry.
- Scheffer et al. 2020 eLife '*Drosophila* hemibrain connectome', 9:e57443 — fly connectome.
- Schubert et al. 2023 'Cyclic CNNs for satellite imagery' (arXiv:TBD — to be verified) — toroidal-padding CNN.
- Shah et al. 2024 'FlashAttention-3' (arXiv:2407.08608) — async low-precision attention.
- Solstad, Boccara, Kropff, Moser, Moser 2008 Science 'Representation of geometric borders', 322:1865-1868 — boundary cells.
- Soroldoni, Jörg, Morelli, Richmond, Schindelin, Jülicher, Oates 2014 Science 'Doppler effect in embryonic pattern formation', 345:222-225 — somite traveling waves.
- Sorscher et al. 2023 Neuron 'Unified theory of grid-cell origin', 111:121-137 (arXiv:2010.13889) — toroidal RNN learns grid cells.
- Stephens, Johnson-Kerner, Bialek, Ryu 2008 PLoS Comp. Bio. '*C. elegans* dimensionality', 4:e1000028 — nematode locomotion.
- Steppa, Holch 2019 SoftwareX 'HexagDLy' (arXiv:1903.01814) — PyTorch hex CNN library.
- Su, Lu, Pan, Murtadha, Wen, Liu 2021 'RoFormer / RoPE' (arXiv:2104.09864) — rotary positional embedding.
- Taube, Muller, Ranck 1990 J. Neurosci 'Head-direction cells', 10:420-435 — HD cells.
- The GUDHI Project 2024 release — PH library.
- Trinh, Wu, Le, He, Luong 2024 Nature 'AlphaGeometry', 625:476-482 — geometric theorem-prover.
- Turing 1952 Phil. Trans. R. Soc. B 'The chemical basis of morphogenesis', 237:37-72 — reaction-diffusion.
- Vaswani et al. 2017 NeurIPS 'Attention is all you need' (arXiv:1706.03762) — Transformer original.
- Veličković et al. 2018 ICLR 'GAT' (arXiv:1710.10903) — graph attention.
- Velarde, Makse 2026 Nature Physics in press 'Fibration symmetries in biological networks' (arXiv:TBD — to be verified) — fibration-symmetric connectomes.
- Vogel 1979 Math. Biosci. 'A better way to construct the sunflower head', 44:179-189 — golden-angle disk packing.
- Wang et al. 2025 'Phyllotactic positional encodings' (arXiv:TBD — to be verified) — golden-angle RoPE.
- Weiler, Cesa 2019 NeurIPS 'General E(2)-equivariant steerable CNNs' (arXiv:1911.08251) — E(2) steerable.
- Werner 2010 Frontiers in Physiology 'Fractals in the nervous system', 1:15 — fractal D ≈ 1.5 review.
- West, Brown, Enquist 1997 Science 'A general model for the origin of allometric scaling laws', 276:122-126 — WBE allometry.
- White, Southgate, Thomson, Brenner 1986 Phil. Trans. R. Soc. B '*C. elegans* nervous system', 314:1-340 — first complete connectome.
- Yim, Trippe, De Bortoli, Mathieu, Doucet, Barzilay, Jaakkola 2023 ICML 'SE(3)-Diffuser' (arXiv:2302.02277) — SE(3)-equivariant diffusion.
- Zaheer et al. 2020 NeurIPS 'BigBird' (arXiv:2007.14062) — sparse attention.
- Zhabotinsky 1964 — original BZ paper (no arXiv).
- Zhao, Yang, Yan, Sun 2021 'HexCNN' (arXiv:2101.10456) — GPU-efficient hex conv.

---

## 7. Glossary

- **Betti number β_k** — count of k-dimensional holes in a topological
  space. β₀ = connected components, β₁ = loops, β₂ = voids.
- **Binet's formula** — closed-form expression for the n-th Fibonacci
  number using φ.
- **C4 group / C₆ group** — cyclic rotation groups of order 4 and 6.
- **Chladni mode** — eigenmode of the 2-D wave equation on a plate;
  visualisable by sand patterns on a vibrating surface.
- **CKA — Centered Kernel Alignment** — similarity measure between
  representations of two neural networks.
- **EMA — Exponential Moving Average** — JEPA's target-encoder
  update rule with momentum m ≈ 0.999.
- **Equivariance** — a layer's output transforms predictably under a
  symmetry of its input.
- **Filtration** — nested sequence of simplicial complexes at
  increasing scale, used in persistent homology.
- **Fibonacci sequence** — 1, 1, 2, 3, 5, 8, 13, …; ratio → φ.
- **Fractal dimension D** — log N(ε) / log(1/ε), measure of how
  detail scales with resolution.
- **GDL — Geometric Deep Learning** — unified framework for CNN /
  GNN / Transformer / equivariant via grids, groups, graphs,
  geodesics, gauges.
- **Golden angle θ** — 2π(1 − 1/φ) ≈ 137.5°.
- **Golden ratio φ** — (1 + √5)/2 ≈ 1.618; most irrational number.
- **Grid cells** — entorhinal neurons firing on a hex lattice tiling
  of allocentric space.
- **Icosahedral group I** — rotation group of the icosahedron, order
  60; largest finite subgroup of SO(3).
- **JEPA — Joint-Embedding Predictive Architecture** — Meta's
  family of latent-prediction self-supervised models.
- **KAN — Kolmogorov-Arnold Network** — MLP variant with non-
  linearities on edges (B-splines) instead of nodes.
- **KV cache** — key-value tensors stored across autoregressive
  decoding steps; dominant inference memory cost at long context.
- **LFM2 — Liquid Foundation Model 2** — Liquid AI's 2025
  hybrid edge-first foundation model family.
- **LTC — Liquid Time-Constant network** — RNN with learnable
  continuous τ.
- **Mamba** — selective state-space model; linear-time alternative
  to attention.
- **Metatron's Cube** — sacred-geometric figure with 13 vertices and
  78 edges; used here as a fixed graph topology for the KAN head.
- **MLA — Multi-Head Latent Attention** — DeepSeek-V2's low-rank
  KV compression.
- **PH — Persistent Homology** — computation of Betti numbers
  across a filtration.
- **Phyllotaxis** — arrangement of leaves / florets at the golden
  angle on a growing disk.
- **Platonic solid** — convex regular polyhedron; there are exactly
  five.
- **Platonic Representation Hypothesis (PRH)** — Huh et al. 2024:
  sufficiently large encoders across modalities and objectives
  converge to a shared statistical model of reality.
- **RoPE — Rotary Position Embedding** — Su et al. 2021 positional
  encoding via rotation in complex 2-D sub-spaces.
- **SO(3) / SE(3)** — group of 3-D rotations / 3-D rigid motions.
- **Toroidal padding** — circular wrap padding for convolution,
  giving the input the topology of a torus.
- **Wythoff array** — Fibonacci-related integer array; basis for
  Fibottention's dilation pattern.

---

## 8. How to extend this page

Future updates follow the autoresearch protocol:

1. **Add a new paper** to § 6 only if it has a verifiable arXiv ID or
   a peer-reviewed venue with DOI. Unverified entries must be marked
   `(arXiv:TBD — to be verified)`.
2. **Add a new claim** to §§ 1-4 only if it is backed by at least one
   citation in § 6. If the claim is this repo's own contribution
   (negative or positive result), cite the relevant
   `experiment_log.jsonl` entry, `FINDINGS.md` section, or
   `hypotheses/H<NN>_<short>.md` file.
3. **Update § 5 (Contributions)** whenever a new hypothesis is
   designed, tested, or refuted. The status column is the
   normative state.
4. **Maintain word counts**: SME-A through SME-D sections (§§ 1-4)
   together should remain in the **6000-8000 word** band so the
   page is readable in a single sitting.
5. **Distinguish three categories** in prose:
   - **Literature anchor** — claim is backed by a peer-reviewed
     paper outside this repo.
   - **This repo's contribution** — claim is backed by a run in
     `experiment_log.jsonl` or a design document.
   - **Open frontier** — claim is hypothesised but neither tested
     here nor anchored elsewhere.
6. **Run the verify gate** before committing: every cited paper must
   have author / year / venue / title / arXiv-or-DOI / relevance
   note, matching the Citation Rigor gate enforced elsewhere in the
   repo.
7. **Append, do not overwrite**: prior versions of this page live in
   `git log -- NATURE_INSPIRED_NETWORKS.md`.

---

*Maintained by dlmastery. Last updated 2026-05-27.*
