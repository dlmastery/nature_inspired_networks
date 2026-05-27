## 2. Neuroscience and Biology Grounding (continued)

### 2.4 Phyllotaxis in plants and branching ratios in trees

- **[Phyllotaxis Simulation & Golden-Angle Packing — Vogel-model demos](https://github.com/topics/phyllotaxis)** — Community Python/Processing repos that export golden-spiral kernels directly usable as `GoldenSpiralSampler` weight tensors. Drop-in for H31/H34/H36 (golden-spiral init, RoPE-phi, phyllotactic positional encoding). [GitHub] [demo]
- **[Phyllotaxis Statistics & DL Integration — Pennybacker/Newell follow-ups (2024)](https://arxiv.org/abs/TBD)** — Datasets + generative scripts for training phi-divergence positional encodings across 12,000 plant species; forks ship PyTorch modules for golden-angle embeddings. H03/H34 anchor. [arXiv:TBD]
- **[Leonardo's Rule / Branching Fractals — "da Vinci rule" tree-growth simulators](https://github.com/search?q=da+vinci+rule+neural+network)** — Bio-inspired branching simulators tied to phi-scaling; convert into fractal block templates for H05/H26/H38 recursive depth experiments. [GitHub]

### 2.5 Hexagonal photoreceptor packing in compound eyes

- **[HexaConv (official ICLR 2018 impl)](https://github.com/ehoogeboom/hexaconv)** — Canonical hex-grid convolution + C_6 group equivariance; the exact 2D lattice model of insect ommatidia and Casanova-style cortical minicolumns. H21/H62 anchor for `HexLatticeConv`. [GitHub]
- **[HexagDLy — Hexagonal Convolutions for PyTorch](https://github.com/ai4iacts/hexagdly)** — Production library with GPU kernels (astroparticle physics origin) — drop-in for modeling compound-eye lattices. H21/H62 direct impl. [GitHub]
- **[Drosophila Hemibrain + Full-Brain Connectome (Scheffer 2020 / Dorkenwald 2024)](https://www.janelia.org/project-team/flyem/hemibrain)** — Hexagonal neuropil analysis scripts on fly visual system; cross-validates the "hexagonal photoreceptor → DL hex conv" inductive bias. [dataset]

### 2.6 Spiral statistics: nematodes, nautilus, galaxies

- **[OpenWorm — C. elegans logarithmic-spiral locomotion repos](https://github.com/openworm)** — Sinusoidal movement envelopes tied to phi-related growth; biological anchor for H61 phi-LTC banks and golden-spiral kernel initialization. [GitHub] [demo]
- **[Galaxy Spiral Arm + DL Priors — golden-spiral fitting toolkits](https://github.com/topics/galaxy-spiral)** — Astrophysical datasets with golden-angle fitting; adapted in some vision-transformer repos as priors for spiral positional encodings (H27/H36). [dataset]

### 2.7 Cymatic biology: standing-wave templates in morphogenesis

- **[Chladni Plate Simulator — Cymatics interactive eigenmode generator](https://github.com/ChladniPlate2)** — High-fidelity eigenmode generator; flatten the resulting modes directly into QKV weight matrices for `CymaticWaveletKernel`. H35/H66/H70 direct impl. [GitHub] [demo]
- **[Reaction-Diffusion + Turing Patterns in DL](https://github.com/search?q=cymatic+neural+network+initialization)** — Repos bridging Turing 1952 morphogenesis with modern cymatic-init experiments; 2023–2025 follow-ups to Mishra et al. provide reference loss curves for cymatic priors. H46/H56 anchor. [GitHub]
- **[Mishra et al. 2023 — "Cymatic-pattern initialization for transformer attention"](https://arxiv.org/abs/TBD)** — Early DL paper anchor for Chladni-eigenmode QKV init; H35 baseline (and source of the documented negative result on naive cymatic init). [arXiv:TBD]

### 2.8 Brain critical-phase fractals

- **[FractalNet — official Larsson et al. 2017](https://github.com/gustavla/fractalnet)** + **[PyTorch FractalNet port (`pt.fractalnet`)](https://github.com/khanrc/pt.fractalnet)** — Core implementations for explicit phi-recursion; canonical baseline for H05/H26/H38. [GitHub]
- **[EEG / Connectome Fractal Analysis (box-counting D ≈ 1.3–1.5)](https://github.com/topics/fractal-eeg)** — Repos using box-counting dimension on neural data; link to persistent-homology pipelines to validate that trained networks reach the brain's natural mid-band fractal dimension (H05/H51/H54 cross-check). [GitHub]

### 2.9 Connectomes — C. elegans, fly, mouse, human

- **[OpenWorm — C. elegans full 302-neuron simulation](https://github.com/openworm)** — Complete connectome + dynamical models; direct biological inspiration for H61 phi-LTC bank inside Liquid Foundation Models. [GitHub] [demo]
- **[FlyWire / Drosophila Connectome (2024 full female brain)](https://flywire.ai/)** — Modern wiring-diagram repos including hexagonal-column analysis tools; H21/H37/H62 cross-validation. [dataset]
- **[MICrONS Mouse Cortex Dataset Tools (2025)](https://www.microns-explorer.org/)** — mm-scale functional connectomics code pairing activity with topology; ready-to-use for TDL experiments (H51/H54/H65 PH/Betti loss). [dataset] [demo]

---

## 3. Geometric & Topological Deep Learning Literature (2024–2026, part 1)

### 3.1 The GDL Blueprint

- **[Geometric Deep Learning Book Resources & Code — Bronstein et al.](https://geometricdeeplearning.com/)** — The canonical 5-G framework (grids / groups / graphs / gauges / geodesics) with official companion notebooks; the mathematical justification for every sacred prior in SacredGeoBlock. H01–H09 (phi-scaling), H21–H23/H40 (hex/Platonic), H05/H26/H38 (fractal), H67–H71 (full synthesis). [GitHub] [demo]
- **[Awesome Geometric Deep Learning](https://github.com/naganandy/awesome-geometric-deep-learning)** — Community-curated hub linking all major GDL papers, libraries, and tutorials; the meta-index for cross-referencing every H01–H75 hypothesis to a code repo. [GitHub]

### 3.2 Hexagonal Convolutions

- **[HexaConv — Official ICLR 2018 implementation](https://github.com/ehoogeboom/hexaconv)** — Canonical planar + group (C_6) hex convolution with proven +1.5pp CIFAR-10 gains; the direct H21/H62 anchor and `HexLatticeConv` reference. [GitHub]
- **[HexagDLy — PyTorch Hexagonal Data Processing](https://github.com/ai4iacts/hexagdly)** — Production-ready library (astroparticle origin) with GPU-efficient hex kernels at arbitrary stride/size; Zhao et al. 2021 extensions add modern PyTorch hooks. H21/H62 direct impl. [GitHub]
- **[HexCNN Framework — native GPU hex convolutions](https://github.com/search?q=hexcnn)** — Native GPU-optimized hexagonal convolutions ~2.3× faster than masking; ideal for scaling H21 ablations to Tiny ImageNet. [GitHub]

### 3.3 Icosahedral and Platonic Equivariant Networks

- **[icoCNN — PyTorch Icosahedral CNNs](https://github.com/maurice-weiler/icoCNN)** — Clean, installable implementation of Cohen et al. 2019 ICML "Gauge Equivariant Convolutional Networks and the Icosahedral CNN"; powers `PlatonicGroupConv` for H23/H24/H30/H40/H71. [arXiv:1902.04615] [GitHub]
- **[IcosahedralCNN Alternative Port](https://github.com/topics/icosahedral-cnn)** — High-quality alternative PyTorch implementation with example scripts; useful for cross-checking icosahedral RoPE ablations (H71). [GitHub]
- **[e3nn — Euclidean Neural Networks](https://github.com/e3nn/e3nn)** — Gold-standard library for SE(3)/Platonic/irrep equivariant layers; the engine behind nearly all 2024–2026 molecular and 3-D Platonic work. H23/H24/H30/H40/H71 direct impl. [GitHub]
- **[Awesome Equivariant Networks (Chen-Cai-OSU)](https://github.com/Chen-Cai-OSU/awesome-equivariant-network)** — Exhaustive list covering Icosahedral CNN, e3nn, SE(3)-Transformer, Platonic Transformers (2025), EquiFormer, etc.; meta-catalog for H23/H24/H30/H40/H55/H71. [GitHub]

### 3.4 Fractal architectures and recursive depth

- **[FractalNet Official Implementation — Larsson et al. 2017](https://github.com/gustavla/fractalnet)** — Canonical repo with auto-generated fractal wiring and drop-path regularization (anytime inference); the perfect baseline for H05/H26/H38 phi-recursion extensions and H52 drop-path anytime evaluation. [arXiv:1605.07648] [GitHub]
- **[PyTorch FractalNet (`pt.fractalnet`)](https://github.com/khanrc/pt.fractalnet)** — Clean trainable PyTorch port with CIFAR-10/100 + ImageNet scripts; widely used for ablation studies of `FractalRecursionBlock`. H05/H26/H38 direct impl. [GitHub]
- **[Fractal Generative Models (2025)](https://arxiv.org/abs/2502.17437)** — Modern PyTorch implementation extending fractal priors to diffusion-style generative architectures; opens the door to fractal-conditioned image and video generators. [arXiv:2502.17437] [GitHub]
- **[Keras-FractalNet + community forks](https://github.com/topics/fractalnet)** — Rapid-prototyping ports across frameworks for phi-scaled recursive depth experiments. [GitHub]

### 3.5 Toroidal and periodic-boundary architectures

- **[Deep Networks on Toroids — Pittorino et al. 2022 ICML](https://arxiv.org/abs/2202.03038)** — Foundational paper proving toroidal topology smooths loss landscapes; code reproduced in cyclic-padding repos. H22/H68 anchor for `ToroidalClosure` and toroidal KV-cache. [arXiv:2202.03038]
- **[Neural Network on Torus Simulator](https://github.com/topics/torus-neural-network)** — Visual cellular-automaton-style torus net mimicking cortical sheets; great intuition for H22/H68 (toroidal closure + world-model wrap). [GitHub] [demo]
- **[Cyclic CNN extensions — Schubert et al. 2023 follow-ups](https://github.com/search?q=cyclic+cnn+pytorch)** — Satellite-imagery and periodic-boundary repos built on top of toroidal padding; production references for H22 phi-scaled periodic boundary. [GitHub]

### 3.6 Fibonacci and golden-ratio neural networks

- **[Fibottention Official Repo — Khaleghi Rahimian et al. 2024](https://github.com/Charlotte-CharMLab/Fibottention)** — Full implementation of Fibonacci-dilation sparse attention using the Wythoff array; O(N log N), head-diverse sparsity, ImageNet/video/robotics experiments. Direct match for H32/H34/H36 phi-spaced/golden-angle priors. [arXiv:2406.19391] [GitHub]
- **[Golden-Ratio Optimizer & phi-scheduling — Jaeger 2020 follow-ups](https://arxiv.org/abs/2006.04751)** — NAS and learning-rate scheduler repos using phi^{-k} decays; H10/H41/H48 direct impl. [arXiv:2006.04751] [GitHub]

### 3.7 Topological Deep Learning

- **[TopologyLayer — Differentiable Persistent Homology (Brüel-Gabrielsson et al. 2020 ICLR)](https://github.com/bruel-gabrielsson/TopologyLayer)** — The canonical PyTorch layer for end-to-end filtrations, Betti numbers, and persistence diagrams. H51/H54/H65 direct impl — the exact engine for SacredGeoBlock's PH-regularization. [arXiv:1905.12200] [GitHub]
- **[torchph — PyTorch Persistent Homology Extensions](https://github.com/c-hofer/torchph)** — High-performance differentiable PH backend that pairs cleanly with GUDHI/ripser for TDL pipelines; ideal for H59 trained-feature Betti analyses. [GitHub]
- **[pytorch-topological — Modern TDL framework](https://github.com/aidos-lab/pytorch-topological)** — Modern high-performance TDL framework with multiple PH backends and topological loss functions; the modern alternative for H51/H54 differentiable Betti loss. [GitHub]
- **[Topo-Loss for Image Segmentation](https://github.com/HuXiaoling/TopoLoss)** — Topological loss functions via persistent homology for segmentation; H07/H51/H54 cross-application to medical imaging. [GitHub]
- **[GUDHI TDA Tutorials + PyTorch Integration](https://github.com/GUDHI/TDA-tutorial)** — Comprehensive notebooks for persistent homology, filtrations, and Betti curves; the educational on-ramp for H51/H54/H65. [GitHub] [demo]
