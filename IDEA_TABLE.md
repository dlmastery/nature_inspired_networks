# IDEA_TABLE — Sacred-Geometry × Deep-Learning design space

> Incremental master table built by chunk-reading four source files in
> `C:\Users\evija\Downloads`:
>
> - `sacred geometry and neural networks.pdf` — the original PDF
> - `sacgeometry.md` — Gemini-style synthesis report
> - `grok-sacred-geometry-deep-learning-transcript.md` — Grok conversation transcript
> - `sacred-geometry-deep-learning-conversation.md` — cleaned conversation
>
> Every distinct idea / hypothesis surfaced in any of the four files is
> listed once. **Status** reflects what is implemented in this repo as of
> commit `e210ac4` (the 11-row CIFAR-10 ablation sweep).
>
> Legend for *Status*:
>
> - **✓ done** — implemented and trained at least once on CIFAR-10
> - **~ partial** — primitive exists (e.g., a function) but no experiment yet
> - **○ not started** — neither code nor experiment
> - **× deferred** — out of scope for v0.1 (recorded for completeness)

---

## Source coverage summary

| source file | distinct hypotheses surfaced | hypotheses also surfaced elsewhere |
|---|---|---|
| `sacred geometry and neural networks.pdf` | 4 polymath hypotheses + tiered datasets | all 4 cross-listed |
| `sacgeometry.md` | SacredGeoBlock arch (4 sub-modules: Fib scale + Fractal Hex Routing + Platonic Fibottention + Topological Betti loss) + 6 lit-anchored sub-claims | overlaps with PDF + Grok |
| `grok-...transcript.md` | **50 numbered hypotheses** (5 thematic groups × 10) | (master list) |
| `sacred-geometry-...conversation.md` | identical 50 + additional Bridging-Modalities (planar icosahedron unfold) + Anytime-property (Drop-Path) | (cleaned version of transcript) |

The transcript's 50 numbered hypotheses are the canonical IDs (H01–H50).
Additional ideas from PDF / sacgeometry.md that do not fall cleanly into
H01–H50 are added as H51+ at the bottom.

---

## Group G1 — Scaling & Growth (Sacred Growth Laws) — Hypotheses H01–H10

| # | ID | Idea (one line) | Status | Where in repo | Proposed experiment / task | Target dir |
|---|---|---|---|---|---|---|
| 1 | **H01** | φ-Compound Scaling — replace EfficientNet α/β/γ with successive φ-ratios; predict ≥10% FLOPs drop at iso-acc | ○ not started | n/a | New: build EfficientNet-style scaling rule (d,w,r = φ^k) and benchmark vs. EfficientNet-B0 on CIFAR-100 + Tiny ImageNet | `ideas/01_phi_compound_scaling/` |
| 2 | **H02** | Fibonacci Depth Progression — block counts per stage follow Fib seq 3-5-8-13-21; predict 1.5–2× faster convergence | ○ not started | n/a | Stack N-stage backbones whose per-stage block counts are Fib vs. uniform vs. linear; measure epochs-to-target | `ideas/02_fib_depth_progression/` |
| 3 | **H03** | Golden Spiral Resolution Scaling — input res grown by golden-angle (137.5°) + φ-multiples; improves ViT rotation robustness | ○ not started | n/a | Train ViT-Tiny with progressive φ-multiplied resolutions (28→45→73→118→…); test rotated-CIFAR | `ideas/03_golden_spiral_resolution/` |
| 4 | **H04** | φ-Self-Similar Width — channel counts multiplied by φ then rounded to nearest Fib | ✓ done (via `fib`/`phi` channel_mode) | `src/sacgeo/priors.py:fibonacci_channels` | Already trained: `sg_chan_fib` / `sg_chan_phi`. Mod-8 rounding collapses them to same widths at n=3; re-run with c0=32 or n=4 to break equivalence | `ideas/04_phi_self_similar_width/` |
| 5 | **H05** | Fractal φ-Recursion — sub-network depth/width scales by 1/φ recursively | ~ partial (FractalNet style at depth=2, no 1/φ rule) | `src/sacgeo/blocks.py:_FractalPath` | Add 1/φ depth-shrink per recursion level; ablate vs. constant-width fractal | `ideas/05_fractal_phi_recursion/` |
| 6 | **H06** | Golden Ratio Bottleneck — channels reduced by 1/φ then expanded by φ in ResNet/Inception | ○ not started | n/a | Add φ-bottleneck variant of `_GenericConv`; compare to standard 1x1 bottleneck | `ideas/06_golden_bottleneck/` |
| 7 | **H07** | φ-Modulated Multi-Scale — FPN pyramid levels spaced by φ (not 2^k); boost medical-segmentation mIoU | ○ not started | n/a | Build φ-spaced FPN; test on MedMNIST segmentation OR Cityscapes-mini | `ideas/07_phi_multi_scale/` |
| 8 | **H08** | Dynamic φ-Growth — adaptive scaling: add layers during training following Fib rule | ○ not started | n/a | Implement layer-add callback; measure cumulative epoch cost to target | `ideas/08_dynamic_phi_growth/` |
| 9 | **H09** | Golden Proportion Parameter Budget — total params allocated 1 : φ : φ² : … across stages | ~ partial (implicit via `fib` widths) | `src/sacgeo/priors.py:fibonacci_channels` | Explicit param-budget allocator that respects total target budget B | `ideas/09_golden_param_budget/` |
| 10 | **H10** | φ-Decay Learning Rate Scheduler — LR follows φ^{-k} per epoch | ○ not started | n/a | Add `PhiDecayLR` scheduler; ablate vs. cosine on CIFAR-10 | `ideas/10_phi_lr_scheduler/` |

---

## Group G2 — Layer / Channel / Neuron Architectures (Sacred Counting) — H11–H20

| # | ID | Idea | Status | Where in repo | Proposed experiment | Target dir |
|---|---|---|---|---|---|---|
| 11 | **H11** | Pure Fibonacci MLP — hidden sizes 8-13-21-34-…; tabular benchmarks | ○ not started | n/a | Build `FibMLP`; benchmark on Higgs UCI (matches sister `autoresearchtabular` interest) | `ideas/11_pure_fib_mlp/` |
| 12 | **H12** | Fib-Channel CNN — successive conv blocks with Fib filter counts + φ-kernel sizes | ✓ done (filter counts only) | `src/sacgeo/models.py:SacredGeoNet` | Already trained; missing: φ-kernel size variant (3 → 5 at every other stage) | `ideas/12_fib_channel_cnn/` |
| 13 | **H13** | Golden Neuron Connectivity — intra-layer connections weighted/pruned by φ-ratios | ○ not started | n/a | Sparse linear layer with mask pruned by 1/φ; measure speed-vs-acc | `ideas/13_golden_connectivity/` |
| 14 | **H14** | Fibonacci Recurrent — RNN/GRU hidden states sized Fib; φ-gated updates | ○ not started | n/a | Fib-sized GRU on synthetic copy-task; long-range dependency benchmark | `ideas/14_fib_recurrent/` |
| 15 | **H15** | φ-Initialized Embedding — token embeddings sampled from golden-spiral 2D projection | ○ not started | n/a | Replace `nn.Embedding.weight = Xavier` with golden-spiral lattice; small-LM perplexity test | `ideas/15_phi_embedding_init/` |
| 16 | **H16** | Fibonacci Head Diversity — attention heads allocated by Fib counts (diverse dilations) | ○ not started | n/a | ViT-Tiny with Fib-count heads (1,1,2,3,5,8 = 20 heads); compare to 12 uniform | `ideas/16_fib_head_diversity/` |
| 17 | **H17** | Golden Ratio Skip Connections — residual skips scaled by φ or 1/φ | ○ not started | n/a | ResNet-20 with skip scale ∈ {1.0, 1/φ, φ-1}; learning-curve compare | `ideas/17_golden_skip_scale/` |
| 18 | **H18** | Fib-Stage Transition — downsampling factors alternate Fib (stride 2 then 3) | ○ not started | n/a | Stride pattern {2,3,2,3} backbone vs. {2,2,2}; CIFAR-100 | `ideas/18_fib_stage_transition/` |
| 19 | **H19** | φ-Neuron Activation Threshold — per-neuron ReLU threshold tuned to local φ-ratios | ○ not started | n/a | Implement `PhiReLU` with learnable per-channel threshold init at 1/φ | `ideas/19_phi_activation_threshold/` |
| 20 | **H20** | Fibonacci Ensemble — ensemble weights follow Fib ratios | ○ not started | n/a | Take last K checkpoints; weighted average by Fib(K-i); compare to SWA | `ideas/20_fib_ensemble/` |

---

## Group G3 — Topologies & Graphs (Hex, Toroidal, Platonic + φ) — H21–H30

| # | ID | Idea | Status | Where in repo | Proposed experiment | Target dir |
|---|---|---|---|---|---|---|
| 21 | **H21** | Hexagonal φ-Packing — native hex grids + neighbor weights φ-modulated | ~ partial (hex mask done, no φ weighting) | `src/sacgeo/priors.py:HexConv2d` | Add φ-radial weighting of the 7 hex taps; measure rot-eq err and CIFAR/rot-CIFAR | `ideas/21_hex_phi_packing/` |
| 22 | **H22** | Toroidal φ-Closure — circular padding + φ-scaled periodic boundary | ✓ done (toroidal pad) / ~ partial (φ scaling missing) | `src/sacgeo/priors.py:toroidal_pad` | Already trained (`sg_only_toroidal`); add φ-periodic variant where wrap distance ∝ φ | `ideas/22_toroidal_phi_closure/` |
| 23 | **H23** | Platonic φ-Graph — Metatron's Cube connectivity, φ edge weights | ○ not started | n/a | GNN with adjacency = Metatron-Cube vertex-incidence matrix; benchmark on ogbg-molhiv | `ideas/23_platonic_phi_graph/` |
| 24 | **H24** | Icosahedral φ-Equivariant — icosa CNN with channel groups in Fib | ○ not started | n/a | Full e2cnn-style icosahedral conv (or geodesic ICOSA unfold per `sacgeometry.md`); test Spherical MNIST | `ideas/24_icosa_phi_equivariant/` |
| 25 | **H25** | Dodecahedral Latent — project latent onto dodeca vertices (20) | ○ not started | n/a | Linear projector to 20-D, normalize to dodeca vertex set; OOD test | `ideas/25_dodeca_latent/` |
| 26 | **H26** | Fractal Toroidal — recursive graphs on toroidal manifolds, φ-scaled | ○ not started | n/a | Combine `_FractalPath` with `toroidal_pad`; depth=3 recursion on torus | `ideas/26_fractal_toroidal/` |
| 27 | **H27** | Golden Spiral Graph — node embeddings via golden spiral in embedding space | ○ not started | n/a | Initialize GraphTransformer node embeddings on spiral; molecular benchmark | `ideas/27_golden_spiral_graph/` |
| 28 | **H28** | Cymatic Hex Resonance — dynamic edge weights oscillating at φ-harmonic freq | ○ not started | n/a | Modulate hex tap weights by cos(ωt + φk·t); add learnable ω; observe stability | `ideas/28_cymatic_hex_resonance/` |
| 29 | **H29** | φ-Small-World — rewiring probability = 1/φ in small-world graphs | ○ not started | n/a | GNN on Watts-Strogatz with p=1/φ vs. p=0.1, 0.5; node-classification | `ideas/29_phi_small_world/` |
| 30 | **H30** | Platonic-Fib Hybrid — icosa/dodeca symmetry + Fib node degrees | ○ not started | n/a | 3D point-cloud net (ModelNet40 reduced) with Fib-degree icosa adjacency | `ideas/30_platonic_fib_hybrid/` |

The previous CIFAR sweep covered partial H21 + H22 + H05 + H12 only.
The C4 group conv we tested is a *proxy* for Platonic equivariance (H24/H30
full version is a strict superset).

---

## Group G4 — Kernels, Attention, Filters — H31–H40

| # | ID | Idea | Status | Where in repo | Proposed experiment | Target dir |
|---|---|---|---|---|---|---|
| 31 | **H31** | Golden Spiral Kernel — conv kernel init as discretized golden spiral | ○ not started | n/a | Discrete log-spiral mask of 5x5 kernel; compare to Gaussian + He init | `ideas/31_golden_spiral_kernel/` |
| 32 | **H32** | **Fibonacci Dilation Attention (Fibottention)** — attn pattern with non-overlapping Fib dilations across heads; O(N log N); uses Wythoff array | ○ not started | n/a | ViT-Tiny with Fibottention masks (2-6% interactions); CIFAR-100 + dense ViT comparison | `ideas/32_fibottention/` |
| 33 | **H33** | Vesica Piscis Filter — overlapping-circle Flower-of-Life multi-path conv kernel | ○ not started | n/a | Multi-path conv where each path's kernel is a circle of radius r_k, r_k offset by half-radius | `ideas/33_vesica_piscis_filter/` |
| 34 | **H34** | Golden Angle Rotary (RoPE-φ) — RoPE frequencies modulated by golden angle | ○ not started | n/a | Replace RoPE base frequencies with phyllotaxis spacing (golden-angle); LM perplexity test | `ideas/34_golden_angle_rotary/` |
| 35 | **H35** | Cymatic Wavelet Kernels — Chladni-eigenmode wavelet init | ✓ done | `src/sacgeo/priors.py:cymatic_init_` + `chladni_modes` | Already trained (`sg_only_cymatic_init`, negative result). Improve: orthonormalize across modes; randomize at correct frequency band | `ideas/35_cymatic_wavelet/` |
| 36 | **H36** | φ-Spiral Positional Encoding — learnable PE trajectory along golden spiral | ○ not started | n/a | PE = (cos(kθ), sin(kθ), k/N) with θ = golden angle; Transformer | `ideas/36_phi_spiral_pe/` |
| 37 | **H37** | Pentagonal φ-Attention — heads grouped in pentagonal sym (dodeca/icosa) | ○ not started | n/a | ViT with 5/10/20 heads in dodeca-vertex symmetry group; rot test | `ideas/37_pentagonal_attention/` |
| 38 | **H38** | Fractal Golden Filter — self-similar kernels at φ-scales | ○ not started | n/a | Multi-scale kernel composed of 3x3 + 5x5 + 8x8 (Fib) at every layer | `ideas/38_fractal_golden_filter/` |
| 39 | **H39** | Harmonic φ-Activation — GELU/SiLU param'd by φ | ○ not started | n/a | Replace GELU with x · sigmoid(x · φ); CIFAR benchmark | `ideas/39_harmonic_phi_activation/` |
| 40 | **H40** | Metatron Kernel Overlap — kernels share weights via Metatron-Cube projections | ○ not started | n/a | Construct overlapping kernel basis from Metatron's 13-circle pattern | `ideas/40_metatron_kernel/` |

The previous sweep covered H35 (negative result) and H17/H34's nearest
neighbor `golden_modulate` (near-no-op).

---

## Group G5 — Optimization, Init, Regularization, NAS — H41–H50

| # | ID | Idea | Status | Where in repo | Proposed experiment | Target dir |
|---|---|---|---|---|---|---|
| 41 | **H41** | Golden Ratio Optimizer — LR/momentum derived from φ | ○ not started | n/a | Implement `PhiAdamW` (β1=1/φ, β2=1/φ²); compare to AdamW on CIFAR | `ideas/41_golden_optimizer/` |
| 42 | **H42** | φ-Weight Initialization — Xavier/He replaced by φ-scaled variance | ○ not started | n/a | He init scaled by sqrt(φ/fan_in); ablate | `ideas/42_phi_weight_init/` |
| 43 | **H43** | Fibonacci Pruning — iterative pruning ratios follow Fib | ○ not started | n/a | Prune {8,13,21,34,55}% of weights iteratively; vs. magnitude+global | `ideas/43_fib_pruning/` |
| 44 | **H44** | Golden Regularization — L1/L2 coef = 1/φ^k per layer | ○ not started | n/a | Per-layer weight decay 1/φ^depth; CIFAR + over-fitting test | `ideas/44_golden_regularization/` |
| 45 | **H45** | Sacred NAS — NAS restricted to φ/Fib/Platonic priors | ○ not started | n/a | Run DARTS-style NAS with channel set ⊂ Fib; cell library = sacred priors | `ideas/45_sacred_nas/` |
| 46 | **H46** | Cymatic Loss — loss with φ-harmonic terms | ○ not started | n/a | Add Fourier-domain MSE between activation power-spectra and target Chladni mode | `ideas/46_cymatic_loss/` |
| 47 | **H47** | φ-Dropout — dropout rates cycling through Fib | ○ not started | n/a | Schedule dropout from 1/φ → 1/φ² → 1/φ³ over training | `ideas/47_phi_dropout/` |
| 48 | **H48** | Golden Momentum Scheduler — momentum decay φ^{-epoch} | ○ not started | n/a | β1 schedule from 1/φ to 1/φ²; vs. constant 0.9 | `ideas/48_golden_momentum/` |
| 49 | **H49** | Platonic Representation Alignment Loss (PRH) — aux loss pulling latents to Platonic/φ-ideal | ○ not started | n/a | Add CKA-to-target loss where target is a fixed Platonic embedding; PRH paper alignment | `ideas/49_prh_alignment_loss/` |
| 50 | **H50** | **Full Sacred Hybrid** — every prior on; the SacredGeoBlock | ✓ done (negative result) | `src/sacgeo/blocks.py:SacredGeoBlock` | Already trained (`sg_full_fib` is WORST: 73.24%, composite 0.6966). Re-run with subset that excludes group+toroidal (the two most damaging single priors). | `ideas/99_mix_all/` |

---

## Group G6 — Topological & Bridging-Modality Ideas (PDF / sacgeometry.md additions) — H51–H60

| # | ID | Idea | Status | Where in repo | Proposed experiment | Target dir |
|---|---|---|---|---|---|---|
| 51 | **H51** | **Topological Betti Loss** — auxiliary loss penalising β-numbers via persistent homology | ~ partial (we compute Betti curves but do not back-prop through them) | `src/sacgeo/topology.py:betti_curve` | Add differentiable PH loss (TopologyLayer / `topo-loss` library) on stage activations; train | `ideas/51_betti_loss/` |
| 52 | **H52** | **Drop-Path / Anytime evaluation** — FractalNet drop-path regularization → multi-depth inference | ○ not started | n/a | Add DropPath to `_FractalPath`; eval at depth=1,2,3; latency-vs-acc curve | `ideas/52_drop_path_anytime/` |
| 53 | **H53** | **2D-3D Icosahedral Unfold Bridge** — GICOPix planar unfold so hex 2D and icosa 3D share rectilinear grid | ○ not started | n/a | Implement planar-icosa unfold; reuse 2D hex conv weights on 3D spherical MNIST | `ideas/53_icosa_unfold_bridge/` |
| 54 | **H54** | **Persistent Homology Regularization on Activations** — track Betti per stage, regularize toward β₀→1 | ~ partial (compute only) | `src/sacgeo/topology.py` | Same as H51 but tracked across layers + target curve | `ideas/54_ph_activation_reg/` |
| 55 | **H55** | **Platonic Transformers (Islam 2025)** — attention groups keyed to Platonic solid sym | ○ not started | n/a | Reference impl of Platonic Transformer; QM9 molecular regression | `ideas/55_platonic_transformer/` |
| 56 | **H56** | **Cymatic-pattern synthetic dataset** — Chladni-frequency dataset for resonance validation | ○ not started | n/a | Generate via 2-D wave equation in SciPy; train classifier; check generalization with frequency leave-one-out | `ideas/56_cymatic_dataset/` |
| 57 | **H57** | **AudioSet × spectrogram cymatic priors** — cross-modal validation | × deferred | n/a | Train CNN on log-mel spectrograms; reuse cymatic kernels | `ideas/57_audio_cymatic_cross/` |
| 58 | **H58** | **C4 max-pool → average-pool** — fix the dominant negative finding from previous sweep | ○ not started (but identified by previous sweep!) | `src/sacgeo/priors.py:GroupConv2d` (max-pool reduction) | Drop-in change: replace `amax(dim=1)` with `mean(dim=1)`; expect +10pp top-1 in `sg_only_group` | `ideas/58_group_avg_pool/` |
| 59 | **H59** | **Trained-feature Betti** — re-run topology on `best.pt` checkpoints (current Betti is fresh-init) | ~ partial (checkpoint save is wired) | `src/sacgeo/runner.py:save_run` | Re-sweep with `best.pt` saving on, then `compute_topology.py` will see trained features | `ideas/59_trained_betti/` |
| 60 | **H60** | **Multi-seed error bars (3-seed sweep)** — uncertainty quantification on every row | ○ not started | n/a | `run_sweep.py --seeds 0 1 2 --skip-existing`; refresh dashboard | `ideas/60_three_seed_uncertainty/` |

---

## Cross-cutting datasets per the source files

The four documents collectively call out these benchmarks. Tier ↔ feasibility ↔ which idea each dataset best tests:

| dataset | tier | feasibility on RTX 4090 Laptop 16GB | best-tested ideas |
|---|---|---|---|
| **CIFAR-10 / CIFAR-100** | T1 | ≤ 1 day / variant | H01, H02, H04, H05, H06, H12, H14, H17, H21, H22, H50, baseline rail |
| **Tiny ImageNet** (200 cls, 64×64) | T1 | 2-3 days / variant | H01, H02, H07, H09, H32 |
| **ImageNet-100 / ImageNet-1k** | T2 | 3-7 days / variant; ImageNet-1k tight | H01, H07, H50, full SacredGeoBlock |
| **MedMNIST v2 (PathMNIST, OrganAMNIST, OCTMNIST, …)** | T1/T3 | hours/dataset | H04, H05, H06, H07, H12, H44 |
| **Spherical MNIST / IcoMNIST / DeepSphere** | T3 | hours-day | H24, H25, H37, H53 |
| **ModelNet40** / **2D-3D-S** | T3 | day-2 | H24, H30, H37, H55 |
| **ogbg-molhiv / MUTAG / PROTEINS** | T3 | hours | H23, H27, H29, H30, H55 |
| **AudioSet spectrograms** | bonus | day-2 | H57 |
| **Synthetic cymatic plates (SciPy wave-eq)** | bonus | hours | H56, H35 |
| **Rotated/OOD variants of CIFAR/ImageNet** | bonus | hours per variant | H24, H37, H58, robustness rail |
| **Higgs UCI (sister project)** | bonus | hours | H11 |

---

## What this repo has actually run, mapped to the table

The 11-row CIFAR-10 sweep (commit `e210ac4`, single seed, 12 epochs) covered:

| run tag | covers ideas | result vs. predicted |
|---|---|---|
| `baseline_resnet20` | rail | top-1 84.78%, leader |
| `baseline_sg_vanilla` | rail (linear channels, no priors) | 82.16% |
| `sg_chan_fib` | H04 / H12 | 80.11%, tied with phi due to mod-8 |
| `sg_chan_phi` | H04 | 80.11%, **equivalence with fib found** |
| `sg_only_hex` | H21 (partial) | 79.32%, mild negative |
| `sg_only_group` | proxy for H24/H30 (only C4) | 69.84% — biggest single negative; **H58 fix identified** |
| `sg_only_fractal` | H05 (partial, no 1/φ) | 82.46% — **only single prior that lifted top-1** |
| `sg_only_toroidal` | H22 (partial, no φ) | 78.05%, negative as predicted |
| `sg_only_cymatic_init` | H35 | 77.44%, **unexpected negative** |
| `sg_only_golden_modulate` | H17 / H34 (partial) | 79.81%, near no-op |
| `sg_full_fib` | H50 | 73.24% — **WORST**, priors conflict |

So out of the 60-idea design space we have:
- **7 ideas trained at least once** (H04, H05 partial, H12, H17/H34 partial, H21 partial, H22 partial, H35, H50)
- **3 ideas wired-but-not-yet-exploited** (Betti curves H51/H54 computed but no loss; H59 checkpoint saving wired; H22 only as 0-pad replacement)
- **50 ideas not started** (everything else)
- **1 idea identified by our negative result that should now be top-priority**: **H58 (group conv: max-pool → avg-pool)**

---

## Proposed directory restructure (next-turn work)

Per your instruction to make each idea independent, auditable, mix-and-matchable:

```
sacgeometry/
├── README.md                       (master)
├── IDEA_TABLE.md                   (this file)
├── ideas/
│   ├── _TEMPLATE/                  ← copy-and-fill scaffold
│   │   ├── README.md               idea statement + hypothesis + lit cite
│   │   ├── IDEA.md                 formal claim + falsifier
│   │   ├── implementation.py       standalone, single-file module
│   │   ├── tests.py                correctness tests
│   │   ├── AUDIT.md                self-audit (weaknesses found)
│   │   ├── IMPROVEMENTS.md         fix log
│   │   ├── VERIFY.md               verification: tests green, sanity checks
│   │   ├── experiment.py           idea-specific experiment strategy
│   │   ├── configs/<exp>.yaml
│   │   ├── experiments/            local run outputs
│   │   ├── results.md              auto-generated
│   │   └── dashboard/              local dashboard
│   ├── 01_phi_compound_scaling/   ← H01
│   ├── 02_fib_depth_progression/  ← H02
│   ├── 03_golden_spiral_resolution/← H03
│   ├── 04_phi_self_similar_width/ ← H04 [MIGRATE existing]
│   ├── 05_fractal_phi_recursion/  ← H05 [MIGRATE existing partial]
│   ├── …                          ← H06–H49
│   ├── 50_baseline_resnet/        ← ResNet-20 literature anchor
│   ├── 51_betti_loss/             ← H51 (TDL loss)
│   ├── 52_drop_path_anytime/      ← H52 (FractalNet drop-path)
│   ├── 53_icosa_unfold_bridge/    ← H53 (2D-3D bridge)
│   ├── 58_group_avg_pool/         ← H58 (fix from negative result)
│   ├── 59_trained_betti/          ← H59 (re-Betti on trained ckpts)
│   ├── 60_three_seed_uncertainty/ ← H60 (error bars)
│   └── 99_mix_all/                ← H50 + chosen subsets
├── src/sacgeo/                    (shared infra — data loaders, dashboard,
│                                   reasoning gates, training engine —
│                                   used by every ideas/ project)
└── …
```

Each `ideas/NNN_<name>/experiment.py` reads its own config and calls
back into `src/sacgeo` for the training engine, dashboard, and gates, so
the modular work does **not** duplicate the training infrastructure —
only the idea-specific architecture / experiment strategy.

A top-level `compose.py` will let you pick a subset of ideas
(e.g., `[04, 05, 58]`) and produce a "mix" model on demand for the final
hybrid sweep.

This is what the next turn would build. The table above is the contract.
