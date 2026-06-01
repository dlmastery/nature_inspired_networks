# IDEA_TABLE — nature-inspired × Deep-Learning design space

> Incremental master table built by chunk-reading four source files in
> `C:\Users\evija\Downloads`:
>
> - `nature-inspired-geometry and neural networks.pdf` — the original PDF
> - `nature_inspired_networks.md` — Gemini-style synthesis report
> - `grok-nature-inspired-deep-learning-transcript.md` — Grok conversation transcript
> - `nature-inspired-deep-learning-conversation.md` — cleaned conversation
>
> Every distinct idea / hypothesis surfaced in any of the four files is
> listed once. **Status** reflects what is implemented in this repo as of
> the completed 10-agent implementation campaign (May 2026).
>
> **Campaign status: 74 of 75 base hypotheses (H01–H75; only H57
> audio-cross-modal deferred) PLUS 9 G8 esoteric-extension hypotheses
> (H76–H84) are implemented with unit tests.** 80 shared src modules,
> 78 test files, 84 hypothesis design docs. 35 tags additionally carry a
> seed-0 CIFAR-10 12-epoch smoke result (`✓ done`); the remaining coded
> hypotheses are `✓ impl` (module + tests, no CIFAR row yet).
>
> Legend for *Status*:
>
> - **✓ done** — implemented + unit-tested AND trained at least once on CIFAR-10 (seed-0 smoke row exists)
> - **✓ impl** — module + unit tests exist and pass, but no CIFAR-10 result yet
> - **~ partial** — primitive exists (e.g., a function) but no experiment yet
> - **○ not started** — neither code nor experiment
> - **× deferred** — out of scope for v0.1 (recorded for completeness)

---

## Source coverage summary

| source file | distinct hypotheses surfaced | hypotheses also surfaced elsewhere |
|---|---|---|
| `nature-inspired-geometry and neural networks.pdf` | 4 polymath hypotheses + tiered datasets | all 4 cross-listed |
| `nature_inspired_networks.md` | NaturePriorBlock arch (4 sub-modules: Fib scale + Fractal Hex Routing + Platonic Fibottention + Topological Betti loss) + 6 lit-anchored sub-claims | overlaps with PDF + Grok |
| `grok-...transcript.md` | **50 numbered hypotheses** (5 thematic groups × 10) | (master list) |
| `nature-inspired-...conversation.md` | identical 50 + additional Bridging-Modalities (planar icosahedron unfold) + Anytime-property (Drop-Path) | (cleaned version of transcript) |

The transcript's 50 numbered hypotheses are the canonical IDs (H01–H50).
Additional ideas from PDF / nature_inspired_networks.md that do not fall cleanly into
H01–H50 are added as H51+ at the bottom.

---

## Group G1 — Scaling & Growth (Nature-Inspired Growth Laws) — Hypotheses H01–H10

| # | ID | Idea (one line) | Status | Where in repo | Proposed experiment / task | Target dir |
|---|---|---|---|---|---|---|
| 1 | **H01** | φ-Compound Scaling — replace EfficientNet α/β/γ with successive φ-ratios; predict ≥10% FLOPs drop at iso-acc | ✓ done (`phi_compound` .8042) | `src/nature_inspired_networks/scaling.py` (`phi_compound_scaling`) | New: build EfficientNet-style scaling rule (d,w,r = φ^k) and benchmark vs. EfficientNet-B0 on CIFAR-100 + Tiny ImageNet | `ideas/01_phi_compound_scaling/` |
| 2 | **H02** | Fibonacci Depth Progression — block counts per stage follow Fib seq 3-5-8-13-21; predict 1.5–2× faster convergence | ✓ done (`fib_depth` .8218) | `src/nature_inspired_networks/scaling.py` (`fibonacci_depths`) | Stack N-stage backbones whose per-stage block counts are Fib vs. uniform vs. linear; measure epochs-to-target | `ideas/02_fib_depth_progression/` |
| 3 | **H03** | Golden Spiral Resolution Scaling — input res grown by golden-angle (137.5°) + φ-multiples; improves ViT rotation robustness | ✓ done (`golden_resize` .8067) | `src/nature_inspired_networks/multi_scale.py` | Train ViT-Tiny with progressive φ-multiplied resolutions (28→45→73→118→…); test rotated-CIFAR | `ideas/03_golden_spiral_resolution/` |
| 4 | **H04** | φ-Self-Similar Width — channel counts multiplied by φ then rounded to nearest Fib | ✓ done (`chan_fib`/`chan_phi` .8011) | `src/nature_inspired_networks/priors.py:fibonacci_channels` | Already trained: `sg_chan_fib` / `sg_chan_phi`. Mod-8 rounding collapses them to same widths at n=3; re-run with c0=32 or n=4 to break equivalence | `ideas/04_phi_self_similar_width/` |
| 5 | **H05** | Fractal φ-Recursion — sub-network depth/width scales by 1/φ recursively | ✓ done (`fractal` .8246) | `src/nature_inspired_networks/blocks.py:_FractalPath` (+ `fractal_phi_shrink`) | Add 1/φ depth-shrink per recursion level; ablate vs. constant-width fractal | `ideas/05_fractal_phi_recursion/` |
| 6 | **H06** | Golden Ratio Bottleneck — channels reduced by 1/φ then expanded by φ in ResNet/Inception | ✓ done (`golden_bottleneck` .6925) | `src/nature_inspired_networks/phi_scaling.py:GoldenBottleneck` | Add φ-bottleneck variant of `_GenericConv`; compare to standard 1x1 bottleneck | `ideas/06_golden_bottleneck/` |
| 7 | **H07** | φ-Modulated Multi-Scale — FPN pyramid levels spaced by φ (not 2^k); boost medical-segmentation mIoU | ✓ done (`phi_multiscale` .8200) | `src/nature_inspired_networks/phi_scaling.py:PhiSpacedFPN` | Build φ-spaced FPN; test on MedMNIST segmentation OR Cityscapes-mini | `ideas/07_phi_multi_scale/` |
| 8 | **H08** | Dynamic φ-Growth — adaptive scaling: add layers during training following Fib rule | ✓ impl | `src/nature_inspired_networks/dynamic_growth.py` | Implement layer-add callback; measure cumulative epoch cost to target | `ideas/08_dynamic_phi_growth/` |
| 9 | **H09** | Golden Proportion Parameter Budget — total params allocated 1 : φ : φ² : … across stages | 🏆 **CERTIFIED n=7 (2026-05-29)** — `sg_only_phi_budget` CIFAR-100 30-ep Δmean +1.24 pp, paired Wilcoxon p=0.0078, **Holm-Bonferroni α'=0.0167 cleared** (k=3 family); promoted SCREENING→EVALUATION per Rule 28. CIFAR-10 12-ep .8554 (screening). ⛰️ **hill-climbed +1.20 pp** (Phase-9a, 2026-05-30, n=3 tuned-vs-tuned) · **iso-tuned-n=7-FAILS-Phase5** (Phase-9f, 2026-06-01: paired Δmean shrinks to +0.66 pp at n_eff=6, Wilcoxon p_one=0.0781, ordinal gate FAILS min L=0.5998 < max B=0.6075; default-config cert stands). | `src/nature_inspired_networks/phi_scaling.py:phi_budget_widths` / `PhiBudgetNet` | Explicit param-budget allocator that respects total target budget B | `ideas/09_golden_param_budget/` |
| 10 | **H10** | φ-Decay Learning Rate Scheduler — LR follows φ^{-k} per epoch | ✓ done (`phi_lr` .7875) | `src/nature_inspired_networks/schedulers.py` (`PhiDecayLR`) | Add `PhiDecayLR` scheduler; ablate vs. cosine on CIFAR-10 | `ideas/10_phi_lr_scheduler/` |

---

## Group G2 — Layer / Channel / Neuron Architectures (Nature-Inspired Counting) — H11–H20

| # | ID | Idea | Status | Where in repo | Proposed experiment | Target dir |
|---|---|---|---|---|---|---|
| 11 | **H11** | Pure Fibonacci MLP — hidden sizes 8-13-21-34-…; tabular benchmarks | ✓ impl | `src/nature_inspired_networks/fib_mlp.py` | Build `FibMLP`; benchmark on Higgs UCI (matches sister `autoresearchtabular` interest) | `ideas/11_pure_fib_mlp/` |
| 12 | **H12** | Fib-Channel CNN — successive conv blocks with Fib filter counts + φ-kernel sizes | ✓ done (`chan_fib` .8011) | `src/nature_inspired_networks/models.py:NaturePriorNet` | Already trained; missing: φ-kernel size variant (3 → 5 at every other stage) | `ideas/12_fib_channel_cnn/` |
| 13 | **H13** | Golden Neuron Connectivity — intra-layer connections weighted/pruned by φ-ratios | ✓ done (`phi_sparse` .7333) | `src/nature_inspired_networks/sparse.py` | Sparse linear layer with mask pruned by 1/φ; measure speed-vs-acc | `ideas/13_golden_connectivity/` |
| 14 | **H14** | Fibonacci Recurrent — RNN/GRU hidden states sized Fib; φ-gated updates | ✓ impl | `src/nature_inspired_networks/fib_recurrent.py` | Fib-sized GRU on synthetic copy-task; long-range dependency benchmark | `ideas/14_fib_recurrent/` |
| 15 | **H15** | φ-Initialized Embedding — token embeddings sampled from golden-spiral 2D projection | ✓ impl | `src/nature_inspired_networks/phi_embedding.py` | Replace `nn.Embedding.weight = Xavier` with golden-spiral lattice; small-LM perplexity test | `ideas/15_phi_embedding_init/` |
| 16 | **H16** | Fibonacci Head Diversity — attention heads allocated by Fib counts (diverse dilations) | ✓ impl | `src/nature_inspired_networks/fib_attention.py` | ViT-Tiny with Fib-count heads (1,1,2,3,5,8 = 20 heads); compare to 12 uniform | `ideas/16_fib_head_diversity/` |
| 17 | **H17** | Golden Ratio Skip Connections — residual skips scaled by φ or 1/φ | ✓ done (`golden_skip` .8163) | `src/nature_inspired_networks/phi_scaling.py:GoldenSkipBlock` | ResNet-20 with skip scale ∈ {1.0, 1/φ, φ-1}; learning-curve compare | `ideas/17_golden_skip_scale/` |
| 18 | **H18** | Fib-Stage Transition — downsampling factors alternate Fib (stride 2 then 3) | ✓ done (`fib_stride` .7255) | `src/nature_inspired_networks/stride.py` | Stride pattern {2,3,2,3} backbone vs. {2,2,2}; CIFAR-100 | `ideas/18_fib_stage_transition/` |
| 19 | **H19** | φ-Neuron Activation Threshold — per-neuron ReLU threshold tuned to local φ-ratios | ✓ done (`phi_relu` .7107) | `src/nature_inspired_networks/phi_threshold.py` | Implement `PhiReLU` with learnable per-channel threshold init at 1/φ | `ideas/19_phi_activation_threshold/` |
| 20 | **H20** | Fibonacci Ensemble — ensemble weights follow Fib ratios | ✓ done (`fib_ensemble` .8011) | `src/nature_inspired_networks/ensemble.py` | Take last K checkpoints; weighted average by Fib(K-i); compare to SWA | `ideas/20_fib_ensemble/` |

---

## Group G3 — Topologies & Graphs (Hex, Toroidal, Platonic + φ) — H21–H30

| # | ID | Idea | Status | Where in repo | Proposed experiment | Target dir |
|---|---|---|---|---|---|---|
| 21 | **H21** | Hexagonal φ-Packing — native hex grids + neighbor weights φ-modulated | ✓ done (`hex` .7932) | `src/nature_inspired_networks/priors.py:HexConv2d` (+ `hex_kernel_radius`) | Add φ-radial weighting of the 7 hex taps; measure rot-eq err and CIFAR/rot-CIFAR | `ideas/21_hex_phi_packing/` |
| 22 | **H22** | Toroidal φ-Closure — circular padding + φ-scaled periodic boundary | ✓ done (`toroidal` .7805) | `src/nature_inspired_networks/priors.py:toroidal_pad` | Already trained (`sg_only_toroidal`); add φ-periodic variant where wrap distance ∝ φ | `ideas/22_toroidal_phi_closure/` |
| 23 | **H23** | Platonic φ-Graph — Metatron's Cube connectivity, φ edge weights | ✓ impl | `src/nature_inspired_networks/platonic_graph.py` | GNN with adjacency = Metatron-Cube vertex-incidence matrix; benchmark on ogbg-molhiv | `ideas/23_platonic_phi_graph/` |
| 24 | **H24** | Icosahedral φ-Equivariant — icosa CNN with channel groups in Fib | ✓ impl | `src/nature_inspired_networks/icosa.py` | Full e2cnn-style icosahedral conv (or geodesic ICOSA unfold per `nature_inspired_networks.md`); test Spherical MNIST | `ideas/24_icosa_phi_equivariant/` |
| 25 | **H25** | Dodecahedral Latent — project latent onto dodeca vertices (20) | ✓ impl | `src/nature_inspired_networks/dodeca_latent.py` | Linear projector to 20-D, normalize to dodeca vertex set; OOD test | `ideas/25_dodeca_latent/` |
| 26 | **H26** | Fractal Toroidal — recursive graphs on toroidal manifolds, φ-scaled | ✓ impl | `src/nature_inspired_networks/fractal_toroidal.py` | Combine `_FractalPath` with `toroidal_pad`; depth=3 recursion on torus | `ideas/26_fractal_toroidal/` |
| 27 | **H27** | Golden Spiral Graph — node embeddings via golden spiral in embedding space | ✓ impl | `src/nature_inspired_networks/spiral_graph.py` | Initialize GraphTransformer node embeddings on spiral; molecular benchmark | `ideas/27_golden_spiral_graph/` |
| 28 | **H28** | Cymatic Hex Resonance — dynamic edge weights oscillating at φ-harmonic freq | ✓ impl | `src/nature_inspired_networks/cymatic_hex.py` | Modulate hex tap weights by cos(ωt + φk·t); add learnable ω; observe stability | `ideas/28_cymatic_hex_resonance/` |
| 29 | **H29** | φ-Small-World — rewiring probability = 1/φ in small-world graphs | ✓ impl | `src/nature_inspired_networks/small_world.py` | GNN on Watts-Strogatz with p=1/φ vs. p=0.1, 0.5; node-classification | `ideas/29_phi_small_world/` |
| 30 | **H30** | Platonic-Fib Hybrid — icosa/dodeca symmetry + Fib node degrees | ✓ impl | `src/nature_inspired_networks/platonic_fib.py` | 3D point-cloud net (ModelNet40 reduced) with Fib-degree icosa adjacency | `ideas/30_platonic_fib_hybrid/` |

The previous CIFAR sweep covered partial H21 + H22 + H05 + H12 only.
The C4 group conv we tested is a *proxy* for Platonic equivariance (H24/H30
full version is a strict superset).

---

## Group G4 — Kernels, Attention, Filters — H31–H40

| # | ID | Idea | Status | Where in repo | Proposed experiment | Target dir |
|---|---|---|---|---|---|---|
| 31 | **H31** | Golden Spiral Kernel — conv kernel init as discretized golden spiral | ✓ done (`golden_spiral_init` .8042) | `src/nature_inspired_networks/inits.py` | Discrete log-spiral mask of 5x5 kernel; compare to Gaussian + He init | `ideas/31_golden_spiral_kernel/` |
| 32 | **H32** | **Fibonacci Dilation Attention (Fibottention)** — attn pattern with non-overlapping Fib dilations across heads; O(N log N); uses Wythoff array | ✓ impl | `src/nature_inspired_networks/fibottention.py` | ViT-Tiny with Fibottention masks (2-6% interactions); CIFAR-100 + dense ViT comparison | `ideas/32_fibottention/` |
| 33 | **H33** | Vesica Piscis Filter — overlapping-circle Flower-of-Life multi-path conv kernel | ✓ impl | `src/nature_inspired_networks/vesica_piscis.py` | Multi-path conv where each path's kernel is a circle of radius r_k, r_k offset by half-radius | `ideas/33_vesica_piscis_filter/` |
| 34 | **H34** | Golden Angle Rotary (RoPE-φ) — RoPE frequencies modulated by golden angle | ✓ impl | `src/nature_inspired_networks/golden_rope.py` | Replace RoPE base frequencies with phyllotaxis spacing (golden-angle); LM perplexity test | `ideas/34_golden_angle_rotary/` |
| 35 | **H35** | Cymatic Wavelet Kernels — Chladni-eigenmode wavelet init | ✓ done (`cymatic_init` .7744) | `src/nature_inspired_networks/priors.py:cymatic_init_` + `chladni_modes` | Already trained (`sg_only_cymatic_init`, negative result). Improve: orthonormalize across modes; randomize at correct frequency band | `ideas/35_cymatic_wavelet/` |
| 36 | **H36** | φ-Spiral Positional Encoding — learnable PE trajectory along golden spiral | ✓ impl | `src/nature_inspired_networks/spiral_pe.py` | PE = (cos(kθ), sin(kθ), k/N) with θ = golden angle; Transformer | `ideas/36_phi_spiral_pe/` |
| 37 | **H37** | Pentagonal φ-Attention — heads grouped in pentagonal sym (dodeca/icosa) | ✓ impl | `src/nature_inspired_networks/pentagonal_attention.py` | ViT with 5/10/20 heads in dodeca-vertex symmetry group; rot test | `ideas/37_pentagonal_attention/` |
| 38 | **H38** | Fractal Golden Filter — self-similar kernels at φ-scales | ✓ impl | `src/nature_inspired_networks/fractal_filter.py` | Multi-scale kernel composed of 3x3 + 5x5 + 8x8 (Fib) at every layer | `ideas/38_fractal_golden_filter/` |
| 39 | **H39** | Harmonic φ-Activation — GELU/SiLU param'd by φ | ✓ done (`phi_activation` .7995) | `src/nature_inspired_networks/activations.py:PhiGELU` | Replace GELU with x · sigmoid(x · φ); CIFAR benchmark | `ideas/39_harmonic_phi_activation/` |
| 40 | **H40** | Metatron Kernel Overlap — kernels share weights via Metatron-Cube projections | ✓ impl | `src/nature_inspired_networks/metatron_kernel.py` | Construct overlapping kernel basis from Metatron's 13-circle pattern | `ideas/40_metatron_kernel/` |

The original 11-row sweep covered H35 (negative result) and H17/H34's nearest
neighbor `golden_modulate` (near-no-op, .7981); the campaign has since added
seed-0 smoke rows for H31 and H39 and full modules for H32/H33/H34/H36/H37/H38/H40.

---

## Group G5 — Optimization, Init, Regularization, NAS — H41–H50

| # | ID | Idea | Status | Where in repo | Proposed experiment | Target dir |
|---|---|---|---|---|---|---|
| 41 | **H41** | Golden Ratio Optimizer — LR/momentum derived from φ | ✓ done (`golden_adam` .5196 — **falsified, worst row**) | `src/nature_inspired_networks/optimizers.py` (`PhiAdamW`) | Implement `PhiAdamW` (β1=1/φ, β2=1/φ²); compare to AdamW on CIFAR | `ideas/41_golden_optimizer/` |
| 42 | **H42** | φ-Weight Initialization — Xavier/He replaced by φ-scaled variance | ✓ done (`phi_init` .7656) | `src/nature_inspired_networks/inits.py` | He init scaled by sqrt(φ/fan_in); ablate | `ideas/42_phi_weight_init/` |
| 43 | **H43** | Fibonacci Pruning — iterative pruning ratios follow Fib | ✓ done (`fib_prune` .8115) | `src/nature_inspired_networks/pruning.py` | Prune {8,13,21,34,55}% of weights iteratively; vs. magnitude+global | `ideas/43_fib_pruning/` |
| 44 | **H44** | Golden Regularization — L1/L2 coef = 1/φ^k per layer | 🏆 **part of CERTIFIED n=7 pair_gm_pdw combo** (2026-05-29); standalone `phi_decay` CIFAR-10 12-ep .7981 (screening). The H09+H48+H44 3-axis stack `pair_gm_pdw` clears α=0.05 Holm-Bonferroni at n=7 (CIFAR-100 30-ep Δmean +1.74 pp, p=0.0078). ⛰️ **hill-climbed +1.80 pp** (Phase-9a `pair_gm_pdw`, 2026-05-30, n=3 tuned-vs-tuned) · **iso-tuned-n=7-FAILS-Phase5** (Phase-9f, 2026-06-01: paired Δmean shrinks to +0.79 pp at n=7, Wilcoxon p_one=0.1094, only 4/7 paired deltas positive, ordinal gate FAILS min L=0.6049 < max B=0.6075; default-config cert stands). | `src/nature_inspired_networks/phi_decay.py` | Per-layer weight decay 1/φ^depth; CIFAR + over-fitting test | `ideas/44_golden_regularization/` |
| 45 | **H45** | Nature-Inspired NAS — NAS restricted to φ/Fib/Platonic priors | ✓ impl | `src/nature_inspired_networks/sacred_nas.py` | Run DARTS-style NAS with channel set ⊂ Fib; cell library = nature-inspired priors | `ideas/45_sacred_nas/` |
| 46 | **H46** | Cymatic Loss — loss with φ-harmonic terms | ✓ impl | `src/nature_inspired_networks/cymatic_loss.py` | Add Fourier-domain MSE between activation power-spectra and target Chladni mode | `ideas/46_cymatic_loss/` |
| 47 | **H47** | φ-Dropout — dropout rates cycling through Fib | ✓ done (`phi_dropout` .8280) | `src/nature_inspired_networks/regularizers.py` | Schedule dropout from 1/φ → 1/φ² → 1/φ³ over training | `ideas/47_phi_dropout/` |
| 48 | **H48** | Golden Momentum Scheduler — momentum decay φ^{-epoch} | 🏆 **part of CERTIFIED n=7 pair_gm_pdw combo** (2026-05-29); standalone `golden_momentum` CIFAR-10 12-ep .8352 (screening). The H09+H48+H44 3-axis stack `pair_gm_pdw` clears α=0.05 Holm-Bonferroni at n=7 (CIFAR-100 30-ep Δmean +1.74 pp, p=0.0078). ⛰️ **hill-climbed +1.80 pp** (Phase-9a `pair_gm_pdw`, 2026-05-30, n=3 tuned-vs-tuned) · **iso-tuned-n=7-FAILS-Phase5** (Phase-9f, 2026-06-01: paired Δmean shrinks to +0.79 pp at n=7, Wilcoxon p_one=0.1094, only 4/7 paired deltas positive, ordinal gate FAILS min L=0.6049 < max B=0.6075; default-config cert stands). | `src/nature_inspired_networks/schedulers.py` | β1 schedule from 1/φ to 1/φ²; vs. constant 0.9 | `ideas/48_golden_momentum/` |
| 49 | **H49** | Platonic Representation Alignment Loss (PRH) — aux loss pulling latents to Platonic/φ-ideal | ✓ impl | `src/nature_inspired_networks/prh_loss.py` | Add CKA-to-target loss where target is a fixed Platonic embedding; PRH paper alignment | `ideas/49_prh_alignment_loss/` |
| 50 | **H50** | **Full Nature-Inspired Hybrid** — every prior on; the NaturePriorBlock | ✓ done (`full_fib` .7324 — negative result, WORST of the original 11) | `src/nature_inspired_networks/blocks.py:NaturePriorBlock` (+ `hybrid_full.py`) | Already trained (`sg_full_fib`, composite 0.6966). Re-run with subset that excludes group+toroidal (the two most damaging single priors). | `ideas/99_mix_all/` |

---

## Group G6 — Topological & Bridging-Modality Ideas (PDF / nature_inspired_networks.md additions) — H51–H60

| # | ID | Idea | Status | Where in repo | Proposed experiment | Target dir |
|---|---|---|---|---|---|---|
| 51 | **H51** | **Topological Betti Loss** — auxiliary loss penalising β-numbers via persistent homology | ✓ impl | `src/nature_inspired_networks/betti_loss.py` | Add differentiable PH loss (TopologyLayer / `topo-loss` library) on stage activations; train | `ideas/51_betti_loss/` |
| 52 | **H52** | **Drop-Path / Anytime evaluation** — FractalNet drop-path regularization → multi-depth inference | ✓ impl | `src/nature_inspired_networks/drop_path.py` | Add DropPath to `_FractalPath`; eval at depth=1,2,3; latency-vs-acc curve | `ideas/52_drop_path_anytime/` |
| 53 | **H53** | **2D-3D Icosahedral Unfold Bridge** — GICOPix planar unfold so hex 2D and icosa 3D share rectilinear grid | ✓ impl | `src/nature_inspired_networks/icosa_unfold.py` | Implement planar-icosa unfold; reuse 2D hex conv weights on 3D spherical MNIST | `ideas/53_icosa_unfold_bridge/` |
| 54 | **H54** | **Persistent Homology Regularization on Activations** — track Betti per stage, regularize toward β₀→1 | ✓ impl | `src/nature_inspired_networks/ph_reg.py` | Same as H51 but tracked across layers + target curve | `ideas/54_ph_activation_reg/` |
| 55 | **H55** | **Platonic Transformers (Islam 2025)** — attention groups keyed to Platonic solid sym | ✓ impl | `src/nature_inspired_networks/platonic_transformer.py` | Reference impl of Platonic Transformer; QM9 molecular regression | `ideas/55_platonic_transformer/` |
| 56 | **H56** | **Cymatic-pattern synthetic dataset** — Chladni-frequency dataset for resonance validation | ✓ impl | `src/nature_inspired_networks/cymatic_dataset.py` | Generate via 2-D wave equation in SciPy; train classifier; check generalization with frequency leave-one-out | `ideas/56_cymatic_dataset/` |
| 57 | **H57** | **AudioSet × spectrogram cymatic priors** — cross-modal validation | × deferred | n/a (no module — only deferred hypothesis) | Train CNN on log-mel spectrograms; reuse cymatic kernels | `ideas/57_audio_cymatic_cross/` |
| 58 | **H58** | **C4 max-pool → average-pool** — fix the dominant negative finding from previous sweep | ✓ done (`group`/proxy .6984 vs `group_avg` .6538) | `src/nature_inspired_networks/blocks.py` (`group_reduce` flag) | Drop-in change: replace `amax(dim=1)` with `mean(dim=1)`; expect +10pp top-1 in `sg_only_group` | `ideas/58_group_avg_pool/` |
| 59 | **H59** | **Trained-feature Betti** — re-run topology on `best.pt` checkpoints (current Betti is fresh-init) | ✓ impl | `src/nature_inspired_networks/trained_betti.py` | Re-sweep with `best.pt` saving on, then `compute_topology.py` will see trained features | `ideas/59_trained_betti/` |
| 60 | **H60** | **Multi-seed error bars (3-seed sweep)** — uncertainty quantification on every row | ✓ impl | `src/nature_inspired_networks/multi_seed.py` | `run_sweep.py --seeds 0 1 2 --skip-existing`; refresh dashboard | `ideas/60_three_seed_uncertainty/` |

---

## Group G7 — Cross-Paradigm Hybrids (sacred priors × Liquid/JEPA/KAN/Transformer/GNN) — H61–H75

15 cross-paradigm fusion hypotheses. All ship as standalone primitive modules
with unit tests; none has a CIFAR-10 smoke row yet (`✓ impl`). H67 is the
flagship full Sacred-Liquid-JEPA-KAN-GNN-Transformer hybrid.

| # | ID | Idea | Status | Where in repo | Proposed experiment | Target dir |
|---|---|---|---|---|---|---|
| 61 | **H61** | Sacred × Liquid-JEPA — nature priors fused into a Liquid/JEPA predictive backbone | ✓ impl | `src/nature_inspired_networks/hybrid_liquid_jepa.py` | Liquid-cell + JEPA latent target with φ-priors; CIFAR/latent-predict | `ideas/61_sacred_liquid_jepa/` |
| 62 | **H62** | Toroidal-KV Hex Attention — toroidal-closed KV cache + hex attention pattern | ✓ impl | `src/nature_inspired_networks/hybrid_toroidal_hex_attn.py` | Toroidal-wrapped KV with hex neighbor attention; ViT-Tiny | `ideas/62_toroidal_kv_hex_attention/` |
| 63 | **H63** | Platonic-Aux Cymatic Teacher — Platonic-aligned student distilled from cymatic teacher | ✓ impl | `src/nature_inspired_networks/hybrid_platonic_cymatic.py` | Aux distillation loss with cymatic teacher targets | `ideas/63_platonic_aux_cymatic_teacher/` |
| 64 | **H64** | Dynamic Growth-Pruning — φ-growth + Fib-pruning co-scheduled | ✓ impl | `src/nature_inspired_networks/hybrid_growth_pruning.py` | Couple H08 growth with H43 pruning in one training loop | `ideas/64_dynamic_growth_pruning/` |
| 65 | **H65** | PH-Betti Collapse Loss — persistent-homology + collapse-gate joint regularizer | ✓ impl | `src/nature_inspired_networks/hybrid_ph_collapse.py` | Combine H51 Betti loss with H83 collapse gate | `ideas/65_ph_betti_collapse_loss/` |
| 66 | **H66** | Cymatic QKV Kernel — cymatic-eigenmode init applied to attention QKV projections | ✓ impl | `src/nature_inspired_networks/hybrid_cymatic_qkv.py` | Init QKV weights from Chladni modes; ViT perplexity/acc | `ideas/66_cymatic_qkv_kernel/` |
| 67 | **H67** | **Full Paradigm Hybrid** — flagship Sacred-Liquid-JEPA-KAN-GNN-Transformer fusion | ✓ impl | `src/nature_inspired_networks/hybrid_full.py` | Compose all paradigm bridges; ablate component-by-component | `ideas/67_full_paradigm_hybrid/` |
| 68 | **H68** | On-Device World Model — compact JEPA-style world model with nature priors | ✓ impl | `src/nature_inspired_networks/hybrid_world_model.py` | Latent rollout world model; on-device latency budget | `ideas/68_ondevice_world_model/` |
| 69 | **H69** | KAN-Metatron Symbolic Head — KAN spline head over Metatron-routed features | ✓ impl | `src/nature_inspired_networks/hybrid_kan_metatron.py` | KAN head + Metatron kernel basis; symbolic-regression probe | `ideas/69_kan_metatron_symbolic_head/` |
| 70 | **H70** | Cymatic Low-Data Curriculum — cymatic-dataset pretrain → low-data finetune curriculum | ✓ impl | `src/nature_inspired_networks/hybrid_cymatic_curriculum.py` | Curriculum from synthetic Chladni plates to CIFAR low-data | `ideas/70_cymatic_low_data_curriculum/` |
| 71 | **H71** | Icosa-RoPE 3D — icosahedral-symmetric 3D rotary positional encoding | ✓ impl | `src/nature_inspired_networks/hybrid_icosa_rope.py` | 3D RoPE keyed to icosa vertices; point-cloud transformer | `ideas/71_icosa_rope_3d/` |
| 72 | **H72** | Fractal-Vesica FFN — fractal multi-path FFN with Vesica-Piscis filters | ✓ impl | `src/nature_inspired_networks/hybrid_fractal_vesica.py` | Replace transformer FFN with fractal-vesica multi-path | `ideas/72_fractal_vesica_ffn/` |
| 73 | **H73** | Golden-Spiral Metatron PE — concat golden-spiral PE + Metatron-routed PE → proj | ✓ impl | `src/nature_inspired_networks/hybrid_spiral_metatron_pe.py` | Concatenated spiral + Metatron PE; sequence-model probe | `ideas/73_golden_spiral_metatron_pe/` |
| 74 | **H74** | Metatron Overlap Tying — single tied conv weight + 13 learnable circle scales | ✓ impl | `src/nature_inspired_networks/hybrid_metatron_tying.py` | Weight-tied Metatron kernel with per-circle scales | `ideas/74_metatron_overlap_tying/` |
| 75 | **H75** | Harmonic Cymatic SwiGLU — PhiGELU gate + cymatic-orthonormal up-projection init | ✓ impl | `src/nature_inspired_networks/hybrid_cymatic_swiglu.py` | SwiGLU with PhiGELU gate + cymatic up_a init; transformer FFN | `ideas/75_harmonic_cymatic_swiglu/` |

---

## Group G8 — Esoteric-Extension Priors (neutral recast of `bonus_hypothesis.md`) — H76–H84

Source: `Downloads/bonus_hypothesis.md` (esoteric H171–H200). Filtered to the
9 ideas with a concrete, citable, CIFAR-scale-implementable mechanism; the
rest were rejected as redundant, mechanism-free, or unspecified (see journal
below). Every artifact is neutral-named per Rule 16; the esoteric origin is
acknowledged in one sentence inside each module/doc only.

| # | ID | Bonus src | Neutral artifact | Idea (one line) | Status | Where in repo |
|---|---|---|---|---|---|---|
| 76 | **H76** | H171 Merkaba | `TetrahedralDualPathBlock` | dual opposing C4 group-conv paths (max vs mean pool), learnable convex merge | ✓ impl + tests; **smoke-wired** as part of dual-path study | `src/nature_inspired_networks/tetra_dualpath.py` |
| 77 | **H77** | H173 Lotus | `RadialSymmetry12Attention` | 12-fold relative-position angular-bias MHA | ✓ impl + tests | `src/nature_inspired_networks/radial12_attention.py` |
| 78 | **H78** | H174 Torus | `ToroidalLatent` | latent → 2-torus embedding (cosθ₁,sinθ₁,cosθ₂,sinθ₂) | ✓ impl + tests | `src/nature_inspired_networks/toroidal_latent.py` |
| 79 | **H79** | H175 Jitterbug | `MorphingGraphLayer` | learnable cuboctahedron↔icosahedron adjacency interp | ✓ impl + tests | `src/nature_inspired_networks/morphing_adjacency.py` |
| 80 | **H80** | H176 Reuleaux | `ConstantWidthConv2d` | constant-width (Reuleaux) isotropic kernel mask | ✓ impl + tests + **sweep row** `sg_only_constant_width` | `src/nature_inspired_networks/constant_width_kernel.py` |
| 81 | **H81** | H177 Vibration | `SinusoidalActivation` | SIREN-style `sin(ωx)`, learnable ω (Sitzmann 2020) | 🏆 **CERTIFIED n=7 (2026-05-29)** — `slot_act_sine` (SLOT variant) CIFAR-100 30-ep Δmean **+1.78 pp**, paired Wilcoxon **p=0.0078**, Holm-Bonferroni α'=0.0167 cleared (k=3 family); promoted SCREENING→EVALUATION per Rule 28. CIFAR-10 12-ep sweep row `sg_only_sine_act` retained as screening. ⛰️ **hill-climbed +2.08 pp** (Phase-9a, 2026-05-30, n=3 tuned-vs-tuned) · **iso-tuned-n=7-FAILS-Phase5** (Phase-9f, 2026-06-01: paired Δmean shrinks to +0.25 pp at n_eff=4 wd=5e-4 baseline neighbour, Wilcoxon p_one=0.3750, ordinal gate FAILS min L=0.6057 < max B=0.6075; wd=2e-3 baseline neighbour is Phase-9e; default-config cert stands). | `src/nature_inspired_networks/sinusoidal_activation.py` |
| 82 | **H82** | H180/181 Masonry | `VoronoiSparseAttention` | irregular Voronoi/Delaunay tessellation sparse-attention mask | ✓ impl + tests | `src/nature_inspired_networks/voronoi_attention.py` |
| 83 | **H83** | H182 Double-slit | `CollapseGatedAttention` | learnable softmax-temperature sharpening (entmax-family) | ✓ impl + tests | `src/nature_inspired_networks/collapse_attention.py` |
| 84 | **H84** | H183 Crystal | `SpectralHopfieldMemory` | modern Hopfield retrieval in FFT eigenmode basis (Ramsauer 2020) | ✓ impl + tests | `src/nature_inspired_networks/spectral_hopfield.py` |

**Rejected from the bonus list (kept here for audit completeness):**

- **H172 Seed-of-Life 7-circle kernel** — redundant with H33 Vesica Piscis +
  H40 Metatron kernel (both already overlapping-circle masked convs).
- **H178 Yin-Yang dual-path** — generic dual-path, subsumed by H76.
- **H179 Bioelectric field priors** — no concrete operational mechanism
  ("voltage routers" is a metaphor, not a layer); not implementable as stated.
- **H184–H200** (Mandala activation axis, Lightcode symbolic regression,
  Geometric Router curriculum, Conscious Collapse reg, Quantum Harmonics loss,
  Pineal Translator gate, …) — only named, never specified. Where a real
  mechanism is implied it duplicates existing work (Quantum Harmonics ≈ H46
  cymatic loss; Conscious Collapse ≈ H83 temperature gate).

The two CNN-droppable members (H80, H81) are wired as Rule-1 atomic post-build
mutators in `runner.post_build_mutators` (override keys `constant_width_kernel`
and `sine_activation`) and carry sweep rows; the attention/latent/graph/memory
members ship as standalone primitives (same convention as G2/G4/G7).

---

## Cross-cutting datasets per the source files

The four documents collectively call out these benchmarks. Tier ↔ feasibility ↔ which idea each dataset best tests:

| dataset | tier | feasibility on RTX 4090 Laptop 16GB | best-tested ideas |
|---|---|---|---|
| **CIFAR-10 / CIFAR-100** | T1 | ≤ 1 day / variant | H01, H02, H04, H05, H06, H12, H14, H17, H21, H22, H50, baseline rail |
| **Tiny ImageNet** (200 cls, 64×64) | T1 | 2-3 days / variant | H01, H02, H07, H09, H32 |
| **ImageNet-100 / ImageNet-1k** | T2 | 3-7 days / variant; ImageNet-1k tight | H01, H07, H50, full NaturePriorBlock |
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

**Implementation campaign (May 2026):** 74 of 75 base hypotheses
(H01–H75; only H57 deferred) plus all 9 G8 hypotheses (H76–H84) now have
a shared src module + passing unit tests (80 modules, 78 test files). On
top of that, **35 tags carry a seed-0 12-epoch CIFAR-10 smoke result**
(the original 11-row sweep + the campaign single-prior/optimizer/init/reg
rows + H80/H81). The seed-0 leaderboard standout is **`phi_budget` (H09)
at 0.8554 — the only variant to beat `baseline_resnet20` (0.8478)**; the
falsified worst row is **`golden_adam` (H41) at 0.5196**. Full seed-0
top-1 ranking (high→low): phi_budget .8554 | baseline_resnet20 .8478 |
golden_momentum(H48) .8352 | phi_dropout(H47) .8280 | fractal(H05) .8246 |
fib_depth(H02) .8218 | baseline_sg_vanilla .8216 | phi_multiscale(H07)
.8200 | golden_skip(H17) .8163 | fib_prune(H43) .8115 | golden_resize(H03)
.8067 | sine_act(H81) .8062 | golden_spiral_init(H31) .8042 |
phi_compound(H01) .8042 | chan_phi/chan_fib(H04/H12) .8011 |
fib_ensemble(H20) .8011 | phi_activation(H39) .7995 | golden_modulate
.7981 | phi_decay(H44) .7981 | hex(H21) .7932 | phi_lr(H10) .7875 |
toroidal(H22) .7805 | cymatic_init(H35) .7744 | phi_init(H42) .7656 |
constant_width(H80) .7595 | phi_sparse(H13) .7333 | full_fib(H50) .7324 |
fib_stride(H18) .7255 | phi_relu(H19) .7107 | group(H58 proxy) .6984 |
golden_bottleneck(H06) .6925 | group_avg(H58) .6538 | golden_adam(H41)
.5196.

The original 11-row CIFAR-10 sweep (commit `e210ac4`, single seed, 12
epochs) covered:

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

So out of the full 84-hypothesis design space we now have:
- **83 hypotheses implemented + unit-tested** (H01–H56, H58–H84; only
  **H57** audio-cross-modal is deferred with no module).
- **35 tags with a seed-0 CIFAR-10 smoke result** (`✓ done`): the rail
  baselines plus H01,H02,H03,H04/H12,H05,H06,H07,H09,H10,H13,H17,H18,
  H19,H20,H21,H22,H31,H35,H39,H41,H42,H43,H44,H47,H48,H50,H58,H80,H81.
- **48 hypotheses `✓ impl` but no CIFAR row yet** (attention / graph /
  latent / memory / cross-paradigm-hybrid modules: H08,H11,H14,H15,H16,
  H23–H30,H32,H33,H34,H36,H37,H38,H40,H45,H46,H49,H51–H56,H59,H60,
  H61–H79,H82,H83,H84).
- **1 deferred**: **H57** (AudioSet cross-modal — out of v0.1 scope).
- The negative-result-driven priority **H58 (group conv max→avg pool)**
  is now implemented as the `group_reduce` flag and carries both seed-0
  rows (`group` .6984, `group_avg` .6538).

---

## Proposed directory restructure (next-turn work)

Per your instruction to make each idea independent, auditable, mix-and-matchable:

```
nature_inspired_networks/
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
├── src/nature_inspired_networks/                    (shared infra — data loaders, dashboard,
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
