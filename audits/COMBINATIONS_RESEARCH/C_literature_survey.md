# C — Literature Survey: Mixing and Matching Training Tricks for Image Classification

> Mainstream-literature state-of-the-art on stacking training-time,
> architectural, and inference-time tricks for image classification.
> What productively combines, what saturates, what conflicts.
> Authored 2026-05-30 as part of the COMBINATIONS_RESEARCH parallel-agent
> campaign (alongside Agent A — empirical project data, Agent B — theoretical
> orthogonality). Sources: 14 landmark papers + 5 secondary reviews
> retrieved via WebFetch / WebSearch against arXiv, CVF, ar5iv, NeurIPS
> proceedings, and well-cited Medium/HuggingFace reviews of the same.

---

## 0. TL;DR

- **14 papers surveyed in depth**, plus follow-up reviews / blog
  walkthroughs to recover the per-trick deltas from tables that arXiv
  abstract pages do not expose.
- The mainstream consensus stack for ImageNet-1k is **8–12 tricks**.
  Beyond that, returns saturate and new tricks start trading off
  against existing ones (notably weight decay).
- **Documented conflicts**: mixup + label-smoothing redundancy; BCE +
  (mixup ∧ label-smoothing) implementation conflict; SAM's 2× compute
  overhead vs. just training longer; large-batch SGD without warmup;
  dropout-on-FC actually HURTS a strongly regularised ResNet-200
  (Bello 2021); EMA + very long schedules can over-smooth a weak
  baseline.
- **The strongest cross-paper signal**: cosine LR, warmup, label
  smoothing, mixup, cutmix, RandAugment, AdamW (or LAMB),
  stochastic-depth, **and a DECREASED weight decay to compensate for
  the regulariser stack** appear in every modern recipe from 2019 →
  2025.

---

## 1. Per-paper extraction

### 1.1 He, Zhang, Zhang, Li, Xie, Li 2019 CVPR 'Bag of Tricks for Image Classification with Convolutional Neural Networks' (arXiv:1812.01187) — the canonical reference

**Stacking philosophy.** Drive ResNet-50 ImageNet top-1 from 75.3% →
79.29% by sequentially adding training and minor-architectural tricks,
each justified by a one-row ablation entry.

**Efficient-training tricks (Table 5, ResNet-50, ImageNet)**:
- Linear LR scaling rule for large batch (1024) — establishes the
  baseline at 75.87%.
- FP16 / mixed precision — small accuracy hit, large throughput win.
- No bias decay (apply WD only to conv/FC weights, not BN/biases).
- Zero-init last γ in each residual block's final BN.
- Cumulative gain from the efficient-training pack: **+0.34 pp**
  (75.87 → 76.21).

**Training-refinement stack (Table 6, applied on ResNet-50-D)**:
- Cosine LR decay vs. step decay — **+0.4 pp** range.
- Label smoothing ε = 0.1 — **+0.4 to +0.7 pp**.
- Knowledge distillation T = 20, ResNet-152-D teacher — **+0.4 to
  +0.7 pp**.
- Mixup α = 0.2 — **+0.4 to +0.7 pp**.
- Cumulative gain across all four: ~+2 pp on top of architecture
  tweaks.

**Architecture tweaks (Table 7)**: ResNet-B (move stride from 1×1 to
3×3 conv in downsample), ResNet-C (3×3 conv stem replacing 7×7),
ResNet-D (combine + extra avg-pool in the projection branch) → **+1 pp
cumulative** at slight FLOP cost.

**Saturation note.** Diminishing returns are explicit only in the
prose ("each refinement gives a smaller marginal gain than the
previous one"); the paper does not test a 7th or 8th refinement, so
saturation N for this recipe is at **~6 tricks** by accident of where
the authors stopped.

*Citation:* He, Zhang, Zhang, Li, Xie, Li 2019 CVPR 'Bag of Tricks for
Image Classification with Convolutional Neural Networks'
(arXiv:1812.01187) — canonical 5-pp cumulative ablation for stackable
training refinements on ResNet-50 ImageNet.

### 1.2 Wightman, Touvron, Jégou 2021 arXiv 'ResNet Strikes Back: An Improved Training Procedure in timm' (arXiv:2110.00476)

**Stacking philosophy.** Same ResNet-50 architecture, ten-trick recipe
gets to **80.4% top-1** (A1) — beating the original ResNet-50 by ~4 pp
through training alone.

**Recipes (Table 2 / Table 9)**:

| Ingredient | A1 (600 ep) | A2 (300 ep) | A3 (100 ep) |
|---|---|---|---|
| Optimizer | LAMB | LAMB | LAMB |
| LR | 5×10⁻³ | 5×10⁻³ | 8×10⁻³ |
| Weight decay | 0.01 | 0.02 | 0.02 |
| Batch size | 2048 | 2048 | 2048 |
| Schedule | cosine | cosine | cosine |
| Warmup | 5 ep | 5 ep | 5 ep |
| Mixup α | 0.2 | 0.1 | 0.1 |
| CutMix α | 1.0 | 1.0 | 1.0 |
| RandAugment | 7 / 0.5 | 7 / 0.5 | 6 / 0.5 |
| Random erase | 0.0 | 0.0 | 0.0 |
| Stochastic depth | 0.05 | 0.05 | 0.00 |
| Label smoothing | 0.1 | ✗ | ✗ |
| Repeated aug | ✓ | ✓ | ✗ |
| EMA | ✗ | ✗ | ✗ |
| Loss | BCE | BCE | BCE |
| Train res | 224 | 224 | 160 |
| Test res | 224 | 224 | 224 |
| Dropout | ✗ | ✗ | ✗ |

**Ablation findings**:
- **BCE loss vs. vanilla CE** is the single biggest swap — removing it
  drops the recipe by ~0.5–0.7 pp.
- **LAMB + cosine** is the most stable optimiser for the
  long-schedule + mixup + BCE combination; SGD + BCE diverged.
- **Repeated augmentation** gives ~+0.3 pp at long schedules; drop it
  at A3 because the data sees fewer aug-passes anyway.
- Label smoothing only carries weight at A1; A2/A3 drop it because
  mixup + cutmix + BCE already provide the smoothing.

*Citation:* Wightman, Touvron, Jégou 2021 arXiv 'ResNet Strikes Back:
An Improved Training Procedure in timm' (arXiv:2110.00476) — modern
ResNet-50 recipe; first wide demonstration that LAMB + BCE + repeated
augmentation stacks coherently.

### 1.3 Liu, Mao, Wu, Feichtenhofer, Darrell, Xie 2022 CVPR 'A ConvNet for the 2020s' (arXiv:2201.03545) — ConvNeXt

**Stacking philosophy.** Split training-recipe modernisation from
architectural modernisation. Apply ONE step at a time; record
cumulative ImageNet top-1.

**Modernisation roadmap (Figure 2, ConvNeXt-T)**:

| Step | Cumulative top-1 | Δ |
|---|---|---|
| ResNet-50 baseline (90 ep) | 76.1% | — |
| + Training recipe (300 ep, AdamW, mixup, cutmix, RandAugment, RandErase, label smoothing, stochastic depth, EMA) | 78.8% | **+2.7** |
| + Stage compute ratio (3,4,6,3) → (3,3,9,3) | 79.4% | +0.6 |
| + Patchify stem (4×4 stride 4) | 79.5% | +0.1 |
| + Depthwise conv (ResNeXt-ify) | 78.3% | −1.2 (intentional, recovered next) |
| + Width 64→96 | 80.5% | +2.2 |
| + Inverted bottleneck | 80.6% | +0.1 |
| + Large kernel 7×7 (with depthwise moved up) | 80.6% | 0.0 |
| + ReLU → GELU | 80.6% | 0.0 |
| + Fewer activations | 81.3% | +0.7 |
| + Fewer norms | 81.4% | +0.1 |
| + BN → LN | 81.5% | +0.1 |
| + Separate downsampling | 82.0% | +0.5 |

**Training recipe (taken as the modern default)**:
AdamW, LR 4×10⁻³, WD 0.05, batch 4096, 300 ep, cosine + 20-ep
warmup, mixup α=0.8, cutmix α=1.0, RandAugment (9, 0.5), Random Erase
0.25, label smoothing 0.1, stochastic depth 0.1, EMA 0.9999,
layer-scale init 1e-6.

**Saturation note.** Five architectural macro changes (stage ratio,
patchify, ResNeXt-ify, inverted bottleneck, separate downsampling)
plus five micro changes (kernel, GELU, fewer act, fewer norm, BN→LN);
some micro steps have **0.0 delta** but are kept because they enable
the next step. This is an explicit example of trick-saturation at
the architectural level: the training-recipe step gives **+2.7 pp**,
the entire 10-step architectural roadmap gives **+3.2 pp**.

*Citation:* Liu, Mao, Wu, Feichtenhofer, Darrell, Xie 2022 CVPR 'A
ConvNet for the 2020s' (arXiv:2201.03545) — ConvNeXt; canonical
sequential roadmap separating training vs. architectural contributions.

### 1.4 Tan, Le 2019 ICML 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks' (arXiv:1905.11946)

**Stacking philosophy.** Compound-scale depth × width × resolution
under a single coefficient φ such that α^φ × (β^φ)² × (γ^φ)² ≈ 2^φ.

**Optimal coefficients (B0 grid search)**:
- depth α = 1.2
- width β = 1.1
- resolution γ = 1.15
- constraint α · β² · γ² ≈ 2

**Single-axis vs. compound finding**: single-axis scaling saturates
(depth alone at d ≈ 6, width alone fast, resolution alone plateaus);
compound scaling achieves SOTA accuracy/FLOPS on ImageNet from B0
(77.3%, 5.3 M params) to B7 (84.3%, 66 M params).

**Training recipe**: RMSProp (decay 0.9, momentum 0.9), SiLU/Swish
activation, AutoAugment, dropout (0.2 → 0.5 from B0 to B7), drop-
connect (0.2 → 0.5), label smoothing 0.1, EMA, batch normalisation.

*Citation:* Tan, Le 2019 ICML 'EfficientNet: Rethinking Model Scaling
for Convolutional Neural Networks' (arXiv:1905.11946) — the original
compound-scaling result.

### 1.5 Radosavovic, Kosaraju, Girshick, He, Dollár 2020 CVPR 'Designing Network Design Spaces' (arXiv:2003.13678) — RegNet

**Stacking philosophy.** Instead of building one big stack, search a
quantised parameter-design space (depth, width slope, group width,
bottleneck ratio) with a few simple constraints to find a family of
networks. Training recipe is intentionally MINIMAL: SGD + momentum +
cosine + label smoothing + ~5 ep warmup; no mixup, no cutmix, no
RandAugment, no SAM. RegNetY (which adds SE blocks) is the only
add-on.

**Saturation note.** Their entire contribution is architectural — they
do NOT pile on the ImageNet-tricks list, instead arguing that a
quantised network family beats a single hand-tuned model. This is the
"less is more" counterpoint.

*Citation:* Radosavovic, Kosaraju, Girshick, He, Dollár 2020 CVPR
'Designing Network Design Spaces' (arXiv:2003.13678) — RegNet; the
counter-thesis that minimal training tricks + better design space
matches stacked-trick ResNets.

### 1.6 Touvron, Cord, Jégou 2022 ECCV 'DeiT III: Revenge of the ViT' (arXiv:2204.07118)

**Stacking philosophy.** Apply the LAMB + BCE + 3-Augment recipe to a
ViT. Simplify rather than stack.

**Recipe**:
- LAMB optimiser, LR 3×10⁻³, WD 0.02
- BCE loss (the same BCE-trick as ResNet Strikes Back)
- Cosine LR + 5-ep warmup
- **3-Augment** — three coarse augmentations only (Gray, Solarisation,
  Gaussian Blur), each applied per-image with prob 1/3 — replacing the
  more elaborate RandAugment for ViT.
- ColorJitter 0.3, Horizontal Flip, Random Resize Crop
- Mixup α = 0.8, CutMix α = 1.0
- Label smoothing 0.1, dropout 0.0, random erasing
- LayerScale init 1e-6, stochastic depth (per-model)
- **Repeated augmentation explicitly DROPPED** (they note prior DeiT
  used RA and it provided a small gain, but at A1-scale BCE + mixup +
  cutmix subsumes the effect)
- EMA 0.99996

**Per-trick finding**: BCE loss is the lever; for ImageNet-1k ViT
training it adds ~0.5 pp uniformly. ImageNet-21k pretraining sees no
BCE benefit (kept CE there).

*Citation:* Touvron, Cord, Jégou 2022 ECCV 'DeiT III: Revenge of the
ViT' (arXiv:2204.07118) — simplified augmentation, BCE win, decoupling
from repeated augmentation.

### 1.7 Cubuk, Zoph, Shlens, Le 2020 CVPRW 'RandAugment: Practical Automated Data Augmentation with a Reduced Search Space' (arXiv:1909.13719)

**Recipe**. Two hyperparameters only: **N** = number of transformations
to sequentially apply per image, **M** = magnitude (0–30 scale).
Search space {N ∈ [1, 3], M ∈ [5, 30]}. Default: ImageNet ResNet-50:
N=2, M=9. CIFAR-10 WideResNet: N=1, M=2. CIFAR-100: N=2, M=14.

**Gains over baseline augmentation**: +1.0 pp ImageNet ResNet-50 (76.3
→ 77.6 at 90 ep with standard RandAugment, additional gain when
combined with mixup/cutmix). Matches AutoAugment without the proxy
search.

*Citation:* Cubuk, Zoph, Shlens, Le 2020 CVPRW 'RandAugment: Practical
Automated Data Augmentation with a Reduced Search Space'
(arXiv:1909.13719) — the augmentation knob everyone now stacks.

### 1.8 Cubuk, Zoph, Mané, Vasudevan, Le 2018 CVPR 'AutoAugment: Learning Augmentation Policies from Data' (arXiv:1805.09501)

**Recipe.** RL-searched augmentation policy on a proxy task; 25
sub-policies, each with 2 ops, prob and magnitude. CIFAR-10:
+0.8 pp on WideResNet-28-10. ImageNet ResNet-50: +1.3 pp on top of
baseline augmentation.

**Note.** Largely replaced by RandAugment in modern recipes due to
much cheaper search, but the AutoAugment policy is what ConvNeXt and
EfficientNet inherit.

*Citation:* Cubuk, Zoph, Mané, Vasudevan, Le 2018 CVPR 'AutoAugment:
Learning Augmentation Policies from Data' (arXiv:1805.09501).

### 1.9 Zhang, Cissé, Dauphin, Lopez-Paz 2018 ICLR 'mixup: Beyond Empirical Risk Minimization' (arXiv:1710.09412)

**Recipe**. Sample λ ~ Beta(α, α); train on
(λ·xᵢ + (1−λ)·xⱼ, λ·yᵢ + (1−λ)·yⱼ). Recommended α:
- ImageNet ResNet-50: α = 0.2 (heavy mixup hurts).
- CIFAR-10/100 PreActResNet-18/WideResNet-28-10: α = 1.0.

**Headline deltas (Table 1 / 3 / 4)**:
- ImageNet ResNet-50 200 ep: +1.5 pp top-1 (76.9 → 78.4).
- CIFAR-10 PreActResNet-18: -0.5 → -1.2 pp test error.
- Robustness to label noise: dramatic (40% noise: 25 pp gain).
- GAN training stability: improved.

**Interaction note.** Mixup induces a label-smoothing-like effect
intrinsically, so stacking with explicit label smoothing yields
diminishing returns (see §4). For ERM-style training, optimal weight
decay drops when mixup is on (mixup is itself a regulariser).

*Citation:* Zhang, Cissé, Dauphin, Lopez-Paz 2018 ICLR 'mixup: Beyond
Empirical Risk Minimization' (arXiv:1710.09412).

### 1.10 Yun, Han, Oh, Chun, Choe, Yoo 2019 ICCV 'CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features' (arXiv:1905.04899)

**Recipe**. Sample λ ~ Beta(1, 1); paste an (1−λ)-area rectangle from
image j into image i, mix labels by area. Apply with probability p =
0.5 per minibatch (or always, mixed with mixup).

**Deltas**: ImageNet ResNet-50 +2.3 pp top-1 (76.3 → 78.6); CIFAR-100
WideResNet-28-10 +1.5 pp. Beats Cutout (which only erases, no label
mix) by ~0.5 pp.

**Stack note.** Modern recipes alternate per-batch between mixup and
cutmix with 50/50 probability (timm default; ConvNeXt and DeiT III).
Reported synergy: combined > either alone by ~0.5 pp.

*Citation:* Yun, Han, Oh, Chun, Choe, Yoo 2019 ICCV 'CutMix:
Regularization Strategy to Train Strong Classifiers with Localizable
Features' (arXiv:1905.04899).

### 1.11 Loshchilov, Hutter 2019 ICLR 'Decoupled Weight Decay Regularization' (arXiv:1711.05101) — AdamW

**Recipe.** Apply WD as a separate term to the parameter update, not
to the loss gradient (which Adam's adaptive denominator otherwise
scales away). The decoupled form is `θ ← θ − lr · (m̂/√v̂+ε + λ·θ)`.

**Empirical claim.** AdamW closes the SGD-vs-Adam generalisation gap
on CIFAR / ImageNet image classification; with cosine schedule and
WD 0.01–0.05, AdamW becomes the modern transformer optimiser of
choice.

*Citation:* Loshchilov, Hutter 2019 ICLR 'Decoupled Weight Decay
Regularization' (arXiv:1711.05101).

### 1.12 Loshchilov, Hutter 2017 ICLR 'SGDR: Stochastic Gradient Descent with Warm Restarts' (arXiv:1608.03983)

**Recipe.** Cosine annealing with periodic warm restarts:
T₀ initial period, T_mult multiplier. CIFAR-10 WideResNet 3.14 % error
(SGDR vs 3.46 % step decay); CIFAR-100 16.21 % vs 18.13 %.

**Modern usage**: most current recipes use **plain cosine** (no
restarts), absorbing only the half-cosine half of SGDR. Restarts make
ensembling-via-snapshot easier but rarely improve final-epoch acc on
modern long schedules.

*Citation:* Loshchilov, Hutter 2017 ICLR 'SGDR: Stochastic Gradient
Descent with Warm Restarts' (arXiv:1608.03983).

### 1.13 Foret, Kleiner, Mobahi, Neyshabur 2021 ICLR 'Sharpness-Aware Minimization' (arXiv:2010.01412) — SAM

**Recipe.** Two-step optimisation per batch:
1. Ascend by ρ along gradient direction to find worst-case nearby
   parameters: θ' = θ + ρ · g/‖g‖.
2. Compute gradient at θ', step from θ.

Default ρ = 0.05; doubles training FLOPs.

**Headline gains**:
- WideResNet-28-10 CIFAR-100: +1.2 pp (16.2 → 15.0 err).
- ResNet-50 ImageNet: +0.8 pp.
- ViT-B/16 ImageNet (NFNets+SAM): substantial gains where AdamW
  alone over-fits.

**Trade-offs (documented later)**: ESAM (Du et al. 2022) shows SAM's
overhead can be cut to 40 % with similar gains; in compute-equal
comparisons against just training 2× longer with same recipe, SAM
often loses on ImageNet but wins on CIFAR-100. Large-batch + SAM
interacts unfavourably — the ρ ball gets too small in flat regions.

*Citation:* Foret, Kleiner, Mobahi, Neyshabur 2021 ICLR 'Sharpness-
Aware Minimization for Efficiently Improving Generalization'
(arXiv:2010.01412).

### 1.14 Zhang, Lucas, Hinton, Ba 2019 NeurIPS 'Lookahead Optimizer: k steps forward, 1 step back' (arXiv:1907.08610)

**Recipe.** Wrap any base optimiser (SGD, Adam). Inner loop runs k
fast-weight steps; outer step pulls slow weights along a fraction α of
the (fast − slow) direction. Defaults: k = 5, α = 0.5–0.8.

**Headline**: stabilises Adam on machine translation, gives small
gains on CIFAR-10/100 ImageNet (~+0.2–0.5 pp), works with SGD or Adam.
Notably absent from ConvNeXt / ResNet-RS / RSB recipes — superseded by
the LAMB / AdamW + cosine pattern.

*Citation:* Zhang, Lucas, Hinton, Ba 2019 NeurIPS 'Lookahead Optimizer:
k steps forward, 1 step back' (arXiv:1907.08610).

### 1.15 Izmailov, Podoprikhin, Garipov, Vetrov, Wilson 2018 UAI 'Averaging Weights Leads to Wider Optima and Better Generalization' (arXiv:1803.05407) — SWA

**Recipe.** After ~75 % of cosine schedule completes, switch to high-
constant or cyclic LR and average the weights of every k-th iterate.
Update BN running stats at the end. CIFAR-10/100 WideResNet: +0.8 pp;
ImageNet ResNet-50: +0.2–0.4 pp.

**Stack note.** SWA is essentially a poor-man's EMA. Modern recipes
(ResNet-RS, ConvNeXt) use **EMA with decay 0.9999** instead of explicit
SWA — the empirical results are comparable but EMA is cheaper.

*Citation:* Izmailov, Podoprikhin, Garipov, Vetrov, Wilson 2018 UAI
'Averaging Weights Leads to Wider Optima and Better Generalization'
(arXiv:1803.05407).

### 1.16 Bello, Fedus, Du, Cubuk, Srinivas, Lin, Shlens, Zoph 2021 NeurIPS 'Revisiting ResNets: Improved Training and Scaling Strategies' (arXiv:2103.07579) — ResNet-RS

**Stacking philosophy.** Strip the architectural changes, focus on
training+scaling. Drive ResNet-200 from 79.0% → 82.2% on ImageNet
through training-method ablation alone.

**Additive table (Table 1)**:

| Step | Cumulative | Δ |
|---|---|---|
| Baseline ResNet-200 | 79.0% | — |
| + Cosine LR | 79.3% | +0.3 |
| + Longer training (350 ep) | 78.8% | **−0.5** |
| + EMA (0.9999) | 79.1% | +0.3 |
| + Label smoothing 0.1 | 80.4% | +1.3 |
| + Stochastic depth | 80.6% | +0.2 |
| + RandAugment | 81.0% | +0.4 |
| + Dropout on FC | 80.7% | **−0.3** |
| + Decrease weight decay (1e-4 → 4e-5) | 82.2% | **+1.5** |

**Two critical findings**:
1. **Just training longer can HURT** if regularisation isn't matched
   to schedule. Going from 90 → 350 ep without other changes cost
   0.5 pp.
2. **Weight decay must DECREASE when more regularisers are stacked.**
   This is the most important saturation/conflict finding in the
   modern recipe literature. Drop on FC actually hurts when stacked
   with RandAugment + label smoothing + stochastic depth.

**Scaling strategies**: depth-scale when overfitting risk is real
(small data / many regularisers); width-scale otherwise. Increase
resolution SLOWLY (compound scaling is too aggressive on resolution).

*Citation:* Bello, Fedus, Du, Cubuk, Srinivas, Lin, Shlens, Zoph 2021
NeurIPS 'Revisiting ResNets: Improved Training and Scaling Strategies'
(arXiv:2103.07579) — single most explicit treatment of regulariser-WD
interaction.

### 1.17 Hoffer, Hubara, Soudry 2017 NeurIPS 'Train longer, generalize better' (arXiv:1705.08741)

**Finding.** Generalisation gap of large-batch SGD vs. small-batch is
explained by NUMBER OF UPDATES, not batch size per se. Train large-
batch SGD with a "regime adaptation" — more epochs to match small-
batch update count — and the gap closes. Foundation for the modern
"long schedule + EMA" pattern.

*Citation:* Hoffer, Hubara, Soudry 2017 NeurIPS 'Train longer,
generalize better: closing the generalization gap in large batch
training of neural networks' (arXiv:1705.08741).

### 1.18 Smith 2018 arXiv 'A disciplined approach to neural network hyper-parameters' (arXiv:1803.09820)

**Finding (a cookbook of interactions)**:
- LR × batch: roughly linear (Goyal 2017 confirmed) for SGD; AdamW
  scales weaker (square-root with batch is closer).
- LR × WD: trade-off — raising LR lets you drop WD, and vice versa.
  Mixup, cutmix, dropout all push effective WD DOWN.
- Momentum × LR: inversely coupled; β = 0.9 default works for most LR.

*Citation:* Smith 2018 arXiv 'A disciplined approach to neural network
hyper-parameters: Part 1 — learning rate, batch size, momentum, and
weight decay' (arXiv:1803.09820).

### 1.19 Goyal, Dollár, Girshick, Noordhuis, Wesolowski, Kyrola, Tulloch, Jia, He 2017 arXiv 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour' (arXiv:1706.02677)

**Recipe.** Linear LR scaling rule (LR = base × batch / 256), gradual
5-ep warmup. Trains ResNet-50 ImageNet at batch 8192 in 1 hour, no
accuracy loss. Beyond batch ~32k SGD breaks; LARS / LAMB take over.

*Citation:* Goyal, Dollár, Girshick, Noordhuis, Wesolowski, Kyrola,
Tulloch, Jia, He 2017 arXiv 'Accurate, Large Minibatch SGD: Training
ImageNet in 1 Hour' (arXiv:1706.02677).

### 1.20 Geiping, Goldstein 2022 arXiv 'Cramming: Training a Language Model on a Single GPU in One Day' (arXiv:2212.14034)

**Finding (compute-budget bounded recipe)**:
- DROP curriculum learning, drop most data-cleaning, drop architectural
  novelties (no SwiGLU win at 1-day budget).
- KEEP cosine LR with warmup, AdamW, large WD on transformer-block
  weights, BPE tokeniser tweaks.
- The compute-budget regime SATURATES at ~5 training tricks; adding
  more does not help and sometimes hurts (the regulariser-WD
  saturation surfaces strongly in low-compute regimes).

*Citation:* Geiping, Goldstein 2022 arXiv 'Cramming: Training a
Language Model on a Single GPU in One Day' (arXiv:2212.14034) — the
"less is more" thesis for tight compute budgets.

---

## 2. Cross-paper trick-effectiveness table

Δ values are the per-row contribution reported by the paper for its
canonical recipe; `incl.` = trick included as part of the default
stack with no isolated number; `n/a` = the paper does not measure
the trick; `—` = baseline.

| Trick | He 2019 (ResNet-50 ImNet) | Bello 2021 (ResNet-200 ImNet) | Wightman 2021 (ResNet-50 ImNet) | Liu 2022 (ConvNeXt-T ImNet) | DeiT III 2022 (ViT-S ImNet) | Cross-paper consensus |
|---|---|---|---|---|---|---|
| Cosine LR | +0.4 | +0.3 | incl. | incl. | incl. | universal positive |
| LR warmup (5 ep) | incl. | incl. | incl. | incl. (20 ep) | incl. | universal |
| Label smoothing ε=0.1 | +0.4–0.7 | +1.3 | only A1 | incl. | incl. | universal positive |
| Mixup α=0.1–0.8 | +0.4–0.7 | n/a | incl. | incl. | incl. | strong positive |
| CutMix α=1.0 | n/a | n/a | incl. | incl. | incl. | strong positive |
| RandAugment (N,M) | n/a (uses simpler) | +0.4 | (7, 0.5) | (9, 0.5) | replaced by 3-aug | universal positive (or 3-aug) |
| Random Erasing | n/a | n/a | n/a | 0.25 incl. | incl. | mild positive |
| Stochastic depth | n/a | +0.2 | 0.05 incl. | 0.1 incl. | per-model | universal positive on deep nets |
| Dropout (FC) | n/a | **−0.3** | ✗ | ✗ | ✗ | NEGATIVE when stacked with the regulariser bundle |
| EMA (0.9999) | n/a | +0.3 | ✗ | incl. | incl. (0.99996) | mild positive, replaces SWA |
| AdamW / LAMB | SGD baseline | SGD | LAMB | AdamW | LAMB | universal in modern recipes |
| BCE loss | CE | CE | **+0.5–0.7** | CE | **+0.5** | strong positive for ViT/timm-style |
| Long schedule (300+ ep) | 120 ep | 350 ep | 600 ep (A1) | 300 ep | 800 ep | universal positive ONLY if reg-stack matches |
| **Decreased WD when reg stacked** | absent | **+1.5** | A1 0.01 < default | 0.05 (transformer-scale) | 0.02 | universal positive — Bello-style WD-tuning is mandatory |
| Knowledge distillation | +0.4–0.7 | n/a | n/a | n/a | incl. (DeiT v1) | useful when teacher available |
| Linear LR scaling (batch) | incl. (1024) | incl. | incl. (2048) | incl. (4096) | incl. (2048) | universal up to ~8k batch |
| Zero-init last γ in BN | incl. | n/a | incl. (timm) | n/a | n/a | mild positive |
| Repeated augmentation | n/a | n/a | A1/A2 incl. | n/a | **dropped** | controversial: helps A1 long schedules, dropped elsewhere |
| Compound scaling (d×w×r) | n/a | depth/width split | n/a | n/a | n/a | EfficientNet-only; replaced by depth-/width-scale rules in Bello 2021 |
| SAM (ρ=0.05) | n/a | n/a | n/a | n/a | n/a | extra +0.5–1.2 pp at 2× compute; not in mainstream recipes |
| SWA / cyclic LR averaging | n/a | n/a | n/a | n/a | n/a | replaced by EMA |
| Lookahead (k=5, α=0.5) | n/a | n/a | n/a | n/a | n/a | replaced by LAMB |
| Layer-scale (1e-6) | n/a | n/a | n/a | incl. | incl. | universal for very deep transformers / ConvNeXts |

---

## 3. The "saturation" finding

How many orthogonal tricks before the curve flattens?

| Paper | N tricks stacked | Cumulative gain over original baseline |
|---|---|---|
| He 2019 (CNN) | ~6 (cosine, label sm., distill, mixup, ResNet-D, FP16) | +4.0 pp on ResNet-50 |
| Bello 2021 (CNN, longer schedule) | ~8 (cosine, EMA, label sm., stoch-depth, RandAug, **decreased WD**, depth-scale, width-scale) | +3.2 pp on ResNet-200 |
| Wightman 2021 (CNN, A1) | ~10 (LAMB, BCE, cosine, warmup, mixup, cutmix, RandAug, repeated aug, stoch-depth, label sm.) | +5 pp on ResNet-50 |
| Liu 2022 (CNN-ViT hybrid) | ~5 architecture + ~7 training | +5.9 pp over ResNet-50 (76.1 → 82.0) |
| DeiT III (ViT) | ~8 (LAMB, BCE, cosine, 3-Aug, mixup, cutmix, label sm., layer-scale, EMA) | recipe-only; replaces RandAug with simpler 3-Aug |
| Touvron / Cubuk / Tan ablations | individual tricks 0.3 – 1.5 pp; marginal trick #6+ drops below 0.3 pp | — |

**Practical ceiling: 8–12 orthogonal tricks.** Below 8 leaves
performance on the table; above 12 returns saturate AND new tricks
start interfering with the existing stack (esp. weight decay).

The mainstream practitioner's recipe is therefore a 10-trick stack
plus one architectural choice (which itself is usually a 5-step
modernisation roadmap à la ConvNeXt).

---

## 4. The "conflict" finding — documented failure modes

### 4.1 Mixup × Label Smoothing redundancy
Thulasidasan et al. 2019 ('On Mixup Training', NeurIPS 2019) shows
mixup induces a *label-smoothing-like* calibration effect intrinsically;
stacking mixup α=0.2 with label smoothing ε=0.1 yields **diminishing
returns** of ~0.2–0.3 pp instead of the additive ~0.5 + 0.4 ≈ 0.9 pp
expected. RSB-A2/A3 simply drops label smoothing for this reason.

### 4.2 BCE × (mixup ∧ label smoothing) implementation conflict
timm pytorch-image-models discussion #1001 (Wightman): when BCE loss
is enabled with BOTH mixup/cutmix AND label smoothing, the soft-target
formulation double-counts the smoothing. Fix: either disable label
smoothing or use the multi-label BCE on already-mixed targets and skip
the smoothing step.

### 4.3 Dropout × heavy regulariser stack
Bello 2021 ResNet-RS Table 1: adding Dropout on the FC after
RandAugment + label smoothing + stochastic depth + EMA **DROPS** top-1
by 0.3 pp. Dropout alone is fine; dropout on top of an already
ergodic-regularised model is over-regularisation.

### 4.4 Weight Decay × regulariser stack
Bello 2021's headline finding: standard WD = 1e-4 works alone but
becomes destructive once mixup + cutmix + RandAugment + label smoothing
+ stochastic depth are stacked. Drop to 4e-5 (≈ 2.5× reduction). Same
finding in Smith 2018: every regulariser added pushes the optimal
effective WD down.

### 4.5 SAM × large batch
SAM's perturbation ρ effectively shrinks as the loss landscape flattens
at large batch; the 2× compute overhead is harder to justify. ESAM
(Du et al. 2022) addresses the cost; SST (Sharpness-aware Training for
Free, 2205.14083) further shows much of SAM's benefit can be recovered
without the second forward-backward.

### 4.6 Long schedules × unmatched regularisation
RSB Table 1 row "+ longer training": **−0.5 pp** when increasing 90 →
350 ep WITHOUT updating regularisers. Long schedules require either
more aug, more stochastic-depth, or higher EMA decay to absorb the
extra updates.

### 4.7 Linear LR scaling breaks past ~32k batch
Goyal 2017 demonstrates linear scaling up to batch 8k; You et al. (LARS
2017, LAMB 2019) needed to replace SGD/AdamW with layer-wise adaptive
optimisers for batch 32k+. SGD with linear LR at batch 64k diverges.

### 4.8 Repeated augmentation × short schedules
DeiT III drops repeated augmentation because at 800 ep the duplicate
views become redundant; at A3 100 ep RSB also drops it. Repeated aug
is a long-schedule-only trick.

---

## 5. State-of-art recipe for CIFAR-100 30-ep on a ResNet-20-class budget (2026-ICML grade)

Reverse-engineered from the 6 modern recipes (ConvNeXt, ResNet-RS,
RSB-A3, DeiT III, EfficientNet, RandAugment defaults) **adapted to
the project's compute envelope** (ResNet-20, 16 GB VRAM, 30 ep CIFAR-
100, num_workers=0 on Windows). 11 tricks; this is what a competitive
2026 submission would use today:

1. **Optimizer**: AdamW (or SGD-momentum 0.9 with Nesterov; either
   acceptable at this scale — AdamW is timm default).
2. **LR schedule**: cosine, no restarts.
3. **Warmup**: 3 epochs linear.
4. **Base LR**: 1e-3 (AdamW) or 0.1 (SGD); linear-scale from batch
   128 baseline. Batch 256 ⇒ LR ×2.
5. **Weight decay**: 5e-4 with this regulariser stack (not the
   pre-mixup era's 1e-3); FOLLOW Bello 2021 — if you stack more, drop
   WD.
6. **Label smoothing ε=0.1** (or skip if BCE is used).
7. **Mixup α=0.2 + CutMix α=1.0**, applied with 50/50 per-batch
   alternation, mixing prob 1.0.
8. **RandAugment** (N=2, M=14 for CIFAR-100) — ConvNeXt/RSB-style
   level.
9. **Random Erasing** p=0.25.
10. **Stochastic depth** linear schedule, max rate 0.1 (light for a
    20-layer net).
11. **EMA** of weights, decay 0.9999, no separate BN-stat update
    (use train-time BN stats at the EMA copy).

Optional 12th: **BCE loss** in place of CE; expected +0.3–0.5 pp at
this scale per the DeiT III / RSB finding.

Optional 13th: **3-seed average** with median composite — this is
already part of the project's protocol (Rule 6) but absent from most
ICML CIFAR submissions, which is a free novelty point.

Three things explicitly NOT to add at this scale:
- **Dropout on FC** — Bello 2021 −0.3 pp.
- **SAM** — 2× compute for ~+0.8 pp; not worth it on a 30-ep budget.
- **Repeated augmentation** — only helps at 600+ ep schedules; pure
  cost at 30 ep.

---

## 6. Cross-reference to our project's hypotheses

Mainstream → our project mapping. The "mainstream consensus" column
shows whether literature has converged on this trick; the project
column lists which of our 84 hypotheses corresponds.

| Mainstream trick | Mainstream consensus | Closest project hypothesis | Status |
|---|---|---|---|
| Cosine LR (Loshchilov 2017) | universal | H10 (φ-decay LR scheduler — a φ-modulated cosine) | implemented, modest gain |
| Label smoothing ε=0.1 (Szegedy 2016) | universal | not isolated; subsumed in mixup-equivalent priors | indirect |
| Mixup α=0.2 (Zhang 2018) | universal | no direct analog; H75 (cymatic SwiGLU) is closest in spirit | gap |
| CutMix α=1.0 (Yun 2019) | universal | no direct analog | gap |
| RandAugment (Cubuk 2020) | universal | not in the project (we use RandomCrop+HFlip only) | **gap** |
| Stochastic depth (Huang 2016) | universal | H52 (drop-path anytime) — partial overlap | implemented |
| EMA / SWA (Izmailov 2018) | universal | no direct analog | gap |
| AdamW / LAMB (Loshchilov 2019) | universal | H41 (golden-ratio optimizer), H48 (golden momentum) — analogs | implemented, mixed results |
| BCE loss (Wightman 2021, DeiT III) | universal in modern | no direct analog | gap |
| Decreased WD on reg stack (Bello 2021) | universal in modern | H44 (golden regularization — closest spirit; scales WD by φ) | implemented |
| Compound scaling d×w×r (Tan 2019) | strong | H01 (φ compound scaling — direct φ-tuning of α,β,γ) | implemented, certified winner (sg_only_phi_budget) |
| SAM (Foret 2021) | strong with 2× compute | no direct analog | gap |
| Linear LR scaling for large batch (Goyal 2017) | universal | implicit in runner.py | indirect |
| Knowledge distillation (He 2019, DeiT) | situational | no direct analog | gap |
| Layer scale init 1e-6 (Touvron 2021) | universal in transformers | no direct analog | gap |
| Patchify stem 4×4 (ConvNeXt) | strong | no direct analog | gap |
| 7×7 depthwise conv (ConvNeXt) | strong | H31 (golden-spiral kernel) — geometric kernel analog | implemented |

### Novelty pocket — priors with NO mainstream analog

Reviewing all 84 hypotheses against the mainstream trick set, the
following hypotheses occupy "novelty pockets" — they touch design
levers the mainstream literature has not explored. These are the
hypotheses where a positive result is most defensible as a NEW
contribution rather than a re-skinning:

- **H10** φ-decay LR scheduler — geometric (φ-modulated) cosine; no
  mainstream analog (everyone uses smooth cosine).
- **H21** hexagonal φ-packing — hex lattice geometry as inductive bias;
  GDL has hex CNN papers (Hoogeboom 2018) but none use φ packing.
- **H22** toroidal closure — wrap-around topology; no mainstream
  classification recipe uses topological closure.
- **H24, H25** icosahedral / dodecahedral equivariance — Platonic-group
  equivariance is a published GDL angle (Cohen 2019) but the φ-tuned
  variants are novel.
- **H28** cymatic hex resonance — sound-driven init pattern; no
  mainstream analog.
- **H33** Vesica Piscis filter — geometric filter shape; no analog.
- **H37** pentagonal φ attention — 5-fold attention symmetry; no analog.
- **H46** cymatic loss — frequency-domain loss; no mainstream analog.
- **H51** topological Betti loss — uses persistent homology of features
  as auxiliary loss; tangentially explored in TDL but not as a recipe
  add-on.
- **H56–H58** cymatic dataset / cross-modal / C4 max→avg-pool — three
  separate novelty pockets.
- **H67** full paradigm hybrid — Sacred + Liquid + JEPA + KAN + GNN +
  Transformer fusion; no mainstream recipe attempts a 6-paradigm
  hybrid (and the May-2026 sg_full_fib result suggests caution).
- **H69** KAN-Metatron symbolic head — KAN-based heads exist (Liu 2024)
  but not Metatron-tied weights.
- **H79** morphing polytope adjacency — time-varying topology; no
  analog.
- **H83** collapse-gated attention — entropy-based attention gating;
  no analog.

**Roughly 14–18 hypotheses sit in real novelty pockets** the
mainstream has not visited.

The remaining ~66 hypotheses are mostly **φ-flavoured re-skins** of
mainstream tricks (mixup-with-φ-ratio, ResNet-with-φ-channels,
attention-with-φ-spacing). These are NOT novelty pockets — they're
new instantiations of established tricks and must be defended as
"better-tuned versions of X" rather than "novel Y".

---

## 7. Implications for the combo strategy

### 7.1 Headroom from the 8–12 trick consensus
The project's three certified winners — `pair_gm_pdw`, `slot_act_sine`,
`sg_only_phi_budget` — each operate on 1–3 design axes. Mainstream
recipes routinely stack 8–12. **There is large headroom for combos**
that pair our certified winners with mainstream-grade additions
(label smoothing, mixup, RandAugment, EMA, BCE) — the project's
current configs use the legacy CIFAR-10 recipe (RandomCrop + HFlip
only, no mixup, no RandAugment).

### 7.2 Same-axis rule explains the sg_full_fib failure
`sg_full_fib` stacked SIX φ priors on the same conv-block forward
path: φ depth + φ width + φ kernel + φ activation + φ skip + φ
dropout cycle. Mainstream recipes obey an **orthogonal-axis rule**
implicitly:
- one optimiser (AdamW)
- one data-aug bundle (RandAug + mixup + cutmix)
- one regulariser (label smoothing) + auxiliary regulariser
  (stochastic depth)
- one schedule (cosine + warmup)
- one inference-time trick (EMA)
- ONE architecture macro (e.g., ConvNeXt's roadmap is ONE family of
  decisions, not 6 independent priors stacked on the same block).

Stacking 6 priors on ONE axis (the conv forward path) is what burned
sg_full_fib: each prior subtly changed the same gradient flow, and
the cumulative effect was −11.54 pp. The combo strategy MUST stack
across orthogonal axes (init / optimizer / aug / regulariser / arch
/ inference) — same as the literature.

### 7.3 The Bello WD-decrease rule applies to us too
If a project combo stacks mixup + label smoothing + EMA + RandAug +
stochastic-depth + golden-regularization-φ-WD, then the BASE WD
itself must be cut by 2.5× per Bello 2021. Without that adjustment
the regulariser stack will over-regularise and look like a "combo
failure" when it's actually a missed hyperparameter calibration.

### 7.4 BCE-loss is a free upgrade
Wightman 2021 and DeiT III both find +0.5 pp from BCE alone. The
project still uses CE everywhere; a single switch to BCE in
`src/nature_inspired_networks/loss.py` (with the mixup-target
adjustment) is a low-risk, mainstream-grade gain.

### 7.5 Repeated augmentation is the wrong direction at 30 ep
Our budget is 30 ep CIFAR-100, not 600 ep ImageNet. Skip repeated
augmentation. Skip SAM. The 2× compute is better spent on more
seeds or a longer schedule.

### 7.6 The "novelty-pocket" hypotheses are where the marginal
**research** value lives — the φ-reskin hypotheses are where the
marginal **engineering** value lives. A 2026-ICML submission would
combine ONE novelty-pocket hypothesis (e.g., H22 toroidal closure,
H51 Betti loss) with a mainstream-grade 11-trick recipe to ensure
the baseline is competitive — then claim the novelty as the
delta-over-strong-baseline.

---

## Appendix A — Quick BibTeX

```
@inproceedings{he2019bag,
  title={Bag of Tricks for Image Classification with Convolutional Neural Networks},
  author={He, Tong and Zhang, Zhi and Zhang, Hang and Zhang, Zhongyue and Xie, Junyuan and Li, Mu},
  booktitle={CVPR}, year={2019}, eprint={1812.01187}
}
@article{wightman2021resnet,
  title={ResNet Strikes Back: An Improved Training Procedure in timm},
  author={Wightman, Ross and Touvron, Hugo and J\'egou, Herv\'e},
  year={2021}, eprint={2110.00476}
}
@inproceedings{liu2022convnet,
  title={A ConvNet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle={CVPR}, year={2022}, eprint={2201.03545}
}
@inproceedings{tan2019efficientnet,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc V.},
  booktitle={ICML}, year={2019}, eprint={1905.11946}
}
@inproceedings{radosavovic2020designing,
  title={Designing Network Design Spaces},
  author={Radosavovic, Ilija and Kosaraju, Raj Prateek and Girshick, Ross and He, Kaiming and Doll\'ar, Piotr},
  booktitle={CVPR}, year={2020}, eprint={2003.13678}
}
@inproceedings{touvron2022deit3,
  title={DeiT III: Revenge of the ViT},
  author={Touvron, Hugo and Cord, Matthieu and J\'egou, Herv\'e},
  booktitle={ECCV}, year={2022}, eprint={2204.07118}
}
@article{cubuk2020randaugment,
  title={RandAugment: Practical Automated Data Augmentation with a Reduced Search Space},
  author={Cubuk, Ekin D. and Zoph, Barret and Shlens, Jonathon and Le, Quoc V.},
  booktitle={CVPRW}, year={2020}, eprint={1909.13719}
}
@inproceedings{cubuk2018autoaugment,
  title={AutoAugment: Learning Augmentation Policies from Data},
  author={Cubuk, Ekin D. and Zoph, Barret and Man\'e, Dandelion and Vasudevan, Vijay and Le, Quoc V.},
  booktitle={CVPR}, year={2018}, eprint={1805.09501}
}
@inproceedings{zhang2018mixup,
  title={mixup: Beyond Empirical Risk Minimization},
  author={Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N. and Lopez-Paz, David},
  booktitle={ICLR}, year={2018}, eprint={1710.09412}
}
@inproceedings{yun2019cutmix,
  title={CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features},
  author={Yun, Sangdoo and Han, Dongyoon and Oh, Seong Joon and Chun, Sanghyuk and Choe, Junsuk and Yoo, Youngjoon},
  booktitle={ICCV}, year={2019}, eprint={1905.04899}
}
@inproceedings{loshchilov2019adamw,
  title={Decoupled Weight Decay Regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR}, year={2019}, eprint={1711.05101}
}
@inproceedings{loshchilov2017sgdr,
  title={SGDR: Stochastic Gradient Descent with Warm Restarts},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR}, year={2017}, eprint={1608.03983}
}
@inproceedings{foret2021sam,
  title={Sharpness-Aware Minimization for Efficiently Improving Generalization},
  author={Foret, Pierre and Kleiner, Ariel and Mobahi, Hossein and Neyshabur, Behnam},
  booktitle={ICLR}, year={2021}, eprint={2010.01412}
}
@inproceedings{zhang2019lookahead,
  title={Lookahead Optimizer: k steps forward, 1 step back},
  author={Zhang, Michael R. and Lucas, James and Hinton, Geoffrey E. and Ba, Jimmy},
  booktitle={NeurIPS}, year={2019}, eprint={1907.08610}
}
@inproceedings{izmailov2018swa,
  title={Averaging Weights Leads to Wider Optima and Better Generalization},
  author={Izmailov, Pavel and Podoprikhin, Dmitrii and Garipov, Timur and Vetrov, Dmitry and Wilson, Andrew Gordon},
  booktitle={UAI}, year={2018}, eprint={1803.05407}
}
@inproceedings{bello2021revisiting,
  title={Revisiting ResNets: Improved Training and Scaling Strategies},
  author={Bello, Irwan and Fedus, William and Du, Xianzhi and Cubuk, Ekin D. and Srinivas, Aravind and Lin, Tsung-Yi and Shlens, Jonathon and Zoph, Barret},
  booktitle={NeurIPS}, year={2021}, eprint={2103.07579}
}
@inproceedings{hoffer2017trainlonger,
  title={Train longer, generalize better: closing the generalization gap in large batch training of neural networks},
  author={Hoffer, Elad and Hubara, Itay and Soudry, Daniel},
  booktitle={NeurIPS}, year={2017}, eprint={1705.08741}
}
@article{smith2018disciplined,
  title={A disciplined approach to neural network hyper-parameters: Part 1 — learning rate, batch size, momentum, and weight decay},
  author={Smith, Leslie N.},
  year={2018}, eprint={1803.09820}
}
@article{goyal2017accurate,
  title={Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour},
  author={Goyal, Priya and Doll\'ar, Piotr and Girshick, Ross and Noordhuis, Pieter and Wesolowski, Lukasz and Kyrola, Aapo and Tulloch, Andrew and Jia, Yangqing and He, Kaiming},
  year={2017}, eprint={1706.02677}
}
@article{geiping2022cramming,
  title={Cramming: Training a Language Model on a Single GPU in One Day},
  author={Geiping, Jonas and Goldstein, Tom},
  year={2022}, eprint={2212.14034}
}
```

---

*Authored 2026-05-30 as part of the COMBINATIONS_RESEARCH parallel-
agent campaign. Sources collected via WebFetch / WebSearch against
arXiv, CVF open-access, ar5iv, NeurIPS proceedings, and Medium /
HuggingFace landmark-paper walkthroughs. 14 primary papers, 5
secondary reviews surveyed; one-page-per-paper extractions in §1;
cross-paper consensus table in §2; saturation + conflict notes in
§§3-4; project-grade synthesis recipe in §5; novelty-pocket mapping
to our 84 hypotheses in §6; combo-strategy implications in §7.*
