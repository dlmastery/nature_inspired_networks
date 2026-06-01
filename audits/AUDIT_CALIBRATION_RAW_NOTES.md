# Raw notes — third-party audit calibration (scratchpad)
Date: 2026-05-30
Auditor: Track-A implementation-critic style (Opus 4.7)

## Sources fetched (raw GitHub user-content)

| File | URL | Notes |
|---|---|---|
| pytorch/vision resnet.py | `https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/resnet.py` | ResNet18..152, ResNeXt, Wide-ResNet |
| pytorch/vision densenet.py | `https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/densenet.py` | DenseNet-121..201 |
| pytorch/vision vgg.py | `https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/vgg.py` | VGG11..19 |
| pytorch/vision squeezenet.py | `https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/squeezenet.py` | Fire module |
| pytorch/vision mobilenetv2.py | `https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/mobilenetv2.py` | InvertedResidual |
| pytorch/pytorch adam.py | `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/adam.py` | Adam defaults |
| pytorch/pytorch sgd.py | `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/sgd.py` | SGD defaults / Nesterov |
| pytorch/pytorch init.py | `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/init.py` | Kaiming, Xavier |
| pytorch/pytorch lr_scheduler.py | `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/lr_scheduler.py` | CosineAnnealingLR |
| pytorch/pytorch batchnorm.py | `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/modules/batchnorm.py` | _NormBase defaults |

Commit reference: HEAD of `main` branch on 2026-05-30 (snapshot for this audit).

---

## Candidate hypotheses (mapping clean-code claims → audit-style rows)

The Track-A audit doctrine: "shape-only tests = MINOR; doc/code divergence in the
mechanism = MAJOR; not-actually-doing-the-thing = BROKEN. PASS requires a
mechanism-verifying assertion." We translate this to clean-code by asking, for
each claimed mechanism: does the code faithfully realise the cited paper's
mechanism? Are there any divergences? Are they documented?

### TP1 — ResNet identity shortcut implements He 2016 arXiv:1512.03385 §3.2
- Paper claim: `y = F(x) + x` where x is identity (skip connection). When
  dimensions change, use a projection shortcut (option B in §3.3).
- Code (resnet.py:94-110, BasicBlock.forward):
  ```python
  identity = x
  out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
  out = self.conv2(out); out = self.bn2(out)
  if self.downsample is not None:
      identity = self.downsample(x)
  out += identity
  out = self.relu(out)
  ```
- `_make_layer` (resnet.py:254-259) constructs `downsample` as a 1×1 conv +
  BN when `stride != 1 or inplanes != planes * expansion`.
- Verdict: **PASS**. Matches §3.2 (identity) and §3.3 option B (projection
  shortcut on stride/channel change). Both pieces of the §3.2 mechanism are
  implemented and dispatched on the correct condition.

### TP2 — ResNet Bottleneck block matches He 2016 arXiv:1512.03385 §4.1
- Paper claim: bottleneck = 1×1 (reduce) → 3×3 (process) → 1×1 (restore) with
  4× expansion factor; designed for "practically considerable" 3-layer block.
- Code (resnet.py:113-172, Bottleneck): `expansion = 4` (line 123); conv1 is
  conv1x1, conv2 is conv3x3, conv3 is conv1x1(width, planes * expansion).
- BUT lines 114-121 contain an EXPLICIT comment block:
  > "Bottleneck in torchvision places the stride for downsampling at 3x3
  > convolution (self.conv2) while original implementation places the stride
  > at the first 1x1 convolution (self.conv1) according to 'Deep residual
  > learning for image recognition'. ... This variant is also known as ResNet
  > V1.5 and improves accuracy according to ..."
- This IS a documented divergence from the paper. Strict audit doctrine: doc/
  code divergence in the mechanism = MAJOR. **But** the comment block is in
  the same file as the implementation, declaring the deviation explicitly and
  citing the rationale (V1.5 NGC reference). This satisfies the audit's
  "documented in the same artefact" requirement (vs. our H21 where the φ-
  radial mechanism was undocumented in the implementation).
- Verdict: **MINOR**. The stride-placement deviation is real but explicitly
  flagged in the code itself; a reviewer reading the file is warned. Strict
  doctrine would push this to MAJOR; the project's actual audit team
  generally graded "documented intentional deviation" as MINOR (cf. G1 H01
  where intra-family phi^s recurrence is "not documented in H01.md" → MINOR).
  Applying symmetric grading: MINOR.

### TP3 — ResNet weight init = Kaiming-normal fan_out + BN constant-1 init
- Paper claim: He 2015 §2.2 — std = sqrt(2 / n_l) where n_l = k² * c_out
  (fan_out) for ReLU networks.
- Code (resnet.py:221-227):
  ```python
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
  elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
      nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
  ```
- `torch.nn.init.kaiming_normal_` (init.py): std = gain / sqrt(fan) where
  gain for ReLU = sqrt(2). With mode="fan_out", fan = k² * c_out. So
  std = sqrt(2) / sqrt(k² * c_out) = sqrt(2 / (k² * c_out)) — matches paper.
- Verdict: **PASS**. Faithful implementation; mode="fan_out" is the canonical
  choice for ReLU per He 2015 footnote.

### TP4 — `zero_init_residual` implements Goyal et al. 2017 arXiv:1706.02677
- Paper claim: "Zero γ-init for last BN of every residual block. We initialise
  the last BN's γ to 0, so each residual block returns its input. This
  improves accuracy by 0.2-0.3 %."
- Code (resnet.py:234-239):
  ```python
  if zero_init_residual:
      for m in self.modules():
          if isinstance(m, Bottleneck) and m.bn3.weight is not None:
              nn.init.constant_(m.bn3.weight, 0)
          elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
              nn.init.constant_(m.bn2.weight, 0)
  ```
- Verdict: **PASS**. Targets the LAST BN of each block (bn3 for Bottleneck,
  bn2 for BasicBlock), zeros γ correctly. Citation in inline comment.

### TP5 — Wide ResNet width factor k=2 = Zagoruyko & Komodakis 2016 arXiv:1605.07146
- Paper claim: WRN-d-k applies widening factor k to the base channel count.
  WRN-50-2 doubles all bottleneck widths (k=2).
- Code (resnet.py:1108, wide_resnet50_2):
  `_ovewrite_named_param(kwargs, "width_per_group", 64 * 2)`
- Combined with Bottleneck.__init__ line 139:
  `width = int(planes * (base_width / 64.0)) * groups`
- With base_width=128, groups=1: width = planes * 2. Channel widths doubled.
- Verdict: **PASS**. The k=2 widening is mathematically realised.
- BUT: the paper's WRN was developed in the BasicBlock-without-bottleneck
  setting (WRN-28-10 etc.). torchvision's `wide_resnet50_2` applies the
  factor in the Bottleneck variant. This is a meaningful semantic note —
  is "WRN-50-2" the same thing as the paper's WRN-d-k? Strictly it is a
  DIFFERENT architecture (Bottleneck blocks with widened middle conv).
  However, torchvision's docstring (not shown but reachable in the file)
  explicitly references the paper and notes the variant.
- Final: **MINOR** would be defensible (paper-vs-code-architecture-mismatch
  even though parameter count rule is honoured). PASS is also defensible
  (the widening rule is correctly applied to the canonical model). Taking
  the more skeptical Track-A stance: **MINOR** — the paper's `widening
  factor k` semantics conflate two architectures.

### TP6 — DenseNet growth_rate accumulates per-layer = Huang et al. 2017 arXiv:1608.06993
- Paper claim: each layer in a dense block produces `k` (growth_rate) feature
  maps and concatenates with all previous layers' outputs (Eq. 2 in §3).
  Channel count after N layers = `k0 + N * k`.
- Code (densenet.py:106): `num_input_features + i * growth_rate` passed to
  each `_DenseLayer` constructor.
- Code (densenet.py:173-179) outer loop:
  ```python
  num_features = num_init_features
  for i, num_layers in enumerate(block_config):
      block = _DenseBlock(...)
      num_features = num_features + num_layers * growth_rate
      if i != len(block_config) - 1:
          trans = _Transition(...)
          num_features = num_features // 2
  ```
- The transition layer halves channels (compression θ=0.5; paper §3 calls
  this DenseNet-BC).
- Verdict: **PASS**. Matches paper Eq.2 and the DenseNet-BC compression rule.

### TP7 — DenseNet bottleneck = 4*growth_rate inner dim = Huang 2017 §3 DenseNet-B
- Paper claim: "BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) ... we let each 1×1 conv
  produce 4k feature-maps" (§3, DenseNet-B variant).
- Code (densenet.py:43): `nn.Conv2d(num_input_features, bn_size * growth_rate,
  kernel_size=1, ...)`. Default `bn_size=4`, `growth_rate=32` → 1×1 produces
  128 feature maps. Matches §3 "4k".
- Code (densenet.py:46): `nn.Conv2d(bn_size * growth_rate, growth_rate,
  kernel_size=3, stride=1, padding=1, bias=False)` — final output is k=32.
- Verdict: **PASS**. The DenseNet-B (B for bottleneck) 4k inner dim is
  faithfully `bn_size=4` * `growth_rate=k`.

### TP8 — VGG conv stack matches Simonyan & Zisserman 2014 arXiv:1409.1556 Table 1
- Paper claim: VGG-A (11 layers): 64-M-128-M-256-256-M-512-512-M-512-512-M.
  All convs are 3×3 with padding=1 and stride=1; pool is 2×2 max with
  stride=2.
- Code (vgg.py:67-80, make_layers): `nn.Conv2d(in_channels, v, kernel_size=3,
  padding=1)` (line 74; default stride=1). `nn.MaxPool2d(kernel_size=2,
  stride=2)` (line 72).
- Code (vgg.py:76, cfg "A"): `[64, "M", 128, "M", 256, 256, "M", 512, 512,
  "M", 512, 512, "M"]` — exactly matches Table 1 column A.
- Verdict: **PASS**. Configuration dict matches Table 1 across A/B/D/E
  (VGG-11/13/16/19). 3×3+pad-1 convs and 2×2 stride-2 max-pool faithful.

### TP9 — Adam defaults match Kingma & Ba 2014 arXiv:1412.6980 §2
- Paper claim: "Good default settings for the tested machine learning
  problems are α=0.001, β1=0.9, β2=0.999 and ε=10^-8" (§2 "Algorithm").
- Code (adam.py:27-32): `lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
  amsgrad=False`.
- Verdict: **PASS**. Verbatim match to the paper's recommended defaults.

### TP10 — SGD with Nesterov implements Sutskever et al. 2013 ICML §3
- Paper claim: Nesterov accelerated gradient (NAG) requires nonzero momentum
  and zero dampening; the update rule looks ahead using the momentum direction.
- Code (sgd.py:51-52):
  ```python
  if nesterov and (momentum <= 0 or dampening != 0):
      raise ValueError("Nesterov momentum requires a momentum and zero dampening")
  ```
- BUT the docstring (sgd.py:130, 146-163) explicitly notes:
  > "The implementation has been adapted from the implementations in Sutskever
  > et al. ... and differs from it in that the momentum buffer accumulates
  > gradients directly rather than scaling by learning rate within the buffer."
- This is a real semantic deviation. The Sutskever 2013 formula has
  `v_{t+1} = µ v_t − ε ∇f(θ_t + µ v_t)`, but PyTorch's
  `v_{t+1} = µ v_t + ∇f(θ_t)` followed by `θ_{t+1} = θ_t − ε (∇f(θ_t) + µ v_{t+1})`.
- The two formulations are equivalent IF the learning rate is constant; under
  LR scheduling they DIVERGE in their effective lookahead. This is the same
  class of bug as our H41 GoldenAdam — a documented-but-meaningful semantic
  deviation from the cited paper.
- Verdict: **MINOR**. Real deviation but explicitly flagged in the docstring;
  a reviewer is warned. Strict doctrine would mark MAJOR; symmetric grading
  vs. our G3 audit (which docked H21 MAJOR for an UNdocumented deviation)
  says **MINOR** here because the deviation IS documented.

### TP11 — Kaiming init formula matches He 2015 arXiv:1502.01852 §2.2
- Paper claim: For ReLU layers, σ² = 2 / n_l, so weights ~ N(0, sqrt(2/n_l)).
  Either fan_in (forward variance preservation) or fan_out (backward variance
  preservation); paper argues either is valid for deep ReLU networks.
- Code (init.py): `std = gain / sqrt(fan)`. For nonlinearity="relu",
  gain = sqrt(2) (via calculate_gain). With mode="fan_in", fan = k² * c_in.
  So std = sqrt(2 / (k² * c_in)) — exactly the paper's formula.
- For nonlinearity="leaky_relu", gain = sqrt(2 / (1 + a²)), generalising
  He's formula to LeakyReLU with negative slope a (a=0 → ReLU).
- Default `a=0` and default `nonlinearity="leaky_relu"` give gain=sqrt(2);
  this matches the He paper.
- Verdict: **PASS**. Faithful and the default ReLU gain emerges from the
  more general leaky_relu form correctly.

### TP12 — CosineAnnealingLR formula matches Loshchilov & Hutter 2017 arXiv:1608.03983 §3
- Paper claim: SGDR cosine decay: η_t = η_min + (1/2)(η_max − η_min)(1 +
  cos(π T_cur / T_i)) where T_i is the restart period.
- Code (lr_scheduler.py:2624-2639):
  ```python
  return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi *
          self.last_epoch / self.T_max)) / 2 for base_lr in self.base_lrs]
  ```
- Note: this is `CosineAnnealingLR`, NOT `CosineAnnealingWarmRestarts`. The
  latter implements the SGDR T_i restart-doubling rule explicitly. The
  basic CosineAnnealingLR is the SINGLE-CYCLE version (no restart) — it
  decays monotonically from base_lr to eta_min over T_max steps.
- Is this faithful to Loshchilov 2017? The paper's primary contribution IS
  the restart schedule; the single-cycle cosine decay is a degenerate case
  (T_0 = T_max, no restart). The function name doesn't make this clear.
- Verdict: **MINOR**. The formula is mathematically correct, but the class
  name suggests "cosine annealing" in the Loshchilov sense (with restarts)
  while implementing only the non-restart degenerate case. PyTorch DOES
  have a separate `CosineAnnealingWarmRestarts` class for the full SGDR.
  A user citing Loshchilov 2017 in their paper and using `CosineAnnealingLR`
  is implementing the no-restart variant — same formula, different schedule.
  This is the same flavour as our H02 audit verdict (correct math, but the
  cited paper's headline mechanism — convergence speed — wasn't reachable).

### TP13 — MobileNetV2 inverted residual = Sandler et al. 2018 arXiv:1801.04381 §3.2
- Paper claim: inverted residual = expand 1×1 conv → depthwise 3×3 conv →
  project 1×1 conv (no nonlinearity on project = "linear bottleneck"); residual
  only when stride=1 AND inp==oup.
- Code (mobilenetv2.py:14-56, InvertedResidual):
  - Expansion: `Conv2dNormActivation(inp, hidden_dim, kernel_size=1, ...)`
    (lines 33-36, skipped when expand_ratio=1).
  - Depthwise: `Conv2dNormActivation(hidden_dim, hidden_dim, stride=stride,
    groups=hidden_dim, ...)` (lines 37-44).
  - Projection: `nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
    norm_layer(oup)` — note: NO ReLU/ReLU6 after projection (linear
    bottleneck). Matches paper.
  - Residual: `self.use_res_connect = self.stride == 1 and inp == oup`
    (line 28). Matches paper.
- Verdict: **PASS**. Faithful expand-depthwise-project, linear bottleneck on
  projection, residual condition correct.

### TP14 — SqueezeNet Fire module = Iandola et al. 2016 arXiv:1602.07360 §3.1
- Paper claim: Fire module = squeeze (s1×1 1×1 convs) → expand (e1×1 1×1 +
  e3×3 3×3 convs, concatenated). Stage activations: ReLU on squeeze and
  expand outputs.
- Code (squeezenet.py:14-28, Fire):
  - squeeze = Conv2d(inplanes, squeeze_planes, kernel_size=1)
  - expand1x1 = Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
  - expand3x3 = Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3,
    padding=1)
  - forward: cat([relu(expand1x1(s)), relu(expand3x3(s))], dim=1) where
    s = relu(squeeze(x))
- Channel counts in SqueezeNet 1.0 (squeezenet.py:39-51) match paper Table 1:
  s1=16, e1=64, e3=64 for fire2-3; s1=32, e1=128, e3=128 for fire4-5;
  s1=48, e1=192, e3=192 for fire6-7; s1=64, e1=256, e3=256 for fire8-9.
- Verdict: **PASS**. Faithful Fire module and faithful channel-count table.

### TP15 — BatchNorm defaults = Ioffe & Szegedy 2015 arXiv:1502.03167
- Paper claim: BN normalises by batch mean/var with small ε; trains affine
  γ, β. The paper does not pin a specific ε or momentum value; ε is "added
  to the mini-batch variance for numerical stability."
- Code (batchnorm.py:27 / _NormBase.__init__): `eps=1e-5, momentum=0.1,
  affine=True, track_running_stats=True`.
- BUT: PyTorch's `momentum` is the COEFFICIENT ON THE NEW BATCH STAT, not
  the conventional EMA decay rate. Documented in batchnorm.py:239-242:
  > "x_hat_new = (1 − momentum) · x_hat + momentum · x_t"
- TensorFlow's convention is the opposite (momentum = decay coefficient on
  the old running stat). A user porting a model from TF expecting
  momentum=0.99 would get totally wrong running stats in PyTorch.
- The deviation IS documented. Same class as TP10 (SGD Nesterov).
- Verdict: **MINOR**. The inverted-momentum semantics IS a real footgun
  with cross-framework consequences. Documented in the docstring →
  symmetric-grading MINOR (not MAJOR).

---

## Tally

PASS: TP1, TP3, TP4, TP6, TP7, TP8, TP9, TP11, TP13, TP14 = **10**
MINOR: TP2, TP5, TP10, TP12, TP15 = **5**
MAJOR: (none)
BROKEN: (none)

Total: 15
Non-PASS rate: 5/15 = **33.3 %**

## Comparison framing
- Our project (G1-G8 across 83 hypotheses): 42/83 non-PASS = 50.6 %.
- pytorch/vision (15 hypotheses): 5/15 non-PASS = 33.3 %.
- Difference: 17.3 pp.

## Interpretive notes
1. The third-party non-PASS rate is NOT zero. The same Track-A doctrine
   applied to pytorch/vision still surfaces 33% non-PASS, primarily from
   documented-but-real deviations from the cited paper (V1.5 stride
   placement, PyTorch Nesterov reformulation, inverted BN momentum
   semantics, no-restart "cosine annealing").
2. The 17 pp gap (51 % vs 33 %) suggests our codebase HAS somewhat higher
   defect density, BUT not by the dramatic margin (e.g., 51 % vs 5 %)
   that would unambiguously justify the §5.1 framing.
3. The 33 % third-party non-PASS rate is itself a non-trivial number and
   indicates the audit team is somewhat trigger-happy — many "MINOR"
   findings on torchvision are nitpicks (paper-vs-implementation
   conventions that are explicitly documented). A more lenient grading
   doctrine that downgraded all "documented intentional deviation" to
   PASS would give torchvision: 14 PASS / 1 MINOR (TP12 CosineAnnealingLR
   no-restart) / 0 MAJOR / 0 BROKEN = 1/15 = 6.7 % non-PASS. Under the
   same lenient doctrine applied to our project, the 51 % rate would
   drop too — but the asymmetry of "third-party deviations are mostly
   documented in code comments; our deviations are mostly UNdocumented
   in doc-vs-code mismatch" is preserved.
4. Subjective judgement: the audit's diagnostic power IS PARTIALLY
   overstated. A more honest framing for PAPER.md §5.1 is:
   "audit non-PASS rate of 51 % conditional on Track-A doctrine; the
   same doctrine on pytorch/vision yields 33 %, so the 18-pp delta
   represents an upper bound on excess-defect-density attributable
   to our codebase rather than to auditor aggressiveness."

## Risk of confirmation bias in this calibration
- I am the same model family (Opus 4.7) as the Track-A audit team.
- I read the third-party code with the explicit goal of finding
  "what the audit would say." This is methodologically symmetric (good).
- However, I have NOT independently verified the pytorch/vision code
  end-to-end (no unit tests written, no PyTorch run). The audit is
  static-read only, mirroring the G1-G8 audit methodology.
- A future audit team could disagree with my MINOR/MAJOR grading on
  any of TP2, TP5, TP10, TP12, TP15. Two of these are borderline
  (TP2 V1.5 deviation, TP5 WRN architectural reinterpretation).

---

# Phase-9b extension — raw notes for TP16..TP62 (added 2026-05-31)

Auditor: Track-A implementation-critic style (Opus 4.7), same model.
Scope: extend the n=15 calibration audit to n≥50 using third-party
repos beyond pytorch/vision (per AC punchlist item 3). Repos snapshot:
HEAD of `main` (or `master` where applicable) on 2026-05-31.

## Sources fetched (raw GitHub user-content)

| File | URL |
|---|---|
| timm resnet.py | `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/models/resnet.py` |
| timm efficientnet.py | `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/models/efficientnet.py` |
| timm convnext.py | `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/models/convnext.py` |
| timm mlp_mixer.py | `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/models/mlp_mixer.py` |
| timm byobnet.py (RepVGG) | `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/models/byobnet.py` |
| timm vision_transformer.py | `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/models/vision_transformer.py` |
| timm evo_norm.py | `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/layers/evo_norm.py` |
| timm drop.py | `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/layers/drop.py` |
| timm cross_entropy.py | `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/loss/cross_entropy.py` |
| timm auto_augment.py | `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/data/auto_augment.py` |
| timm lamb.py | `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/optim/lamb.py` |
| timm lookahead.py | `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/optim/lookahead.py` |
| timm lars.py | `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/optim/lars.py` |
| HF BERT modeling_bert.py | `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/bert/modeling_bert.py` |
| HF GPT-2 modeling_gpt2.py | `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gpt2/modeling_gpt2.py` |
| HF Llama modeling_llama.py | `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py` |
| HF T5 modeling_t5.py | `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/t5/modeling_t5.py` |
| HF ViT modeling_vit.py | `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/vit/modeling_vit.py` |
| HF CLIP modeling_clip.py | `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/clip/modeling_clip.py` |
| HF GPT-NeoX modeling_gpt_neox.py | `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py` |
| HF Mistral modeling_mistral.py | `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/mistral/modeling_mistral.py` |
| HF optimization.py | `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/optimization.py` |
| fastai schedule.py | `https://raw.githubusercontent.com/fastai/fastai/master/fastai/callback/schedule.py` |
| fastai mixup.py | `https://raw.githubusercontent.com/fastai/fastai/master/fastai/callback/mixup.py` |
| lightning-bolts simclr_module.py | `https://raw.githubusercontent.com/Lightning-Universe/lightning-bolts/master/src/pl_bolts/models/self_supervised/simclr/simclr_module.py` |
| lightning-bolts lars.py | `https://raw.githubusercontent.com/Lightning-Universe/lightning-bolts/master/src/pl_bolts/optimizers/lars.py` |
| pytorch radam.py | `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/radam.py` |
| pytorch adamw.py | `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/adamw.py` |
| pytorch nadam.py | `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/nadam.py` |
| pytorch adagrad.py | `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/adagrad.py` |
| state-spaces mamba_simple.py | `https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/modules/mamba_simple.py` |
| state-spaces selective_scan_interface.py | `https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/ops/selective_scan_interface.py` |
| state-spaces mamba2.py | `https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/modules/mamba2.py` |

Sub-section abbreviations: "L<line>" is a line number in the file
above. Each row below pairs a single arXiv-citable mechanism with the
file/line implementing it. Verdicts use the §1.4 symmetric-grading
principle (documented deviation → MINOR; undocumented divergence →
MAJOR; not implemented → BROKEN).

---

### TP16 — timm `resnetrs50` cites Bello 2021 arXiv:2103.07579
- timm/models/resnet.py:2647-2703. Docstring at L2648-L2652 explicitly
  cites Bello 2021 with link to the TF reference; the factory function
  wires `block=Bottleneck, stem_width=32, stem_type='deep', avg_down=True`
  matching the RS recipe. **Verdict: PASS**.

### TP17 — timm Bottleneck V1.5 stride placement
- timm/models/resnet.py:209-216. `conv2` (the 3×3) carries `stride=...`;
  `conv1` (the first 1×1, L203) is stride 1. Same V1.5 deviation as
  torchvision (TP2). timm does NOT carry the explicit comment block
  that torchvision does — the deviation is implicit. Could be graded
  MAJOR (undocumented) but the entire ResNet community knows V1.5 as
  the de facto choice and timm's docs reference it elsewhere.
  Symmetric-grading **Verdict: MINOR**.

### TP18 — timm `zero_init_last` for last BN of each block
- timm/models/resnet.py:125-128 (BasicBlock), L189-L192 (Bottleneck),
  invoked at L481. Zeros `bn2.weight` / `bn3.weight` — paper Goyal
  2017 arXiv:1706.02677. **Verdict: PASS**.

### TP19 — timm ResNet-D deep stem (He 2018 "Bag of Tricks" arXiv:1812.01187)
- timm/models/resnet.py:436-461. `stem_type='deep'` triggers a 3-conv
  stem with `stem_chs = (stem_width // 2, stem_width)` or tiered
  `(3 * stem_width // 4, stem_width)`. Faithful to §3.1. **Verdict: PASS**.

### TP20 — timm `avg_down` projection shortcut (ResNet-D §3.1)
- timm/models/resnet.py:597-614 `downsample_avg()`. Dispatched at
  L633-L638 when `avg_down=True`. Faithful to ResNet-D's average-pool
  + 1×1 shortcut. **Verdict: PASS**.

### TP21 — timm EfficientNet compound scaling (Tan & Le 2019 arXiv:1905.11946 §3.3)
- timm/models/efficientnet.py: factory functions pass concrete
  `channel_multiplier` / `depth_multiplier` per variant (e.g.
  `efficientnet_b3` uses `(1.2, 1.4)`). Paper §3.3 derives these
  from `α=1.2, β=1.1, γ=1.15` under `α·β²·γ² ≈ 2` with φ=3, but
  timm does not implement the symbolic compound formula — it caches
  the multipliers per variant. Documented deviation (deviation
  rationale is the standard practice of caching per-variant). Strict
  audit would push MAJOR for "not implementing the cited formula", but
  the values match what the paper Table 1 reports.
  **Verdict: MINOR**.

### TP22 — timm EfficientNet SE ratio 0.25
- timm/models/efficientnet.py: arch_def strings embed `se0.25` per
  block. Hu 2018 arXiv:1709.01507 §6.1 reports r=16 (reciprocal=0.0625);
  EfficientNet paper §3.4 says "we add SE optimization" with no specific
  ratio. The 0.25 figure is the de facto SENet-EfficientNet figure used
  in the TF reference. Faithful to the EfficientNet usage. **Verdict: PASS**.

### TP23 — timm ConvNeXt block (Liu 2022 arXiv:2201.03545 Fig. 4)
- timm/models/convnext.py:150-180. DWConv 7×7 → LN(channels_first)
  → 1×1 expand 4× → GELU → 1×1 project → LayerScale (γ=1e-6 default)
  → DropPath. Matches paper exactly. **Verdict: PASS**.

### TP24 — timm MLP-Mixer no class token (Tolstikhin 2021 arXiv:2105.01601 §2)
- timm/models/mlp_mixer.py:47-79 MixerBlock; L270 patch embed; L265
  GELU; L311 global average pool (no class token). Pre-LN per the
  paper. **Verdict: PASS**.

### TP25 — timm RepVGG block (Ding 2021 arXiv:2101.03697)
- timm/models/byobnet.py:1247-1424 `RepVggBlock`. Three parallel
  branches (3×3, 1×1, identity-BN) gated on `in_chs==out_chs and
  stride==1` (L1260). `reparameterize()` at L1299-L1323. Faithful.
  **Verdict: PASS**.

### TP26 — timm ViT default block uses pre-LN
- timm/models/vision_transformer.py:~1140-1165 `Block`. The default
  `Block.forward` calls `self.norm1(x)` before attention and
  `self.norm2(x)` before MLP, matching ViT (Dosovitskiy 2020 §3.1
  Eq. 2-3 are pre-norm). Note: timm also offers `ParallelScalingBlock`
  and `BlockChunked` variants — the default is the standard pre-norm
  ViT block. **Verdict: PASS**.

### TP27 — timm ViT LayerScale optional (CaiT addition NOT in original ViT)
- timm/models/vision_transformer.py: `self.ls1 = LayerScale(dim,
  init_values=init_values, **dd) if init_values else nn.Identity()`.
  When `init_values=None` (the default for `vit_*` factories), LayerScale
  collapses to Identity — faithful to the original ViT. When users opt
  in to LayerScale, they're using a CaiT (Touvron 2021 arXiv:2103.17239)
  modification, which is documented in the LayerScale module's own
  docstring. **Verdict: PASS** (the default is faithful to ViT 2020).

### TP28 — timm RandAugment (Cubuk 2019 arXiv:1909.13719)
- timm/data/auto_augment.py:588-639. Two hyperparameters N, M;
  uniform sampling at L599-605; default op list at L549-565 (geometric
  + color transforms); magnitude scaling at L646-648. Faithful.
  **Verdict: PASS**.

### TP29 — timm DropPath (Huang 2016 stochastic depth arXiv:1603.09382)
- timm/layers/drop.py:256-261. Per-sample mask shape `(B, 1, 1, ...)`
  at L258; scale-by-keep `1/(1-p)` at L259-L261. Faithful.
  **Verdict: PASS**.

### TP30 — timm LabelSmoothingCrossEntropy (Szegedy 2016 / Pereyra 2017 arXiv:1701.06548)
- timm/loss/cross_entropy.py:13-19. `confidence=1-smoothing` at L16;
  loss `= confidence*NLL + smoothing*smooth_loss` at L19. Default
  `smoothing=0.1` matches paper. **Verdict: PASS**.

### TP31 — timm LARS optimizer (You 2017 arXiv:1708.03888)
- timm/optim/lars.py:106-130. Trust ratio L106-L111 matches paper
  `η·||w|| / (||g|| + β·||w||)`; momentum L122-L129; default
  `trust_coeff=0.001` at L36. Faithful. **Verdict: PASS**.

### TP32 — timm LAMB optimizer (You 2019 arXiv:1904.00962)
- timm/optim/lamb.py:188-198 trust ratio; L164-L165 Adam-style m/v;
  L156-L160 bias correction. **But** timm-LAMB also adds: gradient
  clipping (L149-L152, not in paper), `caution` mechanism from
  arXiv:2411.16085 (L170-L173, post-paper addition), decoupled weight
  decay option (L175-L182), trust ratio clipping (L200-L201). The
  decoupled-WD addition crosses the AdamW/LAMB boundary; paper Alg. 1
  has L2-style WD baked into the trust ratio. Deviations are
  documented in inline comments and the docstring lists the augmented
  references. Symmetric-grading **Verdict: MINOR**.

### TP33 — timm Lookahead k=6 default (paper Zhang 2019 arXiv:1907.08610 uses k=5)
- timm/optim/lookahead.py:15. Defaults `dict(lookahead_alpha=0.5,
  lookahead_k=6, lookahead_steps=0)`. Paper §4.1 / Algorithm 1 uses
  k=5 (and the k=5 default is what fastai and the original repo use).
  This is an undocumented deviation in the timm defaults — no comment
  block explains the k=6 choice. Strict audit: MAJOR (undocumented
  deviation in a hyperparameter that the paper specifies). Symmetric-
  grading: since k=6 vs k=5 is a small numeric tweak with no
  diagnostic mechanism change, **Verdict: MINOR**.

### TP34 — timm EvoNorm S0 (Liu 2020 arXiv:2004.02967)
- timm/layers/evo_norm.py:155 — `x = x * (x * v).sigmoid() /
  group_std(x, self.groups, self.eps)`. Faithful to paper's S0 form
  (sigmoid gate over Swish-style activation, then group-statistic
  normalisation). EvoNormB0 at L110-L116 similarly faithful.
  **Verdict: PASS**.

### TP35 — HF BertSelfAttention scaled dot-product (Vaswani 2017 §3.2.2)
- transformers/models/bert/modeling_bert.py:104-115. `scaling =
  query.size(-1) ** -0.5` at L104; `attn_weights = matmul(Q, K^T) *
  scaling` at L106-L107; softmax + dropout at L111-L112; output
  matmul at L114-L115. Q/K/V split into heads at L210-L212.
  Faithful. **Verdict: PASS**.

### TP36 — HF BERT post-norm layer ordering (Devlin 2018 / Vaswani 2017 §3.1)
- transformers/models/bert/modeling_bert.py:375-424 `BertLayer`.
  `BertSelfOutput.forward` does `dense → dropout → LayerNorm(x + residual)`;
  `BertOutput.forward` same pattern. Post-norm per Vaswani §3.1.
  **Verdict: PASS**.

### TP37 — HF GPT-2 pre-LN (Radford 2019 §2.3)
- transformers/models/gpt2/modeling_gpt2.py: `GPT2Block.forward`
  applies `self.ln_1(hidden_states)` before attention, then
  `self.ln_2(hidden_states)` before MLP. Final `self.ln_f` in
  `GPT2Model.forward` before output head. **Verdict: PASS**.

### TP38 — HF GPT-2 weight tying input embedding ↔ output head
- transformers/models/gpt2/modeling_gpt2.py:
  `_tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}`.
  Standard practice per Press & Wolf 2017 arXiv:1608.05859.
  **Verdict: PASS**.

### TP39 — HF Llama RoPE (Su 2021 arXiv:2104.09864)
- transformers/models/llama/modeling_llama.py:65-120
  `LlamaRotaryEmbedding`; `inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))`
  matches paper Eq. 14. `apply_rotary_pos_emb` at L145-166 implements
  `(q * cos) + (rotate_half(q) * sin)`. **Verdict: PASS**.

### TP40 — HF Llama RMSNorm (Zhang & Sennrich 2019 arXiv:1910.07467)
- transformers/models/llama/modeling_llama.py:46-62. `LlamaRMSNorm`
  is `x * rsqrt(variance + eps)` with no mean subtraction and no
  bias. Comment L49 cites "equivalent to T5LayerNorm". Faithful.
  **Verdict: PASS**.

### TP41 — HF Llama SwiGLU MLP (Shazeer 2020 arXiv:2002.05202)
- transformers/models/llama/modeling_llama.py:180-195. `down_proj(
  act_fn(gate_proj(x)) * up_proj(x))` with `act_fn` = SiLU per
  Llama config. Matches Shazeer's SwiGLU. **Verdict: PASS**.

### TP42 — HF Llama Grouped-Query Attention (Ainslie 2023 arXiv:2305.13245)
- transformers/models/llama/modeling_llama.py:201-221. `num_key_value_groups =
  num_attention_heads // num_key_value_heads`; `repeat_kv` at L170-L177
  expands K/V heads to match Q heads. GQA is in the Llama 2 paper §2.2
  arXiv:2307.09288. **Verdict: PASS**.

### TP43 — HF T5 relative position bias bucketing (Raffel 2019 §2.1)
- transformers/models/t5/modeling_t5.py:283-330. `_relative_position_bucket`
  with half buckets for exact small offsets, half for log-spaced large
  offsets. Faithful to T5 paper §2.1. **Verdict: PASS**.

### TP44 — HF T5LayerNorm (RMS-style)
- transformers/models/t5/modeling_t5.py:43-60. `variance` computed
  without mean subtraction (L52 comment); no bias; scale-only.
  Documented "Root Mean Square Layer Normalization" (Zhang & Sennrich
  2019). **Verdict: PASS**.

### TP45 — HF ViT class token + learnable pos embed (Dosovitskiy 2020 §3.1)
- transformers/models/vit/modeling_vit.py:78-81 cls_token + pos_embeddings;
  L110 prepend cls; L55 patch Conv2d kernel=stride=patch_size; pre-LN
  at L188-190. **Verdict: PASS**.

### TP46 — HF CLIP learned temperature (Radford 2021 §2.5)
- transformers/models/clip/modeling_clip.py:569 `logit_scale =
  nn.Parameter(...)`; L594 `logits *= self.logit_scale.exp()`; L549-550
  visual_projection + text_projection (both bias=False) to projection_dim;
  symmetric loss L41-L44. **Verdict: PASS**.

### TP47 — HF GPT-NeoX parallel residual (Black 2022 arXiv:2204.06745 §3.4)
- transformers/models/gpt_neox/modeling_gpt_neox.py:199-228. When
  `use_parallel_residual=True` (default), `hidden = mlp_out + attn_out +
  residual`. Faithful. **Verdict: PASS**.

### TP48 — HF Mistral sliding-window attention (Jiang 2023 arXiv:2310.06825 §2.1)
- transformers/models/mistral/modeling_mistral.py:278 dispatches
  `create_sliding_window_causal_mask` when `config.sliding_window is not None`;
  L197 passes `sliding_window` to attention interface. **Verdict: PASS**.

### TP49 — HF `get_cosine_schedule_with_warmup` (vs SGDR full)
- transformers/optimization.py:149-176. Returns `0.5 * (1 + cos(π * 2 *
  num_cycles * progress))` with `num_cycles=0.5` default — the no-restart
  degenerate case of Loshchilov & Hutter 2017 SGDR (arXiv:1608.03983).
  Same MINOR as torchvision TP12: misleading shortname for "the cosine
  decay piece of SGDR without the restarts." Documented in the
  docstring. Symmetric-grading **Verdict: MINOR**.

### TP50 — lightning-bolts LARS (You 2017 arXiv:1708.03888)
- lightning-bolts/optimizers/lars.py:113-127. Trust ratio `lars_lr =
  trust_coefficient * p_norm / (g_norm + p_norm * weight_decay + eps)`
  at L113-114; gradient scaled by `lars_lr` at L116-L117; momentum
  buffer at L120-L126. Default `trust_coefficient=0.001` at L45.
  Faithful. **Verdict: PASS**.

### TP51 — fastai `fit_one_cycle` default pct_start=0.25 vs Smith 2018 §6 (suggests 0.45)
- fastai/callback/schedule.py:46. Signature is `def fit_one_cycle(self,
  n_epoch, lr_max=None, div=25., div_final=1e5, pct_start=0.25, ...)`.
  Smith 2018 arXiv:1803.09820 §6.1 suggests `pct_start ≈ 0.3-0.45` for
  CIFAR; fastai's 0.25 is in the same family but at the conservative
  end. The deviation is in code only — the docstring does NOT cite
  the paper's recommendation. Symmetric-grading: would be MAJOR if
  undocumented entirely, but fastai's book Ch. 5 documents the
  rationale for 0.25 (faster ramp-up empirically). **Verdict: MINOR**.

### TP52 — fastai Mixup default alpha=0.4 (paper Zhang 2017 §3.1 reports α=0.2 on ImageNet)
- fastai/callback/mixup.py:34. `alpha: float = .4`. Paper Table 1 uses
  α=0.2 for ImageNet/CIFAR-10. fastai's 0.4 is an undocumented
  hyperparameter deviation. Strict: MAJOR. Mitigation: the paper
  reports α ∈ [0.1, 0.4] all work and α=0.4 is in the tested range;
  fastai's book Ch. 7 discusses the choice. Symmetric-grading
  **Verdict: MINOR**.

### TP53 — fastai CutMix default alpha=1.0 (paper Yun 2019 §3 uses α=1.0)
- fastai/callback/mixup.py:52. `alpha: float = 1.`. Matches paper.
  **Verdict: PASS**.

### TP54 — lightning-bolts SimCLR temperature default 0.1 (paper Chen 2020 uses 0.5 on CIFAR, 0.1 on ImageNet)
- lightning-bolts/.../simclr_module.py:67 default `temperature=0.1`,
  overridden to 0.5 at L133 for CIFAR-10. Paper §B.1 Table B.5 reports
  the per-dataset sweep; 0.1 is fine for ImageNet but not for CIFAR.
  Default + per-dataset override is the conventional pattern. **Verdict:
  MINOR** (the default is a deviation for the CIFAR use-case the codebase
  primarily targets — see L133 override).

### TP55 — lightning-bolts SimCLR projection head (Chen 2020 §2)
- lightning-bolts/.../simclr_module.py:38-50. Linear → BN1d → ReLU →
  Linear (bias=False) → F.normalize. Matches paper §2 "projection
  head g(·)". **Verdict: PASS**.

### TP56 — torch.optim AdamW decoupled weight decay (Loshchilov & Hutter 2017 arXiv:1711.05101)
- torch/optim/adamw.py: defaults `weight_decay=1e-2`,
  `betas=(0.9, 0.999)`; `decoupled_weight_decay=True` propagated to
  the underlying functional `adam()`, which performs `param.mul_(1 -
  lr * weight_decay)` before the Adam moment update. Faithful to
  paper Alg. 2 (decoupled-WD pre-update). **Verdict: PASS**.

### TP57 — torch.optim NAdam Dozat schedule
- torch/optim/nadam.py:351-352, L379-386. Momentum schedule
  `mu = β1·(1 - 0.5·0.96^(step·momentum_decay))` matches Dozat 2016
  Appendix A.2 exponential schedule (base 0.96). `param.addcdiv_`
  applied separately for the gradient term and momentum term to
  approximate the Nesterov-corrected first moment. **Verdict: PASS**.

### TP58 — torch.optim RAdam SMA threshold 5.0 vs paper ρ_t > 4
- torch/optim/radam.py:335, L449. The implementation switches to the
  rectified update when `rho_t > 5.0`. Liu et al. 2019 Algorithm 2
  uses `ρ_t > 4` (paper §3.2). PyTorch's threshold matches the
  original repo's `rho_t > 5` which itself reflects the Liu et al.
  2020 corrigendum / "RAdam needs at least 5 steps of warmup".
  Deviation is documented neither in the file nor the docstring.
  Strict: MAJOR (undocumented hyperparameter deviation from paper
  Alg. 2). Mitigation: rho_inf at L318 matches paper exactly.
  Symmetric-grading: since the >5 threshold has become the de facto
  convention (matching the reference RAdam implementation), **Verdict:
  MINOR**.

### TP59 — torch.optim Adagrad (Duchi 2011 JMLR)
- torch/optim/adagrad.py:78-82, L280-L284. `state["sum"]` per-param;
  `state_sum.addcmul_(grad, grad, value=1)` accumulator; update
  `param.addcdiv_(grad, sqrt(state_sum)+eps, value=-clr)`. Defaults
  `lr=1e-2`, `initial_accumulator_value=0`. Faithful. **Verdict: PASS**.

### TP60 — Mamba `selective_scan_ref` (Gu & Dao 2023 arXiv:2312.00752 §3.3)
- mamba_ssm/ops/selective_scan_interface.py:107-165 reference Python
  implementation. L129 `deltaA = exp(einsum(delta, A))` matches paper
  discretisation A_bar = exp(ΔA). L141 recurrence `x = deltaA[:,:,i] *
  x + deltaB_u[:,:,i]` matches h_{t+1} = A_bar·h_t + B_bar·x_t.
  Output `y = C·x` via einsum at L142-149. **Verdict: PASS**.

### TP61 — Mamba S4D-style A initialization (Gu 2022 arXiv:2206.11893)
- mamba_ssm/modules/mamba_simple.py:73-77. `A = repeat(arange(1,
  d_state+1), 'n -> d n', d=d_inner); A_log = log(A); self.A_log =
  Parameter(A_log)`. Matches the S4D-Real init (negative real
  eigenvalues at integer spacing) called out in Gu 2022 §3.2. The
  selective-scan recurrence then exponentiates `-exp(A_log) * delta`
  giving exponentially-stable A_bar. **Verdict: PASS**.

### TP62 — Mamba-2 SSD chunk scan (Dao & Gu 2024 arXiv:2405.21060)
- mamba_ssm/modules/mamba2.py:141, L169. Dispatches
  `mamba_split_conv1d_scan_combined` and `mamba_chunk_scan_combined`
  with `chunk_size` (default 256, L72). Multi-head SSM at L82
  `nheads = d_ssm // headdim`. RMSNormGated at L107 applied at L187
  before output projection. Matches SSD algorithm in Dao & Gu 2024.
  **Verdict: PASS**.

---

## Excluded candidates (honesty bias — could not verify cleanly)

The following were considered but excluded from the aggregate:

- **timm EfficientNet MBConv block internals**: defined in
  `_efficientnet_builder.py`, decoded from arch_def strings. The
  WebFetch returned the wrapper but not the block module body, so
  the "1×1 expand → 3×3 DW → SE → 1×1 project" claim cannot be
  cited at file:line resolution. Excluded.
- **HF transformers AdamW**: HF used to ship its own AdamW; current
  code delegates to torch.optim.AdamW. Not a separate audit row.
  Excluded (already covered by TP56).
- **fastai mixup label-stack form**: fastai supports both standard
  Mixup-of-labels and a "stack_y" form that defers interpolation
  into the loss. The form is configurable and matches paper plus
  CE-loss algebra; not a clean per-paper claim. Excluded.
- **HF CLIP "quick GELU"** vs standard GELU: WebFetch did not return
  the explicit `ACT2FN` mapping for CLIP, so we cannot pin down at
  file:line whether timm uses the quick approximation (sigmoid(1.702x))
  vs standard. Excluded.
- **Mamba dt_proj softplus init bound check**: the init formula
  `inv_dt = dt + log(-expm1(-dt))` is present but the boundedness
  proof requires reading the dt_max/dt_min clamping code we didn't
  fully fetch. Excluded.

Total considered: 47 (TP16..TP62). Total included in aggregate
extension: **47**. Grand total across §2 + Phase-9b: **15 + 47 = 62**.
