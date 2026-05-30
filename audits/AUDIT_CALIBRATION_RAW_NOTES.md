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
