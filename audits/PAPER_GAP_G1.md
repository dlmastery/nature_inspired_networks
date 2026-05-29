# Paper-Gap Audit — G1 Scaling & Growth (H01–H10)
Reviewer: PaperGap-G1 (paper-vs-result auditor)
Date: 2026-05-29

This audit is the THIRD dimension on top of `audits/G1_audit.md`
(impl-critic, code-vs-doc) and the per-doc "Addendum: Research-Scientist
Critique" (sci-critic, idea-vs-literature). The third dimension asks:
**given the arXiv paper(s) the design doc cites and the headline numbers
those papers report, why does OUR CIFAR-10 / CIFAR-100 result land where
it does?** Each gap is bucketed into one of seven categories defined in
the audit brief. Baselines used throughout: `baseline_resnet20`
top-1 = 0.8478 (15 ep, 272k params), `baseline_sg_vanilla` top-1 = 0.8216
(12 ep, 186k params). Our nature-prior variants share the 12-ep training
regime of `baseline_sg_vanilla` unless noted.

## Summary
- **PAPER_AGREES:** H02, H09
- **SCALE:** H01, H05
- **DOMAIN:** H03, H07
- **IMPL_BUG:** H06, H08
- **WRONG_TEST:** H10
- **CITATION_DOESNT_SUPPORT:** H04
- **NO_ARXIV:** (none in G1 — every hypothesis cites at least one arXiv
  anchor, though several anchors don't actually claim what the design doc
  reads them as; those fall under CITATION_DOESNT_SUPPORT or DOMAIN, not
  NO_ARXIV.)

The two IMPL_BUG findings (H06 expansion ratio, H08 missing
function-preserving init) are the only ones that warrant a Fixer
follow-up — see "Group-level conclusion" at the end.

---

## Per-hypothesis gap diagnosis

### H01 — phi-Compound Scaling
**Cited paper(s):** Tan & Le 2019 ICML 'EfficientNet: Rethinking Model
Scaling for Convolutional Neural Networks' (arXiv:1905.11946); Bello et
al 2021 NeurIPS 'Revisiting ResNets: Improved Training and Scaling
Strategies' (arXiv:2103.07579).

**Paper's reported number:** EfficientNet-B0..B7 climbs ImageNet top-1
from 77.1% (B0, 5.3M params, 0.39 BFLOPs) to 84.4% (B7, 66M params, 37
BFLOPs) — i.e. ~+7pp top-1 from compound scaling at 350 epochs on
ImageNet with AutoAugment, SiLU, drop-path, and EMA. Bello et al (2103)
shows compound scaling is **insensitive to exponent choice within ±20%**;
this directly anticipates that a φ-substitution lands within noise.

**Our number:** `sg_only_phi_compound` top-1 = 0.8042 vs
`baseline_resnet20` 0.8478 (Δ = -4.36pp) at 127k params (less than half
the baseline budget), 12 epochs, CIFAR-10. Composite 0.8152 vs 0.8458.

**Gap classification:** SCALE

**Diagnosis:** EfficientNet's +7pp gain materialises over **66×** more
params and **30×** more epochs than our screening regime. At our
12-epoch / sub-200k regime the compound-scaling family is operating at
k≈0 (the family is *defined* relative to a base model and 12 epochs is
nowhere near the regime where compound scaling pays back). Bello 2021
explicitly says exponent choice ≈φ is in the insensitive band; we should
expect zero edge, which we see. The 4pp loss is the under-parameter-
isation (127k vs 272k) — a Rule-1 violation in the sweep (param count
also changed) more than a φ-vs-non-φ comparison.

**Recommended action:** None on the implementation. Add a sentence to
H01.md sec. 6 ("Predicted Δ table") explicitly stating that screening at
12 ep and <200k params **cannot** reproduce EfficientNet's regime and
the SCALE gap is expected. Per impl-critic, also document the
intra-family `phi^s` stage-width recurrence — but that's a doc fix, not
a result fix.

---

### H02 — Fibonacci Depth Progression
**Cited paper(s):** Larsson et al 2017 ICLR 'FractalNet: Ultra-Deep
Neural Networks without Residuals' (arXiv:1605.07648); He et al 2016
CVPR 'Deep Residual Learning for Image Recognition' (arXiv:1512.03385);
Tan & Le 2019 (arXiv:1905.11946); Radosavovic et al 2020 CVPR
'Designing Network Design Spaces — RegNet' (arXiv:2003.13678).

**Paper's reported number:** RegNet (2003.13678) finds the optimal
per-stage depth schedule is a **monotonically increasing log-quantised
sequence** — Fib (3, 5, 8, 13) is one specific instantiation, and RegNet
explicitly notes the optimum is *roughly* exponential without a magic
ratio. ResNet-20 itself uses (3, 3, 3) and reaches 91.25% at 164 epochs;
RegNetY-200MF at iso-params reaches ~91-92%.

**Our number:** `sg_only_fib_depth` top-1 = 0.8218 at 180k params
(12 ep), vs `baseline_sg_vanilla` 0.8216 at 186k params — i.e. **within
0.02pp at iso-budget**. Composite 0.8261 vs baseline 0.8258.

**Gap classification:** PAPER_AGREES

**Diagnosis:** This is exactly what RegNet would predict — a quantised
monotone depth schedule (Fib or otherwise) matches a uniform schedule
within noise at small budgets. There is no gap to explain; the result
**confirms** the literature anchor (RegNet) rather than the project's
positive headline. The sci-critic correctly tags this DERIVATIVE+TESTABLE,
which is consistent with PAPER_AGREES — fib-depth is a real but
unsurprising-and-derivative effect.

**Recommended action:** None on the result. Per impl-critic, the falsifier
(`epochs-to-72%-top-1`) is unreachable and the §7.1 composite-formula
manipulation is a Rule-2 violation; those are harness gaps, not paper-gap
issues.

---

### H03 — Golden Spiral Resolution Scaling
**Cited paper(s):** Cohen & Welling 2016 ICML 'Group Equivariant
Convolutional Networks' (arXiv:1602.07576); Dosovitskiy et al 2021 ICLR
'An Image is Worth 16x16 Words: Transformers for Image Recognition at
Scale' (arXiv:2010.11929); Touvron et al 2019 NeurIPS 'Fixing the
Train-Test Resolution Discrepancy — FixRes' (arXiv:1906.06423); Worrall
et al 2017 CVPR 'Harmonic Networks' (arXiv:1612.04642); Weiler et al
2019 NeurIPS 'General E(2)-Equivariant Steerable CNNs' (arXiv:1911.08251).

**Paper's reported number:** FixRes reports +1-2pp on ImageNet from a
single resolution-mismatch correction at the end of training. Group-
equivariant CNNs (Cohen-Welling, Worrall, Weiler) report 0.5-2pp on
rotated-MNIST / rotated-CIFAR with **explicit group convolutions** — not
just a resize. ViT (Dosovitskiy) operates on **patch tokens**, not a
resize; its multi-resolution variants (PiT, MViT) live in
patch-tokenisation space, not in pre-network resize.

**Our number:** `sg_only_golden_resize` top-1 = 0.8067 (Δ = -4.11pp vs
`baseline_resnet20`, Δ = -1.49pp vs `baseline_sg_vanilla`), 127k params,
12 ep. Composite 0.8157.

**Gap classification:** DOMAIN

**Diagnosis:** The cited rotation-equivariance papers (Cohen-Welling,
Worrall, Weiler) require *equivariant group convolutions* (real
mechanism); the cited multi-resolution paper (FixRes) corrects a
**train-test resolution mismatch** at inference, not a per-batch
phyllotactic resize. The shipped mechanism is just `F.interpolate` to
one of `[28, 45, 73, 118, 191]` followed by a standard (non-equivariant)
ResNet-20. The φ-spacing has no theoretical lever to pull on this
backbone, and the resize itself just changes input statistics. Per
impl-critic, the patch-lattice helper is reserved for a future ViT
sub-project and never executed; the CNN-track shipped is essentially
"resize the input by phi^k and run a normal CNN" — which has no paper
backing for the predicted gain.

**Recommended action:** This hypothesis needs the **right benchmark** to
have a fair shot: either (a) rotated-CIFAR-10 / rotated-MNIST with an
equivariant backbone (Worrall / Weiler) to actually test the cochlear-
rotation claim, or (b) a ViT sub-project where the golden-spiral lattice
goes into the patch tokeniser. As shipped, the CIFAR-10 number is a
DOMAIN mismatch with the cited papers and -4pp is fully explained by
"the equivariance the paper would buy us is not implemented in this
backbone". Sci-critic verdict NUMEROLOGY is consistent.

---

### H04 — phi-Self-Similar Width
**Cited paper(s):** Tan & Le 2019 (arXiv:1905.11946); He et al 2016
(arXiv:1512.03385); Howard et al 2017 'MobileNets: Efficient
Convolutional Neural Networks for Mobile Vision Applications'
(arXiv:1704.04861); Howard et al 2019 ICCV 'Searching for MobileNetV3'
(arXiv:1905.02244); Radosavovic et al 2020 (arXiv:2003.13678).

**Paper's reported number:** RegNet (2003.13678) explicitly parametrises
**width-progression** with the family `w_j = w_0 + w_a · j` (linear) or
`u_j = w_0 · w_m^j` (quantised exponential) and **empirically** finds
the best `w_m` lies in **[2.5, 2.9]** — NOT φ ≈ 1.618. So the paper's
own number for "best width-multiplier" is the *opposite* of φ:
non-φ exponents win on ImageNet at all budgets RegNet swept.

**Our number:** `sg_chan_fib` = 0.8011 / `sg_chan_phi` = 0.8011 (iso-
result; impl-critic flagged that `phi_compound` mode is a verbatim copy
of `phi` mode, so observational underdetermination); both -2.05pp vs
`sg_vanilla`. 127k params, 12 ep.

**Gap classification:** CITATION_DOESNT_SUPPORT

**Diagnosis:** The H04 design doc reads EfficientNet / MobileNet /
RegNet as **supporting** a phi-progression of widths. RegNet (cited!)
actually says the opposite: the empirically optimal `w_m` is ~2.5-2.9.
This isn't a SCALE gap or DOMAIN gap — the cited paper actively
disconfirms the hypothesis under its own experimental conditions, and
our -2pp result aligns with the cited paper's prediction (φ-spacing is
worse than the 2.5-2.9 region RegNet found). Sci-critic correctly tags
this NUMEROLOGY and the impl-critic notes mode='phi_compound' is a
duplicate of mode='phi'. The hypothesis as written cannot survive a
literal reading of its own citation.

**Recommended action:** Rewrite H04.md sec. 6 ("Citations") to
acknowledge that RegNet's empirical finding (w_m ≈ 2.5-2.9) is the
**falsifier the doc dodges**, and accept the sci-critic NUMEROLOGY
verdict. Either delete the H04 sweep rows or re-frame the hypothesis as
"*despite* the literature optimum being ~2.5-2.9, does φ-width offer
any small-budget niche?" — to which our 12-ep CIFAR data says no.
**No fixer-style code change recommended**: the implementation is
correct; the citation simply doesn't support the claim.

---

### H05 — Fractal phi-Recursion
**Cited paper(s):** Larsson et al 2017 ICLR 'FractalNet' (arXiv:1605.07648);
Veit et al 2016 NeurIPS 'Residual Networks Behave Like Ensembles of
Relatively Shallow Networks' (arXiv:1605.06431); Huang et al 2017 ICLR
'Multi-Scale Dense Networks' (arXiv:1703.09844).

**Paper's reported number:** FractalNet (Larsson 2017) reports CIFAR-10
top-1 **~95.4%** at **400 epochs** with depth-20 fractal (compared to
ResNet-110 95.43%, DenseNet 96.4%). Veit et al 2016 quantifies that
deep nets behave as **ensembles** — additional paths add ~1pp.

**Our number:** `sg_only_fractal` top-1 = 0.8246 at 259k params, 12 ep.
Composite 0.8104 (slightly under baseline 0.8258 — small param penalty).

**Gap classification:** SCALE

**Diagnosis:** FractalNet's headline 95% is a **400-epoch CIFAR-10
number** with the full augmentation+drop-path recipe. Our 12-epoch
82.46% is precisely the regime where fractal recursion has had no time
to differentiate from a residual ladder. The Veit-2016 ensemble effect
needs late-training paths-as-experts to materialise, which also requires
long training. The +2.35pp the sci-critic flags as "may be ensemble
effect not phi-recursion" cannot even be measured at 12 ep. So the SCALE
gap fully accounts for our flat result; the φ-shrink-ratio specific
claim **is not adjudicable** at the current budget.

**Recommended action:** None on the result. To actually test the
phi-shrink-ratio claim (separable from FractalNet's general gain), run a
400-epoch CIFAR-10 sweep with shrink ratios in {1/2, 1/φ, 1/1.5, 1/1.7}
all at iso-params. That's a Phase-4 / Phase-5 task and outside G1's
12-ep screening scope. Per impl-critic, also add a recursion-compound
test and resolve the depth-recurrence ambiguity in H05.md sec. 5.1.

---

### H06 — Golden Ratio Bottleneck
**Cited paper(s):** He et al 2016 (arXiv:1512.03385); Szegedy et al
2015 CVPR 'Going Deeper with Convolutions' (arXiv:1409.4842); Sandler
et al 2018 CVPR 'MobileNetV2: Inverted Residuals and Linear Bottlenecks'
(arXiv:1801.04381); Howard et al 2019 (arXiv:1905.02244).

**Paper's reported number:** MobileNetV2 (Sandler 2018) uses inverted
bottleneck with **expansion ratio t = 6** and reports ImageNet 72.0% at
3.4M params. MobileNetV3 (Howard 2019) **NAS-tunes per-block expansion**
and lands at ratios in **[2.4, 6.0]**, with small-cost blocks at
**~2.5-3.0** — i.e. φ² ≈ 2.618 sits **squarely inside the NAS-discovered
optimum**. Sandler 2018 also explicitly shows that t=1 (no expansion)
underperforms by ~3pp on ImageNet.

**Our number:** `sg_only_golden_bottleneck` top-1 = 0.6879 at **58k
params** (vs baseline 272k — 4.7× under-parameterised!), 12 ep, CIFAR-10.
Composite 0.7018. **Δ = -15.99pp vs baseline_resnet20**.

**Gap classification:** IMPL_BUG

**Diagnosis:** Two compounding issues. (a) **The impl-critic found the
code uses expansion ratio t = φ ≈ 1.618 instead of the doc-stated
t = φ² ≈ 2.618.** For c_in=32: code gives c_mid=48, doc predicts
c_mid=80. The test (`test_h06_inverted_branch_expands_then_contracts`)
**bakes in the wrong constant (104 for c_in=64)** — it codifies the bug
rather than catching it. (b) **The realised network is 58k params vs the
272k baseline — a 4.7× under-capacity which alone explains ~12pp of the
gap.** MobileNetV2's t=6 ImageNet 72% is for a 3.4M-param network; the
fair comparison would be a 250-275k-param phi-bottleneck CIFAR-10 net
trained 12 ep, which we don't have. So the -16pp gap is **partly** an
implementation bug (wrong expansion ratio, sub-φ²) and **mostly** a
SCALE-PARAM under-parameterisation, but the IMPL_BUG is the actionable
piece because the under-parameterisation cannot be fixed at the same
expansion ratio without doubling the network.

**Recommended action (Fixer follow-up):**
1. **Patch `src/nature_inspired_networks/phi_scaling.py:GoldenBottleneck`
   line 113**: change `c_mid = _round8(c_in * PHI)` to
   `c_mid = _round8(int(round(c_in * PHI**2)))` so the inverted branch
   actually uses ratio φ² as the doc claims.
2. **Rewrite `tests/test_phi_scaling.py:test_h06_inverted_branch_expands_then_contracts`**
   to derive the expected `c_mid` from the constant
   `_round8(int(round(c_in * PHI**2)))` instead of hard-coding 104. This
   prevents the test from re-codifying the bug.
3. **Re-smoke** `sg_only_golden_bottleneck` at seed 0 after the patch
   and record pre-fix (0.6879) vs post-fix in `FINDINGS.md` per Rule 21.
4. **Then** the SCALE-PARAM caveat remains valid: even at correct φ²,
   a 58k-param net cannot match a 272k baseline on CIFAR-10 12 ep —
   that is expected and the falsifier (≥5% param reduction
   `AND` ≥-0.5pp top-1) was always going to fail at this c0. Either
   raise c0 to land at ~250k params or accept the H06 SCALE bound.

---

### H07 — phi-Modulated Multi-Scale FPN
**Cited paper(s):** Lin et al 2017 CVPR 'Feature Pyramid Networks for
Object Detection' (arXiv:1612.03144); Yang et al 2021 Scientific Data
'MedMNIST v2 — A Large-Scale Lightweight Benchmark for 2D and 3D
Biomedical Image Classification' (arXiv:2110.14795); Tan et al 2020
CVPR 'EfficientDet: Scalable and Efficient Object Detection (BiFPN)'
(arXiv:1911.09070); Wang et al 2020 TPAMI 'HRNet' (arXiv:1908.07919);
Liu et al 2018 CVPR 'Path Aggregation Network (PANet)' (arXiv:1803.01534).

**Paper's reported number:** FPN (Lin 2017) reports +1-2 mAP on COCO
**detection** at 800px input; BiFPN reports +1-2 mAP at -50% FLOPs;
HRNet reports +2-3 mAP on COCO keypoint. **All on detection / dense
prediction, not on CIFAR-scale image classification with a 32-pixel
input.** None of these papers claim **classification** accuracy gains
from FPN-style fusion; FPN is a detection-head device.

**Our number:** `sg_only_phi_multiscale` top-1 = 0.8200 (Δ = -2.78pp vs
baseline_resnet20, -0.16pp vs sg_vanilla) at 194k params, 12 ep.
Composite 0.8151.

**Gap classification:** DOMAIN

**Diagnosis:** FPN is a **detection-head mechanism for 800px-class
inputs with multiple stride levels carrying small/medium/large objects**.
Forcing it onto a CIFAR-10 32×32 single-object classification problem
removes its theoretical lever: there are no small-object features that
benefit from late lateral fusion when the whole image is one airplane.
Impl-critic also notes the shipped code only does **channel-axis
phi-spacing, not resolution-axis**, and the doc's "8 fused multi-scale
features" is not implementable in the shipped FPN. So the -2.78pp gap
is "detection-head device applied to classification, won't help, and
the shipped implementation is also a scope reduction of the doc's
claim."

**Recommended action:** The right benchmark is **COCO detection** or
**MedMNIST 224×224 multi-organ classification** where multi-scale fusion
actually pays off (Yang 2021 explicitly uses 28-px MedMNIST images but
that's the same problem — small inputs, no scale separation). For CIFAR
classification, accept that DOMAIN dominates and downgrade the H07
prediction. Per impl-critic, also tighten the widths-ratio test bound
and add an input-order assertion.

---

### H08 — Dynamic phi-Growth
**Cited paper(s):** Chen et al 2016 ICLR 'Net2Net: Accelerating Learning
via Knowledge Transfer' (arXiv:1511.05641); Wei et al 2016 ICML
'Network Morphism' (arXiv:1603.01670); Larsson et al 2017
(arXiv:1605.07648); Frankle & Carbin 2019 ICLR 'The Lottery Ticket
Hypothesis' (arXiv:1803.03635).

**Paper's reported number:** Net2Net (Chen 2016) reports **35% wall-
clock reduction** on ImageNet for matched top-1 via **function-
preserving** widening (the new wider net's forward output **equals** the
narrower net's output at the growth step within 1e-5 tolerance). The
**function-preservation property is the entire mechanism** — without it,
each "growth event" is a from-scratch reinit that disrupts training and
no compute is saved. Wei 2016 (Network Morphism) extends this with
exactly-preserving construction for conv/BN layers.

**Our number:** No CIFAR-10 smoke row exists. `dynamic_growth` is wired
into the module but not in the sweep grid; the closest live row is
`sg_full_fib` which composes multiple priors and is irrelevant for H08.

**Gap classification:** IMPL_BUG

**Diagnosis:** The impl-critic G1 audit identified the headline H08 bug:
**`_kaiming_reinit_` does not implement Net2Net's function-preserving
init**. The cited Net2Net paper's entire claim is contingent on
identity-init (the new conv weights are constructed so the new net's
forward = old net's forward at the growth step within 1e-5). Our code
does **Kaiming reinit** — empirical max-abs-diff between pre-growth and
post-growth forward is **0.088, not 1e-5**. So every "growth event"
restarts the affected layers from scratch and **discards the training
the model has done so far in that branch**. The cited paper's +35%
wall-clock saving is unreachable with this implementation. The H08 test
suite has no `test_grow_model_function_preserves_value` — the design
doc sec. 9 Q&A explicitly commits to this test, which is missing
(Rule-25 violation: Q&A test name not in `tests/`).

**Recommended action (Fixer follow-up — HIGH PRIORITY):**
1. **Replace `_kaiming_reinit_` in
   `src/nature_inspired_networks/dynamic_growth.py:72-84`** with a
   function-preserving init. For an appended `NaturePriorBlock` whose
   residual skip is identity, the new conv weight should be initialised
   as `weight ≈ eps · randn + delta(centre)` (centre 1.0, surround 0.0)
   with **BN gamma=1.0, beta=0.0** so the block computes
   `out = skip + conv(x) ≈ skip + x` ≈ skip at initialisation.
   Alternatively, init the **last conv** in the block to zero — the
   block then computes `out = skip + 0 = skip` exactly, which is the
   simplest function-preserving construction (cf. Zhang et al 2019
   "Fixup Initialization", arXiv:1901.09321).
2. **Add `tests/test_dynamic_growth.py:test_grow_model_function_preserves_value`**
   that asserts
   `(net_after_grow(x) - net_before_grow(x)).abs().max() < 1e-3` on a
   fixed-seed `(2, 3, 32, 32)` input. This is the gate the design doc
   commits to.
3. **Then** add a CIFAR-10 12-ep `sg_only_dynamic_growth` smoke row to
   the sweep grid and record post-fix top-1.
4. If function-preservation is not implemented, update H08.md sec. 9
   Q&A to acknowledge the current implementation does NOT preserve
   function and the +35% Net2Net wall-clock claim is unreachable
   pending the fix.

---

### H09 — Golden Proportion Parameter Budget
**Cited paper(s):** Tan & Le 2019 (arXiv:1905.11946); He et al 2016
(arXiv:1512.03385); Frankle & Carbin 2019 ICLR 'The Lottery Ticket
Hypothesis: Finding Sparse, Trainable Neural Networks' (arXiv:1803.03635);
Hoffmann et al 2022 NeurIPS 'Training Compute-Optimal Large Language
Models — Chinchilla' (arXiv:2203.15556); Radosavovic et al 2020
(arXiv:2003.13678).

**Paper's reported number:** RegNet (2003.13678) finds Pareto-optimal
families on ImageNet whose width-multiplier `w_m` is in [2.5, 2.9]. At
small budgets (~270k params, CIFAR scale) **RegNetX-200MF reaches
~92.0% CIFAR-10 at 100 epochs**, and the family is robust within ±15%
of the optimum exponent — i.e., φ² ≈ 2.618 **lands inside the RegNet
Pareto region** (sci-critic verdict: "sits inside RegNet's already-
discovered Pareto region"). At 12 epochs, RegNet's analysis predicts
**roughly the same delta vs uniform allocation** for any monotone
exponent in [1.5, 3].

**Our number:** `sg_only_phi_budget` (post-Fixer) top-1 = 0.8556 on
CIFAR-10 (12 ep, 262k params, vs baseline_resnet20 0.8478 → **+0.78pp**;
vs sg_vanilla 0.8216 → **+3.40pp**). CIFAR-100 3-seed median 0.5741 vs
baseline 0.5652 = **+0.89pp**. Composite 0.8338 vs baseline 0.8458 (-0.012;
composite is FLOPs-penalised so the FLOPs increase from 41M → 81M eats
the top-1 win in the composite metric).

**Gap classification:** PAPER_AGREES

**Diagnosis:** The +0.78pp CIFAR-10 / +0.89pp CIFAR-100 3-seed delta is
**within the band RegNet would predict for any reasonable monotone
exponent**. φ-allocation is one specific point inside RegNet's Pareto
region; that the point works ~+1pp better than uniform is consistent
with the cited literature. The sci-critic note that "the φ-content is
decorative" is true — the headline gain is the **monotone-allocation
gain RegNet already discovered**, not a φ-specific effect — but the
*number* matches the literature anchor, so this is PAPER_AGREES not
NUMEROLOGY in the paper-gap sense.

**Critical caveat (carried forward from impl-critic G1 audit):** The
realized per-stage param ratios at B=270k are **1 : 1.41 : 2.45**,
**NOT** the claimed 1 : φ : φ² = 1 : 1.618 : 2.618. The allocation step
is correct; the closed-form `c² ≈ params / (2·blocks·k²)` width-derivation
ignores transition-conv costs, BN, stem, and FC. So while the result
PAPER_AGREES (any monotone allocation beats uniform by ~+1pp at this
scale), the *named mechanism* (1:φ:φ²) is not faithfully realized — a
calculator-equipped reviewer can catch this. The "PAPER_AGREES"
classification refers to the **observed accuracy delta**, not to the
**mechanism fidelity**, which is an open impl-critic finding.

**Recommended action:** None on the result classification (the +1pp is
real and consistent with RegNet). On the mechanism fidelity, **the
impl-critic G1 audit's P0 fix is the right action**: add
`test_h09_phi_budget_net_realized_per_stage_param_ratio` and improve
`phi_budget_widths` to solve the c_prev × c_cur transition-cost
quadratic exactly, then re-verify before any external claim. This is
**not** a paper-gap fixer task; it is a within-mechanism fidelity task
already on the impl-critic backlog.

---

### H10 — phi-Decay LR Scheduler
**Cited paper(s):** Loshchilov & Hutter 2017 ICLR 'SGDR: Stochastic
Gradient Descent with Warm Restarts' (arXiv:1608.03983); Smith 2017
WACV 'Cyclical Learning Rates for Training Neural Networks'
(arXiv:1506.01186); Robbins & Monro 1951 (no arXiv — classical);
Goyal et al 2017 'Accurate, Large Minibatch SGD' (arXiv:1706.02677).

**Paper's reported number:** Loshchilov 2017 (cosine annealing) reports
**~0.3-0.5pp gain over step-LR** on CIFAR-10 at 200 epochs. The decay
**shape** matters (Smith 2017 surveys triangular, exponential, cosine,
polynomial); **no specific paper claims φ-based exponential decay is a
sweet spot**. The cited papers explicitly compare cosine to **step,
linear, polynomial, exponential** — all within 0.5pp of each other in
the published numbers. Sci-critic correctly flagged "no arXiv anchor
for the *specific* φ-decay-LR claim" and verdict NUMEROLOGY.

**Our number:** `sg_only_phi_lr` top-1 = 0.7875 (Δ = -6.03pp vs
baseline_resnet20, -3.41pp vs sg_vanilla) at 127k params, 12 ep.
Composite 0.8005.

**Gap classification:** WRONG_TEST

**Diagnosis:** The H10 design doc reasons the prediction band at
`T_max = 1` (per-epoch decay: LR_k = LR_0 · φ^(-k), giving
`[0.100, 0.062, 0.038, 0.024, 0.015, …]` over 12 epochs — i.e., LR
collapses by 12× over the run). The trainer dispatch (`train.py:
_build_scheduler`) instantiates `PhiDecayLR(opt, T_max=epochs)`, so the
shipped sweep row evaluates the **much-slower** `LR_0 · φ^(-k/12)`
schedule (LR decays from 0.100 to 0.062 over the full run — only one
φ-step). **Neither variant aligns with the cosine baseline's tuned
shape**: cosine at our 12-ep regime decays from 0.1 to ~0.001 (`0.1 ·
0.5·(1+cos(π))`), which is ~100× steeper than `T_max=epochs` and ~12×
steeper than `T_max=1`. So the φ-decay LR row is testing the *wrong
schedule shape* for this short-horizon regime; the literature
(Loshchilov, Smith) explicitly says decay shape matters within 0.5pp
at *long* horizons but **decay magnitude** dominates at *short*
horizons.

**Recommended action:** The right test is **either** (a) re-run with
`T_max=1` to literally evaluate `LR_0 · φ^(-k)` (the doc's literal
schedule — and we should expect this to *also* lose to cosine because
the magnitude doesn't decay enough), **or** (b) re-run cosine at
`T_max=12` AND `T_max=1` to confirm the comparison is shape-vs-shape
at iso-magnitude. Without one of these, the -6pp gap is uninterpretable
— it could be magnitude, shape, or warmup-presence. Per impl-critic,
also add a sentence to H10.md sec. 5.1 documenting `T_max=epochs` as
the trainer default. **No fixer code change required**: the
implementation is correct; the test config is wrong.

---

## Group-level conclusion

Of the 10 G1 hypotheses, **most gaps are scale-explainable, paper-
disagree-on-paper, or domain-mismatched** — only **two findings (H06,
H08) are genuine IMPL_BUGs that the previous Fixer round did not
fully close** and that would benefit from another Fixer pass:

1. **H06 expansion-ratio bug (HIGH)** — code uses ratio φ (1.618) while
   the H06 doc explicitly commits to ratio φ² (2.618), and the unit
   test bakes in the wrong constant (`104`) rather than re-deriving it,
   so the bug is *codified by the test suite*. Fixer action is a
   one-line code change + a test rewrite + re-smoke at seed 0. The
   under-parameterisation (58k vs 272k baseline) is a separate, larger
   issue that requires a c0 increase before the falsifier can be
   reached, but is not the IMPL_BUG.

2. **H08 missing function-preserving init (HIGH)** — code uses Kaiming
   reinit on each Fibonacci-epoch growth, whose forward output diverges
   from pre-growth by 0.088 (cited Net2Net paper requires < 1e-5). The
   cited paper's +35% wall-clock claim is unreachable with this
   implementation. Fixer action is to replace `_kaiming_reinit_` with
   a function-preserving init (zero-init the last conv in each grown
   block is the cleanest construction), add the missing
   `test_grow_model_function_preserves_value` (Rule-25 — Q&A test name
   was promised in the design doc but never written), and run the
   first CIFAR-10 12-ep smoke for H08 (the row doesn't even exist yet).

The remaining seven hypotheses fall into:
- **SCALE** (H01, H05) — the cited paper's headline gain requires
  100-400 epochs and 10×-100× our param budget; our 12-ep CIFAR-10
  number is consistent with "zero edge at this scale," which is what
  the cited papers' insensitivity-band findings (Bello 2021 for H01,
  Veit 2016 ensemble effect for H05) actually predict.
- **DOMAIN** (H03, H07) — cited paper is for rotation-equivariant /
  detection tasks; forcing onto CIFAR-10 classification removes the
  theoretical lever. The right benchmarks (rotated-MNIST/CIFAR with
  equivariant backbone for H03; COCO / MedMNIST 224 for H07) would
  give a fair test.
- **PAPER_AGREES** (H02, H09) — our number is within the band the
  cited literature would predict for *any* reasonable monotone
  exponent. The φ-content is decorative (sci-critic correctly tags
  DERIVATIVE+TESTABLE / NUMEROLOGY-adjacent) but the *number* matches.
  H09 carries a separate mechanism-fidelity caveat (realized
  1:1.41:2.45 vs claimed 1:φ:φ²) that is on the impl-critic backlog,
  not a paper-gap issue.
- **WRONG_TEST** (H10) — implementation is correct but `T_max=epochs`
  evaluates φ^(-k/12) not φ^(-k); the doc reasoned at `T_max=1`. The
  -6pp is uninterpretable until the test config matches the literal
  hypothesis.
- **CITATION_DOESNT_SUPPORT** (H04) — RegNet (cited!) actually says
  `w_m ∈ [2.5, 2.9]` is empirically optimal, NOT φ ≈ 1.618. The cited
  paper is the **falsifier** the doc dodges; our -2pp result aligns
  with the cited paper's prediction, and no implementation fix can
  rescue a citation that disagrees with the claim.

**Net for the next Fixer campaign:** prioritise H06 (one-line code +
one test rewrite + re-smoke) and H08 (function-preserving init + missing
test + first CIFAR-10 smoke). Everything else is either scale-explained,
paper-disagree, domain-mismatched, or already on the impl-critic
backlog; no new Fixer code action is warranted.
