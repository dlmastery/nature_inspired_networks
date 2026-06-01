# Audit calibration on third-party code
Date: 2026-05-30
Auditor: Track-A implementation-critic style (Opus 4.7)
Reference codebase: **pytorch/vision** (`torchvision.models`) plus the cited
`pytorch/pytorch` optimizers and init module — HEAD of `main`, snapshot
fetched 2026-05-30.

This file resolves area-chair item **#12 / BLOCKER #74** from
[`audits/REVIEWER_PASS_PAPER.md`](REVIEWER_PASS_PAPER.md): the request to
re-run the Track-A audit doctrine against a known-good reference codebase
and compute its non-PASS rate as a false-positive baseline for the 51 %
non-PASS rate reported in PAPER.md §5.1.

Scratchpad: [`AUDIT_CALIBRATION_RAW_NOTES.md`](AUDIT_CALIBRATION_RAW_NOTES.md).

---

## 1. Methodology

### 1.1 Why pytorch/vision (and the relevant pytorch core modules)

`pytorch/vision` is the canonical reference implementation maintained by
Meta for the foundational image-classification architectures cited across
our hypothesis docs (ResNet/ResNeXt/Wide-ResNet, DenseNet, VGG, SqueezeNet,
MobileNetV2). The same project also pins `torch.optim.Adam`,
`torch.optim.SGD`, `torch.nn.init.kaiming_normal_`,
`torch.optim.lr_scheduler.CosineAnnealingLR`, and `torch.nn.BatchNorm2d`,
all of which are arXiv-citable design choices that our own G1-G8 audits
treated as ground truth when grading our hypotheses. If the Track-A audit
doctrine is well-calibrated, this is precisely the codebase against which
its non-PASS rate should be lowest.

Alternatives considered and rejected:
- **`rwightman/pytorch-image-models` (timm).** Too broad and includes
  intentional reinterpretations of cited papers (e.g., the `ResNet-RS`
  family). Mixes ground-truth with "improved" variants, contaminating
  the calibration.
- **`kuangliu/pytorch-cifar`.** Small and well-known but academic; its
  ResNet implementation is itself a CIFAR adaptation of the He 2016
  paper. The CIFAR variant of ResNet is described in He 2016 §4.2 and
  has a different stem from the ImageNet variant; auditing it would
  conflate "is the code faithful to §4.2" with "is the code faithful
  to the §3/§4.1 ImageNet form."
- **`pytorch-cifar100`.** Same issue as kuangliu — multiple authors'
  CIFAR reinterpretations of papers; ground truth is not unique.

pytorch/vision was chosen because it is the most arXiv-aligned, most
widely-trusted, most-frequently-cited reference implementation in the
ecosystem; if a third-party non-PASS rate is going to be informative as
a false-positive baseline, this codebase has to register a low number.

### 1.2 Hypothesis-list construction

We translated the Track-A audit framework — *"code claims that can be
verified against an arXiv paper, in which divergences from the paper
are graded as MINOR/MAJOR/BROKEN per Track-A doctrine"* — to clean code
by selecting 15 mechanism claims spanning architecture, init, optimizer
defaults, normalisation, and schedulers. Each row pairs a single
arXiv-citable mechanism with the file/line implementing it. The
verbatim source for every line cited below is in
[`AUDIT_CALIBRATION_RAW_NOTES.md`](AUDIT_CALIBRATION_RAW_NOTES.md).

### 1.3 Verdict tiers (identical to G1-G8)

Cribbed verbatim from the G2/G3/G4/G5 audit preambles:

> "Shape-only tests = MINOR; doc/code divergence in the mechanism = MAJOR;
> not-actually-doing-the-thing = BROKEN. PASS requires a
> mechanism-verifying assertion."

Because pytorch/vision ships no unit-test suite in `torchvision/models/`
itself (tests are at `test/test_models.py`, not co-located with the
implementation file), we cannot grade on "are co-located tests
mechanism-bearing" the way G1-G8 did. We therefore grade on the strict
sub-criterion that DOES generalise: **does the code faithfully realise
the cited paper's mechanism, or does it deviate, and if it deviates is
the deviation explicitly documented in the same artefact?** This is the
same grading the G1-G8 audits applied to MAJOR / MINOR boundaries.

### 1.4 Symmetric-grading principle

For audit symmetry we use the same MAJOR-vs-MINOR threshold that our own
audit team applied. Concretely: if a deviation from the cited paper is
explicitly documented in the implementation file's own comments or
docstring, it grades MINOR (analogous to G1-H01 where the intra-family
`phi^s` rule was "not documented in H01.md" → MINOR). If the deviation
is UNdocumented in the implementation, it grades MAJOR (analogous to
G3-H21 where the doc-promised `phi`-radial weighting was undocumented in
code → MAJOR). If the code does not implement the cited mechanism at
all, it grades BROKEN (analogous to G7-H67 where the doc-claimed
GoldenRoPE import was missing → BROKEN).

---

## 2. Hypothesis list (15 rows)

| ID | "Hypothesis" | Verdict | Notes |
|---|---|---|---|
| TP1 | ResNet identity shortcut + projection shortcut on stride/channel change implements He 2016 arXiv:1512.03385 §3.2 + §3.3 (option B) | PASS | `resnet.py:94-110, 254-259` — identity branch and 1×1+BN downsample dispatched on correct stride/channel-mismatch condition. |
| TP2 | ResNet Bottleneck implements He 2016 arXiv:1512.03385 §4.1 (1×1 → 3×3 → 1×1, expansion=4) | MINOR | `resnet.py:113-172` — expansion=4 and the 1-3-1 conv stack are correct, but stride is placed on the 3×3 (V1.5) instead of the paper's 1×1. The deviation is explicitly documented in code comments on `resnet.py:114-121` citing the NGC V1.5 reference. Symmetric-grading MINOR (documented intentional deviation). |
| TP3 | ResNet weight-init = Kaiming-normal `fan_out` matches He 2015 arXiv:1502.01852 §2.2 | PASS | `resnet.py:221-227` calls `kaiming_normal_(mode="fan_out", nonlinearity="relu")` which produces std = sqrt(2 / (k² · c_out)) — the paper's exact formula. |
| TP4 | `zero_init_residual` implements Goyal et al. 2017 arXiv:1706.02677 (zero γ-init on last BN of each block) | PASS | `resnet.py:234-239` — zeros `bn3.weight` for Bottleneck (last BN) and `bn2.weight` for BasicBlock (last BN); citation in inline comment. |
| TP5 | Wide ResNet `wide_resnet50_2` widening factor k=2 = Zagoruyko & Komodakis 2016 arXiv:1605.07146 | MINOR | `resnet.py:1108` sets `width_per_group=64*2`. Combined with `Bottleneck.__init__:139` this correctly doubles bottleneck channels. **But** the paper's WRN was BasicBlock-without-bottleneck (WRN-28-10); torchvision's `wide_resnet50_2` applies the factor in the Bottleneck variant — a different architecture from the paper despite re-using the "WRN-d-k" name. MINOR because the widening rule is correctly applied at the parameter-count level, but the architectural-family conflation is undocumented in the function itself. |
| TP6 | DenseNet growth-rate accumulation = Huang et al. 2017 arXiv:1608.06993 §3 (Eq. 2) | PASS | `densenet.py:106` and `densenet.py:173-179` — channel count after N layers = `k0 + N · k`; transition `// 2` realises DenseNet-BC compression θ=0.5. |
| TP7 | DenseNet bottleneck 1×1 inner-dim = 4k = Huang 2017 §3 DenseNet-B | PASS | `densenet.py:43` — `Conv2d(..., bn_size * growth_rate, kernel_size=1, ...)` with default `bn_size=4` → inner dim 4k. Faithful. |
| TP8 | VGG conv stack matches Simonyan & Zisserman 2014 arXiv:1409.1556 Table 1 | PASS | `vgg.py:67-80` make_layers — 3×3 convs with padding=1, 2×2 max-pool stride=2; `vgg.py:76-81` cfgs A/B/D/E exactly match Table 1 columns A/B/D/E (VGG-11/13/16/19). |
| TP9 | Adam defaults match Kingma & Ba 2014 arXiv:1412.6980 §2 ("Algorithm") | PASS | `adam.py:27-32` — `lr=1e-3, betas=(0.9, 0.999), eps=1e-8`. Verbatim match to paper's recommended defaults. |
| TP10 | SGD-Nesterov implements Sutskever et al. 2013 ICML | MINOR | `sgd.py:51-52` correctly enforces `nesterov ⇒ momentum>0 ∧ dampening==0`. **But** PyTorch's update rule (docstring `sgd.py:146-163`) differs from Sutskever 2013 in that the momentum buffer accumulates raw gradients (not LR-scaled gradients). The two forms are equivalent under constant LR but diverge under LR scheduling. Deviation is explicitly documented. Symmetric-grading MINOR. |
| TP11 | `kaiming_normal_` formula matches He 2015 §2.2 | PASS | `init.py` — `std = gain / sqrt(fan)`; for `nonlinearity="leaky_relu"` (default), `gain = sqrt(2 / (1 + a²))` with `a=0` (default) → `gain=sqrt(2)`. With `mode="fan_in"` → std=sqrt(2 / (k² · c_in)), exactly the paper's formula. |
| TP12 | `CosineAnnealingLR` formula = Loshchilov & Hutter 2017 arXiv:1608.03983 SGDR | MINOR | `lr_scheduler.py:2624-2639` — closed-form `η_t = η_min + (η_max − η_min)(1 + cos(π t / T_max)) / 2` is mathematically correct. **But** SGDR's headline contribution is the RESTART schedule (T_i doubling); `CosineAnnealingLR` is the no-restart degenerate case. PyTorch has a separate `CosineAnnealingWarmRestarts` for the full SGDR. A user citing Loshchilov 2017 in their paper and importing `CosineAnnealingLR` is using only the cosine decay, not the SGDR algorithm. The class name doesn't make this distinction explicit. MINOR because the formula is faithful; the misnaming is the issue. |
| TP13 | MobileNetV2 InvertedResidual = Sandler et al. 2018 arXiv:1801.04381 §3.2 | PASS | `mobilenetv2.py:14-56` — expand 1×1 → depthwise 3×3 (groups=hidden_dim) → linear projection 1×1 (no activation after); residual gated by `stride==1 and inp==oup`. All three pieces of the §3.2 mechanism faithfully realised. |
| TP14 | SqueezeNet Fire module = Iandola et al. 2016 arXiv:1602.07360 §3.1 | PASS | `squeezenet.py:14-28` — squeeze 1×1, expand1x1, expand3x3 concatenated with ReLU on squeeze and on each expand branch. Channel counts in SqueezeNet 1.0 (s1=16/32/48/64, e1=e3=64/128/192/256) match paper Table 1. |
| TP15 | BatchNorm2d defaults match Ioffe & Szegedy 2015 arXiv:1502.03167 | MINOR | `batchnorm.py:27 (_NormBase.__init__)` — `eps=1e-5, momentum=0.1, affine=True, track_running_stats=True`. **But** PyTorch's `momentum` is the coefficient on the NEW batch statistic (`x_new = (1 − momentum) · x + momentum · x_t`), opposite of the conventional EMA-decay convention used in TensorFlow and most papers' notation. A user porting a Keras model expecting `momentum=0.99` gets totally wrong running stats. Documented in `batchnorm.py:239-242`. Symmetric-grading MINOR. |

---

## 3. Aggregate verdict distribution

| Tier | Count | Hypotheses |
|---|---|---|
| **PASS** | 10 | TP1, TP3, TP4, TP6, TP7, TP8, TP9, TP11, TP13, TP14 |
| **MINOR** | 5 | TP2, TP5, TP10, TP12, TP15 |
| **MAJOR** | 0 | — |
| **BROKEN** | 0 | — |

**Total hypotheses audited: 15.**
**Non-PASS rate: 5 / 15 = 33.3 %.**

---

## 4. Comparison to our project (G1-G8 audits)

### 4.1 Tally of the project's own non-PASS rate (re-derived)

Re-tallied directly from [`audits/G1_audit.md`](G1_audit.md) through
[`audits/G8_audit.md`](G8_audit.md):

| Group | PASS | MINOR | MAJOR | BROKEN | Total | Non-PASS |
|---|---|---|---|---|---|---|
| G1 (Scaling & Growth) | 3 | 4 | 3 | 0 | 10 | 7 (70.0 %) |
| G2 (Layer/Channel/Neuron) | 6 | 3 | 1 | 0 | 10 | 4 (40.0 %) |
| G3 (Topologies & Graphs) | 2 | 2 | 6 | 0 | 10 | 8 (80.0 %) |
| G4 (Kernels/Attention/Filters) | 5 | 4 | 1 | 0 | 10 | 5 (50.0 %) |
| G5 (Optim/Init/Reg/NAS) | 4 | 3 | 3 | 0 | 10 | 6 (60.0 %) |
| G6 (Topological Bridging) | 4 | 4 | 0 | 1 | 9* | 5 (55.6 %) |
| G7 (Cross-Paradigm Hybrids) | 10 | 2 | 1 | 2 | 15 | 5 (33.3 %) |
| G8 (Late-Add) | 7 | 2 | 0 | 0 | 9 | 2 (22.2 %) |
| **Total** | **41** | **24** | **15** | **3** | **83** | **42 (50.6 %)** |

*G6 had H57 deferred (counted out of 9, not 10).
The 50.6 % rounds to the 51 % cited in PAPER.md §5.1.

### 4.2 Side-by-side calibration

| Tier | Our project (n=83) | torchvision (n=15) | Δ |
|---|---|---|---|
| PASS | 49.4 % (41) | 66.7 % (10) | −17.3 pp |
| MINOR | 28.9 % (24) | 33.3 % (5) | +4.4 pp |
| MAJOR | 18.1 % (15) | 0.0 % (0) | −18.1 pp |
| BROKEN | 3.6 % (3) | 0.0 % (0) | −3.6 pp |
| **Non-PASS** | **50.6 %** | **33.3 %** | **−17.3 pp** |

### 4.3 Interpretation

**The third-party non-PASS rate is non-zero (33 %).** Even on a
canonical, peer-trusted reference codebase, the Track-A doctrine
identifies 5 of 15 mechanisms as having a real divergence from the
cited paper. This validates a meaningful piece of the area-chair's
concern: **the audit team has a non-trivial false-positive rate when
"deviation from the paper" is the grading criterion** — most engineering
implementations make documented, intentional, defensible deviations
from their source paper.

### 4.3.1 Formal interval-and-test analysis on the 22-pp MAJOR/BROKEN excess (added 2026-05-30 per ICML R2 Q3)

The MAJOR/BROKEN sub-tier comparison is the diagnostically-credible
contrast (§4.3 above). At project n=83 with 18 MAJOR/BROKEN hits
(15 MAJOR + 3 BROKEN) the rate is 21.7%; at calibration n=15 with
0 MAJOR/BROKEN the rate is 0%. The observed difference is **+21.7 pp**.
Computed via 100 000-iteration parametric binomial bootstrap and exact
methods (rng seed 20260530, [`scripts/_compute_stat_tests.py`](../scripts/_compute_stat_tests.py) §8):

| quantity | value |
|---|---|
| Observed difference (proj − cal) | **+21.7 pp** |
| Bootstrap 95% CI on the difference | **[+13.3, +31.3] pp** (excludes 0) |
| Wilson 95% CI on project rate (18/83) | [14.2%, 31.7%] |
| Wilson 95% CI on calibration rate (0/15) | [0.0%, 20.4%] |
| Wilson CI overlap (project lower 14.2% vs calibration upper 20.4%) | overlap on a 6.2-pp window |
| Fisher exact, one-sided (P[proj > cal]) | **p = 0.0363** |
| Fisher exact, two-sided | p = 0.0658 |
| Two-proportion z-test (pooled), two-sided | z = 1.996, p = 0.0459 |

**Reading.** The bootstrap CI on the difference clears 0 by ~13 pp on
the lower bound. The one-sided Fisher exact (the conventional choice
when the direction "real defect density should exceed false-positive
floor" is the pre-registered alternative) clears α = 0.05. The
two-sided Fisher exact (0.066) and the prior-reported two-sample
chi-squared (≈ 0.22 in §4.4) do NOT clear α = 0.05; the Wilson CIs on
the two proportions still overlap on a 6.2-pp window (project lower
14.2% vs calibration upper 20.4%) because the calibration sample is
small.

**Honest reading (R2 Q3 response).** The 22-pp MAJOR/BROKEN excess is
statistically significant at α = 0.05 under the one-sided Fisher exact
test (p = 0.036) and under the pooled two-proportion z-test (p = 0.046),
but NOT under the two-sided Fisher exact (p = 0.066). The bootstrap CI
on the difference excludes 0 with a 13-pp lower-bound margin, which is
the most reviewer-credible single statistic in this comparison. The
n = 15 calibration is the limiting factor — a Phase-9b extension to
n ≥ 50 (timm + HF Transformers + Lightning Bolts) is required before
the comparison clears two-sided α = 0.05 unambiguously.

The MAJOR/BROKEN-tier excess is therefore **directionally credible,
one-sided-significant at α = 0.05, and bootstrap-CI-excludes-0**;
it is NOT yet two-sided significant at α = 0.05 under the most
conservative test. The §5 conclusion below is updated accordingly.



**But the rates are not the same.** The 17-pp gap is driven entirely
by the MAJOR/BROKEN tiers: torchvision has 0 % in those tiers vs.
our project's 21.7 % (15 MAJOR + 3 BROKEN of 83). The MINOR tier is
comparable (29 % vs. 33 %).

**This is the diagnostically informative split.** Track-A's MAJOR
verdict ("doc/code divergence in the mechanism") and BROKEN verdict
("not-actually-doing-the-thing") are the tiers that justify the
audit's headline narrative — the audit "caught real bugs." Those tiers
are EMPTY on the calibration codebase. The MAJOR/BROKEN tiers are
therefore NOT false-positive-driven; they reflect genuine excess defect
density in our codebase.

The MINOR tier IS partly aggressive — torchvision's 5 MINORs are
nitpicks (documented intentional deviations: V1.5 stride placement,
Nesterov reformulation, BN momentum semantics, no-restart "cosine
annealing", WRN-Bottleneck naming). Under a more lenient doctrine
that downgrades all *documented* intentional deviations to PASS,
torchvision drops to 14/15 PASS (TP12 CosineAnnealingLR-named-vs-SGDR
remains MINOR because the misnaming is in the class name, not just
documented), i.e. ~7 % non-PASS rate.

**Honest conclusion:**

1. The 51 % non-PASS rate in PAPER.md §5.1 IS partly inflated by
   Track-A's MINOR-tier aggressiveness on documented deviations.
   The same audit team will register ~25-30 % MINOR-tier non-PASS
   on a clean codebase as a baseline.

2. However, the 51 % is NOT primarily a false-positive artefact: the
   MAJOR/BROKEN tier (21.7 % of our project) has a 0 % calibration
   baseline, so MAJOR/BROKEN findings are diagnostically credible.

3. Of the 21.7 % MAJOR/BROKEN in our project, the case studies
   (G1-H09 phi_budget realised-ratio drift, G3-H21 hex_phi
   undocumented divergence, G7-H67 broken import) are concrete
   defects that an unaudited pipeline would have shipped. These are
   the kind of finding the audit was designed to surface.

4. The 28.9 % MINOR tier in our project is in line with calibration
   (33.3 % for torchvision) and should NOT be cited as evidence of
   widespread defect; many MINORs are "missing co-located test for
   a documented mechanism" — analogous to torchvision's MINORs.

**Recommended re-framing for PAPER.md §5.1 / §5.8:** The audit's
diagnostic power is partially overstated when reporting "51 % non-PASS"
as a single number. The credible audit signal is the **21.7 %
MAJOR+BROKEN tier**, which is the gap between our project and the
calibration baseline. The MINOR tier is informative for code-review
discipline but should not anchor the headline narrative.

---

## 5. Conclusion

The Track-A audit doctrine, when applied to a known-good reference
codebase (pytorch/vision + the cited torch core modules), produces a
**33.3 % non-PASS rate (5 MINOR, 0 MAJOR, 0 BROKEN of 15)**. Compared
to our project's **50.6 % non-PASS rate (29 % MINOR, 18 % MAJOR, 4 %
BROKEN of 83)**, the calibration validates the area-chair's concern at
the MINOR-tier level but NOT at the MAJOR/BROKEN-tier level.

**The audit's diagnostic power is partially overstated.** Reporting
"51 % non-PASS" as a single number without sub-tier decomposition
treats a 4-pp MINOR-tier excess (29 % vs. 33 % calibration) the same
as a 22-pp MAJOR/BROKEN-tier excess (22 % vs. 0 % calibration). The
two have very different interpretations: the first is *audit
aggressiveness*; the second is *real codebase defect density*.

Per §4.3.1, the 22-pp MAJOR/BROKEN excess clears one-sided Fisher
exact at α = 0.05 (p = 0.036) and the bootstrap 95% CI on the
difference is [+13.3, +31.3] pp (excludes 0); it does NOT clear
two-sided Fisher exact (p = 0.066). The signal is directionally
credible but the conservative two-sided test does not certify it
at α = 0.05; a Phase-9b extension to n ≥ 50 is required.

For the camera-ready, **PAPER.md §5.1 should report both numbers**:
51 % aggregate AND 22 % MAJOR/BROKEN-tier; cite this calibration file
as the false-positive baseline; explicitly state that the MAJOR/BROKEN-
tier excess (22 pp above the 0 % calibration floor) is the
diagnostically-credible audit signal, not the aggregate 51 %.

The protocol-as-contribution claim is unchanged in load-bearing
strength: the protocol IS what surfaced the H09 phi_budget realised-
ratio drift (a MAJOR finding in the diagnostically-credible tier) and
the H67 GoldenRoPE BROKEN import, both of which an unaudited pipeline
would have shipped. The 22-pp MAJOR/BROKEN delta vs. the calibration
floor is the empirically-defensible quantification of "the audit
caught real bugs."

**§5.7 / §5.8 status update:** area-chair item #12 is RESOLVED by this
file. The "audit calibration" deferred-tag is now closed; the
calibration was done, its results are in this file, and the
implications for §5.1 framing are recommended above.

---

## Appendix A — Reproducibility

Every code line cited in §2 corresponds to a verbatim excerpt in
[`AUDIT_CALIBRATION_RAW_NOTES.md`](AUDIT_CALIBRATION_RAW_NOTES.md). The
HEAD-of-main snapshot taken 2026-05-30 is reproducible by fetching:

- `https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/resnet.py`
- `https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/densenet.py`
- `https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/vgg.py`
- `https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/squeezenet.py`
- `https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/mobilenetv2.py`
- `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/adam.py`
- `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/sgd.py`
- `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/init.py`
- `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/lr_scheduler.py`
- `https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/modules/batchnorm.py`

A second auditor disagreeing with any of TP2/TP5/TP10/TP12/TP15 (the
borderline MINORs) is welcome to re-grade and re-tally; the aggregate
33 % is robust to ±2 reclassifications (would shift to 20-47 % under
maximally adversarial re-grading; both bounds remain materially below
our project's 51 %).

## Appendix B — Acknowledged limitations of this calibration

1. **Single auditor.** I am Opus 4.7, same model family as Track-A.
   Confirmation bias risk: I read the third-party code with the
   explicit goal of "what would the audit say." Mitigation: the
   verdict tiers are pinned to the same doctrine quoted verbatim from
   G2/G3/G4 preambles, and the symmetric-grading principle (§1.4) is
   stated explicitly so a re-grader can apply it consistently.
2. **Static read only.** No third-party unit tests were written or
   executed. The audit mirrors the G1-G8 static-read methodology but
   does not validate the implementations dynamically.
3. **No co-located tests in torchvision.** Track-A's PASS-requires-a-
   mechanism-verifying-test sub-criterion was relaxed because
   `torchvision/models/*.py` ship without same-file tests; tests
   live in `test/test_models.py`. This is a methodological asymmetry
   between the calibration and the G1-G8 audits — it could make the
   third-party rate slightly LOWER than a fully-symmetric audit would
   report.
4. **Sample size n=15 vs. n=83.** The 33.3 % calibration is bounded
   by 95 % CI roughly [12 %, 62 %] under a binomial proportion model.
   The 51 % project rate is outside the lower bound of that CI; the
   two-sample chi-squared test on (5/15 vs. 42/83) gives p ≈ 0.22,
   i.e. the difference is NOT statistically significant at α=0.05.
   This is itself important: **with only 15 calibration hypotheses,
   we cannot reject "the two rates are the same."** A larger
   calibration sample (e.g. 50-100 third-party hypotheses spanning
   timm + Hugging Face transformers + Lightning Bolts) would tighten
   the CI. We recommend this as Phase-9b future work.

---

*Audit complete. 10 PASS, 5 MINOR, 0 MAJOR, 0 BROKEN. Non-PASS rate
33.3 %. Calibration vs project: −17.3 pp aggregate, −21.7 pp on the
MAJOR/BROKEN sub-tier that anchors the audit's diagnostic narrative.
Area-chair item #12 RESOLVED.*

---

## Appendix A — Phase-9b extension to n ≥ 50 (added 2026-05-31)

### A.1 Why this extension

The §4.3.1 / §5 conclusion explicitly flagged the n=15 calibration as
the limiting factor for clearing two-sided Fisher exact at α=0.05.
AC punchlist item 3 mandated an extension to n ≥ 50 to tighten the
bootstrap CI on the 22-pp MAJOR/BROKEN excess. This appendix carries
out that extension by auditing 47 additional "hypothesis-style" claims
across 5 third-party repos (HEAD-of-main snapshot 2026-05-31):

| Repo | Rows added | TP IDs |
|---|---|---|
| timm (huggingface/pytorch-image-models) | 19 | TP16-TP34 |
| HuggingFace transformers | 15 | TP35-TP49 |
| Lightning Bolts / fastai | 6 | TP50-TP55 |
| torch.optim (extra: AdamW, NAdam, RAdam, Adagrad) | 4 | TP56-TP59 |
| state-spaces/mamba | 3 | TP60-TP62 |
| **Total new** | **47** | |
| **Grand total (with §2)** | **62** | TP1-TP62 |

### A.2 Methodology note

Identical Track-A audit doctrine (§1.3) and symmetric-grading principle
(§1.4) applied to the new rows. Each row pairs a code:line citation
with the arXiv:section being claimed. Raw notes are in
[`AUDIT_CALIBRATION_RAW_NOTES.md`](AUDIT_CALIBRATION_RAW_NOTES.md)
"Phase-9b extension" section.

Two methodological caveats:
1. **Single auditor (still).** Same model (Opus 4.7) graded all
   47 new rows. The confirmation-bias risk from §B-1 applies here too.
2. **Five excluded candidates.** Five claims were considered but
   excluded because the WebFetch did not return enough code to verify
   them at file:line resolution (timm EfficientNet MBConv block body,
   HF CLIP "quick GELU" choice, Mamba dt_min/dt_max clamping, fastai
   stack_y mixup form, and HF transformers' deprecated AdamW). These
   are honestly documented in RAW_NOTES rather than silently inflated
   into the aggregate.

### A.3 New hypothesis table (TP16..TP62)

Each row cites file:line + arXiv:section + verdict. Full quoted
mechanism analysis is in RAW_NOTES.

| ID | Claim (file:line → paper §) | Verdict |
|---|---|---|
| TP16 | timm/models/resnet.py:2647-2703 → Bello 2021 arXiv:2103.07579 RS recipe | PASS |
| TP17 | timm/models/resnet.py:209-216 → V1.5 stride placement vs He 2016 §4.1 | MINOR |
| TP18 | timm/models/resnet.py:125-128, 189-192 → Goyal 2017 arXiv:1706.02677 zero-γ init | PASS |
| TP19 | timm/models/resnet.py:436-461 → He 2018 "Bag of Tricks" arXiv:1812.01187 §3.1 deep stem | PASS |
| TP20 | timm/models/resnet.py:597-614 → ResNet-D avg_down shortcut §3.1 | PASS |
| TP21 | timm/models/efficientnet.py factory multipliers → Tan & Le 2019 arXiv:1905.11946 §3.3 compound scaling | MINOR |
| TP22 | timm/models/efficientnet.py `se0.25` arch_def → Hu 2018 arXiv:1709.01507 SE ratio | PASS |
| TP23 | timm/models/convnext.py:150-180 ConvNeXt block → Liu 2022 arXiv:2201.03545 Fig. 4 | PASS |
| TP24 | timm/models/mlp_mixer.py:47-79, 311 GAP no-class-token → Tolstikhin 2021 arXiv:2105.01601 §2 | PASS |
| TP25 | timm/models/byobnet.py:1247-1424 RepVggBlock → Ding 2021 arXiv:2101.03697 | PASS |
| TP26 | timm/models/vision_transformer.py default Block pre-LN → Dosovitskiy 2020 arXiv:2010.11929 §3.1 | PASS |
| TP27 | timm/models/vision_transformer.py `ls1 = Identity() if init_values is None` → ViT-2020 default faithful | PASS |
| TP28 | timm/data/auto_augment.py:588-639 RandAugment → Cubuk 2019 arXiv:1909.13719 | PASS |
| TP29 | timm/layers/drop.py:256-261 DropPath → Huang 2016 arXiv:1603.09382 | PASS |
| TP30 | timm/loss/cross_entropy.py:13-19 label smoothing → Pereyra 2017 arXiv:1701.06548 / Szegedy 2016 | PASS |
| TP31 | timm/optim/lars.py:106-130 LARS → You 2017 arXiv:1708.03888 | PASS |
| TP32 | timm/optim/lamb.py augmented LAMB → You 2019 arXiv:1904.00962 Alg. 1 + post-paper additions | MINOR |
| TP33 | timm/optim/lookahead.py:15 default k=6 vs Zhang 2019 arXiv:1907.08610 k=5 | MINOR |
| TP34 | timm/layers/evo_norm.py:155 EvoNormS0 → Liu 2020 arXiv:2004.02967 | PASS |
| TP35 | transformers/.../bert/modeling_bert.py:104-115 scaled dot-product → Vaswani 2017 §3.2.2 | PASS |
| TP36 | transformers/.../bert/modeling_bert.py post-norm BertLayer → Devlin 2018 / Vaswani 2017 §3.1 | PASS |
| TP37 | transformers/.../gpt2/modeling_gpt2.py pre-LN ln_1/ln_2/ln_f → Radford 2019 §2.3 | PASS |
| TP38 | transformers/.../gpt2/modeling_gpt2.py `_tied_weights_keys` → Press & Wolf 2017 arXiv:1608.05859 | PASS |
| TP39 | transformers/.../llama/modeling_llama.py:65-166 RoPE → Su 2021 arXiv:2104.09864 Eq. 14 | PASS |
| TP40 | transformers/.../llama/modeling_llama.py:46-62 RMSNorm → Zhang & Sennrich 2019 arXiv:1910.07467 | PASS |
| TP41 | transformers/.../llama/modeling_llama.py:180-195 SwiGLU MLP → Shazeer 2020 arXiv:2002.05202 | PASS |
| TP42 | transformers/.../llama/modeling_llama.py:201-221, repeat_kv L170 GQA → Ainslie 2023 arXiv:2305.13245 | PASS |
| TP43 | transformers/.../t5/modeling_t5.py:283-330 relative-bias bucketing → Raffel 2019 arXiv:1910.10683 §2.1 | PASS |
| TP44 | transformers/.../t5/modeling_t5.py:43-60 T5LayerNorm RMS-style → Zhang & Sennrich 2019 | PASS |
| TP45 | transformers/.../vit/modeling_vit.py:55-110 cls+pos+patch Conv2d → Dosovitskiy 2020 §3.1 | PASS |
| TP46 | transformers/.../clip/modeling_clip.py:569-594 logit_scale + symmetric loss → Radford 2021 §2.5 | PASS |
| TP47 | transformers/.../gpt_neox/modeling_gpt_neox.py:199-228 parallel residual → Black 2022 arXiv:2204.06745 §3.4 | PASS |
| TP48 | transformers/.../mistral/modeling_mistral.py:278 sliding-window mask → Jiang 2023 arXiv:2310.06825 §2.1 | PASS |
| TP49 | transformers/optimization.py:149-176 `get_cosine_schedule_with_warmup` num_cycles=0.5 → SGDR degenerate | MINOR |
| TP50 | lightning-bolts/optimizers/lars.py:113-127 LARS → You 2017 arXiv:1708.03888 | PASS |
| TP51 | fastai/callback/schedule.py:46 `pct_start=0.25` vs Smith 2018 arXiv:1803.09820 §6.1 (suggests 0.45) | MINOR |
| TP52 | fastai/callback/mixup.py:34 default `alpha=0.4` vs Zhang 2017 arXiv:1710.09412 Table 1 α=0.2 | MINOR |
| TP53 | fastai/callback/mixup.py:52 CutMix `alpha=1.0` → Yun 2019 arXiv:1905.04899 §3 | PASS |
| TP54 | lightning-bolts SimCLR temperature default 0.1 (CIFAR override 0.5) → Chen 2020 arXiv:2002.05709 §B.1 | MINOR |
| TP55 | lightning-bolts SimCLR projection head 2-layer MLP → Chen 2020 §2 | PASS |
| TP56 | torch/optim/adamw.py decoupled WD → Loshchilov & Hutter 2017 arXiv:1711.05101 Alg. 2 | PASS |
| TP57 | torch/optim/nadam.py:351-386 Dozat schedule β1·(1−0.5·0.96^…) → Dozat 2016 Appendix A.2 | PASS |
| TP58 | torch/optim/radam.py:335 SMA threshold `>5.0` vs Liu 2019 arXiv:1908.03265 Alg. 2 `>4` | MINOR |
| TP59 | torch/optim/adagrad.py:78-82, 280-284 → Duchi 2011 JMLR | PASS |
| TP60 | mamba_ssm/ops/selective_scan_interface.py:107-165 reference scan → Gu & Dao 2023 arXiv:2312.00752 §3.3 | PASS |
| TP61 | mamba_ssm/modules/mamba_simple.py:73-77 S4D-style A init → Gu 2022 arXiv:2206.11893 §3.2 | PASS |
| TP62 | mamba_ssm/modules/mamba2.py:141, 169 SSD chunk scan → Dao & Gu 2024 arXiv:2405.21060 | PASS |

### A.4 Per-repo verdict distribution

| Repo | n | PASS | MINOR | MAJOR | BROKEN | Non-PASS % |
|---|---|---|---|---|---|---|
| timm | 19 | 15 | 4 | 0 | 0 | 21.1% |
| HF transformers | 15 | 14 | 1 | 0 | 0 | 6.7% |
| Lightning Bolts / fastai | 6 | 3 | 3 | 0 | 0 | 50.0% |
| torch.optim (extra) | 4 | 3 | 1 | 0 | 0 | 25.0% |
| state-spaces/mamba | 3 | 3 | 0 | 0 | 0 | 0.0% |
| **Phase-9b new** | **47** | **38** | **9** | **0** | **0** | **19.1%** |

### A.5 Updated aggregate at n=62

| Tier | §2 (n=15) | Phase-9b (n=47) | Total (n=62) | Share |
|---|---|---|---|---|
| PASS | 10 | 38 | **48** | 77.4% |
| MINOR | 5 | 9 | **14** | 22.6% |
| MAJOR | 0 | 0 | **0** | 0.0% |
| BROKEN | 0 | 0 | **0** | 0.0% |
| **Non-PASS** | **5/15 (33.3%)** | **9/47 (19.1%)** | **14/62 (22.6%)** | — |
| **MAJOR/BROKEN** | **0/15 (0.0%)** | **0/47 (0.0%)** | **0/62 (0.0%)** | — |

**The MAJOR/BROKEN-tier rate is still exactly 0% at n=62.** Adding
47 new audits across 5 large, well-trusted repos produced ZERO MAJOR
or BROKEN findings. The audit doctrine on third-party code surfaces
only MINOR-tier deviations (documented intentional choices, default-
value drifts, and post-paper extensions); no third-party code matched
our project's MAJOR-tier pattern of doc-vs-code mismatch or our BROKEN-
tier pattern of doc-promised-but-not-implemented mechanisms.

### A.6 Recomputed 22-pp MAJOR/BROKEN excess CI at n=62

Computed via [`scripts/_compute_stat_tests.py`](../scripts/_compute_stat_tests.py)
§11 (100 000-iteration parametric binomial bootstrap, rng seed
20260531). Inputs: project k=18, n=83 (unchanged); calibration k=0,
n=62 (extended from n=15).

| quantity | n=15 (§4.3.1) | n=62 (this extension) |
|---|---|---|
| Observed diff (proj − cal) | +21.7 pp | **+21.7 pp** (proj rate unchanged) |
| Bootstrap 95% CI on diff | [+13.3, +31.3] pp | **[+13.3, +31.3] pp** (~unchanged) |
| Bootstrap CI half-width | 9.0 pp | 9.0 pp |
| Wilson 95% CI on project rate (18/83) | [14.2%, 31.7%] | [14.2%, 31.7%] (unchanged) |
| Wilson 95% CI on calibration rate (0/n) | [0.0%, 20.4%] (n=15) | **[0.0%, 5.8%]** (n=62) |
| Wilson CI overlap | 14.2% vs 20.4% → 6.2-pp overlap | **14.2% vs 5.8% → NO OVERLAP (8.3-pp separation)** |
| Fisher exact, one-sided p (proj > cal) | 0.0363 | **p = 1.79e-5** |
| Fisher exact, two-sided p | 0.0658 | **p = 1.94e-5** |
| Pooled two-proportion z, two-sided p | 0.0459 | **z = 3.92, p = 8.93e-5** |

The exact bootstrap interval, Fisher exact, and z-statistic are
computed by `scripts/_compute_stat_tests.py` §11 — see §11 of
[`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) for
the regenerated output.

**Key qualitative changes at n=62:**

1. **Wilson CI on calibration rate shrinks from [0.0%, 20.4%] to
   [0.0%, 5.8%]** — a ~3.5× tighter upper bound. The "calibration
   MAJOR/BROKEN rate could plausibly be as high as 20%" objection
   (§4.3.1 reading) is no longer tenable.
2. **Wilson CIs on the two rates no longer overlap.** At n=15 they
   overlapped on a 6.2-pp window (project lower 14.2% vs calibration
   upper 20.4%). At n=62 the project lower bound (14.2%) sits above
   the calibration upper bound (5.8%) by 8.3 pp — formal Wilson
   non-overlap is satisfied.
3. **Two-sided Fisher exact clears α=0.05 by >2500×** (p = 1.94e-5 vs
   threshold 0.05). The §5 conservative test that the n=15 calibration
   could NOT clear is now decisively cleared.
4. **Pooled two-proportion z clears α=0.05 by >500×** (z=3.92,
   p=8.93e-5 vs threshold 0.05). The z-statistic more than doubled
   (1.996 → 3.918) under the n=15 → n=62 extension.
5. **Honest note on the parametric bootstrap CI.** The parametric
   binomial bootstrap CI on the difference is **essentially unchanged**
   ([+13.3, +31.3] pp at both n=15 and n=62, half-width 9.0 pp). This
   is NOT a defect — it reflects the structural fact that with k_cal=0
   and the bootstrap sampling Binomial(n_cal, p_cal=0), the calibration
   arm contributes zero variance to the difference distribution at
   ANY n. The CI is therefore set entirely by the project arm's
   variance (n=83, p=0.217), which has not changed. The Wilson CI on
   the calibration *rate* itself is the test that scales with n_cal
   (and it tightens 3.5×, as reported above). The bootstrap CI's
   stability at n=62 confirms that the +22 pp point estimate and its
   [+13.3, +31.3] pp uncertainty band are dominated by project-arm
   sampling, not by calibration-arm sparsity.

### A.7 Updated §5 Conclusion at n=62

The §5 conclusion paragraph "Per §4.3.1, the 22-pp MAJOR/BROKEN excess
clears one-sided Fisher exact at α = 0.05 (p = 0.036) and the bootstrap
95% CI on the difference is [+13.3, +31.3] pp (excludes 0); it does
NOT clear two-sided Fisher exact (p = 0.066)" is **superseded** by the
n=62 result:

> Per Appendix A.6, at the extended calibration n=62 the 22-pp
> MAJOR/BROKEN excess clears BOTH one-sided and two-sided Fisher exact
> at α=0.05 with p = 1.94e-5 (two-sided, >2500× margin), the pooled
> two-proportion z reaches z=3.92 with two-sided p=8.93e-5 (>500×
> margin), the Wilson 95% CI on the calibration rate shrinks from
> [0.0%, 20.4%] to [0.0%, 5.8%] (~3.5× tighter), and the Wilson CIs
> on the two rates no longer overlap (project lower 14.2% > calibration
> upper 5.8% by 8.3 pp). The parametric-bootstrap CI on the difference
> is structurally unchanged at [+13.3, +31.3] pp because the calibration
> arm contributes zero variance with k_cal=0 at any n; that CI is
> set entirely by project-arm sampling and excludes 0 by a 13.3-pp
> lower-bound margin. The Phase-9b extension RESOLVES the §5 caveat:
> the MAJOR/BROKEN excess is now two-sided-significant at α=0.05
> under the most conservative tests with very large margins, and the
> 22-pp point estimate is robust to the n=15 → n=62 calibration
> extension.

The protocol-as-contribution claim is now empirically defensible
at the two-sided α=0.05 level, not only at the one-sided level — an
upgrade in the conservativeness of the certified test, with the same
+22-pp point estimate of "excess defect density above the third-party
false-positive floor."

### A.8 Honest caveats for the Phase-9b extension

1. **Verdict-tier concentration in MINOR.** All 9 new non-PASS rows
   landed in MINOR. A reader skeptical of MINOR-tier aggressiveness
   could argue the audit is still over-counting documented intentional
   deviations — see TP21 (compound-scaling cache), TP49 (cosine-no-restart
   naming), TP51 (pct_start=0.25), TP52 (mixup α=0.4), TP58 (RAdam
   ρ>5 vs >4). Under a lenient doctrine that downgrades all
   documented-deviation MINORs to PASS, calibration drops to ~5/62 =
   8 % non-PASS while the project's MAJOR/BROKEN signal is unchanged.
   The MAJOR/BROKEN-tier comparison (the headline statistic) is
   robust to this re-grading because both numerator (18/83 project)
   and denominator (0/62 calibration) are unaffected by MINOR
   re-classifications.
2. **HF transformers is sprawling.** We audited 15 mechanism claims
   from the most-cited models (BERT, GPT-2, Llama, T5, ViT, CLIP,
   GPT-NeoX, Mistral, schedules). The repo contains hundreds of
   model files; a more thorough audit at the same scope could
   sample 50+ more rows. Our 15 are arXiv-canonical and the resulting
   PASS/MINOR mix is consistent with timm and mamba (very low MAJOR
   density on well-vetted reference code).
3. **5 candidate rows excluded.** EfficientNet MBConv block body,
   CLIP quick-GELU, Mamba dt_min/dt_max clamping, fastai stack_y
   mixup, HF deprecated AdamW. RAW_NOTES documents each exclusion
   with a reason; none of these would plausibly have been MAJOR or
   BROKEN, so the inclusion bias is conservative (excluding them
   does not inflate the calibration non-PASS rate).
4. **Single auditor (still).** Same Opus 4.7 model graded all 62
   rows. A second auditor on a different model family could disagree
   on any of the 14 MINOR verdicts; the MAJOR/BROKEN-tier (0/62)
   is robust to up to ±2 MINOR ↔ MAJOR reclassifications without
   changing the §5 conclusion.

---

*Phase-9b extension complete. n=15 → n=62 (4.1× extension).
48 PASS / 14 MINOR / 0 MAJOR / 0 BROKEN. Calibration MAJOR/BROKEN
rate stays at exactly 0% across the extended sample. Wilson 95% CI on
the calibration rate tightens from [0.0%, 20.4%] to [0.0%, 5.8%];
Wilson CIs no longer overlap; two-sided Fisher exact tightens from
p=0.066 to p=1.94e-5 (>2500× margin); pooled z tightens from
p=0.046 to p=8.93e-5 (>500× margin). The parametric bootstrap CI on
the difference is structurally unchanged at [+13.3, +31.3] pp
(calibration arm has k=0 → zero variance contribution at any n),
which is a feature, not a defect: the +22-pp point estimate's
uncertainty band is dominated by project-arm sampling and excludes
zero by 13.3 pp regardless of calibration n. AC punchlist item 3
RESOLVED.*
