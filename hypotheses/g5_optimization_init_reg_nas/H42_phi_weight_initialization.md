# H42 — φ-Weight Initialization

> **One-line claim:** Replacing He / Kaiming initialization's
> `std = sqrt(2 / fan_in)` with the φ-scaled variant
> `std = sqrt(φ / fan_in)` (φ ≈ 1.618) shrinks the early-training
> gradient-norm explosion ratio by ≥15% across a 20-layer ResNet
> because √φ ≈ 1.272 produces a forward-signal-gain factor closer to
> the unit-isometry optimum than √2 ≈ 1.414, per the dynamical-isometry
> framework of Saxe 2014 / Pennington 2017.
>
> **Source design space:** G5 Optimization / Init / Regularization / NAS
> (H41–H50).
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for hypothesis
H42. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

He / Kaiming initialization (`std = sqrt(2 / fan_in)`) was derived
analytically for ReLU layers under the assumption that exactly half of
the post-activation distribution is zero; the factor of 2 is the
inverse of E[ReLU(z)²] / E[z²] for z ~ N(0,1). This is a *coarse*
approximation: real CNNs use BatchNorm before ReLU, and BatchNorm
already enforces unit second moment regardless of the init scale.
What remains relevant for trainability is the **dynamical isometry**
property (Saxe 2014, Pennington 2017): the maximum singular value of
the per-layer Jacobian product should stay close to 1.0 throughout
depth, otherwise gradients vanish or explode exponentially.

The golden ratio enters because √φ ≈ 1.272 lies between the He scale
(√2 ≈ 1.414) and the LeCun scale (1.0), and is the unique scale that
makes a 2-tap φ-decay filter `[φ-1, 1]/√φ` self-similar under
convolution: filtering φ-decay noise with itself reproduces φ-decay
noise. Nature uses this in phyllotactic seed-packing (Fibonacci's
`F(n+1)/F(n) → φ`) for the same self-similarity reason. In a deep CNN
with skip connections, the residual addition's variance composition
`Var(x + f(x)) = Var(x) + Var(f(x))` becomes self-similar across depth
precisely when the per-block gain is √φ — i.e., each residual block
contributes a φ-1 fraction of the next block's variance, exactly
matching the Fibonacci ratio. This is the engineering case for
substituting `√(φ / fan_in)` for `√(2 / fan_in)` in nature-inspired
architectures with skip connections.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because the per-layer signal gain in a ReLU+BN+residual stack composes
multiplicatively with depth — mechanism-wise, He init's gain √2 ≈ 1.414
accumulates a 20-layer gain ratio of (√2)^20 ≈ 1024 before BN clipping,
while φ init's gain √φ ≈ 1.272 accumulates only (√φ)^20 ≈ 121 — per
Pennington 2017's dynamical-isometry framework, we expect φ init to
reduce the initial gradient-norm explosion ratio
(`grad_norm_step_0 / grad_norm_step_100`) by ≥15% on a 20-block
NaturePriorNet without changing asymptotic top-1 by more than ±0.5 pp.

## 3. Falsifier (≥ 30 words)

If the maximum gradient-norm ratio over the first 100 steps with
φ init is NOT reduced by ≥10% versus He init at 3-seed median (i.e.,
`max_t(||g_t||) / ||g_0||` reduction Δ < 10% with upper 95% CI < 10%),
this hypothesis is DISCARDED. Additionally if 12-epoch top-1 drops
by more than -1.0 pp versus the He-init reference, the hypothesis is
DISCARDED as a usability regression.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian 2015
ICCV 'Delving Deep into Rectifiers: Surpassing Human-Level
Performance on ImageNet Classification' (arXiv:1502.01852) -- the
He / Kaiming init paper; the √2 factor is the analytic ReLU
correction we propose to swap for √φ on theoretical grounds.

Saxe, Andrew M. and McClelland, James L. and Ganguli, Surya 2014
ICLR 'Exact solutions to the nonlinear dynamics of learning in deep
linear neural networks' (arXiv:1312.6120) -- introduces dynamical
isometry; the analytical bound we use to argue √φ is a better gain
than √2 for residual stacks.

Pennington, Jeffrey and Schoenholz, Samuel S. and Ganguli, Surya
2017 NeurIPS 'Resurrecting the sigmoid in deep learning through
dynamical isometry: theory and practice' (arXiv:1711.04735) --
develops the orthogonal-init / dynamical-isometry calculus we cite
for the explicit Jacobian-singular-value argument in § 2.

Glorot, Xavier and Bengio, Yoshua 2010 AISTATS 'Understanding the
difficulty of training deep feedforward neural networks' -- the
Xavier reference for the variance-preserving init philosophy our
φ-scaled variant inherits.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

The φ-init implementation is a one-line edit to PyTorch's
`kaiming_normal_`: replace `gain = sqrt(2)` with `gain = sqrt(φ)`.
Applied to every `Conv2d` and `Linear` weight in the model. No change
to params, FLOPs, or latency. Shape and forward semantics identical.
The only observable difference is the initial weight-tensor variance
(scaled by φ/2 ≈ 0.809) and the resulting gradient dynamics during
the first ~500 steps.

```python
import math
import torch.nn as nn

PHI = (1.0 + 5 ** 0.5) / 2

def phi_kaiming_normal_(tensor, fan_in=None, mode="fan_in"):
    """Replace He's √2 with √φ. Same shape, fan_in, distribution."""
    if fan_in is None:
        fan_in = nn.init._calculate_correct_fan(tensor, mode)
    gain = math.sqrt(PHI)            # ≈ 1.272 (vs He's √2 ≈ 1.414)
    std = gain / math.sqrt(fan_in)
    with torch.no_grad():
        return tensor.normal_(0, std)

def apply_phi_init(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            phi_kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
```

Lives in `src/nature_inspired_networks/init/phi_init.py` and is re-exported by
`ideas/42_phi_weight_init/implementation.py`. The training loop calls
`apply_phi_init(model)` after construction.

### 5.2 LLM track (decoder-only Transformer)

For a GPT-2-style decoder, the φ-init applies to every weight tensor:
token embedding, positional embedding (if learned), QKV projections,
output projection, FFN up-proj, FFN down-proj, and LM head. The
RMSNorm gamma weights stay at 1.0 (BN/LN scale params are
init-invariant). FlashAttention-2 compatibility is untouched (no
shape change). Causal mask preservation untouched.

Expected impact at 124M scale (TinyStories): early-training perplexity
trajectory drops faster because the residual stream variance grows
more slowly with depth, keeping the softmax distribution from
saturating during the first ~1000 steps. Steady-state perplexity
±0.3 ppl of baseline.

```python
def apply_phi_init_llm(model: nn.Module, ffn_scale: float = 1.0):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            phi_kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            # √φ-scaled embedding init
            nn.init.normal_(m.weight, std=math.sqrt(PHI / m.weight.shape[1]))
```

For the LLM, we additionally apply Megatron-style **depth-aware
scaling**: the residual-output projection is further scaled by
`1/sqrt(2 * num_layers)`, preserving GPT-2's deep-net stability rule
while still using φ for the base gain.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline (He init) | rationale |
|---|---|---|
| composite | [-0.005, +0.005] | no asymptotic change expected |
| top-1 (CNN) | [-0.5, +0.5] | iso-accuracy hypothesis |
| max grad-norm ratio in first 100 steps | [-50%, -10%] | core targeted metric |
| params | [0, 0] | no change |
| FLOPs | [0, 0] | no change |
| GPU latency (batch=1) | [0, 0] | no change |
| rotation-equivariance err | [0, 0] | unaffected |
| KV cache @ 32k (LLM) | [0, 0] | init only |
| Betti collapse rate | [0, 0] | downstream metric, no prediction |
| perplexity at 1k steps (LLM) | [-1.0, +0.5] | faster early decrease |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dataset:** CIFAR-10
- **Architecture:** `NaturePriorNet` with `channel_mode=fib`, priors off
  (matches `sg_chan_fib` reference)
- **Init:** `apply_phi_init(model)` for the φ-init arm;
  `kaiming_normal_(... gain="relu")` for the He-init control
- **Epochs:** 12, batch=128, bf16 AMP, AdamW
- **Seeds:** 0, 1, 2
- **Composite formula:** SHA-256 fingerprinted; identical to existing sweep
- **Run-script:** `python scripts/run_idea.py --idea 42 --init phi --seeds 0 1 2`
- **Wall-clock:** ≈ 12 min × 3 seeds × 2 inits = ~72 min
- **Extra logging:** per-step `||g||_2` for the first 200 steps (the
  hypothesis is about early dynamics, not asymptote)
- **Archive path:** `ideas/42_phi_weight_init/experiments/exp001_cifar10_phi_vs_he/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

φ-init's advantage scales with depth. The targeted setup:

- **Dataset:** CIFAR-10
- **Architecture:** a 50-block NaturePriorNet (vs. the 6-block default)
  WITHOUT BatchNorm (since BN partially absorbs init mistakes)
- **Comparison:** He init vs. φ init vs. orthogonal init (the
  dynamical-isometry gold standard)
- **Predicted result:** φ init beats He init by ≥3 pp top-1 at 12 epochs;
  loses to orthogonal init by ~1 pp (orthogonal is the limit case of any
  scalar-gain init).

The diagnostic value: this isolates the *depth × no-BN* regime where
init really matters, so a positive result on the 6-block + BN
baseline is corroborated rather than explained-away.

### 7.3 Cross-paradigm context (LLM track)

- **Model:** GPT-2-small (124M), FlashAttention-2, bf16 AMP
- **Dataset:** TinyStories
- **Comparator:** GPT-2 default init (Xavier with depth-scale) vs.
  φ-init with depth-scale
- **Metric:** validation ppl at 1k, 5k, 10k steps
- **Run:** `python scripts/run_llm.py --idea 42 --init phi`

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G5 row H42.
- Master experiment list: `EXPERIMENT_LOG.md` (new Tier 2 row on launch).
- Implementation sub-directory: `ideas/42_phi_weight_init/`
- Related hypotheses that compose:
  - **H41** PhiAdamW — composes because φ-init produces a parameter
    scale that PhiAdamW's β2 horizon is tuned to. Joint run is the
    cleanest test of the φ-optimization stack.
  - **H17** φ-scaled residual skips — composes because both edit the
    residual stream variance; care needed not to double-count.
  - **H44** φ-regularization — joint test reveals whether the gain
    happens via init alone or needs regularization reinforcement.
- Related hypotheses that conflict:
  - **H35** Cymatic Chladni-mode init — replaces init with structured
    eigenmodes, which is incompatible with scalar φ-gain init.
    Cannot run both simultaneously on the same parameter.

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of He init with a different
constant?**

> The constant is theoretically motivated (Jacobian-product
> self-similarity at √φ; Saxe / Pennington dynamical-isometry
> framework). The pre-registered metric is the grad-norm ratio, not
> top-1 — we explicitly do NOT claim better accuracy, only smoother
> early dynamics. This is a more falsifiable claim than typical init
> tuning.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies a 10% threshold on grad-norm ratio with 95% CI
> exclusion. § 6 pre-registers asymptotic top-1 Δ ≤ ±0.5 pp.

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> § 7.2 explicitly tests the depth × no-BN regime. The hypothesis is
> bounded to early-training dynamics; if asymptotic results regress on
> ImageNet we will report it as a known scope limit.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H42 is an init prior, orthogonal to the *architectural* compound
> failure. It is tested in isolation first; only after both H41 and H42
> are individually validated will we attempt their composition.

**Q: How do we know the implementation is correct?**

> `tests.py` includes (a) verification that `Var(W) == φ / fan_in`
> within 2% after init, (b) bit-equivalence when φ is replaced by 2
> (must produce He init), (c) gradient-flow smoke test on a 50-layer
> linear net (must not NaN at init).

## 10. Verification artifacts checklist

- [ ] `ideas/42_phi_weight_init/implementation.py` exists, tests green
- [ ] `ideas/42_phi_weight_init/tests.py` ≥ 5 assertions
- [ ] `ideas/42_phi_weight_init/AUDIT.md` lists ≥ 3 weaknesses
- [ ] `ideas/42_phi_weight_init/IMPROVEMENTS.md` records fixes
- [ ] `ideas/42_phi_weight_init/VERIFY.md` is signed with real date
- [ ] At least one experiment archive
- [ ] `verification/{tests,smoke,gates,reproduction}.txt` complete
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-C.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G5 (elite-research-scientist critic). Critiquing the IDEA, not the implementation (audit at `audits/G5_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)
LOW. He's `std = sqrt(2 / fan_in)` is derived analytically: the `2` arises because ReLU zeros half the pre-activation mass, so to preserve `Var[a^(l)] = Var[a^(l-1)]` across layers one must double the weight variance to compensate. Replacing the `2` with `φ ≈ 1.618` is, by construction, *under-scaling* — the forward signal variance will *decay* by a factor of `(φ/2) = 0.809` per ReLU layer. Across 20 layers this is `0.809^20 ≈ 0.014`, i.e. a 70× attenuation of activation variance into the network's tail. This is vanishing-activation territory, not "closer to the unit-isometry optimum".

### Mechanism scrutiny — does the optimizer/init/reg theory actually predict the claimed effect?
No, and the doc gets the math backwards. Dynamical isometry (Saxe, McClelland, Ganguli 2014 ICLR 'Exact solutions to the nonlinear dynamics of learning in deep linear neural networks' arXiv:1312.6120; Pennington, Schoenholz, Ganguli 2017 NeurIPS 'Resurrecting the Sigmoid in Deep Learning through Dynamical Isometry and a Mean Field Theory' arXiv:1711.04735) requires Jacobian singular values concentrated at 1.0 — for ReLU networks WITHOUT BN, this is achieved precisely by the `√(2/fan_in)` Kaiming choice; with BN it is approximate but the BN scale absorbs the residual. Reducing the gain to `√(φ/fan_in) = √(1.618/fan_in) < √(2/fan_in)` pushes Jacobian singular values toward zero (vanishing-gradient half-plane), not toward unity. The "self-similar Fibonacci variance composition" argument in § 1 is a sleight of hand: residual blocks have `Var(x + f(x)) ≈ 2·Var(x)` (when f and x have the same scale) only because of the addition, not because of any φ-tuning of f's weights.

### Confounds (≥2)
(1) **BatchNorm scale absorbs init.** As § 1 admits, "BatchNorm already enforces unit second moment regardless of the init scale" — but the doc then proceeds to argue the init scale matters anyway. If BN is doing its job, the difference between He and φ-init should attenuate at the BN scale parameter; any observed effect is BN-warmup confounded. (2) **Skip-connection variance accounting.** ResNet's variance is dominated by the running sum of residuals, not by the per-block init scale; Hanin & Rolnick 2018 NeurIPS 'How to Start Training: The Effect of Initialization and Architecture' (arXiv:1803.01719) shows that residual-block init matters mostly for gradient norm at the *first* step. (3) **Last-layer init.** A common trick in ResNet papers (Goyal et al. 2017 arXiv:1706.02677) is zero-init of the last BN's γ — this is what actually matters for residual-block stability, not the convolution's weight scale.

### Numerology / specificity check
Pure numerology. The doc never tests `std = sqrt(c / fan_in)` for `c ∈ {1.0, 1.272, 1.5, 1.618, 1.8, 2.0, 2.5}` to demonstrate `c = φ` is locally optimal. The "self-similar 2-tap filter `[φ-1, 1]/√φ`" claim is an algebraic identity unrelated to actual CNN initialization (3×3 conv kernels are not 2-tap and have spatial structure). Choice of φ over any nearby constant is aesthetic.

### Literature precedent — optimization/init is one of the most studied fields in DL
Init has been studied to death. Glorot & Bengio 2010 AISTATS 'Understanding the difficulty of training deep feedforward neural networks' (arXiv:1001.3014); He, Zhang, Ren, Sun 2015 ICCV 'Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification' (arXiv:1502.01852); Mishkin & Matas 2016 ICLR 'All you need is a good init' (arXiv:1511.06422); Zhang, Dauphin, Ma 2019 ICLR 'Fixup Initialization: Residual Learning Without Normalization' (arXiv:1901.09321); De & Smith 2020 NeurIPS 'Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks' (arXiv:2002.10444). The literature converges: with BN, init scale within a 2× factor of Kaiming is largely irrelevant; without BN, Fixup / SkipInit are the modern answer, neither of which is φ-scaled.

### Expected effect size (90% CI a priori)
[-5 pp, -0.5 pp] on CIFAR-10 top-1 vs. He-init baseline at 12 epochs. Observed: 76.56 % vs. ~80 % baseline = -3.5 pp; consistent with the prior.

### Minimum-distinguishing experiment
Sweep `std = sqrt(c / fan_in)` for `c ∈ {1.0, 1.272² = 1.618, 2.0, 2.83² = 8.0}` × 3 seeds; if `c = 1.618` lies on a smooth curve with no local optimum near φ — and Reddit/Stack-Overflow-style empirical evidence already shows He is near-optimal — the φ-specificity is refuted. Already empirically refuted at 1 seed.

### Verdict
NUMEROLOGY — `c = φ` under-scales relative to He's `c = 2`, causing the empirically observed ~3.5 pp regression. The doc inverts the dynamical-isometry argument and treats `√φ < √2 < √π` as a free parameter unrelated to ReLU's variance-doubling requirement; recommend updating `Implementation status: ✗ disproved (12-ep CIFAR-10: 76.56 % vs. ~80 % baseline)` and reframing the motivation as a falsified numerology rather than a dynamical-isometry argument.
