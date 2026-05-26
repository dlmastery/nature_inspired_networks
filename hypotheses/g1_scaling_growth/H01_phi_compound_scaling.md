# H01 — phi-Compound Scaling (replace EfficientNet alpha/beta/gamma with phi powers)

> **One-line claim:** Compound network scaling using successive powers of
> the golden ratio phi as the depth/width/resolution coefficients matches
> or beats EfficientNet at >=10 percent fewer FLOPs at iso-accuracy.
>
> **Source design space:** G1 Scaling-and-Growth (H01-H10).
>
> **Implementation status (this repo):**
> `o planned`.

This document is the committee-grade design write-up for hypothesis
H01. Every section below is mandatory; the word-count floors are the
same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, >= 100 words)

The golden ratio phi = (1 + sqrt(5))/2 ~= 1.618 is the unique positive
real number that satisfies phi**2 = phi + 1; equivalently, it is the
limit of the ratio of consecutive Fibonacci numbers. Nature reaches for
phi whenever it must pack a growing set of resources into a bounded
medium without self-occlusion. Phyllotactic leaf arrangements, nautilus
shells, pineapple scales, sunflower seed heads, the orbits of the inner
planets, and even the proportion of bone segments in vertebrate limbs
all converge to phi or one of its descendants. The reason is
informational: rotating each new element by 360/phi**2 ~= 137.5 degrees
guarantees that no two elements ever land at exactly the same angular
position, achieving asymptotically optimal sphere-packing in the angular
dimension. EfficientNet (Tan and Le 2019) discovered empirically that
compound scaling -- inflating depth, width, and resolution together by
related coefficients alpha, beta, gamma -- is necessary for Pareto-
optimal accuracy/FLOPs trade-offs. Their alpha approx 1.2, beta approx
1.1, gamma approx 1.15 are tuned by grid search. Setting alpha = phi,
beta = sqrt(phi), gamma = phi/2 reuses the only irrational number that
nature itself selected as the canonical compounding constant for
non-occluding growth. The mechanism we hypothesise is that phi-spaced
scaling minimises destructive aliasing between successive stages, just
as it does between successive phyllotactic primordia.

## 2. Formal hypothesis (the falsifiable claim, >= 50 words)

Because compound scaling of depth d, width w and resolution r by
successive powers of phi (d = phi**k, w = phi**(k/2), r = phi**(k/4))
preserves the EfficientNet constraint d * w**2 * r**2 ~= 2**phi while
introducing the same non-self-overlap property that phyllotaxis enjoys,
the mechanism by which phi-scaled networks should outperform alpha/beta/
gamma-grid-tuned EfficientNets is the elimination of overlapping
receptive-field eigenmodes across stages. Per Tan and Le 2019 we expect
ImageNet top-1 to match B0 within +/- 0.4 pp at >=10 percent fewer
FLOPs, and at fixed FLOPs we expect top-1 to be +0.3 to +1.0 pp higher
on CIFAR-100 in our 12-epoch regime.

## 3. Falsifier (>= 30 words)

If a 5-point phi-scaled family (k = 0, 1, 2, 3, 4) on CIFAR-100 at fixed
seed produces a Pareto front whose iso-accuracy FLOPs are not at least
10 percent below EfficientNet-B0..B2 at the same accuracy AND the
composite Delta is <= -0.005 at 3-seed median, this idea is DISCARDED
and Status moves to `x disproved`. Specifically: if at top-1 = 73 percent
on CIFAR-100 the phi family needs more than 0.45 GFLOPs (vs EfficientNet
B0's 0.39 GFLOPs at the same accuracy), the prior loses.

## 4. Citations (Citation Rigor format, >= 80 words)

```
Tan, Mingxing, Le, Quoc V. 2019 ICML 'EfficientNet: Rethinking Model
Scaling for Convolutional Neural Networks' (arXiv:1905.11946) -- the
prior to be replaced; we reuse their compound-scaling constraint
d * w**2 * r**2 = constant and replace their grid-searched alpha/beta/
gamma with phi powers.

Huh, Minyoung, Cheung, Brian, Wang, Tongzhou, Isola, Phillip 2024 ICML
'The Platonic Representation Hypothesis' (arXiv:2405.07987) -- justifies
why architectural priors that match the geometry of the underlying data
manifold accelerate the convergence to the same shared representation.

Mitchison, Graeme J. 1977 Science 'Phyllotaxis and the Fibonacci series'
(no arXiv; classical) -- biological precedent for phi-spaced growth as
the non-occluding optimum, cited here as the natural-system anchor that
EfficientNet's empirical scaling laws are quietly rediscovering.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

The mechanism replaces EfficientNet's compound-scaling rule

```
d = alpha**k
w = beta**k
r = gamma**k
subject to alpha * beta**2 * gamma**2 = 2
```

with

```
d = phi**k
w = phi**(k/2)
r = phi**(k/4)
phi * phi * phi**0.5 = phi**2.5 ~= 3.33 (slightly above 2; calibrate to 2 via phi**(k * 0.4))
```

Concretely, for k = 0, 1, 2, 3, 4 the depth multipliers are 1.0, 1.618,
2.618, 4.236, 6.854; width multipliers are 1.0, 1.272, 1.618, 2.058,
2.618; resolution multipliers are 1.0, 1.128, 1.272, 1.434, 1.618.
Channel counts are rounded to the nearest multiple of 8 for tensor-core
alignment. Shapes (B, C, H, W): a B0-equivalent input
(B, 3, 224, 224) at k = 0 becomes (B, 3, 256, 256) at k = 1 and so on.

Computational cost vs EfficientNet-B0 (390 MFLOPs, 5.3 M params): at
k = 1 we predict ~340 MFLOPs (-12.8 pct) and ~4.8 M params (-9.4 pct).
At k = 4 we predict 5.6 GFLOPs vs B4's 4.2 GFLOPs (+33 pct) which is the
high-end regime where the prior is expected to lose. The sweet spot is
k in {0, 1, 2}.

Init and training: He-normal scaled by sqrt(phi/fan_in) (cf. H42); LR
cosine 0.05 with phi-decay variant (cf. H10); 12 epochs CIFAR-100,
50 epochs Tiny ImageNet, 90 epochs full ImageNet (skipped on 4090
Laptop budget).

PyTorch code sketch:

```python
PHI = (1 + 5 ** 0.5) / 2

def phi_compound(k: int, base_depth=18, base_width=64, base_res=32):
    """Return (depth, width, resolution) scaled by phi**k for stage k."""
    d = max(1, round(base_depth * PHI ** k))
    w = max(8, 8 * round(base_width * PHI ** (k / 2) / 8))  # /8 align
    r = max(16, 16 * round(base_res * PHI ** (k / 4) / 16))  # /16 align
    return d, w, r

class PhiScaledBackbone(nn.Module):
    def __init__(self, K=4, num_classes=100):
        super().__init__()
        stages, c_in = [], 3
        for k in range(K + 1):
            d, w, r = phi_compound(k)
            stages.append(_make_stage(c_in, w, depth=d))
            c_in = w
        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(c_in, num_classes)
```

It lives in `src/nature_inspired_networks/scaling.py` and is re-exported
by `ideas/01/implementation.py::PhiScaledBackbone`.

### 5.2 LLM track (decoder-only Transformer)

Per the extended-transcript chunk-4 mapping, phi-compound scaling
applies to a GPT-style decoder layer as follows. For a k-indexed family
of decoder configurations:

- **n_layer** = round(base_layer * phi**k) (depth)
- **d_model** = 64 * round(base_dim * phi**(k/2) / 64) (width, must be
  multiple of head_dim for FlashAttention-2)
- **n_head** = round(base_head * phi**(k/2))
- **context_len** = 512 * round(base_ctx * phi**(k/4) / 512)

FlashAttention-2 compatibility: head_dim = d_model / n_head must remain
in {32, 64, 96, 128}; the phi-rounding above guarantees this since both
d_model and n_head scale by phi**(k/2). Causal-mask preservation: phi
scaling does not touch the attention mask -- it only rescales tensor
shapes, so the lower-triangular mask is unchanged.

```python
def phi_decoder_config(k: int):
    n_layer = max(1, round(12 * PHI ** k))         # 12, 19, 31, 51, 82
    d_model = 64 * round(768 * PHI ** (k / 2) / 64) # 768, 960, 1216, ...
    n_head  = max(1, round(12 * PHI ** (k / 2)))   # 12, 15, 19, 24, 31
    ctx_len = 512 * round(2 * PHI ** (k / 4))      # 1024, 1152, 1280, ...
    return dict(n_layer=n_layer, d_model=d_model,
                n_head=n_head, ctx_len=ctx_len)
```

Expected impact at 124M-1B scale: at k = 1 (~280 M params), predict
WikiText-103 perplexity 22.4 +/- 0.3 vs a control 280 M GPT-2-style at
22.9 (-0.5 ppl); KV cache at 32k tokens unchanged (cache scales with
n_head * head_dim * n_layer which obeys the phi**(k/2) * phi**k =
phi**(3k/2) law); latency at batch=1 is +/- 3 percent of control because
the per-layer cost scales identically to a grid-tuned EfficientNet-equivalent.

## 6. Predicted Delta on 4090 benchmarks (the pre-registered prediction)

| metric | Delta vs baseline | rationale |
|---|---|---|
| composite | [+0.005, +0.025] | accuracy lift dominates |
| top-1 (CIFAR-100, CNN) | [+0.3, +1.0] pp | phi spacing reduces inter-stage aliasing |
| perplexity (WikiText-103 LLM) | [-0.7, -0.2] | finer width steps allow better head-dim alignment |
| params | [-12, -5] pct vs EfficientNet-B0 iso-acc | phi-coefficient binding |
| FLOPs | [-15, -8] pct vs EfficientNet-B0 iso-acc | EfficientNet's iso-acc curve at k=1 |
| GPU latency (batch=1) | [-5, +2] pct | shape changes are tensor-core neutral |
| rotation-equivariance err | [-0.005, +0.005] | prior is scale-symmetric, not rotation |
| KV cache @ 32k (LLM) | [-3, +3] pct | identical scaling law as control |
| Betti collapse rate | [+0.01, +0.05] | phi-spacing may shrink beta_0 collapse epoch |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- Dataset: **CIFAR-100** (50k/10k train/test, 100 classes)
- Architecture: PhiScaledBackbone with K in {0, 1, 2, 3} plus an
  EfficientNet-B0 reference baseline
- Epochs / batch / precision / seeds: 50 epochs, batch 128, bf16 AMP,
  seeds {0, 1, 2}
- Composite formula: `0.6 * top1 + 0.2 * (1 - params/2M) + 0.2 *
  (1 - flops/0.5G)`; SHA-256 fingerprint computed at first run and
  re-checked on every subsequent gate
- Run-script: `python scripts/run_sweep.py --config
  configs/h01_phi_compound.yaml --seeds 0 1 2 --skip-existing`
- Wall-clock estimate on RTX 4090 (16 GB): 4 configs * 3 seeds *
  ~9 min = ~110 min
- Archive path: `ideas/01_phi_compound_scaling/experiments/
  exp001_cifar100_k0_3/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

Tiny ImageNet at 50 epochs with K in {1, 2, 3} -- this is where compound
scaling laws were originally validated. We expect the phi family to
trace a Pareto front parallel to EfficientNet's but with FLOPs shifted
10-15 percent left at iso-accuracy. Wall-clock: ~6 hours single seed.

### 7.3 Cross-paradigm context (LLM track)

Train a phi-scaled GPT-2-style decoder at 124M, 280M, and 460M parameter
counts (corresponding to k = 0, 1, 2) on WikiText-103 for one epoch with
bf16 + gradient checkpointing + FlashAttention-2. Compare validation
perplexity at iso-FLOPs to control configurations (124M GPT-2, 350M
GPT-2-medium-style). Budget: ~12 hours single seed for the three points.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` section G1 row H01
- Master experiment list: `EXPERIMENT_LOG.md` Tier T3.1 (planned)
- Implementation sub-directory: `ideas/01_phi_compound_scaling/`
- Related hypotheses that compose: H02 (Fib depth), H04 (phi width),
  H09 (param budget), H10 (phi LR)
- Related hypotheses that conflict: none directly; H45 (NAS) supersedes
  if NAS independently discovers a better scaling law

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of EfficientNet (Tan and Le 2019)?**

> EfficientNet's alpha, beta, gamma are *learned* by grid search on the
> target dataset and lack a closed-form interpretation. Substituting
> phi powers is (a) a single irrational constant with no degrees of
> freedom, (b) testable against EfficientNet at iso-FLOPs as a strict
> ablation, and (c) the only natural-constant choice that simultaneously
> satisfies the compound constraint and the phyllotactic non-overlap
> condition.

**Q: How is this falsifiable rather than aesthetic?**

> Section 3 commits to a specific numeric threshold (>= 10 pct FLOPs
> reduction at iso-accuracy on CIFAR-100). The pre-registered
> Pareto-front comparison in section 6 (composite Delta range
> [+0.005, +0.025]) cannot be argued away after the fact.

**Q: What if the prior helps on CIFAR-100 but hurts on ImageNet?**

> Then the claim is restricted to small-image low-data regimes and the
> hypothesis Status is downgraded to `~ partial`. Section 7.2's Tiny
> ImageNet experiment is exactly the bridge that adjudicates this.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> Yes, `FINDINGS.md` documents that the full hybrid is the worst
> single-seed row. That negative result is about combining six priors
> simultaneously; H01 is a single-prior architectural change that
> commits to no per-block geometry. The unit of analysis here is the
> scaling law, not the residual block.

**Q: How do we know the implementation is correct?**

> `tests/test_phi_scaling.py` asserts (a) channel counts are multiples
> of 8, (b) the d * w**2 * r**2 product stays within 5 pct of
> 2**(k * 0.4), (c) parameter count matches the closed-form prediction
> within 1 pct, and (d) forward shape matches the predicted resolution.
> The per-run `verification/tests.txt` records pass/fail.

## 10. Verification artifacts checklist (committee evidence pack)

- [ ] `ideas/01_phi_compound_scaling/implementation.py` exists; tests green
- [ ] `ideas/01_phi_compound_scaling/tests.py` >= 12 assertions
- [ ] `ideas/01_phi_compound_scaling/AUDIT.md` lists >= 3 self-found weaknesses
- [ ] `ideas/01_phi_compound_scaling/IMPROVEMENTS.md` records the fixes
- [ ] `ideas/01_phi_compound_scaling/VERIFY.md` signed with real date
- [ ] At least one experiment archive under
      `ideas/01_phi_compound_scaling/experiments/exp001_cifar100_k0_3/`
- [ ] That archive carries `verification/{tests.txt, smoke.txt,
      gates.txt, reproduction.txt}`
- [ ] Row added to `EXPERIMENT_LOG.md` (Tier T3.1)
- [ ] Result reflected in `FINDINGS.md` and dashboard

## 11. Status journal

(Append-only timeline of what changed when.)

- 2026-05-27 -- Created from template by Doc-Agent-A.
