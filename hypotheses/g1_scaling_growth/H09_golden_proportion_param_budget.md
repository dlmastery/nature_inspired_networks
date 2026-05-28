# H09 — Golden Proportion Parameter Budget (1 : phi : phi**2 across stages)

> **One-line claim:** Allocating a fixed total parameter budget B across
> network stages in the ratio 1 : phi : phi**2 : phi**3 yields a Pareto
> point dominating uniform or geometric-2 allocation at iso-FLOPs.
>
> **Source design space:** G1 Scaling-and-Growth (H01-H10).
>
> **Implementation status (this repo):**
> `~ partial` (implicit via Fib widths in `priors.py:fibonacci_channels`).

This document is the committee-grade design write-up for hypothesis
H09.

---

## 1. Motivation (sacred / natural intuition, >= 100 words)

When a fixed resource budget must be split across a hierarchy of
processing stages, the optimal allocation depends on how the marginal
returns of each stage compare. For deep networks, deeper stages
process more abstract features that have higher information content
per parameter (because they integrate larger receptive fields). The
phi recurrence phi**(k+1) = phi**k + phi**(k-1) means that allocating
budget in proportion to phi**k automatically gives each stage exactly
the sum of what the two prior stages received, mirroring the
biological resource-inheritance rule (mollusc shells, ammonite
chambers, plant internodes). The total budget is then B = sum_{k=0}^{K}
phi**k = (phi**(K+1) - 1) / (phi - 1), which gives a closed-form
allocator: stage k receives B * (phi - 1) * phi**k / (phi**(K+1) - 1).
This is fundamentally different from the linear, geometric-2, or
EfficientNet-style allocations and provides a parameter-efficient
hierarchy where each stage's expressive capacity is mathematically
locked to the sum of the previous two stages.

## 2. Formal hypothesis (the falsifiable claim, >= 50 words)

Because golden-proportion parameter budget allocation (1 : phi : phi**2
: phi**3) per stage assigns each stage's capacity to be the sum of the
two prior stages, the mechanism by which it should dominate uniform or
geometric-2 allocation at fixed total budget B is that the hierarchical
information-integration cost grows as phi**k (matching biological
resource inheritance) per Tan and Le 2019. We expect a Pareto point on
CIFAR-100 at +0.3 to +0.8 pp top-1 at iso-FLOPs vs uniform allocation.

## 3. Falsifier (>= 30 words)

If a 4-stage phi-budget backbone (params per stage in ratio 1 : phi :
phi**2 : phi**3) on CIFAR-100 at 3-seed median achieves <= +0.0 pp
top-1 vs a uniform-budget backbone at the same total param budget B
and iso-FLOPs (+/- 5 pct), the hypothesis is FALSIFIED.

## 4. Citations (Citation Rigor format, >= 80 words)

```
Tan, Mingxing, Le, Quoc V. 2019 ICML 'EfficientNet: Rethinking Model
Scaling for Convolutional Neural Networks' (arXiv:1905.11946) -- the
prior whose compound scaling implicitly creates a near-phi budget
allocation; H09 makes the allocation rule explicit and closed-form.

He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, Sun, Jian 2016 CVPR 'Deep
Residual Learning for Image Recognition' (arXiv:1512.03385) -- ResNet
allocates more params to deeper stages (16-32-64 channels x
[3, 4, 6, 3] blocks); the implicit ratio is roughly 1 : 4 : 16 : 16,
not phi-spaced.

Frankle, Jonathan, Carbin, Michael 2019 ICLR 'The Lottery Ticket
Hypothesis: Finding Sparse, Trainable Neural Networks' (arXiv:1803.03635)
-- relevant because the lottery-ticket finding suggests param budget
matters less than allocation; H09 commits to a specific allocation
rule and tests whether it dominates the empirical optima.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

For a 4-stage backbone with target B = 1.0M params:

```
phi = 1.618
total_phi = sum(phi**k for k in range(4)) = 1 + 1.618 + 2.618 + 4.236 = 9.472
stage_params = B * phi**k / total_phi
            = [105.6k, 170.8k, 276.5k, 447.1k]
```

Channel counts per stage (assuming each stage has S blocks of
3x3 conv with c_k channels, params ~ S * 9 * c_k**2):

```
c_k = sqrt(stage_params_k / (S * 9))
For S=3: c = [62, 79, 101, 128] (rounded to mod-8 [64, 80, 104, 128])
```

This produces a channel schedule c = [64, 80, 104, 128] very close to
but not identical to standard ResNet (64, 128, 256, 512). The
*difference* is critical: 80 -> 104 -> 128 grows more slowly than
128 -> 256 -> 512, putting more budget in the early/mid stages.

Shapes for input (B, 3, 32, 32) with stride-2 between stages:
stage 0: (B, 64, 32, 32) -- 105k params
stage 1: (B, 80, 16, 16) -- 171k params
stage 2: (B, 104, 8, 8) -- 277k params
stage 3: (B, 128, 4, 4) -- 447k params
total: 1.0M params

PyTorch sketch:

```python
PHI = (1 + 5 ** 0.5) / 2

def phi_budget_channels(B: float, n_stages: int, blocks_per_stage: int):
    total_phi = sum(PHI ** k for k in range(n_stages))
    stage_params = [B * PHI ** k / total_phi for k in range(n_stages)]
    channels = [8 * max(1, round(math.sqrt(sp / (blocks_per_stage * 9))
                                  / 8)) for sp in stage_params]
    return channels

class PhiBudgetBackbone(nn.Module):
    def __init__(self, total_params=1_000_000, n_stages=4,
                 blocks_per_stage=3):
        super().__init__()
        widths = phi_budget_channels(total_params, n_stages, blocks_per_stage)
        stages, c_in = [], 3
        for k, c in enumerate(widths):
            stages.append(_make_stage(c_in, c, blocks_per_stage,
                                        stride=2 if k > 0 else 1))
            c_in = c
        self.stages = nn.Sequential(*stages)
```

Location: `src/nature_inspired_networks/priors.py` (extend
`fibonacci_channels` with explicit budget allocator).

### 5.2 LLM track (decoder-only Transformer)

In a decoder, "stages" do not exist by default. H09's LLM-track
variant maps to **per-layer parameter budget**, i.e., layers earlier
in the stack have fewer params (smaller FFN expansion) and later
layers have more. Specifically: for 12 layers, FFN expansion at layer
k = base_exp * phi**k / mean(phi**k) where mean keeps total params
constant.

For d_model = 768 and target params (FFN only) = 12 * 2 * 768 * 3072 =
56.6M:

```
expansions = [4 * 12 * phi**k / sum(phi**k for k in range(12))
              for k in range(12)]
            ~ [0.36, 0.58, 0.94, 1.52, 2.47, 4.0, 6.46, 10.46,
               16.92, 27.38, 44.30, 71.69]
```

This is impractical (later layers explode). The practical version
caps expansion at 8 and scales early layers down. Alternatively,
allocate budget to *attention* params (d_head) rather than FFN.

```python
def phi_budget_layer_ffn(d_model, n_layers, total_ffn_params):
    weights = [PHI ** k for k in range(n_layers)]
    total = sum(weights)
    return [64 * max(1, round(total_ffn_params * w / (2 * d_model *
                                                       total) / 64))
            for w in weights]
```

FlashAttention-2 compatibility: only FFN width varies; attention is
constant per layer. KV cache: unchanged. Causal mask: preserved.

Expected impact at 124M scale: WikiText-103 ppl improves by 0.2-0.4
because later layers have more capacity for abstract features.

## 6. Predicted Delta on 4090 benchmarks (the pre-registered prediction)

| metric | Delta vs baseline | rationale |
|---|---|---|
| composite | [+0.000, +0.015] | top-1 lift offsets identical params |
| top-1 (CIFAR-100, CNN) | [+0.3, +0.8] pp | better param allocation |
| perplexity (WikiText-103 LLM) | [-0.4, -0.1] | deeper-layer expansion |
| params | [-2, +2] pct | iso-budget by construction |
| FLOPs | [-5, +5] pct | iso-budget keeps FLOPs near-equal |
| GPU latency (batch=1) | [-3, +3] pct | unchanged |
| rotation-equivariance err | [-0.005, +0.005] | not affected |
| KV cache @ 32k (LLM) | [-2, +2] pct | unchanged |
| Betti collapse rate | [+0.01, +0.04] | better hierarchy |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- Dataset: **CIFAR-100**
- Architecture: 4-stage backbones at fixed B = 1.0M params; conditions
  {uniform allocation, geometric-2 (1 : 2 : 4 : 8), phi-budget
  (1 : phi : phi**2 : phi**3), reverse-phi (front-loaded)}
- Epochs / batch / precision / seeds: 50 epochs, batch 128, bf16,
  seeds {0, 1, 2}
- Composite formula: existing project formula; SHA-256 pinned
- Run-script: `python scripts/run_sweep.py --config
  configs/h09_phi_budget.yaml --seeds 0 1 2`
- Wall-clock: 4 configs * 3 seeds * ~10 min = ~120 min
- Archive: `ideas/09_golden_param_budget/experiments/
  exp001_cifar100_budget/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

CIFAR-100 at multiple budget points (B = 0.25M, 0.5M, 1.0M, 2.0M);
predict a Pareto front consistently above the uniform allocation
curve. Wall-clock: ~4 hours single seed.

### 7.3 Cross-paradigm context (LLM track)

124M decoder with phi-per-layer FFN expansion (capped at 8) on
WikiText-103, 1 epoch. Compare ppl to uniform-expansion control.
Budget: ~5 hours.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` section G1 row H09
- Master experiment list: `EXPERIMENT_LOG.md` (new T3.x row)
- Implementation sub-directory: `ideas/09_golden_param_budget/`
- Related hypotheses that compose: H01 (compound), H02 (Fib depth),
  H04 (phi width), H06 (golden bottleneck)
- Related hypotheses that conflict: any fixed-allocation rule

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of EfficientNet's width
coefficient?**

> EfficientNet uses a *global* width multiplier across all stages, not
> a *per-stage* phi-ratio allocator. H09 commits to a closed-form
> per-stage rule and tests whether it dominates EfficientNet's grid-
> searched flat scaling.

**Q: How is this falsifiable rather than aesthetic?**

> Section 3 commits to >= +0.0 pp top-1 lift at iso-budget AND iso-
> FLOPs. Either side failing falsifies.

**Q: What if the prior helps on CIFAR-100 but hurts on ImageNet?**

> Section 7.2 sweeps a Pareto front. Generalisation to ImageNet is
> out of single-GPU budget but the Pareto pattern at CIFAR-100 should
> hold qualitatively.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> Yes for combining six per-block priors at one block. H09 is a
> *global* allocation prior orthogonal to per-block geometry.

**Q: How do we know the implementation is correct?**

> `tests/test_phi_budget.py` asserts (a) total params within 5 pct of B,
> (b) consecutive stage ratios approach phi, (c) channel counts are
> mod-8.

## 10. Verification artifacts checklist (committee evidence pack)

- [ ] `ideas/09_golden_param_budget/implementation.py`; tests green
- [ ] `ideas/09_golden_param_budget/tests.py` >= 10 assertions
- [ ] `ideas/09_golden_param_budget/AUDIT.md` >= 3 weaknesses
- [ ] `ideas/09_golden_param_budget/IMPROVEMENTS.md`
- [ ] `ideas/09_golden_param_budget/VERIFY.md` signed
- [ ] Experiment archive
- [ ] `verification/{tests.txt, smoke.txt, gates.txt, reproduction.txt}`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md`

## 11. Status journal

- 2026-05-27 -- Created from template by Doc-Agent-A.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G1 (elite-research-scientist critic). Critiquing the IDEA, not the implementation (that audit lives at `audits/G1_audit.md`). **Hardest scrutiny applied — this is the CIFAR-10/100 cross-dataset survivor.***

### Prior plausibility (independent of nature-inspired framing)
**LOW for the φ-mechanism, but the *positive empirical result* must be taken seriously and re-explained.** This is the only hypothesis with a verified cross-dataset positive in the campaign. That fact is a *constraint* on the critique, not a license to accept the φ-mechanism — it forces me to find a *better* explanation than "phi works."

### Mechanism scrutiny — does the claimed mechanism predict the effect?
The "because" clause: *"each stage's capacity to be the sum of the two prior stages."* This is *not* what the proposed allocation does. The allocation is **geometric** (stage k gets B · φ^k / Σφ^k), not Fibonacci-additive. The closed form makes stage_3 = stage_2 · φ, not stage_3 = stage_2 + stage_1. The doc conflates the Fibonacci *limit ratio* with the Fibonacci *recurrence*; only the ratio is implemented. The "biological resource inheritance" prose is therefore unsupported by the actual implementation.

### Confounds — what else could explain a positive (or negative) result?
**This is the critical section.** Here are five alternative explanations the doc does not address:

1. **Stage-3 shrinkage as implicit regularisation.** Per user's special instruction: 1:φ:φ² ≈ 1:1.6:2.6 vs canonical ResNet 1:2:4 doubling. The φ-ratio gives a **smaller stage-3 channel count** (2.6×base vs 4×base). At 12-30 epoch budgets, smaller stage-3 channels = less over-fitting capacity at the most-abstract stage = implicit regularisation. This is the **most likely true mechanism** and it has *nothing to do with φ* — any ratio in [1.5, 2.8] would likely produce the same regularising effect.

2. **Spatial-resolution × channel-count product.** At c0=64 with stride-2 between stages, stage-3 features are 4×4×128 (φ) vs 4×4×512 (doubling) — the φ variant has 4× fewer parameters at the bottleneck, which is exactly the lottery-ticket prediction (Frankle & Carbin 2019, arXiv:1803.03635) for over-parameterised CIFAR networks: prune the late stages.

3. **FLOPs-per-stage redistribution.** Late-stage stride-2 means each stage costs FLOPs ∝ c_k² × (H/2^k)². The doubling rule (c_k=2^k·c0) keeps stage FLOPs constant; the φ rule (c_k=φ^k·c0) gives *decreasing* stage FLOPs (φ²/4 < 1), shifting compute earlier. *Early-compute-heavy* networks are known to converge faster at small epoch budgets (Hoffmann et al 2022 *Chinchilla* arXiv:2203.15556 makes the analogous argument for LLMs).

4. **Mod-8 quantisation alignment**: at c0=64, the schedule [64, 104, 168, 272] happens to align with mod-8 better than [64, 128, 256, 512] in fewer cases — irrelevant on tensor cores but possibly different on the 4090's specific kernel selection heuristic.

5. **Composite-formula weight on params.** The project's composite penalises params; reducing total params by ~15% from doubling to φ-budget gives a "free" composite boost independent of top-1.

### Numerology check — does φ specifically matter?
**This is the key kill-or-confirm.** Per user's instruction:

**Three-way ablation, CIFAR-100, 30 epochs, 3 seeds:**
- **A**: 1:1.6:2.6 (≈ 1:φ:φ²) at iso-total-params
- **B**: 1:1.6:4 (same stage-1/2 as A, doubled stage-3) — *isolates whether stage-3 shrinkage is the win*
- **C**: 1:2:4 (canonical doubling) at iso-total-params

If A and C are within 0.3pp but B regresses by ≥0.5pp, the win is *stage-3 shrinkage* (mechanism #1 above). If A beats both B and C by ≥0.5pp, then the *whole schedule shape* matters but still not φ specifically. If A ≈ B ≈ C, the entire H09 result is noise.

**Additional control**: replace ratio φ (1.618) with 1.5, 1.7, 1.8, 1.9 — if any ratio in [1.5, 1.9] matches φ within 0.3pp, φ-specificity dies.

### Literature: precedent or rediscovery?
**Direct precedent**: RegNet (Radosavovic et al 2020, arXiv:2003.13678) explicitly parameterises *per-stage width progression* w_k = c_0 · u·k + c_0 (linear) and *quantised* variants. Their NAS search found optimal "width slope" parameters that imply per-stage ratios of ~1.5-2.5 depending on FLOPs budget — *exactly* the range where 1:φ:φ² sits. **H09 is functionally a single point in RegNet's design space, located in its empirically-discovered Pareto-optimal region.** Lin et al 2017 *Focal Loss / RetinaNet* implicitly use similar non-doubling channel schedules. Compound scaling (Tan & Le 2019) is the parent.

### Expected effect size — skeptical a-priori re-prediction
The CIFAR-10/100 win is real. **My re-prediction with the correct controls**: the win is *not* φ-specific. Any per-stage ratio in [1.5, 1.9] will produce the same composite gain to within ±0.3pp. At strictly iso-(total-params, total-FLOPs, depth), Δ(top-1) vs the best non-φ schedule ∈ [−0.2, +0.2] pp (90% CI).

### Minimum-distinguishing experiment
**The four-way ablation above**: {1:1.5:2.25, 1:φ:φ², 1:1.7:2.89, 1:1.9:3.61, 1:2:4}, all at iso-total-params, CIFAR-100, 30 epochs, 3 seeds. Cost: 5 configs × 3 seeds × ~10 min = ~150 min. **If the φ row does not Pareto-dominate the {1.5, 1.7, 1.9} rows by ≥0.5pp top-1 or ≥0.005 composite, the H09 verdict moves from NOVEL to NUMEROLOGY-with-real-mechanism-elsewhere.**

### Verdict
**DERIVATIVE+TESTABLE** (provisionally) — The empirical result is real, but it is *almost certainly* explained by stage-3 shrinkage acting as implicit regularisation at small epoch budgets (mechanism #1) and FLOPs-redistribution to early stages (mechanism #3), not by φ. The hypothesis as currently stated should be **re-framed as "test whether per-stage ratio in [1.5, 2.0] generally beats doubling at small epoch budgets"** — for which there is already RegNet precedent. The user's recommended kill-or-confirm ablation (1:1.6:2.6 vs 1:1.6:4 vs 1:2:4) is exactly the right test; until that ablation is run, the φ-specificity claim is unsupported by data that the doc *claims* supports it.
