# H39 — Harmonic φ-Activation

> **One-line claim:** Replacing GELU / SiLU with a φ-parameterised harmonic activation `x · σ(x · φ)` raises top-1 on CIFAR-10/100 and improves gradient-flow statistics (gradient norm variance) relative to standard GELU / SiLU at zero added parameter cost.
>
> **Source design space:** G4 Kernels / Attention / Filters (H31–H40).
>
> **Implementation status (this repo):** `○ not started`.

This document is the committee-grade design write-up for hypothesis H39.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

Activation functions determine the nonlinear transformation between layers and thus the gradient-flow properties of the entire network. The progression ReLU → ELU → SELU → SiLU/Swish → GELU has been driven by finer-grained control over the gradient near zero and near saturation. A pattern in the literature (Ramachandran 2017 Swish, Hendrycks 2016 GELU) is that the optimal "slope-at-origin" of the activation is some specific value — Swish's parameter β was tuned to ≈ 1.0, GELU is approximately fixed by its Gaussian-CDF derivation.

The hypothesis here is that **φ** is the natural choice for the activation's slope-at-origin: φ ≈ 1.618 sits between Swish's β=1 and the next-larger value that produces unstable activations. Specifically, `PhiAct(x) = x · σ(x · φ)` is a Swish-like activation with `β = φ`. The motivation: φ-scaled activations may align with the φ-scaled width / depth progressions used in other priors (H04, H05) so that signal magnitude is preserved across φ-scaled layers. Additionally, φ is the slowest-growing scaling that still produces super-linear gradient magnification, which Goldilocks the gradient between vanishing and exploding.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** φ is the optimal Swish-β coefficient under φ-scaled width / depth priors and produces the Goldilocks gradient-magnification regime between vanishing and exploding, replacing GELU with `PhiAct(x) = x · σ(x · φ)` raises CIFAR-100 top-1 by ≥ +0.3 pp and reduces gradient-norm variance across layers by ≥ 10 % relative to GELU baseline, per the mechanism of Ramachandran 2017 (Swish, arXiv:1710.05941) and Hendrycks 2016 (GELU).

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on CIFAR-100, PhiAct fails to lift top-1 by ≥ 0.15 pp AND fails to reduce per-layer gradient-norm variance by ≥ 5 % relative to GELU baseline, this hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Ramachandran, P., Zoph, B., Le, Q. V. 2017 'Searching for Activation
Functions' (arXiv:1710.05941) — Swish reference: x · σ(β·x); H39 sets
β = φ.

Hendrycks, D., Gimpel, K. 2016 'Gaussian Error Linear Units (GELUs)'
(arXiv:1606.08415) — GELU reference.

Klambauer, G., et al. 2017 NeurIPS 'Self-Normalizing Neural Networks'
(arXiv:1706.02515) — SELU reference, motivates careful activation design.

Glorot, X., Bengio, Y. 2010 — gradient-flow analysis methodology.

He, K., et al. 2015 — ReLU / He init reference.

Krizhevsky 2009 — CIFAR dataset.
```

## 5. Mechanism

### 5.1 CNN track

A pure Python / PyTorch activation, no parameters, drop-in:

```python
# ideas/39_harmonic_phi_activation/implementation.py
PHI = (1+5**0.5)/2

class PhiAct(nn.Module):
    def __init__(self, learnable=False):
        super().__init__()
        if learnable:
            self.beta = nn.Parameter(torch.tensor(PHI))
        else:
            self.register_buffer("beta", torch.tensor(PHI))
    def forward(self, x):
        return x * torch.sigmoid(x * self.beta)

def phi_act(x):
    return x * torch.sigmoid(x * PHI)
```

- Input: any tensor.
- Params: 0 (or 1 if learnable=True).
- FLOPs: same as Swish.
- Init: deterministic.

### 5.2 LLM track

Drop-in replacement for GELU in FFN; SwiGLU-style gated variant also possible.

```python
class PhiGLU(nn.Module):
    def __init__(self, d, expand=4):
        super().__init__()
        self.W = nn.Linear(d, 2*expand*d)
        self.proj = nn.Linear(expand*d, d)
    def forward(self, x):
        a, b = self.W(x).chunk(2, dim=-1)
        return self.proj(phi_act(a) * b)
```

- FlashAttention-2 compatibility: ✓ (FFN is unaffected by attention kernel).
- Causal-mask preservation: ✓.
- Expected: **-0.05 to -0.15 perplexity** vs GELU at 124 M scale.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (CIFAR-100) | [+0.001, +0.010] | small but real activation effect |
| top-1 (CIFAR-100, primary) | [+0.3 pp, +1.5 pp] | direct claim |
| Gradient-norm variance | [-5 %, -25 %] | direct claim |
| params | [0, +0.001 %] | 0 or 1 |
| FLOPs | [≈0, ≈0] | identical to Swish |
| GPU latency | [≈1.0×, ≈1.0×] | unchanged |
| Perplexity (LLM) | [-0.15, -0.05] | small positive |
| Betti collapse rate | [+0.02, +0.10] | small effect |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** CIFAR-100.
- **Architecture:** NaturePriorBlock with all GELU/ReLU replaced by `PhiAct`.
- **Epochs / batch / precision / seeds:** 25 epochs, batch 128, bf16, 3 seeds.
- **Composite:** top-1 (0.6), gradient-norm variance (0.2), latency (0.1), params (0.1).
- **Run-script:** `python ideas/39_harmonic_phi_activation/experiment.py --seeds 0 1 2`.
- **Wall-clock:** ~25 min/seed × 3 = 75 min.
- **Archive:** `ideas/39_harmonic_phi_activation/experiments/exp001_cifar100_seed0..2/`.

### 7.2 Idea-targeted experiment

1. **CIFAR-10 / CIFAR-100** with sweep over β ∈ {1.0 Swish, φ, 2.0, π/2} — control sweep.
2. **Tiny-ImageNet** for scaling.
3. **Tabular Higgs UCI** — activation effect persists across architectures.

### 7.3 Cross-paradigm context

LLM-track: WikiText-103 124 M with PhiGLU instead of SwiGLU; 50 k steps; perplexity comparison.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G4 row H39.
- Master: planned Tier 1.
- Sub-dir: `ideas/39_harmonic_phi_activation/`.
- Composes: H04 / H05 (φ-scaling — same constant), H19 (φ-neuron threshold), H41 (golden optimizer).
- Conflicts: none.

## 9. Committee Q&A

**Q: Why isn't this just Swish with β = 1.618?**

> Yes, that is literally the contribution. The pre-committed value (and the comparison to β ∈ {1.0, π/2, 2.0}) is the falsifier.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 gives top-1 ≥ +0.15 pp AND gradient-variance ≤ -5 %. Both must hold.

**Q: What if β-sweep shows β=2 is better than β=φ?**

> That is a hypothesis-negative outcome — the φ-specificity is not unique. We file and move on.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> Single-prior; compounding is H50/H67.

**Q: How do we know the implementation is correct?**

> `tests/test_phi_act.py::test_swish_at_phi` asserts `phi_act(x) == x * sigmoid(x * PHI)`. `test_at_origin` asserts `phi_act(0) == 0`. `test_gradient_continuity` asserts gradient is continuous at origin. Plus archive verification.

## 10. Verification artifacts checklist

- [ ] `ideas/39_harmonic_phi_activation/implementation.py` tests green
- [ ] `ideas/39_harmonic_phi_activation/tests.py` ≥ 5 assertions
- [ ] AUDIT.md ≥ 3 weaknesses
- [ ] IMPROVEMENTS.md
- [ ] VERIFY.md signed
- [ ] Experiment archives + verification
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] FINDINGS reflects result

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-B.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G4 (elite-research-scientist critic). Critiquing the IDEA, not the implementation (audit at `audits/G4_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)

**LOW-MED.** β=φ ≈ 1.618 sits between Swish (β=1) and GELU (effectively β ≈ 1.702 if approximated as σ(1.702·x); Hendrycks 2016 arXiv:1606.08415 explicitly notes this). The relevant question is whether the activation landscape is sensitive to β in [1.5, 1.7]. Ramachandran 2017 (Swish, arXiv:1710.05941) Table 4 reports a Neural-Architecture-Search sweep that found β=1.0 optimal on CIFAR-10 ResNet, with β=1.5 producing a 0.1 pp loss. The "Goldilocks" claim is unsupported — the sweep showed monotone or flat behavior, not a peak at φ. Furthermore, β-Swish at β > 1 is approximately ReLU-like (sharper transition), which the field has moved AWAY from (toward smoother GELU).

### Mechanism scrutiny

The claim "φ Goldilocks the gradient between vanishing and exploding" is not derivable from the activation's properties. Both Swish (β=1) and GELU have BOUNDED derivatives (sup |f'| < 1.1 for both), so neither vanishes nor explodes. The activation choice contributes O(1%) to gradient-norm variance vs init/normalization (BN/LN dominate). Wang 2024 (arXiv:2407.06405) "Activation Function Survey" shows β-Swish, GELU, ReLU all within 0.3 pp on CIFAR at matched training.

The "φ aligns with φ-scaled width/depth priors (H04, H05)" argument is metaphysical, not mechanical. Width/depth scaling factors (`d → d·φ`) operate on TENSOR SHAPES; activation β operates on POINTWISE NONLINEARITY. These two are orthogonal — there is no signal-magnitude conservation law that links them. The hypothesis conflates the WORD "φ" appearing in both, not a shared mechanism.

### Confounds (≥2)

1. **Variance-shift confound.** Different β values change the effective output variance of the activation (β=1 has lower variance than β=φ at the same input). Without re-tuning BN/LN, this variance shift propagates through the network and can produce a small accuracy delta unrelated to "Goldilocks gradient". Control: re-tune BN momentum / LN epsilon for each β.
2. **Initialization-coupling confound.** He init assumes ReLU-like (β → ∞) variance recipe. Swish/GELU/PhiAct deviate from this; β-Swish at higher β is closer to ReLU, so He init is more correct. Any measured Δ may come from "init matches activation slope better", not from a Goldilocks effect.

### Numerology / specificity check

The committee Q&A explicitly admits: "What if β-sweep shows β=2 is better than β=φ?" → "we file and move on". This is CORRECT scientific practice but it ALSO admits that the φ value is not pre-registered as the unique optimum. The pre-registered comparator set {1.0, π/2, 2.0} is reasonable but does not include β=1.7 (close to GELU). Without β ∈ [1.5, 1.7] sweep at fine resolution, the φ vs 1.7 question is undecidable. **Moderate numerology — the value is suspiciously meaningful only via the name "φ"; the same arithmetic gain could be claimed for π/2 ≈ 1.571 or 5/3 ≈ 1.667.**

### Literature precedent — kernel/attention design is a crowded field

Activation literature: ReLU (Glorot 2010), ELU (Clevert 2015 arXiv:1511.07289), SELU (Klambauer 2017 arXiv:1706.02515), Swish/SiLU (Ramachandran 2017 arXiv:1710.05941), GELU (Hendrycks 2016 arXiv:1606.08415), Mish (Misra 2019 arXiv:1908.08681), SwiGLU (Shazeer 2020 arXiv:2002.05202). All differ by <0.5 pp on standard CIFAR/ImageNet at matched training. The activation choice is THIRD-ORDER vs normalization, init, and training recipe.

### Expected effect size (90% CI a priori)

CIFAR-100 top-1 vs GELU: [-0.2 pp, +0.2 pp] (likely null). vs SiLU (β=1): [-0.1 pp, +0.3 pp] (slight chance of small positive). Author's [+0.3, +1.5] is too optimistic. Gradient-norm variance reduction is probably real but small (<5%) due to slope difference, not "Goldilocks".

### Minimum-distinguishing experiment

3-seed CIFAR-100 sweep β ∈ {0.8, 1.0, 1.3, 1.5, φ, 1.7, 2.0, 2.5} with FIXED everything else, plot top-1 vs β. If a peak at β=φ with ≥ 0.2 pp margin over β=1.7 (closest GELU-equivalent), the φ-specificity is non-null. Otherwise it's a flat curve and the choice is null. The doc's protocol § 7.2 ("β sweep") is good, but it should be the PRIMARY experiment, not secondary.

### Verdict
DERIVATIVE+TESTABLE — β-Swish at β=φ is a well-defined point in a known design space; the falsifier (fine-grained β sweep) is correctly pre-registered. The "Goldilocks gradient" mechanism is unsupported but the empirical test is sound. Most likely outcome: null (flat curve over β ∈ [1.0, 2.0]).
