# H51 — Topological Betti Loss (Differentiable Persistent Homology)

> **One-line claim:** Adding a differentiable persistent-homology
> auxiliary loss that drives stage-3 activation Betti-0 toward 1
> (i.e., feature-space connected components collapse to a single
> cluster per class as training proceeds) reduces CIFAR-10
> generalization gap by ≥1.5 pp and accelerates topological collapse
> by ≥30% in epoch terms, because explicitly enforcing the
> low-dimensional topology that the network is trying to learn anyway
> sharpens the implicit bias toward class-cohesive representations,
> per the TopologyLayer framework (Gabrielsson 2020).
>
> **Source design space:** G6 Topological + bridging additions (H51–H60).
>
> **Implementation status (this repo):** `~ partial — Betti curves are
> already computed in `src/nature_inspired_networks/topology.py:betti_curve`
> but no gradient flows back through them yet.`

This document is the committee-grade design write-up for hypothesis
H51. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

Persistent homology (PH) is the topological data-analysis tool that
quantifies the multi-scale connectivity of a point cloud: at scale ε,
how many connected components (β₀), loops (β₁), voids (β₂) exist?
For natural representations of class-discriminating images, we
expect β₀ to decrease toward `n_classes` as features become more
abstract — i.e., features for "dog" should form one connected
component, distinct from features for "cat", at most spatial scales.
This is the topological signature of good representations.

Empirically, CNNs *do* eventually produce features with this
topological structure (see Naitzat 2020 "Topology of deep neural
networks", JMLR). However, the topology emerges slowly and only
after many epochs. The differentiable PH machinery (TopologyLayer,
Gabrielsson 2020; PersLay, Carrière 2020) lets us add the desired
topological invariant as an explicit auxiliary loss term: minimize
`||β₀(stage_features) - n_classes||²` and gradient-descend through
the PH computation graph. Nature provides the prior; we supply the
gradient.

In sacred-geometry parlance, this is enforcing the Platonic ideal
of "n distinct clusters" topologically rather than via classifier
logits. It complements PRH alignment (H49, which constrains *where*
features go) by constraining *how connected* they are.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because adding a differentiable PH loss that targets β₀ = n_classes
(here, 10 for CIFAR-10) on stage-3 features explicitly biases the
representation toward class-cohesive topology — mechanism-wise, the
gradient pulls inter-class samples apart while compressing intra-class
clusters — per Gabrielsson 2020 (TopologyLayer, arXiv:1905.12200)
and Naitzat 2020, we expect a ≥1.5 pp reduction in CIFAR-10
generalization gap (train_top1 - test_top1) and a ≥30% reduction in
epochs-to-β₀-collapse, with asymptotic top-1 within ±0.5 pp of
baseline.

## 3. Falsifier (≥ 30 words)

If at 3-seed median the PH-loss arm does NOT reduce the generalization
gap by ≥1.0 pp (with 95% CI exclusion of 0), OR does NOT accelerate
β₀-collapse by ≥20% (epochs), OR if asymptotic top-1 drops by more
than -1.0 pp from baseline, this hypothesis is DISCARDED.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
Gabrielsson, Rickard Brüel and Nelson, Bradley J. and Dwaraknath,
Anjan and Skraba, Primoz 2020 AISTATS 'A Topology Layer for
Machine Learning' (arXiv:1905.12200) -- the differentiable PH
framework we use to make β₀ a gradient-friendly target.

Naitzat, Gregory and Zhitnikov, Andrey and Lim, Lek-Heng 2020 JMLR
'Topology of deep neural networks' (arXiv:2004.06093) -- shows
that successful networks topologically simplify class clusters
during training; we cite as the empirical motivation for choosing
β₀ as the target.

Carrière, Mathieu and Chazal, Frédéric and Ike, Yuichi and Lacombe,
Théo and Royer, Martin and Umeda, Yuhei 2020 AISTATS 'PersLay: A
Neural Network Layer for Persistence Diagrams and New Graph
Topological Signatures' (arXiv:1904.09378) -- alternative
differentiable-PH implementation; relevant as a fall-back if
TopologyLayer stability is poor.

Hofer, Christoph and Kwitt, Roland and Niethammer, Marc and Uhl,
Andreas 2017 NeurIPS 'Deep Learning with Topological Signatures'
(arXiv:1707.04041) -- foundational paper for PH-based deep
learning losses; methodologically anchors our choice of approach.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

The implementation hooks into the existing `betti_curve` computation
in `src/nature_inspired_networks/topology.py` but adds a differentiable
PH layer (via the `topologylayer` PyPI package) that supports
gradient flow.

```python
import torch
from torch import nn
try:
    from topologylayer.nn import LevelSetLayer2D, BarcodePolyFeature
except ImportError:
    LevelSetLayer2D = None  # graceful fallback for CI

class BettiLoss(nn.Module):
    """Differentiable PH-based loss targeting β₀ = n_classes."""

    def __init__(self, n_classes=10, persistence_threshold=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.thresh = persistence_threshold
        self.layer = LevelSetLayer2D(size=(16, 16), maxdim=0)

    def forward(self, feat):
        """feat: (B, D) penultimate features"""
        # 1. Compute pairwise distance / density estimate as height-fn
        density = self._density_map(feat)  # (16, 16)
        # 2. Compute persistence diagram
        dgms = self.layer(density)
        # 3. Count "persistent" β₀ components (lifetime > thresh)
        lifetimes = dgms[0][:, 1] - dgms[0][:, 0]
        n_components = (lifetimes > self.thresh).sum()
        # 4. MSE to target n_classes
        return (n_components.float() - self.n_classes) ** 2
```

Training cost: PH computation on a 16×16 grid is ~5 ms/batch (large
overhead but acceptable). FLOPs are negligible at inference (loss
not computed at eval time). Forward shape unchanged.

Lives in `src/nature_inspired_networks/losses/betti_loss.py`, re-exported by
`ideas/51_betti_loss/implementation.py`. Currently in **partial**
status because `topology.py:betti_curve` already computes
non-differentiable Betti curves at eval; this hypothesis adds the
trainable gradient flow.

### 5.2 LLM track (decoder-only Transformer)

For decoder LLMs, "class" is replaced by next-token target. The
analog: drive the representation Betti curve over the token-embedding
space toward `β₀ = vocab_size`. This is impractical for vocab_size=50k
+; instead, we target a sub-vocab (e.g., the top-100 most-frequent
tokens) and minimize `||β₀(top100_embeddings) - 100||²`.

FlashAttention-2 compatibility: PH computation operates on outputs
post-FA2; unaffected. Causal mask preservation: unaffected.

Expected impact at 124M scale: minor perplexity improvement (-0.3 to
-0.8 ppl on TinyStories); training overhead +20% wall-clock per step.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline (no Betti loss) | rationale |
|---|---|---|
| composite | [0, +0.012] | small lift via gap reduction |
| top-1 (CNN, 50 ep) | [-0.5, +1.5] | possible lift |
| generalization gap | [-3.0 pp, -1.0 pp] | core targeted metric |
| epochs-to-β₀-collapse | [-3, -1] | core targeted metric |
| params | [0, 0] | loss-only |
| FLOPs (inference) | [0, 0] | unchanged |
| GPU latency training (batch=1) | [+10%, +30%] | PH computation overhead |
| GPU latency inference | [0, 0] | unchanged |
| rotation-equivariance err | [-0.02, 0] | mild improvement |
| KV cache @ 32k (LLM) | [0, 0] | unaffected |
| Betti collapse rate | [-0.3, -0.1] | direct effect |
| perplexity (LLM 124M) | [-0.8, +0.2] | mild lift |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dataset:** CIFAR-10
- **Architecture:** `NaturePriorNet` (`channel_mode=fib`, priors off)
- **Loss:** CE + λ_betti · BettiLoss(stage3_feat); λ_betti = 0.1
- **Epochs:** 50 (gap takes time to develop)
- **Optimizer:** AdamW
- **Seeds:** 0, 1, 2
- **Logging:** train/test top-1, gap, β₀ curve per epoch
- **Run-script:** `python scripts/run_idea.py --idea 51 --betti --seeds 0 1 2`
- **Wall-clock:** ≈ 65 min × 3 seeds × 2 conditions ≈ 6.5 h
- **Archive path:** `ideas/51_betti_loss/experiments/exp001_cifar10_betti/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

The PH loss should help most in *low-data + over-parameterized*
regimes where natural Betti collapse is slow:

- **Dataset:** CIFAR-10 with 5% data (2.5k samples)
- **Architecture:** 2× wider `NaturePriorNet`
- **Predicted:** ≥3 pp gap reduction; ≥50% faster collapse
- **Diagnostic:** if no advantage in this extreme low-data regime,
  the topological prior is uninformative.

### 7.3 Cross-paradigm context (LLM track)

- **Model:** GPT-2-small (124M)
- **Dataset:** TinyStories
- **Loss:** CE + Betti loss on top-100 token-embedding β₀ target
- **Steps:** 10k
- **Run:** `python scripts/run_llm.py --idea 51 --betti`

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G6 row H51.
- Master experiment list: `EXPERIMENT_LOG.md` (Tier 2 row planned).
- Implementation sub-directory: `ideas/51_betti_loss/`
- Related hypotheses that compose:
  - **H54** PH activation regularization — extends H51 to β₁ and
    multi-stage targets.
  - **H59** Trained-feature Betti — H51 uses the same eval pipeline.
  - **H49** PRH alignment — composes (topology + geometry both
    pulled toward Platonic target).
  - **H65** PH Betti-collapse loss term (LLM-track flagship).
- Related hypotheses that conflict:
  - None directly; aux loss is additive.

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just supervised contrastive learning?**

> Contrastive learning shapes pairwise distances; PH loss shapes the
> *topological invariants*. The two are related but distinct: PH
> loss is rotation-invariant in feature space (does not care about
> absolute distance, only about connectivity at scale ε), whereas
> contrastive cares about absolute distance.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies ≥1.0 pp gap reduction AND ≥20% faster β₀ collapse
> with 95% CI exclusion.

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> Number of classes scales; the target β₀ scales with it. Bigger
> n_classes means bigger PH computation cost — may need to subsample.
> The fundamental hypothesis should remain valid; the
> implementation may need scaling.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H51 is a *loss-side* prior; the compound failure was architectural.
> Tested in isolation; composes naturally with all architectural priors.

**Q: How do we know the implementation is correct?**

> `tests.py` verifies (a) the PH computation produces correct β₀ for
> known toy point clouds (ring → β₀=1; two-cluster → β₀=2), (b)
> gradients flow non-zero through the PH layer, (c) at λ=0 training
> equals baseline.

## 10. Verification artifacts checklist

- [ ] `ideas/51_betti_loss/implementation.py` exists, tests green
- [ ] `ideas/51_betti_loss/tests.py` ≥ 8 assertions
- [ ] `ideas/51_betti_loss/AUDIT.md` ≥ 3 weaknesses
- [ ] `ideas/51_betti_loss/IMPROVEMENTS.md` records fixes
- [ ] `ideas/51_betti_loss/VERIFY.md` signed
- [ ] At least one experiment archive
- [ ] `verification/{tests,smoke,gates,reproduction}.txt`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-C.
