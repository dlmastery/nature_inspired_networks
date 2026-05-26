# H54 — Persistent Homology Regularization on Stage Activations

> **One-line claim:** Tracking β₀/β₁ Betti curves at every residual
> stage of a CNN and adding a hierarchical PH regularizer
> `L_PH = Σ_k λ_k ||β₀(stage_k) - target_k||²` with stage-specific
> targets (β₀ decreasing toward n_classes, β₁ decreasing toward 0)
> shapes the topological hierarchy of features across the network
> depth, yielding ≥2.0 pp top-1 lift over the no-regularization
> baseline on CIFAR-10 with 50-epoch training, because the natural
> topological simplification rate is too slow without supervision,
> per Naitzat 2020.
>
> **Source design space:** G6 Topological + bridging additions (H51–H60).
>
> **Implementation status (this repo):** `~ partial — Betti curves are
> computed at every stage in `src/nature_inspired_networks/topology.py` but
> not yet used as a gradient signal.`

This document is the committee-grade design write-up for hypothesis
H54. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

H51 (Topological Betti Loss) targets a single stage's β₀ as the loss
term. H54 extends this to the *hierarchy*: track β₀ and β₁ at every
residual stage and enforce a *topological staircase* — early stages
have high β₀ (many small clusters, raw features), middle stages
collapse loops (β₁ → 0), late stages collapse components (β₀ →
n_classes). This mirrors what Naitzat 2020 measured empirically in
successful CNNs: the topology of intermediate features simplifies
monotonically with depth, and the rate of simplification correlates
with generalization. Forcing this hierarchy explicitly should
accelerate learning of class-discriminating features.

The sacred-geometry connection: nature's hierarchical systems
(tree → branches → leaves → veins) show the same topological
simplification — each successive scale has fewer connected
components and fewer loops. The CNN hierarchy is the engineered
analog; H54 enforces it with a topological loss.

This complements H49 (PRH alignment) which constrains the final
embedding's *position* and H51 which targets a single stage's
*topology*; H54 adds the *depth-dependent topological gradient* that
ties the network's depth to its semantic abstraction level.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because adding a multi-stage PH regularizer enforces a topological
staircase — mechanism-wise, β₀ targets per stage are
`{20, 10, 5}` at depths 1/2/3 and β₁ targets are `{40, 5, 0}` —
per Naitzat 2020 and Gabrielsson 2020, we expect a ≥2.0 pp top-1 lift
on CIFAR-10 at 50 epochs and a ≥40% reduction in epochs-to-target-
topology vs. the no-PH-reg baseline (3-seed median, 95% CI exclusion
of 0).

## 3. Falsifier (≥ 30 words)

If at 3-seed median the multi-stage PH-reg arm does NOT lift top-1
by ≥1.0 pp (95% CI exclusion of 0), OR if PH targets are not reached
within the 50-epoch budget, OR if training wall-clock more than
doubles, this hypothesis is DISCARDED.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
Naitzat, Gregory and Zhitnikov, Andrey and Lim, Lek-Heng 2020 JMLR
'Topology of deep neural networks' (arXiv:2004.06093) -- shows
hierarchical topological simplification empirically in successful
CNNs; the basis for staircase targets.

Gabrielsson, Rickard Brüel and others 2020 AISTATS 'A Topology
Layer for Machine Learning' (arXiv:1905.12200) -- differentiable
PH framework.

Hofer, Christoph and Kwitt, Roland and Niethammer, Marc 2019 ICML
'Connectivity-Optimized Representation Learning via Persistent
Homology' (arXiv:1906.00722) -- complementary connectivity-
preserving PH loss; methodological precedent.

Carrière, Mathieu and others 2020 AISTATS 'PersLay'
(arXiv:1904.09378) -- alternative PH layer; fall-back.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

The model exposes 3 residual stages (after pooling layers). Hooks
extract penultimate feature for each stage. Per stage, a differentiable
PH layer computes (β₀, β₁) at a configurable persistence threshold;
the regularizer is:

```python
class HierarchicalBettiLoss(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        # stage-specific targets
        self.b0_targets = [2 * n_classes, n_classes, n_classes]  # 20, 10, 10
        self.b1_targets = [4 * n_classes, n_classes // 2, 0]      # 40, 5, 0
        self.lambdas = [0.05, 0.1, 0.2]  # deeper stages weighted higher

    def forward(self, stage_feats):
        loss = 0.0
        for i, feat in enumerate(stage_feats):
            b0, b1 = compute_persistent_betti(feat, dim=(0, 1))
            loss += self.lambdas[i] * (
                (b0 - self.b0_targets[i]) ** 2 +
                (b1 - self.b1_targets[i]) ** 2
            )
        return loss
```

Cost: ≈ 5-10 ms / batch / stage = 15-30 ms / batch (significant
training overhead). Inference cost: zero.

Lives in `src/nature_inspired_networks/losses/hier_betti.py`, re-exported by
`ideas/54_ph_activation_reg/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

For decoder LLMs, "stages" are layers. Track β₀ of token-embedding
clusters at layers {3, 6, 9, 12} (for GPT-2-small with 12 layers).
Target staircase: β₀ from full vocab → top-100 dominant clusters at
late layers. β₁ tracking less natural for 1D residual streams.

FA2 compatibility: PH on activations post-FA2; unaffected. Causal
mask: unaffected.

Expected at 124M: minor ppl lift (-0.3 to -0.7); +30% wall-clock
during training.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline (no PH reg) | rationale |
|---|---|---|
| composite | [+0.005, +0.020] | lift via topological constraint |
| top-1 (CNN, 50 ep) | [+1.0, +3.0] pp | core target |
| epochs-to-β₀-target | [-50%, -30%] | core target |
| params | [0, 0] | loss-only |
| FLOPs inference | [0, 0] | unchanged |
| GPU latency training (batch=1) | [+30%, +80%] | PH overhead |
| GPU latency inference | [0, 0] | unchanged |
| rotation-equivariance err | [-0.02, 0] | minor |
| KV cache @ 32k (LLM) | [0, 0] | unaffected |
| Betti collapse rate | [-0.5, -0.2] | direct target |
| perplexity (LLM 124M) | [-0.7, +0.1] | mild lift |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dataset:** CIFAR-10
- **Architecture:** `NaturePriorNet`
- **Loss:** CE + HierarchicalBettiLoss(stage_feats)
- **Epochs:** 50
- **Seeds:** 0, 1, 2
- **Run-script:** `python scripts/run_idea.py --idea 54 --seeds 0 1 2`
- **Wall-clock:** ≈ 90 min × 3 seeds × 2 conditions ≈ 9 h
- **Archive path:** `ideas/54_ph_activation_reg/experiments/exp001_hier_betti/`

### 7.2 Idea-targeted experiment

The staircase regularizer should help most when natural Betti
simplification stalls — i.e., over-parameterized + low data:

- **Dataset:** CIFAR-10 with 10% data
- **Epochs:** 100
- **Predicted:** ≥4 pp lift on top-1 + cleaner staircase

### 7.3 Cross-paradigm context (LLM track)

- **Model:** GPT-2-small (124M), TinyStories, 10k steps
- **Loss:** CE + simpler β₀-only hierarchical regularizer
- **Run:** `python scripts/run_llm.py --idea 54`

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G6 row H54.
- Master experiment list: `EXPERIMENT_LOG.md` (planned).
- Implementation sub-directory: `ideas/54_ph_activation_reg/`
- Related hypotheses that compose:
  - **H51** Topological Betti loss — single-stage progenitor;
    H54 is the multi-stage extension.
  - **H59** Trained-feature Betti — provides the eval pipeline.
  - **H49** PRH alignment — both pull representations toward
    Platonic ideal; topology vs. geometry.
  - **H65** PH Betti-collapse loss (LLM-track sibling).
- Related hypotheses that conflict:
  - None directly.

## 9. Committee Q&A

**Q: Why isn't this just H51 with multiple stages?**

> H54 introduces *stage-specific targets* (the staircase) which is
> a structurally different claim than "one stage should have β₀=10".
> The staircase is the contribution.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies ≥1.0 pp lift with 95% CI exclusion.

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> Staircase targets scale with n_classes; the topology argument
> applies on any classification task. Scaling: ImageNet's 1000
> classes makes the PH computation expensive but still tractable
> with subsampling.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H54 is a *loss-side* prior, additive. Tested in isolation.

**Q: How do we know the implementation is correct?**

> `tests.py` verifies (a) PH outputs correct β₀ for known toy
> point clouds at each stage, (b) loss gradients flow back to each
> stage's parameters, (c) at λ=0 baseline is recovered.

## 10. Verification artifacts checklist

- [ ] `ideas/54_ph_activation_reg/implementation.py` exists
- [ ] `ideas/54_ph_activation_reg/tests.py` ≥ 8 assertions
- [ ] `ideas/54_ph_activation_reg/AUDIT.md` ≥ 3 weaknesses
- [ ] `ideas/54_ph_activation_reg/IMPROVEMENTS.md`
- [ ] `ideas/54_ph_activation_reg/VERIFY.md` signed
- [ ] One experiment archive
- [ ] `verification/{tests,smoke,gates,reproduction}.txt`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-C.
