# H49 — Platonic Representation Alignment Loss (PRH Auxiliary)

> **One-line claim:** Adding an auxiliary CKA-based alignment loss
> pulling the penultimate-layer features of a CIFAR-10 model toward a
> fixed Platonic embedding (12-vertex dodecahedron projected through
> a CLIP-pretrained image encoder) accelerates convergence to top-1 ≥
> 82% by ≥20% (epochs) because the Platonic-Representation-Hypothesis
> (Huh, Cheung, Wang, Isola 2024) predicts a universal target
> embedding toward which large networks converge, and explicitly
> aligning to it short-circuits the implicit alignment process.
>
> **Source design space:** G5 Optimization / Init / Regularization / NAS
> (H41–H50).
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for hypothesis
H49. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The Platonic Representation Hypothesis (PRH, Huh et al. 2024,
arXiv:2405.07987) provides empirical evidence that across modalities
(vision, language) and objectives (supervised, self-supervised), the
representations learned by sufficiently large neural networks
converge to a shared statistical model of reality. They measure this
via mutual nearest-neighbor agreement between text and image models —
representations align as scale grows. The conjecture is that there
exists a "Platonic" canonical embedding to which all sufficiently
capable networks converge.

The 12-vertex dodecahedron is the largest fully symmetric Platonic
solid embeddable in 3D space (icosahedron's dual). Its vertex
positions form a maximally well-spread set on the sphere — the
geometric analog of an information-theoretically optimal
representation. The connection to PRH is: if there is a universal
target, projecting onto dodecahedron vertices is the symmetry-
respecting, information-spread-maximizing parameterization of that
target in 3D. In higher dimensions, the analog is the 600-cell or
hyper-dodecahedron, but for didactic clarity we work with 3D
dodecahedron-vertex targets and embed them via a CLIP encoder.

The engineering hypothesis: rather than waiting for scale to
implicitly align representations toward the PRH target, we add an
auxiliary loss that *explicitly* pulls features toward a fixed
projection of a CLIP embedding onto the dodecahedron vertex set.
This is a Platonic-anchored regularizer.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because the PRH (Huh 2024) predicts a universal target embedding,
and aligning to that target via a CKA loss is mechanistically
equivalent to applying a structured prior — mechanism-wise, the loss
`L_align = 1 - CKA(features, target)` pulls the penultimate layer
toward a fixed Platonic-projected CLIP embedding — we expect a ≥20%
reduction in epochs-to-82%-top-1 on CIFAR-10 (3-seed median, 95% CI
exclusion of 0% improvement) at λ_align = 0.1, while asymptotic top-1
stays within ±1.0 pp of baseline.

## 3. Falsifier (≥ 30 words)

If at 3-seed median the PRH-aligned arm does NOT reach 82% top-1 at
least 20% faster (i.e., epoch count) than the no-alignment control
(95% CI upper bound must exceed 20% speedup), OR if asymptotic
12-epoch top-1 drops by more than -1.5 pp from the baseline, this
hypothesis is DISCARDED.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
Huh, Minyoung and Cheung, Brian and Wang, Tongzhou and Isola,
Phillip 2024 ICML 'The Platonic Representation Hypothesis'
(arXiv:2405.07987) -- the foundational PRH paper; cited as the
empirical justification for the existence of a universal target
embedding worth aligning to.

Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and
Hinton, Geoffrey 2019 ICML 'Similarity of Neural Network
Representations Revisited' (arXiv:1905.00414) -- introduces CKA
(centered kernel alignment); the metric we minimize to align
representations.

Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others 2021
ICML 'Learning Transferable Visual Models From Natural Language
Supervision' (arXiv:2103.00020) -- CLIP; the source of the
pretrained target embedding we project onto Platonic vertices.

Tian, Yonglong and Krishnan, Dilip and Isola, Phillip 2020 ECCV
'Contrastive Multiview Coding' (arXiv:1906.05849) -- foundational
work on multi-view representation alignment, the methodological
precursor of CKA-based alignment.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

Setup at training time:

1. **Pre-compute target embeddings.** For each CIFAR-10 image, run
   it through a frozen CLIP ViT-B/16 image encoder, project the
   512-D embedding onto the 20 nearest dodecahedron-vertex
   directions, and store as a `(N_train, 20)` target tensor.
2. **At each training step:** compute the model's penultimate-layer
   feature (size 168 for the standard `NaturePriorNet`). Project to
   20 dim via a small learnable linear head. Compute CKA between
   batch features and pre-computed targets:

```python
def cka(X, Y, eps=1e-6):
    """Centered Kernel Alignment between feature matrices X, Y."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XX = (X @ X.T)
    YY = (Y @ Y.T)
    XY = (X @ Y.T)
    return (XY.norm()**2 + eps) / (XX.norm() * YY.norm() + eps)

class PRHAlignmentLoss(nn.Module):
    def __init__(self, target, feat_dim=168, target_dim=20):
        super().__init__()
        self.register_buffer("target", target)
        self.proj = nn.Linear(feat_dim, target_dim, bias=False)

    def forward(self, feat, batch_idx):
        """feat: (B, feat_dim); batch_idx: indices into target."""
        z = self.proj(feat)
        t = self.target[batch_idx]
        return 1.0 - cka(z, t)
```

3. **Compose with classification loss.**
   `L = L_ce + λ_align · L_align(feat, idx)`.

Cost at training: one CLIP encoder pre-computation (~5 min one-time
on CIFAR-10), then per-batch CKA (negligible — O(B²) with B=128).
Cost at inference: **zero** — alignment loss is training-only.

Lives in `src/nature_inspired_networks/losses/prh_alignment.py`, re-exported by
`ideas/49_prh_alignment_loss/implementation.py`. The targets are
pre-cached at `data/prh_targets/cifar10_dodeca.pt`.

### 5.2 LLM track (decoder-only Transformer)

For a 124M GPT-2-small on TinyStories: the PRH target is the
text-embedding produced by a pretrained sentence-transformer
(MiniLM-L12), projected onto a higher-dimensional Platonic analog
(the 600-cell has 120 vertices in 4D; we use the first 120 PCA-
components of MiniLM as the target).

Per extended-transcript chunk-6: this auxiliary loss is also
discussed as the "Geometric JEPA" target — same auxiliary loss
applied during pretraining of an LLM produces topological alignment
to the universal embedding.

FlashAttention-2 compatibility: alignment loss operates on residual
stream post-FA2; unaffected. Causal mask preservation: unaffected.

Expected impact at 124M (TinyStories, 10k steps): -0.5 to -1.0 ppl
faster lift; asymptotic ppl within ±0.3 of baseline.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline (no PRH loss) | rationale |
|---|---|---|
| composite | [0, +0.015] | speed gain composite-positive |
| top-1 (CNN, 12 ep) | [-1.0, +1.5] pp | mild lift |
| epochs-to-82%-top-1 | [-3, -1] | core targeted metric |
| params | [+0.3%, +0.5%] | projection head adds tiny params |
| FLOPs | [+0.1%, +0.3%] | training-only addition |
| GPU latency (batch=1, inference) | [0, 0] | training-only |
| rotation-equivariance err | [-0.01, 0] | minor |
| KV cache @ 32k (LLM) | [0, 0] | unaffected |
| Betti collapse rate | [-0.1, 0] | accelerated topological alignment |
| perplexity (LLM 124M) | [-1.0, +0.2] | targeted lift via faster alignment |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dataset:** CIFAR-10
- **Pre-cache:** `python scripts/precache_clip_targets.py
  --dataset cifar10 --model clip-vit-b16 --project dodeca`
  (~5 min one-time on RTX 4090)
- **Architecture:** `NaturePriorNet` (`channel_mode=fib`, priors off)
- **Loss:** CE + λ_align · PRHAlignmentLoss
- **λ sweep:** {0.0, 0.05, 0.1, 0.3} to find optimum
- **Epochs:** 12, batch=128, bf16
- **Seeds:** 0, 1, 2 (for chosen λ)
- **Logging:** per-epoch top-1 trajectory, alignment-loss value
- **Run-script:** `python scripts/run_idea.py --idea 49 --lambda 0.1 --seeds 0 1 2`
- **Wall-clock:** ≈ 5 min precache + 12 min × 3 seeds × 4 conditions ≈ 150 min
- **Archive path:** `ideas/49_prh_alignment_loss/experiments/exp001_cifar10_prh/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

PRH alignment should help most where data scarcity slows implicit
convergence:

- **Dataset:** CIFAR-10 with 10% data
- **Epochs:** 50
- **Predicted:** ≥3 pp top-1 lift over no-alignment control (CLIP's
  pretraining provides "free" data via the target)
- **Diagnostic:** if no advantage on low-data regime, the PRH target
  is not providing useful priors and the hypothesis fails its
  natural-fit scenario.

### 7.3 Cross-paradigm context (LLM track)

- **Model:** 124M GPT-2-small
- **Dataset:** TinyStories
- **Target:** pre-cached MiniLM-L12 embeddings projected onto 120-D
  600-cell vertex set
- **Steps:** 10k
- **Run:** `python scripts/run_llm.py --idea 49 --lambda 0.1`
- **Expected:** ≥0.5 ppl faster lift at 5k steps

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G5 row H49.
- Master experiment list: `EXPERIMENT_LOG.md` (planned).
- Implementation sub-directory: `ideas/49_prh_alignment_loss/`
- Related hypotheses that compose:
  - **H25** Dodeca latent — H49 *enforces* the dodeca-latent
    structure via loss; H25 *parameterizes* it. They compose
    perfectly.
  - **H46** Cymatic loss — both auxiliary losses; jointly tested as
    a multi-target regularization strategy.
  - **H63** Platonic projection aux + cymatic teachers (LLM) —
    composition extends to LLM track.
  - **H67** Full paradigm hybrid — PRH alignment is the auxiliary
    loss component.
- Related hypotheses that conflict:
  - None directly; aux loss is additive.

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just CLIP knowledge distillation?**

> KD aligns logits or softmax probabilities; PRH alignment is in
> *feature space* and projects onto a *Platonic vertex set* (not
> arbitrary embeddings). The Platonic projection is the
> contribution.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies 20% epoch reduction at 95% CI exclusion. λ must be
> non-trivial (≥ 0.05).

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> CLIP was pretrained on a superset of ImageNet, so alignment
> targets are more reliable there. § 7.2 tests low-data regime;
> ImageNet should be a stronger positive, not weaker.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H49 is a *loss-side* prior; the compound failure was architectural.
> Tested in isolation; later composed with H46 cymatic loss to test
> dual-aux-loss interaction.

**Q: How do we know the implementation is correct?**

> `tests.py` verifies (a) CKA is symmetric and in [0,1], (b) target
> cache covers all training examples, (c) gradients flow through the
> projection head, (d) at λ=0 training equals baseline (bit-equiv).

## 10. Verification artifacts checklist

- [ ] `ideas/49_prh_alignment_loss/implementation.py` exists, tests green
- [ ] `ideas/49_prh_alignment_loss/tests.py` ≥ 8 assertions
- [ ] `ideas/49_prh_alignment_loss/AUDIT.md` ≥ 3 weaknesses
- [ ] `ideas/49_prh_alignment_loss/IMPROVEMENTS.md` records fixes
- [ ] `ideas/49_prh_alignment_loss/VERIFY.md` signed
- [ ] At least one experiment archive
- [ ] `verification/{tests,smoke,gates,reproduction}.txt`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-C.
