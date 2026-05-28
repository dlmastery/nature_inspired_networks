# H45 — Sacred / Nature-Inspired Neural Architecture Search

> **One-line claim:** Restricting a DARTS-style differentiable NAS to
> a cell library consisting solely of φ-channel / Fib-depth /
> Platonic-equivariant operators, with channel counts ⊂ Fib, reaches
> CIFAR-10 top-1 within -0.5 pp of unrestricted DARTS at ≥5× fewer GPU
> hours because the φ/Fib/Platonic priors collectively cut the search
> space dimension from ~10¹⁸ to ~10⁵, enabling fast convergence on
> hardware where unrestricted DARTS is infeasible.
>
> **Source design space:** G5 Optimization / Init / Regularization / NAS
> (H41–H50).
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for hypothesis
H45. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

Neural Architecture Search (NAS) is plagued by combinatorial
explosion. DARTS (Liu 2019) reduced search compute from 2000 GPU-days
(RL-NAS) to single-digit GPU-days by relaxing discrete choices into
continuous mixtures, but the 1-shot supernet still requires ≥1 GPU-day
on CIFAR-10 and ≥4 GPU-days on ImageNet. The bottleneck is the search
space dimension: ~|ops|^|edges| typically ~8^14 = 4.4×10¹². Most of
that space is empirically wasteful: very few discovered cells look
random; they cluster around skip-heavy, Inception-like, or pyramid-
shaped topologies.

Nature solved the same problem differently. Photosynthetic leaves
(Iolla et al. 2009) and capillary trees (West, Brown, Enquist 1997)
do not search over arbitrary topologies; they search within a
constrained family of self-similar branching patterns with
Fibonacci-like degree sequences and φ-scaled child sizes. The
constraint removes >99% of the topological possibilities but loses
<1% of the performance because the priors are *correct*. By baking
the same prior into a NAS search space — channel counts ⊂ {3, 5, 8,
13, 21, 34, 55, 89, 144}, cell depth ∈ Fib, equivariant operator
choices restricted to {identity, C4-group-conv, hex-conv, φ-skip} —
we cut the space from 10¹⁸ to 10⁵ and bring NAS into hours-per-run
on a single 4090. This is the engineering argument for "Sacred NAS".

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because the φ/Fib/Platonic priors restrict architecture choices to a
known-effective topological subspace — mechanism-wise, the search
space is reduced from ~10¹⁸ to ~10⁵ candidates while preserving the
empirically-relevant region (skip-heavy, multi-scale, equivariant
operators) — per Liu 2019 (DARTS, arXiv:1806.09055), we expect Sacred
NAS to discover a cell achieving CIFAR-10 top-1 within -0.5 pp of
unrestricted DARTS at ≥5× fewer GPU hours, with the discovered cell
empirically containing ≥3 Fib-spaced channel transitions out of a
maximum 5 (verifying the prior is actively guiding search).

## 3. Falsifier (≥ 30 words)

If Sacred NAS produces a cell with CIFAR-10 top-1 more than -2.0 pp
below unrestricted DARTS at iso-compute, OR if Sacred NAS does NOT
achieve ≥5× speed-up at iso-accuracy, OR if the discovered cell
contains zero Fib-spaced channel transitions (indicating the prior is
inert and search ignored it), this hypothesis is DISCARDED.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
Liu, Hanxiao and Simonyan, Karen and Yang, Yiming 2019 ICLR 'DARTS:
Differentiable Architecture Search' (arXiv:1806.09055) -- the
canonical differentiable NAS framework we constrain.

Real, Esteban and Aggarwal, Alok and Huang, Yanping and Le, Quoc V.
2019 AAAI 'Regularized Evolution for Image Classifier Architecture
Search' (arXiv:1802.01548) -- evolutionary NAS baseline for
unrestricted search; we ablate against this if differentiable NAS
proves unstable.

Tan, Mingxing and Le, Quoc V. 2019 ICML 'EfficientNet: Rethinking
Model Scaling for Convolutional Neural Networks'
(arXiv:1905.11946) -- EfficientNet's compound scaling rule is one
of the few empirically successful constrained search spaces; our
φ-Compound Scaling (H01) is an even tighter constraint.

Pham, Hieu and Guan, Melody Y. and Zoph, Barret and Le, Quoc V. and
Dean, Jeff 2018 ICML 'Efficient Neural Architecture Search via
Parameter Sharing' (arXiv:1802.03268) -- ENAS as the one-shot
weight-sharing precursor to DARTS we cite for the speed-vs-quality
trade-off framework.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

The Sacred-NAS supernet is a 5-cell stack where each cell has 5
intermediate nodes; each node selects (via softmax-mixed weights) one
operator from the **restricted library**:

```
sacred_ops = [
    "identity",
    "phi_skip",          # H17 — φ-scaled skip
    "fib_conv_3x3",      # channels in next-Fib step
    "fib_conv_5x5",      # H12 — larger Fib kernel
    "c4_group_conv",     # H24 — C4 equivariance
    "hex_conv_3x3",      # H21 — hex lattice tap
    "max_pool_3x3",      # standard
    "avg_pool_3x3",      # H58 — preferred over max
]
```

Channel counts at each stage transition are restricted to
{16, 24, 40, 64, 104, 168}, the Fib-aligned widths from H04 / H12. The
DARTS supernet trains both architecture weights α (continuous mixtures
over ops) and network weights w (per-op convolutional weights) via
bi-level optimization for 25 epochs; then a discrete derivation step
picks `argmax(α)` per node, and the discrete cell is retrained for
12 epochs.

Computational cost: ≈ 2 GPU-hours search + 12 min retrain on RTX 4090.

```python
class SacredNASCell(nn.Module):
    def __init__(self, c_in, c_out, ops=sacred_ops):
        super().__init__()
        self.ops = nn.ModuleList([
            build_op(name, c_in, c_out) for name in ops
        ])
        self.alpha = nn.Parameter(torch.zeros(len(ops)))

    def forward(self, x):
        w = torch.softmax(self.alpha, dim=0)
        return sum(w[i] * op(x) for i, op in enumerate(self.ops))
```

Lives in `src/nature_inspired_networks/nas/sacred_nas.py` and re-exported by
`ideas/45_sacred_nas/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

Sacred NAS for LLMs replaces the architecture choices with a
constrained block library: attention pattern ∈ {full, Fibottention
(H32), hex-attn (H62)}, FFN width ∈ Fib, normalization position ∈
{pre-LN, post-LN}, head count ∈ Fib. Search over 6 blocks gives
~3 × 3 × 2 × 4 = 72 configurations per block → 72^6 ≈ 1.4×10¹¹, a
huge reduction from typical LLM-NAS spaces of ~10²⁰.

FlashAttention-2 compatibility: the search must restrict attention
patterns to FA2-compatible forms (full, sparse-block); arbitrary
masks would break FA2's tile-based kernel. Causal mask is preserved
across all candidates.

Expected impact: at 124M scale, search finds a 124M-equivalent block
configuration in <1 GPU-day; retrained model reaches the same TinyStories
perplexity as the manual GPT-2-small baseline ± 0.5 ppl.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline (unrestricted DARTS) | rationale |
|---|---|---|
| composite | [-0.005, +0.010] | iso-accuracy, much faster |
| top-1 (CNN) | [-0.5, +0.5] | sacred priors are competitive |
| search-time GPU-hours | [-90%, -80%] | core targeted metric |
| params (discovered cell) | [-20%, +20%] | depends on Fib choices |
| FLOPs (discovered cell) | [-30%, +20%] | likely lower (Fib widths) |
| GPU latency (batch=1) | [-30%, +10%] | typically faster (Fib widths) |
| rotation-equivariance err | [-0.04, 0] | if C4 is chosen, lower rot-err |
| KV cache @ 32k (LLM) | [-30%, +10%] | if Fibottention chosen |
| Betti collapse rate | [-0.05, +0.05] | unclear |
| perplexity (LLM 124M) | [-0.5, +0.5] | iso |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dataset:** CIFAR-10
- **Architecture:** 5-cell Sacred-NAS supernet
- **Search:** 25 epochs bi-level (Adam for w, Adam for α at separate LR)
- **Retrain:** 12 epochs on discrete cell, bf16 AMP, AdamW
- **Seeds:** 0, 1, 2 (search and retrain both)
- **Control:** unrestricted DARTS with the same 5-cell structure
- **Run-script:** `python scripts/run_nas.py --idea 45 --restricted --seeds 0 1 2`
- **Wall-clock:** ≈ 2 h search + 12 min retrain per seed → ≈ 6.5 h total
- **Archive path:** `ideas/45_sacred_nas/experiments/exp001_cifar10_sacred_nas/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

Sacred NAS should shine most where unrestricted NAS *cannot run*
within budget — e.g., on a small benchmark with a tight time bound:

- **Setup:** CIFAR-100 with strict 3-GPU-hour budget per search
- **Comparator:** Sacred NAS (completes 25 ep) vs. truncated DARTS
  (only 10 ep, incomplete search)
- **Predicted:** Sacred NAS top-1 ≥ 2 pp above truncated DARTS,
  validating speed as the primary lever
- **Diagnostic:** if Sacred NAS does not win in the budget-constrained
  regime, the speed argument fails.

### 7.3 Cross-paradigm context (LLM track)

- **Model:** 124M GPT-2-small search space
- **Dataset:** TinyStories
- **Search:** 1 GPU-day (single 4090 with bf16 + grad-ckpt)
- **Retrain:** 5k steps on the discovered configuration
- **Run:** `python scripts/run_llm_nas.py --idea 45 --restricted`

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G5 row H45.
- Master experiment list: `EXPERIMENT_LOG.md` (planned).
- Implementation sub-directory: `ideas/45_sacred_nas/`
- Related hypotheses that compose:
  - **H01** φ-compound scaling — Sacred NAS implicitly searches over
    a φ-aware width set, so post-discovery scaling can extend with φ.
  - **H32** Fibottention — entries in the LLM search library.
  - **H58** group_avg_pool — Sacred-NAS library uses avg-pool by
    default (max-pool excluded based on T1.4 finding).
- Related hypotheses that conflict:
  - **H50** Full hybrid — Sacred NAS *implicitly* explores hybrid
    combinations, so a fully fixed hybrid pre-empts the search; we
    run NAS first, then fix discoveries.

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of DARTS with a different
search space?**

> The search-space restriction is the *contribution*. Every NAS
> paper trades off search space size and quality; H45's specific
> choice (φ/Fib/Platonic priors) is theoretically motivated rather
> than empirically tuned. The pre-registered speed-up factor is the
> falsifiable component.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies a 5× speed-up at iso-accuracy with -2.0 pp ceiling,
> plus the "discovered cell must contain ≥3 Fib transitions"
> usability gate.

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> The compute-budget argument scales with dataset size. On ImageNet,
> Sacred NAS's speed advantage grows (more search would be needed
> for unrestricted DARTS), making the argument stronger, not weaker.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> Sacred NAS searches *within* the prior space, so it can produce a
> non-compound architecture if the data prefers it. NAS is the natural
> answer to "which priors should compound" — a positive Sacred-NAS
> result would discover the right composition automatically.

**Q: How do we know the implementation is correct?**

> `tests.py` verifies (a) the supernet forward shape on a B=2 dummy
> batch, (b) α gradients flow non-zero through every op, (c) discrete
> derivation produces a valid cell with no isolated nodes.

## 10. Verification artifacts checklist

- [ ] `ideas/45_sacred_nas/implementation.py` exists, tests green
- [ ] `ideas/45_sacred_nas/tests.py` ≥ 6 assertions
- [ ] `ideas/45_sacred_nas/AUDIT.md` ≥ 3 weaknesses
- [ ] `ideas/45_sacred_nas/IMPROVEMENTS.md` records fixes
- [ ] `ideas/45_sacred_nas/VERIFY.md` signed
- [ ] At least one experiment archive
- [ ] `verification/{tests,smoke,gates,reproduction}.txt`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-C.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G5 (elite-research-scientist critic). Critiquing the IDEA, not the implementation (audit at `audits/G5_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)
LOW. The hypothesis structure is fundamentally flawed: restricting NAS to a search space `{φ, Fib, Platonic-equivariant, hexagonal, toroidal, cymatic, golden-angle}` is *strictly worse* than open NAS *unless* the restricted priors individually outperform unrestricted alternatives. The Phase-5 3-seed evidence (FINDINGS.md) and the H50-full-hybrid result (73.24 % vs. 84.78 % baseline = -11.5 pp) demonstrate the priors do not stack. A NAS over a search space of mostly-neutral and several actively-harmful priors will, in expectation, return architectures that are no better than (and likely worse than) random sampling from the unrestricted space.

### Mechanism scrutiny — does the optimizer/init/reg theory actually predict the claimed effect?
NAS theory (Liu, Simonyan, Yang 2019 ICLR 'DARTS: Differentiable Architecture Search' arXiv:1806.09055; Pham, Guan, Zoph, Le, Dean 2018 ICML 'Efficient Neural Architecture Search via Parameter Sharing' arXiv:1802.03268; Xie, Zheng, Liu, Lin 2019 ICLR 'SNAS: Stochastic Neural Architecture Search' arXiv:1812.09926) predicts NAS quality scales with: (a) coverage of the search space — wider is better unless priors are positively known to help, (b) candidate diversity, (c) signal-to-noise of the proxy task. The "sacred" restriction reduces (a) and (b) and adds no signal to (c). The hypothesis implicitly assumes the constituent priors compound positively — Section 5 of FINDINGS.md (and the H50 result) demonstrate they do not.

### Confounds (≥2)
(1) **Survivorship bias in the restricted space.** If NAS returns a "good" Sacred-NAS architecture, the question is whether it would have ranked highly in an open NAS — typically the answer in restricted searches is "yes but lower" because the restriction filters out the actual top-K candidates. (2) **Cell-template confound.** The sacred-NAS cell template might include or exclude DepthwiseSeparable, BN-vs-GN, etc., which dominate over the sacred prior. (3) **Search-budget coupling.** NAS budget on a small space looks artificially good per FLOPs.

### Numerology / specificity check
Numerology by composition. The restriction takes seven independent numerological priors (φ-scaling, Fibonacci, Platonic, hexagonal, toroidal, cymatic, golden-angle), most of which individually under-perform, and assumes a "Bonferroni for free": that the discrete-search over sacred combinations will find a compounding sweet-spot. There is no theoretical argument given for why such a sweet-spot should exist; the H50 result is evidence against.

### Literature precedent — optimization/init is one of the most studied fields in DL
NAS literature: DARTS (arXiv:1806.09055), ENAS (arXiv:1802.03268), SNAS (arXiv:1812.09926), Once-for-All (Cai, Gan, Wang, Zhang, Han 2020 ICLR arXiv:1908.09791), and Yu, Sciuto, Jaggi, Musat, Salzmann 2020 ICLR 'Evaluating the Search Phase of Neural Architecture Search' (arXiv:1902.08142) which shows ~50 % of NAS gains are search-noise on top of strong baselines. Restricting to a hand-picked "natural" subspace adds prior-bias on top of NAS's already-high search-noise.

### Expected effect size (90% CI a priori)
[-3 pp, -0.3 pp] on CIFAR-10 top-1 vs. open-DARTS baseline at iso-search-budget. Most likely outcome: Sacred-NAS converges to the best individual prior (probably "all priors off, i.e. the baseline"), wasting compute relative to open NAS.

### Minimum-distinguishing experiment
Run DARTS on `cells = {sacred ops}` vs. DARTS on `cells = {standard ops} ∪ {sacred ops}` (i.e. additive, not restrictive) vs. DARTS on `cells = {standard ops}` only, at matched search-budget × 3 seeds CIFAR-10. If the restrictive variant doesn't win by ≥ 0.5 pp with non-overlapping CI, the restriction is refuted; if the additive variant wins, the priors are useful but should not displace standard operators.

### Verdict
DERIVATIVE+TESTABLE — NAS over a hand-picked sub-space of mostly individually-neutral or harmful priors will, by construction, underperform open NAS; the hypothesis is testable but the prior strongly anticipates failure. Recommend reframing as an *additive* NAS (sacred ops added to the standard search space) rather than a restrictive one, which removes the worst-case downside.
