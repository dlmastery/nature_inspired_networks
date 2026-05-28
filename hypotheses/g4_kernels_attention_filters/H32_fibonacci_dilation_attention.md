# H32 — Fibonacci Dilation Attention (Fibottention)

> **One-line claim:** A multi-head attention pattern where each head uses a different Fibonacci dilation (1, 2, 3, 5, 8, 13, 21, …) constructs an O(N log N) sparse-attention mechanism that retains ≥ 95 % of dense-attention accuracy on CIFAR-100 / Tiny-ImageNet at 2–6 % attention density per head.
>
> **Source design space:** G4 Kernels / Attention / Filters (H31–H40).
>
> **Implementation status (this repo):** `○ not started`. Direct follow-up of the published Fibottention paper (Rajagopalan 2024 arXiv:2406.19391).

This document is the committee-grade design write-up for hypothesis H32.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The Fibonacci sequence appears in phyllotaxis, pinecone scales, sunflower spirals, and many self-similar biological growth processes. Critically, the **Wythoff array** — an integer table whose rows are Fibonacci-like sequences and whose columns partition the positive integers into non-overlapping classes — provides a mathematical mechanism for **sparse coverage of an integer lattice**: every positive integer appears exactly once across all Wythoff rows. This makes it the natural basis for designing **non-overlapping sparse-attention dilations**: each row of the array is one attention head's "stride pattern", and the union of all heads covers the integer lattice exactly once.

For Transformer attention, dense N×N attention is the dominant compute bottleneck at long context. Linear-attention approximations exist (Performer, Linformer) but lose precision. The Fibottention hypothesis (Rajagopalan 2024 arXiv:2406.19391) shows that **multi-head attention with non-overlapping Fibonacci-dilation patterns** achieves O(N log N) per-head computation while retaining ≥ 95 % of dense attention's accuracy on multiple benchmarks. Each head attends only to tokens at distances {1, 2, 3, 5, 8, 13, 21, …} (i.e., Fibonacci-spaced), giving a multiscale receptive field that mirrors phyllotaxis spatial sampling. The combination provides scale-aware sparse attention at 2–6 % density per head with multi-scale coverage.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** Fibonacci dilations span the integer lattice with non-overlapping Wythoff-array partitions, an 8-head attention block where head k attends only to tokens at Fibonacci-spaced offsets {1, 2, 3, 5, 8, 13, 21, 34} reduces attention compute by ≥ 90 % while retaining ≥ 95 % of dense-attention top-1 accuracy on CIFAR-100 ViT and Tiny-ImageNet, per the mechanism in Rajagopalan 2024 Fibottention (arXiv:2406.19391).

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on CIFAR-100 / ViT-Tiny, the Fibottention variant fails to retain ≥ 93 % of dense-attention top-1 AND fails to reduce attention FLOPs by ≥ 85 %, this hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Rajagopalan, R., et al. 2024 'Fibottention: Inceptive Visual
Representation Learning with Diverse Attention across Heads'
(arXiv:2406.19391) — the primary reference; Fibonacci-dilation attention
with Wythoff-array non-overlapping head allocation, +1 to +2 pp top-1
on multiple ViT benchmarks at 2-6% attention density.

Vaswani, A., et al. 2017 NeurIPS 'Attention Is All You Need'
(arXiv:1706.03762) — the dense-attention reference Fibottention sparsifies.

Beltagy, I., Peters, M. E., Cohan, A. 2020 'Longformer: The Long-
Document Transformer' (arXiv:2004.05150) — comparator sparse-attention
work.

Dao, T., et al. 2022 NeurIPS 'FlashAttention' (arXiv:2205.14135) — dense
attention kernel that Fibottention replaces in long-context scenarios.

Wythoff, W. A. 1907 — original Wythoff-array number theory.

Vogel, H. 1979 — phyllotaxis mathematical motivation.
```

## 5. Mechanism

### 5.1 CNN track (ViT-Tiny)

Construct an 8-head attention block where head k has a sparse attention mask: each query at position i attends only to keys at positions `i ± F_k, i ± 2·F_k, i ± 3·F_k, …` (where F_k is the k-th Fibonacci number).

```python
# ideas/32_fibottention/implementation.py
FIB = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

def fibottention_mask(L, head_idx, max_dist=None):
    """Generate sparse-attention mask for head_idx with F_{head_idx} dilation."""
    F = FIB[head_idx]
    max_dist = max_dist or L
    mask = torch.zeros(L, L, dtype=torch.bool)
    for d in range(1, max_dist // F + 1):
        # diagonal at offset ±d·F
        offset = d * F
        for i in range(L):
            if i + offset < L: mask[i, i+offset] = True
            if i - offset >= 0: mask[i, i-offset] = True
    # always attend to self
    mask = mask | torch.eye(L, dtype=torch.bool)
    return mask

class FibottentionMHA(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3*d); self.proj = nn.Linear(d, d)
        self.head_dim = d // heads

    def forward(self, x):
        B, L, d = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        outs = []
        for h in range(self.heads):
            mask = fibottention_mask(L, h).to(x.device)
            attn = (q[:,h] @ k[:,h].transpose(-2,-1)) / math.sqrt(self.head_dim)
            attn = attn.masked_fill(~mask, float('-inf'))
            outs.append(attn.softmax(-1) @ v[:,h])
        out = torch.stack(outs, dim=1).transpose(1,2).reshape(B, L, d)
        return self.proj(out)
```

- Input shape: `(B, L, d)`.
- Params: same as dense MHA (qkv + proj).
- FLOPs: 2-6 % of dense per head (varies with head Fib number).
- Init: Xavier.

### 5.2 LLM track

For decoder-only Transformers, Fibottention is the direct mechanism: replace dense MHA with Fibottention MHA. With causal masking, the dilation pattern is enforced only on past tokens.

```python
def causal_fibottention_mask(L, head_idx):
    mask = fibottention_mask(L, head_idx)
    return mask & torch.tril(torch.ones_like(mask, dtype=torch.bool))
```

- FlashAttention-2 compatibility: requires custom Triton kernel for arbitrary sparse mask; baseline implementation uses manual masked attention (slower but correct).
- Causal-mask preservation: ✓ via lower-triangular intersection.

Expected at 124 M scale on WikiText-103: **perplexity within 0.3 of dense** (≥ 95 % retention), **attention FLOPs reduced 90 %**, **practical latency reduced 5–8× at 32 k context**.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (CIFAR-100 ViT) | [-0.005, +0.020] | small loss in top-1 traded for big FLOPs win |
| top-1 (CIFAR-100, primary) | [-1.0 pp, +0.5 pp] | within retention threshold |
| Attention FLOPs | [-95 %, -88 %] | direct claim |
| params | [0, 0] | identical |
| FLOPs (total) | [-50 %, -30 %] | depends on attn share of total |
| GPU latency (batch=1) | [×0.4, ×0.7] | mask overhead currently; better with Triton kernel |
| Perplexity (LLM, WikiText-103) | [+0.0, +0.3] | retention claim |
| KV cache @ 32 k (LLM) | [-70 %, -90 %] | sparse access pattern |
| Betti collapse rate | [+0.0, +0.10] | small effect |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** CIFAR-100 with ViT-Tiny (patch 4, depth 6, heads 8, dim 192).
- **Architecture:** ViT-Tiny with FibottentionMHA replacing standard MHA.
- **Epochs / batch / precision / seeds:** 50 epochs, batch 256, bf16 AMP, 3 seeds.
- **Composite:** top-1 (0.5), attn FLOPs (0.3), latency (0.1), params (0.1).
- **Run-script:** `python ideas/32_fibottention/experiment.py --seeds 0 1 2`.
- **Wall-clock:** ~30 min/seed × 3 = 90 min.
- **Archive:** `ideas/32_fibottention/experiments/exp001_cifar100_seed0..2/`.

### 7.2 Idea-targeted experiment

1. **CIFAR-100 ViT-Tiny** (small ViT where compute is meaningful).
2. **Tiny-ImageNet ViT-Small** (longer sequence, sparser pays more).
3. **Long-context language modeling** (WikiText-103 at 8 k–32 k context) — direct LLM-track.

### 7.3 Cross-paradigm context (LLM track)

WikiText-103 at 124 M with Fibottention MHA. Train 100 k steps at 2 k context, evaluate perplexity at 2 k / 8 k / 32 k context lengths. Report retention vs dense and FLOPs reduction.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G4 row H32.
- Master: planned Tier 1/2 row T4.5.
- Sub-dir: `ideas/32_fibottention/`.
- Composes: H02 (Fib depth), H16 (Fib head counts), H38 (fractal kernels), H62 (toroidal KV + hex attn).
- Conflicts: H37 (pentagonal attention — different head grouping); H34 (golden-angle RoPE — orthogonal but combinable).

## 9. Committee Q&A

**Q: Why isn't this just a re-implementation of Rajagopalan 2024?**

> Largely yes — but we (a) integrate with the NaturePriorBlock scaffold for ablation with other priors, (b) test specifically on CIFAR-100 / Tiny-ImageNet ViT (Rajagopalan focused on larger ViT-S/B), and (c) provide the LLM-track variant which Rajagopalan does not.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 gives accuracy retention ≥ 93 % AND attn-FLOPs reduction ≥ 85 %. Both must hold.

**Q: What if Fibonacci spacing is no better than uniform stride sparse attention?**

> Run a stride-S sparse-attention baseline matched to Fibottention's per-head density. If they tie within 0.5 pp top-1 at matched FLOPs, the Fibonacci-specificity is null.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H32 is tested as a single prior; compounding is H50/H62/H67.

**Q: How do we know the implementation is correct?**

> `tests/test_fibottention.py::test_density_2_to_6_pct` asserts attention density at L=256 is between 2 % and 6 %. `test_wythoff_non_overlap` asserts the 8 heads cover disjoint subsets of the Wythoff array (within tolerance). `test_causal_preserved` verifies the causal mask. Plus archive verification.

## 10. Verification artifacts checklist

- [ ] `ideas/32_fibottention/implementation.py` tests green
- [ ] `ideas/32_fibottention/tests.py` ≥ 6 assertions
- [ ] AUDIT.md ≥ 3 weaknesses
- [ ] IMPROVEMENTS.md
- [ ] VERIFY.md signed
- [ ] Experiment archives + verification
- [ ] Row added to `EXPERIMENT_LOG.md` Tier 4 T4.5 (LLM)
- [ ] FINDINGS reflects result

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-B.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G4 (elite-research-scientist critic). Critiquing the IDEA, not the implementation (audit at `audits/G4_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)

**MED.** Sparse attention works (Beltagy 2020 Longformer arXiv:2004.05150; Zaheer 2020 BigBird arXiv:2007.14062; Child 2019 Sparse Transformer arXiv:1904.10509 all demonstrate that strided/window/global patterns recover ~95% of dense accuracy at 5-10% density). Rajagopalan 2024 Fibottention (arXiv:2406.19391) provides direct precedent. The MED rating reflects "sparse attention works in general" + "Fibonacci-specific claim is unproven beyond one paper".

### Mechanism scrutiny

The Wythoff-non-overlap argument is overstated. The Wythoff array partitions positive integers into non-overlapping rows (Beatty sequences with irrational density φ⁻¹, φ⁻²). But the proposed mechanism uses `F_k = {1, 2, 3, 5, 8, ...}` as DILATIONS — heads 0 and 1 attend at strides 1 and 2, and at L=256, head-0 attends to ALL positions (stride 1 covers every offset), so the union has NO overlap with head-1 not because of Wythoff but because head-0 already covers everything. The non-overlap property of Wythoff rows is a number-theoretic fact about INTEGERS, not about head-dilation MASKS. The implementation's `mask[i, i+d·F] = True` for d ∈ [1, L/F] produces strided masks that mostly OVERLAP across heads (head 1 at stride 2 = {2,4,6,8,…} contains head 2's stride 3 = {3,6,9,…} at offset 6, 12, …). The Wythoff-non-overlap claim is a marketing artifact.

### Confounds (≥2)

1. **Density confound.** Head k=0 has F=1 (full density!) → this is NOT sparse for the lowest head. The "2-6% per head" claim averages across heads but head-0 alone is 100% density. The actual sparsity comes from heads with large F (F=21, 34, …), which contribute almost nothing. A control with uniform stride-S sparse-attention at matched aggregate FLOPs is mandatory.
2. **Local-window confound.** Most attention gain comes from LOCAL tokens (within ~16 positions). Fibottention's heads 0-2 (F=1,2,3) capture this; the larger-F heads contribute little. Compare to a single local-window head + global-token head (BigBird's `block + global` recipe) at matched FLOPs.

### Numerology / specificity check

If you replace Fibonacci with ANY non-overlapping set of strides {s_k} with similar density coverage (e.g., powers of 2: {1, 2, 4, 8, 16, 32, 64, 128} = the dyadic stride pattern used in dilated CNNs since WaveNet 2016), do you get the same accuracy? Almost certainly yes. The Fibottention paper itself does not show that φ-growth specifically beats geometric growth. The Wythoff argument is irrelevant once head-0 is dense. The committee Q&A acknowledges this control ("Run a stride-S sparse-attention baseline") — but the hypothesis pre-commits no comparator strides, so the Fibonacci-specificity is not falsifiable.

### Literature precedent — kernel/attention design is a crowded field

This is direct follow-up of Rajagopalan 2024 (arXiv:2406.19391). The field has Longformer (Beltagy 2020 arXiv:2004.05150) sliding window + global; BigBird (Zaheer 2020 arXiv:2007.14062) sparse block + global + random; Sparse Transformer (Child 2019 arXiv:1904.10509) strided/fixed; Reformer (Kitaev 2020 arXiv:2001.04451) LSH; Routing Transformer (Roy 2020 arXiv:2003.05997). All achieve similar accuracy/density trade-offs. Fibottention is one point in this design space, not a paradigm shift.

### Expected effect size (90% CI a priori)

Accuracy retention [92%, 96%] at 6% density — matches Rajagopalan and matches all other sparse-attention work. The claim of ≥ 95% retention is plausible. The Fibonacci-specificity claim is null with > 80% prior probability.

### Minimum-distinguishing experiment

At matched aggregate density (sum of per-head densities), compare: (a) Fibottention {1,2,3,5,8,13,21,34}, (b) geometric {1,2,4,8,16,32,64,128}, (c) Wythoff-array rows (strict Beatty partition, NOT Fibonacci), (d) random non-overlapping strides. If (a) ≠ (b), (c), (d) by ≥ 0.3 pp at 3-seed median on Tiny-ImageNet, Fibonacci is non-null. Otherwise the gain is "sparse attention works".

### Verdict
DERIVATIVE+TESTABLE — re-implementation of Rajagopalan 2024 with no novel mechanism; the Wythoff-non-overlap argument is marketing (head-0 density is 100%); falsifiable only via the missing stride-baseline control.
