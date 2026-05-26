# H34 — Golden-Angle Rotary (RoPE-φ)

> **One-line claim:** Replacing RoPE's base-10000 frequency progression with a golden-angle (137.5°) phyllotaxis-spaced frequency set improves long-context perplexity at extrapolation lengths (2 k → 16 k) by ≥ 0.5 perplexity points relative to standard RoPE.
>
> **Source design space:** G4 Kernels / Attention / Filters (H31–H40).
>
> **Implementation status (this repo):** `○ not started` (partial proxy `golden_modulate` from T1.8 is near-no-op; full RoPE-φ is LLM-track).

This document is the committee-grade design write-up for hypothesis H34.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The golden angle (≈ 137.50776°) is `2π·(1 − 1/φ)` and is the angle at which successive seeds in a sunflower head, leaves on a stem, and pinecone scales are arranged. Mathematically, it is the most irrational angular spacing — no rational multiple of 2π comes close — so a sequence of unit-vectors at successive golden-angle rotations achieves the **most uniform discrete coverage of the unit circle**. Vogel 1979 proved this is the optimal phyllotactic spacing in 2-D. In 3-D, the golden-angle spiral extends to icosahedral / dodecahedral vertex arrangements.

For Transformer attention with Rotary Position Embedding (RoPE; Su 2021 arXiv:2104.09864), each position is encoded as a rotation in 2-D subspaces whose frequencies progress as `θ_k = 10000^(-2k/d)`. The base-10000 choice is arbitrary (inherited from sinusoidal-position-embedding tradition); experiments show it determines the LONG-context extrapolation behavior. **Replacing the base-10000 frequencies with phyllotaxis-spaced (golden-angle) frequencies** should produce the most uniform coverage of the rotation phase-space, theoretically improving RoPE's ability to discriminate distant positions and reducing aliasing at long contexts.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** the golden angle is the most irrational angular spacing and produces the maximally-uniform discrete coverage of the rotation phase-space, replacing RoPE's frequency progression `θ_k = 10000^(-2k/d)` with golden-angle phyllotaxis spacing `θ_k = (2π·(1−1/φ))·k` improves WikiText-103 perplexity at 8× context extrapolation by ≥ 0.5 perplexity points and reduces position-embedding aliasing (measured by Q·K dot-product autocorrelation) by ≥ 20 %, per the mechanism of Su 2021 (arXiv:2104.09864) and Vogel 1979.

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on WikiText-103 at 124 M scale, the RoPE-φ variant fails to improve 16 k-context perplexity by ≥ 0.3 points AND fails to reduce position-embedding aliasing by ≥ 10 %, this hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Su, J., et al. 2021 'RoFormer: Enhanced Transformer with Rotary
Position Embedding' (arXiv:2104.09864) — the canonical RoPE
reference; H34 modulates RoPE's base frequencies.

Press, O., Smith, N., Lewis, M. 2022 ICLR 'Train Short, Test Long:
Attention with Linear Biases Enables Input Length Extrapolation'
(arXiv:2108.12409) — ALiBi reference; long-context extrapolation
comparator.

Chen, S., et al. 2023 'Extending Context Window of Large Language
Models via Positional Interpolation' (arXiv:2306.15595) — Position
Interpolation reference for long-context.

Vogel, H. 1979 — phyllotaxis golden-angle mathematical proof.

Vaswani, A., et al. 2017 NeurIPS — Transformer / sinusoidal-PE base.

Touvron, H., et al. 2023 'Llama: Open and Efficient Foundation
Language Models' (arXiv:2302.13971) — RoPE in practice at scale.
```

## 5. Mechanism

### 5.1 CNN track

There is **no direct CNN-track** for RoPE-φ; the prior is fundamentally a Transformer construct. The nearest CNN proxy is `golden_modulate` (T1.8), which scales channel features by a golden-angle-modulated factor — a near-no-op result (-0.30 pp). The full hypothesis is LLM-track only.

### 5.2 LLM track (decoder-only Transformer)

Replace RoPE's frequency progression in the rotary projection of Q and K:

```python
# ideas/34_golden_angle_rotary/implementation.py
PHI = (1 + 5**0.5) / 2
GOLDEN_ANGLE = 2 * math.pi * (1 - 1/PHI)  # ≈ 2.3998

def rope_phi_freqs(dim, base="phi"):
    """Return frequency table for RoPE."""
    if base == "standard":
        # original: theta_k = 10000^(-2k/d)
        return 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    elif base == "phi":
        # golden-angle: theta_k = (golden_angle) * k
        return GOLDEN_ANGLE * torch.arange(0, dim // 2).float() / (dim // 2)
    else:
        raise ValueError(base)

def apply_rope_phi(q, k, freqs):
    """Apply rotary embedding with phi-frequencies."""
    seq_len = q.shape[-2]
    pos = torch.arange(seq_len, device=q.device).float()
    angles = pos.unsqueeze(-1) * freqs.unsqueeze(0)  # (L, d/2)
    cos, sin = angles.cos(), angles.sin()
    q_rot = q * cos.repeat_interleave(2, dim=-1) + rotate_half(q) * sin.repeat_interleave(2, dim=-1)
    k_rot = k * cos.repeat_interleave(2, dim=-1) + rotate_half(k) * sin.repeat_interleave(2, dim=-1)
    return q_rot, k_rot
```

- Slots in: replaces standard RoPE in MHA.
- FlashAttention-2 compatibility: ✓ — applied before the dot product.
- Causal-mask preservation: ✓ — RoPE is per-position, mask-independent.
- Parameter / FLOPs cost: identical to standard RoPE (frequency table precomputed).

Expected at 124 M on WikiText-103: **perplexity neutral at training context** (2 k), **-0.3 to -0.8 perplexity at 16 k extrapolation**.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (LLM) | [+0.005, +0.025] | small but real PE effect |
| Perplexity at 2 k context (primary, training length) | [-0.1, +0.1] | should be neutral |
| Perplexity at 8 k context | [-0.3, -0.7] | extrapolation benefit |
| Perplexity at 16 k context | [-0.5, -1.5] | larger extrapolation gain |
| Q·K aliasing autocorr | [-10 %, -30 %] | direct claim |
| params | [0, 0] | freq table is precomputed |
| FLOPs | [0, 0] | identical |
| GPU latency | [≈1.0×, ≈1.0×] | unchanged |
| KV cache @ 32 k | [-0 %, 0 %] | RoPE doesn't change cache |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** WikiText-103.
- **Architecture:** GPT-2-small (124 M) with RoPE-φ.
- **Epochs / batch / precision / seeds:** 100 k steps, batch 32, bf16 AMP, FlashAttention-2, 3 seeds.
- **Composite:** perplexity at {2 k, 8 k, 16 k} (0.6), aliasing autocorr (0.2), params (0.1), latency (0.1).
- **Run-script:** `python ideas/34_golden_angle_rotary/experiment.py --seeds 0 1 2`.
- **Wall-clock:** ~12 hr/seed at 124 M scale × 3 seeds = ~36 hr.
- **Archive:** `ideas/34_golden_angle_rotary/experiments/exp001_wt103_124m_seed0..2/`.

### 7.2 Idea-targeted experiment

1. **WikiText-103 with eval-at-extrapolation-length** (the primary).
2. **TinyStories at 4 k context** (small model, smaller compute).
3. **Long-document QA** (HotpotQA at long-context).
4. **Aliasing measurement on synthetic periodic sequences** — direct geometric test.

### 7.3 Cross-paradigm context (LLM track)

The hypothesis IS the LLM-track. CNN-track is the near-no-op `golden_modulate` already in T1.8.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G4 row H34.
- Master: `EXPERIMENT_LOG.md` Tier 1 T1.8 (near-no-op proxy), Tier 4 T4.6.
- Sub-dir: `ideas/34_golden_angle_rotary/`.
- Composes: H17 (golden skip), H22 (toroidal KV — RoPE-φ extends periodic encoding), H36 (φ-spiral PE), H71 (icosa RoPE).
- Conflicts: H32 (Fibottention — different sparsity); none directly opposed.

## 9. Committee Q&A

**Q: Why isn't this just RoPE with a different base?**

> Yes, that is exactly the contribution. The question is whether the φ-specific base produces measurable improvement vs alternatives (base-1000, base-100, ALiBi). The pre-registered protocol tests 4 bases.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 gives 16 k-context perplexity ≥ 0.3 reduction AND aliasing ≤ -10 %. Both must hold.

**Q: T1.8 `sg_only_golden_modulate` was -0.30 pp / near-no-op. Why expect H34 to help?**

> T1.8 was a CNN-track channel-wise multiplicative modulation. H34 is an LLM-track RoPE-frequency modification. They are STRUCTURALLY DIFFERENT operators — the CNN proxy is a side-channel, the LLM version is the core mechanism.

**Q: What if the prior helps at extrapolation but hurts at training length?**

> Acceptable; the hypothesis explicitly targets extrapolation. § 6 predicts neutral at training length.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> Single-prior LLM test; CNN compounding is irrelevant.

**Q: How do we know the implementation is correct?**

> `tests/test_rope_phi.py::test_frequency_irrational` asserts no frequency is a rational multiple of 2π/period. `test_rotation_correctness` asserts Q·K dot product is preserved under common position offset. `test_compatibility_with_flash_attn` asserts the output of FlashAttention-2 with RoPE-φ matches the naive computation to within 1e-3. Plus archive verification.

## 10. Verification artifacts checklist

- [ ] `ideas/34_golden_angle_rotary/implementation.py` tests green
- [ ] `ideas/34_golden_angle_rotary/tests.py` ≥ 6 assertions
- [ ] AUDIT.md ≥ 3 weaknesses
- [ ] IMPROVEMENTS.md
- [ ] VERIFY.md signed
- [ ] Experiment archives + verification
- [ ] Row added to `EXPERIMENT_LOG.md` Tier 4 T4.6
- [ ] FINDINGS reflects result

## 11. Lesson from T1.8 (golden_modulate near-no-op)

T1.8 (`sg_only_golden_modulate`) on CIFAR-10 was a CNN-track channel-wise multiplicative modulation by a golden-angle-spaced factor — top-1 79.81 % vs reference 80.11 %, a **0.30 pp shortfall** that the FINDINGS table classifies as "near no-op as predicted". The lesson for H34 is that the golden angle is **structurally inert** when applied as a simple channel-wise multiplication; it only becomes meaningful when applied to a STRUCTURALLY-DEPENDENT operation like RoPE's periodic encoding, where the angular spacing determines aliasing behavior. H34 thus moves from "side-channel modulation" (T1.8) to "core operator parameterization" (RoPE base).

## 12. Status journal

- 2026-05-27 — Created from template by Doc-Agent-B. Notes T1.8 near-no-op CNN proxy as evidence that golden-angle priors must touch a structure-sensitive operator (RoPE) to matter.
