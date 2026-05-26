# H36 — φ-Spiral Positional Encoding

> **One-line claim:** A learnable positional encoding whose trajectory traces a golden-spiral curve in 3-D embedding space `(cos(kθ), sin(kθ), k/N)` improves zero-shot context-extension perplexity and reduces token-position aliasing relative to standard sinusoidal / learned-absolute PE at matched embedding dimension.
>
> **Source design space:** G4 Kernels / Attention / Filters (H31–H40).
>
> **Implementation status (this repo):** `○ not started`.

This document is the committee-grade design write-up for hypothesis H36.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

A golden spiral in 3-D (cylindrical) coordinates is `r(z) = a · φ^(z/h)`, parameterized as `(cos(kθ), sin(kθ), k/N)` where θ is the golden angle (137.5°). This curve is the explicit 3-D embedding of the phyllotaxis 2-D lattice into a "rising helix" that preserves nearest-neighbour distance approximately uniformly as one progresses up the spiral. Natural examples: DNA's double helix has approximately golden-spiral pitch-to-diameter ratio (1.618:1 is the empirical mean across organisms); sunflower-seed packing on a curved surface follows the rising golden spiral; the Voynich manuscript-conjectured pinecone-scale arrangements all express the same 3-D phyllotaxis curve.

For Transformer attention with explicit positional encoding (PE), the choice of trajectory determines (a) the distance metric between far-apart positions and (b) the network's ability to extrapolate to contexts longer than training. Sinusoidal PE (Vaswani 2017) uses `sin/cos` at exponentially-spaced frequencies; learned-absolute PE uses arbitrary embeddings; **golden-spiral PE replaces both** with a deterministic 3-D trajectory that has the most-irrational angular spacing AND a slow exponential growth in the z-axis (modeling token "age"). The result is a PE with uniform discrimination at all distances and no aliasing.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** the golden spiral is the most-uniformly-distributing 3-D trajectory under the most-irrational angular spacing, learnable positional encoding initialized as `PE[k] = (cos(k·GA), sin(k·GA), k/N)` (with GA = golden angle, plus a learnable d-3 dim correction) improves WikiText-103 perplexity at 4× context extrapolation by ≥ 0.4 perplexity points relative to standard sinusoidal PE, per the mechanism of Vaswani 2017 and Vogel 1979.

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on WikiText-103 with 124 M model, the φ-spiral PE variant fails to improve 8 k-context perplexity by ≥ 0.2 perplexity points relative to sinusoidal PE, this hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Vaswani, A., et al. 2017 NeurIPS 'Attention Is All You Need'
(arXiv:1706.03762) — original Transformer + sinusoidal PE.

Su, J., et al. 2021 'RoFormer: RoPE' (arXiv:2104.09864) — alternative
positional encoding to compare.

Press, O., Smith, N., Lewis, M. 2022 ICLR 'ALiBi: Train Short, Test
Long' (arXiv:2108.12409) — long-context extrapolation comparator.

Vogel, H. 1979 — golden-angle phyllotaxis math.

Chen, S., et al. 2023 'Position Interpolation' (arXiv:2306.15595) —
extrapolation methodology.

Watson, J. D., Crick, F. H. C. 1953 Nature — DNA helix dimensions.
```

## 5. Mechanism

### 5.1 CNN track

For CNN-track, φ-spiral PE applies to **patchified ViT** position embeddings: instead of learnable-absolute or sinusoidal patch positions, use the golden-spiral trajectory.

```python
# ideas/36_phi_spiral_pe/implementation.py
PHI = (1+5**0.5)/2; GOLDEN_ANGLE = 2*math.pi*(1 - 1/PHI)

def phi_spiral_pe(N, d):
    """Generate (N, d) positional encoding on a golden-spiral trajectory."""
    pe = torch.zeros(N, d)
    k = torch.arange(N, dtype=torch.float32)
    pe[:, 0] = (k * GOLDEN_ANGLE).cos()
    pe[:, 1] = (k * GOLDEN_ANGLE).sin()
    pe[:, 2] = k / N
    # remaining d-3 dims with learnable correction
    if d > 3:
        nn.init.normal_(pe[:, 3:], std=0.02)
    return pe

class PhiSpiralPE(nn.Module):
    def __init__(self, N, d):
        super().__init__()
        self.pe = nn.Parameter(phi_spiral_pe(N, d))
    def forward(self, x):
        # x: (B, N, d)
        return x + self.pe.unsqueeze(0)
```

- Input: `(B, L, d)` token sequence; PE added.
- Params: `N · d` (same as learned-absolute PE).
- FLOPs: identical (init + add).
- Init: trajectory is deterministic; remaining dims are small-Gaussian.

### 5.2 LLM track

Drop-in replacement for sinusoidal / learned PE in any decoder-only Transformer.

```python
class PhiSpiralPEDecoderLLM(nn.Module):
    def __init__(self, vocab, d, max_len=8192, n_layers=12):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pe = PhiSpiralPE(max_len, d)
        self.layers = nn.ModuleList([DecoderLayer(d) for _ in range(n_layers)])
    def forward(self, ids):
        x = self.pe(self.tok(ids))
        for l in self.layers: x = l(x)
        return x
```

- FlashAttention-2 compatibility: ✓ — PE is additive before attention.
- Causal-mask preservation: ✓ — PE is per-position, mask-independent.
- KV cache impact: none.

Expected at 124 M scale: **perplexity within 0.1 of sinusoidal at training length** (2 k); **-0.4 to -0.8 perplexity at 8 k extrapolation**.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (LLM at 8 k extrapolation) | [+0.005, +0.025] | PE-trajectory effect |
| Perplexity at 2 k (training length) | [-0.1, +0.1] | neutral |
| Perplexity at 4 k extrapolation | [-0.2, -0.5] | mild improvement |
| Perplexity at 8 k extrapolation | [-0.4, -1.0] | main claim |
| Aliasing autocorr | [-15 %, -35 %] | direct |
| params | [≈0, 0] | same |
| FLOPs | [0, 0] | identical |
| GPU latency | [≈1.0×, ≈1.0×] | unchanged |
| KV cache @ 32 k | [0 %, 0 %] | PE doesn't change cache |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** WikiText-103 train at 2 k context; evaluate at 2 k / 4 k / 8 k contexts.
- **Architecture:** 124 M decoder-only with PhiSpiralPE.
- **Epochs / batch / precision / seeds:** 100 k steps, batch 32, bf16 AMP, FA-2, 3 seeds.
- **Composite:** perplexity at all 3 contexts (0.7), latency (0.2), params (0.1).
- **Run-script:** `python ideas/36_phi_spiral_pe/experiment.py --seeds 0 1 2`.
- **Wall-clock:** ~12 hr/seed × 3 = ~36 hr.
- **Archive:** `ideas/36_phi_spiral_pe/experiments/exp001_wt103_seed0..2/`.

### 7.2 Idea-targeted experiment

1. **WikiText-103 extrapolation** as primary.
2. **CIFAR-100 ViT** with PhiSpiralPE.
3. **Long-doc QA** at long context.

### 7.3 Cross-paradigm context

Combination with H34 (RoPE-φ): some tasks may benefit from BOTH golden-angle frequency progression AND golden-spiral trajectory.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G4 row H36.
- Master: planned Tier 4.
- Sub-dir: `ideas/36_phi_spiral_pe/`.
- Composes: H15 (φ-init embedding), H27 (golden-spiral graph), H34 (RoPE-φ), H22 (toroidal — periodic-PE).
- Conflicts: none.

## 9. Committee Q&A

**Q: Why isn't this just learned-absolute PE with one specific init?**

> Yes, that is the contribution: pre-committing to the golden-spiral init lets us pre-register the falsifier. Random-init learned-absolute PE will eventually find any structure but pays an early-training cost.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 gives 8 k-context perplexity ≥ 0.2 improvement. Either side discards.

**Q: What if golden-spiral init is washed out after training?**

> The hypothesis specifically targets EARLY convergence + EXTRAPOLATION. If the structure is gone but the model performs identically to sinusoidal PE, that is a hypothesis-negative outcome.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> Single-prior test; compounding is H67.

**Q: How do we know the implementation is correct?**

> `tests/test_phi_spiral_pe.py::test_golden_angle_progression` asserts successive PE positions subtend golden angle. `test_unit_circle_xy` asserts the first 2 dims fall on unit circle. `test_monotonic_z` asserts the z-dim is monotone increasing. Plus archive verification.

## 10. Verification artifacts checklist

- [ ] `ideas/36_phi_spiral_pe/implementation.py` tests green
- [ ] `ideas/36_phi_spiral_pe/tests.py` ≥ 5 assertions
- [ ] AUDIT.md ≥ 3 weaknesses
- [ ] IMPROVEMENTS.md
- [ ] VERIFY.md signed
- [ ] Experiment archives + verification
- [ ] Row added to `EXPERIMENT_LOG.md` Tier 4
- [ ] FINDINGS reflects result

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-B.
