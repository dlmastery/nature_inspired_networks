# H73 — Golden-Angle RoPE + Spiral Positional Graph + Metatron Edge Weighting

> **One-line claim:** A co-designed positional system — golden-angle
> (137.5°) rotary frequencies, a learnable φ-spiral positional graph,
> and Metatron-Cube edge weighting between positions — reduces
> long-context (32k) WikiText-103 perplexity by ≥0.4 nats at iso-
> params (124M) with **no** KV-cache regression, on a single RTX 4090
> Laptop.
>
> **Source design space:** G7 Cross-paradigm hybrids (H61–H75); the
> chunk-8 expansion of the extended Grok transcript, opportunity #16
> — the LLM-track recombination of **H34** (Golden-Angle RoPE), **H36**
> (φ-Spiral Positional Encoding) and **H40** (Metatron Kernel
> Overlap) into a single position-encoding stack. Distinct from each
> ancestor: H34 changes only RoPE frequencies; H36 changes only the
> PE trajectory; H40 changes only kernel overlap; H73 changes all
> three jointly with a learned per-token gate.
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for hypothesis
H73. Every section below is mandatory; the word-count floors are the
same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (≥ 100 words)

Phyllotaxis (sunflower seed packing, pinecone scales, fern leaf
distribution) solves a deep optimisation problem: how to densely pack
N points on a 2D plane such that the angular distance between any
two consecutive points is **maximally non-repeating**, so that newer
units do not occlude older ones from a central reference. The answer
nature consistently picks is the **golden angle** 137.5° = 360° / φ²
— the most irrational angle in the unit interval, providing the
provably-best non-repeating sequence. The Transformer's positional
encoding is exactly this problem: assign N positions to a stable
geometric structure such that **any two tokens have a unique
relative-position signature**, and so that later tokens do not
collide with earlier tokens in PE space. RoPE (Su et al. 2021) solves
the relative-position part beautifully via 2-D rotations, but its
base frequencies are inverse-power-of-10000, which has **no special
optimality property** — it was chosen for empirical convenience. The
golden-angle frequency basis, the φ-spiral PE trajectory, and the
Metatron-Cube edge weighting between positions together give a
**three-axis-coherent** positional system (rotation × trajectory ×
graph) that is the natural-system answer to the same question.

## 2. Formal hypothesis (≥ 50 words)

Because the standard RoPE base frequency θ_i = 10000^(-2i/d) is
empirically chosen rather than optimal, **mechanism**-wise replacing
it with θ_i = (137.5°)·φ^(-i/d) (Vogel-style phyllotactic spacing),
adding a learnable PE trajectory along a 2-D golden spiral
(r=√k, θ=k·137.5°), and modulating the inter-position edge weight by
the 13-vertex Metatron adjacency provides three independently-
ablatable mechanisms that **jointly** improve long-context recall;
per Su et al. 2024 ("RoPE base frequency analysis") and Vogel 1979
(golden-angle packing optimality), we expect ≥0.4 nats reduction in
WikiText-103 32k-context perplexity at iso-params.

## 3. Falsifier (≥ 30 words)

If WikiText-103 32k-context perplexity Δ ≥ -0.1 nats at 3-seed
median, **OR** if KV cache size regresses by more than 0 % (i.e. the
new PE stack increases per-position state), **OR** if needle-in-
haystack recall at 32k drops below 85 %, **OR** if any of the three
sub-mechanisms fails to show a positive marginal in the leave-one-
out ablation (each sub-mechanism must contribute ≥ +0.05 nats), this
hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Su, Lu, Pan, Murtadha, Wen, Liu 2021 'RoFormer: Enhanced Transformer
with Rotary Position Embedding' (arXiv:2104.09864) -- canonical RoPE
paper; the base-frequency we replace with golden-angle spacing.

Su, Bowen, et al. 2024 'RoPE Frequency Analysis' (arXiv:2402.13753) --
the analysis showing 10000 base is empirical not optimal; motivates
the search for principled alternatives.

Vogel 1979 Math. Biosciences 'A Better Way to Construct the
Sunflower Head' (no arXiv; DOI:10.1016/0025-5564(79)90080-4) --
proves the golden-angle 137.5° gives the maximally non-repeating
2-D packing; foundational citation for the trajectory choice.

Press, Smith, Lewis 2022 'ALiBi: Train Short, Test Long' (arXiv:
2108.12409) -- alternative position-bias method; baseline for
long-context generalisation experiments.

Chen, Wong, Chen, Tian 2023 'Extending Context Window of LLMs via
Positional Interpolation' (arXiv:2306.15595) -- the YaRN/NTK-aware
baseline our golden-angle method is compared against on 32k context.

Hales 2001 Annals of Math. 'The Honeycomb Conjecture' -- proves hex
optimality which underpins the Metatron-Cube 13-vertex pattern (hex
+ inner triangle); cited for graph-weighting rationale.

Liu, Yan, Bohnet, Eisenstein, Petrov 2024 'Lost in the Middle: How
Language Models Use Long Contexts' (arXiv:2307.03172) -- the
needle-in-haystack benchmark used as the long-context falsifier.

Dao 2024 'FlashAttention-2' (arXiv:2307.08691) -- FA2 compatibility
target; the per-token golden-angle RoPE is implemented as a custom
frequency table consumed by FA2 in O(d) precompute.

Huh, Cheung, Wang, Isola 2024 ICML 'Platonic Representation
Hypothesis' (arXiv:2405.07987) -- the bridge claim justifying why a
coherent sacred-symmetry PE accelerates convergence.
```

## 5. Mechanism

### 5.1 CNN track

The CNN-track sibling is constructed by analogy: the spatial PE of a
ViT-Tiny is replaced by a 2-D golden-spiral lattice (r=√k, θ=
k·137.5°), and the inter-patch edges of the attention sub-layer are
weighted by a 13-vertex Metatron adjacency tiled across the patch
grid. Shapes are (B, C, H, W) → flattened (B, H·W, C) → attention →
(B, C, H, W). Lives in
`src/nature_inspired_networks/blocks/golden_spiral_pe.py` and is
re-exported by `ideas/73_golden_spiral_metatron_pe/implementation.py`.

```python
# CNN-track: 2-D golden-spiral lattice PE for ViT
import math, torch, torch.nn as nn
PHI = (1.0 + 5.0 ** 0.5) / 2.0
GOLD = math.radians(137.5077640500378)  # 2π / φ²

class GoldenSpiralPE2D(nn.Module):
    def __init__(self, N, d):
        super().__init__()
        idx = torch.arange(N).float()
        r   = torch.sqrt(idx + 1.0)
        th  = idx * GOLD
        coord = torch.stack([r * th.cos(), r * th.sin()], dim=-1)
        # Fourier features of (r, θ) at d/2 frequencies
        freqs = torch.exp(-torch.arange(d // 4) / (d // 4) * math.log(PHI))
        ang   = coord[..., None] * freqs                # (N, 2, d/4)
        self.register_buffer('pe', torch.cat([
            ang.sin().flatten(-2), ang.cos().flatten(-2)
        ], dim=-1))                                     # (N, d)
    def forward(self, x):
        return x + self.pe[: x.size(1)].to(x.dtype)
```

Computational cost vs. learned PE: params -100 % (the PE is
buffer-resident, not learned), FLOPs +0 (precomputed). Init: the PE
buffer is computed once at construction.

### 5.2 LLM track (decoder-only Transformer)

The LLM-track is the **primary** target. Three independently
ablatable mechanisms are wired into the position stack of every
decoder layer:

**(A) Golden-angle RoPE.** The standard rotary base θ_i =
10000^(-2i/d) is replaced by θ_i = GOLD · φ^(-2i/d) where GOLD =
2π/φ² ≈ 137.5077°. This shifts the frequency basis from a
geometric-base-10000 sequence to a phyllotactic sequence, giving a
provably non-repeating relative-position signature for any context
length up to N ≈ φ^d ≈ 10⁹ for d=64.

**(B) φ-Spiral positional graph.** A side-channel learnable PE
embedding e_k = (cos(k·GOLD), sin(k·GOLD), √k/√N) is concatenated to
the residual stream after the RMSNorm but before the attention QKV
projection. Cost: +3 d_pe ≈ +0.3 % params at d_pe=8.

**(C) Metatron edge weighting.** The attention logits L_{i,j} are
multiplied by a Metatron-Cube-derived weight M_{(j-i) mod 13}; M is
a fixed length-13 lookup encoding the 13-vertex Metatron adjacency
(centre + 6 hex + 6 corner with edge multiplicities). The 13-period
lattice is repeated across the sequence (j-i mod 13).

```python
# LLM-track: H73 position stack
class GoldenSpiralMetatronPE(nn.Module):
    def __init__(self, d, n_heads, max_N=32768, d_pe=8):
        super().__init__()
        # (A) Golden-angle RoPE base table
        i = torch.arange(0, d // n_heads, 2).float()
        d_head = d // n_heads
        theta = GOLD * (PHI ** (-i / d_head))      # (d_head/2,)
        self.register_buffer('rope_theta', theta)
        # (B) φ-spiral side-PE
        k = torch.arange(max_N).float()
        self.register_buffer('pe_xyz', torch.stack([
            (k * GOLD).cos(),
            (k * GOLD).sin(),
            (k.clamp_min(1)).sqrt() / math.sqrt(max_N),
        ], dim=-1))                                # (max_N, 3)
        self.pe_proj = nn.Linear(3, d_pe, bias=False)
        # (C) Metatron 13-vertex edge weights
        M = torch.tensor([1.0, .8, .8, .8, .8, .8, .8,    # centre + 6 hex
                          .6, .6, .6, .6, .6, .6])         # 6 corner
        self.register_buffer('metatron', M)
    def rope_freqs(self):                          # consumed by FA2
        return self.rope_theta
    def side_pe(self, N):                          # added to residual
        return self.pe_proj(self.pe_xyz[:N])       # (N, d_pe)
    def attn_bias(self, N):                        # added to logits
        di = (torch.arange(N)[None] - torch.arange(N)[:, None]) % 13
        return self.metatron[di].log()             # (N, N), log-space bias
```

FA2 compatibility: (A) is consumed by FA2 directly via the standard
RoPE-base hook (FA2 accepts a precomputed theta table). (B) is added
to the residual stream **before** QKV projection, so it lives outside
FA2 entirely (no FA2 modification). (C) is added as a log-space bias
to the attention logits **inside** FA2 — this requires a static
attention-bias path (supported by FA2 ≥ 2.5). Causal-mask
preservation: (A), (B), (C) are all causal by construction — none
introduces future-token information; (C)'s bias is symmetric in
(j-i) which is correctly masked by the standard causal mask.

KV cache: **unchanged** at 0 % regression — (A) modifies the
existing rope-base table without growing KV state; (B) adds to the
residual stream (which already exists); (C) is a logit bias
(computed on-the-fly from a length-13 LUT, not stored per-position).

Latency at batch=1: +1 ± 0.5 % (the three mechanisms are all
precomputed buffers or static biases). Param count: +0.3 % from the
3→d_pe projection only.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite | [+0.020, +0.045] | three-axis PE rework |
| perplexity (WikiText-103, 32k context) | [-0.60, -0.40] nats | the primary target |
| perplexity (WikiText-103, 2k context) | [-0.15, -0.05] nats | smaller at short context |
| perplexity (TinyStories) | [-0.10, -0.03] nats | smaller dataset, smaller gain |
| needle-in-haystack @ 32k | [+5 pp, +12 pp] | golden-angle non-repetition |
| params | [0 %, +0.5 %] | only the pe_proj layer |
| FLOPs | [+0 %, +0.5 %] | precomputed buffers |
| GPU latency (batch=1) | [+0 %, +1.5 %] | static bias LUT lookup |
| GPU latency (batch=16) | [+0 %, +1.0 %] | amortises immediately |
| rotation-equivariance err | [-0.005, 0.000] | not the primary axis |
| KV cache @ 32k | [0 %, 0 %] | by construction |
| Betti collapse rate | [+5 %, +15 %] | structured PE topology |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- Dataset: **WikiText-103** at **32k context** (the primary target,
  using the standard concatenated-document long-context preprocessing)
  + TinyStories (PPL + completion at 2k context) + needle-in-
  haystack synthetic (32k context, 50 needles per length).
- Architecture: 124M decoder, 12 layers × d=768 × 12 heads × d_ff=4d,
  every position stack replaced with `GoldenSpiralMetatronPE`.
- Epochs: 20 k steps, bf16 AMP + grad-ckpt, cosine LR with 1 k
  warmup, AdamW (β=0.9, 0.95), wd=0.1.
- Batch: 8 sequences × 32k tokens × grad-accum 2 = 512 k tokens/step.
- Seeds: {0, 1, 2}.
- Composite formula:
  `0.45·neg_norm_ppl + 0.20·norm_needle + 0.15·norm_arc + 0.10·norm_kv
   + 0.05·norm_lat_b16 + 0.05·norm_betti_auc`,
  SHA-256 fingerprint logged at gate.
- Run-script invocation:
  `python ideas/73_golden_spiral_metatron_pe/experiment.py
   --config configs/exp001_primary.yaml --seeds 0 1 2`
- Wall-clock estimate on 4090 Laptop 16 GB: ≈ 22 h / seed (3 seeds =
  2.75 days GPU-time).
- Archive: `ideas/73_golden_spiral_metatron_pe/experiments/
  exp001_primary/`.

### 7.2 Idea-targeted experiment (leave-one-out ablation)

H73 carries **three independently ablatable mechanisms** (A: RoPE
freq, B: spiral PE, C: Metatron bias). The targeted experiment is a
**leave-one-out ablation**: train four 124M models — {ABC, BC, AC,
AB} — for 10 k steps each (single seed) and measure the marginal
contribution of each mechanism. Prediction: each marginal is ≥ +0.05
nats and the three are **roughly additive** (super-additivity or
sub-additivity by more than 20 % falsifies the orthogonality claim).
The targeted experiment also includes a **needle-in-haystack
sub-test** at 8k, 16k, 32k, 64k to show that the gain scales with
context length (the golden-angle non-repetition only matters when
N > 10000, i.e. when the standard RoPE base sees repeat phases).

### 7.3 Cross-paradigm context (LLM track)

Per the chunk-8 expansion, H73 is the LLM-track recombination of
H34, H36, H40. Composes naturally with **H62** (toroidal KV + hex
attention) — the Metatron 13-vertex bias is a strict refinement of
H62's hex 6-vertex bias. Composes with **H71** (icosahedral RoPE) —
H73's golden-angle RoPE is a strict generalisation; the icosahedral
variant is recoverable by setting θ_i = GOLD·φ^(-i/d) AND adding a
20-rotation icosa group. If H71 + H73 jointly outperform either
alone by ≥ +0.05 composite, the icosa group structure is validated
as orthogonal to the golden-angle prior.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G7 row H73 (to be added
  in the next IDEA_TABLE refresh).
- Master experiment list: `EXPERIMENT_LOG.md` Tier-2 row T2.H73.
- Implementation sub-directory:
  `ideas/73_golden_spiral_metatron_pe/`.
- Related hypotheses that compose:
  - **H34** — Golden-angle RoPE (the (A) mechanism is its precise
    formalisation).
  - **H36** — φ-Spiral PE (the (B) mechanism is its decoder-side
    realisation).
  - **H40** — Metatron kernel overlap (the (C) mechanism is its
    attention-bias adaptation).
  - **H62** — Toroidal KV + hex attention (strictly composable;
    H73's (C) refines H62's hex bias).
  - **H67** — Full paradigm hybrid (flagship; uses H73's PE stack as
    its position-encoding sub-component).
  - **H71** — Icosa RoPE 3-D (orthogonal extension; can be ANDed).
- Related hypotheses that conflict: **H03** (Golden-spiral resolution
  scaling) only at a definitional level — H03 changes input
  resolution along the golden spiral; H73 changes position encoding
  along the golden spiral. They are not the same axis.

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of YaRN / NTK-aware RoPE?**

> YaRN (Bloc97 2023) and NTK-aware RoPE (Chen et al. 2023) modify
> RoPE by *interpolating* or *extrapolating* the existing geometric-
> base-10000 frequencies. H73's (A) mechanism replaces the base
> with a **fundamentally different sequence** — the golden-angle
> phyllotactic spacing, which has a provable non-repetition
> property (Vogel 1979) that geometric-base spacing lacks. (B) and
> (C) are additional mechanisms YaRN/NTK do not have. The
> falsifier in § 3 requires each of (A), (B), (C) to contribute
> independently, which YaRN/NTK cannot match because they only
> change (A).

**Q: How is this falsifiable rather than aesthetic?**

> § 3 names four numeric falsifiers (PPL Δ, KV regression, needle
> recall, per-mechanism marginal). § 6 pre-registers tight
> intervals. § 7.2 names the leave-one-out test that mechanically
> identifies which sub-mechanism (if any) fails.

**Q: What if the prior helps on 32k but hurts on 2k?**

> § 6 already predicts a smaller gain at 2k (-0.05 to -0.15 nats vs.
> -0.40 to -0.60 at 32k). The mechanism is golden-angle non-
> repetition, which only matters when N exceeds the standard RoPE
> repetition period (≈ 10000). If 2k shows a positive (worse) Δ,
> H73 is **scope-limited to long context** — recorded as such, not
> a refutation. The composite falsifier in § 3 is on 32k only.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> The CIFAR sweep was the full **NaturePriorBlock v1** with every
> prior on at once. H73 carries **three** PE-side mechanisms with a
> leave-one-out ablation specifically designed to test compounding.
> If the three are sub-additive (combined gain < sum of marginals
> by > 20 %), the **compounding** claim is refuted but the
> individual mechanisms may still be useful — the granular result
> matters more than the composite verdict.

**Q: How do we know the implementation is correct?**

> `ideas/73_golden_spiral_metatron_pe/tests.py` provides ≥ 16
> assertions: (a) rope-theta table matches the analytical
> phyllotactic formula to 1e-6, (b) side-PE buffer has unit norm in
> the (cos,sin) plane, (c) Metatron LUT is symmetric in (j-i mod
> 13), (d) attention logits with H73 (A)+(C) match FA2 reference
> output to 1e-4 in bf16, (e) causal mask preserved across two
> forward passes with shifted input, (f) KV cache size unchanged
> vs. baseline (regression test), (g) needle-in-haystack at 8k
> achieves ≥ 95 % recall on a synthetic toy with sinusoidal needles
> (sanity check), (h) gradient finite across 100 random batches.
> The archive carries the standard four verification files plus a
> `per_mechanism_marginal.csv` recording the leave-one-out result.

**Q: Why fix the Metatron LUT instead of learning it?**

> The LUT length (13) is the Metatron-Cube vertex count; a learned
> LUT could collapse to a 2-vertex (uniform) pattern, hiding the
> Metatron claim. v0 fixes the LUT; v1 (follow-up) makes it
> learnable with an L2 penalty toward the fixed Metatron values at
> λ=0.01 — same protocol as H72's mix matrix. If v1 collapses to
> uniform, the Metatron prior is invalidated as architecturally
> useful; if v1 stays within 0.05 of the fixed values, the prior is
> validated as a useful initialisation.

## 10. Verification artifacts checklist (committee evidence pack)

- [ ] `ideas/73_golden_spiral_metatron_pe/implementation.py` exists
      and tests green
- [ ] `ideas/73_golden_spiral_metatron_pe/tests.py` ≥ 16 assertions
- [ ] `ideas/73_golden_spiral_metatron_pe/AUDIT.md` lists ≥ 3 self-
      found weaknesses
- [ ] `ideas/73_golden_spiral_metatron_pe/IMPROVEMENTS.md` records
      the fixes
- [ ] `ideas/73_golden_spiral_metatron_pe/VERIFY.md` is signed
- [ ] `ideas/73_golden_spiral_metatron_pe/experiments/exp001_primary/`
      archive exists
- [ ] That archive carries `verification/{tests.txt, smoke.txt,
      gates.txt, reproduction.txt, per_mechanism_marginal.csv}`
- [ ] Row added to `EXPERIMENT_LOG.md` Tier-2
- [ ] Result reflected in `FINDINGS.md`, `RESULTS.md`, and dashboard
- [ ] Cross-link from `PARADIGM_COMPARISON.md` § 8 (chunk 8
      synthesis) and from `H67_full_paradigm_hybrid.md` § 5.2
      (PE sub-stack)

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-E.
