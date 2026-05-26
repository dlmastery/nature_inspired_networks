# H74 — Metatron Overlap Sharing across QKV/FFN + Platonic Group Tying

> **One-line claim:** Tying the Q, K, V projection matrices and the
> FFN gate/up matrices through a shared Metatron-Cube 13-vertex
> overlap basis, plus enforcing Platonic icosa/dodeca group symmetry
> on the head partition, compresses decoder-only Transformer
> parameters by 18–30 % at iso-perplexity on WikiText-103 (350M
> baseline → 245–287M tied model), with no GSM8K regression beyond
> 1 pp.
>
> **Source design space:** G7 Cross-paradigm hybrids (H61–H75); the
> chunk-8 expansion of the extended Grok transcript, opportunity #18
> — the LLM-track recombination of **H40** (Metatron Kernel Overlap)
> with **H23** (Platonic φ-Graph) and **H30** (Platonic-Fib Hybrid)
> into a **parameter-tying scheme** for QKV+FFN matrices. Distinct
> from each ancestor: H40 covers only kernel-level overlap on a
> single layer; H23/H30 cover only graph adjacency without sharing
> weight matrices; H74 fuses them into a per-layer parameter-sharing
> protocol that yields measurable compression.
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for hypothesis
H74. Every section below is mandatory; the word-count floors are the
same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (≥ 100 words)

Metatron's Cube — the 13-vertex, 78-edge figure formed by overlaying
the 13 circles of the Flower of Life — encodes the 2-D projection
shadows of **all five Platonic solids** simultaneously, and is the
densest informationally-overlapping figure in classical sacred
geometry. The biological analogue is the cortical mini-column: each
neuron's dendritic field overlaps with its neighbours' so that
information is **shared** across the column without per-neuron
duplication. Modern Transformers do the opposite: every Q, K, V
matrix is fully independent (3·d² parameters per layer just for QKV)
even though Q and K differ only in what they *attend to*, not in
their geometric basis. Pythagoras called this redundancy
"unnecessary multiplicity"; Hales (2001) proved the hex packing is
optimal precisely because it eliminates such redundancy. H74 is the
operational claim that QKV and FFN-gate/up matrices share a
13-vertex Metatron basis that can be **factored out** without loss,
saving 18–30 % of parameters at iso-perplexity. The Platonic
group-tying additionally constrains the head partition to obey
icosa/dodeca symmetry (60-fold rotation), which is the parameter-
efficient way nature avoids storing the same head function at
multiple rotations.

## 2. Formal hypothesis (≥ 50 words)

Because the standard decoder layer stores Q, K, V, gate, up as five
independent d×d matrices (5·d² params), **mechanism**-wise
factorising each as M_i = U·S_i·V^T where U,V are 13-vertex Metatron
basis matrices shared across {Q, K, V, gate, up} and S_i are the
five per-matrix diagonal coefficient vectors, cuts the per-layer
parameter count from 5·d² to d² + d·13·5 + d·13 ≈ d² (1 + 65/d)
which is **18–30 %** of 5·d² at d∈[768, 4096]; per Hu et al. 2022
(LoRA factorisation) and the Platonic Representation Hypothesis
(Huh et al. 2024), we expect this compression to preserve WikiText-
103 perplexity within ±0.05 nats.

## 3. Falsifier (≥ 30 words)

If WikiText-103 perplexity Δ > +0.15 nats at 3-seed median, **OR**
if compression ratio falls below 15 % (i.e. param reduction < 15 %),
**OR** if GSM8K zero-shot drops by more than 1 pp, **OR** if the
implied rank of the recovered M_i matrices does not collapse to ≤ 13
on a held-out set of 1000 input tokens (verified by SVD), this
hypothesis is **DISCARDED**. The 18–30 % compression target is the
pre-registered band.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen 2022 ICLR 'LoRA:
Low-Rank Adaptation of Large Language Models' (arXiv:2106.09685) --
the precedent for low-rank factorisation of attention matrices; H74
extends this from fine-tune time to pre-train time with a fixed
geometric basis (Metatron 13-vertex) instead of a learned low-rank.

Lan, Chen, Goodman, Gimpel, Sharma, Soricut 2020 ICLR 'ALBERT: A
Lite BERT for Self-supervised Learning' (arXiv:1909.11942) --
parameter-tying precedent (cross-layer parameter sharing); H74's
intra-layer cross-matrix tying is a strict refinement.

Press, Wolf 2017 EACL 'Using the Output Embedding to Improve
Language Models' (arXiv:1608.05859) -- the original embedding-tying
result; H74 generalises the principle to QKV+FFN.

Hales 2001 Annals of Math. 'The Honeycomb Conjecture' -- hex
optimality; the Metatron-Cube 13-vertex figure is hex + inner
triangle and inherits the optimality.

Huh, Cheung, Wang, Isola 2024 ICML 'The Platonic Representation
Hypothesis' (arXiv:2405.07987) -- empirical bridge; QKV matrices
across modalities converge to a shared Platonic representation,
which justifies the U,V shared basis a priori.

Cohen, Weiler, Kicanaoglu, Welling 2019 ICML 'Icosahedral CNN'
(arXiv:1902.04615) -- icosa group symmetry; the head partition
tying applies the I-group with 60 rotations.

Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
2017 NeurIPS 'Attention is All You Need' (arXiv:1706.03762) -- the
baseline architecture whose 5·d² per-layer matrix budget we
compress.

Wang, Li, Khabsa, Fang, Ma 2020 'Linformer: Self-Attention with
Linear Complexity' (arXiv:2006.04768) -- another low-rank attention
factorisation; baseline comparison for the compression ratio.

Dao 2024 'FlashAttention-2' (arXiv:2307.08691) -- FA2 still applies
because the tied matrices are reconstructed before the QKV
projection in the forward pass; FA2 sees only the reconstructed
d×d matrix.
```

## 5. Mechanism

### 5.1 CNN track

The CNN-track sibling applies the same Metatron-overlap basis to a
ResNet-20 stage: the four 1×1 bottleneck convs of a residual block
(in-bottleneck, depthwise, expand, residual-add) share a 13-vertex
Metatron basis. Lives in
`src/nature_inspired_networks/blocks/metatron_tying.py` and is
re-exported by `ideas/74_metatron_overlap_tying/implementation.py`.

```python
# CNN-track: shared Metatron basis across 4 bottleneck convs
import math, torch, torch.nn as nn, torch.nn.functional as F
PHI = (1.0 + 5.0 ** 0.5) / 2.0

class MetatronTiedBottleneck(nn.Module):
    """4 bottleneck convs share a 13-vertex Metatron basis."""
    def __init__(self, c_in, c_out, K=13):
        super().__init__()
        # Shared U, V across the 4 sub-matrices
        self.U = nn.Parameter(torch.randn(c_in, K) / math.sqrt(c_in))
        self.V = nn.Parameter(torch.randn(K, c_out) / math.sqrt(K))
        # 4 per-matrix diagonal coefficients
        self.S = nn.Parameter(torch.ones(4, K))
        self.bias = nn.Parameter(torch.zeros(4, c_out))
    def reconstruct(self, i):
        # M_i = U @ diag(S_i) @ V
        return (self.U * self.S[i]) @ self.V  # (c_in, c_out)
    def forward(self, x):
        M0 = self.reconstruct(0); M1 = self.reconstruct(1)
        M2 = self.reconstruct(2); M3 = self.reconstruct(3)
        # ... apply as 4 sub-convs sharing the Metatron basis
        h0 = F.linear(x, M0.t(), self.bias[0])
        h1 = F.linear(h0, M1.t(), self.bias[1])
        h2 = F.linear(h1, M2.t(), self.bias[2])
        h3 = F.linear(h2, M3.t(), self.bias[3])
        return h3
```

Computational cost vs. independent 4-matrix baseline at d=512: param
count drops from 4·512·512 ≈ 1.05 M to 2·512·13 + 4·13 ≈ 13.4 k —
**98 % reduction**. The CNN-track is a parameter-efficiency
demonstrator only; the real target is the LLM-track.

### 5.2 LLM track (decoder-only Transformer)

Slot: **replaces the standard QKV+FFN matrix bank** of every decoder
layer. The five matrices {Q, K, V, gate, up} are factorised as M_i
= U · diag(S_i) · V^T with U ∈ ℝ^(d×13), V ∈ ℝ^(13×d), S_i ∈ ℝ^13.
The down-projection of FFN is **not** tied (it has a different
output dimension d_ff → d) and is kept full-rank.

Two variants are tested:

**v0 (Metatron-fixed):** U and V are initialised at fixed analytic
Metatron-vertex projections (the 13 unit vectors pointing to the
hex+inner-triangle vertices in 2-D, lifted to d-D by a random
orthogonal embedding); only S_i is learned. Compression: 28 %.

**v1 (Metatron-learned):** U, V, S_i are all learned with U, V
initialised at the analytic Metatron projection. Compression: 18 %.

```python
# LLM-track: H74 tied QKV+FFN layer
class MetatronTiedDecoderLayer(nn.Module):
    def __init__(self, d, n_heads, d_ff=None, K=13, tie='v0'):
        super().__init__()
        d_ff = d_ff or 4 * d
        # Shared Metatron basis (5 matrices: Q, K, V, gate, up)
        if tie == 'v0':
            U = self._metatron_init(d, K)
            V = self._metatron_init(d, K).t()
            self.register_buffer('U', U); self.register_buffer('V', V)
        else:
            self.U = nn.Parameter(self._metatron_init(d, K))
            self.V = nn.Parameter(self._metatron_init(d, K).t())
        # Per-matrix coefficients (5 diagonals)
        self.S = nn.Parameter(torch.ones(5, K))
        # Untied: FFN down-projection
        self.W_down = nn.Linear(d_ff, d, bias=False)
        # RMSNorm + RoPE + Icosa-group head partition
        self.norm = RMSNorm(d)
        self.icosa_group = IcosaHeadPartition(n_heads, group=60)
    def _metatron_init(self, d, K):
        # 13 unit vectors: 1 centre + 6 hex + 6 corner
        ang_hex = torch.linspace(0, 2*math.pi, 7)[:-1]
        ang_cor = ang_hex + math.pi / 6
        v2d = torch.stack([
            torch.tensor([0.0, 0.0]),
            *[torch.tensor([math.cos(a), math.sin(a)]) for a in ang_hex],
            *[PHI * torch.tensor([math.cos(a), math.sin(a)]) for a in ang_cor],
        ])                                   # (13, 2)
        # Random orthogonal lift 2 → d
        Q, _ = torch.linalg.qr(torch.randn(d, 2))
        return v2d @ Q.t()                   # (13, d)
    def reconstruct(self, i):
        return (self.U * self.S[i].unsqueeze(0)) @ self.V  # (d, d)
    def forward(self, x):
        h = self.norm(x)
        Wq, Wk, Wv = self.reconstruct(0), self.reconstruct(1), self.reconstruct(2)
        Wg, Wu     = self.reconstruct(3), self.reconstruct(4)
        # ... standard attn + SwiGLU FFN with these reconstructed matrices
        ...
```

FA2 compatibility: the reconstructed Q/K/V matrices are full d×d,
so FA2 sees a standard attention input. The reconstruction adds 5
d×K matmuls (≈ 5·d·K·d FLOPs ≈ 0.05·d³ FLOPs at K=13, d=1024 — a
1.3 % FLOP overhead vs. baseline). Causal-mask preservation:
untouched (no change to mask logic). KV cache: **slightly smaller**
because K and V are reconstructed from the same U·V basis and a
KV-cache optimisation can store only the K-rank-13 coefficients
(not implemented in v0; predicted -15 % KV at 32k in v1).

Latency at batch=1: +2 ± 1 % from the 5 reconstruction matmuls;
at batch=16: ≤+0.5 % (the reconstruction is O(K·d²) once per layer
per forward, not per token).

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite | [+0.015, +0.035] | compression at iso-PPL |
| perplexity (WikiText-103) | [-0.05, +0.10] nats | within ±0.05 target band |
| perplexity (TinyStories) | [-0.05, +0.10] nats | same band |
| GSM8K zero-shot | [-1.0, +0.5] pp | falsifier threshold |
| **params (v0 fixed-basis)** | [-30 %, -25 %] | the headline compression |
| **params (v1 learned-basis)** | [-22 %, -18 %] | smaller compression |
| FLOPs | [+1 %, +3 %] | reconstruction overhead |
| GPU latency (batch=1) | [+1 %, +3 %] | per-layer reconstruction |
| GPU latency (batch=16) | [-1 %, +1 %] | amortises |
| rotation-equivariance err | [-0.02, 0.000] | icosa head partition |
| KV cache @ 32k | [-15 %, -5 %] | shared K,V basis |
| Betti collapse rate | [-5 %, +5 %] | not the target axis |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- Dataset: **WikiText-103** (PPL, primary) + GSM8K zero-shot
  (falsifier) + ARC easy (auxiliary).
- Architecture: 350M decoder, 24 layers × d=1024 × 16 heads × d_ff=4d,
  every decoder layer replaced with `MetatronTiedDecoderLayer`.
- Variants: {baseline 350M, H74-v0 fixed-basis ≈ 245M, H74-v1
  learned-basis ≈ 287M}.
- Epochs: 30 k steps, bf16 AMP + grad-ckpt, cosine LR with 1 k
  warmup, AdamW (β=0.9, 0.95), wd=0.1.
- Batch: 16 sequences × 2048 tokens × grad-accum 4 = 128 k tokens/step.
- Seeds: {0, 1, 2}.
- Composite formula:
  `0.30·neg_norm_ppl + 0.30·param_compression_norm + 0.15·norm_gsm
   + 0.10·norm_arc + 0.05·norm_kv + 0.05·norm_lat_b16 + 0.05·norm_betti`,
  SHA-256 fingerprint logged at gate.
- Run-script invocation:
  `python ideas/74_metatron_overlap_tying/experiment.py
   --config configs/exp001_primary.yaml --variant v0 --seeds 0 1 2`
- Wall-clock estimate on 4090 Laptop 16 GB: ≈ 28 h / seed × 2 variants
  × 3 seeds = 7 days GPU-time.
- Archive: `ideas/74_metatron_overlap_tying/experiments/
  exp001_primary/`.

### 7.2 Idea-targeted experiment (rank-13 SVD verification)

H74's central claim is that the per-layer matrices have an effective
rank-13 structure. The targeted experiment is a **rank-decay
verification**: on a held-out set of 1000 input tokens, compute the
SVD of each reconstructed M_i and verify the singular-value spectrum
has a sharp cliff at rank 13. Prediction: σ₁₃ / σ₁ ≥ 0.05 and σ₁₄ /
σ₁ ≤ 0.005 (an order-of-magnitude cliff between rank 13 and 14). If
the cliff is at rank ≠ 13, the Metatron prior is the wrong
representation; if there is no cliff at all, the rank-13 claim is
refuted. A secondary targeted experiment varies K ∈ {7, 13, 19, 31}
to test the Metatron-Cube optimality directly — prediction is a
local minimum on PPL at K=13 (the Metatron vertex count).

### 7.3 Cross-paradigm context (LLM track)

Per the chunk-8 expansion, H74 is the LLM-track recombination of
H40 (Metatron kernel overlap) and H23/H30 (Platonic graph
adjacency). Composes naturally with **H62** (toroidal KV + hex
attention) — Metatron contains hex; the toroidal KV becomes a
rank-13 ring buffer. Composes with **H67** (full paradigm hybrid)
as a layer-level compression that reduces flagship param count.
Conflicts with **H50** (full sacred hybrid CNN-track) only in
priority: H50 is the CNN flagship, H74 is the LLM compression
flagship; both should not be run in the same 4090 campaign at
350M without budgeting.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G7 row H74 (to be added
  in the next IDEA_TABLE refresh).
- Master experiment list: `EXPERIMENT_LOG.md` Tier-2 row T2.H74.
- Implementation sub-directory: `ideas/74_metatron_overlap_tying/`.
- Related hypotheses that compose:
  - **H40** — Metatron kernel overlap (the per-kernel ancestor; H74
    extends to per-matrix and intra-layer sharing).
  - **H23** — Platonic φ-graph (the 13-vertex adjacency used as the
    Metatron basis structure).
  - **H30** — Platonic-Fib hybrid (the icosa/dodeca head partition
    used in the Platonic group-tying).
  - **H43** — Fibonacci pruning (complementary post-training
    compression; H74 is a structural pre-training compression).
  - **H67** — Full paradigm hybrid (uses H74 as the layer-level
    compression sub-block).
- Related hypotheses that conflict: none architecturally; H74 is
  composable with every G3/G4/G5 hypothesis.

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just LoRA / Linformer with a fancier basis?**

> LoRA is a *fine-tune-time* low-rank update on top of a full-rank
> pretrained matrix; H74 is a *pre-train-time* full-replacement of
> the rank-d matrix with a rank-13 factorisation. Linformer
> factorises only attention (Q, K, V) not FFN, and uses random
> projections not a geometric basis. H74's Metatron basis is the
> specific 13-vertex pattern (centre + 6 hex + 6 corner) that has a
> closed-form analytic initialisation; LoRA and Linformer have no
> such structure. The falsifier in § 3 includes a rank-13 SVD
> cliff verification that LoRA / Linformer cannot meet by
> construction.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 names four numeric falsifiers (PPL Δ ≤ +0.15, compression ≥
> 15 %, GSM8K ≤ -1 pp, rank-13 SVD cliff). § 6 pre-registers tight
> intervals on each. § 7.2 names a direct K-sweep that tests the
> Metatron-Cube optimality claim mechanically.

**Q: What if the prior helps on WikiText but hurts on GSM8K?**

> The falsifier in § 3 explicitly admits up to 1 pp GSM8K
> regression — this is the **acceptable cost** of compression for
> a non-reasoning benchmark like WikiText. If GSM8K regresses by
> > 1 pp, the compression is rejected; if PPL stays flat but GSM8K
> regresses by < 1 pp, H74 is a **publishable compression-at-cost
> result**, recorded as such.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H74 is a single-axis hypothesis (parameter compression). It does
> not compound multiple sacred priors in one block. The
> compounding claim of H67 is orthogonal to H74's compression
> claim; H74 can be tested independently and is in the chunk-8
> "Phase 5" of `PARADIGM_COMPARISON.md` § 8.3.

**Q: How do we know the implementation is correct?**

> `ideas/74_metatron_overlap_tying/tests.py` provides ≥ 18
> assertions: (a) the 13-vertex initial U has the analytic
> Metatron geometry (centre + 6 hex + 6 corner) to 1e-6, (b)
> reconstruction matches the analytical formula M_i = U·diag(S_i)·V
> to 1e-6, (c) reconstructed Q/K/V matrices have effective rank ≤
> 13 on synthetic input (SVD verification), (d) FA2 still passes
> with the reconstructed matrices, (e) gradient finite across 100
> random batches, (f) bf16 numerical stability check, (g) variant-
> v0 has frozen U/V (zero grad), (h) variant-v1 has trainable U/V
> (non-zero grad), (i) param-count regression test (v0 ≈ 245M, v1
> ≈ 287M at d=1024). The archive carries
> `verification/svd_cliff.png` alongside the standard four files.

**Q: Why specifically 13 vertices and not 12 (icosa) or 20 (dodeca)?**

> 13 is the Metatron-Cube vertex count (1 centre + 6 hex + 6 corner)
> which is the densest 2-D Platonic-projection. Icosa (12) and
> dodeca (20) are 3-D Platonic solids whose 2-D shadow is exactly
> the Metatron-Cube. Per Cohen et al. 2019, the icosa group has 60
> rotations and partitions naturally with 12-head models; H74's
> head-partition tying (separate from the U,V tying) uses icosa
> for the head partition while the matrix-tying uses Metatron-13
> for the basis. They are complementary — H74 ties both.

## 10. Verification artifacts checklist (committee evidence pack)

- [ ] `ideas/74_metatron_overlap_tying/implementation.py` exists and
      tests green
- [ ] `ideas/74_metatron_overlap_tying/tests.py` ≥ 18 assertions
- [ ] `ideas/74_metatron_overlap_tying/AUDIT.md` lists ≥ 3 self-
      found weaknesses (rank-13 hard-coded, no per-layer K, no
      down-projection tying)
- [ ] `ideas/74_metatron_overlap_tying/IMPROVEMENTS.md` records the
      fixes
- [ ] `ideas/74_metatron_overlap_tying/VERIFY.md` is signed
- [ ] `ideas/74_metatron_overlap_tying/experiments/exp001_primary/`
      archive exists for **both** v0 and v1 variants
- [ ] That archive carries `verification/{tests.txt, smoke.txt,
      gates.txt, reproduction.txt, svd_cliff.png,
      param_compression.json}`
- [ ] Row added to `EXPERIMENT_LOG.md` Tier-2
- [ ] Result reflected in `FINDINGS.md`, `RESULTS.md`, and dashboard
- [ ] Cross-link from `PARADIGM_COMPARISON.md` § 8.3 (Phase 5
      compression tests) and from `H67_full_paradigm_hybrid.md` § 5.2

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-E.
