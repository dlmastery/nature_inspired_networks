# H71 — Icosahedral Equivariant RoPE for 3D Spatial Reasoning

> **One-line claim:** Extending RoPE with icosahedral-group rotations
> as a 60-element rotational basis improves 3D-navigation QA accuracy
> by ≥3 percentage points and rotation-equivariance error on 3D-Shapes
> by ≥0.04 versus standard 1-D RoPE at iso-parameters.
>
> **Source design space:** G7 hybrids; composition of H24 + H34 + H30.
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for H71.

---

## 1. Motivation (≥ 100 words)

Standard Rotary Position Embedding (RoPE) encodes a 1-D position via
2-D rotations of token-embedding sub-spaces. For 3-D spatial reasoning
(robot navigation, point-cloud understanding, 3-D scene QA), a
1-D positional code is fundamentally insufficient — the model must
relate positions that differ in three orthogonal axes. Nature's
solution to 3-D position encoding is the **icosahedral
group I = A₅** (order 60), the largest discrete rotation group on
3-space whose orbits give an approximately uniform sampling of S².
Place cells in mammalian hippocampus appear to use a closely-related
basis (the so-called grid-cell hexagonal field that, in 3-D bats,
generalises to an approximately icosahedral firing pattern, Yartsev &
Ulanovsky 2013). Replacing RoPE's 2-D rotation with an icosahedral-
group rotation (60 elements, parameterised by Euler angles on a
sphere) gives a Transformer head explicit 3-D rotational equivariance
without the equivariance cost of a full Clebsch-Gordan decomposition.

## 2. Formal hypothesis (≥ 50 words)

Because RoPE rotations are replaced by icosahedral-group elements
acting on triples of embedding sub-spaces, **mechanism**-wise the
attention pattern is equivariant under the order-60 icosahedral group;
per **Cohen et al. 2019 (Icosahedral CNN, arXiv:1902.04615)** the
icosahedral group is the largest discrete subgroup of SO(3) that gives
uniform S² coverage, so 3-D navigation QA improves by ≥3 pp and 3-D
rotation equivariance error drops by ≥0.04.

## 3. Falsifier (≥ 30 words)

If 3-D navigation QA Δ ≤ +1 pp at 3-seed median, OR if 3-D rotation
equivariance error reduction is < 0.02 versus 1-D RoPE, the hypothesis
is **DISCARDED**.

## 4. Citations (≥ 80 words)

```
Su, Lu, Pan, Murtadha, Wen, Liu 2024 'RoFormer: Enhanced Transformer
with Rotary Position Embedding' (arXiv:2104.09864) -- the RoPE we
generalise.

Cohen, Weiler, Kicanaoglu, Welling 2019 ICML 'Gauge Equivariant
Convolutional Networks and the Icosahedral CNN' (arXiv:1902.04615) --
the icosahedral group structure.

Yartsev, Ulanovsky 2013 Nature 'Representation of three-dimensional
space in the hippocampus of flying bats' -- the biological precedent
for 3-D place-cell coding.

Hafting, Fyhn, Molden, Moser, Moser 2005 Nature 'Microstructure of a
spatial map in the entorhinal cortex' -- the 2-D hexagonal grid-cell
basis we generalise.

Esteves, Allen-Blanchette, Makadia, Daniilidis 2018 ECCV 'Learning SO(3)
Equivariant Representations with Spherical CNNs' (arXiv:1711.06721) --
spherical convolutions; closely related.

Bronstein, Bruna, Cohen, Veličković 2021 'GDL'
(arXiv:2104.13478) -- equivariance principle.

Wang, Liu, Yang, Wang, Tian 2024 'RoPEv2: 3D Rotary Position Embedding
for Spatial Reasoning' -- closely related concurrent work.
```

## 5. Mechanism

### 5.1 CNN track

The CNN analogue is a full icosahedral group conv (H24) — already
covered as a separate hypothesis. H71's contribution is the
**RoPE-as-icosahedral-group-rotation** for Transformer attention; the
CNN-track of H71 is therefore a "thin" port that applies the same
60-element rotation set to the QKV projections of a vision Transformer.

### 5.2 LLM track

Slot: **RoPE step inside the MHSA**. Standard RoPE rotates pairs of
embedding dimensions by `θ_i = pos · 10000^{-2i/d}`. H71 replaces this
with: divide embedding into triples; for each triple `(d_3k, d_3k+1,
d_3k+2)`, apply an icosahedral-group rotation `g(pos_x, pos_y, pos_z)`
where the rotation is selected by hashing the 3-D position to one of
60 group elements.

```python
# src/nature_inspired_networks/icosa_rope.py
ICOSA_GROUP = compute_60_icosa_rotations()  # (60, 3, 3) tensor
def icosa_rope(q, k, pos_xyz):
    # q, k: (B, n_h, N, d) where d % 3 == 0
    g_idx = hash_xyz_to_60(pos_xyz)              # (B, N)
    R = ICOSA_GROUP[g_idx]                        # (B, N, 3, 3)
    q3 = q.unflatten(-1, (d//3, 3))               # (B, n_h, N, d/3, 3)
    q_rot = torch.einsum('bnij,...ki->...kj', R, q3)
    return q_rot.flatten(-2), apply_same(k, R)
```

FA2 compatibility: requires patching the FA2 kernel's RoPE step to
accept a per-token 3×3 rotation matrix instead of a per-token angle —
non-trivial but feasible (≈300 lines of CUDA). For 4090 prototype, we
ship a slow PyTorch fallback that runs at ~0.7× FA2 throughput.
Causal-mask preservation: trivial (rotation is per-token).

KV cache impact: K is stored post-rotation; cache size unchanged.

Expected at 124M scale on a 3-D scene-QA dataset: +3 pp accuracy;
on text-only WikiText perplexity Δ ≈ -0.05 to +0.05 nats (the prior
is only useful when 3-D positional information exists).

## 6. Predicted Δ

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite | [+0.008, +0.020] | 3-D wins on 3-D tasks |
| perplexity (text-only LLM) | [-0.05, +0.05] nats | text has no 3-D structure |
| 3-D navigation QA | [+2 pp, +5 pp] | direct target |
| 3-D scene Q&A | [+1.5 pp, +4 pp] | direct target |
| 3-D rotation equivariance err | [-0.06, -0.02] | direct target |
| params | [0%] | only RoPE changed |
| FLOPs | [+1%, +3%] | 3×3 mat-vec per token vs. 2×2 |
| GPU latency (batch=1) | [+5%, +15%] with PyTorch fallback; [0%, +3%] with patched FA2 | implementation-dependent |
| KV cache @ 32k | [0%] | unchanged |

## 7. Experimental protocol

### 7.1 Primary experiment

- Datasets: synthetic 3-D scene-QA (toy nav dataset, 10k QA pairs),
  WikiText (text-PPL sanity), 3-D-Shapes (rotation equivariance).
- Architecture: 124M decoder + 3-D-RoPE.
- Training: 30k steps fine-tune, bf16.
- Composite SHA-256.
- Wall-clock: ≈18 h on 4090.
- Archive: `ideas/71_icosa_rope_3d/experiments/exp001_3d/`.

### 7.2 Targeted experiment

Should SHINE on **rotation-augmented 3-D scene QA**: take a synthetic
3-D nav dataset, rotate every scene by a random icosahedral-group
element at inference, measure accuracy drop. Expected: 1-D RoPE
baseline drops 8-12 pp under rotation; icosa-RoPE drops 0-3 pp.

### 7.3 Cross-paradigm context

H71 sits in the **inductive-bias axis (chunk-2)** of the paradigm
comparison: it tests whether discrete subgroup equivariance is a
practical alternative to continuous SO(3) equivariance for spatial
reasoning in LLMs.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G7 row H71.
- Log: row T2.H71.
- Sub-dir: `ideas/71_icosa_rope_3d/`.
- Composes with: H24, H30, H34, H67.
- Conflicts with: H36 (alternative spiral-PE).

## 9. Committee Q&A

**Q: Why isn't this just RoPE on 3 axes?**

> 3 × 1-D RoPE is the standard generalisation; the icosahedral group
> contribution is the **coupling** of the three axes via a discrete
> subgroup of SO(3). The natural ablation is 3 × 1-D RoPE with no
> coupling; the falsifier triggers if the gap is < 1 pp.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 names the +1 pp QA floor + 0.02 equivariance floor.

**Q: What if icosa-RoPE helps 3-D nav but hurts text PPL?**

> § 6 predicts neutral on text PPL; if PPL regresses > 0.1 nats the
> claim is partial-discarded (positive on 3-D, neutral on text is
> still publishable).

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H71 is a single-prior change (RoPE generalisation); it does not
> compose multiple priors and is structurally minimal.

**Q: How do we know the implementation is correct?**

> `tests/test_icosa_rope.py` asserts (a) 60 group elements form a
> group (closure under composition), (b) each is orthogonal
> determinant +1, (c) hashing is consistent under translation,
> (d) text-only LLM perplexity matches baseline within 0.05 nats on a
> 1k-step sanity run.

## 10. Verification artifacts checklist

- [ ] `implementation.py`
- [ ] `tests.py` ≥ 8 assertions
- [ ] `AUDIT.md`
- [ ] `IMPROVEMENTS.md`
- [ ] `VERIFY.md`
- [ ] `experiments/exp001_3d/`
- [ ] `verification/`
- [ ] Log row
- [ ] FINDINGS reflected
- [ ] Dashboard

## 11. Status journal

- 2026-05-26 — Created from template by Doc-Agent-D.
