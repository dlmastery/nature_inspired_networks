# H82 — Voronoi Sparse Attention

> **One-line claim:** A fixed Delaunay-adjacency sparse attention mask
> recovers most dense-attention quality at planar (≈6N) edge cost.
>
> **Source design space:** G8 Esoteric Extensions.
>
> **Implementation status (this repo):** `✓ done` (primitive + tests).

---

## 1. Motivation (≥ 100 words)

Voronoi tessellations are nature's space-partitioning default: the
territories of cells in epithelial tissue, the polygonal cracking of
drying mud, the interlocking irregular stones of Inca "polygonal masonry",
and the columnar basalt of the Giant's Causeway all approximate Voronoi
cells, because partitioning a plane by nearest-seed minimises boundary
energy. The dual of a Voronoi diagram is the Delaunay triangulation, which
connects cells that share a boundary — a planar graph whose mean vertex
degree is ≈6 regardless of N. Using Delaunay adjacency as an attention
mask gives an irregular, locally-adaptive sparse connectivity: dense
clusters of seed points form high-degree hubs, sparse regions stay
low-degree, unlike a uniform sliding window. The esoteric "polygonal
masonry" interlocking-topology motif is the mystical motivation; the
implementation is a standard seeded Delaunay-adjacency sparse attention
mask, with a kNN proxy when SciPy is unavailable.

## 2. Formal hypothesis (≥ 50 words)

Because attention is restricted to Delaunay-adjacent token pairs (plus
self), each token attends to O(1) neighbours instead of O(N), reducing the
effective attention cost from O(N²) toward O(6N); mechanism-wise the
planar tessellation preserves local connectivity while pruning long-range
redundancy. Per Child 2019, fixed sparse masks retain most dense-attention
accuracy at far lower cost, so we expect ≤1 pp accuracy loss at
substantially reduced attention density.

## 3. Falsifier (≥ 30 words)

If the Voronoi-masked attention loses > 2 pp top-1 versus dense attention
on a patch-token ViT-Tiny CIFAR-10 run at matched compute, while the mask
density is below 20% of N², the sparsity is too aggressive and the
hypothesis is DISCARDED.

## 4. Citations (≥ 80 words)

Child, R., Gray, S., Radford, A., Sutskever, I. 2019 'Generating Long
Sequences with Sparse Transformers' (arXiv:1904.10509) — shows fixed
sparse attention patterns recover most dense-model quality at sub-
quadratic cost, the core justification for replacing dense attention with
a Delaunay mask. Beltagy, I., Peters, M. E., Cohan, A. 2020 'Longformer:
The Long-Document Transformer' (arXiv:2004.05150) — local+global sparse
attention; the irregular Voronoi mask generalises its fixed local window
to a data-agnostic but spatially-irregular planar neighbourhood with
self-loops, retaining locality without a fixed band width.

## 5. Mechanism

### 5.1 CNN / vision track
`VoronoiSparseAttention(d_model, n_tokens, n_heads, seed)` builds a
seeded `(N,N)` Delaunay-adjacency buffer once; per forward, disallowed
pairs get `-inf` before softmax. Mask density ≈6N/N² (tested < 20% at
N=64). Replaces a ViT block's dense MHA; FLOPs are still computed densely
in this reference (the mask is applied, not fused), so the win is
quality-at-fixed-pattern, not yet wall-clock — a fused kernel is future
work. Lives at `src/nature_inspired_networks/voronoi_attention.py`.

### 5.2 LLM track (decoder-only)
The Delaunay mask can be intersected with the causal mask (keep only
`j ≤ i` Delaunay edges) to give a causal sparse pattern; KV cache holds
only retained keys per query in a fused implementation. Compatible with
block-sparse FlashAttention given a static mask.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. dense | rationale |
|---|---|---|
| composite | [−0.01, +0.003] | sparsity may cost a little on small N |
| top-1 (CNN) | [−2.0, +0.2] pp | locality retained, long-range pruned |
| params | [0, 0] | mask is a buffer |
| attention density | [−85%, −80%] | ≈6N of N² edges |
| FLOPs (fused, future) | [−70%, −85%] | only with a block-sparse kernel |

## 7. Experimental protocol

### 7.1 Primary
ViT-Tiny on CIFAR-10, dense MHA → VoronoiSparseAttention (n_tokens =
num_patches+1), 12-epoch smoke, bf16, batch 256, seeds 0/1/2. Archive:
`ideas/82_voronoi_attention/experiments/exp001_smoke/`.

### 7.2 Targeted (where it should shine)
Long-sequence (N≥1024) tasks where dense O(N²) is prohibitive and a
planar O(6N) mask is the only feasible option — the sparsity is the
point, not a handicap.

### 7.3 Cross-paradigm (LLM)
Causal-intersected Voronoi mask on a 124M decoder at 4k context; measure
perplexity and memory vs. dense.

## 8. Cross-references

- Parent: G8 esoteric extensions.
- Sibling sparse/graph: `src/nature_inspired_networks/sparse.py`,
  `small_world.py` (H29), `fibottention.py`.
- Composes with: H77 (12-fold bias on the retained edges).

## 9. Committee Q&A

**Q: Why isn't this just Longformer/BigBird?**
> Those use a fixed band + global tokens; the Voronoi mask is an
> irregular, locally-adaptive planar neighbourhood (degree varies by seed
> density), a different and data-agnostic sparsity prior.

**Q: How is this falsifiable?**
> §3: > 2 pp loss vs. dense at < 20% density discards it.

**Q: The reference isn't actually faster — why claim sparsity?**
> Honest scope: the mask is applied, not fused, so the present win is
> quality-at-pattern; wall-clock requires a block-sparse kernel (§6 marks
> the FLOPs row "future").

**Q: Priors don't compound — why bother?**
> Single-prior unit, measured against dense attention, not in a hybrid.

**Q: How do we know it is correct?**
> `tests/test_voronoi_attention.py` (5 tests): `(N,N)` symmetric mask with
> self-loops, sparsity < 20% of N² with ≥1 neighbour each, masked forward
> shape preserved + finite, seed determinism, and the SciPy-free kNN
> fallback path for small N.

## 10. Verification checklist

- [x] Primitive `voronoi_attention.py` exists, tests green (5/5).
- [x] Adjacency symmetric with self-loops (tested).
- [x] Sparsity far below N² (tested).
- [x] Seed-deterministic mask (tested).
- [x] SciPy import guarded with kNN fallback (tested).
- [ ] Experiment archive (deferred — attention module, no sweep row).

## 11. Status journal

- 2026-05-27 — Created; primitive + 5 unit tests green.
