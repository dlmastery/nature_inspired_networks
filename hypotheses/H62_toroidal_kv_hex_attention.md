# H62 — Toroidal KV-Cache + Hex Attention Graph + Fib Pruning

> **One-line claim:** Replacing the dense N² attention graph with a
> hex-lattice φ-edge-weighted graph whose KV is wrapped on a toroidal
> manifold and pruned by Fibonacci-ratio magnitude cuts WikiText-103
> 128k-context KV memory by ≥40% with perplexity Δ ≤ +0.2 nats.
>
> **Source design space:** G7 Cross-paradigm hybrids; extends H21 + H22 +
> H43 + H28 (chunk-6 efficiency synthesis).
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for H62.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

Hexagons are the unique 2-D lattice that simultaneously (i) maximises
nearest-neighbour count per unit area (six contacts vs. four for a
square), (ii) minimises per-area perimeter (the honeycomb conjecture,
proved by Hales 2001), and (iii) tessellates with the smallest number
of distinct tile shapes — one. Bees, fruit-fly retinas, and basalt
columns all converge on the same lattice because it is the
information-theoretically optimal packing for dense local interaction.
Long-context LLMs face the same packing problem: an N² attention matrix
is dense, and the dominant memory cost at long context is the KV cache.
Wrapping the cache on a torus exploits the fact that **positional
modular arithmetic is exactly the symmetry group of a torus**, so
rotational shifts within the cache become free; Fibonacci-ratio pruning
respects the natural-system observation that biological synapses are
sparsified in coarse-to-fine cycles (≈8%, 13%, 21%, 34%, 55%). The
three priors compose: a hex graph is sparser than dense attention, a
torus is the natural wrap manifold for that hex graph, and Fib pruning
is the natural sparsification schedule.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because hex-lattice attention reduces interactions per token from O(N)
to O(7) per hop, **mechanism**-wise the KV graph is 6-regular and
multi-hop reachability covers the full N tokens in O(log_φ N) hops; per
**Hoogeboom et al. 2018 (HexaConv, arXiv:1803.02108)** the hex lattice
preserves 6-fold rotational equivariance and per **Liquid AI 2025
(LFM2, arXiv:2511.23404)** wrapping the cache on a torus permits
modular-shift O(1) updates, so 128k-context KV memory drops by ≥40%
with perplexity Δ ≤ +0.2 nats.

## 3. Falsifier (≥ 30 words)

If 128k-context KV-cache memory reduction is < 25% OR perplexity
regression exceeds +0.4 nats at 3-seed median on WikiText-103, this
hypothesis is **DISCARDED**. Composite Δ ≤ -0.005 also discards.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Hoogeboom, Peters, Cohen, Welling 2018 ECCV 'HexaConv'
(arXiv:1803.02108) -- canonical hex-lattice convolution; we adopt its
discrete-Fourier formulation on the hex grid for the attention graph.

Pittorino, Ferraro, Perugini, Feinauer, Baldassi, Borghesi, Cecchini,
Zecchina 2022 Phys. Rev. E 'Deep networks on toroids: removing
symmetries reveals the structure of flat regions in the landscape'
(arXiv:2202.03038) -- justification for wrapping a learned manifold on
a torus to exploit translation invariance.

Han, Mao, Dally 2016 ICLR 'Deep Compression: Pruning, Trained
Quantization and Huffman Coding' (arXiv:1510.00149) -- the magnitude-
pruning protocol whose iterative schedule we set to Fibonacci ratios.

Liquid AI 2025 arXiv 'LFM2' (arXiv:2511.23404) -- 192 MB KV at 32k as
our LLM baseline; we beat it on KV memory at 128k.

Hales 2001 Annals of Math. 'The Honeycomb Conjecture' -- proof that
the hex lattice minimises per-area perimeter; the analytic justification
for hex attention over square attention.

Rastegari, Ordonez, Redmon, Farhadi 2024 'Fibottention: Inceptive Visual
Representation Learning with Diverse Attention Across Heads'
(arXiv:2406.19391) -- per-head Fibonacci dilation; we adopt its
Wythoff-array construction for hex-attention head distribution.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

The CNN analogue is a hex-grid attention module replacing a standard
self-attention layer: tokens live on a hex grid (after a learned
permutation), each token attends to its 6 hex neighbours plus itself
(7-tap kernel), and the KV state is stored with toroidal wrap so the
last and first columns/rows are adjacent. Shapes: input (B, C, H, W);
permute to hex coords (B, C, H, W) but with shifted-row addressing;
hex-attn output same shape. Params delta: -22% vs. dense self-attn (we
keep a 7-tap weight matrix). FLOPs delta: -50% at H=W=32.

```python
# src/nature_inspired_networks/hex_attn.py
class HexToroidalAttn(nn.Module):
    def __init__(self, d, n_heads, prune_ratio=0.21):
        super().__init__()
        self.heads = n_heads
        self.qkv = nn.Linear(d, 3*d)
        self.proj = nn.Linear(d, d)
        # 7-tap hex offsets: center + 6 neighbours
        self.register_buffer('hex_offsets', torch.tensor(
            [[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,-1],[-1,1]]))
        self.prune_ratio = prune_ratio
    def forward(self, x):                            # (B, N, d) N = H*W
        # toroidal-wrap addressing
        ...
        return self.proj(out)
```

### 5.2 LLM track (decoder-only Transformer)

Slot: **replaces the MHSA sub-layer**. Token positions are mapped onto
a 1-D hex strip (axial hex coordinate), KV cache is stored in a ring
buffer with modular index update (the torus), and pruning is applied
once per **Fibonacci-spaced epoch** (8%, 13%, 21%, 34%, 55% cumulative
sparsity).

FlashAttention-2 compatibility: hex attention can re-use FA2 by
pre-computing a sparse causal mask whose support is exactly the hex
neighbours within the causal cone; the support pattern is static so
FA2's `bias_mask` accepts it without per-step recomputation. Causal
mask preservation: the hex offsets are filtered to keep only
non-positive temporal offsets.

KV cache analysis: at 128k context, baseline GPT-2-style cache is
≈768 MB (24 layers × 1024 d × 128k × 2 bytes); hex 7-tap + 21%
Fib-pruning gives ≈430 MB (-44%).

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite | [+0.015, +0.030] | KV win dominates; PPL near-flat |
| perplexity (LLM) | [-0.10, +0.20] nats | sparsity costs <0.2 nats per HexaConv |
| params | [-1%, -3%] | qkv unchanged; output proj unchanged; minus pruned weights |
| FLOPs | [-30%, -55%] | 7-tap vs N-tap |
| GPU latency (batch=1) | [-15%, -35%] | sparse pattern fits FA2 |
| KV cache @ 128k | [-50%, -40%] | hex + Fib prune compound |
| Betti collapse rate | [+5%, +10%] | sparser graph collapses cycles faster |

## 7. Experimental protocol

### 7.1 Primary experiment

- Dataset: **WikiText-103** progressive 8k → 16k → 32k → 64k → 128k.
- Architecture: 124M decoder (12 × 768 × 12), then 350M.
- Training: 200k steps total with progressive context schedule
  (40k steps per context length), bf16, grad-ckpt.
- Composite: `0.5*norm_kv + 0.4*neg_norm_ppl + 0.1*norm_latency`, SHA-256.
- Wall-clock: ≈48 h on 4090 Laptop (16 GB) for 124M; 350M cap at 96 h.
- Archive: `ideas/62_toroidal_kv_hex_attn/experiments/exp001_progressive/`.

### 7.2 Idea-targeted experiment

The prior should SHINE at **>32k context with repetitive long-range
structure** — e.g., copy-task variants and PG19 long-document
perplexity. We add a synthetic toroidal-shifted copy task (16k tokens
that wrap once) where the hex-toroidal variant should improve recall
≥20 pp over dense attention.

### 7.3 Cross-paradigm context (LLM track)

H62 sits in the **efficiency-axis** of the paradigm comparison: it
exchanges Transformer's O(N²) for HexConv's O(N) at the cost of one hex
re-permutation per layer. The toroidal wrap echoes LFM2's ring-buffer
KV, and the Fib pruning is a discrete realisation of the curriculum
sparsification reported for V-JEPA 2's mask schedule.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G7 row H62.
- Master log: `EXPERIMENT_LOG.md` Tier-2 row T2.H62.
- Sub-dir: `ideas/62_toroidal_kv_hex_attn/`.
- Composes with: H21, H22, H28, H43, H67.
- Conflicts with: H37 (pentagonal head grouping uses an alternative
  sparsity pattern).

## 9. Committee Q&A

**Q: Why isn't this just HexaConv on tokens?**

> HexaConv is 2-D spatial; H62 is 1-D token-stream wrapped on a torus
> with Fibonacci pruning. The composition is new.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 names the 25% KV-cut + 0.4-nat PPL ceiling.

**Q: What if hex sparsity hurts at 8k but helps at 128k?**

> § 7.2 schedules the experiment progressively to detect exactly that
> crossover; the composite is computed at each context length and the
> Pareto frontier is the deliverable.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> Acknowledged. H62 stacks 4 priors but each is independently
> ablatable: dropping toroidal wrap is a hex-attn-only run, dropping
> hex is a toroidal-cache-only run, dropping Fib pruning is a no-prune
> run.

**Q: How do we know the implementation is correct?**

> `tests/test_hex_toroidal_attn.py` checks (a) 6-fold rotational
> equivariance at H=W=12, (b) causal-mask preservation, (c) toroidal
> wrap correctness via random index shifts, (d) sparsity exactly equals
> Fib-schedule target.

## 10. Verification artifacts checklist

- [ ] `ideas/62_toroidal_kv_hex_attn/implementation.py` green
- [ ] `tests.py` ≥ 10 assertions
- [ ] `AUDIT.md` ≥ 3 weaknesses
- [ ] `IMPROVEMENTS.md`
- [ ] `VERIFY.md` signed
- [ ] `experiments/exp001_progressive/`
- [ ] `verification/{tests,smoke,gates,reproduction}.txt`
- [ ] EXPERIMENT_LOG row
- [ ] FINDINGS reflected
- [ ] Dashboard updated

## 11. Status journal

- 2026-05-26 — Created from template by Doc-Agent-D.
