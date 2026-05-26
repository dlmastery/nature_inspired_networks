# H55 — Platonic Transformers (Islam 2025 Reference Implementation)

> **One-line claim:** Implementing the Platonic Transformer (Islam
> 2025), in which attention heads are partitioned by Platonic-solid
> vertex incidence (icosahedron's 12 vertices → 12 heads, dodecahedron's
> 20 → 20 heads), yields a strictly equivariant Transformer that
> matches a vanilla Transformer's CIFAR-10 / QM9 accuracy at iso-params
> while reducing rotation-equivariance error by ≥0.04 on rotated test
> sets, validating the Platonic head-grouping as a useful inductive
> bias on 3D-symmetric data.
>
> **Source design space:** G6 Topological + bridging additions (H51–H60).
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for hypothesis
H55. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The Platonic Transformer (Islam 2025) is a recent proposal that
introduces strict Platonic-symmetry into the multi-head attention
mechanism: attention heads are not independent random projections,
but are *partitioned and constrained* by the vertex incidence of a
Platonic solid. For a 12-head model, the heads correspond to the 12
vertices of the icosahedron and are required to respect the
icosahedral rotation group's action — rotating the input by an
icosa-symmetry element permutes the heads correspondingly.

This is a 3D-equivariant Transformer in the strict GDL sense (Bronstein
2021): the model is built from operations that commute with the group
action, so the representations are guaranteed to be equivariant up to
permutation of heads. On 3D-symmetric data (QM9 molecules, ShapeNet
parts), this gives the same inductive-bias benefit that icosahedral
CNNs (H24) give on spherical inputs, with the added flexibility of
Transformer architectures (no fixed grid).

The sacred-geometry framing is direct: the Platonic solids are nature's
canonical symmetry groups; mapping attention heads to their vertex
sets is the natural way to *enforce* those symmetries inside a
Transformer. H55 is the reference implementation of this idea in
this repo.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because partitioning attention heads by Platonic-vertex incidence
imposes strict rotation-group equivariance on the model — mechanism-
wise, the head outputs are permuted by group elements that rotate
the input — per Islam 2025 (Platonic Transformer) and Bronstein 2021
GDL, we expect a Platonic Transformer to match a vanilla Transformer's
accuracy on QM9 at iso-params while reducing rotation-equivariance
error on rotated test sets by ≥0.04 (3-seed median, 95% CI exclusion
of 0).

## 3. Falsifier (≥ 30 words)

If the Platonic Transformer's accuracy on QM9 atomization energy
drops by more than -0.5 kcal/mol from the vanilla Transformer
baseline (a meaningful regression), OR if the rotation-equivariance
error reduction is < 0.02 (95% CI exclusion of 0.02), this
hypothesis is DISCARDED.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
Islam, M. and others 2025 (forthcoming) 'Platonic Transformers' --
the reference paper for H55; defines the Platonic-vertex head
partitioning we implement.

Bronstein, Michael M. and Bruna, Joan and Cohen, Taco S. and
Veličković, Petar 2021 arXiv 'Geometric Deep Learning: Grids,
Groups, Graphs, Geodesics, and Gauges' (arXiv:2104.13478) -- GDL
manifesto; the theoretical framework for group-equivariant
architectures we cite.

Cohen, Taco S. and Geiger, Mario and Köhler, Jonas and Welling,
Max 2019 ICML 'Gauge Equivariant Convolutional Networks and the
Icosahedral CNN' (arXiv:1902.04615) -- the icosahedral group's
introduction to deep learning; methodological precursor.

Vaswani, Ashish and others 2017 NeurIPS 'Attention Is All You Need'
(arXiv:1706.03762) -- the original Transformer; vanilla baseline.

Schütt, Kristof T. and others 2017 NIPS 'SchNet — A continuous-
filter convolutional neural network for modeling quantum
interactions' (arXiv:1706.08566) -- the canonical QM9 reference
baseline.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

The Platonic Transformer is *not* a CNN; we adapt it to the CIFAR-10
classification task by patchifying the image (16×16 patches → 196
tokens) and feeding through a Platonic-Transformer encoder. Each
Platonic head is computed as:

```python
class PlatonicMultiheadAttention(nn.Module):
    def __init__(self, d_model, group="icosa"):
        super().__init__()
        self.d_model = d_model
        # group = "tetra" (4-head), "cube" (8-head), "octa" (6-head),
        #         "icosa" (12-head), "dodeca" (20-head)
        self.n_heads = {"tetra": 4, "cube": 8, "octa": 6,
                        "icosa": 12, "dodeca": 20}[group]
        self.d_head = d_model // self.n_heads
        # vertex coordinates as group-equivariant projection
        self.vertex_coords = get_platonic_vertices(group)  # (n_heads, 3)
        # ... Q/K/V projections with shared weights across symmetry orbits
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # standard attention; heads enumerated by Platonic vertices
        ...
```

The strict equivariance constraint is enforced by weight-tying across
group orbits: heads connected by a group element share Q/K/V
parameters. For icosahedron, the 60-element rotation group acts on 12
heads producing 12/12 = 1 free Q/K/V matrix shared across all heads
(but with vertex-rotation applied at forward time).

Compute cost: same FLOPs as a vanilla 12-head attention at iso-d_model.
Params: reduced by symmetry-group-size factor (5×–10× fewer
parameters).

Lives in `src/nature_inspired_networks/transformer/platonic_attention.py`,
re-exported by `ideas/55_platonic_transformer/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

For decoder LLMs, replace `MultiheadAttention` with
`PlatonicMultiheadAttention(group="icosa")`. Number of heads becomes
12 (for icosa); LLM choices to match: ensure d_model is divisible by
12 (GPT-2-small has d_model=768 = 12 × 64, perfect fit).

FlashAttention-2 compatibility: the Platonic attention's standard
softmax / scaled-dot-product is FA2-compatible; the weight-tying
applies at parameter level, not at kernel call. Causal mask preserved.

Expected at 124M: matches vanilla GPT-2-small ppl on TinyStories
(±0.3 ppl); reduces ZeroShot rotational-symmetry probes (synthetic
diagnostics) by ≥0.05.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline (vanilla 12-head) | rationale |
|---|---|---|
| composite | [-0.005, +0.015] | iso to mild lift |
| top-1 (CIFAR-10) | [-1.0, +1.0] pp | iso-acc hypothesis |
| QM9 MAE (kcal/mol) | [-0.3, +0.1] | mild improvement |
| params | [-50%, -30%] | symmetry-tied weights |
| FLOPs | [0, 0] | same as vanilla |
| GPU latency (batch=1) | [0, +5%] | slight overhead from vertex math |
| rotation-equivariance err | [-0.10, -0.04] | core targeted metric |
| KV cache @ 32k (LLM) | [0, 0] | unchanged |
| Betti collapse rate | [-0.05, +0.05] | minor |
| perplexity (LLM 124M) | [-0.3, +0.3] | iso |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dataset:** QM9 atomization-energy regression (130k molecules)
- **Architecture:** Platonic Transformer (icosa, 12 heads) vs. vanilla
  Transformer (12 heads) at matched d_model=128, 6 layers, ~600k params
- **Epochs:** 100
- **Seeds:** 0, 1, 2
- **Run-script:** `python scripts/run_idea.py --idea 55 --dataset qm9 --seeds 0 1 2`
- **Wall-clock:** ≈ 3 h × 3 seeds × 2 conditions ≈ 18 h
- **Archive path:** `ideas/55_platonic_transformer/experiments/exp001_qm9/`

### 7.2 Idea-targeted experiment

Strongest demonstration on rotated test sets:

- **Dataset:** rotated-MNIST (synthetic; arbitrary 3D rotations applied)
- **Architecture:** small Platonic-Transformer on MNIST
- **Predicted:** rotation-equivariance error reduced by ≥0.10
- **Diagnostic:** if no equivariance gain on the natural setting,
  the implementation has a bug.

### 7.3 Cross-paradigm context (LLM track)

- **Model:** 124M GPT-2-small with PlatonicMultiheadAttention(icosa)
- **Dataset:** TinyStories
- **Metric:** val ppl and rotational-probe consistency
- **Run:** `python scripts/run_llm.py --idea 55 --attn platonic_icosa`

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G6 row H55.
- Master experiment list: `EXPERIMENT_LOG.md` (planned).
- Implementation sub-directory: `ideas/55_platonic_transformer/`
- Related hypotheses that compose:
  - **H24** Icosahedral φ-equivariant (full CNN version).
  - **H37** Pentagonal φ-attention — sibling head-grouping; H55 is
    the strict-equivariant version.
  - **H71** Icosa RoPE — composes (Platonic heads + Platonic RoPE).
- Related hypotheses that conflict:
  - **H32** Fibottention — different head-grouping principle (Fib
    dilations); cannot run both on the same model.

## 9. Committee Q&A

**Q: Why isn't this just a re-implementation of Islam 2025?**

> The contribution is the integration with this repo's framework +
> the CIFAR-10 / rotated-MNIST diagnostics + the LLM-track sibling
> experiment.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies QM9 MAE bound + ≥0.02 equivariance reduction.

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> Platonic equivariance is most useful on 3D-symmetric data (QM9,
> ShapeNet). On CIFAR-10 we expect iso-acc; on rotated tasks the
> equivariance pays off. Scope is explicit.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H55 is a Transformer-architectural prior; the previous sweep was
> CNN-only. The compound-failure framing does not extend.

**Q: How do we know the implementation is correct?**

> `tests.py` verifies (a) rotating the input by an icosa-group element
> produces the predicted head permutation at the output, (b) parameter
> count matches the weight-tied symmetry-group prediction.

## 10. Verification artifacts checklist

- [ ] `ideas/55_platonic_transformer/implementation.py` exists
- [ ] `ideas/55_platonic_transformer/tests.py` ≥ 8 assertions
- [ ] `ideas/55_platonic_transformer/AUDIT.md` ≥ 3 weaknesses
- [ ] `ideas/55_platonic_transformer/IMPROVEMENTS.md`
- [ ] `ideas/55_platonic_transformer/VERIFY.md` signed
- [ ] One experiment archive
- [ ] `verification/{tests,smoke,gates,reproduction}.txt`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-C.
