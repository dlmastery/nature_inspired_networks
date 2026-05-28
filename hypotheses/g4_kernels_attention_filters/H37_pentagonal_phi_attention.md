# H37 — Pentagonal φ-Attention

> **One-line claim:** Grouping multi-head attention heads in pentagonal / dodecahedral symmetry (5 / 10 / 20 heads) with edge weights modulated by φ improves rotation-equivariance error and zero-shot accuracy on 3-D / spatially-symmetric tasks versus uniform-head allocation at matched total head count.
>
> **Source design space:** G4 Kernels / Attention / Filters (H31–H40).
>
> **Implementation status (this repo):** `○ not started`.

This document is the committee-grade design write-up for hypothesis H37.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The pentagon and the dodecahedron are the canonical 5-fold-symmetry geometric forms; they appear in starfish anatomy, the C5 viral capsid axis, the dodecahedral phase of certain liquid crystals, and the Penrose-tile aperiodic packings. The pentagonal symmetry group D5 (order 10) is the unique cyclic-mirror group that does NOT tile the plane periodically — yet it tiles the surface of the dodecahedron and the icosahedron exactly. The connection to φ is direct: every diagonal of a regular pentagon is exactly φ times its side, the pentagon's vertex coordinates contain φ in closed form, and the angles 36°, 72°, 108° are all related by φ.

For Transformer multi-head attention, the standard practice is uniform-head allocation (e.g., 8 / 12 heads). **Pentagonal head allocation** groups heads in 5 / 10 / 20 — matching the dodecahedral / icosahedral vertex counts. Each pentagon's 5 heads attend to a different "facet" of the rotation group, with edge weights between heads following the dodecahedral edge graph (Petersen graph for 10 heads, full dodeca for 20). This is a structural prior that may help on 3-D / spatial / molecular tasks where pentagonal symmetry is data-aligned.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** pentagonal / dodecahedral symmetry is the highest-symmetry 5-fold group and is connected to φ via diagonal/side ratios, partitioning multi-head attention into 5 / 10 / 20 heads with dodecahedral-graph inter-head connections and φ-edge weighting raises rotated-CIFAR-10 top-1 by ≥ +1 pp and reduces rotation-equivariance error by ≥ 0.025 relative to uniform 8 / 12 / 16 head allocation, per the mechanism of Cohen 2019 icosahedral CNN.

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on rotated-CIFAR-10, the pentagonal-head ViT-Tiny fails to lift top-1 by ≥ 0.5 pp AND fails to reduce rot-eq-err by ≥ 0.015 versus the 8-head uniform baseline, this hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Cohen, T. S., et al. 2019 ICML 'Icosahedral CNN' (arXiv:1902.04615) —
icosahedral equivariance reference, of which dodecahedral is dual.

Dosovitskiy, A., et al. 2021 ICLR 'An Image is Worth 16x16 Words'
(arXiv:2010.11929) — ViT reference H37 modifies.

Penrose, R. 1974 — Penrose-tiling pentagonal aperiodic packing.

Petersen, J. 1898 — Petersen graph (10-vertex 3-regular dodecahedron-
edge graph).

Vaswani, A., et al. 2017 — original MHA reference.

Krizhevsky 2009 — CIFAR dataset citation.
```

## 5. Mechanism

### 5.1 CNN track (ViT-Tiny)

Partition the 10-head MHA into 2 pentagonal groups of 5 heads each, with Petersen-graph inter-head connectivity (each head has 3 "neighbour heads" in the Petersen sense). Apply φ-weighted aggregation across the Petersen edges before the final projection.

```python
# ideas/37_pentagonal_attention/implementation.py
PHI = (1+5**0.5)/2

def petersen_adjacency():
    """Return (10, 10) Petersen-graph adjacency."""
    # Outer pentagon: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner pentagram: 5-7, 7-9, 9-6, 6-8, 8-5
    # Cross edges: 0-5, 1-6, 2-7, 3-8, 4-9
    edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
             (5,7),(7,9),(9,6),(6,8),(8,5),
             (0,5),(1,6),(2,7),(3,8),(4,9)]
    A = torch.zeros(10, 10)
    for i, j in edges: A[i,j] = A[j,i] = 1.0
    return A

class PentagonalMHA(nn.Module):
    def __init__(self, d, heads=10):
        super().__init__()
        assert heads == 10, "Pentagonal MHA requires 10 heads"
        self.heads = heads
        self.head_dim = d // heads
        self.qkv = nn.Linear(d, 3*d); self.proj = nn.Linear(d, d)
        A = petersen_adjacency()
        self.register_buffer("petersen", A + torch.eye(10))
        self.phi_weight = 1.0 / PHI

    def forward(self, x):
        B, L, d = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, 10, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        out = attn.softmax(-1) @ v  # (B, 10, L, head_dim)
        # Mix heads via Petersen-phi weighting before projection
        out_mixed = torch.einsum('bhld,gh->bgld', out, self.petersen * self.phi_weight + (1-self.phi_weight)*torch.eye(10).to(out.device))
        out_flat = out_mixed.transpose(1,2).reshape(B, L, d)
        return self.proj(out_flat)
```

- Params: same as standard 10-head MHA (Petersen mixing is parameter-free).
- FLOPs: extra 10×10 mixing per token; negligible.
- Init: Xavier.

### 5.2 LLM track

For decoder-only Transformers, pentagonal head allocation applies the same Petersen-graph mixing at every MHA. The hypothesis is that 3-D / spatial reasoning tasks benefit; text-only benchmarks may be neutral.

Expected at 124 M: **perplexity within 0.2 of dense**, neutral on text; **+1-3 pp** on 3-D spatial reasoning tasks (ARC-3D, robot-nav-QA).

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (rotated-CIFAR) | [+0.005, +0.025] | rotation-aligned |
| top-1 (rotated-CIFAR, primary) | [+1.0 pp, +3.0 pp] | direct claim |
| top-1 (upright CIFAR) | [-0.5 pp, +0.5 pp] | neutral on isotropic |
| Rotation-equivariance err | [-0.02, -0.05] | direct geometric prediction |
| params | [0, 0] | same |
| FLOPs | [+1 %, +3 %] | small Petersen mixing |
| GPU latency | [≈1.0×, ≈1.05×] | trivial |
| Perplexity (LLM) | [+0.0, +0.2] | retention claim |
| KV cache | [0, 0] | unchanged |
| Betti collapse rate | [+0.05, +0.15] | symmetric prior aids consolidation |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** rotated-CIFAR-10 (train upright, eval random 0-90° rotations).
- **Architecture:** ViT-Tiny with PentagonalMHA (10 heads instead of 8).
- **Epochs / batch / precision / seeds:** 50 epochs, batch 256, bf16, 3 seeds.
- **Composite:** top-1 (0.4), rot-eq-err (0.3), latency (0.15), params (0.15).
- **Run-script:** `python ideas/37_pentagonal_attention/experiment.py --seeds 0 1 2`.
- **Wall-clock:** ~30 min/seed × 3 = 90 min.
- **Archive:** `ideas/37_pentagonal_attention/experiments/exp001_rotcifar_seed0..2/`.

### 7.2 Idea-targeted experiment

1. **Rotated-CIFAR-10 / Tiny-ImageNet** — rotation testbed.
2. **Spherical MNIST** — 5-fold symmetry alignment.
3. **3-D point-cloud (ModelNet40)** — pentagonal-icosahedral structure.

### 7.3 Cross-paradigm context

LLM-track on a 3-D-spatial benchmark (ARC-3D-mini) at 124 M with pentagonal MHA.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G4 row H37.
- Master: planned Tier 3.
- Sub-dir: `ideas/37_pentagonal_attention/`.
- Composes: H16 (Fib head diversity), H23 (Platonic graph), H24 (icosa CNN), H25 (dodeca latent), H30 (Platonic-Fib hybrid).
- Conflicts: H32 (Fibottention — 8 heads with Fib dilation, structural conflict on head allocation).

## 9. Committee Q&A

**Q: Why isn't this just a 10-head MHA with extra mixing?**

> The mixing is STRUCTURED as the Petersen graph (the unique 3-regular, vertex-transitive graph on 10 nodes that embeds the dodecahedral edges), not arbitrary. Random 10×10 mixing is the natural control.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 gives top-1 ≥ +0.5 pp AND rot-eq-err ≤ -0.015. Both must hold.

**Q: What if 12-head uniform performs identically?**

> The pre-registered comparison is 8-head uniform (standard ViT-Tiny). 10-head uniform is a control; 12-head uniform is a third control with extra heads. The Petersen-graph mixing is the variable.

**Q: T1.4 showed group conv (4-fold symmetry) was -10 pp. Why expect 5-fold symmetry to help?**

> Because (a) we apply it as HEAD MIXING (not orbit reduction with max-pool), (b) we test on rotated data not upright, (c) ViT attention is structurally different from group conv.

**Q: How do we know the implementation is correct?**

> `tests/test_pentagonal.py::test_petersen_structure` asserts the graph has 15 edges and is 3-regular. `test_petersen_vertex_transitive` verifies the automorphism group has order 120. `test_phi_weight_applied` asserts the mixing matrix has off-diagonal `1/φ`. Plus archive verification.

## 10. Verification artifacts checklist

- [ ] `ideas/37_pentagonal_attention/implementation.py` tests green
- [ ] `ideas/37_pentagonal_attention/tests.py` ≥ 5 assertions
- [ ] AUDIT.md ≥ 3 weaknesses
- [ ] IMPROVEMENTS.md
- [ ] VERIFY.md signed
- [ ] Experiment archives + verification
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] FINDINGS reflects result

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-B.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G4 (elite-research-scientist critic). Critiquing the IDEA, not the implementation (audit at `audits/G4_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)

**LOW.** The proposed mechanism applies a CONSTANT 10×10 mixing matrix (`A·(1/φ) + I·(1 − 1/φ)`) after attention softmax. This mixing is a LINEAR head-recombination — equivalent to a fixed orthogonal-projection layer between attention and output proj. It is exactly the kind of "linear-mixing-doesn't-help" intervention that the literature has tested and rejected (head pruning, head importance studies — Voita 2019 arXiv:1905.09418 found that heads are largely redundant; explicit linear mixing layers are SUBSUMED by the existing output projection W_O). The Petersen-graph structure is COSMETIC since the mixing is then projected by a dense `proj` Linear that can undo any mixing.

### Mechanism scrutiny

The Petersen mixing is a CONSTANT matrix `M` applied as `out_mixed[g] = sum_h M[g,h] · out[h]`, then `output = W_O · out_mixed`. By matrix associativity, this is equivalent to using `W'_O = W_O · M` as the output projection — i.e., the Petersen mixing IS ABSORBED INTO W_O during training. The model can learn ANY 10×10 mixing through W_O alone; pre-multiplying by the Petersen matrix is a no-op once W_O has trained. The "structural prior" is a deception.

The 3-D / molecular-task claim is unsupported: text and natural images do NOT have 5-fold rotational symmetry. Natural images have approximate horizontal-flip symmetry (Olshausen 1996) and small in-plane rotation tolerance (RandRotation augmentation). Imposing 5-fold mixing on heads buys nothing on data without 5-fold structure. The T1.4 group-conv negative result (−10 pp) is direct evidence; the rebuttal that "head mixing differs from orbit reduction" is true but irrelevant: BOTH impose symmetry that the data does not have.

### Confounds (≥2)

1. **Absorption-into-W_O confound.** As above, the fixed Petersen mixing is annihilated by the trainable output projection W_O during training. After training, the model is INDISTINGUISHABLE from a standard 10-head MHA with a different W_O init. Control: train without Petersen mixing but with the same W_O init.
2. **10-vs-8 head confound.** The baseline uses 8 heads; the variant uses 10. With d=192 and 10 heads, head_dim = 19.2 (NON-INTEGER!) — the implementation must use d=190 or d=200. This is a structural change unrelated to pentagonal symmetry. Control: 10-head dense MHA WITHOUT Petersen mixing, matched d.

### Numerology / specificity check

The φ-edge weighting is a single scalar `1/φ ≈ 0.618` on off-diagonal entries. This is one number. Replace 0.618 with 0.5, 0.7, 0.3, or any value in [0.3, 0.9] and the mixing matrix has identical RANK and similar conditioning — the φ-specificity is undetectable. The Petersen graph is the relevant structural choice; φ-edge-weight is decoration. Furthermore, the Petersen graph is NOT the dodecahedron's edge graph — the dodecahedron has 20 vertices and 30 edges; the Petersen graph has 10 vertices and 15 edges. The Petersen graph is the KNESER GRAPH KG(5,2), with NO direct geometric correspondence to the dodecahedron. The author has conflated different objects. **High numerology score.**

Beyond numerology: 5-fold rotational symmetry is NOT a property of CIFAR-10 or text data. Imposing it as a head-mixing structure is data-misaligned. The pre-existing T1.4 group-conv result (−10 pp) is strong prior evidence that symmetry constraints HURT on data without that symmetry.

### Literature precedent — kernel/attention design is a crowded field

Head-mixing / head-importance literature: Voita 2019 (arXiv:1905.09418) "Analyzing Multi-Head Self-Attention" — many heads can be pruned; Michel 2019 (arXiv:1905.10650) "Are Sixteen Heads Really Better than One?" — head redundancy. Talking-Heads Attention (Shazeer 2020 arXiv:2003.02436) does learnable head-mixing — the closest precedent — and finds marginal gains (~0.5 pp). The Petersen-fixed-mixing variant is a SPECIAL CASE of Talking-Heads with a fixed pre-attention matrix and is dominated by the learnable case.

### Expected effect size (90% CI a priori)

On rotated-CIFAR-10: [-1.5 pp, +0.5 pp] (likely negative or null vs 8-head baseline; the 10-head + Petersen change is a structural noise). On upright CIFAR: [-1.0 pp, +0.5 pp]. The author's [+1.0, +3.0] is implausibly large.

### Minimum-distinguishing experiment

Train 3 seeds on rotated-CIFAR-10 ViT-Tiny: (a) 8-head dense MHA, (b) 10-head dense MHA (no mixing), (c) 10-head + Petersen-1/φ mixing, (d) 10-head + RANDOM 3-regular 10-vertex mixing, (e) 10-head + Talking-Heads LEARNABLE mixing. If (c) > (b), (d) by ≥ 0.3 pp at matched FLOPs, the Petersen-pentagonal claim is non-null. If (c) ≈ (b) or (e) dominates, it is null. The doc's existing committee Q&A predicts (b) and "12-head uniform" as controls but does not pre-register them in § 7.

### Verdict
NUMEROLOGY — the Petersen mixing is absorbed into W_O during training (matrix associativity); φ-edge-weight is a single scalar that any value in [0.3, 0.9] could replace; Petersen graph is NOT the dodecahedron edge graph (it's Kneser KG(5,2)); data has no 5-fold symmetry. The T1.4 group-conv −10 pp negative is direct refuting evidence the doc dismisses.
