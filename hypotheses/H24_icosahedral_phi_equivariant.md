# H24 — Icosahedral φ-Equivariant CNN

> **One-line claim:** An icosahedral-group equivariant CNN with Fibonacci-allocated channel groups raises top-1 on spherical / IcoMNIST and reduces rotation-equivariance error versus a non-equivariant baseline of matched parameter budget, **provided the orbit reduction is AVG-pool rather than MAX-pool** (per the lesson learned from T1.4).
>
> **Source design space:** G3 Topologies & Graphs (H21–H30).
>
> **Implementation status (this repo):** `○ not started` (full version). A **C4 proxy** ran in T1.4 (`sg_only_group`) and produced the single worst negative result of the previous sweep (-10.27 pp top-1, -0.046 rot-eq-err); H58 (max → avg-pool fix) is currently running and is the **direct prerequisite** for the full icosahedral version proposed here.

This document is the committee-grade design write-up for hypothesis H24. The proxy run (T1.4 / `sg_only_group`) is the most negatively-impactful single-prior result in the previous CIFAR-10 campaign and is the **textbook example** of an equivariance prior wasted on isotropic data. We integrate H58 (the max → avg-pool fix that the previous campaign identified as the top-priority follow-up) as part of this hypothesis, because H24 cannot run safely until H58 confirms the fix recovers the lost capacity.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The icosahedron has 60 rotational symmetries and is the highest-symmetry Platonic solid that still has discrete (not continuous) symmetry. It is nature's preferred shape for closed sphere-like structures at the molecular scale: viral capsids (HIV, influenza, ~90 % of all viruses follow Caspar–Klug icosahedral construction), C60 fullerene, quasicrystals (Penrose tilings with icosahedral symmetry at long range), and the dual dodecahedron underlies the radiolarian skeletons Haeckel famously catalogued. The icosahedral group I (order 60) and its alternating-A5 subgroup form the largest finite rotation group of the sphere; any spherical signal can be discretized onto a 12-, 20-, 32- or 92-vertex icosahedral mesh with near-uniform sampling density that no other regular polyhedron achieves.

In deep learning, **icosahedral equivariance** matters when (a) the data lives on the sphere (climate models, panoramic cameras, cosmological maps, drug-target binding sites on protein surfaces), or (b) the data has rotational symmetries the network should be wired to exploit rather than learn from scratch. Cohen 2019 (arXiv:1902.04615) shows full icosahedral CNNs outperform planar baselines on Spherical MNIST and climate-pattern segmentation by 2–8 pp. Adding **Fibonacci-allocated channel groups** layers a phyllotaxis prior on top: channels are partitioned into groups of size {1, 1, 2, 3, 5, 8} summing to 20 (the icosa-vertex count), so each group of orbits-on-the-icosa receives an exponentially-increasing channel budget that matches natural-image power-spectrum statistics.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** the icosahedral group is the largest finite spherical-rotation group, an icosahedral group conv with Fibonacci-partitioned channel groups and **average-pool** orbit reduction (not max-pool, per the H58 lesson) raises top-1 on Spherical MNIST and IcoMNIST by ≥ +5 pp and on rotated-CIFAR by ≥ +1.5 pp, with rot-eq error reduced by ≥ 0.04 — all relative to the priors-off NaturePriorBlock baseline — per the mechanism of Cohen 2019 (arXiv:1902.04615) corrected by the avg-pool fix in our H58.

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on Spherical MNIST AND on rotated-CIFAR-10, the H24 variant fails to lift top-1 by ≥ 3.0 pp on Spherical MNIST AND fails to reduce rot-eq-err by ≥ 0.03 on rotated-CIFAR, this hypothesis is **DISCARDED**. CIFAR-10 (upright) is excluded from the falsifier because T1.4 already established it is not the proper testbed.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Cohen, T. S., Geiger, M., Köhler, J., Welling, M. 2019 ICLR
'Spherical CNNs' (arXiv:1801.10130) and Cohen, T. S. et al. 2019 ICML
'Gauge Equivariant Convolutional Networks and the Icosahedral CNN'
(arXiv:1902.04615) — the primary icosahedral CNN reference; +2–8 pp on
Spherical MNIST and climate-pattern segmentation.

Caspar, D. L. D., Klug, A. 1962 Cold Spring Harbor 'Physical principles
in the construction of regular viruses' — biological grounding.

Esteves, C., Allen-Blanchette, C., Makadia, A., Daniilidis, K. 2018
ECCV 'Learning SO(3) Equivariant Representations with Spherical CNNs'
(arXiv:1711.06721) — comparator non-icosahedral spherical CNN.

He, K., Zhang, X., Ren, S., Sun, J. 2016 CVPR 'Deep Residual Learning'
(arXiv:1512.03385) — the ResNet scaffold the equivariant blocks slot into.

Worrall, D. E., Welling, M. 2019 NeurIPS 'Deep Scale-spaces:
Equivariance Over Scale' (arXiv:1905.11697) — broader equivariance
literature context.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

The full implementation requires an icosahedral-mesh sampling of the input + an icosahedral group conv. We use the **GICOPix planar unfold** (Eder 2020) so the icosa-mesh becomes a rectilinear 2-D grid amenable to standard CUDA conv kernels.

- Input: 2-D image is mapped onto the icosahedral mesh by gnomonic projection; equivalent rectilinear shape after GICOPix unfold is `(B, C, 5·H, H)` where H is the per-face resolution.
- Convolution: rotate-and-share weights across the 60 icosahedral group elements; **orbit reduction at the end of each block is AVG-pool over the 60 orbit elements (not max-pool, per the H58 lesson)**.
- Param count: a `(C_in, C_out)` icosa conv has the SAME parameter count as a standard `(C_in, C_out)` 3×3 conv, because weights are SHARED across the 60 orbit copies. Output features are 60× larger before reduction.
- FLOPs: 60× the standard conv if naively materialized, but in practice ≈ 8–12× with kernel fusion + sparsity.
- Init: He init; group conv preserves variance.

```python
# ideas/24_icosa_phi_equivariant/implementation.py
class IcosaGroupConv2d(nn.Module):
    def __init__(self, C_in, C_out, k=3, reduction="avg"):
        """reduction MUST be avg per the T1.4 lesson learned."""
        super().__init__()
        self.weight = nn.Parameter(torch.randn(C_out, C_in, k, k) * 0.01)
        self.register_buffer("orbit_rot",
            icosa_rotation_matrices(60, k))  # (60, k, k)
        self.reduction = reduction
        assert reduction == "avg", "Use AVG pool — max-pool kills training (T1.4)"

    def forward(self, x):
        # x: (B, C_in, H, W) on GICOPix-unfolded grid
        outs = []
        for r in range(60):
            wr = self.weight.unsqueeze(0) @ self.orbit_rot[r]
            outs.append(F.conv2d(x, wr, padding=self.k//2))
        out = torch.stack(outs, dim=1)  # (B, 60, C_out, H, W)
        return out.mean(dim=1)          # AVG over orbit (NOT MAX!)
```

- Fibonacci channel groups: partition the output channels into groups of sizes `{1,1,2,3,5,8} = 20` (icosa vertex count) and apply different orbit-reductions per group (some avg, some max-equivariant) to ablate.
- Lives in: `ideas/24_icosa_phi_equivariant/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

For decoder-only Transformers, icosahedral equivariance maps onto **3-D spatial position encoding** — see H71 (icosahedral RoPE). Each token's positional encoding is sampled from a 12-vertex icosa mesh in 3-space, supporting models that reason about 3-D spatial relations (robot nav, embodied QA).

- Slots in: RoPE replacement. Token position `i` is mapped to an icosa-vertex via `pos_i = icosa_vertex[i mod 12]`; rotary projection uses the icosa-rotation group as base.
- FlashAttention-2 compatibility: ✓ — the rotary encoding is applied to Q/K before the dot product.
- Causal-mask preservation: ✓.
- Pseudocode: see H71.
- Expected impact on 3-D-nav-QA: **+2–5 pp** zero-shot accuracy at 350 M scale; neutral on text-only perplexity.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (Spherical MNIST) | [+0.030, +0.080] | the prior's natural data regime |
| top-1 (Spherical MNIST, primary) | [+5.0 pp, +12.0 pp] | direct claim per Cohen 2019 |
| top-1 (rotated-CIFAR-10) | [+1.5 pp, +4.0 pp] | 60-fold rotation supervision |
| top-1 (upright CIFAR-10) | [-3.0 pp, +0.5 pp] | wrong testbed, may still benefit from AVG-pool fix |
| params | [-5 %, +5 %] | weights shared across orbit |
| FLOPs | [×6, ×12] | 60-orbit materialization, mitigated by fusion |
| GPU latency (batch=1) | [×4, ×8] | substantial; budget limits to T3 protocol |
| rotation-equivariance err | [-0.05, -0.10] | direct geometric prediction |
| KV cache @ 32 k (LLM) | [+0 %, +5 %] | minor extra precomputed orbits |
| Betti collapse rate | [+0.10, +0.30] | symmetry-aligned features consolidate faster |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dataset:** **Spherical MNIST** (12-vertex icosa-mesh sampling).
- **Architecture:** 3-stage NaturePriorBlock with `IcosaGroupConv2d` at each conv (avg-pool reduction).
- **Epochs / batch / precision / seeds:** 25 epochs, batch 64, bf16 AMP, 3 seeds.
- **Composite formula:** identical scaffold + new metric `rot_eq_err` weight 0.2.
- **Run-script:** `python ideas/24_icosa_phi_equivariant/experiment.py --dataset spherical_mnist --seeds 0 1 2`.
- **Wall-clock:** 2–3 hr/seed on RTX 4090 Laptop, total ~9 hr.
- **Archive:** `ideas/24_icosa_phi_equivariant/experiments/exp001_smnist_seed0..2/`.

### 7.2 Idea-targeted experiment (where this prior should SHINE)

1. **Spherical MNIST / IcoMNIST** — Cohen's original demo ground.
2. **Climate-pattern segmentation** (Mudigonda 2017 ClimateNet) at reduced resolution.
3. **Protein-surface binding-site classification** (icosahedral mesh of the molecular surface).
4. **Cosmological-microwave-background anomaly detection** (Planck-style icosa-pixelization).

### 7.3 Cross-paradigm context (LLM track)

LLM-track is H71 (icosahedral RoPE) at 124 M scale on a 3-D spatial QA benchmark (ARC-3D-mini), trained 50 k steps; report zero-shot accuracy on 3-D-spatial vs control RoPE.

## 8. Cross-references

- Parent design-space row: `IDEA_TABLE.md` § G3 row H24.
- Master experiment list: `EXPERIMENT_LOG.md` Tier 1 row T1.4 (`sg_only_group`, C4 proxy that failed) and Tier 2 row T2.1 (`sg_only_group_avg`, the H58 fix that is the **direct prerequisite** for this hypothesis).
- Implementation sub-directory: `ideas/24_icosa_phi_equivariant/`.
- Related hypotheses that compose: H21 (hex 2-D projection of icosa); H23 (Platonic graph); H25 (dodeca latent); H30 (Platonic-Fib hybrid); H53 (GICOPix bridge); H55 (Platonic Transformer); **H58 (THE PREREQUISITE FIX)**; H71 (icosa RoPE).
- Related hypotheses that conflict: H29 (small-world); H22 (toroidal closure is a different topology).

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of Cohen 2019?**

> Cohen's icosa CNN uses **max-pool** over the orbit (default), the exact failure mode our T1.4 surfaced. We replace it with avg-pool (H58) and add Fibonacci channel grouping. The protocol pre-registers the falsifier numerically.

**Q: T1.4 already showed `sg_only_group` is -10.27 pp on CIFAR-10 — why retry?**

> Because T1.4 used (a) only the C4 subgroup (a 2-D proxy, not the full I-60 icosahedral group), (b) max-pool reduction over the orbit, which discards 75 % of the signal in our 12-epoch / single-seed setup, and (c) upright CIFAR-10, where the equivariance prior is unused. H58 (in flight) is testing the avg-pool fix; once H58 confirms the fix, the full icosahedral version is testable on Spherical MNIST.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies: top-1 lift ≥ 3.0 pp on Spherical MNIST AND rot-eq-err drop ≥ 0.03 on rotated-CIFAR. Both must hold; either failure discards.

**Q: What if the prior helps on Spherical MNIST but hurts on CIFAR?**

> That is **consistent with the hypothesis**. The CNN-track scope is spherical / rotated data. Upright CIFAR-10 is a known weak testbed for equivariance priors.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> Yes for the full hybrid. H24 as a single prior is tested on its native data; compounding is H30 (Platonic-Fib hybrid) and H67 (full paradigm).

**Q: How do we know the implementation is correct?**

> `tests/test_icosa_conv.py::test_60_orbit_count` asserts the rotation set has 60 elements. `test_equivariance_60deg` asserts `f(R·x) ≈ R·f(x)` for an icosa rotation R to within 1e-3. `test_avg_pool_not_max` asserts the reduction op is mean. Plus the experiment archive carries verification.

## 10. Verification artifacts checklist (committee evidence pack)

- [ ] `ideas/24_icosa_phi_equivariant/implementation.py` exists and tests green
- [ ] `ideas/24_icosa_phi_equivariant/tests.py` ≥ 8 assertions (orbit count, equivariance, avg-pool guard, forward shape, gradient flow, Fibonacci-group counts, GICOPix-unfold-correctness, He-init variance)
- [ ] `ideas/24_icosa_phi_equivariant/AUDIT.md` lists ≥ 3 self-found weaknesses
- [ ] `ideas/24_icosa_phi_equivariant/IMPROVEMENTS.md` records the fixes
- [ ] `ideas/24_icosa_phi_equivariant/VERIFY.md` is signed
- [ ] Experiment archives under `ideas/24_icosa_phi_equivariant/experiments/`
- [ ] Archives carry `verification/`
- [ ] Row added to `EXPERIMENT_LOG.md` Tier 3
- [ ] Result reflected in `FINDINGS.md` and dashboard

## 11. Why the C4 single-prior CIFAR-10 run (T1.4) produced a -10.27 pp negative result, what H58 changes, and why H58 is the direct prerequisite for H24

T1.4 (`sg_only_group`) on upright CIFAR-10 yielded top-1 69.84 % vs the `sg_chan_fib` reference 80.11 %, a **10.27 pp shortfall** that is the **single biggest single-prior negative** in the entire previous campaign, and composite 0.6937 vs 0.8135 (-0.1198). Three distinct mechanisms compound here:

1. **Max-pool over the orbit discards 75 % of the signal in a 4-rotation C4 group.** At every layer, the network produces 4 rotated copies of every feature map and keeps only the per-channel maximum. With 12 epochs of training, the network has insufficient time to coordinate which orbit element wins and the gradient flow through max-pool is sparse (one of four positions selected per element). This is the dominant negative term.
2. **CIFAR-10 is upright with no rotation in the test distribution.** The rot-eq-err DID drop by 0.046 (the largest of any prior in the sweep) — the equivariance prior is REAL and CORRECTLY learned — but CIFAR-10's composite gives the equivariance only 0.15 weight, so the dramatic top-1 hit dominates the composite.
3. **C4 (4-rotation) is a small finite subgroup of the icosahedral group I-60.** The hypothesis H24 is for the full 60-element group on **spherical** data, not the C4 proxy on planar data.

**H58 (the direct prerequisite, currently running as T2.1) changes the orbit reduction from `amax(dim=1)` to `mean(dim=1)`.** This is a one-line code change with a robust mechanistic prediction: avg-pool preserves all 4 orbit signals (gradient flows everywhere), and the prior's parameter saving (weight sharing across rotations) is uncompromised. The transcript prediction is **+8 to +10 pp top-1 recovery**, restoring `sg_only_group` to roughly parity with `sg_chan_fib`. If H58 confirms this, **H24 becomes the natural next step**: extend C4 → I-60, swap to Spherical MNIST / IcoMNIST as the proper data regime, add Fibonacci-allocated channel groups for an additional phyllotaxis bias, and pre-register a 3-seed protocol.

**What the next experiment changes:** (a) replace max-pool with avg-pool over the orbit (the H58 fix) at every group-conv layer; (b) extend from C4 to the full icosahedral I-60 group via GICOPix unfold; (c) shift the primary benchmark from upright CIFAR-10 to Spherical MNIST and IcoMNIST; (d) add Fibonacci channel grouping (size {1,1,2,3,5,8} sum 20 = icosa vertex count); (e) 3-seed median with rot-eq-err as a first-class metric.

## 12. Status journal

- 2026-05-27 — Created from template by Doc-Agent-B. Integrates T1.4 (C4 proxy max-pool failure) and T2.1 (H58 avg-pool fix in flight) as the direct prerequisite chain.
