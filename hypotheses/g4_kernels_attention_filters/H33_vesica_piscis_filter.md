# H33 — Vesica Piscis Filter

> **One-line claim:** A multi-path convolutional filter where each path's effective receptive field is a circular disc of radius r_k spaced at φ-multiples (the Vesica Piscis / Flower-of-Life overlap pattern) raises top-1 on natural-image benchmarks and improves multi-scale feature detection versus a parameter-matched standard multi-branch baseline.
>
> **Source design space:** G4 Kernels / Attention / Filters (H31–H40).
>
> **Implementation status (this repo):** `○ not started`.

This document is the committee-grade design write-up for hypothesis H33.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The Vesica Piscis — the lens-shaped intersection of two equal circles whose centers each lie on the other's circumference — is one of the most ancient geometric symbols, appearing in early Christian iconography (Christ-in-mandorla), Pythagorean mathematics, and modern Flower-of-Life sacred-geometry symbolism. Mathematically, the Vesica Piscis encodes √3 (the ratio of its height to width) and √2 (in nested-circle Flower-of-Life recursions), and it is the basis pattern of the Flower-of-Life — a tiling of 19 overlapping circles whose intersections produce all five Platonic-solid vertices when projected. The pattern represents **multi-scale overlapping reception**: at every point, multiple-radius receptive fields intersect.

For deep learning, multi-path / multi-scale convolutional architectures (Inception, Big-Little Net, HRNet, OctConv) demonstrate that **overlapping receptive fields at different scales** improve accuracy by simultaneously detecting features at multiple scales. The Vesica Piscis filter is the explicit geometric realization of this principle: each conv path has a circular mask of radius r_k where r_k follows φ-multiples, and the centers of the paths are offset by half-radius to produce Vesica-Piscis intersections. Compared to Inception (parallel 1×1, 3×3, 5×5), the Vesica Piscis filter uses **continuous-radius circular masks** rather than square kernels, which align better with the rotation-isotropic statistics of natural images (Olshausen 1996).

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** the Vesica Piscis pattern is the canonical overlap of multi-scale circular receptive fields and the Flower-of-Life tiling spans multiple φ-scales simultaneously, a 4-path Vesica Piscis convolutional filter with radii {1, φ, φ², φ³} pixels raises CIFAR-100 top-1 by ≥ +0.8 pp and improves Inception-style multi-scale feature detection (measured by per-scale activation entropy) versus a parameter-matched Inception baseline, per the mechanism of Szegedy 2015 (Inception, arXiv:1512.00567) and Olshausen 1996.

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on CIFAR-100, the Vesica Piscis variant fails to lift top-1 by ≥ 0.4 pp AND fails to improve per-scale activation entropy by ≥ 0.05 nats relative to Inception baseline at matched params, this hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Szegedy, C., et al. 2015 CVPR 'Going Deeper with Convolutions'
(arXiv:1409.4842) and Szegedy 2016 CVPR 'Rethinking the Inception
Architecture' (arXiv:1512.00567) — Inception / multi-branch reference
H33 specializes to Vesica Piscis radii.

Olshausen, B. A., Field, D. J. 1996 Nature — natural-image
log-spiral / circular-receptive-field motivation.

Chen, Y., et al. 2019 CVPR 'Drop an Octave: Reducing Spatial
Redundancy in CNNs with Octave Convolution' (arXiv:1904.05049) —
OctConv multi-scale reference, comparator.

Wang, X., et al. 2019 ICCV 'Big-Little Net: An Efficient Multi-Scale
Feature Representation for Visual and Speech Recognition'
(arXiv:1807.03848) — multi-scale comparator.

Krizhevsky 2009 — CIFAR-100 dataset citation.
```

## 5. Mechanism

### 5.1 CNN track

Implement `VesicaPiscisConv2d` as 4 parallel convolutions with circular masks of radii {1, φ, φ², φ³} ≈ {1, 1.618, 2.618, 4.236} pixels, centered at appropriate Vesica-Piscis offsets.

```python
# ideas/33_vesica_piscis_filter/implementation.py
PHI = (1+5**0.5)/2

def circular_mask(k, r):
    """Return (k, k) binary mask: 1 inside circle of radius r centered at k//2."""
    y, x = torch.meshgrid(torch.arange(k), torch.arange(k), indexing='ij')
    cx = cy = k // 2
    return ((y - cy)**2 + (x - cx)**2 <= r**2).float()

class VesicaPiscisConv2d(nn.Module):
    def __init__(self, C_in, C_out, n_paths=4):
        super().__init__()
        radii = [PHI ** i for i in range(n_paths)]
        # max kernel size determined by largest radius
        k = 2 * int(math.ceil(radii[-1])) + 1
        masks = [circular_mask(k, r).view(1, 1, k, k) for r in radii]
        self.k = k
        self.register_buffer("masks", torch.cat(masks, dim=0))
        # one conv per path with shared (C_in, C_out//n_paths) weights
        self.convs = nn.ModuleList([
            nn.Conv2d(C_in, C_out // n_paths, k, padding=k//2)
            for _ in range(n_paths)
        ])
        for i, c in enumerate(self.convs):
            c.weight.data *= self.masks[i:i+1]

    def forward(self, x):
        outs = []
        for i, c in enumerate(self.convs):
            # re-apply mask to weight (preserves through training)
            w_masked = c.weight * self.masks[i:i+1]
            outs.append(F.conv2d(x, w_masked, c.bias, padding=self.k//2))
        return torch.cat(outs, dim=1)
```

- Input shape: `(B, C_in, H, W)`. Output: `(B, C_out, H, W)`.
- Params: 4 paths × `(C_in, C_out/4, k, k)`; total per-path slightly less than dense `(C_in, C_out, k, k)` because masks zero ~20-40 % of weights.
- FLOPs: similar to a parameter-matched 5x5 conv.
- Init: He init, post-multiplied by mask.

### 5.2 LLM track

For decoder-only Transformers, Vesica Piscis maps onto **multi-scale window attention**: each attention head has a different window radius {1, φ, φ², φ³} (rounded to integers ≈ {1, 2, 3, 4}) of tokens it attends to. This is a sparse-multiscale variant of H32 Fibottention, with each head having a CONTIGUOUS window rather than DILATED stride.

```python
class VesicaWindowMHA(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.windows = [int(round(PHI ** i)) for i in range(heads)]
        ...
```

Expected at 124 M scale: **perplexity within 0.3 of dense**, attention FLOPs **-60 to -80 %**, latency **-30 to -50 %**.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (CIFAR-100) | [+0.005, +0.020] | multi-scale prior pays off on diverse imagery |
| top-1 (CIFAR-100, primary) | [+0.8 pp, +2.5 pp] | direct claim |
| Per-scale activation entropy | [+0.05, +0.15] nats | direct geometric prediction |
| params | [-15 %, -5 %] | mask zeros ~20-40 % of weights |
| FLOPs | [-10 %, +5 %] | mask still costs apply ops |
| GPU latency | [×1.0, ×1.4] | extra paths cost |
| Perplexity (LLM) | [+0.0, +0.3] | retention claim |
| KV cache @ 32 k (LLM) | [-50 %, -75 %] | windowed sparse |
| Betti collapse rate | [+0.05, +0.20] | multi-scale aids consolidation |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** CIFAR-100.
- **Architecture:** NaturePriorBlock with `VesicaPiscisConv2d` replacing 3×3 conv at each stage.
- **Epochs / batch / precision / seeds:** 25 epochs, batch 128, bf16, 3 seeds.
- **Composite:** top-1 (0.5), latency (0.2), params (0.15), entropy (0.15).
- **Run-script:** `python ideas/33_vesica_piscis_filter/experiment.py --seeds 0 1 2`.
- **Wall-clock:** ~25 min/seed × 3 = 75 min.
- **Archive:** `ideas/33_vesica_piscis_filter/experiments/exp001_cifar100_seed0..2/`.

### 7.2 Idea-targeted experiment

1. **CIFAR-100** as primary multi-scale benchmark.
2. **PathMNIST / dermatology images** where cell sizes vary multiscale.
3. **Cityscapes-mini segmentation** where multi-scale receptive matters.

### 7.3 Cross-paradigm context (LLM track)

WikiText-103 124 M with Vesica-window MHA. 50 k steps, perplexity + latency at multi-context lengths.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G4 row H33.
- Master: planned Tier 1/2.
- Sub-dir: `ideas/33_vesica_piscis_filter/`.
- Composes: H07 (φ multi-scale FPN), H32 (Fibottention), H38 (fractal golden filter).
- Conflicts: none.

## 9. Committee Q&A

**Q: Why isn't this just Inception with different kernel sizes?**

> Vesica Piscis uses CIRCULAR masks (continuous-radius) on a single common kernel size; Inception uses SQUARE kernels at multiple sizes (1×1, 3×3, 5×5). The geometric prior is circle-rotation-equivariant; Inception's is not.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 gives top-1 ≥ +0.4 pp AND entropy ≥ +0.05 nats. Both must hold.

**Q: What if the prior helps on CIFAR-100 but is no better than Inception on ImageNet?**

> Tested via § 7.2 targeted dataset list; scaling to ImageNet is T2-T3.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> Single-prior test; compounding is H50/H67.

**Q: How do we know the implementation is correct?**

> `tests/test_vesica.py::test_circular_mask_radius` asserts mask matches `(y-cy)² + (x-cx)² ≤ r²`. `test_phi_spaced_radii` asserts radii follow φ powers. `test_rotation_equivariance_45deg` asserts circular masks are rotation-equivariant within rasterization tolerance. Plus archive verification.

## 10. Verification artifacts checklist

- [ ] `ideas/33_vesica_piscis_filter/implementation.py` tests green
- [ ] `ideas/33_vesica_piscis_filter/tests.py` ≥ 6 assertions
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

**LOW-MED.** Multi-scale Inception-style filters work (Szegedy 2015 arXiv:1409.4842), but the specific claim — that CIRCULAR masks with φ-spaced radii outperform parameter-matched Inception square kernels — has no precedent. The "rotation isotropy of natural-image statistics" argument is wrong (natural images have STRONG horizontal/vertical bias from gravity-aligned scenes; oriented Gabor RFs reflect this — Olshausen 1996). Multi-scale works for SCALE coverage, not for "rotation isotropy".

### Mechanism scrutiny

The φ-spaced radii {1, 1.618, 2.618, 4.236} rasterized to integer-pixel circular masks become {radius-1 disc = 5 pixels, radius-φ disc = 9 pixels, radius-φ² disc = 21 pixels, radius-φ³ disc = 57 pixels}. The φ-spacing is ENTIRELY destroyed by integer rasterization: a disc of radius 1.618 contains the same 9 pixels as a disc of radius 1.5 or radius 1.9. The "continuous-radius" claim is false at any practical kernel size. To meaningfully discriminate φ from non-φ radii, the kernel must be ≥ 16×16 (where radius 4.236 vs 4.5 differ by 1-2 pixels in disc area). At k=9 (the implementation's largest), the φ-spacing degenerates to coarse {tiny, small, medium, large} discs — same as ANY 4-stage radius progression.

### Confounds (≥2)

1. **Channel split confound.** `C_out // n_paths` per branch reduces each branch's expressive capacity. A dense `(C_in, C_out, 5, 5)` conv has more capacity than 4× `(C_in, C_out/4, 9, 9)` masked convs at matched params, depending on the mask sparsity. Any measured Δ may come from channel-grouping effects (similar to ResNeXt cardinality; Xie 2017 arXiv:1611.05431) rather than from the radius progression.
2. **Mask-zeroed-weight confound.** The implementation re-applies the mask at every forward (`c.weight * self.masks[i]`), which means the gradient through the masked positions is zero — those weights never update from their He init. This is a known failure mode in masked-conv literature (the weights drift due to weight decay even though their masked output is zero, then the mask suddenly applies and the kernel changes shape over training). Control: hard-prune masked positions at init only, do NOT re-apply mask.

### Numerology / specificity check

{1, φ, φ², φ³} ≈ {1, 1.6, 2.6, 4.2} is a φ-geometric progression. Compare to {1, 2, 3, 4} arithmetic, {1, 2, 4, 8} powers-of-2, {1, √2, 2, 2√2}, {1, e, e², e³}. At k=9 raster grid, all of these produce the SAME 4 discrete disc areas (small, medium, large, max). The φ-specificity is undetectable at any kernel size used in practice. **High numerology score.**

### Literature precedent — kernel/attention design is a crowded field

Multi-branch / multi-scale conv designs include: Inception (Szegedy 2015 arXiv:1409.4842), Inception-ResNet (Szegedy 2016 arXiv:1602.07261), ResNeXt (Xie 2017 arXiv:1611.05431), Big-Little Net (Wang 2019 arXiv:1807.03848), OctConv (Chen 2019 arXiv:1904.05049), HRNet (Sun 2019 arXiv:1908.07919), Res2Net (Gao 2021 arXiv:1904.01169). None use circular masks; all use square kernels at different sizes. The literature consensus is that "circular vs square" is a third-order effect; the first-order effect is scale coverage.

### Expected effect size (90% CI a priori)

[-0.2 pp, +0.6 pp] CIFAR-100 top-1 vs parameter-matched Inception. The author's [+0.8, +2.5] is too optimistic. Most likely outcome: ~match Inception within 0.3 pp.

### Minimum-distinguishing experiment

Compare: (a) Vesica Piscis circular {1, φ, φ², φ³}, (b) circular {1, 2, 3, 4} arithmetic, (c) circular {1, 2, 4, 8} geometric, (d) Inception square {1, 3, 5, 7}, all at matched params and FLOPs. If (a) > (b), (c) by ≥ 0.3 pp, φ-spacing is non-null. If (a) ≈ (d), the "circular vs square" claim is null.

### Verdict
NUMEROLOGY — at any rasterization grid k ≤ 16, the φ-spaced radii {1, 1.6, 2.6, 4.2} discretize to the same disc set as ANY 4-step monotonic radius progression. The "circular = rotation isotropic" argument misrepresents natural-image statistics (Olshausen RFs are oriented Gabors, NOT isotropic).
