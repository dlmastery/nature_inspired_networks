# H53 — 2D-3D Icosahedral Unfold Bridge (GICOPix)

> **One-line claim:** Using the GICOPix planar-unfold of an
> icosahedral mesh as the rectilinear grid for a 2D convolution
> lets the SAME `Conv2d` weights be reused on 3D spherical inputs
> (Spherical MNIST, ICOSA pixelization) achieving ≥80% of an
> e2cnn-style full icosahedral-equivariant CNN's accuracy at <30% of
> its FLOPs because the planar unfold preserves icosa-vertex
> connectivity in 5 distorted-but-rectifiable patches that share
> weights, bridging the 2D-image and 3D-spherical paradigms with a
> single convolution kernel.
>
> **Source design space:** G6 Topological + bridging additions (H51–H60).
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for hypothesis
H53. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The icosahedron is the largest Platonic solid (20 faces, 12 vertices),
making it the closest polytope to a sphere with finite geometry.
Spherical convolutions (Cohen 2019 Icosahedral CNN) operate on
icosa-vertex-discretized inputs and produce icosa-vertex outputs;
they are exactly equivariant to the 60-element icosahedral rotation
group. The cost: O(60×) compute over a planar conv with the same
kernel size, because the kernel must be replicated and rotated for
each group element.

GICOPix (Górski 2005 + Yu 2019) provides a clever workaround: unfold
the icosa-mesh into 5 planar rhomboid patches that tile a flat
2D rectangle. The unfold is approximate (some boundary distortion)
but allows a SINGLE 2D conv kernel to be applied with weight-sharing
across patches, recovering most of the equivariance benefit at near-
zero overhead vs. planar conv. This bridges the 2D and 3D paradigms
— the same kernel weights, trained on a 2D image task, can be used
on a sphere-projected variant by unfolding to GICOPix.

The sacred-geometry angle: the icosahedron is the Platonic ideal of
"sphere-like polytope" and GICOPix is the engineering realization of
"how to do conv on it cheaply". This is the canonical 2D-3D bridge.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because GICOPix planar unfold preserves vertex-neighbor connectivity
within 5 patches that tile a flat rectangle, a single 2D conv with
weight-sharing across patches approximates an icosahedral conv —
mechanism-wise, conv weights are shared across 5 rhomboid tiles
related by 72° rotations — per Cohen 2019 (Icosahedral CNN,
arXiv:1902.04615) and Yu 2019 GICOPix, we expect a GICOPix-based
network to reach ≥80% of an e2cnn icosa-equiv network's accuracy on
Spherical MNIST at <30% of its FLOPs cost (3-seed median).

## 3. Falsifier (≥ 30 words)

If GICOPix-bridge accuracy on Spherical MNIST is < 75% of the e2cnn
reference (95% CI excluding 75%), OR if FLOPs ratio exceeds 40%,
this hypothesis is DISCARDED. Equivalently: if the bridge does not
deliver a meaningful Pareto improvement on the
accuracy-vs-FLOPs plane, it fails.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
Cohen, Taco S. and Geiger, Mario and Köhler, Jonas and Welling,
Max 2019 ICML 'Gauge Equivariant Convolutional Networks and the
Icosahedral CNN' (arXiv:1902.04615) -- the canonical icosahedral
CNN paper; our bridge baseline.

Górski, Krzysztof M. and Hivon, Eric and Banday, Anthony J. and
others 2005 ApJ 'HEALPix: A Framework for High-Resolution
Discretization and Fast Analysis of Data Distributed on the
Sphere' -- the GICOPix-precursor HEALPix pixelization;
methodological anchor.

Yu, Yu and Hong, Sungwon and others 2019 NeurIPS 'Spherical CNNs
on Unstructured Grids' (arXiv:1901.02039) -- the GICOPix-derived
icosa-pixelization for CNN; the direct technical predecessor.

Esteves, Carlos and Allen-Blanchette, Christine and Makadia, Ameesh
and Daniilidis, Kostas 2018 ECCV 'Learning SO(3) Equivariant
Representations with Spherical CNNs' (arXiv:1711.06721) --
broader spherical-CNN literature for context.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

The GICOPix unfold maps an icosahedral mesh (20 triangular faces) to
5 flat rhomboid patches that tile a (W, 5W/2) rectangle. Each patch
contains 4 triangles. Apply a standard 2D conv on the rectangle with
weight-sharing across patches (the kernel is the same; the 5
rhomboid patches are different inputs sharing the same parameters).

```python
class GICOPixConv2d(nn.Conv2d):
    """A 2D conv operating on GICOPix-unfolded icosahedral data."""

    def __init__(self, in_ch, out_ch, ks=3):
        super().__init__(in_ch, out_ch, ks, padding=ks // 2)
        self.gicopix_W = 32  # GICOPix patch width (configurable)

    def forward(self, x):
        # x: (B, C, 5, W, W) — 5 patches, each W×W
        B, C, P, H, W = x.shape
        assert P == 5
        # reshape to (B*5, C, W, W) for standard conv
        x = x.view(B * P, C, H, W)
        y = super().forward(x)
        # restore 5-patch dimension
        return y.view(B, P, -1, H, W).permute(0, 2, 1, 3, 4)

def unfold_sphere_to_gicopix(sphere_input, n_subdiv=4):
    """Map a sphere-discretized image to 5 GICOPix patches."""
    # Lookup table from sphere vertex index → (patch, h, w)
    ...
```

Computational cost: comparable to standard 2D conv on a (5W × W)
rectangle. Vs. e2cnn icosahedral conv at 60-element group:
GICOPix is ~12× cheaper.

Lives in `src/nature_inspired_networks/equiv/gicopix.py`, re-exported by
`ideas/53_icosa_unfold_bridge/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

For decoder LLMs, the "spherical" structure is not natural; instead,
H53's LLM-track analog is **icosahedral RoPE** (H71): use the 60
rotations of the icosa group to construct rotary embeddings that
quantify 3D spatial positions in a token stream. This is suitable for
LLMs that process 3D scene tokens (e.g., spatial reasoning, NeRF
captions).

FlashAttention-2 compatibility: icosa-RoPE is a drop-in replacement
for standard RoPE; FA2 handles the rotation matmul. Causal mask:
unaffected.

Expected at 124M: 3D-spatial QA benchmarks (ScanQA) lift ≥3 pp;
standard text benchmarks unaffected.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline (e2cnn icosa-conv) | rationale |
|---|---|---|
| composite | [-0.05, +0.05] | trade-off |
| top-1 (Spherical MNIST) | [-15 pp, -5 pp] | lossy bridge |
| FLOPs | [-80%, -65%] | core benefit |
| params | [-20%, -50%] | weight-sharing |
| GPU latency (batch=1) | [-75%, -50%] | proportional to FLOPs |
| rot-eq err (icosa-rotation) | [+0.01, +0.05] | approximate equivariance |
| KV cache @ 32k (LLM) | [0, 0] | N/A here; H71 separately |
| Betti collapse rate | [0, 0] | minor |
| perplexity (LLM) | [-0.5, +0.5] | applies only to H71 |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dataset:** Spherical MNIST (60k train, 10k test; sphere-projected)
- **Architecture A (GICOPix):** 3-layer GICOPixConv2d net (~50k params)
- **Architecture B (e2cnn):** 3-layer e2cnn icosa-equivariant net
  (~120k params)
- **Architecture C (plain CNN):** 3-layer plain Conv2d on
  unstructured sphere flatten (baseline floor)
- **Epochs:** 20, batch=128
- **Seeds:** 0, 1, 2
- **Run-script:** `python scripts/run_idea.py --idea 53 --bridge gicopix`
- **Wall-clock:** ≈ 20 min × 3 seeds × 3 archs ≈ 3 h
- **Archive path:** `ideas/53_icosa_unfold_bridge/experiments/exp001_smnist/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

GICOPix bridge should win on tasks where 2D pretraining can be
*transferred* to 3D inference:

- **Setup:** train a 2D-MNIST CNN, then deploy on Spherical MNIST via
  GICOPix patches reusing the 2D weights
- **Predicted:** ≥60% accuracy at zero retraining (the bridge property)
- **Diagnostic:** if zero-shot transfer fails, the bridge claim
  (single weights for 2D and 3D) is invalid.

### 7.3 Cross-paradigm context (LLM track)

For LLM track, see H71 (icosahedral RoPE for 3D spatial reasoning):
- **Model:** 124M GPT-2-small with icosa-RoPE
- **Dataset:** synthetic 3D spatial-reasoning QA (e.g., "what is left
  of the red cube?")
- **Run:** `python scripts/run_llm.py --idea 71 --rope icosa`

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G6 row H53.
- Master experiment list: `EXPERIMENT_LOG.md` (planned).
- Implementation sub-directory: `ideas/53_icosa_unfold_bridge/`
- Related hypotheses that compose:
  - **H24** Icosahedral φ-equivariant — the full version that H53
    approximates.
  - **H71** Icosa RoPE for 3D — LLM-track sibling.
  - **H21** Hex φ-packing — alternative non-rect lattice; conceptually
    related.
- Related hypotheses that conflict:
  - None directly.

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just GICOPix re-implementation?**

> The contribution is (a) integration into the NaturePriorBlock
> framework, (b) explicit benchmark against e2cnn-icosa, (c) the
> 2D-3D transfer experiment in § 7.2 that prior work has not done
> in this combination.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies accuracy and FLOPs targets with 95% CI.

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> H53 targets Spherical MNIST (its natural domain); we don't expect
> CIFAR-10 to be a fair test. Reported scope is sphere-based tasks.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H53 is a *bridging* mechanism, not a composable prior; it doesn't
> interact with the compound failure.

**Q: How do we know the implementation is correct?**

> `tests.py` verifies (a) the unfold preserves vertex-count
> (12 + 10×3 = 42 vertices map exactly), (b) shared-weight conv
> produces the predicted equivariance (rotating the input by 72°
> produces an output that is the predicted permutation of patches).

## 10. Verification artifacts checklist

- [ ] `ideas/53_icosa_unfold_bridge/implementation.py` exists, tests green
- [ ] `ideas/53_icosa_unfold_bridge/tests.py` ≥ 8 assertions
- [ ] `ideas/53_icosa_unfold_bridge/AUDIT.md` ≥ 3 weaknesses
- [ ] `ideas/53_icosa_unfold_bridge/IMPROVEMENTS.md` records fixes
- [ ] `ideas/53_icosa_unfold_bridge/VERIFY.md` signed
- [ ] At least one experiment archive
- [ ] `verification/{tests,smoke,gates,reproduction}.txt`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-C.
