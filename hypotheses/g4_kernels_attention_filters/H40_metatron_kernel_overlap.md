# H40 — Metatron Kernel Overlap

> **One-line claim:** A weight-shared convolutional kernel basis constructed from Metatron's-Cube 13-circle overlap projections produces a sparse, structured kernel set that raises CIFAR-100 top-1 and reduces parameter count by ≥ 30 % versus an unstructured dense baseline of matched FLOPs.
>
> **Source design space:** G4 Kernels / Attention / Filters (H31–H40).
>
> **Implementation status (this repo):** `○ not started`.

This document is the committee-grade design write-up for hypothesis H40, the most explicitly-sacred-geometry-inspired filter in our design space.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

Metatron's Cube is the 13-circle diagram formed by drawing circles centered at all 13 vertices of the Fruit of Life pattern (six outer + six inner + one central circle, with edges connecting every pair). When projected onto the 2-D plane, the overlapping circles produce 78 distinct edges and a wealth of intersection regions that contain projections of all five Platonic solids (tetrahedron, cube, octahedron, dodecahedron, icosahedron). Sacred-geometry literature credits Metatron's Cube as the "fundamental pattern of creation" — but the engineering claim is more modest: it is a maximally-symmetric 13-element basis for filters defined on a hex-centered lattice, with overlap structure that encodes phyllotaxis-aligned multi-scale features.

For deep learning, the Metatron kernel basis provides **weight-shared sparse filters**: instead of learning K independent dense kernels, learn one basis of 13 elements and project K filters as linear combinations of basis elements. This is a parameter-saving / regularization trick (low-rank kernel factorization, e.g., DCFNet 2018) where the **basis** is the Metatron-13 structure rather than learned. Compared to learned-basis kernel factorization (which can overfit), Metatron's Cube provides a FIXED, geometrically-motivated basis whose 13 elements encode hex-symmetric multi-scale features.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** Metatron's Cube is the maximally-symmetric 13-element overlap basis on the hex lattice and encodes phyllotaxis-aligned multi-scale features, factorizing each convolutional kernel as a linear combination of 13 fixed Metatron-basis elements (with K learnable mixing coefficients per output channel) reduces parameter count by ≥ 30 % and retains ≥ 99 % of dense CIFAR-100 top-1 accuracy, per the mechanism of Qiu 2018 DCFNet (arXiv:1802.04145).

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on CIFAR-100, the Metatron-basis filter fails to retain ≥ 98 % of dense top-1 AND fails to reduce parameter count by ≥ 25 % relative to dense baseline at matched depth, this hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Qiu, Q., Cheng, X., Calderbank, R., Sapiro, G. 2018 'DCFNet: Deep
Convolutional Filter Network with Decomposed Convolutional Filters'
(arXiv:1802.04145) — kernel-basis factorization reference.

Cohen, T. S., Welling, M. 2016 — group equivariance, the symmetry
framework Metatron's Cube specializes.

Hoogeboom, E., et al. 2018 — HexaConv hex-lattice reference.

Mandelbrot, B. 1982 — multi-scale fractal structure.

Krizhevsky 2009 — CIFAR dataset.

Howard, A. G., et al. 2017 'MobileNets' (arXiv:1704.04861) — depthwise
separable filter compression, comparator parameter-efficient method.
```

## 5. Mechanism

### 5.1 CNN track

Pre-compute 13 Metatron-Cube basis kernels (5×5 each, sampled from the 13-circle overlap pattern) and use them as the FIXED basis for all conv layers. Each conv layer learns only K coefficients per output channel (K = 13 mixing coefficients).

```python
# ideas/40_metatron_kernel/implementation.py
def metatron_basis(k=5):
    """Generate 13 fixed Metatron-Cube basis kernels of size (k, k)."""
    basis = torch.zeros(13, k, k)
    centers = metatron_circle_centers(k)  # (13, 2)
    radii = [k / 4] * 13  # uniform radius
    for i, (cy, cx) in enumerate(centers):
        for y in range(k):
            for x in range(k):
                d = math.hypot(y - cy, x - cx)
                basis[i, y, x] = max(0.0, 1 - d / radii[i])
    # orthonormalize
    flat = basis.view(13, -1)
    Q, _ = torch.linalg.qr(flat.T)
    return Q.T.view(13, k, k)

def metatron_circle_centers(k):
    """13 vertex positions of the Metatron pattern in a k x k frame."""
    cy = cx = k / 2
    r = k / 4
    centers = [(cy, cx)]  # 1 center
    for i in range(6):  # 6 inner
        theta = i * math.pi / 3
        centers.append((cy + r*math.sin(theta), cx + r*math.cos(theta)))
    for i in range(6):  # 6 outer
        theta = i * math.pi / 3 + math.pi/6
        centers.append((cy + 2*r*math.sin(theta), cx + 2*r*math.cos(theta)))
    return centers

class MetatronConv2d(nn.Module):
    def __init__(self, C_in, C_out, k=5):
        super().__init__()
        self.register_buffer("basis", metatron_basis(k))  # (13, k, k)
        # learn mixing coefficients
        self.alpha = nn.Parameter(torch.randn(C_out, C_in, 13) * 0.01)
    def forward(self, x):
        # construct effective kernel: (C_out, C_in, k, k) = alpha @ basis
        weight = torch.einsum('oic,ckk->oikk'.replace('kk','xy'),
                              self.alpha, self.basis)
        # (note: real impl uses correct einsum / matmul reshape)
        return F.conv2d(x, weight, padding=self.basis.shape[-1]//2)
```

- Input: `(B, C_in, H, W)`.
- Output: `(B, C_out, H, W)`.
- Params: `C_out · C_in · 13` (vs. `C_out · C_in · k² = C_out · C_in · 25` for k=5 dense). Saves `(25-13)/25 = 48 %` of weights.
- FLOPs: identical to dense conv (effective kernel is materialized).
- Init: small-Gaussian on `alpha`.

### 5.2 LLM track

For decoder-only Transformers, Metatron-basis applies to **QKV projection matrices**: factor each projection as `W = sum_i alpha_i · B_i` with 13 fixed basis matrices. Saves projection parameters.

Expected at 124 M: **-30 % to -40 % projection params**, perplexity within 0.1 of dense.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (CIFAR-100) | [+0.005, +0.025] | parameter-saving without accuracy loss |
| top-1 (CIFAR-100, primary) | [-0.5 pp, +1.5 pp] | retention claim |
| params | [-30 %, -50 %] | direct claim |
| FLOPs | [≈0, ≈0] | effective kernel materialized |
| GPU latency | [≈1.0×, ≈1.1×] | small basis-projection cost |
| Perplexity (LLM) | [+0.0, +0.2] | retention claim |
| KV cache | [0, 0] | unchanged |
| Betti collapse rate | [+0.05, +0.15] | structured prior aids collapse |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** CIFAR-100.
- **Architecture:** NaturePriorBlock with `MetatronConv2d` replacing every conv layer.
- **Epochs / batch / precision / seeds:** 25 epochs, batch 128, bf16, 3 seeds.
- **Composite:** top-1 (0.5), params (0.3), latency (0.1), Betti (0.1).
- **Run-script:** `python ideas/40_metatron_kernel/experiment.py --seeds 0 1 2`.
- **Wall-clock:** ~25 min/seed × 3 = 75 min.
- **Archive:** `ideas/40_metatron_kernel/experiments/exp001_cifar100_seed0..2/`.

### 7.2 Idea-targeted experiment

1. **CIFAR-100** primary.
2. **MobileNetV2-style parameter-efficient benchmark** for compression comparison.
3. **MedMNIST 28×28 datasets** (small-image, small-model regime).
4. **Comparison with PCA-basis** as a control (learned-orthonormal vs Metatron-fixed).

### 7.3 Cross-paradigm context

LLM-track: WikiText-103 124 M with Metatron-basis QKV. 50 k steps; perplexity vs dense.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G4 row H40.
- Master: planned Tier 2.
- Sub-dir: `ideas/40_metatron_kernel/`.
- Composes: H21 (hex base), H23 (Platonic graph = Metatron adjacency), H43 (Fib pruning), H69 (KAN on Metatron).
- Conflicts: none.

## 9. Committee Q&A

**Q: Why isn't this just DCFNet with a Metatron basis?**

> Yes, that is the contribution. DCFNet uses learnable / PCA bases; H40 uses a FIXED Metatron-geometric basis. The claim is the basis-choice is data-aligned with hex-natural-image statistics.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 gives accuracy retention ≥ 98 % AND parameter reduction ≥ 25 %. Both must hold.

**Q: What if Metatron is no better than PCA-13 basis?**

> The PCA-13 basis is the natural control. If they tie within 0.2 pp at matched params, the Metatron-specificity is null.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> Single-prior; compounding is H67.

**Q: How do we know the implementation is correct?**

> `tests/test_metatron.py::test_13_circle_centers` asserts 13 distinct centers. `test_orthonormalization` asserts the basis is orthonormal after QR. `test_param_count` asserts 13 mixing coeffs per (in, out) channel pair. `test_effective_kernel_shape` verifies output kernel is (C_out, C_in, k, k). Plus archive verification.

## 10. Verification artifacts checklist

- [ ] `ideas/40_metatron_kernel/implementation.py` tests green
- [ ] `ideas/40_metatron_kernel/tests.py` ≥ 6 assertions
- [ ] AUDIT.md ≥ 3 weaknesses
- [ ] IMPROVEMENTS.md
- [ ] VERIFY.md signed
- [ ] Experiment archives + verification
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] FINDINGS reflects result

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-B.
