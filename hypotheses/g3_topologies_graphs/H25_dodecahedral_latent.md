# H25 — Dodecahedral Latent

> **One-line claim:** Projecting the penultimate latent vector onto the 20 vertices of a unit-dodecahedron and normalising to that vertex set improves out-of-distribution detection AUC and rotation-stability versus an unconstrained latent of matched dimension.
>
> **Source design space:** G3 Topologies & Graphs (H21–H30).
>
> **Implementation status (this repo):** `○ not started`.

This document is the committee-grade design write-up for hypothesis H25.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The dodecahedron is the dual of the icosahedron — 20 vertices, 12 pentagonal faces, 30 edges — and is the largest Platonic solid by face count. It famously appears in Roman dodecahedron artifacts (likely calendrical instruments), Plato's Timaeus (associated with the cosmos), and pyrite crystal habits. Mathematically, the dodecahedron is the canonical realization of icosahedral symmetry on vertices instead of faces; its 20 vertices form three concentric tetrahedra and span 3-space with golden-ratio coordinates `(±1, ±1, ±1), (0, ±1/φ, ±φ), (±1/φ, ±φ, 0), (±φ, 0, ±1/φ)`. Every vertex coordinate is a power of φ — the dodeca IS the geometric realization of the golden ratio in 3-space.

For deep learning, projecting a latent vector onto the dodecahedral vertex set gives a 20-codeword **discrete-ish bottleneck** that is rotation-equivariant under the icosahedral group, has built-in φ scaling, and is exactly the right dimensionality (20) for many CIFAR-100 / ImageNet-100 / few-shot benchmarks. The motivation parallels vector-quantization (VQ-VAE) and prototype networks: a small fixed-vertex codebook regularizes the latent space, encourages clustering, and provides an interpretable readout (each dodeca vertex is a Platonic "prototype"). Adding the golden-ratio coordinate structure layers in an explicit φ prior that supports our broader research program.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** the dodecahedron is the maximum-vertex Platonic solid with φ-coordinate structure and 60-fold icosahedral symmetry, projecting the penultimate latent vector onto 20 dodeca-vertex prototypes raises out-of-distribution-detection AUC on CIFAR-10 vs CIFAR-100-as-OOD by ≥ +2 pp and reduces latent-rotation-instability by ≥ 0.05 relative to an unconstrained latent of dimension 20, per the mechanism of van den Oord 2017 (VQ-VAE; arXiv:1711.00937) and Cohen 2019 icosahedral equivariance.

## 3. Falsifier (≥ 30 words)

If, at 3-seed median, the dodecahedral-latent variant fails to lift OOD-AUC by ≥ 1.0 pp on (CIFAR-10 train, CIFAR-100 OOD) AND fails to reduce latent-rotation-instability by ≥ 0.03 relative to the unconstrained baseline, this hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
van den Oord, A., Vinyals, O., Kavukcuoglu, K. 2017 NeurIPS
'Neural Discrete Representation Learning' (arXiv:1711.00937) — the
canonical VQ-VAE / prototype-bottleneck reference; supports the
"discrete prototype regularizes the latent" claim.

Cohen, T. S., et al. 2019 ICML 'Gauge Equivariant Convolutional
Networks and the Icosahedral CNN' (arXiv:1902.04615) — icosahedral
equivariance (the dodecahedron is dual to the icosahedron).

Snell, J., Swersky, K., Zemel, R. 2017 NeurIPS 'Prototypical Networks'
(arXiv:1703.05175) — prototype-based classification motivation.

Hendrycks, D., Gimpel, K. 2017 ICLR 'A Baseline for Detecting
Misclassified and Out-of-Distribution Examples in Neural Networks'
(arXiv:1610.02136) — OOD-AUC evaluation methodology.

Huh, M., Cheung, B., Wang, T., Isola, P. 2024 ICML 'The Platonic
Representation Hypothesis' (arXiv:2405.07987) — the meta-argument that
sufficiently large networks converge to a Platonic representation;
this hypothesis provides an explicit Platonic prior.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

After the final pooled feature `f ∈ R^d` (`d = 128` typical), add a learnable projection `W ∈ R^{d × 20}` to produce coefficients `c = W·f`, then compute the latent as a soft assignment to dodecahedral vertices: `z = Σ_i softmax(c)_i · v_i` where `v_i ∈ R^3` are the 20 dodeca vertices in φ-coordinate form.

- Input: `(B, d)` post-pool feature.
- Output: `(B, 3)` projected onto dodeca, plus `(B, 20)` soft-assignment vector for classification head.
- Params: `d·20 = 2560` per dodeca head.
- FLOPs: negligible.

```python
# ideas/25_dodeca_latent/implementation.py
PHI = (1 + 5**0.5) / 2
def dodeca_vertices():
    """20 dodecahedron vertices in golden-ratio coordinates."""
    v = []
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                v.append([s1, s2, s3])
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            v.append([0, s1/PHI, s2*PHI])
            v.append([s1/PHI, s2*PHI, 0])
            v.append([s2*PHI, 0, s1/PHI])
    return torch.tensor(v, dtype=torch.float32)  # (20, 3)

class DodecaLatent(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, 20)
        self.register_buffer("V", dodeca_vertices())
    def forward(self, f):
        c = self.proj(f).softmax(dim=-1)
        z = c @ self.V  # (B, 3) in dodeca space
        return z, c
```

### 5.2 LLM track (decoder-only Transformer)

For decoder-only Transformers, dodeca-latent is the **last-layer auxiliary head**: a 20-vertex prototype layer on top of the residual stream that provides an interpretable "Platonic readout" useful for analysis (e.g., does the network's final state cluster on dodeca vertices?). Combined with H49 (PRH alignment loss).

- Slots in: after final layernorm, parallel to LM head.
- Causal-mask preservation: ✓ (parallel head).
- FlashAttention-2 compatibility: ✓.
- Pseudocode:

```python
class DodecaReadout(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, 20, bias=False)
        self.register_buffer("V", dodeca_vertices())
    def forward(self, h):  # h: (B, T, d)
        c = self.proj(h).softmax(dim=-1)
        z = c @ self.V
        return z  # (B, T, 3)
```

- Expected impact at 124 M: **interpretability boost** (UMAP shows clustering on dodeca vertices per H49); perplexity neutral.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (CIFAR-10) | [-0.005, +0.010] | primary metric is OOD-AUC not top-1 |
| top-1 (CIFAR-10) | [-1.0 pp, +0.5 pp] | bottleneck may slightly hurt accuracy |
| OOD-AUC (CIFAR-10 vs CIFAR-100) | [+2.0 pp, +5.0 pp] | direct claim, prototype regularization |
| latent-rotation-instability | [-0.05, -0.10] | dodeca symmetry stabilizes rotation |
| params | [+1 %, +3 %] | small extra projection |
| FLOPs | [+0.5 %, +1 %] | negligible |
| GPU latency (batch=1) | [≈1.0×, ≈1.05×] | trivial |
| Betti collapse rate | [+0.10, +0.30] | discrete bottleneck accelerates collapse |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dataset:** CIFAR-10 train, CIFAR-100 as OOD (Hendrycks 2017 protocol).
- **Architecture:** NaturePriorBlock + DodecaLatent head replacing the standard global-pool → fc head.
- **Epochs / batch / precision / seeds:** 12 epochs, batch 128, bf16 AMP, 3 seeds.
- **Composite formula:** weighted sum top-1 (0.4), OOD-AUC (0.4), latency (0.1), params (0.1).
- **Run-script:** `python ideas/25_dodeca_latent/experiment.py --seeds 0 1 2`.
- **Wall-clock:** 12 min/run × 3 seeds ≈ 36 min on RTX 4090 Laptop.
- **Archive:** `ideas/25_dodeca_latent/experiments/exp001_cifar_seed0..2/`.

### 7.2 Idea-targeted experiment (where this prior should SHINE)

1. **Few-shot CIFAR-FS / mini-ImageNet** — 20-vertex prototype matches few-shot N-way setup.
2. **Long-tail CIFAR-LT** — prototype regularization helps minority classes.
3. **Spherical MNIST** — dodeca latent is equivariant to icosa rotation.
4. **OOD detection benchmarks (SVHN, LSUN, Places365)** — classic Hendrycks panel.

### 7.3 Cross-paradigm context (LLM track)

LLM-track: WikiText-103 at 124 M with DodecaReadout aux head + H49 PRH alignment loss. Train 50 k steps, evaluate (a) perplexity, (b) UMAP of final residual states colored by dodeca-vertex assignment, (c) zero-shot probe accuracy on syntactic tasks.

## 8. Cross-references

- Parent design-space row: `IDEA_TABLE.md` § G3 row H25.
- Master experiment list: not yet in `EXPERIMENT_LOG.md` (planned Tier 3).
- Implementation sub-directory: `ideas/25_dodeca_latent/`.
- Related hypotheses that compose: H24 (icosa CNN — dodeca is dual), H23 (Platonic graph), H30 (Platonic-Fib hybrid), H49 (PRH alignment), H37 (pentagonal attention — dodeca faces are pentagons).
- Related hypotheses that conflict: none directly.

## 9. Committee Q&A

**Q: Why isn't this just VQ-VAE with 20 codewords?**

> Identical-count, different structure: VQ-VAE codewords are LEARNED arbitrary vectors; our 20 vertices are FIXED at the dodecahedral geometry in φ-coordinates. The prior is structural rather than data-driven.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies: OOD-AUC ≥ +1.0 pp AND rotation-instability ≤ -0.03. Either failure discards.

**Q: What if dodecahedral structure is no better than a random 20-prototype set?**

> The ablation compares against a random-20-prototype baseline at matched dim. If random is within 0.5 pp of dodeca, the φ-coordinate structure has not added value.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H25 is tested as a single prior; H50/H67 test compounding.

**Q: How do we know the implementation is correct?**

> `tests/test_dodeca.py::test_vertex_count` asserts |V|=20. `test_phi_coords` verifies golden-ratio coordinate structure. `test_icosa_symmetry` asserts the 60-element icosa group permutes the vertex set. Plus experiment-archive verification.

## 10. Verification artifacts checklist

- [ ] `ideas/25_dodeca_latent/implementation.py` exists and tests green
- [ ] `ideas/25_dodeca_latent/tests.py` ≥ 5 assertions
- [ ] `ideas/25_dodeca_latent/AUDIT.md` lists ≥ 3 self-found weaknesses
- [ ] `ideas/25_dodeca_latent/IMPROVEMENTS.md` records the fixes
- [ ] `ideas/25_dodeca_latent/VERIFY.md` signed
- [ ] Experiment archives present
- [ ] Archives carry `verification/`
- [ ] Row added to `EXPERIMENT_LOG.md` Tier 3
- [ ] Result reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-B.
