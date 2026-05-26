# H26 — Fractal Toroidal

> **One-line claim:** A network whose `_FractalPath` recursive substructure operates on a toroidally-padded manifold raises top-1 on tiled / wrap-aware data and improves Betti-collapse rate versus a planar fractal of matched depth and parameter budget.
>
> **Source design space:** G3 Topologies & Graphs (H21–H30).
>
> **Implementation status (this repo):** `○ not started` (composed prior; parts exist).

This document is the committee-grade design write-up for hypothesis H26, the **composition** of H05 (fractal recursion, +2.35 pp single positive) and H22 (toroidal padding).

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

Fractal-toroidal geometry appears in cosmology (the topology of the universe under some compact-3-torus cosmological models is fractal at small scales and toroidal at large), in fluid dynamics (vortex rings have toroidal topology with fractal turbulent boundary layers), and in chaos theory (the Mandelbrot set's toroidal Julia subsets). Sacred-geometry literature emphasizes the "torus of life" with internal fractal self-similar pattern (the Flower-of-Life recursion living on a toroidal manifold). The DL motivation is the **composition argument**: H05 (fractal) was the only single prior that LIFTED top-1 in the previous sweep (+2.35 pp), and the wrap-aware version of toroidal closure (H22's full version) is structurally complementary because fractal recursion provides feature-multiplicity within scales while toroidal closure removes boundary artifacts across scales. Together, they exercise scale + wrap simultaneously.

The biological inspiration is the cortical sheet: it is fractally folded (gyrification index ≈ 2.4, fractal dimension ≈ 2.6) and topologically equivalent to a torus when including the corpus callosum and brain-stem connections (the surface is a sphere with handles). Fractal-toroidal architectures may align with this combined geometry.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** fractal recursion provides scale-self-similar multiplicity and toroidal closure removes boundary artifacts, composing `_FractalPath` recursive substructure on top of a toroidally-padded grid raises top-1 on tiled-CIFAR by ≥ +3 pp (i.e., greater than the sum of H05 and H22 single-prior effects) and accelerates the Betti-collapse rate by ≥ 0.10, per the mechanism of Larsson 2017 FractalNet (arXiv:1605.07648) composed with Pittorino 2022 toroidal landscapes (arXiv:2202.03038).

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on tiled-CIFAR-10, the fractal-toroidal variant fails to lift top-1 by ≥ 2.0 pp over a parameter-matched planar-fractal baseline AND fails to accelerate Betti-β₁ collapse by ≥ 0.05 versus the baseline, this hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Larsson, G., Maire, M., Shakhnarovich, G. 2017 ICLR 'FractalNet:
Ultra-Deep Neural Networks without Residuals' (arXiv:1605.07648) — the
fractal-recursion reference; supports the depth-multiplicity argument.

Pittorino, F., et al. 2022 ICLR Workshop 'Deep Networks on Toroids'
(arXiv:2202.03038) — toroidal landscape flatness argument.

Mandelbrot, B. 1982 'The Fractal Geometry of Nature' — the foundational
fractal-geometry reference; nature's preference for self-similar
recursion.

Edelsbrunner, H., Letscher, D., Zomorodian, A. 2000 'Topological
Persistence and Simplification' — persistent-homology reference behind
the Betti-collapse metric.

Caspar, D. L. D., Klug, A. 1962 — icosahedral assembly principles
(tangential, but motivates toroidal closure on biological surfaces).
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

Combine `_FractalPath` (depth=2, 1/φ width-shrink per recursion) with `ToroidalConv2d` at every convolution.

- Input: `(B, C, H, W)`.
- `_FractalPath(depth=2)` produces parallel branches of conv-conv vs single-conv, joined by mean.
- Each conv inside the fractal block is replaced by `ToroidalConv2d` (circular pad, optional φ-scaling).
- Param count: same as H05 fractal (≈2× the priors-off baseline due to fractal multiplicity).

```python
# ideas/26_fractal_toroidal/implementation.py
class FractalToroidalBlock(nn.Module):
    def __init__(self, C, depth=2, phi_shrink=True):
        super().__init__()
        self.depth = depth
        if depth == 1:
            self.path = ToroidalConv2d(C, C, 3)
        else:
            inner_C = int(round(C / PHI)) if phi_shrink else C
            self.deep = nn.Sequential(
                FractalToroidalBlock(inner_C, depth-1),
                FractalToroidalBlock(inner_C, depth-1),
            )
            self.shallow = ToroidalConv2d(C, inner_C, 3)
            self.up = ToroidalConv2d(inner_C, C, 1) if inner_C != C else nn.Identity()
    def forward(self, x):
        if self.depth == 1: return self.path(x)
        return self.up((self.deep(self.shallow(x)) + self.shallow(x)) / 2)
```

### 5.2 LLM track (decoder-only Transformer)

For decoder-only Transformers, fractal-toroidal composes H52 (FractalNet drop-path anytime) with H22 (toroidal KV cache). Result: a model with **anytime inference** (drop-path lets you sample at multiple depths) AND **constant-memory long context** (toroidal KV).

- Slots in: FFN replacement (fractal) + KV cache (toroidal).
- FlashAttention-2 compatibility: ✓.
- Causal-mask: preserved.
- Expected: KV cache constant; perplexity slightly worse than non-fractal at fixed depth but anytime-eval gives latency/accuracy trade-off curve.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (tiled-CIFAR) | [+0.020, +0.060] | composition of two complementary priors |
| top-1 (tiled-CIFAR, primary) | [+3.0 pp, +6.0 pp] | non-additive composition |
| top-1 (upright CIFAR) | [+1.0 pp, +3.0 pp] | fractal alone was +2.35; toroidal hurts but less when fractal helps |
| params | [+90 %, +110 %] | inherits fractal's 2× cost |
| FLOPs | [+90 %, +110 %] | same |
| GPU latency (batch=1) | [×2.5, ×3.5] | fractal + toroidal-pad compound latency |
| Betti collapse rate | [+0.10, +0.30] | fractal recursion accelerates β collapse |
| KV cache @ 32 k (LLM) | [-50 %, -80 %] | toroidal KV constant |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** tiled-CIFAR-10.
- **Architecture:** 3-stage NaturePriorBlock with `FractalToroidalBlock` at each stage.
- **Epochs / batch / precision / seeds:** 12 epochs, batch 96 (fractal cost), bf16 AMP, 3 seeds.
- **Composite:** identical to T1.5 scaffold.
- **Run-script:** `python ideas/26_fractal_toroidal/experiment.py --seeds 0 1 2`.
- **Wall-clock:** ~20 min/seed × 3 = 60 min.
- **Archive:** `ideas/26_fractal_toroidal/experiments/exp001_seed0..2/`.

### 7.2 Idea-targeted experiment

1. Tiled-CIFAR / tiled-Tiny-ImageNet (wrap-aware).
2. Spherical MNIST with toroidal projection.
3. Fluid-dynamics surrogate (small simulated turbulence dataset).

### 7.3 Cross-paradigm context (LLM track)

Fractal-toroidal Transformer at 124 M on WikiText-103 with anytime evaluation at depths {1, 2, 4} and toroidal KV at 8 k / 16 k / 32 k contexts.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G3 row H26.
- Experiment list: planned Tier 2/3.
- Implementation sub-dir: `ideas/26_fractal_toroidal/`.
- Composes: H05 (fractal), H22 (toroidal), H52 (anytime), H62 (LLM toroidal KV).
- Conflicts: H24 (icosa requires planar gnomonic, not toroidal).

## 9. Committee Q&A

**Q: Why isn't this just FractalNet + circular pad?**

> Adds 1/φ depth-shrink to the fractal recursion AND φ-scaled toroidal wrap; both are novel relative to the cited papers, and the composition is pre-registered.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 gives top-1 ≥ +2 pp AND Betti-β₁ collapse ≥ +0.05. Both must hold.

**Q: What if H05 alone is enough?**

> H05 is the control. If H26 - H05 < 1.0 pp at matched params, the toroidal addition is null and the hypothesis fails its compositional claim.

**Q: Compounding has historically failed (H50). Why expect it here?**

> H50 stacked SIX priors that span DIFFERENT geometric assumptions; H26 stacks TWO priors that share scale-recursive geometry. The compounding failure is plausibly priors-set-specific.

**Q: How do we know the implementation is correct?**

> `tests/test_fractal_toroidal.py::test_recursive_depth` asserts depth=2 yields the expected branch count. `test_phi_shrink` asserts inner-channel = round(C/φ). `test_circular_pad` verifies wrap-around. Plus archive verification.

## 10. Verification artifacts checklist

- [ ] `ideas/26_fractal_toroidal/implementation.py` exists and tests green
- [ ] `ideas/26_fractal_toroidal/tests.py` ≥ 6 assertions
- [ ] `ideas/26_fractal_toroidal/AUDIT.md` lists ≥ 3 weaknesses
- [ ] `ideas/26_fractal_toroidal/IMPROVEMENTS.md` records fixes
- [ ] `ideas/26_fractal_toroidal/VERIFY.md` signed
- [ ] Experiment archives present
- [ ] Archives carry `verification/`
- [ ] Row added to `EXPERIMENT_LOG.md` Tier 2/3
- [ ] Result reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-B.
