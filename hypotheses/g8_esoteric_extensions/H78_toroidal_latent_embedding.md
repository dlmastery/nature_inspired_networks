# H78 — Toroidal Latent Embedding

> **One-line claim:** Projecting a latent onto T² (two angle→(cos,sin)
> circles) yields a bounded, periodic code superior for cyclic factors.
>
> **Source design space:** G8 Esoteric Extensions.
>
> **Implementation status (this repo):** `✓ done` (primitive + tests).

---

## 1. Motivation (≥ 100 words)

Many factors of variation in real data are intrinsically cyclic:
orientation (0 and 2π are the same), phase, hue, day-of-week, periodic
texture. A Euclidean latent represents these badly — it must "tear" the
circle somewhere, creating an artificial discontinuity where 359° and 1°
land far apart. The torus T² is the product of two circles, the natural
manifold for a pair of independent angular factors, and it is compact, so
the code is bounded without any explicit regularisation. Embedding an
angle θ as the unit-circle pair (cos θ, sin θ) is the standard isometric,
discontinuity-free encoding; doing it for two angles places the latent on
T² ⊂ R⁴. The esoteric "torus field" energy motif is the mystical
motivation; the implementation is a plain angle-to-(cos,sin) embedding
onto a product of two circles, which is exactly how hyperspherical/
toroidal VAEs encode periodic structure.

## 2. Formal hypothesis (≥ 50 words)

Because the bottleneck maps the input to two angles and embeds each as a
unit-circle pair, the latent is constrained to lie on the compact, smooth
manifold T², which removes the wrap-around discontinuity a Euclidean code
suffers on cyclic factors; per Davidson 2018, compact-manifold latents
improve representation of periodic variation, so we expect equal or better
downstream accuracy on cyclic-factor tasks at a tiny parameter cost, and a
bounded latent norm by construction.

## 3. Falsifier (≥ 30 words)

If, on a rotation-angle regression task whose target is genuinely cyclic,
the T² bottleneck does not reduce angular MAE by ≥ 0.02 rad versus an
equal-width Euclidean bottleneck at 3-seed median, the periodic prior adds
no value and the hypothesis is DISCARDED.

## 4. Citations (≥ 40 words)

Davidson, T. R., Falorsi, L., De Cao, N., Kipf, T., Tomczak, J. M. 2018
UAI 'Hyperspherical Variational Auto-Encoders' (arXiv:1804.00891) —
establishes that embedding latents on compact manifolds (circles and
spheres, whose products give tori) yields bounded, periodic codes that
better model cyclic factors of variation, directly motivating the T²
bottleneck implemented here.

## 5. Mechanism

### 5.1 CNN / vision track
A `ToroidalLatent(in_dim, out_dim)` bottleneck after global-average-pool:
`Linear(in→2)` → `(cos t1, sin t1, cos t2, sin t2)` (R⁴ on T²) →
`Linear(4→out)`. Params: `(in·2+2)+(4·out+out)` — a few hundred, far
under a dense bottleneck. The intermediate is guaranteed unit-norm per
circle (tested). Lives at
`src/nature_inspired_networks/toroidal_latent.py`; the helper
`toroidal_distance` gives the wrap-aware geodesic.

### 5.2 LLM track (decoder-only)
Slots as a periodic positional/factor code: map a scalar position to T²
and concatenate the R⁴ embedding to token features, giving a bounded,
period-aware positional signal (RoPE is a related but rotation-pair
formulation). No KV-cache change; negligible FLOPs.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite | [−0.003, +0.004] | inert on isotropic CIFAR, helps cyclic tasks |
| top-1 (CNN) | [−0.2, +0.3] pp | bottleneck capacity slightly limited |
| params | [+0.05%, +0.2%] | two small Linear maps |
| FLOPs | [+0.01%, +0.1%] | tiny |
| angular MAE (cyclic task) | [−0.05, −0.02] rad | wrap-free code |

## 7. Experimental protocol

### 7.1 Primary
ResNet-20 + T² bottleneck before the head, CIFAR-10 12-epoch smoke, bf16,
batch 256, seeds 0/1/2. Archive:
`ideas/78_toroidal_latent/experiments/exp001_smoke/`.

### 7.2 Targeted (where it should shine)
Rotated-MNIST angle regression / orientation prediction, where the target
is exactly a circular quantity — the T² code is data-aligned.

### 7.3 Cross-paradigm (LLM)
Use as a periodic positional code on a 124M decoder; compare perplexity
on a synthetic periodic-token corpus vs. learned absolute positions.

## 8. Cross-references

- Parent: G8 esoteric extensions.
- Sibling latent: `src/nature_inspired_networks/dodeca_latent.py` (H25).
- Composes with: H79 (graph on toroidal node codes), H77 (latent head).

## 9. Committee Q&A

**Q: Why isn't this just a sinusoidal positional encoding?**
> Sinusoidal PEs are fixed; here the two angles are *learned* from the
> input via a Linear, and the latent is an information bottleneck on T²,
> not a positional table.

**Q: How is this falsifiable?**
> §3: a cyclic-target angular-MAE non-improvement of < 0.02 rad discards
> it; the comparison is an equal-width Euclidean bottleneck.

**Q: What if it hurts on non-cyclic data?**
> Expected — the claim is scoped to cyclic factors (§7.2). On isotropic
> CIFAR it is at best inert; the smoke row is a screen.

**Q: Priors don't compound — why bother?**
> Single-prior unit of analysis, measured against an equal-capacity
> Euclidean bottleneck, not inside a hybrid.

**Q: How do we know it is correct?**
> `tests/test_toroidal_latent.py` (5 tests): forward shape, on-T²
> unit-norm per circle, wrap-around distance, seed determinism, and a
> bad-shape regression guard.

## 10. Verification checklist

- [x] Primitive `toroidal_latent.py` exists, tests green (5/5).
- [x] Embedded points lie on T² (unit norm per circle, tested).
- [x] `toroidal_distance` respects wraparound (tested).
- [x] Forward deterministic given seed (tested).
- [ ] Experiment archive (deferred — latent module, no sweep row).

## 11. Status journal

- 2026-05-27 — Created; primitive + 5 unit tests green.
