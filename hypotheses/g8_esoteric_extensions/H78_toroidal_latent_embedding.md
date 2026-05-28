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

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G8 (elite-research-scientist critic). Critiquing
the IDEA, not the implementation (audit at `audits/G8_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)

MED — but only for *cyclic-target* tasks. The mathematical content is
solid: T² as a product of two circles is the natural latent manifold
for two independent angular factors (Davidson, Falorsi, De Cao, Kipf,
Tomczak 2018 UAI 'Hyperspherical Variational Auto-Encoders'
(arXiv:1804.00891); Rezende, Papamakarios, Racaniere, Albergo,
Kanwar, Shanahan, Cranmer 2020 ICML 'Normalizing Flows on Tori and
Spheres' (arXiv:2002.02428)). For CIFAR-10 image classification —
which is intrinsically *non-cyclic* — squeezing the latent down to
R⁴ (two cos/sin pairs) is a 4-D information bottleneck, almost
certainly *destructive* for a 10-way classifier built on a backbone
whose typical penultimate dim is 64–512.

### Mechanism scrutiny — does the NEUTRAL recast match the cited real technique?

YES. Embedding angles as (cos θ, sin θ) is the standard wrap-aware
encoding (Gemici, Rezende, Mohamed 2016 'Normalizing Flows on
Riemannian Manifolds' (arXiv:1611.02304)). The Davidson 2018
(arXiv:1804.00891) hyperspherical VAE is the canonical cited prior
for compact-manifold latents. The author's recast matches the
literature exactly when used as a VAE bottleneck on a cyclic
factor.

### Does the esoteric origin contaminate the implementation or framing?

PARTIALLY. The "torus field" / "tube-torus energy" framing motivates
inserting a torus latent into a CIFAR-10 *classifier*, where there
is no cyclic target. The implementation is sound; the *application*
is contaminated. For CIFAR-10 the doc itself notes the latent will
be "inert on isotropic CIFAR" (§6) — at which point one must ask why
this hypothesis is on the CIFAR-10 sweep at all rather than scoped
to rotated-MNIST / orientation regression (which §7.2 acknowledges).

### Confounds (≥2)

1. **Bottleneck capacity loss.** A R⁴ latent is a hard
   information ceiling: CIFAR-10's information-theoretic lower
   bound is `log₂(10) ≈ 3.32` bits, but the *features* needed to
   reach 90 %+ accuracy clearly require more than 4 continuous
   dimensions to express. Any accuracy drop conflates "no cyclic
   factor present" with "bottleneck too narrow".
2. **Linear vs. T² head.** The block sandwiches the T² between
   two `Linear` layers; the second Linear can in principle invert
   any rotation on T², so the T² constraint is bypassable through
   the outer linear unless the bottleneck is held strictly to R⁴.
3. **No rotated-CIFAR baseline.** Without a rotated-input evaluation
   the T² bottleneck cannot demonstrate its claimed cyclic
   advantage.

### Numerology / specificity check

The "torus" framing is plausible for orientation but the doc does
not justify *two* angles vs. one circle (S¹) or four (T⁴) or a
sphere (S²). The choice of T² rather than S² (3-D embedding) seems
to follow "torus is mystical / unified field" framing rather than
information-theoretic argument. The angular degree of freedom in
CIFAR-10 is closer to S¹ (in-plane rotation) than T², making T²
overspecified.

### Literature precedent — was the neutral recast already known?

YES, well-established. Davidson 2018 (arXiv:1804.00891) for
hyperspherical VAEs, Falorsi, De Haan, Davidson, Cao, Weiler,
Forré, Cohen 2018 'Explorations in Homeomorphic Variational
Auto-Encoding' (arXiv:1807.04689) for general homogeneous-space
latents, Rezende 2020 (arXiv:2002.02428) for tori in flows. The
2-torus VAE is undergraduate-level literature now.

### Expected effect size (90% CI a priori)

CIFAR-10 12-ep top-1 vs. equal-width Euclidean bottleneck:
[−2.5 pp, +0.2 pp]. The strong negative skew reflects the R⁴
capacity ceiling. Rotated-MNIST angular MAE: [−0.08, −0.02] rad
(plausibly genuine gain on a cyclic target).

### Minimum-distinguishing experiment

Rotated-MNIST angle regression with two arms: (a) R⁴ Euclidean
bottleneck, (b) T² bottleneck. The hypothesis is alive only if
angular MAE on (b) < (a) by ≥ 0.02 rad at 3-seed median. CIFAR-10
is *not* the discriminating task — running it as a hypothesis
smoke is wasted compute given the doc's own §6 row predicting
inert behaviour.

### Verdict

DERIVATIVE+TESTABLE — Davidson 2018 / Rezende 2020 toroidal latent
applied to CIFAR-10, where it is predicted inert; the falsifier is
sharp on the rotated-MNIST cyclic task but the CIFAR sweep row is
the wrong experiment.
