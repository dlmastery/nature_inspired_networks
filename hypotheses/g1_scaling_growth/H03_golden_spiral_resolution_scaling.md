# H03 — Golden Spiral Resolution Scaling (input res grown by 137.5 deg + phi multiples)

> **One-line claim:** Training ViT-Tiny with progressively phi-multiplied
> input resolutions (28 -> 45 -> 73 -> 118 -> 191) along the golden spiral
> increases rotated-CIFAR robustness by >= 3 pp at iso-compute over
> conventional doubling resolutions.
>
> **Source design space:** G1 Scaling-and-Growth (H01-H10).
>
> **Implementation status (this repo):**
> `o planned`.

This document is the committee-grade design write-up for hypothesis H03.

---

## 1. Motivation (sacred / natural intuition, >= 100 words)

The golden spiral is the unique logarithmic spiral whose growth rate
per quarter turn equals phi. It governs the arrangement of sunflower
seeds, the curve of nautilus shells, the cochlea of the inner ear, and
the spiral of certain galaxies. Each successive turn of the spiral is
related to its predecessor by the same multiplicative factor (phi),
and each seed is offset from its predecessor by the golden angle
360 / phi**2 = 137.508 degrees. This produces the densest non-
overlapping angular packing in the plane and the densest non-self-
similar spatial packing on a 2D surface. Vision transformers process
images at fixed input resolution and the choice of intermediate
resolutions is typically a power-of-two cascade (224 -> 112 -> 56 ->
28 -> 14). The hypothesis here is that the geometric prior best suited
to rotation-invariance and scale-invariance is not power-of-two but
phi-progression, because rotation by 137.5 degrees combined with phi-
scaling forms an exact rotational lattice -- the same lattice that
evolution selected for the cochlea, an organ whose primary job is
exactly to be both rotation- and scale-equivariant.

## 2. Formal hypothesis (the falsifiable claim, >= 50 words)

Because golden-spiral resolution scaling -- successive resolutions
multiplied by phi and patches rotated by the golden angle 137.5 deg --
imposes the cochlear rotation-scale lattice on the encoder, the
mechanism through which it should improve robustness is enforcing
rotation-equivariance at every scale transition (the cochlea's
informational optimum). Per Cohen and Welling 2016 we expect rotated-
CIFAR top-1 to improve by 3-5 pp at iso-compute, with a corresponding
decrease in rotation-equivariance error of 0.04-0.08.

## 3. Falsifier (>= 30 words)

If a 5-resolution phi-progression ViT-Tiny (28, 45, 73, 118, 191) does
NOT beat a power-of-two ViT-Tiny (28, 56, 112, 224) on rotated-CIFAR-10
by at least 3 pp top-1 at 3-seed median AND its rotation-equivariance
error is not reduced by >= 0.03, the hypothesis is DISCARDED and Status
moves to `x disproved`.

## 4. Citations (Citation Rigor format, >= 80 words)

```
Cohen, Taco S., Welling, Max 2016 ICML 'Group Equivariant Convolutional
Networks' (arXiv:1602.07576) -- the foundational rotation-equivariant
network. Our phi-progression is a continuous-rotation cousin: rather
than discretising rotation to C4 or C8, we choose a single irrational
rotation (137.5 deg) such that all rotation orbits are dense.

Dosovitskiy, Alexey, Beyer, Lucas, Kolesnikov, Alexander, Weissenborn,
Dirk, Zhai, Xiaohua, Unterthiner, Thomas, Dehghani, Mostafa, Minderer,
Matthias, Heigold, Georg, Gelly, Sylvain, Uszkoreit, Jakob, Houlsby,
Neil 2021 ICLR 'An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale' (arXiv:2010.11929) -- the ViT backbone we modify;
patch grid replaced by golden-spiral lattice at each resolution stage.

Vogel, Helmut 1979 Math Biosciences 'A better way to construct the
sunflower head' -- the canonical golden-angle = 137.508 deg derivation
that motivates our 137.5 deg patch rotation between stages.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

Although H03 is naturally a ViT idea, the CNN-track variant grows the
input image through a 4-stage resize pyramid where resize ratios are
phi**1, phi**2, phi**3, phi**4 from a 28x28 base. Each stage operates
at its native resolution (28, 45, 73, 118) with patches sampled along
the golden-spiral lattice rather than the rectangular grid.

Shapes: (B, 3, 28, 28) -> stage_1 -> (B, C_1, 28, 28); the *next* stage
resizes input to (B, 3, 45, 45). Stages are not connected by
downsampling -- each is an independent multi-resolution view.

Cost vs ViT-Tiny iso-compute: total tokens across 4 resolutions =
49 + 81 + 169 + 289 = 588 vs ViT-Tiny single-res 196 tokens. Compute
is ~3x; we offset by reducing depth from 12 to 4 layers per stage.

PyTorch sketch:

```python
PHI = (1 + 5 ** 0.5) / 2
GOLDEN_ANGLE = 360.0 / PHI ** 2  # 137.508 deg

def golden_spiral_lattice(N: int):
    """N points on a unit disk along the golden spiral."""
    theta = torch.arange(N) * GOLDEN_ANGLE * math.pi / 180
    r = torch.sqrt(torch.arange(N).float() / N)
    return torch.stack([r * theta.cos(), r * theta.sin()], dim=-1)

class GoldenSpiralResize(nn.Module):
    def __init__(self, resolutions=(28, 45, 73, 118)):
        super().__init__()
        self.resolutions = resolutions
    def forward(self, x):  # (B, C, H0, W0)
        return [F.interpolate(x, size=(r, r), mode='bilinear',
                              align_corners=False)
                for r in self.resolutions]
```

It lives in `src/nature_inspired_networks/multi_scale.py`, re-exported
by `ideas/03_golden_spiral_resolution/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

For LMs, "resolution" maps to **context length**. The phi-progression
schedule sets context lengths at training-curriculum stages:
512, 829, 1342, 2172, 3514, 5687. Each stage trains for a fixed token
count; transitioning between stages preserves the model weights but
extends positional encodings via RoPE base-frequency rescaling (cf.
H34). Patch rotation by 137.5 deg has no direct analogue but the closest
mapping is **token-stride rotation**: at each curriculum stage the
positional encoding's base frequency is shifted by 137.5 deg / (2 *
context_len), producing a non-overlapping coverage of the rotary
frequency space.

FlashAttention-2 compatibility: context_len changes are supported as
long as the sliding-window or block-sparse mask is recomputed. Causal
mask: preserved. KV cache: grows linearly with context_len as expected.

```python
class PhiContextSchedule:
    def __init__(self, base_ctx=512, num_stages=6):
        self.contexts = [int(base_ctx * PHI ** k)
                         for k in range(num_stages)]
    def context_at(self, step, steps_per_stage):
        return self.contexts[min(step // steps_per_stage,
                                  len(self.contexts) - 1)]
```

Expected impact at 124M scale: WikiText-103 ppl unchanged at training
end (curriculum gets to the same place), but long-context (8k+) ppl
improves by 1.5-3.0 ppl because intermediate resolutions train the
model on a denser frequency lattice.

## 6. Predicted Delta on 4090 benchmarks (the pre-registered prediction)

| metric | Delta vs baseline | rationale |
|---|---|---|
| composite | [-0.005, +0.020] | rot-eq gains balanced against compute |
| top-1 (rot-CIFAR, CNN) | [+3.0, +5.5] pp | rotation lattice matched to test |
| perplexity (long-ctx LLM) | [-3.0, -1.5] | denser rotary-freq curriculum |
| params | [-5, +0] pct | depth reduced to offset multi-res |
| FLOPs | [+10, +40] pct | multi-resolution training |
| GPU latency (batch=1) | [+15, +35] pct | extra resolution streams |
| rotation-equivariance err | [-0.08, -0.04] | core target |
| KV cache @ 32k (LLM) | [+0, +5] pct | unchanged |
| Betti collapse rate | [0.00, +0.05] | unclear effect |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- Dataset: **CIFAR-10** with rotated-CIFAR-10 as eval-only OOD
- Architecture: ViT-Tiny scaffold; conditions {power-of-two ViT-T,
  phi-progression 4-resolution ViT-T, phi-progression + golden-spiral
  patch lattice ViT-T}
- Epochs / batch / precision / seeds: 100 epochs, batch 256, bf16,
  seeds {0, 1, 2}
- Composite formula: `0.4 * top1_clean + 0.3 * top1_rotated +
  0.15 * (1 - flops/2G) + 0.15 * (1 - rot_eq_err)`; SHA-256 pinned
- Run-script: `python scripts/run_sweep.py --config
  configs/h03_golden_spiral.yaml --seeds 0 1 2`
- Wall-clock: 3 configs * 3 seeds * ~25 min = ~225 min (3.75 h)
- Archive: `ideas/03_golden_spiral_resolution/experiments/
  exp001_rotcifar_vit_tiny/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

Spherical MNIST and IcoMNIST -- datasets where rotation is the
intrinsic axis of variation. Predict +5-8 pp top-1 vs power-of-two ViT
baseline. Wall-clock: ~6 hours on 4090 single seed.

### 7.3 Cross-paradigm context (LLM track)

124M decoder, WikiText-103, with phi-context curriculum [512, 829,
1342, 2172] over 4 epochs. Measure perplexity at native and at 8k
extrapolated context. Compare to fixed-512 baseline. Budget: ~8 hours.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` section G1 row H03
- Master experiment list: `EXPERIMENT_LOG.md` (new T6.1+ rows)
- Implementation sub-directory: `ideas/03_golden_spiral_resolution/`
- Related hypotheses that compose: H01 (phi compound), H34 (golden
  RoPE), H36 (phi-spiral PE), H24 (icosahedral)
- Related hypotheses that conflict: power-of-two scaling literature

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of multi-resolution ViTs
(MViT/Swin)?**

> MViT downsamples by 2x at each stage in a strictly hierarchical
> fashion. H03 trains independent multi-resolution streams at phi-
> spaced sizes and combines them via attention pooling, not pyramid
> downsampling. The discriminator is the resolution ratio (phi vs 2)
> and the patch lattice (golden-spiral vs Cartesian).

**Q: How is this falsifiable rather than aesthetic?**

> Section 3 commits to >= 3 pp rotated-CIFAR top-1 gain and >= 0.03
> reduction in rotation-equivariance error.

**Q: What if the prior helps on rotated-CIFAR but hurts on clean CIFAR?**

> Section 7.1's composite weights clean top-1 (0.4) higher than rotated
> top-1 (0.3) so a clean-accuracy regression is heavily penalised.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> The previous sweep did not test rotation-equivariance evaluation;
> this hypothesis is exactly the experiment that the negative-result
> follow-up was waiting for.

**Q: How do we know the implementation is correct?**

> `tests/test_golden_spiral.py` asserts (a) consecutive lattice points
> are separated by 137.5 deg +/- 1e-3, (b) consecutive radii ratio
> approaches sqrt(2/pi), (c) total lattice mass = N, (d) forward shape.

## 10. Verification artifacts checklist (committee evidence pack)

- [ ] `ideas/03_golden_spiral_resolution/implementation.py` exists; tests green
- [ ] `ideas/03_golden_spiral_resolution/tests.py` >= 12 assertions
- [ ] `ideas/03_golden_spiral_resolution/AUDIT.md` >= 3 weaknesses
- [ ] `ideas/03_golden_spiral_resolution/IMPROVEMENTS.md` fixes
- [ ] `ideas/03_golden_spiral_resolution/VERIFY.md` signed
- [ ] Experiment archive populated
- [ ] `verification/{tests.txt, smoke.txt, gates.txt, reproduction.txt}`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Result reflected in `FINDINGS.md`

## 11. Status journal

- 2026-05-27 -- Created from template by Doc-Agent-A.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G1 (elite-research-scientist critic). Critiquing the IDEA, not the implementation (that audit lives at `audits/G1_audit.md`).*

### Prior plausibility (independent of nature-inspired framing)
**LOW.** The hypothesis conflates *two unrelated geometric properties* (multi-scale resolution and angular rotation lattice) and claims a single mechanism — rotation-equivariance via cochlear-spiral analogy — explains both. The cochlea is not rotation-equivariant; it is a *frequency-position decomposer along a 1D curve*. Citing Cohen & Welling 2016 (arXiv:1602.07576) which explicitly works in **discrete groups (C4, C8)** as justification for an *irrational* rotation (137.5°) misreads that paper: their entire framework requires the rotation group to be finite or generated by a finite-index subgroup of SO(2).

### Mechanism scrutiny — does the claimed mechanism predict the effect?
The "because" clause: *"imposes the cochlear rotation-scale lattice on the encoder ... enforcing rotation-equivariance at every scale transition."* Three problems: (i) the implementation merely resizes inputs to (28, 45, 73, 118) via `F.interpolate` — there is no rotation operator anywhere in the forward pass; (ii) bilinear interpolation is not rotation-equivariant under irrational angles; (iii) the "patch lattice" code generates 2D points on a disk but the model never uses them — the lattice is decorative.

### Confounds — what else could explain a positive (or negative) result?
1. **Multi-resolution ensemble**: any 4-resolution model will beat a 1-resolution model on rotated-CIFAR (cf. Touvron et al 2019 *FixRes* arXiv:1906.06423); the resolution *spacing* is irrelevant.
2. **3× compute**: the doc concedes 3× tokens — most of the rotated-CIFAR gain may be raw compute, not the φ-spacing.
3. **TestRot ≠ TrainRot**: rotated-CIFAR is a robustness test, not an equivariance test — a model can be more *robust* without being more *equivariant*.

### Numerology check — does φ specifically matter?
**No.** The resolution cascade [28, 45, 73, 118] is rounded to integers from φ-multiples; the cascade [28, 50, 90, 160] (ratio 1.8) or [28, 42, 63, 95] (ratio 1.5) would be operationally indistinguishable. **Kill-or-confirm**: compare cascades at ratios {1.5, φ≈1.618, 1.8, 2.0} with all other knobs frozen. If rot-CIFAR Δ across the four is < 1pp, φ is decorative.

### Literature: precedent or rediscovery?
**Direct precedent**: Singh & Davis 2018 *An Analysis of Scale Invariance in Object Detection — SNIP* (arXiv:1711.08189) and Touvron et al 2019 *FixRes* (arXiv:1906.06423) both study non-power-of-two multi-resolution training. For rotation-equivariance the canonical citations are Worrall et al 2017 *Harmonic Networks* (arXiv:1612.04642) and Weiler et al 2019 *General E(2)-Equivariant Steerable CNNs* (arXiv:1911.08251). None of these benefited specifically from φ-spacing.

### Expected effect size — skeptical a-priori re-prediction
Doc predicts +3.0 to +5.5 pp on rot-CIFAR. My prior: any 4-resolution model gives +1 to +3pp on rot-CIFAR via ensemble robustness; φ-spacing specifically contributes Δ ∈ [−0.3, +0.5] pp (90% CI). The big number in the doc is the *multi-resolution* effect, mis-attributed to φ.

### Minimum-distinguishing experiment
**Three configs, rot-CIFAR-10, 30 epochs, 3 seeds**: (i) single-res 28×28 ViT-T; (ii) 4-res power-of-two [28,56,112,224] ViT-T; (iii) 4-res φ-spaced [28,45,73,118] ViT-T. If (ii) and (iii) are within 0.5pp, φ is decorative. The doc's protocol compares (i) directly to (iii) which inflates the apparent effect.

### Verdict
**NUMEROLOGY** — The φ-spacing is decorative atop a real but well-known effect (multi-resolution training improves rotated robustness). The cochlear-rotation-equivariance mechanism is unsupported and the patch-lattice code is never executed in the forward pass.
