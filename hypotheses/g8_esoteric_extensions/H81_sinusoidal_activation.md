# H81 — SinusoidalHarmonicActivation (SIREN-style periodic activation with learnable frequency)

> **One-line claim:** Replacing a CNN's ReLU activations with a
> SIREN-style sinusoidal activation `sin(ω·x)` carrying a learnable
> per-activation frequency `ω` (init 1.0) is at worst neutral on
> CIFAR-10 top-1 (within ±0.5 pp of ReLU at 12 ep) and improves the
> fidelity of high-frequency / fine-texture features, measured as ≥1 pp
> higher accuracy on a high-frequency-emphasised test split.
>
> **Source design space:** G8 esoteric extensions; concretely
> implementable bonus idea backed by a strong real citation. Esoteric
> origin (acknowledged once): the "everything is vibration / harmonic"
> framing motivated trying a periodic activation; the operational object
> is exactly the published SIREN activation.
>
> **Implementation status (this repo):** `● implemented` —
> `src/nature_inspired_networks/sinusoidal_activation.py` +
> `tests/test_sinusoidal_activation.py` (7 assertions, green).

---

## 1. Motivation (≥ 100 words)

Piecewise-linear activations (ReLU) are spectrally biased: networks
built from them learn low-frequency components of a target function far
faster than high-frequency ones (the "spectral bias" of Rahaman et al.
2019), and their second derivatives are zero almost everywhere, so they
struggle to represent signals whose derivatives matter. Sitzmann et al.
2020 (SIREN) showed that a sinusoidal activation `sin(ω·x)` fixes both:
periodic activations represent a signal *and* its derivatives with high
fidelity, dramatically improving implicit-neural-representation quality
on images, audio, and PDE residuals. The per-layer frequency ω is the
key knob — SIREN initialises the first layer's ω to 30 to cover the
input's frequency content, with a careful weight init to keep
pre-activations in `[-1, 1]`. For a CIFAR-scale classifier the
hypothesis is narrower and more conservative: making ω *learnable*
(init 1.0, so `sin(x) ≈ x` near the origin and the network starts close
to a smooth identity) lets each activation discover its own preferred
frequency, potentially capturing fine texture that ReLU's spectral bias
suppresses, while a drop-in swap helper keeps the architecture
otherwise identical. The esoteric "harmonic vibration" motivation maps
cleanly onto this well-cited, falsifiable activation.

## 2. Formal hypothesis (≥ 50 words)

Because ReLU networks exhibit spectral bias toward low frequencies and
have vanishing higher-order derivatives, **mechanism**-wise replacing
ReLU with a learnable-frequency sinusoid `sin(ω·x)` should let each
activation tune its own frequency band *because* gradient descent can
raise ω where high-frequency detail is informative and lower it (toward
the near-identity `sin(x)≈x`) elsewhere; we therefore predict CIFAR-10
top-1 within ±0.5 pp of the ReLU baseline at equal architecture and a
measurable gain on high-frequency-emphasised inputs.

## 3. Falsifier (≥ 30 words)

If, at 3-seed median, the sine-activation network scores **more than
0.5 pp below** the ReLU baseline on standard CIFAR-10 (12 ep) **AND**
shows no advantage (within noise) on a high-frequency test split,
**OR** if training diverges (loss NaN) on any seed at `omega_init=1.0`
with the standard recipe, the hypothesis is **DISCARDED**. `omega_init`
is fixed at 1.0 in v0; the SIREN `omega_init=30.0` first-layer recipe is
the targeted follow-up.

## 4. Citations (Citation Rigor format, ≥ 40 words)

```
Sitzmann, Martel, Bergman, Lindell, Wetzstein 2020 NeurIPS 'Implicit
Neural Representations with Periodic Activation Functions (SIREN)'
(arXiv:2006.09661) -- the canonical sinusoidal activation sin(omega*x)
this module implements; establishes the omega-frequency hyper-parameter
and the fidelity advantage over ReLU for signals and their derivatives.

Rahaman, Baratin, Arpit, Draxler, Lin, Hamprecht, Bengio, Courville
2019 ICML 'On the Spectral Bias of Neural Networks'
(arXiv:1806.08734) -- documents ReLU networks' low-frequency learning
bias, the deficiency a periodic activation is hypothesised to mitigate.

Tancik, Srinivasan, Mildenhall, Fridovich-Keil, Raghavan, Singhal,
Ramamoorthi, Barron, Ng 2020 NeurIPS 'Fourier Features Let Networks
Learn High Frequency Functions in Low Dimensional Domains'
(arXiv:2006.10739) -- corroborates that injecting sinusoidal structure
helps networks fit high-frequency content, supporting the mechanism.
```

## 5. Mechanism

### 5.1 CNN track

`SinusoidalActivation` computes `sin(ω·x)` element-wise. `ω` is a single
learnable scalar by default (init `omega_init=1.0`), or a per-channel
learnable vector when `num_channels=C` is supplied (broadcasts along the
channel dim). A `learnable=False` flag stores ω as a fixed buffer to
recover the original SIREN treatment for ablation. `swap_relu_with_sine`
walks a model recursively and replaces every `nn.ReLU` (including nested
and `inplace=True` ReLUs) with a fresh `SinusoidalActivation`, leaving
all other modules untouched. Input shape is preserved (element-wise).
At init with `omega=1.0`, `sin(x) ≈ x` for small pre-activations so the
swapped network starts near a smooth-identity perturbation of the ReLU
network rather than in a chaotic high-frequency regime.

### 5.2 LLM track

Periodic activations in a Transformer FFN are plausible but
under-explored and prone to instability at high ω; out of scope for v0.
The same `SinusoidalActivation` module would slot into an FFN by
swapping the GELU/SiLU, with `omega_init=1.0` for stability.

## 6. Predicted Δ on CIFAR-10 (pre-registered)

| metric | Δ vs. ReLU baseline | rationale |
|---|---|---|
| top-1 standard (12 ep) | [-0.5, +0.8] pp | near-identity init, mild gain |
| top-1 high-freq split | [+0.5, +2.0] pp | periodic act fits HF detail |
| params | [+0.0 %, +0.01 %] | one ω scalar per activation |
| FLOPs | [+0.0 %, +0.2 %] | sin vs. max, negligible |
| learned ω (mean, final) | [0.8, 3.0] | activations raise frequency a bit |

## 7. Experimental protocol

1. **Unit tests (Phase 0):** `tests/test_sinusoidal_activation.py` — 7
   assertions: forward shape preserved; `sin(ω·0)=0` at origin; ω is a
   learnable `nn.Parameter` receiving gradient; swap helper removes ALL
   ReLU (incl. nested) and forward still runs on a tiny CNN; periodicity
   `act(x)≈act(x+2π/ω)`; per-channel ω broadcasts (ω=0 channel → 0);
   `learnable=False` stores ω as a buffer. All green.
2. **CIFAR-10 SOTA smoke (Phase 1):** unchanged ReLU baseline ≥ 80 % @
   12 ep.
3. **CIFAR-10 hypothesis smoke (Phase 2):** one sweep row flips a
   `sine_activation` flag that calls `swap_relu_with_sine(model)` after
   build; a second row sets `omega_init=30.0` on the stem only (SIREN
   recipe). Report standard and high-frequency-split accuracy.

## 8. Cross-references

- Activation-axis sibling to H39 (harmonic φ-activation) and H75
  (harmonic-cymatic SwiGLU); H81 is the pure-SIREN, no-φ control.

## 9. Verification checklist

- [x] `src/nature_inspired_networks/sinusoidal_activation.py` exists.
- [x] `SinusoidalActivation` + `swap_relu_with_sine` implemented.
- [x] `tests/test_sinusoidal_activation.py` ≥ 4 assertions (has 7), green.
- [x] `sin(0)=0` and periodicity (mechanism), ω-grad (mechanism), swap
      helper on a CNN (regression), per-channel broadcast + buffer (edge)
      asserted.
- [x] `# TODO runner wiring:` block describes the post-build swap call
      without touching `models.py` / `runner.py`.
- [ ] Phase-2 CIFAR-10 sweep row (lead wires the flag).

## 10. Status journal

- 2026-05-27 — Implemented + tested (7/7 green) as a standalone G8
  primitive. Scalar and per-channel ω, learnable and fixed variants
  validated.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G8 (elite-research-scientist critic). Critiquing
the IDEA, not the implementation (audit at `audits/G8_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)

LOW for classification. SIREN (Sitzmann, Martel, Bergman, Lindell,
Wetzstein 2020 NeurIPS 'Implicit Neural Representations with
Periodic Activation Functions' (arXiv:2006.09661)) was *engineered*
for implicit-representation regression problems (image / SDF / NeRF
coordinate-to-value mapping), where representing high-frequency
content of a target *function* is the whole point. Image
classification is a discriminative task; the target is a class
label, not a high-frequency signal. The spectral-bias literature
(Rahaman, Baratin, Arpit, Draxler, Lin, Hamprecht, Bengio, Courville
2019 ICML 'On the Spectral Bias of Neural Networks' (arXiv:
1806.08734)) shows ReLU networks bias toward *low-frequency
generalisation* — which is *helpful*, not harmful, for
classification. Switching to a high-frequency activation may
generalise worse, not better.

### Mechanism scrutiny — does the NEUTRAL recast match the cited real technique?

PARTIALLY. The `sin(ω·x)` activation is exactly SIREN's. But SIREN
requires careful weight initialisation (uniform `[−√(6/fan_in)/ω,
+√(6/fan_in)/ω]`) to keep pre-activations in `[−1, 1]`, and ω₀=30
on the *first* layer with ω=1 on subsequent layers. The doc's
`omega_init=1.0` everywhere bypasses the canonical SIREN recipe
entirely — recovering at best a "near-identity smooth perturbation
of ReLU" rather than SIREN's actual high-frequency capability. The
80.62 % empirical result (cited in task prompt) is consistent with
*underrating* SIREN by using ω=1 throughout: SIREN's ω₀=30 first
layer is what gives it its frequency reach.

### Does the esoteric origin contaminate the implementation or framing?

PARTIALLY. The "everything is vibration / harmonic" framing leads to
the choice of `sin` over the equally plausible `cos`, `tanh-of-sine`,
or `sin²` — none of which would be different empirically but each of
which has equal "vibration" claim. More importantly, the framing
elevates sinusoidal activation above more relevant alternatives for
classification (GELU, SiLU, Mish — Misra 2019 'Mish: A Self
Regularized Non-Monotonic Activation Function' (arXiv:1908.08681))
which all outperform ReLU on CIFAR-10 with proper recipes.

### Confounds (≥2)

1. **ω-init choice.** ω=1 makes `sin(x) ≈ x` near origin — the
   network starts as a near-linear identity perturbation of ReLU,
   not as a SIREN. The 80.62 % result conflates "sin activation
   poor for classification" with "ω=1 is the wrong init for
   SIREN".
2. **Weight init unchanged.** SIREN's weight init (uniform with ω-
   dependent scale) is required for the pre-activation
   distribution to land in `[−1, 1]`. The implementation reuses
   ReLU-era He init — every forward pass starts in the chaotic
   regime where `sin` is approximately white noise.
3. **No optimiser change.** SIREN typically uses Adam with reduced
   learning rate; standard CIFAR-10 SGD at lr=0.1 likely destroys
   the sine network within the first epoch via NaN-adjacent
   activations.

### Numerology / specificity check

Sine is the natural eigenfunction of the Laplacian, so the
"vibration" framing has *some* mathematical content. But the
hypothesis does not distinguish sine from cosine, from
`sin(x)·exp(-x²)` (Gabor), or from `tanh(sin(x))` — all are
"vibrational". The specificity to `sin(ω·x)` comes from SIREN, not
from the cymatic motif.

### Literature precedent — was the neutral recast already known?

YES, but in a different task family. SIREN (arXiv:2006.09661) is
canonical for implicit representations. Periodic activations for
classification have been tried sporadically (Parascandolo, Huttunen,
Virtanen 2017 ICLR Workshop 'Taming the Waves: Sine as Activation
Function in Deep Neural Networks' (https://openreview.net/forum?id=
Sks3zF9eg)) with consistently modest-to-negative results. So the
classification-track prior is *known* but with documented poor
performance.

### Expected effect size (90% CI a priori)

CIFAR-10 12-ep top-1 vs. ReLU baseline: [−6 pp, +0.5 pp] at
`omega_init=1.0`. With proper SIREN recipe (ω₀=30 stem,
ω-dependent weight init): [−3 pp, +1 pp]. The 80.62 % observed
result fits the low-init prediction.

### Minimum-distinguishing experiment

Three-arm sweep: (a) ReLU baseline, (b) sine ω=1 with He init
(current implementation), (c) sine ω₀=30 stem + SIREN init
elsewhere. The hypothesis "sine activation helps classification"
lives only if (c) ≥ (a) at 3-seed median. If (c) < (a) too, sine
is genuinely unsuited to classification and the SIREN paper's
implicit-representation scope was the right scope.

### Verdict

DERIVATIVE+TESTABLE — SIREN (arXiv:2006.09661) applied to
classification with a sub-optimal ω-init that hobbles the prior;
the proper SIREN-recipe arm is the only experiment that genuinely
tests the hypothesis.
