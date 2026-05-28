# H12 — Fib-Channel CNN (Fib filter counts + phi-kernel sizes alternating 3 -> 5)

> **One-line claim:** Successive conv blocks with Fibonacci filter counts
> and alternating phi-kernel sizes (3 -> 5 every other stage) lift CIFAR
> top-1 by 0.5-1.5 pp over uniform-kernel constant-channel baselines.
>
> **Source design space:** G2 Layer/Channel/Neuron (H11-H20).
>
> **Implementation status (this repo):**
> `✓ done` (filter counts only; phi-kernel-size variant missing).

This document is the committee-grade design write-up for hypothesis
H12. **Experiment data exists via T1.1 sg_chan_fib (single seed).**

---

## 1. Motivation (sacred / natural intuition, >= 100 words)

H12 composes H04's Fibonacci channel widths with a kernel-size
alternation: every other stage uses a kernel size that grows by phi
(3, 5, 8 = round(3 * phi), 13 = round(8 * phi)). The biological
precedent is the receptive-field growth of V1 -> V2 -> V4 visual
cortex, where the receptive-field radius approximately doubles per
area but the *cell density* drops to match -- the joint widening of
receptive field and reduction in neuron count keeps total wiring cost
constant. Translating to CNNs: as channel count grows by phi (per
H04), kernel size should also grow by phi to maintain a constant
wiring cost per layer. The two priors compose: Fibonacci channels +
phi-growing kernels. The hypothesis is that this joint prior produces
a Pareto improvement over either prior alone.

## 2. Formal hypothesis (the falsifiable claim, >= 50 words)

Because Fibonacci-channel CNNs with alternating phi-spaced kernel
sizes (3 -> 5 -> 8 -> 13) impose joint width-and-kernel growth at the
phi rate, the mechanism by which they should beat constant-kernel
Fibonacci channels (the T1.1 sg_chan_fib variant) is matching the
biological joint width-kernel growth rule of cortical visual areas.
Per LeCun et al 1989 and biological neuroscience we expect CIFAR-10
top-1 to lift by 0.5-1.5 pp above sg_chan_fib reference.

## 3. Falsifier (>= 30 words)

If a Fibonacci-channel + phi-kernel CNN does NOT lift CIFAR-10 top-1
by >= +0.3 pp at 3-seed median over the existing T1.1 sg_chan_fib
result (80.11 pct), the joint prior is FALSIFIED. The kernel-size
component is the new claim under test.

## 4. Citations (Citation Rigor format, >= 80 words)

```
LeCun, Yann, Boser, Bernhard, Denker, John S., Henderson, Donnie,
Howard, R.E., Hubbard, Wayne, Jackel, Lawrence D. 1989 Neural
Computation 'Backpropagation Applied to Handwritten Zip Code
Recognition' (no arXiv; classical) -- foundational CNN paper with
the original 5x5 kernel; H12 reintroduces 5x5 at later stages.

Simonyan, Karen, Zisserman, Andrew 2015 ICLR 'Very Deep Convolutional
Networks for Large-Scale Image Recognition' (arXiv:1409.1556) -- VGG's
uniform 3x3 kernels; the constant-kernel baseline that H12 disrupts.

He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, Sun, Jian 2016 CVPR 'Deep
Residual Learning for Image Recognition' (arXiv:1512.03385) -- ResNet
uses 3x3 throughout; some bottleneck variants use 1x1+3x3+1x1. H12
tests the 3 -> 5 -> 8 phi-spaced kernel cascade.

Tan, Mingxing, Le, Quoc V. 2020 ICML 'EfficientNetV2' (arXiv:2104.00298)
-- uses 3x3 throughout the body; relevant baseline.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

3-stage backbone with Fibonacci channels [21, 34, 55] (or mod-8
variants [24, 40, 56]) and alternating kernel sizes:

- Stage 0: kernel 3, channels 21 -- params 9*21*21 = 3969
- Stage 1: kernel 5, channels 34 -- params 25*34*34 = 28900
- Stage 2: kernel 3, channels 55 -- params 9*55*55 = 27225
  (alternation: 3, 5, 3 OR 3, 5, 8 if extended)

For 4 stages [21, 34, 55, 89] with kernels [3, 5, 3, 5]:
- params per stage: 9*21*21=3969, 25*34*34=28900, 9*55*55=27225,
  25*89*89=198025 -- total 258k
- vs constant-kernel-3 variant: 3969 + 10404 + 27225 + 71289 = 113k

The kernel-5 stages dramatically increase params. Honest interpretation:
H12 is *not* a parameter-saving prior; it is a *capacity-redistribution*
prior. The hypothesis is that the larger receptive field at
intermediate stages matters more than the param cost.

Cost vs T1.1 (127k params): the phi-kernel variant would be ~258k
params (about 2x). This is a significant cost increase.

```python
PHI = (1 + 5 ** 0.5) / 2

def phi_kernel_sequence(n_stages, base=3):
    """3, 5, 3, 5, ... with optional 3 -> 5 -> 8 cascade."""
    ks = []
    for k in range(n_stages):
        if k == 0:
            ks.append(base)
        elif k % 2 == 1:
            ks.append(max(3, 2 * round(ks[-1] * PHI / 2) + 1))  # odd
        else:
            ks.append(base)
    return ks

class FibChannelPhiKernel(nn.Module):
    def __init__(self, channels=[21, 34, 55, 89], stride_at=(1, 2, 2, 2)):
        super().__init__()
        kernels = phi_kernel_sequence(len(channels))
        layers, c_in = [], 3
        for k, (c, ks, s) in enumerate(zip(channels, kernels, stride_at)):
            layers += [nn.Conv2d(c_in, c, ks, padding=ks // 2,
                                  stride=s), nn.ReLU()]
            c_in = c
        self.stages = nn.Sequential(*layers)
```

Existing impl: `src/nature_inspired_networks/models.py:NaturePriorNet`
uses Fib channel counts but constant kernel 3. The H12 v2 extension
adds the alternating kernel-size variant.

### 5.2 LLM track (decoder-only Transformer)

For decoders, "kernel size" maps to **attention window** in sparse
attention. H12 LLM-track maps to Fibonacci-channel feed-forward layers
(per H11) plus phi-spaced **attention window sizes** per layer:

- layer 0: full attention (no window, or window=N)
- layer 1: window = phi * something
- layer 2: full attention
- layer 3: window = phi**2 * something
- ...

This is a non-standard architecture; the closest reference is
Longformer (Beltagy et al 2020).

```python
class FibFFNPhiWindowDecoder(nn.Module):
    def __init__(self, n_layers=12, d_model=768, ffn_widths=None,
                 window_seq=None):
        super().__init__()
        ffn_widths = ffn_widths or fib_widths(6, n_layers)
        window_seq = window_seq or [None if k % 2 == 0 else
                                     int(512 * PHI ** ((k - 1) // 2))
                                     for k in range(n_layers)]
        self.layers = nn.ModuleList()
        for w, win in zip(ffn_widths, window_seq):
            self.layers.append(_DecoderBlock(d_model, ffn_dim=w,
                                              window=win))
```

FlashAttention-2 compatibility: block-sparse window masks are natively
supported. KV cache: per-layer; layers with window have constant cache,
layers without window have full cache.

Expected impact at 124M scale: WikiText-103 ppl mixed depending on
which layers get windows; long-context ppl improves by 1-2 due to
window stages.

## 6. Predicted Delta on 4090 benchmarks (the pre-registered prediction)

| metric | Delta vs baseline | rationale |
|---|---|---|
| composite | **observed (Fib channels only): -0.0123** | T1.1 vs vanilla |
| top-1 (CIFAR-10, CNN) | [+0.3, +1.5] pp over T1.1 ref | phi-kernel addition |
| perplexity (WikiText-103 LLM) | [-0.4, +0.2] | depends on window placement |
| params | [+90, +110] pct over T1.1 | kernel 5 doubles params |
| FLOPs | [+90, +110] pct | proportional |
| GPU latency (batch=1) | [+30, +60] pct | larger kernels |
| rotation-equivariance err | [-0.01, 0.0] | larger kernels see more rotation |
| KV cache @ 32k (LLM) | [-30, -10] pct | windowed layers smaller |
| Betti collapse rate | [+0.02, +0.05] | larger receptive fields |

**Observed (single seed, T1.1 sg_chan_fib, Fib channels only):**

```
sg_chan_fib  top-1 80.11%  params 127k  latency 4.43 ms  composite 0.8135
```

The Fib-channel-only version is below baseline_vanilla (82.16 pct).
The H12 hypothesis is that adding the phi-kernel alternation will
recover and exceed the vanilla baseline. The kernel-5 cost (+100 pct
params) is significant.

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- Dataset: **CIFAR-10**
- Architecture: NaturePriorNet with conditions {Fib channels + kernel
  3 throughout (T1.1 reproduction), Fib channels + alternating 3/5,
  Fib channels + ascending 3/5/3/5, constant 32 channels + kernel 3
  (baseline)}
- Epochs / batch / precision / seeds: 12 epochs, batch 128, bf16,
  seeds {0, 1, 2}
- Composite formula: existing project formula; SHA-256 pinned
- Run-script: `python scripts/run_sweep.py --config
  configs/h12_fib_phi_kernel.yaml --seeds 0 1 2`
- Wall-clock: 4 configs * 3 seeds * ~10 min = ~120 min
- Archive: `ideas/12_fib_channel_cnn/experiments/exp001_phi_kernel/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

CIFAR-100 with larger images (resize to 64x64) -- larger receptive
fields should help when image content has long-range structure.
Predict +1-2 pp lift. Wall-clock: ~3 hours single seed.

### 7.3 Cross-paradigm context (LLM track)

124M decoder with Fib-FFN + alternating phi-window attention on
WikiText-103, 1 epoch. Budget: ~5 hours.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` section G2 row H12
- Master experiment list: `EXPERIMENT_LOG.md` T1.1 (Fib-channel-only
  done); H12.v2 (phi-kernel addition) queued
- Implementation sub-directory: `ideas/12_fib_channel_cnn/`
- Related hypotheses that compose: H04 (phi width), H38 (fractal
  golden filter), H11 (Fib MLP)
- Related hypotheses that conflict: VGG/ResNet uniform-3x3

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of mixed-kernel CNNs (e.g.,
Inception, MixNet)?**

> Inception uses 1x1+3x3+5x5 *in parallel* within a block. MixNet uses
> learned kernel-size mixtures. H12 alternates kernel sizes
> *sequentially* across stages with phi-spaced sizes (3, 5, 8). The
> sequential phi cascade is the distinguishing factor.

**Q: How is this falsifiable rather than aesthetic?**

> Section 3 commits to >= +0.3 pp top-1 over T1.1.

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> Section 7.2 is the CIFAR-100/64x64 bridge.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> The Fib-channel part of H12 is already in T1.1. The kernel-size
> alternation is the new component under test. The H12 v2 result
> separates the two components.

**Q: How do we know the implementation is correct?**

> Existing tests cover Fib channels. New tests assert kernel sizes
> follow `phi_kernel_sequence` and forward shape preservation.

## 10. Verification artifacts checklist (committee evidence pack)

- [ ] `ideas/12_fib_channel_cnn/implementation.py`; tests green
- [ ] `ideas/12_fib_channel_cnn/tests.py` >= 10 assertions
- [ ] `ideas/12_fib_channel_cnn/AUDIT.md` >= 3 weaknesses
- [ ] `ideas/12_fib_channel_cnn/IMPROVEMENTS.md`
- [ ] `ideas/12_fib_channel_cnn/VERIFY.md` signed
- [ ] T1.1 archive (exists) + new exp001 archive
- [ ] `verification/{tests.txt, smoke.txt, gates.txt, reproduction.txt}`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md`

## 11. Status journal

- 2026-05-27 -- Created from template by Doc-Agent-A.
- (previous) -- T1.1 sg_chan_fib at single seed: Fib channels only, no
  phi-kernel addition. Top-1 80.11 pct, composite 0.8135.
- (planned) -- exp001 adds the alternating-kernel phi variant.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G2 (elite-research-scientist critic). Critiquing the IDEA, not the implementation (that audit lives at `audits/G2_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)
LOW-MED. The Fib-channel-only ablation (T1.1 sg_chan_fib) already underperforms the constant-channel vanilla baseline (-2.05 pp top-1) at single seed. The doc proposes to **add another phi-derived knob** (alternating kernel sizes) on top of a *failed* prior to recover the loss — this is double-counting the same numerological hypothesis. Mixed-kernel CNNs (Inception, MixNet) work because of *parallel multi-scale feature extraction*, not sequential phi-spaced alternation; sequential kernel alternation has been tried under the name "kernel size search" by Tan, Pang, Le 2019 ICML 'MixConv: Mixed Depthwise Convolutional Kernels' (arXiv:1907.09595) and the optimum is found by NAS, not by phi.

### Mechanism scrutiny
The "joint width-and-kernel growth at the phi rate" claim is post-hoc. The biological precedent (V1→V2→V4 receptive-field growth) actually shows RF growth by ~2x per cortical area and ~5x cumulatively (Smith, Singh, Williams, Greenlee 2001 Cerebral Cortex), not phi^k. The doc's own cost analysis (Sec 5.1) admits the kernel-5 stages produce a +100 % param cost — the hypothesis then claims this is "capacity redistribution," which is unfalsifiable without separating the kernel-size and channel-count axes.

### Confounds (≥ 2 alternatives)
(1) Larger kernels straightforwardly improve accuracy in low-data regimes (3x3 vs 5x5 on CIFAR-10: Liu, Mao, Wu, Feichtenhofer, Darrell, Xie 2022 CVPR 'A ConvNet for the 2020s' arXiv:2201.03545 uses 7x7); the proposed comparison conflates this well-known effect with phi-spacing. (2) Parameter-count: +100 % params trivially helps a tiny CIFAR-10 network. (3) Effective receptive field: alternating stride-1 kernels 3,5,3,5 gives ERF = 1+2+4+6+8 = 21px on 32x32 CIFAR, which is nearly global — comparing against a 3x3-only ERF of 9px is unfair.

### Numerology check
Yes — kernels [3, 5, 3, 5] vs [3, 7, 3, 7] vs [3, 5, 7, 9] should all produce the same direction of effect. The phi-cascade (3 → round(3φ)=5 → round(5φ)=8) is just "increasing kernel size by ~60% per alternation," which can be replicated by any growth factor in [1.4, 1.8].

### Literature precedent
This is a rediscovery of multi-scale convolution. Szegedy et al 2015 CVPR 'Going Deeper with Convolutions' (arXiv:1409.4842) — Inception used parallel 1x1, 3x3, 5x5. Tan, Pang, Le 2019 ICML 'MixConv' (arXiv:1907.09595) — learned kernel-size mixtures. Ding, Zhang, Han, Ding, Sun 2022 CVPR 'Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs' (arXiv:2203.06717) — large kernels are best monotonically, not in phi-pairs. Trockman, Kolter 2022 ICLR 'Patches Are All You Need?' (arXiv:2201.09792) — ConvMixer also uses single large kernel.

### Expected effect size (90% CI a priori)
On CIFAR-10 12-epoch at +100% params: Δtop-1 = [+0.3, +1.5] pp from raw capacity, NOT from phi. Iso-param control with constant kernel-5 throughout would dominate the phi-alternation by [-0.3, +0.3] pp (i.e., be statistically indistinguishable). The doc's predicted "+0.3 to +1.5 pp over T1.1" is plausible but entirely attributable to params.

### Minimum-distinguishing experiment
Iso-param control: match params by reducing channel counts on the kernel-5 stages so total params equals the constant-3 baseline. Then compare {3,3,3,3}, {3,5,3,5}_phi-spaced, {5,5,5,5}_constant, {3,5,7,9}_linear-growth. If phi-spacing has no advantage over linear-growth at iso-params, H12 is numerology.

### Verdict
NUMEROLOGY — the phi-kernel-spacing addition is indistinguishable from "any moderate kernel-size growth" and the existing Fib-channel evidence is already negative.

