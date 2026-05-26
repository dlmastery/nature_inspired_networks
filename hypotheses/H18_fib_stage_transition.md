# H18 — Fib-Stage Transition (downsampling factors alternate Fib: stride 2, 3, 2, 3)

> **One-line claim:** A 4-stage backbone with alternating stride
> {2, 3, 2, 3} (Fib pair) yields a finer downsampling cascade than the
> standard {2, 2, 2, 2}, lifting CIFAR-100 top-1 by 0.3-0.8 pp.
>
> **Source design space:** G2 Layer/Channel/Neuron (H11-H20).
>
> **Implementation status (this repo):**
> `o planned`.

This document is the committee-grade design write-up for hypothesis
H18.

---

## 1. Motivation (sacred / natural intuition, >= 100 words)

Standard CNNs downsample by stride 2 at each stage, halving the
spatial resolution four times to produce a final feature map at 1/16
of the input resolution. This power-of-two cascade is computationally
convenient but produces a coarse step from one stage to the next.
Biological visual systems (the retinal -> LGN -> V1 -> V2 -> V4
cascade) use intermediate downsampling ratios closer to 2.5x at each
step, achieving a finer scale-space coverage. The Fibonacci pair 2, 3
satisfies the recurrence 2 + 3 = 5 and produces alternating
downsampling ratios that average phi (geometric mean of 2 and 3 is
sqrt(6) = 2.449, close to phi**2 = 2.618). The hypothesis is that
{2, 3, 2, 3} downsampling preserves more spatial information at
intermediate scales than uniform {2, 2, 2, 2} and produces better
features for medium-resolution classification tasks.

## 2. Formal hypothesis (the falsifiable claim, >= 50 words)

Because Fibonacci-stride downsampling ({2, 3, 2, 3} produces resolution
cascade 1, 1/2, 1/6, 1/12, 1/36 instead of 1, 1/2, 1/4, 1/8, 1/16), the
mechanism by which it should improve CIFAR-100 top-1 is matching the
retinal-cortical scale-space cascade more closely than power-of-two
downsampling. Per LeCun et al 1998 LeNet (originally used non-uniform
strides) we expect a 0.3-0.8 pp top-1 lift at iso-FLOPs.

## 3. Falsifier (>= 30 words)

If a {2, 3, 2, 3}-stride 4-stage backbone does NOT match or exceed the
{2, 2, 2, 2}-stride baseline within +/- 0.3 pp top-1 on CIFAR-100 at
3-seed median, AND has no other compensating advantage (FLOPs, params,
latency), the hypothesis is FALSIFIED.

## 4. Citations (Citation Rigor format, >= 80 words)

```
LeCun, Yann, Bottou, Leon, Bengio, Yoshua, Haffner, Patrick 1998 Proc
IEEE 'Gradient-Based Learning Applied to Document Recognition' (no
arXiv; classical) -- the LeNet paper that originally used non-uniform
strides; H18 returns to that practice with a Fibonacci stride pair.

Simonyan, Karen, Zisserman, Andrew 2015 ICLR 'Very Deep Convolutional
Networks for Large-Scale Image Recognition' (arXiv:1409.1556) -- VGG's
uniform stride-2 cascade is the standard baseline.

He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, Sun, Jian 2016 CVPR 'Deep
Residual Learning for Image Recognition' (arXiv:1512.03385) -- ResNet
uses uniform stride 2 across stages.

Sandler, Mark, Howard, Andrew, Zhu, Menglong, Zhmoginov, Andrey, Chen,
Liang-Chieh 2018 CVPR 'MobileNetV2' (arXiv:1801.04381) -- shows that
non-uniform strides can be more parameter-efficient in some regimes.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

For CIFAR-100 input (B, 3, 32, 32), 4-stage backbone with c0 = 32:

Standard {2, 2, 2, 2} (but skipping first stage stride): 1 -> 1/2 ->
1/4 -> 1/8 = 32, 16, 8, 4.

Fibonacci {2, 3, 2, 3} (cumulative): 1 -> 1/2 -> 1/6 -> 1/12 -> 1/36.
For input 32, this gives 32, 16, ~5, ~3, ~1. The final 1x1 spatial
resolution is the same as global average pooling and works for small
inputs.

For Tiny ImageNet (64x64): 64, 32, 11, 6, 2.

For ImageNet (224x224): 224, 112, 38, 19, 7. Note: cumulative product
2*3*2*3 = 36, and 224/36 = 6.22, rounded to 7. The {2,2,2,2} cascade
gives 224/16 = 14, so Fib-stride has *less* final spatial resolution
(7x7 vs 14x14) which is closer to the eventual GAP target.

Shapes for {2, 3, 2, 3} on (B, 3, 32, 32):
stage 0: (B, 32, 16, 16) [stride 2]
stage 1: (B, 64, 6, 6)   [stride 3, rounded]
stage 2: (B, 128, 3, 3)  [stride 2]
stage 3: (B, 256, 1, 1)  [stride 3, rounded]

Stride-3 convolutions need kernel_size >= stride; we use kernel_size =
3 with stride = 3 and padding = 0. This produces a non-overlapping
3x3 receptive field at each step.

```python
class FibStrideBackbone(nn.Module):
    def __init__(self, c0=32, n_stages=4, strides=(2, 3, 2, 3)):
        super().__init__()
        widths = [c0 * 2 ** k for k in range(n_stages)]
        stages, c_in = [], 3
        for k, (c, s) in enumerate(zip(widths, strides)):
            stages.append(nn.Sequential(
                nn.Conv2d(c_in, c, kernel_size=max(3, s),
                          stride=s, padding=0 if s == 3 else 1),
                nn.BatchNorm2d(c),
                nn.ReLU()
            ))
            c_in = c
        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(c_in, 100)
    def forward(self, x):
        x = self.stages(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)
```

Cost vs {2, 2, 2, 2}: stride-3 stages have wider effective receptive
field (3 vs 2) and dwell at intermediate resolution longer, increasing
FLOPs at those stages by ~50 pct. Total FLOPs +20-30 pct. Params
unchanged because channel counts are identical.

Location: `src/nature_inspired_networks/stride.py:FibStrideBackbone`,
re-exported by `ideas/18_fib_stage_transition/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

LMs do not downsample tokens in the standard architecture. H18's LLM-
track variant maps to **hierarchical token compression** (cf. Hourglass
Transformers, Nawrot et al 2021): tokens are pooled by Fibonacci
factors at each stage. For an input of 1024 tokens:

stage 0: 1024 tokens (full)
stage 1: 512 tokens (pool by 2)
stage 2: 170 tokens (pool by 3)
stage 3: 85 tokens (pool by 2)
stage 4: 28 tokens (pool by 3)

This produces a multi-resolution token hierarchy with cumulative
compression 1024 / 28 = 36 (vs power-of-2 compression 1024 / 64 = 16).

FlashAttention-2 compatibility: standard; pooling is a separate op.
Causal mask: must be recomputed per-stage. KV cache: dramatically
reduced (sum over stages of seq_len = 1024 + 512 + 170 + 85 + 28 ~=
1819 vs 4 * 1024 = 4096 for non-pooling).

Expected impact at 124M scale: WikiText-103 ppl regresses by 0.5-1.0
(information loss from aggressive pooling), but long-context (32k)
inference latency drops by ~40 pct.

## 6. Predicted Delta on 4090 benchmarks (the pre-registered prediction)

| metric | Delta vs baseline | rationale |
|---|---|---|
| composite | [-0.005, +0.015] | accuracy vs FLOPs trade-off |
| top-1 (CIFAR-100, CNN) | [+0.3, +0.8] pp | finer cascade |
| perplexity (WikiText-103 LLM) | [+0.5, +1.0] | pooling-info-loss |
| params | [-2, +2] pct | unchanged |
| FLOPs | [+15, +30] pct | stride-3 dwell |
| GPU latency (batch=1) | [+10, +25] pct | proportional |
| rotation-equivariance err | [-0.005, +0.005] | not affected |
| KV cache @ 32k (LLM, with pool) | [-50, -30] pct | hierarchical compression |
| Betti collapse rate | [+0.02, +0.05] | finer cascade |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- Dataset: **CIFAR-100**
- Architecture: 4-stage backbone; conditions {{2,2,2,2} uniform,
  {2,3,2,3} Fib, {3,2,3,2} reverse-Fib, {2,3,5,3} extended-Fib}
- Epochs / batch / precision / seeds: 50 epochs, batch 128, bf16,
  seeds {0, 1, 2}
- Composite formula: existing project formula; SHA-256 pinned
- Run-script: `python scripts/run_sweep.py --config
  configs/h18_fib_stride.yaml --seeds 0 1 2`
- Wall-clock: 4 configs * 3 seeds * ~10 min = ~120 min
- Archive: `ideas/18_fib_stage_transition/experiments/
  exp001_cifar100_fib_stride/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

Tiny ImageNet at 64x64 -- intermediate resolutions matter more for
medium-size inputs. Predict +0.5-1.5 pp lift. Wall-clock: ~3 hours
single seed.

### 7.3 Cross-paradigm context (LLM track)

124M Hourglass-style decoder with Fib-stride pooling on WikiText-103,
1 epoch. Compare ppl + KV cache to non-pool baseline. Budget: ~5 hours.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` section G2 row H18
- Master experiment list: `EXPERIMENT_LOG.md` (new T3.x row)
- Implementation sub-directory: `ideas/18_fib_stage_transition/`
- Related hypotheses that compose: H01 (compound), H02 (Fib depth),
  H03 (golden spiral resolution)
- Related hypotheses that conflict: uniform-stride literature

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of LeNet's non-uniform strides?**

> LeNet uses non-uniform strides empirically. H18 commits to the
> *specific* Fibonacci-pair {2, 3} cascade. The hypothesis is whether
> this particular pair dominates other non-uniform choices.

**Q: How is this falsifiable rather than aesthetic?**

> Section 3 commits to +/- 0.3 pp top-1 match-or-exceed AND
> compensating efficiency advantage.

**Q: What if the prior helps on CIFAR-100 but hurts on ImageNet?**

> Section 7.2 is the Tiny ImageNet bridge.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H18 is a stride-pattern prior, orthogonal to the per-block geometry
> priors tested previously.

**Q: How do we know the implementation is correct?**

> `tests/test_fib_stride.py` asserts (a) cumulative spatial product
> matches predicted, (b) forward shape at each stage matches, (c)
> stride-3 convolutions use the right padding.

## 10. Verification artifacts checklist (committee evidence pack)

- [ ] `ideas/18_fib_stage_transition/implementation.py`; tests green
- [ ] `ideas/18_fib_stage_transition/tests.py` >= 10 assertions
- [ ] `ideas/18_fib_stage_transition/AUDIT.md` >= 3 weaknesses
- [ ] `ideas/18_fib_stage_transition/IMPROVEMENTS.md`
- [ ] `ideas/18_fib_stage_transition/VERIFY.md` signed
- [ ] Experiment archive
- [ ] `verification/{tests.txt, smoke.txt, gates.txt, reproduction.txt}`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md`

## 11. Status journal

- 2026-05-27 -- Created from template by Doc-Agent-A.
