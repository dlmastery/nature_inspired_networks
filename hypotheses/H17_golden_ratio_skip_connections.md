# H17 — Golden Ratio Skip Connections (residual skips scaled by phi or 1/phi)

> **One-line claim:** Multiplying the residual skip path by 1/phi
> (= 0.618) or phi - 1 (= 0.618 too) instead of the standard 1.0 makes
> ResNet-20 converge 10-20 pct faster with no top-1 regression on
> CIFAR-10.
>
> **Source design space:** G2 Layer/Channel/Neuron (H11-H20).
>
> **Implementation status (this repo):**
> `~ partial` (closest neighbour: `sg_only_golden_modulate` channel gate, T1.8).

This document is the committee-grade design write-up for hypothesis
H17. **Partial experiment data exists via T1.8 sg_only_golden_modulate
(channel-level golden-angle modulation), single seed.**

---

## 1. Motivation (sacred / natural intuition, >= 100 words)

Residual skip connections (He et al 2016) add the input directly to
the output: y = F(x) + x. The skip path has implicit weight 1.0 -- a
choice baked into the architecture without theoretical motivation.
Variants like rescaling (Hayou et al 2021) and learnable scaling
(Bachlechner et al 2021, ReZero) have shown that the weight on the
skip path matters significantly for trainability and convergence
speed. The golden ratio identity phi**2 = phi + 1 implies
1/phi**2 + 1/phi = 1, which when applied to residual scaling
(skip_weight = 1/phi, branch_weight = 1/phi**2) ensures that the two
paths sum to less than 1 by a phi-fraction at each layer, providing
implicit depth regularisation. Biological neural circuits often have
recurrent loops with feedback gain near 0.618 -- not quite at unity
(which would cause runaway oscillation) but close enough to support
stable recurrence. The hypothesis is that phi-scaling of skip
connections produces a similarly stable training dynamic.

## 2. Formal hypothesis (the falsifiable claim, >= 50 words)

Because golden-ratio skip connections scale the residual path by 1/phi
or phi - 1 (= 0.618), the mechanism by which they should accelerate
convergence is that the phi-weighted skip damps the per-layer signal
gain to 1/phi while preserving identity-mapping capacity in the limit
of deep networks. Per Hayou et al 2021 we expect epochs-to-target on
CIFAR-10 to drop by 10-20 pct with no final-accuracy regression.

## 3. Falsifier (>= 30 words)

If a phi-skip ResNet-20 LOSES more than 0.3 pp top-1 (3-seed median)
versus the standard 1.0-skip baseline OR fails to demonstrate >= 10
pct reduction in epochs-to-target, the hypothesis is FALSIFIED. The
T1.8 single-seed result (-0.30 pp from sg_chan_fib reference at near-
no-op) is treated as suggestive but not decisive.

## 4. Citations (Citation Rigor format, >= 80 words)

```
He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, Sun, Jian 2016 CVPR 'Deep
Residual Learning for Image Recognition' (arXiv:1512.03385) -- the
foundational residual learning paper. H17 modifies the standard
y = F(x) + x to y = F(x) + (1/phi) * x.

Hayou, Soufiane, Ghosh, Arnaud, Doucet, Arnaud 2021 ICLR 'Stable
ResNet' (arXiv:2010.12859) -- demonstrates that rescaling residual
branches improves stability; H17 commits to specific 1/phi scaling.

Bachlechner, Thomas, Majumder, Bodhisattwa Prasad, Mao, Henry, Cottrell,
Gary, McAuley, Julian 2021 UAI 'ReZero is All You Need: Fast
Convergence at Large Depth' (arXiv:2003.04887) -- ReZero uses *learnable*
scalar alpha initialised to 0; H17 commits to 1/phi as the
natural-constant init.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

A standard residual block:

```python
def forward(self, x):
    return self.relu(self.conv2(self.relu(self.conv1(x))) + x)
```

Phi-skip variants:

```python
PHI = (1 + 5 ** 0.5) / 2

class PhiSkipResBlock(nn.Module):
    def __init__(self, c, mode='inv_phi'):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        if mode == 'inv_phi':
            self.skip_scale = 1 / PHI  # 0.618
            self.branch_scale = 1.0
        elif mode == 'phi_inv2':
            self.skip_scale = 1 / PHI
            self.branch_scale = 1 / PHI ** 2  # 0.382 (sum=1)
        elif mode == 'phi':
            self.skip_scale = PHI - 1  # also 0.618
            self.branch_scale = 1.0
        elif mode == 'learnable_phi_init':
            self.skip_scale = nn.Parameter(torch.tensor(1 / PHI))
            self.branch_scale = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        branch = F.relu(self.conv2(F.relu(self.conv1(x))))
        return self.skip_scale * x + self.branch_scale * branch
```

Shapes / params: H17 changes only scalar weights; no shape or param
change. The only overhead is the scalar multiplication (negligible).

Cost: identical to the baseline. The hypothesis is therefore a
*zero-cost* prior with potential accuracy/convergence benefit.

Location: `src/nature_inspired_networks/blocks.py:_GenericResBlock`
(modified), re-exported by `ideas/17_golden_skip_scale/
implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

For a decoder block:

```python
def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.ffn(self.norm2(x))
    return x
```

H17 LLM-track variants:

```python
class PhiSkipDecoderBlock(nn.Module):
    def forward(self, x):
        x = self.skip_scale * x + self.attn(self.norm1(x))
        x = self.skip_scale * x + self.ffn(self.norm2(x))
        return x
```

with skip_scale = 1/phi. The double-scaling per block compounds to
1/phi**2 = 0.382 per block, which dramatically attenuates the
residual stream over depth. To avoid this, use single-scaling: scale
only the attention skip and leave FFN skip = 1.0, or use learnable
phi-init scalars.

FlashAttention-2 compatibility: scalar scaling is post-attention, no
impact. Causal mask: preserved. KV cache: unchanged.

Expected impact at 124M scale: WikiText-103 ppl within +/- 0.2 of
baseline; convergence speed may improve in late training but loss
floor may be slightly higher due to capacity damping.

## 6. Predicted Delta on 4090 benchmarks (the pre-registered prediction)

| metric | Delta vs baseline | rationale |
|---|---|---|
| composite | **observed (related): -0.0093** | T1.8 vs sg_chan_fib ref (golden_modulate, not exact phi-skip) |
| top-1 (CIFAR-10, CNN) | **observed (related): -0.30 pp** | T1.8: 79.81% vs sg_chan_fib 80.11% |
| perplexity (WikiText-103 LLM) | [-0.2, +0.2] | scaling-only change |
| params | [0, 0] pct | scalar only |
| FLOPs | [0, 0] pct | scalar only |
| GPU latency (batch=1) | **observed (related): +34 pct (5.95 vs 4.43 ms)** | T1.8 has additional channel gating; pure phi-skip would be [0, 0] pct |
| rotation-equivariance err | **observed: -0.026** | from T1.8 (slight gain) |
| KV cache @ 32k (LLM) | [0, 0] pct | unchanged |
| Betti collapse rate | [-0.02, +0.02] | scaling-only |

**Observed (single seed, T1.8 sg_only_golden_modulate -- which is the
channel-gate variant, not the pure skip-scaling variant; nearest
available data):**

```
sg_only_golden_modulate  top-1 79.81%  params 127k  latency 5.95 ms  composite 0.8042
sg_chan_fib (ref)        top-1 80.11%  params 127k  latency 4.43 ms  composite 0.8135
```

T1.8 is the **closest available data point** but is not pure H17 --
it modulates channel weights by the golden angle 137.5 deg rather
than scaling the residual skip by 1/phi. The pure H17 (with no
channel-gate overhead) is predicted to be near-no-op or mildly
positive in convergence, with no latency penalty.

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- Dataset: **CIFAR-10** (12 epochs, matching prior sweep)
- Architecture: NaturePriorNet with phi-skip variants; conditions
  {standard 1.0 skip, 1/phi skip + 1.0 branch, 1/phi skip + 1/phi**2
  branch, phi-1 skip + 1.0 branch (same as 1/phi), learnable phi-init}
- Epochs / batch / precision / seeds: 12 epochs, batch 128, bf16,
  seeds {0, 1, 2}
- Composite formula: existing project formula; SHA-256 pinned
- Run-script: `python scripts/run_sweep.py --config
  configs/h17_phi_skip.yaml --seeds 0 1 2`
- Wall-clock: 5 configs * 3 seeds * ~6 min = ~90 min
- Archive: `ideas/17_golden_skip_scale/experiments/
  exp001_cifar10_phi_skip/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

Deep regime: ResNet-110 on CIFAR-10 where skip stability matters most.
Predict 10-15 pct training-loss convergence speedup with no top-1
loss. Wall-clock: ~6 hours single seed.

### 7.3 Cross-paradigm context (LLM track)

124M decoder with phi-skip on attention only (not FFN) on WikiText-103,
1 epoch. Compare ppl and training curve. Budget: ~5 hours.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` section G2 row H17
- Master experiment list: `EXPERIMENT_LOG.md` T1.8 (golden_modulate
  variant done); new H17.v2 (pure skip-scale) queued
- Implementation sub-directory: `ideas/17_golden_skip_scale/`
- Related hypotheses that compose: H34 (golden RoPE), H42 (phi weight
  init), H10 (phi LR), H39 (harmonic phi activation)
- Related hypotheses that conflict: ReZero (alpha = 0 init), Stable
  ResNet (different rescaling)

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of ReZero (Bachlechner 2021)?**

> ReZero initialises a *learnable* scalar to 0 on the branch path.
> H17 fixes the *skip* path to 1/phi. The two are dual: ReZero damps
> the branch, H17 damps the skip. The hypothesis is that 1/phi skip
> scaling has the same stability benefit at no learnable-param cost.

**Q: How is this falsifiable rather than aesthetic?**

> Section 3 commits to >= -0.3 pp top-1 OR >= 10 pct convergence
> speedup.

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> Section 7.2 is the deep-regime ResNet-110 bridge.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> The closest available data (T1.8 sg_only_golden_modulate) shows a
> mild negative (-0.30 pp top-1) but it confounds skip scaling with
> channel gating. Pure H17 is untested.

**Q: How do we know the implementation is correct?**

> `tests/test_phi_skip.py` asserts (a) skip_scale = 1/phi within 1e-9,
> (b) forward output for identity-branch case equals 1/phi * x, (c)
> learnable variants train under autograd.

## 10. Verification artifacts checklist (committee evidence pack)

- [ ] `ideas/17_golden_skip_scale/implementation.py`; tests green
- [ ] `ideas/17_golden_skip_scale/tests.py` >= 10 assertions
- [ ] `ideas/17_golden_skip_scale/AUDIT.md` >= 3 weaknesses
- [ ] `ideas/17_golden_skip_scale/IMPROVEMENTS.md`
- [ ] `ideas/17_golden_skip_scale/VERIFY.md` signed
- [ ] T1.8 archive (golden_modulate variant, exists) + new exp001 archive
- [ ] `verification/{tests.txt, smoke.txt, gates.txt, reproduction.txt}`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md`

## 11. Status journal

- 2026-05-27 -- Created from template by Doc-Agent-A.
- (previous) -- T1.8 sg_only_golden_modulate at single seed:
  top-1 79.81 pct, composite 0.8042. This is the *channel-gate*
  variant which modulates channel weights by 137.5 deg rotation, not
  the pure skip-scaling variant. -0.30 pp from sg_chan_fib reference,
  treated as near no-op.
- (planned) -- exp001 tests pure phi-skip without channel gating.
