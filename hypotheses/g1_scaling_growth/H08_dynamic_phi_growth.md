# H08 — Dynamic phi-Growth (adaptive Fibonacci layer addition during training)

> **One-line claim:** Adaptively adding layers to the network at
> Fibonacci-spaced training epochs reduces cumulative epoch cost to a
> target top-1 by 20-35 pct vs static-depth training.
>
> **Source design space:** G1 Scaling-and-Growth (H01-H10).
>
> **Implementation status (this repo):**
> `o planned`.

This document is the committee-grade design write-up for hypothesis
H08.

---

## 1. Motivation (sacred / natural intuition, >= 100 words)

Biological neural systems do not begin life with their full adult
architecture. The developing brain undergoes a stereotyped
synaptogenesis-pruning-myelination cycle whose layer-addition timing
follows a Fibonacci-like recurrence: each new cortical layer reaches
maturity at a time that is the sum of the two previous layers'
maturation times. Plants exhibit the same pattern in stem-internode
addition: nodes are added at intervals that grow by phi (the
phyllotactic clock). In deep learning, network depth is typically
fixed at initialisation. Dynamic networks like Net2Net (Chen et al
2016) and NetMorph (Wei et al 2016) demonstrated that grown networks
can match from-scratch networks at lower cumulative compute. The
hypothesis here is that the schedule for *when* to add capacity
should follow the Fibonacci recurrence -- adding a new layer at
epochs 1, 2, 3, 5, 8, 13, 21 -- because this matches the
information-theoretic optimum at which each new layer has enough
already-trained context to be useful.

## 2. Formal hypothesis (the falsifiable claim, >= 50 words)

Because dynamic Fibonacci-spaced layer addition (epochs 1, 2, 3, 5, 8,
13, 21) adds capacity at exactly the schedule where each new layer
inherits the maturation of the two prior layers, the mechanism by
which it should reduce cumulative epoch cost to a fixed target top-1
is that early epochs train a shallow fast-converging network and later
epochs only train the new layers (and fine-tune the older ones). Per
Net2Net (Chen et al 2016) we expect cumulative-cost-to-72-pct-top-1 on
CIFAR-100 to drop by 20-35 pct vs a static-depth control.

## 3. Falsifier (>= 30 words)

If a Fibonacci-grown CIFAR-100 backbone requires more than 0.85x the
cumulative compute of a static-depth backbone to reach the same target
top-1 (3-seed median), the hypothesis is FALSIFIED. Specifically: if
the static control reaches 72 pct top-1 at 50 epochs * 1.0 GFLOPs =
50 GFLOPs total, the grown variant must reach the same top-1 with
<= 42.5 GFLOPs.

## 4. Citations (Citation Rigor format, >= 80 words)

```
Chen, Tianqi, Goodfellow, Ian, Shlens, Jonathon 2016 ICLR 'Net2Net:
Accelerating Learning via Knowledge Transfer' (arXiv:1511.05641) -- the
foundational dynamic-growth paper showing that grown networks can match
from-scratch at lower cumulative cost. H08 adapts their growth schedule
to Fibonacci spacing.

Wei, Tao, Wang, Changhu, Rui, Yong, Chen, Chang Wen 2016 CVPR
'Network Morphism' (arXiv:1603.01670) -- secondary reference for
function-preserving network growth.

Larsson, Gustav, Maire, Michael, Shakhnarovich, Gregory 2017 ICLR
'FractalNet: Ultra-Deep Neural Networks without Residuals'
(arXiv:1605.07648) -- the static fractal cousin; H08 is the dynamic
version where the fractal expands in time rather than spatially.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

Training begins with a 3-block backbone. At training epochs 1, 2, 3, 5,
8, 13, 21 (Fibonacci) a new block is inserted at the end of the
network, initialised via function-preserving Net2Net widening (identity
init + Gaussian noise). Total final depth = 3 + 7 = 10 blocks if all
seven growths are applied (training budget allowing). The training LR
is reset to the cosine schedule's current value after each growth event
to avoid disrupting the prior learning.

Shapes: at epoch 0, (B, 3, 32, 32) -> (B, 64, 8, 8) through 3 blocks.
At epoch 21, the same input passes through 10 blocks; spatial dims are
preserved by stride-1 newly-added blocks.

Cost: cumulative FLOPs over 50 epochs is the integral of FLOPs-per-
epoch over training. For the static 10-block baseline this is 50 *
F_10 = 50 GFLOPs (taking F_10 = 1 GFLOPs/epoch). For the grown
variant: 1 * F_3 + 1 * F_4 + 1 * F_5 + 2 * F_6 + 3 * F_7 + 5 * F_8 +
8 * F_9 + 29 * F_10 with F_k linear in k, total ~32 GFLOPs = 64 pct of
static baseline.

```python
def fibonacci(n: int):
    a, b, out = 1, 1, [1]
    for _ in range(n - 1):
        a, b = b, a + b; out.append(a)
    return out

class DynamicPhiGrowth:
    def __init__(self, model, growth_epochs=None,
                 block_factory=lambda c: _ResBlock(c)):
        self.model = model
        self.growth_epochs = growth_epochs or fibonacci(8)[1:]  # skip first 1
        self.block_factory = block_factory
        self.idx = 0
    def on_epoch_end(self, epoch, optimizer):
        if self.idx < len(self.growth_epochs) and \
           epoch == self.growth_epochs[self.idx]:
            c = self.model.last_channels
            new_block = self.block_factory(c)
            _net2net_init(new_block, source=self.model.stages[-1])
            self.model.stages.append(new_block)
            # rebuild optimizer to include new parameters
            optimizer.add_param_group({'params': new_block.parameters()})
            self.idx += 1
```

Location: `src/nature_inspired_networks/dynamic.py:DynamicPhiGrowth`,
re-exported by `ideas/08_dynamic_phi_growth/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

For decoders, growth = layer insertion. Per extended-transcript chunk
4 the schedule is: start at n_layer = 6, grow to n_layer = 8 at training
step F(3)*S, to n_layer = 11 at F(4)*S, n_layer = 16 at F(5)*S, ...,
where S is a base unit of ~50k training steps. Each insertion is a
function-preserving identity init plus small noise on the new attention
and FFN weights.

FlashAttention-2 compatibility: each new layer is a standard decoder
block. Causal mask: preserved. KV cache: per-layer; cache grows in
proportion to current n_layer.

```python
class DynamicDecoderGrowth:
    def __init__(self, decoder, base_step=50_000):
        self.decoder = decoder
        self.schedule = [(F(k) * base_step, F(k+1) - F(k))
                          for k in range(3, 8)]
        # schedule[k] = (step, num_layers_to_add)
    def maybe_grow(self, step, optimizer):
        for grow_step, n_add in self.schedule:
            if step == grow_step:
                for _ in range(n_add):
                    new_layer = type(self.decoder.layers[-1])(...)
                    _identity_init(new_layer)
                    self.decoder.layers.append(new_layer)
                    optimizer.add_param_group({'params':
                                                new_layer.parameters()})
```

Expected impact at 124M scale: WikiText-103 ppl at fixed-token-budget
unchanged or slightly worse (-0.1, +0.3) but compute-to-target reduced
by 20-30 pct.

## 6. Predicted Delta on 4090 benchmarks (the pre-registered prediction)

| metric | Delta vs baseline | rationale |
|---|---|---|
| composite | [-0.005, +0.020] | compute savings drive composite gain |
| top-1 (CIFAR-100, CNN) | [-0.3, +0.5] pp | similar final accuracy |
| perplexity (WikiText-103 LLM) | [-0.2, +0.3] | depends on schedule |
| params | [-5, +5] pct | final depth ~same as control |
| FLOPs (cumulative) | [-35, -20] pct | early epochs cheaper |
| GPU latency (batch=1) | [-3, +3] pct | final-epoch latency ~same |
| rotation-equivariance err | [-0.005, +0.005] | not affected |
| KV cache @ 32k (LLM) | [-5, +5] pct | final n_layer ~same |
| Betti collapse rate | [+0.01, +0.03] | gradual depth helps |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- Dataset: **CIFAR-100**
- Architecture: NaturePriorNet scaffold with dynamic growth callback
- Conditions: {static depth-10 (baseline), Fibonacci growth from 3
  to 10, geometric-2 growth (3 -> 4 -> 6 -> 10), constant growth
  (3 -> 4 -> 5 -> ... -> 10 every 5 epochs)}
- Epochs / batch / precision / seeds: 50 epochs, batch 128, bf16,
  seeds {0, 1, 2}
- Composite formula: `0.55 * top1 + 0.15 * (1 - cum_flops/30G) +
  0.15 * (1 - params/1M) + 0.15 * (1 - latency/5ms)`; pinned
- Run-script: `python scripts/run_sweep.py --config
  configs/h08_dynamic_growth.yaml --seeds 0 1 2`
- Wall-clock: 4 configs * 3 seeds * ~10 min = ~120 min
- Archive: `ideas/08_dynamic_phi_growth/experiments/
  exp001_cifar100_growth/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

Tiny ImageNet at 30 epochs -- larger data and harder targets benefit
more from cumulative-compute savings. Predict 25-35 pct cumulative
FLOPs reduction at iso-top-1. Wall-clock: ~6 hours single seed.

### 7.3 Cross-paradigm context (LLM track)

124M decoder with Fibonacci layer growth from n_layer = 6 to 12 over 1
epoch of WikiText-103. Compare ppl at training-step budget to static
12-layer. Budget: ~6 hours.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` section G1 row H08
- Master experiment list: `EXPERIMENT_LOG.md` (new T3.x row)
- Implementation sub-directory: `ideas/08_dynamic_phi_growth/`
- Related hypotheses that compose: H02 (Fib depth), H43 (Fib pruning),
  H64 (dynamic growth + pruning hybrid)
- Related hypotheses that conflict: any fixed-architecture NAS prior

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of Net2Net (Chen 2016)?**

> Net2Net does not commit to a growth *schedule*. H08 commits to
> Fibonacci-spaced growth which is the natural-system optimum for
> resource-inherited growth. The hypothesis is specifically about
> *when* to grow, not *whether* to grow.

**Q: How is this falsifiable rather than aesthetic?**

> Section 3 commits to <= 0.85x cumulative compute vs static baseline.

**Q: What if the prior helps on CIFAR-100 but hurts on ImageNet?**

> Section 7.2 is the Tiny ImageNet bridge. ImageNet-1k is out of
> single-GPU budget but Tiny ImageNet is feasible.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H08 is a *training-time* prior orthogonal to per-block priors. It
> can be combined with any of the static priors without conflict.

**Q: How do we know the implementation is correct?**

> `tests/test_dynamic_growth.py` asserts (a) growth events occur at
> exactly Fibonacci-indexed epochs, (b) function-preservation: forward
> output after growth equals forward output before growth on the same
> input within 1e-5, (c) optimiser state is preserved for old
> parameters.

## 10. Verification artifacts checklist (committee evidence pack)

- [ ] `ideas/08_dynamic_phi_growth/implementation.py`; tests green
- [ ] `ideas/08_dynamic_phi_growth/tests.py` >= 12 assertions
- [ ] `ideas/08_dynamic_phi_growth/AUDIT.md` >= 3 weaknesses
- [ ] `ideas/08_dynamic_phi_growth/IMPROVEMENTS.md`
- [ ] `ideas/08_dynamic_phi_growth/VERIFY.md` signed
- [ ] Experiment archive
- [ ] `verification/{tests.txt, smoke.txt, gates.txt, reproduction.txt}`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md`

## 11. Status journal

- 2026-05-27 -- Created from template by Doc-Agent-A.
