# H16 — Fibonacci Head Diversity (attention heads allocated by Fib counts)

> **One-line claim:** A ViT-Tiny with attention heads grouped in Fibonacci
> counts (1, 1, 2, 3, 5, 8 = 20 heads, with the corresponding dilations)
> matches uniform-12-head ViT-Tiny at +0.5-1.0 pp top-1 on CIFAR-100.
>
> **Source design space:** G2 Layer/Channel/Neuron (H11-H20).
>
> **Implementation status (this repo):**
> `o planned`.

This document is the committee-grade design write-up for hypothesis
H16.

---

## 1. Motivation (sacred / natural intuition, >= 100 words)

The number of attention heads in a Transformer is a design choice with
no strong theoretical basis. Standard choices (8, 12, 16) come from
empirical practice. Recent work (Voita et al 2019, Michel et al 2019)
showed that many heads are redundant and can be pruned with no
accuracy loss, suggesting that head allocation matters more than head
count. Biological multi-frequency processing in the auditory cortex
arranges critical bands at Fibonacci-spaced frequencies; the visual
cortex arranges complex cells at Fibonacci-spaced spatial frequencies.
The hypothesis is that allocating heads by Fibonacci count -- 1 head
of dilation 1, 1 head of dilation 2, 2 heads of dilation 3, 3 heads
of dilation 5, 5 heads of dilation 8, 8 heads of dilation 13, total
20 heads -- imposes the natural-system multi-frequency arrangement
on attention, increasing functional diversity and reducing redundancy.

## 2. Formal hypothesis (the falsifiable claim, >= 50 words)

Because Fibonacci-allocated attention heads (Fib counts at Fib
dilations) impose the cortical multi-frequency band allocation on the
attention layer, the mechanism by which they should beat uniform-head
ViT-Tiny is reduced head redundancy and matched spatial-frequency
coverage. Per Voita et al 2019 we expect CIFAR-100 top-1 to lift by
0.5-1.0 pp at near-iso-FLOPs (slightly more heads but smaller per-head
dimension).

## 3. Falsifier (>= 30 words)

If a Fibonacci-head ViT-Tiny (20 heads, Fib counts at Fib dilations)
does NOT match uniform-12-head ViT-Tiny on CIFAR-100 within +0.0 pp
top-1 at 3-seed median, the hypothesis is FALSIFIED.

## 4. Citations (Citation Rigor format, >= 80 words)

```
Voita, Elena, Talbot, David, Moiseev, Fedor, Sennrich, Rico, Titov,
Ivan 2019 ACL 'Analyzing Multi-Head Self-Attention: Specialized Heads
Do the Heavy Lifting, the Rest Can Be Pruned' (arXiv:1905.09418) --
the head-pruning paper that motivates non-uniform head allocation.

Michel, Paul, Levy, Omer, Neubig, Graham 2019 NeurIPS 'Are Sixteen
Heads Really Better than One?' (arXiv:1905.10650) -- complementary
analysis showing head redundancy; H16 reduces redundancy by Fibonacci
diversity.

Dosovitskiy, Alexey et al 2021 ICLR 'An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale' (arXiv:2010.11929) --
ViT scaffold. Standard 12 heads at d_model = 192 in ViT-Tiny.

Rao, Surya Narayanan, et al 2024 'Fibottention: Inceptive Visual
Representation Learning with Diverse Attention via Wythoff Array'
(arXiv:2406.19391) -- closest reference: Fibonacci dilation attention.
H16 reduces this to head count + per-head dilation only.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

Heads are an attention concept; the CNN analogue is **multi-dilation
parallel convolution**. For a 3x3 conv block, replace with K parallel
3x3 convs at Fibonacci dilations [1, 2, 3, 5, 8] and concatenate.

```python
class FibDilatedConv(nn.Module):
    def __init__(self, c_in, c_out, dilations=(1, 2, 3, 5, 8)):
        super().__init__()
        assert c_out % len(dilations) == 0
        c_per = c_out // len(dilations)
        self.convs = nn.ModuleList([
            nn.Conv2d(c_in, c_per, 3, padding=d, dilation=d)
            for d in dilations
        ])
    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], dim=1)
```

### 5.2 LLM / ViT track (the natural home)

Standard ViT-Tiny: d_model = 192, n_head = 12, d_head = 16, n_layer = 12.
Total attention params per layer: 4 * 192 * 192 = 147k.

Fibonacci head allocation: 20 heads at d_head = 12 (so total =
20 * 12 = 240, project back to 192 with linear). The 20 heads split
into Fibonacci groups by dilation:

- 1 head, dilation 1 (full attention)
- 1 head, dilation 2 (sliding window of stride 2)
- 2 heads, dilation 3
- 3 heads, dilation 5
- 5 heads, dilation 8
- 8 heads, dilation 13

Total dilation-pattern interaction sparsity is Fibonacci-cascaded
following the Wythoff array (per arXiv:2406.19391). The d_head = 12 is
not a power of 2; FlashAttention-2 requires d_head in {32, 64, 96, 128}
so we use d_head = 32 and total = 640, then project to 192. Total
attention params per layer: 192*640 + 640*192 = 247k (+68 pct vs
uniform-12).

Causal mask: per-head dilation mask is causal by construction.
KV cache: per-head; total cache grows ~67 pct.

```python
PHI = (1 + 5 ** 0.5) / 2

class FibHeadAttention(nn.Module):
    def __init__(self, d_model=192, head_dilations=None, d_head=32):
        super().__init__()
        # default: Fib counts at Fib dilations -- 20 heads total
        head_dilations = head_dilations or (
            [1]*1 + [2]*1 + [3]*2 + [5]*3 + [8]*5 + [13]*8
        )
        self.n_head = len(head_dilations)
        self.d_head = d_head
        self.dilations = head_dilations
        self.qkv = nn.Linear(d_model, 3 * self.n_head * d_head)
        self.out = nn.Linear(self.n_head * d_head, d_model)
    def forward(self, x):  # (B, T, D)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(*t.shape[:-1], self.n_head, self.d_head)
                   for t in qkv]
        outs = []
        for h in range(self.n_head):
            d = self.dilations[h]
            mask = _causal_dilated_mask(x.shape[1], d)
            outs.append(F.scaled_dot_product_attention(
                q[:, :, h], k[:, :, h], v[:, :, h], attn_mask=mask))
        return self.out(torch.cat(outs, dim=-1).flatten(-2))
```

Expected impact at 124M decoder scale: WikiText-103 ppl improves by
0.3-0.6; KV cache +50 pct; latency +30 pct.

## 6. Predicted Delta on 4090 benchmarks (the pre-registered prediction)

| metric | Delta vs baseline | rationale |
|---|---|---|
| composite | [-0.005, +0.015] | accuracy lift balanced against compute |
| top-1 (CIFAR-100, ViT-T) | [+0.5, +1.0] pp | head diversity |
| perplexity (WikiText-103 LLM) | [-0.6, -0.3] | reduced head redundancy |
| params | [+30, +70] pct | more heads, larger projections |
| FLOPs | [+25, +60] pct | proportional |
| GPU latency (batch=1) | [+20, +50] pct | per-head sequential ops |
| rotation-equivariance err | [-0.015, -0.005] | dilation diversity |
| KV cache @ 32k (LLM) | [+40, +70] pct | more heads |
| Betti collapse rate | [+0.03, +0.06] | diverse heads compress better |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- Dataset: **CIFAR-100**
- Architecture: ViT-Tiny scaffold (d_model = 192, n_layer = 12)
- Conditions: {uniform 12 heads, Fib 20 heads at Fib dilations, Fib
  count at uniform dilation, uniform 20 heads at Fib dilations}
- Epochs / batch / precision / seeds: 100 epochs, batch 256, bf16,
  seeds {0, 1, 2}
- Composite formula: existing project formula; SHA-256 pinned
- Run-script: `python scripts/run_sweep.py --config
  configs/h16_fib_head.yaml --seeds 0 1 2`
- Wall-clock: 4 configs * 3 seeds * ~30 min = ~360 min (6 h)
- Archive: `ideas/16_fib_head_diversity/experiments/
  exp001_cifar100_fib_head/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

ImageNet-100 (subset of ImageNet-1k) where head diversity matters
more due to higher class count. Predict +1-2 pp lift. Wall-clock:
~24 hours single seed.

### 7.3 Cross-paradigm context (LLM track)

124M decoder with Fib-head allocation on WikiText-103, 1 epoch.
Compare ppl + KV cache to uniform-head control. Budget: ~6 hours.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` section G2 row H16
- Master experiment list: `EXPERIMENT_LOG.md` (new T4.x row)
- Implementation sub-directory: `ideas/16_fib_head_diversity/`
- Related hypotheses that compose: H32 (Fibottention), H34 (golden
  RoPE), H37 (pentagonal attention)
- Related hypotheses that conflict: standard uniform-head ViT

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of Fibottention (arXiv:2406.19391)?**

> Fibottention is a *single* sparse-attention pattern across all heads
> using the Wythoff array. H16 allocates heads with Fibonacci *counts*
> per dilation group, which is orthogonal to Fibottention's pattern
> within a head. The two can be combined.

**Q: How is this falsifiable rather than aesthetic?**

> Section 3 commits to >= +0.0 pp top-1 at 3-seed median.

**Q: What if the prior helps on CIFAR-100 but hurts on ImageNet?**

> Section 7.2 is the ImageNet-100 bridge. If ImageNet-100 regresses,
> scope is restricted to small-dataset regimes.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H16 is a ViT-track prior; the CNN CIFAR sweep did not include ViT
> backbones. The previous compound-failure result does not apply.

**Q: How do we know the implementation is correct?**

> `tests/test_fib_head.py` asserts (a) head counts match Fib indices
> [1, 1, 2, 3, 5, 8], (b) per-head dilation matches the lookup, (c)
> causal-dilated mask is correctly lower-triangular.

## 10. Verification artifacts checklist (committee evidence pack)

- [ ] `ideas/16_fib_head_diversity/implementation.py`; tests green
- [ ] `ideas/16_fib_head_diversity/tests.py` >= 12 assertions
- [ ] `ideas/16_fib_head_diversity/AUDIT.md` >= 3 weaknesses
- [ ] `ideas/16_fib_head_diversity/IMPROVEMENTS.md`
- [ ] `ideas/16_fib_head_diversity/VERIFY.md` signed
- [ ] Experiment archive
- [ ] `verification/{tests.txt, smoke.txt, gates.txt, reproduction.txt}`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md`

## 11. Status journal

- 2026-05-27 -- Created from template by Doc-Agent-A.
