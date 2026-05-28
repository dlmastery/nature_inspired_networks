# H38 — Fractal Golden Filter

> **One-line claim:** A multi-scale convolutional filter composed of stacked 3×3, 5×5, and 8×8 (Fibonacci-sized) kernels with φ-scaled weight ratios applied at every conv layer raises top-1 on natural-image benchmarks and accelerates Betti-collapse versus a single-scale baseline of matched parameter budget.
>
> **Source design space:** G4 Kernels / Attention / Filters (H31–H40).
>
> **Implementation status (this repo):** `○ not started`.

This document is the committee-grade design write-up for hypothesis H38.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

Fractal self-similarity at φ-scales is one of nature's most universal organization principles: cortical folding, vascular branching, lung-alveoli structure, river networks, lightning strikes, fern leaves, and coastline statistics all show fractal dimension between 2.0 and 2.6 with self-similar substructure at successive scales. The mathematical signature is that the structure looks similar at every magnification with a fixed scale ratio — and that ratio, empirically, is often φ or close to it (Mandelbrot 1982).

For deep learning, multi-scale kernel architectures (Inception, OctConv, HRNet) demonstrate that simultaneous processing at multiple kernel sizes improves accuracy. **H38 is the explicit φ-scaled / Fibonacci-sized realization**: at every conv layer, three parallel kernels of sizes {3, 5, 8} (consecutive Fibonacci numbers) process the input, with the three branches' weights scaled by `{1, 1/φ, 1/φ²}` to favor the local kernel while still receiving signal from the larger scales. This is the fractal generalization of the Vesica Piscis filter (H33) and shares its multi-scale motivation, but adds the **explicit Fibonacci kernel-size progression** and **φ-decaying weight ratios** at every layer.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** φ-scaled fractal self-similarity is empirically universal in natural images and Fibonacci kernel-size progressions match log-spaced receptive-field statistics, applying a 3-branch fractal-φ filter `{3×3, 5×5, 8×8}` with weight ratios `{1, 1/φ, 1/φ²}` at every conv layer raises CIFAR-100 top-1 by ≥ +1 pp and accelerates Betti-collapse rate by ≥ 0.10 versus a single-scale 5×5 baseline at matched parameter budget, per the mechanism of Larsson 2017 FractalNet (arXiv:1605.07648) and Mandelbrot 1982.

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on CIFAR-100, the fractal-golden filter fails to lift top-1 by ≥ 0.5 pp AND fails to accelerate Betti-β₁ collapse by ≥ 0.05 versus the parameter-matched 5×5 baseline, this hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Larsson, G., Maire, M., Shakhnarovich, G. 2017 ICLR 'FractalNet'
(arXiv:1605.07648) — primary fractal-network reference.

Mandelbrot, B. 1982 'The Fractal Geometry of Nature' — fractal self-
similarity in nature.

Szegedy, C., et al. 2015 — Inception multi-branch reference.

Chen, Y., et al. 2019 — OctConv multi-scale.

Wang, X., et al. 2019 — Big-Little Net.

Edelsbrunner, H. 2000 — persistent-homology Betti collapse metric.

Krizhevsky 2009 — CIFAR-100 dataset.
```

## 5. Mechanism

### 5.1 CNN track

At every conv layer, run 3 parallel branches with kernels {3, 5, 8} and combine outputs with weights `{1, 1/φ, 1/φ²}` (normalized to sum to 1).

```python
# ideas/38_fractal_golden_filter/implementation.py
PHI = (1+5**0.5)/2
FIB_K = [3, 5, 8]

class FractalGoldenFilter(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        weights = torch.tensor([1.0, 1/PHI, 1/PHI**2])
        weights = weights / weights.sum()
        self.register_buffer("branch_w", weights)
        # split channels across branches to keep total params similar
        Cs = [C_out // 3, C_out // 3, C_out - 2*(C_out // 3)]
        self.branches = nn.ModuleList([
            nn.Conv2d(C_in, c, k, padding=k//2) for c, k in zip(Cs, FIB_K)
        ])

    def forward(self, x):
        outs = []
        for w, branch in zip(self.branch_w, self.branches):
            outs.append(w * branch(x))
        return torch.cat(outs, dim=1)
```

- Input shape: `(B, C_in, H, W)`.
- Output: `(B, C_out, H, W)` with channels split per branch.
- Params: roughly equal to a single 5×5 of the same C_out (the 3×3 branch saves params; the 8×8 branch costs more; nets out near 5×5).
- FLOPs: parallel branches; aggregate ≈ 1.0–1.4× single-5×5.
- Init: He init per branch, scaled by `1/sqrt(branch_w)`.

### 5.2 LLM track

For decoder-only Transformers, the fractal-golden filter applies to **multi-scale FFN**: replace the standard 2-layer FFN with 3 parallel branches of width {d, d·φ, d·φ²} mixed by `{1, 1/φ, 1/φ²}` weights. Each branch processes the input at a different "scale" of width.

```python
class FractalGoldenFFN(nn.Module):
    def __init__(self, d, expansion=4):
        super().__init__()
        sizes = [int(d*expansion), int(d*expansion*PHI), int(d*expansion*PHI**2)]
        weights = torch.tensor([1.0, 1/PHI, 1/PHI**2])
        self.branch_w = weights / weights.sum()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(d, s), nn.GELU(), nn.Linear(s, d))
            for s in sizes
        ])
    def forward(self, x):
        return sum(w*b(x) for w, b in zip(self.branch_w, self.branches))
```

Expected at 124 M scale: **-0.05 to -0.15 perplexity**, +25-40 % parameters.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (CIFAR-100) | [+0.005, +0.020] | multi-scale prior |
| top-1 (CIFAR-100, primary) | [+1.0 pp, +3.0 pp] | direct claim |
| Betti collapse rate | [+0.10, +0.30] | direct claim |
| params | [-5 %, +20 %] | depends on Fib split |
| FLOPs | [+10 %, +40 %] | 8×8 branch is expensive |
| GPU latency | [×1.3, ×1.7] | three branches |
| Perplexity (LLM) | [-0.15, -0.05] | small positive |
| KV cache | [0, 0] | unchanged |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** CIFAR-100.
- **Architecture:** NaturePriorBlock with `FractalGoldenFilter` replacing 3×3 conv.
- **Epochs / batch / precision / seeds:** 25 epochs, batch 128, bf16, 3 seeds.
- **Composite:** top-1 (0.5), latency (0.2), params (0.15), Betti collapse (0.15).
- **Run-script:** `python ideas/38_fractal_golden_filter/experiment.py --seeds 0 1 2`.
- **Wall-clock:** ~35 min/seed × 3 = 105 min.
- **Archive:** `ideas/38_fractal_golden_filter/experiments/exp001_cifar100_seed0..2/`.

### 7.2 Idea-targeted experiment

1. **CIFAR-100** for multi-scale benchmark.
2. **Tiny-ImageNet** for larger receptive-field testing.
3. **Cityscapes-mini** for multi-scale segmentation.

### 7.3 Cross-paradigm context

LLM-track: WikiText-103 124 M with FractalGoldenFFN. 50 k steps.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G4 row H38.
- Master: planned Tier 2/3.
- Sub-dir: `ideas/38_fractal_golden_filter/`.
- Composes: H05 (fractal recursion — positive single result), H07 (φ multi-scale), H33 (Vesica Piscis), H32 (Fibottention).
- Conflicts: H31 (golden-spiral kernel — different kernel structure).

## 9. Committee Q&A

**Q: Why isn't this just Inception with Fibonacci kernel sizes?**

> The explicit additions are (a) Fibonacci kernel SIZE PROGRESSION (not arbitrary {1, 3, 5}), (b) φ-decaying WEIGHT RATIOS across branches, (c) full-coverage at EVERY layer (Inception uses multi-scale only at specific stages).

**Q: How is this falsifiable rather than aesthetic?**

> § 3 gives top-1 ≥ +0.5 pp AND Betti collapse ≥ +0.05. Both must hold.

**Q: What if 5×5 single-scale matches at the same compute budget?**

> The control is a parameter-matched 5×5 baseline. If they tie, the multi-scale prior fails.

**Q: T1.5 fractal alone was +2.35 pp on CIFAR-10 but 2× params. Why expect fractal filter to be cheaper?**

> T1.5 is fractal-RECURSION (replicates the block structure). H38 is fractal-FILTER (parallel branches inside one block); the parameter scaling is different (1.0-1.4× rather than 2×).

**Q: How do we know the implementation is correct?**

> `tests/test_fractal_filter.py::test_fib_kernel_sizes` asserts kernels are {3,5,8}. `test_phi_decay_weights` asserts branch weights are {1, 1/φ, 1/φ²} (normalized). `test_output_shape` asserts spatial dims preserved. Plus archive verification.

## 10. Verification artifacts checklist

- [ ] `ideas/38_fractal_golden_filter/implementation.py` tests green
- [ ] `ideas/38_fractal_golden_filter/tests.py` ≥ 5 assertions
- [ ] AUDIT.md ≥ 3 weaknesses
- [ ] IMPROVEMENTS.md
- [ ] VERIFY.md signed
- [ ] Experiment archives + verification
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] FINDINGS reflects result

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-B.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G4 (elite-research-scientist critic). Critiquing the IDEA, not the implementation (audit at `audits/G4_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)

**MED.** Multi-scale conv is a WELL-ESTABLISHED idea (Inception, OctConv, HRNet) that consistently shows modest gains. {3, 5, 8} kernel sizes are a reasonable multi-scale set. The MED rating reflects "multi-scale works" + "the Fibonacci/φ-specificity is unjustified" + "8×8 kernels are expensive enough to dominate latency".

### Mechanism scrutiny

The 8×8 kernel is the dominant FLOPs contributor (8² = 64 vs 3² + 5² = 34; the 8×8 branch alone exceeds the other two combined). The author's table predicts FLOPs [+10%, +40%] and latency [×1.3, ×1.7] — substantial costs. The matched-parameter comparison is against a "single-scale 5×5 baseline" (which has 25 FLOPs/pixel), but the variant uses 3²+5²+8² = 98 FLOPs/pixel summed across branches. THIS IS NOT FLOP-MATCHED. To match FLOPs, the baseline must be a 9×9 or larger single-kernel — at which point the comparison becomes "deep multi-scale vs deep wide single-scale", and the literature consensus is that single-scale wide kernels (ConvNeXt: Liu 2022 arXiv:2201.03545 uses 7×7) MATCH multi-scale on CIFAR/ImageNet.

The φ-decaying weight ratio `{1, 1/φ, 1/φ²}` ≈ `{0.62, 0.24, 0.14}` after normalization. This means the 3×3 branch contributes 62% of the output, 5×5 contributes 24%, 8×8 contributes 14%. The 8×8 branch is paying 64 FLOPs/pixel for 14% of output mass — a TERRIBLE compute/output ratio. The φ-decay PUNISHES the high-FLOPs branch. The hypothesis predicts gains but the mechanism actively SUPPRESSES the most expensive branch — internally inconsistent.

### Confounds (≥2)

1. **Channel-split confound.** `C_out // 3` per branch reduces each branch's capacity. The 8×8 branch with `C_out/3` channels is comparable to a 1/3-width conv. Total params depend on the C_out/3 split; small deviations from "matched params" can be misleading.
2. **Branch-weight confound.** The fixed `{1, 1/φ, 1/φ²}` weights cannot be undone by a downstream layer because the channels are CONCATENATED (not added). Any subsequent BatchNorm scales-per-channel, but the cross-branch ratio is fixed by the concat order. Control: equal weights {1/3, 1/3, 1/3} or learnable weights.

### Numerology / specificity check

{3, 5, 8} are Fibonacci numbers but they are ALSO {odd, odd, even} = {small, medium, large} — any 3 ascending kernel sizes (e.g., {3, 5, 7} as in Inception-v3, {3, 5, 9} arithmetic) would test "multi-scale" equally. The Fibonacci specificity is undetectable. The φ-decaying weights {1, 0.62, 0.38} are also unfalsifiable: any weights with monotonic decay produce similar mixing. The combination "Fibonacci kernels + φ decay" stacks TWO numerological choices, neither individually justified. **High numerology score.**

The Larsson 2017 FractalNet (arXiv:1605.07648) citation is misapplied: FractalNet is about FRACTAL DEPTH RECURSION (block self-similarity), NOT about Fibonacci kernel sizes inside a single block. The "fractal" framing is rebranding.

### Literature precedent — kernel/attention design is a crowded field

Multi-scale conv: Inception-v1/v2/v3 (Szegedy 2015 arXiv:1409.4842, 2016 arXiv:1512.00567), Inception-ResNet (Szegedy 2016 arXiv:1602.07261), Res2Net (Gao 2021 arXiv:1904.01169), HRNet (Sun 2019 arXiv:1908.07919), Big-Little Net (Wang 2019 arXiv:1807.03848). FractalNet (Larsson 2017 arXiv:1605.07648) is about depth recursion. Recent work on LARGE kernels (ConvNeXt: Liu 2022 arXiv:2201.03545; RepLKNet: Ding 2022 arXiv:2203.06717 uses 31×31) finds wide single-kernel often matches multi-scale. No precedent for Fibonacci-sized multi-scale.

### Expected effect size (90% CI a priori)

On CIFAR-100 at matched FLOPs (not matched params!): [-0.3 pp, +0.8 pp] top-1 vs Inception {1,3,5} or ConvNeXt-7. The author's [+1.0, +3.0] is too optimistic by ~3×. Latency cost [×1.3, ×1.7] is the major issue.

### Minimum-distinguishing experiment

3-seed CIFAR-100 at matched FLOPs: (a) single 5×5, (b) single 7×7, (c) Inception {1,3,5} equal-weight, (d) fractal {3,5,8} with {1, 1/φ, 1/φ²}, (e) fractal {3,5,8} with EQUAL weights, (f) {3,5,7} arithmetic with φ-decay. If (d) > (e), (f) by ≥ 0.3 pp, the Fibonacci+φ combo is non-null. Most likely (d) ≈ (e) ≈ (f) ≈ (c) within 0.5 pp.

### Verdict
DERIVATIVE+TESTABLE — multi-scale conv works and the design is implementable, but the Fibonacci kernel sizes and φ-weight decay are NOT pre-registered as falsifiable against equal-weight + arithmetic-progression controls. The 8×8 branch with 14% output weight is internally inconsistent (paying full FLOPs for fractional contribution). FractalNet citation is misapplied (FractalNet is depth recursion, not kernel scale).
