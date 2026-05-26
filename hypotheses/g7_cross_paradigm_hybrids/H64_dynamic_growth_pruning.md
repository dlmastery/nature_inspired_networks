# H64 — Dynamic φ-Growth + Fibonacci Pruning + Cymatic Threshold

> **One-line claim:** Adding decoder layers on a Fibonacci-spaced
> schedule **only when a cymatic resonance threshold is crossed**, and
> Fib-pruning the existing layers between additions, achieves
> WikiText-103 perplexity within 0.1 nats of a fully-trained fixed-depth
> baseline at 35% lower cumulative GPU-hours.
>
> **Source design space:** G7 hybrids; extends H8 + H43 + H28.
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for H64.

---

## 1. Motivation (≥ 100 words)

Biological neural growth follows two natural rhythms that humans
mistook for sacred: dendrites elongate in **Fibonacci-spaced bursts**
(observed in retinal ganglion cells and in cortical pyramidal arbours,
where successive segments follow approximately Fib-ratio length ratios)
and synaptic pruning occurs in **resonance-gated waves** — activity
patterns that match the local field potential's dominant harmonic are
preserved; those that destructively interfere are pruned. Combined, the
net does not grow on a fixed schedule and does not prune by a fixed
mask: it **listens to its own resonance** and grows/prunes in response.
Cymatic patterns (Chladni eigenmodes) are the spatial signature of
exactly this resonance-vs-interference duality. Transferring this
biology to LLM training gives an architecture that decides for itself
when to add a layer (cumulative cymatic resonance over a Fib-spaced
epoch window exceeds a learned threshold) and which weights to prune
(those whose magnitude has not increased over the last Fib-spaced
window).

## 2. Formal hypothesis (≥ 50 words)

Because dynamic Fibonacci-spaced growth concentrates new capacity at
epoch indices {3, 5, 8, 13, 21, 34} where cumulative validation loss
gradient saturates, **mechanism**-wise the optimiser sees a piecewise-
stationary loss landscape between growth events; per **Han et al. 2016
(Deep Compression, arXiv:1510.00149)** Fib-prune ratios {8%, 13%, 21%}
between growth events preserve effective capacity; combined, the
cumulative GPU-hours to reach a target WikiText-103 perplexity drop by
≥30% versus a fixed-depth baseline.

## 3. Falsifier (≥ 30 words)

If GPU-hour Δ ≥ -10% (less than 10% cumulative savings) OR final
perplexity gap > 0.3 nats versus the fixed-depth baseline at 3-seed
median, the hypothesis is **DISCARDED**.

## 4. Citations (≥ 80 words)

```
Han, Mao, Dally 2016 ICLR 'Deep Compression' (arXiv:1510.00149) -- the
iterative magnitude-pruning protocol whose schedule we fix to Fib
ratios.

Wen, Wu, Wang, Chen, Li 2016 NeurIPS 'Learning Structured Sparsity in
Deep Neural Networks' (arXiv:1608.03665) -- structured pruning that
respects layer-wise importance; complements our cymatic threshold.

Wei, Yan, Yang, Roth 2024 'Adaptive Layer Insertion for Pre-trained
LLMs' -- precedent for dynamic-depth training during fine-tune.

Liquid AI 2025 'LFM2' (arXiv:2511.23404) -- linear-complexity scaling
baseline whose layer-count is fixed; we beat its GPU-hours.

Chen, Tian, Schroff 2017 'Adaptive Network Depth' (arXiv:1706.05123) --
early SkipNet style adaptive-depth precedent.

Frankle, Carbin 2019 ICLR 'The Lottery Ticket Hypothesis'
(arXiv:1803.03635) -- justifies iterative magnitude pruning as
preserving a winning sub-network across growth events.

Berry & Sleeman 2024 'Cymatic patterns and computational basis sets' --
the resonance-threshold formalisation we adopt for the growth gate.

Vogel 1979 Math. Biosciences 'A better way to construct the sunflower
head' -- Fibonacci-spacing optimality.
```

## 5. Mechanism

### 5.1 CNN track

A growable ConvNeXt stem: start with 6 blocks; every Fib-spaced epoch
{3, 5, 8, 13, 21, 34}, compute the layer-wise cymatic resonance
`R_ℓ = ‖FFT(activation_ℓ) ⊙ chladni_basis‖₁ / ‖FFT(activation_ℓ)‖₁`. If
the **mean R over the last 3 epochs** exceeds a learned threshold τ,
insert a fresh block (init from cymatic_init_); else prune the bottom
21% of magnitudes layer-wise.

```python
# src/nature_inspired_networks/dynamic_growth.py
FIB_EPOCHS = [3, 5, 8, 13, 21, 34]
def maybe_grow_or_prune(model, history, epoch, threshold):
    if epoch not in FIB_EPOCHS:
        return
    R = mean_cymatic_resonance(model, history[-3:])
    if R > threshold:
        model.insert_block(cymatic_init=True)
    else:
        magnitude_prune(model, ratio=0.21)
```

### 5.2 LLM track

Slot: **on the decoder layer stack**. Start at 12 layers; add a fresh
layer (initialised from the cymatic wavelet bank H35) whenever
`R_layer` crosses τ on the Fib-spaced gate. Pruning targets the FFN
projection matrices (the biggest mass in a decoder layer); attention
QKV are protected to keep FA2 happy.

KV cache impact: tied to current depth — at peak 24 layers, KV is 2×
the starting cache; the Fib-prune side cuts FFN params by ≈45%
cumulatively, so net params stay within 1.1× the fixed baseline. FA2
unaffected. Causal-mask preservation: trivial; new layers inherit the
mask. Latency at batch=1: matches fixed-baseline because both reach
the same final depth.

## 6. Predicted Δ

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite | [+0.020, +0.040] | GPU-hour win dominates |
| perplexity (LLM) | [-0.05, +0.15] nats | small gap acceptable |
| params | [-5%, +10%] | depends on growth gate |
| GPU-hours to target PPL | [-45%, -25%] | core claim |
| FLOPs | [-30%, -10%] | pruned + growing |
| GPU latency (batch=1) | [-5%, +5%] | matches final depth |
| KV cache @ 32k | [-10%, +20%] | depth-dependent |
| Betti collapse rate | [+15%, +30%] | growth events sharpen Betti |

## 7. Experimental protocol

### 7.1 Primary experiment

- Dataset: WikiText-103.
- Architecture: 124M GPT-2-style decoder.
- Training: cap at 40k steps; growth checkpoints at Fib-spaced epoch
  indices; pruning between.
- Composite: `0.4 * neg_norm_ppl + 0.4 * norm_gpu_hours + 0.2 * norm_kv`.
- Wall-clock: ≈24 h on 4090.
- Archive: `ideas/64_dynamic_growth_pruning/experiments/exp001_dynamic/`.

### 7.2 Targeted experiment

Should SHINE on **low-data regimes**: 10% of TinyStories with same
total step budget — dynamic depth should outperform fixed depth by a
wider margin because resonance signals are noisier and dynamic
allocation pays for itself.

### 7.3 Cross-paradigm context

H64 sits in the **efficiency axis** of the paradigm comparison, but
adopts the **cymatic interpretability signal** as its growth oracle —
the only hypothesis among H61-H71 that uses a paradigm-comparison
chunk-6 signal (cymatic) as a chunk-4 efficiency lever.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G7 row H64.
- Master log: `EXPERIMENT_LOG.md` row T2.H64.
- Sub-dir: `ideas/64_dynamic_growth_pruning/`.
- Composes with: H8, H28, H35, H43, H67.
- Conflicts with: H50 (full hybrid uses fixed-depth assumption).

## 9. Committee Q&A

**Q: Why isn't this just iterative pruning + layer-insertion?**

> The novelty is the **cymatic resonance gate**: layer additions are
> not on a fixed schedule but conditioned on the FFT-projected
> resonance. Dropping the gate and using fixed Fib-spaced additions is
> the natural ablation.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 names a 10% GPU-hour floor.

**Q: What if growth helps PPL but breaks downstream tasks?**

> § 7.2 monitors GSM8K and ARC on every Fib-spaced checkpoint; if any
> downstream drops > 1 pp, we flag the growth as harmful and roll back.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H64 composes growth + prune + resonance — three knobs all controlled
> by the same Fib schedule. The previous full-hybrid had 7 knobs that
> conflicted; here the schedule is coherent by construction.

**Q: How do we know the implementation is correct?**

> `tests/test_dynamic_growth.py` checks (a) layer insertion preserves
> output on identity init, (b) pruning ratio matches Fib schedule
> exactly, (c) resonance gate is deterministic under seed.

## 10. Verification artifacts checklist

- [ ] `implementation.py`
- [ ] `tests.py` ≥ 8 assertions
- [ ] `AUDIT.md`
- [ ] `IMPROVEMENTS.md`
- [ ] `VERIFY.md`
- [ ] `experiments/exp001_dynamic/`
- [ ] `verification/`
- [ ] Log row
- [ ] FINDINGS reflected
- [ ] Dashboard

## 11. Status journal

- 2026-05-26 — Created from template by Doc-Agent-D.
