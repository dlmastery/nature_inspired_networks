# H66 — Cymatic Wavelet QKV Kernel with Dynamic Resonance

> **One-line claim:** Initialising **and** φ-harmonically modulating the
> Q, K, V projection matrices with a Chladni-eigenmode wavelet bank
> reduces WikiText-103 perplexity at 350M scale by ≥0.3 nats while
> preserving FlashAttention-2 compatibility.
>
> **Source design space:** G7 hybrids; composition of H28 + H35 + H46.
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for H66.

---

## 1. Motivation (≥ 100 words)

Chladni eigenmodes — the standing-wave patterns sand assumes on a
vibrating plate — form an **orthonormal basis** for 2-D scalar fields
with the additional property that the basis functions are spatially
multi-scale and naturally exhibit nested nodal lines (zero-crossings).
Two natural systems use exactly this basis: the basilar membrane in
mammalian hearing decomposes audio into a Chladni-like frequency
lattice, and visual cortex V1 simple cells are well-approximated by
Gabor wavelets that are themselves smoothed Chladni modes on a circular
plate. The Q, K, V projection matrices in a Transformer are the
**only learned bases inside an attention layer**; replacing the random
Gaussian initialisation with a φ-spaced Chladni-mode bank gives the
attention head a built-in inductive bias toward multi-scale frequency
decomposition. Dynamic φ-harmonic modulation (rotating the bank phase
during training) is the cymatic analogue of position-dependent gauge —
it allows the bank to "tune" to the data's resonant frequencies.

## 2. Formal hypothesis (≥ 50 words)

Because the QKV matrices are initialised on a φ-spaced Chladni
eigenmode basis with K modes, **mechanism**-wise attention heads
inherit a multi-scale frequency prior that is orthogonal at init and
remains near-orthogonal through fine-tuning; per **Bracewell 1986
(Fourier Transform and its Applications)** and the **Cymatic-pattern
arguments of chunk-6**, this prior accelerates convergence and reduces
final perplexity by ≥0.3 nats at 350M scale on WikiText-103.

## 3. Falsifier (≥ 30 words)

If WikiText-103 perplexity Δ ≥ -0.10 nats at 3-seed median, OR if QKV
projections lose >50% of their initial orthogonality during fine-tune
(suggesting the prior is washed out), the hypothesis is **DISCARDED**.

## 4. Citations (≥ 80 words)

```
Chladni 1787 'Entdeckungen über die Theorie des Klanges' -- the
historical basis; we use the modern numerical solver of the wave
equation for square plates.

Daubechies 1992 SIAM 'Ten Lectures on Wavelets' -- the orthonormal-
wavelet-basis theory we adopt for the Chladni-mode bank.

Saxe, McClelland, Ganguli 2014 ICLR 'Exact solutions to the nonlinear
dynamics of learning in deep linear neural networks'
(arXiv:1312.6120) -- justifies orthogonal initialisation as a faster
convergence prior.

Dao 2024 'FlashAttention-2: Faster Attention with Better Parallelism'
(arXiv:2307.08691) -- compatibility target; cymatic init is just a
weight-init change so FA2 unaffected.

Vaswani et al. 2017 NeurIPS 'Attention is All You Need'
(arXiv:1706.03762) -- the architecture we modify.

Tancik, Srinivasan, Mildenhall, Fridovich-Keil, Raghavan, Singhal,
Ramamoorthi, Barron, Ng 2020 NeurIPS 'Fourier Features Let Networks
Learn High Frequency Functions in Low Dimensional Domains'
(arXiv:2006.10739) -- closely-related Fourier-feature initialisation
that inspired our φ-spacing.

Rastegari et al. 2024 'Fibottention' (arXiv:2406.19391) -- Fib
dilations per head; we use it to set the per-head Chladni-mode index.
```

## 5. Mechanism

### 5.1 CNN track

Replace QKV initialisation in any attention block with `cymatic_init_`
already wired in the repo (H35). Add a **per-step φ-harmonic phase**
multiplier applied to the V output (only V — Q/K are kept geometric to
avoid double-rotation): `V_t = V * (1 + ε·sin(ω·step + φ·layer_idx))`.

Params delta: 0 (init only; the phase multiplier reuses ω as a single
scalar per layer). FLOPs delta: +0.2%. Init-time cost: O(d²) once.

```python
# src/nature_inspired_networks/cymatic_qkv.py
def cymatic_init_qkv(qkv_weight, n_modes=12, phi_spacing=True):
    d_out, d_in = qkv_weight.shape  # 3*d, d
    modes = chladni_modes(n_modes, d_in, phi_spacing=phi_spacing)  # (K, d)
    coefs = torch.randn(d_out, n_modes) * (1.0 / math.sqrt(n_modes))
    qkv_weight.data = coefs @ modes
```

### 5.2 LLM track

Slot: **QKV projection of every MHSA**. The cymatic basis is built
once at model construction; the runtime change is the per-step
φ-harmonic phase multiplier on V. FA2 compatibility: complete — only
the weight values change. Causal-mask preservation: trivial — the mask
is unmodified. Latency at batch=1: +1% (the phase multiplier is one
scalar broadcast).

Expected at 350M on WikiText-103: -0.3 to -0.5 nats; on TinyStories:
-0.2 to -0.4 nats; on GSM8K zero-shot: +0.5 pp.

## 6. Predicted Δ

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite | [+0.012, +0.022] | PPL win + zero-cost mass |
| perplexity (LLM) | [-0.5, -0.2] nats | orthogonal cymatic init speeds convergence |
| params | [0%] | init-only |
| FLOPs | [+0.5%] | one extra multiply per V projection |
| GPU latency (batch=1) | [0%, +2%] | trivial |
| KV cache @ 32k | [0%] | unchanged |
| Betti collapse rate | [+5%, +15%] | basis is orthonormal |

## 7. Experimental protocol

### 7.1 Primary experiment

- Dataset: WikiText-103.
- Architecture: 350M decoder, 24×1024×16.
- Train: 20k steps, bf16, cosine LR.
- Composite SHA-256.
- Wall-clock: ≈18 h on 4090.
- Archive: `ideas/66_cymatic_qkv_kernel/experiments/exp001_init/`.

### 7.2 Targeted experiment

Should SHINE on **early-training convergence**: report perplexity at
1k / 5k / 10k / 20k steps. Expected: cymatic-init beats random-init by
≥0.5 nats at 1k steps (then the gap narrows but persists).

### 7.3 Cross-paradigm context

H66 is the **mechanism-chunk** hypothesis — chunk-3 of the paradigm
comparison treats wavelet bases as an axis distinct from
attention/recurrence/spline. We test that prediction on the
Transformer paradigm specifically.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G7 row H66.
- Log: row T2.H66.
- Sub-dir: `ideas/66_cymatic_qkv_kernel/`.
- Composes with: H28, H35, H46, H67.
- Conflicts with: H42 (alternative weight-init).

## 9. Committee Q&A

**Q: Why isn't this just Fourier-features init?**

> Chladni modes on a square plate are the natural-system specialisation
> of Fourier features with **nodal-line constraints** that random
> Fourier features lack. φ-spacing across the K modes is the
> sacred-geometry contribution.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 names the -0.10 nat ceiling + the orthogonality-preservation
> falsifier.

**Q: What if cymatic init helps at GPT-2-small but hurts at 1B?**

> Saxe et al. predict orthogonal init benefits decay with scale; we
> scope claim to 350M and below.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H35 (cymatic init alone) was a single-prior negative on CIFAR; H66
> is a deliberate test of the same prior on a **different paradigm**
> (attention, not conv). A regime-specific positive would refute the
> "cymatic always hurts" reading of FINDINGS.md.

**Q: How do we know the implementation is correct?**

> `tests/test_cymatic_qkv.py` checks (a) cymatic_init produces an
> orthonormal basis with cosine sim < 0.05 between modes,
> (b) qkv_weight Frobenius norm matches He init within 5%, (c) FA2
> output equality to a vanilla init on identity inputs.

## 10. Verification artifacts checklist

- [ ] `implementation.py`
- [ ] `tests.py` ≥ 7 assertions
- [ ] `AUDIT.md`
- [ ] `IMPROVEMENTS.md`
- [ ] `VERIFY.md`
- [ ] `experiments/exp001_init/`
- [ ] `verification/`
- [ ] Log row
- [ ] FINDINGS reflected
- [ ] Dashboard

## 11. Status journal

- 2026-05-26 — Created from template by Doc-Agent-D.
