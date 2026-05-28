# H77 — Radial 12-fold Symmetry Attention

> **One-line claim:** A fixed 12-fold relative-position angular head bias
> adds a structural inductive bias to MHA at zero extra parameters.
>
> **Source design space:** G8 Esoteric Extensions (neutral re-castings of
> bonus ideas).
>
> **Implementation status (this repo):** `✓ done` (primitive + tests).

---

## 1. Motivation (≥ 100 words)

Twelve-fold radial symmetry is the densest integer rosette that tiles the
angular plane with clean divisors (12 = 2·2·3): it is the clock dial, the
dodecagonal quasicrystal, the 12 face-normals of the rhombic dodecahedron,
and the 12 nearest neighbours of a close-packed sphere (the kissing number
in 3-D). In each case the 12-fold partition is the maximum-symmetry way to
distribute directions about a centre so that no direction is privileged.
For multi-head attention the heads are exactly such a bank of directions
over the sequence; the standard practice allocates them uniformly with no
angular structure. Imposing a 12-fold radial arrangement gives each head a
distinct, evenly-spaced angular phase, encouraging the bank to cover
relative-position frequencies isotropically rather than redundantly. The
esoteric "Lotus of Life" 12-petal rosette is the mystical motivation; the
implementation is a neutral, parameter-free relative-positional bias.

## 2. Formal hypothesis (≥ 50 words)

Because each head ``h`` receives the fixed relative-position bias
``(1/φ)·cos(2π(h mod 12)/12 + 2π(j−i)/N)`` added to its pre-softmax
logits, the attention bank is forced to span 12 evenly-spaced angular
phases of the relative-position signal; mechanism-wise this is a
parameter-free Fourier-feature prior on ``j−i``. Per Shaw 2018, only
relative (not absolute) biases alter softmax outputs, so we expect a
small but non-zero change in attention maps and a modest accuracy gain on
sequence tasks where periodic relative structure matters, at exactly zero
added parameters.

## 3. Falsifier (≥ 30 words)

If, on a patch-token ViT-Tiny CIFAR-10 run, the 12-fold radial bias yields
composite Δ ≤ −0.005 versus the identical model with ``radial=False`` at
3-seed median, the structural bias is net-harmful and the hypothesis is
DISCARDED. A no-op (Δ within ±0.002) downgrades it to "inert prior".

## 4. Citations (≥ 80 words)

Vaswani, A., Shazeer, N., Parmar, N. 2017 NeurIPS 'Attention Is All You
Need' (arXiv:1706.03762) — the canonical multi-head attention this module
extends; we add a structural per-head relative bias without changing the
QKV/softmax core. Shaw, P., Uszkoreit, J., Vaswani, A. 2018 NAACL
'Self-Attention with Relative Position Representations' (arXiv:1803.02155)
— proves that relative-position biases (unlike a constant per-row shift,
which softmax cancels) change attention weights, which is exactly why the
12-fold bias is made a function of ``j−i`` rather than a per-head constant.

## 5. Mechanism

### 5.1 CNN / vision track
Input patch-token sequence ``(B, N, D)`` → ``(B, N, D)``. The bias tensor
``(12k, N, N)`` is recomputed per forward from the buffer ``head_angles``
(period 12) and the relative index ``j−i``; it adds zero parameters and
``O(h·N²)`` FLOPs (same order as the logits it adds to). Init is unchanged
(standard QKV). Lives at
`src/nature_inspired_networks/radial12_attention.py`.

### 5.2 LLM track (decoder-only)
Slots into the MHSA logit stage exactly like a relative-position bias
(ALiBi-style), before the causal mask and softmax. Causal masking is
preserved (the additive bias is applied before the mask). It is
FlashAttention-2-incompatible in fused form (needs an explicit bias
tensor) but cheap at small context; expected effect is a small perplexity
change with no KV-cache cost (bias is recomputed, not cached).

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite | [−0.004, +0.006] | structural prior, may be inert on isotropic data |
| top-1 (CNN) | [−0.3, +0.5] pp | small periodic-relative-pos gain |
| params | [0, 0] | parameter-free bias |
| FLOPs | [+0.5%, +2%] | extra (h·N²) additive term |
| GPU latency (batch=1) | [+1%, +4%] | bias recompute per forward |

## 7. Experimental protocol

### 7.1 Primary
ViT-Tiny on CIFAR-10, n_heads forced to 12, 12-epoch smoke, bf16 AMP,
batch 256, seeds 0/1/2, composite per `eval.py:COMPOSITE_FORMULA`.
Archive: `ideas/77_radial12_attention/experiments/exp001_smoke/`.

### 7.2 Targeted (where it should shine)
Synthetic periodic-sequence classification where the label depends on a
relative period dividing 12 — the 12-fold bias is data-aligned there.

### 7.3 Cross-paradigm (LLM)
124M decoder, swap relative bias in, measure WikiText-103 perplexity
delta vs. RoPE-only baseline.

## 8. Cross-references

- Parent design space: G8 esoteric extensions.
- Sibling 5-fold module: `src/nature_inspired_networks/pentagonal_attention.py` (H37).
- Composes with: H82 (sparse mask), H78 (latent head).

## 9. Committee Q&A

**Q: Why isn't this just pentagonal_attention with 12 instead of 5?**
> Same family, different fold; 12 is the densest clean angular divisor and
> matches kissing-number / dodecagonal structure. It is deliberately a
> faithful generalisation, validated against the 5-fold sibling's test.

**Q: How is this falsifiable rather than aesthetic?**
> See §3: a 3-seed composite Δ ≤ −0.005 vs. the ``radial=False`` fallback
> discards it; the fallback is the exact ablation.

**Q: What if it helps on CIFAR-10 but hurts elsewhere?**
> The claim is scoped to periodic-relative-position structure (§7.2); the
> CIFAR row is a screen, not the headline.

**Q: Hasn't the prior sweep shown priors don't compound?**
> This is a single-prior unit of analysis (parameter-free), measured
> against its own off-switch, not inside a hybrid.

**Q: How do we know the implementation is correct?**
> `tests/test_radial12_attention.py` (5 tests): shape, multiple-of-12
> guard, buffer-not-Parameter, cyclic phase permutation, and
> ``radial=False`` exactly equals hand-rolled plain MHA.

## 10. Verification checklist

- [x] Primitive `radial12_attention.py` exists, tests green (5/5).
- [x] Bias is a registered buffer, not a Parameter (tested).
- [x] ``radial=False`` matches plain MHA bit-for-bit (tested).
- [x] n_heads multiple-of-12 enforced (tested).
- [ ] Experiment archive (deferred — attention module, no sweep row).

## 11. Status journal

- 2026-05-27 — Created; primitive + 5 unit tests green.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G8 (elite-research-scientist critic). Critiquing
the IDEA, not the implementation (audit at `audits/G8_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)

LOW. Patch-token sequences on CIFAR-10 have a 2-D spatial relative
structure (Δx, Δy), not a 1-D angular periodicity. Indexing the bias
by `j−i` (a *linearised* token index) imposes a *spurious* angular
periodicity along a raster-scan order — `j−i = 6` connects a patch
to the patch six raster steps later, which is geometrically
arbitrary. The 12-fold "angular phase" assignment to heads is
imposed on a quantity (`j−i`) that has no angular meaning to begin
with.

### Mechanism scrutiny — does the NEUTRAL recast match the cited real technique?

Partially. Per-head fixed relative-position bias is standard
(ALiBi — Press, Smith, Lewis 2022 ICLR 'Train Short, Test Long:
Attention with Linear Biases Enables Input Length Extrapolation'
(arXiv:2108.12409) — uses a per-head linear-in-distance bias). The
Shaw 2018 NAACL 'Self-Attention with Relative Position
Representations' (arXiv:1803.02155) citation is correct that
relative biases alter softmax outputs. But the *12-fold sinusoidal*
phase pattern is not the canonical ALiBi/Shaw choice; canonical
choices are monotone decay (ALiBi) or learned per-offset embeddings
(Shaw). The cosine bias is closer to RoPE (Su 2024 Neurocomputing
'RoFormer: Enhanced Transformer with Rotary Position Embedding'
(arXiv:2104.09864)) without the Q/K rotation that makes RoPE work.

### Does the esoteric origin contaminate the implementation or framing?

YES. The number 12 has no defensible link to CIFAR-10 patch tokens.
The doc lists "clock dial, dodecagonal quasicrystal, 12 face-normals
of the rhombic dodecahedron, kissing number 12 in 3-D" as
motivation — none of which apply to a 2-D image patch raster. The
12-fold prior is "Lotus of Life" numerology projected onto a head
count.

### Confounds (≥2)

1. **`n_heads` constraint.** Forcing `n_heads` to be a multiple of
   12 is itself an architectural change vs. the canonical ViT-Tiny
   `n_heads=3`. Any observed gain may be the head-count change, not
   the bias.
2. **Raster vs. 2-D.** A genuine angular bias for image patches
   should be a function of `arctan2(Δy, Δx)` on the 2-D patch grid,
   not `j−i` on the linearised sequence. The current
   implementation's "12-fold radial" framing is geometrically
   incoherent.
3. **`1/φ` scaling.** The `(1/φ)·cos(...)` bias magnitude is a free
   parameter masquerading as a constant — a learned scale would
   absorb it. φ appears for aesthetic reasons, not functional ones.

### Numerology / specificity check

The "12 is the densest integer rosette" claim privileges 12 over 6,
8, 10, or 16 with no falsifiable test. CIFAR-10's 64 tokens have
no internal 12-fold structure (`64 mod 12 = 4`). A fair comparison
sweeping `n_fold ∈ {4, 6, 8, 12, 16}` would expose whether 12 is
special or whether *any* fold ≥ some threshold suffices. The 12 is
chosen for "Lotus of Life", not for any property of CIFAR-10
patches.

### Literature precedent — was the neutral recast already known?

YES. The neutral recast is a sinusoidal relative-position bias —
essentially a non-rotary variant of RoPE (Su 2024 arXiv:2104.09864)
or a fixed (non-learned) version of Shaw 2018 (arXiv:1803.02155).
ALiBi (Press 2022 arXiv:2108.12409) achieves the same parameter-free
goal with a *monotone* bias and is the established baseline. Adding
"12-fold" buys nothing the literature has not already explored.

### Expected effect size (90% CI a priori)

ViT-Tiny CIFAR-10 12-ep top-1 vs. `radial=False`: [−1.0 pp, +0.3 pp].
The slightly negative skew reflects that an imposed periodicity on a
linearised raster is more likely to confuse than help on
non-periodic image patches.

### Minimum-distinguishing experiment

Sweep `n_fold ∈ {3, 4, 6, 8, 12, 16}` on ViT-Tiny CIFAR-10. If 12 is
not a clear peak (i.e. accuracy is monotone or flat in `n_fold`),
the "12 is special" claim is falsified and the prior reduces to "a
sinusoidal relative bias", already known. Pair with the canonical
ALiBi baseline to check whether *any* fixed bias beats *no* fixed
bias.

### Verdict

NUMEROLOGY — The 12-fold specificity has no defensible link to
CIFAR-10 patch geometry; the neutral primitive (a fixed sinusoidal
relative bias) is real but the "12" comes from the Lotus motif, not
the data.
