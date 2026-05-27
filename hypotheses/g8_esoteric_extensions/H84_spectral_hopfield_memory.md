# H84 — Spectral Hopfield Associative Memory

> **One-line claim:** A modern Hopfield layer that matches queries in an
> exact-invertible FFT eigenmode basis retrieves on vibrational structure.
>
> **Source design space:** G8 Esoteric Extensions.
>
> **Implementation status (this repo):** `✓ done` (primitive + tests).

---

## 1. Motivation (≥ 100 words)

A crystal lattice "remembers" through its normal modes of vibration: a
perturbation is decomposed into eigenmodes (phonons), and the structure
relaxes back along those modes. The Fourier basis is precisely the
eigenbasis of the discrete Laplacian / shift operator, so matching two
signals by their frequency spectra is matching them on their vibrational-
mode content rather than their raw sample values. Modern continuous
Hopfield networks (Ramsauer et al. 2020) already store patterns and
retrieve the nearest one via a softmax readout — mathematically identical
to attention — with exponential storage capacity. Performing the
similarity comparison in the rfft spectral basis biases retrieval toward
patterns that share spectral (periodic / textural) structure, which is
natural for signals dominated by mode content. The esoteric "crystal
vibration memory" motif is the mystical motivation; the implementation is
a modern Hopfield layer reparameterised by an exact-invertible FFT.

## 2. Formal hypothesis (≥ 50 words)

Because the retrieval similarity `β·⟨rfft(q), rfft(xᵢ)⟩` is computed in
the FFT eigenmode basis — an exact linear reparameterisation of the signal
inner product — the softmax readout `softmax(β X q)X` concentrates on the
stored pattern whose vibrational spectrum best matches the query;
mechanism-wise, per Ramsauer 2020, increasing β sharpens this toward exact
single-pattern recall, and the `irfft∘rfft` round-trip is lossless, so
spectral matching never discards information.

## 3. Falsifier (≥ 30 words)

If, with β ≥ 20 and a query within 1% noise of a stored pattern, the
retrieved vector's nearest stored pattern is NOT the query's source
pattern (associative recall fails), or if `irfft(rfft(x))` deviates from
`x` by > 1e-4, the mechanism is broken and the hypothesis is DISCARDED.

## 4. Citations (≥ 40 words)

Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M. 2020
'Hopfield Networks is All You Need' (arXiv:2008.02217) — establishes the
modern continuous Hopfield update `softmax(β X_stored q) X_stored` with
exponential storage capacity and one-step retrieval; this module
implements that update and reparameterises the matching space by an
exact-invertible rfft, which leaves the energy landscape isometric.

## 5. Mechanism

### 5.1 CNN / vision track
`SpectralHopfieldMemory(dim, beta)`: `store(patterns: (M,dim))` caches the
bank as a buffer; `retrieve(query: (B,dim))` maps query and bank through
rfft (stacking real+imag → spectral feature), scores with β, softmaxes,
and mixes the *signal-domain* patterns. Used as a denoising/prototype head
over pooled features. Cost O(M·dim·log dim) for the FFTs plus O(B·M)
matmul. Lives at `src/nature_inspired_networks/spectral_hopfield.py`.

### 5.2 LLM track (decoder-only)
A Hopfield retrieval layer over a learned key-pattern bank, slotting where
an external-memory / retrieval head would; the spectral basis biases recall
toward periodic token-feature structure. Matching is causal-agnostic (it
attends to a fixed pattern bank, not the sequence), so no causal-mask or
KV-cache interaction.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. raw-basis Hopfield | rationale |
|---|---|---|
| recall accuracy (assoc. task) | [−0%, +5%] | spectral bias helps periodic patterns |
| composite (CNN head) | [−0.005, +0.004] | task-dependent |
| params | [0, 0] | pattern bank is a buffer |
| FLOPs | [+O(dim·log dim)] | two FFTs per retrieval |

## 7. Experimental protocol

### 7.1 Primary
Associative-recall synthetic benchmark: store M random patterns, query
with noisy copies, measure recall@1 vs. β; compare spectral vs. raw-basis
matching. Archive:
`ideas/84_spectral_hopfield/experiments/exp001_recall/`.

### 7.2 Targeted (where it should shine)
Patterns that are periodic/textural (e.g. sinusoidal mixtures): the
spectral basis should beat raw-basis matching because the discriminative
information lives in a few modes.

### 7.3 Cross-paradigm (LLM)
Retrieval-memory head on a 124M decoder over a frozen key-pattern bank;
measure perplexity vs. a no-memory baseline.

## 8. Cross-references

- Parent: G8 esoteric extensions.
- Related: `src/nature_inspired_networks/cymatic_*` (Chladni/FFT modes),
  `pentagonal_attention.py` (attention-as-retrieval analogue).
- Composes with: H82 (sparse retrieval), H78 (toroidal pattern codes).

## 9. Committee Q&A

**Q: Why isn't this just attention / vanilla modern Hopfield?**
> It is modern Hopfield at its core (acknowledged, cited); the
> contribution is the exact-invertible spectral reparameterisation of the
> matching space, which changes *which* patterns win without distorting
> the energy landscape (rfft is linear and losslessly invertible).

**Q: How is this falsifiable?**
> §3: failed associative recall at β≥20 with a near-exact query, or a
> round-trip error > 1e-4, discards it — both are directly tested.

**Q: What if spectral matching hurts on non-periodic patterns?**
> Expected to be at best inert there (the basis is a rotation); the §7.2
> targeted task is periodic patterns where it should win.

**Q: Priors don't compound — why bother?**
> Single-prior unit (the spectral basis), measured against raw-basis
> Hopfield, not inside a hybrid.

**Q: How do we know it is correct?**
> `tests/test_spectral_hopfield.py` (5 tests): associative recall returns
> the stored source pattern, output shape matches query rank, higher β
> places more softmax mass on the match (sharper recall), the FFT
> round-trip is lossless, and retrieval before `store()` raises.

## 10. Verification checklist

- [x] Primitive `spectral_hopfield.py` exists, tests green (5/5).
- [x] Associative recall of a stored pattern (tested).
- [x] Output shape matches query (tested).
- [x] High β sharper than low β (tested).
- [x] `irfft(rfft(x)) ≈ x` lossless (tested).
- [ ] Experiment archive (deferred — memory module, no sweep row).

## 11. Status journal

- 2026-05-27 — Created; primitive + 5 unit tests green.
