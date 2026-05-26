# H57 — Audio / Cymatic Cross-Modal Validation

> **One-line claim:** Reusing the cymatic kernel bank from H35
> (Chladni-eigenmode init) as the first-layer convolution of a
> log-mel-spectrogram audio classifier on AudioSet's "music" subset
> achieves ≥2 pp top-1 lift over the He-init baseline because audio
> spectrograms — like cymatic patterns — are direct visualizations of
> standing-wave resonance, providing a cross-modal validation that
> the cymatic priors capture a genuine physical regularity rather
> than a CIFAR-specific accident.
>
> **Source design space:** G6 Topological + bridging additions (H51–H60).
>
> **Implementation status (this repo):** `× deferred (out of scope for
> v0.1 single-GPU budget; design recorded for completeness).`

This document is the committee-grade design write-up for hypothesis
H57. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

Cymatic patterns and audio spectrograms are *the same physical
phenomenon viewed in two media*: a Chladni plate visualizes 2D
standing waves with sand, while a log-mel spectrogram visualizes 1D
sound waves' time-frequency decomposition with pixel brightness.
Both reveal the eigenfunctions of the Laplace operator restricted to
the relevant geometry (square plate vs. time-frequency rectangle).
A filter bank that captures cymatic eigenmodes should therefore also
capture audio spectrogram regularities, providing cross-modal
evidence that the prior is real.

This is the canonical *cross-modal validation* — if the same prior
helps on both audio and visual cymatic data, the prior captures a
physics-level regularity rather than a dataset accident. The Platonic
Representation Hypothesis (Huh 2024) further argues that
sufficiently-trained models across modalities converge to the same
representation; explicit cross-modal priors should accelerate this.

The deferred status acknowledges the single-GPU budget: AudioSet at
full scale requires 1–2 GPU-days per run, beyond the v0.1 envelope.
The design is recorded so that a future scale-up can drop it in.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because log-mel spectrograms are time-frequency standing-wave
visualizations and cymatic kernels are 2D-Laplacian eigenfunctions —
mechanism-wise, the cymatic filter bank pre-aligns the first-layer
filters with spectrogram resonance structure — per Hershey 2017
(VGGish AudioSet baseline) and Chladni 1787, we expect ≥2 pp top-1
lift on the music-classification subtask of AudioSet over He-init
baseline (3-seed median), validating cymatic priors cross-modally.

## 3. Falsifier (≥ 30 words)

If cymatic-init does NOT lift AudioSet music top-1 by ≥1 pp vs. the
He-init control (95% CI exclusion of 0), OR if the lift exists but
disappears after the first 5 epochs (suggesting init-bias only,
not a learned property), this hypothesis is DISCARDED.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
Hershey, Shawn and Chaudhuri, Sourish and Ellis, Daniel P. W. and
others 2017 ICASSP 'CNN Architectures for Large-Scale Audio
Classification' (arXiv:1609.09430) -- VGGish AudioSet baseline; we
use this as the architectural reference.

Gemmeke, Jort F. and Ellis, Daniel P. W. and Freedman, Dylan and
others 2017 ICASSP 'AudioSet: An Ontology and Human-Labeled
Dataset for Audio Events' -- the AudioSet dataset reference.

Chladni, Ernst F. F. 1787 (Leipzig) 'Entdeckungen über die Theorie
des Klanges' -- 1787 origin of cymatic patterns; foundational
reference.

Huh, Minyoung and others 2024 ICML 'The Platonic Representation
Hypothesis' (arXiv:2405.07987) -- cross-modal convergence
hypothesis we test.

Kong, Qiuqiang and others 2020 IEEE 'PANNs: Large-Scale Pretrained
Audio Neural Networks' (arXiv:1912.10211) -- modern AudioSet SOTA
reference; cited for context.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track (this is the primary track for H57)

The audio classifier processes log-mel spectrograms (128 mel bins ×
998 time frames at 10s clips). First-layer conv is 7×7 with 64
filters; we apply the cymatic-init from H35 to these filters using
the first 8 Chladni modes per spatial dimension.

```python
def cymatic_init_audio(conv: nn.Conv2d, n_modes=8):
    """Initialize a 7×7 conv kernel with Chladni eigenmodes."""
    out_ch, in_ch, kH, kW = conv.weight.shape
    modes = [(m, n) for m in range(1, n_modes+1) for n in range(1, n_modes+1)]
    assert len(modes) >= out_ch
    for o in range(out_ch):
        m, n = modes[o]
        # Sample the (m, n) mode on a 7×7 grid
        xs = torch.linspace(0, 1, kW)
        ys = torch.linspace(0, 1, kH)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        kern = torch.sin(m * math.pi * X) * torch.sin(n * math.pi * Y)
        # broadcast across input channels (replicate or randomize)
        conv.weight.data[o] = kern.unsqueeze(0).expand(in_ch, kH, kW) / in_ch
    # add small Gaussian noise to break ties
    conv.weight.data += torch.randn_like(conv.weight.data) * 1e-3
```

Forward shape unchanged. Init-only modification. Inference latency:
unchanged.

Lives in `src/nature_inspired_networks/init/cymatic_audio.py`, re-exported by
`ideas/57_audio_cymatic_cross/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

For audio-language models (e.g., speech-LLM), the analog is to use
H66 cymatic-QKV-init on a transformer that processes mel-spectrogram
patches. Not the primary scope of H57.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline (He init audio CNN) | rationale |
|---|---|---|
| composite | [-0.005, +0.020] | mild lift |
| top-1 (AudioSet music subset) | [+1.0, +5.0] pp | core target |
| epochs-to-target | [-30%, -10%] | faster |
| params | [0, 0] | init only |
| FLOPs | [0, 0] | unchanged |
| GPU latency (batch=1) | [0, 0] | unchanged |
| rotation-equivariance err | [0, 0] | N/A for audio |
| KV cache @ 32k (LLM) | [0, 0] | N/A |
| Betti collapse rate | [-0.05, 0] | minor |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment (deferred)

- **Dataset:** AudioSet "music" subset (~50k clips, 10 sub-classes)
- **Architecture:** VGGish-style CNN, 5 conv blocks, ~1.5M params
- **Pre-processing:** 10s audio → 128 mel-bins × 998 time-frames spectrogram
- **Epochs:** 25
- **Seeds:** 0, 1, 2
- **Comparator:** He-init baseline; cymatic-init at conv1
- **Run-script:** `python scripts/run_idea.py --idea 57 --dataset audioset_music`
- **Wall-clock:** ≈ 6 h × 3 seeds × 2 inits = ~36 h (this is the
  deferral reason)
- **Archive path:** `ideas/57_audio_cymatic_cross/experiments/exp001_audioset/`

### 7.2 Idea-targeted experiment (deferred)

The cleanest test: a *single resonance-classification* synthetic
audio dataset (pure-tones + harmonics) where cymatic-init should
clearly win. This is the audio analog of H56's Chladni-pattern
dataset.

### 7.3 Cross-paradigm context

H57 is itself a cross-paradigm bridge between audio and vision.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G6 row H57.
- Master experiment list: `EXPERIMENT_LOG.md` Tier 6 row T6.6 (deferred).
- Implementation sub-directory: `ideas/57_audio_cymatic_cross/`
- Related hypotheses that compose:
  - **H35** Cymatic Init — the prior being cross-modally validated.
  - **H56** Cymatic Pattern Dataset — visual sibling.
  - **H46** Cymatic Loss — additional cymatic prior.
  - **H66** Cymatic QKV Kernel — LLM-track sibling for audio-LM.
- Related hypotheses that conflict:
  - None directly.

## 9. Committee Q&A

**Q: Why isn't this just standard audio CNN with different init?**

> The init is *physics-derived* (2D Laplacian eigenmodes) and the
> hypothesis is *cross-modal*: validates that cymatic priors capture
> resonance regularity present in audio. If both visual cymatic
> patterns AND audio spectrograms benefit, the prior is real.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies ≥1 pp lift with 95% CI + persistence through
> training (not just init-bias).

**Q: What if the prior helps on CIFAR but hurts on AudioSet?**

> That would falsify the cross-modal validation — interesting
> negative result, would refine cymatic-prior claims to vision-only.

**Q: Why is this deferred?**

> AudioSet at full scale is ~36 GPU-hours per condition; out of
> single-GPU v0.1 budget. Design recorded; runs when budget allows
> or on shared compute.

**Q: How do we know the implementation is correct?**

> `tests.py` verifies (a) cymatic_init_audio produces the predicted
> filters when queried at known (m, n), (b) shape preservation, (c)
> spectrogram loader correctness on a known-pitch sine wave (peak at
> expected mel bin).

## 10. Verification artifacts checklist

- [ ] `ideas/57_audio_cymatic_cross/implementation.py` exists
- [ ] `ideas/57_audio_cymatic_cross/tests.py` ≥ 6 assertions
- [ ] `ideas/57_audio_cymatic_cross/AUDIT.md` ≥ 3 weaknesses
- [ ] `ideas/57_audio_cymatic_cross/IMPROVEMENTS.md`
- [ ] `ideas/57_audio_cymatic_cross/VERIFY.md` signed
- [ ] One experiment archive (deferred)
- [ ] `verification/{tests,smoke,gates,reproduction}.txt`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-C; marked deferred
  per single-GPU budget. Design retained for future scale-up.
