# H46 — Cymatic Loss (Fourier-Domain Chladni-Harmonic Term)

> **One-line claim:** Adding a small (λ ≈ 0.05) Fourier-domain MSE term
> that pulls the power-spectrum of intermediate activations toward the
> nearest Chladni-eigenmode of the same spatial dimensions yields
> ≥0.5 pp top-1 lift on CIFAR-10 because the natural Chladni harmonics
> are the eigenfunctions of the 2D Laplacian — the same operator that
> dominates the smoothness prior in convolutional networks — and
> projecting activations onto them sharpens edge responses without
> introducing aliasing.
>
> **Source design space:** G5 Optimization / Init / Regularization / NAS
> (H41–H50).
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for hypothesis
H46. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The classic Chladni-plate experiment (Chladni 1787) demonstrates that
a vibrating square plate forms standing-wave nodal lines exactly
matching the eigenfunctions of the 2D Laplace operator
`Δu + λu = 0` on a Dirichlet-bounded square. These eigenfunctions are
`u_{m,n}(x,y) = sin(mπx) sin(nπy)`, with eigenvalues `λ_{m,n} = (m² +
n²)π²`. They form a complete orthonormal basis for functions on the
square; any image can be expanded in this basis. Crucially, the
basis is sparse for natural images: most of the energy lives in
low-(m, n) modes, with high-frequency modes dominated by noise.

Convolutional networks implicitly learn the same decomposition — early
filters resemble low-(m, n) Chladni modes, deeper filters more
oriented — but with no explicit regularization toward this basis. A
**cymatic loss** that nudges activation power-spectra toward
Chladni-mode targets imposes the smoothness prior that nature gives
for free. This is the spectral analog of Tikhonov regularization, but
with Chladni modes as the chosen smooth basis instead of the
generic Laplacian eigen-decomposition. The expected benefit is
sharper edge responses (the modes naturally have sharp nodal lines)
without the aliasing that ad-hoc spectral filtering introduces.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because adding a Fourier-domain regularization term that targets
Chladni eigenmodes constrains intermediate activations to live near
the dominant 2D-Laplacian eigenfunctions — mechanism-wise, the loss
`L_cym = ||FFT(activations)|^2 - target_chladni_spectrum||²`
penalizes deviations from the smooth nodal-line basis — per Chladni
1787 and the FFT-domain regularization arguments of Rahaman 2019
(spectral bias of NNs, arXiv:1806.08734), we expect a ≥0.5 pp lift in
CIFAR-10 top-1 at λ_cym = 0.05 with no measurable hit to latency
(loss is computed at training time only).

## 3. Falsifier (≥ 30 words)

If at 3-seed median the cymatic-loss arm does NOT lift top-1 by
≥0.3 pp vs. the no-cymatic-loss control (95% CI must exclude 0), OR
if the optimal λ_cym is found to be ≤ 0.005 (essentially zero —
indicating the term is helping only as numerical noise), this
hypothesis is DISCARDED.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
Chladni, Ernst F. F. 1787 (Leipzig) 'Entdeckungen über die Theorie
des Klanges' -- the original 1787 publication establishing the
nodal-line patterns we use as eigenfunction targets.

Rahaman, Nasim and Baratin, Aristide and Arpit, Devansh and others
2019 ICML 'On the Spectral Bias of Neural Networks'
(arXiv:1806.08734) -- shows neural networks naturally learn
low-frequency components first; our cymatic loss accelerates this
implicit bias.

Tancik, Matthew and Srinivasan, Pratul and others 2020 NeurIPS
'Fourier Features Let Networks Learn High Frequency Functions in
Low Dimensional Domains' (arXiv:2006.10739) -- complementary
result; we cite it for the inverse case (we want to regularize
toward low-freq Chladni modes, not lift high-freq).

Mallat, Stéphane 2012 PAMI 'Group Invariant Scattering' -- relates
spectral decompositions to invariance properties; cited as the
mathematical foundation for the FFT-domain loss approach.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

The cymatic loss operates on the spatial feature map at one
designated layer (typically the output of the second residual block,
which has spatial dimensions 16×16 on CIFAR-10). At each step:

1. Compute the per-channel 2D FFT of the feature map.
2. Compute `|FFT|²` — the power spectrum.
3. Compute the target spectrum as a weighted sum of Chladni modes
   `(m, n) ∈ {(1,1), (1,2), (2,1), (2,2), (1,3), (3,1)}` with weights
   proportional to `1 / λ_{m,n}` (inverse-eigenvalue, emphasizing
   low-frequency).
4. MSE between observed spectrum and target spectrum, summed across
   channels, averaged across batch.
5. Add `λ_cym · L_cym` to the classification loss.

Pseudocode:

```python
import torch, math

def chladni_target(H, W, modes=((1,1),(1,2),(2,1),(2,2),(1,3),(3,1))):
    """Return target |FFT|² spectrum of size (H, W)."""
    target = torch.zeros(H, W)
    for (m, n) in modes:
        # Power at frequency (m, n) ∝ 1/λ
        weight = 1.0 / (m**2 + n**2)
        # FFT shifts so DC is at (0,0)
        target[m % H, n % W] += weight
    return target / target.sum()

def cymatic_loss(feat, target=None):
    """Compute MSE between feat's power-spectrum and Chladni target."""
    B, C, H, W = feat.shape
    if target is None:
        target = chladni_target(H, W).to(feat.device)
    spec = torch.fft.fft2(feat).abs() ** 2  # (B, C, H, W)
    spec_avg = spec.mean(dim=(0, 1))  # average over batch and channels
    spec_norm = spec_avg / spec_avg.sum()
    return torch.nn.functional.mse_loss(spec_norm, target)
```

Cost at training time: one 2D FFT per layer per batch (negligible — the
forward pass FFT is microseconds). Cost at inference: **zero** — the
loss term is training-only.

Lives in `src/nature_inspired_networks/losses/cymatic.py`, re-exported by
`ideas/46_cymatic_loss/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

For decoder LLMs, "spatial" Chladni modes don't directly apply (the
attention dimension is 1D). The LLM-track analog is a **1D
DCT/DST-domain loss** targeting the cosine basis modes
`cos(kπt/T)`, the 1D analog of Chladni modes. Applied to the residual
stream activations at one layer, regularizing toward smooth
positional structure.

FlashAttention-2 compatibility: the loss is computed on activations
post-FA2, so kernel compatibility is preserved. Causal mask
preservation untouched.

Expected impact at 124M: minor perplexity lift (-0.2 to -0.5 ppl on
TinyStories); no latency change.

```python
def dct_target(T, modes=(1,2,3,4)):
    target = torch.zeros(T)
    for k in modes:
        target[k] += 1.0 / k**2
    return target / target.sum()

def cymatic_loss_1d(feat):
    # feat: (B, T, D) residual stream
    B, T, D = feat.shape
    dct = torch.fft.fft(feat, dim=1).abs() ** 2
    spec = dct.mean(dim=(0, 2))
    return F.mse_loss(spec / spec.sum(), dct_target(T).to(feat.device))
```

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline (λ_cym=0) | rationale |
|---|---|---|
| composite | [+0.002, +0.012] | targeted small lift |
| top-1 (CNN) | [+0.3, +1.5] pp | targeted lift |
| params | [0, 0] | loss term only |
| FLOPs | [0, 0] | inference-time identical |
| GPU latency (batch=1) | [0, 0] | inference-time identical |
| rotation-equivariance err | [-0.01, +0.01] | minor effect |
| KV cache @ 32k (LLM) | [0, 0] | unaffected |
| Betti collapse rate | [-0.05, +0.05] | possibly accelerated |
| perplexity (LLM 124M) | [-0.5, -0.1] | targeted lift |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dataset:** CIFAR-10
- **Architecture:** `NaturePriorNet` (`channel_mode=fib`, priors off)
- **Loss:** classification + `λ_cym × cymatic_loss(stage2_features)`
- **λ_cym sweep:** {0.005, 0.05, 0.5} to find the optimum
- **Epochs:** 12, batch=128, AdamW, bf16 AMP
- **Seeds:** 0, 1, 2 (for the chosen λ)
- **Control:** identical training without the cymatic loss term
- **Run-script:** `python scripts/run_idea.py --idea 46 --lambda 0.05`
- **Wall-clock:** ≈ 12 min × 3 seeds × 4 conditions ≈ 144 min
- **Archive path:** `ideas/46_cymatic_loss/experiments/exp001_cifar10_cymatic/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

Cymatic loss should help most on **synthetic Chladni-pattern data**
(H56), where the data itself is built from these exact modes:

- **Dataset:** synthetic Chladni patterns (256×256, 16 frequency classes)
- **Architecture:** small CNN with cymatic loss at every layer
- **Predicted:** ≥5 pp lift over no-cymatic-loss control
- **Diagnostic:** if it does not help on data literally made of
  Chladni modes, the loss is broken; if it helps there but not on
  CIFAR, the prior is dataset-mismatched.

### 7.3 Cross-paradigm context (LLM track)

- **Model:** 124M GPT-2-small
- **Dataset:** TinyStories
- **Loss:** standard CE + `λ × cymatic_loss_1d(residual_stream)` at
  layer 6
- **Steps:** 10k
- **Run:** `python scripts/run_llm.py --idea 46 --lambda 0.05`

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G5 row H46.
- Master experiment list: `EXPERIMENT_LOG.md` (planned).
- Implementation sub-directory: `ideas/46_cymatic_loss/`
- Related hypotheses that compose:
  - **H35** Cymatic Wavelet Init — both use Chladni modes (init vs.
    loss); composes naturally.
  - **H56** Cymatic Pattern Dataset — provides the targeted-experiment
    benchmark.
  - **H66** Cymatic QKV Kernel Bank (LLM-track) — composes; QKV
    init + cymatic-loss reinforce each other.
  - **H49** PRH alignment loss — composes; both are auxiliary
    spectral-domain losses.
- Related hypotheses that conflict:
  - None directly; auxiliary losses are additive.

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just spectral regularization?**

> Generic spectral regularization (penalize high-frequency components)
> is unprincipled; the target spectrum is arbitrary. Cymatic loss
> picks a *specific* basis (2D Laplacian eigenmodes on a square) with
> 234 years of physical pedigree and a known sparsity structure for
> natural images. The choice of basis is the claim.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies ≥0.3 pp lift with 95% CI exclusion. Optimal λ must
> be ≥0.005 to count as non-trivial.

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> The Chladni basis assumes near-square spatial dims; ImageNet's
> larger receptive fields may shift the dominant modes. We'd report
> as scope-limited and not extrapolate.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H46 is a *loss-side* prior; the compound failure was architectural.
> Tested in isolation; later composed with H35 cymatic-init to test
> whether two cymatic priors compound differently than architectural
> priors.

**Q: How do we know the implementation is correct?**

> `tests.py` verifies (a) `chladni_target` sums to 1, (b) the FFT2
> applied to a known sin(πx) sin(πy) image produces a single non-zero
> coefficient at (1,1) within float tolerance, (c) gradient flow
> through the loss is non-zero.

## 10. Verification artifacts checklist

- [ ] `ideas/46_cymatic_loss/implementation.py` exists, tests green
- [ ] `ideas/46_cymatic_loss/tests.py` ≥ 6 assertions
- [ ] `ideas/46_cymatic_loss/AUDIT.md` ≥ 3 weaknesses
- [ ] `ideas/46_cymatic_loss/IMPROVEMENTS.md` records fixes
- [ ] `ideas/46_cymatic_loss/VERIFY.md` signed
- [ ] At least one experiment archive
- [ ] `verification/{tests,smoke,gates,reproduction}.txt`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-C.
