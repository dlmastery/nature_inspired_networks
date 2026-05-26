# H56 — Cymatic Pattern Synthetic Dataset

> **One-line claim:** Generating a synthetic 256×256 image dataset of
> Chladni-eigenmode plate-vibration patterns at 64 driving frequencies,
> each labeled with the (m, n) mode index, produces a controlled
> resonance-classification benchmark on which the cymatic-init prior
> (H35) and cymatic-loss term (H46) should both score ≥+10 pp top-1
> over a plain-CNN baseline, providing the *natural-fit* benchmark
> that CIFAR-10 fails to be for cymatic-prior validation.
>
> **Source design space:** G6 Topological + bridging additions (H51–H60).
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for hypothesis
H56. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The previous CIFAR-10 sweep showed cymatic-init (T1.7,
`sg_only_cymatic_init`) was *negative* (top-1 77.44%, -2.67 pp vs.
reference). This was unexpected — Chladni modes are the eigenfunctions
of the 2D Laplacian and should be a useful filter-bank for natural
images. The leading hypothesis for the negative result is **dataset
mismatch**: CIFAR-10 natural images have power-law spectra dominated
by very low frequencies; the Chladni modes' nodal-line structure
matches *resonance phenomena*, not natural-image statistics.

H56 builds the natural-fit dataset where cymatic priors SHOULD work
by construction. Generate Chladni-plate vibration patterns by solving
the 2D wave equation `∂²u/∂t² = c²∇²u` on a square plate with
Dirichlet boundary conditions, driving at frequencies that excite
specific (m, n) modes. The dataset contains 64 classes (one per
distinct mode within a frequency band), 1000 images per class,
256×256 resolution. Augmentation: small rotation, color (the modes
are inherently visualized with colored fluid), small displacement.

This dataset is to cymatic priors what Spherical MNIST is to icosa-
equivariant CNNs: the controlled benchmark where the prior should
shine. If cymatic-init and cymatic-loss do NOT lift performance here,
the priors are demonstrably broken; if they do lift here but not on
CIFAR, the previous negative result is scope-confirmed as a
data-mismatch result, not a primitives-broken result.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because the synthetic Chladni-pattern dataset is *constructed* from
the same 2D-Laplacian eigenmodes that cymatic-init produces, the
cymatic priors should be statistically optimal for this task —
mechanism-wise, cymatic-init pre-aligns the filter bank with the
target eigenfunctions before training begins; cymatic-loss
regularizes activations toward the target spectrum — per Chladni
1787 and Rahaman 2019 (spectral bias), we expect cymatic-init + loss
to lift top-1 by ≥10 pp vs. plain-CNN baseline (3-seed median).

## 3. Falsifier (≥ 30 words)

If cymatic-init + cymatic-loss combined do NOT lift top-1 by ≥5 pp
on Chladni-dataset classification vs. plain-CNN baseline (95% CI
exclusion of 0), this hypothesis (and *both* H35 and H46) is
DISCARDED at the natural-fit benchmark.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
Chladni, Ernst F. F. 1787 (Leipzig) 'Entdeckungen über die Theorie
des Klanges' -- the 1787 origin of cymatic patterns; foundational
reference for the synthetic dataset.

Rahaman, Nasim and others 2019 ICML 'On the Spectral Bias of
Neural Networks' (arXiv:1806.08734) -- spectral bias of CNNs;
relevant for the eigenmode-classification difficulty argument.

LeCun, Yann and Cortes, Corinna and Burges, Christopher J. C.
1998 MNIST handwritten digit database -- methodological precedent
for synthetic-classification benchmarks (we follow the same
construction philosophy).

Karras, Tero and others 2018 ICLR 'Progressive Growing of GANs'
(arXiv:1710.10196) -- methodological reference for synthetic
high-resolution image generation pipelines.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track — dataset generation

The dataset is *generated*, not learned. Generation procedure:

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def generate_chladni_pattern(m, n, size=256, noise_level=0.05):
    """Solve the steady-state 2D wave equation Δu + (m² + n²)π²u = 0."""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    u = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
    # Visualize nodal lines: sand accumulates where u = 0
    img = np.abs(u)
    # Threshold to get sand pattern
    img = (img < 0.05).astype(np.float32)
    # Add Gaussian blur (physical sand has thickness)
    img = gaussian_filter(img, sigma=1.5)
    # Add noise
    img += np.random.randn(*img.shape) * noise_level
    return img

# Generate dataset
modes = [(m, n) for m in range(1, 9) for n in range(1, 9)]  # 64 modes
for (m, n) in modes:
    for i in range(1000):
        img = generate_chladni_pattern(m, n)
        save(f"chladni/{m}_{n}/img_{i:04d}.png", img)
```

The dataset is also augmented with **synthetic class-mixed images**
(superposition of two modes) for harder classification.

After dataset generation, training is standard CIFAR-style:

```python
class ChladniDataset(Dataset):
    def __init__(self, root, transform=None):
        # load 64-class structured dataset
        ...
```

Lives in `src/nature_inspired_networks/data/chladni.py`, re-exported by
`ideas/56_cymatic_dataset/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

The dataset can also be tokenized for a vision-language model: 1D
DCT coefficients of each Chladni pattern, fed to a small transformer
for mode classification. Test: do the cymatic priors (H66 cymatic-
QKV) help on this 1D DCT classification?

This is a stretch use case; the main H56 contribution is visual.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ for cymatic-init+loss vs. plain-CNN | rationale |
|---|---|---|
| composite | [+0.05, +0.15] | large lift on natural-fit data |
| top-1 (Chladni, 64 classes) | [+10 pp, +25 pp] | core target |
| epochs-to-90%-top-1 | [-50%, -30%] | faster convergence |
| params | [0, 0] | init-only |
| FLOPs | [0, 0] | unchanged |
| rotation-equivariance err | [-0.02, +0.02] | minor |
| KV cache @ 32k (LLM) | [0, 0] | N/A |
| Betti collapse rate | [-0.1, +0.1] | minor |
| perplexity (LLM, if used) | [-0.5, +0.5] | minor |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Step 1:** generate dataset (≈ 1h one-time on CPU; 64k images)
- **Architecture A:** plain CNN (3 conv layers, He init)
- **Architecture B:** plain CNN with cymatic-init (H35)
- **Architecture C:** B + cymatic-loss (H46, λ=0.05)
- **Architecture D:** B + cymatic-loss + Platonic alignment (H49)
- **Epochs:** 30
- **Seeds:** 0, 1, 2
- **Run-script:** `python scripts/run_idea.py --idea 56 --seeds 0 1 2`
- **Wall-clock:** ≈ 1h generation + 30 min × 3 seeds × 4 archs ≈ 7 h
- **Archive path:** `ideas/56_cymatic_dataset/experiments/exp001_chladni_classify/`

### 7.2 Idea-targeted experiment

The strongest version: classify *unseen* frequency modes via
zero-shot inference. Holds out modes (1,5)..(5,1) during training;
evaluates on these test modes. Tests whether cymatic-init learns the
*generative principle* rather than memorizing patterns.

### 7.3 Cross-paradigm context (LLM track)

Tokenize Chladni patterns as DCT coefficient sequences; classify with
a small transformer + H66 cymatic-QKV init.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G6 row H56.
- Master experiment list: `EXPERIMENT_LOG.md` Tier 6 row T6.5
  (planned).
- Implementation sub-directory: `ideas/56_cymatic_dataset/`
- Related hypotheses that compose:
  - **H35** Cymatic Wavelet Init — H56 is the natural-fit benchmark.
  - **H46** Cymatic Loss — same.
  - **H66** Cymatic QKV Kernel (LLM-track sibling).
  - **H70** Cymatic resonance curriculum — uses H56 patterns.
  - **H57** AudioSet cross-modal — H56 is the visual analog.
- Related hypotheses that conflict:
  - None directly; dataset is a benchmark.

## 9. Committee Q&A

**Q: Why isn't this just MNIST with different content?**

> The dataset is *constructed* from the prior's underlying physics
> (2D Laplacian eigenmodes). This makes it the *natural-fit* benchmark
> for cymatic priors — analogous to Spherical MNIST being the
> natural-fit benchmark for spherical CNNs.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies ≥5 pp lift with 95% CI exclusion; if the priors fail
> on their own natural benchmark, they are demonstrably broken.

**Q: What if the prior helps on Chladni but hurts on ImageNet?**

> That is exactly the expected scope; the previous CIFAR negative is
> the data-mismatch result, not the broken-primitive result. H56
> confirms.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H56 is a dataset, not a prior. The compound-failure framing applies
> when testing priors on H56.

**Q: How do we know the implementation is correct?**

> `tests.py` verifies (a) generated images at (m, n) have exactly
> m × n connected sand-line regions, (b) FFT of the pattern peaks at
> (m, n), (c) class label corresponds to the generator parameters.

## 10. Verification artifacts checklist

- [ ] `ideas/56_cymatic_dataset/implementation.py` exists
- [ ] `ideas/56_cymatic_dataset/tests.py` ≥ 8 assertions
- [ ] `ideas/56_cymatic_dataset/AUDIT.md` ≥ 3 weaknesses
- [ ] `ideas/56_cymatic_dataset/IMPROVEMENTS.md`
- [ ] `ideas/56_cymatic_dataset/VERIFY.md` signed
- [ ] One experiment archive
- [ ] `verification/{tests,smoke,gates,reproduction}.txt`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-C.
