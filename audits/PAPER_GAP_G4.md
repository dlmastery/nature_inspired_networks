# PAPER_GAP_G4 — Kernels / Attention / Filters (H31–H40)

> Reviewer doctrine: anchor the doc's headline claim to the cited
> arXiv paper, then diagnose why our CIFAR-10 12-ep number does (or
> does not) reproduce the paper's regime. Verdict tiers:
> **NO_ARXIV** — no paper backs the specific claim;
> **CITATION_DOESNT_SUPPORT** — cited paper points the other way;
> **DOMAIN** — paper's regime is wrong domain/scale for CIFAR;
> **SCALE** — claim is real at large scale but invisible at 12 ep;
> **WRONG_TEST** — protocol cannot distinguish the effect;
> **IMPL_BUG** — implementation provably mismatches paper math;
> **REPRODUCED** — we recover the paper's effect.

Baselines (12-ep CIFAR-10, seed 0): `baseline_resnet20` top-1 **0.8478**,
`baseline_sg_vanilla` (phi_budget channel ladder base) **0.8216**.

---

## Summary

| H | Tag                              | top1   | Δ vs RN20 | Δ vs sg_vanilla | Paper-gap verdict          |
|---|----------------------------------|--------|-----------|-----------------|----------------------------|
| H31 | sg_only_golden_spiral_init    | 0.8057 | −4.21 pp  | −1.59 pp        | NO_ARXIV (+ post-fix re-run pending) |
| H32 | (no CIFAR row — ViT/sparse)   | —      | —         | —               | DOMAIN                     |
| H33 | (no CIFAR row)                | —      | —         | —               | NOT_RUN                    |
| H34 | (no CIFAR row — LM/RoPE)      | —      | —         | —               | CITATION_DOESNT_SUPPORT    |
| H35 | sg_only_cymatic_init          | 0.7764 | −7.14 pp  | −4.52 pp        | NO_ARXIV + WRONG_TEST + latent IMPL_BUG (cycle path) |
| H36 | (no CIFAR row — PE)           | —      | —         | —               | DOMAIN                     |
| H37 | (no CIFAR row — ViT)          | —      | —         | —               | DOMAIN                     |
| H38 | (no CIFAR row)                | —      | —         | —               | NOT_RUN (+ latent IMPL_BUG: half-pixel crop) |
| H39 | sg_only_phi_activation        | 0.7995 | −4.83 pp  | −2.21 pp        | SCALE + WRONG_TEST         |
| H39 | slot_act_phi (on phi_budget)  | 0.8537 | +0.59 pp  | +3.21 pp        | REPRODUCED (slot context)  |
| H40 | (no CIFAR row)                | —      | —         | —               | NOT_RUN                    |

**Counts:** NO_ARXIV = 2 (H31, H35) · CITATION_DOESNT_SUPPORT = 1 (H34) ·
DOMAIN = 3 (H32, H36, H37) · SCALE = 1 (H39) · WRONG_TEST = 2 (H35, H39) ·
IMPL_BUG candidates = 2 (H35 cycle-path, H38 even-kernel crop; H31 spiral
already patched by Fixer-InitFilter) · NOT_RUN = 3 (H33, H38, H40) ·
REPRODUCED = 1 (H39 in `slot_act_phi` slot composition).

---

## H31 — Golden-Spiral Conv Init — VERDICT: **NO_ARXIV**

- **Anchor:** He init (He 2015 arXiv:1502.01852) is the gold-standard
  fan-in Gaussian init. No arXiv paper of which we are aware claims that
  a *golden-spiral* multiplicative mask on top of He init outperforms
  He on CIFAR-10 ResNet-20. Sci-critic rated H31 NUMEROLOGY.
- **Our result:** 0.8057 (−4.21 pp vs RN20). The previous Fixer-InitFilter
  patch corrected the spiral formula (uniform-angle + generic-exp → true
  `r ∝ φ^(θ/(π/2))` with golden-angle phyllotaxis), so the number is now
  on the *real* φ-spiral. The post-fix re-run must be recorded in
  FINDINGS as the canonical number for H31.
- **Gap diagnosis:** even with the math correct, the structured mask
  modulates He variance per-filter and so locally violates He's i.i.d.
  Gaussian assumption. Without a paper-backed mechanism connecting
  golden-angle filter geometry to optimisation dynamics or
  generalisation on natural images, this is a NUMEROLOGY hypothesis;
  no paper-gap closure is expected.
- **IMPL_BUG?:** Fixed (formula). Q&A-test `test_phi_growth` /
  `test_golden_angle_step` must exist in `tests/` (Rule 25).

---

## H32 — Fibottention (Fibonacci Dilated Attention) — VERDICT: **DOMAIN**

- **Anchor:** Longformer (Beltagy 2020 arXiv:2004.05150), BigBird
  (Zaheer 2020 arXiv:2007.14062), Sparse Transformer (Child 2019
  arXiv:1904.10509). All three show sparse attention matches dense at
  2–6 % density on **long-context** sequences (4k–32k tokens).
- **Our result:** no CIFAR row — the hypothesis is ViT-track.
- **Gap diagnosis:** CIFAR images are 32×32 (≤ 1024 patch tokens at
  patch=1, typically 64 at patch=4). Sparse attention's regime starts
  at sequence lengths where dense attention is memory-bound; CIFAR is
  too small to need sparsity. Even on long-context LMs, sparse matches
  dense — it does not *beat* it on accuracy. DOMAIN mismatch + no
  expected lift on CIFAR. To produce a meaningful row, run on ViT
  with long-sequence ImageNet-21k pretraining or a tokenised LM
  benchmark.

---

## H33 — Vesica Piscis Filter — VERDICT: **NOT_RUN**

- **Anchor:** Multi-scale kernels (Szegedy 2015 Inception arXiv:1409.4842),
  OctConv (Chen 2019 arXiv:1904.05049). Both support multi-scale
  reception fields generically.
- **Our result:** no CIFAR row.
- **Gap diagnosis:** the mechanism is "row of 3 horizontally-shifted
  discs" — closer to a 1-D Flower-of-Life tile than the canonical
  2-disc vesica (audit G4 §H33 MINOR). To resolve the paper-gap, run
  `sg_only_vesica_piscis` smoke at seed 0. If the disc layout is fixed
  to satisfy `|c₁ − c₂| = r`, the claim becomes a special-case multi-
  scale Inception block — at best a marginal Inception substitute.

---

## H34 — Golden-Angle Rotary (RoPE-φ) — VERDICT: **CITATION_DOESNT_SUPPORT**

- **Anchor:** RoPE (Su 2021 arXiv:2104.09864). Base θ = 10000 is the
  canonical hyperparameter. The long-context extension literature
  (Llama-3 base=500k, YaRN, LongRoPE) uniformly **increases** the base
  to stretch the period; H34 sets base = φ ≈ 1.618, which **shrinks**
  the per-pair period to ~1.6, aliasing positions inside a single
  rotation cycle by token ~4.
- **Our result:** no CIFAR row — LM-track.
- **Gap diagnosis:** the cited paper's optimisation arrow points in
  the opposite direction. The H34 claim is therefore not supported by
  the cited literature — quite the contrary. To honestly evaluate
  RoPE-φ one must benchmark on a transformer LM at sequences ≥ 1024,
  which CIFAR does not cover. The hypothesis is mathematically clean
  (rotation is orthogonal, audit G4 §H34 PASS) but its directionality
  is unsupported.

---

## H35 — Cymatic / Chladni Init — VERDICT: **NO_ARXIV + WRONG_TEST + IMPL_BUG (latent)**

- **Anchor:** Closest peer-reviewed analogue is mean-field structured
  init (Schoenholz 2016 arXiv:1611.01232) and the broader edge-of-
  chaos init literature. No arXiv claims Chladni / 2-D wave-equation
  eigenmodes as conv init beat He on CIFAR.
- **Our result:** 0.7764 (−7.14 pp vs RN20, −4.52 pp vs sg_vanilla).
  Strong negative.
- **Gap diagnosis:**
  1. **NO_ARXIV** — no paper-backed claim that QR-orthonormalised
     sin·sin eigenmodes preserve He variance through ReLU on RGB
     natural-image conv stacks.
  2. **WRONG_TEST** — at 12 ep the init's effect on the variance
     dynamics is dominated by the rapid LR schedule; only longer-run
     loss-landscape probes would distinguish He from Chladni.
  3. **IMPL_BUG (latent)** — `chladni_modes_banded` cycles literal
     duplicate modes when `n_modes > k²` (audit G4 §H35 MINOR). For
     `out_c=32, k=3` ⇒ `k²=9 < 32`, the cycle path fires in real
     training and silently violates the orthonormality contract that
     the doc *promises* is the load-bearing mechanism. The negative
     result may partially be the duplicate-mode signal, not the
     Chladni-prior signal.
- **Fix:** raise on `n_modes > k²` OR sign-flip duplicates
  deterministically; add regression test; re-run.

---

## H36 — φ-Spiral Positional Encoding — VERDICT: **DOMAIN**

- **Anchor:** Vaswani 2017 (Transformer PE), Su 2021 (RoPE),
  phyllotaxis literature (no specific arXiv for golden-spiral PE in
  transformers).
- **Our result:** no CIFAR row — transformer PE track.
- **Gap diagnosis:** PE matters at sequence lengths where positional
  ambiguity is real; CIFAR-as-image with CNN backbone does not use a
  sequential PE at all. The audit confirms the math is correct
  (PASS); to evaluate the paper-gap we need a ViT or small LM run.

---

## H37 — Pentagonal φ-Attention — VERDICT: **DOMAIN**

- **Anchor:** Cohen 2019 Icosahedral CNN (arXiv:1902.04615), Vaswani
  2017, Dosovitskiy 2021 ViT (arXiv:2010.11929). Icosahedral
  equivariance is for spherical signals; pentagonal head grouping in
  attention has no specific arXiv backing.
- **Our result:** no CIFAR row — ViT-only.
- **Gap diagnosis:** the audit shows the relative-position bias is
  correctly wired (PASS, not the constant-additive trap). But the
  effect requires a multi-head transformer with n_heads divisible by
  5 and patch tokenisation, which CIFAR-on-ResNet does not exercise.
  Run a tiny ViT on CIFAR or a ViT-tiny on ImageNet to close the gap.

---

## H38 — Fractal Golden Filter (3+5+8 paths) — VERDICT: **NOT_RUN + IMPL_BUG (latent)**

- **Anchor:** FractalNet (Larsson 2017 arXiv:1605.07648), Inception
  (Szegedy 2015), OctConv (Chen 2019). Multi-scale kernel literature
  supports the *idea* of summing Fibonacci-sized paths.
- **Our result:** no CIFAR row.
- **Gap diagnosis:**
  - The k=8 (even) path crops the trailing row/col after
    `pad = k//2 = 4`, introducing a **half-pixel offset** between the
    k=3/k=5 paths and the k=8 path (audit G4 §H38 MINOR). The
    "fractal alignment across Fibonacci scales" framing is broken on
    the largest path. **IMPL_BUG candidate.**
  - To honestly run the hypothesis, switch to `(3, 5, 7)` (odd-only)
    or apply asymmetric padding `F.pad(x, [3,4,3,4])` for k=8.
- **Fix:** patch padding, re-add regression test for centre-of-mass
  alignment, then smoke.

---

## H39 — PhiGELU — VERDICT: **SCALE + WRONG_TEST (sg_only); REPRODUCED (slot_act_phi)**

- **Anchor:** GELU (Hendrycks 2016 arXiv:1606.08415) and SiLU/Swish
  (Ramachandran 2017 arXiv:1710.05941). β = φ ≈ 1.618 sits between
  SiLU (β=1) and GELU's approximation (β ≈ 1.702). The specific
  choice β = φ has no specific arXiv backing.
- **Our result (sg_only_phi_activation):** 0.7995 (−4.83 pp vs RN20,
  −2.21 pp vs sg_vanilla).
- **Our result (slot_act_phi, on phi_budget base):** 0.8537 (+0.59 pp
  vs RN20, **+3.21 pp** vs sg_vanilla).
- **Gap diagnosis:**
  - At 12 ep on a vanilla ResNet-20, PhiGELU vs ReLU/SiLU/GELU are
    near-equivalent in the activation literature; the protocol cannot
    distinguish β=1, φ, 1.702 — **SCALE + WRONG_TEST**.
  - In the slot composition on a phi_budget channel base,
    `slot_act_phi` beats the vanilla baseline by +3.21 pp and the
    RN20 baseline by +0.59 pp — consistent with the activation-as-
    smooth-gating literature where β between 1 and 2 is broadly
    beneficial. This is the **REPRODUCED** branch: H39 is a real but
    small effect, observable only when stacked on top of a usable
    base.
- **Implication:** the headline H39 claim should be reframed as
  "activation choice β=φ is a benign drop-in replacement for ReLU
  comparable to SiLU/GELU; visible only in compositional settings".
  Not a stand-alone winner; not a NUMEROLOGY loss either.

---

## H40 — Metatron Kernel Overlap — VERDICT: **NOT_RUN**

- **Anchor:** DCFNet (Qiu 2018 arXiv:1802.04145), HexaConv (Hoogeboom
  2018 arXiv:1803.02108), Cohen-Welling 2016. Structured-basis
  convolutions are paper-supported in the equivariance literature.
- **Our result:** no CIFAR row.
- **Gap diagnosis:** the audit confirms the 1+6+6 disc geometry is
  correct (PASS) and the basis decomposition is well-conditioned.
  Run `sg_only_metatron_kernel` at seed 0 to populate the row. The
  honest expected effect is small at 12 ep (basis-conv generally
  needs longer schedules to differentiate from free conv).

---

## Group-level conclusions

1. **G4 has the strongest implementations** of all groups (audit:
   5 PASS, 4 MINOR, 1 MAJOR, 0 BROKEN) — but **also the most paper-gap
   problems** because half the hypotheses target transformer / RoPE
   regimes that CIFAR cannot exercise. DOMAIN dominates.
2. **NO_ARXIV** hypotheses (H31, H35) are the candidates most at risk
   of being NUMEROLOGY in disguise; H31's Fixer patch closes the math
   gap but does not produce a paper-backed mechanism.
3. **IMPL_BUG candidates** in G4: **H35 cycle-path** (`n_modes > k²`
   duplicates), **H38 even-kernel crop** (half-pixel shift on k=8
   path), **H31 already patched** (spiral formula) — these are the
   only true implementation-vs-paper drifts.
4. **The H39 result is the most informative**: stand-alone PhiGELU
   does not move the 12-ep needle, but PhiGELU-in-a-composition does
   — vindicating the "drop-in activation" framing and matching the
   SiLU/GELU literature where activation differences emerge only
   under sufficient stacking.
5. **Recommended next runs** (to close paper-gaps): re-smoke H31 with
   the Fixer patch, smoke H33/H38/H40 once H38 padding is fixed, and
   defer H32/H34/H36/H37 to ViT/LM tracks where their papers actually
   live.

— end PAPER_GAP_G4 —
