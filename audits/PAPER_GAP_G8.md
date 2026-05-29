# Paper-gap audit — Group G8 (Esoteric Extensions, H76–H84)

Auditor: paper-gap reviewer (Opus 4.7), 2026-05-29.
Scope: 9 neutral-recast hypotheses (H76 Tetrahedral DualPath, H77
Radial-12 Attention, H78 Toroidal Latent, H79 Morphing Polytope,
H80 Constant-Width / Reuleaux Conv, H81 Sinusoidal / SIREN, H82
Voronoi Sparse Attention, H83 Collapse-Gated Attention, H84 Spectral
Hopfield).

Doctrine: per hypothesis (1) what the cited paper actually
demonstrated, (2) what our implementation does, (3) what the CIFAR
row shows, (4) classification (PAPER_AGREES, PAPER_DISAGREES,
PAPER_AGREES_TANGENTIALLY, NO_ARXIV, CITATION_DOESNT_SUPPORT, DOMAIN,
IMPL_BUG, NO_DATA).

## Summary

| H   | classification              | CIFAR row(s)                        | one-line gap                                                              |
|-----|-----------------------------|-------------------------------------|---------------------------------------------------------------------------|
| H76 | NO_DATA                     | none                                | impl correct; no row                                                      |
| H77 | NO_DATA                     | none                                | impl correct (Shaw 2018 relative bias); no row                            |
| H78 | NO_DATA                     | none                                | T² embedding correct; no row                                              |
| H79 | NO_DATA                     | none                                | morphing GCN adjacency correct; no row                                    |
| H80 | NO_ARXIV / PAPER_DISAGREES  | sg_only_constant_width 0.7595       | no specific Reuleaux-conv paper; mask drops ~50% taps → –8.83 pp          |
| H80b| PAPER_AGREES (cymatic init) | slot_init_cymatic 0.8540 (3-seed grad) | Chladni init lifted classification; cited spectral-bias supports         |
| H81 | PAPER_AGREES_TANGENTIALLY   | sg_only_sine_act 0.8062 / slot_act_sine 0.8556 ; CIFAR-100 3-seed median 0.5784 winner | SIREN cited for signal reconstruction, transfers to classification |
| H82 | NO_DATA                     | none                                | Delaunay / kNN attention; no row                                          |
| H83 | NO_DATA                     | none                                | softplus-τ + collapse; no row                                             |
| H84 | NO_DATA                     | none                                | spectral Hopfield; no row                                                 |

Tier counts:
- PAPER_AGREES: 1 (H80b cymatic init slot variant)
- PAPER_AGREES_TANGENTIALLY: 1 (H81 SIREN)
- NO_ARXIV / PAPER_DISAGREES: 1 (H80 Reuleaux constant-width conv)
- NO_DATA: 7 (H76, H77, H78, H79, H82, H83, H84) — and the H80b
  cymatic variant is the same module-family as H80 but with a
  different config flag

IMPL_BUG candidates: **0** in G8 (no broken modules — G8 audit closed
with 7 PASS / 2 MINOR / 0 MAJOR / 0 BROKEN).

---

## H76 — TetrahedralDualPathBlock — NO_DATA

- **Anchor citation.** Cohen & Welling 2016 ICML (arXiv:1602.07576)
  group-equivariant CNN; convex-merge gating literature.
- **Our implementation.** Two GroupConv2d paths (`reduce='max'` and
  `reduce='mean'`) fused by `β·A + (1−β)·B` with learnable β through
  sigmoid. G8 audit PASS with mechanism-verifying tests.
- **CIFAR row.** None.
- **Classification: NO_DATA.** Composition of two correct group-conv
  reductions; no head-to-head measurement yet.

---

## H77 — RadialSymmetry12Attention — NO_DATA

- **Anchor citation.** Shaw, Uszkoreit, Vaswani 2018 NAACL
  "Self-Attention with Relative Position Representations"
  (arXiv:1803.02155).
- **Our implementation.** Relative bias `bias[h,i,j] = cos(angles[h] +
  2π·(j−i)/L) / PHI`, added to logits before softmax. G8 audit PASS.
- **CIFAR row.** None — radial-12 attention is a transformer-stack
  prior, not exercised by ResNet-20 sweep.
- **Classification: NO_DATA.**

---

## H78 — ToroidalLatent — NO_DATA

- **Anchor citation.** Toroidal-embedding literature (e.g., Davidson
  et al. 2018 "Hyperspherical VAEs" arXiv:1804.00891; Falorsi et al.
  2018 "Reparameterizing Distributions on Lie Groups"
  arXiv:1903.02958).
- **Our implementation.** `(θ₁,θ₂) → (cos θ₁, sin θ₁, cos θ₂, sin θ₂)
  ∈ T² ⊂ ℝ⁴`. G8 audit PASS; unit-norm invariants verified.
- **CIFAR row.** None.
- **Classification: NO_DATA.**

---

## H79 — MorphingPolytopeAdjacency — NO_DATA

- **Anchor citation.** Kipf & Welling 2017 ICLR GCN
  (arXiv:1609.02907); Fuller "jitterbug" geometric framing.
- **Our implementation.** `A(t) = (1−t)·A_cubocta + t·A_icosa`,
  sigmoid-gated learnable t, symmetric-normalised GCN. G8 audit PASS.
- **CIFAR row.** None.
- **Classification: NO_DATA.**

---

## H80 — ConstantWidthConv2d (Reuleaux) — NO_ARXIV / PAPER_DISAGREES

- **Anchor citation.** No specific arXiv claims Reuleaux-mask conv
  outperforms standard 3×3. The hex-CNN literature (Hoogeboom et al.
  2018 arXiv:1803.02108) supports *masked-kernel* ideas in general,
  but not the Reuleaux mask specifically.
- **Our implementation.** `reuleaux_mask(k)` = intersection of 3 disks
  at equilateral-triangle vertices. The mask zeros out ~50% of the
  3×3 taps (corners suppressed). Applied at every forward as
  `weight · mask`. G8 audit PASS — impl is mathematically faithful
  to the Reuleaux definition.
- **CIFAR row.** `sg_only_constant_width_seed0`: **top1 = 0.7595**,
  composite = 0.7629.
- **Baseline.** `baseline_resnet20_seed0`: **top1 = 0.8478**,
  composite = 0.8457.
- **Delta = −8.83 pp** vs ResNet-20 baseline.
- **Classification: NO_ARXIV / PAPER_DISAGREES.** No arXiv ever
  claimed Reuleaux masking outperforms standard 3×3 conv on image
  classification; the −8.83 pp drop is entirely explainable as
  mask-capacity loss (zeroing out the corner taps of the 3×3 kernel
  removes ~50% of learnable parameters per spatial position). The
  implementation is correct; the *hypothesis* (Reuleaux constant
  width is a useful inductive bias for classification) is empirically
  FALSIFIED by our measurement, and was never supported by a cited
  paper to begin with.

---

## H80b (cymatic-init variant) — PAPER_AGREES

The H80 family also contains a cymatic-init variant exercised as
`slot_init_cymatic_seed0` (top1 0.8540, composite 0.8254; CIFAR-100
3-seed graduate per FINDINGS at median 0.85 / post-fix winner).
This is a different mechanism (Chladni-band initialisation rather
than Reuleaux masking) and aligns with the **spectral-bias**
literature (Rahaman et al. 2019 arXiv:1806.08734) that prescribes
mid-frequency initialisation to counteract low-frequency bias.
**Classification: PAPER_AGREES.** The cited spectral-bias mechanism
predicts a faster fit on mid-frequency targets, and our +0.62 pp
over baseline (0.8540 vs 0.8478 on CIFAR-10 seed 0) is consistent
with that prediction. Logged here for completeness; tracked in detail
under the H45-H50 G5 cymatic family.

---

## H81 — SinusoidalActivation (SIREN) — PAPER_AGREES_TANGENTIALLY

- **Anchor citation.** Sitzmann, Martel, Bergman, Lindell, Wetzstein
  2020 NeurIPS "Implicit Neural Representations with Periodic
  Activation Functions" (arXiv:2006.09661). SIREN was demonstrated for
  **coordinate networks / implicit representations / signal
  reconstruction**: fitting RGB images, audio, 3-D shapes, video. The
  paper reports near-perfect reconstruction fidelity. SIREN was
  **never claimed** to help classification — its mechanism (periodic
  activations with carefully-scaled ω₀ for spectral-bias mitigation)
  is targeted at fitting smooth coordinate-indexed signals.
- **Our implementation.** `sin(ω·x)` with learnable ω. G8 audit
  MINOR — the default `omega_init=1.0` is not the canonical SIREN
  `ω₀=30` first-layer recipe, but the swap helper exposes it via
  `swap_relu_with_sine(model, omega_init=…)`. The Sitzmann citation
  is fully formatted (arXiv:2006.09661).
- **CIFAR-10 rows.**
  - `sg_only_sine_act_seed0`: **top1 = 0.8062**, composite = 0.8197.
  - `slot_act_sine_seed0`: **top1 = 0.8556**, composite = 0.8618.
  - Baseline `baseline_resnet20_seed0`: top1 = 0.8478, composite =
    0.8458.
  - `slot_act_sine` is **+0.78 pp** over baseline on CIFAR-10 seed 0.
- **CIFAR-100 3-seed.** Per FINDINGS / Phase-5 graduation, the
  sine-act slot is a 3-seed median winner at top1 ≈ **0.5784**.
- **Classification: PAPER_AGREES_TANGENTIALLY.** SIREN was never
  designed for classification — Sitzmann's results are signal-fitting
  benchmarks. That our slot-variant (a single ReLU→sine swap inside
  one block) lifts CIFAR-10 by +0.78 pp and is a CIFAR-100 3-seed
  median winner is **surprising relative to the cited paper's scope**.
  The transfer is plausibly explained by the spectral-bias-mitigation
  mechanism: classification features benefit from richer high-frequency
  representations in the same way reconstruction does. But we should
  flag in the paper that this is an off-label use of SIREN — the cited
  arXiv paper did not predict the classification lift.

---

## H82 — VoronoiSparseAttention — NO_DATA

- **Anchor citation.** Sparse-attention literature (Child et al. 2019
  "Generating Long Sequences with Sparse Transformers"
  arXiv:1904.10509; Beltagy et al. 2020 Longformer arXiv:2004.05150).
  Voronoi/Delaunay adjacency on attention is not a standard published
  pattern.
- **Our implementation.** `voronoi_adjacency(N, seed)` via `scipy.spatial.Delaunay`
  with symmetric kNN fallback. G8 audit PASS.
- **CIFAR row.** None.
- **Classification: NO_DATA.**

---

## H83 — CollapseGatedAttention — NO_DATA

- **Anchor citation.** Martins & Astudillo 2016 ICML "Sparsemax"
  (arXiv:1602.02068); Peters, Niculae, Martins 2019 "Sparse
  Sequence-to-Sequence Models" (arXiv:1905.05702). Both precedents
  for temperature- / sparsity-controlled softmax.
- **Our implementation.** `τ = softplus(τ_raw) + 1e-3` (floor present);
  optional `collapse ∈ [0,1]` interpolates toward one-hot argmax.
  G8 audit PASS — including verification that the 1e-3 floor is wired
  on both `tau` property and forward path.
- **CIFAR row.** None.
- **Classification: NO_DATA.**

---

## H84 — SpectralHopfieldMemory — NO_DATA

- **Anchor citation.** Ramsauer et al. 2020 NeurIPS "Hopfield
  Networks is All You Need" (arXiv:2008.02217).
- **Our implementation.** `[Re(rfft) ‖ Im(rfft)]` feature stack, β-
  weighted softmax over scores, readout in the signal domain. G8
  audit MINOR — the "isometry" framing in the docstring overstates
  the relationship (rfft non-uniformly weights DC/Nyquist vs other
  bins), but retrieval works empirically (`test_associative_recall`
  passes at β=20).
- **CIFAR row.** None.
- **Classification: NO_DATA.** The Hopfield retrieval mechanism is
  faithful at the *similarity-ranking* level; the spectral "isometry"
  framing is a doc gap (G8 audit), not a paper-gap with the cited
  arXiv (Ramsauer 2020 does not claim spectral re-parameterisation).

---

## Group-level conclusion

G8 has the **cleanest implementation track record** of the three
groups (zero BROKEN, zero MAJOR per the underlying audit). The paper
gap is dominated by **NO_DATA** (7 of 9 hypotheses have no CIFAR row
because the modules are transformer-stack / GCN / Hopfield primitives
not exercised by the ResNet-20 sweep), with two empirical rows on
H80 (Reuleaux PAPER_DISAGREES at −8.83 pp, but no arXiv ever claimed
the win) and H80b/H81 (cymatic init / sine activation — both
PAPER_AGREES or PAPER_AGREES_TANGENTIALLY, with H81 the most
notable surprise: SIREN was designed for signal reconstruction yet
lifts CIFAR-100 3-seed median to 0.5784, making it a Phase-5 winner).

**Note for paper writing.** The H81 SIREN finding is the strongest
G8 result and should be flagged as PAPER_AGREES_TANGENTIALLY (off-
label transfer) rather than PAPER_AGREES — the cited paper's scope
was signal reconstruction, not image classification. The mechanism
(periodic activations for spectral-bias mitigation) plausibly
transfers, and our 3-seed numbers confirm the transfer, but Sitzmann
et al. did not predict the classification lift in advance.
