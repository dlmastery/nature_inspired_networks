# Paper-gap audit — Group G6 (Topological / Bridging hypotheses, H51–H60)

Auditor: paper-gap reviewer (Opus 4.7), 2026-05-29.
Scope: 9 hypotheses (H57 deferred — not audited). Cross-check between
each design doc's anchor citation, the live implementation, and any
CIFAR-10 ablation row that bears on the claim.

Doctrine: per hypothesis (1) what the cited paper actually demonstrated,
(2) what our implementation does, (3) what the CIFAR row (if any)
shows, (4) classification of the gap (PAPER_AGREES, PAPER_DISAGREES,
PAPER_AGREES_TANGENTIALLY, NO_ARXIV, CITATION_DOESNT_SUPPORT, DOMAIN,
IMPL_BUG, IMPL_SURROGATE, NO_DATA).

## Summary

| H   | classification         | CIFAR row              | one-line gap                                                                    |
|-----|------------------------|------------------------|---------------------------------------------------------------------------------|
| H51 | IMPL_SURROGATE         | NO_DATA                | smooth cdist+sort surrogate, not real PH; cited paper used gudhi/Ripser         |
| H52 | NO_DATA                | NO_DATA                | DropPath impl correct (Fixer-verified); no CIFAR row run                        |
| H53 | DOMAIN                 | NO_DATA                | DeepSphere/GICOPix are spherical (sky/climate); CIFAR is flat 2-D               |
| H54 | IMPL_SURROGATE         | NO_DATA                | PH-reg uses H51 surrogate; not real persistent homology                         |
| H55 | IMPL_BUG               | NO_DATA                | Platonic head_bias identically 0 → vanilla MHA; Islam2025 is molecular anyway   |
| H56 | CITATION_DOESNT_SUPPORT| NO_DATA                | drum-membrane eigenmodes used, not Chladni plate spectra                        |
| H57 | DEFERRED               | DEFERRED               | not audited                                                                     |
| H58 | PAPER_DISAGREES        | sg_only_group(_avg)    | mean reduce FALSIFIED (0.6538 vs max 0.6984, –4.46 pp)                          |
| H59 | NO_DATA                | NO_DATA                | trained-Betti utility; no CIFAR ablation row                                    |
| H60 | NO_DATA                | NO_DATA                | 3-seed uncertainty utility; no CIFAR ablation row                               |

Tier counts (excluding H57):
- IMPL_BUG candidates: **1** (H55)
- IMPL_SURROGATE: 2 (H51, H54)
- DOMAIN mismatch: 1 (H53)
- CITATION_DOESNT_SUPPORT: 1 (H56)
- PAPER_DISAGREES: 1 (H58)
- NO_DATA (impl correct, just no row): 3 (H52, H59, H60)

---

## H51 — Betti loss (`src/nature_inspired_networks/betti_loss.py`)

- **Anchor citation.** Brüel-Gabrielsson, Nelson, Dwaraknath, Lio,
  Carlsson, Leskovec 2019 "A Topology Layer for Machine Learning"
  (arXiv:1906.00722). The cited TopologyLayer uses **real persistent
  homology** computed by `gudhi`/Ripser through a vertex/edge filtration
  with a true matrix-reduction backward.
- **Our implementation.** `differentiable_persistence` sorts the
  upper-triangle of `cdist` and treats *gap between consecutive edges*
  as a persistence proxy. `_soft_count_above(lifetimes, t)` =
  `sigmoid(32·(x−t)).sum()` returns a real-valued β-surrogate. This is
  a smooth gap-based heuristic, **not** Vietoris–Rips persistence —
  G6 audit explicitly flagged it as "Rips-skeleton heuristic".
- **CIFAR row.** None. H51 is exposed as an auxiliary loss; no sweep
  row currently turns it on.
- **Classification: IMPL_SURROGATE.** The cited paper's results
  depended on a true PH backend (gudhi). Our surrogate cannot reproduce
  Brüel-Gabrielsson et al.'s claims because the underlying persistence
  computation is mathematically different. Not an IMPL_BUG (the
  surrogate is correctly implemented and the design doc flags it as a
  surrogate), but the empirical gap to the cited paper is unbridgeable
  without integrating a real PH library.

---

## H52 — DropPath (`src/nature_inspired_networks/drop_path.py`)

- **Anchor citation.** Larsson, Maire, Shakhnarovich 2017 ICLR
  "FractalNet: Ultra-Deep Neural Networks without Residuals"
  (arXiv:1605.07648) — drop-path matches ResNet on CIFAR at 400 epochs;
  Huang et al. 2016 ECCV "Deep Networks with Stochastic Depth"
  (arXiv:1603.09382) is the survival-probability backbone.
- **Our implementation.** Per-sample timm convention with `(B,1,1,...,1)`
  Bernoulli mask divided by `keep_prob` — correct expectation rescale.
  `FractalDropPath.drop_probs = [p_max·k/(n−1)]` monotone non-decreasing.
  Fixer/G6-audit verified.
- **CIFAR row.** None. Drop-path is a regulariser ingredient, not its
  own ablation row.
- **Classification: NO_DATA.** Implementation matches the cited
  papers; we just never ran the head-to-head row. The DERIVATIVE+TESTABLE
  sci-rating is fair. If the cited paper's 400-epoch ResNet-match is
  the falsifier, we have not yet run that experiment.

---

## H53 — Icosa unfold (`src/nature_inspired_networks/icosa_unfold.py`)

- **Anchor citation.** Cohen, Geiger, Köhler, Welling 2018/2019
  "Spherical CNNs" (arXiv:1801.10130); Defferrard et al. 2018
  "DeepSphere" (arXiv:1810.12186); Yu et al. 2019 NeurIPS GICOPix; HEALPix
  Górski et al. 2005. **All of these target spherical data**: climate
  fields, sky surveys, molecular surfaces.
- **Our implementation.** A deterministic bijective permutation
  mapping a 12-icosa-vertex vector into a planar (4,3) grid suitable
  for `Conv2d` weight reuse. The G6 audit confirmed the permutation
  is bijective but the "great-circle adjacency" claim breaks because
  the 12 vertex z-coordinates split as `{±1, ±0.618, 0×4}`, which does
  not factor into 4 bands of 3.
- **CIFAR row.** None.
- **Classification: DOMAIN.** Even if the unfold preserved adjacency
  exactly, CIFAR-10 images live on a flat 2-D torus/plane, not on the
  sphere. The cited papers' wins (rotation-equivariance on spherical
  fields) do not transfer to flat 32×32 RGB images — there is no
  spherical structure to exploit. The hypothesis is mis-targeted, not
  mis-implemented.

---

## H54 — PH activation regulariser (`src/nature_inspired_networks/ph_reg.py`)

- **Anchor citation.** Same TopologyLayer literature as H51 (Brüel-
  Gabrielsson 2019 arXiv:1906.00722; Naitzat, Zhitnikov, Lim 2020 JMLR
  "Topology of Deep Neural Networks" arXiv:2004.06093; Hofer, Kwitt,
  Niethammer 2019 ICML).
- **Our implementation.** `register_forward_hook` captures named-stage
  outputs; `loss()` runs each through `BettiLoss` (the H51 surrogate).
  Hooks are removed correctly. G6 audit flagged a silent-zero-loss
  foot-gun if `loss()` is called before forward.
- **CIFAR row.** None.
- **Classification: IMPL_SURROGATE.** Because H54 wraps H51's gap-
  based surrogate, it inherits the same gap to the cited PH literature.
  An empirical CIFAR row with H54 would not be a faithful test of
  topology-regularised classification as published; it would be a test
  of *our gap-surrogate-regularised* classification.

---

## H55 — Platonic Transformer (`src/nature_inspired_networks/platonic_transformer.py`)

- **Anchor citation.** Islam, Welling et al. 2025 "Platonic
  Transformers: A Solid Choice for Equivariance" (arXiv:2510.03511).
  Their target dataset is **molecular property prediction**: QM9
  energies, MD17 force-fields. The "Platonic" symmetry exploits SO(3)
  point-group structure on 3-D molecular conformations.
- **Our implementation.** `PlatonicAttention` computes a per-head bias
  as `gram.mean(dim=-1)` where `gram = coords @ coords.T` over the
  vertex set of a Platonic solid. For every vertex-transitive Platonic
  solid the vertices sum to zero (centroid 0), so
  `gram[i,:].mean() = (1/n)·v_i·Σ_j v_j = 0`. The G6 audit verified
  empirically: head_bias is `[0,0,…,0]` for tetra, octa, icosa, dodeca.
  The module is bit-equivalent to vanilla `nn.MultiheadAttention` with
  a forced head count.
- **CIFAR row.** None.
- **Classification: IMPL_BUG.** The head bias — the only mechanism
  carrying the Platonic prior — is mathematically void. Even if it
  were non-zero, the cited paper's domain (molecular SE(3)) does not
  match CIFAR-10 (planar 2-D images), so a corrected impl would still
  be DOMAIN-mismatched. The Islam 2025 citation also lacks an arXiv ID
  in the doc bibliography (Rule-4 violation flagged in G6 audit).

---

## H56 — Cymatic dataset (`src/nature_inspired_networks/cymatic_dataset.py`)

- **Anchor citation.** Chladni 1787 (the founder of plate-vibration
  patterns); Rahaman et al. 2019 ICML "On the Spectral Bias of Neural
  Networks" (arXiv:1806.08734) for the spectral-bias linkage.
- **Our implementation.** `generate_cymatic_pattern` produces
  `sin(m·π·x_grid · freq) · sin(n·π·y_grid · freq)` — i.e. the
  separable solution to the **2-D rectangular drum-membrane**
  Helmholtz problem with Dirichlet boundary, *not* the Chladni plate
  eigenmodes (which obey the biharmonic equation with free-edge
  boundary). True Chladni patterns have nodal-line structure dictated
  by the bi-Laplacian eigenfunctions, NOT the separable sin×sin
  drum-membrane modes. SciCritic flagged this in the H56 addendum.
- **CIFAR row.** None.
- **Classification: CITATION_DOESNT_SUPPORT.** Chladni-1787 is cited
  as motivation but the dataset is drum-membrane modes. The cited
  Rahaman spectral-bias work is about Fourier-frequency learning and
  *is* tangentially supported (the dataset DOES exercise progressively
  higher Fourier modes). The "Chladni" branding is the gap; the
  spectral-bias science is intact.

---

## H57 — Audio cross-modal

**DEFERRED.** Not audited per instructions. No CIFAR row applies
(audio dataset, not image).

---

## H58 — GroupConv2d reduce='max'/'mean' (`priors.py`)

- **Anchor citation.** Cohen, Welling 2016 ICML "Group Equivariant
  Convolutional Networks" (arXiv:1602.07576) — group-convolution
  pools over the orbit, typically via `mean` (the orbit average is the
  G-invariant projection) or `max` (oriented-response argmax). The
  paper itself uses mean for the invariant feature map.
- **Our implementation.** `GroupConv2d` builds a `(G,O,I,k,k)` rot90
  stack and reduces along the orbit dim by either `mean` or `amax`.
  Both code paths are correct.
- **CIFAR row.** Two rows exist:
  - `sg_only_group` (max reduce): top1 = **0.6984**, composite = 0.6937
  - `sg_only_group_avg` (mean reduce): top1 = **0.6538**, composite = 0.6513
  - **Delta = −4.46 pp** in favour of max-reduce on CIFAR-10 ResNet-20.
- **Classification: PAPER_DISAGREES.** Cohen & Welling 2016 advocate
  mean as the invariant orbit projection. Our CIFAR-10 ResNet-20
  measurement falsifies that prescription for this specific
  configuration: max wins by 4.46 pp. The doc has been updated to
  reflect the empirical finding (G6-audit confirms "the falsification
  is properly documented in the docstring"). This is a healthy
  PAPER_DISAGREES — the prior literature's recipe does NOT win here,
  and we documented why (max acts as a soft argmax over oriented
  responses, retaining the strongest signal rather than averaging it
  with the 3 weaker ones).

---

## H59 — Trained-feature Betti (`src/nature_inspired_networks/trained_betti.py`)

- **Anchor citation.** Naitzat et al. 2020 JMLR (arXiv:2004.06093);
  Hofer et al. 2017 NeurIPS "Deep Learning with Topological
  Signatures"; Bauer 2021 "Ripser" (arXiv:1908.02518). These papers
  compute *trained-feature* topology via real PH on extracted features.
- **Our implementation.** Loads checkpoint via `state_dict`, calls
  `collect_features`, then `betti_curve(feats, rel_thresh=...)`. G6
  audit flagged the `strict=False` state_dict load as a silent-failure
  hazard but the math is correct.
- **CIFAR row.** None — this is a post-hoc analysis utility, not a
  training-time prior.
- **Classification: NO_DATA.** The mechanism is faithful to the cited
  literature (real Betti curves via Ripser-style filtration on trained
  features); we simply have no CIFAR row that uses it as a
  classification feature. The G6-audit's MINOR finding (strict=False)
  is a robustness issue, not a paper-gap issue.

---

## H60 — Three-seed uncertainty (`src/nature_inspired_networks/multi_seed.py`)

- **Anchor citation.** Henderson et al. 2018 AAAI "Deep Reinforcement
  Learning that Matters" (arXiv:1709.06560); Bouthillier et al. 2019
  NeurIPS workshop "Unreproducible Research is Reproducible";
  Loshchilov & Hutter 2019 ICLR AdamW. These prescribe **sample-std
  with Bessel's correction** and bootstrap CIs over multiple seeds.
- **Our implementation.** Sample std `(n−1)` denominator;
  `bootstrap_ci` resamples with replacement. G60 audit confirmed
  correct.
- **CIFAR row.** Not a per-experiment row — H60 is the project's
  3-seed re-run discipline (used downstream for FINDINGS error bars).
- **Classification: NO_DATA.** Mechanism faithful to literature;
  no paper-gap. The 3-seed discipline is exercised on Phase-5
  winners (H09 phi_budget, H81 sine_act, H80 slot_init_cymatic per
  FINDINGS).

---

## Group-level conclusion

The G6 paper gap is dominated by **surrogate-vs-real-PH** drift (H51,
H54), **domain mismatch** (H53 spherical CNNs forced onto flat 2-D
images; H55 molecular SE(3) forced onto image classification), and
**one PAPER_DISAGREES win for max-reduce over the literature's mean-
reduce prescription** (H58, −4.46 pp). H55 is the sole IMPL_BUG
candidate (head_bias identically 0). Three hypotheses (H52, H59, H60)
have correct implementations but no CIFAR rows to compare against —
they remain NO_DATA until run.
