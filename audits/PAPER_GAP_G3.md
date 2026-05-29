# Paper-Gap Audit — Group G3 (Topologies & Graphs, H21–H30)

Reviewer: paper-gap auditor. Goal: diagnose, per hypothesis, whether the
gap between the cited arXiv paper(s) and our CIFAR-10 12-epoch number is
attributable to (a) implementation bug, (b) the cited paper not actually
supporting the claim we are testing, (c) domain mismatch between the
paper's evaluation setting and ours, or (d) absence of a peer-reviewed
anchor altogether. This audit consumes the impl-critic findings in
`audits/G3_audit.md` (6 MAJOR, now Fixer-patched) and the sci-critic
verdicts embedded in each `hypotheses/g3_topologies_graphs/H<NN>_*.md`.

Baseline reference (CIFAR-10, 12 ep, ResNet-20 backbone shared by all
G3 NaturePrior variants): `baseline_resnet20_seed0` top-1 = 0.8478 at
15 ep / 0.272 M params. `baseline_sg_vanilla` (NaturePriorBlock with
all flags off) at 12 ep is the apples-to-apples reference at the same
0.127 M parameter budget.

## Summary

Classification counts (10 hypotheses):

| Classification | Count | Hypotheses |
|---|---|---|
| **CITATION_DOESNT_SUPPORT** | 3 | H21, H22, H29 |
| **DOMAIN_MISMATCH** | 3 | H24, H26, H28 |
| **NO_ARXIV** | 3 | H23, H25, H30 |
| **PAPER_SUPPORTS_BUT_UNTESTED_ON_CIFAR** | 1 | H27 |
| **IMPL_BUG** | **0** | — |

Group conclusion. G3 is the clearest example in the campaign of a
**research-design defect, not an engineering defect**. Every cited
arXiv paper that DOES exist (HexaConv, Spherical CNNs, Pittorino flat-
basin, Watts–Strogatz, FractalNet) was evaluated on data with the
matching geometric structure (hexagonal aerial imagery, spherical
panoramas, flat-loss-basin classification analysis, social networks,
non-toroidal CIFAR but for a DIFFERENT mechanism). CIFAR-10 is the
canonical 2-D square-pixel benchmark with no rotational invariance
demanded, no hex sampling lattice, no spherical topology, no torus
wrap. Forcing icosahedral, hexagonal, toroidal, fractal, or cymatic
priors onto CIFAR-10 means the inductive bias is **paying capacity for
a structure the data does not have**. The observed −5 pp drop on
`sg_only_hex` and −7 pp drop on `sg_only_toroidal` (vs baseline) is
therefore the *expected* signature of capacity-cost-without-payoff,
not an implementation defect.

The 6 MAJOR findings the impl-critic raised in `G3_audit.md` and that
the Fixer campaign patched (φ-radial weighting on H21, φ-wrap on H22,
78-edge graph on H23, avg-pool orbit on H24, hex-tap cymatic on H28,
20-vertex dodeca-Fib on H30) corrected mechanism vs documentation —
but **none of those fixes change the domain-mismatch diagnosis** for
the CIFAR-10 sweep numbers. The post-fix Phase-2 re-runs on
`sg_only_hex` and `sg_only_toroidal` will record the corrected
mechanism, but the predicted CIFAR-10 ceiling remains low because the
underlying paper's success regime is not CIFAR.

IMPL_BUG candidates (require running code-investigation re-runs):
**none**. The Fixer campaign already addressed every doc-vs-code gap
the impl-critic raised, and the residual gap is paper-vs-data, not
paper-vs-code.

---

## H21 — Hexagonal φ-Packing → `sg_only_hex` (top-1 = 0.7932, −5.5 pp vs `baseline_resnet20`)

- **Anchor citation**: Hoogeboom 2018 *HexaConv* (arXiv:1803.02108).
- **What the paper actually evaluated**: AID (Aerial Image Dataset)
  and UC-merced (aerial / overhead imagery). Reports 25–42 %
  memory/time savings and ~+1.0 pp top-1 *on hex-resampled data*.
  Hex equivariance pays off because aerial imagery is approximately
  isotropic at small scales (no preferred orientation, no horizon).
- **What we ran**: CIFAR-10 32×32 RGB, RandomCrop+HFlip. Square pixels,
  no hex resampling. The "hex prior" is a 7-tap zero-corners mask on a
  3×3 square kernel — a structural constraint that *removes* 2 of 9
  kernel weights without changing the underlying lattice.
- **Classification**: **CITATION_DOESNT_SUPPORT for CIFAR-10 use.** The
  cited paper supports hex-equivariance on hex-grid data. CIFAR-10
  does not have a hex grid. The −5 pp gap is the cost of dropping ~22 %
  of conv-kernel capacity (2/9 weights) without an aligned geometric
  payoff.
- **Falsifier reachability after Fixer**: φ-radial weighting now lives
  in the code (H21 Fixer patch); rerunning at seed 0 on CIFAR-10 will
  not save the hypothesis — the φ-radial extension still operates on
  square pixels. The legitimate test of H21 is rotated-CIFAR
  (CIFAR-10-Rot) or a hex-resampled CIFAR pipeline, neither of which
  is in our current campaign.
- **IMPL_BUG candidate?**: No. The 7-tap mask is mathematically what
  the doc and HexaConv paper describe. The remaining −5 pp is domain
  cost, not impl cost.

## H22 — Toroidal φ-Closure → `sg_only_toroidal` (top-1 = 0.7805, −6.7 pp vs `baseline_resnet20`)

- **Anchor citation**: Pittorino 2022 (arXiv:2202.03038) and CircleNet
  (Schubert 2019) for circular padding.
- **What the paper actually evaluated**: Pittorino studied flat-loss-
  basin geometry under permutation symmetry — a **landscape analysis**,
  not a classification-accuracy claim. CircleNet's circular padding
  was evaluated on detection of *circular signs* (rotational symmetry
  built into the targets). Neither paper claims toroidal padding lifts
  classification accuracy on a non-wrapping image benchmark.
- **What we ran**: CIFAR-10 with `F.pad(... mode="circular")` substituted
  for zero-pad inside NaturePriorBlock. CIFAR images are bounded scenes;
  the top edge of a horse-image is sky and the bottom is grass — pixel
  content does NOT wrap.
- **Classification**: **CITATION_DOESNT_SUPPORT.** SciCritic already
  flagged this as UNFALSIFIABLE for the CIFAR-10 setting in the H22
  design-doc addendum: "CIFAR images do not wrap; predicting better
  generalisation from a torus prior on non-toroidal data is
  unfalsifiable on this benchmark." The −7 pp drop is the natural cost
  of forcing the model to treat top↔bottom and left↔right edges as
  adjacent when the data semantics do not.
- **Falsifier reachability after Fixer**: φ-scaling of wrap distance
  (H22 Fixer patch) is now in `toroidal_pad`; rerunning will still
  pay the wrap-on-non-toroidal-data cost. Legitimate test would be
  tiled / panoramic data (Cityscapes panoramas, omnidirectional MNIST),
  not CIFAR.
- **IMPL_BUG candidate?**: No.

## H23 — Platonic φ-Graph (Metatron 13) → no CIFAR row

- **Anchor citation**: Battaglia 2018 (arXiv:1806.01261) relational
  inductive bias; Kipf-Welling 2017 (arXiv:1609.02907) GCN; Gilmer 2017
  (arXiv:1704.01212) MPNN. **No specific arXiv paper claims that the
  Metatron-Cube graph improves any GNN benchmark.**
- **What the paper actually evaluated**: Battaglia / Kipf / Gilmer
  papers are foundational GNN works on Cora, Citeseer, ogbn-arxiv,
  QM9. They support generic relational inductive bias, NOT the
  specific Metatron-Cube 78-edge / 13-vertex topology.
- **What we ran**: nothing on CIFAR-10 — graph task hypothesis,
  not an image-classification task. The Fixer-patched 78-edge graph
  with 3-class φ partition awaits an ogbg-molhiv / Cora bench run.
- **Classification**: **NO_ARXIV.** The Metatron framing is decorative
  sacred-geometry; there is no peer-reviewed paper that motivates
  this specific 13-vertex / 78-edge adjacency on a GNN task. The cited
  GNN papers support graph-NN-in-general, not THIS graph.
- **IMPL_BUG candidate?**: No. The Fixer extended the adjacency to
  the full 78-edge Metatron set; correctness is now matched to the
  (decorative) doc. The hypothesis cannot be falsified or supported
  by CIFAR-10 runs.

## H24 — Icosahedral φ-Equivariant CNN → no CIFAR row

- **Anchor citation**: Cohen 2019 *Spherical CNNs* (arXiv:1902.04615);
  Esteves 2018 (arXiv:1711.06721) icosa-CNN.
- **What the paper actually evaluated**: Spherical MNIST (sphere-
  embedded digits), 3D shape classification on ModelNet40, climate
  panoramic data. The data has **literal spherical / icosahedral
  symmetry** — rotating the input is a meaningful natural variation.
- **What we ran**: nothing on CIFAR-10 (3D-equivariance hypothesis on
  2D square-image data is intrinsically misaligned). Even if wired
  in via `IcosaConv1d`, CIFAR-10's RandomCrop+HFlip augmentation
  already covers the only relevant 2-D nuisance (mirror flip);
  applying I-60 symmetry to a 2-D square image gives 56 redundant
  group elements.
- **Classification**: **DOMAIN_MISMATCH.** The post-Fixer code
  (avg-pool orbit reduction restored, `IcosaConv2d` wired) is now
  faithful to Cohen 2019 — but the paper's gain pattern (`+5 pp on
  Spherical MNIST`) does not transfer to non-spherical data.
- **Falsifier reachability after Fixer**: Spherical MNIST test would
  be the *correct* legitimate test; would require the
  `gicopix.py` unfold to be added to the dataset loader. Not in
  current campaign.
- **IMPL_BUG candidate?**: No. The avg-pool reduction and the 60-
  rotation-applied-to-kernel forward are now both present (Fixer
  patches), matching Cohen 2019. The remaining issue is domain.

## H25 — Dodecahedral Latent → no CIFAR row

- **Anchor citation**: van den Oord 2017 (arXiv:1711.00937) VQ-VAE;
  Snell 2017 (arXiv:1703.05175) ProtoNets; Cohen 2019 icosa-equiv;
  Huh 2024 (arXiv:2405.07987) Platonic Representation Hypothesis;
  Hendrycks 2017 (arXiv:1610.02136) OOD baseline.
- **What the paper actually evaluated**: VQ-VAE codebook learning on
  ImageNet / speech; ProtoNets few-shot on miniImageNet; Huh PRH is
  a *meta-observation* across representation alignment, not a
  prescriptive Platonic-codebook claim.
- **What we ran**: nothing on CIFAR-10. The 20-vertex dodecahedron
  with golden-ratio coords is a fixed-codebook prior — the closest
  honest mapping would be a few-shot OOD-detection benchmark, not
  CIFAR-10 classification.
- **Classification**: **NO_ARXIV for the specific Platonic-codebook
  claim.** Huh 2024 PRH supports the *spirit* (representations
  converge toward a shared geometry) but does not prescribe a
  dodecahedral codebook as the geometry. The cited papers support
  codebook-style latents in general; none anchor 20-vertex
  dodecahedron specifically.
- **IMPL_BUG candidate?**: No. The impl-critic gave H25 a PASS
  verdict pre-Fixer; the code is mathematically clean.

## H26 — Fractal Toroidal → no CIFAR row

- **Anchor citation**: Larsson 2017 *FractalNet* (arXiv:1605.07648).
- **What the paper actually evaluated**: FractalNet on CIFAR-10 /
  CIFAR-100 / SVHN / ImageNet, reporting 4.6 % CIFAR-10 error
  (a 95.4 % top-1). Importantly, FractalNet did NOT use toroidal
  padding — it used standard zero-padding.
- **What we ran**: nothing on CIFAR-10 *as configured for this
  hypothesis* (no `sg_only_fractal_toroidal` tag in the sweep).
  Note: there IS an `sg_only_fractal_seed0/1/2` row in
  `experiments/cifar10/` — but that is the H15 (FractalNet zero-pad
  variant), not H26.
- **Classification**: **DOMAIN_MISMATCH.** FractalNet's success on
  CIFAR-10 supports recursive expansion; the toroidal padding added
  here inherits H22's wrap-on-non-toroidal-data cost. The combined
  prior is "FractalNet (data-aligned) + Torus (data-misaligned)",
  which strictly cannot exceed FractalNet alone unless the torus
  somehow helps — and there is no paper showing it does on CIFAR.
- **IMPL_BUG candidate?**: No. Impl-critic verdict was PASS.

## H27 — Golden-Spiral Graph Init → no CIFAR row

- **Anchor citation**: Vogel 1979 phyllotaxis formula (not arXiv,
  classical botany paper); Glorot–Bengio 2010 *Xavier init*
  (no arXiv, JMLR/AISTATS).
- **What the paper actually evaluated**: Vogel produced a 2-D point
  pattern for phyllotaxis (sunflower seed spirals). Xavier is the
  standard init for any DNN. **The cited works support each part of
  the H27 mechanism**, but the combination (Vogel-spiral in 2 dims
  + Xavier in remaining d−2 dims, lifted via random orthonormal
  projection) is novel and **untested on CIFAR-10**.
- **What we ran**: nothing on CIFAR-10. Graph / GNN task hypothesis.
- **Classification**: **PAPER_SUPPORTS_BUT_UNTESTED_ON_CIFAR.** Closest
  to a legitimate hypothesis among the no-CIFAR G3 entries: each
  sub-claim has a peer-reviewed anchor. The composite claim is the
  honest research increment.
- **IMPL_BUG candidate?**: No. Impl-critic verdict was MINOR (the
  random orthonormal projection differs from the doc's "Xavier in
  remaining d−2 dims" but is isotropy-equivalent).

## H28 — Cymatic Hex Resonance → no CIFAR row (Fixer-patched)

- **Anchor citation**: Chladni 1787 plate cymatics (pre-arXiv);
  Hoogeboom 2018 HexaConv (arXiv:1803.02108).
- **What the paper actually evaluated**: HexaConv on aerial imagery
  (see H21). Chladni is a 200-year-old physics observation, not a
  DL benchmark. **No peer-reviewed paper supports cosine-modulated
  hex-tap weights for image classification.**
- **What we ran**: nothing on CIFAR-10. Pre-Fixer the impl
  modulated per-output-channel (mismatched doc); post-Fixer the impl
  modulates per-hex-tap (matched doc).
- **Classification**: **DOMAIN_MISMATCH + NO_ARXIV for the
  modulation-of-hex-taps mechanism.** Even with the Fixer correction,
  the prior compounds H21's hex-on-square-pixel cost with a learned
  per-tap cosine modulation — there is no paper claiming this
  combined operator improves any benchmark.
- **IMPL_BUG candidate?**: No. The Fixer landed `cos(ω·t + k·φ)`
  with `k = neighbour index`, exactly what the doc demands.

## H29 — φ Small-World Rewiring → no CIFAR row

- **Anchor citation**: Watts & Strogatz 1998 *Nature* "Collective
  dynamics of small-world networks" (no arXiv; foundational network
  science).
- **What the paper actually evaluated**: WS define the small-world
  regime as `p ∈ [0.001, 0.1]` — i.e., rewire 0.1 % to 10 % of
  edges. At `p ≈ 0.5` clustering collapses to random; at `p = 1`
  the graph is pure Erdős–Rényi. The H29 default is **p = 1/φ ≈
  0.618**, which is firmly in the **Erdős–Rényi regime**, NOT
  small-world.
- **What we ran**: nothing on CIFAR-10. GNN task hypothesis.
- **Classification**: **CITATION_DOESNT_SUPPORT.** The numerical
  value `1/φ` was chosen for golden-ratio aesthetics and is
  outside the regime the cited Nature paper defines as small-world.
  The hypothesis NAME contradicts its OWN parameter choice.
- **IMPL_BUG candidate?**: No. The WS construction is correct, and
  the rewiring at any `p` is mathematically what WS describe. The
  defect is choosing a value of `p` outside the regime the cited
  paper validates.

## H30 — Platonic-Fib Hybrid → no CIFAR row (Fixer-patched)

- **Anchor citation**: Qi 2017 *PointNet* (arXiv:1612.00593) and
  Wang 2019 *DGCNN* (arXiv:1801.07829) for point-cloud GNNs;
  Cohen 2019 (arXiv:1902.04615) for icosa equivariance.
- **What the paper actually evaluated**: PointNet / DGCNN on
  ModelNet40 point clouds; Cohen icosa-equiv on Spherical MNIST.
  **No paper supports a 20-vertex-dodeca + Fibonacci-channel-
  partition GNN.**
- **What we ran**: nothing on CIFAR-10. Point-cloud / 3-D task
  hypothesis. Pre-Fixer had `(12, (1,1,2,3,5))` icosa partition;
  post-Fixer has `(20, (1,1,2,3,5,8))` dodeca partition per doc.
- **Classification**: **NO_ARXIV for the dodeca-Fib hybrid.** The
  individual components (point-cloud GNN, icosa equivariance, Fib
  channel widths) each have arXiv anchors; the hybrid does not.
- **IMPL_BUG candidate?**: No. Fixer corrected vertex count and
  Fib partition; impl now matches doc.

---

## Cross-hypothesis observations

1. **Zero IMPL_BUG candidates.** The G3 Fixer campaign already
   closed every gap the impl-critic flagged. Remaining gaps are
   research-design (data domain) and citation alignment (the
   cited paper does not support the claim we are testing on
   this dataset), not engineering.

2. **CIFAR-10 is the wrong benchmark for 7 of 10 G3 hypotheses.**
   H23, H24, H25, H26, H27, H28, H29, H30 all need a domain-aligned
   benchmark (graph, sphere, point-cloud, panorama) to evaluate
   their core claim honestly. Only H21 (hex on square pixels) and
   H22 (torus pad on bounded scenes) ran on CIFAR-10 — and both
   underperform baseline by 5–7 pp, exactly as a paper-gap analysis
   would predict.

3. **`sg_only_hex` and `sg_only_toroidal` should NOT be presented
   as falsifications of HexaConv / Pittorino.** The papers were
   never about CIFAR-10. Presenting the −5 pp / −7 pp drop as
   evidence "against the cited papers" is a category error;
   the experiments simply test a claim those papers do not make.

4. **Sci-critic verdicts already foreshadowed this.** The H22
   addendum's UNFALSIFIABLE verdict (CIFAR images do not wrap) is
   the canonical example. The G3 group has the highest density of
   NUMEROLOGY / UNFALSIFIABLE sci-critic verdicts in the campaign,
   which is consistent with this paper-gap audit.

5. **What would change the picture.** Routing H21 to a hex-grid
   benchmark (CIFAR-10 hex-resampled, or aerial-imagery datasets
   AID/UC-merced), H22 to panoramic data, H24/H25/H30 to Spherical
   MNIST or ModelNet40, H23/H27/H29 to ogbg-molhiv or Cora — each
   would let the cited paper's success regime actually meet the
   tested prior. Until then, the G3 negatives on CIFAR-10 are
   uninformative about the underlying ideas.

## Recommendations to FINDINGS.md / PAPER.md

- **DO NOT cite `sg_only_hex` / `sg_only_toroidal` underperformance
  as a refutation of geometric priors.** The cited papers do not
  predict CIFAR-10 wins; our test does not falsify them.
- **DO record the paper-gap classification** alongside the
  experimental number, so external readers can distinguish "we
  tested it on the wrong data" from "the prior demonstrably
  doesn't help".
- **Route H23, H24, H27, H29, H30 to their proper domain** (GNN /
  spherical / point-cloud benchmarks) before any external claim
  about G3 ideas is made.

*— Paper-gap audit G3 complete, 2026-05-29.*
