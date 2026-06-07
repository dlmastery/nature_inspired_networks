# Negative Results

**Status:** structural skeleton per
[SYNTHESIS_100.md item B17](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md).
**Date filed:** 2026-06-06.
**Completion target:** Week 12 of the 12-week plan (paper-rewrite agent
fills remaining rows after Waves 0-4 land).

---

## Purpose

The 84-hypothesis substrate produced 7-10 priors that survived to Phase-8
screening and Phase-9i diagnostic (the exact count depends on which
reframing is binding post-iso-FLOPs re-test). The **other 74+** priors
were not promoted. Reviewer 3 (R3 #6 + R4 #29) and the
[SYNTHESIS_100.md item B17](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md)
framing flagged that those 74 negative results are buried in the
project's archive and need first-class reporting.

Per [CLAUDE.md Rule 36](../CLAUDE.md), every non-promoted prior carries one
of four verdicts:

- **FALSIFIED** — the pre-registered falsifier landed on the right
  dataset at the right scale; the prior is empirically refuted.
- **NUMEROLOGY** — sci-critic flagged the hypothesis as π-style after-the-fact
  curve-fitting; either FALSIFIED on a real test or untestable in
  principle. Distinct from FALSIFIED because the **mechanism** does not
  justify the prediction even before data is seen.
- **UNTESTED_ON_RIGHT_DATASET** — Rule-36 designated verdict for any
  hypothesis whose pre-registered falsifier specified a dataset that
  wasn't in the sweep. Common case: priors with toroidal / hex /
  rotation symmetry tested on upright CIFAR (where the symmetry is
  not load-bearing).
- **DERIVATIVE** — empirically plausible but the mechanism is a known
  prior-art special case (e.g., H09 vs RegNet Pareto region). Honest
  authorship requires explicit disclosure.

Each row below cites the audit (`audits/G<X>_audit.md` if applicable),
the design doc, and the resolving evidence file.

---

## Group G1 — Scaling & Growth (H01-H10)

Five representative entries; remaining 5 rows are TODO Week-12.

| Hypothesis ID | Name | Pre-registered falsifier | Verdict | Audit ref |
|---|---|---|---|---|
| H01 | φ compound scaling | Top1 lift vs EfficientNet compound at iso-FLOPs CIFAR-100 ≥ +0.5 pp at n=3 | UNTESTED_ON_RIGHT_DATASET — pre-reg required ImageNet compound-scaling testbed; only CIFAR-100 ran. Phase-9j n=7 deferred. | `audits/G1_audit.md` |
| H02 | Fibonacci depth progression | Sub-Pareto depth grid shows φ-progression dominates linear at iso-FLOPs | FALSIFIED on CIFAR-10 12-ep screening (n=1); not promoted to CIFAR-100. Sci-critic verdict NUMEROLOGY (no derivation that Fib depths help under cosine schedule). | `audits/G1_audit.md` |
| H05 | Fractal φ recursion | Multi-scale data (Camelyon17) lift ≥ +1 pp vs ResNet-20 | UNTESTED_ON_RIGHT_DATASET — only CIFAR ran. [SYNTHESIS_100.md B9](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md) re-tests on Camelyon17 in Block B. | — |
| H08 | Dynamic φ growth | Function-preserving grow_model lifts intermediate-epoch top1 | FALSIFIED_AT_MECHANISM_TEST. Fixer-Growth patch landed but mechanism-pinning test (function preservation under grow) weak. [SYNTHESIS_100.md B19](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md) demotes from contributions. | `audits/G1_audit.md` |
| H09 | Golden Proportion Parameter Budget | Pareto-dominates RegNetX-200MF at iso-FLOPs on Imagenette/Tiny-ImageNet/ImageNet-100 | DERIVATIVE+TESTABLE per sci-critic. Open under Wave-1 pre-registration ([B12](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md)). Phase-9i Δ at non-iso-FLOPs not binding. | `audits/G1_audit.md` |
| _TODO Week-12_ | Remaining 5 G1 rows (H03, H04, H06, H07, H10) | — | — | — |

---

## Group G2 — Layer / Channel / Neuron (H11-H20)

Five representative entries; remaining 5 rows are TODO Week-12.

| Hypothesis ID | Name | Pre-registered falsifier | Verdict | Audit ref |
|---|---|---|---|---|
| H11 | Pure Fibonacci MLP | Top1 ≥ baseline at iso-params on CIFAR-100 | FALSIFIED at CIFAR-10 screening; sci-critic NUMEROLOGY (Fib-width MLPs are width-vs-depth Pareto-dominated by regular widths). | `audits/G2_audit.md` |
| H13 | Golden neuron connectivity | Sparse-connectivity lift on CIFAR-100 at n=3 | FALSIFIED at screening; -2 pp vs dense baseline. | `audits/G2_audit.md` |
| H15 | φ-initialized embedding | LLM perplexity lift on small-scale token model | UNTESTED_ON_RIGHT_DATASET — image-classification substrate; no LM run. | — |
| H17 | Golden ratio skip connections | Δ vs identity skip at iso-FLOPs CIFAR-100 | DERIVATIVE per sci-critic (1/φ skip-scaling sits inside the published learnable-skip Pareto region). | `audits/G2_audit.md` |
| H20 | Fibonacci ensemble | Lifts test top1 over equal-weight ensemble at iso-compute | UNTESTED_ON_RIGHT_DATASET — no ensemble experiments in the screening matrix. | — |
| _TODO Week-12_ | Remaining 5 G2 rows (H12, H14, H16, H18, H19) | — | — | — |

---

## Group G3 — Topologies & Graphs (H21-H30)

Five representative entries; remaining 5 rows are TODO Week-12.

| Hypothesis ID | Name | Pre-registered falsifier | Verdict | Audit ref |
|---|---|---|---|---|
| H21 | Hexagonal φ packing | Lift on hex-symmetric task (aerial / tiled texture) at iso-FLOPs | UNTESTED_ON_RIGHT_DATASET — only upright CIFAR ran; SYNTHESIS_100.md B6 schedules AID re-test. | `audits/G3_audit.md` |
| H22 | Toroidal φ closure | Lift on wrap-aware dataset (tiled-CIFAR) at iso-FLOPs | UNTESTED_ON_RIGHT_DATASET — pre-reg called for tiled-CIFAR; only upright CIFAR ran. SYNTHESIS_100.md B7/E11 schedules tiled-CIFAR re-test. | `audits/G3_audit.md` |
| H23 | Platonic φ graph | Vertex-transitive graph attention lift on small-world benchmark | NUMEROLOGY per sci-critic — head-bias provably zero under vertex-transitive centroid identity (R3 #4 caught this). | `audits/G3_audit.md` |
| H24 | Icosahedral φ equivariant | Equivariance error < ε on Spherical MNIST AND lift on rotated test set | UNTESTED_ON_RIGHT_DATASET — Spherical MNIST never ran. Subsumed by H71 in Wave-4. | — |
| H28 | Cymatic hex resonance | Spectrogram task lift over Xavier init | UNTESTED_ON_RIGHT_DATASET — only CIFAR ran; SYNTHESIS_100.md B14 schedules UrbanSound8K re-test. | — |
| _TODO Week-12_ | Remaining 5 G3 rows (H25, H26, H27, H29, H30) | — | — | — |

---

## Group G4 — Kernels / Attention / Filters (H31-H40)

Five representative entries; remaining 5 rows are TODO Week-12.

| Hypothesis ID | Name | Pre-registered falsifier | Verdict | Audit ref |
|---|---|---|---|---|
| H31 | Golden spiral kernel | Top1 lift over He-init at iso-FLOPs CIFAR-100 at n=3 | FALSIFIED at CIFAR-10 screening (-0.4 pp vs He init); sci-critic NUMEROLOGY for the spiral-init derivation. | `audits/G4_audit.md` |
| H33 | Vesica Piscis filter | Lift on rotation-augmented CIFAR at iso-FLOPs | UNTESTED_ON_RIGHT_DATASET — only upright CIFAR ran. | — |
| H34 | Golden angle rotary | LM perplexity lift on small-scale token model | UNTESTED_ON_RIGHT_DATASET — image substrate; no LM run. | — |
| H36 | φ-spiral positional encoding | LM perplexity lift over sinusoidal PE | UNTESTED_ON_RIGHT_DATASET — image substrate. | — |
| H38 | Fractal golden filter | Multi-scale data lift at iso-FLOPs | UNTESTED_ON_RIGHT_DATASET — only CIFAR ran; even-k-pad-fix landed but not re-validated on multi-scale. | `audits/G4_audit.md` |
| _TODO Week-12_ | Remaining 5 G4 rows (H32, H35, H37, H39, H40) | — | — | — |

---

## Group G5 — Optimization / Init / Regularization / NAS (H41-H50)

Six representative entries; remaining 4 rows are TODO Week-12.

| Hypothesis ID | Name | Pre-registered falsifier | Verdict | Audit ref |
|---|---|---|---|---|
| H41 | Golden ratio optimizer | β-only AdamW(β2=1/φ) shows non-convergence behaviour at ≥ 100 ep per Reddi 2018 | UNTESTED_ON_RIGHT_DATASET — Reddi prediction needs ≥ 100 ep on hard task; only 12-ep screening ran. Phase-9j n=7 deferred. | `audits/G5_audit.md` |
| H42 | φ weight initialization | Top1 ≥ He init at iso-FLOPs CIFAR-100 | FALSIFIED on CIFAR-10 screening; -1.2 pp vs He init. | `audits/G5_audit.md` |
| H43 | Fibonacci pruning | Iterated-pruning lift over magnitude pruning | FALSIFIED_AT_MECHANISM_TEST. EMA-load test caught silent strict=False fallback (D14). [SYNTHESIS_100.md B19](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md) demotes from contributions. | `audits/G5_audit.md` |
| H44 | Golden regularization | Layer-graded WD lifts CIFAR-100 at iso-FLOPs | CONDITIONAL — phi_decay_wd carries part of `pair_gm_pdw` signal; isolation pending [SYNTHESIS_100.md E6](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md) 2³ factorial. | `audits/G5_audit.md` |
| H47 | φ dropout | Cyclic-period dropout lifts CIFAR-100 generalization | DERIVATIVE per sci-critic (cyclic dropout schedules are published). | `audits/G5_audit.md` |
| H50 | Full sacred hybrid | Stacked-all priors lift at iso-FLOPs | FALSIFIED — `sg_full_fib` showed -11.54 pp on CIFAR-10 (CLAUDE.md Rule 23 cautionary tale). | `audits/G5_audit.md` |
| _TODO Week-12_ | Remaining 4 G5 rows (H45, H46, H48, H49) | — | — | — |

---

## Group G6 — Topological + Bridging (H51-H60)

Five representative entries; remaining 5 rows are TODO Week-12.

| Hypothesis ID | Name | Pre-registered falsifier | Verdict | Audit ref |
|---|---|---|---|---|
| H51 | Topological Betti loss | Generalization-gap reduction at iso-compute | DERIVATIVE per sci-critic (PH-based regularization is published). Currently 2× slowdown under autocast (D15). | `audits/G6_audit.md` |
| H53 | Icosa unfold bridge | Equivariance error < ε on Spherical MNIST | FALSIFIED_AT_MECHANISM_TEST — equivariance test (E9 / SYNTHESIS_100) shows it is augmentation, not steerable convolution. Renamed per E8. | `audits/G6_audit.md` |
| H55 | Platonic Transformers | Vertex-transitive attention lift | NUMEROLOGY — head bias provably zero under vertex-transitive centroid identity. Islam 2025 citation pending hallucination check (F18). | `audits/G6_audit.md` |
| H56 | Cymatic pattern dataset | Synthetic vibration-mode dataset improves CIFAR-100 transfer | UNTESTED_ON_RIGHT_DATASET — dataset never constructed. | — |
| H57 | Audio-cymatic cross-modal | Cross-modal alignment lift | UNTESTED_ON_RIGHT_DATASET — implementation deferred (no module per INDEX). | — |
| _TODO Week-12_ | Remaining 5 G6 rows (H52, H54, H58, H59, H60) | — | — | — |

---

## Group G7 — Cross-Paradigm Hybrids (H61-H75)

Six representative entries (G7 is the largest group at 15 hypotheses);
remaining 9 rows are TODO Week-12.

| Hypothesis ID | Name | Pre-registered falsifier | Verdict | Audit ref |
|---|---|---|---|---|
| H61 | Sacred-Liquid-JEPA hybrid | Multi-paradigm fusion lift on CIFAR-100 | UNTESTED — implementation complexity exceeded budget; deferred. | — |
| H62 | Toroidal KV + hex attention | LM perplexity lift on small-scale token model | UNTESTED_ON_RIGHT_DATASET — image substrate. | — |
| H67 | Full paradigm hybrid | All-paradigm stack lifts CIFAR-100 | FALSIFIED_AT_MECHANISM_TEST — shape-only tests passed; mechanism-pinning tests caught broken implementation. [SYNTHESIS_100.md B19](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md) demotes from contributions. | `audits/G7_audit.md` |
| H69 | KAN-Metatron symbolic head | Symbolic head lift on small reasoning task | UNTESTED — no reasoning-task substrate in screening. | — |
| H71 | IcosaRoPE3D | Lift ≥ +3 pp at rotated-test Spherical MNIST n=5 paired | OPEN — sole NOVEL+TESTABLE sci-critic survivor. Wave-4 pre-registered ([B5/B18](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md)). | — |
| H74 | Metatron overlap tying | Weight-tying lift at iso-params CIFAR-100 | FALSIFIED_AT_MECHANISM_TEST — shape-only tests passed but mechanism-pinning weak. [SYNTHESIS_100.md B19](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md) demotes from contributions. | `audits/G7_audit.md` |
| _TODO Week-12_ | Remaining 9 G7 rows (H63, H64, H65, H66, H68, H70, H72, H73, H75) | — | — | — |

---

## Group G8 — Esoteric Extensions (H76-H84)

Five representative entries; remaining 4 rows are TODO Week-12.

| Hypothesis ID | Name | Pre-registered falsifier | Verdict | Audit ref |
|---|---|---|---|---|
| H76 | Tetrahedral dual-path | Dual-stream lift at iso-FLOPs CIFAR-100 | UNTESTED — implementation deferred. | — |
| H77 | Radial-symmetry-12 attention | Rotation-aware attention lift on rotated CIFAR | UNTESTED_ON_RIGHT_DATASET. | — |
| H79 | Morphing polytope adjacency | Dynamic-graph lift at iso-FLOPs | UNTESTED. | — |
| H81 | Sinusoidal activation | Lift over ReLU at iso-FLOPs CIFAR-100 | DERIVATIVE — this is Sitzmann 2020 SIREN replication (arXiv:2006.09661). Control 2 shows tanh > sine by +0.48 pp. [SYNTHESIS_100.md E4](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md) demotes `slot_act_sine` from the abstract triple. | — |
| H83 | Collapse-gated attention | Lift on rare-class task at iso-FLOPs | UNTESTED. | — |
| _TODO Week-12_ | Remaining 4 G8 rows (H78, H80, H82, H84) | — | — | — |

---

## TODO — paper-rewrite agent (Week-12)

Fill in the remaining ~36 rows above following the same row format. For
each row, pull the pre-registered falsifier from the design doc
`hypotheses/g<N>_<group>/H<NN>_*.md`, the verdict from
[`audits/G<X>_audit.md`](../audits/) or the sci-critic addendum block in the
design doc, and the resolving evidence file from `experiments/`,
`experiments_modern/`, or `ideas/<NN>/experiments/`.

Order of completion:

1. Rows with audit references already in `audits/` (highest signal).
2. Rows where the design doc carries a sci-critic addendum block.
3. Rows where only the design doc + pre-registered falsifier exist
   (UNTESTED verdicts).

---

## Cross-references

- Pre-registration index: [`pre-registration/README.md`](../pre-registration/README.md)
- Falsifier contract: [`FALSIFIERS.md`](FALSIFIERS.md)
- Hypothesis index: [`hypotheses/INDEX.md`](../hypotheses/INDEX.md)
- Audit ledgers: [`audits/`](../audits/)
- IDEA_TABLE master status: [`hypotheses/IDEA_TABLE.md`](../hypotheses/IDEA_TABLE.md)
- 100-improvement synthesis: [`SYNTHESIS_100.md`](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md)
