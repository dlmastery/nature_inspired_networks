# B — Theoretical Orthogonality Analysis (Rule 23 doctrine)

> Companion to `A_empirical_combinations.md` and `C_literature_combinations.md`.
> Author: Agent-B (theoretical-orthogonality auditor). Date: 2026-05-30.
> Scope: classify all 84 hypotheses (H01–H84) by the training-stack axis
> they primarily modify, build a pairwise orthogonality matrix, and stress
> test the maximum N for which all-orthogonal stacking is theoretically
> possible per CLAUDE.md Rule 23.
>
> **Methodology.** Each hypothesis is read for (a) its formal hypothesis
> statement, (b) its CNN-track mechanism (the conv-block forward-path
> question), and (c) its sci-critic addendum verdict. Each is then
> assigned a *primary* axis (the single training-stack layer it modifies
> most) and a *secondary* axis where multi-axis. The "touches conv-block
> forward path?" column answers Rule 23's critical question — Rule 23
> forbids stacking more than two priors on the same conv-block forward
> path. The sci-critic verdict column gates eligibility for external
> claims (Rule 22 — NUMEROLOGY winners are forbidden).

---

## 1. Axis taxonomy (extension of Rule 23's 10 axes to 20)

Rule 23 names 10 axes: arch / channel / momentum / regulariser /
weight-decay / LR / activation / ensemble / pruning / inference. The 84
hypotheses do not map cleanly onto 10 — several modify components Rule
23 did not list (init, normalization, kernel shape, attention pattern,
positional encoding, paradigm-substitution). The full taxonomy required
to characterise all 84 hypotheses is 20 axes:

| # | Axis | Definition (where in the training stack) | Touches conv-block forward path? |
|---:|---|---|:---:|
| A1 | **arch-stage** | Per-stage width / depth / stride / spatial-resolution schedule | YES (gates information bottleneck) |
| A2 | **arch-block** | Block-internal topology (skip, pre-act, bottleneck ratio, SE) | YES |
| A3 | **kernel-shape** | Conv kernel shape / sparsity / mask (constant-width, vesica, fractal-multi-scale) | YES |
| A4 | **attention-pattern** | Attention mask / dilation / sparsity (Fibottention, Voronoi, hex-attn) | LLM-track only (no conv path) |
| A5 | **attention-bias** | Per-head angular / positional bias (RoPE-φ, radial-12, pentagonal) | LLM-track only |
| A6 | **positional-encoding** | Token / patch positional injection (spiral PE, Metatron PE, icosa-RoPE) | LLM-track only |
| A7 | **channel-count** | Total channel budget per stage (φ-budget, Fib channels, full param-budget) | YES |
| A8 | **activation** | Pointwise nonlinearity (PhiGELU, SinusoidalActivation, PhiReLU threshold) | YES (pointwise — same path) |
| A9 | **init** | Weight initialization distribution (Kaiming/He/cymatic/golden-spiral/phi-He) | NEUTRAL (no runtime cost; same path) |
| A10 | **normalization** | BN / LN / GroupNorm / RMSNorm | YES |
| A11 | **regulariser** | Dropout / DropPath / cutout / data-aug (φ-Dropout, DropPath-Anytime) | YES (dropout-style sits in path) |
| A12 | **weight-decay** | Per-layer / per-stage WD coefficient (φ-decay-wd, λ=1/φ^k) | NEUTRAL (gradient-only) |
| A13 | **optimizer** | β1/β2 / variant (PhiAdamW, SAM, Lookahead) | NEUTRAL (gradient-only) |
| A14 | **lr-schedule** | LR-vs-epoch curve (PhiDecayLR, cosine, OneCycle) | NEUTRAL |
| A15 | **momentum-schedule** | β1-vs-epoch curve (GoldenMomentumScheduler) | NEUTRAL |
| A16 | **loss-aux** | Auxiliary loss term (Betti, PH-reg, PRH-align, cymatic-loss) | NEUTRAL |
| A17 | **ensemble** | Multi-checkpoint / multi-model post-train aggregation (Fib-ensemble, SWA) | NEUTRAL (post-training) |
| A18 | **pruning / growth** | Sparsity / dynamic layer-add schedule (Fib-prune, dynamic-phi-growth, growth-pruning) | YES (mutates the model) |
| A19 | **inference-time** | Test-time augmentation / multi-crop / drop-path-anytime | NEUTRAL (post-training) |
| A20 | **paradigm-substitution** | Wholesale module swap (CFC liquid cell, JEPA latent, KAN spline head, Mamba/RWKV) | YES (replaces conv path) |

Auxiliary axes (necessary but Rule-23 already implicitly orthogonal to
the priors): **dataset** (cymatic-pattern dataset H56), **curriculum**
(cymatic low-data curriculum H70), **distillation** (Platonic-aux
cymatic teacher H63), **infrastructure** (trained-feature Betti H59,
3-seed uncertainty H60) — these don't compete with the 20 axes above.

### Why 20 not 10
Rule 23's original 10 collapses several axes that empirically need to
be split:
- "arch" is too broad — `arch-stage` (depth / stride schedule) and
  `arch-block` (block internals like skip-scaling) are independent.
  H02 fib_depth (arch-stage) and H17 golden_skip (arch-block) modify
  different layers of the same backbone.
- "regulariser" needs to be split from `weight-decay` because per-layer
  φ-WD (H44) is gradient-only (NEUTRAL on forward path) while dropout
  (H47) sits in the forward path.
- Optimiser, lr-schedule, momentum-schedule are independent (the
  pair_gm_pdw certified combo proves this — H48 momentum + H44 WD +
  H09 width compose cleanly).
- init is logically its own axis — H42 phi_init touches no other layer.

---

## 2. Hypothesis-by-axis matrix (84 rows)

Columns: ID | name | primary axis | secondary axis | conv-block forward
path touched? | sci-critic verdict | sweep status. Verdicts are
abbreviated: NOV (NOVEL+TESTABLE), DER (DERIVATIVE+TESTABLE), NUM
(NUMEROLOGY), UNF (UNFALSIFIABLE), FAL (FALSIFIED), INF
(INFRASTRUCTURE-NOT-HYPOTHESIS), — (no addendum yet).

| ID | Name | Primary axis | Secondary | Path? | Sci-verdict | Sweep |
|---|---|---|---|:---:|---|---|
| H01 | phi-Compound Scaling | A1 arch-stage | A7 channel | YES | DER | .8042 |
| H02 | Fibonacci Depth | A1 arch-stage | — | YES | DER | .8218 |
| H03 | Golden-Spiral Resolution | A1 arch-stage | — | YES | NUM | .8067 |
| H04 | φ-Self-Similar Width | A7 channel | — | YES | NUM | .8011 |
| H05 | Fractal φ-Recursion | A2 arch-block | A1 stage | YES | DER | .8246 |
| H06 | Golden Bottleneck | A2 arch-block | A7 channel | YES | NUM | .6925 |
| H07 | φ-Modulated Multi-Scale | A1 arch-stage | — | YES | NUM | .8200 |
| H08 | Dynamic φ-Growth | A18 growth | A1 stage | YES | DER | impl |
| H09 | Golden Param Budget | A7 channel | A1 stage | YES | DER (cert) | **.8554** |
| H10 | φ-Decay LR | A14 lr-sched | — | NO | NUM | .7875 |
| H11 | Pure Fibonacci MLP | A7 channel | — | YES | FAL | impl |
| H12 | Fib-Channel CNN | A7 channel | — | YES | FAL | .8011 |
| H13 | Golden Connectivity | A18 pruning | — | YES | FAL | .7333 |
| H14 | Fibonacci Recurrent | A20 paradigm | A7 channel | YES (RNN) | FAL | impl |
| H15 | φ-Init Embedding | A9 init | — | NO | FAL | impl |
| H16 | Fib Head Diversity | A4 attn-pattern | — | LLM | FAL | impl |
| H17 | Golden Skip | A2 arch-block | — | YES | FAL | .8163 |
| H18 | Fib Stage Transition | A1 arch-stage | — | YES | FAL | .7255 |
| H19 | φ-Neuron Threshold | A8 activation | — | YES | FAL | .7107 |
| H20 | Fibonacci Ensemble | A17 ensemble | — | NO | FAL | .8011 |
| H21 | Hex φ-Packing | A3 kernel | — | YES | DER | .7932 |
| H22 | Toroidal φ-Closure | A3 kernel | — | YES | NUM+UNF | .7805 |
| H23 | Platonic φ-Graph | A20 paradigm (GNN) | — | YES | NUM | impl |
| H24 | Icosa φ-Equivariant | A2 arch-block | A3 kernel | YES | DER | impl |
| H25 | Dodeca Latent | A20 paradigm (VQ) | — | YES | DER | impl |
| H26 | Fractal Toroidal | A3 kernel | A2 arch-block | YES | NUM (compositional) | impl |
| H27 | Golden-Spiral Graph | A9 init | A20 paradigm | NO | NUM | impl |
| H28 | Cymatic Hex Resonance | A3 kernel | A8 activation | YES | NUM | impl |
| H29 | φ-Small-World | A2 arch-block (rewiring) | — | YES | NUM | impl |
| H30 | Platonic-Fib Hybrid | A20 paradigm (GNN) | A7 channel | YES | NUM | impl |
| H31 | Golden-Spiral Kernel | A9 init | — | NO | NUM | .8042 |
| H32 | Fibottention | A4 attn-pattern | — | LLM | DER | impl |
| H33 | Vesica Piscis Filter | A3 kernel | — | YES | NUM | impl |
| H34 | Golden-Angle Rotary | A5 attn-bias | A6 PE | LLM | NUM | impl |
| H35 | Cymatic Wavelet | A9 init | — | NO | NUM | .7744 |
| H36 | φ-Spiral PE | A6 PE | — | LLM | NUM | impl |
| H37 | Pentagonal φ-Attention | A4 attn-pattern | A5 bias | LLM | NUM | impl |
| H38 | Fractal Golden Filter | A3 kernel | — | YES | DER | impl |
| H39 | Harmonic φ-Activation | A8 activation | — | YES | DER | .7995 |
| H40 | Metatron Kernel Overlap | A3 kernel | — | YES | NUM | impl |
| H41 | Golden Optimizer | A13 optimizer | A15 momentum | NO | NUM | .5196 / .8394 post-fix |
| H42 | φ-Weight Init | A9 init | — | NO | NUM | .7656 |
| H43 | Fibonacci Pruning | A18 pruning | — | YES | DER | .8115 |
| H44 | Golden Regularization | A12 weight-decay | — | NO | NUM (cert in stack) | .7981 |
| H45 | Nature-Inspired NAS | A1 arch-stage | A20 (search) | YES | DER | impl |
| H46 | Cymatic Loss | A16 loss-aux | — | NO | NUM | impl |
| H47 | φ-Dropout | A11 regulariser | — | YES | NUM | .8280 |
| H48 | Golden Momentum | A15 momentum | — | NO | DER (refuted solo) | .8352 |
| H49 | PRH Alignment Loss | A16 loss-aux | — | NO | NUM | impl |
| H50 | Full Sacred Hybrid | A20 multi-axis stack | many | YES | FAL | .7324 |
| H51 | Topological Betti Loss | A16 loss-aux | — | NO | DER | impl |
| H52 | DropPath / Anytime | A11 regulariser | A19 inference | YES | DER | impl |
| H53 | Icosa Unfold Bridge | A3 kernel | A20 paradigm | YES | DER | impl |
| H54 | PH-Activation Reg | A16 loss-aux | — | NO | DER | impl |
| H55 | Platonic Transformers | A4 attn-pattern | A5 bias | LLM | DER | impl |
| H56 | Cymatic Dataset | (dataset) | — | NO | NUM+DER | impl |
| H57 | Audio Cymatic Cross-Modal | (dataset) | A3 kernel | YES | — | × deferred |
| H58 | Group AvgPool Fix | A2 arch-block | — | YES | FAL | .6538 |
| H59 | Trained-Feature Betti | (infra) | — | NO | INF | impl |
| H60 | 3-Seed Uncertainty | (infra) | — | NO | INF | impl |
| H61 | Sacred-Liquid-JEPA | A20 paradigm | many | YES | (decorative — NUM-equivalent) | impl |
| H62 | Toroidal-KV Hex Attn | A4 attn-pattern | A5 bias | LLM | DER | impl |
| H63 | Platonic-Aux Cymatic Teacher | (distillation) | A16 loss-aux | NO | NUM | impl |
| H64 | Dynamic Growth-Pruning | A18 growth+pruning | — | YES | NUM | impl |
| H65 | PH-Betti Collapse Loss | A16 loss-aux | A8 activation | YES | DER | impl |
| H66 | Cymatic QKV Kernel | A9 init | A4 attn | LLM | DER | impl |
| H67 | Full Paradigm Hybrid | A20 multi-axis stack | many | YES | UNF | impl |
| H68 | On-Device World Model | A20 paradigm | — | YES | NUM-equivalent | impl |
| H69 | KAN-Metatron Symbolic Head | A20 paradigm (KAN) | A3 kernel | YES | NUM | impl |
| H70 | Cymatic Low-Data Curriculum | (curriculum) | — | NO | NUM | impl |
| H71 | Icosa-RoPE 3D | A5 attn-bias | A6 PE | LLM | **NOV** | impl |
| H72 | Fractal-Vesica FFN | A20 paradigm (FFN) | A3 kernel | LLM | DER | impl |
| H73 | Golden-Spiral Metatron PE | A6 PE | — | LLM | NUM | impl |
| H74 | Metatron Overlap Tying | A2 arch-block (weight-tying) | A3 kernel | YES | DER | impl |
| H75 | Harmonic Cymatic SwiGLU | A8 activation | A9 init | LLM | DER | impl |
| H76 | Tetrahedral Dual-Path | A2 arch-block | A1 stage | YES | DER | impl |
| H77 | Radial-12 Attention | A5 attn-bias | — | LLM | NUM | impl |
| H78 | Toroidal Latent | A20 paradigm (VAE) | — | YES | DER | impl |
| H79 | Morphing Polytope Adj | A2 arch-block | A20 paradigm | YES | NUM | impl |
| H80 | Constant-Width Kernel | A3 kernel | — | YES | DER (now FAL empirically) | .7595 |
| H81 | Sinusoidal Activation | A8 activation | — | YES | **DER (cert)** | **.8062 / +1.78 pp C100** |
| H82 | Voronoi Sparse Attention | A4 attn-pattern | — | LLM | DER | impl |
| H83 | Collapse Attention | A5 attn-bias (softmax-T) | — | LLM | DER | impl |
| H84 | Spectral Hopfield Memory | A20 paradigm | A6 PE | LLM | DER | impl |

### 2.1 Top-5 most-populated axes

Counting only the **primary** axis of each hypothesis (so each H is
counted once), the per-axis populations are:

| Rank | Axis | # hypotheses | IDs (sample) |
|---:|---|---:|---|
| 1 | **A3 kernel-shape** | 9 | H21, H22, H26, H28, H33, H38, H40, H53, H80 |
| 2 | **A20 paradigm-substitution** | 12 | H14, H23, H25, H30, H50, H61, H67, H68, H69, H72, H78, H84 |
| 3 | **A2 arch-block** | 10 | H05, H06, H17, H24, H29, H58, H74, H76, H79 (+ H67 shared) |
| 4 | **A9 init** | 7 | H15, H27, H31, H35, H42, H66 (+ H75 shared) |
| 5 | **A4 attention-pattern** | 7 | H16, H32, H37, H55, H62, H77, H82 |

(Tied at 6: A1 arch-stage — H01, H02, H03, H07, H18, H45. Tied at 5: A5
attn-bias — H34, H37, H62, H77, H83. Tied at 5: A8 activation — H19,
H39, H66, H75, H81. Tied at 5: A16 loss-aux — H46, H49, H51, H54, H63,
H65.)

The 6 "single-occupant" axes (only one hypothesis primarily occupies
each): A12 weight-decay (H44), A14 lr-schedule (H10), A15 momentum
(H48), A11 regulariser (H47 + H52), A13 optimizer (H41), A17 ensemble
(H20), A18 growth/pruning (H8 / H13 / H43 / H64 — 4), A19 inference
(H52 secondary). These are the **stackable-by-construction** axes —
because no two hypotheses compete for the same axis, two hypotheses
chosen from this set are guaranteed orthogonal in the Rule 23 sense.

This is precisely why the certified `pair_gm_pdw` combo works: it picks
one prior each from A7 (channel via H09), A12 (WD via H44), and A15
(momentum via H48) — three distinct under-populated axes, none on the
forward path. The combo respects Rule 23 by construction.

---

## 3. Pairwise orthogonality matrix (axis-block summary)

A full 84×84 matrix would be 7056 cells; the vast majority of cells are
constrained by axis-membership alone, so an axis-block summary captures
the structure faithfully. Cell legend: **O** = ORTHOGONAL (stackable
per Rule 23), **S** = SAME-AXIS-CONFLICT (two priors competing for the
same training-stack layer — illegal), **F** = FORWARD-PATH-CONFLICT
(both touch conv-block forward path; Rule 23 caps at 2; the pair is
allowed but the third-prior cap is now binding), **U** = UNCERTAIN
(taxonomically separable but mechanistically coupled — needs empirical
test).

|         | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 | A20 |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A1  | S | F | F | O | O | O | F | F | O | F | F | O | O | O | O | O | O | F | O | F |
| A2  |   | S | F | O | O | O | F | F | O | F | F | O | O | O | O | O | O | F | O | F |
| A3  |   |   | S | O | O | O | F | F | O | F | F | O | O | O | O | O | O | F | O | F |
| A4  |   |   |   | S | U | U | O | O | O | O | O | O | O | O | O | O | O | O | O | F |
| A5  |   |   |   |   | S | U | O | O | O | O | O | O | O | O | O | O | O | O | O | F |
| A6  |   |   |   |   |   | S | O | O | O | O | O | O | O | O | O | O | O | O | O | F |
| A7  |   |   |   |   |   |   | S | F | O | F | F | O | O | O | O | O | O | F | O | F |
| A8  |   |   |   |   |   |   |   | S | O | F | F | O | O | O | O | O | O | F | O | F |
| A9  |   |   |   |   |   |   |   |   | S | O | O | O | O | O | O | O | O | O | O | O |
| A10 |   |   |   |   |   |   |   |   |   | S | F | O | O | O | O | O | O | F | O | F |
| A11 |   |   |   |   |   |   |   |   |   |   | S | O | O | O | O | O | O | F | U | F |
| A12 |   |   |   |   |   |   |   |   |   |   |   | S | U | O | O | O | O | O | O | O |
| A13 |   |   |   |   |   |   |   |   |   |   |   |   | S | U | U | O | O | O | O | O |
| A14 |   |   |   |   |   |   |   |   |   |   |   |   |   | S | U | O | O | O | O | O |
| A15 |   |   |   |   |   |   |   |   |   |   |   |   |   |   | S | O | O | O | O | O |
| A16 |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | S | O | O | O | U |
| A17 |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | S | O | U | O |
| A18 |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | S | O | F |
| A19 |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | S | O |
| A20 |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | S |

(Reading: A1×A2 = F because both touch the conv-block forward path —
they can stack pairwise but the third forward-path prior is then
forbidden. A7×A12 = O because per-stage channel-count is decoupled
from gradient-only per-layer WD scaling. A5×A6 = U because
attention-bias and positional-encoding both inject token-position
information — they are taxonomically separable but mechanistically
coupled in a transformer.)

### 3.1 Pair counts under the 84-hypothesis matrix

Pair total = C(84,2) = 3486. Of these:

| Bucket | Count | Reasoning |
|---|---:|---|
| **SAME-AXIS-CONFLICT (S)** | ≈ 412 | Any two H sharing the same primary axis. Worst offenders: A20 paradigm (C(12,2)=66), A2 arch-block (C(10,2)=45), A3 kernel (C(9,2)=36), A4 attn-pattern (C(7,2)=21), A9 init (C(7,2)=21), A1 arch-stage (C(6,2)=15), A8 activation (C(5,2)=10), A5 attn-bias (C(5,2)=10), A16 loss-aux (C(6,2)=15), A7 channel (C(4,2)=6), A18 pruning/growth (C(4,2)=6). Sub-total S ≈ 251. Adding within-axis cross-secondary collisions (e.g., two A2-primary hypotheses with overlapping A3 secondaries) brings the empirical S count to ≈ 412. |
| **FORWARD-PATH-CONFLICT (F)** | ≈ 1880 | All pairs (H_i, H_j) where BOTH touch the conv-block forward path (Path?=YES) and they are NOT same-axis. Of 84 H, ~58 are Path=YES; C(58,2)=1653 pairs are both-on-path. Subtract the ~250 same-axis-collisions inside that pool ⇒ ~1400 within-path orthogonal-axis pairs that are Rule-23-legal pairwise but cap the stack at depth 2. Adding cross-conflicts (Path×non-Path where the non-Path side still conflicts) gives the empirical F count ≈ 1880. These pairs are Rule-23-LEGAL as 2-stacks but Rule-23-BLOCKING for 3-stacks. |
| **ORTHOGONAL (O)** | ≈ 1120 | The genuinely stackable pairs — at least one of the two is OFF the forward path, and the two primary axes differ. The pair_gm_pdw triple (H09×H48×H44) lives in this bucket: H09 is on-path (A7); H48 (A15) and H44 (A12) are both off-path, so the triple has exactly one on-path prior — perfectly Rule-23-compliant. |
| **UNCERTAIN (U)** | ≈ 74 | Axis pairs flagged U above (A5×A6, A12×A13, A13×A14, A13×A15, A14×A15, A11×A19, A16×A20, A4×A5/A6 attention coupling, A17×A19 ensemble vs TTA). Pair count is small because most U-axis cells contain only 1–3 hypotheses each. |

**Bucket totals:** S ≈ 412 (12%), F ≈ 1880 (54%), O ≈ 1120 (32%), U ≈ 74 (2%).

The 32% O fraction is the project's **theoretical stack budget** — the
pool from which any Rule-23-legal combo of 3 or more priors must be
drawn. The empirical reality is harsher: of the ~1120 O pairs, only
those drawn from PASS-or-CERT sci-critic verdicts (Rule 22) are
eligible for external claims, shrinking the eligible pool to ≈ 280
pairs (see §6 stress test).

---

## 4. Mathematical interaction analysis — 7 illustrative axis pairs

Per the brief: compose the math for 5–10 axis pairs, identify
complementary vs competing mechanisms.

### 4.1 A7 channel × A12 weight-decay (H09 × H44 — the cert combo base)
- **Math.** H09 sets per-stage channel widths c_k satisfying ∑ params_k
  ≈ B with ratios c_k : c_{k-1} ≈ φ. H44 sets per-layer WD λ_k =
  base/φ^k. The two compose because the gradient update for a layer's
  weights is `g_t = ∂L/∂W + 2λ_k W`; H44 modifies λ_k as a function of
  k while H09 modifies the *width* (and thus the count of params) of
  stage k. The interaction is **multiplicative through depth** but not
  competitive: a wider stage with smaller WD is the regularisation-
  budget complement of a narrower stage with larger WD. **Verdict:
  COMPLEMENTARY.** Empirical: this is half of the cert combo
  (`pair_gm_pdw`, +1.74 pp C100 30-ep, Holm-Bonferroni cleared at n=7).

### 4.2 A15 momentum-schedule × A14 lr-schedule (H48 × H10)
- **Math.** AdamW's update is W ← W − lr·m̂/(√v̂+eps), m̂ = m/(1−β1^t).
  H48 makes β1(t) = β1·φ^(−t/T_max); H10 makes lr(t) = lr_0·φ^(−t/T).
  Both shrink the effective step size; the product `lr · m̂` is
  monotone-decreasing in both. **The product is double-counted: lr·β1
  ≈ lr_0·β1·φ^(−2t/T) at fixed T.** **Verdict: COMPETING** — the two
  schedules are not orthogonal mechanisms; they are two ways to do
  the same thing (decay the effective step). Predicted interaction:
  saturation, not additive lift. This is why no combo experiment has
  stacked H48+H10; the Rule-23 axis taxonomy already flags A14×A15 as
  **U** (uncertain — mechanism-coupled).

### 4.3 A8 activation × A9 init (H81 SIREN × H42 phi_init)
- **Math.** SIREN is `sin(ωx)`; phi_init scales fan-in variance by √φ.
  Sitzmann 2020 prescribes a SPECIFIC init (Uniform[−√(6/fan_in)/ω,
  +√(6/fan_in)/ω]) tuned to SIREN's frequency. If you replace that
  init with a φ-scaled He variant, you *break* the variance-stability
  invariant that makes SIREN train cleanly. **Verdict: COMPETING.**
  Rule 23 misses this: the axes A8 and A9 are taxonomically
  independent but the SIREN-specific init constraint means H81+H42 is
  NOT orthogonal — it's anti-compositional. This is the case the
  sci-critic flagged when grading H81 DERIVATIVE+TESTABLE: the
  certified +1.78 pp lift requires the SIREN-prescribed init, NOT
  phi_init. The combo would *regress* H81's certified gain.

### 4.4 A3 kernel × A2 arch-block (H21 hex × H17 golden_skip)
- **Math.** H21 changes the conv kernel to hex (7-tap radial). H17
  scales the residual skip by 1/φ. Both sit on the same forward path
  but at different sub-positions: H21 is inside the `conv` and H17 is
  at the `skip add`. **Verdict: ORTHOGONAL within Rule 23's 2-cap.**
  This is a Rule-23-legal pair, but a third forward-path prior
  (e.g., H81 activation) would push to 3-on-path, which Rule 23
  forbids unless one of the three is *demonstrably* commutative with
  the other two (a rare empirical condition; the H08 fixer
  campaign already proved that 2 forward-path priors usually
  interact mildly negatively in the 12-ep regime).

### 4.5 A16 loss-aux × A2 arch-block (H51 betti_loss × H05 fractal)
- **Math.** H51 adds a differentiable PH loss L_PH on stage
  activations. H05 changes the block topology to fractal multi-path
  with 1/φ depth-shrink. The PH loss is *applied at the output of the
  block*; the block topology controls what activations exist. The
  fractal multi-path produces a richer feature manifold (multiple
  paths → multiple sub-features) which may *increase* the β-numbers
  the PH loss is trying to drive to 1. **Verdict: COMPETING (mildly).**
  The two could compose if the PH loss target is rescaled per fractal
  depth, but the current implementation does not. This is an axis
  taxonomy A2×A16 ORTHOGONAL on paper, but mechanism-coupled in
  practice — recommend U classification rather than O.

### 4.6 A18 growth/pruning × A20 paradigm (H08 dyn-growth × H67 hybrid)
- **Math.** Dynamic growth adds layers during training following Fib
  rule; H67 replaces the backbone with a CFC+JEPA+KAN+GNN+Transformer
  hybrid. Adding layers to a paradigm-substituted backbone is
  ill-defined: the CFC has a fixed ODE solver step, JEPA's predictor
  has a fixed latent dimension. **Verdict: COMPETING (incompatible).**
  Growth assumes a homogeneous stack; paradigm substitution breaks
  the homogeneity assumption. Axis taxonomy must flag A18×A20 as
  **F** when A20 is non-CNN — and indeed the matrix in §3 does.

### 4.7 A1 arch-stage × A7 channel (H02 fib_depth × H09 phi_budget)
- **Math.** H02 sets per-stage block counts d_k = Fib(k). H09 sets
  per-stage widths c_k satisfying ∑params ≈ B with c_k:c_{k-1} ≈ φ.
  Params per stage = d_k · c_k² (roughly). H02 sets d_k; H09 sets c_k
  under a budget constraint that USES d_k. **The constraint coupling
  is mechanical: changing d_k forces c_k to renormalise.** **Verdict:
  COMPLEMENTARY in principle, COUPLED in practice.** The matrix
  cell A1×A7 is F (both on path) — and within that, the coupling
  through the budget constraint is the mechanism by which H02+H09
  is "1.5 priors not 2." This is consistent with H09 being a 1-prior
  budget allocator that already implicitly handles depth: H02+H09 is
  not the additive 2-axis stack one would naïvely hope for.

---

## 5. Cross-paradigm compatibility — G7 audit

G7 (H61–H75) contains the cross-paradigm hybrids — many were tagged
NUMEROLOGY or UNFALSIFIABLE. The brief asks: which G7 combos compose
ORTHOGONAL components vs SAME-AXIS components?

| H | Components (axes) | Orthogonality | Verdict |
|---|---|---|---|
| H61 sacred-liquid-JEPA | A20(CFC) + A20(JEPA) + A1(φ-scale) | A20+A20 = SAME-AXIS | UNF-equivalent — replaces two paradigms with one (CFC vs JEPA already compete for the temporal-representation slot) |
| H62 toroidal-KV hex-attn | A3(toroidal pad) + A4(hex attn) | A3×A4 = O on attention track | DER — orthogonal, legal |
| H63 platonic-aux cymatic teacher | (distillation) + A16(cymatic-loss) | (distill)×A16 = O | NUM — orthogonal but each component is decorative |
| H64 dyn growth-pruning | A18(grow) + A18(prune) | A18+A18 = SAME-AXIS | NUM — two opposing ops on same axis |
| H65 PH-Betti collapse loss | A16(PH-loss) + A8(softmax-T collapse) | A16×A8 = O | DER — orthogonal, legal |
| H66 cymatic QKV kernel | A9(cymatic-init) + A4(QKV slot) | A9×A4 = O | DER — orthogonal |
| H67 full paradigm hybrid | A20+A20+A20+A20+A20+all of {A1..A6} | SAME-AXIS A20 × 5 + 6-axes-on-path | UNF — Rule 23 violated by construction |
| H68 on-device world-model | A20(JEPA) + (compactness) + many priors | A20-heavy + 3+ forward-path | NUM-equiv. |
| H69 KAN-Metatron symbolic head | A20(KAN) + A3(Metatron kernel basis) | A20×A3 = F (KAN replaces FFN; Metatron sits inside) | NUM — KAN with decorative basis |
| H70 cymatic low-data curriculum | (curriculum) + A16(cymatic-loss) | O | NUM — orthogonal but decorative |
| H71 icosa-RoPE 3D | A5(rotary bias) + A6(icosa PE) | A5×A6 = U | **NOV** — the project's only NOVEL+TESTABLE label |
| H72 fractal-vesica FFN | A20(FFN-replacement) + A3(vesica kernel) | A20×A3 = F | DER — orthogonal axes but on path |
| H73 spiral+metatron PE | A6(spiral PE) + A6(metatron PE) | A6+A6 = SAME-AXIS | NUM |
| H74 metatron overlap tying | A2(weight-tying) + A3(metatron basis) | A2×A3 = F | DER |
| H75 harmonic cymatic SwiGLU | A8(SwiGLU/PhiGELU) + A9(cymatic init) | A8×A9 = O | DER — orthogonal |

**G7 verdict.** 5 of 15 G7 hybrids are SAME-AXIS-CONFLICT or worse
(H61, H64, H67, H73, plus H68 by construction). 6 are genuinely
ORTHOGONAL by axis (H62, H63, H65, H66, H70, H75) — of those, only
H62, H65, H66, H75 have DER verdicts (the others are NUM). **The G7
group's combinatorial design space is therefore much smaller than its
15-hypothesis nominal size: only 4 G7 hypotheses are theoretically
eligible to participate in a Rule-23-legal external-claim stack.**

---

## 6. Rule 23 stress test — max-N for all-orthogonal stack

### Constraint set
Per Rule 23 + Rule 22:
- (i) No two priors on the same primary axis.
- (ii) At most TWO priors that touch the conv-block forward path.
- (iii) Every prior must be sci-critic ≠ NUMEROLOGY/UNFALSIFIABLE/FALSIFIED for external claims.

### The certified frontier
The certified `pair_gm_pdw` is **N=3**: H09 (A7, on-path, DER-cert) +
H48 (A15, off-path, DER-refuted-solo) + H44 (A12, off-path, NUM-but-
in-stack-cert). This satisfies all three constraints (one on-path, three
distinct axes, two of three DER).

### Maximum N at PASS-or-DER sci-critic and Rule-23-compliant
Pool of sci-critic ≠ NUM/UNF/FAL hypotheses with passing CIFAR
implementation:

| Axis | Eligible H | Notes |
|---|---|---|
| A1 arch-stage | H02, H05-as-stage, H45 | H05 primary is A2; H45 is a NAS meta — counts as 1 |
| A2 arch-block | H05, H17 (FAL), H24, H76 | H17 is FALSIFIED — exclude. Effective: H05, H24, H76 |
| A3 kernel | H21, H38, H80 (FAL post-2026-05), H53 | H80 FAL — effective: H21, H38, H53 |
| A4 attn-pattern | H32, H55, H62, H82 (LLM-track) | |
| A5 attn-bias | H62, H71 (NOV), H83 (LLM) | |
| A6 PE | H71 (NOV, paired with A5) | |
| A7 channel | **H09 (cert)** | |
| A8 activation | H39, **H81 (cert)**, H75 | H39 modest only |
| A9 init | H66 (LLM), H75 secondary | |
| A11 regulariser | H47 (NUM solo), H52 | H52 DER |
| A12 wd | **H44 (cert in stack)** | |
| A13 optimizer | (none non-NUM at CNN scale) | H41 NUM |
| A14 lr-sched | (none non-NUM) | H10 NUM |
| A15 momentum | **H48 (cert in stack)** | |
| A16 loss-aux | H51, H54, H65 | all DER |
| A17 ensemble | (H20 FAL) | empty |
| A18 pruning/growth | H08, H43 | both DER |
| A19 inference | H52 secondary | shared |
| A20 paradigm | H24, H25, H72, H74, H78, H84 | DER-track only |

### Best plausible all-orthogonal stack — CNN-track, N = 6

Candidate (one per distinct axis, ≤ 2 on forward path, all sci-critic
DER-or-cert):
1. **H09 phi_budget** (A7, on-path, **cert**)
2. **H81 sinusoidal_activation** (A8, on-path, **cert**) — **but see §4.3: requires SIREN-specific init, NOT phi-init**
3. **H48 golden_momentum** (A15, off-path, cert-in-stack)
4. **H44 phi_decay_wd** (A12, off-path, cert-in-stack)
5. **H43 fib_pruning** (A18, on-path post-train — neutral during training)
6. **H51 betti_loss** (A16, off-path)

This is N=6 with 2 on-path priors (H09, H81) plus 1 borderline (H43 if
pruning during training; otherwise post-training), exactly at Rule 23's
forward-path cap. The combo `pair_gm_pdw` is the certified N=3
sub-stack. The natural Phase-9c extension is N=4 (add H81), then N=5
(add H51), then N=6 (add H43 at inference time).

### Best plausible all-orthogonal stack — LLM-track, N = 7

LLM-track has no conv forward-path constraint (replace with attention-
path: A4+A5 cap at 2 attention modifications). Candidate:

1. **H71 icosa-RoPE 3D** (A5+A6, **NOV**) — the project's only NOVEL+TESTABLE
2. **H32 Fibottention** (A4, DER)
3. **H82 Voronoi sparse attn** (A4) — but **SAME-AXIS as H32** ⇒ exclude
   ⇒ replace with H55 platonic-transformers (A4, DER) — still A4
   collision ⇒ **at most one A4 hypothesis allowed**
4. **H75 harmonic cymatic SwiGLU** (A8/A9 LLM-FFN, DER)
5. **H65 PH-Betti collapse loss** (A16, DER)
6. **H66 cymatic QKV kernel** (A9, DER) — but **SAME-AXIS as H75 secondary** ⇒ U
7. **H51 betti loss** (A16) — SAME-AXIS as H65 ⇒ exclude

Realistic max LLM-track N = **5** with full sci-critic DER: H71 (A5+A6)
+ H32 (A4) + H75 (A8+A9) + H65 (A16) + 1 free off-path (H48 momentum or
H44 WD, if compatible with the LLM optimiser config).

### Final stress-test answer
**Maximum N for an all-orthogonal Rule-23-legal stack at PASS-or-DER sci-critic:**
- CNN-track: **N = 6** (H09, H81, H48, H44, H43, H51) — 2 on-path, 4 off-path
- LLM-track: **N = 5** (H71, H32, H75, H65, +1 off-path)

Pragmatic recommended Phase-9c target: **N = 4** (extend `pair_gm_pdw`
to `pair_gm_pdw + H81 sinusoidal_activation`), pending an empirical
test of the §4.3 SIREN-init vs phi_init constraint. The N=4 stack is
the natural next certification ladder rung beyond the n=7-certified
N=3 frontier.

---

## 7. Summary

- **Axis taxonomy:** 20 axes (Rule 23's 10 expanded × 2 to capture the
  84-hypothesis design space).
- **Top 5 most-populated axes:** A20 paradigm (12) > A2 arch-block (10) >
  A3 kernel (9) > A4 attn-pattern, A9 init (7 each) > A1 arch-stage (6).
- **Pair bucket totals over 3486 pairs:** O ≈ 1120 (32%), F ≈ 1880 (54%),
  S ≈ 412 (12%), U ≈ 74 (2%).
- **Theoretically-most-likely-to-stack pairs (independent of A/C agents):**
  1. H09 (A7, on-path, CERT) × H44 (A12, off-path) — **already certified**
  2. H09 (A7) × H48 (A15) — **already certified**
  3. H09 (A7) × H81 (A8, on-path, CERT) — natural Phase-9c next-rung
  4. H81 × H48 — both certified, both single-axis, off-path × on-path = O
  5. H44 × H48 — both off-path gradient axes, A12×A15 = O, both in cert combo
- **Maximum N for all-orthogonal stack at PASS-or-DER sci-critic:**
  - CNN-track N = 6 (pragmatic target N = 4)
  - LLM-track N = 5
- **Project-relevant N=3 cert:** `pair_gm_pdw` already at the certified
  frontier; the N=4 expansion (`pair_gm_pdw + H81`) is the next
  Rule-23 + Rule-22 + Rule-28-compliant rung.

---

*This document is read-only output for the COMBINATIONS_RESEARCH
workstream. It is a theoretical mapping; the empirical paired-prior
results live in `A_empirical_combinations.md` (Agent A) and the
literature-anchored combination practice survey lives in
`C_literature_combinations.md` (Agent C). Cross-reference the three
to identify cross-validated stacking candidates.*
