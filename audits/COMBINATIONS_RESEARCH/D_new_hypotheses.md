# D — Synthesis + 8 Proposed Combo Hypotheses for Phase-9d

> Synthesis of the three deep-research deliverables (A empirical
> stackability, B theoretical orthogonality, C literature survey)
> plus 8 NEW combo-hypothesis proposals for the project's next
> experimental wave (Phase-9d). Authored 2026-05-30 by the synthesis +
> new-hypothesis-proposer agent. Cross-references: paper/FINDINGS.md,
> paper/STATISTICAL_TESTS.md (n=7 certification), hypotheses/IDEA_TABLE.md
> (84-hypothesis status table), CLAUDE.md Rules 22/23/28/35-37.

---

## 1. Synthesis of A + B + C

Three independent research lenses converge on the same core finding:
**productive compounding of nature-inspired priors requires orthogonal
axes, and the empirical ceiling at our compute budget is shallow (N≈3)**.
Agent A's experimental ledger documents exactly one super-additive pair
(`pair_gm_pdw`, +1.74 pp at n=7 CIFAR-100 30-ep, Holm-Bonferroni cleared)
and shows the additive ladder `combo2..combo8` saturates by N=3 then
inflects negative at N=5 (the φ-LR scheduler is the single most destructive
axis). Agent B's 20-axis taxonomy classifies 32 % of the 84-hypothesis
pair space as "ORTHOGONAL by construction" and shows the certified
`pair_gm_pdw` (H09 A7 × H48 A15 × H44 A12) is one of only ~280 sci-critic-
eligible orthogonal triples; the theoretical N-cap is **6 CNN-track / 5
LLM-track** at PASS-or-DER sci-critic. Agent C's 14-paper literature
survey shows mainstream image-classification recipes routinely stack
**8-12 orthogonal training tricks** (cosine, warmup, label-smoothing,
mixup, cutmix, RandAugment, stochastic-depth, EMA, AdamW, decreased-WD,
BCE) — a regime our project's default ResNet-20 + RandomCrop+HFlip
baseline misses by ~5-10 pp on ImageNet-scale tasks. The Bello-2021
"decrease WD when stacking regularisers" rule is the single most
important interaction in the entire literature.

**The key tension.** Mainstream stacks 11 tricks; we stack 1-3 nature-
inspired priors on top of an under-tuned 0-trick baseline. Two
interpretations: (i) **PESSIMISTIC** — our +1.74 pp lift over an
under-tuned baseline disappears once we stack the 11-trick mainstream
recipe (the BLOCKER #13 area-chair concern). (ii) **OPTIMISTIC** —
our priors live on orthogonal axes the mainstream hasn't visited; they
should stack productively on top of the 11-trick recipe (Rule 23
empirically validated by `pair_gm_pdw` on 3 distinct axes). Phase-9a
hill-climb partially refuted (i) by showing the leader-vs-baseline gap
persists in the tuned regime (+1.20 / +1.80 / +2.08 pp). **The bridge
this proposal builds: Phase-9d tests Class-I "bridge combos" that put
a certified nature-inspired prior on top of a 7-9-trick modern-recipe
baseline, directly measuring whether the +1.74 pp signal survives
when the baseline is genuinely competitive.**

---

## 2. Strategic framework for new hypotheses

Five complementary classes structure the Phase-9d proposal space:

- **Class I — Bridge combos.** Stack one or two certified nature-
  inspired priors on top of the mainstream 11-trick recipe. Tests
  whether priors deliver marginal lift beyond a properly-tuned modern
  baseline. This is the single most important test for an ICML
  submission — it directly answers the area-chair's "you just beat
  an under-tuned baseline" objection. Risk: priors might be redundant
  with mainstream regularisers (cf. mixup + label-smoothing
  redundancy from C §4.1).
- **Class II — Novelty-pocket combos.** Combine 2+ priors from C's
  14-18 no-mainstream-analog list. Tests the theoretically-orthogonal
  nature-only design space — H22 toroidal, H51 Betti loss, H46
  cymatic loss, H37 pentagonal attention, etc. These deliver maximum
  research novelty when they work; minimum risk of being a
  "φ-flavoured re-skin of X."
- **Class III — Saturation-extension combos.** N=4, N=5 orthogonal
  stacks of certified winners. The obvious next rung is `pair_gm_pdw
  + slot_act_sine` (architecture / momentum / WD / activation =
  N=4). B §6 identifies this as the natural pragmatic Phase-9c target;
  empirically A §2 shows N=4 was where the original additive ladder
  inflected negative, so this is a high-information test.
- **Class IV — Domain-stretch combos.** Combine a certified winner
  with a domain-stretching prior on the dataset where that prior
  should actually shine — e.g., H71 IcosaRoPE3D on rotated CIFAR,
  H21 hex on rotated/topology-respecting data. Two hypotheses tested
  for the price of one experimental dollar.
- **Class V — Cross-paradigm combos.** Bring CFC/JEPA/KAN/Mamba-style
  alternative compute into a certified base. Highest novelty,
  highest risk (cf. C `sg_full_fib` cautionary tale and H67
  UNFALSIFIABLE verdict).

---

## 3. The 8 proposals

Each proposal is a full design-doc-quality entry; the top-3 also have
draft hypothesis design docs at `hypotheses/g9_combo_winners/H_NN_*.md`.

---

### 3.1 `combo_bridge_modern` (H85) — Class I (BRIDGE)

**Composition.** Mainstream 7-trick recipe (AdamW + cosine + 3-ep
warmup + label-smoothing 0.1 + mixup α=0.2 + cutmix α=1.0 +
RandAugment N=2 M=14 + stochastic-depth 0.1) + H09 phi_budget +
H48 golden_momentum + H44 phi_decay_wd. **Axes:** mainstream
{A14 lr-sched, A1 stage-ratio, A11 reg, A11 reg, A11 reg, A11 reg,
A11 reg, A2 block} ⊕ nature {A7 channel, A15 momentum, A12 wd}.

**Orthogonality check.** B's 20-axis matrix: the 7 mainstream tricks
together touch 5 distinct axes (A14 + A11 + A2 + warmup-of-A14 +
WD-of-A12); the 3 nature priors touch A7, A15, A12. **The single
collision is A12 weight-decay** (mainstream WD vs H44 φ-decay-wd),
which is exactly the Bello-2021 interaction. Rule 23 forward-path
cap: mainstream uses 1 on-path (stochastic depth) + 1 reg
(mixup/cutmix are data-side, not on conv path); nature contributes
1 on-path (H09). Total on-path = 2 (compliant).

**Stackability evidence.** Empirical (A §4 LOO): pdw-marginal-in-
combo ≈ 0 → the H44 prior is replaceable by mainstream WD-tuning
without loss; gm-marginal-in-combo = +0.69 pp → genuinely additive.
Theoretical (B §1, §4.1): H09 × mainstream-cosine = O (different
axes). Literature (C §1.16 Bello): the 11-trick recipe needs WD
decreased by 2.5× when stacking regularisers — H44's per-layer φ-
decay schedule may serve as that decrease *automatically*.

**Predicted Δ vs baseline + Δ vs strongest existing winner.**
Baseline = current under-tuned ResNet-20 (CIFAR-100 30-ep mean
0.5612). Mainstream-recipe baseline (M11 ≈ 7-trick): predicted
**+3 to +5 pp over current baseline** (per C §5 mainstream
expectation at ResNet-20 scale, 30-ep budget). On top of M11:
adding `pair_gm_pdw` predicted **+0.5 to +1.5 pp** (smaller than
the +1.74 pp solo because mainstream stochastic-depth + cosine-WD-
schedule absorb part of H44+H48's contribution). Total vs current
baseline: **+3.5 to +6.5 pp**. Δ vs strongest existing winner
(`slot_act_sine` +1.78 pp): **+1.7 to +4.7 pp**. Honest caveat
per A §2's N=3 saturation finding: the +0.5-1.5 marginal on top
of M11 may be 0 (priors redundant with mainstream regularisers).

**Sci-critic anticipated verdict.** DERIVATIVE+TESTABLE. Each
component is published; the novelty is the *composition* (no paper
combines φ-budget + golden-momentum + φ-decay-WD on top of the
ConvNeXt-class 7-trick recipe), and the empirical test is sharp.

**Falsifier.** If `combo_bridge_modern` ≤ M11 baseline + 0.0 pp at
n=3 CIFAR-100 30-ep, the "nature-inspired priors stack on top of
mainstream" claim is refuted (the +1.74 pp solo gain was an
under-tuned-baseline artefact, BLOCKER #13 fully confirmed).

**GPU cost estimate.** CIFAR-100 30-ep × 7 seeds @ ~22 min/seed ≈
2.6 GPU-h. Adding RandAugment + mixup may add 10-15 % per epoch →
~3.0 GPU-h. Plus M11 baseline at n=7: +2.6 GPU-h. Total ≈ **6 GPU-h**.

**Class:** I (bridge).

**Risk.** Sub-additive — mixup + label-smoothing + decreased-WD may
already saturate the regularisation budget; pair_gm_pdw's
contribution could vanish (cf. C §4.4 Bello WD interaction). If
H81 SIREN is added, the SIREN-specific init constraint (B §4.3)
must be respected.

---

### 3.2 `combo_bridge_lite` (H86) — Class I (BRIDGE, half-stack)

**Composition.** Mainstream 4-trick (AdamW + cosine + label-smoothing
0.1 + mixup α=0.2) + `slot_act_sine` (H09 phi_budget + H81 SIREN).
**Axes:** {A14 + A11(LS) + A11(mixup) + A13 AdamW} ⊕ {A7 + A8}.

**Orthogonality check.** No axis collision; 4 distinct mainstream
axes + 2 nature axes. On-path priors: H09 (A7) + H81 (A8) +
mixup (data-side, not on conv path). Total on-path = 2 (cap
honored). The known SIREN-init constraint (B §4.3) is respected by
preserving the SIREN init prescription (NOT swapping to φ-init).

**Stackability evidence.** Empirical (A §6 SLOT ablations): the
activation slot is orthogonal to width-allocation by 4-of-5
mainstream test rows; sine-activation lifts +0.54 pp over pb-alone
on CIFAR-100. Theoretical (B §6): H81+H09 is in the N=6 max-N
stack. Literature (C §7.3): the mainstream 4-trick recipe is the
"safe minimum" addition; adding mixup alone is +1.5 pp on ImageNet
ResNet-50 (Zhang 2018).

**Predicted Δ.** Baseline = current under-tuned: predicted **+2.5 to
+4 pp** (4-trick recipe gives ~2-3 pp at 30-ep budget, slot_act_sine
gives +1.78 pp on its own). Δ vs slot_act_sine alone: **+1 to +2 pp**.
Δ vs M4-only baseline: **+1 to +2 pp** (the slot_act_sine signal
should mostly survive at 30 ep, modest erosion from mixup-induced
label smoothing).

**Sci-critic anticipated verdict.** DERIVATIVE+TESTABLE. The
component recipe is published; the composition with H81 SIREN is
not (no paper combines SIREN with mixup-cutmix-AdamW for CIFAR
classification).

**Falsifier.** If `combo_bridge_lite` ≤ M4-only baseline + 0.0 pp at
n=3 CIFAR-100 30-ep, H81 SIREN does not stack with the lite-modern
recipe — the certified +1.78 pp solo was under-tuned-baseline.

**GPU cost estimate.** CIFAR-100 30-ep × 7 seeds × 22 min ≈ 2.6
GPU-h. With mixup CPU overhead: ~3 GPU-h. **3 GPU-h.**

**Class:** I (bridge, lite).

**Risk.** Mixup's intrinsic label-smoothing effect may compete with
SIREN's high-frequency expressiveness; mixup smooths labels +
SIREN sharpens features → unclear interaction. Lower stack height
than 3.1 (smaller WD interaction risk).

---

### 3.3 `combo_n4_pair_slot` (H87) — Class III (SATURATION EXT.) ★ TOP-3

**Composition.** H09 phi_budget + H48 golden_momentum + H44
phi_decay_wd + H81 sinusoidal_activation. **Axes:** A7 + A15 + A12
+ A8. Four certified winners on four distinct axes.

**Orthogonality check.** B §6 explicitly nominates this as the
project's natural Phase-9c next-rung. Axes A7, A15, A12, A8 are all
distinct; on-path priors are H09 (A7) + H81 (A8) = 2 (Rule 23 cap
honored). The B §4.3 SIREN-init caveat applies — H81's published
init (Sitzmann 2020) must NOT be swapped to phi-init; the
implementation already preserves SIREN init via the swap helper
(verified A §6).

**Stackability evidence.** Empirical (A §8): both `pair_gm_pdw` and
`slot_act_sine` are independently certified at n=7 on CIFAR-100;
their predicted-additive sum is +3.52 pp but A's N=3 saturation
finding caps realistic compound at ~+2.0-2.5 pp. Theoretical
(B §7): max-N N=6 stack includes both. Literature (C §3): mainstream
saturation point at this stack-depth is 8-12 tricks; an N=4 stack
of nature-inspired priors is below the saturation curve.

**Predicted Δ.** Δ vs current baseline: **+1.5 to +2.5 pp** (NOT the
naive sum of +1.74 + +1.78 = +3.52; A §2's empirical finding shows
the N=3→N=4 marginal first goes mildly negative). Δ vs strongest
existing winner (`slot_act_sine` +1.78 pp): **−0.3 to +0.7 pp** —
this is the binary question this experiment answers: do the two
certified winners stack productively or saturate flat?

**Sci-critic anticipated verdict.** DERIVATIVE+TESTABLE.
Components individually published; the N=4 composition is novel.

**Falsifier.** If `combo_n4_pair_slot` < max(`pair_gm_pdw`,
`slot_act_sine`) − 0.3 pp at n=7 CIFAR-100 30-ep, the N=3
saturation finding extends to N=4 (two certified winners do NOT
stack productively above the N=3 ceiling).

**GPU cost estimate.** CIFAR-100 30-ep × 7 seeds × 22 min ≈ 2.6
GPU-h. **2.6 GPU-h.**

**Class:** III (saturation extension).

**Risk.** A's N=3 saturation finding (combo4 was the first
negative-marginal row in the additive ladder) suggests this combo
could land *flat* or even mildly negative vs the better solo
winner. This is itself a high-information empirical answer — it
would close the "more orthogonal axes ⇒ more lift" question.

---

### 3.4 `combo_novelty_betti_torus` (H88) — Class II (NOVELTY POCKET) ★ TOP-3

**Composition.** H09 phi_budget + H22 toroidal closure + H51
topological Betti loss. **Axes:** A7 channel + A3 kernel
(toroidal-padding-of-conv) + A16 loss-aux.

**Orthogonality check.** B's matrix: A7×A3 = F (both on path —
allowed pairwise, third on-path forbidden); A7×A16, A3×A16 = O.
On-path priors: H09 (A7) + H22 (A3) = 2 (cap honored). Loss-aux
H51 is off-path. C §6 novelty pocket: H22 and H51 BOTH have no
mainstream analog. The combo therefore tests two novelty-pocket
priors stacked on the certified-winner H09 base.

**Stackability evidence.** Empirical: H22 solo is −6.73 pp on
CIFAR-10 (negative because CIFAR doesn't wrap and the base scaffold
was NaturePrior, not ResNet-20). H51 has no CIFAR row yet (impl
only). The bet: on the H09 phi_budget base scaffold, H22 may be
near-neutral (the scaffold is correct now), and H51 adds a
geometric-regularisation signal that's complementary to H09's
allocator. Theoretical (B §4.5): A16×A2 = U (mildly coupled in
practice); H22 is A3, not A2, so the coupling is weaker. Literature
(C §6 novelty pocket): no mainstream recipe combines toroidal
padding with persistent-homology regularisation.

**Predicted Δ.** Δ vs current baseline: **+0.5 to +1.5 pp** (H09
contributes its +1.24 pp solo; H22 modulates around 0 pp on the
correct scaffold; H51 contributes 0 to +0.5 pp from PH regulariser).
Δ vs `sg_only_phi_budget` (+1.24 pp): **−0.5 to +0.3 pp** — this
experiment's primary value is testing whether the novelty pocket
priors deliver real signal or are decorative.

**Sci-critic anticipated verdict.** DERIVATIVE+TESTABLE for H51
(Betti loss is a published TDL technique, Hofer 2017); H22 likely
NUMEROLOGY individually (C9-flagged) but in compound with H51 the
combination tests a published topological-regularisation framing.

**Falsifier.** If `combo_novelty_betti_torus` ≤ `sg_only_phi_budget`
− 0.3 pp at n=3 CIFAR-100 30-ep, the novelty-pocket priors (H22,
H51) do NOT add value on the certified H09 base.

**GPU cost estimate.** CIFAR-100 30-ep × 7 seeds × ~25 min (Betti
loss adds ~15 % overhead via differentiable PH library) ≈ **3 GPU-h**.

**Class:** II (novelty pocket).

**Risk.** Two unproven priors — possible the Δ is null or negative;
high research value if positive (first toroidal × Betti × budget
result). H22 mechanism-coupling with H09 (B §4.7 stage×channel
coupling) is a known concern.

---

### 3.5 `combo_novelty_cymatic_pentagonal` (H89) — Class II (NOVELTY POCKET)

**Composition.** H09 phi_budget + H46 cymatic loss + H37 pentagonal
φ-attention (CNN-track: 5-fold rotational symmetry constraint on
the final conv stage). **Axes:** A7 + A16 + A4 (treating pentagonal-
sym constraint as an attention-like radial bias for CNN; B
catalog says A4 for LLM, here we use a CNN-track adaptation).

**Orthogonality check.** A7 × A16 = O; A7 × A4 = O (per B's table
A4 row, off-path on CNN track when applied at the final feature-
aggregation stage). On-path: H09 + (pentagonal-sym in final stage
counts as light on-path) = 1.5; well within Rule 23 cap.

**Stackability evidence.** Empirical: neither H46 nor H37 has a
CIFAR row yet — both are `✓ impl` only. Theoretical (B §1): A7
(channel allocator) × A16 (frequency-domain loss) × A4 (radial-
symmetric attention) are 3 distinct under-populated axes.
Literature (C §6 novelty pocket): H46 cymatic loss + H37 pentagonal
attention have NO mainstream analogs.

**Predicted Δ.** Δ vs current baseline: **+0.3 to +1.5 pp** (H09's
+1.24 pp + 0 to +0.3 from each novelty prior; risk of mild
negative interaction from the late-stage symmetry constraint
limiting representation). Δ vs `sg_only_phi_budget`: **−0.7 to +0.3 pp**.

**Sci-critic anticipated verdict.** NUMEROLOGY for the cymatic-loss
component (frequency-domain MSE vs Chladni modes is hard to
falsify mechanistically); DERIVATIVE for pentagonal-sym (Cohen 2019
Platonic-equivariant CNNs is the literature anchor). Combined:
NUMEROLOGY-leaning — caveat that Rule 22 prevents external claims
unless the *combination* delivers measurable lift.

**Falsifier.** If Δ ≤ `sg_only_phi_budget` − 0.3 pp at n=3 CIFAR-100,
both novelty priors are decorative additions.

**GPU cost estimate.** CIFAR-100 30-ep × 3 seeds × ~24 min (cymatic
FFT loss adds ~10 % per epoch) ≈ **1.2 GPU-h**.

**Class:** II (novelty pocket).

**Risk.** Two NUMEROLOGY-leaning priors; this is the "lottery
ticket" of the proposed wave. Approach with single-seed smoke
before committing to n=7.

---

### 3.6 `combo_n5_pair_slot_betti` (H90) — Class III (SATURATION EXT., DEEP)

**Composition.** H09 + H48 + H44 + H81 + H51 Betti loss. **Axes:**
A7 + A15 + A12 + A8 + A16. Five distinct axes.

**Orthogonality check.** All 5 axes distinct. On-path: H09 (A7)
+ H81 (A8) = 2 (cap honored). H51 (A16) is off-path. B §6 N=6
max-N stack includes all 5 priors.

**Stackability evidence.** Built on the (still-untested but
predicted-positive) N=4 base of proposal 3.3 by adding the (still-
untested) Betti loss from proposal 3.4. **Conditional on 3.3 and
3.4 each succeeding**, this tests N=5.

**Predicted Δ.** Δ vs current baseline: **+1 to +2.5 pp** (extrapolation
from N=4 prediction + small Betti contribution; A §2 saturation
suggests this caps at ~+2 pp). Δ vs N=4 (proposal 3.3): **−0.5
to +0.5 pp** — empirical answer to the "how deep can we stack"
question.

**Sci-critic anticipated verdict.** DERIVATIVE+TESTABLE if 3.3
and 3.4 succeed; NUMEROLOGY-by-association if either fails.

**Falsifier.** If Δ < N=4 base − 0.3 pp at n=3 CIFAR-100, the
saturation point is N=4 (not N=5).

**GPU cost estimate.** CIFAR-100 30-ep × 3 seeds × ~25 min (Betti
overhead) ≈ **1.25 GPU-h** at n=3 screening; **3 GPU-h** at n=7.

**Class:** III (saturation extension, deep).

**Risk.** **CONDITIONAL on proposals 3.3 + 3.4 succeeding.** Run
only after both Class-I/III base experiments land. Otherwise this
is a research wager on the weaker of the two component-trees.

---

### 3.7 `combo_domain_icosa_rotation` (H91) — Class IV (DOMAIN STRETCH) ★ TOP-3

**Composition.** H09 phi_budget + H24 icosahedral equivariant
conv + H71 IcosaRoPE3D positional encoding — evaluated on the
ROTATED-CIFAR-100 dataset (random 360° rotations) at 30 ep.
**Axes:** A7 + A2 (block-internal C5/icosa lift) + A5 (positional
bias for 3D rotation symmetry).

**Orthogonality check.** A7×A2 = F (both on path — paired allowed);
A7×A5 = O; A2×A5 = O on this dataset. On-path: H09 + H24 = 2
(cap honored). H71 is dataset-side positional encoding (off-path
during conv). This is the **only domain-stretch combo on the
project's only NOVEL+TESTABLE prior (H71, per B §5)** combined
with the certified H09 winner.

**Stackability evidence.** Empirical: H24 has no CIFAR row yet
(impl only); H71 is the project's only NOVEL+TESTABLE sci-verdict.
The bet: on rotated-CIFAR-100, the rotational-invariance prior of
H24 (icosa-equivariant conv) plus H71 (3D-icosa-RoPE positional
encoding) should outperform any non-equivariant baseline INCLUDING
the certified `pair_gm_pdw` (which has no rotation prior). The
combination tests two priors at the dataset where they should win.
Literature (C §6 novelty pocket): full-icosa equivariance is a
published GDL angle (Cohen 2019), and H71 is the project's
NOVEL+TESTABLE outlier.

**Predicted Δ.** Δ vs `baseline_resnet20` on ROTATED-CIFAR-100:
**+5 to +15 pp** (massive lift expected because the baseline has
no rotation prior; published Cohen 2019 results show 5-20 pp on
rotated benchmarks). Δ vs `pair_gm_pdw` on rotated-CIFAR-100:
**+3 to +10 pp** (pair_gm_pdw has no rotation prior either).
**This is the highest predicted Δ in the entire Phase-9d wave
because it tests on the dataset where the prior should shine.**

**Sci-critic anticipated verdict.** NOVEL+TESTABLE inherited from
H71 (B §5 verdict); H24 DER; H09 cert. Combined: NOVEL+TESTABLE.

**Falsifier.** If Δ ≤ `pair_gm_pdw` on rotated-CIFAR-100 + 0.0 pp
at n=3, icosa-equivariance does NOT transfer to rotated-CIFAR-100.
This is a strong falsifier — rotated-CIFAR is the canonical
rotation-equivariance benchmark.

**GPU cost estimate.** Rotated-CIFAR-100 generation: ~10 min
one-time. Training: 30 ep × 7 seeds × ~30 min (icosa-conv is
~30 % slower) ≈ **3.5 GPU-h**. Plus baseline + `pair_gm_pdw` on
rotated-CIFAR-100 at n=3 each: ≈ **2 GPU-h**. Total ≈ **5.5
GPU-h**.

**Class:** IV (domain stretch).

**Risk.** Rotated-CIFAR-100 is not in the canonical Phase-1-5
pipeline; data-loader work required. H24 icosa-conv has no
existing CIFAR smoke row — possible the implementation has bugs
revealed only at training. Verification: launch single-seed
smoke first.

---

### 3.8 `combo_paradigm_kan_phi` (H92) — Class V (CROSS-PARADIGM, SPECULATIVE)

**Composition.** H09 phi_budget + H69 KAN-Metatron symbolic head
(replace the final classification head with a KAN spline layer
over Metatron-routed features) + H44 phi_decay_wd. **Axes:** A7
+ A20 (paradigm replacement of head) + A12.

**Orthogonality check.** A7 × A20 = F (both on path, since head
replacement IS the forward path); A7 × A12 = O; A20 × A12 = O.
On-path: H09 + KAN head = 2 (cap honored). A20-axis is the
"wholesale module swap" axis — H69 replaces the FC head with a
KAN spline head, while keeping the rest of the network
H09-allocated.

**Stackability evidence.** Empirical: H69 has no CIFAR row (impl
only); its sci-critic verdict was NUMEROLOGY (B §5). The bet:
NUMEROLOGY was for the standalone H69 with no anchor; combining
with the certified H09 base and the gradient-side H44 may
elevate the configuration to DERIVATIVE because the components
that empirically WORK (H09, H44) carry the model. Literature
(C §6 novelty pocket): KAN heads exist (Liu 2024) but not with
Metatron-tied weights and not on top of φ-budget.

**Predicted Δ.** Δ vs current baseline: **−1 to +1 pp**. KAN
heads on CIFAR are mostly neutral to mildly negative on small
data per the 2024 KAN literature (Liu 2024 finds KAN benefit only
at very high data or symbolic-regression tasks). This is the
*explicit* high-risk-low-expected-value proposal in the wave.

**Sci-critic anticipated verdict.** NUMEROLOGY-leaning (H69 was
NUMEROLOGY solo); DERIVATIVE if the KAN head delivers measurable
fit improvement.

**Falsifier.** If Δ < `sg_only_phi_budget` − 0.5 pp at n=3, the
KAN-Metatron head is decorative on top of phi_budget.

**GPU cost estimate.** CIFAR-100 30-ep × 3 seeds × ~28 min (KAN
spline head adds ~20 % per-step overhead) ≈ **1.4 GPU-h**.

**Class:** V (cross-paradigm).

**Risk.** Highest of the 8. Both H69 NUMEROLOGY individually AND
A20 paradigm-substitution caveat (B §4.6) — H08 dynamic-growth
× H67 hybrid was flagged COMPETING (incompatible). KAN heads on
CIFAR are mostly neutral per Liu 2024.

---

## 4. Priority ranking + execution plan

| Rank | Tag | Class | Predicted Δ vs baseline | GPU-h | Δ-per-GPU-h | Priority |
|---|---|---|---|---|---|---|
| 1 | `combo_n4_pair_slot` (3.3) | III | +1.5 to +2.5 pp | 2.6 | 0.6-1.0 pp/h | **TOP-3 ★** |
| 2 | `combo_domain_icosa_rotation` (3.7) | IV | +5 to +15 pp (on rotated) | 5.5 | 0.9-2.7 pp/h | **TOP-3 ★** |
| 3 | `combo_novelty_betti_torus` (3.4) | II | +0.5 to +1.5 pp | 3.0 | 0.17-0.5 pp/h | **TOP-3 ★** |
| 4 | `combo_bridge_modern` (3.1) | I | +3.5 to +6.5 pp | 6.0 | 0.58-1.08 pp/h | Wave 2 |
| 5 | `combo_bridge_lite` (3.2) | I | +2.5 to +4 pp | 3.0 | 0.83-1.33 pp/h | Wave 2 |
| 6 | `combo_n5_pair_slot_betti` (3.6) | III | +1 to +2.5 pp | 3.0 | 0.33-0.83 pp/h | Wave 3 (conditional) |
| 7 | `combo_novelty_cymatic_pentagonal` (3.5) | II | +0.3 to +1.5 pp | 1.2 | 0.25-1.25 pp/h | Wave 3 |
| 8 | `combo_paradigm_kan_phi` (3.8) | V | −1 to +1 pp | 1.4 | −0.7 to +0.7 pp/h | Future work |

**Priority justification.**
- **Rank 1 (`combo_n4_pair_slot`).** Cheapest, most-informative
  test of saturation hypothesis. Either confirms A's N=3
  saturation extends to N=4 (closes the question) or breaks it
  (new strongest winner). Either outcome is publishable.
- **Rank 2 (`combo_domain_icosa_rotation`).** Highest predicted Δ
  in the wave (+5 to +15 pp), tests the project's only NOVEL+
  TESTABLE prior (H71) on its natural-fit dataset. Even at 5.5
  GPU-h this is the highest expected-value-per-hour after
  weighting by novelty impact.
- **Rank 3 (`combo_novelty_betti_torus`).** Tests three novelty-
  pocket priors at once on a low-cost design. Modest predicted
  Δ but exclusive research-novelty payload.

**Phase-9d execution plan.**

| Wave | Tags | Total GPU-h | Trigger |
|---|---|---|---|
| **9d-1 (immediate)** | 3.3, 3.7, 3.4 | 11.1 | Phase-9c complete |
| **9d-2 (conditional)** | 3.1, 3.2 (run together to share M7 baseline) | 9 | At least 2 of wave-1 land Δ > +0.3 pp |
| **9d-3 (future)** | 3.6, 3.5 | 4.2 | Wave-1's 3.3 + 3.4 land positive |
| **future work** | 3.8 | 1.4 | Conditional on H69 hill-climb signal |
| **TOTAL (full wave)** | 8 hypotheses | **25.7 GPU-h** | |

Wave 9d-1 alone is the **minimum viable Phase-9d** — 11.1 GPU-h
delivers the three highest-information experiments.

---

## 5. Pre-registration footprint (Rule 36)

For each proposal, the headline cell-config preregistered before
launch (committed to git with hash). The pre-registration is a
contract per Rule 36: the SCREENING/EVALUATION classification is
fixed in writing before seeds are drawn.

| Tag | Dataset | Epochs | LR | WD | BS | Optimizer | Seeds | Tier (pre-reg) |
|---|---|---|---|---|---|---|---|---|
| 3.3 `combo_n4_pair_slot` | CIFAR-100 | 30 | 3e-3 | 5e-4 | 128 | AdamW | 0..6 | **EVALUATION** (n=7) |
| 3.7 `combo_domain_icosa_rotation` | rotated-CIFAR-100 (±180°) | 30 | 3e-3 | 5e-4 | 128 | AdamW | 0..6 | **EVALUATION** (n=7) on rot dataset |
| 3.4 `combo_novelty_betti_torus` | CIFAR-100 | 30 | 3e-3 | 5e-4 | 128 | AdamW | 0..2 | **SCREENING** (n=3) — promote on success |
| 3.1 `combo_bridge_modern` | CIFAR-100 | 30 | 1e-3 | 5e-4 (Bello-tuned) | 256 | AdamW | 0..6 | **EVALUATION** (n=7) |
| 3.2 `combo_bridge_lite` | CIFAR-100 | 30 | 1e-3 | 5e-4 | 256 | AdamW | 0..2 | **SCREENING** (n=3) |
| 3.6 `combo_n5_pair_slot_betti` | CIFAR-100 | 30 | 3e-3 | 5e-4 | 128 | AdamW | 0..2 | **SCREENING** (n=3) |
| 3.5 `combo_novelty_cymatic_pentagonal` | CIFAR-100 | 30 | 3e-3 | 5e-4 | 128 | AdamW | 0..2 | **SCREENING** (n=3) |
| 3.8 `combo_paradigm_kan_phi` | CIFAR-100 | 30 | 3e-3 | 5e-4 | 128 | AdamW | 0..2 | **SCREENING** (n=3) |

The cell-config for 3.1/3.2 is at the Bello-2021 mainstream tuning
(LR 1e-3, BS 256, WD calibrated to regulariser stack) — different
from the rest of the wave (3e-3/128) because the recipe demands it.

Composite formula fingerprint preserved:
`d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`
(Rule 2). Pre-registration commit hash to be added on launch.

---

## 6. Anti-recommendations (DO NOT TEST)

Five combos that look tempting but should be rejected on A/B/C-
sourced grounds:

1. **`combo_plr_anything`** — any combo containing H10 φ-decay LR
   scheduler. Empirical: every H10-containing pair in A §3 is
   ANTAGONISTIC (−3 to −5 pp). Theoretical: B §4.2 shows H10
   competes mechanically with H48 (both decay the effective step,
   double-counting). Literature: C §5 mainstream uses smooth cosine,
   never φ^{-k} decay. **Reject on 3-of-3 sources.**
2. **`combo_full_sg_revival`** — the historical `sg_full_fib`
   (H21+H24+H05+H22+H35+H17) at any scale or seed count. Empirical:
   A §5 documents the −11.54 pp catastrophe explicitly; six priors
   on the same conv-block forward path is Rule 23 violation by
   construction. **Reject on Rule 23.**
3. **`combo_h81_with_phi_init`** — replacing the SIREN-prescribed
   init in `slot_act_sine` with H42 φ-init. Theoretical (B §4.3):
   the A8 SIREN activation has a published, frequency-tuned init
   constraint; phi-init violates He-variance preservation and
   would *regress* the certified +1.78 pp gain. Empirical: A §6's
   `slot_init_phi` confirms phi-init causes −2.10 pp on the
   phi_budget base. **Reject on cross-source confirmation.**
4. **`combo_h67_hybrid_revival`** — H67 Full Paradigm Hybrid
   (CFC + JEPA + KAN + GNN + Transformer fused) at any seed
   count. Sci-critic verdict: UNFALSIFIABLE by construction (B §5).
   Rule 22 explicitly forbids external claims from UNFALSIFIABLE
   priors. **Reject on Rule 22.**
5. **`combo_h24_h25_h30_platonic_stack`** — three A20 paradigm
   priors (icosa, dodeca, Platonic-GNN) stacked. B §1 lists A20 as
   the most-populated axis (12 hypotheses); stacking 3 priors
   from the same primary axis is a SAME-AXIS-CONFLICT (B §3.1, S
   bucket). **Reject on Rule 23 same-axis discipline.**

---

## 7. Bonus — Top-3 design docs

Full committee-grade design docs drafted for the top-3 proposals:

- [`hypotheses/g9_combo_winners/H87_combo_n4_pair_slot.md`](../../hypotheses/g9_combo_winners/H87_combo_n4_pair_slot.md)
- [`hypotheses/g9_combo_winners/H91_combo_domain_icosa_rotation.md`](../../hypotheses/g9_combo_winners/H91_combo_domain_icosa_rotation.md)
- [`hypotheses/g9_combo_winners/H88_combo_novelty_betti_torus.md`](../../hypotheses/g9_combo_winners/H88_combo_novelty_betti_torus.md)

Each carries the 11 mandatory sections per `hypotheses/_TEMPLATE.md`:
motivation ≥100w, formal hypothesis ≥50w, numeric falsifier, multi-
paper citations (Rule 4 format), mechanism (CNN-track + LLM-track
where applicable), predicted Δ table, 3-part experimental protocol,
cross-references to A/B/C + parent winners, ≥4 Committee Q&A,
verification checklist, status journal.

---

*Authored 2026-05-30 by synthesis + new-hypothesis-proposer agent.
Read-only across the codebase apart from this file +
`hypotheses/g9_combo_winners/H{87,88,91}_*.md`. Cross-references:
[A](A_empirical_stackability.md), [B](B_theoretical_orthogonality.md),
[C](C_literature_survey.md), [FINDINGS](../../paper/FINDINGS.md),
[STATISTICAL_TESTS](../../paper/STATISTICAL_TESTS.md),
[IDEA_TABLE](../../hypotheses/IDEA_TABLE.md).*
