# PIPELINE COMPLETE — 2026-06-01

> **Status: GPU PIPELINE CLOSED.** Phases 9a, 9b, 9c, 9d, 9e, 9f, 9g are all complete. Phase-9h items are filed but unlaunched (workshop / D&B-track submission is unblocked; main-track conditional on Phase-9h closure of Control 3a + Control 2 tanh extension).

This marker documents the final-state defensibility envelope of the `nature_inspired_networks` repository as of 2026-06-01 17:02 (PT) when the last Phase-9g control sweep finished. Per CLAUDE.md Rule 11, every prior Phase-9 sub-phase's commit history is preserved; this file is the **closure narrative**, not a re-litigation of prior phases.

---

## 1. Phase-9 sub-phase roll-up

| sub-phase | start | end | scope | output |
|---|---|---|---|---|
| **9a** | 2026-05-30 AM | 2026-05-30 PM | Per-hypothesis hill-climb (cube: lr × wd × bs × optimizer; budget 25) on baseline + 3 leaders. | `ideas/{00,09,91,92}/hillclimb_results.json`; per-hypothesis `ideas/<NN>/dashboard/index.html`; [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §7. |
| **9b** | 2026-05-30 PM | 2026-05-31 AM | n=62 audit-calibration extension on third-party code subsample (R2 Q3 / AC item 3). | [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §8 + §11; [`audits/AUDIT_CALIBRATION_THIRD_PARTY.md`](AUDIT_CALIBRATION_THIRD_PARTY.md). |
| **9c** | 2026-05-31 AM | 2026-05-31 PM | Magnitude-test addendum (permutation + paired-t for R1 BLOCKER #3 — Wilcoxon-at-floor concern). | [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §9. |
| **9d** | 2026-05-31 PM | 2026-06-01 AM | R-D synthesis combo hypothesis pre-registration (H87 / H88 / H91 design docs landed in `hypotheses/g9_combo_winners/`). | 3 hypothesis docs + design templates. |
| **9e** | 2026-06-01 AM | 2026-06-01 morning | Wave-1 of combo sweep: n=3 each on the three R-D-synthesis combos. | [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §12; cells under `experiments/{cifar100,rotated_cifar100}/combo_*_seed*/`. |
| **9f** | 2026-06-01 morning | 2026-06-01 afternoon | n=7 iso-tuned (lr=3e-3 wd=5e-4 bs=128 AdamW) extension of baseline + 3 leaders at the hill-climbed cell (R2 BLOCKER #13 closure path). | [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §10; Phase-5 ordinal gate FAIL for all three winners at iso-tuned n=7. |
| **9g** | 2026-06-01 07:29 PT | 2026-06-01 17:02 PT | Reviewer-flagged controls (C1 non-φ 3-axis / C2 activation ablation / C3a tuned baseline hill-climb / C4 H71 IcosaRoPE3D ablation). | [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §13; cells under `experiments/{cifar100,rotated_cifar10}/*`; this closure marker. |

**Total Phase-9 elapsed GPU-time:** ~35–40 h on the single RTX 4090 Laptop, distributed across the 7 sub-phases above (≈ 9.5 h for Phase-9g alone; the rest accumulated across 9a–9f). The auto-checkpoint loop (Rule 20) preserved every intermediate output across the campaign.

---

## 2. Reviewer / AC / REBUTTAL items addressed (closure tally)

### ICML R1 (Reviewer 1) blockers

- **R1 BLOCKER #1 — POSI / family-size correction.** ADDRESSED — [`PAPER.md`](../PAPER.md) §5.5 POSI paragraph + [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §3 (Holm-Bonferroni k=3 confirmatory family clear) + §9 (paired-t magnitude diagnostic clears the POSI k=49 bound for 2/3 winners).
- **R1 BLOCKER #3 — Wilcoxon-at-floor magnitude gap.** ADDRESSED — [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §9 (paired-t + permutation magnitude test; t = 5.43 to 9.06; p_one = 5×10⁻⁵ to 8×10⁻⁴).

### ICML R2 (Reviewer 2) blockers and questions

- **R2 BLOCKER #13 — mixed-bs / iso-tuned baseline concern.** PARTIALLY VALIDATED — Phase-9f n=7 iso-tuned extension shows Δ-shrinkage from default-config +1.24/+1.74/+1.78 pp to +0.66/+0.79/+0.25 pp paired; Phase-5 ordinal gate FAILS at iso-tuned n=7 for all three winners (max iso-tuned baseline = 0.6075). Phase-9g Control 3a's tuned-baseline n=1 result (0.5984) further reinforces this picture. [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §10 (Phase-9f) + §13.3 (Control 3a).
- **R2 Q1 — non-φ 3-axis control (Control 1) + SIREN-without-φ confound (Control 2) + tuned RegNetX comparator (Control 3b).** Controls 1, 2, 3a CLOSED; Control 3b refused by launch allowlist and re-filed as Phase-9h. Controls 1+2 honest verdicts: **φ-specific story PARTIALLY REFUTED** (3-axis structure carries ~61 % of `pair_gm_pdw`'s lift; φ residual not statistically certified at n=3) and **SIREN-specific story REFUTED** (`slot_act_tanh` beats `slot_act_sine` by +0.48 pp paired with 3/3 positive deltas). [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §13.1 + §13.2.
- **R2 Q2 — batch-size confound between hill-climbed baseline (bs=256) and leaders (bs=128).** ADDRESSED via Phase-9f n=7 iso-tuned (both arms at bs=128 lr=3e-3 wd=5e-4 AdamW). [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §10.

### ICML R3 (Reviewer 3) questions

- **R3 W2 / AC item #2 — cross-family auditor concordance.** PARTIALLY ADDRESSED — [`audits/CROSS_FAMILY_HONEST_REAUDIT.md`](CROSS_FAMILY_HONEST_REAUDIT.md) (8/10 strict concordant; 10/10 defect-existence concordant); honest limitation: all passes remain Opus 4.7. True non-Claude cross-family closure (GPT-5 / Gemini 3 Pro) deferred to Phase-9i future work. [`PAPER.md`](../PAPER.md) §5.7.
- **R3 Q-cross-domain-H71 — first empirical test of the sole NOVEL+TESTABLE sci-critic survivor.** ADDRESSED via Phase-9g Control 4: `h71_icosa_rope3d_vit_tiny_rotcifar10` n=3 mean 0.6525 vs `vit_tiny_1d_rope_rotcifar10` n=1 = 0.6507, Δ = +0.18 pp INCONCLUSIVE. [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §13.4.

### Area-chair punchlist

- **AC item 2 (cross-family auditor concordance).** PARTIALLY ADDRESSED (see R3 W2 above).
- **AC item 3 (audit-calibration sample-size to n≥50).** ADDRESSED — Phase-9b n=62 extension; [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §11.

### REBUTTAL items

- All 13 items in [`audits/ICML_REVIEWS_2026-05-30/REBUTTAL.md`](ICML_REVIEWS_2026-05-30/REBUTTAL.md) are addressed in the form they were promised: each blocker has either an explicit data point (Phase-9a/b/c/f/g) or an explicit filed-as-Phase-9h future-work entry.

---

## 3. Final defensibility envelope (three regimes, honest)

### 3.1 Default-config n=7 cert (lr=1e-3 wd=5e-4 bs=256 AdamW) — **CERTIFIED**

| arm | n=7 mean | Δ vs baseline | paired Wilcoxon p_one | Holm-Bonferroni α' = 0.0167 cleared? |
|---|---:|---:|---:|:---:|
| `pair_gm_pdw` | 0.5786 | +1.74 pp | 0.0078 | **YES** |
| `slot_act_sine` | 0.5790 | +1.78 pp | 0.0078 | **YES** |
| `sg_only_phi_budget` | 0.5736 | +1.24 pp | 0.0078 | **YES** |
| `baseline_resnet20` (rail) | 0.5612 | — | — | — |

**Formal statistical statement that remains valid for the paper.** ([`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §1–§6).

### 3.2 Iso-tuned-cell n=7 (lr=3e-3 wd=5e-4 bs=128 AdamW; `slot_act_sine` at wd=2e-3) — **NOT CERTIFIED (HONEST FAIL)**

| arm | n | Δmean paired | paired Wilcoxon p_one | Phase-5 ordinal gate |
|---|---:|---:|---:|:---:|
| `pair_gm_pdw` | 7 | +0.79 pp | 0.1094 | FAIL |
| `sg_only_phi_budget` | 6 (seed-3 excluded as <30 ep) | +0.66 pp | 0.0781 | FAIL |
| `slot_act_sine` | 4 (wd=2e-3 baseline-neighbour) | +0.25 pp | 0.3750 | FAIL |

**Honestly reported. No iso-tuned-cell paired Wilcoxon clears α=0.05, let alone Holm-Bonferroni. Default-config cert remains the formal claim; iso-tuned regime cannot be re-certified at NeurIPS-α with this sample size. Phase-9g future work: n=15+ iso-tuned extension.** ([`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) §10.)

### 3.3 Phase-9g Controls — **PARTIAL REFUTATION OF SPECIFIC-MECHANISM INTERPRETATIONS**

| control | finding | impact on paper |
|---|---|---|
| C1 non-φ 3-axis | paired Δ +0.61 pp pair_gm_pdw vs nonphi, 2/3 positive, p_one=0.25 | **φ-specific interpretation PARTIALLY REFUTED** (3-axis structure carries the bulk of the lift) |
| C2 activation | `slot_act_tanh` BEATS sine by +0.48 pp paired (3/3 positive) | **SIREN-specific interpretation REFUTED** (generic activation engineering, not SIREN) |
| C3a tuned baseline (n=1) | tuned vanilla `baseline_resnet20` = 0.5984; sits +1.94 to +2.48 pp ABOVE all three winners' default-config n=7 means | **PROVISIONAL: tuned baseline numerically beats all three priors** — 3-seed extension is Phase-9h headline diagnostic |
| C4 H71 IcosaRoPE3D vs 1D-RoPE | Δ=+0.18 pp (n=3 vs n=1) INCONCLUSIVE | H71 untested NOVEL+TESTABLE status preserved; n=3 1D-RoPE extension filed Phase-9h |

---

## 4. Honest assessment by submission track

### 4.1 Workshop / D&B-track defensibility — **CLEARLY DEFENSIBLE**

The paper's load-bearing contribution at the workshop / D&B track is the **protocol** (dual-track audit + Fixer + per-experiment-page + auto-checkpoint loop, packaged as 17 content-agnostic skills under [`skills/`](../skills/)). That contribution is independent of the empirical accuracy claims. The protocol's calibration data (51 % non-PASS impl-critic, 22-pp MAJOR/BROKEN excess at one-sided Fisher p=0.036, 1/81 NOVEL+TESTABLE sci-critic rate, H09 phi_budget realised-ratio drift case study) holds regardless of whether the iso-tuned / tuned-baseline regimes ultimately preserve the priors' lift. **Workshop / D&B track: clearly defensible at the screening compute budget. The paper offers the community portable infrastructure for distinguishing signal from numerology in large autoresearch design spaces, with a worked example.**

### 4.2 Main-track defensibility — **CONDITIONAL ON PHASE-9H**

The paper's empirical headline ("three Phase-8 candidates pass paired Wilcoxon at α=0.05 under Holm-Bonferroni") **remains true as a formal statistical statement at the default-config slice** (matched-config baseline at lr=1e-3 wd=5e-4 bs=256 AdamW). However, the Phase-9f n=7 iso-tuned + Phase-9g Control 3a n=1 tuned-baseline evidence jointly suggests that **the priors' lift does NOT robustly transfer to properly-tuned baseline regimes** — the φ-specific and SIREN-specific interpretations are also partially / fully refuted by Controls 1 and 2.

**Honest main-track verdict:** the empirical cert is *defensible at the default-config slice* but its *interpretation* is now significantly more uncertain. Main-track acceptance is conditional on (a) Phase-9h's 3-seed re-run of `baseline_resnet20_tuned_lr0.01_wd0.0005` either falling below the priors (preserving the empirical claim) or above the priors (downgrading the empirical claim to "default-config artifact"); (b) the n=7 paired tanh-vs-sine Control 2 closure; (c) the rotated_CIFAR-100 ResNet-20 baseline that closes Control 4 / H91. **Without Phase-9h, the paper is INTERNAL QA PASS / EXTERNAL WEAK_REJECT (per [`paper/REVIEWER_CHECKLIST.md`](../paper/REVIEWER_CHECKLIST.md) Acceptance gate, unchanged since 2026-05-30).** The Phase-9g controls REINFORCE the WEAK_REJECT status — they did not flip the paper to ACCEPT — but they DO close the Phase-9 sub-phase chain at every reviewer-flagged item, leaving Phase-9h as the single remaining external blocker.

### 4.3 Pipeline-close criterion (per Rule 11)

Per [Rule 11 — periodic GitHub checkpoint](../CLAUDE.md#rule-11) and the auto-checkpoint loop (Rule 20), the GPU pipeline closes here: every cell produced by Phase-9a → Phase-9g is committed, pushed, and dashboard-indexed. No GPU job is in-flight. The next action (Phase-9h Control 3a 3-seed closure) is queued but unlaunched — the operator can launch it independently when ready, without losing any prior-phase state.

---

## 5. Filed Phase-9h work (single binding diagnostic + cleanup items)

The following items are **filed but unlaunched** as Phase-9h; none is required for the workshop / D&B-track submission. Each is listed with the cheapest-experiment cost to close it.

1. **Phase-9h headline (~0.25 GPU-h):** 3-seed re-run of `baseline_resnet20_tuned_lr0.01_wd0.0005` at default-config recipe to resolve the Control 3a n=1 vs winners-n=7 asymmetry. If the tuned-baseline n=3 mean reproduces ≥ 0.59, the priors' main-track empirical claim downgrades to "default-config artifact"; if it falls below ~0.575, the cert's interpretation strengthens.
2. **Phase-9h Control 3b (~1 GPU-h):** RegNetX-200MF shrunk to 270k params, 3 seeds, default recipe — the literature-canonical Pareto-region comparator for `sg_only_phi_budget`'s DERIVATIVE+TESTABLE sci-critic verdict.
3. **Phase-9h Control 2 closure (~3 GPU-h):** n=7 paired extension of `slot_act_tanh` vs `slot_act_sine` to settle whether the +0.48 pp tanh advantage at n=3 holds at NeurIPS-α.
4. **Phase-9h Control 4 closure (~2 GPU-h):** 3-seed extension of `vit_tiny_1d_rope_rotcifar10` to match the n=3 IcosaRoPE3D arm for a fair paired Wilcoxon on the H71 mechanism.
5. **Phase-9h rotated baseline (~1 GPU-h):** rotated_CIFAR-100 ResNet-20 baseline at the matched recipe so H91 (`combo_domain_icosa_rotation`) has a fair Δ reference.
6. **Phase-9h iso-tuned n=15 (~10 GPU-h):** the principled re-certification path for the iso-tuned regime — extending both arms from n=7 to n=15+ at the iso-tuned cell to tighten σ_iso below the default-config σ.

**Total Phase-9h cost: ~17 GPU-h** — within a single overnight batch on the RTX 4090 Laptop. The pipeline is CLOSED in the sense that every reviewer-flagged item has either a data point or a filed future-work entry; the pipeline is **NOT** closed in the sense that all Phase-9h work is finished. The latter is the operator's decision; this marker documents the state at which the operator can step away from active GPU work with the full Phase-9 chain preserved.

---

## 6. Cross-references

- [`CLAUDE.md`](../CLAUDE.md) — Rules 1–27 (especially Rule 11 / 20 checkpoint discipline, Rule 22 dual-track audit gate, Rule 28 screening-vs-evaluation tier).
- [`PAPER.md`](../PAPER.md) — §5.5 + §5.5.6 splice (Phase-9g closeout) + §7.3 (limitations updated through Phase-9g).
- [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md) — Sections 1–13 (n=7 cert / hill-climb / audit-calibration / magnitude / iso-tuned / Wave-1 combos / Phase-9g controls).
- [`paper/FINDINGS.md`](../paper/FINDINGS.md) — 2026-05-29 PM PROMOTION + 2026-06-01 update + 2026-06-01 PM Wave-1 + 2026-06-01 evening Phase-9g.
- [`paper/REVIEWER_CHECKLIST.md`](../paper/REVIEWER_CHECKLIST.md) — Sections A–I, H1–H8 hill-climb evidence + iso-tuned closeout + Phase-9g controls.
- [`controls/PLAN.md`](../controls/PLAN.md) — pre-registered control specs (C1–C4) + launch allowlist that gated the Phase-9g sweep.
- [`audits/ICML_REVIEWS_2026-05-30/REBUTTAL.md`](ICML_REVIEWS_2026-05-30/REBUTTAL.md) — rebuttal item tracker; Phase-9g closes the open items listed there.
- [`scripts/run_control_sweeps.py`](../scripts/run_control_sweeps.py) — the launch orchestrator that ran Phase-9g.
- [`logs/controls_phase9g_relaunch_20260601_072933.log`](../logs/controls_phase9g_relaunch_20260601_072933.log) — the raw Phase-9g run log (07:29 → 17:02 PT).

---

*Generated 2026-06-01 evening. The GPU pipeline is closed at Phase-9g; Phase-9h is filed and ready to launch when the operator decides. Workshop / D&B-track defensibility: clearly held. Main-track defensibility: conditional on Phase-9h, with the Phase-9g controls reinforcing the WEAK_REJECT status by partially refuting specific-mechanism interpretations while leaving the default-config formal cert intact.*
