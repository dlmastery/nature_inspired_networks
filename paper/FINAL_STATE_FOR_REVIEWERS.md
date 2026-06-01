# Final State for ICML 2027 Reviewers
Date: 2026-05-31
Bundle status: ACTIVE PIPELINE (Phase-9f + Wave-1 + Controls 1-4 + Cross-Family Re-Audit in flight)
Read time: ~10 minutes

> A reviewer-facing executive summary that does not lie. Everything below
> traces to an existing artifact + commit SHA; nothing claims a sweep has
> completed that has not. The full evidence stack lives across `PAPER.md`,
> `paper/FINDINGS.md`, `paper/STATISTICAL_TESTS.md`, `paper/AUDIT_SUMMARY.md`,
> `audits/AUDIT_CALIBRATION_THIRD_PARTY.md`, `audits/CROSS_FAMILY_HONEST_REAUDIT.md`,
> `audits/ICML_REVIEWS_2026-05-30/{R1..R4,AC_synthesis,REBUTTAL}.md`, and
> `paper/REVIEWER_CHECKLIST.md` — ~10 000 lines combined. This file is the
> 10-minute version.

---

## TL;DR (~200 words)

We submit a **methodological contribution**: a dual-track skeptical audit
+ Fixer-with-mechanism-verifying-test + per-experiment-page protocol for
LLM-agent-implemented autoresearch campaigns, codified as 28 normative
rules and seven content-agnostic skills. The protocol's signature catch
on its 84-hypothesis nature-inspired-priors calibration substrate is
**H09 phi_budget**: an unaudited pipeline would have published a
CIFAR-100 +1.53 pp lift produced by a network whose realised
stage-parameter ratio was 1:1.41:2.45, not the doc-claimed 1:φ:φ²
(commit `519cdf3`).

**Certified empirical claims** (CIFAR-100 30-ep, post-fix code, n=7
default-config, 2026-05-29 PM): three Phase-8 candidates — `pair_gm_pdw`,
`slot_act_sine`, `sg_only_phi_budget` — pass paired Wilcoxon p=0.0078
under Holm-Bonferroni α'=0.0167 across the k=3 confirmatory family,
with paired-t p ∈ [5×10⁻⁵, 8×10⁻⁴] confirming magnitude. Three honest
caveats: (a) the iso-tuned-cell extension at n=3 cannot itself
re-certify and the Phase-5 ordinal gate fails there for all three;
(b) one of the three (`sg_only_phi_budget`) does NOT clear strict
POSI k=49; (c) all auditors are Claude Opus 4.7.

**Pending GPU work in flight at submission**: Phase-9f iso-tuned n=7
extension, Wave-1 combos H87/H88/H91 (~11 GPU-h), Controls 1–4
(~31.75 GPU-h). The headline and the protocol-as-contribution claim
are independent of these.

---

## 1. The certified empirical claims (n=7 default-config CIFAR-100 30-ep)

Baseline CIFAR-100 seeds 0..6, mean 0.5612, σ = 0.453 pp (n=7).
Source: `paper/STATISTICAL_TESTS.md` §1, commit `8e1fdab` family.

| tag | Δmean | Wilcoxon p_one | 95 % bootstrap CI on Δmean | Holm-Bonferroni α'=0.0167 | Paired-t p_one (df=6) | POSI k=49 (paired-t < 0.001) | Iso-tuned n=3 status |
|---|---:|---:|---|:---:|---:|:---:|:---:|
| `pair_gm_pdw` (H09+H48+H44) | **+1.74 pp** | 0.0078 | [+1.42, +2.09] pp | **YES** | 5.1 × 10⁻⁵ | **YES** | Δmean +1.59 pp; CI [+0.43, +2.62]; ordinal gate FAIL (tied 0.6057) |
| `slot_act_sine` (H81 SIREN) | **+1.78 pp** | 0.0078 | [+1.38, +2.18] pp | **YES** | 1.2 × 10⁻⁴ | **YES** | Δmean +1.68 pp; CI [+0.49, +2.77]; ordinal gate FAIL |
| `sg_only_phi_budget` (H09 post-fix) | **+1.24 pp** | 0.0078 | [+0.84, +1.67] pp | **YES** | 8.1 × 10⁻⁴ | **NO** (POSI-uncertified) | Δmean +1.16 pp; CI [−0.04, +2.30] **contains 0**; ordinal gate FAIL |

All three winners produced 7/7 positive paired deltas → Wilcoxon W=0,
floor (1/2)⁷ = 0.0078 attained. Paired-t magnitude tests
(`paper/STATISTICAL_TESTS.md` §9; commit `3f501a3`) sit 3–4 orders
below the Wilcoxon floor, confirming the lift is many σ above zero.

## 2. Statistical envelope across 3 regimes

`paper/STATISTICAL_TESTS.md` §§1 (default n=7), §7 (hill-climbed n=3),
§10 (iso-tuned n=3 baseline-extension, 2026-05-31, commit `d6d4739`).

| regime | n | baseline mean (top1) | σ_baseline (pp) | Outside 2σ_baseline? | Holm-Bonferroni floor cleared? | Phase-5 ordinal gate |
|---|---:|---:|---:|:---:|:---:|:---:|
| Default-config (lr=1e-3, wd=5e-4, bs=256, AdamW) | 7 | 0.5612 | 0.453 | **YES — all three Δs (+1.24/+1.74/+1.78) clear 2σ=0.91 pp** | **YES (0.0078 < 0.0167)** | **PASS** for all 3 |
| Hill-climbed best (Phase-9a, BLOCKER #13 refutation) | 3 | 0.5929 (bs=256) | 0.967 | Borderline (Δmean +0.79 / +1.22 / +1.31 vs 2σ=1.93) | NO (n=3 floor 0.125 > 0.0167) | Median-level pass; formal Wilcoxon at floor |
| Iso-tuned cell (Phase-9f Section 10; baseline at bs=128) | 3 | 0.5937 | **1.14** (2.5× wider) | NO (2σ_iso = 2.28; Δs +1.16/+1.59/+1.68) but **YES** vs 2σ_default = 0.91 pp | NO (n=3 floor 0.125 > 0.0167) | **FAIL** for all 3 (max baseline 0.6057 ≥ min leader) |

Directional positive Δ is preserved across all three regimes for all
three winners. The **default-config n=7 result is the formal claim**;
the iso-tuned n=3 extension is a robustness check that cannot itself
re-certify at NeurIPS α. **Phase-9f is the re-certification path**
(n=7+ extension at iso-tuned cell — see §8).

## 3. The methodological contribution

The protocol is the deliverable. Encoded as **CLAUDE.md Rules 20–28**
(28 normative rules total) plus seven content-agnostic skills in
[`skills/`](../skills/): `autoresearch-multi-agent-dispatch`,
`autoresearch-critic-team`, `autoresearch-scicritic-team`,
`autoresearch-fixer-campaign`, `autoresearch-combo-ladder`,
`autoresearch-per-experiment-page`, `autoresearch-auto-checkpoint-loop`
(commit `3ec6c64`). The four moving parts: (i) **dual-track audit** —
8 + 8 parallel disjoint-scoped Critic agents grade code-vs-doc fidelity
and scientific merit independently; (ii) **Fixer campaign with
mechanism-verifying-test contract (Rule 21)** — every code patch ships
with a regression test that would have caught the original bug;
(iii) **screening-vs-evaluation tiering (Rule 28)** — n=1 screening
rows are prospectively labelled SCREENING before negatives are
observed; only n=7 + Phase-5 gate rows count as EVALUATION;
(iv) **dual-track gate (Rule 22)** — external claims must pass BOTH
the impl-critic PASS bar AND the sci-critic non-NUMEROLOGY /
non-UNFALSIFIABLE bar. The 84-hypothesis design space is the
**calibration substrate**, not a recommended library; cross-domain
portability is **claimed by construction**, not demonstrated (open
future work, see §6).

## 4. Auditor self-grading concern + closure

The §1.3 caveat: all auditors share model family (Opus 4.7). Two
complementary closures land along orthogonal axes.

**Closure A — third-party-code calibration extended to n=62**
([`audits/AUDIT_CALIBRATION_THIRD_PARTY.md`](../audits/AUDIT_CALIBRATION_THIRD_PARTY.md)
Appendix A; `paper/STATISTICAL_TESTS.md` §11; commit `e6f1f18`).
Same Track-A doctrine applied to mechanism claims from
`pytorch/vision` (n=15) + `timm` (n=19) + HuggingFace Transformers
(n=15) + Lightning Bolts / fastai (n=6) + `torch.optim` extra (n=4) +
`state-spaces/mamba` (n=3) = **n=62**. Headline:

| metric | n=15 (§4.3.1) | n=62 (Appendix A) |
|---|---|---|
| Calibration MAJOR/BROKEN rate | 0/15 (0 %) | **0/62 (0 %)** |
| Wilson 95 % CI calibration | [0.0 %, 20.4 %] | **[0.0 %, 5.8 %]** (3.5× tighter) |
| Wilson CIs overlap with project [14.2 %, 31.7 %]? | YES (6.2-pp window) | **NO** (8.3-pp separation) |
| Fisher exact two-sided | p = 0.066 (does NOT clear α=0.05) | **p = 1.94 × 10⁻⁵** (clears by >2500×) |
| Pooled two-proportion z | z = 1.996, p = 0.046 | **z = 3.918, p = 8.93 × 10⁻⁵** |

The 22-pp MAJOR/BROKEN-tier excess is now two-sided significant at
α=0.05 by a very large margin; point estimate is robust to calibration
n.

**Closure B — cross-family methodologically-diverse re-audit**
([`audits/CROSS_FAMILY_HONEST_REAUDIT.md`](../audits/CROSS_FAMILY_HONEST_REAUDIT.md);
commit `8f0f431`). 10 of the 18 MAJOR/BROKEN findings re-audited
across 3 distinct methods (property-based / mechanism-trace /
paper-math), stratified across G1–G8 with all 3 originally BROKEN
included. **8/10 strict CONCORDANT, 10/10 defect-existence CONCORDANT**.
The 2 partial-discordances are *finding-additions* (H47 wrap-mod-5,
H67 1024→13 adaptive-pool), NOT finding-revocations — the
methodological-diversity probe surfaces NEW concerns the original
audit missed. **Honest gap**: this is NOT a non-Claude external
auditor; that remains Phase-9e future work (no API access in current
execution environment).

## 5. Negative results (equal prominence per Rule 9)

- **H50 `sg_full_fib` — −11.54 pp** below ResNet-20 baseline at 12-ep
  CIFAR-10 (all six priors stacked on the same conv-block forward
  path). The cautionary tale that motivated CLAUDE.md Rule 23
  (orthogonal-axes-only compounding). `paper/FINDINGS.md` §"The
  compound failure".
- **H41 `golden_adam` — REQUALIFIED 2026-05-29 (β-only ~−1 pp, NOT
  −33 pp)**. The pre-fix −32.82 pp catastrophe was an
  **eps-confound** (`eps = 1/φ⁴ ≈ 0.146` dominated Adam's
  denominator at CIFAR gradient scales). Fixer-Opt (`8aa0430`)
  restored stock `eps = 1e-8`; post-fix β-only top1 = 0.8394 (Δ ≈ −1
  pp at 12-ep CIFAR-10). Clean β-only Reddi-regime test is asymptotic
  (β2<0.95 non-convergence at 100+ epochs); deferred to Phase-9
  long-horizon β2 sweep.
- **H80 Reuleaux constant-width — −8.83 pp on CIFAR-10 12-ep**
  (`sg_only_constant_width` 75.95 % vs baseline 84.78 %). Constant-
  width 3×3 conv mask is too aggressive on the small kernels. Given
  equal billing to H50 in `paper/FINDINGS.md`.

## 6. Honest limitations

The seven explicit limitations enumerated in `PAPER.md` §7.3:

1. **Single-seed for most screening rows** — only baseline / phi_budget
   / golden_momentum carry 3-seed on CIFAR-10; rest are n=1.
2. **Baseline sits 6.5 pp below 164-ep SOTA** — our 12-ep ResNet-20
   reaches 84.78 %; the canonical 164-ep recipe (He CVPR 2016)
   reaches 91.25 %. Phase-8 winners' +1.24 to +1.78 pp CIFAR-100
   lifts are < 1/8 the gap-to-SOTA.
3. **Auditor still all-Claude** — partially neutralised by §4
   closures A + B; the GPT-5 / Gemini 3 Pro pass on the same 10
   findings is open Phase-9e work.
4. **H71 IcosaRoPE3D (sole NOVEL+TESTABLE survivor) is untested**
   except via H91 combo (queued in Wave-1).
5. **No transformer-track experiments** — half the 84 hypotheses
   target attention backbones; none tested.
6. **No ImageNet-scale validation** — all empirical work is CIFAR-10
   / CIFAR-100.
7. **POSI on Phase-8 family selection** — k=3 confirmatory family is
   post-screening; k=49 strict POSI bound is reported separately
   (paired-t clears 0.001 for two winners; `sg_only_phi_budget`
   does not).

Plus (added 2026-05-31):

8. **Iso-tuned-cell extension at n=3 has wide baseline σ** (1.14 pp
   vs default σ=0.453 pp at n=7). Δs of +1.16 to +1.68 pp fall
   INSIDE 2σ_iso = 2.28 pp; Phase-5 ordinal gate FAILS at iso-tuned
   n=3 for all three winners.
9. **Rule 23 (orthogonal-axes-only) was derived from n=1 screening
   data**; replication at n=7 is Phase-9d (R2 W6 conceded).
10. **HARKing acknowledgement** for the screening-vs-evaluation
    distinction (`PAPER.md` §7.3.1). Rule 28 codifies the discipline
    prospectively.

## 7. ICML reviewer reception (transparent)

Four-reviewer simulated ICML 2027 main-track pass
([`audits/ICML_REVIEWS_2026-05-30/`](../audits/ICML_REVIEWS_2026-05-30/),
commits `6445744` / `ed5d2fa` / `0dd7940` / `6546938` / `2a70901`).

| Reviewer | Focus | Rating | Headline complaint |
|---|---|---|---|
| R1 | Theoretical / statistical rigor | 4/10 | POSI k=49 BLOCKER; Wilcoxon-at-floor = sign test |
| R2 | Empirical rigor | 5/10 | Controls 1–4 READY but UNLAUNCHED; BS=128 confound |
| R3 | Methodology | 5/10 main / 7/10 D&B | Incremental novelty; content-agnostic claim by construction |
| R4 | Writing / framing | 5/10 | 848-line PAPER too long; abstract 440 words |
| **Mean** | | **4.75 / 10** | |
| **AC synthesis** | | **5/10 main / 7/10 D&B** | Weak-Reject main, Accept-with-revisions D&B / Methods workshop |

**Doc-only revisions landed** before submission (no GPU work, all
data-driven splices):

- `3f501a3` — bootstrap CI + Fisher exact on 22-pp excess; paired
  permutation + paired-t on n=7 winners (R1 BLOCKER #3, R2 Q3).
- `5e930e3` — PAPER.md compressed to 318 lines; abstract to 199 words;
  [VERIFY] tags resolved; Bronstein 2021 added; POSI re-framing
  (R1 Q1); Rule-23 n=1 caveat sentence (R2 W6); §5.5 chain
  consolidated; Appendix C dropped (R4 A–G).
- `1f0f904` — REBUTTAL.md (1400 words, within 1500-word ICML limit)
  addressing 6/6 doc-side BLOCKERs.
- `51b40e1` — Rule-27 regression + pyproject BOM (post-rebuttal
  sniff-test fixes).

Post-revision rebuttal addresses every doc-side BLOCKER from R1, R2,
R3, R4 (16 of 16 items in the aggregate fix table); GPU-side filings
(Controls 1–4, Phase-9b/c/d/e/f) are held until explicit user
authorisation and tracked in §8.

## 8. Pending GPU work (in-flight, 2026-05-31)

| Pipeline | Status | What it delivers | GPU-h |
|---|---|---|---:|
| Phase-9f iso-tuned n=7 extension (`.launch_phase9f.ps1`) | running | n=7 baseline + leaders at iso-tuned (bs=128 lr=3e-3 wd=5e-4 AdamW) cell. Wilcoxon floor drops from n=3 0.125 to n=7 0.0078; re-certifies in iso-tuned regime. Splices into `paper/STATISTICAL_TESTS.md` §10 + PAPER.md §5.5. | ~14 |
| Phase-9e Wave-1 H87 + H88 + H91 (`.launch_phase9e_wave1.ps1`, commits `b0eb863` + `0dc587b`) | queued (script ready) | N=4 saturation combo (H87), novelty-pocket combo Betti+toroidal (H88), domain-stretch H71 icosa-RoPE3D smoke (H91). 11.1 GPU-h. | ~11 |
| Controls 1–4 (`controls/PLAN.md`, READY) | queued (gated on Wave-1 outcomes) | Non-φ 3-axis stack (Control 1, defends `pair_gm_pdw`); activation ablation (Control 2, defends `slot_act_sine`); tuned-ResNet-20 + RegNetX-200MF (Control 3, defends H09); H71 ViT-Tiny smoke (Control 4). | ~31.75 |
| Phase-9b audit calibration n ≥ 50 | **COMPLETE** at n=62 (commit `e6f1f18`) | Resolves the §5.8 caveat at α=0.05 two-sided. | done |
| Cross-family external auditor (Phase-9e) | OPEN (no API access in current env) | GPT-5 / Gemini 3 Pro re-audit of same 10 findings. | n/a (~$20 API + 5 h human) |
| Cross-domain replication on `autoresearchtabular` (R3 W3) | OPEN | Tests "content-agnostic" claim empirically. | ~$30 API + 1 day agent-time |

**Total in-flight GPU**: ~57 GPU-h. The post-pipeline auto-splice agent
folds Phase-9f + Wave-1 + Controls 1–4 outcomes into PAPER + dashboard
+ FINDINGS when Controls 1–4 lands. **None of the pending GPU work
changes the formal claim** — the default-config n=7 cert + protocol-
as-contribution claim are pre-pipeline.

## 9. The 8 NEW combo hypotheses (R-D synthesis)

`audits/COMBINATIONS_RESEARCH/D_new_hypotheses.md` (commit `c3b01c2`),
synthesising Agents A (empirical stackability) + B (theoretical 20-axis
orthogonality) + C (14-paper literature survey).

| # | Tag | Class | Composition | Predicted Δ | GPU-h | Status |
|---|---|---|---|---|---:|---|
| 1 | `combo_n4_pair_slot` (H87) | III saturation-ext | H09+H48+H44+H81 | +1.5 to +2.5 pp vs baseline; −0.3 to +0.7 vs best solo | 2.6 | **Wave-1 queued** |
| 2 | `combo_domain_icosa_rotation` (H91) | IV domain-stretch | H09+H24+H71 on rotated-CIFAR-100 | +5 to +15 pp vs baseline (sole NOVEL+TESTABLE on its natural dataset) | 5.5 | **Wave-1 queued** |
| 3 | `combo_novelty_betti_torus` (H88) | II novelty-pocket | H09+H22+H51 | +0.5 to +1.5 pp; tests Betti-loss + toroidal pad | 3.0 | **Wave-1 queued** |
| 4 | `combo_bridge_modern` (H85) | I bridge | 7-trick mainstream + H09+H48+H44 | +3.5 to +6.5 pp (tests BLOCKER #13 directly) | 6.0 | Wave-2 future |
| 5 | `combo_bridge_lite` (H86) | I bridge | 4-trick mainstream + H09+H81 | +2.5 to +4 pp | 3.0 | Wave-2 future |
| 6 | `combo_n5_pair_slot_betti` | III sat-ext | N=5 of certified + H51 | +1 to +2.5 pp | 3.0 | Wave-3 future |
| 7 | `combo_novelty_cymatic_pentagonal` (H89) | II novelty | H09+H46+H37-CNN | +0.3 to +1.5 pp | 1.2 | Wave-3 future |
| 8 | `combo_paradigm_kan_phi` (H92) | V cross-paradigm | H09+H69+H44 (KAN head) | −1 to +1 pp (high-risk) | 1.4 | Future-work |

Wave-1 (H87+H88+H91) is the **minimum viable Phase-9d**, 11.1 GPU-h,
running now. The 5 future-work proposals carry full design docs with
falsifiers + GPU costs in `audits/COMBINATIONS_RESEARCH/D_new_hypotheses.md`.

## 10. Navigation pointers

| Artifact | Path | Commit (representative) |
|---|---|---|
| Paper body | `PAPER.md` | `5e930e3` (compressed 318 lines) |
| Findings + verdict tier summary | `paper/FINDINGS.md` | `3d0d77e` (iso-tuned closeout) |
| Statistical-tests sourcebook | `paper/STATISTICAL_TESTS.md` | §0–§11 (`d6d4739` + `e6f1f18`) |
| Audit summary (Tracks A + B + C) | `paper/AUDIT_SUMMARY.md` | `c99b19b` + appendices |
| Third-party audit calibration n=62 | `audits/AUDIT_CALIBRATION_THIRD_PARTY.md` | `e6f1f18` |
| Cross-family methodologically-diverse re-audit | `audits/CROSS_FAMILY_HONEST_REAUDIT.md` | `8f0f431` |
| ICML 2027 simulated review pack | `audits/ICML_REVIEWS_2026-05-30/{R1..R4,AC_synthesis,REBUTTAL}.md` | `1f0f904` / `2a70901` |
| Reviewer checklist (42 of 42 internal QA PASS; external WEAK-REJECT-main) | `paper/REVIEWER_CHECKLIST.md` | `ea52d01` |
| The 28 normative rules | `CLAUDE.md` Rules 1–27 + Rule 28 (screening-vs-evaluation) | latest HEAD |
| 7 content-agnostic skills | `skills/autoresearch-*` | `3ec6c64` |
| Controls 1–4 plan (READY, unlaunched) | `controls/PLAN.md` + `scripts/run_control_sweeps.py` | latest HEAD |
| 8 new combo hypotheses (R-D synthesis) | `audits/COMBINATIONS_RESEARCH/D_new_hypotheses.md` | `c3b01c2` |
| Wave-1 launch script | `.launch_phase9e_wave1.ps1` | `0dc587b` |
| Phase-9f launch script | `.launch_phase9f.ps1` | latest HEAD |
| H09 Fixer commit (the signature catch) | `src/nature_inspired_networks/phi_scaling.py` | `519cdf3` |
| Composite-formula fingerprint | `src/nature_inspired_networks/eval.py:COMPOSITE_FORMULA` SHA-256 `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893` | (Rule 2) |
| Live dashboard | https://dlmastery.github.io/nature_inspired_networks/ | latest |

## 11. Frequently asked questions (FAQ for area chairs)

**Q1. Is the n=7 cert robust to multiple-comparisons?**
A1. **Family-of-3 confirmatory family**: YES. Paired Wilcoxon
p=0.0078 < Holm α'=0.0167 for all three; paired-t p ∈ [5×10⁻⁵, 8×10⁻⁴].
**Strict POSI k=49 screening family** (per R1 BLOCKER #1): only
`pair_gm_pdw` and `slot_act_sine` clear (paired-t p < 0.001);
`sg_only_phi_budget` is **POSI-uncertified but family-of-3 certified**
and labelled as such in PAPER.md §5.5. Both interpretations are
reported.

**Q2. Is the 22-pp MAJOR/BROKEN audit excess statistically credible?**
A2. **YES at α=0.05 two-sided by very large margin.** Phase-9b
n=15 → n=62 extension (commit `e6f1f18`): Fisher exact two-sided
p = 1.94 × 10⁻⁵ (>2500× margin past α=0.05); pooled two-proportion
z = 3.92, p = 8.93 × 10⁻⁵ (>500× margin); Wilson 95 % CI on
calibration MAJOR/BROKEN rate shrinks to [0.0 %, 5.8 %] and no
longer overlaps the project Wilson CI [14.2 %, 31.7 %]. The
bootstrap CI on the difference is structurally unchanged at
[+13.3, +31.3] pp because the calibration arm contributes zero
variance with k_cal=0 at any n.

**Q3. What's the bundle's track recommendation?**
A3. **ICML D&B-track / Methods-Workshop accept** (mean reviewer
4.75/10; AC 5/10 main, 7/10 D&B). Main-track conditional on
Phase-9e/9f/9g GPU sweep landing + cross-domain replication + true
cross-family external auditor. AC's verdict: "Weak Reject for main;
Accept-with-revisions for D&B / Reproducibility & Meta-Research
workshop" with five camera-ready punchlist items (see
`audits/ICML_REVIEWS_2026-05-30/AC_synthesis.md` §"Highest-leverage
actionable items").

**Q4. Will Phase-9f cert recover `sg_only_phi_budget` at iso-tuned?**
A4. **Open.** The default-config n=7 cert stands as the formal claim
regardless of Phase-9f outcome. At iso-tuned n=3 the
`sg_only_phi_budget` Δmean is +1.16 pp with 95 % bootstrap CI
[−0.04, +2.30] (contains 0); at n=7 with √(7/3) ≈ 1.53× tighter
variance the CI would likely shift to roughly [+0.4, +1.9] pp, but
this is forecast, not result. If Phase-9f delivers 7/7 positive Δ
at iso-tuned for all three, the Wilcoxon floor drops to 0.0078 and
the iso-tuned regime also clears Holm α'=0.0167.

**Q5. Should reviewers wait for Phase-9f / Wave-1 / Controls?**
A5. **No.** The default-config n=7 cert claim and the protocol-as-
contribution claim are independent of those sweeps. Pending GPU
work strengthens robustness margins (Phase-9f), tests prior-stacking
saturation (Wave-1), and tests the "you just beat an under-tuned
baseline" objection (Controls 1–4 + Wave-2 bridge combos), but does
not change the headline. The §5.5 framing already explicitly labels
`pair_gm_pdw` as "candidate, confound-open" and `slot_act_sine` as
"SIREN-replication-misclassified-as-nature-inspired."

**Q6. What's the load-bearing intellectual contribution?**
A6. **The Fixer + mechanism-verifying-test contract (Rule 21)**, not
the n=7 numbers. AC synthesis: "the integration of dual-track audit
+ Fixer-mechanism-test + per-experiment-page + Rule-28 screening-vs-
evaluation into a single executable refuse-to-launch protocol …
addresses the specific pathology that LLM-implementer-agents
optimise for 'tests pass' and so cut the shape-only-test corner —
a pathology R3 considers a research finding in its own right."
H09's 12.6 % realised-ratio drift is the cleanest single
demonstration that an unaudited pipeline would have shipped a
wrong headline (`PAPER.md` §1.2 + `paper/AUDIT_SUMMARY.md` §"H09
the headline rescue").

**Q7. Is `slot_act_sine` a "nature-inspired" finding?**
A7. **No, and the paper concedes this** (R1 §F + AC). `slot_act_sine`
swaps ReLU for `sin(ω·x)` with ω=1, consistent with Sitzmann et al.
NeurIPS 2020 SIREN's known activation-engineering benefit; the
+1.78 pp Δ has **no φ content**. The honest reading is "the
protocol surfaced a SIREN-class candidate that an unaudited pipeline
would have mis-attributed to φ-content" — a protocol-positive
finding, not a nature-inspired-prior finding.

**Q8. What's the most-honestly-caveated single sentence in the
paper?**
A8. PAPER.md §1 abstract closing: *"We report these as illustrative
protocol output, with three caveats: baseline sits 6.5 pp below
164-ep SOTA; most single-prior rows remain n=1; all agents share
a model family (Opus 4.7)."* The conclusion echoes: *"whether any
of the protocol's surviving candidates becomes an externally
defensible empirical claim is decided in Phase-9, not in this
submission."*

**Q9. What's the cleanest single existence proof of the
methodological contribution?**
A9. **H09 phi_budget**. Pre-fix realised stage-parameter ratio was
1:1.41:2.45, not the doc-claimed 1:φ:φ² = 1:1.618:2.618 — a
12.6 % drift at stage 1, caught by Track-A audit and corrected by
Fixer-PhiScaling (commit `519cdf3`) which added an integer-search
allocator whose post-fix max error is 0.43 %. The mechanism-pinning
test that ships with the fix would have caught the bug if it had
been written first. The pre-fix CIFAR-100 3-seed median (58.05 %)
is ~0.6 pp HIGHER than the post-fix (57.41 %), consistent with
"the broken realised ratio happened to land a fortuitously-high
seed-0 result" — the protocol caught the headline number that
would otherwise have shipped.

**Q10. Is the protocol-as-contribution claim genuinely
demonstrable?**
A10. **By construction yes, empirically not yet across domains.**
The seven content-agnostic skills in [`skills/`](../skills/) expose
a content-agnostic templating API; domain-specific content lives in
prompt-templates. R3 W3 conceded: the skills are CIFAR-conditional
in their parameter defaults but not in their templating layer.
**Cross-domain demonstration is open future work** (R3 Q2 / §7.4-6):
run `autoresearch-critic-team` against the sister-repo
`dlmastery/autoresearchtabular` Higgs UCI codebase and report the
verdict distribution. If the tabular replication produces an
analogous distribution AND surfaces at least one defect, the
content-agnostic claim is empirically supported.

---

*Generated 2026-05-31 in support of the ICML 2027 D&B-track / Methods-
Workshop submission. All numbers trace to existing artifacts; pending
GPU work is honestly labelled in §8. The protocol-as-contribution
claim stands independent of in-flight sweeps; the n=7 default-config
cert is the formal empirical claim. Negative results, auditor
limitations, and HARKing caveats are reported with equal prominence
to positive findings per CLAUDE.md Rule 9.*
