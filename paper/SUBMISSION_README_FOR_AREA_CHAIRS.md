# Submission README — for Area Chair
Submission title: *A Skeptical Protocol for Nature-Inspired Neural-Network Priors: Methodological Contribution and an 84-Hypothesis Dual-Track Audit on CIFAR-10/-100*
Track preference: ICML 2027 — bifurcated submission (Datasets & Benchmarks primary; Methods workshop fallback; main-track contingent on Phase-9 GPU pipeline closure)

## 30-second pitch
We present a dual-track audit + Fixer-with-mechanism-verifying-test + per-experiment-page protocol for LLM-agent-implemented autoresearch — codified as 28 normative rules and seven content-agnostic skills — whose signature catch on its 84-hypothesis nature-inspired-priors calibration substrate is **H09 phi_budget**: an unaudited pipeline would have published a CIFAR-100 +1.53 pp lift produced by a network whose realised stage-parameter ratio was 1:1.41:2.45, not the doc-claimed 1:φ:φ² (commit `519cdf3`). **Certified**: three Phase-8 candidates pass paired Wilcoxon p=0.0078 under Holm-Bonferroni α'=0.0167 across a k=3 confirmatory family at n=7 default-config CIFAR-100; paired-t magnitude tests sit 3-4 orders below the Wilcoxon floor (commit `8e1fdab`/`3f501a3`). **Pending**: Phase-9f iso-tuned n=7 re-certification (~14 GPU-h), Wave-1 combos (~11 GPU-h), Controls 1-4 (~31.75 GPU-h), and a true non-Claude external auditor are AUTO-EXECUTING and explicitly not in the formal claim.

## Why this paper belongs in the venue
- **Protocol-as-contribution** — the integration of dual-track audit + Fixer-mechanism-test contract + Rule-28 screening-vs-evaluation tiering + per-experiment-page is well above bar for D&B / Reproducibility & Meta-Research workshop (R3 verdict 7/10 D&B; AC `audits/ICML_REVIEWS_2026-05-30/AC_synthesis.md`).
- **Certified empirical signal at NeurIPS-α** — three winners clear paired Wilcoxon W=0 (n=7, p=0.0078) under Holm-Bonferroni α'=0.0167; paired-t p ∈ [5×10⁻⁵, 8×10⁻⁴] confirms magnitude (`paper/STATISTICAL_TESTS.md` §1, §9; commit `3f501a3`). This is the formal claim — independent of the still-running GPU pipelines.
- **Audit-calibration backbone** — third-party `pytorch/vision`+`timm`+HuggingFace+Lightning Bolts+`torch.optim`+`mamba` calibration at **n=62** (0/62 MAJOR/BROKEN) yields Fisher exact two-sided p=1.94×10⁻⁵ vs the project's 18/83 — 22-pp tier-separated excess clears α=0.05 by >2500× (commit `e6f1f18`; `audits/AUDIT_CALIBRATION_THIRD_PARTY.md` Appendix A).

## Why it might NOT belong in main track (honest)
- **D&B / methods-workshop fit** — every individual piece is a port of established practice (mutation testing, two-stage clinical trials, pre-registration); novelty is in the *integration*, not any single component (R3 W1, AC §"AC recommendation justification").
- **GPU pipeline still in flight at submission** — Phase-9f iso-tuned n=7 extension (BS=128 leaders vs BS=256 baseline confound; R2 Q1), Wave-1 H87/H88/H91 combos, Controls 1-4 (non-φ stack / activation ablation / tuned-ResNet-20+RegNet / H71 ViT-Tiny) are READY but the splice has not landed (`paper/FINAL_STATE_FOR_REVIEWERS.md` §8).
- **Single-model-family auditor** — implementer, impl-critic, sci-critic, Fixer, and calibration auditor are all Claude Opus 4.7. Closure A (n=62 third-party calibration, commit `e6f1f18`) and Closure B (cross-family methodologically-diverse re-audit on 10 of 18 MAJOR/BROKEN, 8/10 strict CONCORDANT, 10/10 defect-existence CONCORDANT, commit `8f0f431`) are partial; a true non-Claude external auditor (GPT-5 / Gemini 3 Pro) remains Phase-9e open work.

## Reviewer-assignment recommendations
- **Stats / statistical-rigor reviewer** — Wilcoxon-at-floor identity with sign test, Holm-Bonferroni family construction, POSI k=49 vs k=3 confirmatory family, paired-t magnitude tests, bootstrap CI on Δmean (R1's profile; should be comfortable with two-stage clinical-trial framing).
- **Empirical-ML reviewer** — CIFAR-100 30-ep on ResNet-20 class, baseline 0.5612 (n=7), Bag-of-Tricks envelope intuition, BS=128 vs BS=256 hill-climb confound, gap-to-164-ep-SOTA (6.5 pp) magnitude calibration (R2's profile).
- **Meta-research / autoresearch / reproducibility reviewer** — Fixer-mechanism-test contract, Rule-28 screening-vs-evaluation tiering, per-experiment-page discipline, content-agnostic skills, cross-domain portability claims (R3's profile; this is the natural advocate).
- **LLM-agent methodology reviewer** — model-family-monoculture risk, "make tests pass" shape-only-assertion pathology, mutation-testing port into LLM-implements-LLM-audits regime, auditor self-grading limit (R1 §F + R3 §4.6).

## The "this is what we claim" table
| Claim | Evidence | Strength | Caveat | Pending |
|---|---|---|---|---|
| Protocol catches a real bug an unaudited pipeline would publish | H09 phi_budget realised ratio 1:1.41:2.45 vs doc 1:1.618:2.618 (12.6 % drift); fix `519cdf3` restores 1:1.623:2.629 (0.43 % max err); mechanism-pinning test added | LOAD-BEARING (existence proof, R1 concedes) | Single example on this codebase | Cross-domain skill replication open (R3 §6 item 1) |
| Three Phase-8 candidates certify at NeurIPS-α under Holm | Paired Wilcoxon p=0.0078 (W=0, n=7) for `pair_gm_pdw` / `slot_act_sine` / `sg_only_phi_budget`; Holm α'=0.0167; bootstrap CI ≥ 2σ_baseline; commit `8e1fdab`+`3f501a3` | CERTIFIED (default-config n=7) | k=3 is post-screening (R1 BLOCKER POSI); `sg_only_phi_budget` does NOT clear strict POSI k=49 nor iso-tuned ordinal gate | Phase-9f n=7 iso-tuned re-cert AUTO-EXECUTING (~14 GPU-h) |
| Audit's 22-pp MAJOR/BROKEN-tier excess is diagnostically credible | 0/62 third-party MAJOR/BROKEN vs 18/83 project; Fisher p=1.94×10⁻⁵; pooled-z p=8.93×10⁻⁵; Wilson CIs no longer overlap (commit `e6f1f18`) | STATISTICALLY CREDIBLE at α=0.05 by >2500× | All auditors share model family (Opus 4.7) | True non-Claude external re-audit open (~$20 API, 5 h) |

## The "this is what we DO NOT claim" table
| Non-claim | Why we disown it | Where acknowledged |
|---|---|---|
| "General nature-inspired NN advance" — that φ/Platonic/fractal priors broadly outperform mainstream baselines | 51 % impl-critic non-PASS, 1/81 NOVEL+TESTABLE sci-verdict; H50 `sg_full_fib` lost −11.54 pp; H80 Reuleaux −8.83 pp; `slot_act_sine` is a SIREN replication, not a φ-claim (§5.5.2) | Abstract caveat triple; `paper/FINDINGS.md` §"The compound failure"; AC concession list |
| ImageNet-scale or transformer-track generalisation | 10/84 hypotheses target attention backbones, none tested; all empirical work is CIFAR-10/-100 30-ep ResNet-20-class; baseline sits 6.5 pp below 164-ep SOTA | PAPER.md §7.3 limitations 5+6; `paper/FINAL_STATE_FOR_REVIEWERS.md` §6 |
| Content-agnostic skills empirically demonstrated cross-domain | Skills are content-agnostic *by construction* in the templating layer; CIFAR-conditional in parameter defaults; no sister-repo replication yet executed | §1.1 contribution 3 (concession landed); R3 W3; AC concession #3 |

## Pre-rebuttal sniff-test status
- AC mean 4.75/10 (R1 4 + R2 5 + R3 5 main / 7 D&B + R4 5); AC final 5/10 main, 7/10 D&B — Weak Reject main / Accept-with-revisions D&B (`audits/ICML_REVIEWS_2026-05-30/AC_synthesis.md`).
- Post-rebuttal doc-side BLOCKERs addressed in:
  - `8ba3b28` + `4223d94` + `5e930e3` + `1f0f904` — abstract compression to 199 words; PAPER.md to 318 lines; H09 elevator-pitch lift; POSI re-framing; Bronstein 2021 added; [VERIFY] tags resolved; REBUTTAL (1400 words).
  - `51b40e1` + `abc0ba4` + `c99b19b` + `b3e0f37` — Rule-27 regression + pyproject BOM + AUDIT_SUMMARY appendices + sniff-test cleanups.
  - `e6f1f18` — n=15 → n=62 calibration extension (resolves α=0.05 marginality).
  - `8f0f431` — cross-family methodologically-diverse re-audit on 10 of 18 MAJOR/BROKEN (8/10 strict CONCORDANT, 10/10 defect-existence CONCORDANT; partial closure on R3 W2 / AC item #2).
- Authors' position: main-track-acceptable contingent on Phase-9f/9e/9g GPU pipeline landing (~50 GPU-h, AUTO-EXECUTING; `paper/FINAL_STATE_FOR_REVIEWERS.md` §8). D&B-track-ready *now*.

## Navigation
- 10-min version: [`paper/FINAL_STATE_FOR_REVIEWERS.md`](FINAL_STATE_FOR_REVIEWERS.md) (commit `255eaf1`)
- Full paper: [`paper/icml2027/main.tex`](icml2027/main.tex) (commit `7cc5d99`)
- Figures (publication-quality): [`paper/figures/fig{1..6}.pdf`](figures/) (commit `910ea65`)
- ICML 4 reviews + AC + Rebuttal: [`audits/ICML_REVIEWS_2026-05-30/{R1..R4,AC_synthesis,REBUTTAL}.md`](../audits/ICML_REVIEWS_2026-05-30/)
- Stats appendix (§1 n=7 default, §7 hill-climbed, §8 paired-t, §9 magnitude, §10 iso-tuned, §11 calibration): [`paper/STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md)
- Audit calibration n=62: [`audits/AUDIT_CALIBRATION_THIRD_PARTY.md`](../audits/AUDIT_CALIBRATION_THIRD_PARTY.md) (commit `e6f1f18`)
- Cross-family re-audit: [`audits/CROSS_FAMILY_HONEST_REAUDIT.md`](../audits/CROSS_FAMILY_HONEST_REAUDIT.md) (commit `8f0f431`)
- The 28 normative rules: [`CLAUDE.md`](../CLAUDE.md) Rules 1-27 + Rule 28 (screening-vs-evaluation).
- The 7 content-agnostic skills: [`skills/autoresearch-*`](../skills/) (commit `3ec6c64`).
- Live dashboard: https://dlmastery.github.io/nature_inspired_networks/.

## Frequently misunderstood points
1. **`slot_act_sine` is not a φ-claim.** It is a SIREN replication (Sitzmann 2020); we surface it as a Phase-8 winner only because the protocol's worst-leader-seed Phase-5 ordinal gate caught it. Honestly labelled φ-prior-neutral in PAPER.md §5.5.2; AC explicitly recommended demoting it from the abstract triple, which the post-rebuttal abstract does.
2. **`sg_only_phi_budget` cleared family-of-3 Holm but not strict POSI k=49.** We report both bounds (`paper/STATISTICAL_TESTS.md` §1 + §9); strict POSI k=49 requires paired-t p < 0.001 — `pair_gm_pdw` and `slot_act_sine` clear it, `sg_only_phi_budget` does not (p=8.1×10⁻⁴). Honest in §5.5; FAQ Q1.
3. **The audit's 51 % non-PASS rate is NOT the headline.** The headline is the 22-pp MAJOR/BROKEN-tier excess vs the n=62 third-party calibration arm (Fisher p=1.94×10⁻⁵), with the tier-separation explicit: MINOR-tier audit-aggressiveness is bounded by calibration, MAJOR/BROKEN-tier is where the diagnostic signal lives. The non-PASS aggregate is informationally less useful than the tier-stratified split.
4. **All auditors are Claude Opus 4.7.** This is the §1.3 binding caveat. Two complementary partial closures land along orthogonal axes: (A) third-party-code calibration extended to n=62 (commit `e6f1f18`), (B) cross-family methodologically-diverse re-audit using property-based / mechanism-trace / paper-math methods on 10 of 18 MAJOR/BROKEN (commit `8f0f431`). The true non-Claude external auditor (GPT-5 / Gemini 3 Pro on the same 10 findings) remains Phase-9e open work — no API access in current execution environment.
