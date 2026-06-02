# Submission README — for Area Chair
Submission title: *A Self-Auditing LLM-Agent Autoresearch Protocol That Catches Its Own Headline Drift: Audit-Calibration on 62 Third-Party Hypotheses and an 84-Hypothesis Nature-Inspired-Priors Case Study*
Track preference: ICML 2027 — bifurcated submission (Datasets & Benchmarks primary; Methods workshop fallback; main-track contingent on Phase-9 GPU pipeline closure)

## 2-minute pitch — "the audit catches its own bullshit"

We present a self-auditing LLM-agent autoresearch protocol. The headline empirical claim is **audit-calibration**: on a 62-hypothesis third-party-code substrate (`pytorch/vision` + `timm` + HF Transformers + Lightning Bolts + `torch.optim` + `state-spaces/mamba`), the protocol's implementation-critic doctrine registers **0/62 MAJOR/BROKEN**; on its own 84-hypothesis nature-inspired-priors substrate, **18/83 MAJOR/BROKEN** — a **22-pp tier-separated excess** with **Fisher exact two-sided p = 1.94 × 10⁻⁵** (clears α=0.05 by ≈ 2500×; Wilson 95% CIs non-overlapping by 8.3-pp; `audits/AUDIT_CALIBRATION_THIRD_PARTY.md` Appendix A; commit `e6f1f18`). The protocol's MAJOR/BROKEN tier is statistically distinguishable from a clean-code floor by a very large margin.

The protocol's load-bearing double-barrelled existence proof is that it **catches its own bullshit**: (a) it caught H09 phi_budget's 12.6% realised-stage-ratio drift before any external claim shipped (commit `519cdf3`); (b) it caught its OWN three-prior default-config headline collapsing under iso-tuned-LR. The three Phase-8 candidates that cleared paired Wilcoxon at α'_Holm=0.0167 on n=7 matched-recipe seeds are *outperformed by a properly LR-tuned vanilla ResNet-20 baseline by +2.27 to +2.81 pp* (Mann-Whitney p_one ∈ [0.0083, 0.0111]; no rank overlap; `paper/STATISTICAL_TESTS.md` §14). **An unaudited / un-iso-tuned-tested pipeline would have published the default-config n=7 cert as "the priors help" and stopped there.**

The 84-hypothesis nature-inspired-priors design space is the **case-study substrate** on which the protocol is calibrated, not the headline contribution. The three Phase-8 priors are honestly demoted to "screened candidates that survive matched-recipe certification but not iso-tuned-LR." The default-config cert is preserved as a worked example of protocol output and a formally-correct matched-recipe statement, but it is the **secondary** empirical claim.

## Why this paper belongs in the venue
- **Audit-calibration backbone** — n=62 third-party calibration with Fisher exact two-sided p=1.94×10⁻⁵ clears α=0.05 by >2500×; pooled-z p=8.93×10⁻⁵; Wilson CIs non-overlapping. The methodological claim is empirically backed (commit `e6f1f18`).
- **Protocol-as-contribution with a load-bearing self-falsification existence proof** — Phase-9h tuned-baseline binding diagnostic shows the protocol catches its own headline drift, not just other people's bugs. AC `audits/ICML_REVIEWS_2026-05-30/AC_synthesis.md` 7/10 D&B.
- **Secondary empirical signal at NeurIPS-α (matched-recipe only)** — three winners clear paired Wilcoxon W=0 (n=7, p=0.0078) under Holm-Bonferroni α'=0.0167 at default-config; paired-t p ∈ [5×10⁻⁵, 8×10⁻⁴] confirms magnitude. Reported as worked example of protocol output, with explicit Phase-9h iso-tuned demotion.

## Why it might NOT belong in main track (honest)
- **D&B / methods-workshop fit** — every individual piece is a port of established practice (mutation testing, two-stage clinical trials, pre-registration); novelty is in the *integration*, not any single component (R3 W1, AC §"AC recommendation justification").
- **Single-LLM-family auditor — the structural limit.** Implementer, impl-critic, sci-critic, Fixer, and calibration auditor are all Claude Opus 4.7. Closure A (n=62 calibration) and Closure B (cross-family methodologically-diverse re-audit, 8/10 strict CONCORDANT, 10/10 defect-existence CONCORDANT) are partial; true non-Claude external auditor (GPT-5 / Gemini 3 Pro) remains Phase-9e open work.
- **CIFAR-only scale — the structural empirical limit.** All experiments are CIFAR-10/-100 on ResNet-20-class; baseline 6.5 pp below 164-ep SOTA; no ImageNet, no transformer-track training.

## Reviewer-assignment recommendations
- **Stats / statistical-rigor reviewer** — Fisher exact / pooled-z / Wilson CI on the 22-pp MAJOR/BROKEN excess, Mann-Whitney + bootstrap CI on the Phase-9h tuned-baseline binding, Holm-Bonferroni at k=3 on the matched-recipe default-config cert.
- **LLM-agent methodology reviewer** — single-LLM-family auditor risk, "make tests pass" shape-only-assertion pathology, mutation-testing port into LLM-implements-LLM-audits regime, Track-A vs Track-B disjoint-scope independence, Fixer mechanism-pinning-test contract (Rule 21).
- **Meta-research / autoresearch / reproducibility reviewer** — per-experiment-page discipline, Rule-28 screening-vs-evaluation tiering, content-agnostic skills, cross-domain portability claims (R3's natural advocate).
- **Empirical-ML reviewer** — CIFAR-100 30-ep ResNet-20 class, baseline 0.5612 (n=7) at default-config vs 0.6017 (n=3) at tuned cell, the 6.5-pp gap-to-164-ep-SOTA magnitude calibration.

## The "this is what we claim" table (audit-calibration leads)
| Claim | Evidence | Strength | Caveat | Pending |
|---|---|---|---|---|
| **Audit's 22-pp MAJOR/BROKEN excess is statistically distinguishable from a clean-code floor** | 0/62 third-party MAJOR/BROKEN vs 18/83 project; Fisher p=1.94×10⁻⁵; pooled-z p=8.93×10⁻⁵; Wilson CIs no longer overlap (commit `e6f1f18`) | **HEADLINE — STATISTICALLY CREDIBLE at α=0.05 by >2500×** | All auditors share model family (Opus 4.7) | True non-Claude external re-audit open (~$20 API, 5 h) |
| **Protocol catches its own headline drift (self-falsification existence proof)** | (a) H09 phi_budget realised ratio 1:1.41:2.45 vs doc 1:1.618:2.618 fixed `519cdf3`; (b) Phase-9h tuned-baseline n=3 mean 0.6017 beats all 3 winners by +2.27 to +2.81 pp at Mann-Whitney p_one ∈ [0.0083, 0.0111] (`paper/STATISTICAL_TESTS.md` §14) | **METHODOLOGICAL HEADLINE — LOAD-BEARING** | Single case on this codebase | Cross-domain skill replication open |
| Three Phase-8 candidates certify at NeurIPS-α under Holm at the matched-recipe default-config slice | Paired Wilcoxon p=0.0078 (W=0, n=7) for `pair_gm_pdw` / `slot_act_sine` / `sg_only_phi_budget`; Holm α'=0.0167; commit `8e1fdab`+`3f501a3` | **SECONDARY — CERTIFIED matched-recipe** | Does NOT survive iso-tuned-LR (Phase-9h above demotes); `sg_only_phi_budget` also fails strict POSI k=49 | Symmetric iso-tuned n=7 paired close-out (Phase-9i) ~5 GPU-h |

## The "this is what we DO NOT claim" table
| Non-claim | Why we disown it | Where acknowledged |
|---|---|---|
| "General nature-inspired NN advance" — that φ/Platonic/fractal priors broadly outperform mainstream baselines | 51 % impl-critic non-PASS, 1/81 NOVEL+TESTABLE sci-verdict; H50 `sg_full_fib` lost −11.54 pp; H80 Reuleaux −8.83 pp; `slot_act_sine` is a SIREN replication; Phase-9h tuned baseline beats all three winners | PAPER.md abstract; §5 (self-falsification); `paper/FINDINGS.md` "The compound failure"; AC concession list |
| Priors help at iso-tuned conditions | Phase-9h n=3 tuned-baseline binding cell beats all 3 winners by +2.27 to +2.81 pp at Mann-Whitney p_one ∈ [0.0083, 0.0111] | PAPER.md §5.2 + §6.5; `paper/STATISTICAL_TESTS.md` §14 |
| ImageNet-scale or transformer-track generalisation | 10/84 hypotheses target attention backbones, none tested; all empirical work is CIFAR-10/-100 30-ep ResNet-20-class | PAPER.md §7 limitations; `paper/FINAL_STATE_FOR_REVIEWERS.md` §7 |
| Content-agnostic skills empirically demonstrated cross-domain | Skills are content-agnostic *by construction* in the templating layer; CIFAR-conditional in parameter defaults; no sister-repo replication yet executed | §1.1 contribution; R3 W3; AC concession #3 |

## Pre-rebuttal sniff-test status
- AC mean 4.75/10 (R1 4 + R2 5 + R3 5 main / 7 D&B + R4 5); AC final 5/10 main, 7/10 D&B — Weak Reject main / Accept-with-revisions D&B.
- Post-rebuttal doc-side BLOCKERs addressed in:
  - `8ba3b28` + `4223d94` + `5e930e3` + `1f0f904` — abstract compression; PAPER.md to 318 lines; H09 elevator-pitch lift; POSI re-framing; Bronstein 2021 added; [VERIFY] tags resolved; REBUTTAL (1400 words).
  - `e6f1f18` — n=15 → n=62 calibration extension (resolves α=0.05 marginality; **promoted to empirical headline in this reframe**).
  - `8f0f431` — cross-family methodologically-diverse re-audit on 10 of 18 MAJOR/BROKEN (8/10 strict CONCORDANT, 10/10 defect-existence CONCORDANT).
  - **2026-06-01 reframe** — PhD-grade critique landed; audit-calibration result promoted to headline; priors demoted to case-study substrate; PAPER.md re-organized so §4 (audit calibration) + §5 (self-falsification existence proof) precede §6 (84-hypothesis case study).

## Navigation
- 10-min version: [`paper/FINAL_STATE_FOR_REVIEWERS.md`](FINAL_STATE_FOR_REVIEWERS.md)
- Full paper: [`paper/icml2027/main.tex`](icml2027/main.tex)
- Figures: [`paper/figures/fig{1..6}.pdf`](figures/)
- ICML 4 reviews + AC + Rebuttal: [`audits/ICML_REVIEWS_2026-05-30/{R1..R4,AC_synthesis,REBUTTAL}.md`](../audits/ICML_REVIEWS_2026-05-30/)
- Stats appendix (§§1, 7, 8, 9, 10, 11 calibration n=62, 13 Controls, 14 Phase-9h tuned-baseline binding): [`paper/STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md)
- Audit calibration n=62: [`audits/AUDIT_CALIBRATION_THIRD_PARTY.md`](../audits/AUDIT_CALIBRATION_THIRD_PARTY.md) (commit `e6f1f18`)
- Cross-family re-audit: [`audits/CROSS_FAMILY_HONEST_REAUDIT.md`](../audits/CROSS_FAMILY_HONEST_REAUDIT.md) (commit `8f0f431`)
- The 28 normative rules: [`CLAUDE.md`](../CLAUDE.md) Rules 1-27 + Rule 28 (screening-vs-evaluation).
- The 7 content-agnostic skills: [`skills/autoresearch-*`](../skills/) (commit `3ec6c64`).
- Live dashboard: https://dlmastery.github.io/nature_inspired_networks/.

## Frequently misunderstood points
1. **The headline is the audit-calibration result (Fisher p=1.94×10⁻⁵ at n=62), not the 3 priors.** The reframe landed 2026-06-01: the priors don't survive iso-tuned-LR (Phase-9h tuned baseline beats all three by +2.27 to +2.81 pp), so they are honestly demoted to case-study status. The strongest empirical claim is the audit's MAJOR/BROKEN tier statistical distinguishability from a clean-code floor.
2. **`slot_act_sine` is not a φ-claim.** It is a SIREN replication (Sitzmann 2020); we surface it as a Phase-8 winner only because the protocol's worst-leader-seed Phase-5 ordinal gate caught it. Honestly labelled φ-prior-neutral in PAPER.md §6.5.
3. **`sg_only_phi_budget` cleared family-of-3 Holm but not strict POSI k=49.** We report both bounds; strict POSI k=49 requires paired-t p < 0.001 — `pair_gm_pdw` and `slot_act_sine` clear it, `sg_only_phi_budget` does not (p=8.1×10⁻⁴). Honest in §6.5; FAQ Q1.
4. **The audit's 51 % non-PASS rate is NOT the headline.** The headline is the 22-pp MAJOR/BROKEN-tier excess vs the n=62 third-party calibration arm (Fisher p=1.94×10⁻⁵). MINOR-tier audit-aggressiveness is bounded by calibration (29% project vs 34% calibration are comparable). MAJOR/BROKEN-tier is where the diagnostic signal lives.
5. **All auditors are Claude Opus 4.7.** This is the §1.3 binding caveat and the leading structural limit. Two complementary partial closures land along orthogonal axes: (A) third-party-code calibration extended to n=62 (commit `e6f1f18`), (B) cross-family methodologically-diverse re-audit using property-based / mechanism-trace / paper-math methods on 10 of 18 MAJOR/BROKEN (commit `8f0f431`). True non-Claude external auditor remains Phase-9e open work.
