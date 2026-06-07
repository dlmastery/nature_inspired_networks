# Falsifier Contract

**Status:** binding falsifier contract per [CLAUDE.md Rule 36](../CLAUDE.md)
and [SYNTHESIS_100.md item C15](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md).
**Date filed:** 2026-06-06.

---

## Purpose

This document is the project's pre-committed answer to the reviewer
question "what would falsify this paper?" Reviewer 3 (R3 #24) flagged
the implicit answer — after the Phase-9h pivot — as "nothing." That
pivot was post-hoc analysis-branch multiplication
([SYNTHESIS_100.md item A7](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md))
and reads to external reviewers as the "garden of forking paths"
(Gelman & Loken 2013). Closing that critique requires a pre-registered,
public list of concrete results that WILL refute specific paper claims
when (and if) they land.

Each row below is a hypothesis-result pair. The **falsification
condition** is the exact empirical result that refutes the corresponding
**claim**; **status** OPEN means the experiment is pre-registered but
unrun. When a result lands, the row's status updates to RESOLVED with
the commit hash of the merged result, and the verdict (REFUTED / CONFIRMED
/ CONDITIONAL) is logged. **Rows are APPEND-ONLY**
([CLAUDE.md Rule 3](../CLAUDE.md) spirit + Rule 38 audit-ledger discipline)
— a row is never edited after filing, only annotated.

---

## Row format

| F# | Wave | Claim | Falsification condition | Status |
|---|---|---|---|---|

`F#` is the stable identifier (cited in PAPER.md / FINDINGS.md /
pre-registration docs). `Wave` cross-references the
[`pre-registration/`](../pre-registration/) document. `Claim` is the
exact externally-facing claim. `Falsification condition` is the
pre-committed empirical result that refutes the claim. `Status` ∈
{OPEN, RESOLVED-REFUTED, RESOLVED-CONFIRMED, RESOLVED-CONDITIONAL,
DEFERRED}.

---

## Falsifier table

| F# | Wave | Claim | Falsification condition | Status |
|---|---|---|---|---|
| F1 | Wave-0 | The modern 11-trick recipe (Bello 2021 + Wightman 2021) transfers to small-scale image benchmarks beyond CIFAR. | If Wave-0 best recipe lands < 0.85 median top1 on Imagenette at n=5, 10 ep, the recipe does NOT transfer; the project's downstream waves require a deeper recipe debug before any prior claim is interpretable. | OPEN |
| F2 | Wave-1 | Nature-inspired priors Pareto-dominate the literature (RegNetX-200MF / ConvNeXt-V2-Femto / ViT-S/16) on Imagenette at iso-FLOPs ~1 G FLOPs at 160². | If at iso-FLOPs RegNetX-200MF beats H09 φ-budget by ≥ 0.5 pp median top1 at n=5 paired Wilcoxon, the H09-Pareto-dominates-RegNet claim is REFUTED. φ-budget collapses to "rediscovery of RegNet's Pareto region" without dominance. | OPEN |
| F3 | Wave-2 | Iso-FLOPs nature-inspired priors carry a +1 pp directional lift on 200-class Tiny-ImageNet at the modern recipe. | If Wave-2 paired Wilcoxon p_one > 0.05 at n=5 with iso-FLOPs at modern recipe for ALL 3 priors (`pair_gm_pdw`, `slot_act_sine`, `sg_only_phi_budget`), the priors-survive-iso-tuning claim is REFUTED. The Phase-9i n=3 result is a CIFAR-100-specific artifact, not a generalizable prior. | OPEN |
| F4 | Wave-3 | The Wave-2 winning prior transfers to ImageNet-100 against a ResNet-50 modern-recipe baseline at iso-FLOPs at 160². | If ImageNet-100 ResNet-50 + winning-prior loses to ResNet-50 baseline by ≥ 0.5 pp median top1 at n=3, there is NO ImageNet transfer; the prior is "small-scale only" and the paper's headline reframes around Wave-2 as the largest-scale surviving claim. | OPEN |
| F5 | Wave-4 | H71 IcosaRoPE3D — the sole NOVEL+TESTABLE sci-critic survivor — lifts non-equivariant ViT-Tiny on rotated-test Spherical MNIST by ≥ +3 pp at n=5 paired. | If H71 lifts the non-equivariant baseline by < +1 pp median (or loses) at n=5 paired Wilcoxon, the IcosaRoPE3D prior is REFUTED on its load-bearing benchmark. The abstract's third-winner slot pivots to hex AID (B6) or H22 toroidal tiled-CIFAR (B7); if neither lifts, the paper collapses to a 2-winner abstract. | OPEN |
| F6 | Audit-calibration | Cross-family auditor agreement: GPT-5 / Gemini-3-Pro reach > 60% verdict-agreement with the Claude-family auditor on 10 MAJOR/BROKEN findings, validating the audit-doctrine as cross-family-robust. | If GPT-5 / Gemini-3-Pro 10-finding verdict-agreement rate < 60% on MAJOR/BROKEN, the audit-calibration headline is INVALID — same-family self-grading circularity (R1 #03 / R4 #08) is the operative effect, not a substrate-quality signal. PAPER.md §6 retracts the "audit doctrine separates project-quality from production-quality" framing. | OPEN |
| F7 | Statistical | The Phase-9j n=7 paired Wilcoxon at the modern recipe iso-FLOPs clears Holm-Bonferroni at k=3 for at least one prior. | If Phase-9j n=7 paired Wilcoxon p_one > 0.05/3 ≈ 0.0167 at modern recipe iso-FLOPs for ALL 3 priors, the "priors lift at iso-FLOPs at modern recipe" claim is REFUTED at the EVALUATION-tier. Phase-9i is reframed as a CIFAR-100-specific, non-iso-FLOPs sign-test artifact. | OPEN |
| F8 | Whole-paper | The 5-reviewer-burst (R1 ICLR / R2 ICML / R3 NeurIPS / R4 elite-researcher / R5 lab-lead) returns at least 1 ACCEPT after the 12-week plan executes. | If a re-run of the same 5-reviewer-burst on the post-12-week PAPER.md returns 5 REJECTs again, the project's external-resubmission strategy is REFUTED. Either the 12-week plan was insufficient, or the underlying claim (iso-FLOPs nature-inspired priors lift) does not survive contact with modern architectures + statistics. Honest pivot: workshop submission with the protocol contribution as headline. | OPEN |

---

## Resolution log

When a row resolves, append a `## F#` section here with the commit hash
of the resolving run, the verdict, the seed numbers, and the pointer to
the merged `pre-registration/wave*.md` result block.

_No rows resolved as of 2026-06-06._

---

## Cross-references

- Pre-registration index: [`pre-registration/README.md`](../pre-registration/README.md)
- Statistical-rigor floor: [`STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md)
- Negative-results catalogue: [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md)
- 100-improvement synthesis: [`SYNTHESIS_100.md`](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md)
- [CLAUDE.md Rules 28, 35, 36, 38](../CLAUDE.md)
- Gelman & Loken 2013 "The garden of forking paths"
