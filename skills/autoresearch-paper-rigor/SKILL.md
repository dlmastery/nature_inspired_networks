---
name: autoresearch-paper-rigor
description: Use BEFORE making any "winner" / "outside seed noise" / "statistically significant" claim in PAPER.md, FINDINGS.md, README, or dashboard headline. Enforces the statistical-rigor floor (paired Wilcoxon + 95% bootstrap CI + Holm-Bonferroni), the n=3-is-screening / n≥7-is-evaluation seed-count floor, pre-registration of the screening-vs-evaluation distinction (no post-hoc HARKing), the empirically-derived noise band per dataset, the UNTESTED_ON_RIGHT_DATASET verdict tier, the abstract-must-carry-limitations discipline, and the internal-contradiction self-audit.
metadata:
  rules_enforced: [28, 35, 36, 37]
  added: 2026-05-29
  origin_audit: audits/REVIEWER_PASS_PAPER.md, STATISTICAL_TESTS.md
---

# Skill — Statistical rigor floor + pre-registration discipline

## When to use

- BEFORE writing any "winner", "outside seed noise", or
  "statistically significant" sentence in the paper, FINDINGS,
  README, or dashboard ribbon.
- BEFORE running the n=3 → n=7 escalation sweep — pre-register
  the success criterion in git.
- AFTER any audit produces a "WEAK_REJECT" verdict — downgrade
  the artefact within the same commit and re-issue the claim
  with the rigor required.
- BEFORE recording a NUMEROLOGY / FALSIFIED verdict on a hypothesis
  — confirm the dataset matches the pre-registered falsifier.

## Pillar 1 — The statistical-rigor floor (Rule 35)

Every empirical claim using ANY of the phrases below MUST report
the four items in the contract:

**Trigger phrases:** "winner", "outside seed noise", "statistically
significant", "beats baseline", "Phase-N positive", "lead", "lift",
"survives at α=...".

**Contract (mandatory, all four):**

1. **Paired Wilcoxon signed-rank** (or paired t-test with explicit
   normality justification — Shapiro-Wilk p-value).
   - For 3 paired seeds: Wilcoxon W∈{0,1,...,6}; smallest two-
     sided p achievable is 0.25 (for W=0 or W=6).
   - This means **n=3 cannot achieve p<0.05 under Wilcoxon** —
     n=3 is SCREENING, period.
   - For n=7 paired seeds: W=0 yields p=0.0156 two-sided, which
     clears Holm-Bonferroni α'=0.0167 on a 3-hypothesis family.

2. **95 % bootstrap CI on the pp delta** (≥10,000 resamples,
   percentile method).
   ```python
   import numpy as np
   def bootstrap_ci(leader, baseline, n_boot=10000, seed=0):
       rng = np.random.default_rng(seed)
       deltas = leader - baseline
       boots = rng.choice(deltas, size=(n_boot, len(deltas)), replace=True).mean(axis=1)
       return np.percentile(boots, 2.5), np.percentile(boots, 97.5)
   ```

3. **Holm-Bonferroni correction** across the sweep family.
   - K = number of hypothesis-vs-baseline comparisons in the sweep.
   - α' = α / (K - i + 1) for the i-th smallest p, in ascending p
     order; reject if p_i ≤ α'.
   - For the 3-winner family (`pair_gm_pdw`, `slot_act_sine`,
     `sg_only_phi_budget`) at α=0.05: α'_1 = 0.0167, α'_2 = 0.025,
     α'_3 = 0.05.

4. **Empirically-derived noise band per dataset** — NOT a rule-of-
   thumb "±0.5 pp". Derive σ_seed from the project's own
   multi-seed baseline runs per dataset and report:
   - CIFAR-10 baseline ResNet-20 3-seed top-1 σ = ... (cite the
     run paths in `paper/STATISTICAL_TESTS.md`).
   - CIFAR-100 baseline ResNet-20 3-seed top-1 σ = ... .
   - A lift "outside seed noise" must exceed `2σ_seed` (or be
     explicitly characterized at the smaller multiplier).

### Minimum-seed floor table

| family size K | α_target | needed two-sided p | minimum n (paired Wilcoxon) |
|---:|---:|---:|---:|
| 1  | 0.05 | 0.05   | 6 (W=0 ⇒ p=0.0313) |
| 3  | 0.05 | 0.0167 | 7 (W=0 ⇒ p=0.0156) |
| 5  | 0.05 | 0.01   | 8 |
| 10 | 0.05 | 0.005  | 9 |

**n=3 is SCREENING** under any defensible test. **n≥7 is the
EVALUATION minimum** for a 3-hypothesis family at α=0.05 under
Holm-Bonferroni.

## Pillar 2 — Pre-registration of screening vs evaluation (Rule 36)

The classification of any sweep row as SCREENING vs EVALUATION
MUST be pre-registered BEFORE the sweep runs. Operational pattern:

```yaml
# ideas/<NN>/experiments/exp<NNN>_<tag>/pre-registration.yaml
tag: pair_gm_pdw
classification: EVALUATION           # one of: SCREENING | EVALUATION
sweep_family: phase8_winners         # K = 3 hypotheses share this family
holm_bonferroni_k: 3
alpha_target: 0.05
seeds_planned: [0, 1, 2, 3, 4, 5, 6] # n=7 to clear α'=0.0167
success_criterion: |
  Paired Wilcoxon W ≤ 1 vs baseline_resnet20_cifar100;
  95% bootstrap CI lower bound > 0;
  Holm-Bonferroni α'_i ≤ 0.0167 (smallest p in family).
falsifier_dataset: CIFAR-100
pre_registered_at: 2026-05-29T14:32:00Z
pre_registered_commit: <SHA>
author: dlmastery
```

The pre-registration commit SHA is referenced in PAPER.md and
FINDINGS.md when the claim is reported. Without pre-registration:

- Any reclassification of a row as "screening" after seeing it
  lose is **HARKing** — Hypothesizing After Results are Known —
  a BLOCKER-level finding.
- Any redefinition of the success criterion after the sweep
  completes is HARKing.

### Distinguish ordinal-margin from Δmean

When reporting a lift, report BOTH statistics unambiguously:

> `pair_gm_pdw` lifts the CIFAR-100 ordinal margin
> `min(leader_seeds) − max(baseline_seeds) = +0.99 pp`
> AND the Δmean `mean(leader) − mean(baseline) = +1.34 pp`.
> These are different statistics; the first is a non-parametric
> ordinal-gate, the second is the sample mean of paired
> differences. The Wilcoxon test below is computed on paired
> differences (the Δmean basis); the ordinal margin is reported
> for cross-reference with the Phase-5 ordinal gate, NOT as the
> primary statistical claim.

## Pillar 3 — Dataset-aware verdict tiers (Rule 36)

A hypothesis whose pre-registered falsifier specifies a dataset
that ISN'T in the sweep cannot earn a NUMEROLOGY / FALSIFIED
verdict from that sweep. Add the verdict tier:

| verdict | meaning |
|---|---|
| NOVEL+TESTABLE | passed sci-critic, awaiting empirical evidence |
| DERIVATIVE+TESTABLE | rediscovers literature mechanism but lift is reproducible |
| NUMEROLOGY | passed audit but mechanism is decorative φ-coincidence |
| UNFALSIFIABLE | mechanism cannot be tested under any feasible protocol |
| FALSIFIED | pre-registered falsifier triggered |
| **UNTESTED_ON_RIGHT_DATASET** | falsifier specified dataset X; sweep ran on dataset Y; verdict deferred |

Example: H22 (toroidal φ-closure) pre-registers "tiled-texture or
wrap-aware synthetic dataset" as falsifier. The sweep ran on
upright CIFAR-10. The correct verdict is
`UNTESTED_ON_RIGHT_DATASET`, NOT `NUMEROLOGY` — that re-labelling
is one of the BLOCKER-level findings in
`audits/REVIEWER_PASS_PAPER.md`.

## Pillar 4 — Limitations in the abstract; internal-contradiction audit (Rule 37)

### Limitations IN THE ABSTRACT

Material limitations MUST appear in the abstract, not buried in
§7. Specifically:

- Single-seed sweep rows ("the 35-row CIFAR-10 sweep is n=1 per
  row").
- Baseline-below-SOTA gap ("our 12-ep baseline is 6.5 pp below
  the 164-ep SOTA").
- Auditor-self-grading circularity ("implementer + critic + sci-
  critic + fixer agents share a model family").
- Untested NOVEL+TESTABLE hypothesis ("H71 is listed as the sole
  NOVEL+TESTABLE survivor and has no CIFAR smoke").
- CIFAR-as-wrong-testbed admission ("CIFAR is not the correct
  testbed for several equivariance hypotheses").

### Internal-contradiction self-audit

BEFORE marking the paper "done", read it end-to-end. Verify:

1. The abstract's claim about contributions matches the
   conclusion's claim about contributions.
2. The introduction's "we are not making broad claims" stance is
   NOT contradicted by §5.5 marketing the same broad claim.
3. The conclusion's "no NOVEL+TESTABLE-AND-impl-PASS hypothesis
   emerged" statement is NOT contradicted by an abstract or §5.5
   that markets a winner.
4. The "Phase-8 winner" framing is NOT contradicted by the
   "12/30-ep budget is the screening regime" caveat.
5. Every section number that appears more than once
   (e.g., duplicated `### 5.5`) is renumbered.

If ANY of these contradictions exists, the paper is not done.
The 2026-05-29 reviewer audit (`audits/REVIEWER_PASS_PAPER.md`)
counted THREE distinct internal contradictions in PAPER.md.

## Pillar 5 — Auditor-self-grading circularity disclosure

When any audit-derived rate (e.g., "51 % non-PASS audit verdict")
is reported, include the calibration disclosure:

> The implementer, critic, sci-critic, and fixer agents share a
> model family (Claude Opus 4.7). The 51 % non-PASS rate has not
> been calibrated against a known-good reference codebase. A
> future-work item is to re-run the audit protocol on a third-
> party reference (timm ResNet, pytorch-cifar100) and report the
> non-PASS rate as a false-positive baseline. The reader should
> interpret the 51 % figure as descriptive, not diagnostic, until
> that calibration is reported.

This disclosure goes in the paper section that reports the audit
rate, NOT in §7 limitations.

## Anti-patterns

- **"Outside seed noise" without paired Wilcoxon + bootstrap CI +
  Holm-Bonferroni.** Trigger phrases bind the contract.
- **Reporting Δmean only when ordinal margin tells a different
  story.** Always report both.
- **Reclassifying a losing row as "screening" after the fact.**
  HARKing. BLOCKER-level. Pre-register or accept the negative.
- **NUMEROLOGY verdict on a hypothesis whose pre-registered
  falsifier specified a different dataset.** Use
  UNTESTED_ON_RIGHT_DATASET.
- **Marketing the positive without the calibrated negative in the
  abstract.** Equal prominence (per Rule 32 spirit applied to the
  paper).
- **n=3 "winner" reported without the SCREENING tag.** n=3 is
  screening.

## Cross-references

- CLAUDE.md Rules 28 (screening-vs-evaluation), 35 (rigor floor),
  36 (pre-registration + dataset-aware verdicts), 37 (no self-
  grading banners + circularity disclosure).
- `audits/REVIEWER_PASS_PAPER.md` — origin findings.
- `paper/STATISTICAL_TESTS.md` — the project's pre-computed
  Wilcoxon / bootstrap / Holm-Bonferroni values.
- `paper/REVIEWER_CHECKLIST.md` — Sections H + I gates that
  consume this skill's outputs.
- `skills/autoresearch-per-hypothesis-hillclimb/` — the n=3 → n=7
  escalation protocol.
- `skills/autoresearch-scicritic-team/` — the verdict tiers
  augmented with UNTESTED_ON_RIGHT_DATASET.
- `skills/autoresearch-dashboard-comprehension/` — the visual
  arm of the same Rule-28 framing.
