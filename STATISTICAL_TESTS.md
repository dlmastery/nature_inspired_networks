# Statistical tests on the dual-track-audit + Fixer campaign

Author: stats-rigor pass, 2026-05-29.
Input: `experiments/{cifar10,cifar100}/*/metrics.json`
Computation: `scripts/_compute_stat_tests.py` (deterministic seed 20260529 for bootstrap RNG).
Purpose: address area-chair reviewer pass items
[`audits/REVIEWER_PASS_PAPER.md`](audits/REVIEWER_PASS_PAPER.md) **#B-130/134/138/194/198** —
"outside seed noise" headline claims that are not yet backed by paired tests,
confidence intervals, or multiple-comparisons correction. The reviewer's
required revision items **3 (formal statistical tests)**, **20 (multiple
comparisons)**, **194 (single-seed limitation)** are all addressed below.

This file is read-only with respect to `metrics.json`; it does not amend any
existing experimental result. It records new analyses on the existing data.
Where conclusions materially change a PAPER.md claim, that is flagged at the
bottom of §5 — but the paper itself is **not** edited by this pass (Rule 3
analog: analyses are append-only).

---

## 1. Per-claim formal tests (CIFAR-100, n=3 each)

CIFAR-100 baseline_resnet20 seeds (s0,s1,s2) = (0.5615, 0.5652, 0.5662),
mean = 0.5643, median = 0.5652, **σ = 0.248 pp**. 2σ ≈ 0.50 pp.

| Claim | Leader top1 (s0,s1,s2) | Leader median | Δmedian | Δmean | Paired Wilcoxon W | p one-sided | p two-sided | 95% bootstrap CI on Δmean | Phase-5 ordinal-gate α | Reject H0 at α=0.05? | Reject at Holm-Bonferroni α'=0.05/3=0.0167? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **pair_gm_pdw** (+1.34 pp claim) | 0.5786, 0.5789, 0.5761 | 0.5786 | +1.34 pp | +1.36 pp | 0.00 | 0.1250 | 0.2500 | [+1.11 pp, +1.63 pp] | 0.125 | **NO** | **NO** |
| **slot_act_sine** (+1.32 pp claim) | 0.5796, 0.5784, 0.5766 | 0.5784 | +1.32 pp | +1.39 pp | 0.00 | 0.1250 | 0.2500 | [+1.13 pp, +1.67 pp] | 0.125 | **NO** | **NO** |
| **sg_only_phi_budget** (+0.89 pp post-fix) | 0.5741, 0.5775, 0.5687 | 0.5741 | +0.89 pp | +0.91 pp | 0.00 | 0.1250 | 0.2500 | [+0.44 pp, +1.36 pp] | 0.125 | **NO** | **NO** |

The paired Wilcoxon W = 0 in every row because all three per-seed deltas are
strictly positive — the test statistic is the sum of negative-rank
magnitudes, which is zero. **At n=3 the exact one-sided p achieves its
theoretical floor of (1/2)^3 = 0.125 and cannot go lower.** This is the
strongest possible result the Phase-8 design can produce. It is **NOT**
strong enough to clear α=0.05, and structurally not strong enough to clear
the Holm-Bonferroni-corrected α'=0.0167.

### 1.1 Interpretation

- **pair_gm_pdw**: All three seed-paired deltas are positive and the
  bootstrap 95% CI [+1.11, +1.63] pp does not contain 0 → strong effect-size
  signal. But the paired Wilcoxon p=0.1250 is the n=3 floor and exceeds
  α=0.05 by ≥2.5×. The empirical lift is real-looking, but the **sample
  size is too small to formally reject H0 at any NeurIPS-conventional α**.
- **slot_act_sine**: Same situation; CI [+1.13, +1.67] pp excludes 0, but
  p_one=0.125 is the floor. Headline framing as "outside seed noise" is
  numerically defensible against the **CIFAR-100 baseline σ=0.248 pp** (the
  effect is ~5.5σ_base in magnitude) but not statistically defensible
  against the formal H0 at α=0.05.
- **sg_only_phi_budget** (post-fix): CI [+0.44, +1.36] pp excludes 0 by a
  comfortable margin (lower bound 1.5× the baseline σ). However, the **PAPER.md:138
  lead-floor claim of +0.25 pp (min-leader − max-baseline = 0.5687 − 0.5662)
  is a different quantity than the CI on Δmean.** The "+0.25 pp lead floor"
  is the project's Phase-5 ordinal-gate quantity, NOT a difference-of-means
  estimate. The +0.25 pp number is the worst-case ordinal margin; the
  best-estimate Δmean is +0.91 pp with the bootstrap CI as quoted. Both
  numbers should be reported; conflating them is the issue the reviewer
  flagged. Importantly, **0 is OUTSIDE the bootstrap CI**, so phi_budget's
  CIFAR-100 lift is the only one of the three with a defensible Δmean signal
  that survives the lower-bound-of-CI check at 95% (the other two have
  even larger margins). All three nonetheless fail the formal p<0.0167 test.

---

## 2. The Phase-5 worst-leader-seed > best-baseline-seed gate

The project's Phase-5 ordinal gate accepts a candidate as a winner when, on
n=3 seeds, the worst leader seed strictly beats the best baseline seed:

    pass_5  :=  min({leader_s}) > max({baseline_s}),  |leaders| = |baselines| = 3.

Under the SIGN-TEST characterization (seeds are matched pairs, sign of
delta), the probability that all 3 paired deltas are positive under H0 is

    P(all 3 sgn(d_s) = +)  =  (1/2)^3  =  0.125  =  α_gate.

The project uses unpaired min/max wording but the seeds ARE seed-aligned,
so the matched-pair sign-test interpretation is the correct one. **The
Phase-5 ordinal gate is therefore equivalent to a one-sided sign test at
α≈0.125, NOT at α=0.05.** This is 2.5× looser than the NeurIPS-standard
α=0.05 single-test bar and ~7.5× looser than the Holm-Bonferroni-corrected
α' across the 3-claim family.

To achieve a one-sided sign-test at α≤0.05 we need n ≥ 5
(P(all 5 +) = 1/32 = 0.03125). For paired Wilcoxon, n ≥ 6 is required for
the smallest one-sided p to drop below 0.05 (with the appropriate exact
distribution). **The project's claimed Phase-8 winners cannot clear α=0.05
at n=3, by any test, even when every seed favors the leader.** They can
only clear α=0.125.

This is not a flaw of the data — it is a fundamental small-sample limit.
The fix is more seeds (n ≥ 7 to clear Holm-Bonferroni at α=0.05 across
k=3 tests).

---

## 3. Multiple-comparisons correction (Holm-Bonferroni)

### 3.1 CIFAR-10 35-row screening sweep (n=1 each)

The 35-row CIFAR-10 12-ep screening sweep tests up to 35 simultaneous
null hypotheses. Family-wise α=0.05 under Bonferroni requires per-test
α'_Bonf = 0.05 / 35 ≈ **0.00143**. At n=1 seed per tag, the smallest
paired p-value achievable is 0.5 (one observation, two-sided sign test).
**No CIFAR-10 screening row can clear ANY corrected α at n=1.** The
35-row sweep is exploratory by mathematical necessity; no row of it
constitutes a confirmed positive or confirmed negative on its own.

This vindicates the paper's §7.3.1 reclassification of the CIFAR-10 sweep
as "screening" — the data does not support evaluation-level claims either
way. However, the reviewer's HARKing critique still stands: the
reclassification was post-hoc. The proper fix is to declare the sweep
exploratory **at pre-registration**, not to redefine it after observing
the negatives.

### 3.2 Phase-8 CIFAR-100 family (k=3 simultaneous tests)

Family-wise α=0.05 under Bonferroni requires per-test
α'_Bonf = 0.05 / 3 ≈ **0.0167**. Under Holm step-down: sort p-values
ascending, smallest test must clear α/3 = 0.0167, second must clear
α/2 = 0.025, third must clear α/1 = 0.05.

At n=3 the **theoretical minimum p_one-sided is 0.125**, which exceeds
α'_Holm = 0.0167 for all three positions in the step-down sequence.
Equivalently: **for k=3 simultaneous tests at α=0.05 family-wise, no
combination of n=3-seed paired Wilcoxon results can clear Holm-Bonferroni**,
regardless of effect size.

### 3.3 Sample-size requirement to clear Holm at α=0.05

To clear Holm-Bonferroni with k=3 at α=0.05 we need each p ≤ 0.05/k = 0.0167.

- **Paired sign test:** need n ≥ 6, since P(all 6 +) = 1/64 = 0.0156.
- **Paired Wilcoxon:** need n ≥ 7 with all positive deltas. One-sided exact
  p at n=7, all ranks negative-rank sum = 0, is 1/128 = 0.0078.

**The Phase-8 family must be re-run with n ≥ 7 seeds** before any
Holm-corrected α=0.05 claim is achievable for the three-winner family.

### 3.4 H09 pre-fix vs post-fix as a single test (k=1)

The reviewer also asked about H09 phi_budget pre-fix vs post-fix as a
distinct family. Treated as a single pre-registered question (k=1), no
Holm correction is needed and the pre-fix–post-fix delta is governed by
the standard paired Wilcoxon at n=3 with p_min=0.125 — also insufficient
at α=0.05. Treated as part of the 3-claim Phase-8 family (k=3) it inherits
the §3.2 conclusion.

---

## 4. Single-seed noise floor estimate (CIFAR-10 12-ep)

### 4.1 Per-tag 3-seed σ on CIFAR-10 12-ep

11 tags carry 3-seed coverage on CIFAR-10 12-ep:

| Tag | seed0 | seed1 | seed2 | mean | std (pp) |
|---|---|---|---|---|---|
| baseline_resnet20 | 0.8478 | 0.8339 | 0.8346 | 0.8388 | **0.783** |
| baseline_sg_vanilla | 0.8216 | 0.8295 | 0.8226 | 0.8246 | 0.430 |
| sg_chan_fib | 0.8011 | 0.8074 | 0.8121 | 0.8069 | 0.552 |
| sg_chan_phi | 0.8011 | 0.8074 | 0.8121 | 0.8069 | 0.552 |
| sg_only_cymatic_init | 0.7764 | 0.7580 | 0.7634 | 0.7659 | 0.946 |
| sg_only_fractal | 0.8246 | 0.8170 | 0.8246 | 0.8221 | 0.439 |
| sg_only_golden_modulate | 0.7981 | 0.7832 | 0.7792 | 0.7868 | 0.996 |
| sg_only_group | 0.6984 | 0.6993 | 0.7033 | 0.7003 | 0.261 |
| sg_only_hex | 0.7932 | 0.7993 | 0.7929 | 0.7951 | 0.361 |
| sg_only_phi_budget | 0.8556 | 0.8548 | 0.8551 | 0.8552 | 0.040 |
| sg_only_toroidal | 0.7805 | 0.7865 | 0.7743 | 0.7804 | 0.610 |

**Pooled CIFAR-10 12-ep σ = 0.607 pp** (RMS of per-tag stds).
**2σ_pooled ≈ 1.21 pp**, i.e. for a single-seed CIFAR-10 Δ to be
distinguishable from zero at the 95% Gaussian-approximation level, the
magnitude must exceed ~1.2 pp.

The paper's stated "±0.5 pp is within seed noise" rule of thumb (PAPER.md:209)
is **OPTIMISTIC** relative to this empirical pooled σ — ±0.5 pp corresponds
to ~0.82σ_pooled, well inside the noise distribution rather than at its
edge. A more honest rule of thumb is **"within ±1.2 pp is plausibly seed
noise at n=1"**.

The baseline_resnet20 σ=0.783 pp is itself substantial and is the right
σ to use specifically for the H09 comparison (because phi_budget itself
has σ=0.040 pp on CIFAR-10 — anomalously tight, possibly a data-augmentation
or seed-injection artifact deserving its own follow-up).

### 4.2 35-row screening Δ distribution at seed-0

Baseline seed-0 CIFAR-10 12-ep top1 = 0.8478. Across all 58 non-baseline
seed-0 tags present in `experiments/cifar10/`:

- mean Δtop1 = **−4.07 pp** (most tags hurt the baseline)
- median Δtop1 = **−3.19 pp**
- 90th percentile Δtop1 = +0.63 pp
- 95th percentile Δtop1 = +0.79 pp
- 99th percentile Δtop1 = +0.96 pp
- mean |Δ| = 4.40 pp; max |Δ| = 19.40 pp
- fraction of |Δ| > 2σ_pooled = 36/58 = **62.1%**.

The distribution is heavily skewed negative — most priors hurt. The
positive tail is narrow: even the 99th percentile Δ (+0.96 pp) is INSIDE
the 2σ_pooled noise band. **At n=1, no single CIFAR-10 12-ep tag has
prima-facie statistical credibility as a positive.**

### 4.3 H09 phi_budget specifically

- **CIFAR-10 12-ep seed-0:** phi_budget top1=0.8556, Δ vs baseline_seed0
  = +0.78 pp. Compared to 2σ_pooled = 1.21 pp, this is **INSIDE** the
  noise band. The pre-fix +0.78 pp single-seed lift is not by itself
  evidence of effect.
- **CIFAR-10 12-ep 3-seed paired Wilcoxon (post-fix):** Δmean = +1.64 pp
  (the post-fix run has tightened phi_budget to a much narrower seed
  distribution; baseline σ unchanged). Paired Wilcoxon one-sided p=0.125
  (the n=3 floor); two-sided p=0.25. Achieves the n=3 floor; does NOT
  clear α=0.05 even single-comparison.

---

## 5. Conclusion — claim-by-claim verdict under proper testing

The three Phase-8 "winner" claims, evaluated under formal statistical
rigor:

| Claim | Δmedian | Bootstrap 95% CI excludes 0? | Paired Wilcoxon p_one | α=0.05 single-test | α=0.0167 Holm (k=3) | Verdict |
|---|---|---|---|---|---|---|
| pair_gm_pdw | +1.34 pp | YES | 0.125 (floor) | FAIL | FAIL | **WEAK** — large effect, n too small |
| slot_act_sine | +1.32 pp | YES | 0.125 (floor) | FAIL | FAIL | **WEAK** — large effect, n too small |
| sg_only_phi_budget | +0.89 pp | YES | 0.125 (floor) | FAIL | FAIL | **WEAK** — moderate effect, n too small |

**All three "winners" FAIL the formal Holm-Bonferroni-corrected test at
α=0.05.** All three have effect sizes that are visually compelling
(CIs that exclude 0 with comfortable margins), but the n=3 design is
structurally incapable of producing a paired Wilcoxon p ≤ 0.0167 — the
exact-distribution floor is 0.125.

### 5.1 What the paper currently claims vs what the statistics support

| PAPER.md location | Current language | Justified language under §1–4 above |
|---|---|---|
| L130 (pair_gm_pdw lead +0.99 pp) | "outside seed noise" | Δmean=+1.36 pp (95% CI [+1.11, +1.63]); p=0.125 (n=3 floor); FAILS Holm α'=0.0167 |
| L134 (slot_act_sine lead +1.04 pp) | "outside seed noise" | Δmean=+1.39 pp (95% CI [+1.13, +1.67]); p=0.125 (n=3 floor); FAILS Holm α'=0.0167 |
| L138 (phi_budget lead +0.25 pp) | "outside seed noise" | Δmean=+0.91 pp (95% CI [+0.44, +1.36]); +0.25 pp is the worst-case ordinal margin, not Δmean; p=0.125; FAILS Holm |
| L194 (single-seed limitation) | "Only 3 tags carry 3-seed error bars" | Per §3.1, the entire 35-row sweep cannot clear ANY Bonferroni-corrected α at n=1 |
| L198 (HARKing concern) | §7.3.1 reclassifies screen as exploratory | The exploratory framing is statistically defensible but must be PRE-REGISTERED, not post-hoc |

### 5.2 Material implications for PAPER.md

The following PAPER.md headlines are **statistically unsupported** under
NeurIPS-conventional α=0.05 with Holm-Bonferroni correction across the
Phase-8 family:

- **pair_gm_pdw as a "winner":** The Δmean effect is the strongest in the
  campaign but does not clear formal α at n=3. Reframe as "Δmean=+1.36 pp
  (95% bootstrap CI [+1.11, +1.63]); paired Wilcoxon one-sided p=0.125
  (n=3 exact-distribution floor); does NOT clear Holm-Bonferroni
  α'=0.0167. Phase-9 re-run at n ≥ 7 seeds required for formal
  evaluation."
- **slot_act_sine as a "winner":** Same.
- **sg_only_phi_budget +0.89 pp / +0.25 pp lead:** Same; the +0.25 pp
  "lead floor" should be labeled "worst-case ordinal margin under the
  Phase-5 gate at α=0.125", not "outside seed noise". The proper Δmean
  is +0.91 pp with a CI quoted.

### 5.3 The +0.25 pp phi_budget claim, specifically (reviewer item #138)

`min(leader) − max(baseline) = 0.5687 − 0.5662 = +0.25 pp`. This is the
worst-case ordinal margin; 0 is **excluded** from this margin (because
the margin is positive). But the more informative quantity is the
bootstrap CI on Δmean = [+0.44 pp, +1.36 pp], which also excludes 0.
**The +0.25 pp number itself is not inside the bootstrap CI on Δmean
([+0.44, +1.36]) because +0.25 < +0.44.** This is a clue that the paper
is conflating two different statistics:

- (a) the ordinal-margin quantity (worst-case seed gap), which is +0.25 pp;
- (b) the mean-difference estimate (Δmean), which is +0.91 pp.

The paper should report BOTH and label them. The ordinal margin is the
Phase-5 gate output; the Δmean is the proper effect-size estimate.

### 5.4 Recommended PAPER.md revisions (consistent with reviewer item #3)

1. Replace every "outside seed noise" phrase with the bootstrap CI plus
   paired Wilcoxon p plus an explicit Holm-Bonferroni-failure disclosure.
2. Promote the +0.25 / +0.99 / +1.04 pp "lead floor" numbers to a
   companion column labeled **"worst-case ordinal margin under Phase-5
   gate, α=0.125 (= (1/2)^3 sign test)"** so the reader knows the gate
   is not at α=0.05.
3. Add an explicit "Phase-8 sample-size limitation" paragraph: n=3 is
   below the floor needed for any α=0.05 formal test, by exactly the
   factor 0.125 / 0.05 = 2.5×. The strongest possible Phase-8 outcome
   under the present design is the one already observed.
4. Add a Phase-9 pre-registration note: re-run pair_gm_pdw, slot_act_sine,
   sg_only_phi_budget at **n=7 seeds** on CIFAR-100 30-ep so the paired
   Wilcoxon can clear Holm-Bonferroni α'=0.0167 even after correction
   across k=3. This is the minimum sample size that makes the headline
   claims potentially defensible.

---

## 6. Reproducibility

Run `scripts/_compute_stat_tests.py` (read-only with respect to
`metrics.json`; deterministic bootstrap seed 20260529) to regenerate
every number in this file. Scipy ≥ 1.11 required (the venv currently
runs 1.17.1). The script is self-contained and emits a markdown-formatted
report to stdout; this file is the curated/interpreted version.

---

*End of statistical-rigor pass. The protocol is the deliverable; honest
statistics on the existing data is part of that protocol. The Phase-8
winners remain the strongest empirical signal in the campaign, but the
n=3 sample size structurally prevents them from clearing α=0.05 formal
tests. Phase-9 at n ≥ 7 is the next milestone needed before any external
claim survives a hostile peer-reviewer pass on the statistics.*
