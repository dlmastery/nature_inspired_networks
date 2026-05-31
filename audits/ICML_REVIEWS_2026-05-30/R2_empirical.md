# ICML 2027 — Reviewer 2 (Empirical / Experimental Rigor)

Paper: "A Skeptical Protocol for Nature-Inspired Neural-Network Priors:
Methodological Contribution and an 84-Hypothesis Dual-Track Audit on
CIFAR-10/-100"

Repository snapshot: `C:\Users\evija\sacgeometry`, branch `main`, head
of `git log` at review start `ea27415` ("Auto-checkpoint: post-fix
campaign (tick 23)"). Required reading covered in full: `PAPER.md`,
`paper/{FINDINGS,STATISTICAL_TESTS,SOTA_COMPARISON,LIMITATIONS}.md`,
`controls/PLAN.md`, `audits/AUDIT_CALIBRATION_THIRD_PARTY.md`,
`audits/REVIEWER_PASS_PAPER.md`, `experiments/EXPERIMENT_LOG.md`,
`hypotheses/IDEA_TABLE.md`, three hypothesis design docs spanning
G1/G4/G7 (H09, H35, H67/H71), three idea hill-climb result files
(`ideas/{00_baseline_resnet20,09_phi_budget,91_pair_gm_pdw,92_slot_act_sine}/hillclimb_results.json`)
and a representative per-experiment page
(`dashboard/experiments/cifar100__pair_gm_pdw_seed0.html` index).

---

## Summary

This paper offers a real piece of methodological infrastructure (the
dual-track audit + Fixer + mechanism-test + per-experiment-page
machinery) and is admirably honest about its own caveats — most
notably the post-hoc HARKing concession at §7.3.1 and the model-family
self-grading caveat at §1.3. The 2026-05-29 PM n=7 extension on the
three Phase-8 winners is a real upgrade over the n=3 framing the area
chair reviewed: paired Wilcoxon p=0.0078 actually clears
Holm-Bonferroni α'=0.0167 across the k=3 family on CIFAR-100 30-ep.
The third-party-code audit calibration (`AUDIT_CALIBRATION_THIRD_PARTY.md`)
is a genuinely useful addition and the MAJOR/BROKEN-tier 0% calibration
floor is the most diagnostically credible single number in the paper.

But — and as an empirical reviewer this is my central message — the
paper's headline empirical claims are still **screening-budget-only,
controls-still-not-launched, and confounded by a hyperparameter axis
(batch size) the authors do not engage**. Below.

## Strengths

1. **The n=7 Wilcoxon/Holm-Bonferroni promotion is real and correctly
   computed** (`STATISTICAL_TESTS.md` §1 + §3). At n=7 with 7/7
   positive paired deltas the exact one-sided paired Wilcoxon floor
   is (1/2)^7 = 0.0078, and Holm step-down at k=3 demands ≤ 0.0167 on
   the smallest — both arithmetic and bookkeeping match. The
   bootstrap CIs ([+0.84, +1.67] for `sg_only_phi_budget`, [+1.42,
   +2.09] for `pair_gm_pdw`, [+1.38, +2.18] for `slot_act_sine`)
   exclude 0 by ≥ 2σ_baseline.
2. **Audit calibration on third-party code now exists.**
   `AUDIT_CALIBRATION_THIRD_PARTY.md` applies the Track-A doctrine to
   15 mechanisms in pytorch/vision + torch core. 10 PASS / 5 MINOR /
   0 MAJOR / 0 BROKEN gives a 22-pp MAJOR/BROKEN excess in the project
   versus a 0% calibration floor — the diagnostically credible part of
   the 51%-non-PASS narrative. The methodology section is candid about
   the n=15 → 95% CI [12%, 62%] and the χ² p≈0.22 non-significance.
3. **Reproducibility scaffolding is unusually complete.** §4.4
   hyperparameter table, composite formula + SHA-256 in §2, per-
   experiment archive directories, Rule-26 thread caps, `controls/PLAN.md`
   per-control GPU-cost estimates, and explicit pre-registration
   template (Appendix B). The cold-reader reproduction path in
   Appendix C is the kind of operator quick-reference top venues
   rarely get.

## Weaknesses (empirical-rigor focus)

1. **Four reviewer-flagged controls are READY but NOT RUN; that
   substitution is not equivalent.** `controls/PLAN.md` documents
   Controls 1–4 (non-φ 3-axis stack, activation ablation,
   tuned-ResNet-20 + RegNetX-200MF, H71 ViT-Tiny smoke) with
   commit-stamped wiring (Sets 1–4 landed) and a ~31.75 GPU-h cost
   budget — but no sweep has fired. The paper's §5.5.1 explicitly
   labels `pair_gm_pdw` "candidate, confound-open"; §5.5.2 says
   `slot_act_sine` is "φ-prior-neutral" until the activation sweep
   runs. The third-party-code audit (22-pp MAJOR/BROKEN excess)
   addresses a *different* concern (audit false-positive rate), not
   the *attribution* question of whether φ-content explains the lifts.
   Item #12 was independent of items #4/#5/#13 in the area chair's
   list, and conflating them in the rebuttal logic doesn't close the
   gap. Three Holm-cleared winners with their φ-content still
   formally unattributable is a contradiction the paper acknowledges
   but cannot resolve in this submission.
2. **Batch-size is an uncontrolled confound in the §5.5.4 hill-climb
   "BLOCKER #13 refutation".** All three winners' hill-climbed
   best_config lands at BS=128 (`hillclimb_results.json` for
   `91_pair_gm_pdw`, `92_slot_act_sine`, `09_phi_budget`); the
   hill-climbed baseline best lands at BS=256 (`00_baseline_resnet20`
   best_config). The §4.4 default-config certification ran every tag
   at BS=256. So the §5.5 n=7 result is "leaders at BS=256 beat
   baseline at BS=256" while §5.5.4 is "leaders at BS=128 beat
   baseline at BS=256". The hill-climbed gap is therefore partly a
   BS=256→128 effect for the leaders that the baseline doesn't enjoy.
   The cube is the SAME for all four tags, so the asymmetry is
   *empirically discovered*, not an artifact of design — but the
   paper does not separate "the prior helps at fixed BS" from "the
   prior + small-BS regularisation helps together." A BS=128 baseline
   row would settle it cheaply and is conspicuously missing.
3. **n=7 certification is conditional on screening budget; n=3
   hill-climbed best fails its own gate; n=7 hill-climbed best is
   future work.** `STATISTICAL_TESTS.md` §7 is honest: at n=3 the
   exact paired-Wilcoxon floor is 0.125, which cannot clear
   α'=0.0167. The two winners with positive lower-bound bootstrap CIs
   at hill-climbed best (`pair_gm_pdw` [+0.15, +1.99],
   `slot_act_sine` [+0.20, +2.23]) have CIs that scrape zero; the
   third (`sg_only_phi_budget` [−0.32, +1.76]) contains 0. The paper
   correctly does NOT call §5.5.4 a re-certification, but the
   Abstract banner ("first formally-certified empirical claims at
   NeurIPS-standard α") still asserts certification at the screening
   compute budget only. ICML readers will read "certified" and miss
   the "AT THIS BUDGET" qualifier without harder typography.
4. **CIFAR-10 12-ep claims are dead but not retracted.** §6.1 / §6.2
   / §6.3 still report a 35-row CIFAR-10 12-ep screen as evidence;
   `STATISTICAL_TESTS.md` §5 reports the 99th-percentile single-seed
   Δ across 58 non-baseline tags is +0.96 pp — INSIDE the 2σ_pooled =
   1.21 pp band. The H09 CIFAR-10 +0.76 pp screening lift the
   pre-revision paper headlined sits inside the noise band by
   construction. The post-hoc reframing of these as "screening data,
   not evaluation" is acknowledged HARKing (§7.3.1), but the rows
   still appear in §7.2.1's combo-ladder table at full type weight
   (combo5 −5.66 pp, etc.) as if those n=1 numbers carry signal
   about Rule-23-violation dynamics. A combo-ladder where every
   row is n=1 cannot adjudicate compound-stack failure modes.
5. **Seed independence is not guaranteed.** Rule 6 sets
   `cudnn.benchmark=True` intentionally (CLAUDE.md §2). At
   `cudnn.benchmark=True`, conv-algorithm selection is
   workload-dependent and non-deterministic across runs, so seed-N
   runs are NOT bit-reproducible and the "paired" assumption of
   paired Wilcoxon assumes seeds are exchangeable across leader and
   baseline arms. The paper's pairing is sound at the
   *random-initialisation × data-order* level but not at the
   *cuDNN-algorithm-choice* level — a minor empirical caveat the
   methods section should state explicitly, alongside the
   Windows/4090-Laptop hardware contract.
6. **`pair_gm_pdw` (3 axes) "refuting" `sg_full_fib` (6 axes) is
   uncontrolled.** §7.2.1's framing of orthogonal-axis compounding as
   "3 is fine, 6 is too many" relies on comparing 3-axis stacks to
   6-axis stacks, but the missing rows are 4-axis and 5-axis
   orthogonal stacks. The combo-ladder ostensibly does this, but at
   n=1 (point 4 above); and combo5 = `plr` (phi-LR-schedule) appears
   to dominate the catastrophic drop, suggesting the failure mode is
   axis-specific (LR-schedule) not depth-of-stack. The paper's own
   §7.2.1 ("removing `plr` from combo8 *also* hurts (−1.13 pp)") is
   evidence for *interaction* between axes, not orthogonality.
   Rule 23 is built on the H50 anecdote and one n=1 combo ladder.
7. **Post-fix re-run of negatives (Rule 21) is incomplete on
   non-leader hypotheses.** §1.1 Contribution 2 reports "Fixers
   correct 22 hypotheses across 8 commits", but the post-fix re-runs
   that matter for the screening-tier narrative — does H41
   `golden_adam` move from −33pp to −1pp at *every* seed, not seed 0?
   does H08 dynamic-growth's Kaiming-reinit fix change its
   verdict? — are reported only at single seed. The Phase-7 survival
   table in `paper/FINDINGS.md` lists pre/post deltas but every
   non-leader row is n=1, so the "STABLE / WEAKENED / EMERGED" labels
   carry no statistical weight. Rule 21 is correctly stated; it has
   not been fully executed at the rigor level Rule 28 demands.
8. **No cross-domain transfer evidence for the protocol claim.**
   §1.1 Contribution 3 promises "portable infrastructure" via seven
   content-agnostic skills; §7.4-6 admits cross-domain replication is
   future work. The skills directory exists (Rule 10 audit-pass), but
   the protocol's diagnostic power on a tabular / FX / medical
   codebase is unknown. The sister-repo list (`autoresearchtabular`,
   `autoresearchspy`, etc.) is a planning artifact, not evidence.

## Questions for the authors

Q1 — **Will you launch Controls 1–4 from `controls/PLAN.md` before the
camera-ready, even at single seed?** Reading the plan: 31.75 GPU-h on
the laptop 4090, spread over five evenings. The READY status is
landed; the missing piece is `--launch`. Without Control 1, the
`pair_gm_pdw` claim cannot promote past "candidate, confound-open" and
the §5.5 promotion banner conflicts with §5.5.1's labelling.

Q2 — **Can you add a BS=128 baseline row at default LR/wd to the n=7
sweep?** This is one row × 7 seeds ≈ 3.5 GPU-h. It separates "the
prior helps" from "the prior+BS=128 helps" in the hill-climbed regime.
If the BS=128 baseline closes the gap with `pair_gm_pdw`-at-BS=128 to
within bootstrap-CI overlap, the §5.5.4 refutation of BLOCKER #13 is
itself confounded.

Q3 — **What's the bootstrap CI on the 22-pp MAJOR/BROKEN-tier excess
at n=15 calibration vs n=83 project?** The Fisher exact p≈0.07 is
"marginal but in the direction"; if you bootstrap the proportion
difference and report its 95% CI, the headline of §5.8 (calibrating
the 51% non-PASS rate) becomes more concrete. A second n≈30 calibration
sample on `timm` or HF Transformers (mentioned as Phase-9b) would also
tighten the CI substantially — and you could probably do it in a day
with the same audit doctrine.

## Recommendation

**Borderline reject (5/10)**, leaning toward **workshop accept** or
**ICML Datasets-and-Benchmarks** if the four planned controls run
before the rebuttal. The protocol-as-contribution framing now holds
together; the n=7 Holm-Bonferroni certification is a real upgrade
since the area-chair pass; but the empirical headline still rests on
(a) controls that the paper itself says are required and that are
READY but not LAUNCHED, (b) a batch-size confound that the
hill-climbed-best refutation of BLOCKER #13 introduces without
addressing, and (c) the screening compute budget (12-ep / 30-ep)
qualifier that the certification banner downplays. The audit-
calibration third-party-code result is the most valuable single
addition since the prior reviewer pass; the missing iso-budget
non-φ controls are the largest remaining empirical gap. With Controls
1+2 launched at single seed and the BS=128 baseline added, this
becomes a strong accept; without them, it remains a methods-paper
case study at workshop scale.

---

*Reviewer 2 (empirical) — submitted 2026-05-30.*
