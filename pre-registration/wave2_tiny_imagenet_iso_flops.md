# Wave-2 Pre-Registration — Tiny-ImageNet 200-class iso-FLOPs comparison

**Date filed:** 2026-06-06
**Commit hash:** _to be filled at commit time and propagated to PAPER.md §5 before any seed launches_
**Status:** PRE-REGISTERED (no seeds run yet)
**Depends on:** Wave-1 PROMOTE for at least one prior
**SYNTHESIS_100.md items:** B3 + C1 + C3
**Estimated wall-clock:** ~80 GPU-h on RTX 4090 Laptop, 3 calendar weeks

## Hypothesis under test

Iso-FLOPs nature-inspired priors lift the modern recipe baseline on
Tiny-ImageNet (200 classes, 64×64 native) at n=5 paired Wilcoxon with
Holm-Bonferroni k=3. Tiny-ImageNet 200-class is the sweet spot for a
single-4090 study — CIFAR-100 100-class is the noise-floor zone for
+1 pp claims (Phase-9i Wilcoxon-floor confound), and ImageNet-100 is
deferred to Wave-3. The three priors under test are the Wave-1
SURVIVORS (or {`pair_gm_pdw`, `slot_act_sine`, `sg_only_phi_budget`}
if Wave-1 results promote them as a set under the alternative
PROMOTE-CONDITIONAL branch).

## Recipe (exact)

- **Architectures:** baseline + the (up to 3) Wave-1 SURVIVOR priors at
  iso-FLOPs ~1.0 G. Baseline = the Wave-1 strongest non-prior literature
  architecture (likely RegNetX-200MF post width-pin per Wave-1 outcome).
  Priors carry their YAML overlays (e.g.,
  `configs/cifar100_modern_200ep_pair_gm_pdw.yaml` updated to Tiny-ImageNet).
- **Dataset:** Tiny-ImageNet 200-class (Yao & Miller 2015 Stanford CS231n).
  Train 100k, val 10k. Native 64×64 (no upscaling — the iso-FLOPs band
  is set at 64² input).
- **Augmentation:** the Wave-0 PROMOTED recipe re-tuned to 64×64 (RandAugment
  N=2 M=14 is verified at 64² in the literature; Random Erasing kept at
  p=0.25).
- **Optimizer:** AdamW, lr=5.0e-4, wd=5.0e-4.
- **Scheduler:** cosine, warmup 5 ep, 80 ep total.
- **Seeds:** 0, 1, 2, 3, 4 (n=5) for SCREENING; **extended to n=7** for any
  prior that clears the n=5 decision rule (Phase-9j n=7 confirmation,
  SYNTHESIS_100.md C1).
- **Epochs:** 80.
- **Hardware:** laptop RTX 4090 16 GB. bf16 AMP, batch 256, num_workers 0,
  OMP/MKL caps per CLAUDE.md Rule 26. **Headline-mode** seeding per D4
  (`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
  `worker_init_fn=seed_worker`).

## Decision rule (pre-committed)

- **IF** at n=5 a prior's paired Wilcoxon p_one ≤ 0.0167 (= 0.05/3 Holm
  k=3 at the most stringent step) AND its BCa 95% bootstrap CI on Δmean
  excludes 0 **THEN** EXTEND that prior to n=7 and re-test under the
  EVALUATION-tier gate.
- **IF** at n=7 the prior's paired Wilcoxon p_one ≤ 0.0167 AND the BCa
  95% CI on Δmean excludes 0 **THEN** PROMOTE to Wave-3 (ImageNet-100).
- **IF** at n=5 the prior's CI includes 0 **THEN** FALSIFIED on
  Tiny-ImageNet at iso-FLOPs modern recipe.
- **IF** all 3 priors are FALSIFIED **THEN** the iso-FLOPs nature-inspired
  claim is refuted at the 200-class scale; PAPER.md §5 reframes around
  the negative result.

POSI correction (CLAUDE.md Rule 35 / SYNTHESIS_100 C2): even a n=7 PROMOTE
is reported with the disclosure that the screening universe was k≈40-76
across the Phase-0 → Phase-8 funnel; post-selection-inference α' ≈ 0.001
at k=40 — the Wave-2 PROMOTE is α=0.0167 EVALUATION but does NOT clear
the POSI floor. Wave-3 is the path to a POSI-clearable claim.

## Statistical test

- **Test:** paired Wilcoxon signed-rank, paired permutation test (n=5
  exact p achievable ≈ 0.031 minimum, n=7 ≈ 0.008). Paired-t DROPPED at
  n ≤ 6 per SYNTHESIS_100 C5.
- **α:** 0.05 two-sided; one-sided used only when the prior's pre-registered
  prediction is directional (all 3 priors here are directional +Δ).
- **Multiple-comparison correction:** Holm-Bonferroni k=3 (3 priors).
  POSI α' ≈ 0.001 at k=40 disclosed.
- **Auxiliary:** 10⁴-resample BCa bootstrap 95% CI on Δmean per prior.
  Empirical noise band derived from the n=7 baseline-only seeds (CLAUDE.md
  Rule 35).
- **Required n for stated power:** at α=0.05 two-sided, σ=0.45 pp,
  Δ=1.0 pp, 80% power, n≥7 (`paper/STATISTICAL_TESTS.md` §3 derivation).

## Expected wall-clock

- 80 ep at ~1 G FLOPs and batch 256 on the 4090 ≈ ~4 GPU-h per seed.
- 4 arms × 5 seeds × 4 GPU-h = 80 GPU-h SCREENING tier.
- + 4 arms × 2 extra seeds × 4 GPU-h = ~32 GPU-h n=7 extension for
  surviving priors (folded into the budget — total ~110 GPU-h if all
  priors extend, ~80 if none do).

## Pre-registered analysis plan

1. Compute per-(arm, seed) top1 at epoch 80. **Headline-mode seeded.**
2. Apply iso-FLOPs band check (re-verify each arm at [0.95, 1.05] G FLOPs).
3. Compute per-prior paired Wilcoxon vs baseline at n=5.
4. Apply Holm-Bonferroni k=3 at α=0.05.
5. Compute 10⁴-resample BCa bootstrap 95% CI on Δmean per prior.
6. For each prior with p_one ≤ 0.0167 AND CI excludes 0, EXTEND to n=7.
7. Re-test at n=7 under the EVALUATION-tier gate.
8. Cross-family auditor pass on the n=7 SURVIVOR mechanism (GPT-5 +
   Gemini 3 Pro, ~$50 API total per SYNTHESIS_100 C3); verdict logged in
   `audits/CROSS_FAMILY_HONEST_REAUDIT.md` Section 5.
9. POSI disclosure block written into PAPER.md §5 for any PROMOTE.

## What would falsify this

1. Paired Wilcoxon p_one > 0.05 at n=5 for all 3 priors with iso-FLOPs
   at the modern recipe — priors do not survive iso-FLOPs tuning.
2. n=7 extension produces a paired Wilcoxon p_one > 0.05/3 = 0.0167 for
   every prior that cleared n=5 — the n=5 SCREENING was an n=5 fluke.
3. The cross-family auditor (GPT-5 / Gemini-3-Pro) flags BROKEN /
   NUMEROLOGY on the SURVIVOR prior's mechanism — the audit-doctrine
   defence collapses.
4. Iso-FLOPs band check at re-launch finds any arm > 1.05 G — the
   comparison is malformed.

## Cross-references

- SYNTHESIS_100.md Block B item [B3](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md),
  Block C items [C1](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md),
  [C2](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md),
  [C3](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md)
- FALSIFIERS.md rows F3 and F7
- CLAUDE.md Rules 28, 33, 35, 36
- `paper/STATISTICAL_TESTS.md` §3 (n≥7 derivation)
- Yao & Miller 2015 Tiny-ImageNet
- Berk 2013 arXiv:1306.1107 POSI
- Holm 1979 Scandinavian Journal of Statistics (Holm-Bonferroni)
