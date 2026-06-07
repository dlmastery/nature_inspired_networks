# Wave-3 Pre-Registration — ImageNet-100 FFCV (the headline experiment)

**Date filed:** 2026-06-06
**Commit hash:** _to be filled at commit time and propagated to PAPER.md §5 before any seed launches_
**Status:** PRE-REGISTERED (no seeds run yet)
**Depends on:** Wave-2 PROMOTE at n=7 for at least one prior
**SYNTHESIS_100.md item:** B4
**Estimated wall-clock:** ~80 GPU-h on RTX 4090 Laptop, 3 calendar weeks

## Hypothesis under test

The Wave-2 SURVIVING prior transfers to ImageNet-100 at iso-FLOPs against
a ResNet-50 modern-recipe baseline. SYNTHESIS_100.md frames this as
"the figure that makes the paper publishable at top-tier" because every
nature-inspired DL paper at NeurIPS / ICLR / ICML needs an ImageNet
number. ImageNet-100 (Tian 2020 arXiv:1906.05849, the
100-class subset) is the laptop-feasible analogue of full ImageNet-1k;
FFCV (Leclerc 2022 arXiv:2110.01077) brings per-seed wall-clock at 160²
to ~13 GPU-h on a 4090 Laptop, making the experiment fit within the
12-week plan budget.

## Recipe (exact)

- **Architectures (2):**
  - `resnet50_modern_in100` — ResNet-50 (He 2016 arXiv:1512.03385) at
    160² input, `timm` reference implementation, modern-recipe 11-trick
    training.
  - `resnet50_<wave2_survivor>_modern_in100` — the same ResNet-50 with
    the Wave-2 SURVIVING prior applied. The exact application is the
    pre-registered overlay of the prior on the ResNet-50 stage hierarchy
    (e.g., for H09 φ-budget: pin per-stage channels in 1:φ:φ²:φ³ ratio at
    total params ~25 M; for `pair_gm_pdw`: golden-momentum + phi_decay_wd
    on the ResNet-50 optimizer; iso-FLOPs band [0.95, 1.05] of the
    baseline measured FLOPs).
- **FLOPs measurement:** `fvcore.FlopCountAnalysis` at input (1, 3, 160, 160).
  Both arms pinned to the SAME measured FLOPs ± 5% (tighter band than
  Wave-1/2 because ResNet-50 scale tolerates the constraint).
- **Dataset:** ImageNet-100 (Tian 2020 arXiv:1906.05849), the 100-class
  subset of ImageNet-1k commonly used as the laptop-scale ImageNet
  benchmark. Train ~130k, val ~5k. **FFCV-packaged** for fast loading
  (sister `autoresearchimage` repo has a clean FFCV pipeline per
  SYNTHESIS_100 B4).
- **Augmentation:** the Wave-0 PROMOTED recipe re-tuned to 160² (the
  ImageNet-tuned 11-trick recipe is the original Bello 2021 / Wightman
  2021 calibration — at 160² it should land near published).
- **Optimizer:** AdamW, lr=2.5e-4 (= Wave-2 lr / 2 for the larger model
  per Wightman 2021 §3.2), wd=5.0e-4.
- **Scheduler:** cosine, warmup 5 ep, 100 ep total.
- **Seeds:** 0, 1, 2 (n=3). N=3 here is justified by the Wave-2 n=7
  EVALUATION already certifying the prior — Wave-3 tests TRANSFER, not
  the prior itself. Per SYNTHESIS_100 §5 B4, n=3 at ImageNet-100 is
  more compute than the entire CIFAR-100 Phase-9i.
- **Epochs:** 100.
- **Hardware:** laptop RTX 4090 16 GB. bf16 AMP, batch 192 (the largest
  that fits ResNet-50 at 160² + FFCV in 16 GB VRAM), num_workers 0,
  OMP/MKL caps per CLAUDE.md Rule 26. Headline-mode seeding per D4.

## Decision rule (pre-committed)

- **IF** the prior arm's median top1 exceeds the baseline median by
  ≥ +0.5 pp AND the paired Wilcoxon p_one ≤ 0.05 at n=3 (with the
  acknowledgement that n=3 Wilcoxon floor is p=0.125, so the practical
  rule is "all 3 prior seeds beat their paired baseline seed" =
  sign-test α=0.125) **THEN** the prior TRANSFERS to ImageNet-100; the
  paper's headline becomes "iso-FLOPs nature-inspired prior lifts
  ResNet-50 on ImageNet-100 at modern recipe by Δ pp (n=3, sign test
  α=0.125, EVALUATION-tier via Wave-2 n=7 prior confirmation)."
- **IF** the prior arm's median lifts by < +0.5 pp OR loses to baseline
  on ≥ 1 seed **THEN** the prior DOES NOT TRANSFER to ImageNet-100;
  PAPER.md reframes around Wave-2 as the highest-resolution surviving
  claim, with an honest note that ImageNet-100 did not lift.
- **IF** the prior arm loses by ≥ -0.5 pp on median **THEN** the prior is
  FALSIFIED on ImageNet-100; the Wave-2 PROMOTE is reframed as
  "dataset-scale-specific lift up to 200-class."

## Statistical test

- **Test:** paired Wilcoxon signed-rank (n=3 floor p=0.125), paired
  permutation. Per CLAUDE.md Rule 35 n=3 is SCREENING-tier; here it is
  rebrandable as EVALUATION-of-TRANSFER ONLY BECAUSE Wave-2 n=7 already
  certified the prior at the smaller scale. This rebranding is
  pre-registered HERE and any future deviation is HARKing per Rule 36.
- **α:** 0.05 two-sided where achievable; otherwise the n=3 sign-test
  floor of 0.125 is reported as such.
- **Multiple-comparison correction:** k=1 (single prior under test);
  POSI does NOT apply because Wave-2 already paid the family selection
  cost.
- **Auxiliary:** 10⁴-resample BCa bootstrap 95% CI on Δmean (DiCiccio &
  Efron 1996 — bootstrap at n=3 under-covers true parameter by ~2×;
  disclosed per SYNTHESIS_100 C6).
- **Required n for stated power:** n=3 is TRANSFER-tier; n=5 would clear
  Wilcoxon p_one ≤ 0.03 if all 5 positive, but ImageNet-100 wall-clock
  blocks the extension within the 12-week budget. n=3 is the explicit
  compromise.

## Expected wall-clock

- 100 ep at 160² ResNet-50 + FFCV on the 4090 ≈ ~13 GPU-h per seed.
- 2 arms × 3 seeds × 13 GPU-h = 78 GPU-h.
- + 2 days of FFCV-pipeline setup (sister-repo parity per SYNTHESIS_100 B4).

## Pre-registered analysis plan

1. Compute per-(arm, seed) top1 at epoch 100. **Headline-mode seeded.**
2. Verify iso-FLOPs band at re-launch (both arms must land in
   [0.95, 1.05] × baseline FLOPs).
3. Compute paired Wilcoxon p_one at n=3 (with floor disclosure).
4. Compute BCa 95% bootstrap CI on Δmean (with under-coverage disclosure).
5. Compute the sign-test p (= 0.125 / 0.25 / 0.5 depending on positives).
6. Plot per-epoch top1 curves; verify EMA + non-EMA agreement (D6).
7. Report the verdict against the pre-committed decision rule.
8. NO post-hoc n-extension to n=4/5/6 — that would be sequential testing
   without α-spending correction (SYNTHESIS_100 C13).

## What would falsify this

1. The prior arm loses to baseline by ≥ -0.5 pp on median top1 — prior
   is refuted on ImageNet-100.
2. The prior arm wins by < +0.5 pp on median top1 — the lift collapses
   at ImageNet scale and the paper headline reverts to Wave-2 evidence.
3. The iso-FLOPs band check fails at re-launch (either arm > 1.05 of
   baseline FLOPs) — the comparison is malformed; experiment refused.
4. Either arm's training fails to reach the project's ImageNet-100
   pre-registered floor of top1 ≥ 0.78 at n=3 median — recipe debug
   reverts to Wave-0-equivalent at 160² before Wave-3 is retried.

## Cross-references

- SYNTHESIS_100.md Block B item [B4](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md)
- FALSIFIERS.md row F4
- CLAUDE.md Rules 28, 35, 36
- Tian 2020 arXiv:1906.05849 "Contrastive Multiview Coding" (ImageNet-100
  100-class subset)
- He 2016 arXiv:1512.03385 ResNet
- Leclerc 2022 arXiv:2110.01077 FFCV
- Sister `autoresearchimage` FFCV pipeline (Camelyon17 reference)
