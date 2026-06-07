# Wave-0 Pre-Registration — Imagenette recipe validation

**Date filed:** 2026-06-06
**Commit hash:** _to be filled at commit time and propagated to PAPER.md §5 before any seed launches_
**Status:** PRE-REGISTERED (no seeds run yet)
**SYNTHESIS_100.md item:** B1
**Estimated wall-clock:** ~5 GPU-h on RTX 4090 Laptop, 1 calendar week

## Hypothesis under test

The "modern 11-trick" recipe (AdamW + cosine + linear warmup + WD +
label-smoothing + stochastic depth + Mixup α=0.2 + CutMix α=1.0 + RandAugment
N=2 M=14 + Random Erasing p=0.25 + EMA 0.9999) that
[`configs/cifar100_modern_200ep.yaml`](../configs/cifar100_modern_200ep.yaml)
encodes is ImageNet-tuned (Bello 2021 arXiv:2103.07579 + Wightman 2021
arXiv:2110.00476). [R5 BLOCKER 2 / synthesis A4] reports the recipe
landed at median CIFAR-100 top1 = 0.6360 — below the project's own
pre-registered floor of 0.68 (`convergence/PLAN.md:117`). Wave-0 tests
whether the recipe transfers to Imagenette (10 classes, 13k images, native
160×160) BEFORE any nature-inspired prior result is interpreted on top of
it. If the recipe transfers to Imagenette but fails on CIFAR-100, the
defect is in the recipe-to-32×32 transfer (Müller & Hutter 2021), not in
the modern-recipe family itself.

## Recipe (exact)

- **Architecture:** ResNet-20-Imagenette (3 stages × 3 BasicBlocks, channels
  64 → 128 → 256, head global-avg-pool + linear-1000). Width is scaled
  up vs the CIFAR ResNet-20 to keep parameters ~3 M, comparable to
  literature Imagenette small-model results.
- **Dataset:** Imagenette v2-160 (https://github.com/fastai/imagenette).
  Train/val split as published. Image side 160×160; train aug crop=160.
- **Augmentation:** three recipe variants, each pre-registered.
  - `recipe_legacy` — RandomCrop(160, padding=4) + HFlip + Normalize ONLY.
  - `recipe_modern_naive` — the 11-trick CIFAR-100 recipe as-is
    (RandAugment N=2 M=14, Random Erasing p=0.25, Mixup α=0.2, CutMix α=1.0,
    EMA 0.9999).
  - `recipe_modern_cifar_tuned` — He-2019 / Tuning-Playbook CIFAR-tuned
    variant per SYNTHESIS_100.md item A4: RandAugment N=1 M=9, no Random
    Erasing, Mixup α=0.1, EMA 0.9999, warmup 5 ep.
- **Optimizer:** AdamW, lr=5.0e-4, wd=5.0e-4, β=(0.9, 0.999).
- **Scheduler:** cosine, warmup 5 ep, 10 ep total.
- **Seeds:** 0, 1, 2, 3, 4 (n=5).
- **Epochs:** 10 (recipe screening — purpose is to identify the working
  recipe, not to converge to absolute SOTA).
- **Hardware:** laptop RTX 4090 16 GB. bf16 AMP, batch 256, num_workers 0,
  OMP/MKL caps per CLAUDE.md Rule 26.

## Decision rule (pre-committed)

The Imagenette small-model published band at 10 epochs is ≥0.90 top1
(fast.ai leaderboard, ResNet-style models). Decision rule, evaluated on the
median top1 across the 5 seeds per recipe:

- **IF** any recipe lands median top1 ≥ 0.90 **THEN** that recipe is
  PROMOTED to Wave-1 as the working modern recipe; pre-register its name
  in `wave1_imagenette_iso_flops_pareto.md` before any Wave-1 seed launches.
- **IF** the best of three lands in [0.85, 0.90) **THEN** the recipe is
  CONDITIONAL — proceed to Wave-1 with a 1-paragraph caveat in PAPER.md
  noting Wave-0 recipe sat below the small-model band.
- **IF** all three land below 0.85 **THEN** the recipe is REFUSED.
  Pivot to a deeper recipe debug (RandAugment magnitude sweep at
  fixed N=1; Mixup α sweep ∈ {0.0, 0.1, 0.2, 0.4}). No Wave-1 launches.

## Statistical test

- **Test:** none required at Wave-0; this is recipe screening, not a
  paired claim. Per-recipe median + IQR reported.
- **α:** N/A.
- **Multiple-comparison correction:** N/A (the three recipes are not a
  hypothesis family — Wave-0 is descriptive).
- **Required n for stated power:** n=5 per recipe is sufficient for the
  median + IQR descriptive estimate; the IQR is reported as the
  Wave-0 empirical noise band per CLAUDE.md Rule 35.

## Expected wall-clock

- ~20 min per seed × 5 seeds × 3 recipes = ~5 GPU-h total.

## Pre-registered analysis plan

1. Compute per-seed top1 at epoch 10 for each recipe.
2. Report median ± IQR per recipe.
3. Apply the decision rule above to identify the working recipe.
4. The working recipe's YAML is committed as
   `configs/imagenette_modern_10ep.yaml` BEFORE any Wave-1 launch.
5. No post-hoc retuning. If the rule's CONDITIONAL or REFUSED branch fires,
   the recipe debug is itself a new pre-registration
   (`wave0b_recipe_debug.md`), not an inline extension of Wave-0.

## What would falsify this

Three results that would refute the "modern recipe transfers" claim
within Wave-0's scope:

1. All 3 recipes land < 0.85 top1 at Imagenette 10 ep n=5 — the modern
   11-trick recipe does NOT transfer at 160×160.
2. `recipe_modern_naive` lands ≥ 0.90 but `recipe_modern_cifar_tuned`
   lands < 0.85 — the CIFAR-targeted retuning (A4) is the wrong direction.
3. `recipe_legacy` Pareto-dominates both modern recipes at Imagenette
   — the modern recipe is not the right baseline for this benchmark class.

## Cross-references

- SYNTHESIS_100.md Block B item [B1](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md)
- FALSIFIERS.md row F1
- CLAUDE.md Rules 13, 26, 35, 36
- Bello 2021 arXiv:2103.07579 "Revisiting ResNets"
- Wightman 2021 arXiv:2110.00476 "ResNet Strikes Back"
- Müller & Hutter 2021 NeurIPS "TrivialAugment"
- Imagenette v2-160 https://github.com/fastai/imagenette
