# Wave-1 Pre-Registration — Imagenette iso-FLOPs Pareto frontier

**Date filed:** 2026-06-06
**Commit hash:** _to be filled at commit time and propagated to PAPER.md §5 before any seed launches_
**Status:** PRE-REGISTERED (no seeds run yet)
**Depends on:** Wave-0 PROMOTE or CONDITIONAL outcome
**SYNTHESIS_100.md items:** B2 + B12
**Estimated wall-clock:** ~50 GPU-h on RTX 4090 Laptop, 2 calendar weeks

## Hypothesis under test

Nature-inspired priors Pareto-dominate the modern-architecture literature
on Imagenette at iso-FLOPs. Specifically: the H09 φ-budget allocation
(w_m=φ≈1.618) Pareto-dominates Radosavovic 2020 RegNetX's published
optimum (w_m∈[2.5, 2.9]) at the same FLOP budget on Imagenette top1.
SYNTHESIS_100.md item B12 frames this as the H09-vs-RegNet honest claim;
SYNTHESIS_100.md item B2 establishes the 4-architecture iso-FLOPs grid
that lets a reviewer judge whether the prior Pareto-dominates the
literature, not just the project's own ResNet-20 baseline.

## Recipe (exact)

- **Architectures (4):** all pinned to ~1.0 G FLOPs at 160×160 input.
  - `resnet20_imagenette_w160` — the project's ResNet-20-Imagenette,
    width-scaled to ~1.0 G FLOPs at 160².
  - `regnetx_200mf` — Radosavovic 2020 RegNetX-200MF (literature anchor
    for H09); `timm` implementation at default config (~200 MF FLOPs at
    224², so we use the `_w50` width-multiplied variant to land at 1.0 G
    at 160² — exact multiplier pre-committed before first launch).
  - `convnextv2_femto` — Woo 2023 ConvNeXt-V2-Femto, `timm` implementation,
    width-scaled to ~1.0 G at 160².
  - `vit_s16` — Dosovitskiy 2021 ViT-Small/16, `timm` implementation,
    patch=16, 6 layers, embed_dim=192, heads=3, width-scaled to ~1.0 G
    at 160².
- **FLOPs measurement:** `fvcore.FlopCountAnalysis` at input shape
  (1, 3, 160, 160). Pre-committed iso-FLOPs band [0.95, 1.05] G FLOPs.
  Any architecture that cannot be width-pinned into the band is DROPPED
  from Wave-1, not silently re-pinned.
- **Dataset:** Imagenette v2-160. Same train/val split as Wave-0.
- **Augmentation:** the Wave-0 PROMOTED recipe, byte-identical YAML.
- **Optimizer:** AdamW, lr=5.0e-4, wd=5.0e-4.
- **Scheduler:** cosine, warmup 5 ep, 50 ep total.
- **Seeds:** 0, 1, 2, 3, 4 (n=5).
- **Epochs:** 50.
- **Hardware:** laptop RTX 4090 16 GB. bf16 AMP, batch 256, num_workers 0,
  OMP/MKL caps per CLAUDE.md Rule 26.

## Decision rule (pre-committed)

The honest comparison is the iso-FLOPs Pareto frontier on the
(top1_median, params_M) plane:

- **IF** ResNet-20-Imagenette + H09 φ-budget allocation lies on the
  Pareto frontier AND Pareto-dominates RegNetX-200MF in BOTH median top1
  AND params (n=5 paired Wilcoxon p_one < 0.0167 = 0.05/3 with Holm
  k=3 over {RegNet, ConvNeXt-V2-Femto, ViT-S/16}) **THEN** H09 is
  PROMOTED to a Wave-1 surviving nature-inspired claim.
- **IF** H09 is Pareto-comparable (within the empirically-derived noise
  band per CLAUDE.md Rule 35) to RegNetX-200MF **THEN** H09 is
  CONDITIONAL — reported as "rediscovery of RegNet's Pareto region with
  exact φ-ratio as the constrained allocation rule." Wave-2 still runs.
- **IF** RegNetX-200MF Pareto-dominates H09 by ≥ 0.5 pp at iso-FLOPs
  (n=5 paired) **THEN** H09 is FALSIFIED on Imagenette and is dropped
  from the Wave-2 hypothesis set.

The decision rule for the other two priors (`pair_gm_pdw`, `slot_act_sine`)
follows the same template, evaluated independently with Holm k=3
correction across the three priors.

## Statistical test

- **Test:** paired Wilcoxon signed-rank on the 5 seed-paired top1 deltas
  between each prior and the strongest non-prior baseline at iso-FLOPs
  (= the best of {RegNetX-200MF, ConvNeXt-V2-Femto, ViT-S/16}).
- **α:** 0.05 two-sided, one-sided used only when the pre-registered
  prediction is directional (φ-budget prediction is directional → +Δ).
- **Multiple-comparison correction:** Holm-Bonferroni at k=3 (3 priors).
- **Auxiliary:** 10⁴-resample BCa bootstrap 95% CI on the pp delta per
  CLAUDE.md Rule 35.
- **Required n for stated power:** n=5 is the Wilcoxon minimum that lets
  a one-sided test land at p ≤ 0.03 with all 5 seeds positive; this is
  SCREENING-tier statistics, EVALUATION-tier (n≥7) carries forward into
  Wave-2 for any Wave-1 SURVIVOR.

## Expected wall-clock

- 50 ep at ~1 G FLOPs and batch 256 on the 4090 ≈ ~2.5 GPU-h per seed.
- 4 architectures × 5 seeds × 2.5 GPU-h = 50 GPU-h.

## Pre-registered analysis plan

1. Compute per-seed top1 at epoch 50 for each (architecture, seed) cell.
2. Compute per-architecture median top1 + 95% bootstrap CI.
3. Compute per-prior paired Wilcoxon vs the strongest non-prior baseline.
4. Apply Holm-Bonferroni k=3 to the three prior p-values.
5. Plot the iso-FLOPs Pareto frontier on (top1_median, params_M) and on
   (top1_median, FLOPs_M) — the second plot verifies the iso-FLOPs
   pinning is faithful.
6. Tabulate the decision-rule outcome per prior.
7. No post-hoc reclassification of SCREENING ↔ EVALUATION.

## What would falsify this

1. RegNetX-200MF Pareto-dominates H09 φ-budget by ≥ 0.5 pp at iso-FLOPs
   (n=5 paired) — H09 is refuted on Imagenette.
2. None of the 3 priors clear Holm-Bonferroni at α=0.05 — the iso-FLOPs
   nature-inspired claim does not survive contact with modern architectures
   on Imagenette.
3. Any architecture cannot be width-pinned into [0.95, 1.05] G FLOPs —
   the iso-FLOPs comparison is malformed and Wave-1 reverts to a
   recipe / architecture redesign.

## Cross-references

- SYNTHESIS_100.md Block B items
  [B2](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md) and
  [B12](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md)
- FALSIFIERS.md row F2
- CLAUDE.md Rules 28, 35, 36
- Radosavovic 2020 arXiv:2003.13678 "Designing Network Design Spaces" (RegNet)
- Woo 2023 arXiv:2301.00808 "ConvNeXt V2"
- Dosovitskiy 2021 arXiv:2010.11929 "ViT"
- `timm` library (Wightman 2019) for the three literature architectures
- fvcore for FLOPs measurement (https://github.com/facebookresearch/fvcore)
