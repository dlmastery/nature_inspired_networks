# Pre-registration

This directory contains commit-hashed, time-stamped pre-registrations
for every experiment whose result is intended for an externally-facing
artefact (PAPER.md, FINDINGS.md, README badge, dashboard headline).

## Why pre-registration

[CLAUDE.md Rule 36](../CLAUDE.md) requires that the classification of any
sweep row as SCREENING vs EVALUATION be pre-registered BEFORE the sweep
runs, with the commit hash referenced in the resulting paper / FINDINGS
entry. Post-hoc reclassification of a row as "screening" after seeing
it lose is HARKing (Hypothesizing After Results are Known) and is a
BLOCKER-level finding under the project's audit doctrine.

The 2026-06-06 reviewer-five synthesis
([`audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md`](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md))
elevates pre-registration from a CLAUDE.md rule to a paper-level
contract — every external reviewer reads the Phase-9 cascade as
"garden of forking paths" (Gelman & Loken 2013) and only commit-hashed
pre-registrations close that critique.

## File format

Each `wave*.md` follows the template in
[`SYNTHESIS_100.md`](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md)
Agent C section: hypothesis, exact recipe, pre-committed decision rule,
statistical test + α + correction, expected wall-clock, pre-registered
analysis plan, falsification conditions, cross-references.

Once a wave's seeds finish, the document is appended (NEVER edited —
CLAUDE.md Rule 3) with a `## Result` block carrying the seed numbers,
the resolved verdict against the pre-committed decision rule, and the
commit hash of the merged result.

## Index

- [`wave0_imagenette_recipe_validation.md`](wave0_imagenette_recipe_validation.md)
  — validate the modern recipe transfers to Imagenette before any prior claim (~5 GPU-h).
- [`wave1_imagenette_iso_flops_pareto.md`](wave1_imagenette_iso_flops_pareto.md)
  — iso-FLOPs Pareto frontier across ResNet-20 / RegNetX-200MF /
  ConvNeXt-V2-Femto / ViT-S/16 on Imagenette (~50 GPU-h).
- [`wave2_tiny_imagenet_iso_flops.md`](wave2_tiny_imagenet_iso_flops.md)
  — Tiny-ImageNet 200-class iso-FLOPs comparison, baseline + 3 priors
  at n=5 paired Wilcoxon + Holm-Bonferroni (~80 GPU-h).
- [`wave3_imagenet100_ffcv.md`](wave3_imagenet100_ffcv.md)
  — ImageNet-100 FFCV at 160² for the Wave-2 winning prior + ResNet-50
  baseline (~80 GPU-h).
- [`wave4_h71_spherical_mnist.md`](wave4_h71_spherical_mnist.md)
  — H71 IcosaRoPE3D on Spherical MNIST 60×60, rotated test set, n=5
  paired vs non-equivariant ViT-Tiny (~15 GPU-h).

All five waves are PRE-REGISTERED but UNRUN as of 2026-06-06.

## Cross-references

- Falsifier contract: [`paper/FALSIFIERS.md`](../paper/FALSIFIERS.md)
- Statistical-rigor floor: [`paper/STATISTICAL_TESTS.md`](../paper/STATISTICAL_TESTS.md)
- Hardware contract: [`CLAUDE.md`](../CLAUDE.md) §2 and Rule 26
- 12-week plan: [`SYNTHESIS_100.md`](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md) §10
