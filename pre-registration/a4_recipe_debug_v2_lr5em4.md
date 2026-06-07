# Pre-Registration — A4-v2 Modern-Recipe Debug, single-knob LR halve

**Date filed:** 2026-06-07
**Triggering result:** A4-v1 seed 0 top1 = 0.6747 (DIRECTIONAL per `pre-registration/a4_recipe_debug_he2019.md` decision rule; ∈ [0.66, 0.68))
**Commit hash to be filled at commit time.**
**Status:** PRE-REGISTERED, no seeds run yet
**Cross-reference:** `audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md` Block A item A4, `pre-registration/a4_recipe_debug_he2019.md` (this is the v2 follow-up explicitly pre-committed by v1)

## Result that triggered this v2

A4-v1 seed 0 (He-2019/Playbook CIFAR recipe, single seed, full 200 ep):
- top1 = **0.6747** ← landed in DIRECTIONAL band [0.66, 0.68)
- train_top1_clean = 0.7773
- generalization_gap = 0.103 (clean train top-1 minus test top-1) — material overfitting
- composite = 0.5886 (FLOPs-extended fingerprint `b73e8bbf…` ✓)
- FLOPs check PASSED (measured 41.22 M, target 41.22 M)
- train_seconds = 12114 (3.37 h, on plan)

The recipe-debug lifted +3.97 pp (0.635 → 0.6747) — a substantial directional confirmation that the He-2019 augmentation set transfers better to CIFAR-100/ResNet-20. The remaining 0.5 pp gap to PROMOTE could be:
- LR too high — cosine annealing arrives at a noisy minimum
- Optimizer choice (AdamW + small ResNet sometimes underperforms SGD)
- Insufficient regularization given the gap=10pp overfitting signal

Per the v1 pre-registration's binding rule, ONE additional knob is investigated before n=3 launch.

## The single knob to change

**LR: 1.0e-3 → 5.0e-4** (halve).

Synthesis A4 wave explicitly pre-committed this as the first fallback. Rationale: at AdamW + cosine on small ResNets, a 2× lower peak LR widens the convergence basin (Loshchilov & Hutter 2019 §5; Wightman 2021 §3.2 reports CIFAR-100 ResNet-20 prefers lr=5e-4 at AdamW + 200 ep). Everything else stays at the A4-v1 He-2019 recipe values.

| Knob | A4-v1 (DIRECTIONAL 0.6747) | A4-v2 (this config) |
|---|---|---|
| Optimizer | AdamW | AdamW (same) |
| Scheduler | cosine | cosine (same) |
| **LR (peak)** | **1.0e-3** | **5.0e-4** |
| Warmup epochs | 10 | 10 (same) |
| Weight decay | 5e-4 | 5e-4 (same) |
| Label smoothing | 0.1 | 0.1 (same) |
| RandAugment (N, M) | (1, 9) | (1, 9) (same) |
| Random Erasing p | 0.0 | 0.0 (same) |
| Mixup α | 0.1 | 0.1 (same) |
| CutMix α | 0.0 | 0.0 (same) |
| EMA decay | 0.9999 | 0.9999 (same) |
| Epochs | 200 | 200 (same) |
| Batch | 256 | 256 (same) |
| headline_mode | true | true (same) |
| flops_target | 41224448 | 41224448 (same) |

## Decision rule (pre-committed, binding)

After A4-v2 seed 0 lands:

| Outcome | Action |
|---|---|
| **top1 ≥ 0.68** | **PROMOTE** — LR halving was the missing piece. Launch v2 seeds 1+2 for n=3. v2 recipe becomes the modern recipe for all subsequent waves. |
| **top1 ≥ A4-v1 + 0.005 (≥ 0.6797)** | **STRONG DIRECTIONAL** — even though below 0.68, the additional ~0.5 pp lift means LR halving is a real signal. Pre-register A4-v3 with one MORE knob: SGD with momentum 0.9 (operator may also greenlight launching v2 seeds 1+2 to capture the lift). |
| **top1 ∈ [A4-v1 − 0.005, A4-v1 + 0.005] (i.e., ∈ [0.6697, 0.6797))** | **NULL** — LR halving did nothing material. Recipe-debug strategy is hitting diminishing returns on augmentation/LR. Pivot to A4-v3 with optimizer change (SGD + momentum) instead. |
| **top1 < 0.6697 (below A4-v1 by ≥0.005)** | **REGRESSION** — LR halving HURT. Stop the recipe-debug iteration. Either (a) accept A4-v1 at 0.6747 as the practical baseline and proceed to A1 with this slightly-below-floor baseline + an honest disclosure, or (b) pivot to a fundamentally different debug (architecture / optimizer family). |

A1 (iso-FLOPs prior re-test) remains BLOCKED on PROMOTE or operator override.

## Seeds (pre-committed sequence)

1. Seed 0 runs first (~3.5 GPU h).
2. Decision gate fires.
3. If PROMOTE: seeds 1+2 sequentially (each ~3.5 h).

Total if PROMOTE: ~10.5 GPU h.

## What would falsify

If A4-v2 seed 0 lands in REGRESSION (<0.6697), LR halving is refuted as the missing piece and the recipe-debug strategy moves to optimizer-family exploration next.

If A4-v2 seed 0 PROMOTEs (≥0.68), LR halving is the confirmed missing piece — A4-v1 alone wasn't enough but the He-2019 augmentation set + halved LR closes the floor.

## Cross-references

- `pre-registration/a4_recipe_debug_he2019.md` (the v1 that triggered v2)
- `audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md` A4
- `paper/FALSIFIERS.md` row F1
- `experiments_modern_debug/cifar100/baseline_resnet20_he2019_debug_seed0/metrics.json` (the triggering result)
