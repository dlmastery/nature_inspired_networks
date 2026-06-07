# Pre-Registration — A4 Modern-Recipe Debug (He-2019 / Playbook CIFAR variant)

**Date filed:** 2026-06-06
**Commit hash to be filled at commit time:** (see git log post-launch)
**Status:** PRE-REGISTERED, no seeds run yet
**Cross-reference:** `audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md` Block A item A4

## Hypothesis under test

The Bello-2021/Wightman-2021 11-trick recipe at `configs/cifar100_modern_200ep.yaml`
is ImageNet-tuned and under-performs the project's own pre-registered floor of
0.68 on CIFAR-100 (current Phase-9i baseline median: 0.6350, 3.3 pp below floor).
R5 BLOCKER 2 identified the root causes: RandAugment(N=2, M=14) is the ImageNet
setting and too aggressive at 32×32; Random Erasing has near-zero effect at 32×32
(Müller & Hutter 2021); Mixup α=0.2 + CutMix α=1.0 simultaneously over-regularises.

The He-2019 / DeepLearningTuningPlaybook CIFAR-specific recipe should close
this gap. Predicted Δ vs current modern recipe: **+3-5 pp**.

## Recipe (exact, frozen as of commit time)

| Knob | Bello/Wightman (current) | He-2019/Playbook (this config) |
|---|---|---|
| Optimizer | AdamW | AdamW (same) |
| Scheduler | cosine | cosine (same) |
| Warmup epochs | 5 | **10** |
| Weight decay | 5e-4 | 5e-4 (same) |
| Label smoothing | 0.1 | 0.1 (same) |
| RandAugment (N, M) | (2, 14) | **(1, 9)** |
| Random Erasing p | 0.25 | **0.0 (DROPPED)** |
| Mixup α | 0.2 | **0.1** |
| CutMix α | 1.0 | **0.0 (DROPPED)** |
| Mixup/CutMix alternation prob | 0.5 | **1.0 (no CutMix)** |
| EMA decay | 0.9999 | 0.9999 (same) |
| Epochs | 200 | 200 (same) |
| Batch | 256 | 256 (same) |
| LR | 1e-3 | 1e-3 (same) |

## Architecture (frozen)

ResNet-20 (He 2016, the project baseline). Same as current modern recipe.
`params=278,324`, `flops=41,224,448`. The A3 FLOP-target check (synth) refuses
to launch if measured FLOPs deviate from `41224448 ± 10%`.

## Determinism (frozen)

`headline_mode: true` in YAML → runner sets `torch.use_deterministic_algorithms(True)`,
`cudnn.deterministic=True`, `cudnn.benchmark=False`, and installs
`worker_init_fn=seed_worker` on the DataLoader. Two seeded runs are bit-identical
(verified via `tests/test_headline_mode.py`).

## Seeds (pre-committed sequence)

1. **Seed 0** runs first (~3.5 GPU h).
2. Decision gate (see below) fires.
3. If gate PASS → seeds 1 + 2 launch sequentially (each ~3.5 GPU h).

Total expected GPU-h if gate PASSes: ~10.5 h.

## Decision rule (pre-committed, binding)

After seed 0 lands:

| Outcome | Action |
|---|---|
| **top1 ≥ 0.68** | **PROMOTE** — recipe clears PLAN.md floor. Launch seeds 1 + 2 for n=3. Replaces the current modern recipe for all subsequent waves. |
| **top1 ∈ [0.66, 0.68)** | **DIRECTIONAL** — recipe improved but not enough. Investigate one additional knob (e.g., LR=5e-4) before launching n=3. Document the next debug iteration in `pre-registration/a4_recipe_debug_v2.md` BEFORE running. |
| **top1 ∈ [0.64, 0.66)** | **WEAK** — small improvement. Halt automatic progression. Operator must explicitly authorize next step. Likely root cause is deeper than augmentation. |
| **top1 < 0.64** | **FALSIFIED** — He-2019 recipe variant is NOT the fix. The recipe-debug hypothesis is refuted at this knob set. Log the negative result in `paper/NEGATIVE_RESULTS.md` and re-investigate (LR schedule? optimizer choice? warmup curve?). |

A1 (iso-FLOPs prior re-test) is **blocked on PROMOTE**. Priors must not be re-run
until the baseline clears the floor — that's the entire point of Block A.

## Statistical analysis (post-n=3, pre-committed)

- Report: top1 mean, σ, paired-t vs `experiments_modern/cifar100/baseline_resnet20_seed{0,1,2}` (the legacy modern baseline).
- Test: paired Wilcoxon (n=3, will hit floor p=0.125 — that's a sign test).
- Bootstrap: skipped at n=3 (synth C6).
- Conclusion language: "directional lift confirmed at n=3 sign test; n≥7 confirmation deferred to Wave-2 modern-recipe Tiny-ImageNet cell."

## Expected wall-clock

| Phase | GPU-h |
|---|---|
| Seed 0 (decision-gate) | ~3.5 |
| Seeds 1 + 2 (if PROMOTE) | ~7.0 |
| **Total if PROMOTE** | **~10.5** |
| **Total if FALSIFIED at seed 0** | **~3.5** |

## What would falsify this

(Closes synth C15 row F1.)

The recipe-debug hypothesis is **refuted** if seed 0 top1 < 0.64 AT THE SAME LR
+ optimizer + epochs. The hypothesis is **partially refuted** if seed 0 lands in
[0.64, 0.66): then dropping Bello tricks helps but doesn't close the gap, and
the modern recipe vs CIFAR mismatch has a deeper cause.

## What this does NOT test

- LR schedule shape (cosine vs cosine-restart vs OneCycle).
- Optimizer choice (AdamW vs SGD-with-momentum vs LAMB).
- Architecture knobs (depth, width, stage count).
- Different RandAugment policies (TrivialAugment, AugMix).

If A4 falsifies, the next pre-registered debug iteration (A4-v2) will pick ONE of
the above to investigate, not all of them.

## Cross-references

- `audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md` Block A item A4 (BLOCKER)
- `audits/REVIEWER_FIVE_2026-06-06/05_BigTechLabLeader.md` BLOCKER 2
- `convergence/PLAN.md` line 117 (the 0.68 floor that the current recipe missed)
- `paper/FALSIFIERS.md` row F1 (this binds the falsification condition)
- `tests/test_modern_recipe.py` (recipe primitives' regression tests)
- `tests/test_headline_mode.py` (determinism plumbing)
