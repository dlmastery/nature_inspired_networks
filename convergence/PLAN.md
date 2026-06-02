# Convergence-regime sweep PLAN — modern 11-trick recipe

> Drafted 2026-06-01 as the laptop-realistic response to PhD-critique
> items **B** ("modern-recipe baseline at convergence") and **2**
> ("convergence-regime re-test of winners"). NOT yet launched —
> user authorisation gated. The plan deliberately stops short of
> full server-grade ImageNet sweeps; everything below fits on the
> 4090 Laptop (16 GB) over 3 overnight sessions.

## 0. Scope and motivation

The 30-epoch CIFAR-100 numbers reported in `FINDINGS.md` were collected
under a legacy recipe: AdamW + cosine + label smoothing + RandomCrop +
HFlip only. PhD-critique items B and 2 argue, correctly, that this is
NOT the convergence regime — Bello 2021 (Revisiting ResNets) and
Wightman 2021 (ResNet Strikes Back) show that 80-95% of the absolute
top-1 lift available to a CIFAR-scale ResNet comes from the 11-trick
modern recipe at 200+ epochs, not from architecture changes. A
winning hypothesis at 30 ep over a weak baseline could simply be the
baseline-too-weak artefact. The fix is to re-test the three top
hypotheses against a modern-recipe baseline that has had enough
epochs to actually converge.

The 11 tricks (Bello 2021):

| # | Trick                  | Wiring point                          |
|---|------------------------|---------------------------------------|
| 1 | AdamW                  | `train.Trainer._build_optimizer`      |
| 2 | Cosine LR              | `train._build_scheduler`              |
| 3 | Linear LR warmup       | `train._build_scheduler` (new)        |
| 4 | Weight decay 5e-4      | `TrainConfig.weight_decay`            |
| 5 | Label smoothing 0.1    | `TrainConfig.label_smoothing`         |
| 6 | Stochastic depth       | `drop_path.FractalDropPath` (H52)     |
| 7 | Mixup α=0.2            | `mixup.mixup_batch` (NEW)             |
| 8 | CutMix α=1.0           | `cutmix.cutmix_batch` (NEW)           |
| 9 | RandAugment (2, 14)    | `randaugment.build_randaugment` (NEW) |
|10 | Random Erasing p=0.25  | `random_erasing.build_random_erasing` (NEW) |
|11 | EMA decay 0.9999       | `ema.ModelEMA` (NEW)                  |

All five NEW pieces ship with unit tests in `tests/test_modern_recipe.py`
(see Phase 0 below).

## 1. Hardware budget (CLAUDE.md §2 + Rule 26)

* RTX 4090 Laptop, 16 GB VRAM, Windows 11
* bf16 AMP, batch 256, `num_workers=0`
* Environment caps before EVERY long run:

  ```powershell
  $env:KMP_DUPLICATE_LIB_OK = "TRUE"
  $env:OMP_NUM_THREADS = 2
  $env:MKL_NUM_THREADS = 2
  ```

* Auto-checkpoint loop (Rule 20) must be active alongside any run >15 min.

Wall-clock budget per run on the 4090 Laptop (ResNet-20, bf16, batch 256):

| Run length | Per-epoch  | Total (1 seed) | Total (3 seeds) |
|------------|-----------:|---------------:|----------------:|
| 50 ep smoke| ~110 s     | ~1.5 GPU h     | ~4.5 GPU h      |
| 200 ep     | ~115 s     | ~6.5 GPU h     | ~19.5 GPU h     |

The 11-trick recipe adds ~10-12% per-step overhead vs. legacy (RandAugment
on PIL + Mixup batch ops + EMA shadow update). The 4090 Laptop's
thermal envelope sustains ~95% peak utilisation indefinitely with the
laptop on AC and the OMP/MKL caps in place (Rule 26).

## 2. Wave matrix

### Wave-A — smoke validation (~3 GPU h, ~2 overnight)

Goal: confirm the 11-trick recipe wires correctly end-to-end and pick
the trick subset that the laptop budget supports. Uses the 50-epoch
smoke config (`configs/cifar100_modern_smoke.yaml`).

| Tag                                  | Run length | Goal |
|--------------------------------------|-----------:|------|
| `baseline_resnet20_modern_smoke`     | 50 ep      | Full 11-trick recipe; expected top1 ∈ [0.62, 0.68]. |
| `baseline_resnet20_legacy_smoke`     | 50 ep      | Legacy recipe (no mix, no RA/RE/EMA); expected top1 ∈ [0.55, 0.61]. |
| `baseline_resnet20_5trick_smoke`     | 50 ep      | AdamW + cosine + warmup + LS + RA only (drop mix + RE + EMA). Cheaper recipe sanity. |
| `pair_gm_pdw_modern_smoke`           | 50 ep      | Winner #1 under modern recipe. |
| `slot_act_sine_modern_smoke`         | 50 ep      | Winner #2 under modern recipe. |
| `sg_only_phi_budget_modern_smoke`    | 50 ep      | Winner #3 under modern recipe. |

Decision tree (post-Wave-A):

1. If `baseline_resnet20_modern_smoke` < 0.62 → wiring bug. Inspect
   `experiments/cifar100/baseline_resnet20_modern_smoke_seed0/history.json`
   and the `test_top1_ema` column. STOP and debug; do NOT proceed.
2. If `baseline_resnet20_modern_smoke` ≥ 0.62 AND
   `baseline_resnet20_legacy_smoke` < `baseline_resnet20_modern_smoke`
   by ≥ 4 pp → recipe is doing real work. Proceed to Wave-B.
3. If the gap is < 4 pp → either the smoke is too short (try 80 ep) or
   the recipe doesn't help this regime. Investigate before Wave-B.

CLI (one tag at a time; serial on the single GPU):

```powershell
.\.venv\Scripts\python -m nature_inspired_networks.runner `
  --config configs\cifar100_modern_smoke.yaml `
  --tag baseline_resnet20_modern_smoke --seed 0 `
  --root experiments
```

### Wave-B — 200-epoch modern-recipe baseline (~6 GPU h, n=3)

Goal: establish the convergence-regime baseline. Three seeds for the
error bar that any external claim requires (Rule 19 Phase 5).

| Tag                                | Seeds | Wall-clock |
|------------------------------------|-------|-----------:|
| `baseline_resnet20_modern_200ep`   | 0,1,2 | ~19.5 GPU h |

Expected pass band: **top1 ∈ [0.68, 0.72] median**. Honest estimate
based on Bello 2021 Table 3 (ResNet-50 at 350 ep on ImageNet shows
+5.8 pp from 11 tricks; CIFAR ResNet-20 sees a comparable absolute lift
because the underlying recipe-gap is larger on shorter runs). The
literature reports ResNet-20-style nets on CIFAR-100 at ~0.66-0.68
under the legacy recipe and ~0.71-0.73 under the He-2019 / Bello-2021
modern recipe; ResNet-20 is small enough that EMA + Mixup may give
slightly less lift than ResNet-50, hence the [0.68, 0.72] band.

CLI:

```powershell
foreach ($seed in 0,1,2) {
  $env:KMP_DUPLICATE_LIB_OK = "TRUE"
  $env:OMP_NUM_THREADS = 2
  $env:MKL_NUM_THREADS = 2
  .\.venv\Scripts\python -m nature_inspired_networks.runner `
    --config configs\cifar100_modern_200ep.yaml `
    --tag baseline_resnet20_modern_200ep --seed $seed `
    --root experiments
}
```

Decision tree (post-Wave-B):

1. If Wave-B median top1 > 0.70 → modern recipe wires correctly AND
   the baseline is competitive with the literature. Wave-C is
   meaningful: any winner that beats this baseline by > 0.5 pp at
   convergence is a real result.
2. If Wave-B median top1 ∈ [0.68, 0.70] → recipe works but is
   right at the bottom of the expected band. Wave-C still meaningful
   but the burden of proof on winners is higher (need ≥ 1.0 pp lift to
   be confident over the seed-noise floor).
3. If Wave-B median top1 < 0.68 → recipe is under-performing relative
   to the literature. Debug BEFORE Wave-C (it would be wasteful to
   compare winners against a weak baseline — the same trap critique
   item B was warning about).

### Wave-C — 200-epoch modern-recipe winners (~12 GPU h, 3 winners × n=1 seed initial)

Goal: do the three winners lift over the modern-recipe baseline at
convergence?

| Tag                                   | Seeds (initial) | Override snippet |
|---------------------------------------|-----------------|------------------|
| `pair_gm_pdw_modern_200ep`            | 0               | `model: NaturePrior`, `channel_mode: fib`, `momentum_schedule: golden`, `phi_decay_wd: true`, `phi_decay_base: 5.0e-4`, all `flags: false` |
| `slot_act_sine_modern_200ep`          | 0               | `model: resnet20`, `channel_mode: linear`, `slot_activation: sine`, `omega_init: 1.0`, all `flags: false` |
| `sg_only_phi_budget_modern_200ep`     | 0               | `model: phi_budget`, `channel_mode: phi`, `phi_budget_total: 16`, `phi_budget_n_stages: 4`, `phi_budget_mode: float` |

Per Rule 19 Phase 5, **only winners that retain ≥ 0.5 pp lift over Wave-B
get the n=3 seed re-run** (additional ~12 GPU h per winner). Cheap to
test, expensive to confirm — start with 1 seed each.

Wall-clock (1 seed × 3 winners): ~19.5 GPU h. n=3 confirmation for 1 lifting
winner: ~13 GPU h additional.

CLI pattern (per winner; override the variant fields inline via a
per-tag YAML overlay if the existing build_matrix dispatcher doesn't
already know the tag — see `scripts/run_sweep.py` for the override
table). Example for `pair_gm_pdw`:

```powershell
# Create a one-shot overlay config from cifar100_modern_200ep.yaml +
# the pair_gm_pdw override stanza, then launch:
.\.venv\Scripts\python -m nature_inspired_networks.runner `
  --config configs\cifar100_modern_200ep_pair_gm_pdw.yaml `
  --tag pair_gm_pdw_modern_200ep --seed 0 `
  --root experiments
```

Decision tree (post-Wave-C):

| Outcome                                          | Action |
|--------------------------------------------------|--------|
| Winner ≥ +0.5 pp over Wave-B median              | n=3 confirmation seeds; update FINDINGS.md headline. |
| Winner within ±0.5 pp of Wave-B                  | Verdict: NEUTRAL at convergence — the original 30-ep "win" was likely a baseline-too-weak artefact. Update FINDINGS.md with the post-fix vs convergence-regime table. |
| Winner < Wave-B − 0.5 pp                         | Verdict: REGRESSION at convergence — the prior was masking under-fitting; remove from headline claims. |

### Wave-D (optional) — multi-benchmark expansion (~15 GPU h)

Goal: address critique item 6 (multi-benchmark). Test Wave-C winners
on Imagenette (10-class, 32×32) and Tiny-ImageNet-200 (200-class,
64×64). NOT in the initial authorisation request — recommend gating on
Wave-C results.

| Tag                                              | Dataset         | Approx GPU-h |
|--------------------------------------------------|-----------------|-------------:|
| `baseline_resnet20_modern_imagenette_200ep`      | imagenette-160  | ~3 GPU h     |
| `<winner>_modern_imagenette_200ep` × N           | imagenette-160  | ~3 GPU h × N |
| `baseline_resnet20_modern_tinyimagenet_100ep`    | tiny-imagenet   | ~6 GPU h     |
| `<winner>_modern_tinyimagenet_100ep` × N         | tiny-imagenet   | ~6 GPU h × N |

Requires adding `imagenette` and `tinyimagenet` loaders to `data.py`
(separate task, ~half day) — NOT in this plan.

## 3. Total budget

| Wave | Hours | Seeds | Coverage |
|------|------:|-------|----------|
| A    | ~3    | 1     | 6 smoke configs at 50 ep                          |
| B    | ~19.5 | 3     | modern baseline at convergence                    |
| C    | ~19.5 | 1     | 3 winners under modern recipe                     |
| C2 (if winners confirmed) | ~13 | 2 more | n=3 on lifting winners                |
| **A+B+C**     | **~42 GPU h** | — | core critique B + 2 answer |
| D (optional)  | +~15-30 GPU h | — | multi-benchmark expansion        |

3 overnight sessions on the laptop = ~24 GPU h. Wave-A+B can run in
the first overnight + the next afternoon; Wave-C consumes the second
overnight; Wave-C2 confirmation seeds the third overnight.

## 4. Pre-flight gate (Rule 13)

Before launching Wave-A, the existing CIFAR-10 SOTA-smoke MUST still
pass on the current head:

```powershell
.\.venv\Scripts\python -m nature_inspired_networks.runner `
  --config configs\cifar10_sota_smoke.yaml --tag smoke --seed 0
```

Expected: top1 ≥ 0.80 at 12 ep. The modern-recipe changes preserve the
legacy path byte-for-byte when `mixup_alpha=cutmix_alpha=ema_decay=warmup_epochs=randaugment_n=random_erasing_p=0`,
so the SOTA-smoke (which sets none of these) must continue to land in
the expected band.

The new `tests/test_modern_recipe.py` provides 8 regression tests
(including a `test_train_loop_legacy_path_unchanged` that asserts
the trainer construct path is unchanged when modern-recipe knobs are
unset). All 8 must be green before Wave-A:

```powershell
.\.venv\Scripts\python -m pytest tests/test_modern_recipe.py -v
```

## 5. Authorisation checklist

- [ ] User confirms: 3 overnight sessions on the 4090 Laptop are budgeted
- [ ] User confirms: auto-checkpoint loop will run alongside (Rule 20)
- [ ] User confirms: Rule 11 commit-push cadence is in place
- [ ] User confirms: `tests/test_modern_recipe.py` green
- [ ] User confirms: CIFAR-10 SOTA-smoke green on current head
- [ ] User signs off on the Wave-A → Wave-B → Wave-C decision tree

Once all six checkboxes are signed, the agent has authorisation to
launch Wave-A. Wave-B is gated on the Wave-A decision tree; Wave-C is
gated on the Wave-B decision tree.

---

*Last updated: 2026-06-01. Plan is laptop-realistic; no ImageNet, no
server-grade sweeps. The 11-trick recipe is implemented in
`src/nature_inspired_networks/{mixup,cutmix,random_erasing,randaugment,ema}.py`
+ wired through `train.py` and `runner.py`. The plan is the
deliverable until user-authorised; the runs are secondary.*
