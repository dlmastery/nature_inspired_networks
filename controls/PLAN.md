# Reviewer-flagged control sweep plan

> Status: **planning + dry-run scaffolding only** — no sweeps launched yet.
>
> Source: `audits/REVIEWER_PASS_PAPER.md` items 4, 5, 13, 14 (the four
> control sweeps that the NeurIPS-grade reviewer required before any
> empirical headline can stand). This document specifies hyperparameter
> matrices, expected outcomes, GPU-cost estimates, and per-control
> implementation-gap status. The companion script
> `scripts/run_control_sweeps.py` operationalises the plan with
> `--dry-run` as the default, so executing it does NOT launch any
> training. Authorisation to launch must be explicit via `--launch`.

The four controls map one-to-one to the four reviewer counter-arguments
that the paper as written cannot defend:

| Control | Defends | Reviewer item | Total runs |
|---------|---------|---------------|------------|
| 1 | `pair_gm_pdw` (3-axis φ stack)             | item 4  | 3   |
| 2 | `slot_act_sine` (SIREN activation)         | item 5  | 12  |
| 3 | H09 `phi_budget` (RegNet rediscovery risk) | item 13 | 18  |
| 4 | H71 `IcosaRoPE3D` (sole NOVEL+TESTABLE)    | item 14 | 3   |

Per-control specifications below.

---

## Control 1 — Non-φ 3-axis regularizer stack

**Defends:** `pair_gm_pdw` (PAPER.md:130–134; FINDINGS.md headline).
**Reviewer counter-argument (item 4, §F bullet 3):** the +1.34 pp lift
could be explained by **any** three orthogonal regularization knobs
tuned away from baseline defaults. The φ-content is decorative.

**Mechanism of the control:** keep the 3-axis structure of
`pair_gm_pdw` but replace each axis's φ-derived knob with the **closest
non-φ analog at iso-params / iso-recipe cost**.

| Axis | `pair_gm_pdw` (φ) | Control 1 (non-φ) |
|------|-------------------|--------------------|
| Width allocator | `phi_budget` (1:φ:φ², widths [37, 48, 61]) | **uniform allocator** at the same param-budget B=270k, n_stages=3; all stages at the average width (~50, exact value resolved by the allocator). |
| Momentum | `momentum_schedule=golden` (β1 anneals 1/φ→1/φ² via H48 exponential) | **constant β1=0.4** (close to the H48 final value 0.382 but with no schedule). |
| Per-layer wd | `phi_decay_wd=True` (wd = base_wd / φ^k, geometric) | **linear-decay wd** (wd = base_wd · (1 - k/(K-1))/2, arithmetic; same total wd budget over the block stack at iso-base_wd). |

**Expected outcomes:**

- If Δ ≈ +1 pp on CIFAR-100 (matching `pair_gm_pdw`'s +1.34 pp lift),
  the φ-content is **decorative** — the paper must drop the
  `pair_gm_pdw` headline.
- If Δ ≈ 0 pp or negative, the φ-content has **real signal**, and the
  paper can defend the headline.
- If Δ is intermediate (~+0.3 to +0.7 pp), the φ-content is partial —
  most of the lift is from "any 3-axis regularization" but a fraction
  is φ-specific; the paper must report the decomposition.

**Sweep tag:** `pair_nonphi_3axis_cifar100_seed{0,1,2}` (1 row × 3 seeds).

**GPU cost on RTX 4090 Laptop:**

- Reference: `pair_gm_pdw_seed0` on CIFAR-100 took ~50 min (30 ep,
  batch 256, bf16, phi_budget ~268k params).
- Uniform allocator at the same param budget produces similar-width
  3-stage net (~50/50/50); FLOPs and latency within ±5%.
- **3 runs × ~50 min ≈ 2.5 GPU-h.**

**Implementation-gap status: READY (2026-05-30).**

All three primitives have landed in the source tree:

1. `phi_budget_widths(..., budget_mode='uniform')` — added to
   `src/nature_inspired_networks/phi_scaling.py` via the new
   `_uniform_budget_widths` helper; `PhiBudgetNet` now routes both
   modes through the unified allocator. Tests: 4 new in
   `tests/test_phi_scaling.py` (`test_budget_mode_uniform_returns_equal_widths`,
   `test_budget_mode_uniform_param_count_matches_phi_at_iso_target`,
   `test_budget_mode_phi_default_unchanged`,
   `test_budget_mode_rejects_invalid_mode`).
2. `train.TrainConfig.const_beta1` — new field plus the
   `Trainer._pin_beta1` static helper that bypasses the H48
   `GoldenMomentumScheduler` when set; re-applied after
   growth_pruning rebuilds the optimizer. `runner.run_one` forwards
   `cfg['const_beta1']`. Tests: 3 new in
   `tests/test_integration_optimizer.py`
   (`test_const_beta1_holds_value_across_epochs`,
   `test_const_beta1_compatible_with_golden_momentum_beta2_decay`,
   `test_const_beta1_pin_beta1_static_helper`).
3. `phi_decay.linear_decay_param_groups` — arithmetic per-layer
   weight-decay schedule `wd_k = base_wd · max(0, 1 - depth_factor · k / max_depth)`.
   Symmetric to `phi_decay_param_groups`; same coverage / no-
   duplicate invariants. Tests: 6 new in `tests/test_phi_decay.py`
   (`test_linear_decay_param_groups_strictly_decreasing_wd`,
   `test_linear_decay_param_groups_first_group_equals_base_wd`,
   `test_linear_decay_param_groups_iso_total_wd_to_phi_decay_at_matched_total`,
   `test_linear_decay_covers_all_params_no_duplicates`,
   `test_linear_decay_rejects_invalid_depth_factor`).

Commit: `2d29f18` (Set 1).

---

## Control 2 — Non-sine activation ablation

**Defends:** `slot_act_sine` (PAPER.md:138 Phase-8 winner +1.04 pp lead
floor).

**Reviewer counter-argument (item 5, §F bullet 4):** SIREN at ω=1 is
just smooth bounded `sin(x)`. Replacing ReLU with `sin(x)` is a known
activation-engineering trick (Sitzmann 2020 NeurIPS) with no
golden-ratio content. To claim `slot_act_sine` as a winner, sine must
outperform **all** of {tanh, softplus, gelu, swish} at the same SLOT
substitution recipe by ≥0.5 pp at 3 seeds.

**Sweep matrix:** the SLOT ablation already runs on the
`phi_budget` base (see `scripts/run_sweep.py` line 467). Add four new
SLOT rows that swap ReLU for the named activation; `slot_act_sine`
(already 3-seed CIFAR-100) is the comparison.

| Tag | Activation | Wired? |
|-----|------------|--------|
| `slot_act_sine` (existing)     | `sin(ω x)`, ω=1.0 init | YES — `swap_relu_with_sine` |
| `slot_act_tanh` (new)          | `tanh(x)`              | NO  — needs generic swap helper |
| `slot_act_softplus` (new)      | `softplus(x)`          | NO  — needs generic swap helper |
| `slot_act_gelu` (new)          | `gelu(x)`              | NO  — needs generic swap helper |
| `slot_act_swish` (new)         | `x · sigmoid(x)` (SiLU; β=1)       | NO  — needs generic swap helper |

The reviewer's exact list is {tanh, softplus, gelu, swish, sin}; the
existing `slot_act_phi` (PhiGELU at β=φ) is intentionally **NOT** part
of this control because it is itself a φ-derivative.

**Expected outcomes:**

- If `slot_act_sine` median > all four non-φ alternatives by ≥0.5 pp,
  the `slot_act_sine` claim survives — and the paper should mention
  that the winning activation is SIREN, not a φ-derivative.
- If any of {tanh, softplus, gelu, swish} matches or beats sine within
  ±0.5 pp, drop `slot_act_sine` from headlines per item 5 of the
  reviewer's required revisions.

**Sweep tags:** `slot_act_{tanh,softplus,gelu,swish}_cifar100_seed{0,1,2}`
(4 rows × 3 seeds = **12 runs**).

**GPU cost on RTX 4090 Laptop:**

- Reference: `slot_act_sine_seed0` on CIFAR-100 took ~52 min (slightly
  slower than ReLU baseline due to autograd through sin).
- tanh / softplus / gelu / swish: comparable or marginally cheaper
  (GELU has erf overhead, swish has sigmoid). Average ≈ 50 min.
- **12 runs × ~50 min ≈ 10 GPU-h.**

**Implementation-gap status: READY (2026-05-30).**

The generic helper has landed in
`src/nature_inspired_networks/activations.py`:

- `swap_relu_with(model, factory)` — recursively replaces every
  `nn.ReLU` with a fresh `factory()` instance.
- `SLOT_ACTIVATION_FACTORIES` table covers the `{tanh, softplus,
  gelu, swish, silu}` aliases the reviewer's Control 2 matrix
  requires. Sine and PhiGELU remain on their dedicated helpers so
  the `omega_init` / `beta_init` wiring is preserved byte-for-byte.
- `runner.post_build_mutators` has a new `slot_activation` branch
  that dispatches to `swap_relu_with(model, factory)` for standard
  aliases and delegates to the existing helpers for `sine` / `phi`.
  Unknown aliases raise `ValueError` (Rule 7 — no silent fall-
  through).

Tests: 8 new in `tests/test_activations.py`
(`test_swap_relu_with_replaces_all_relus_in_resnet20`,
`test_swap_relu_with_preserves_other_layers`,
`test_slot_activation_{tanh,softplus,gelu,swish}_dispatches_correctly`,
`test_slot_activation_unknown_rejected`,
`test_slot_activation_factories_table_coverage`).

Commit: `1d09411` (Set 2).

---

## Control 3 — Tuned ResNet-20 + RegNetX-200MF baseline

**Defends:** H09 `phi_budget` (PAPER.md:179 cited as eligible for
external claims; line 23 admits "rediscovery of RegNet Pareto region").

**Reviewer counter-argument (item 13, §F bullet 9):** the current
baseline is an **untuned** ResNet-20. A +0.78 pp lift over an untuned
baseline is uninterpretable. The control must replace the baseline with
(a) a tuned ResNet-20 (best LR, wd found by coordinate-descent at the
30-ep CIFAR-100 horizon) and (b) RegNetX-200MF (the literature analog
with parameter budget matched to phi_budget's ~268k → closest RegNetX
size: 200MF ≈ 2.7M params is too large; **RegNetY-200MF** or a custom
RegNetX shrunk to 270k params is the iso-compute alternative).

**Sweep matrix (two sub-controls):**

### 3a. Tuned ResNet-20 hill-climb

Coordinate-descent over (lr, weight_decay) at single seed=0, then
3-seed at the best (lr, wd) pair.

| lr  | wd     | Hill-climb? | Wired? |
|-----|--------|-------------|--------|
| 0.003 | 1e-4  | yes (seed 0) | YES (existing cfg keys) |
| 0.003 | 5e-4  | yes (seed 0) | YES |
| 0.003 | 1e-3  | yes (seed 0) | YES |
| 0.01  | 1e-4  | yes (seed 0) | YES |
| 0.01  | 5e-4  | yes (seed 0) | YES |
| 0.01  | 1e-3  | yes (seed 0) | YES |
| 0.03  | 1e-4  | yes (seed 0) | YES |
| 0.03  | 5e-4  | yes (seed 0) | YES |
| 0.03  | 1e-3  | yes (seed 0) | YES |
| 0.1   | 1e-4  | yes (seed 0) | YES |
| 0.1   | 5e-4  | yes (seed 0) | YES |
| 0.1   | 1e-3  | yes (seed 0) | YES |
| best   | best   | 3-seed (final) | YES |

Total: **12 seed-0 hill-climb runs + 3 seed runs at best = 15 runs.**

### 3b. RegNetX-200MF at iso-params

RegNetX-200MF (Radosavovic 2020) — ~2.7M params at default. To match
phi_budget's 270k budget, either:

- Option A: take RegNetX-200MF and shrink `bottleneck_ratio` /
  `group_width` until params hit 270k ±5%. Use `torchvision.models.regnet_x_200mf`
  with custom `RegNetParams`.
- Option B: use the smallest stock RegNetX (RegNetX-200MF ≈ 2.7M) and
  accept the 10× param disparity, reporting it openly.

This plan goes with **Option A** to preserve iso-params comparison.
3 seeds at the chosen RegNetX config.

**Sweep tags:**

- Hill-climb: `baseline_resnet20_tuned_lr{LR}_wd{WD}_cifar100_seed0`
  (12 rows × 1 seed = 12 runs).
- Best 3-seed: `baseline_resnet20_tuned_cifar100_seed{0,1,2}`
  (1 row × 3 seeds = 3 runs).
- RegNetX: `baseline_regnetx_200mf_cifar100_seed{0,1,2}` (1 row × 3
  seeds = 3 runs).
- **Total: 18 runs.**

**GPU cost on RTX 4090 Laptop:**

- ResNet-20 untuned (existing baseline) on CIFAR-100 30 ep ≈ 35 min.
- Hill-climb: 12 × 35 min ≈ **7 GPU-h.**
- Best 3-seed: 3 × 35 min ≈ **1.75 GPU-h.**
- RegNetX-200MF shrunk to 270k params ≈ similar FLOPs to phi_budget
  ≈ 50 min × 3 seeds ≈ **2.5 GPU-h.**
- **Total: ~11.25 GPU-h.**

**Implementation-gap status: READY (2026-05-30).**

- (3a) Tuned ResNet-20 hill-climb: **READY** — `lr` and
  `weight_decay` are first-class cfg keys; the orchestrator emits
  N+3 calls to `run_one` with different lr/wd overrides. Unchanged.
- (3b) RegNetX-200MF: **READY**. `models.build_regnetx(name,
  num_classes, target_params)` now builds stock RegNetX-200MF (via
  the canonical Radosavovic 2020 Table 9 init parameters,
  `BlockParams.from_init_params(depth=13, w_0=24, w_a=36.44,
  w_m=2.49, group_width=8)`) and a shrunk variant via
  `width_multiplier_search(target_params)`. The shrunk variant
  binary-searches a uniform `(w_0, w_a)` scale (with `w_0`
  quantised to a multiple of 8) until the realised param count
  lands within +/- 5 % of the target. `build_model` dispatches
  `regnetx_200mf` / `regnetx_200mf_shrunk`; `runner._MODEL_BUILD_KW`
  forwards `regnetx_param_budget`. NOTE: torchvision >= 0.21
  dropped the `regnet_x_200mf` factory helper (smallest stock
  factory is now 400MF), so we re-implement the 200MF init in our
  own dispatch — the Radosavovic 2020 paper is the authoritative
  source. The import error surfaces with a clear "install
  torchvision >= 0.15" hint when the BlockParams import fails.

Tests: 6 new in `tests/test_models.py`
(`test_regnetx_200mf_builds`,
`test_regnetx_param_count_at_default_is_at_least_2M`,
`test_regnetx_width_multiplier_search_hits_target_within_5_percent`,
`test_build_regnetx_shrunk_via_build_model_dispatch`,
`test_build_model_rejects_unknown_name`,
`test_width_multiplier_search_rejects_invalid_target`).

Commit: `00b79e7` (Set 3).

---

## Control 4 — H71 IcosaRoPE3D ViT-Tiny smoke

**Defends:** H71 IcosaRoPE3D (PAPER.md:111 cited as the SOLE
NOVEL+TESTABLE survivor; §7.4 admits no CIFAR smoke yet).

**Reviewer counter-argument (item 14, §F):** the sole NOVEL+TESTABLE
hypothesis is **never empirically tested**. A paper listing H71 as a
contribution while admitting it has no number is "we have an idea,
trust us." Even a single-seed smoke that produces a falsifiable number
converts H71 from "research proposal" to "we tested our best idea."

**Sweep matrix:**

- Architecture: ViT-Tiny (12 layers, 192 dim, 3 heads). H71 design doc
  §7.1 specifies "124M decoder" for the primary experiment; that scale
  is impractical on a 4090 Laptop in <18 h. We downscale to ViT-Tiny
  for the smoke (~5.7M params, runs in ~2 h at 30 ep on CIFAR-10).
- Dataset: **rotated-CIFAR-10** — apply `torchvision.transforms.
  RandomRotation(degrees=180)` to both train and test sets. (H71's
  design doc §7.1 specifies synthetic 3-D scene-QA; a 2-D rotation-
  equivariance smoke on rotated CIFAR-10 is the cheapest proxy for
  the rotation-equivariance falsifier in §3 of the design doc.)
- Two RoPE variants compared (Rule 1 atomic): standard 1-D RoPE vs.
  H71 IcosaRoPE3D. Both swap the same MHSA RoPE slot; everything else
  is identical.

**Sweep tags:**

- `h71_icosa_rope3d_vit_tiny_rotcifar10_seed{0,1,2}` (3 runs).
- (Reference: `vit_tiny_1d_rope_rotcifar10_seed0` — single seed
  baseline. NOT in the headline control but the orchestrator will
  print the row so the reviewer can confirm we ran the head-to-head.)

**Expected outcomes:**

- If IcosaRoPE3D ≥ 1-D RoPE by ≥1 pp at n=3 (H71 falsifier §3 floor),
  H71 graduates from "research proposal" to "tested falsifier passed".
- If Δ < 1 pp, H71 is **DISCARDED** per its own falsifier.

**GPU cost on RTX 4090 Laptop:**

- ViT-Tiny on CIFAR-10 (32×32 → 8×8 patches × 12 layers × 192 dim):
  ~2 GPU-h per 30-ep run at batch 256, bf16.
- Add 1-D RoPE reference run (seed 0 only) ≈ 2 GPU-h.
- **Total: 4 runs × ~2 h ≈ 8 GPU-h.**

**Implementation-gap status: READY (2026-05-30).**

All three primitives have landed:

1. **ViT-Tiny model** — new file
   `src/nature_inspired_networks/vit_tiny.py` (~330 LOC). Minimal
   ViT-Tiny class (`patch_embed` → 12 × `{LayerNorm, MHSA-with-RoPE,
   MLP}` → `LayerNorm` → `Linear` head). `MultiHeadSelfAttention`
   dispatches on `rope_kind` ∈ `{"none", "rope1d", "icosa3d"}`:
   - `"rope1d"` rotates the largest even-`D` prefix and passes the
     odd tail (head_dim=33 → 32 rotated + 1 pass-through) — documented
     in the helper docstring.
   - `"icosa3d"` instantiates `IcosaRoPE3D(head_dim=33)` whose default
     `base=PHI` matches the H71 design doc.
   `build_vit_tiny(...)` is the factory used by `models.build_model`;
   the runner forwards `vit_*` cfg keys via `_MODEL_BUILD_KW`.
2. **rotated-CIFAR-10 dataset loader** — `data.py` adds
   `rotated_cifar_loaders` and the `load_rotated_cifar10` alias.
   Train pipeline samples uniformly over the four cardinal angles
   `(0, 90, 180, 270)` via `T.functional.rotate`; eval pipeline
   applies ALL four rotations as deterministic test-time augmentation
   via `_RotatedCIFAR` (10_000 base × 4 angles = 40_000 eval items),
   yielding the rotation-equivariance-aware top-1. `load_dataset`
   routes `rotated_cifar10` / `rotcifar10` / `rotated_cifar100`.
3. **head_dim constraint** — `ViTTiny.__init__` enforces
   `head_dim % 3 == 0` unconditionally. The Control 4 cfg uses
   `embed=198, heads=6, head_dim=33` (the choice was 6 × 33 = 198
   per the cfg YAML, picked over the alternative 4 × 48 = 192 to
   keep heads close to canonical; the alternative is also tested).

Tests: 9 new in `tests/test_vit_tiny.py` + 8 new in
`tests/test_data.py`.

Commit: `<set4_sha>` (Set 4 — pending commit at end of this report).

---

## Aggregate GPU-cost summary (RTX 4090 Laptop)

| Control | Runs | Per-run | Total |
|---------|------|---------|-------|
| 1 — non-φ 3-axis             | 3   | ~50 min | 2.5 h |
| 2 — activation ablation      | 12  | ~50 min | 10.0 h |
| 3 — tuned baseline + RegNetX | 18  | ~35–50 min | 11.25 h |
| 4 — H71 ViT-Tiny smoke       | 4   | ~2 h | 8.0 h |
| **TOTAL**                    | **37** | — | **~31.75 GPU-h** |

A continuous 32-h sweep is impractical without the Rule-26 thread caps
and Rule-20 auto-checkpoint loop; the recommended cadence is one
control per evening (5 evenings) with the auto-checkpoint loop running
alongside.

## Per-control implementation-gap summary

| Control | Status | Blocking work |
|---------|--------|---------------|
| 1 | **READY** (2026-05-30) | All three primitives landed: (a) `phi_budget_widths(..., budget_mode='uniform')`, (b) `TrainConfig.const_beta1` + `Trainer._pin_beta1`, (c) `phi_decay.linear_decay_param_groups`. Commit `2d29f18`. |
| 2 | **READY** (2026-05-30) | Generic `activations.swap_relu_with(model, factory)` + `SLOT_ACTIVATION_FACTORIES` table + `runner.post_build_mutators` `slot_activation` dispatch. Commit `1d09411`. |
| 3 | **READY** (2026-05-30) — 3a + 3b | `models.build_regnetx` + `width_multiplier_search` shrink RegNetX-200MF to the requested param budget within +/- 5 %. Commit `00b79e7`. |
| 4 | **READY** (2026-05-30) | `vit_tiny.ViTTiny` (head_dim=33, embed=198, 6 heads) + `data.rotated_cifar_loaders` (4 cardinal angles, all-4 TTA on eval). Commit pending below. |

## How to run (after wiring patches land)

```powershell
# Default is --dry-run: prints the plan, does NOT launch.
.\.venv\Scripts\python scripts\run_control_sweeps.py --control all

# Launch a specific control (after the relevant wiring patches land):
.\.venv\Scripts\python scripts\run_control_sweeps.py --control 1 --launch
.\.venv\Scripts\python scripts\run_control_sweeps.py --control 2 --launch
.\.venv\Scripts\python scripts\run_control_sweeps.py --control 3 --launch
.\.venv\Scripts\python scripts\run_control_sweeps.py --control 4 --launch
```

The script is dry-run by default; an explicit `--launch` flag is
required to actually spawn any `run_one(...)` call. The `--launch`
path also enforces the Rule-13 pre-flight: a fresh CIFAR-100 baseline
30-ep run must pass the expected band (top1 ≥ 0.60) before any
variant launches.

---

*Plan authored 2026-05-29 by Control-Sweep-Designer agent. Not
launched. Wiring gaps documented per-control. Authorisation required
via `--launch` flag.*
