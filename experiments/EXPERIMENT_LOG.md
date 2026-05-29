# EXPERIMENT_LOG — master long list

> Append-only, single source of truth for every experiment ever launched
> against `dlmastery/nature_inspired_networks`. Each row corresponds to one
> archived sub-directory under `ideas/<NN>/experiments/expNNN_<short>/`.
>
> Status legend:
> - **✓ done** — completed; metrics + artifacts archived
> - **▶ running** — background task in flight
> - **⏸ queued** — config + reasoning written; awaiting GPU
> - **○ planned** — design only; no config yet
> - **✗ failed** — crashed or rejected by gates; see `error.txt` in archive
> - **♻ superseded** — overtaken by a later run; kept for history
>
> One commit per row update. The checkpoint is the deliverable.

## Tier 0 — Literature anchors

| # | Idea | Tag | Dataset | Status | Top-1 | Params | Composite | Archive |
|---|---|---|---|---|---|---|---|---|
| L0 | Baseline ResNet-20 (He 2015) | `baseline_resnet20` | CIFAR-10 | ✓ done | 84.78 % | 272 k | 0.8458 | `experiments/cifar10/baseline_resnet20_seed0/` |
| L1 | Baseline NaturePriorNet (no priors, linear channels) | `baseline_sg_vanilla` | CIFAR-10 | ✓ done | 82.16 % | 186 k | 0.8258 | `experiments/cifar10/baseline_sg_vanilla_seed0/` |

## Tier 1 — Single-prior CIFAR-10 ablation (the original 11-run sweep)

| # | Idea | Tag | Dataset | Status | Top-1 | Params | Composite | Archive |
|---|---|---|---|---|---|---|---|---|
| T1.1 | H04 phi/fib width — `fib` channel mode, priors off | `sg_chan_fib` | CIFAR-10 | ✓ done | 80.11 % | 127 k | 0.8135 | `experiments/cifar10/sg_chan_fib_seed0/` |
| T1.2 | H04 phi width — `phi` channel mode, priors off | `sg_chan_phi` | CIFAR-10 | ✓ done | 80.11 % | 127 k | 0.8152 | `experiments/cifar10/sg_chan_phi_seed0/` |
| T1.3 | H21 hex-only with fib channels | `sg_only_hex` | CIFAR-10 | ✓ done | 79.32 % | 127 k | 0.7941 | `experiments/cifar10/sg_only_hex_seed0/` |
| T1.4 | H24 (proxy: C4 group conv max-pool) | `sg_only_group` | CIFAR-10 | ✓ done | 69.84 % | 127 k | 0.6937 | `experiments/cifar10/sg_only_group_seed0/` |
| T1.5 | H05 fractal recursion at depth=2 | `sg_only_fractal` | CIFAR-10 | ✓ done | 82.46 % | 259 k | 0.8104 | `experiments/cifar10/sg_only_fractal_seed0/` |
| T1.6 | H22 toroidal padding | `sg_only_toroidal` | CIFAR-10 | ✓ done | 78.05 % | 127 k | 0.7768 | `experiments/cifar10/sg_only_toroidal_seed0/` |
| T1.7 | H35 cymatic (Chladni-mode) init | `sg_only_cymatic_init` | CIFAR-10 | ✓ done | 77.44 % | 127 k | 0.7883 | `experiments/cifar10/sg_only_cymatic_init_seed0/` |
| T1.8 | H17/H34 golden-angle channel gate | `sg_only_golden_modulate` | CIFAR-10 | ✓ done | 79.81 % | 127 k | 0.8042 | `experiments/cifar10/sg_only_golden_modulate_seed0/` |
| T1.9 | H50 full hybrid (all 6 priors on) | `sg_full_fib` | CIFAR-10 | ✓ done | 73.24 % | 259 k | 0.6966 | `experiments/cifar10/sg_full_fib_seed0/` |

## Tier 2 — Negative-result follow-ups (in-flight + queued)

| # | Idea | Tag | Dataset | Status | Top-1 | Composite | Verdict |
|---|---|---|---|---|---|---|---|
| T2.1 | **H58 — C4 group max → avg pool fix** | `sg_only_group_avg` | CIFAR-10 | ✓ done | **65.38 %** | **0.6597** | **DISCARD** — mean-pool *hurts* worse than max (-4.46 pp). Hypothesis falsified. |
| T2.2 | H58 + H50 — full hybrid with avg-pool reduction | `sg_full_fib_avg` | CIFAR-10 | ✓ done | **66.86 %** | **0.6432** | **DISCARD** — confirms T2.1; mean-pool consistently worse (-6.38 pp). |
| T2.3 | H59 — trained-feature Betti on T1.* using newly saved checkpoints | n/a | CIFAR-10 | ⏸ queued | depends on T2.1/T2.2; checkpoint saving wired but legacy runs lack `best.pt` |
| T2.4 | H60 — 3-seed re-sweep for error bars on the 11-row matrix | various | CIFAR-10 | ⏸ queued | `run_sweep.py --seeds 0 1 2 --skip-existing` |
| T2.5 | H05.v2 — fractal with explicit 1/φ depth shrink per recursion | `sg_fractal_phi_shrink` | CIFAR-10 | ⏸ queued | extends T1.5 with proper 1/φ rule |
| T2.6 | H22.v2 — toroidal with φ-scaled periodic wrap distance | `sg_toroidal_phi` | CIFAR-10 | ⏸ queued | extends T1.6 with φ multiplier |
| T2.7 | H21.v2 — hex with φ-radial weighting of the 7 taps | `sg_hex_phi_weight` | CIFAR-10 | ⏸ queued | extends T1.3 with H21 full version |
| T2.8 | H50.LOO — leave-one-out from full hybrid (6 rows) | `sg_loo_no_*` | CIFAR-10 | ⏸ queued | `run_sweep.py --full` |

## Tier 3 — Scaled / scope-extended experiments

| # | Idea | Tag | Dataset | Status | Notes |
|---|---|---|---|---|---|
| T3.1 | Best Tier-1/2 variant on CIFAR-100 (100 classes) | `<top>_cifar100` | CIFAR-100 | ○ planned | 25 epochs ≈ 12 min on 4090 each |
| T3.2 | MedMNIST PathMNIST (28 px, 9 classes) | `*_pathmnist_28` | MedMNIST | ○ planned | loader ready in `data.py` |
| T3.3 | MedMNIST OrganAMNIST (28 px, 11 classes) | `*_organamnist_28` | MedMNIST | ○ planned |  |
| T3.4 | MedMNIST OCTMNIST (28 px, 4 classes) | `*_octmnist_28` | MedMNIST | ○ planned |  |
| T3.5 | Tiny ImageNet (64 px, 200 classes) | `*_tinyimagenet` | Tiny ImageNet | ○ planned | ~2–3 day/run; load on demand |
| T3.6 | ImageNet-100 subset | `*_imagenet100` | IN-100 | ○ planned | 3–7 day/run on 4090 Laptop |

## Tier 4 — LLM-track per extended-transcript (new design space)

Per extended-transcript chunks 4-8: each H01–H50 has a decoder-only Transformer
sibling with PyTorch pseudocode + WikiText-103 / TinyStories / GSM8K
benchmarks.

| # | Idea | Tag | Dataset | Status | Notes |
|---|---|---|---|---|---|
| T4.1 | H01-LLM φ-compound scaling on GPT-2-small (124 M) | `llm_phi_scale` | WikiText-103 | ○ planned | bf16 + grad-ckpt + FlashAttention-2 |
| T4.2 | H02-LLM Fibonacci depth progression | `llm_fib_depth` | WikiText-103 | ○ planned |  |
| T4.3 | H21-LLM hex sparse attention graph | `llm_hex_attn` | WikiText-103 | ○ planned | with toroidal KV |
| T4.4 | H22-LLM toroidal KV cache | `llm_toroidal_kv` | WikiText-103 | ○ planned | constant-memory long-context test |
| T4.5 | H32-LLM Fibottention (Fib-dilated attention) | `llm_fibottention` | WikiText-103 / TinyStories | ○ planned | extends arXiv:2406.19391 |
| T4.6 | H34-LLM golden-angle RoPE | `llm_golden_rope` | WikiText-103 | ○ planned |  |
| T4.7 | H49-LLM Platonic alignment auxiliary loss | `llm_prh_align` | TinyStories | ○ planned |  |
| T4.8 | H50-LLM full hybrid decoder block | `llm_sgblock_v2` | TinyStories + GSM8K | ○ planned |  |

## Tier 5 — Cross-paradigm hybrids (H61–H71)

| # | Idea | Tag | Status | Notes |
|---|---|---|---|---|
| T5.1 | **H61** Sacred-Liquid-JEPA decoder block | `liquid_jepa_phi` | ○ planned | 350M on TinyStories |
| T5.2 | **H62** Toroidal KV + hex attention + Fib-pruning | `toroidal_hex_attn` | ○ planned | constant-memory 128 k context |
| T5.3 | **H63** Platonic projection aux + cymatic teachers | `platonic_cymatic_teacher` | ○ planned |  |
| T5.4 | **H64** Dynamic φ-growth + Fib-pruning | `dyn_growth_prune` | ○ planned |  |
| T5.5 | **H65** Differentiable Betti-collapse loss | `betti_loss` | ○ planned | wired but not trained yet |
| T5.6 | **H66** Cymatic wavelet QKV kernel bank | `cymatic_qkv` | ○ planned |  |
| T5.7 | **H67** Full Sacred-Liquid-JEPA-KAN-GNN-Transformer | `full_paradigm` | ○ planned | flagship |
| T5.8 | **H68** On-device world-model + toroidal closure | `world_model_torus` | ○ planned |  |
| T5.9 | **H69** KAN edges on Metatron graphs (symbolic head) | `kan_metatron_head` | ○ planned |  |
| T5.10 | **H70** Cymatic resonance low-data curriculum | `cymatic_curr` | ○ planned |  |
| T5.11 | **H71** Icosahedral RoPE for 3D spatial reasoning | `icosa_rope` | ○ planned |  |
| T5.12 | **H72** Fractal+Vesica FFN | `llm_fractal_vesica_ffn` | ○ planned | ext H05+H33 inside decoder FFN |
| T5.13 | **H73** Golden-spiral+Metatron PE | `llm_golden_spiral_metatron_pe` | ○ planned | ext H34+H36+H40 |
| T5.14 | **H74** Metatron overlap tying | `llm_metatron_overlap_tying` | ○ planned | ext H40+H23/H30 weight tying |
| T5.15 | **H75** Harmonic+cymatic SwiGLU | `llm_harmonic_cymatic_swiglu` | ○ planned | ext H19+H39+H35 inside FFN |

## Tier 6 — Robustness & specialized validation (bonus)

| # | Idea | Dataset | Status |
|---|---|---|---|
| T6.1 | Rotated CIFAR-10 (group-conv should now help) | rotCIFAR-10 | ○ planned |
| T6.2 | Spherical / IcoMNIST | IcoMNIST | ○ planned |
| T6.3 | ModelNet40 3D point cloud | ModelNet40 | ○ planned |
| T6.4 | ogbg-molhiv graph benchmark | OGB | ○ planned |
| T6.5 | Synthetic Chladni cymatic patterns | synthetic | ○ planned |
| T6.6 | AudioSet log-mel spectrograms (cross-modal) | AudioSet | ○ planned |
| T6.7 | Higgs UCI tabular (sister-project rail) | Higgs | ○ planned |

## Campaign log — 2026-05-27 — 35-tag CIFAR-10 Phase-2 smoke

> Dated campaign block (Rule 3: append-only). This logs the completed
> implementation + CIFAR-10 Phase-2 screening campaign. The tiered
> long-list above remains the canonical master index; this block
> summarises the single largest sweep run to date.

**Scope of the now-complete build.** 80 shared modules under
`src/nature_inspired_networks/`, exercised by 78 `tests/test_*.py` files
(~780+ unit tests, all green), backing 84 hypothesis design docs across
thematic groups g1–g8.

**Phase-2 smoke sweep.** 35 tags, seed 0, 12 epochs each, RTX 4090
Laptop, bf16. **Zero failures** across all 35 runs. Wall-clock:
~8559 s for the main 34-tag block + ~791 s for the G8 esoteric-extension
tag = ~9350 s (~2.6 h) total. 12 epochs is a **screening budget**
(Phase-2 broad scan), not a converged number — graduates re-run at the
full recipe in Phase 4.

**Seed-0 leaderboard (top-1, 12 ep).** Only **one** variant beat the
ResNet-20 baseline:

| rank | tag | H# | top-1 | note |
|---|---|---|---|---|
| 1 | `phi_budget` | H09 | **0.8554** | **ONLY variant > baseline** (+0.76 pp) |
| 2 | `baseline_resnet20` | — | 0.8478 | reference |
| 3 | `golden_momentum` | H48 | 0.8352 | |
| 4 | `phi_dropout` | H47 | 0.8280 | |
| 5 | `fractal` | H05 | 0.8246 | |
| 6 | `fib_depth` | H02 | 0.8218 | @180k params |
| 7 | `baseline_sg_vanilla` | — | 0.8216 | priors-off baseline |
| 8 | `phi_multiscale` | H07 | 0.8200 | |
| 9 | `golden_skip` | H17 | 0.8163 | |
| 10 | `fib_prune` | H43 | 0.8115 | |
| 11 | `golden_resize` | H03 | 0.8067 | |
| 12 | `sine_act` | H81 | 0.8062 | |
| 13 | `golden_spiral_init` | H31 | 0.8042 | |
| 13 | `phi_compound` | H01 | 0.8042 | |
| 15 | `chan_phi` / `chan_fib` / `fib_ensemble` | H04/— | 0.8011 | |
| 18 | `phi_activation` | H39 | 0.7995 | |
| 19 | `golden_modulate` | H17/H34 | 0.7981 | |
| 19 | `phi_decay` | H44 | 0.7981 | |
| 21 | `hex` | H21 | 0.7932 | |
| 22 | `phi_lr` | H10 | 0.7875 | |
| 23 | `toroidal` | H22 | 0.7805 | |
| 24 | `cymatic_init` | H35 | 0.7744 | |
| 25 | `phi_init` | H42 | 0.7656 | |
| 26 | `constant_width` | H80 | 0.7595 | |
| 27 | `phi_sparse` | H13 | 0.7333 | |
| 28 | `full_fib` | H50 | 0.7324 | all-priors hybrid |
| 29 | `fib_stride` | H18 | 0.7255 | |
| 30 | `phi_relu` | H19 | 0.7107 | |
| 31 | `group` | H24 | 0.6984 | C4 group conv |
| 32 | `golden_bottleneck` | H06 | 0.6925 | @58k params |
| 33 | `full_fib_avg` | H50+H58 | 0.6686 | |
| 34 | `group_avg` | H58 | 0.6538 | |
| 35 | `golden_adam` | H41 | 0.5196 | **FALSIFIED** — φ-scheduled Adam betas collapse training |

**Verdicts.**
- **Positive:** `phi_budget` (H09) is the only variant to clear the
  ResNet-20 baseline at 12 ep (+0.76 pp). It graduates.
- **Falsification:** `golden_adam` (H41) at 0.5196 top-1 — φ-derived
  Adam β-schedule destabilises optimisation; recorded as a negative
  result.
- **Read:** at a 12-epoch screening budget the priors mostly do not beat
  a well-tuned baseline; the signal is in the cross-row ablation deltas,
  not the asymptote. Convergence (Phase 4) decides graduates.

**Phase-4 CIFAR-100 shortlist.** The heavy-hitters that graduate to the
≥30-epoch CIFAR-100 phase: `phi_budget` (H09), `fib_depth` (H02),
`fractal` (H05), `golden_momentum` (H48), plus `baseline_resnet20` as
the carried-forward reference.

## Header-line conventions

- One row = one archive sub-directory = one `metrics.json`.
- Append-only. Never edit a past row's numbers; bump `<short>_v2` and add a
  new row if a result is superseded.
- The `Tag` column matches the `tag` field in `experiment_log.jsonl`.
- The `Archive` column is the directory; the README inside has the full
  story (hypothesis, prediction, actual, verdict, learning).
