# Big-Tech Lab Lead Critique — R5
Date: 2026-06-06 · Lens: engineering + reproducibility + ImageNet-viability + hire-decision

## Hire / no-hire verdict

**No-hire for a research-engineer / RS role on my team today. Hire for a senior infra / research-tooling role on a 3-month conversion plan.** I would not bring this person into a research scientist seat — the empirical instincts are broken in a way that four reviewer-passes have already named (motivated narrative migration, post-hoc HARKing, headline drift). I would *seriously* consider them as a senior research engineer or eval-infra hire: the 38-rule discipline, the auto-checkpoint loop, the SHA-256 fingerprint, the 780-test corpus, the Fixer-with-mechanism-pinning-test contract, the chunk-by-chunk source ledger, and the consistent commit cadence are all hallmarks of an engineer I would want owning my team's eval harness. Internal-ship verdict: **do not** ship this as a research framework, because the framework's own validation regime contradicts its claims (Phase-9h falsified the priors, Phase-9i restored them at iso-modern but at 2× the FLOPs of the baseline, and the public README still leads with "84 hypotheses + 35 smoke + 3 winners"). But absolutely **do** lift the Fixer-mechanism-pinning-test pattern, the per-experiment archive discipline, and the auto-checkpoint loop into an internal eval-infra package. Net: one strong engineering hire, one weak science hire, in the same body.

## Reproducibility verdict

**A fresh engineer on my team would NOT hit the n=3 modern-recipe headline (priors +1.00 to +1.24 pp over baseline 0.6360) in 24 h on a 4090 — and would likely conclude the headline is wrong.** The headline iso-modern 200-ep config (`configs/cifar100_modern_200ep.yaml`) is committed as the *baseline* config only. The three prior overlays the paper claims (`pair_gm_pdw_modern_200ep`, `slot_act_sine_modern_200ep`, `sg_only_phi_budget_modern_200ep`) **do not exist as YAML files in `configs/`** — `convergence/PLAN.md` says "create a one-shot overlay" without committing one. The runs at `experiments_modern/cifar100/{pair_gm_pdw,slot_act_sine,sg_only_phi_budget}_seed{0,1,2}/metrics.json` therefore have **no checked-in config provenance**. Worse, the prior runs all carry `params=267,658, flops=80,816,212` vs baseline `params=278,324, flops=41,224,448` — the priors are at **2× the FLOPs and ~3.5× the train-wall-clock** (12.3-17.2 ks vs 12.5 ks per seed for baseline — wait, that's roughly equal at 200 ep; but the legacy 30-ep prior runs are 3.5× the legacy baseline, see `experiments/cifar100/pair_gm_pdw_seed0/metrics.json` train_seconds=2804s vs baseline 795s at same 30 ep). The headline gap of +1.00 pp is on a *non-iso-compute* comparison and the configs aren't reproducible from a clean clone.

A 24-h cold-clone reproduction would land at: SOTA-smoke ✓ (the 80% band is real, that test works); 30-ep CIFAR-100 baseline reproducing 0.561 ✓ (legacy); 30-ep CIFAR-100 priors blocked at "no committed override config" ✗; modern 200-ep baseline blocked or producing 0.636 (below the [0.68, 0.72] band the PLAN.md says is the floor for proceeding) ✗. The honest cold-clone outcome is "engineer cannot reproduce headline because (a) overlay configs missing, (b) modern-recipe baseline is below the project's own stated success band, (c) the 'priors lift' is 2× FLOPs which the composite metric is too gentle to penalize."

## Bugs and engineering findings

**BLOCKER 1 — Non-iso-compute "iso-recipe" comparison in the Phase-9i headline.**
- `experiments_modern/cifar100/baseline_resnet20_seed0/metrics.json`: `params=278324`, `flops=41,224,448`, `latency_ms=3.74`
- `experiments_modern/cifar100/sg_only_phi_budget_seed0/metrics.json`: `params=267658`, `flops=80,816,212`, `latency_ms=3.04`
- `experiments_modern/cifar100/pair_gm_pdw_seed0/metrics.json`: `params=267658`, `flops=80,816,212`, `latency_ms=3.45`
- `experiments_modern/cifar100/slot_act_sine_seed0/metrics.json`: `params=267659`, `flops=80,816,212`, `latency_ms=3.32`

The priors are run at **~2× the FLOPs** of the baseline. The paper sells §5.3 / §6.5 / §9 as "iso-modern-recipe + iso-convergence" and the FINDINGS table lists "Δ +1.00 to +1.24 pp" without disclosing the FLOP gap. The composite metric `top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms)` only penalises params and latency, not FLOPs — so the prior wins on `composite` (0.649 vs 0.634) despite consuming 2× the compute. **Fix:** pin the priors to the baseline's 41.2M-FLOP envelope (e.g., reduce the φ-budget total or H09's n_stages until FLOPs match within ±5%) and re-run. Either the lift survives or — more likely — it collapses, because what's being measured is a depth/compute lift, not a φ-prior lift. As-is, the headline is a compute artifact wearing a prior costume. **File the FINDINGS / PAPER reframe immediately; this is the single most consequential finding in this review.**

**BLOCKER 2 — Modern-recipe baseline is below the project's own stated success band; sweep proceeded anyway.**
- `convergence/PLAN.md` line 117: "Expected pass band: **top1 ∈ [0.68, 0.72] median**."
- `convergence/PLAN.md` line 148-150: "If Wave-B median top1 < 0.68 → recipe is under-performing relative to the literature. Debug BEFORE Wave-C (it would be wasteful to compare winners against a weak baseline — the same trap critique item B was warning about)."
- Actual measured baseline median: `(0.635, 0.6383, 0.6348)` → median 0.6350, **3.3 pp below the floor**.

The project's own pre-registered decision tree said STOP and debug. Instead, Wave-C launched, Wave-C winners landed at 0.645-0.649, the FINDINGS got a "Phase-9i corrective binding" splice, and the paper now claims the priors are vindicated. **This is the textbook critique-item-B trap that PLAN.md explicitly warned against.** The mean lift +1.00 pp is exactly the kind of margin a still-underfit baseline produces over a slightly-more-converged variant (the priors have 2× FLOPs and are still in their high-loss regime when the baseline plateaus). **Fix:** debug the recipe collapse (likely culprits: warmup_epochs=5 / 200ep with bs=256 is only ~9800 steps, EMA warmup helps but doesn't close the gap; RandAugment N=2 M=14 is the ImageNet setting and is too aggressive for 32×32 CIFAR per Müller & Hutter 2021; Mixup α=0.2 + CutMix α=1.0 + label-smoothing-off may be over-regularising at this scale).

**BLOCKER 3 — `train_top1_final` and `generalization_gap` are silently broken under Mixup/CutMix.**
- `src/nature_inspired_networks/train.py:393` reports `acc = (logits.argmax(1) == target).float().mean()` where `target = y_a if mixed else y`. Under Mixup λ ≈ 0.6, the dominant label is `y_a` but the image is 60% `y_a` + 40% `y_b` — accuracy on `y_a` is structurally bounded by ~λ even for a perfect classifier. Result: `train_top1 ≈ 0.45` and `test_top1 ≈ 0.635` in every modern-recipe run, so `generalization_gap = max(0, 0.45 − 0.635) = 0.0` for all 12 runs.
- `eval.py:535` defines gap as `max(0.0, train_top1_final - top1)` — this floor at 0 silently hides the inversion. A reviewer reading the metrics.json would conclude there is *zero* generalization gap on a 200-ep CIFAR-100 ResNet-20, which is impossible under any normal training regime.
- **Fix:** when `mixed=True`, either (a) compute `train_top1` on the un-mixed last-epoch eval pass, or (b) drop the field and report only `test_top1`. The current value is a metric whose meaning silently changes with the recipe.

**BUG 4 — Mixup λ-flip bias is undocumented and untested.**
- `src/nature_inspired_networks/mixup.py:71-74`: `if lam < 0.5: lam = 1.0 - lam`.
- The comment calls this "purely cosmetic — symmetric in expectation." It is not. It collapses Beta(0.2, 0.2) (a deeply U-shaped distribution) to its right half. The *effective* alpha for the dominant-label loss weight is no longer Beta(α, α) — it's the folded distribution. For Beta(0.2, 0.2) this halves the variance of effective-mixing and changes the entropy of the supervision. timm does this too — but timm explicitly tags it as a convention that affects Mixup-CutMix coin-flip balance, not a no-op. Test `test_mixup_lambda_in_unit_interval` does not check `lam >= 0.5`.
- **Fix:** add an assertion `assert lam >= 0.5` in the test (or remove the flip; modern timm defaults to NOT flipping and instead permuting label order in the loss). Document the convention in the docstring.

**BUG 5 — CutMix `rand_bbox` allows degenerate zero-area boxes; test does not catch.**
- `src/nature_inspired_networks/cutmix.py:43-53`: at `lam ≈ 0.95`, `cut_h = int(32 * sqrt(0.05)) = int(7.15) = 7`. Centred at random pixel, often clipped so the integer-rounded box can degenerate to `(x1=x2, y1=y2)` when `cy ± 3` lands at image-boundary edges.
- `cutmix_batch` correctly guards `if (x2 - x1) > 0 and (y2 - y1) > 0:` but the test `test_cutmix_box_within_image_bounds` only asserts `0 <= x1 <= x2`, allowing `x1 == x2`. The degenerate case happens silently and shifts the effective Beta-α distribution by hard-clipping λ_eff = 1.0 ~5-10% of the time.
- **Fix:** add a test that asserts `lam_eff < 1.0 OR (x2 > x1 AND y2 > y1)` at `alpha=1.0` over 1000 trials, and assert the degenerate rate matches the analytical expectation. Right now the augmentation is shifted in distribution and no test surfaces it.

**BUG 6 — `apply_random_erasing_to_batch` is broken under `inplace=True`.**
- `src/nature_inspired_networks/random_erasing.py:32`: `T.RandomErasing(..., inplace=True)`.
- `random_erasing.py:61-63`: `apply_random_erasing_to_batch` does `out = x.clone(); for i in range(out.size(0)): out[i] = erasing(out[i])`. With `inplace=True` and `out[i]` being a *view* into `out`, the in-place erase modifies the view's underlying tensor — fine here because of the `.clone()`. **But:** the test `test_random_erasing_in_place` asserts `bool(diff.all())` at `p=1.0` (every image differs). With `T.RandomErasing` at `scale=(0.02, 0.33), ratio=(0.3, 3.3)` there is a non-zero chance of `_get_params` returning `(i, j, h, w) = (0, 0, 0, 0)` which is a no-op (newer torchvision); the test depends on a specific torchvision version's _get_params behaviour and will break silently on a different version. The training pipeline applies RandomErasing *after* Normalize with `value=0` — meaning the erased pixels are exactly at the per-channel mean (NOT zero in pixel space). The doctring claim "equivalent to the per-channel mean of CIFAR after normalisation which is exactly 0" is wrong: post-Normalize zero in `(x − μ)/σ` space corresponds to `x = μ` in pixel space — yes mean — but the comment says "exactly 0" without clarifying which space. Minor but a reviewer will land here.
- **Fix:** clarify the docstring; add a test parameterised over torchvision 0.17 / 0.18 / 0.19 _get_params behaviour.

**BUG 7 — `set_seed` does not enable deterministic algorithms, does not seed DataLoader workers.**
- `runner.py:38-41`: sets `random`, `numpy`, `torch.manual_seed`, `cudnn.benchmark = True`. Missing: `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic = True`, DataLoader `generator` kwarg, `worker_init_fn` for any `num_workers > 0` path.
- `data.py:54-82` uses `num_workers=4` as default for `cifar_loaders` (Windows hardware contract forces 0, but Linux users will inherit 4 silently). Without `worker_init_fn`, each worker gets a different RNG state every run, making the headline numbers unreproducible on Linux.
- `cudnn.benchmark = True` is explicitly opted-in (Rule 6), so cuDNN picks different kernels per shape per run — making bit-exact reproduction impossible even on the same hardware.
- **Fix:** Rule 6 should distinguish "fast-mode" (benchmark=True, no determinism) from "headline-mode" (deterministic=True, benchmark=False, worker_init_fn=seed_worker). Every reported headline should be in headline-mode; only screening sweeps in fast-mode. The README's "set_seed(seed)" claim of reproducibility is overstated.

**BUG 8 — RandAugment seeding broken.**
- `src/nature_inspired_networks/randaugment.py` wraps `T.RandAugment(N, M)`. torchvision RandAugment uses the global `torch` RNG via `torch.randint` inside `_augmentation_space` — meaning RandAugment IS seedable, but only if the DataLoader's `__getitem__` runs in the main process (`num_workers=0`). Under `num_workers>0` each worker has an unseeded RandAugment RNG. There is no `worker_init_fn` set.
- **Fix:** add `worker_init_fn=lambda wid: torch.manual_seed(seed + wid)` to the DataLoader on both the CIFAR-10/100 and rotated paths.

**BUG 9 — EMA finalization clobbers running BN buffers with shadow's stale buffers.**
- `train.py:496-502`: after fit, the live model loads the EMA shadow's full `state_dict` (params + buffers). EMA's buffers are COPIED (not blended — see ema.py:131-136), but the *last copy happened at the last optimizer step* which is mid-augmentation. The running BN statistics at the last train batch may differ from a clean EMA-of-running-mean — and loading them onto the live model overwrites the BN running stats that `topk_accuracy` then evaluates. If the last batch had a heavy Mixup/CutMix/RandAugment shift, the loaded BN running mean is shifted relative to a clean eval.
- The "right" thing in timm/Bello is to either (a) update BN buffers on the EMA model with a single pass through the train set in eval mode (BatchNorm momentum override), or (b) maintain EMA-of-buffers separately. The current code does neither.
- **Fix:** add a `_recalibrate_bn(model, train_loader)` pass before evaluating the EMA-loaded model (timm has `update_bn_stats`).

**BUG 10 — `cudnn.benchmark = True` in `set_seed` makes seed sweeping uninformative.**
- `runner.py:41` toggles benchmark globally. When sweeping seeds {0,1,2,3,4,5,6}, cuDNN picks different kernels per (shape, hardware, run) so the seeds carry *both* RNG noise and kernel-choice noise. The reported σ ≈ 0.45 pp on the n=7 default cert may be substantially inflated by kernel-choice noise and the +1.24 pp signal is on the edge of that band.
- **Fix:** for headline runs, set `cudnn.deterministic=True; cudnn.benchmark=False`. Re-report σ.

**BUG 11 — `cudnn.benchmark` set inside `set_seed` is a global side effect.**
- Calling `set_seed(0)` then `set_seed(1)` to compare seeds doesn't reset cudnn benchmark state (autotune cache persists across calls in the same process). This is OK for the runner's one-process-one-run pattern but breaks any multi-tag-in-one-process orchestrator.

**BUG 12 — `composite_score` has untested edge cases.**
- `eval.py:130`: `params_M = max(0.001, params / 1e6)`. With `params=267,658` this is 0.268 M which is fine. With `params=0` (untrained / pure-identity model in a degenerate hypothesis), the floor kicks in and you get `log10(0.001) = -3 → +0.05 * 3 = +0.15` boost to composite. A buggy model that fails to register parameters gets a *bonus* to composite. No test for this.
- **Fix:** raise on `params < 1000` rather than silently floor.

**BUG 13 — Append-only `experiment_log.jsonl` has no integrity check.**
- `runner.py:344-352`: opens the file in `"a"` mode, no fsync, no flock. A concurrent sweep (despite Rule 11 implying serial runs) could interleave incomplete JSON lines. No assertion on uniqueness of `(tag, seed)` pairs.
- **Fix:** add a `tag_seed_set` check + flock + line-length sanity at append.

**BUG 14 — `evaluate_full` recomputes `topk_accuracy` AFTER EMA load, but `epochs_to_target` was set during fit on the un-EMA-loaded model.**
- `train.py:488` increments `epochs_to_target` based on `te["top1"]` (raw model, pre-EMA-load). Then `evaluate_full` (eval.py:528-545) re-evaluates `model` after EMA has been loaded onto it. The `epochs_to_target` field thus measures *raw* model convergence; the `top1` field measures *EMA* model accuracy. These two fields are conceptually mismatched in the same metrics.json.
- **Fix:** report both `epochs_to_target_raw` and `epochs_to_target_ema` separately, or document the mismatch.

**BUG 15 — H09 phi_budget integer search at `phi_budget_total=16, n_stages=4` is heavily quantised.**
- The H09 fix (commit `519cdf3`) claims 0.43% max error in realised ratio 1:1.623:2.629. But this is the integer-rounded width tuple over a small budget. The `phi_budget_total=16` setting in `cifar100_modern_200ep` overlays produces widths around `16 → 26 → 42 → 68`, with rounding error compounding. Without checking that the realised widths actually produce φ²-ratio, you can't claim the H09 mechanism is "isolated" — and the 2× FLOPs increase shows the search isn't iso-budget either.
- **Fix:** re-run with explicit width tuple in the metrics.json so the reviewer can verify.

**BUG 16 — Citation Rigor regex over-matches; under-matches.**
- `reasoning.py:31`: `CITATION_RE_ID = re.compile(r"(?:arXiv|bioRxiv):\s*\d{4}\.\d{4,5}")` — does not match the modern arXiv ID format with category prefix (e.g., `cs.LG/0610087` legacy). More importantly, it accepts `arXiv:9999.99999` (any 4-5 digit suffix) without semantic check. The "gate" is mostly a syntactic shape check.
- `reasoning.py:32`: `CITATION_RE_DASH = re.compile(r"(?:—|--)\s*\S")` — accepts a single non-whitespace char after the dash. "relevance note" floor is one character. A reviewer reading "— X" would consider that a defect not a citation.

**BUG 17 — TrainConfig float comparisons leak hyperparam coupling.**
- `train.py:223`: `_mixing_active = (mixup_alpha > 0.0) or (cutmix_alpha > 0.0)`. The label-smoothing-drop happens on `_mixing_active`, regardless of whether mixing fires on a given batch. So in a 50/50 alternation regime, the trainer hard-disables label smoothing for *all* batches even when neither Mixup nor CutMix is the active op (e.g., when both αs are > 0 and the coin lands but lam degenerates). This is timm convention but it means the recipe is *less* than 11-trick whenever the coin "no mix" case is reached.

**BUG 18 — Pruning code path silently drops EMA finalization.**
- `train.py:498-502`: load with `strict=True`, fall back to `strict=False` on `RuntimeError`. The fallback is silent and only triggered when pruning has appended `weight_orig`/`weight_mask` — meaning H43 fibonacci_prune runs *never* fully load EMA shadow on the final model. The reported top1 is then a hybrid: pruned raw model with *partial* EMA buffer overwrite. No log, no flag.
- **Fix:** log every key that fails strict load; raise on mismatched-shape `weight_orig` keys.

**BUG 19 — `_extract_penultimate_features` returns BF16 tensor under autocast; BettiLoss runs cdist in fp32 via `.float()` cast on a *detached-from-graph* path.**
- `train.py:378-381`: `feats = self._extract_penultimate_features(x_eff); ... feats.float()`. Under bf16 autocast `feats` was bf16, then float() upcasts to fp32. But `_extract_penultimate_features` calls `model.stagewise_features(x_eff)` which RE-RUNS the forward — *doubling the per-step compute* when Betti is active. There's no caching of the autocast forward.
- **Fix:** cache `stagewise_features` from the main forward and reuse.

**BUG 20 — 20+ TODOs in source modules are documented as "runner wiring left for integration pass."**
- `grep TODO src/nature_inspired_networks/` returns 20 hits across `collapse_attention, dodeca_latent, dynamic_growth, fib_attention, fib_mlp, fib_recurrent, icosa, morphing_adjacency, phi_embedding, radial12_attention, platonic_graph, small_world, spectral_hopfield, tetra_dualpath, toroidal_latent`.
- These are not "TODO comments" in the typical sense — they are *unwired* modules that ship with tests, are counted in "84 hypotheses implemented" claim, but are never reachable from the runner. The runner can't dispatch to them. The implementation count is inflated.
- **Fix:** either wire them or explicitly mark "documented-but-not-executed" in IDEA_TABLE.md.

## Top 3 fatal engineering flaws

1. **Non-iso-compute headline (BLOCKER 1).** The Phase-9i "lift" runs the priors at ~2× the baseline's FLOPs. Composite metric doesn't penalise FLOPs. A reviewer who diffs the metrics.json sees this in 90 seconds. Every external claim in the paper is built on this confound.

2. **Modern-recipe baseline failed its own pre-registered floor (BLOCKER 2).** PLAN.md said "STOP if Wave-B median < 0.68". Wave-B landed at 0.6360. The project proceeded to Wave-C and now sells the result. This is a self-falsification of the protocol's own discipline — the *exact* failure mode CLAUDE.md Rule 28's pre-registration was supposed to prevent. The fact that no one in the auto-checkpoint loop or the auditor team caught this means the protocol does not work as advertised. R4 already named this; my contribution is the file:line.

3. **Reproducibility plumbing is fast-mode (benchmark=True, no determinism, no worker seeding); headline σ is contaminated by kernel-choice noise.** Combined with cuDNN nondeterminism and missing `use_deterministic_algorithms(True)`, a fresh engineer running the same config gets a non-bit-exact run, and the +1 pp signal is inside the kernel-choice band. No headline number in this repo is bit-reproducible. This is fine for *screening* (Rule 28); it is not fine for an "EVALUATION-tier" claim. The 38-rule discipline writes about reproducibility but the code skipped the boring half.

## 25+ concrete improvements

| # | What's wrong | What to do | Severity | Priority |
|---|---|---|---|---|
| 1 | Phase-9i prior runs at 2× baseline FLOPs (BLOCKER 1) | Pin priors to baseline 41M FLOPs via budget reduction; re-run n=3. If lift survives → real. If not → retract. | XL | P0 |
| 2 | Modern baseline 0.6360 below pre-reg floor 0.68 (BLOCKER 2) | Debug recipe: drop RandAugment to (N=1, M=9), drop Random Erasing entirely (Müller&Hutter show no benefit at 32×32), increase warmup to 10 ep. Re-baseline before any prior claim. | XL | P0 |
| 3 | `set_seed` doesn't enable determinism; cudnn.benchmark=True (BUG 7, 10) | Add `headline_mode=True` to set_seed: deterministic=True, benchmark=False, worker_init_fn for DataLoader. Re-run n=7 default cert in headline mode; report σ. | L | P0 |
| 4 | Mixup λ-flip silently shifts effective Beta distribution (BUG 4) | Document or remove; add test asserting `lam >= 0.5` post-flip; track effective alpha. | M | P1 |
| 5 | CutMix degenerate boxes pollute λ_eff distribution (BUG 5) | Add 1000-trial test asserting box non-degeneracy rate ≥ 0.9 at α=1; reject α-CutMix pairs with high degeneracy. | M | P1 |
| 6 | EMA finalization clobbers BN buffers (BUG 9) | Add `_recalibrate_bn(model, train_loader)` post-EMA-load; one fwd pass over train set in eval mode with BN momentum=None. | L | P1 |
| 7 | `train_top1`/`generalization_gap` silently wrong under mixing (BUG 3) | When mixed=True, also evaluate the model on un-mixed train set once per epoch; report both. | M | P1 |
| 8 | `composite_score` floors params at 0.001M (BUG 12) | Raise on params < 1000; tighten composite assertions. | S | P2 |
| 9 | 20 TODO "runner wiring" comments in src (BUG 20) | Mark unwired modules in IDEA_TABLE.md as "module-only, not in sweep matrix"; reduce "implemented" count from 74 to actual reachable count. | L | P1 |
| 10 | Citation Rigor regex over/under-matches (BUG 16) | Tighten to require arXiv 4.4-5 digit ID + at least 5 words in relevance clause + author-year-style ID also matched. | S | P2 |
| 11 | append-only log has no integrity check (BUG 13) | Add flock + fsync; assert uniqueness of (tag, seed) at append. | M | P2 |
| 12 | randaugment per-worker RNG (BUG 8) | Set `worker_init_fn=seed_worker` on every DataLoader; document it. | M | P1 |
| 13 | Composite metric is project-specific and obscures Pareto | Drop composite from headline tables; show top1, params, FLOPs, latency separately. | L | P1 |
| 14 | Modern recipe is ImageNet-tuned RandAugment/RandomErasing at 32×32 | Adopt the He-2019 / DeepLearningTuningPlaybook CIFAR-specific recipe: RandAugment(1, 9), no RandomErasing, Mixup α=0.1, no CutMix, EMA 0.9999, 200 ep. Re-baseline. | L | P0 |
| 15 | ResNet-20 in 2026 is not a publishable scaffold; need ImageNet evidence | See SOTA-migration path below. | XL | P0 |
| 16 | No iso-FLOPs control for any "prior wins" claim | Add iso-FLOPs column to every leaderboard; reject any winner not iso-FLOPs ±5%. | XL | P0 |
| 17 | No multi-task / multi-dataset transfer for any winner | Replicate top-3 winners at Imagenette n=3 50 ep. ~9 GPU-h. Reject any tag that doesn't transfer. | L | P0 |
| 18 | Auto-checkpoint loop commits every ~10 min producing 1000+ commits | Add a `--squash-checkpoints` post-campaign step: `git rebase -i` collapse all auto-commits into one labeled-by-tag commit. Current history is unreadable. | M | P2 |
| 19 | 217 per-experiment HTML pages + 35 hypothesis × 8 group pages = aesthetic, not informative | Cut to 1 aggregate + 3 winner pages + 3 falsifier pages. R4 already named this; my contribution is "what to keep". | M | P2 |
| 20 | The runner has no `--dry-run` to validate config without launching | Add `--dry-run` that builds the model, runs 1 batch, prints memory, prints latency, prints expected wall-clock — no full fit. | S | P2 |
| 21 | The runner has no `--resume` for crash recovery beyond the auto-checkpoint loop | Save optimizer + scheduler + RNG state every epoch to `state.ckpt`; on launch, resume if present. | M | P2 |
| 22 | TrainConfig has 25+ fields; no schema validation | Use `pydantic.BaseModel` with Field constraints (ge=0 for alpha, le=1 for decay). Currently a bad config silently runs to completion before failing. | M | P2 |
| 23 | EMA + pruning silently falls back to non-strict load (BUG 18) | Log every dropped key; raise if shape mismatch. | M | P2 |
| 24 | BettiLoss path doubles forward compute under bf16 (BUG 19) | Cache `stagewise_features` from the main forward; reuse for Betti term. | M | P2 |
| 25 | Dashboard pages duplicate the leaderboard; show stale "screening" verdicts after Phase-9i corrections | Add a `last_updated` field per page; auto-invalidate if `experiments/cifar100/<tag>_seed*/metrics.json` mtime exceeds page mtime. | M | P2 |
| 26 | "780+ tests" badge is misleading — shape-only tests dominate | Tag every test with `@pytest.mark.{shape,mechanism,regression}` and report the breakdown on the badge. Mechanism count will be ~80, not 780. | L | P1 |
| 27 | `cudnn.benchmark = True` is set globally inside `set_seed` (BUG 10/11) | Move to a separate `enable_fast_mode()` helper; default to deterministic for headline runs. | M | P1 |
| 28 | Composite SHA-256 fingerprint enforces the wrong invariant | The fingerprint guards string identity, not semantic identity. A rename "params_M" → "params_million" would change the SHA but keep the same metric. Replace with a tuple `(operation_graph_hash, weight_tuple)`. | S | P2 |
| 29 | No external auditor in the loop (R4 also flagged) | Spend $20 of GPT-5/Gemini-3 API on the 10 MAJOR/BROKEN findings; report agreement. | L | P0 |
| 30 | Repo root has 4 files per Rule 31 but `paper/`, `audits/`, `experiments/`, `experiments_modern/` all collide as semi-roots | Move `experiments_modern/` under `experiments/modern/`; one of these directories' results is unreferenced by FINDINGS.md. | S | P2 |
| 31 | No FLOP target check at config-load time | Add a `flops_target` field to YAML; runner asserts measured FLOPs are within ±10% before launching the full fit. Would have caught BLOCKER 1. | L | P0 |
| 32 | `convergence/PLAN.md` lists override snippets but they aren't checked in as YAML | Commit `configs/cifar100_modern_200ep_{pair_gm_pdw,slot_act_sine,sg_only_phi_budget}.yaml` for reproducibility. | M | P0 |

## SOTA-benchmark migration path under laptop-4090 constraint

The single most important section. The right benchmark suite for this work on a single 4090 Laptop with 16 GB VRAM, sequenced to make the paper credible at a top-tier venue. Total budget ~250 GPU-h, ~10 weeks at 25 h/week. The author already has the auto-checkpoint loop and per-experiment archive — the infrastructure pays for itself here.

**Stage 1 — Validate the recipe on a real small dataset (1 week, ~25 GPU-h).** The current modern-recipe baseline failed PLAN.md's own band. Before anything else: re-baseline.

- **Imagenette-160 (10 classes, 160×160)** with ResNet-20-CIFAR-shaped width but 160×160 input → about 7M effective FLOPs per image, ~1.2 h per 10-ep run on a 4090 Laptop bf16 batch 128. Run 5 seeds × 10 ep × 3 recipes (legacy, modern-naive ImageNet-tuned, modern-CIFAR-tuned-per-Tuning-Playbook). Goal: identify the recipe that lands inside the published Imagenette small-model band (90%+ for ResNet-class). **5 h total. Wave-0.**
- If Wave-0 modern-CIFAR-tuned recipe lands in band, that recipe replaces the current `configs/cifar100_modern_*.yaml`. Re-baseline CIFAR-100 at n=3 200 ep. If baseline now lands in [0.68, 0.72]: proceed. If not: file a "the recipe doesn't transfer to 32×32" honest finding and pivot to Imagenette as the primary substrate.
- **Total Wave-0: ~25 GPU-h.**

**Stage 2 — Iso-FLOPs Pareto frontier on Imagenette (2 weeks, ~50 GPU-h).** This is what the paper is missing: a real Pareto curve.

- 4 models at iso-FLOPs (matched to ResNet-20-CIFAR ~41M FLOPs scaled to 160×160 ≈ 1.0 G FLOPs): ResNet-20-Imagenette, RegNetX-200MF (the literature anchor for H09 that the paper avoids), ConvNeXt-V2-Femto, ViT-Small at patch-16 160×160 (~22M params). 5 seeds each, 50 ep, modern-CIFAR-tuned recipe.
- This single table gives the reader a clean Pareto frontier and a fair RegNet comparator the paper currently lacks. The H09 phi_budget claim is then evaluated as: "does φ-allocation Pareto-dominate RegNetX-200MF at this FLOP budget?" If yes, real result. If no, honest negative.
- **Total Wave-1: ~50 GPU-h.**

**Stage 3 — Tiny-ImageNet (200 classes, 64×64) for the priors paper (3 weeks, ~80 GPU-h).** The 200-class regime is where ResNet-style priors actually matter (CIFAR-10 noise floor too high; CIFAR-100 100 classes too few; Tiny-ImageNet is the sweet spot for a single-4090 study).

- Baseline ResNet-20-TinyImagenet at iso-FLOPs to Imagenette (~1.0 G), 5 seeds, 80 ep, ~4 h per seed = 20 h.
- Three priors (pair_gm_pdw, slot_act_sine, sg_only_phi_budget) at iso-FLOPs, 5 seeds each = 60 h.
- Statistical test: paired Wilcoxon n=5, paired-bootstrap CI, Holm-Bonferroni k=3, plus iso-FLOPs Pareto report. If priors clear at iso-FLOPs on a 200-class dataset, the paper has a real headline result.
- **Total Wave-2: ~80 GPU-h.**

**Stage 4 — ImageNet-100 at 160×160 with FFCV (3 weeks, ~80 GPU-h).** Yes, on a 4090 Laptop. FFCV's optimized pipeline + bf16 + batch 256 makes ResNet-50 at 160×160 do an epoch in ~8 min on a 4090; 100 epochs is ~13 h per seed. Take the iso-FLOPs Pareto winner from Wave-2 and graduate it to ImageNet-100, 3 seeds, ~40 h. Baseline ResNet-50 at 160², 3 seeds, ~40 h.

- Setup cost: ~2 days to FFCV-package ImageNet-100. Use the `state-spaces/Mamba` codebase as the FFCV reference; it has a clean Mamba+FFCV pipeline that adapts to ConvNet.
- This is the experiment that turns the paper from "12-ep CIFAR-10 ablation" into "actual ML." The wave-3 result is the headline figure.
- **Total Wave-3: ~80 GPU-h.**

**Stage 5 — Spherical MNIST for H71 IcosaRoPE3D (1 week, ~15 GPU-h).** The sole NOVEL+TESTABLE sci-critic survivor in the entire 84-hypothesis substrate is H71 and it's untested. R4 said run it; I agree, and here's how on the laptop:

- Spherical MNIST 60×60 (e3nn standard), ViT-Tiny patch-12, rotated test set, 100 ep, 5 seeds. ~3 h per seed = 15 h total.
- If H71 lifts a non-equivariant ViT-Tiny by +3 pp on a rotation-equivariance task, the paper has a real "we found an actual prior that helps where the symmetry exists" result. Honest negative is also publishable.
- **Total Wave-4: ~15 GPU-h.**

**Total roadmap: 250 GPU-h, ~10 weeks at 25 GPU-h/week on a single 4090 Laptop.** At completion: the paper carries (a) an iso-FLOPs Imagenette/Tiny-ImageNet/ImageNet-100 Pareto report against RegNetX/ConvNeXt-V2-Femto/ViT-Small; (b) a clean H71 result on Spherical MNIST; (c) the protocol-as-meta-research framing of the original paper with the priors as a *secondary* case study where one of them survived and seven fell. This is the credible paper R4 also called out, with a concrete laptop-realistic budget.

**The single least credible piece of the current submission, by a large margin, is that it has zero ImageNet evidence and yet claims to be moving the field forward.** A 4090 Laptop with FFCV can land an ImageNet-100 result in 40 GPU-h. The fact that this hasn't been done in 6 weeks of campaign is the single biggest signal to a top-tier reviewer that the author is optimising for paper-shipping over knowledge-creation. The fix is to stop running new CIFAR-100 phases and start the FFCV ImageNet-100 pipeline tomorrow. If the priors don't lift on ImageNet-100, that is itself a publishable result and a much more honest paper.

## Closing note

The engineering tooling is genuinely good. The Fixer-with-mechanism-pinning-test contract (`test_phi_budget_realised_ratio` in particular) is a beautiful primitive that I would lift into my team's eval infra tomorrow. The auto-checkpoint loop is the right pattern. The 38 rules contain at least 10 load-bearing operational disciplines I would copy. The reasoning-blob word-count gates are heavy-handed but mechanical, and mechanical is the right answer for LLM-agent pipelines.

But the empirical story is broken in three nested ways: (a) the headline comparison isn't iso-compute (BLOCKER 1); (b) the modern recipe under-performs its own pre-registered floor and the campaign proceeded anyway (BLOCKER 2); (c) reproducibility plumbing is fast-mode while reporting "EVALUATION-tier" claims (BUG 7/10). Any one of these is a publication-blocker. All three together mean the priors-survive-iso-modern-recipe headline does not survive a 30-minute diff of metrics.json files.

R4 said: "stop running experiments for 48 hours and write the protocol paper." I would add: do those 48 hours, then spend the next 250 GPU-h on the SOTA-migration path above. The result of that program is one of two papers: a *protocol paper* with credible ImageNet evidence on the priors-as-case-study, or a *retraction paper* explaining why CIFAR-only results are systematically unreliable for LLM-agent pipelines. Both are publishable. The current trajectory is neither.

Decision: would not hire as RS today. Would hire as senior eval-infra engineer on a 90-day conversion plan: if Wave-0 + Wave-1 (75 GPU-h) lands a real iso-FLOPs Pareto report by end of conversion, the candidate has earned the RS seat. If not, the conversion ends and the work stays in eval-infra. The discipline is real; the science instincts need a forcing function.
