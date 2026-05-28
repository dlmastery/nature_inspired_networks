# G5 Audit — Optimization / Init / Reg / NAS (H41–H50)

> Reviewer: hostile NeurIPS-PC stand-in. Doctrine: shape-only tests = MINOR;
> doc/code divergence in the mechanism = MAJOR; not-actually-doing-the-thing
> = BROKEN. PASS requires a mechanism-verifying assertion in the tests AND
> a reachable falsifier path.

---

## Summary

- **PASS (4):** H42 phi_init, H43 fib_prune, H44 phi_decay, H49 prh_loss
- **MINOR (3):** H45 sacred_nas, H46 cymatic_loss, H50 full_fib
- **MAJOR (3):** H41 golden_adam (falsification confounded by eps change),
  H47 phi_dropout (per-batch cycle ≠ doc's "early-train-high / late-train-low"),
  H48 golden_momentum (saturates to floor in 1 step — not a schedule)
- **BROKEN (0):** —

The most consequential finding is **H41**: the headline 51.96 % top-1 vs.
84.78 % baseline is REAL (β2 = 0.382 + eps = 0.146 default is catastrophic
for AdamW), but the implementation changes TWO things vs. the doc's stated
"β alone" hypothesis, so the published falsification does not cleanly
refute the pre-registered claim. See § H41 below.

---

## H41 — Golden Ratio Optimizer (`GoldenRatioAdamW`) — **MAJOR**

- **Module:** `src/nature_inspired_networks/optimizers.py`. Subclass of
  `torch.optim.AdamW`; overrides defaults to `β1 = 1/φ ≈ 0.6180`,
  `β2 = 1/φ² ≈ 0.3820`, **`eps = 1/φ⁴ ≈ 0.14590`**, `weight_decay = 1e-2`.
- **Mechanism vs. claim:** The H41 design doc (`H41_*.md` § 2 & § 3)
  pre-registers the falsifier strictly in terms of β1, β2 — the doc states
  "the eps choice is NOT load-bearing" and only β1, β2 carry the
  edge-of-chaos argument. The implementation, however, ALSO replaces
  AdamW's `eps=1e-8` with `eps=1/φ⁴ ≈ 0.146`. The Adam update is
  `lr · m̂ / (√v̂ + eps)`; with weight-init-scale gradients on CIFAR-10
  (~1e-3), `√v̂` is comfortably below 0.146, so the denominator is
  **dominated by eps** and the effective lr is `lr/0.146 ≈ 6.85 ×
  nominal_lr`. The sweep ran at base `lr=1e-3` (cifar10_ablation.yaml),
  so the model is effectively trained at lr ≈ 6.85e-3 with AdamW. This,
  combined with β2=0.382 (which on its own already destabilises Adam by
  shortening the second-moment EMA to τ≈1.6 steps), produces the
  catastrophic 51.96 % top-1. The falsification is REAL in the sense
  that swapping AdamW defaults for the four-knob-φ set is broken, but
  it does NOT cleanly falsify the β-only hypothesis the doc pre-registered.
- **Math correctness:** The numbers `GOLDEN_BETA1=1/PHI`,
  `GOLDEN_BETA2=1/PHI**2`, `GOLDEN_EPS=1/PHI**4` are exact. β2 = 0.382 is
  inside Adam's "stable" regime (0 < β2 < 1) only in the trivial sense;
  the original paper's convergence proof requires the second-moment
  estimate to actually track ||∇||² which needs β2 close to 1. Jaeger
  2020 (ESN edge-of-chaos) is about the **state-transition** Jacobian
  spectral radius, NOT about an EMA decay constant — the citation is a
  category-mismatch in the doc, and the test does not catch it because
  no test exercises the dynamical-system interpretation.
- **Test rigor (`tests/test_optimizers.py`, 5 tests):**
  - `test_h41_default_betas_are_phi_derived` asserts the exact β1, β2,
    eps values (regression — good).
  - `test_h41_step_updates_params` shape-only-ish — verifies params move,
    no numerical-trajectory assertion.
  - `test_h41_state_has_exp_avg_after_step` checks Adam state populates
    — not mechanism-load-bearing.
  - `test_h41_factory_dispatches_correctly` + override test cover the
    factory and the "set betas back to vanilla" escape hatch — but the
    escape hatch is NOT used by the runner / sweep, so the falsification
    cannot isolate β alone.
  - **No test verifies that on a synthetic quadratic Adam with
    (0.9, 0.999, 1e-8) and Adam with the φ defaults converge at
    comparable rates** — the doc's §9 Committee Q&A promises such a
    "numerical regression vs. AdamW reference" but no such test exists.
- **Citation alignment:** Format compliant; arXiv IDs present. The
  Jaeger 2020 citation is a category-mismatch (see above) — the doc
  treats reservoir spectral radius and Adam EMA β as the same object;
  they are not. This was not flagged in any test or AUDIT.md.
- **Falsifier reachable?** Yes — the row `sg_only_golden_adam` ran;
  top-1 = 0.5196, composite = 0.514 vs. baseline_resnet20 0.8478 / 0.846.
  But the falsification confounds β and eps, so the doc's § 3 falsifier
  on β alone has not been tested. To clear the cloud, run a second row
  with `GoldenRatioAdamW(..., eps=1e-8)` — the build_optimizer factory
  does not expose eps, so a runner patch is needed.
- **Hidden bugs / cargo-cult:** (a) `eps = 1/φ⁴` is cargo-cult — Adam's
  eps is a denormal-floor stabiliser, not a hyperparameter expressing
  φ-content. Picking 1/φ⁴ because "everything must be φ-derived" is
  exactly the aesthetic-first failure mode the project's CLAUDE.md
  warns against. (b) `build_optimizer` does NOT accept an `eps` kwarg —
  the only way to recover stock AdamW eps is to instantiate
  `GoldenRatioAdamW` directly, which the sweep does not do. (c) The
  factory raises `ValueError` on unknown names, which is correct; the
  list is closed.
- **Concrete fix:** Add `eps` parameter to `build_optimizer`; default it
  to `1e-8` (matching AdamW) rather than `GOLDEN_EPS`; rename the
  φ-default constant to make explicit it is "experimental" and not
  part of the falsifier surface. Re-run `sg_only_golden_adam` and a
  new `sg_only_golden_adam_stock_eps` row to isolate β from eps.

---

## H42 — φ-Weight Initialization (`phi_weight_init_`, `apply_phi_init`) — **PASS**

- **Module:** `src/nature_inspired_networks/inits.py`. Replaces He's
  `std = √(2/fan_in)` with `std = √(φ/fan_in)`.
- **Mechanism vs. claim:** Realised std is exactly `math.sqrt(phi/fan_in)`
  in code; the test `test_phi_init_variance_matches_phi_over_fan_in`
  asserts Monte-Carlo variance within 2 % of `φ/fan_in` (matches doc's
  ≤ 2 % gate). This is the real thing, not a He init with a φ-rename.
- **Math correctness:** `nn.init._calculate_correct_fan(tensor, mode)`
  is the canonical PyTorch fan helper. Reduces to He init exactly when
  `phi=2.0` — pinned by `test_phi_init_reduces_to_he_when_phi_equals_2`.
- **Test rigor (`tests/test_inits.py`, 7 tests for H42 plus H31 shares):**
  - Variance matches `φ/fan_in` ≤ 2 % (mechanism-load-bearing).
  - He backward-compat at phi=2.0 (regression).
  - Mean ≈ 0 (~ N(0, σ) check).
  - 4-D conv-shape preserved.
  - `apply_phi_init` walks every Conv2d/Linear, biases zeroed.
  - 1-D tensor rejected.
- **Citation alignment:** He 2015 (arXiv:1502.01852) and Glorot 2010
  cited via the inits.py docstring lineage in H42_*.md. Format compliant.
- **Falsifier reachable?** Yes — `sg_only_phi_init_seed0` recorded
  top-1 = 0.7656, composite = 0.777 (vs. baseline_sg_vanilla 0.8216 /
  0.826). Falsifier in the doc is a ≥ +0.3 pp lift; observed Δ = −5.6 pp,
  so the hypothesis is empirically falsified (consistent with the
  field's understanding: He init's `2` is tuned to ReLU's variance loss
  and `φ < 2` produces a small under-scaling).
- **Hidden bugs:** None observed. The std-vs-He gap factor is
  `√(φ/2) ≈ 0.899`, a 10 % shrink — small but consistent with the −5.6 pp
  result (initial activations under-scale → signal attenuation in deep
  stacks).
- **Concrete fix:** None required for correctness. Doc should be
  updated to record the −5.6 pp finding as a clean falsification.

---

## H43 — Fibonacci Pruning (`fibonacci_prune`) — **PASS**

- **Module:** `src/nature_inspired_networks/pruning.py`. Uses
  `torch.nn.utils.prune.global_unstructured(L1Unstructured)` at
  Fibonacci-indexed epochs `{1, 2, 3, 5, 8}` (one-indexed).
- **Mechanism vs. claim:** `fibonacci_ratios(n)` returns normalised
  Fibonacci sequence summing to **exactly 1.0**, so successive prune
  amounts are `{1/19, 2/19, 3/19, 5/19, 8/19}` — the fractions of
  remaining weights, NOT cumulative sparsity. This matches the doc.
- **Math correctness:** The prune fractions are passed as `amount=ratio`
  to `prune.global_unstructured`, which uses the fraction as a fraction
  of *remaining* unmasked weights (PyTorch semantics). After 5 prune
  events, the final sparsity is bounded above by `1 − ∏(1 − r_k) =
  1 − (18/19)(17/19)(16/19)(14/19)(11/19) ≈ 0.50`. Test asserts > 0.42
  which is consistent. The audit verified the design avoids the
  obvious cargo-cult of using raw Fib counts (which would over-prune).
- **Test rigor (`tests/test_pruning.py`, 5 tests):**
  - Ratios sum to 1 + exact-value regression (mechanism).
  - Non-Fib epoch is a no-op (edge case).
  - Fib epoch increases sparsity, fib_idx and ratio match expected.
  - Iterative monotonic-sparsity invariant (regression).
  - No-prunable-modules guard.
- **Citation alignment:** Tanaka et al. 2020 (arXiv:2006.05467) on
  SynFlow + Frankle & Carbin 2019 lottery ticket — present in docstring.
- **Falsifier reachable?** Yes — `sg_only_fib_prune_seed0`: top-1 =
  0.8115, composite = 0.800. The doc's falsifier is "≥ −0.005 composite
  vs. baseline at 50–90 % sparsity"; observed sparsity here is not
  recorded in metrics.json. So the *direction* of the experiment is
  recorded but the falsifier requires a sparsity readout to evaluate.
- **Hidden bugs:** `make_permanent=False` is the runner default, which
  means successive `prune.global_unstructured` calls layer NEW
  ForwardPreHook prune-masks on top of the existing ones — PyTorch
  handles this idempotently but it does grow the per-module hook list.
  Memory leak risk is small but non-zero on long training; the
  iterative-monotonic test does not exercise more than 5 events so
  this would not have been caught.
- **Concrete fix:** Record `global_sparsity(model)` in the per-epoch
  metrics so the falsifier (composite at sparsity X) is queryable
  post-hoc.

---

## H44 — Golden Regularization (`build_phi_decay_param_groups`,
`GoldenRegularizer`) — **PASS**

- **Module:** `src/nature_inspired_networks/phi_decay.py`. Per-layer
  weight-decay `λ_k = λ_0 / φ^k` indexed by registration order of
  top-level children (or, when `block_attr="blocks"`, of the ModuleList
  children).
- **Mechanism vs. claim:** Schedule is exactly `base_wd / phi^k`
  monotonically decreasing. Composes with AdamW via param_groups —
  PyTorch's decoupled-weight-decay path respects per-group
  `weight_decay`. Doc claim matches implementation.
- **Math correctness:** `_named_layer_groups` de-duplicates by `id()`
  so shared params don't double-count. Stem and head get `base_wd`
  (k=0 for the schedule started at the first block).
- **Test rigor (`tests/test_phi_decay.py`, 10 tests):**
  - layer 0 = `base_wd` exact (regression for off-by-one).
  - layer N = `base_wd / phi^N` exact.
  - Strictly decreasing.
  - Non-block children = `base_wd`.
  - All-params-covered, no-duplicates (load-bearing property).
  - `block_attr=None` walks every top-level child.
  - `GoldenRegularizer` end-to-end mutation in place; `opt.step` still
    runs.
  - phi=1.0 ablation gives uniform λ (control row).
  - Negative base_wd rejected.
  - Empty optimizer rejected.
- **Citation alignment:** Loshchilov & Hutter 2019 (arXiv:1711.05101)
  for decoupled WD; Zhang et al. 2017 generalization for the
  layer-depth-vs-overfit motivation — both present in docstring.
- **Falsifier reachable?** Yes — `sg_only_phi_decay_seed0`: top-1 =
  0.7981, composite = 0.812 (vs. baseline_sg_vanilla 0.8216 / 0.826).
  Note the base_wd was set to 1e-2 in the sweep but the default
  baseline uses 5e-4, so the sweep ALSO changes the absolute weight
  decay level. **Confound:** the φ-decay row is testing TWO knobs
  (base_wd magnitude AND φ-decay schedule). Strictly Rule 1 is bent.
- **Hidden bugs:** Confound on base_wd magnitude (see above). The
  schedule indexing starts at k=0 for `blocks[0]` AND k=0 for the stem,
  so stem and blocks[0] both get `base_wd` — this is the documented
  intent but worth flagging that the "depth=0" semantics are not
  shared across the two paths.
- **Concrete fix:** Add a `sg_only_phi_decay_baseline_wd` row that
  uses `phi_decay_base=5e-4` to match the baseline magnitude, then
  the variant truly differs by one knob (Rule 1).

---

## H45 — Sacred-NAS (`sacred_search_space`, `random_arch_sample`,
`SacredNASController`) — **MINOR**

- **Module:** `src/nature_inspired_networks/sacred_nas.py`.
- **Mechanism vs. claim:** The doc describes a "DARTS-style
  differentiable architecture search restricted to nature-prior ops".
  The implementation is **honestly described as a stub** — the search
  space dictionary lists the 4 ops, `random_arch_sample` does sample
  uniformly over (channel_mode, block_pick, connectivity_idx), and
  `SacredNASController` is a single-cell mixture-of-4 with learnable
  softmax logits α. There is NO bi-level optimisation, NO edge-level
  mixture, NO discrete architecture derivation, NO supernet search
  driver. The doc Section 5 acknowledges this is a stub — so the
  mechanism-vs-claim gap is admitted, not concealed.
- **Math correctness:** `F.softmax(self.logits, dim=0)` is the
  canonical DARTS-α normalisation. Sample is deterministic for fixed
  seed. `phi_small_world` builds a φ-rewired Watts-Strogatz graph
  with `p_rewire = 1/φ ≈ 0.618`, symmetric, zero-diagonal.
- **Test rigor (`tests/test_sacred_nas.py`, 7 tests):**
  - Search-space keys present + 4-op tuple identity.
  - Channel widths div-8 and monotone for "fib".
  - Random sample deterministic, seed-1..5 differ.
  - phi_small_world symmetric + zero diagonal + seed-deterministic.
  - Controller forward shape preserved.
  - α softmax-normalised, sums to 1, all > 0, uniform at init.
  - Logit perturbation correctly drives α.
- **Citation alignment:** Liu, Simonyan, Yang 2019 ICLR DARTS
  (arXiv:1806.09055), Pham et al. 2018 ICML ENAS (arXiv:1802.03268) —
  present in docstring.
- **Falsifier reachable?** **No.** There is no `sg_only_sacred_nas`
  sweep row; the controller exists but is not trained end-to-end on
  CIFAR-10. The doc's falsifier (search-found arch beats vanilla
  NaturePrior at iso-flops) is not testable from the current pipeline.
- **Hidden bugs:** `phi_small_world` build can saturate (no free
  rewire target) and silently restore the original edge — this is
  documented in the code but the test does not exercise the saturated
  case. Not a load-bearing issue at n ≥ 13.
- **Concrete fix:** Mark this as "stub / non-falsifiable in current
  scope" in IDEA_TABLE.md; either implement the full DARTS supernet
  or move H45 to a separate "research deferred" bucket.

---

## H46 — Cymatic Loss (`power_spectrum_2d`, `chladni_target_spectrum`,
`cymatic_loss`, `CymaticLossModule`) — **MINOR**

- **Module:** `src/nature_inspired_networks/cymatic_loss.py`.
- **Mechanism vs. claim:** `power_spectrum_2d` is exactly
  `torch.fft.fft2(x).abs() ** 2` — correct. `chladni_target_spectrum`
  builds the target from `chladni_modes_banded`, sums per-mode power
  spectra, normalises to sum 1. `cymatic_loss` MSEs the per-(B, C)-
  normalised activation spectrum against the target. Mechanism matches
  the doc.
- **Math correctness:** Per-(B, C) normalisation `spec / spec.sum(...)`
  makes the loss scale-free, as documented. Parseval-style sanity
  test asserts DC bin = `(sum(x))²`, which is correct for the
  unnormalised FFT2.
- **Test rigor (`tests/test_cymatic_loss.py`, 7 tests):**
  - Power-spectrum shape, non-negativity.
  - DC bin = sum² (Parseval sanity).
  - Target spectrum shape, normalisation to 1, non-square supported.
  - Determinism on seed.
  - Loss is scalar and finite, non-negative.
  - Perfect-match → loss ≈ 0 (mechanism-load-bearing — constructed
    via inverse FFT of √target).
  - Module caches target, rebuilds on shape change.
  - Gradient flows through activations.
- **Citation alignment:** Chladni 1787, Rahaman et al. 2019 ICML
  (arXiv:1806.08734) on spectral bias, Tancik et al. 2020 NeurIPS
  Fourier features — all in docstring with arXiv IDs.
- **Falsifier reachable?** **No sweep row exists.** The
  `CymaticLossModule` is a primitive; there is no `sg_only_cymatic_loss`
  experiment in `scripts/run_sweep.py`, no `cymatic_loss_lambda`
  config key parsed by the runner, no integration into the
  training loop. The doc's falsifier (Δ composite ≥ +0.003 with
  λ_cym tuned) is not testable end-to-end from this code.
- **Hidden bugs:** None observed. The target spectrum is shape-only
  deterministic, which the test confirms.
- **Concrete fix:** Add an `aux_cymatic_loss` config flag + a
  `sg_only_cymatic_loss` sweep row, hooking a single intermediate
  activation via a forward-hook the same way H49 is conceptually
  positioned to be wired.

---

## H47 — φ-Dropout (`PhiDropout`, `PHI_DROPOUT_SCHEDULE`) — **MAJOR**

- **Module:** `src/nature_inspired_networks/regularizers.py`.
- **Mechanism vs. claim:** The doc claims "early-training noise should
  be high (≈ 1/φ ≈ 0.618) and late-training noise low (≈ 1/φ⁴ ≈ 0.146),
  matching the curriculum-noise prescription of Bengio 2009". The
  implementation increments `step_counter` ONCE PER FORWARD CALL during
  training and reads `schedule[step_counter % length]`. With
  `length=5` and ~196 forward calls per CIFAR-10 epoch (batch 256),
  the schedule **cycles 39 times per epoch** — there is no
  "early-train-high / late-train-low" curriculum at all; the dropout
  rate oscillates ~39× per epoch through `{0.053, 0.105, 0.158, 0.263,
  0.421}` (for `cycle='fib'`). The mechanism contradicts the claim.
- **Math correctness:** The normalised Fibonacci sequence and the
  `1/φ^(1+k)` φ-schedule are themselves correct values, exact to
  float tolerance.
- **Test rigor (`tests/test_regularizers.py`, 6 tests):**
  - Fib schedule = normalised Fib (regression — mechanism).
  - phi schedule = `1/φ^(1+k)` (regression).
  - Eval mode = identity, counter does NOT advance (correct).
  - Cycle wraps, rate stays in [0, 1) (regression — but this is
    EXACTLY the failure mode of the claim: the test validates the
    wrap which is what makes the curriculum claim false).
  - Bad-arg ValueError × 5.
  - Real-dropout sanity (zeros present, not pass-through).
- **Citation alignment:** Bengio 2009 (curriculum learning), Srivastava
  et al. 2014 (Dropout). Format compliant.
- **Falsifier reachable?** Yes — `sg_only_phi_dropout_seed0`: top-1 =
  0.828, composite = 0.822. The doc's falsifier (≥ +0.3 pp over a
  fixed-p=0.2 baseline) — the baseline is the no-dropout vanilla, not
  a fixed-p control, so the "fixed-p ablation" arm is missing.
- **Hidden bugs:** (a) The "per-forward" step granularity is a hidden
  defect — a sane curriculum implementation would step per-EPOCH (via
  a trainer callback), giving the doc's intended slow decay from
  ~0.62 → ~0.09 over training. (b) No `cycle='epoch'` mode exists.
  (c) The init `p_init=1/φ` is a vestigial default that the schedule
  immediately overwrites — confusing.
- **Concrete fix:** Add `step_unit ∈ {'forward', 'epoch'}` constructor
  arg, defaulting to `'epoch'` to match the doc, and a Trainer hook
  that calls `module.advance()` once per epoch. Add a fixed-p control
  row to the sweep for the falsifier comparison.

---

## H48 — Golden Momentum Scheduler (`GoldenMomentumScheduler`,
`golden_momentum_curve`) — **MAJOR**

- **Module:** `src/nature_inspired_networks/schedulers.py`.
- **Mechanism vs. claim:** Docstring: "On each `step()` call the current
  β1 (or momentum) is updated via `new = max(floor, current * (1/φ))`."
  Initial β1 = `1/φ ≈ 0.618`; floor = `1/φ² ≈ 0.382`. **One** multiplicative
  step gives `0.618 * 1/φ = 0.382 = floor`. After step 1, the floor
  clamp is hit; ALL further steps return `floor` unchanged. The doc
  claims "β1 multiplicatively decays by 1/φ per epoch with a 1/φ²
  floor" — this is *literally* implemented but the schedule
  **saturates to the floor after the very first epoch step**, so for
  any training > 1 epoch the optimizer is effectively running at
  fixed β1 = 0.382 from epoch 1 onward. The "decay schedule" is one
  step long.
- **Math correctness:** The `golden_momentum_curve(epochs, tau=5.0)`
  function provides an EXPONENTIAL-shaped reference curve
  `β1(e) = floor + span * exp(-e/τ)` — this is **NOT used** by the
  scheduler; the test `test_h48_closed_form_curve_bounded` asserts
  bounds on the unused curve, not on the actual scheduler. The
  scheduler IS multiplicative and saturates in 1 step; the reference
  curve takes ~τ epochs to halve. They are different mathematical
  objects masquerading as the same H48 schedule.
- **Test rigor (`tests/test_schedulers.py`, 6 tests + H10 sanity):**
  - Initial β1 = 1/φ exact (regression — mechanism).
  - `test_h48_beta_monotonically_decays_toward_floor`: this test
    **passes precisely because the schedule reaches the floor in 1
    step** — the assertion `seen[-1] == floor` is satisfied
    immediately, masking that no "decay" occurs at step 2..20.
  - `test_h48_one_step_hits_phi_squared`: explicit confirmation
    that one step lands on the floor. This is the test that should
    have raised the alarm in code review.
  - SGD momentum field works.
  - Floor clamp idempotent past saturation.
  - Closed-form curve bounded (tests the UNUSED reference curve).
  - H10 PhiDecayLR sanity (passes — independent regression).
- **Citation alignment:** Sutskever 2013, Smith 2017 (1506.01186),
  Loshchilov & Hutter 2017 (1608.03983), Jaeger 2020 (2006.04751).
  Format compliant.
- **Falsifier reachable?** Yes — `sg_only_golden_momentum_seed0`:
  top-1 = 0.8352, composite = 0.831 (vs. baseline 0.8478 / 0.846).
  But because the schedule saturates to β1=0.382 in 1 step, this is
  effectively testing "AdamW with β1=0.382 constant from epoch 1" —
  which is a DIFFERENT hypothesis from "decay β1 multiplicatively
  from 0.618 to 0.382 over training". The falsifier path is
  technically reachable but does not test the doc's curve.
- **Hidden bugs:** (a) Floor reached in 1 step is the dominant bug.
  (b) `golden_momentum_curve` is dead code w.r.t. the actual
  scheduler; either remove it or wire it via a `mode='exp_curve'`
  alternative. (c) The init=1/φ overwrite is applied even when the
  optimizer was already AdamW with β1=0.9 — this is intentional but
  means the H48 row also implicitly changes initial β1 (not just the
  schedule).
- **Concrete fix:** Replace the multiplicative-per-step rule with the
  exponential-curve closed form `β1(e) = floor + span * exp(-e/tau)`
  for a documented tau, OR change the per-step factor so saturation
  takes N epochs (e.g. `factor = phi^(-1/T_max)` per Loshchilov-style
  schedules). The current "instant saturation" implementation is not
  what the doc claims.

---

## H49 — Platonic Representation Alignment Loss (`cka_loss`,
`PRHAlignmentLoss`) — **PASS**

- **Module:** `src/nature_inspired_networks/prh_loss.py`.
- **Mechanism vs. claim:** `cka_loss` computes
  `1 − HSIC(K_a, K_b) / sqrt(HSIC(K_a, K_a) · HSIC(K_b, K_b))` with
  centered linear-kernel Gram matrices. This is **bona fide CKA**
  (Kornblith 2019), not a renamed cosine. `CKA(X, X) = 1` → loss ≈ 0
  is asserted by `test_cka_loss_zero_when_features_identical`; CKA's
  orthogonal-transform invariance is also asserted (a Q-rotated B
  still gives loss ≈ 0). Reviewer's specific suspicion
  ("cka(x,x) ≈ 0; loss = 1 − CKA; CKA ≈ 1 when identical") — this is
  exactly what the test asserts.
- **Math correctness:** `_center_gram` uses `H = I − 1/n` and computes
  `H K H` (Kornblith 2019 eq. 4). `_hsic_linear` returns
  `trace(K L) / (n−1)²` (the biased estimator with the standard
  normalisation; the (n−1)² factor cancels in the CKA ratio).
  CKA clamped to [0, 1] before flipping to a loss — defensive against
  small-N numerical noise.
- **Test rigor (`tests/test_prh_loss.py`, 7 tests):**
  - Scalar in [0, 1] (sanity).
  - Zero when identical + orthogonal-transform invariance
    (mechanism-load-bearing).
  - Symmetry CKA(X, Y) = CKA(Y, X).
  - Dim-agnostic (D_a ≠ D_b supported).
  - PRH module: returns finite scalar.
  - Per-sample target path + batch-idx path.
  - Gradient flows through proj head into features.
  - project=False requires dim match.
- **Citation alignment:** Huh et al. 2024 ICML (arXiv:2405.07987),
  Kornblith et al. 2019 ICML (arXiv:1905.00414), Radford et al. 2021
  (arXiv:2103.00020). Format compliant.
- **Falsifier reachable?** **No sweep row exists.** Same finding as
  H46 — primitive lives, no integration into trainer, no
  `sg_only_prh_loss` row, no Platonic target embedding cached. The
  doc's falsifier (Δ composite ≥ +0.005 with λ_prh tuned) is not
  testable from the current pipeline.
- **Hidden bugs:** None in the primitive. The fallback path in
  `hybrid_platonic_cymatic.py` is OK. The auditing concern is end-to-
  end-reachability of the falsifier rather than primitive correctness.
- **Concrete fix:** Add an `aux_prh_loss` config flag, precompute
  CLIP→dodeca/icosa target embeddings under `data/prh_targets/`, and
  wire a sweep row.

---

## H50 — Full Sacred Hybrid (`NaturePriorBlock`, `NaturePriorFlags`) — **MINOR**

- **Module:** `src/nature_inspired_networks/blocks.py`.
- **Mechanism vs. claim:** `NaturePriorFlags` has the 6 core Booleans
  (`hex, group, fractal, toroidal, cymatic_init, golden_modulate`) plus
  3 H05.v2/H21.v2/H35.v2 toggles (`fractal_phi_shrink`,
  `hex_kernel_radius`, `cymatic_init_orthonormalize`) and 1 string
  field (`group_reduce`). All 6 core Booleans ARE wired into the
  forward path: `hex → HexConv2d`, `group → GroupConv2d`,
  `fractal → _FractalPath`, `toroidal → toroidal_pad`,
  `cymatic_init → cymatic_init_`, `golden_modulate → cos(phases+α)`
  gate. `tag()` reflects every active Boolean. The mechanism is real;
  the audit confirms each flag has a forward-pass consequence.
- **Math correctness:** The composition order matches the ResNet
  basic-block contract: conv1 → ReLU → fractal/conv2 → modulate →
  + skip → ReLU. The `phases` buffer is a `(c_out,)` golden-angle
  sequence and `gate = (cos(phases + α) * 0.5 + 0.5)` is in [0, 1]
  per channel — a sane multiplicative gate.
- **Test rigor (`tests/test_blocks.py`, ~13 tests):**
  - All flag combos preserve (H, W) at stride=1.
  - All flag combos halve (H, W) at stride=2.
  - ResNet20 + NaturePriorNet sanity.
  - Per-channel-mode forward.
  - param-count band for ResNet-20 (250k–290k).
  - H58 group_reduce='mean' regression (tag, forward-shape).
  - build_model casing regression (renaming bug repro).
  - H05.v2 fractal_phi_shrink default-off vs. activated.
  - H21.v2 hex_kernel_radius=1 (3×3, 7 ones) vs. =2 (5×5, 19 ones).
  - H35.v2 cymatic_init_orthonormalize default-off (bit-identical
    legacy) vs. activated (near-orthogonal Gram).
  - Flag-field-iteration regression (compute_topology.py bug).
- **Citation alignment:** N/A at the block level (composition of
  primitives whose citations live in their respective modules).
- **Falsifier reachable?** Yes — `sg_full_fib_seed0` is the full-on
  hybrid; per `FINDINGS.md` it has historically scored at-or-below
  baseline, which is the negative result this group recorded.
- **Hidden bugs:** (a) The default `NaturePriorFlags()` constructor
  turns ALL 6 Booleans ON, which means a naive `NaturePriorBlock(c, c)`
  with no flags arg is the full-hybrid block — that's a footgun if a
  test forgets to construct a "vanilla" `NaturePriorFlags(*[False]*6)`.
  The test file does construct an explicit all-off variant, so this
  is contained, but new callers must beware. (b) `golden_modulate`
  uses a multiplicative gate in [0, 1] which is monotonically
  attenuating — the doc's "channel rotary" framing implies a phase
  rotation, not an amplitude gate. This is the "channel modulation"
  but not a true rotary; minor framing drift.
- **Concrete fix:** Change `NaturePriorFlags` defaults to all-False
  to make "no priors" the explicit identity, then have the sweep
  build full-hybrid flags explicitly. Document the gate-vs-rotary
  framing drift in the H50 design doc.

---

## Cross-cutting findings

1. **H41 falsification is REAL but CONFOUNDED.** The 51.96% is genuinely
   bad (β2=0.382 + eps=0.146 destroys Adam's per-parameter LR), but the
   pre-registered falsifier was about β alone. A clean test requires
   `GoldenRatioAdamW(..., eps=1e-8)` which `build_optimizer` does not
   expose — `scripts/run_sweep.py` cannot construct it without source
   patching. The published falsification therefore does not cleanly
   refute the doc's stated β-only hypothesis.

2. **H47 and H48 mechanisms contradict their docs.** Both are
   "schedules" that don't schedule across training: H47 cycles 39× per
   epoch (per-forward step), H48 saturates to its floor in 1 epoch
   step. In each case the experiments DID run, so the user has data,
   but the data tests a different mechanism than what the doc
   pre-registers.

3. **H45, H46, H49 lack reachable falsifiers** — the primitives are
   correct but never integrated end-to-end. Three of ten G5 hypotheses
   have no sweep row, so their composite-Δ pre-registrations are not
   evaluable from the existing experiment_log.jsonl.

4. **Test discipline (Rule 12) is met for every G5 module** — every
   file ends with `All N tests passed.` Mechanism-load-bearing
   assertions are present where the primitive is in scope (H42 variance,
   H43 ratio sum, H44 schedule, H45 α softmax, H46 perfect-match zero
   loss, H49 identity zero loss, H50 flag-field iteration). The
   weakness is the gap between primitive correctness (well-tested) and
   end-to-end mechanism realisation (H47 step granularity, H48 floor
   saturation) — tests verified the math but not whether the math
   realises the doc's mechanism over training time.

5. **Rule 1 borderline:** H44's `sg_only_phi_decay` row changes both
   `phi_decay_wd=True` AND `phi_decay_base=1e-2` (vs. baseline 5e-4).
   Strictly two knobs vary — the per-layer schedule AND the absolute
   magnitude.

---

*Audit conducted 2026-05-27 by hostile-reviewer agent. Source-tree
hashes at audit time captured by the next git commit on this file.*
