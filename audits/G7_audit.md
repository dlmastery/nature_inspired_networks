# G7 Audit — Cross-Paradigm Hybrids (H61–H75)

Reviewer: NeurIPS-PC-grade audit, heightened skepticism for composition claims.
Date: 2026-05-27. Repo: `dlmastery/nature_inspired_networks`.

## Summary

15 hypotheses audited. Verdict tally:

| Tier   | Count | IDs                                                |
|--------|------:|----------------------------------------------------|
| PASS   |    10 | H61, H62, H66, H68, H69, H70, H71, H72, H73, H75   |
| MINOR  |     2 | H63, H65                                           |
| MAJOR  |     1 | H64                                                |
| BROKEN |     2 | H67, H74                                           |

The G7 group is heterogeneous: most single-axis hybrids (a primitive + an
auxiliary loss, or two primitives stacked) genuinely compose what their
design doc claims. The two BROKEN hypotheses are the ones that depend on
**multi-primitive composition** (H67 "everything on") or on a
**mathematically substantive parameter-tying scheme** (H74). Both fail in
ways that the unit tests do not catch because the tests are
implementation-shape tests rather than design-doc-fidelity tests.

---

## Per-hypothesis findings

### H61 — Sacred Liquid JEPA — **PASS**

`src/nature_inspired_networks/hybrid_liquid_jepa.py` + `tests/test_hybrid_liquid_jepa.py`.

- `LiquidCFCCell.forward` implements the claimed damped-oscillator update
  `h + dt * (-h/τ + W·tanh(x))` (lines 124–126) with `dt = 1/φ`
  (line 92) and softplus-positive `τ` (line 100). Test
  `test_cfc_cell_dt_defaults_to_one_over_phi` asserts the φ step exactly.
- `SacredLiquidJEPA.forward` chains `encode → cfc → predictor` and
  `jepa_loss` correctly detaches the target.
- `NaturePriorBlock` is genuinely instantiated `n_blocks` times and
  exercised in `encode` (lines 199–202).
- Tests cover shape, statefulness, φ-step constant, and stop-grad.
- Minor: only a single CFC step per forward; for a "recurrence" claim, a
  multi-step test (e.g., 3 forward calls with carried state) would
  strengthen rigor. Not blocking.

### H62 — Toroidal-KV Hex Attention — **PASS**

`hybrid_toroidal_hex_attn.py` + `test_hybrid_toroidal_hex_attn.py`.

- `_toroidal_neighbour_grid` does call `toroidal_pad(grid, pad)` from
  `priors` (line 134) — toroidal closure is real.
- `hex_kernel_mask` is registered as buffer and used in
  `scores.masked_fill(mask == 0, -inf)` (lines 116, 166) — the hex
  window is genuinely applied to the attention softmax.
- Test `test_hex_mask_buffer_corners_are_zero` verifies the 7-of-9
  hex pattern; `test_non_square_n_falls_back_gracefully` exercises the
  1-D circular fallback.
- Note: non-square N silently degrades the "toroidal" prior to 1-D
  circular padding. Acceptable since the fallback is documented.

### H63 — Platonic Auxiliary Cymatic Teacher — **MINOR**

`hybrid_platonic_cymatic.py` + `test_hybrid_platonic_cymatic.py`.

- `PlatonicCymaticTeacher.forward` returns `(logits, student_act)`; the
  CKA distillation loss is exposed as a separate helper. The trainer
  is responsible for combining it with CE — there is **no** in-module
  composite loss helper. Acceptable, but the doc claims "training-time
  auxiliary loss is the CKA distance" so the absence of a packaged
  `composite_loss(logits, targets, student_act)` callable invites
  silent ablation by a trainer that forgets to add the auxiliary term.
- `cka_distillation_loss` uses external `prh_loss.cka_loss` if
  available, otherwise an internal CKA implementation. The internal
  CKA is implemented correctly (centred Gram numerator / Frobenius
  product denominator, returns `1 - CKA` as a distance).
- The "Platonic" claim comes from `NaturePriorBlock` (whose
  `group=True` flag toggles a Platonic group conv). H63 doesn't expose
  a per-flag assertion that the Platonic prior is on; this is a weak
  attestation but the flag default in `NaturePriorFlags` controls it.
- **Recommendation**: add a `composite_loss` helper and a test
  asserting both the CE *and* CKA terms contribute non-trivially.

### H64 — Dynamic Growth + Pruning Cycle — **MAJOR**

`hybrid_growth_pruning.py` + `test_hybrid_growth_pruning.py`.

Mechanical wiring is correct:
- `DynamicGrowthCallback` is invoked at `grow_epochs` (line 168) and
  param count increases (test `test_grow_event_increases_param_count`).
- `fibonacci_prune` is invoked at `prune_epochs` (lines 183–188) and
  sparsity rises (`test_prune_event_increases_sparsity`).
- Default schedules `(3,5,8,13)` and `(4,6,9,14)` are disjoint and
  Fibonacci-adjacent — matches the doc.

But two issues prevent a PASS:

1. **Schedule-Fibonacci mapping is opaque**. The prune epoch is
   translated to a `fake_epoch = FIB_SCHEDULE[idx] - 1` (line 182).
   This is brittle: `FIB_SCHEDULE` is imported lazily and its layout
   is not under H64's control. If upstream renames or reorders
   `FIB_SCHEDULE`, H64 silently maps to wrong ratios and the test
   (which only checks `sparsity_after > sparsity_before`) won't catch
   it.

2. **No trainer integration**. The hybrid is a callback orchestrator,
   but the runner doesn't call `sched.step(epoch, model)` anywhere we
   can find in `scripts/run_sweep.py` or `runner.py`. (Out of audit
   scope to fix, but documented here so the empirical CIFAR result
   for H64 is not over-interpreted — if no trainer wires the
   schedule, the H64 experiment reduces to a plain ResNet.) Verify by
   grepping the runner for `GrowthPruningSchedule`.

   Verification grep: `grep -rn "GrowthPruningSchedule" scripts/ src/nature_inspired_networks/runner*` → **no hits found in scope**. The hybrid is not wired into the training loop. The CIFAR-10 ablation row for H64 (if any) is therefore not actually exercising H64's mechanism.

### H65 — Persistent-Homology Betti-Collapse Loss — **MINOR**

`hybrid_ph_collapse.py` + `test_hybrid_ph_collapse.py`.

- `BettiCollapseLoss.forward` correctly composes `(β₀-1)² + (1/φ)·spectral`
  (lines 162–166) and returns a `parts` dict — composition is real.
- `BettiLoss.estimate_betti` is genuinely called (line 162); β₀
  surrogate is the differentiable PH count.
- Optional `cymatic_loss` import: `_HAVE_CYMATIC_LOSS = True` is
  verified (file exists, has `cymatic_loss` symbol at line 108), but
  the hybrid looks for `cymatic_consistency_loss` or
  `spectral_consistency` first, falling through to `cymatic_loss`.
  This works; however, the fallback path
  `_spectral_consistency_fallback` (penalising high-frequency energy)
  has the OPPOSITE sign from a "consistency" loss — it penalises
  energy outside the band but does not pull energy *toward* the
  Chladni mid-band. Borderline correct as a *regulariser*; mislabelled
  as "consistency".
- Test `test_collapse_is_zero_when_beta_0_matches_target` asserts
  collapse term vanishes with a tight cluster. Test
  `test_default_spectral_weight_is_one_over_phi` pins the spectral
  weight.
- **Recommendation**: rename the fallback to
  `_spectral_highfreq_penalty` or actually wire the external
  `cymatic_loss(target_mode, feat)` signature (the external function
  exists at line 108 of `cymatic_loss.py`).

### H66 — Cymatic QKV Kernel — **PASS**

`hybrid_cymatic_qkv.py` + `test_hybrid_cymatic_qkv.py`.

- `cymatic_init_linear_` constructs a `Conv2d(1, out_dim, k)` proxy,
  calls `cymatic_init_(dummy, orthonormalize=True, band=(2,5),
  seed=s)`, and flattens the first `in_dim` entries back into the
  Linear weight (lines 95–100). The init is genuinely Chladni-seeded.
- Q, K, V each get a *distinct* seed (0, 1, 2) so the three
  projections are independent — `test_qkv_projections_differ_after_init`
  proves this.
- `test_cymatic_init_linear_changes_weights` verifies the init is
  not a no-op.
- Forward is standard MHA; cymatic init is the entire mechanism, and
  it actually fires.

### H67 — Full Paradigm Hybrid — **BROKEN**

`hybrid_full.py` + `test_hybrid_full.py`.

**Two of the six advertised priors silently no-op in production.**

1. **`GoldenRoPE` class does not exist**. The optional import
   `from .golden_rope import GoldenRoPE` (line 73) raises
   `ImportError` because `golden_rope.py` exports
   `apply_golden_rope` and `golden_angle_rope_freqs` as functions —
   no class named `GoldenRoPE`. Verified by import:
   ```
   ImportError: cannot import name 'GoldenRoPE' from
   'nature_inspired_networks.golden_rope'
   ```
   `_HAVE_GOLDEN_ROPE` is therefore `False`, and the model always
   uses `_SinusoidalPE` — i.e., **standard Vaswani 2017 absolute
   sinusoidal PE**, not Golden RoPE. The "golden RoPE" prior is
   never engaged.

2. **MetatronGraphLayer construction fails**. Line 184 calls
   `MetatronGraphLayer(width)` with a single positional argument,
   but the constructor (`platonic_graph.py:121-128`) requires
   `(in_dim, out_dim)`. Verified:
   ```
   TypeError: MetatronGraphLayer.__init__() missing 1 required
   positional argument: 'out_dim'
   ```
   This raises inside the outer `try: ... except Exception:` block,
   which silently swaps in `nn.Identity()` (line 187). The
   "Platonic graph" prior is therefore never engaged either.

3. **`which_priors_active` lies**. The property (lines 198–207)
   *hardcodes* `nature_prior_blocks=True, fibottention_attention=True,
   cymatic_qkv_init=True, liquid_cfc=True` without inspecting the
   model. It does not iterate `self.modules()`, does not introspect
   the actual init, and does not check forward instrumentation. The
   test `test_each_load_bearing_prior_is_engaged` is therefore a
   tautology — it asserts hardcoded `True == True`. Only the two
   "optional" priors (`golden_rope`, `platonic_graph`) are reported
   based on actual state, and those are both `False` in practice
   (see #1 and #2 above).

4. **Liquid CFC is degenerate**. Line 243 calls `self.cfc(z, None)`
   exactly once per forward pass with `h=None` (zero-init). One step
   from zero state is `dt * W·tanh(z)` plus the new (identical-)decay
   — i.e., a learned MLP-ish projection. The temporal-recurrence
   nature of the LiquidCFC is not exercised because there is no
   sequence loop; H67's forward sees a single image, not a sequence.
   So the "Liquid CFC" prior collapses to an affine + nonlinearity.

Net effect: of the six priors H67 claims to compose, only the
**NaturePriorBlock encoder, Fibottention attention, and cymatic-init
QKV** genuinely fire. Golden RoPE → sinusoidal PE; Platonic graph →
identity; Liquid CFC → MLP. The "all-on stress test" is in fact a
half-on stress test.

5. Test `test_qkv_init_is_actually_cymatic` (line 82–89) asserts
   `model.attention.qkv.weight != fresh_xavier.weight`, but ANY
   non-Xavier init satisfies this (including the original
   Fibottention default). The test does not actually verify the
   weights came from `cymatic_init_`. A stronger test would
   construct two `Fibottention` modules — one with cymatic, one
   without — and compare against the cymatic-only instance.

**Recommendation (out of scope for this audit)**:
- Either land a `GoldenRoPE` class in `golden_rope.py` or replace
  the import with `apply_golden_rope` (function-style).
- Fix `MetatronGraphLayer(width, width)` two-arg call.
- Make `which_priors_active` actually inspect modules and forward
  hooks.
- Wrap LiquidCFC in an explicit unroll (T frames or T patch tokens)
  if the recurrence claim is to be empirically falsifiable.

### H68 — On-Device World Model — **PASS**

`hybrid_world_model.py` + `test_hybrid_world_model.py`.

- `GoldenSkipBlock` is instantiated three times (stage0/1/2). Test
  `test_world_model_phi_constant_present_in_skip_block` verifies
  `alpha == 1/φ` on each stage.
- Fibonacci channels: `fibonacci_channels(c0, 3, mode='fib')` is
  called (line 125). Test `test_world_model_fibonacci_widths_monotonic`
  verifies monotonic / divisible-by-8 widths.
- Drop-path: optional import works (`drop_path.py:DropPath` exists),
  so `_HAS_DROP_PATH=True`; even if not, the fallback class is a
  correct stochastic-depth implementation.
- GRU predictor + cosine-distance JEPA loss with stop-grad. Tests
  cover the sequential forward.
- Minor: H68's design doc mentions JEPA but the recurrent predictor
  is a plain GRUCell, not a transformer. The H68 doc explicitly says
  "single-layer GRU" so this is intentional.

### H69 — KAN-Metatron Symbolic Head — **PASS**

`hybrid_kan_metatron.py` + `test_hybrid_kan_metatron.py`.

- `MetatronGraphLayer(node_dim, node_dim)` constructed correctly
  (line 131) — both args provided.
- `MetatronGraphLayer.forward` is called in the head's forward
  (line 160), genuinely applying the 13-node message passing.
- `KANEdge` is a per-output learnable lookup spline; the
  `test_kan_edge_silu_at_init_inside_domain` test verifies init
  matches SiLU exactly at the knot positions.
- `d_out` independent splines (line 133): test
  `test_kan_metatron_head_param_count_includes_d_out_splines`
  verifies count.
- Per-output spline loop (lines 164–166) is O(d_out) Python — not
  vectorised, but functionally correct.

### H70 — Cymatic Low-Data Curriculum — **PASS**

`hybrid_cymatic_curriculum.py` + `test_hybrid_cymatic_curriculum.py`.

- `α(epoch) = max(0, 1 - epoch/warmup_epochs)`. Test
  `test_alpha_decays_monotonically_to_zero` covers the schedule.
- Forward composes `α·cymatic + (1-α)·CE` (line 187). Tests
  `test_forward_pure_ce_at_warmup_end` and
  `test_forward_pure_cymatic_at_start` verify both endpoints.
- Optional `cymatic_loss` external module is wrapped with a 2-D
  signature adapter (lines 156–167); fallback
  `default_cymatic_loss` does FFT-MSE between channel-mean spectrum
  and the mean Chladni target — both genuine cymatic structure.
- `chladni_modes_banded` registered as buffer (line 148).

### H71 — Icosahedral 3-D RoPE — **PASS**

`hybrid_icosa_rope.py` + `test_hybrid_icosa_rope.py`.

- `icosa_vertices_unit()` returns the canonical 12-vertex icosahedron
  unit vectors. Test asserts norm == 1.
- `_rodrigues_apply` implements
  `v·cos + (k×v)·sin + k(k·v)(1-cos)` correctly.
- `forward` reshapes `(B,H,N,D) → (B,H,N,D/3,3)`, applies
  per-triple Rodrigues rotation around axis `verts[t mod 12]` with
  angle `pos · φ^(-t)`. Test
  `test_icosa_rope_preserves_norm_per_triple` confirms norm
  preservation (pure rotation). Test
  `test_icosa_rope_position_zero_is_identity` confirms identity at
  pos=0.
- Test `test_icosa_rope_axes_cycle_through_12` verifies the 12-axis
  cycle for `head_dim > 36`. Good rigor.

### H72 — Fractal Vesica FFN — **PASS**

`hybrid_fractal_vesica.py` + `test_hybrid_fractal_vesica.py`.

- `VesicaPiscisConv2d` is genuinely used as the up-projection in
  three parallel paths (line 98–104).
- Radii `b, b/φ, b/φ²` (line 51); test
  `test_2d_ffn_three_paths_with_phi_shrunk_radii` verifies.
- φ-GELU activation `x · σ(φ·x)` (line 121) — the H39 φ-Swish.
- Hidden dim defaults to `round(c · φ)`; test verifies.
- 1-D variant uses masked Conv1d (multiplying conv weights by a
  radial mask before applying) — proper sparse-mask implementation,
  not a learned-zero workaround.

### H73 — Golden Spiral × Metatron PE — **PASS**

`hybrid_spiral_metatron_pe.py` + `test_hybrid_spiral_metatron_pe.py`.

- `fallback_golden_spiral_pe` implements the Vogel construction
  `r=√k, θ=k·GOLDEN_ANGLE` with φ-frequency layered features
  (lines 73–84). Test verifies the first row.
- `MetatronGraphLayer(d_metatron_node, d_metatron_node)` constructed
  correctly (line 134).
- Forward concatenates spiral + flattened Metatron output then
  projects to `d_model` (lines 174–180) — both signals genuinely
  feed the final projection.
- Test `test_spiral_metatron_pe_gradients_reach_seed_and_proj`
  verifies grad flow to both Metatron seed and final proj.

### H74 — Metatron Overlap Tying — **BROKEN**

`hybrid_metatron_tying.py` + `test_hybrid_metatron_tying.py`.

**The "13-circle overlap" composition is mathematically vacuous.**

The forward computes `eff_w = W · sum(α_c)` then runs a single
`F.conv2d(x, eff_w, ...)`. Lines 117–148. The 13 alphas only appear
through their **sum**: a single scalar `s = Σα_c`. The doc and
class-docstring's headline claim is "13 Conv2d outputs sharing one
underlying weight tensor", but in fact the implementation is a single
Conv2d whose kernel is `W` multiplied by a scalar gate. There are no
13 distinct circle masks, no per-circle offsets, no per-circle bias —
nothing that would make the 13 alphas non-degenerate.

The 13 alphas are reparameterisation-redundant: replacing all 13
alphas with a single scalar `s` would produce **bit-identical**
outputs and the same loss landscape (modulo a 13-vs-1 LR effective
rescaling).

The test `test_tied_conv_effective_weight_scales_with_alpha_sum`
literally proves the equivalence:
```python
eff_w = conv.effective_weight()
expected = conv.weight * conv.alphas.sum()
assert torch.allclose(eff_w, expected)
```
i.e., the test passes precisely because the 13 alphas collapse to
their sum. The author appears to have noticed this and asserted it
without recognising the design failure.

The "param_compression_ratio() ≈ 0.92" claim (line 121–134) is
vacuous — there is no untied bank to compress, because the model
never instantiated 13 distinct convs in the first place. A truly
13-tied scheme would either:
- shift the weight tensor by 13 distinct hex-overlap offsets
  (e.g., `Σ_c conv2d(x, W, padding=offset_c)`), or
- mask `W` by 13 distinct circle-overlap masks
  (e.g., `Σ_c conv2d(x, W · mask_c)`), or
- apply 13 distinct angular phase rotations (DFT-style).

None of these distinct-per-circle structures appear in the
implementation.

**Recommendation (out of audit scope)**: rework forward to
`y = Σ_c α_c · F.conv2d(x, W · mask_c[c], ...)` with 13 fixed
per-circle masks derived from `metatron_cube_adjacency()`. Then the
13 alphas become non-redundant and the compression claim is
meaningful.

### H75 — Harmonic Cymatic SwiGLU — **PASS**

`hybrid_cymatic_swiglu.py` + `test_hybrid_cymatic_swiglu.py`.

- `PhiGELU(beta=φ)` is the gate. Test
  `test_swiglu_uses_phigelu_gate` asserts both `isinstance(PhiGELU)`
  and `beta.item() == PHI`. Good.
- `_cymatic_init_linear_` uses a 5×5 Conv2d proxy with
  `cymatic_init_(orthonormalize=True, band=(2,5))`, then collapses
  via the central tap and L2-renormalises rows to He-equivalent
  norm. Test `test_swiglu_up_a_rows_have_he_equivalent_norm`
  verifies the row norms match `sqrt(2/in)·sqrt(in)`.
- SwiGLU forward: `down(gate(up_a x) * up_b x)`, classic GLU
  composition (lines 142–145).
- Test `test_swiglu_up_a_differs_from_default_kaiming` distinguishes
  the cymatic init from the default. Reasonable rigor; could be
  stronger by verifying mid-band energy concentration in `up_a`.

---

## Group concerns (cross-cutting)

1. **Optional-import fallbacks silently degrade hybrids**. Five G7
   hybrids (H63, H65, H67, H70, H73) wrap their primary primitive in
   `try/except ImportError`. In production:
   - H63: external `prh_loss.cka_loss` — internal fallback is
     bit-equivalent for centered inputs (OK).
   - H65: external `cymatic_loss` exists but H65 looks for *wrong
     function names* (`cymatic_consistency_loss`,
     `spectral_consistency`); it does find `cymatic_loss` as a last
     fallback (OK).
   - H67: BOTH optional priors (Golden RoPE, Platonic Graph) silently
     fall back due to API drift (no `GoldenRoPE` class;
     `MetatronGraphLayer` constructor mismatch). See H67 finding.
   - H70: external `cymatic_loss` is wrapped with an adapter; OK.
   - H73: external `spiral_pe.golden_spiral_pe` — fallback is correct
     Vogel construction.
   - Lesson: optional imports should fail LOUD in test mode (e.g., a
     `STRICT_HYBRID=1` env var that raises rather than falls back),
     so degradation doesn't masquerade as success on the CI dashboard.

2. **Hardcoded `which_priors_active`-style attestations**. H67's
   property is the worst offender, but the pattern of "expose a
   boolean flag dict that the test asserts" is fragile because the
   constructor stores the dict at construction time. A more robust
   pattern is `which_priors_active(self)` walking
   `self.modules()` + `self.named_buffers()` to detect each prior by
   class membership or buffer signature.

3. **Loss composition is not packaged**. H63 (CKA distillation),
   H65 (Betti collapse), H70 (cymatic curriculum) all expose loss
   helpers but only H70 actually combines with CE in-module. H63 and
   H65 require the trainer to remember to add them — an invitation
   to silent ablation. Recommend each of these expose a
   `composite_loss(logits, targets, *aux_inputs)` callable so the
   per-experiment hyperparameter is the auxiliary weight, not the
   "did the trainer remember to add the term".

4. **Shape-only tests dominate**. Twelve of fifteen G7 test files
   pass with shape and finite checks plus one or two stronger
   property tests. H67's `test_qkv_init_is_actually_cymatic` is the
   archetypal weak test — it asserts the weight is *not* a freshly
   Xavier-init weight, which is trivially true for any non-Xavier
   init. Strong tests should differentiate the *claimed* mechanism
   from a counterfactual (e.g., a SwiGLU with default init vs
   cymatic init must produce statistically different mid-band FFT
   energy).

5. **H64 schedule wiring is opaque to the runner**. Per Rule 14 the
   hybrid is a content-agnostic glue layer; per Rule 13 + Rule 19
   the runner must actually invoke `step()` at each epoch. Grep
   shows no runner integration. Either an experiment script under
   `ideas/<NN>/experiment.py` (out of audit scope) wires it, or the
   H64 ablation row reduces to a vanilla baseline. Verify before
   citing H64 empirics.

---

## Follow-ups

In priority order:

1. **H67 (BROKEN)**: fix the `GoldenRoPE` import and the
   `MetatronGraphLayer(width)` call. Replace hardcoded
   `which_priors_active` with a module-introspection version.
   Wrap LiquidCFC in a sequence loop (e.g., treat tokens as a
   sequence of length N and roll the CFC for T steps) so the
   recurrence is actually exercised.
2. **H74 (BROKEN)**: redesign forward so the 13 alphas multiply 13
   distinct circle-shifted or circle-masked Conv2d outputs. Without
   this, the H74 hypothesis is mathematically equivalent to a
   single-Conv2d-with-scalar-gate baseline.
3. **H64 (MAJOR)**: confirm via grep that the runner / experiment
   script actually invokes `GrowthPruningSchedule.step` per epoch.
   If not, either wire it up or mark H64 as "not empirically
   evaluated" in `IDEA_TABLE.md`.
4. **H63 / H65 (MINOR)**: package a `composite_loss(...)` helper that
   combines the auxiliary term with CE at the documented weight, so
   the trainer cannot silently omit the auxiliary.
5. **Cross-cut**: a unit-test pass should add at least one
   "differentiates-from-counterfactual" test per hybrid (e.g.,
   "this hybrid with prior X disabled produces statistically
   different outputs than the full hybrid").
6. **Cross-cut**: introduce a `STRICT_HYBRID=1` env var that
   converts all `try/except ImportError` fallbacks into hard
   `ImportError`s in CI runs, surfacing API drift like H67's.

---

*Last updated: 2026-05-27. Audit doctrine: NeurIPS-PC, heightened
skepticism for cross-paradigm composition. Verdict tiers: PASS /
MINOR / MAJOR / BROKEN.*
