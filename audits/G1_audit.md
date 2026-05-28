# G1 audit — Scaling & Growth (H01–H10)
Reviewer: Critic-G1 (expert critic)
Date: 2026-05-27

## Summary
- **PASS:** H01, H03, H10 (qualified — see notes)
- **MINOR:** H02, H04, H05, H07
- **MAJOR:** H06, H08, H09
- **BROKEN:** (none)

The single most damning finding is **H09**: the project's only defensible
positive headline (CIFAR-10 85.54 / CIFAR-100 58.05 3-seed median) is
produced by `PhiBudgetNet` whose **realized per-stage param ratios drift
materially from the claimed 1 : phi : phi²** (measured ~1 : 1.41 : 2.45
at B=270k vs. claim 1 : 1.618 : 2.618), and whose **tests assert the
allocation-step ratio but never the realized-net ratio**. The headline
is not necessarily wrong, but the *named mechanism* (1:phi:phi²) is not
faithfully realized by the model that produced the result. Reviewers
will catch this; we should too. See H09 below.

## Per-hypothesis findings

### H01 — phi-Compound Scaling
**Module:** `src/nature_inspired_networks/scaling.py`
(`phi_compound`, `phi_compound_channels`, `fibonacci_sequence`)
**Verdict:** PASS

**Mechanism check:** `phi_compound(k, base_depth=18, base_width=64,
base_res=32)` correctly implements `d = round(base_depth * phi**k)`,
`w = round8(base_width * phi**(k/2))`, `r = round16(base_res *
phi**(k/4))` (scaling.py:48–53). At k=0 this reduces to the base
triple. The intra-family stage schedule (`phi_compound_channels`)
returns `[round8(w0 * phi**s) for s in range(n_stages)]`
(scaling.py:71) — i.e., WITHIN a family the stage widths grow by
phi^s, not phi^(s/2). This is a separate scaling rule from the
family-level phi^(k/2) and is **not documented in H01.md sec. 5.1**
(which only gives the across-family rule). Not wrong, but undocumented.

**Math correctness:** `PHI = (1.0 + 5.0**0.5) / 2.0` is canonical
(priors.py:16). The `_round8` and div-16 floors are correct. Negative
k still respects the 8/16 floors via `max(8, …)` and `max(16, …)`.

**Test rigor:** `tests/test_scaling.py:test_h01_phi_compound_grows_by_phi_powers`
asserts `d == round(18 * PHI**k)` for k=1..3 — a real mechanism
assertion, not shape-only. `test_h01_phi_compound_k0_is_identity`
catches an off-by-one boundary. `test_h01_phi_compound_channels_div8_and_monotone`
verifies monotonicity. Six H01 tests total, all mechanism-bearing.

**Citation alignment:** Tan & Le 2019 arXiv:1905.11946 IS the
compound-scaling paper being modified. Correct.

**Bugs / cargo-cult:** None observed.

**Concrete fix:** Document the intra-family stage-width recurrence
(phi^s) in H01.md sec. 5.1 alongside the family-level recurrence so a
reviewer reading just the doc can derive every width the code
produces. No code change needed.

---

### H02 — Fibonacci Depth Progression
**Module:** `src/nature_inspired_networks/scaling.py`
(`fibonacci_sequence`, `fibonacci_depths`, `resolve_blocks_schedule`)
**Verdict:** MINOR

**Mechanism check:** `fibonacci_sequence(n)` returns the canonical
`(1, 1)`-start sequence (scaling.py:84–92). `fibonacci_depths(n_stages=4,
start_index=3)` correctly produces `[3, 5, 8, 13]` (the H02 spec
schedule). `resolve_blocks_schedule` correctly dispatches uniform /
fib / linear.

**Math correctness:** Verified. `fibonacci_sequence(8)` ==
`[1, 1, 2, 3, 5, 8, 13, 21]` (test_scaling.py:99) and
`fibonacci_depths(5, 2)` == `[2, 3, 5, 8, 13]`.

**Test rigor:** Seven H02 tests cover the explicit Fibonacci values,
both start-index variants, all three modes of
`resolve_blocks_schedule`, the list-length-mismatch ValueError, the
end-to-end NaturePriorNet wiring smoke, and the default-mode
regression. Strong.

**Citation alignment:** Larsson et al. 2017 arXiv:1605.07648 is the
correct FractalNet reference (closest analogue). He et al. 2016 is
the ResNet anchor. Correct.

**Bugs / cargo-cult:** None at the primitive layer. **However**, the
H02 falsifier (section 3) requires a **convergence-speed metric**
(`epochs-to-72-pct-top-1`), and there is no place in the runner where
that metric is emitted or scored. The composite formula
in sec. 7.1 ("0.15 * (1 - epochs_to_72/50)") is unimplemented. This
means the **falsifier is not currently reachable** with the existing
gates: a Fibonacci-depth run can only be scored on the static top-1,
which H02 explicitly says is NOT the key novelty. **MINOR** because
the primitive itself is correct; the harness gap demotes it.

**Concrete fix:** Either (a) implement an `epochs_to_target` metric
hook in the runner and a `convergence_composite` formula behind a new
fingerprint, or (b) downgrade the H02 status journal to acknowledge
that only the iso-accuracy/iso-params dimension is currently
evaluable.

---

### H03 — Golden Spiral Resolution Scaling
**Module:** `src/nature_inspired_networks/multi_scale.py`
**Verdict:** PASS

**Mechanism check:** `GOLDEN_ANGLE_DEG = 360.0 / (PHI**2)` ≈ 137.508
(multi_scale.py:29) — correct. `golden_spiral_resolutions(28, 5)`
returns `[28, 45, 73, 118, 191]` matching the H03 canonical schedule
(test verified). `golden_spiral_lattice(n)` uses `theta = k *
golden_angle, r = sqrt(k/n)` — exact Vogel 1979 phyllotactic
construction (multi_scale.py:73-75).

**Math correctness:** `int(28 * PHI**1) = int(45.30) = 45`,
`int(28 * PHI**2) = int(73.30) = 73`, etc. Floor matches the doc
sequence. `GOLDEN_ANGLE_RAD = 2*pi*(1 - 1/phi)` ≈ 2.39996 — equivalent
to `2*pi/phi**2` (verified).

**Test rigor:** Nine H03 tests: angle constants, canonical schedule
exact equality, monotonicity with `align`, lattice points-on-disk,
n=1 edge case (origin), resize shape across all four interp modes,
3-D input rejection, end-to-end NaturePriorNet wiring, and default-
no-resize regression. Excellent — mechanism-verifying.

**Citation alignment:** Vogel 1979, Dosovitskiy et al. 2021. Correct
for the phyllotactic-lattice + ViT axes; Cohen-Welling 2016 is
mentioned only in the doc (the CNN-track here is *not* equivariant
beyond the resize).

**Bugs / cargo-cult:** The doc claims a *patch lattice* (golden-spiral
sampling pattern) component but the code only ships a **resize**
module — there is no per-token golden-spiral patching wired into a
ViT. The lattice helper exists but isn't used by any model. This is a
**scope reduction not a bug**: the H03 CNN-track variant ships as
documented; the ViT lattice piece is parked. No falsifier blocker.

**Concrete fix:** Add an explicit note to H03.md sec. 5.1 stating
"the lattice helper is reserved for a future ViT idea sub-project; the
current sweep row evaluates only the input resize". No code change.

---

### H04 — phi-Self-Similar Width
**Module:** `src/nature_inspired_networks/priors.py:fibonacci_channels`
**Verdict:** MINOR

**Mechanism check:** `fibonacci_channels(c0, n, mode='fib')` uses fib
sequence `[1, 2, 3, 5, 8, 13, ...]` (starting `[1, 2]`, base=fib[1]=2)
and computes `c = round(c0 * fib[k+1] / 2)` round-to-8 (priors.py:34–
41). For c0=32, n=4: produces `[32, 48, 80, 128]` ✓.
`mode='phi'` produces `[round8(c0 * phi**k)]` — for c0=32 n=4:
`[32, 48, 80, 136]`.

**Math correctness:** Both modes are mathematically faithful to their
respective recurrences. The H04 doc's "mod-8 collapse at c0=16, n=3"
is **reproducible exactly** here (`fib` and `phi` both yield
`[16, 24, 40]`). That is the **acknowledged methodological lesson**
in the doc, not a bug — but the doc says the planned re-run should
use **c0=32, n=4** to separate the variants.

**Test rigor:** `test_fibonacci_channels_phi_grows_geometrically`
asserts `1.2 < b/a < 2.2` (test_priors.py:39–42) — this is **far too
loose**; `1.2 < 2 < 2.2` would let a 2x doubling baseline pass as
"phi-grown". The bound should be tightened to `1.5 < b/a < 1.75`
to be mechanism-discriminating. `test_fibonacci_channels_fib_monotonic_and_div8`
only asserts monotonicity, not the Fibonacci ratio (e.g., `(F[k+1] /
F[k])` should approach phi from above/below alternately). **No test
asserts that `mode='fib'` actually produces a Fibonacci-derived
sequence** rather than any monotone div-8 sequence — a regression
where someone replaces `fib[k+1] / 2` with `2**k` would pass all
existing tests.

**Citation alignment:** EfficientNet (Tan & Le 2019), ResNet (He
2016), MobileNets (Howard 2017). All correct.

**Bugs / cargo-cult:**
- **`phi_compound` mode is a verbatim copy of `phi` mode** (priors.py:
  45–51 with the comment "identical to 'phi' at k=0 but kept as a
  distinct mode so the H01 sweep row carries a non-overlapping tag").
  This is a **documentation/labelling hack**, not a real H01 mechanism
  implementation in this function — H01's true mechanism lives in
  `scaling.phi_compound_channels`. A sweep row tagged `phi_compound`
  via `fibonacci_channels(mode='phi_compound')` is **bit-identical to
  the `phi` row**. If experiments rely on this tag to distinguish H01
  from H04, the experimental design is **observationally
  underdetermined**.

**Concrete fix:**
1. Tighten `test_fibonacci_channels_phi_grows_geometrically` to
   `1.55 < b/a < 1.75` (around `PHI ± 0.07`).
2. Add `test_fibonacci_channels_fib_matches_fibonacci_recurrence`
   that asserts `widths[k] / widths[k-1] → phi` as k → ∞.
3. Either delete the `phi_compound` mode (it adds nothing over
   `phi`) or make it actually call `scaling.phi_compound_channels`
   so H01-tagged sweep rows really invoke the H01 width recurrence.

---

### H05 — Fractal phi-Recursion
**Module:** `src/nature_inspired_networks/blocks.py:_FractalPath`
**Verdict:** MINOR

**Mechanism check:** `_FractalPath` with `phi_shrink=True` shrinks
the recursive branch B's intermediate width by `int(c_out / PHI)`
with floor 8 (blocks.py:130–134). Recursion is handled by passing
`phi_shrink=self.phi_shrink` to the recursive call, so each level
shrinks again. Verified empirically: depth-3, c_out=64 yields b1 width
39 (64/phi), inner b2.b1 width 24 (39/phi). The recursion correctly
compounds the 1/phi shrink rule.

**Math correctness:** `c_mid = max(8, int(c_out / PHI))` uses **`int`
(floor)**, not `round`. For c_out=64, `int(64/PHI) = int(39.55) = 39`,
which is a sub-8-multiple. The output `c_out` is reached via a 1x1
projection (`b_project`), so downstream contract is preserved — but
the 39-channel intermediate breaks the div-8 tensor-core alignment
that the rest of the codebase enforces. Minor performance concern
on RTX 4090 (sub-8 channel widths force generic kernels).

**Test rigor:** `tests/test_blocks.py:test_h05v2_fractal_phi_shrink_activates_when_set`
asserts `fp.b1.conv.weight.shape[0] == expected_mid` for **one
recursion level only** (level 1 from c_out=64). **No test verifies
that successive recursion levels each shrink by 1/phi** — the
mechanism is only spot-checked at the outermost level. A regression
where the recursive call dropped `phi_shrink` would not be caught.

The doc also claims the recursion shrinks **both depth and width**
by 1/phi per level. The code shrinks width by 1/phi but **depth by
the integer recurrence "decrement by 1"** — the doc's `d_k = round(d_0
/ phi**k)` would give depths `[8, 5, 3, 2, 1]` for d_0=8, but the
code's recursion just decrements `depth=1, 2, 3...`. For default
fractal_depth=2 the two interpretations coincide; at depth>=4 they
diverge.

**Citation alignment:** Larsson et al. 2017 (FractalNet) — correct.
Murray 1926 is a name-drop without an arXiv counterpart; that's fine
under Rule 4 (the citation is well-known classical).

**Bugs / cargo-cult:** No silent randomness. `b_project` (a 1x1 conv
+ BN) is correctly registered as a sub-module. No `@torch.no_grad()`
inside forward.

**Concrete fix:**
1. Add `test_h05_fractal_phi_shrink_compounds_across_levels` that
   exercises depth-3 fractal and asserts inner-level widths follow
   `[c0, c0/phi, c0/phi**2, …]` with the 8-floor.
2. Replace `int(c_out / PHI)` with `8 * max(1, round(c_out / PHI / 8))`
   for tensor-core alignment (or document the deliberate sub-8 choice).
3. Resolve the depth-recurrence ambiguity in H05.md sec. 5.1 — either
   commit to "each recursion level drops depth by 1" (current code)
   or implement the `round(d_0 / phi**k)` depth schedule.

---

### H06 — Golden Ratio Bottleneck
**Module:** `src/nature_inspired_networks/phi_scaling.py:GoldenBottleneck`
**Verdict:** MAJOR

**Mechanism check:** Non-inverted: `c_mid = round8(c_in / PHI)` (line
115) — matches the doc's "reformulated phi bottleneck contract 1/phi".
Inverted: **`c_mid = round8(c_in * PHI)`** (line 113) — uses expansion
ratio `phi ≈ 1.618`, **NOT `phi² ≈ 2.618` as the H06 design doc
explicitly states** in sec. 5.1 ("MobileNetV2-style with phi**2
expansion instead of 6"). For c_in=32: code gives c_mid=48, doc
predicts c_mid=80. The code-vs-doc divergence is real and
quantitatively large (factor of phi off).

**Math correctness:** Both round-to-8 helpers are equivalent and
correct. `c_in/PHI` for c_in=64 → 39.55 → round8=40. ✓ But the
inverted ratio mismatch is the headline issue.

**Test rigor:** `test_h06_inverted_branch_expands_then_contracts`
asserts `c_mid == 104` for c_in=64 inverted (test_phi_scaling.py:54).
`64 * PHI = 103.55 → round8 = 104` matches the **code**'s ratio of
phi. **The test bakes in the wrong ratio** — it codifies the
divergence rather than catching it. A reviewer reading the test will
conclude "code matches doc"; both reviewer and committee would be
wrong. This is the worst kind of test pathology: it locks the bug in.

The non-inverted test `test_h06_mid_channel_is_c_div_phi_rounded_to_8`
correctly asserts `c_mid == 40` for c_in=64 (40 = round8(64/PHI)) —
that path is OK.

**Citation alignment:** ResNet (He 2016), Inception (Szegedy 2015),
MobileNetV2 (Sandler 2018). Correct.

**Bugs / cargo-cult:** The `param_count` method on `GoldenBottleneck`
just delegates to `sum(p.numel() for p in self.parameters())` —
harmless. The H06 falsifier (>=5% param reduction) is reachable via
`test_h06_regression_param_count_under_resnet_at_iso_depth`, which
asserts `n_gold < n_base` (any reduction at all) — far weaker than
the >=30% claim in the test comment ("Require AT LEAST 30%
reduction") which is **commented but not asserted**. The actual
assertion is just `<`. Inconsistency between comment and assertion.

**Concrete fix:**
1. Either fix the code (`c_mid = _round8(int(round(c_in * PHI**2)))`
   for inverted) **OR** rewrite H06.md sec. 5.1 to commit to
   expansion-ratio phi (1.618) instead of phi**2 (2.618). The doc
   actually walks through phi**2 = 2.618 = 6/phi^0.5… so the code
   change is the more honest fix.
2. Replace `test_h06_inverted_branch_expands_then_contracts`'s
   hard-coded `104` with `expected = _round8(int(round(c_in * PHI**2)))`
   so the test re-derives the doc's ratio rather than baking in the
   bug.
3. Strengthen `test_h06_regression_param_count_under_resnet_at_iso_depth`
   to actually assert `(n_base - n_gold) / n_base >= 0.05` (the H06
   falsifier value), not just `n_gold < n_base`.

---

### H07 — phi-Modulated Multi-Scale FPN
**Module:** `src/nature_inspired_networks/phi_scaling.py:PhiSpacedFPN`
**Verdict:** MINOR

**Mechanism check:** `phi_pyramid_widths(c0, n)` correctly returns
`[round8(c0 * phi**k)]`. `PhiSpacedFPN` lateral 1x1s, top-down
upsample, reproject across phi-ratioed widths, smooth 3x3s — all
present and shape-correct (phi_scaling.py:194–245).

**Math correctness:** Verified. Default `c0=16, n=4` → `[16, 24, 40,
64]`. Top-down ordering uses `merged[n-1] = laterals[n-1]` (deepest)
then walks down (correct FPN convention assuming `feats[0]` is
highest-resolution, deepest is `feats[-1]`).

**Test rigor:** Five H07 tests: widths ratio check (1.2..2.2 — same
loose bound as H04, **MINOR test concern**), per-level spatial
preservation, uniform-baseline branch, input-length-mismatch
rejection, regression-vs-uniform widths. None of the tests exercise
the actual top-down **fusion content** — only shape and width.

The H07 doc's section 5.1 talks about "8 fused multi-scale features
at near-phi spacing" via interpolation between adjacent strides at
1/phi fractions. The shipped code does **not** generate intermediate
phi-spaced strides; it just lateral-1x1's the input strides and
top-downs them. The "phi-spacing" in the shipped version is **in the
channel-width axis only, not in the resolution axis** — yet the
hypothesis is named "multi-scale" and the doc emphasises strides.

**Citation alignment:** Lin et al. 2017 (FPN), MedMNIST (Yang 2021),
Greenwood 1990 (cochlear). Correct.

**Bugs / cargo-cult:**
- The input-list ordering is **undocumented** in code asserts. A user
  who passes feats in deepest-first order will get coordinate-flipped
  outputs (the assert only checks length). MINOR robustness gap.
- The doc says the top-down fusion sums 8 levels; code sums n levels
  (4 by default). The "8 interpolated levels" experimental condition
  in sec. 7.1 is **not implementable with the shipped code**.

**Concrete fix:**
1. Either rewrite H07.md sec. 5.1 / 7.1 to match what the code does
   (phi-spaced channel widths, n native strides) OR implement the
   intermediate-stride interpolation the doc promises.
2. Tighten the widths-ratio test bound to `1.55 < b/a < 1.75`.
3. Add an input-order assertion: `assert all(feats[k].shape[-1] >=
   feats[k+1].shape[-1] for k in range(n-1))` in `PhiSpacedFPN.forward`.

---

### H08 — Dynamic phi-Growth
**Module:** `src/nature_inspired_networks/dynamic_growth.py`
**Verdict:** MAJOR

**Mechanism check:** `fib_growth_schedule(n_events=4, start_index=3)`
returns `(3, 5, 8, 13)` — correct. `DynamicGrowthCallback.step(epoch)`
correctly fires on Fib epochs only (verified by
`test_callback_triggers_only_on_fib_epochs`).

`grow_model` appends `n_extra_blocks` `NaturePriorBlock` instances
to the deepest stage and **Kaiming-reinitialises** the new conv/linear
weights (`_kaiming_reinit_`, dynamic_growth.py:72–84).

**Math correctness:** Schedule correct. The function-preservation
claim is the issue (see below).

**Test rigor:** Six H08 tests cover schedule, idempotency,
shape-preservation of forward output, and monotonic param growth.
**None verifies value preservation under growth.** The H08 design
doc sec. 9 Q&A "How do we know the implementation is correct?"
commits to: "*function-preservation: forward output after growth
equals forward output before growth on the same input within 1e-5*".
**The shipped code does NOT satisfy this** — Kaiming-reinit produces
a brand-new conv whose initial activation IS NOT identity. Empirical
verification with a minimal-flags NaturePriorNet on a (2, 3, 32, 32)
input shows the max-abs-diff between pre-growth and post-growth
forwards is **0.088**, not 1e-5. The implementation grows the model
*from-scratch* at each Fib epoch rather than via Net2Net-style
function-preserving widening, but the H08 mechanism explicitly cites
Net2Net (Chen et al. 2016) as the function-preservation reference.

This means: **(a)** the implementation diverges from the cited
mechanism (Net2Net function-preserving init becomes Kaiming
from-scratch init), and **(b)** the test that the design doc
specifies as the implementation-correctness gate is missing. The
falsifier (cumulative compute reduction) is **not even reachable**
without identity-init growth because every reinit disrupts the
training loss curve — the network effectively restarts mid-training.

**Citation alignment:** Chen 2016 (Net2Net), Wei 2016 (Network
Morphism), Larsson 2017 (FractalNet). Correct **as citations**, but
the implementation doesn't actually use Net2Net's function-preserving
init — the cited mechanism is not realized.

**Bugs / cargo-cult:** `_kaiming_reinit_` walks `module.modules()`
and re-inits BN gamma/beta to their PyTorch defaults — fine, but
silently undoes the BN warm-up that other code paths may want.

**Concrete fix:**
1. **Implement function-preserving (Net2Net) init** in
   `_kaiming_reinit_` (or replace it): the new conv's weight should
   be initialised so that the output is approximately the identity
   plus small noise. For a `c → c` conv, init `weight ≈ small_eps *
   randn + delta(centre)` (centre 1.0, surround 0.0) and BN
   gamma=1.0, beta=0.0; the residual skip then carries the signal
   unchanged through the new block.
2. Add `test_grow_model_function_preserves_value` that asserts
   `(y_after - y_before).abs().max() < 1e-3` (or whatever tolerance
   the chosen identity-init achieves) on a fixed-seed input. This is
   the single most important missing test in G1.
3. If function-preserving init is not implemented, **update H08.md
   sec. 9 Q&A** to acknowledge that the current implementation does
   NOT preserve function on growth events and the cumulative-compute
   savings claim is contingent on a future fix.

---

### H09 — Golden Proportion Parameter Budget
**Module:** `src/nature_inspired_networks/phi_scaling.py`
(`phi_budget_allocations`, `phi_budget_widths`, `PhiBudgetNet`)
**Verdict:** MAJOR

**This is the headline-defending hypothesis. Per the audit brief,
audited especially carefully.**

**Mechanism check:**
- `phi_budget_allocations(B, n)` correctly computes shares =
  `(phi - 1) * phi**k / (phi**n - 1)` (phi_scaling.py:263–264). For
  B=1_000_000, n=4 the output is `[105_573, 170_820, 276_393,
  447_214]`, and pairwise ratios are `1 : 1.618 : 2.618 : 4.236`
  (verified — within 5e-6 of phi**k). The allocation step is correct.
- `phi_budget_widths(B, n, kernel=3, blocks_per_stage=2)` derives
  per-stage channel widths from `c = sqrt(params_k / (2 *
  blocks_per_stage * 9))` then round-to-8. For B=270_000, n=3,
  blocks=2 → widths `[40, 48, 64]`.
- `PhiBudgetNet` builds stages with the above widths.

**Math correctness:** **Here is the critical drift.** The closed-form
`c² ≈ params / (2 * blocks * k²)` models ONLY the intra-stage 3x3
convs of identical-width blocks; it ignores (a) the c_in → c_out
**transition conv** between stages whose cost scales as `c_prev *
c_cur` (asymmetric), (b) the **1x1 skip projection** when widths
change (c_prev * c_cur), (c) all BatchNorm parameters, (d) the stem
and FC head.

Realized per-stage params at B=270_000, n=3, mode='phi':
- stem: 1_160
- stage 0 (width 40): 57_920
- stage 1 (width 48): 81_888
- stage 2 (width 64): 141_952
- fc: 650
- **total: 283_570** (5% over the target 270_000)

Stage param ratios: `1 : 1.414 : 2.451`. **Claim: 1 : phi : phi² =
1 : 1.618 : 2.618.** Drift is roughly **−13% at stage 1, −6% at
stage 2** from the claimed mechanism. The discrepancy comes from the
quadratic-only approximation in `phi_budget_widths` (the early stages
are dominated by their c_prev * c_cur transition cost, and the
round-to-8 quantises away phi precision at small widths).

**Why this matters for the headline:** H09's hypothesis is the
*specific* 1:phi:phi² ratio. If the realized network actually
allocates 1 : 1.41 : 2.45, the experiment is no longer testing
"phi**k allocation"; it is testing "some monotone allocation that is
between uniform and phi-spaced". A reviewer with a calculator will
catch this.

A separate concern: the **uniform baseline produces widths `[48, 48,
48]` with 256_666 params** vs phi mode's 283_570 — so the comparison
is **not iso-parameter** (`+10%` for the phi variant). The H09
falsifier requires `iso-budget AND iso-FLOPs (+/- 5 pct)` — the
shipped configuration violates the +/-5% iso-budget condition by 2x.

**Test rigor:**
- `test_h09_allocations_sum_to_budget` — checks the **allocation
  step**, with B=1M (large enough that rounding noise is negligible).
  Allocation IS correct; this test confirms it.
- `test_h09_regression_phi_ratio_holds` — checks the **share** ratios
  at B=10M. Same caveat.
- `test_h09_canonical_widths_consume_budget_approximately` — only
  asserts widths are div-8, monotone, and length n. **Does not check
  realized param count at all.**
- `test_h09_phi_vs_uniform_budget_modes_differ` — checks widths
  differ, not realized params.

**Crucial missing test:** there is no test that builds `PhiBudgetNet`
and asserts `(realised_stage_params[k] / realised_stage_params[0]) ≈
phi**k`. The test suite proves the **mathematician's allocation** is
correct (which it is); it does **not** prove the **realized network's
allocation** matches the hypothesis (which it doesn't, by ~13%).

**Citation alignment:** EfficientNet (Tan 2019), ResNet (He 2016),
Lottery Ticket (Frankle & Carbin 2019). Correct.

**Bugs / cargo-cult:**
- `c_sq = max(64.0, params_k / denom_factor)` (line 293) — the `64.0`
  floor silently caps tiny stages at width 8 (sqrt(64)=8). For small
  B this masks the phi schedule entirely. Not caught by any test.
- `out[-1] += drift` on the allocation rounding repair (line 269) —
  silently dumps up-to-(n-1) units of rounding error into the last
  stage. For B=270k, n=3 this adds up to 2 params into stage 2; for
  B=1M, n=8 it could add 7. Minor.

**Concrete fix (HIGHEST PRIORITY in G1):**
1. Add `test_h09_phi_budget_net_realized_per_stage_param_ratio` that
   constructs `PhiBudgetNet(B_total=270_000, n_stages=3,
   budget_mode='phi')`, measures `sum(p.numel() for p in
   net.stages[k].parameters())` for k=0,1,2, and asserts
   `realized_ratios[k] ≈ phi**k` within (say) 10%.
2. Add the same assertion at B=1M, n=4 where rounding noise is
   smaller.
3. If the realized ratio doesn't match: improve `phi_budget_widths`
   to account for the c_prev × c_cur transition cost (solve the
   quadratic exactly for each stage given the prior stage's width)
   and re-verify before re-running the headline 3-seed CIFAR
   experiments. **This is the audit's most concerning finding for
   the project's defensibility.**
4. Also gate `PhiBudgetNet` construction with a `total_params_drift
   <= 0.05` invariant so any future width-derivation tweak that
   breaks the iso-budget contract fails loudly.

---

### H10 — phi-Decay LR Scheduler
**Module:** `src/nature_inspired_networks/schedulers.py:PhiDecayLR`
**Verdict:** PASS (with code-vs-doc note)

**Mechanism check:** `get_lr()` returns `base_lr * phi ** (-k /
T_max)` clamped at `lr_floor` (schedulers.py:63–67). Subclasses
`_LRScheduler` correctly so `last_epoch` advances on `.step()`.

**Math correctness:** Verified to 1e-9 by
`test_h10_phi_decay_step_k_equals_phi_neg_k_over_Tmax`. The floor
clamps correctly to `1e-3` after 100 steps with T_max=1
(`test_h10_phi_decay_floor_clamps_long_horizon`).

**Test rigor:** Eight H10 tests covering initial LR, the
`base_lr * phi^(-k/T_max)` formula, raw `phi^(-k)` recovery at
T_max=1, the lr_floor clamp, T_max=0 rejection, monotonic
non-increase, trainer dispatch, and unknown-scheduler typo
rejection. Mechanism-verifying — strong.

**Citation alignment:** Loshchilov & Hutter 2017 (SGDR — cosine),
Smith 2017 (cyclical LR), Robbins & Monro 1951. Correct.

**Bugs / cargo-cult:** None.

**Code-vs-doc nuance:** The H10 design doc sec. 5.1 walks through
the schedule with **T_max=1** (so LR_k = LR_0 * phi^(-k) per-epoch,
giving `[0.100, 0.062, 0.038, …]` over 12 epochs). The trainer
dispatch in `train.py:_build_scheduler` instantiates `PhiDecayLR(opt,
T_max=epochs)` — so by default a 12-epoch run gets a **much slower
decay** (`base * phi^(-k/12)`, falling by a factor of just phi over
the full run). This is not a bug, but it means the "phi-decay
LR sweep row" actually evaluates `phi^(-k/12)` not `phi^(-k)`. The
doc's prediction band (top-1 ± 0.2pp of cosine) was reasoned at the
T_max=1 schedule. Either rerun with T_max=1 to test the literal
hypothesis, or update H10.md sec. 5.1 to use T_max=epochs and note
the corresponding schedule shape.

**Concrete fix:** Add a sentence to H10.md sec. 5.1 stating "the
trainer default is `T_max=epochs`, which gives `base * phi^(-k/epochs)`
— the raw `phi^(-k)` schedule is recovered by `T_max=1` and is the
literal hypothesis under test". Either keep the default or add a
`T_max=1` row to the H10 config sweep.

---

## Group-level concerns

1. **Test culture: shape-only tests are common.** H03 and H10 have
   strong mechanism-verifying tests. H04, H05, H07 lean heavily on
   shape + monotonicity. The looseness bound `1.2 < b/a < 2.2` for
   "approximately phi" in `test_fibonacci_channels_phi_grows_geometrically`
   and `test_h07_phi_pyramid_widths_grow_geometrically_in_phi` lets
   a 2x doubling baseline (`b/a = 2.0`) pass as "phi-grown". This is
   too permissive for a hypothesis whose entire claim hinges on the
   specific ratio phi vs 2.

2. **Tests baking in wrong constants.** H06's
   `test_h06_inverted_branch_expands_then_contracts` hard-codes
   `c_mid == 104` (= `round8(c_in * PHI)`), which **codifies the
   code's wrong expansion ratio**. When the test asserts the wrong
   constant verbatim, the test no longer serves as a guard against
   the mechanism drift; it becomes part of the drift.

3. **Mechanism implementations that don't match the cited paper.**
   H08 cites Net2Net (function-preserving init) but implements
   Kaiming reinit (function-restarting init). H06 cites MobileNetV2
   t=6 vs t=phi**2 but implements t=phi. H04's `phi_compound` mode
   is a literal duplicate of `phi` mode. These are the kinds of
   findings that would justify a "Major Revision" decision at a
   serious venue.

4. **Headline-defence gap (H09).** The project's only positive
   headline is produced by `PhiBudgetNet`. The headline's named
   mechanism (1:phi:phi² stage param ratio) is approximate not exact
   in the realized network, and the test suite verifies the
   *allocator function* but not the *realized network*. The
   discrepancy is real (~13% at stage 1) and a calculator-equipped
   reviewer can demonstrate it from public test code. The +0.005
   composite delta margin that "wins" H09 needs to be re-evaluated
   against the actual realized allocation rather than the claimed
   one.

5. **Falsifier reachability.** H02's convergence-speed falsifier
   ("epochs to 72% top-1") has no metric hook in the runner. H08's
   cumulative-compute falsifier depends on function-preserving growth
   that isn't implemented. Two of the ten G1 hypotheses cannot be
   adjudicated by the existing harness.

## Recommended follow-ups (prioritized)

1. **(P0, headline-defending)** Add a realized-per-stage-param-ratio
   test for `PhiBudgetNet` (H09). Verify against the 1:phi:phi²
   claim. If the realized ratio is materially off, either improve
   `phi_budget_widths` to solve the quadratic exactly (including
   transition costs) or rewrite H09.md to acknowledge the
   approximation. **Do this before any external claim** about the
   85.54 / 58.05 headline.

2. **(P0, code-vs-doc)** Decide and fix the H06 inverted ratio:
   either `c_mid = round8(c_in * PHI**2)` in code, or "expansion =
   phi" in doc. Update the H06 test to derive the expected `c_mid`
   from the constant rather than hard-code `104`.

3. **(P0, missing test)** Implement and assert function-preserving
   growth in H08 (Net2Net-style identity init). The H08 design doc
   explicitly commits to this test as the correctness gate.

4. **(P1, test rigor)** Tighten the "approximately phi" ratio bounds
   from `1.2 < b/a < 2.2` to `1.55 < b/a < 1.75` across H04, H07,
   and any other helper that claims phi-growth. Add a fib-recurrence
   test for `fibonacci_channels(mode='fib')`.

5. **(P1, dead branch)** Either delete `priors.fibonacci_channels(
   mode='phi_compound')` (it's a verbatim copy of `mode='phi'`) or
   make it dispatch to `scaling.phi_compound_channels`. Today the
   `phi_compound` sweep tag is observationally indistinguishable
   from `phi`.

6. **(P2, doc fidelity)** Add a code-implementation note to each of
   H05.md (recursion-depth semantics), H07.md (no intermediate
   strides), H10.md (T_max default is epochs not 1), and H03.md
   (only the resize is shipped; the lattice helper is reserved).

7. **(P2, falsifier hook)** Add an `epochs_to_target` metric to the
   runner so H02's convergence-speed falsifier becomes reachable.

