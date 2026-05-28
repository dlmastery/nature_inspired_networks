# G6 audit — Topological / Bridging hypotheses (H51–H60, H57 deferred)

Auditor: NeurIPS-PC reviewer (Opus 4.7), skeptical pass.
Scope: 9 hypotheses (H57 audio_cross_modal DEFERRED — not audited).
Doctrine: per hypothesis (1) mechanism vs claim; (2) math correctness;
(3) test rigor (shape-only=MINOR, mechanism-verifying=PASS);
(4) citation alignment; (5) falsifier reachable; (6) hidden bugs.

## Summary

| H  | tier   | one-line headline |
|----|--------|-------------------|
| H51 | PASS  | smooth-sigmoid persistence surrogate; backward confirmed differentiable |
| H52 | PASS  | timm-style per-sample DropPath; scaling + monotone schedule verified |
| H53 | MINOR | bijective unfold; "latitude-band" claim breaks down on z={0,0,0,0} |
| H54 | MINOR | hooks register correctly; silent zero-loss path if forward precedes register |
| H55 | **BROKEN** | head_bias is identically 0 for every Platonic group — module is vanilla MHA |
| H56 | MINOR | formula correct; docstring swaps h/w roles vs code; 16-class table OK |
| H57 | DEFERRED | not audited per instructions |
| H58 | PASS  | reduce='mean' is true orbit mean; reduce='max' is amax; both shape-checked |
| H59 | MINOR | strict=False state_dict load + key-fallback can silently skip mismatches |
| H60 | PASS  | sample std (n-1); bootstrap actually resamples with replacement |

Tier counts (excluding H57): PASS 4, MINOR 4, MAJOR 0, **BROKEN 1**, DEFERRED 1.

---

## H51 — BettiLoss (`src/nature_inspired_networks/betti_loss.py`)

- **Mechanism vs claim.** Doc admits this is a *smooth surrogate*, not real PH.
  `differentiable_persistence` builds `pairs = (sorted_edges[:-1], sorted_edges[1:])`
  on the upper-triangle of `cdist`. Persistence per pair = `pairs[:,1]-pairs[:,0]`,
  i.e. the *gap* between consecutive sorted edges. β₀,β₁,β₂ counts come from
  `_soft_count_above(lifetimes, t)` — `sigmoid(32*(x-t)).sum()`. Fully smooth,
  no hard thresholding.
- **Differentiability of backward.** `torch.sort` returns sorted values whose
  gradient propagates through the index permutation; the subsequent sigmoid-sum
  is smooth. Test `test_betti_loss_backward_nonzero_grad` confirms
  `x.grad.abs().sum() > 0`. Verified: the surrogate IS differentiable.
- **Math.** β₀ = 1 + soft_count(lifetimes > t); β₁ uses φ·t; β₂ uses φ²·t — the
  φ-scaling matches the docstring claim. Note: "β₁ = number of cycles" is at
  best a heuristic — using the *gap-to-next-edge* as persistence is NOT the
  true Rips persistence (which requires matrix reduction). Doc honestly flags
  this as "Rips-skeleton heuristic" — acceptable.
- **Tests.** 8 tests including mechanism-verifying ones:
  `test_two_cluster_higher_loss_than_single_blob` (regression that two
  clusters → larger β₀=1 loss than a single blob). MECHANISM-VERIFYING.
- **Citations.** Gabrielsson 2020 AISTATS + Naitzat 2020 JMLR — both with
  full author/venue/arXiv format. PASS.
- **Hidden bugs.** None found. Degenerate N<2 returns `(1,2)` zeros tensor.
- **Verdict: PASS.**

---

## H52 — DropPath (`src/nature_inspired_networks/drop_path.py`)

- **Mechanism vs claim.** Per-sample timm convention: Bernoulli mask of shape
  `(B,1,1,...,1)`, divide by `keep_prob`. `x * (mask / keep_prob)` — correct
  expectation rescaling.
- **Schedule monotonicity.** `FractalDropPath.drop_probs = [p_max*k/(n-1)]` —
  monotone non-decreasing from 0 → p_max. `test_fractal_drop_path_schedule_monotone`
  explicitly checks this. N==1 corner case returns `[0.0]` (no zero-div).
- **anytime_forward.** Properly checks `hasattr(model,'set_max_depth')` and
  raises AttributeError otherwise. State restored after sweep. `was_training`
  flag honoured.
- **Tests.** 7 tests including `test_droppath_drops_samples_in_train_mode`
  which verifies survivors are rescaled to exactly 2.0 with p=0.5. PASS-grade
  mechanism check.
- **Citations.** Larsson 2017 ICLR + Huang 2016 ECCV (stochastic depth) —
  both with arXiv. PASS.
- **Hidden bugs.** Note: in `FractalDropPath.forward`, when shapes mismatch
  (`y.shape != x.shape`), the wrapper falls through (`x = drop(y)`), which
  skips the residual. Documented but worth noting for callers.
- **Verdict: PASS.**

---

## H53 — IcosaUnfold (`src/nature_inspired_networks/icosa_unfold.py`)

- **Mechanism vs claim.** Doc claims a 12→(4,3) "great-circle adjacency"
  preserving permutation derived from latitude bands of 3 vertices each.
- **Reality.** The 12 icosa vertex z-coordinates are
  `{±1, ±1, ±0.618, ±0.618, 0, 0, 0, 0}` — i.e. an *equatorial band of 4
  vertices* (z=0) plus 2-2-2-2 bands at z=±1, ±0.618. Slicing
  `argsort(z, desc)[band*3:(band+1)*3]` into 4 bands of 3 forces a non-
  geometric cut: band 0 = `{z=1, z=1, z=0.618}`, band 1 = `{z=0.618, z=0, z=0}`,
  band 2 = `{z=0, z=0, z=-0.618}`, band 3 = `{z=-0.618, z=-1, z=-1}`. The
  "great-circle traversal" claim does not match the actual latitude structure.
- **What is preserved.** Permutation IS bijective — `test_permutation_length_and_bijective`
  passes. `fold(unfold(x)) == x` exactly. So *as a permutation* it is valid
  glue between a 12-vector and a (4,3) planar grid; downstream Conv2d weight
  reuse works (param-count test passes).
- **Math.** No mathematical error in the permutation itself; the failure is
  in the *interpretation* the docstring sells.
- **Citations.** Cohen 2019 ICML + Yu 2019 NeurIPS GICOPix + HEALPix 2005.
  Properly formatted. PASS-grade citation set.
- **Tests.** Six tests, all shape/bijection. None verifies adjacency claim
  (and a faithful test would FAIL).
- **Verdict: MINOR.** The permutation is a deterministic bijection (which is
  all downstream code requires), but the design-doc claim of preserving
  great-circle vertex adjacency is unsupported — the planar grid does not
  honour icosa edge structure.

---

## H54 — PHActivationRegularizer (`src/nature_inspired_networks/ph_reg.py`)

- **Mechanism vs claim.** `register_forward_hook` on each named stage
  (defaults `("stage1","stage2","stage3")`); hook stores `output` in
  `_captured[idx]` (no `detach()`, so grad flows back through activations).
  After forward, `loss()` walks `_captured`, runs each tensor through
  `BettiLoss`, weights by `lambdas[i]`, sums. `clear_hooks()` properly
  iterates `h.remove()` on each handle.
- **Differentiability.** Test `test_loss_backward_updates_cnn_params` confirms
  `.backward()` populates `stage*.weight.grad` (non-zero for at least one
  stage). PASS-grade.
- **Hidden behaviour to flag.** `loss()` returns `torch.zeros((), requires_grad=False)`
  when `_captured` is empty. If a caller (a) constructs the regularizer but
  forgets to call `.register(...)`, or (b) computes `reg.loss()` *before*
  the first forward pass, they get a silent 0 — the auxiliary regularization
  vanishes with no warning. Test `test_no_capture_yields_zero_loss` documents
  but does not warn against this.
- **Stage-name skipping.** Missing attrs (`getattr(model, name, None)`) are
  silently skipped — the doc warns about this and `test_missing_stage_silently_skipped`
  covers it. Acceptable.
- **Lambda schedule.** Default `(0.05 + 0.15*k/(n-1))` — properly deepening
  per the H54 design doc.
- **Citations.** Naitzat 2020 JMLR + Gabrielsson 2020 AISTATS + Hofer 2019
  ICML, all arXiv-formatted. PASS.
- **Tests.** 7 tests including `test_clear_hooks_removes_all` which verifies
  post-clear forward pass does NOT repopulate the buffer. Mechanism-verifying.
- **Verdict: MINOR.** Silent-zero-loss path is a foot-gun: a misconfigured
  trainer would train without the regularizer and never know. Recommend
  a warning if `loss()` is called with `_n_registered > 0` but `_captured`
  is empty.

---

## H55 — PlatonicTransformer (`src/nature_inspired_networks/platonic_transformer.py`)

- **CRITICAL FINDING.** The "Platonic-specific" head bias is identically
  zero for every Platonic group.

  ```python
  coords = platonic_vertex_coords(group)            # (n_heads, 3), unit-norm
  gram = coords @ coords.t()                        # (n_heads, n_heads)
  bias = gram.mean(dim=-1)                          # (n_heads,)
  ```

  For vertex-transitive Platonic solids, the vertex coordinates sum to zero
  (the centroid is the origin). Therefore
  `gram[i,:].mean() = (1/n) * v_i · sum_j v_j = (1/n) * v_i · 0 = 0`
  for every vertex `i`. Empirically verified:

  | group  | head_bias |
  |--------|-----------|
  | tetra  | [0, 0, 0, 0] |
  | octa   | [0]*6 |
  | icosa  | [0]*12 |
  | dodeca | [≈0]*20 (numerical 0) |

  The buffer is registered (`self.register_buffer("head_bias", bias)`),
  added to logits (`logits + self.head_bias.view(1,n_heads,1,1)`), and has
  exactly zero numerical effect. **`PlatonicAttention` is bit-equivalent
  to a vanilla `nn.MultiheadAttention(n_heads=12 or 20)`** — only the
  forced head count is Platonic. The Q/K/V projections are standard
  unconstrained `nn.Linear` weights with no Platonic symmetry constraint.

- **Tests.** Zero of the 7 tests check that the head bias actually does
  anything non-trivial; they only check shape, head count, and divisibility.
  A mechanism test ("logits with vs without head_bias are different on the
  same input") would FAIL — they would be exactly equal. **Test rigor is
  shape-only and HIDES the broken inductive bias.**

- **Citations.** Islam 2025 "Platonic Transformers" — listed as
  *forthcoming* (no arXiv) which violates Citation Rigor (Rule 4) — no
  arXiv ID. Vaswani 2017 + Bronstein 2021 GDL are properly formatted.
  Citation block partial-PASS but the load-bearing reference is unverifiable.

- **Falsifier reachability.** The hypothesis claims Platonic head-grouping
  is the load-bearing prior; if it produces zero effect, the falsifier
  ("CIFAR-10 top-1 ≥ vanilla MHA baseline") is trivially false — the
  module IS the baseline. Any positive composite result on a CIFAR sweep
  would have to come from the n_heads choice or run-to-run variance, NOT
  from Platonic symmetry.

- **Verdict: BROKEN.** The intended inductive bias is mathematically void.
  Fix options: (1) use a different orbit-respecting bias (e.g. log-cosine
  of all pairwise dot products as a (n_heads,n_heads,1,1) attention-bias
  matrix instead of a scalar per head); (2) clarify scope as
  "MHA-with-Platonic-head-count" and rename accordingly.

---

## H56 — CymaticDataset (`src/nature_inspired_networks/cymatic_dataset.py`)

- **Formula.** `generate_cymatic_pattern` produces
  `sin(m*pi*x_grid * freq) * sin(n*pi*y_grid * freq)` on a cell-centred
  grid with `x in [0,1]`, `y in [0,1]` — matches `u_{m,n}(x,y) = sin(m*pi*x/h)*sin(n*pi*y/w)`
  ONLY if x,y are normalised; the docstring is sloppy about the `/h` `/w`
  factors (the code divides x by `w` and y by `h`, but the doc says the
  opposite). Internally consistent, externally confusingly documented.
- **16-class table.** `_class_to_mode(cls)` returns
  `(2 + cls//4, 2 + cls%4)` with `cls in [0,16)` — exactly the `(m,n) in [2..5]^2`
  table claimed. Verified `test_dataset_length_and_item_type` checks label
  in `[0,16)`.
- **Determinism.** `CymaticDataset.__getitem__` uses
  `torch.Generator().manual_seed(seed + idx)` so item `i` is reproducible.
  Test `test_cymatic_dataset_deterministic` confirms.
- **Citations.** Chladni 1787 (the founder) + Rahaman 2019 ICML
  spectral-bias. Chladni has no arXiv (it's an 18th-century treatise),
  which is acceptable historical reference; Rahaman is fully formatted.
- **Tests.** 7 tests including value-range, distinguishability of
  distinct modes, deterministic seed, and label-coverage (400 draws → ≥10
  distinct labels). Mechanism-verifying.
- **Verdict: MINOR.** Documentation swaps the `/h` `/w` roles relative
  to the implementation. The implementation is correct; readers of the
  docstring would be confused. No behavioural defect.

---

## H57 — Audio Cross-Modal

**DEFERRED — not audited per audit instructions.**

---

## H58 — GroupConv2d reduce='max'/'mean' (`src/nature_inspired_networks/priors.py`)

- **Mechanism.** `_orbit()` builds a `(G,O,I,k,k)` stack of `rot90` copies
  of the weight (4 copies for c4, 8 for d4). Forward reshapes to
  `(G*O,I,k,k)` and runs a single `F.conv2d`. Outputs are reshaped to
  `(B,G,O,H,W)` then reduced along the orbit dim:
  ```python
  if self.reduce == "mean":
      y = y.mean(dim=1)
  else:
      y = y.amax(dim=1)
  ```
  This is a genuine arithmetic mean (NOT a sum), divided by G=4 (c4) or
  G=8 (d4). `amax` is a per-position max over the 4/8 oriented copies.
- **H58 history.** Doc admits the "mean is better than max" intuition
  was **falsified** by the prior CIFAR-10 sweep — max actually wins by
  4–6 pp on CIFAR-10 because it acts as a soft argmax preserving the
  strongest oriented response. The code keeps both reductions accessible.
- **Citations.** N/A in this file (priors.py is the shared infra module;
  H58 doc lives in hypotheses/g6 and is the canonical citation site).
- **Tests.** `test_group_conv_reduce_max_and_mean_h58` checks (a) shape
  for both reductions, (b) outputs differ between max and mean (sanity),
  (c) finite. `test_group_conv_invalid_reduce_rejected` exercises the
  guard. Mechanism-verifying.
- **Verdict: PASS.** Both reductions are implemented correctly; the
  empirical falsification is properly documented in the docstring.

---

## H59 — Trained-Feature Betti (`src/nature_inspired_networks/trained_betti.py`)

- **Mechanism.** `compute_trained_betti` loads `state_dict` from a
  `.pt` checkpoint (with fallback over `("model_state","state_dict","model")`
  keys), moves model to device, calls
  `collect_features(model, dataloader, device=device, n_points=...)`,
  then `betti_curve(feats, rel_thresh=rel_thresh)`. Re-keys the parallel
  b0/b1 lists by stage name.
- **Risk: silent state_dict load.** Line 140: `model.load_state_dict(state, strict=False)`.
  If the checkpoint has totally mismatched keys (e.g. a different
  architecture's weights), `strict=False` silently drops the unmatched
  weights and the model remains at **fresh-init**, invalidating the
  "trained" claim. There is no return-value check or
  `_IncompatibleKeys` inspection. The test
  `test_checkpoint_load_handles_plain_state_dict` saves & reloads the
  *same* model so this hazard is not exercised.
- **n_points_for.** Trivial `return int(n_samples)`; raises on n<1.
  The forward reference in `compute_trained_betti` (line 143 calls
  `n_points_for` before it is defined at line 159) is safe in Python
  because lookup is at call time. Confirmed by `test_n_points_for_validation`.
- **Citations.** Naitzat 2020 + Hofer 2017 + Bauer 2021 Ripser. Full
  arXiv format. PASS.
- **Tests.** 6 tests covering 4-stage naming, b0/b1 ints, NaN-free,
  plain-state_dict loading, missing-file error, and n_points_for guard.
  Reasonable but doesn't exercise the strict=False hazard.
- **Verdict: MINOR.** Recommend `load_state_dict(state, strict=True)`
  (or capture the `_IncompatibleKeys` result and raise on missing/unexpected
  ≠ ∅) so a wrong-architecture checkpoint can't silently produce
  fresh-init "trained" Betti curves.

---

## H60 — Three-Seed Uncertainty (`src/nature_inspired_networks/multi_seed.py`)

- **Sample std.** `var = sum((x-mean)**2 for x in values) / (n-1)`,
  `std = sqrt(var)`. Sample std with Bessel's `(n-1)` denominator —
  correct.
- **Bootstrap.** `bootstrap_ci` uses `random.Random(seed)`, draws
  `n_boot` resamples with replacement (`rng.choice(vals)` n times per
  resample), computes per-resample mean, then sorts and picks
  α/2 and 1−α/2 quantiles. Empty input → `(NaN,NaN)`. Correct.
- **Key intersection.** `aggregate_seeds` intersects keys across all
  dicts and skips any key that is non-numeric in any seed — prevents
  silent partial averaging.
- **Citations.** Henderson 2018 AAAI + He 2016 CVPR + Loshchilov 2019
  ICLR + Bouthillier 2019 NeurIPS workshop. All arXiv-formatted (the
  workshop entry lacks an arXiv ID but cites the venue properly).
- **Tests.** 7 tests covering shape, empty input, partial keys,
  bootstrap bounds & ordering, mean inside CI, empty bootstrap, and
  markdown table formatting. Mechanism-verifying.
- **Verdict: PASS.**

---

## Most damning findings

1. **H55 Platonic head_bias is identically zero** — the entire "Platonic
   symmetry orbit" inductive bias is mathematically void because
   vertex-transitive solids have centroid 0. The module is provably
   equivalent to vanilla MHA with a forced head count. Every existing
   test is shape-only and conceals this. The Islam 2025 citation lacks
   an arXiv ID (Rule 4 violation).
2. **H53 "great-circle adjacency" claim is unsupported** — the 12 icosa
   vertices have z-distribution {±1, ±0.618, 0×4}, which does not
   decompose into 4 bands of 3. The 4×3 unfold is a bijection but
   the documented adjacency preservation is fiction.
3. **H59 silent state_dict load (`strict=False`)** — a wrong-architecture
   checkpoint will silently load nothing, and the "trained Betti curves"
   will be computed on fresh-init weights. No test exercises this hazard;
   in production this could invalidate every downstream topology claim
   keyed off `compute_trained_betti`.

## Commit info

See follow-up commit on `audits/G6_audit.md`.
