# G2 Audit — Layer / Channel / Neuron (H11–H20)

> Reviewer: hostile NeurIPS-PC stand-in. Doctrine: shape-only tests = MINOR;
> doc/code divergence in the mechanism = MAJOR; not-actually-doing-the-thing
> = BROKEN. PASS requires a mechanism-verifying assertion in the tests.

---

## Summary

- **PASS (6):** H11 fib_mlp, H13 phi_sparse, H15 phi_embedding, H17 golden_skip,
  H19 phi_relu, H20 fib_ensemble
- **MINOR (3):** H12 fib_channel_cnn, H16 fib_attention, H18 fib_stride
- **MAJOR (1):** H14 fib_recurrent
- **BROKEN (0):** —

---

## H11 — Pure Fibonacci MLP — **PASS**

- **Module:** `src/nature_inspired_networks/fib_mlp.py` (`FibMLP`,
  `default_fib_hidden`).
- **Mechanism check:** Stacked Linear layers with hidden sizes
  `[8, 13, 21, 34, 21, 13, 8]` (a Fib "diamond"). Activation defaults to
  PhiGELU. Mechanism matches the doc *spirit* (Fibonacci-sized hidden
  layers) — the diamond shape is a sane CNN-runner-less choice consistent
  with sec. 5.1's ±25% param target.
- **Math correctness:** Linear chain is straightforward; `param_count`
  helper enables a closed-form check; ValueError on input_dim<1,
  output_dim<1, or hidden 0.
- **Test rigor (`tests/test_fib_mlp.py`):** 6 tests — forward shape,
  canonical Fib diamond `[8,13,21,34,21,13,8]` (mechanism assertion),
  closed-form param count, drop-in equivalence under uniform sizes,
  invalid-input rejection, gradient finite-grad backprop.
- **Citation alignment:** Arik–Pfister 2021, Gorishniy 2021, Baldi 2014
  — all present in code docstring with arXiv IDs, format compliant.
- **Falsifier reachable?** Only via a tabular runner (Higgs UCI) that is
  documented as "TODO runner wiring" in the source. Mechanism itself
  reachable; *falsifier evidence* will require the tabular dispatcher
  promised in the TODO.
- **Bugs / cargo-cult:** None observed. Activation default = PhiGELU
  conflates H11 with H39; the drop-in test pins activation=ReLU when
  comparing against vanilla, which keeps the regression honest.
- **Concrete fix:** Document in the design doc that the *primary* schedule
  is the diamond `[8,13,21,34,21,13,8]` rather than the ascending
  `[8,13,21,34,55]` shown in sec. 5.1; this drift-vs-doc is the only
  cosmetic concern.

---

## H12 — Fib-Channel CNN + Phi-Kernel — **MINOR**

- **Module:** `src/nature_inspired_networks/models.py:NaturePriorNet`
  (channel widths via `priors.fibonacci_channels`).
- **Mechanism check:** Channel widths follow Fibonacci ratios at the
  stage level — `priors.fibonacci_channels(mode='fib')` generates
  `c0 * F_{k+1}/F_1` rounded to multiples of 8. **The phi-kernel
  alternation cascade (3 → 5 → 8) promised in doc sec. 5.1 is NOT
  implemented** — the design doc itself flags `(filter counts only;
  phi-kernel-size variant missing)`. The Fib-channel half of the
  hypothesis is implemented; the kernel-cascade half is missing.
- **Math correctness:** Fibonacci sequence generation in
  `priors.py:fibonacci_channels` indexes from F_1=1, F_2=2 — a slight
  deviation from the canonical (1, 1, 2, 3, …) but stays monotonic
  and `% 8 == 0`. `_round8` enforces minimum 8.
- **Test rigor (`tests/test_priors.py`, `tests/test_blocks.py`):**
  - `test_fibonacci_channels_fib_monotonic_and_div8` asserts only
    monotonicity and `%8==0` — never asserts widths match a specific
    Fib subsequence (so a non-Fib geometric sequence could pass).
  - `test_NaturePriorNet_forward_each_channel_mode` is shape-only across
    `{fib, phi, linear}` and does not assert mode-dependent widths.
  - There is no test that actual stage widths equal `[16, 24, 40]` (or
    the analogue from `fibonacci_channels(16, 3, 'fib')`).
  - No phi-kernel test exists because the variant doesn't exist.
- **Citation alignment:** LeCun 1989, Simonyan & Zisserman 2015, He
  2016, Tan & Le 2020 — present in design doc; the implementation
  itself draws only on the Fibonacci-Net 2025 / EfficientNet compound
  references in the priors.py docstring (sufficient).
- **Falsifier reachable?** The phi-kernel falsifier (+0.3 pp lift over
  T1.1) is NOT reachable from any code path that exists — the cascade
  variant is unbuilt.
- **Bugs / cargo-cult:** None in the parts that exist; the omission is
  the issue.
- **Concrete fix:** (a) Add a test that pins `fibonacci_channels(16, 3,
  'fib') == [16, 24, 40]` (or whatever the rounded output actually is)
  so a future refactor of the Fib generator can't silently drift to
  geometric/phi growth. (b) Implement `FibChannelPhiKernel`
  with alternating `kernels=[3, 5, 3, 5]` and register a sweep row
  before claiming the H12 falsifier was tested. Until then the H12
  status in the doc should be `~ partial`, not `done`.

---

## H13 — phi-Sparse Linear / Conv — **PASS**

- **Module:** `src/nature_inspired_networks/sparse.py`
  (`PhiSparseLinear`, `PhiSparseConv2d`, `magnitude_prune_to_phi`,
  `PhiSparseNaturePriorNet`).
- **Mechanism check:** `DEFAULT_DENSITY = 1.0 / PHI ≈ 0.618` and
  `phi_sparse_mask` constructs the mask as `(uniform < density)` so the
  expected fraction of **kept** weights is `1/φ`. This matches the
  doc's "fraction p = 1/phi kept" wording (NOT inverted to 0.618
  sparsity). `magnitude_prune_mask` keeps top-`density` by `|W|`.
  Forward uses `W * mask`, so gradient on masked-out positions is
  identically 0.
- **Math correctness:** Pristine. Constraint `0 < density ≤ 1`,
  Bernoulli sampling with optional `Generator`, top-k threshold via
  `torch.topk`.
- **Test rigor (`tests/test_phi_sparse.py`):** 13 tests including:
  `DEFAULT_DENSITY ≈ 1/φ` (numeric pin), mask binary `{0, 1}`, observed
  density on a 256×256 draw within 0.01 of `1/φ` (statistical pin),
  forward equals `F.linear(x, W*M, b)` (mechanism pin), `density=1.0`
  recovers dense, invalid density rejected, magnitude pruning keeps
  the top fraction (sorted check `kept.min ≥ dropped.max`), conv-2D
  shape and stride-2, mask-then-`reset_mask_magnitude` preserves
  weights but changes mask, gradient on masked positions equals 0
  exactly (mechanism regression), end-to-end NaturePriorNet variant
  and `build_model` registration.
- **Citation alignment:** Frankle–Carbin 2019, Han 2015, Markram 2015 —
  all in the module docstring with arXiv IDs / DOIs (the Markram
  reference has no arXiv but is correctly attributed).
- **Falsifier reachable?** Yes — `natureprior_phi_sparse` is registered
  with the dispatcher so a CIFAR-10 sweep row can trigger it.
- **Bugs / cargo-cult:** None. `magnitude_prune_to_phi` constructs the
  sparse layer with `strategy='ones'`, then overwrites the mask in
  `with torch.no_grad()` — clean.
- **Concrete fix:** None required. Optional follow-up: add a test that
  the random mask's `effective_param_count / (out*in)` matches density
  more tightly than the current 0.40–0.85 band (which is loose for
  16×8 = 128 cells but allows a true `0.5` bug to pass).

---

## H14 — Fibonacci Recurrent — **MAJOR**

- **Module:** `src/nature_inspired_networks/fib_recurrent.py`
  (`FibGRU`).
- **Mechanism check:** Doc sec. 5.1' prescribes **"update gate bias
  initialised to `logit(1/φ) ≈ -0.481`"** (`b_z.data.fill_(logit(1/phi))`).
  The implementation does **not** modify the bias at all — it
  re-derives the gate value and applies a *multiplicative* rescale
  `z_phi = z * (1/φ)` so the update probability is clamped to
  `[0, 1/φ]`. This is a real mechanism divergence: a multiplicative
  cap on `z` is mathematically *not* equivalent to a bias-init shift
  (which biases the *pre-sigmoid* logit; the cap saturates from above
  at `1/φ`). The code's docstring honestly describes what it does,
  but it does not match the design doc's primary recipe.
- **Math correctness:** The `_step` re-derivation correctly reuses the
  cell's `weight_ih`, `weight_hh`, `bias_ih`, `bias_hh` — parameter
  parity with `nn.GRUCell` is preserved. The rescale `z * 1/φ` is
  algebraically clean. Hidden sizes `[8, 13, 21, 34]` are Fibonacci.
- **Test rigor (`tests/test_fib_recurrent.py`):** 6 tests — forward
  shape, default sizes are Fib (mechanism pin), phi_gate=True differs
  from phi_gate=False after weight transfer (mechanism *change* pin),
  gradient flow at depth, rejects wrong rank / wrong feature dim,
  h0=None equals h0=zeros. **The tests do not verify the actual
  *numerical* identity `z_phi == z / φ` and they do not catch the
  doc-prescribed bias-init mechanism being absent.**
- **Citation alignment:** Cho 2014, Hochreiter–Schmidhuber 1997,
  Miller 1956 — all present in module docstring with arXiv where
  applicable.
- **Falsifier reachable?** No — `FibGRU` is documented as "TODO runner
  wiring" (no sequence-data dispatcher in the runner), so the
  copy-task / PTB falsifier cannot be checked. Even if it were, the
  result would be measuring a different mechanism than the doc
  promises.
- **Bugs / cargo-cult:** The doc claims "phi-biased update gates" as
  the mechanism, but the code never touches `bias_ih[hidden_size:
  2*hidden_size]`. Doc-prescribed regime is not exercisable.
- **Concrete fix:** Either (a) implement the doc-prescribed
  `bias_ih[hidden:2*hidden].fill_(log((1/φ)/(1 - 1/φ)))` init in
  `FibGRU.__init__` and toggle it via an additional `phi_bias_init`
  flag, OR (b) amend the design doc sec. 5.1' to state explicitly
  that the multiplicative rescale is the canonical mechanism and
  retire the "bias to logit(1/φ)" prescription. Add a test that pins
  `bias_ih[h:2h]` to the expected logit value (or that pins the
  numerical equality `output_phi == output_off when z multiplied by
  1/phi`) so the doc and code can never silently diverge again.

---

## H15 — phi-Initialised Embedding — **PASS**

- **Module:** `src/nature_inspired_networks/phi_embedding.py`
  (`PhiEmbedding`, `golden_spiral_embedding_init_`).
- **Mechanism check:** `GOLDEN_ANGLE_RAD = 2π(1 - 1/φ) = 2π/φ²` (the
  two are algebraically identical via `1/φ² = 1 - 1/φ`, both ≈ 137.508°).
  2-D sunflower lattice `(√(k+1) cos(kθ), √(k+1) sin(kθ))` projected
  via Haar-orthonormal `Q ∈ R^{d×2}` (QR of random Gaussian). The
  resulting embedding is rank-2 in d_model and preserves the
  phyllotactic angular spacing.
- **Math correctness:** Float64 internally; default scale heuristic
  `1/√(mean_r²)` is documented and tested.
- **Test rigor (`tests/test_phi_embedding.py`):** 7 tests — weight
  shape preserved, seed-determinism, *rank-2 structure pinned via SVD*
  `s[0] > 10 * s[2]` (mechanism pin — this is the key assertion that
  rules out an isotropic Gaussian masquerading), unit-mean row norm,
  no two embeddings co-linear (`max off-diag cosine < 1`), drop-in
  shape parity with `nn.Embedding`, init actually changes weights,
  embedding_dim<2 rejection, golden-angle within (137.5, 137.51) deg
  (the headline numeric pin).
- **Citation alignment:** Mu–Viswanath 2018, Mikolov 2013, Vogel 1979
  — all properly attributed with arXiv where applicable.
- **Falsifier reachable?** LLM falsifier (WikiText-103 ppl drop)
  requires a 124M-decoder dispatcher not wired into the runner. The
  *mechanism* itself is fully testable and pinned.
- **Bugs / cargo-cult:** None. Float64→`emb_layer.weight.dtype` cast
  is correctly placed after the projection.
- **Concrete fix:** None. (Optional: an *additional* test asserting
  that consecutive vocab indices (rows `k` and `k+1`) project to
  vectors whose 2-D pre-image is rotated by exactly 137.5° — would
  give the strongest possible mechanism pin, but the rank-2 SVD test
  is sufficient.)

---

## H16 — Fibonacci Head Diversity — **MINOR**

- **Module:** `src/nature_inspired_networks/fib_attention.py`
  (`FibMultiheadAttention`).
- **Mechanism check:** Head allocation `[1, 1, 2, 3, 5, 8]` summing to
  20 (Fibonacci counts), per-group dilations `[1, 2, 3, 5, 8, 13]`
  (Fib indices). The per-head attention mask is
  `(j - i) % d == 0` — i.e. token `i` attends to tokens at indices
  `i mod d`. This is the standard sparse-attention "dilation"
  interpretation (Fibottention, Longformer-strided). The doc's
  description of "sliding window of stride 2" is sloppy wording — the
  actual implementation is index-subsampling, which is the right thing.
- **Math correctness:** Mask is `(seq_len, seq_len)` bool per head;
  `True == KEEP`. `i = i` is included because `diff == 0` is `% d == 0`
  for any `d ≥ 1`. SDPA receives `(1, H, N, N)` broadcast mask.
- **Test rigor (`tests/test_fib_attention.py`):** 7 tests — forward
  shape, head counts/dilations pinned to canonical lists, fib=False
  forward shape ok, fib=True vs fib=False produces different outputs
  (but *both head count and dilations differ*, so this is not a clean
  isolation of dilation effect), indivisible embed_dim rejected,
  mismatched schedule lengths rejected, gradient flow, `_attn_mask`
  returns None when all dilations are 1. **There is no test that
  asserts the per-head mask actually has the right structure — i.e.,
  that head 2 (group "dilation 2") attends to exactly the
  even-difference positions, that head 19 (group "dilation 13")
  attends to positions whose `(j-i) % 13 == 0`, and that the mask
  cell count per row equals `⌈N/d⌉`.**
- **Citation alignment:** Voita 2019, Michel 2019, Dosovitskiy 2021,
  Rao 2024 — all present in module docstring with arXiv IDs.
- **Falsifier reachable?** ViT scaffold ("build_vit_model") is
  documented as TODO in the file footer. CIFAR-100 ViT-T sweep cannot
  run today.
- **Bugs / cargo-cult:** The doc claims "Causal mask: per-head
  dilation mask is causal by construction" — this is **false** for the
  implementation, which is bidirectional (uses `abs(j - i) % d`-style
  modulo — actually `(j - i) % d`, but `%` on negative ints in Python
  is well-defined, so `j < i` still attends). For ViT this is correct;
  for the LLM track this is **not** causal and the LLM falsifier would
  need a causal-conjunction with the lower triangle.
- **Concrete fix:** Add a focused test: build a mask with
  `_per_head_dilation_mask(seq_len=12, [1, 1, 2], [1, 2, 3])` and
  assert head 0 is all-True, head 1 is True at even differences,
  head 2 is True at differences in {-9, -6, -3, 0, 3, 6, 9}, etc. Add
  a `causal: bool = False` flag to the constructor that AND-masks
  with the lower triangle so the LLM track is actually reachable.

---

## H17 — Golden Ratio Skip Connections — **PASS**

- **Module:** `src/nature_inspired_networks/phi_scaling.py`
  (`GoldenSkipBlock`, `GoldenSkipResNet`).
- **Mechanism check:** `alpha = nn.Parameter(torch.tensor(1/φ))` —
  a single scalar per block, learnable by default, with init
  exactly `1/φ ≈ 0.618`. Forward: `y + alpha * skip(x)` (branch
  weight 1.0, skip weight `alpha`). Matches doc sec. 5.1 `mode='inv_phi'`
  recipe (skip_scale=1/φ, branch_scale=1). The block correctly
  **excludes** the H17/H34 confound (no `phases` buffer, no cosine
  channel-modulation) — the regression test pins this.
- **Math correctness:** Stride-2 projection skip via 1×1 conv. Alpha
  numel=1. Gradient flows through alpha when trainable=True; alpha is
  a buffer (no grad) when trainable=False. Both branches covered.
- **Test rigor (`tests/test_phi_scaling.py`):** Within H17.pure
  section, 6 tests — alpha init `|alpha - 1/φ| < 1e-6` (numeric pin),
  forward shape, trainable=False registers alpha as buffer
  (no param), init=1.0 recovers vanilla skip, stride=2 path
  through projection, **regression test that `alpha.numel() == 1`
  and `phases` buffer is absent** (confound-exclusion mechanism pin),
  9 GoldenSkipBlocks in the network with each alpha at `1/φ`,
  grad flows / frozen-mode comparison.
- **Citation alignment:** He 2016, Hayou 2021, Bachlechner 2021 —
  all present in module docstring with arXiv IDs.
- **Falsifier reachable?** Yes — `golden_skip` is dispatched through
  `build_phi_model`; CIFAR-10 sweep row can call it directly.
- **Bugs / cargo-cult:** None observed.
- **Concrete fix:** None. The H17.pure module is a textbook example
  of how a hypothesis ought to look (single learnable scalar,
  numeric init pin, confound-exclusion regression test).

---

## H18 — Fibonacci Stage Transition — **MINOR**

- **Module:** `src/nature_inspired_networks/stride.py`
  (`FibStrideNaturePriorNet`, `fib_stride_schedule`).
- **Mechanism check:** Default schedule `(1, 2, 3)` (3-stage CIFAR
  adaptation). `fib_stride_schedule(n_stages, pair=(2,3))` alternates
  the Fib pair `(2, 3)` after the first stage. Forward applies the
  stride as the `stride` arg to the first conv of each stage; rest of
  the stage is stride-1.
- **Math correctness:** `predicted_spatial_cascade(32) == [32, 32, 16, 6]`
  matches what the test pins. The formula `floor((h + 2 - 3) / s) + 1`
  is correct for kernel=3, padding=1.
- **Test rigor (`tests/test_fib_stride.py`):** 11 tests — default
  schedule `(1, 2, 3)` pinned, 3-stage / 4-stage / 5-stage alternation
  pinned (mechanism), pair `(2, 5)` branch, first_stage_stride!=1
  branch, invalid-args rejection, forward shape on CIFAR, predicted
  cascade equals actual stage shapes (mechanism regression: catches
  off-by-one in padding), adaptive-pool collapses to 1×1, custom
  stride tuple `(1, 2, 5)`, registered with `build_model`, baseline
  uniform `(1, 2, 2)` schedule still works.
- **Citation alignment:** LeCun 1998, Simonyan 2015, He 2016, Sandler
  2018 — all present in design doc (the module docstring is lean and
  refers back to the hypothesis doc).
- **Falsifier reachable?** Yes — `natureprior_fib_stride` is wired
  through the dispatcher.
- **Bugs / cargo-cult:** Doc sec. 5.1 prescribes a 4-stage
  `{2, 3, 2, 3}` cascade and discusses `kernel_size = max(3, stride)`
  with `padding=0 if s==3 else 1` (so a stride-3 conv has kernel-3,
  padding-0). The implementation uses `kernel_size=3, padding=1,
  stride=3`, which produces a *different* spatial size than the
  doc's `(B, 64, 6, 6)` example. The implementation's choice is
  documented honestly in `_downsampled_size` and is internally
  consistent (cascade=[32, 32, 16, 6]). This is doc-vs-code drift in
  the padding convention, not a math bug; the test pins the actual
  behaviour.
- **Concrete fix:** Reconcile the design doc and the implementation:
  either (a) update doc sec. 5.1 to state `padding=1` for stride-3
  and revise the (B, 64, 6, 6) shape to (B, 64, 11, 11), or (b)
  add a constructor flag `stride3_padding: int = 1` so the
  `padding=0` variant from the doc can be exercised as an ablation
  cell.

---

## H19 — phi-Threshold ReLU — **PASS**

- **Module:** `src/nature_inspired_networks/phi_threshold.py`
  (`PhiReLU`, `PhiAdaptiveReLU`, `PhiReLUNaturePriorNet`).
- **Mechanism check:** `tau = nn.Parameter(torch.full((num_channels,),
  1/φ))` — **per-channel** learnable threshold (shape `(C,)`, NOT a
  scalar), correctly broadcasted across 2/3/4/5-D inputs. Forward:
  `F.relu(x - tau)`. Exactly the doc recipe.
- **Math correctness:** Broadcast helper picks the right axis layout
  per ndim. `PhiAdaptiveReLU` (the second variant) tracks per-channel
  EMA mean and applies `tau = φ * mu`; integer `num_batches_tracked`
  is updated correctly.
- **Test rigor (`tests/test_phi_threshold.py`):** 12 tests — tau
  init = 1/φ (numeric pin **and** shape pin `(8,)`), forward equals
  `relu(x - tau)` for 2-D / 3-D / 4-D layouts (mechanism pin **and**
  exact-value pin), invalid ndim rejected, gradient `d(sum)/dτ = -1`
  per surviving channel (mechanism regression: catches the
  "tau is scalar" or "tau detached" bug), per-channel independence
  pinned by setting `[0, 0.5, 1, 2]` thresholds against constant input
  0.7 and checking `[0.7, 0.2, 0, 0]`, PhiAdaptiveReLU eval-mode
  uses running_mean, training updates the EMA, dynamic rule matches
  doc's `where(x > φ·mean, x, 0)`, bad-args rejection, end-to-end
  variant with `build_model` registration.
- **Citation alignment:** He 2015, Trottier 2017, Lu 2019 — all in
  module docstring with arXiv IDs.
- **Falsifier reachable?** Yes — `natureprior_phi_relu` is wired
  through the dispatcher.
- **Bugs / cargo-cult:** None observed.
- **Concrete fix:** None. (Optional: add an end-to-end "dead-channel
  rate" probe test — the doc's secondary falsifier — but this is
  out-of-scope for a unit test.)

---

## H20 — Fibonacci Ensemble — **PASS**

- **Module:** `src/nature_inspired_networks/ensemble.py`
  (`FibEnsemble`, `FibEMA`, `fibonacci`, `fib_weights`).
- **Mechanism check:** `fibonacci(8) == [1, 1, 2, 3, 5, 8, 13, 21]`
  (matches doc); average is `sum(w_i * cp_i) / sum(w_i)` so weights
  *normalise* to a probability distribution and the result is the
  Fib-weighted **average of weights** (NOT logit-averaging at
  inference). The implementation correctly **averages weights**, as
  required by SWA / EMA conventions, not output predictions.
  `FibEMA` uses `w_new = F_K / sum(F_1..F_K) = 21/54 ≈ 0.389` for
  K=8 — matches doc.
- **Math correctness:** Float32 accumulator (avoids fp16 underflow
  during summation), result cast back to source dtype on output.
  Integer buffers (BatchNorm `num_batches_tracked`) are
  forwarded as the most-recent value (correct — averaging an int64
  tracker would yield garbage).
- **Test rigor (`tests/test_fib_ensemble.py`):** 14 tests — Fibonacci
  sequence pinned (1, 1, 2, 3, 5, 8, 13, 21) and sum=54 (numeric
  mechanism pin), negative-n rejected, normalised weights sum to 1
  and `w[-1] == 21/54`, unnormalised weights pinned to `[1,1,2,3,5,
  8]` for K=6, FIFO eviction on K+1 updates (mechanism), averaging K
  identical state-dicts equals the input (regression), 3-update
  manual weighted mean pinned (`(1*[1,2] + 1*[3,4] + 2*[5,6])/4 =
  [14,18]/4`), partial-buffer uses trailing weights (mechanism
  branch), empty-buffer rejection, bad-K rejection, integer-buffer
  most-recent semantics (mechanism edge case), end-to-end
  `load_into` real `nn.Linear`, FibEMA new-weight `21/54` numeric
  pin, FibEMA convergence on constant state (regression), FibEMA
  pre-update query rejection.
- **Citation alignment:** Izmailov 2018, Polyak–Juditsky 1992, Caron
  2021 — all in module docstring with arXiv IDs.
- **Falsifier reachable?** Yes — the wrapper is a stand-alone post-
  training utility; any sweep that saves checkpoints can feed it.
- **Bugs / cargo-cult:** None. The "average WEIGHTS not OUTPUTS"
  question raised in the suspicion list is correctly resolved by
  the design: this is a checkpoint-state-dict averaging tool whose
  output is a loaded model, so inference happens *once* on the
  averaged weights, not as a multi-model ensemble at inference.
  This is the SWA contract (Izmailov 2018), not an output-bagging
  ensemble — and the doc is explicit about this.
- **Concrete fix:** None.

---

## Group-level concerns

1. **Runner wiring debt.** H11 (tabular MLP), H14 (sequence GRU), H15
   (LM embedding), H16 (ViT) all advertise "TODO runner wiring" in
   their module footers. Until those dispatchers exist, the
   falsifiers for those four hypotheses are *unreachable from a
   single sweep command* — verification is limited to unit tests.
   This is documented honestly but bears watching. Rule 9 (every
   experiment archive has a very detailed README) is hard to honour
   when the experiment cannot be launched.

2. **Doc-vs-code drift in mechanism specifics.** Three hypotheses
   (H14 phi-gate as multiplicative rescale vs. doc's bias init; H18
   stride-3 padding=1 vs doc's padding=0; H11 default `[8,13,21,34,
   21,13,8]` diamond vs doc's `[8,13,21,34,55]` ascent) all show
   the same pattern: implementation chose a defensible variant of
   the doc's prescription without amending the doc. Rule 18
   (committee-grade docs) is undermined by this drift; a hostile
   reviewer would call out the H14 case as a *different hypothesis
   under the same name*.

3. **Shape-only/loose-bounds tests where mechanism would be cheap.**
   - H12: `test_fibonacci_channels_fib_monotonic_and_div8` does not
     pin the actual widths — a geometric or pyramidal sequence
     would pass.
   - H16: `test_fib_different_dilations_produce_different_outputs`
     conflates *count* differences with *dilation* differences; the
     test cannot tell which axis caused the diff.
   - The H16 per-head mask structure (the central mechanism — does
     head 4 actually attend at dilation 5?) has no direct assertion.

4. **Confound exclusion done well in H17, missing elsewhere.** H17
   has a dedicated regression test that pins the absence of the
   `phases` buffer (the H34 confound). H12 (channel widths vs.
   `phi` mode), H14 (rescale vs. bias-init), H16 (per-head mask
   vs. uniform attention with weight scrambling) lack analogous
   confound-exclusion tests.

5. **PHI numeric pin discipline is excellent.** Every module that
   imports `PHI` uses `PHI = (1 + 5**0.5) / 2` (no `1.618` literal
   drift seen). `DEFAULT_DENSITY`, `PHI_RECIPROCAL`,
   `GOLDEN_ANGLE_RAD` all derive from `PHI` rather than typing the
   constant. No PHI vs. 1.618 drift found in the audited set.

---

## Recommended follow-ups (in priority order)

1. **(MAJOR, H14)** Reconcile FibGRU's `phi_gate` mechanism with the
   design doc. Either implement the bias-init path or amend the doc
   to declare the multiplicative rescale canonical. Add a test that
   pins the chosen path numerically.

2. **(MINOR, H12)** Add `test_fibonacci_channels_pins_expected_widths`
   that asserts `fibonacci_channels(16, 3, 'fib')` returns the exact
   list of widths the rest of the project assumes. Implement (or
   formally retire) the `FibChannelPhiKernel` cascade variant.

3. **(MINOR, H16)** Add an explicit mask-structure test that calls
   `_per_head_dilation_mask` with hand-picked counts/dilations and
   asserts the bool pattern. Add a `causal` flag so the LLM track is
   reachable.

4. **(MINOR, H18)** Resolve the stride-3 padding discrepancy between
   doc sec. 5.1 (padding=0) and code (padding=1).

5. **(All G2 hypotheses)** Wire `build_tabular_model`,
   `build_seq_model`, `build_vit_model`, `build_llm_model`
   dispatchers — until then, H11 / H14 / H15 / H16 falsifiers cannot
   be reached from the sweep runner and remain "unit-test only".

6. **(Process)** A standing rule: every hypothesis whose doc claims
   `done` must have either a CIFAR sweep row or a dedicated test
   that exercises the same code path the falsifier would. Today H12
   `done` flag is partially aspirational (Fib channels yes,
   phi-kernel cascade no), which a hostile PC member would mark as
   over-claiming.

*Audit complete. 6 PASS, 3 MINOR, 1 MAJOR, 0 BROKEN.*
