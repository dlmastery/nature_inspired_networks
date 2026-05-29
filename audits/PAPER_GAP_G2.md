# Paper-Gap Audit — G2 (Layer / Channel / Neuron, H11–H20)

> Third audit dimension after `audits/G2_audit.md` (impl-critic: code-vs-doc)
> and the sci-critic addenda in each design doc (idea-vs-novelty). This file
> diagnoses the gap between each hypothesis's **cited arXiv result** and the
> **actual experimental number** we measured on CIFAR-10 (12-epoch baseline
> = 82.16 % top-1 on `baseline_sg_vanilla`, seed 0).
>
> Reviewer: hostile NeurIPS-PC stand-in.
> Verdict tiers: PAPER_AGREES / SCALE / DOMAIN / IMPL_BUG / WRONG_TEST /
> CITATION_DOESNT_SUPPORT / NO_ARXIV.
> Date: 2026-05-29.

---

## Summary

Counts per classification across the 10 G2 hypotheses (multiple tags map to
the same hypothesis are counted once per CIFAR row evaluated):

| Classification | Count | Hypotheses |
|---|---|---|
| PAPER_AGREES | 0 | — |
| SCALE | 0 | — |
| DOMAIN | 4 | H11, H14, H15, H16 (no CIFAR row — paper task is tabular / RNN / LM / ViT) |
| IMPL_BUG | 2 | **H18 fib_stride** (-9.6 pp), **H19 phi_relu** (-11.1 pp) |
| WRONG_TEST | 2 | H13 phi_sparse (sparse-from-scratch ≠ paper's prune-then-fine-tune), H17.gate golden_modulate (12 ep too short for cosine schedule) |
| CITATION_DOESNT_SUPPORT | 2 | H17 golden_skip (1/φ specifically), H19 phi_relu (Koch "0.6 of dynamic range" misquote) |
| NO_ARXIV | 4 | H12 phi-kernel half, H17 (1/φ-specific claim), H18 fib stride, H20 fib_ensemble (vs SWA) |
| PAPER_AGREES (qualified) | 1 | H20 fib_ensemble — SWA literature predicts ~+0.0..+0.5 pp; we got -2 pp, consistent with "no signal from short-horizon SWA" |

(Categories overlap; one hypothesis can carry both NO_ARXIV for the specific
φ claim *and* CITATION_DOESNT_SUPPORT for an over-broad attribution. The
total of column 2 above is therefore > 10.)

**Top-2 IMPL_BUG candidates (highest priority for the Fixer queue):**

1. **H19 phi_relu** — per-channel τ_init = 1/φ ≈ 0.618 cuts off ~62 % of
   the post-BN signal in early epochs. Empirical -11.1 pp vs baseline.
   The implementation matches the doc; the doc is what's wrong (factually
   misquotes Koch 1999 — see CITATION_DOESNT_SUPPORT below). Fix: either
   τ_init=0 (PReLU-style) or τ_init scaled to per-channel post-BN std,
   not a φ-derived absolute constant.
2. **H18 fib_stride** — schedule `(1, 2, 3)` with stride-3 + kernel-3 +
   padding-1 yields spatial cascade `[32, 32, 16, 6]`. The 16 → 6
   downsample loses ~62 % of the spatial information in a single layer
   (vs the standard 16 → 8 stride-2 which loses 75 % across two layers).
   Empirical -9.6 pp vs baseline. Fix: either reconcile to doc's
   padding=0 + smaller spatial collapse, or drop stride-3 at the last
   stage (use `(1, 2, 2)` Fibonacci-disguised-as-uniform).

**Commit:** see end of file.

---

## H11 — Pure Fibonacci MLP — **DOMAIN**

- **Cited arXiv:** Arik & Pfister 2021 TabNet (arXiv:1908.07442), Gorishniy
  2021 "Revisiting Deep Learning Models for Tabular Data" (arXiv:2106.11189),
  Baldi 2014 Higgs DNN (arXiv:1402.4735).
- **What the paper reports:** Higgs-UCI AUC ≈ 0.876–0.881 for MLP /
  ResNet-tabular at ~1 M params (Gorishniy 2021 Table 2). TabNet reports
  similar AUC at lower param counts.
- **What we observed:** No CIFAR row — H11 is a tabular task and the
  runner has no `build_tabular_model` dispatcher (the G2 impl-audit flags
  this as "TODO runner wiring").
- **Diagnosis:** DOMAIN — the hypothesis targets tabular MLP; forcing it
  onto CIFAR would be category-confused. The paper agrees *in principle*
  (Fibonacci diamond `[8,13,21,34,21,13,8]` is parameter-comparable to the
  Gorishniy MLP); we just haven't run the right task.
- **Action:** Either wire the tabular dispatcher (Phase-4 work) or
  formally retire the H11 sweep claim until then. No code patch needed.

---

## H12 — Fib-Channel CNN (+ phi-Kernel cascade) — **NO_ARXIV (kernel half) / PAPER_AGREES (channel half)**

- **Cited arXiv:** LeCun 1989, Simonyan & Zisserman 2015 (arXiv:1409.1556),
  He 2016 (arXiv:1512.03385), Tan & Le 2020 EfficientNet
  (arXiv:1905.11946).
- **What the paper reports:** Compound scaling sweeps width *uniformly*
  (constant ratio across stages). EfficientNet uses `1.2^φ` width
  multipliers across the family but uniform within a network. No paper
  specifically prescribes a *Fibonacci-ratio width cascade* within a single
  network as a deliberate inductive bias.
- **What we observed:** H12 has no dedicated CIFAR row; the closest proxy
  is `baseline_sg_vanilla` at 82.16 % (which uses
  `priors.fibonacci_channels` widths but no other priors). This is on par
  with ResNet-20 baselines at 12 epochs (≈ 80–82 %) — meaning the
  Fib-channel cascade is **not detectably worse** than uniform widths at
  matched parameter count.
- **Diagnosis:** PAPER_AGREES at the channel-cascade half (Fib widths are
  a *defensible* width schedule, not a known winner or loser). NO_ARXIV
  for the phi-kernel alternation cascade (`kernels=[3, 5, 3, 5]`), which
  the impl-audit flags as **unimplemented** — there is no specific paper
  endorsing this cascade either, so the "missing implementation" loophole
  is not a paper-vs-result gap; it's a paper-doesn't-exist gap.
- **Action:** Either build the phi-kernel cascade and run the dedicated
  sweep row, OR retire the kernel-cascade falsifier from the H12 claim.

---

## H13 — phi-Sparse Linear / Conv (density 1/φ ≈ 0.618) — **WRONG_TEST / DOMAIN**

- **Cited arXiv:** Frankle & Carbin 2018 Lottery Ticket Hypothesis
  (arXiv:1803.03635), Han 2015 Deep Compression (arXiv:1510.00149),
  Markram 2015 (DOI, neuroscience).
- **What the paper reports:** Lottery Ticket prunes a *trained* network
  to ≤ 20 % density and recovers full accuracy *after rewinding and
  retraining the surviving subnetwork*. Han 2015 Deep Compression
  achieves AlexNet ImageNet accuracy at ~10 % density via iterative
  prune-then-fine-tune. **Neither paper trains-from-scratch at fixed 1/φ
  density** — both rely on either (a) a trained dense initial model
  whose pruning mask is reused, or (b) iterative magnitude pruning with
  fine-tuning. The specific *density* 1/φ ≈ 0.618 is not from either
  paper; it's our project's φ-derived choice.
- **What we observed:** `sg_only_phi_sparse_seed0` — **top-1 = 0.7333**
  (composite 0.6953). Δ = -8.83 pp vs baseline (0.8216). FLOPs unchanged
  (zero-masked weights still computed by dense GEMM).
- **Diagnosis:** **WRONG_TEST + DOMAIN.** Two compounding gaps:
  - WRONG_TEST: 12-epoch CIFAR-10 from-scratch with a static random
    Bernoulli mask is the *opposite* of Frankle's "train dense → find
    winning ticket → rewind" protocol. Our test omits the train-dense
    step and the rewind step, so we are evaluating "constant-mask SGD
    with 38 % of weights gone", not "lottery ticket sub-network".
  - DOMAIN: the density 1/φ ≈ 0.618 is much higher than the 0.05–0.20
    range where Lottery Ticket actually shows interesting behaviour;
    0.618 lives in the dense-baseline-matching regime where pruning
    literature consistently reports "no signal".
- **Action:** Either (a) reframe H13 as "random fixed φ-density mask
  underperforms dense by 9 pp on CIFAR-10/12ep" (a confirmed negative;
  this is already what the sci-critic verdict captures), or (b) re-run
  H13 under a proper Lottery-Ticket protocol (train dense for 100 ep,
  prune to 1/φ density via magnitude, rewind, retrain) before claiming
  the φ-specific density has any meaning. Without (b) the result is
  uninterpretable as a literature comparison.

---

## H14 — Fibonacci Recurrent (FibGRU with phi-gate) — **DOMAIN / NO_ARXIV**

- **Cited arXiv:** Cho 2014 GRU (arXiv:1406.1078), Hochreiter & Schmidhuber
  1997 LSTM, Miller 1956 (psychology, no arXiv).
- **What the paper reports:** Cho 2014 introduces the GRU but says
  nothing about hidden-size schedules; Miller's `7 ± 2` is a
  cognitive-load number, not a depth prescription. There is **no specific
  arXiv** advocating Fibonacci hidden sizes `[8, 13, 21, 34]` or a
  multiplicative cap `z_phi = z * (1/φ)` on the update gate. The impl
  itself diverges from the design doc (multiplicative rescale vs the doc's
  bias-init `logit(1/φ)`) — see G2_audit MAJOR finding.
- **What we observed:** No CIFAR row — H14 is a sequence task and the
  runner has no `build_seq_model` dispatcher.
- **Diagnosis:** DOMAIN (CIFAR is not the right testbed) + NO_ARXIV (the
  specific phi-gate and Fib-hidden mechanisms are aesthetic, not
  paper-supported).
- **Action:** Defer until the sequence-task dispatcher is wired and
  retire the "phi-gate as bias-init" prescription in the design doc to
  match the implementation (or vice versa — see G2_audit).

---

## H15 — phi-Initialised Embedding (golden-spiral sunflower) — **DOMAIN / NO_ARXIV**

- **Cited arXiv:** Mu & Viswanath 2018 (arXiv:1702.01417), Mikolov 2013
  (arXiv:1310.4546), Vogel 1979 (botanical journal, no arXiv).
- **What the paper reports:** Mu & Viswanath 2018 finds word-embedding
  matrices have anisotropic structure that hurts downstream performance;
  removing top principal components helps. Mikolov 2013 word2vec uses
  random uniform initialisation, NOT a φ-derived sunflower lattice.
  Glorot/Xavier 2010 (arXiv:1001.3014) is the standard embedding init.
  **No paper advocates a 2-D golden-spiral lattice projected into d_model
  via Haar-orthonormal as an embedding initialisation.**
- **What we observed:** No CIFAR row — H15 is an LM task (WikiText
  perplexity) and the runner has no `build_llm_model` dispatcher.
- **Diagnosis:** DOMAIN + NO_ARXIV. The mechanism is a coherent piece of
  geometry, but no specific paper claims it improves perplexity.
- **Action:** Defer until LM dispatcher is wired; meanwhile the unit
  tests fully exercise the rank-2 structure (good) but no falsifier
  evidence exists.

---

## H16 — Fibonacci Head Diversity (sparse multi-head attention) — **DOMAIN / NO_ARXIV**

- **Cited arXiv:** Voita 2019 (arXiv:1905.09418), Michel 2019
  (arXiv:1905.10650), Dosovitskiy 2021 ViT (arXiv:2010.11929), Rao 2024
  Fibottention.
- **What the paper reports:** Voita and Michel both find that *uniform*
  head counts (8 or 12) contain many prunable heads (a substantial
  fraction). Dosovitskiy ViT-T uses 3 heads, ViT-S uses 6, ViT-B uses
  12 — all uniform. **No paper specifically advocates Fibonacci-count
  head allocation `[1, 1, 2, 3, 5, 8]`** at the architectural level.
  Rao 2024 "Fibottention" is the closest match (Fibonacci-spaced
  sparse-attention windows), but it's a sparsity/dilation pattern paper,
  not a head-count paper.
- **What we observed:** No CIFAR row — H16 is a ViT task and the runner
  has no `build_vit_model` dispatcher.
- **Diagnosis:** DOMAIN + NO_ARXIV. The Fibottention paper is the
  closest support but it doesn't claim the *count* schedule we encode;
  it claims a *dilation* schedule that we partly inherit. The impl-audit
  flags that the per-head-mask structure has no direct assertion in the
  test suite.
- **Action:** Defer until the ViT dispatcher is wired; meanwhile add the
  mask-structure test recommended by the impl-audit.

---

## H17 — Golden Ratio Skip Connections (skip_scale = 1/φ) — **CITATION_DOESNT_SUPPORT / NO_ARXIV**

### H17.pure — `golden_skip` (single learnable α init at 1/φ)

- **Cited arXiv:** He 2016 ResNet (arXiv:1512.03385), Hayou 2021
  Stable-ResNet (arXiv:2010.01950), Bachlechner 2020 ReZero
  (arXiv:2003.04887), and adjacently Zhang 2019 Fixup
  (arXiv:1901.09321), Nguyen 2019 ScaleNorm (arXiv:1910.05895).
- **What the papers actually report:** Fixup, ReZero, and ScaleNorm all
  derive their skip scales from **stability analysis at deep networks**
  (typically `1/√L` or `0`, learnable). ReZero specifically initialises
  the residual scalar to **0** (NOT 1/φ) so the network starts as the
  identity. Hayou 2021 derives a depth-dependent geometric mean. **No
  paper concludes that a depth-independent constant 1/φ ≈ 0.618 is a
  principled scale**; the choice of 1/φ is aesthetic.
- **What we observed:** `sg_only_golden_skip_seed0` — **top-1 = 0.8163**
  (composite 0.8014). Δ = -0.53 pp vs baseline. Within seed noise.
- **Diagnosis:** **CITATION_DOESNT_SUPPORT** for the *specific* 1/φ
  claim. The cited papers (Fixup/ReZero/ScaleNorm) endorse "learnable
  small skip scale" generically; our 1/φ init falls into the broad
  basin those papers map and lands in their predicted range (within
  seed noise of baseline). So the experimental *result* is consistent
  with the literature — the *claim* "1/φ is special" is not.
- **Action:** Reframe H17 as "learnable skip scalar with neutral init —
  the value 1/φ has no advantage over 0 (ReZero) or 1 (vanilla)". This
  is what the sci-critic addendum already says (verdict
  DERIVATIVE+TESTABLE). The result is honest; the framing must be too.

### H17.gate — `golden_modulate` (channel-wise gate, distinct mechanism)

- **What we observed:** `sg_only_golden_modulate_seed0` — **top-1 =
  0.7981** (composite 0.8042). Δ = -2.35 pp vs baseline.
- **Diagnosis:** WRONG_TEST. At 12 epochs the channel-gate hasn't
  saturated; cosine LR + 12 ep is too short to give a per-channel gate
  schedule its working set. A 100+ ep run is needed to adjudicate.
  Combined with CITATION_DOESNT_SUPPORT for the 1/φ-init specifically.
- **Action:** Either extend H17.gate to 30+ epochs before claiming a
  verdict, or label it "preliminary" in FINDINGS.

---

## H18 — Fibonacci Stage Transition (stride schedule `(1, 2, 3)`) — **IMPL_BUG / NO_ARXIV**

- **Cited arXiv:** LeCun 1998, Simonyan 2015 (arXiv:1409.1556), He 2016
  (arXiv:1512.03385), Sandler 2018 MobileNetV2 (arXiv:1801.04381).
- **What the papers report:** Every cited architecture uses **uniform
  stride-2 downsampling** at stage transitions (ResNet, VGG, MobileNet).
  No standard CIFAR-10 architecture uses a stride-3 conv anywhere in
  the trunk. Stride-3 in the *final* downsample stage is unusual
  because it produces a very small spatial map (~6×6 from 32×32 input)
  before global pooling.
- **What we observed:** `sg_only_fib_stride_seed0` — **top-1 = 0.7255**
  (composite 0.6872). Δ = -9.61 pp vs baseline. Param count and FLOPs
  match baseline within noise (259 k / 200 M) — so this is not a
  capacity issue.
- **Diagnosis:** **IMPL_BUG (top-2 priority).** The schedule
  `(1, 2, 3)` with `kernel=3, padding=1` produces the cascade
  `[32, 32, 16, 6]` (verified in the G2 impl-audit and pinned by
  `test_predicted_cascade_equals_actual`). The jump 16 → 6 is too
  aggressive: it forces the network to commit to a coarse 6×6 grid
  before the head, losing per-pixel information that the baseline
  retains at 8×8. The design doc *describes* a different shape
  `(B, 64, 6, 6)` (also small but with a stride-3 + padding-0 + kernel-3
  cascade that lands at 11×11 instead), and the impl currently doesn't
  match the doc's padding convention either.
- **Action:**
  1. Test the doc-prescribed padding-0 stride-3 variant
     (`(B, 64, 11, 11)` cascade) — should narrow the gap by ~3–4 pp
     based on receptive-field accounting.
  2. Or replace the schedule with `(1, 2, 2)` (still "Fibonacci-derived"
     since `2 = F_3` repeated) and recover near-baseline.
  3. Run a controlled ablation `(1, 2, 2)` vs `(1, 2, 3)` to attribute
     the -9.6 pp drop to the stride-3 step specifically. Until then
     the H18 negative is not interpretable as a paper-vs-result gap;
     it's a paper-doesn't-cover-this-architecture gap.

---

## H19 — phi-Threshold ReLU (per-channel τ init = 1/φ) — **CITATION_DOESNT_SUPPORT / IMPL_BUG**

- **Cited arXiv:** He 2015 PReLU (arXiv:1502.01852), Trottier 2017 PELU
  (arXiv:1605.09332), Lu 2019 dying-ReLU analysis (arXiv:1903.06733),
  and the design doc cites Koch 1999 *Biophysics of Computation* with
  the claim that biological interneuron firing thresholds sit at
  "≈ 0.6 of dynamic range".
- **What the papers actually report:** PReLU (He 2015) introduces a
  *negative-slope* learnable parameter α (NOT a positive-side threshold
  τ); α is initialised to 0.25, not 1/φ. PELU (Trottier 2017) is a
  parametric ELU. Lu 2019 quantifies dying-ReLU and concludes that
  *higher* thresholds *increase* the dying-ReLU rate. **None of these
  papers advocate a positive threshold τ ≈ 0.618 on a post-BN signal**.
  The Koch 1999 attribution is **factually wrong**: typical mammalian
  cortical neuron thresholds sit at roughly 10–15 mV above resting
  potential, ≈ 0.11–0.13 of the action-potential dynamic range (Koch
  1999 §1.2 and §1.4) — not 0.6. The sci-critic flagged this as a
  citation misquote.
- **What we observed:** `sg_only_phi_relu_seed0` — **top-1 = 0.7107**
  (composite 0.6733). Δ = -11.09 pp vs baseline. Worst G2 result.
- **Diagnosis:** **CITATION_DOESNT_SUPPORT + IMPL_BUG (top-2
  priority).** The implementation faithfully realises what the doc
  prescribes (τ_init = 1/φ ≈ 0.618 per-channel, broadcasted, learnable)
  — so the impl-audit verdict (PASS) is correct *for what the doc
  asks*. The result is catastrophically bad because **the doc asks for
  the wrong thing**: cutting the post-BN signal at +0.618 removes
  roughly 62 % of standard-normal mass, leaving the network to learn
  from the top 38 % of activations only. The learnable τ presumably
  reduces toward 0 during training, but 12 epochs is not enough for
  per-channel τ to descend that far while also fitting the
  classification head.
- **Action:**
  1. Replace τ_init = 1/φ with τ_init = 0 (recovering ReLU) and
     verify the learnable τ drifts up or down rather than starting at
     a punitive level. This is the minimal Fixer patch.
  2. Or rescale τ to per-channel post-BN std: τ_init = 0.618 × σ_c
     where σ_c is estimated at init. This honours the "0.6 of
     dynamic range" intent without imposing it on an unscaled signal.
  3. Correct the Koch 1999 attribution in the design doc to the actual
     ≈ 0.11 figure (or remove the biological framing entirely and
     keep the φ-as-numerology framing the sci-critic recommends).

---

## H20 — Fibonacci Ensemble (Fib-weighted state-dict averaging) — **NO_ARXIV / PAPER_AGREES (qualified)**

- **Cited arXiv:** Izmailov 2018 SWA (arXiv:1803.05407), Polyak &
  Juditsky 1992 (no arXiv), Caron 2021 DINO (arXiv:2104.14294).
- **What the papers report:** SWA uses **uniform weights** across the
  averaging window. Polyak averaging is also uniform. DINO uses an
  exponential moving average with a momentum coefficient. **No paper
  specifically advocates Fibonacci weights `[1, 1, 2, 3, 5, 8, 13, 21]`
  normalised to a probability distribution** as the averaging schedule.
  SWA's known gain on CIFAR-10 is ~+0.2–0.5 pp at long horizons (150+
  epochs); at 12 epochs SWA is not expected to help and may slightly
  hurt because the trajectory hasn't entered the SWA-favourable
  flat-minima regime.
- **What we observed:** `sg_only_fib_ensemble_seed0` — **top-1 =
  0.8011** (composite 0.8132). Δ = -2.05 pp vs baseline.
- **Diagnosis:** **NO_ARXIV** for the Fib-weighted *specific* schedule;
  **PAPER_AGREES (qualified)** for the broader weight-averaging family.
  The -2 pp deficit at 12 epochs is consistent with SWA's known
  behaviour: short-horizon weight averaging *hurts* before the
  trajectory enters the SWA basin. There is no paper-vs-result gap
  here — the SWA literature explicitly warns that short-horizon
  averaging is not the SWA contract.
- **Action:** Re-run H20 at ≥ 100 epochs with the averaging starting
  after epoch ≈ 75 % of training (the SWA recipe). At that horizon,
  Fib-weighted vs uniform SWA becomes a meaningful comparison. At 12
  ep we are measuring "weight averaging in the wrong regime", not
  "Fib vs uniform schedule".

---

## Group-level themes

1. **Half of G2 has no arXiv-specific support for the φ/Fib *specific*
   choice.** H12 phi-kernel cascade, H14 phi-gate value, H15 sunflower
   embedding init, H16 Fib head counts, H17 1/φ-specific skip init,
   H18 stride-3, H20 Fib-weighted SWA — all of these are *aesthetic*
   φ/Fib applications on top of *real* mechanisms (channel scaling,
   gating, skip scaling, downsampling, weight averaging). The G2
   sci-critic addenda already mark most of them DERIVATIVE / NUMEROLOGY.
   The paper-gap audit just confirms there's no specific paper to
   defend the φ-specific choice against.

2. **Two implementation bugs surfaced as IMPL_BUG (H18, H19).** Both
   produce empirical -9 to -11 pp drops vs baseline that are not
   attributable to scale, domain, or short-horizon training; they're
   attributable to (a) too-aggressive downsampling (H18) and (b) a
   punitive per-channel threshold init (H19). Both have clear Fixer
   patches.

3. **Two domain-confused tests (H13, H17.gate) report negatives that
   the literature explicitly predicts.** Frankle 2018 does not endorse
   random-mask sparse-from-scratch; SWA does not endorse short-horizon
   averaging. We knew the negatives before we ran. The honest path
   is to label these as "wrong protocol" in FINDINGS.

4. **Four DOMAIN hypotheses (H11, H14, H15, H16) are not runnable
   today** because the runner has no tabular / sequence / LM / ViT
   dispatcher. These four would need substantial harness work before
   a paper-vs-result gap can even be measured. The G2 impl-audit's
   "TODO runner wiring" finding is the precursor to this audit's
   inability to score them.

5. **The Koch 1999 misquote (H19) is the most serious citation
   integrity issue in G2.** Rule 4 (citation rigor) requires the
   *cited paper* to actually claim what the doc claims. The "0.6 of
   dynamic range" quote does not appear in Koch 1999 as advertised;
   typical neuron thresholds are roughly an order of magnitude
   lower. This violates Rule 4 directly and should be corrected in
   the H19 design doc before any external claim references the
   biological motivation.

---

## Recommended Fixer / re-run queue (priority order)

1. **(P0, IMPL_BUG)** H19 phi_relu: patch τ_init to 0 (or to
   per-channel σ scaling). Re-run sg_only_phi_relu_seed0 and update
   FINDINGS with pre-fix vs post-fix table per Rule 21.
2. **(P0, IMPL_BUG)** H18 fib_stride: reconcile padding convention to
   the doc OR replace `(1, 2, 3)` with a less-aggressive cascade.
   Re-run sg_only_fib_stride_seed0 and update FINDINGS.
3. **(P0, Rule 4)** Correct the Koch 1999 misquote in
   H19_phi_neuron_activation_threshold.md.
4. **(P1)** Rewrite H13, H17, H17.gate, H20 FINDINGS entries to
   acknowledge the protocol mismatch (rather than claim a clean
   negative). Sci-critic verdicts already capture this; FINDINGS
   should mirror.
5. **(P2)** Build runner dispatchers for tabular / sequence / LM /
   ViT so H11, H14, H15, H16 falsifiers become reachable.
6. **(P2)** Add the missing phi-kernel cascade for H12 OR retire the
   kernel-cascade falsifier from the H12 claim.

---

*Audit complete. 10 hypotheses scored. 0 PAPER_AGREES (strict),
1 PAPER_AGREES (qualified, H20), 0 SCALE, 4 DOMAIN, 2 IMPL_BUG,
2 WRONG_TEST, 2 CITATION_DOESNT_SUPPORT, 4 NO_ARXIV (overlapping).*
