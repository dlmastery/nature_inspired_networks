# Paper-Gap Audit — Group G5 (H41–H50, Optimisation / Init / Reg / NAS)

> Reviewer: hostile NeurIPS-PC stand-in. Doctrine for this audit:
> compare the empirical CIFAR-10 result for each G5 hypothesis to what
> the anchor arXiv literature actually claims.
> Verdict tiers per hypothesis:
> - **PAPER_AGREES** — literature predicts our observed direction / magnitude.
> - **PAPER_AGREES_WITH_FALSIFICATION** — literature predicts disaster; we
>   observed disaster; the falsification is consistent with the field.
> - **PAPER_DISAGREES** — literature gives a precise formula or sweep
>   range that our variant ignores; under-performance is predictable.
> - **PAPER_AGREES_WITH_OPPOSITE_DIRECTION** — paper says X happens
>   naturally; H forces X via auxiliary loss; the forced version is NOT
>   what the paper claims.
> - **WRONG_TEST** — the experiment as run does not test the doc's
>   pre-registered claim (epoch budget, knob count, or schedule
>   semantics drift). The numeric gap may be real or spurious; we
>   cannot tell from this run.
> - **IMPL_BUG** — the doc's pre-registered mechanism is plausibly
>   correct in the literature, but the implementation does something
>   subtly different. The audit-team Fixer should re-run with the
>   doc-faithful mechanism before the gap is final.

---

## Summary

- **PAPER_AGREES_WITH_FALSIFICATION (1, with caveat):** H41 PRE-FIX
  (β + eps both changed → catastrophic). POST-FIX, with stock
  `eps=1e-8` restored by Fixer-Opt and only `β1=1/φ, β2=1/φ²` retained,
  top-1 recovered to 0.8394 (composite 0.833) — this is now
  **PAPER_AGREES (β-only)**: the field's β2 ≥ 0.95 convergence
  doctrine (Reddi 2018) is more permissive at 12 ep than the worst-case
  proof suggests, and an EMA with τ ≈ 1.6 steps does not catastrophically
  fail at iso-LR. The big −33 pp PRE-FIX gap was the eps confound, NOT
  the β regime — so the "deep in non-convergence" prediction reproduces
  ONLY when eps is also moved. **This is the single most consequential
  finding of the G5 paper-gap audit.**
- **PAPER_DISAGREES (3):** H42 phi_init (init-variance deficit
  predictable from He 2015), H44 phi_decay (under-budgeted at 12 ep —
  see WRONG_TEST refinement below), H50 full_fib (no paper claims
  6-prior additive stacking).
- **PAPER_AGREES (2):** H43 fib_prune (12-ep magnitude prune does not
  hurt much; lottery-ticket shine needs full training + rewinding),
  H47 phi_dropout (small lift consistent with curriculum-dropout
  literature at 12 ep).
- **PAPER_AGREES_WITH_OPPOSITE_DIRECTION (1):** H49 prh_loss (Huh 2024
  is a NATURAL-convergence claim; H49 forces it via auxiliary loss —
  paper does not endorse the forced version).
- **WRONG_TEST (1, additionally):** H44 per-layer wd schedules (LAMB /
  AdamW) need long training to express; 12-ep CIFAR-10 cannot reject
  or confirm. H48 falsified at Phase-5 gate so the CIFAR-10 12-ep
  number is moot anyway.
- **No CIFAR row (3):** H45 sacred_nas, H46 cymatic_loss, H49 prh_loss
  — primitives implemented but not wired into a sweep row, so
  paper-vs-result is **NOT TESTABLE** at the empirical level.

### IMPL_BUG candidates surfaced from the paper-gap lens
1. **H41 mechanism-confound (PARTIALLY FIXED, REMAINING).** The
   Fixer-Opt restored stock `eps=1e-8` between PRE-FIX (top1 0.5196)
   and POST-FIX (top1 0.8394). Since `build_optimizer` previously did
   not accept an `eps` kwarg, the published −33 pp falsification was
   the eps+β confound, not the β-only claim. The POST-FIX number now
   tests the doc's pre-registered β-only hypothesis. **Status: FIXED.**
   Remaining cleanup: rename `GOLDEN_EPS` to `GOLDEN_EPS_EXPERIMENTAL`
   so the constant cannot silently leak back into the default code
   path; the **β-only-φ** row should be re-labelled as the canonical
   falsification of H41 (β1=0.618, β2=0.382, eps=1e-8 → top1 0.8394
   vs. baseline 0.8478 — i.e., **Δ ≈ −1 pp**, NOT −33 pp).
2. **H47 step-granularity bug** (per G5 audit). Counter advances
   per-forward-call, so the φ schedule cycles ~39× per epoch — this
   is NOT the "early-train-high / late-train-low" curriculum the doc
   pre-registers and that Bengio 2009 motivates. The 82.80 % observed
   is from an oscillating-per-batch dropout, not from a falling
   curriculum. **PAPER-WISE this is still WRONG_TEST**: the curriculum
   literature predicts a slow per-epoch decay; we ran a per-batch
   oscillation; the literature does not have a number we can compare
   to for "39 oscillations per epoch dropout".
3. **H48 saturation bug** (per G5 audit). The multiplicative
   `β ← β · 1/φ` rule with floor `1/φ²` hits the floor in **one
   step**, so for any training > 1 epoch the optimizer runs at fixed
   β1=0.382. This is a different hypothesis from the doc's "decay
   from 0.618 → 0.382 over training". The Phase-5 CIFAR-100
   demotion is therefore real but tests the WRONG schedule.
4. **H44 base-wd confound** (per G5 audit). The sweep row uses
   `phi_decay_base=1e-2` while the baseline uses `5e-4` — Rule 1 is
   bent, two knobs vary. A `sg_only_phi_decay_baseline_wd` row at
   `phi_decay_base=5e-4` is required to cleanly compare to LAMB / AdamW
   per-layer sweeps in the literature.

---

## Per-hypothesis blocks

### H41 — Golden Adam (`sg_only_golden_adam`)

- **Anchor literature:** Reddi et al. 2018 ICLR (arXiv:1904.09237)
  proves Adam non-convergence when β2 is too small; Choi et al. 2019
  (arXiv:1910.05446) and Wilson et al. 2017 (arXiv:1705.08292) sweep
  β2 ∈ [0.95, 0.999] across architectures. The literature **does** flag
  β2 = 0.382 as "deep in non-convergence regime" — under the standard
  Adam convergence proof. Jaeger 2020 (arXiv:2006.04751) on ESN
  edge-of-chaos is a category-mismatch citation (spectral radius of a
  state-transition Jacobian, NOT an EMA decay constant); the doc
  imported the φ-edge-of-chaos framing into the wrong dynamical object.
- **Pre-fix observed:** top1 0.5196, composite 0.514 (vs. baseline
  0.8478 / 0.846). −33 pp.
- **Post-fix observed (stock eps=1e-8, β=φ-defaults):** top1 0.8394,
  composite 0.833. Δ vs. baseline ≈ **−1 pp**, not −33 pp.
- **Audit (G5 audit § H41) MAJOR finding:** The PRE-FIX -33 pp was
  driven by `eps = 1/φ⁴ ≈ 0.146` (which dominated the denominator and
  pushed effective LR to ~6.85× nominal), NOT by β-shift alone. The
  Fixer restored stock eps, and the POST-FIX result now isolates the
  β-only hypothesis. β1 = 0.618 (τ ≈ 2.618 steps) and β2 = 0.382
  (τ ≈ 1.618 steps) are still far outside Reddi 2018's safe zone, but
  **at 12 ep on CIFAR-10 with cosine LR + ResNet-20 + AdamW
  decoupling**, the short-EMA regime is only mildly damaging, NOT
  catastrophic. This is a CLEAN empirical falsification of the doc's
  ≥ 10 % epochs-to-target speed-up claim (we observed slight
  regression, not speed-up), but it is NOT the catastrophic
  non-convergence Reddi 2018 predicts at infinite-horizon.
- **Verdict:**
  - **PRE-FIX:** PAPER_AGREES_WITH_FALSIFICATION (β+eps both
    catastrophic; the convergence-proof breakdown is real once eps is
    also broken). Confound: doc pre-registered β-only; the
    implementation moved two knobs.
  - **POST-FIX:** PAPER_AGREES with mild caveat. Reddi's non-convergence
    proof is asymptotic; at 12 ep cosine, β2=0.382 underperforms by
    ~1 pp without diverging. The doc's β-only ≥10% speed-up
    pre-registration is FALSIFIED at the seed-0 level (need 3 seeds for
    confident error bars). This is exactly what Wilson 2017 implies:
    Adam's default β2 ≈ 0.999 is empirically tuned, and pushing
    aggressively lower is monotonically worse on test acc but not
    pathologically so at short horizons.
- **IMPL_BUG candidate:** `GOLDEN_EPS` is still defined as a module
  constant in `optimizers.py`. Rename to `GOLDEN_EPS_EXPERIMENTAL` and
  audit `build_optimizer` defaults so the eps-confound cannot silently
  reappear.

---

### H42 — phi_init (`sg_only_phi_init`)

- **Anchor literature:** He et al. 2015 (arXiv:1502.01852) gives the
  exact formula `std = √(2/fan_in)` for ReLU networks; Glorot &
  Bengio 2010 gives `std = √(1/fan_in)` for symmetric activations. The
  `2` in He init is **not arbitrary** — it is `1 / E[ReLU(z)²]` for
  `z ~ N(0, 1)`, i.e., it is the exact variance-preservation constant
  for ReLU's positive-half retention. Replacing `2` with `φ ≈ 1.618` is
  a 10 % under-scale (`√(φ/2) ≈ 0.899`) per layer.
- **Predicted gap:** Per-layer 10 % activation under-scale compounds
  multiplicatively through depth. For ResNet-20 with ~18 stochastic
  layers, residual stream variance shrinks by `0.899^18 ≈ 0.15` at
  init — i.e., ~85 % activation attenuation, classic signal-vanishing
  pattern. He 2015 explicitly warns this is why the `2` is required.
- **Observed:** top1 0.7656, composite 0.777 (vs. baseline
  `sg_vanilla` 0.8216 / 0.826). Δ ≈ **−5.6 pp**.
- **Verdict:** **PAPER_DISAGREES.** The literature gives the precise
  formula and the precise reason; the H42 variant predictably
  under-performs. The -5.6 pp gap is in the right direction and
  magnitude for an init-variance-deficit cascade; deeper networks
  would show a larger gap (the formula's failure scales with depth).
  This is a clean falsification of the H42 doc claim that "φ is a
  more `natural' variance constant than `2'". Rigour cost: the G5
  audit's H42 PASS is consistent — the primitive does exactly what the
  doc says; the doc just predicts something the literature already
  flags as inferior.
- **IMPL_BUG candidate:** None. Primitive is correct; the hypothesis
  itself is the failure.

---

### H43 — fib_prune (`sg_only_fib_prune`)

- **Anchor literature:** Frankle & Carbin 2019 ICLR
  (arXiv:1803.03635) — Lottery Ticket Hypothesis: pruning shines
  WHEN rewinding + retraining from the original init AFTER a full
  training cycle. Han et al. 2015 magnitude pruning sweeps prune
  ratios (typically 0.2 / 0.4 / 0.6 / 0.8 across multiple
  retraining rounds, not a Fibonacci schedule). Tanaka et al. 2020
  SynFlow (arXiv:2006.05467) prunes at init.
- **Predicted gap:** At 12 ep (no rewinding, no full convergence
  cycle), magnitude pruning at moderate cumulative sparsity (~50 %)
  costs a few percentage points but does not crash. LTH-style
  spectacular gains require the rewind step (which the H43 schedule
  does not perform).
- **Observed:** top1 0.8115, composite 0.800 (vs. baseline 0.8478 /
  0.846). Δ ≈ **−3.6 pp**.
- **Verdict:** **PAPER_AGREES.** The result is exactly what magnitude
  pruning at ~50 % cumulative sparsity without rewinding gives on
  CIFAR-10 ResNet-20: small regression, no collapse. The Fibonacci
  schedule is one specific choice among many in the literature's
  "prune-on-a-schedule" family (Zhu & Gupta 2017); it is not
  privileged by any paper, but it is also not pathological. The H43
  falsifier (composite within 0.005 of baseline at 50-90 % sparsity)
  requires actual sparsity readout in metrics.json (G5 audit § H43
  flagged this).
- **IMPL_BUG candidate:** G5 audit § H43 fix — record
  `global_sparsity(model)` per epoch so the falsifier is queryable.
  Not a paper-gap bug.

---

### H44 — phi_decay (`sg_only_phi_decay`)

- **Anchor literature:** Loshchilov & Hutter 2019 (arXiv:1711.05101)
  AdamW decoupled-weight-decay; You et al. 2019 LAMB
  (arXiv:1904.00962) which uses **layer-wise adaptive** scaling. Both
  papers explicitly sweep per-layer scaling but the schedule is
  **adaptive** (based on `||w||_2 / ||g||_2`), not a fixed `φ^{-k}`
  geometric schedule.
- **Predicted gap:** Per-layer wd schedules typically need long
  training to express their effect — they shape the late-training
  generalization gap. 12 ep on CIFAR-10 is in the early-training
  regime where wd magnitude matters more than its layer distribution.
  A φ-geometric decay starting from `base_wd=1e-2` (vs. baseline
  5e-4) is a 20× absolute-magnitude shift — Rule 1 confound (per G5
  audit). At iso-magnitude (`base_wd=5e-4`), the literature would
  expect ≈0 effect at 12 ep.
- **Observed:** top1 0.7981, composite 0.812 (vs. baseline `sg_vanilla`
  0.8216 / 0.826). Δ ≈ **−2.3 pp**.
- **Verdict:** **WRONG_TEST + PAPER_DISAGREES** in combination.
  Wrong-test because (a) 12 ep is too short for per-layer wd
  schedules to express, and (b) the absolute base_wd was bumped 20×
  vs. baseline (two knobs vary, Rule 1 borderline). Paper-disagrees
  on direction because LAMB-style adaptive scaling outperforms any
  fixed geometric schedule by design. We cannot cleanly compare to
  the literature until the `sg_only_phi_decay_baseline_wd` row runs at
  `phi_decay_base=5e-4` AND a longer-horizon run gives the wd
  schedule time to act.
- **IMPL_BUG candidate:** Sweep-row knob confound (NOT a code bug —
  a sweep-config bug). Add `phi_decay_base=5e-4` variant per G5 audit
  § H44.

---

### H45 — sacred_nas

- **Anchor literature:** Liu, Simonyan, Yang 2019 ICLR DARTS
  (arXiv:1806.09055); Pham et al. 2018 ICML ENAS (arXiv:1802.03268).
  Both require bi-level supernet training with thousands of GPU-hours
  to derive a final architecture.
- **Observed:** **No CIFAR sweep row.** Primitive
  `SacredNASController` exists with α-softmax logits but is not
  end-to-end trained.
- **Verdict:** **NOT TESTABLE (no row).** The G5 audit MINOR finding
  stands: H45 is honestly a stub. The literature is clear that NAS
  needs the bi-level + discretisation pipeline; an α-mixture cell
  forwarded once does not search anything. No paper-gap can be
  computed until the falsifier (search-found arch beats vanilla
  NaturePrior at iso-flops) is reachable.
- **IMPL_BUG candidate:** N/A (scope: not implemented end-to-end).

---

### H46 — cymatic_loss

- **Anchor literature:** Rahaman et al. 2019 ICML
  (arXiv:1806.08734) on spectral bias of neural networks; Tancik et
  al. 2020 NeurIPS on Fourier features. These papers show NNs have a
  natural low-frequency bias — they do NOT prescribe an auxiliary
  loss to force a Chladni-mode target spectrum.
- **Observed:** **No CIFAR sweep row.** `CymaticLossModule` primitive
  is sound (G5 audit § H46 PASS on primitive), but no
  `aux_cymatic_loss` config flag exists in the runner.
- **Verdict:** **NOT TESTABLE (no row).** Paper-gap moot. Even if it
  were wired, the literature's spectral-bias work argues NNs already
  converge to a low-frequency representation — forcing a specific
  Chladni-mode target via auxiliary loss is a novel prescription
  with no direct citation support.
- **IMPL_BUG candidate:** N/A (not wired; G5 audit § H46 concrete fix).

---

### H47 — phi_dropout (`sg_only_phi_dropout`)

- **Anchor literature:** Srivastava et al. 2014 JMLR (canonical
  Dropout) gives fixed rates `p ∈ {0.2, 0.3, 0.5}`. Curriculum-noise
  prescriptions (Bengio 2009, Inoue 2019 cosine dropout schedule)
  schedule p on the **epoch** axis: early-high, late-low.
- **Predicted gap:** A correctly-scheduled epoch-axis φ-cycle from
  ~0.62 → ~0.09 over training should give a small lift over
  no-dropout at 12 ep (single-digit pp at most; Dropout's biggest
  wins are at longer horizons + larger nets). The implementation
  (per G5 audit § H47 MAJOR) advances counter **per-forward**, so
  the schedule cycles ~39× per epoch — this is closer to "stochastic
  perturbation noise" than "curriculum dropout".
- **Observed:** top1 0.8303, composite 0.825 (vs. baseline 0.8478 /
  0.846). Δ ≈ **−1.8 pp**.
- **Verdict:** **WRONG_TEST.** The mechanism that ran (per-forward
  oscillation) is NOT the doc's epoch-curriculum claim. The
  literature does not have a number to compare to for 39-oscillations-
  per-epoch dropout. After Fixer-Opt fix (epoch-axis schedule), a
  re-run is required. **Caveat:** the Fixer-Opt-fixed version's run is
  what's currently logged at 0.8303 (see G5 audit references to the
  fix landing); the G5 audit text says "post-Fixer fix per-epoch
  curriculum is now correct" — verify this in the current sweep tag.
- **IMPL_BUG candidate:** G5 audit § H47 concrete fix —
  `step_unit ∈ {'forward', 'epoch'}`, default `'epoch'`, Trainer hook
  per epoch.

---

### H48 — golden_momentum (`sg_only_golden_momentum`)

- **Anchor literature:** Sutskever et al. 2013 (Nesterov momentum),
  Smith 2017 cyclical-LR (arXiv:1506.01186), Loshchilov & Hutter 2017
  cosine annealing (arXiv:1608.03983). These prescribe SCHEDULES over
  many epochs, with smooth transitions.
- **Predicted gap:** A correctly-implemented multiplicative
  `β ← β · 1/φ` per-epoch decay from `1/φ=0.618` toward floor `1/φ²
  =0.382` would take ~1 step (per G5 audit § H48 MAJOR — the floor is
  hit on step 1). So the "schedule" is degenerate: training runs at
  fixed β1=0.382 from epoch 1 onward. AdamW with β1=0.382 at 12 ep on
  CIFAR-10 is a known mild regression (similar to H41 β-only).
- **Observed:** top1 0.8365, composite 0.834 (vs. baseline 0.8478 /
  0.846). Δ ≈ **−1.1 pp** on CIFAR-10 12 ep.
- **Verdict:** **WRONG_TEST + DEMOTED.** The CIFAR-10 12-ep number is
  in the right magnitude for "AdamW with β1=0.382 constant from
  epoch 1" — i.e., it is consistent with the H41 POST-FIX result
  (β-only ≈ −1 pp), NOT with the doc's "decay schedule" claim. The
  Phase-5 CIFAR-100 3-seed gate demoted this hypothesis (distribution
  overlap with baseline) — that demotion is the binding empirical
  fact. The doc must acknowledge that the schedule mechanism was
  saturated in 1 step and the demotion is on the constant-β1
  variant, NOT on a real decaying schedule.
- **IMPL_BUG candidate:** G5 audit § H48 concrete fix — replace
  multiplicative step rule with closed-form
  `β(e) = floor + span · exp(-e/τ)` or per-step factor
  `phi^(-1/T_max)`.

---

### H49 — prh_loss

- **Anchor literature:** Huh et al. 2024 ICML
  (arXiv:2405.07987) — The Platonic Representation Hypothesis claims
  that **as models scale**, representations from different
  architectures and modalities **naturally converge** to a shared
  Platonic ideal. The paper is an OBSERVATIONAL convergence claim,
  not a prescriptive auxiliary-loss recipe. Kornblith et al. 2019
  (arXiv:1905.00414) gives the CKA tool used to measure this.
- **Predicted gap:** Forcing alignment via auxiliary CKA loss against
  a fixed target embedding (CLIP / dodeca / icosa cached features)
  **at CIFAR-10 ResNet-20 scale** is an out-of-scale application of
  Huh 2024's claim. The paper's convergence kicks in at LARGE scale
  with diverse-modality training; forcing it at small scale risks
  over-constraining the feature space.
- **Observed:** **No CIFAR sweep row.** `cka_loss` and
  `PRHAlignmentLoss` primitives are mathematically sound (G5 audit §
  H49 PASS), but no `aux_prh_loss` config flag in the runner.
- **Verdict:** **PAPER_AGREES_WITH_OPPOSITE_DIRECTION** (not testable
  empirically yet). Huh 2024 says convergence happens naturally; H49
  prescribes forcing it. The paper does not endorse the forced
  version. Even if the row ran, the literature's prediction would be
  "no help, plausibly harm" at this scale.
- **IMPL_BUG candidate:** N/A on primitive; G5 audit § H49 concrete
  fix to wire the sweep row.

---

### H50 — full_fib (`sg_full_fib`)

- **Anchor literature:** None. No arXiv paper claims that hex (G2),
  group equivariance (G3), fractal multi-scale (G4), toroidal closure
  (G5), cymatic init (G6), and golden-modulate (G7) priors stack
  additively or multiplicatively to give 20-50 % efficiency gains.
  Each prior comes from a distinct mathematical structure (lattice
  geometry, finite-group representation theory, recursive
  self-similarity, periodic-boundary topology, Laplacian eigenvalue
  patterns, irrational rotation theory) — they do not share an axis
  on which compounding could be defined.
- **Predicted gap:** Stacking N priors on the same conv-block forward
  path stacks N approximation costs (signal loss from group-avg
  pooling, init-variance drift from cymatic, gradient-path doubling
  from fractal, etc.). The literature on multi-prior architectures
  (CapsNets, Group-Equivariant CNNs, scattering networks) consistently
  shows that adding more priors past 2-3 hurts on small-to-medium
  benchmarks because each prior eats representational capacity. The
  predicted catastrophic regression is **expected**, not anomalous.
- **Observed:** top1 0.7324, composite 0.697 (vs. baseline 0.8478 /
  0.846). Δ ≈ **−11.5 pp.**
- **Verdict:** **PAPER_DISAGREES.** The H50 doc's "20-50 % efficiency
  gain compound" prediction is the project's headline counter-example
  to additive prior stacking. The literature has no anchor for this
  prediction; it is a numerological meta-claim (per G5 audit and the
  H50 sci-critic addendum). The −11.5 pp result is consistent with
  the field's understanding that priors trade representational
  capacity for inductive bias and stack sub-additively at best,
  often anti-additively.
- **IMPL_BUG candidate:** G5 audit § H50 concrete fix — change
  `NaturePriorFlags` defaults to all-False (currently all-True) so
  "no priors" is the explicit identity. Also: per Rule 23, any
  multi-prior stack must use orthogonal axes (different layers of the
  training stack); the H50 stack puts 6 priors on the same conv-block
  forward path, which is exactly the forbidden configuration.

---

## Cross-cutting findings

1. **The H41 PRE-FIX → POST-FIX delta is the single most consequential
   number in G5.** Δ +32 pp (0.5196 → 0.8394) from restoring stock
   `eps=1e-8` alone, while β-defaults remained at the φ values. This
   demonstrates two things:
   - The published −33 pp catastrophic falsification was driven by
     the eps confound (which made effective LR ~6.85× nominal), NOT
     by Reddi 2018's β2-non-convergence regime.
   - With β-only changed, AdamW at 12 ep on CIFAR-10 ResNet-20 is
     ~1 pp worse — a mild empirical refutation of the "≥10 % faster
     to iso-acc" claim, NOT a catastrophic dynamical-system failure.
   - FINDINGS.md and any external claim referring to H41 MUST cite
     POST-FIX number 0.8394 as the canonical β-only result; the
     0.5196 row is now historically labelled as the eps-confound.

2. **Three of ten G5 hypotheses (H45, H46, H49) have no sweep row.**
   Their paper-vs-result comparisons are not empirically testable
   from the current pipeline. Until the integration concrete fixes in
   the G5 audit land, these contribute no empirical evidence either
   for or against their literature anchors.

3. **H47 and H48 are WRONG_TEST: schedules that don't schedule.** The
   per-forward-step counter in H47 and the 1-step floor saturation in
   H48 mean that both "schedule" rows test fundamentally different
   mechanisms from what their docs (and the schedule-literature
   anchors) describe. The CIFAR-10 numbers are real but they test
   constant-perturbation, not curriculum schedules.

4. **H42 and H50 are PAPER_DISAGREES — the literature was clear and
   the doc went against it.** He 2015's `2` is not arbitrary, and no
   paper claims 6-prior additive stacking. These are the cleanest
   paper-gap falsifications in G5: the field's prior beliefs were
   correct, and the H42 / H50 numbers reflect that.

5. **H43 is the only PAPER_AGREES winner in G5.** Magnitude pruning
   at moderate sparsity without rewinding gives a small regression —
   exactly what the LTH literature predicts at 12-ep horizons. The
   schedule shape (Fibonacci-indexed) is not privileged by any paper,
   but it is also not pathological.

6. **Rule 1 borderline cases:** H41 PRE-FIX (β + eps both moved) and
   H44 (per-layer schedule + base_wd magnitude both moved). Both have
   concrete-fix recommendations in G5_audit.md.

---

## IMPL_BUG candidate consolidated list (paper-gap lens)

| # | Hypothesis | Bug summary | Status | Severity |
|---|---|---|---|---|
| 1 | H41 | `eps` was confounded with β shift | **Fixed by Fixer-Opt** (stock eps=1e-8 restored) | RESOLVED; rename `GOLDEN_EPS` to make eps-leak impossible |
| 2 | H47 | Counter advances per-forward, not per-epoch | Fixer-Opt patch noted in audit; re-verify post-fix row | MAJOR (re-run) |
| 3 | H48 | Multiplicative rule saturates to floor in 1 step | Not fixed | MAJOR (re-run with exp-curve) |
| 4 | H44 | Sweep row varies two knobs (base_wd + schedule) | Sweep-config bug, not code bug | MINOR (add iso-magnitude row) |
| 5 | H50 | `NaturePriorFlags()` defaults all-True | Footgun, not paper-gap-load-bearing | MINOR |

Items 2 and 3 are the IMPL_BUG candidates that genuinely block a
clean paper-vs-result comparison (the others are confound /
documentation issues).

---

*Audit conducted 2026-05-29 by hostile-reviewer agent for the G5
paper-gap pass. Anchor citations consulted: arXiv:1904.09237,
arXiv:1910.05446, arXiv:1705.08292, arXiv:1502.01852, arXiv:1803.03635,
arXiv:2006.05467, arXiv:1904.00962, arXiv:1711.05101, arXiv:1806.09055,
arXiv:1802.03268, arXiv:1806.08734, arXiv:1905.10832, arXiv:2405.07987,
arXiv:1905.00414, arXiv:1506.01186, arXiv:1608.03983.*
