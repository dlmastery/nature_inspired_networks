# CIFAR-10 Smoke Plan (Phase 2 of Rule 19)

> Planning document — produced by Smoke-Planner, 2026-05-26. No code edits.
> Authoritative tables: [`IDEA_TABLE.md`](IDEA_TABLE.md),
> [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md). Sweep matrix lives in
> [`scripts/run_sweep.py`](scripts/run_sweep.py) `build_matrix()`.

Per **Rule 19 Phase 2** the smoke must cover **ALL 75 hypotheses** before any
graduate to CIFAR-100 (Phase 4). We currently have 13 rows covering ~8
hypotheses. This plan defines the *next 15* rows plus the three v2 fixes
identified by Code-X / Code-Y, ordered by (a) implementation cost ascending,
(b) expected information gain descending — the cheap broad scan precedes the
expensive deep dive.

---

## Status today

The 13 rows already in `build_matrix()` (curated path; 11 rows + 2 H58
follow-ups), all on CIFAR-10, 12 ep, seed 0:

| # | Tag | Covers | Top-1 | Composite | Status |
|---|---|---|---|---|---|
| 1 | `baseline_resnet20` | rail / Rule-13 anchor (He 2015) | 84.78 % | 0.8458 | ✓ leader |
| 2 | `baseline_sg_vanilla` | rail (NaturePrior, all priors off, linear) | 82.16 % | 0.8258 | ✓ |
| 3 | `sg_chan_fib` | H04 fib widths | 80.11 % | 0.8135 | ✓ tie w/ phi (mod-8 collapse) |
| 4 | `sg_chan_phi` | H04 phi widths | 80.11 % | 0.8152 | ✓ tie w/ fib |
| 5 | `sg_only_hex` | H21 partial (mask, no φ) | 79.32 % | 0.7941 | ✓ mild neg |
| 6 | `sg_only_group` | H24 proxy (C4, max-pool) | 69.84 % | 0.6937 | ✓ biggest single neg |
| 7 | `sg_only_fractal` | H05 partial (depth=2, no 1/φ) | 82.46 % | 0.8104 | ✓ **only single prior that lifted** |
| 8 | `sg_only_toroidal` | H22 partial (circ pad, no φ) | 78.05 % | 0.7768 | ✓ neg as predicted |
| 9 | `sg_only_cymatic_init` | H35 | 77.44 % | 0.7883 | ✓ unexpected neg |
| 10 | `sg_only_golden_modulate` | H17 / H34 partial | 79.81 % | 0.8042 | ✓ near no-op |
| 11 | `sg_full_fib` | H50 full hybrid | 73.24 % | 0.6966 | ✓ WORST |
| 12 | `sg_only_group_avg` | H58 (max → mean fix) | 65.38 % | 0.6597 | ✓ DISCARDED (-4.46 pp) |
| 13 | `sg_full_fib_avg` | H58 + H50 | 66.86 % | 0.6432 | ✓ DISCARDED (-6.38 pp) |

**Coverage as of today:** 8 of 75 hypotheses ≥ partially smoked
(H04, H05.partial, H12.implicit, H17/H34.proxy, H21.partial, H22.partial,
H35, H50, H58). **67 hypotheses remain unsmoked.**

---

## What's blocked on implementation

Hypotheses where no primitive exists in `priors.py` / `blocks.py` — each needs
a code change before a smoke row can be added. Cost = S (≤ 30 LoC, single
file), M (new module + tests), L (new model class or training-loop hook).

| H-ID | Short name | What's missing | Cost |
|---|---|---|---|
| H02 | Fib depth progression | model factory currently uses fixed stage layout; needs `stage_blocks=[3,5,8,13]` arg in `NaturePriorNet` | S |
| H06 | Golden bottleneck | new `BottleneckConv(c_in, c_mid=c_in/φ, c_out)` module in `priors.py`; rewire `_GenericConv` to use it under a flag | M |
| H08 | Dynamic φ-growth | training-loop callback to add stages at epochs `[Fib(k)·E/Fib(K)]`; touches `runner.py` | L |
| H10 | φ-decay LR | new `PhiDecayLR(lr0, gamma=1/PHI)` in `optim.py`; CLI plumbing | S |
| H13 | Golden sparse connectivity | new `PhiSparseLinear` w/ mask pruned by 1/φ; only used in classifier head | M |
| H16 | Fib head diversity | requires a ViT path (none today); defer until Tier-4 LLM track | L |
| H19 | φ ReLU threshold | new `PhiReLU(nn.Module)` with learnable threshold init at 1/φ | S |
| H23 | Platonic graph (Metatron) | full GNN backbone — out of scope for CIFAR; smoke on ogbg later (Tier-3) | L |
| H24 | Icosahedral equivariant | `e2cnn`-style icosa conv; very heavy | L |
| H25 | Dodeca latent | linear projector + dodeca-vertex normalization; small new module | S |
| H28 | Cymatic hex resonance | time-dependent hex tap weights; needs `t` arg in forward | M |
| H29 | φ small-world | GNN backbone (defer) | L |
| H31 | Golden-spiral kernel init | new `golden_spiral_init_(conv)` next to `cymatic_init_` | S |
| H32 | Fibottention | ViT path required; defer to Tier-4 | L |
| H33 | Vesica-piscis filter | multi-circle mask basis; new `vesica_kernel_mask(k)` + multi-path wrap | M |
| H36 | φ-spiral PE | needs Transformer path; defer Tier-4 | L |
| H37 | Pentagonal attention | ViT path required; defer | L |
| H38 | Fractal golden filter | multi-scale {3,5,8} kernel ensemble; new `FractalKernelConv` | M |
| H39 | Harmonic φ activation | `x * sigmoid(x * φ)` — one-liner in `priors.py` | S |
| H40 | Metatron kernel overlap | tied-weight kernel basis; new `MetatronConv` | M |
| H41 | φ AdamW | new `PhiAdamW` (β1=1/φ, β2=1/φ²) in `optim.py` | S |
| H42 | φ weight init | replace `kaiming_normal_(gain=√2)` w/ `gain=√φ`; one-liner flag | S |
| H43 | Fib pruning | iterative-magnitude pruner + scheduler; new module | M |
| H44 | Golden regularization | per-layer weight-decay `1/φ^depth` in optimizer; one-line param-group split | S |
| H46 | Cymatic loss | Fourier-domain aux loss term; touches `runner.py` loss | M |
| H47 | φ dropout | new `PhiDropoutScheduler` cycling rates 1/φ → 1/φ² → 1/φ³ | S |
| H48 | Golden momentum | β1 schedule `φ^-epoch`; small optimizer hook | S |
| H49 | PRH alignment loss | CKA-to-Platonic-target auxiliary loss | M |
| H51 / H54 | Differentiable Betti loss | wire `topo-loss` or `TopologyLayer`; back-prop through PH | L |
| H52 | DropPath anytime | DropPath into `_FractalPath`; tiny | S |
| H53 | Icosa unfold bridge | GICOPix planar unfold; new geometry module | M |
| H55 | Platonic Transformer | full ViT-equivariant — defer Tier-4 | L |
| H61 – H75 | Cross-paradigm hybrids | depends on Tier-4 LLM scaffold | L (mostly) |

The unsmoked-but-already-implementable set is small: only rows that flip
existing `NaturePriorFlags` or change `channel_mode` are zero-code. Everything
else needs at least an **S** patch.

---

## Next batch — top 15 hypotheses to add NEXT

Ranked by (a) implementation cost ascending, (b) expected information gain
descending. The first 5 are **S-cost** and can ship in a single Code-Fixer
turn. Rows 6 – 12 are **M-cost** and need a small new primitive each. Rows
13 – 15 are M / L and complete the CIFAR-feasible portion of the design space.

### Cheap (S — same-day; flag flip or ≤ 30 LoC)

#### 1. H58.v2 / H21.v2 — Hex with φ-radial tap weighting
- Already-implementable? **No** — needs a radial weight buffer in `HexConv2d`.
- Cost: **S** (≤ 20 LoC). Multiply `mask` by `1/φ^r` where r = hex ring.
- Config delta: new flag `hex_phi_weight: bool = False` on `NaturePriorFlags`.
- Sweep tag: `sg_hex_phi_weight` (already queued as T2.7 in EXPERIMENT_LOG).
- Why interesting: H21 partial *already* lost 0.79 pp; full version with φ
  weighting is the falsifier — either it recovers, or H21 is dead.

#### 2. H42 — φ-scaled weight initialization
- Already-implementable? **No** — one-line patch (gain `√φ` instead of `√2`).
- Cost: **S** (~5 LoC + flag).
- Config delta: `phi_init: bool` flag → `kaiming_normal_(gain=math.sqrt(PHI))`.
- Sweep tag: `sg_phi_init`.
- Why interesting: cheapest possible test of "φ improves trainability";
  Pennington 2017 dynamical-isometry prediction; orthogonal to every other
  prior we've tested.

#### 3. H39 — Harmonic φ activation
- Already-implementable? **No** — new activation module.
- Cost: **S** (one-line: `x * torch.sigmoid(x * PHI)`).
- Config delta: `harmonic_phi_act: bool` flag → swaps ReLU in `_GenericConv`.
- Sweep tag: `sg_harmonic_phi_act`.
- Why interesting: orthogonal axis (activation, not conv); GELU/SiLU family
  has well-known smooth-gating benefits — testing whether φ specifically wins.

#### 4. H17 — Golden ratio skip scale (proper, not the modulate proxy)
- Already-implementable? **No** — `skip_scale` arg on `NaturePriorBlock`.
- Cost: **S** (~10 LoC: `self.skip_scale = 1/PHI`).
- Config delta: `phi_skip: bool` flag → multiplies `identity` by 1/φ.
- Sweep tag: `sg_phi_skip`.
- Why interesting: H17.proxy (`golden_modulate`) was near no-op; this is the
  *strict* H17. ESN-style spectral-radius argument; cheapest test of skip-path
  damping vs. He's free 1.0 skip.

#### 5. H52 — DropPath anytime
- Already-implementable? **Partial** — `_FractalPath` exists; add `DropPath`.
- Cost: **S** (~15 LoC: stochastic-depth on branch B).
- Config delta: `drop_path: float = 0.0` on `NaturePriorFlags`.
- Sweep tag: `sg_fractal_droppath`.
- Why interesting: fractal was the *only* single prior that *lifted* top-1
  (82.46 %); DropPath is the canonical FractalNet 2017 regulariser — likely
  to push it past baseline. High information gain at trivial cost.

#### 6. H10 — φ-decay LR scheduler
- Already-implementable? **No** — needs `PhiDecayLR` in optim.
- Cost: **S** (~25 LoC scheduler class + CLI plumbing).
- Config delta: `lr_schedule: "phi_decay"` (alongside existing `cosine`).
- Sweep tag: `sg_phi_lr`.
- Why interesting: orthogonal to architecture; tests whether the φ-prior
  helps *training dynamics* even when the model is vanilla ResNet-20.

#### 7. H47 — φ-dropout schedule
- Already-implementable? **No** — scheduler hook + param.
- Cost: **S** (~15 LoC, callback in runner).
- Config delta: `dropout_schedule: "phi_cycle"`.
- Sweep tag: `sg_phi_dropout`.
- Why interesting: only regularisation-axis test in the current sweep.

#### 8. H48 — Golden momentum schedule
- Already-implementable? **No** — optimizer hook.
- Cost: **S** (~10 LoC).
- Config delta: `momentum_schedule: "phi"` (β1: 1/φ → 1/φ²).
- Sweep tag: `sg_phi_momentum`.
- Why interesting: pairs with H10, isolates momentum from LR.

### Mid-cost (M — half-day; new primitive + unit tests)

#### 9. H06 — Golden ratio bottleneck
- Already-implementable? **No** — new `BottleneckConv` module.
- Cost: **M** (new module + tests; ~80 LoC).
- Config delta: `bottleneck: "phi"` channel option on `NaturePriorBlock`.
- Sweep tag: `sg_phi_bottleneck`.
- Why interesting: ResNet-50's bottleneck reduces FLOPs ~6×; φ-ratio version
  is the principled "Why 4×? Why not φ²?" probe. Could *reduce* params at
  similar acc → strong composite gain.

#### 10. H02 — Fibonacci depth progression
- Already-implementable? **No** — `stage_blocks` arg on model factory.
- Cost: **M** (model factory change + tests; ~60 LoC).
- Config delta: `stage_blocks: [3, 5, 8]` (Fib) vs default `[3, 3, 3]`.
- Sweep tag: `sg_fib_depth`.
- Why interesting: changes total parameter count (need iso-param control row);
  tests whether 1.5-2× convergence claim survives smoke epochs.

#### 11. H31 — Golden-spiral kernel init
- Already-implementable? **No** — new `golden_spiral_init_()` next to cymatic.
- Cost: **M** (~50 LoC + tests).
- Config delta: `spiral_init: bool` flag.
- Sweep tag: `sg_spiral_init`.
- Why interesting: H35 cymatic_init was a surprise negative; spiral-init is
  the second-most natural "geometric" init and would disambiguate "init
  geometry hurts" from "Chladni-specific hurts."

#### 12. H38 — Fractal golden filter (multi-scale {3,5,8} kernel)
- Already-implementable? **No** — new multi-scale conv module.
- Cost: **M** (~80 LoC + tests).
- Config delta: `multi_scale_kernel: bool` flag.
- Sweep tag: `sg_fractal_golden_filter`.
- Why interesting: pairs with H05 (which lifted); tests whether multi-scale
  *kernels* (not just multi-scale *paths*) inherit the fractal advantage.

#### 13. H33 — Vesica-piscis filter
- Already-implementable? **No** — circle-mask basis + multi-path wrap.
- Cost: **M** (~70 LoC + tests).
- Config delta: `vesica: bool` flag (replaces `hex` mask with vesica).
- Sweep tag: `sg_vesica`.
- Why interesting: alternative geometric prior on the same axis as hex;
  direct comparison against H21.

#### 14. H49 — PRH (Platonic representation alignment) loss
- Already-implementable? **No** — auxiliary CKA loss.
- Cost: **M** (~60 LoC; runner loss term).
- Config delta: `prh_aux_weight: float`.
- Sweep tag: `sg_prh_aux`.
- Why interesting: the *only* auxiliary-loss prior in the current design
  space; orthogonal to every architecture flag.

#### 15. H40 — Metatron kernel overlap
- Already-implementable? **No** — tied-weight conv basis.
- Cost: **M** (~100 LoC + tests).
- Config delta: `metatron_tie: bool` flag.
- Sweep tag: `sg_metatron_tie`.
- Why interesting: parameter-tying is currently absent from the sweep;
  reduces param count by ~30 % at iso-FLOPs — strong composite lever.

---

## v2 fixes Code-X / Code-Y identified

Three concrete bugs in already-implemented priors. Fixing them re-tests an
already-smoked hypothesis under the *correct* implementation; the original
negative result cannot be trusted until each fix lands.

### Fix A — H05 fractal lacks explicit 1/φ depth shrink (Code-X)

- **Bug:** `_FractalPath` recurses with constant channel count; H05 demands
  `c_at_depth_k = c_in / PHI^k`. Current depth=2 path keeps `c_out` constant.
- **Priority:** HIGH. H05 is *the* positive single prior in the existing
  sweep (+0.30 pp), so the corrected variant is the most likely lifted
  composite of any row we've planned.
- **Fix sketch:** in `_FractalPath.__init__`, when `depth > 1`, allocate
  `c_inner = max(8, int(round(c_out / PHI)))` for `self.b2`; add bridging
  `1x1` to project back to `c_out` before the mean merge.
- **Expected delta vs. buggy:** +0.5 to +1.5 pp top-1 if H05 is real; small
  param reduction (≈ -8 %) as a side benefit.
- **Sweep tag:** `sg_fractal_phi_shrink` (already queued as T2.5).

### Fix B — H21 hex_kernel_mask 3×3 is only 180°-symmetric, not 6-fold (Code-Y)

- **Bug:** The current 3×3 mask
  ```
  1 1 0
  1 1 1
  0 1 1
  ```
  has 2-fold rotational symmetry (axial-offset hex emulation) but NOT the
  6-fold symmetry the hex prior actually wants. HexaConv 2018 uses two
  alternating offset-row stencils; we use one.
- **Priority:** MEDIUM. H21 was a mild negative (-0.79 pp) — the bad mask
  is a plausible cause. Could fully flip the verdict.
- **Fix sketch:** average two opposite-offset masks (the current mask and
  its horizontal flip) on alternating rows, OR upgrade to the radius-2
  mask at k=5 which IS 6-fold symmetric. Document both variants.
- **Expected delta vs. buggy:** +0.3 to +1.0 pp; rotational-equivariance
  error should drop ~30 %.
- **Sweep tag:** `sg_hex_6fold` (new row; combine with H21.v2 φ-weight as
  `sg_hex_full` for the full-H21 row).

### Fix C — H35 cymatic_init_ lacks orthonormalization (Code-X)

- **Bug:** `cymatic_init_` draws `coef ~ N(0,1)` then L2-normalises — so the
  basis is normalised per (o,i) pair but the basis itself (`chladni_modes`)
  is not orthonormalised. Modes can have non-trivial overlap (e.g., (1,2)
  and (2,1) are mirrored, not orthogonal in general). This *plus* the
  fan-in scaling `√(2/fan_in)` likely cancels each other.
- **Priority:** MEDIUM-LOW. H35 was a clear negative (-2.67 pp vs. baseline
  NaturePrior); fixing it might recover the gap.
- **Fix sketch:** Gram-Schmidt over `chladni_modes` output before
  `cymatic_init_` consumes it; flatten each mode to a vector, QR-decompose,
  reshape back.
- **Expected delta vs. buggy:** +0.5 to +1.5 pp on `sg_only_cymatic_init`;
  if it doesn't move, H35 is genuinely dead and we move on.
- **Sweep tag:** `sg_cymatic_init_ortho`.

---

## Concrete next-turn execution plan

A short checklist for the Code-Fixer agent to ship the next batch of smoke
rows. Each step is a single commit (Rule 11). Steps within a phase are
independent and can be reordered; phases are strictly ordered.

### Phase A — v2 fixes (clears trust on existing rows)

1. **Fix C — H35 ortho.** Patch `chladni_modes` with Gram-Schmidt; add
   regression test in `tests/test_priors.py` asserting orthonormality.
2. **Fix B — H21 6-fold mask.** Add `hex_kernel_mask_6fold(k=3)` (averaged
   offset variant); make `HexConv2d` accept a `mask_variant: str` arg
   defaulting to legacy. Unit test for 6-fold rotational symmetry under
   `torch.rot90`.
3. **Fix A — H05 1/φ depth shrink.** Patch `_FractalPath` to allocate
   `c/φ` channels at inner recursion; add 1x1 projection. Unit test that
   param count matches `c_in + c_out + c_out/φ + c_out` budget.
4. Append three new rows to `build_matrix()`:
   `sg_cymatic_init_ortho`, `sg_hex_6fold`, `sg_fractal_phi_shrink`.
5. Run sweep over the three new tags + the three buggy rows for direct
   comparison (`--only sg_cymatic_init_ortho sg_hex_6fold sg_fractal_phi_shrink`).
6. Append rows to `EXPERIMENT_LOG.md` Tier-2; update `IDEA_TABLE.md` status
   for H05 / H21 / H35 to `~ partial-v2` or `✓ done`.
7. Commit + push.

### Phase B — Cheap S-cost batch (8 new rows)

8. Add `NaturePriorFlags` fields: `hex_phi_weight`, `phi_init`, `harmonic_phi_act`,
   `phi_skip`, `drop_path`. Update `tag()` for new fields.
9. Implement each toggle (each ≤ 30 LoC):
   - `hex_phi_weight` → multiplicative buffer in `HexConv2d.__init__`.
   - `phi_init` → branch in `_GenericConv` init.
   - `harmonic_phi_act` → swap `F.relu` for `x * sigmoid(x*PHI)`.
   - `phi_skip` → `self.skip_scale = 1/PHI` in `NaturePriorBlock`.
   - `drop_path` → stochastic-depth in `_FractalPath` branch B.
10. Add `PhiDecayLR`, `PhiDropoutScheduler`, `phi_momentum_schedule` to
    `optim.py` (new module if absent) + CLI plumbing in runner.
11. Add unit tests in `tests/test_priors.py` and `tests/test_blocks.py`
    for each flag (shape + on/off equivalence + the H42 gain assertion).
12. Add 8 rows to `build_matrix()`:
    `sg_hex_phi_weight`, `sg_phi_init`, `sg_harmonic_phi_act`, `sg_phi_skip`,
    `sg_fractal_droppath`, `sg_phi_lr`, `sg_phi_dropout`, `sg_phi_momentum`.
13. Run sweep (~75 min on RTX 4090 Laptop at 12 ep).
14. Append rows to `EXPERIMENT_LOG.md` Tier-1 v2 section. Update IDEA_TABLE
    status for H17 / H39 / H42 / H47 / H48 / H10 / H52.
15. Commit + push.

### Phase C — Mid-cost M batch (7 new rows)

16. New module `priors.py:BottleneckConv` (H06) with `(c, c/φ, c)` channels.
17. New init `golden_spiral_init_(conv)` (H31).
18. New module `FractalKernelConv` (H38) — multi-scale {3,5,8} ensemble.
19. New mask `vesica_kernel_mask(k)` (H33).
20. New module `MetatronConv` (H40) — tied-weight basis.
21. Model factory: `stage_blocks` arg + Fib default option (H02).
22. New runner loss term `prh_aux_loss` (H49).
23. Unit tests for each (shape, equivalence-when-off, smoke forward pass).
24. Add 7 rows to `build_matrix()`:
    `sg_phi_bottleneck`, `sg_fib_depth`, `sg_spiral_init`,
    `sg_fractal_golden_filter`, `sg_vesica`, `sg_prh_aux`, `sg_metatron_tie`.
25. Run sweep (~75 min).
26. Append rows to `EXPERIMENT_LOG.md`. Update IDEA_TABLE for
    H06 / H02 / H31 / H38 / H33 / H49 / H40.
27. Commit + push.

### Phase D — Dashboard + findings refresh

28. `python scripts/build_dashboard.py` → refresh dashboard.
29. `python scripts/build_report.py` → refresh RESULTS.md narratives.
30. Append a "Round 2 sweep — 18 new rows" section to `FINDINGS.md`
    identifying top-K composite performers (Rule 19 Phase 3 trigger).
31. Update `IDEA_TABLE.md` running tally: target ≥ 28 / 75 ideas at least
    partially smoked after this batch.
32. Commit + push.

### Notes for Code-Fixer

- **Rule 1**: Each new row flips exactly one flag relative to its baseline.
  Combinations (e.g., `phi_init + harmonic_act`) are NOT in this batch.
- **Rule 12**: Every new module ships with `tests/test_<module>.py` covering
  shape, on/off equivalence, and a regression test for the bug class it
  prevents.
- **Rule 11**: One commit per row appended to `build_matrix()`; one commit
  per implementation file. Push before *every* sweep launch.
- **Rule 13**: Before launching Phase B / C sweeps, re-run
  `configs/cifar10_sota_smoke.yaml` and confirm ≥ 80 % top-1 at 12 ep.
- **Rule 8**: Each row's archive directory must contain README.md,
  `config.yaml`, `reasoning.json`, and `run_seed0/{metrics, history, best.pt}`.
- **Rule 19 phase order**: do NOT launch CIFAR-100 on any of these rows
  yet — that is Phase 4 work after Phase 3 check-in dashboard.

### Out-of-batch deferrals

The following are flagged in `IDEA_TABLE.md` but explicitly **deferred** to
Tier-4 (LLM track) or Tier-3 (graphs / spheres): H14, H15, H16, H23, H24,
H25, H29, H32, H34, H36, H37, H53, H55, H56, H57, H61–H75. CIFAR-10 cannot
falsify them — wrong dataset for the inductive bias.

H08 (dynamic φ-growth) and H51 / H54 (differentiable Betti loss) are
deferred to a later batch because they touch the training loop and the
runner; they don't fit the "single flag flip + new primitive" cadence of
this batch.

---

*Plan author: Smoke-Planner. No code touched. Hand off to Code-Fixer with
Phase A as the next-turn target.*
