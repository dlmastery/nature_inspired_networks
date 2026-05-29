# NEXT_SWEEP_PRIORITY.md

Priority queue for the next CIFAR-10 / CIFAR-100 / transformer-track sweep
launch, ranked by expected information gain per GPU hour. Authored
2026-05-28 by the Next-Sweep-Designer agent after the Fixer campaign
closed (Fixer-Priors, Fixer-Growth, Fixer-InitFilter, Fixer-Graphs).
Rule references: Rule 1 (one config change), Rule 11 (commit cadence),
Rule 13 (SOTA smoke first), Rule 19 (phased CIFAR-10 → CIFAR-100),
Rule 21 (post-fix re-run discipline), Rule 22 (dual-track audit),
Rule 23 (orthogonal axes in compounds).

---

## Tier 1 — Post-fix-capability rows (added to `build_matrix`)

These five rows exercise APIs introduced by Fixer agents that previously
had **no** sweep row. All are Rule-1 atomic single-prior deltas vs
`baseline_sg_vanilla`. Launch on CIFAR-10 seed 0, 12-epoch quick smoke
(`configs/cifar10_quick.yaml`). Combined budget: ~5 × 8 min ≈ 40 min on
the 4090.

| # | Tag | Capability tested | Launchable now? |
|---|-----|-------------------|-----------------|
| 1 | `sg_only_hex_phi_radial` | H21 — `HexConv2d(phi_radial=True)` reweights honeycomb by phi^{-r} | **No** — needs `NaturePriorFlags.hex_phi_radial`, `make_flags` forward, `_GenericConv` pass-through |
| 2 | `sg_only_toroidal_phi_scaled` | H22 — `toroidal_pad(phi_scaled=True)` damps wrap by 1/phi | **No** — needs `NaturePriorFlags.toroidal_phi_scaled`, `make_flags` forward, `_GenericConv.forward` plumbing |
| 3 | `sg_only_cymatic_hex` | H28 — per-tap golden-angle phases on hex mask | **No** — needs `NaturePriorFlags.cymatic_hex`, `make_flags` forward, `_GenericConv` HexConv2d→CymaticHexConv swap |
| 4 | `sg_only_dynamic_growth` | H08 — function-preserving fib-scheduled growth (BN gamma=0) | **No** — needs `TrainConfig.growth_schedule`, `Trainer._epoch_loop` step + optimizer rebuild, `runner.run_one` callback wire-up |
| 5 | `sg_only_fractal_filter` | H38 — Fibonacci-kernel `(3,5,8)` parallel filter | **No** — needs `NaturePriorFlags.fractal_filter`, `_GenericConv` Conv2d→FractalGoldenFilter swap |

Wiring effort: each is a single-file (or two-file for #4) additive patch
with no impact on existing rows (defaults preserve byte-for-byte legacy
behaviour). See inline `# TODO runner wiring:` comments in
`scripts/run_sweep.py` for the precise lines to touch. Recommended
landing order: 1, 2 (priors.py-driven, smallest); then 3, 5 (block
swaps); finally 4 (trainer callback, largest patch).

---

## Tier 2 — G7 hybrid smoke / skip verdicts

Source: `audits/G7_audit.md` (PASS / MINOR / MAJOR / BROKEN per
hypothesis). All "PASS" hybrids could plausibly compete; verdicts below
are conditioned on the implementation audit *and* the corresponding
sci-critic addendum (Rule 22).

| Hypothesis | Audit | Verdict | One-line smoke/skip |
|------------|-------|---------|---------------------|
| H62 — Toroidal-KV hex attention | PASS | **smoke** | CIFAR-10 + ViT-Tiny scaffold (when available); toroidal KV is a single-attention-module change — cheap to verify and orthogonal to the channel-mode axis. |
| H66 — Cymatic-orthonormal QKV init | PASS | **smoke** | Init-only delta, no forward-path change → trivially Rule-1; budget 1 GPU run; sci-critic verdict pending but DERIVATIVE+TESTABLE expected. |
| H72 — Fractal-vesica FFN | PASS | **smoke** | FFN swap inside a transformer block; needs ViT scaffold. Risk: depth × multi-path compute can blow batch. Smoke at d_model=192, depth=6. |
| H75 — Harmonic cymatic SwiGLU | PASS | **skip-for-now** | High-quality PASS but compositional with H72; recommend smoking H72 first and revisiting H75 only if H72 ≥ baseline +0.5pp. |

H64 (MAJOR — not wired into runner) and H67/H74 (BROKEN) are deferred
pending re-design; do not include in the next sweep.

---

## Tier 3 — Transformer-track scaffold (H71 IcosaRoPE3D)

H71 is the only G7 hybrid with a **NOVEL+TESTABLE** sci-critic verdict
and no scaffold yet. The CNN-track sweep cannot exercise it because the
mechanism (per-token 3-D rotary embedding over icosahedral vertices)
only applies to attention.

**4-step plan for the ViT-Tiny + H71 smoke:**

1. **Scaffold** (`src/nature_inspired_networks/vit_tiny.py`, new file):
   build a ViT-Tiny (d_model=192, depth=12, heads=3, patch=4) for
   CIFAR-10 32×32. Standard sinusoidal PE on the baseline. Re-use
   existing `nn.MultiheadAttention` so the IcosaRoPE3D module can be
   inserted as a Q/K rotator.
2. **Config** (`configs/cifar10_vit_tiny.yaml` + `cifar10_vit_tiny_sota_smoke.yaml`):
   AdamW lr=5e-4, cosine, 50 epochs, label smoothing=0.1, RandAugment,
   bs=256. Expected baseline top-1 ≥ 78% at 50 ep on CIFAR-10 (Rule 13
   SOTA smoke band).
3. **Wire H71** (`models.py` + `runner.py`): add `vit_tiny_icosa_rope`
   model name; in build, replace per-head Q/K projection output with
   `IcosaRoPE3D(...)`. Add a sweep row `vit_tiny_baseline` and
   `vit_only_icosa_rope_3d` (Rule 1 atomic).
4. **Smoke** seed=0, 50 epochs. Expected delta per H71 design doc:
   ≥ +0.5pp top-1 on CIFAR-10 with no parameter cost. If smoke passes
   the SOTA band, graduate to a 3-seed re-run before any external
   claim.

Budget: ~3 h for the scaffold + 2 × 1 h smoke runs.

---

## Tier 4 — G3 graph hypotheses (H24, H30) — DEFERRED

Fixer-Graphs corrected the rotation matrix for icosahedral GNN
operators. **However**, H24 (icosa-graph message passing) and H30
(spiral-graph attention) are MLP / GNN primitives, not CNN drop-ins —
they cannot meaningfully be smoked on CIFAR-10 image classification
without a graph backbone scaffold. Defer until a GNN benchmark
(QM9 / Cora / OGB-products) is selected for the autoresearch image
sister-project. Add to the backlog as a separate sister-project issue
on `dlmastery/autoresearchimage`.

---

## Tier 5 — 3-seed CIFAR-10 re-runs for error bars

Per Rule 19 Phase-5, headline numbers require seed-median composite
over `--seeds 0 1 2`. Current status (CIFAR-10):

- Multi-seed (≥ 3 seeds): `baseline_resnet20`, `sg_only_phi_budget`.
- Single-seed: every other tag in the matrix.

**Recommended next batch** (in priority order, post-Fixer):

1. `combo3_pb_gm_pd`, `combo4_pb_gm_pd_pdw`, `combo5_*` — the
   combo-ladder rows above the current 1-prior winner.
2. The Tier-2 LOO subtractive rows (`loo_no_*`) on top of
   `combo8_pb_gm_pd_pdw_plr_fe_sa_fp` — already cheap LOO.
3. The 5 new Tier-1 post-fix rows (once their wiring lands).
4. `sg_only_golden_spiral_init`, `sg_only_phi_activation`,
   `sg_only_phi_decay` — three single-prior rows whose seed-0 result
   was within 0.5 pp of baseline (so the verdict is currently
   inconclusive, and a 3-seed median would settle the question).
5. Selected pair-interaction rows (`pair_gm_pdw`, `pair_pd_plr`) if
   their seed-0 number is super-additive vs the corresponding combo2.

Budget: ~12 rows × 2 extra seeds × 12 min = ~5 GPU h.

---

## Launch order summary (single weekend, ~12 GPU h)

1. **Now** (~40 min): smoke the 5 Tier-1 post-fix rows **after** their
   runner-wiring patches land. Commit + push between each launch
   (Rule 11) and use the auto-checkpoint loop (Rule 20) for the
   multi-hour Tier-3 + Tier-5 blocks.
2. **+1 h**: ViT-Tiny baseline SOTA smoke (Tier 3, step 2).
3. **+3 h**: H71 scaffold + smoke (Tier 3, steps 3 + 4).
4. **+5 h**: G7 hybrid smokes — H62, H66, H72 (Tier 2).
5. **+12 h** (overnight): 3-seed re-sweep of Tier-5 selection.

After every block: regenerate dashboards (`scripts/build_dashboard.py`,
`scripts/build_report.py`), refresh `RESULTS.md`, append a journal
entry to the relevant hypothesis docs, commit + push (Rule 11).
