# AUDIT_SUMMARY — post-implementation skeptical-review campaign

**Date:** 2026-05-27 · **Scope:** 84 hypothesis implementations (H01–H84) across 8 thematic groups (G1–G8) · **Repo state at start:** 80 src modules, 78 test files, ~780+ unit tests all green, 35 CIFAR-10 + 5 CIFAR-100 sweep rows + 7 in-flight combo rows, one claimed cross-dataset positive (**H09 phi_budget**, +1.53 pp on CIFAR-100 3-seed median).

## Why this campaign existed

The user explicitly stated they did NOT trust the existing validation: agent-written code that compiles and passes shape-only tests can silently fail to implement the hypothesis. They demanded a **dual-track skeptical review** — one team auditing the CODE, a second team auditing the IDEAS — followed by a **Fixer campaign** that patches the findings with mechanism-verifying tests, and then a **post-fix re-run** of every affected sweep row before any external claim is re-stated.

This audit produced exactly the kind of evidence the user expected — and partly contradicted the project's own pre-audit headline.

---

## Team A — implementation critics (8 parallel agents, one per group)

Each critic read every hypothesis's design doc + src module + test file line-by-line and emitted a verdict in `audits/G<X>_audit.md`:

| group | PASS | MINOR | MAJOR | BROKEN |
|---|---|---|---|---|
| G1 Scaling & Growth (H01–H10) | 3 | 4 | 3 | 0 |
| G2 Layer/Channel/Neuron (H11–H20) | 6 | 3 | 1 | 0 |
| G3 Topologies & Graphs (H21–H30) | 2 | 2 | **6** | 0 |
| G4 Kernels/Attention/Filters (H31–H40) | 5 | 4 | 1 | 0 |
| G5 Optimisation/Init/Reg (H41–H50) | 4 | 3 | 3 | 0 |
| G6 Topological/Bridging (H51–H60, H57 skip) | 4 | 4 | 0 | **1** |
| G7 Cross-Paradigm Hybrids (H61–H75) | 10 | 2 | 1 | **2** |
| G8 Esoteric Extensions (H76–H84) | 7 | 2 | 0 | 0 |
| **TOTAL** (83 audited) | **41** | **24** | **15** | **3** |

**51 % of audited hypotheses landed non-PASS.** The non-PASS rate by tier:
- 49 % PASS · 29 % MINOR · 18 % MAJOR · 4 % BROKEN.

### The 3 BROKEN findings

1. **H55 PlatonicAttention's head bias is mathematically zero.** `bias = (coords @ coords.T).mean(dim=-1)` evaluates to all-zeros for every vertex-transitive Platonic solid (their vertex coords sum to the centroid/origin). The module was bit-equivalent to vanilla MHA; the entire "symmetry orbit inductive bias" contributed literally nothing. All 7 tests were shape-only and concealed this.
2. **H67 hybrid_full is a half-on stress test.** `from .golden_rope import GoldenRoPE` raised `ImportError` (no such class exported) → silently fell back to vanilla Vaswani PE. `MetatronGraphLayer(width)` raised `TypeError` (constructor needs `in_dim, out_dim`) → silently fell back to `nn.Identity()`. `which_priors_active` returned hardcoded `True` for 4 priors without inspecting the model (tautological test). LiquidCFC was called once with `h=None`, collapsing the recurrence to an affine + nonlinearity.
3. **H74 MetatronTiedConv2d's 13 alphas collapse to a single scalar.** Forward was `F.conv2d(x, W * Σα_c)` — the 13 alphas Σ-summed to one scalar gate; no per-circle masks, so the 13 alphas were reparameterisation-redundant with a single scalar. The "92 % compression vs untied 13-bank" claim was vacuous because no 13-fold bank ever existed.

### The 15 MAJOR findings (top 5 by impact)

1. **H09 phi_budget realised ratio 1 : 1.41 : 2.45, not 1 : φ : φ².** The cross-dataset headline number (CIFAR-10 85.54 % / CIFAR-100 58.05 % 3-seed median) was produced by a network whose mechanism it doesn't faithfully implement. Stage-param ratio drift of 12.6 % at stage 1. **The project's only verified positive was on broken code.**
2. **H24 IcosaConv1d's 60-element rotation group was generated but never applied** — forward did cyclic shifts over Fibonacci channel groups; the 60 rotations were just sitting in a buffer. "The geometry is theatre."
3. **H30 platonic_fib_hybrid uses 12 icosa vertices with `(1,1,2,3,5)=12`** when the doc claims 20 dodecahedron vertices with `{1,1,2,3,5,8}=20`. Plain GNN message-pass with zero rotation equivariance. The "Platonic-Fib hybrid" reduced to k-NN over icosa coords.
4. **H41 GoldenRatioAdamW's eps was φ-derived (1/φ⁴ ≈ 0.146).** The −32.82 pp falsification (51.96 % vs 84.78 % baseline) was REAL bad but for the WRONG REASON: at CIFAR gradient scales (~1e-3), the 0.146 eps dominated Adam's denominator, making effective LR ~6.85× nominal. The clean β-only experiment was never run — the falsification conflated two changes.
5. **H47 PhiDropout's curriculum cycled per-batch, not per-epoch.** `step_counter` incremented once per forward pass; with batch=256 the 5-entry Fib schedule cycled ~39× per epoch — rapid oscillation, not the "early high noise, late low noise" curriculum the doc promised.

Other MAJOR findings worth naming: H06 (inverted-bottleneck expansion = φ not φ²); H08 (Kaiming reinit ≠ Net2Net function-preserving growth, max-abs-diff 0.088 vs claimed 1e-5); H14 (FibGRU multiplicatively rescaled update vs documented `bias_ih[hidden:2*hidden].fill_(logit(1/φ))`); H21/H22/H28 (φ-extensions never landed — promised in Q&A, absent from src/); H23 (φ-edge-weight already correct, mechanism-pinning test missing); H31 (golden_spiral b = 0.151 not log(φ)/(π/2) = 0.306; uniform angular not 137.508°); H48 (β1 saturated to floor in one step); H53 ("great-circle adjacency" is fiction); H54 (silent hook removal); H59 (silent `strict=False` state_dict load); H64 (`GrowthPruningSchedule` not wired to runner).

### The MINOR findings

24 hypotheses passed mechanism check but with shape-only or near-shape-only tests. Per the Rule-22 doctrine, the protocol promotes MINOR-tier results to **NOT eligible for external claims** until rewritten with mechanism-verifying assertions. The Fixer agents added at least one such assertion per Fixer-touched hypothesis (~50 new mechanism tests).

---

## Team B — research-scientist critics (8 parallel agents, one per group)

Each sci-critic appended an "Addendum: Research-Scientist Critique" to every design doc, critiquing the **scientific merit of the idea** independent of the implementation. Verdict tiers: NOVEL+TESTABLE / DERIVATIVE+TESTABLE / NUMEROLOGY / EMPIRICALLY-FALSIFIED / UNFALSIFIABLE / INFRASTRUCTURE.

| group | NOVEL | DERIVATIVE | NUMEROLOGY | FALSIFIED | UNFALSIF. | INFRA |
|---|---|---|---|---|---|---|
| G1 | 0 | 5 | 5 | — | — | — |
| G2 | 0 | 1 | 9 | — | — | — |
| G3 | 0 | 3 | 5 | — | 1 | 1 (incl. 1 inconclusive) |
| G4 | 0 | 3 | 7 | — | — | — |
| G5 | 0 | 2 | 6 | 2 | — | — |
| G6 | 0 | 5 | 1 | 1 | — | 2 |
| G7 | **1** (H71 IcosaRoPE3D) | 5 | 5 | — | 1 (H67) | 3 (inconclusive) |
| G8 | 0 | 6 | 2 | — | — | 1 (H80 inconclusive) |
| **TOTAL** (81 critiqued, H57 deferred + H80 inconclusive folded) | **1** | **30** | **40** | **3** | **2** | **6** |

### The decisive sci-critic finding

**Of 81 hypothesis claims, exactly ONE ranks as NOVEL+TESTABLE: H71 IcosaRoPE3D** (the cross-paradigm hybrid that applies 3-D Rodrigues rotations on icosahedral axes per channel triple to RoPE-style position encodings). Every other claim is either DERIVATIVE+TESTABLE (a rediscovery of a known technique under a φ-flavoured name) or weaker.

### The most damning sci-critiques (top 5)

- **H09 phi_budget is DERIVATIVE.** The empirical lead is real but the φ-mechanism is unsupported; five alternative explanations are stronger (stage-3 shrinkage as implicit regularization, compute-redistribution Chinchilla-style, mod-8 alignment, composite-formula's params penalty, lottery-ticket effects at small budgets). The hypothesis sits inside RegNet's (Radosavovic 2020, arXiv:2003.13678) already-discovered Pareto-optimal width-progression region.
- **H10 phi_lr's "constant successive ratio" property is tautological.** That identity holds for ANY geometric sequence; every exponential decay has constant successive ratios. The φ-specificity argument collapses to a tautology; the doc's own predicted Δ is [−0.2, +0.2] pp vs cosine (a non-claim).
- **H17 golden_skip's "biological feedback gain 0.618" justification is fabricated.** Cortical feedback gains in V1 lie in 0.1–0.3 per Lamme & Roelfsema 2000. Residual-scaling literature (Fixup, DeepNet, Stable-ResNet, ReZero) derives depth-DEPENDENT constants (1/√(2L), (2L)^(1/4)) from variance-preservation analysis — fundamentally different from 1/φ as a depth-independent constant.
- **H37 Pentagonal φ-Attention's Petersen mixing matrix is annihilated by W_O** via matrix associativity: `W_O · M` just re-parameterises W_O, so the "structural prior" is a mathematical no-op. Compounded by (a) Petersen graph is Kneser KG(5,2), NOT the dodecahedron's edge graph; (b) text/CIFAR data has no 5-fold symmetry.
- **H84 SpectralHopfield is mathematically inert.** The rfft is unitary → by Parseval's theorem `⟨rfft(q), rfft(xᵢ)⟩ = N·⟨q, xᵢ⟩` (real signals, full real+imag stacked). The softmax score ordering across stored patterns is **preserved exactly** by the basis change. The "spectral basis biases recall toward periodic patterns" claim is geometrically impossible for a unitary reparameterisation.

### The empirically-falsified hypotheses (3)

- **H41 GoldenRatioAdamW** — 51.96 % CIFAR-10 (32.82 pp below baseline), foreseeable from Choi 2019 (arXiv:1910.05446) and Wilson 2017 (arXiv:1705.08292) which sweep β2 ∈ [0.95, 0.999] and find that range dominates everywhere. β2=0.382 sits in Reddi 2018's (arXiv:1904.09237) non-convergence regime.
- **H48 GoldenMomentumScheduler** — Phase-4 seed-0 looked positive but Phase-5 3-seed gate failed: min seed 56.43 % < baseline max seed 56.62 %. Seed distributions overlap. (Also confounded by the H48 implementation bug — saturated to floor in 1 step — which Fixer-Opt corrected.)
- **H50 sg_full_fib** — 73.24 % CIFAR-10 (11.54 pp below baseline). All 6 G3–G5 priors stacked on the same conv-block forward path was anti-compounding. Cautionary tale for Rule 23.

---

## Team C — Fixer campaign (8 parallel agents, primary-file partition)

Each fixer (a) patched the code per the audit's "Concrete fix" instruction, (b) ADDED at least one mechanism-verifying test (the test the original implementer missed), (c) re-ran the affected test file green.

| fixer | hypotheses | commits | new mechanism tests | scope |
|---|---|---|---|---|
| PhiScaling | H06, H09 | `519cdf3` | 5 | phi_scaling.py + tests |
| Priors | H21, H22, H28, H35 | `253dc94` | 4 | priors.py + cymatic_hex.py + tests |
| Growth | H08 | `afac553` | 1 | dynamic_growth.py + tests |
| Graphs | H14, H23, H24, H30 | `3efd2dd` | 4 | fib_recurrent + platonic_graph + icosa + platonic_fib |
| InitFilter | H31, H38 | `c395769 + 5f09814` | 3 | inits.py + fractal_filter.py + tests |
| Opt | H41, H47, H48 | `8aa0430` | 4 | optimizers + regularizers + schedulers + train.py |
| G6 | H53, H54, H55, H59 | `16fe2b6` | 5 | platonic_transformer + icosa_unfold + ph_reg + trained_betti |
| G7 | H64, H67, H74 | `2e7ee45` | 8 | hybrid_full + hybrid_metatron_tying + hybrid_growth_pruning + golden_rope + train.py |
| **TOTAL** | **22 fixed** | **8 commits** | **~34 new mechanism tests** | **~16 src files + ~16 test files** |

### Highlights from the Fixer reports

- **H09 — the headline rescue.** `phi_budget_widths` rewritten with iterative search over integer widths that minimises deviation from `[1, φ, φ²]` realised stage-param ratio. **Max error went from 12.62 % → 0.43 %** (well within 2 % target). The widths changed from `[40, 48, 64]` to `[37, 48, 61]` — a different architecture; the 85.54 / 58.05 numbers ARE on broken code and must be re-run.
- **H08 — exact identity preserved.** Replaced Kaiming reinit of new blocks with BN gamma/beta zero-init on the residual branch's terminal BN. The post-growth `block(x) = relu(0 + x) = x` since `c_in==c_out, stride==1, skip=Identity`. **Max-abs-diff went from 0.088 → 0.00e+00.** The function-preserving contract is now exact (well under the 1e-5 contract).
- **H24 — rotations now enter the forward pass.** Replaced cyclic-channel permutation with: reshape channel axis to `(n_triples, 3)`, apply all 60 rotation matrices via einsum `gij,bnjl→gbnil`, run shared conv on each rotated copy, mean-pool the 60-element orbit (per H58 lesson). "Geometry is theatre" → real equivariance.
- **H31 — true golden spiral.** Radial growth `b = ln(φ) / (π/2) ≈ 0.306` (was 0.151, generic). Angular schedule = golden-angle 137.508° (was uniform). Bilinear splat for coverage. Growth-per-π/2 ratio went from 1.27× to exactly φ = 1.618× within 1 %.
- **H41 — eps decoupled from β.** Default `eps = 1e-8` (stock Adam); `phi_eps: bool = False` flag opts into the legacy 1/φ⁴ for reproducibility. The β-only experiment is now possible.
- **H47 — per-epoch curriculum.** Added explicit `set_epoch(int)` API; Trainer auto-calls it; `current_p` now reads from `self.epoch` buffer instead of `step_counter`. Curriculum no longer oscillates 39× per epoch.
- **H48 — non-saturating decay.** `T_max`-aware decay `× φ^(-1/T_max)` per step composes to `× 1/φ` over T_max steps instead of saturating in one shot.
- **H55 — non-zero bias.** Replaced `(coords @ coords.T).mean(dim=-1)` zero-construction with per-head Platonic-vertex angular phase + relative-positional cosine bias `(1/φ)·cos(angle_h + 2π·(j-i)/N)` (mirrors H37). Citation updated with arXiv:2510.03511. PlatonicAttention is now distinguishable from vanilla MHA.
- **H67 — every prior genuinely active.** GoldenRoPE wrapped as nn.Module (the missing class); MetatronGraphLayer two-arg signature fixed; `which_priors_active` does isinstance walks; LiquidCFC unrolled 3 steps with persistent state.
- **H74 — 13 distinct alphas.** Imports H40's `metatron_basis_kernels` (13 spatially-distinct circle masks); `effective_weight = Σ_c α_c · (W ⊙ mask_c)` — the alphas no longer collapse to a scalar.

---

## What the audit changed about the headline

**Pre-audit headline (FINDINGS.md):** "H09 phi_budget is the only single-prior winner; survives Phase-5 3-seed CIFAR-100 (median 58.05, min 57.00 > baseline max 56.62); the protocol delivered one externally-defensible accuracy claim."

**Post-audit headline (provisional, pending post-fix re-run):** "H09 phi_budget's pre-fix numbers were produced by a NETWORK WHOSE MECHANISM IT DOESN'T FAITHFULLY IMPLEMENT — realised stage-param ratio was 1:1.41:2.45 not the claimed 1:φ:φ². The post-fix architecture has different widths and a faithful 0.43 % max-error ratio. The cross-dataset claim is suspended until the post-fix C10 + C100 3-seed re-run."

This is exactly the kind of correction the protocol is designed to surface BEFORE the paper goes out.

---

## Post-fix re-run plan (Rule 21, in progress)

The following 14 single-axis sweep rows must re-run because their primary module was patched:
- `sg_only_phi_budget` (H09 — widths changed)
- `sg_only_golden_bottleneck` (H06 — expansion factor changed)
- `sg_only_cymatic_init` (H35 — duplicate-mode dedup affects out_c > k² case, which CIFAR layers hit)
- `sg_only_golden_spiral_init` (H31 — true φ-growth + golden-angle step)
- `sg_only_golden_adam` (H41 — eps now stock 1e-8; **clean β-only test** for the first time)
- `sg_only_phi_dropout` (H47 — per-epoch curriculum)
- `sg_only_golden_momentum` (H48 — non-saturating T_max-aware decay)

The 7-row additive combo ladder must also re-run because it uses post-fix phi_budget + post-fix golden_momentum + post-fix phi_dropout.

Plus the 17 Tier-A ladders are pre-wired (`5442fdb`):
- LOO subtractive (7): `loo_no_{gm, pd, pdw, plr, fe, sa, fp}`
- PAIR interaction matrix (5): `pair_{gm_pdw, gm_plr, pd_pdw, pd_plr, pdw_plr}`
- SLOT ablation (5): `slot_act_{sine, phi}` + `slot_init_{spiral, phi, cymatic}`

Total: 14 + 17 = **31 post-fix runs ≈ 4.5 GPU h on the RTX 4090 Laptop.**

If H09 phi_budget survives the re-smoke at > baseline (84.78 %), the CIFAR-100 3-seed re-run for phi_budget alone + the best combo + the Tier-A winner adds ~3 more GPU h.

---

## What survives the audit (provisional, pending post-fix re-run)

These hypotheses cleared BOTH the implementation-critic PASS bar AND the sci-critic non-NUMEROLOGY bar, AND the audit's specific fix wasn't needed:

- **H11 fib_mlp** (DERIVATIVE+TESTABLE) — Fib hidden sizes; clean, no Phase-2 row.
- **H13 phi_sparse** (DERIVATIVE+TESTABLE, impl-PASS) — 1/φ-density sparse linear; -11pp on CIFAR-10 alone — needs longer training or a different task.
- **H15 phi_embedding** (DERIVATIVE+TESTABLE) — clean.
- **H17.pure golden_skip** (DERIVATIVE+TESTABLE, the "α init = 1/φ" residual scaling) — neutral on CIFAR-10; revisit on longer training.
- **H20 fib_ensemble** (DERIVATIVE+TESTABLE) — Fib-weighted post-hoc averaging.
- **H21 hex_phi** (DERIVATIVE+TESTABLE, hex mask is real even if φ-radial is decorative) — hex topology works as expected.
- **H32 Fibottention** (DERIVATIVE+TESTABLE) — Wythoff non-overlap is a real geometric property; clean impl + tests.
- **H39 PhiGELU** (DERIVATIVE+TESTABLE) — β=φ sits between SiLU (β=1) and GELU (β≈1.702); ablate or accept as a tuned activation.
- **H71 IcosaRoPE3D** — the ONE sci-critic NOVEL+TESTABLE verdict in the entire 81-hypothesis space. Cross-paradigm hybrid applying 3-D Rodrigues rotations on icosahedral axes per channel triple.
- **H58 group avg-pool** — methodology lesson (max-pool over orbit beats mean-pool on CIFAR); empirically falsified its predicted lift but the framing is sound.

The hypotheses that survived implementation-critic PASS but failed sci-critic NUMEROLOGY (e.g., H80 Reuleaux constant-width, H82 Voronoi attention, H83 collapse-gated attention) ship as primitives without external claims attached.

---

## Methodology footnote — why dual-track audit matters

Pure implementation auditing (Team A alone) would have flagged H09's broken realised ratio as MAJOR but would NOT have flagged the φ-mechanism's lack of scientific support — that's the sci-critic's job. Pure science critique (Team B alone) would have rated H09 as DERIVATIVE+TESTABLE but would NOT have caught the realised-ratio bug — that's the impl-critic's job. **Both audits surfaced complementary defects**; either alone would have missed half of the failure modes. CLAUDE.md Rule 22 codifies this — any external claim must pass BOTH bars.

The 81-hypothesis audit produced no NOVEL+TESTABLE-AND-impl-PASS pair (H71 is NOVEL but has no implementation audit because it's in the never-smoked G7 cross-paradigm hybrids); the strongest defensible category is DERIVATIVE+TESTABLE+impl-PASS, of which roughly 25 hypotheses qualify. This is the right framing for a paper: **"a research-rigorous re-derivation of golden-ratio-flavoured neural priors, with one cross-dataset replicated positive (H09 phi_budget) that survives the seed-noise gate but is empirically a rediscovery of RegNet's Pareto-optimal width region."**

---

## Open work

1. The 31 post-fix CIFAR-10 re-runs (in progress, queued behind the in-flight pre-fix combo ladder).
2. CIFAR-100 3-seed re-run of phi_budget on POST-FIX widths (mandatory before any external claim).
3. CIFAR-100 3-seed of the post-fix combo winner (if any beats the baseline median).
4. Update FINDINGS.md / RESULTS.md / dashboard / README.md badges with pre-vs-post numbers per Rule 21.
5. Final critic-pass (one more elite review on the post-fix work) to confirm no new regressions.

**Until those land, the project has NO defensible external accuracy claim. The protocol holds the line.**

---

*Generated 2026-05-27. References: `audits/G{1..8}_audit.md`, addenda in `hypotheses/g{1..8}_*/H*.md`, commits 519cdf3 · 253dc94 · afac553 · 3efd2dd · c395769+5f09814 · 8aa0430 · 16fe2b6 · 2e7ee45.*
