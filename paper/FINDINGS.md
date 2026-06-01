# FINDINGS — nature_inspired_networks curated CIFAR-10 sweep (seed 0, 12 epochs)

> ## 🏆 2026-05-29 PM PROMOTION — Phase-8 winners → EVALUATION tier (CERTIFIED at α=0.05 Holm-Bonferroni, n=7)
>
> The Phase-8 family was extended from n=3 to **n=7 seeds** on CIFAR-100
> 30-ep. All three winners produced 7/7 positive paired deltas against
> baseline, yielding paired Wilcoxon W=0 with exact one-sided p=(1/2)^7
> = **0.0078** — below the Holm-Bonferroni-corrected per-test α'=0.05/3
> = 0.0167 demanded for the k=3 simultaneous-test family. **All three
> winners CLEAR α=0.05 under Holm-Bonferroni step-down.**
>
> | tag | C100 mean (n=7) | Δmean | 95% bootstrap CI | Wilcoxon p | Holm cleared? |
> |---|---:|---:|---:|---:|:---:|
> | `pair_gm_pdw` (H09+H48+H44) | **0.5786** | **+1.74 pp** | [+1.42, +2.09] pp | 0.0078 | **YES** |
> | `slot_act_sine` (H81 SIREN) | **0.5790** | **+1.78 pp** | [+1.38, +2.18] pp | 0.0078 | **YES** |
> | `sg_only_phi_budget` (H09 post-fix) | **0.5736** | **+1.24 pp** | [+0.84, +1.67] pp | 0.0078 | **YES** |
> | `baseline_resnet20` (rail, n=7) | 0.5612 | — | — | — | — |
>
> Per CLAUDE.md [Rule 28](../CLAUDE.md#rule-28), the three winners are
> **PROMOTED from SCREENING to EVALUATION tier** on 2026-05-29 PM. These
> are the project's **first formally-certified empirical claims at
> NeurIPS-standard α=0.05**.
>
> **Honest caveat (preserved):** 12-ep CIFAR-10 and 30-ep CIFAR-100 are
> still NOT the convergence regime. The certification holds AT THIS
> COMPUTE BUDGET. Whether the lift survives at the literature-canonical
> 164-ep training + tuned RegNetX-200MF baseline (Phase-9a hill-climb)
> is open. The case-study framing of PAPER.md §5.5 is unchanged: the
> protocol's screen + n=7 evaluation surfaced three candidates that
> pass formal rigor at the screening budget; the φ-content vs
> any-three-orthogonal-axes attribution question remains.
>
> Full per-claim certification table + Holm step-down derivation:
> [`paper/STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md) §0–§3.
>
> ### 🔬 2026-06-01 update — Phase-9f n=7 iso-tuned extension complete (cert STANDS at default-config; iso-tuned regime NOT re-certified)
>
> Phase-9f n=7 iso-tuned extension complete. The iso-tuned cell
> (lr=3e-3, wd=5e-4, bs=128, AdamW; `slot_act_sine` at wd=2e-3) was
> extended to n=7 seeds on both the baseline and the three leaders.
> **Iso-tuned-cell Δ shrinks to +0.54 to +0.79 pp** (Δmean_unpaired
> vs default-config Δmean +1.24 to +1.78 pp). Paired Wilcoxon at
> iso-tuned n=7: `pair_gm_pdw` W=4.0 p_one=0.1094 (4/7 positive
> deltas), `sg_only_phi_budget` W=3.0 p_one=0.0781 (n=6 after seed-3
> exclusion as <30 ep), `slot_act_sine` W=2.0 p_one=0.3750 (n=4 at
> wd=5e-4 baseline neighbour). **Phase-5 ordinal gate FAILS at
> iso-tuned n=7 for all three winners** (max iso-tuned baseline =
> 0.6075 at seed=3; min iso-tuned leader seeds 0.5998 / 0.6049 /
> 0.6057 all ≤ 0.6075). **The default-config n=7 cert STANDS as the
> formal claim of the paper**, but the iso-tuned-regime equivalent
> CANNOT be certified at NeurIPS-α with this sample size. **R2
> BLOCKER #13 concern partially validated** — at iso-tuned bs=128
> lr=3e-3 the baseline gets the same tuning love and the relative
> lift shrinks substantially. Phase-9g (n=15+ iso-tuned extension)
> is the principled re-certification path; Phase-9e (wd=2e-3
> baseline neighbour for `slot_act_sine`) is the related closure.
> Full Phase-9f closeout table:
> [`paper/STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md) §10.

> ## ⛰️ 2026-05-30 PM — Hill-climbed-best regime confirms the n=7 default-config certification (Phase-9a, BLOCKER #13 partial refutation)
>
> Phase-9a ran a per-hypothesis coordinate hill-climb (cube:
> lr × weight_decay × batch_size × optimizer, budget 25, see
> [`scripts/run_hillclimb.py`](../scripts/run_hillclimb.py))
> independently on `baseline_resnet20` and on each of the three n=7
> winners on CIFAR-100 30-ep. Each tag's hill-climbed best_config was
> re-run on seeds 0/1/2 and recorded in `ideas/<NN>/hillclimb_results.json`.
>
> | tag | hill-climbed best_config | top1 median (n=3) | Δmedian vs hill-climbed baseline |
> |---|---|---:|---:|
> | `baseline_resnet20` | lr=3e-3 wd=5e-4 bs=256 AdamW | **0.5929** | — (rail) |
> | `sg_only_phi_budget` | lr=3e-3 wd=5e-4 bs=128 AdamW | 0.6049 | **+1.20 pp** |
> | `pair_gm_pdw` | lr=3e-3 wd=5e-4 bs=128 AdamW | 0.6109 | **+1.80 pp** |
> | `slot_act_sine` | lr=3e-3 wd=2e-3 bs=128 AdamW | 0.6137 | **+2.08 pp** |
>
> The priors carry signal in BOTH tuning regimes — default-config (n=7,
> Δmean +1.24 / +1.74 / +1.78 pp, Holm-Bonferroni cleared) AND
> hill-climbed-best (n=3, Δmedian +1.20 / +1.80 / +2.08 pp). The
> area-chair concern (BLOCKER #13) that "any properly-tuned baseline
> would close the gap" is qualitatively refuted at this compute budget.
>
> **n=3 at hill-climbed best does NOT clear Holm-Bonferroni by itself.**
> The exact paired-Wilcoxon one-sided floor at n=3 is (1/2)^3 = 0.125,
> above α'_Holm = 0.0167. The formal certification of the paper remains
> the n=7 default-config result (banner above). The hill-climb is an
> ADDITIVE robustness extension. An n=7 hill-climbed extension (Phase-9c)
> would deliver full Holm-Bonferroni clearance in the tuned regime and
> is filed as future work.
>
> Full Section 7 statistical analysis (Wilcoxon, bootstrap CI, Phase-5
> ordinal gate at n=3):
> [`paper/STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md) §7.
> Splice in PAPER.md: [`§5.5.4`](../PAPER.md#554--hill-climbed-best-regime-phase-9a-2026-05-30--blocker-13-refutation).

> ## 🌐 2026-05-31 — Iso-tuned-comparison closeout (Phase-9a baseline-extension, n=3) — SUPERSEDED by Phase-9f n=7 (see 2026-06-01 update in the PROMOTION block above)
>
> The Phase-9a hill-climb's Section-7 comparison used the hill-climbed
> baseline at bs=256 vs the three leaders' hill-climbed bests at bs=128
> — confounding "prior helps" with "bs=128 helps the baseline." The
> 2026-05-31 baseline-extension filled out the baseline at the iso-tuned
> cell (lr=3e-3, wd=5e-4, bs=128, AdamW) to n=3 so prior-vs-non-prior
> can be measured at matched bs/lr/wd/optimizer.
>
> | tag | iso-tuned cell | n | mean | σ (pp) | Δmean vs baseline (iso-tuned) | 95% boot CI | Phase-5 gate |
> |---|---|---:|---:|---:|---:|---|:---:|
> | `baseline_resnet20` | lr=3e-3 wd=5e-4 bs=128 AdamW | 3 | 0.5937 | **1.14** | — (rail) | — | — |
> | `sg_only_phi_budget` | lr=3e-3 wd=5e-4 bs=128 AdamW | 3 | 0.6053 | 0.57 | **+1.16 pp** | [−0.04, +2.30] pp **(contains 0)** | FAIL |
> | `pair_gm_pdw` | lr=3e-3 wd=5e-4 bs=128 AdamW | 3 | 0.6096 | 0.34 | **+1.59 pp** | [+0.43, +2.62] pp | FAIL (tie 0.6057) |
> | `slot_act_sine` | lr=3e-3 wd=2e-3 bs=128 AdamW | 3 | 0.6105 | 0.57 | **+1.68 pp** | [+0.49, +2.77] pp | FAIL |
>
> **Closeout verdict.** The certified-at-default-config n=7 claims
> (banner at top of file) STAND. The default-config baseline σ at n=7
> is small (0.453 pp), the Δs of +1.24 / +1.74 / +1.78 pp exit
> 2σ_default = 0.91 pp, and the paired Wilcoxon n=7 floor (0.0078) clears
> Holm-Bonferroni α'=0.0167.
>
> **The iso-tuned-cell extension at n=3** is a robustness check that
> confirms directional positive Δ for all three winners across the
> hyperparameter regime (every winner's iso-tuned mean exceeds the
> iso-tuned baseline mean), but cannot itself re-certify at NeurIPS α
> because (a) σ_iso = 1.14 pp at n=3 is 2.5× wider than σ_default, (b)
> the Phase-5 ordinal gate FAILS for all three winners (max baseline =
> 0.6057, min leaders 0.5998 / 0.6057 / 0.6039 ≤ 0.6057), and (c) the
> n=3 Wilcoxon floor (0.125) cannot clear Holm-Bonferroni α'=0.0167.
>
> Phase-9f (n=7+ iso-tuned baseline-AND-leader extension) is the
> principled re-certification path at the tuned regime. The
> `slot_act_sine` wd=2e-3 baseline-neighbour extension is filed as
> Phase-9e.
>
> Exclusion criterion (Rule 3-compatible, transparent at the analysis
> layer): `sg_only_phi_budget__hc_lr3em3_wd5em4_bs128_optAdamW_seed3`
> ran 2 epochs (top1=0.2148) as a diagnostic-budget hill-climb-search
> cell, NOT a 30-ep evaluation seed; excluded from the n=3 iso-tuned
> stats above. Underlying metrics.json is unchanged per Rule 3.
>
> Full Section 10 statistical analysis (Wilcoxon, bootstrap CI, Phase-5
> ordinal gate at iso-tuned n=3, 2σ_iso vs 2σ_default comparison):
> [`paper/STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md) §10.
> Splice in PAPER.md: §5.5 iso-tuned sub-paragraph + §7.3 limitations +
> §7.4 future work item 12.

> ## ✅ PHASE-8 FINAL VERDICT (2026-05-29 AM, n=3) — THREE replicated cross-dataset positives (now superseded by n=7 certification above)
>
> The audit + Fixer + post-fix re-run + 3-seed CIFAR-100 graduation cycle
> is complete. **Three independent hypotheses cleared the worst-leader-seed
> > best-baseline-seed Phase-5 gate** on CIFAR-100 (30 ep, 3 seeds):
>
> | tag | C100 median | min seed | baseline max | Δ vs baseline median |
> |---|---:|---:|---:|---:|
> | `pair_gm_pdw` (H09+H48+H44 orthogonal stack) | **0.5786** | 0.5761 | 0.5662 | **+1.34 pp** |
> | `slot_act_sine` (H81 SIREN, single prior) | **0.5784** | 0.5766 | 0.5662 | **+1.32 pp** |
> | `sg_only_phi_budget` (H09, post-fix 1:1.623:2.629) | **0.5741** | 0.5687 | 0.5662 | **+0.89 pp** |
> | `baseline_resnet20` (rail) | 0.5652 | — | — | — |
>
> All three lead floors are strictly above the baseline ceiling →
> **leads are outside seed noise**. The post-fix H09 median (57.41 %) is
> ~0.6 pp LOWER than the pre-fix broken-architecture median (58.05 %),
> which is exactly what an honest audit predicts — the broken realised
> ratio happened to land a fortuitously-high seed-0 number. The corrected
> mechanism gives a more modest but statistically robust lead. *The
> 2026-05-29 PM n=7 extension subsequently elevates this from
> "screening-gate pass" to "formally certified at α=0.05 Holm-Bonferroni"
> — see the PROMOTION block above.*
>
> **The dual-track audit RATIFIED, not destroyed, the project**:
> - the audit caught a headline produced by broken code BEFORE publication
> - the Fixer campaign corrected 18 BROKEN/MAJOR findings with mechanism-
>   verifying tests
> - 3 of the 4 post-fix Phase-8 candidates cleared the C100 3-seed gate
> - `pair_gm_pdw` (the orthogonal-axis combo) is the project's first
>   experimentally-verified evidence of *prior compounding* — directly
>   refuting the H50 sg_full_fib catastrophic monolithic-hybrid result
> - H81 SIREN's matched performance shows that a SINGLE well-chosen
>   activation can be as effective as a 3-prior orthogonal stack
>
> Per CLAUDE.md [Rule 22](../CLAUDE.md#rule-22) (dual-track gate), all three winning hypotheses
> additionally satisfy: (a) impl-critic verdict ≠ MAJOR/BROKEN, and (b)
> sci-critic verdict ≠ NUMEROLOGY/UNFALSIFIABLE. The accuracy claims are
> defensible.
>
> Pre-AUDIT-NOTICE section below documents the pre-fix journey for
> archival completeness.

> ## ⚠ AUDIT NOTICE (2026-05-27)
>
> A dual-track audit (impl-critic + sci-critic, 8 + 8 parallel agents)
> + 8-agent Fixer campaign just completed. **Three previously-stated
> headlines in this document are NOW PROVISIONAL pending the post-fix
> re-run** (queued, ~4.5 GPU h):
>
> 1. **H09 phi_budget cross-dataset positive (CIFAR-10 85.54 % /
>    CIFAR-100 58.05 % 3-seed median, +1.53 pp).** The pre-fix
>    network's realised stage-parameter ratio was 1:1.41:2.45, NOT
>    the claimed 1:φ:φ²=1:1.618:2.618 — a 12.6 % drift. Fixer-PhiScaling
>    (commit `519cdf3`) corrected the allocator; post-fix realised
>    ratio is 1:1.623:2.629 (0.43 % max error). The architecture changed
>    (widths `[40,48,64]` → `[37,48,61]`). The 85.54 / 58.05 numbers
>    are NOT representative of the corrected network; the re-run is
>    mandatory before any external claim is restated ([Rule 21](../CLAUDE.md#rule-21)).
> 2. **H41 GoldenRatioAdamW falsification at 51.96 % top-1.** The
>    falsification was real but invalid as stated — `eps = 1/φ⁴ ≈ 0.146`
>    dominated Adam's denominator at CIFAR gradient scales, making the
>    effective LR ~6.85× nominal. The "β-only" experiment was never
>    cleanly run. Fixer-Opt (`8aa0430`) restored stock eps=1e-8 as
>    default; the β-only test runs as part of the post-fix campaign.
> 3. **H48 GoldenMomentumScheduler.** β1 saturated to the 1/φ² floor
>    after a single step — the "schedule" was one step long; CIFAR-100
>    Phase-5 distribution overlap demoted it to neutral; the audit also
>    found the implementation was non-monotonic. Fixer-Opt corrected to
>    T_max-aware `× φ^(-1/T_max)` per step.
>
> **Cumulative audit defect tally:** 3 BROKEN + 15 MAJOR + 24 MINOR
> impl-critic findings across 83 hypotheses (51 % non-PASS); 40
> NUMEROLOGY + 3 EMPIRICALLY-FALSIFIED + 2 UNFALSIFIABLE sci-critic
> verdicts across 81 hypotheses; ZERO NOVEL+TESTABLE sci verdicts on
> any implemented + smoked hypothesis (H71 IcosaRoPE3D is NOVEL+TESTABLE
> but has no CIFAR row). See [`AUDIT_SUMMARY.md`](../paper/AUDIT_SUMMARY.md) and
> [`audits/G{1..8}_audit.md`](audits/) for the full account.
>
> Until the post-fix re-run lands, **the project has NO defensible
> external accuracy claim**. The historical findings below remain as
> the pre-audit record; the post-fix verdict will be added as a new
> section after the orchestrator (`scripts/launch_postfix_campaign.sh`,
> task #88) completes.

> ## 🔍 METHODOLOGICAL CAVEAT — screening vs evaluation (2026-05-29)
>
> The 12-epoch CIFAR-10 single-config single-seed protocol used for
> every per-hypothesis row in this document is a **screening filter**,
> not an evaluation. Single-config single-seed numbers conflate
> "the hypothesis is bad" with "the hypothesis is bad *at our specific
> config*". Real evaluation requires per-hypothesis hill-climbing
> (Phase 9, ahead) across the natural knobs of each prior (β2 for
> H41, base_wd for H44, dropout floor for H47, T_max-step for H48,
> width-rounding for H09, etc.), 3-seed re-runs at the best config
> found, and then the Phase-5 worst-leader-seed > best-baseline-seed
> gate on CIFAR-100.
>
> The **only hypothesis-level claims that survive this distinction
> today are the three Phase-8 winners**, each of which IS **n=7
> CIFAR-100 evaluated and CERTIFIED at α=0.05 Holm-Bonferroni**
> (2026-05-29 PM promotion): `pair_gm_pdw`, `slot_act_sine`,
> `sg_only_phi_budget` (post-fix). Verdict tier: **CERTIFIED, n=7**.
> Every other hypothesis-level statement in this document — including
> "H41 falsified", "H42 PAPER_DISAGREES", "H44 borderline", "H50
> catastrophic", "H47/H48 WRONG_TEST", etc. — is reclassified as
> **SCREENING DATA**, not evaluation. The screening signal is real
> (a hypothesis that flunks at one config has *prima facie* evidence
> against it) but it is NOT yet a falsification at the level required
> for an external claim. The Phase-9 hill-climb skill (in flight) is
> the mechanism by which each screened-negative hypothesis gets a
> fair re-test at its own most-favourable config before any final
> verdict is recorded.

> **The pre-audit headline (historical, preserved verbatim):** at the
> 12-epoch / 127–272 k-parameter scale on CIFAR-10, the nature-inspired
> priors do *not* compound. The full hybrid (`sg_full_fib`, all six
> flags on) is the worst NaturePrior variant in the sweep, not the
> best. The autoresearch protocol delivered exactly what it is supposed
> to: a falsifiable negative result with clear single-prior
> decomposition. [Above audit notice supersedes the H09 / H41 / H48
> sub-claims within the expansion campaign + Phase-4 / Phase-5
> sections.]

## Expansion campaign — 2026-05-27 — 20 new hypotheses smoked (seed 0, 12 ep)

After the implementation campaign brought the repo from 7 to 74/75
hypothesis implementations, all 20 newly-wired CIFAR-10-droppable tags
were smoked at seed 0 / 12 epochs (zero failures, 8559 s total). The
result **sharpens, not overturns, the original headline**: of 20 new
single-change variants, exactly **one beats the ResNet-20 baseline on
top-1**, and **none beats it on composite** (the baseline's 272 k params
at 84.78 % remain the composite leader at 0.8458).

### The one positive: H09 φ-Proportion Parameter Budget

| tag | hyp | top-1 | params | composite | verdict |
|---|---|---|---|---|---|
| `sg_only_phi_budget` | H09 | **85.54 %** | 284 k | 0.8429 | **only variant > baseline top-1 (+0.76 pp)** |

`phi_budget` allocates the per-stage parameter budget in a `1 : φ : φ²`
ratio rather than the uniform doubling of a stock ResNet. It is the
**single strongest CIFAR-10 lead the project has produced** and the
clear #1 candidate to graduate to CIFAR-100 (Phase 4). The composite is
fractionally below baseline only because it carries 11 k more params;
on raw accuracy it wins. Pre-registered falsifier (top-1 ≤ baseline)
is **not** met → hypothesis survives.

### The clean falsification: H41 Golden-Ratio AdamW — REQUALIFIED 2026-05-29

> **CORRECTION (2026-05-29, from `audits/PAPER_GAP_G5.md`).** The
> "clean falsification" headline as originally stated was wrong: it
> conflated two simultaneous changes. The pre-fix
> `GoldenRatioAdamW` shipped both `eps = 1/φ⁴ ≈ 0.146` AND
> `β1 = 1/φ`, `β2 = 1/φ²` — and the catastrophic −33 pp collapse was
> driven **entirely by the eps change** (which dominated Adam's
> denominator at CIFAR gradient scales ~1e-3, pushing effective LR
> to ~6.85× nominal), NOT by the β change. After Fixer-Opt
> (commit `8aa0430`) restored stock `eps = 1e-8` and kept ONLY the
> β-shift, the post-fix CIFAR-10 12-ep top-1 is **0.8394** vs
> baseline 0.8478 → Δ ≈ **−1 pp**, not −33 pp. The Reddi 2018
> non-convergence prediction is asymptotic; it does NOT yet manifest
> at the 12-ep screening horizon. **Revised verdict: WEAKLY NEGATIVE
> at 12-ep CIFAR-10 screening; the β-only mild regression is
> consistent with Wilson 2017 ("default β2 ≈ 0.999 is empirically
> tuned, lower is monotonically worse but not pathologically so at
> short horizons"); a clean β-only falsification at the Reddi-2018
> non-convergence scale needs 100+-epoch training and is deferred to
> the Phase-9 hill-climb at varied β2.** The 0.5196 row is now
> historically labelled as the eps-confound; the canonical β-only
> H41 number is 0.8394.

| tag | hyp | pre-fix top-1 (eps+β confound) | post-fix top-1 (β-only) | verdict |
|---|---|---:|---:|---|
| `sg_only_golden_adam` | H41 | 51.96 % (eps=1/φ⁴) | **83.94 %** (eps=1e-8, β=φ-defaults) | WEAKLY NEGATIVE at 12 ep; Reddi-2018 β2 non-convergence prediction deferred to Phase-9 long-horizon sweep |

Setting Adam's `β1 = 1/φ ≈ 0.618`, `β2 = 1/φ² ≈ 0.382` sits far below
the empirically-stable regime (`0.9 / 0.999`) and the EMA time-constant
for the second moment becomes τ ≈ 1.6 steps. Reddi et al. 2018
(arXiv:1904.09237) prove Adam non-convergence in this regime
asymptotically; Choi 2019 (arXiv:1910.05446) and Wilson 2017
(arXiv:1705.08292) sweep β2 ∈ [0.95, 0.999] and find that range
dominates everywhere. At the 12-ep CIFAR-10 screening horizon with
cosine LR + AdamW decoupling and stock eps, the β-only regression is
only ~1 pp — the asymptotic non-convergence proof does not yet bite.
The original `eps = 1/φ⁴ ≈ 0.146` setting (which is what made the
pre-fix row catastrophic) inflated effective LR by ~6.85× and
saturated the optimiser; that was a confound, NOT the β claim.
**Revised verdict: WEAKLY NEGATIVE (screening), pending Phase-9
β2-sweep at longer training horizons before any final falsification
is recorded.** The "DISCARD as a universal recipe" framing remains
appropriate at the screening level, but the catastrophic-collapse
narrative was eps-driven and is retired.

### The efficiency story: H02 Fibonacci depth + the φ-channel family

| tag | hyp | top-1 | params | composite |
|---|---|---|---|---|
| `sg_only_fib_depth` | H02 | 82.18 % | **180 k** | 0.8261 |
| `sg_only_golden_resize` | H03 | 80.67 % | 127 k | 0.8157 |
| `sg_only_phi_compound` | H01 | 80.42 % | 127 k | 0.8152 |

`fib_depth` reaches the #3 composite slot at **0.66× the baseline
parameter count** — the priors that touch *scaling* (depth schedule,
channel schedule) are consistently the most parameter-efficient family,
even when raw top-1 trails. Worth a CIFAR-100 look as the "small-model"
graduate.

### Mid-pack (neutral, within ±2 pp of the no-prior scaffold)

`golden_momentum` (83.52), `phi_dropout` (82.80), `phi_multiscale`
(82.00), `golden_skip` (81.63), `fib_prune` (81.15), `golden_spiral_init`
(80.42), `fib_ensemble` (80.11), `phi_activation` (79.95), `phi_decay`
(79.81), `phi_lr` (78.75) — all land in the neutral band: a real
implementation, no reliable lift at 12 ep. These are "needs-more-epochs
or needs-the-right-dataset" rather than falsified.

### New strong negatives (join group/cymatic/toroidal)

`golden_adam` (51.96 pre-fix / **83.94 post-fix β-only**, see correction above — the −33pp was eps-confound, the β-only Δ is only ~−1 pp at 12 ep), `group_avg` (65.38), `golden_bottleneck`
(69.25, but at **0.21× params** — efficiency outlier worth a second
look), `phi_relu` (71.07), `fib_stride` (72.55), `phi_sparse` (73.33),
`phi_init` (76.56). The φ-init and sparse-connectivity priors actively
hurt at this scale.

### G8 esoteric-extension tags (smoked 2026-05-27)

| tag | hyp | top-1 | params | composite | verdict |
|---|---|---|---|---|---|
| `sg_only_sine_act` | H81 SIREN sin(ωx) | 80.62 % | 127 k | 0.8197 | neutral — lands mid-pack, beats every φ-channel variant; ω=1 near-identity start trains stably |
| `sg_only_constant_width` | H80 Reuleaux | 75.95 % | 127 k | 0.7629 | mild negative — constant-width mask is too aggressive on 3×3 convs (drops ~4 of 9 taps), losing kernel capacity the isotropy gain doesn't recover at this scale |

`sine_act` is the better-behaved of the two and could merit a second
row at the SIREN-canonical `ω=30` first-layer recipe. The other 7 G8
modules (radial-12 / toroidal-latent / morphing-graph / voronoi-attn /
collapse-attn / spectral-hopfield) ship as standalone primitives
without a CNN sweep row, matching the G2/G4/G7 convention.

### Phase-4 CIFAR-100 graduation shortlist (by this campaign)

1. **H09 `phi_budget`** — only top-1 winner; clear #1.
2. **H02 `fib_depth`** — best param-efficiency (comp 0.8261 @ 180 k).
3. **H05 `fractal`** — the original campaign's only positive single prior.
4. **H48 `golden_momentum`** — closest neutral to baseline (83.52).
5. `baseline_resnet20` rail (always carried).

## Phase 4 — CIFAR-100 graduation results (2026-05-27, seed 0, 30 ep)

The 5-model shortlist trained on CIFAR-100 at 30 epochs (zero failures,
4939 s total; `baseline_resnet20` ran first as the Rule-13 pre-flight and
landed in-band at 56.15 % / 84.14 % top-5).

| rank | tag | hyp | top-1 | top-5 | composite | params | Δ top-1 vs baseline |
|---|---|---|---|---|---|---|---|
| 1 | `sg_only_phi_budget` | H09 | **58.05 %** | 85.37 % | **0.5815** | 289 k | **+1.90 pp** |
| 2 | `sg_only_golden_momentum` | H48 | **56.99 %** | 84.79 % | 0.5676 | 278 k | **+0.84 pp** |
| 3 | `baseline_resnet20` | — | 56.15 % | 84.14 % | 0.5568 | 278 k | (ref) |
| 4 | `sg_only_fib_depth` | H02 | 51.78 % | 81.44 % | 0.5187 | **184 k** | -4.37 pp (0.66× params) |
| 5 | `sg_only_fractal` | H05 | 50.72 % | 81.09 % | 0.4881 | 263 k | -5.43 pp |

**The cross-dataset verdict — H09 φ-Proportion Parameter Budget is a
genuine, replicated positive.** It is the **only hypothesis that beats the
ResNet-20 baseline on BOTH CIFAR-10 (85.54 % vs 84.78 %) and CIFAR-100
(58.05 % vs 56.15 %)**. Allocating per-stage parameters in a `1 : φ : φ²`
ratio rather than the uniform doubling of a stock ResNet consistently helps
at this scale — the single most defensible result the project has produced.
Pre-registered falsifier (≤ baseline on the harder dataset) is **not met**
→ hypothesis survives a second, independent test.

**H48 Golden-Momentum scheduler also graduates.** Neutral on CIFAR-10
(83.52, -1.26 pp) but **+0.84 pp on CIFAR-100** — a φ-decayed β1 schedule
appears to help more as task difficulty rises. Promoted to Phase 5.

**H02 fib_depth / H05 fractal do NOT graduate on raw accuracy** (-4.4 /
-5.4 pp). `fib_depth` retains its efficiency narrative (0.66× params) but
the accuracy cost is real on CIFAR-100; both drop out of the headline track.

**Phase-5 (3-seed error bars) shortlist:** `phi_budget`, `golden_momentum`,
+ `baseline_resnet20` rail. No external accuracy claim will be made until
the 3-seed median ± spread confirms the phi_budget lead is outside noise.

## Phase 5 — 3-seed CIFAR-100 error bars (2026-05-27, 30 ep)

| tag | s0 | s1 | s2 | **median** | mean | std | range |
|---|---|---|---|---|---|---|---|
| `baseline_resnet20` | 56.15 | 56.52 | 56.62 | **56.52** | 56.43 | 0.20 | 0.47 |
| **`sg_only_phi_budget`** | 58.05 | 58.63 | 57.00 | **58.05** | 57.89 | 0.67 | **1.63** |
| `sg_only_golden_momentum` | 56.99 | 56.76 | 56.43 | **56.76** | 56.73 | 0.23 | 0.56 |

### Decisive seed-noise test (Phase-5 gate before any external claim)

The protocol's gate: **does the lead survive the seed-noise envelope?**
Specifically, is the leader's WORST seed still above the baseline's BEST
seed? (A weaker formulation: do the per-tag {min, max} intervals fail to
overlap?)

- **`phi_budget` min = 57.00 % > `baseline_resnet20` max = 56.62 %**.
  Even the worst phi_budget seed beats the best baseline seed by +0.38 pp.
  Median advantage: **+1.53 pp** (58.05 vs 56.52). Mean advantage: +1.46 pp.
  Phi_budget's seed-std (0.67) is larger than baseline's (0.20) — the
  φ-allocation amplifies the seed-to-seed spread — but the floor still
  clears the baseline ceiling. **The lead is outside seed noise. ✓**
- **`golden_momentum` min = 56.43 % < `baseline_resnet20` max = 56.62 %**.
  The seed distributions overlap; the Phase-4 +0.84 pp seed-0 advantage
  shrinks to **+0.24 pp at the median** and the worst momentum seed
  underperforms two of three baseline seeds. **Within noise. Demoted.**

### Final verdicts

- **H09 φ-Proportion Parameter Budget — SURVIVES Phase 5.** The only
  prior in the project that beats the ResNet-20 baseline on *both* CIFAR-10
  (85.54 vs 84.78, single seed) *and* CIFAR-100 (58.05 vs 56.52, 3-seed
  median, lead outside noise). The "1 : φ : φ²" per-stage parameter
  allocation is a real, replicated, seed-robust positive at this scale.
  This is the project's first defensible single-prior win.
- **H48 Golden-Momentum Scheduler — DEMOTED.** Phase-4 seed-0 win was
  within seed noise. Recorded as neutral; further investigation would need
  longer training or a different scheduler interaction to be revisited.
- **All other hypotheses tested at Phase-5 depth:** unchanged from earlier
  verdicts (H02 fib_depth efficiency-only, H05 fractal neutral-positive on
  C10 only, H41 golden_adam falsified [REQUALIFIED 2026-05-29 — see
  correction above: post-fix β-only top-1 is 0.8394, not 0.5196; the
  −33pp was eps-driven; H41 is now WEAKLY NEGATIVE at 12-ep screening
  rather than a clean catastrophic falsification, pending Phase-9
  long-horizon β2 sweep], etc.).

### What Phase 5 proved

The 6-phase autoresearch pipeline (unit tests → SOTA smoke → 35-tag C10
smoke → top-K selection → C100 graduation → 3-seed error bars) ran
end-to-end through 35 + 5 + 6 = **46 GPU runs** spanning two datasets and
produced exactly one externally defensible accuracy claim plus several
documented falsifications. **The protocol is the deliverable** — and it
worked exactly as designed: most "golden-ratio fixes everything" claims
died (H41 catastrophically [REQUALIFIED 2026-05-29: the catastrophic
collapse was eps-confound; β-only post-fix is mild −1 pp at 12 ep —
clean β-only falsification deferred to Phase-9], H48 quietly), and the one survivor was made
to prove it twice on independent data and three times on independent
seeds before earning the headline.

---

## Phase 7 — Pre-fix vs post-fix comparison (24 of 31 tags, 2026-05-28)

The audit + Fixer campaign (Rules 21, 22) re-ran every previously
landed single-axis and combo tag against the corrected hypothesis
modules. The pre-fix numbers come from commit `282cddb` ("Pre-fix
combo ladder complete (7 rows, broken phi_budget + 1-step-saturating
golden_momentum)"); the post-fix numbers are the current working
tree under `experiments/cifar10/<tag>_seed0/metrics.json`. The 10
Tier-A LOO / PAIR tags did NOT exist pre-fix — they were authored as
the corrected ladder's importance-of-removal probes. Survival
buckets: **STRENGTHENED** post − pre > +0.20 pp on top-1,
**STABLE** within ±0.20 pp, **WEAKENED** post − pre < −0.20 pp,
**EMERGED** new tag with post-fix > baseline 84.78 %,
**NEW (negative)** new tag with post-fix < baseline. All numbers are
CIFAR-10 top-1, 12-epoch, seed 0.

| tag | hyp | pre-fix top1 | post-fix top1 | Δ (pp) | survives? |
|---|---|---:|---:|---:|---|
| `sg_only_phi_budget` | H09 | 85.54 | 85.56 | +0.02 | STABLE |
| `sg_only_golden_bottleneck` | H10 | 69.25 | 68.79 | −0.46 | WEAKENED |
| `sg_only_cymatic_init` | H29 | 77.44 | 77.64 | +0.20 | STABLE |
| `sg_only_golden_spiral_init` | H30 | 80.42 | 80.57 | +0.15 | STABLE |
| `sg_only_golden_adam` | H41 | 51.96 | 83.94 | **+31.98** | STRENGTHENED |
| `sg_only_phi_dropout` | H44 | 82.80 | 83.03 | +0.23 | STRENGTHENED |
| `sg_only_golden_momentum` | H48 | 83.52 | 83.65 | +0.13 | STABLE |
| `combo2_pb_gm` | C2 | 86.14 | 85.62 | −0.52 | WEAKENED |
| `combo3_pb_gm_pd` | C3 | 86.42 | 85.66 | −0.76 | WEAKENED |
| `combo4_pb_gm_pd_pdw` | C4 | 85.80 | 85.44 | −0.36 | WEAKENED |
| `combo5_pb_gm_pd_pdw_plr` | C5 | 81.59 | 79.78 | −1.81 | WEAKENED |
| `combo6_pb_gm_pd_pdw_plr_fe` | C6 | 85.23 | 84.90 | −0.33 | WEAKENED |
| `combo7_pb_gm_pd_pdw_plr_fe_sa` | C7 | 85.05 | 85.29 | +0.24 | STRENGTHENED |
| `combo8_pb_gm_pd_pdw_plr_fe_sa_fp` | C8 | 84.08 | 84.96 | +0.88 | STRENGTHENED |
| `loo_no_gm` | LOO−gm | N/A — new tag | 84.27 | — | NEW (negative) |
| `loo_no_pd` | LOO−pd | N/A — new tag | 85.37 | — | EMERGED |
| `loo_no_pdw` | LOO−pdw | N/A — new tag | 84.97 | — | EMERGED |
| `loo_no_plr` | LOO−plr | N/A — new tag | 83.83 | — | NEW (negative) |
| `loo_no_fe` | LOO−fe | N/A — new tag | 84.03 | — | NEW (negative) |
| `loo_no_sa` | LOO−sa | N/A — new tag | 84.82 | — | EMERGED |
| `loo_no_fp` | LOO−fp | N/A — new tag | 85.29 | — | EMERGED |
| `pair_gm_pdw` | P(gm,pdw) | N/A — new tag | **85.85** | — | EMERGED |
| `pair_gm_plr` | P(gm,plr) | N/A — new tag | 80.36 | — | NEW (negative) |
| `pair_pd_pdw` | P(pd,pdw) | N/A — new tag | 85.24 | — | EMERGED |

### Survival counts

- **STRENGTHENED:** 4 (`golden_adam`, `phi_dropout`, `combo7`, `combo8`)
- **STABLE:** 4 (`phi_budget`, `cymatic_init`, `golden_spiral_init`,
  `golden_momentum`)
- **WEAKENED:** 6 (`golden_bottleneck`, `combo2`, `combo3`, `combo4`,
  `combo5`, `combo6`)
- **EMERGED:** 6 (`pair_gm_pdw`, `loo_no_pd`, `loo_no_fp`,
  `loo_no_pdw`, `loo_no_sa`, `pair_pd_pdw`)
- **NEW (negative):** 4 (`loo_no_plr`, `loo_no_fe`, `loo_no_gm`,
  `pair_gm_plr`)

### Analysis

**H09 phi_budget survives the corrected mechanism.** The single-axis
post-fix number (85.56 %, +0.02 pp Δ) sits squarely on top of the
pre-fix headline (85.54 %), confirming that the **mechanism fix did
not erase the gain**. The Phase-5 verdict — phi_budget as the
project's first defensible single-prior win, replicated on CIFAR-100
across three seeds — remains intact. The corrected "1 : φ : φ²"
per-stage allocation is reproducible, and the small param drop
(283.6 k → 262.1 k) under the corrected width-rounding actually
improves the param-efficient composite slightly.

**H48 golden_momentum and H44 phi_dropout still help — but only
inside the right combo.** golden_momentum's solo post-fix number
moved from 83.52 → 83.65 (within noise), still −1.91 pp below the
baseline; the fix saved its mechanism without rescuing the prior on
its own. Stacked onto phi_budget, the combo2 row weakened slightly
(86.14 → 85.62) because the pre-fix combo2 inherited an unsupported
boost from the broken phi_budget initialisation. The **best stack in
the entire post-fix sweep is `pair_gm_pdw` at 85.85 %** (phi_budget +
golden_momentum + phi_dropout-weight) — beating every single prior
including phi_budget alone by +0.29 pp and beating the corrected
combo2 by +0.23 pp. phi_dropout solo moved STRENGTHENED (82.80 →
83.03) and shows up in two of the four EMERGED LOO/pair rows.

**Combo ladder post-fix: monotone-add is dead, the *axis* matters.**
The pre-fix combo ladder showed monotone gain from C2 → C3 (86.14 →
86.42) and a cliff at C5 (`+plr`, −4.83 pp). Post-fix the C2-to-C4
plateau is gone (85.62, 85.66, 85.44) and the C5 cliff is even
deeper post-fix (−5.88 pp from C4). C7 and C8 STRENGTHENED slightly,
meaning that once `plr` has already collapsed accuracy, adding
self-attention (`sa`) and feature-pyramid (`fp`) repairs some of the
damage. The pattern: **`plr` (phi-LR-schedule) is the single most
destructive axis** and a `fe` (fib-ensemble) on its own is also
net-negative; `gm`, `pd`, `pdw`, `sa`, `fp` are all neutral-to-mildly
positive in the right context.

**Tier-A LOO/PAIR signal — which removal hurts the most?** Reading
`combo8 = 84.96 %` and the seven LOO rows: removing `gm` drops to
84.27 (−0.69 pp), removing `plr` drops to 83.83 (−1.13 pp),
removing `fe` drops to 84.03 (−0.93 pp). Removing `pd`, `pdw`, `sa`,
`fp` all sit at-or-above 84.82 (no harm to mild improvement, i.e.
those four axes are net-zero inside combo8). **The most-damaging
removal is `plr` (−1.13 pp), followed by `fe` (−0.93 pp) and `gm`
(−0.69 pp).** Counterintuitively, although `plr` was the **most
destructive axis** when *added* to the C4 stack, once the network
has compensated for it through C7 + C8, **removing it costs the
most** — the surrounding stack has reorganised around its presence.

**Provisional Phase-8 winners.** Two tags clear both gates (post-fix
top-1 > baseline 84.78 % AND best-of-class for their tier):

1. **`pair_gm_pdw` — 85.85 %** (best single combo in the post-fix
   sweep, +1.07 pp over baseline, beats every solo prior). Three
   orthogonal axes: phi_budget (architecture) + golden_momentum
   (optimiser) + phi_dropout-weight-decay (regulariser). Promote to
   3-seed CIFAR-100 Phase-8.
2. **`sg_only_phi_budget` — 85.56 %** (best single-prior post-fix,
   already cleared Phase-5 on CIFAR-100). Continues as the
   single-axis control for the 3-seed CIFAR-100 head-to-head.

`combo3_pb_gm_pd` (85.66 %) is a candidate alternative if `pair_gm_pdw`
fails the CIFAR-100 seed-noise gate; both share gm + a phi_dropout
variant and differ only in the weight-decay vs dropout coupling.

---

## Final ranking (composite descending, single seed) — original 11-row campaign

| rank | tag | top-1 | params | latency ms | composite | Δ vs `sg_chan_fib` |
|---|---|---|---|---|---|---|
| 1 | `baseline_resnet20` | 84.78 % | 272 k | 4.03 | **0.8458** | (different scaffold) |
| 2 | `baseline_sg_vanilla` | 82.16 % | 186 k | 4.42 | 0.8258 | +0.0123 |
| 3 | `sg_chan_phi` | 80.11 % | 127 k | 4.11 | 0.8152 | +0.0017 |
| 4 | `sg_chan_fib` | 80.11 % | 127 k | 4.43 | **0.8135** | — (reference) |
| 5 | `sg_only_fractal` | 82.46 % | 259 k | 7.42 | 0.8104 | -0.0031 |
| 6 | `sg_only_golden_modulate` | 79.81 % | 127 k | 5.95 | 0.8042 | -0.0093 |
| 7 | `sg_only_hex` | 79.32 % | 127 k | 7.55 | 0.7941 | -0.0194 |
| 8 | `sg_only_cymatic_init` | 77.44 % | 127 k | 4.14 | 0.7883 | -0.0252 |
| 9 | `sg_only_toroidal` | 78.05 % | 127 k | 9.34 | 0.7768 | -0.0367 |
| 10 | `sg_only_group` | 69.84 % | 127 k | 9.75 | **0.6937** | -0.1198 |
| 11 | `sg_full_fib` | 73.24 % | 259 k | 20.02 | **0.6966** | -0.1169 |

## Single-prior decomposition vs the `sg_chan_fib` reference

| prior on alone | Δ top-1 | Δ latency (ms) | Δ rot-eq err | verdict |
|---|---|---|---|---|
| `hex` | **-0.79 pp** | +3.12 (1.7×) | -0.022 | mild negative — mask overhead exceeds isotropy gain at this scale |
| `group` (C4) | **-10.27 pp** | +5.32 (2.2×) | **-0.046** (large) | strong negative on top-1, but rot-eq drops as theory predicts — equivariance prior is unused by the data |
| `fractal` | **+2.35 pp** | +2.99 (1.7×) | -0.036 | **only single prior that lifts top-1**; pays in 2× params |
| `toroidal` | **-2.06 pp** | +4.91 (2.1×) | -0.025 | negative as predicted — CIFAR images do not wrap |
| `cymatic_init` | **-2.67 pp** | -0.29 (≈) | -0.011 | unexpected negative — Chladni-mode init hurt rather than helped |
| `golden_modulate` | **-0.30 pp** | +1.52 (1.3×) | -0.026 | near no-op as predicted |

## The compound failure

`sg_full_fib` (all six priors ON simultaneously):

- **top-1: 73.24 %** (-6.87 pp below `sg_chan_fib` reference; -11.54 pp
  below `baseline_resnet20`)
- **latency: 20.02 ms** (5× the baseline)
- **composite: 0.6966** (worst in sweep, tied with `sg_only_group`)

## H58 follow-up — the avg-pool fix DISCARDED

The first campaign's top-priority follow-up was **H58**: replace the C4
group convolution's max-pool reduction with mean-pool. Pre-registered
prediction: the +5 to +10 pp top-1 recovery on `sg_only_group` would
prove that "max-pool over the 4-rotation orbit throws away 75 % of the
signal." The 2-row sweep (`sg_only_group_avg`, `sg_full_fib_avg`)
trained on 2026-05-27 with checkpoint saving enabled returned the
following:

| variant | reduce | top-1 | params | composite | Δ vs same-arch max |
|---|---|---|---|---|---|
| `sg_only_group` | max | 69.84 % | 127 k | 0.6937 | (ref) |
| `sg_only_group_avg` | mean | **65.38 %** | 127 k | **0.6597** | **-4.46 pp** |
| `sg_full_fib` | max | 73.24 % | 259 k | 0.6966 | (ref) |
| `sg_full_fib_avg` | mean | **66.86 %** | 259 k | **0.6432** | **-6.38 pp** |

**Verdict: DISCARD.** Mean-pool over the orbit *hurts* worse than
max-pool — the opposite of the prediction. The "discards 75 % signal"
intuition was wrong: max-pool over rotated copies actually preserves
the strongest evidence at every spatial location (a soft argmax over
orientations), while mean-pool *dilutes* discriminative features by
averaging out the response-vs-orientation mismatch. The autoresearch
protocol delivered exactly what it should: a clean falsifiable
negative against a confidently-stated hypothesis.

**New direction for H24 (Platonic equivariance):** the fix is not the
reduction operator but the data. C4-equivariant features are useful on
data with rotational variance (rotated CIFAR, IcoMNIST, spherical
MNIST) — not on canonically-oriented CIFAR-10. The next experiment
should test the *same* `sg_only_group` (max-pool) variant on
rotated-CIFAR-10 where the equivariance prior is data-aligned.

## Trained-feature Betti (first data)

The H58 sweep produced the first `best.pt` checkpoints, enabling
trained-feature Betti curves for the first time. Two interesting
observations:

| variant | β₀ per stage (trained) | β₀ per stage (fresh-init) |
|---|---|---|
| `sg_only_group_avg` | [121, 127, 127, 127] | [93, 71, 122, 100] |
| `sg_full_fib_avg` | [123, 124, 127, 127] | [123, 124, 127, 127]* |

Trained models *increase* β₀ relative to fresh init — they cluster
features by class, producing more isolated connected components at the
relative-threshold band. β₁ also rises (one 1-D hole detected in
`sg_full_fib_avg`), consistent with class-prototype loop structures.
This inverts the naive "topology simplification" reading and motivates
a follow-up where Betti is tracked **per epoch** to see whether the
β-curves cross the random baseline at convergence.

The naive additive-deltas prediction (sum of single-prior Δs ≈ -13 pp
top-1) would put `sg_full_fib` at ≈ 67 %; observed 73.24 % is actually
*better* than the additive prediction, suggesting some priors do
recover *some* loss when combined — but the net is still strongly
negative. The hex+group+toroidal latency stack (each adds 1.7–2.2×)
multiplies into 5× total latency.

## What this means

1. **The source PDF's 20–50 % efficiency claim does not survive a
   12-epoch CIFAR-10 ablation on this block.** Either the priors
   need a different composition (not all six at once on a single
   block), more training (12 epochs is short), or a different
   dataset where the priors' inductive biases match the data
   geometry.

2. **C4 group conv is the dominant negative term.** With C4
   max-pool over the 4-rotation orbit, we are throwing away 75 %
   of the signal at every layer. The rotation-equivariance gain
   (rot-eq err drops by 0.046) is real, but CIFAR-10 does not
   reward it because the test images are not rotated.

3. **Fractal recursion is the only single prior that lifts top-1.**
   At 2× parameter cost the +2.35 pp lift is hard to justify, but
   it is the one prior that materially helps representation
   quality at this scale.

4. **`baseline_sg_vanilla` (the NaturePriorBlock scaffold with all
   priors OFF and linear channels) is competitive.** It sits at
   82.16 % top-1 with 186 k params vs ResNet-20's 272 k — i.e.,
   the NaturePriorBlock scaffold itself (BN placement, skip-connection
   structure) is fine. The priors do most of the harm.

## Open axes for a follow-up campaign

- [ ] Run with `--seeds 0 1 2` to get error bars; the deltas above
      are single-seed.
- [ ] Run `sg_loo_no_*` leave-one-out from the full hybrid
      (`scripts/run_sweep.py --full`) to identify which single
      prior is responsible for the most damage when combined.
- [ ] Test the full hybrid on a dataset where the priors are
      data-aligned: e.g., **rotated CIFAR-10** (group conv should
      now help) or **spherical / IcoMNIST** (Platonic equivariance
      should pay off).
- [ ] Re-run sg_only_group with **average pooling** over the orbit
      instead of max-pool — max-pool's 75 % signal loss may be the
      single biggest source of the C4 damage.
- [ ] Save `best.pt` checkpoints (already wired) and re-run
      `compute_topology.py` for **trained-feature** Betti curves;
      the current β-curves are on fresh-init features.
- [ ] Scale to MedMNIST PathMNIST (data loader already in
      `src/nature_inspired_networks/data.py`).

## Reproduce

```powershell
$env:SSL_CERT_FILE = ".\.venv\Lib\site-packages\certifi\cacert.pem"
.\.venv\Scripts\python -u scripts\run_sweep.py `
   --config configs\cifar10_quick.yaml --seeds 0 --skip-existing
.\.venv\Scripts\python scripts\compute_topology.py --seeds 0
.\.venv\Scripts\python scripts\build_dashboard.py
.\.venv\Scripts\python scripts\build_report.py
start dashboard\dashboard.html
```

Total wall-clock for the 11-run sweep on RTX 4090 Laptop (16 GB):
**3853 s = 64 min** (single seed, 12 epochs each).
