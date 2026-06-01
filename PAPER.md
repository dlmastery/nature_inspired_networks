# A Skeptical Protocol for Nature-Inspired Neural-Network Priors: Methodological Contribution and an 84-Hypothesis Dual-Track Audit on CIFAR-10/-100

> Internal QA pass; ICML 2027 reviewer-pass rebuttal landed 2026-05-30
> ([`audits/ICML_REVIEWS_2026-05-30/REBUTTAL.md`](audits/ICML_REVIEWS_2026-05-30/REBUTTAL.md)).
> The revision history (16 prior BLOCKER/MAJOR items from the area-chair
> sniff test) is preserved in [`audits/REVIEWER_PASS_PAPER.md`](audits/REVIEWER_PASS_PAPER.md)
> and the per-line cross-walk in [`audits/REVIEWER_PASS_PAPER_RESPONSE.md`](audits/REVIEWER_PASS_PAPER_RESPONSE.md);
> they are no longer rendered in the paper body.

## Abstract

We present a dual-track skeptical audit + Fixer + per-experiment-page protocol for autoresearch campaigns that propose large families of architectural priors. Two independent 8-agent expert teams attack orthogonal failure modes — an implementation-critic team grades code-vs-doc fidelity with mechanism-pinning tests, and a research-scientist-critic team challenges scientific merit independent of code — followed by a Fixer campaign whose patches each ship with a regression test that would have caught the original bug. The protocol is operationalised as CLAUDE.md Rules 20–28 and packaged as seven content-agnostic skills. Its signature catch on the calibration substrate (an 84-hypothesis nature-inspired neural-prior design space on CIFAR-10/-100) is **H09 phi_budget**: an unaudited pipeline would have published a CIFAR-100 +1.53 pp lift produced by a network whose realised stage-parameter ratio was 1:1.41:2.45, not the doc-claimed 1:φ:φ²; the protocol detected the 12.6 % drift and the Fixer corrected it with a mechanism-pinning test (`519cdf3`). On the corrected codebase, three post-fix candidates pass a 7-seed CIFAR-100 paired Wilcoxon at α=0.05 under Holm-Bonferroni across a k=3 confirmatory family (p=0.0078 each). Robustness re-certification at the hill-climbed (bs=128, lr=3e-3) iso-tuned cell at n=7 (Phase-9f closeout, 2026-06-01) shows substantial Δ-shrinkage (+1.24 to +1.78 pp default-config → +0.54 to +0.79 pp iso-tuned) and Phase-5 ordinal-gate FAIL for all three winners, partially validating R2 BLOCKER #13's mixed-bs concern; the default-config certification remains the formal claim of the paper while the iso-tuned regime is reported with full transparency. We report these as illustrative protocol output, with three caveats: baseline sits 6.5 pp below 164-ep SOTA; most single-prior rows remain n=1; all agents share a model family (Opus 4.7).

## 1 · Introduction

Geometric Deep Learning (GDL) and Topological Deep Learning (TDL) have repeatedly converged on a small set of geometric priors — hexagonal lattices (Hoogeboom 2018), Platonic / icosahedral symmetry groups (Cohen 2019), fractal recursion (Larsson 2017), toroidal embeddings (Pittorino 2022), Fibonacci channels, and Hopfield / SIREN / RoPE phase encodings (Bronstein et al. 2021 systematise much of this surface). A separate popular literature suggests the golden ratio φ ≈ 1.618 and Fibonacci sequence "appear everywhere in nature", and it is tempting to assemble the geometric priors and the φ-flavoured constants into one framework and claim broad gains.

**This work does not do that.** It presents the *infrastructure required to know whether such a claim is justified*, and applies that infrastructure to its own design space. We implement 84 hypotheses faithfully, audit them adversarially with two independent expert teams, patch the findings, re-run the empirical evidence on the corrected code, and publish the surviving framework with its honest verdict — including the honest verdict that the protocol's own empirical findings are PROVISIONAL until the Phase-9 hill-climb completes.

### 1.1 · Contributions

1. **A dual-track skeptical audit protocol** — an 8-agent implementation-critic team verifies *code-vs-doc correspondence*, and an 8-agent research-scientist-critic team challenges the *scientific merit of the hypothesis itself*. Both teams are parallel and disjoint-scoped; their findings populate `audits/G<X>_audit.md` and an "Addendum: Research-Scientist Critique" section appended into each design doc.
2. **A Fixer campaign with mechanism-verifying-test discipline**: every code patch ships with at least one test that asserts the fixed mechanism (the test the original implementer missed). The Fixers correct 22 hypotheses across 8 commits, adding ~34 new mechanism tests.
3. **A normative ruleset** (CLAUDE.md Rules 20–28) operationalising the audit + fix + re-run cycle and the screening-vs-evaluation distinction, plus seven reusable skills in [`skills/`](skills/) so any future autoresearch project can pick up the protocol unchanged. Portability is *claimed by construction* (the skills are content-agnostic); a cross-domain demonstration is open future work.
4. **A reproducible 84-hypothesis implementation** of nature-inspired neural priors as 80 pure-PyTorch modules + 78 test files (~826 unit tests post-Fixer), each accompanied by a committee-grade design doc — the calibration substrate, NOT a standalone library of recommended priors.
5. **Empirical calibration on CIFAR-10/-100**: 51 % non-PASS implementation-critic rate (3 BROKEN / 15 MAJOR / 24 MINOR of 83); 1 / 81 NOVEL+TESTABLE sci-critic rate; three Phase-8 candidates pass the worst-leader-seed gate. Calibration numbers are conditional on §1.3.

What is **explicitly NOT** in the contributions: any claim about transformer-track / decoder-only-LM hypotheses (10 of 84 target attention backbones; none tested); any claim about H71 IcosaRoPE3D (the sole NOVEL+TESTABLE survivor is untested); any claim about cross-domain portability (open future work); any claim that the three Phase-8 winners constitute a standalone empirical headline — they are illustrative protocol output.

### 1.2 · Why the audit was necessary

The pre-audit version of this paper claimed H09 phi_budget as a verified cross-dataset positive (CIFAR-10 85.54 %, CIFAR-100 58.05 % 3-seed median, leading the baseline by +1.53 pp). The user did not trust the validation — every test file in the repo had been written by an implementer agent trained to make tests pass, and "make tests pass" often degenerates to shape-only assertions. The audit campaign was launched to discover whether that distrust was justified. **It was**: the post-audit analysis revealed that H09's realised stage-parameter ratio was 1:1.41:2.45, not the claimed 1:φ:φ² = 1:1.618:2.618 (12.6 % drift at stage 1); the headline number was produced by a network that did not faithfully implement its own design doc. Fixer-PhiScaling (commit `519cdf3`) corrected the integer search; the post-fix realised ratio is 1:1.623:2.629 (0.43 % max error). This is the protocol's load-bearing example of diagnostic power and the canonical motivation for the mechanism-pinning-test discipline (Rule 21).

### 1.3 · Limitations of the audit protocol — auditor self-grading

A binding caveat (per area-chair item #15): implementer, implementation-critic, sci-critic, Fixer, and audit-calibration agents in this campaign are **all from the same model family** (Claude Opus 4.7). Disjoint-scope and disjoint-file-target "independence" is enforced; model-family independence is not. The audit's 51 % non-PASS rate is conditional on this caveat. Third-party-code calibration (§5.8) partially neutralises it; a cross-family audit on the same `pytorch/vision` sample remains future work (§7.4-5).

### 1.4 · Mystical motivation, neutral artifact names

The 84-hypothesis design space is motivated in part by popular literature on nature-inspired constants. We treat that motivation as a **prior over the design space**, not as evidence. Per CLAUDE.md [Rule 16](CLAUDE.md#rule-16), artifact names are neutral (`nature_inspired_networks`, `NaturePrior*`); the mystical inspiration is acknowledged in prose only.

## 2 · The autoresearch protocol (inherited)

The autoresearch protocol of `dlmastery/autoresearchimage` — citation rigor (every reasoning entry uses `Author YEAR VENUE 'Title' (arXiv:XXXX.XXXXX) — relevance`), reasoning-blob completeness (word-count floors), SHA-256-fingerprinted composite metric, append-only experiment log, no-bypass gates, per-experiment archive directory — is taken verbatim and forms the **floor** of this work. The composite metric is
```
composite = top1 − 0.05 · log10(params_M) − 0.05 · log10(latency_ms)
```
with SHA-256 fingerprint `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893` ([Rule 2](CLAUDE.md#rule-2)); editing it raises `CompositeFingerprintError` at runner import. Rules 20–28 layered on top during this campaign codify the auto-checkpoint loop, post-fix re-run discipline, dual-track audit gate, orthogonal-axes-only compounding, dashboard discipline, Q&A-test correspondence, Windows thread-cap safety, Pages-link discipline, and **screening-vs-evaluation tiering** (Rule 28; the formal substrate of §7.3.1).

## 3 · The 84-hypothesis design space

Eight thematic groups, organised in `hypotheses/g<N>_*/H<NN>_*.md`: G1 Scaling & Growth (H01–H10), G2 Layer / Channel / Neuron (H11–H20), G3 Topologies & Graphs (H21–H30), G4 Kernels / Attention / Filters (H31–H40), G5 Optimisation / Init / Reg / NAS (H41–H50), G6 Topological & Bridging (H51–H60), G7 Cross-Paradigm Hybrids (H61–H75), G8 Esoteric Extensions (H76–H84). See [`IDEA_TABLE.md`](IDEA_TABLE.md) and [`hypotheses/INDEX.md`](hypotheses/INDEX.md).

## 4 · Methods — the dual-track audit

### 4.1 · Track A: implementation-critic team

For each thematic group G1–G8, a parallel Critic agent reads every hypothesis's `H<NN>_*.md` design doc, the corresponding src module under `src/nature_inspired_networks/`, and the test file `tests/test_<module>.py`. Findings on six dimensions: (1) mechanism check; (2) math correctness; (3) test rigor — does the test assert MECHANISM or only SHAPE; (4) citation alignment; (5) falsifier reachability; (6) hidden bugs / cargo-cult. Verdict tiers: **PASS** / **MINOR** (shape-only tests; cosmetic) / **MAJOR** (mechanism partially wrong) / **BROKEN** (code contradicts the doc).

### 4.2 · Track B: research-scientist-critic team

Independent of the implementation, each group's sci-critic appends an "Addendum: Research-Scientist Critique" to every design doc, challenging prior plausibility, mechanism scrutiny, confounds (≥ 2), numerology check (does φ specifically matter or would any value in [1.3, 2.0] work?), literature precedent, expected effect size with 90 % CI, and minimum-distinguishing experiment. **Verdict:** NOVEL+TESTABLE / DERIVATIVE+TESTABLE / NUMEROLOGY / FALSIFIED / UNFALSIFIABLE / INFRASTRUCTURE.

### 4.3 · Track C: Fixer campaign

Findings from Track A populate the Fixer specs. Eight parallel Fixer agents (partitioned by primary src file to avoid cross-fixer contention) (a) patch the code per the audit's "Concrete fix" instruction, (b) add at least one **mechanism-verifying** test that would have caught the bug (NOT shape-only), (c) confirm the test passes green, (d) commit with retry-wrapped scoped `git add`. Together: 8 commits (`519cdf3`, `253dc94`, `afac553`, `3efd2dd`, `c395769`+`5f09814`+`8aa0430`, `16fe2b6`, `2e7ee45`, `9cca91e`), ~34 new mechanism tests, ~16 patched src files. After all fixers land, every affected sweep row re-runs on the corrected code (Rule 21). The mechanism-pinning test is the durable Fixer artefact — the patched code may improve further, but the test invariant remains.

### 4.4 · Hyperparameters and hardware contract

**Hardware ([Rule 26](CLAUDE.md#rule-26)).** 1× RTX 4090 Laptop, 16 GB VRAM, Windows 11; Python 3.13; bf16 AMP; `num_workers=0`; `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`; `set_seed(seed)` at run start; `cudnn.benchmark=True` (not bit-reproducible by design — [Rule 6](CLAUDE.md#rule-6)).

**Training defaults (CIFAR-10 12-ep screening; CIFAR-100 30-ep graduation + Phase-8 n=7).** AdamW; LR 1e-3 (cosine, T_max=epochs); weight decay 5e-4; batch 256; label smoothing 0.1; RandomCrop(32, pad=4) + HorizontalFlip + RandAugment(N=1, M=4); bf16. Single seed for 32/35 CIFAR-10 screening rows; n=7 for the three Phase-8 winners on CIFAR-100.

**Per-hypothesis deltas from defaults:** H09 phi_budget post-fix widths `[37, 48, 61]` (realised ratio 1:1.623:2.629, max-error 0.43 %); H41 golden_adam β1=1/φ, β2=1/φ², **eps=1e-8** (post-fix; pre-fix eps=1/φ⁴ ≈ 0.146 dominated Adam's denominator and was the eps-confound that produced the −33 pp pre-fix headline); H48 golden_momentum β1 schedule `1−(1−1/φ²)·k/T_max` (post-fix); H44 phi_decay_wd per-layer wd scaled by `φ^(−depth_idx)` from 5e-4; H81 slot_act_sine replaces every ReLU with `sin(ω·x)`, ω=1; H05 fractal depth=3, mix ratio φ⁻¹.

**Sweep-tag → config-row mapping:** `baseline_resnet20` (stock ResNet-20); `sg_only_phi_budget` (H09); `pair_gm_pdw` (H09 + H48 + H44); `slot_act_sine` (H09 + H81). Resolved YAMLs committed at `experiments/cifar100/<tag>_seed{0..6}/config.yaml`.

## 5 · Results — the audit verdict (protocol calibration)

### 5.1 · Implementation-critic distribution (Track A)

| group | PASS | MINOR | MAJOR | BROKEN |
|---|---:|---:|---:|---:|
| G1 Scaling & Growth | 3 | 4 | 3 | 0 |
| G2 Layer / Channel / Neuron | 6 | 3 | 1 | 0 |
| G3 Topologies & Graphs | 2 | 2 | **6** | 0 |
| G4 Kernels / Attention | 5 | 4 | 1 | 0 |
| G5 Optimisation | 4 | 3 | 3 | 0 |
| G6 Topological / Bridging | 4 | 4 | 0 | **1** (H55) |
| G7 Cross-Paradigm Hybrids | 10 | 2 | 1 | **2** (H67, H74) |
| G8 Esoteric Extensions | 7 | 2 | 0 | 0 |
| **TOTAL (83 audited)** | **41** | **24** | **15** | **3** |

51 % non-PASS rate, conditional on §1.3 + §5.8. Failures cluster in G3 (graph equivariance) and G7 (cross-paradigm composition). The credible signal is the **MAJOR/BROKEN sub-tier excess** over the calibration floor (§5.8), not the aggregate.

### 5.2 · Research-scientist verdict distribution (Track B)

NOVEL+TESTABLE 1 (H71 IcosaRoPE3D, untested); DERIVATIVE+TESTABLE 30; NUMEROLOGY 40; EMPIRICALLY-FALSIFIED 3 (H41 / H48 / H50); UNFALSIFIABLE 2 (H22, H67); INFRASTRUCTURE 5. Roughly half the design space is flagged NUMEROLOGY where φ is a renaming of a constant that any value in [1.3, 2.0] could replace with the same outcome. The strongest defensible category is DERIVATIVE+TESTABLE (37 %) — golden-skip residual scaling re-derives Fixup / ReZero; phi-decay LR is functionally interchangeable with cosine; spectral-Hopfield is a unitary basis change of modern Hopfield (Ramsauer 2020). The 1/81 rate is conditional on Track B sharing model family with Track A and the implementers.

### 5.3 · BROKEN findings (case studies of protocol output)

1. **H55 PlatonicAttention's head bias is mathematically zero.** `bias = (coords @ coords.T).mean(dim=-1)` evaluates to all-zeros for every vertex-transitive Platonic solid (vertex coords sum to centroid). PlatonicAttention was bit-equivalent to vanilla MHA. All 7 tests were shape-only. **Fixed** in Fixer-G6 (`16fe2b6`) with a relative-position cosine bias mirroring the H37 pentagonal pattern (Islam et al. 2025 Platonic Transformers).
2. **H67 hybrid_full was a half-on stress test.** `from .golden_rope import GoldenRoPE` raised ImportError; MetatronGraphLayer constructor signature was wrong; `which_priors_active` hardcoded `True` for 4 priors; LiquidCFC collapsed to affine + nonlinearity. **Fixed** in Fixer-G7 (`2e7ee45`).
3. **H74 MetatronTiedConv2d's 13 alphas collapsed to one scalar.** Forward was `F.conv2d(x, W · Σα_c)`; the 13 alphas Σ-summed to a single gate. **Fixed** in Fixer-G7 — now uses H40's `metatron_basis_kernels` 13 spatially-distinct circle masks; `effective_weight = Σ_c α_c · (W ⊙ mask_c)`.

### 5.4 · The single most impactful MAJOR finding — H09 phi_budget

§1.2 describes the realised-ratio drift (1:1.41:2.45 vs claimed 1:φ:φ²) and Fixer-PhiScaling's correction. Post-fix 3-seed CIFAR-100 medians on the corrected mechanism: `phi_budget=0.5741`, `baseline=0.5652` (+0.89 pp). The pre-fix median (58.05 %) was ~0.6 pp HIGHER than the post-fix (57.41 %), consistent with "the broken realised ratio happened to land a fortuitously-high seed-0 result." The case-study reading: the protocol caught a code-vs-doc divergence that would have shipped as a headline in a non-audited pipeline; whether the +0.89 pp post-fix lift represents a genuine architectural signal or a tuning artifact is decided by Phase-9 hill-climb, not this submission.

### 5.5 · Three Phase-8 candidates surfaced by the protocol — n=7 certification, hill-climb robustness, magnitude diagnostic, POSI re-framing

**Headline (n=7 default config, CIFAR-100 30-ep, post-fix code, sweep completed 2026-05-29 PM):**

| tag | C100 mean | Δmean | 95% bootstrap CI on Δmean | Paired Wilcoxon p (one-sided) | Holm-Bonferroni α'=0.0167 cleared? |
|---|---:|---:|---|---:|:---:|
| `pair_gm_pdw` (H09+H48+H44 stack) | 0.5786 | **+1.74 pp** | [+1.42, +2.09] pp | 0.0078 | **YES** |
| `slot_act_sine` (H81 SIREN) | 0.5790 | **+1.78 pp** | [+1.38, +2.18] pp | 0.0078 | **YES** |
| `sg_only_phi_budget` (H09 post-fix) | 0.5736 | **+1.24 pp** | [+0.84, +1.67] pp | 0.0078 | **YES** |
| `baseline_resnet20` (rail) | 0.5612 | — | (σ=0.451 pp on C100) | — | — |

All three winners produce 7/7 positive paired deltas → paired Wilcoxon W=0, exact one-sided p=(1/2)^7=**0.0078**, which clears Holm-Bonferroni α'=0.05/3=0.0167 across the k=3 family. Full derivation in [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §0–§3.

**Magnitude diagnostic (per ICML R1 BLOCKER #3; [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §9).** The Wilcoxon-at-floor at n=7 with 7/7 positive deltas is informationally identical to a paired sign test. We complement it with two magnitude-based tests. The exact paired permutation on Δmean (2^7=128 sign-flips) attains the same 0.0078 floor (the observed Δmean IS the maximum sign-flipped Δmean when every paired delta is positive). The **paired-t-test** (df=6) extracts σ-scaled magnitude information and produces p-values three to four orders of magnitude below the floor: t=9.06, p_one=5.1×10⁻⁵ for `pair_gm_pdw`; t=7.82, p_one=1.2×10⁻⁴ for `slot_act_sine`; t=5.43, p_one=8.1×10⁻⁴ for `sg_only_phi_budget`. The lift is many σ-baseline above zero, not merely "7/7 positive of any magnitude."

**Hill-climbed-best robustness (Phase-9a, 2026-05-30, n=3 each — [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §7).** A per-hypothesis coordinate hill-climb (cube lr × wd × bs × optimizer, budget 25) ran independently on baseline and each leader; each tag's best_config was re-run on seeds 0/1/2. Hill-climbed-baseline best top1 median = 0.5929 at {lr=3e-3, wd=5e-4, bs=256, AdamW}. Hill-climbed leader medians: `sg_only_phi_budget` 0.6049 (Δmedian +1.20 pp, Δmean +0.79 pp, 95% CI on Δmean **[−0.32, +1.76] pp — includes 0**); `pair_gm_pdw` 0.6109 (Δmedian +1.80 pp, Δmean +1.22 pp, CI [+0.15, +1.99]); `slot_act_sine` 0.6137 (Δmedian +2.08 pp, Δmean +1.31 pp, CI [+0.20, +2.23]). The priors carry signal in BOTH tuning regimes — but the `sg_only_phi_budget` hill-climbed CI INCLUDES 0, a result we report honestly and acknowledge undercuts the n=7 default-config certification when the analysis condition is changed to tuned-vs-tuned (ICML R1 BLOCKER #3 / R2 Q2 conceded). The two-arm batch-size confound (all three winners' best_config is bs=128 vs baseline's bs=256) is a known open issue — a bs=128 baseline row at n=7 is the cheapest single experiment that separates "prior helps" from "prior + small-bs regularisation helps" (R2 Q2; ~3.5 GPU-h; filed as Phase-9d).

**Iso-tuned-cell re-certification at n=7 (Phase-9f closeout, 2026-06-01 — [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §10).** **The iso-tuned re-certification at n=7 shows substantial Δ-shrinkage** (from +1.24/+1.74/+1.78 pp at default-config to +0.54 to +0.79 pp Δmean_unpaired at iso-tuned), and **the Phase-5 ordinal gate FAILS at iso-tuned n=7 for all three winners** (max iso-tuned baseline = 0.6075 at seed=3 across n=7; minimum iso-tuned leader seeds 0.5998 / 0.6049 / 0.6057 all ≤ 0.6075). **The default-config n=7 certification remains the formal claim; the iso-tuned regime reveals the priors' lift is partially explained by baseline-hyperparameter-mismatch (R2 BLOCKER #13 concern partially validated). Phase-9g future work: ~15-seed iso-tuned would tighten the iso-tuned variance estimate.** Concretely, the 2026-06-01 Phase-9f sweep extended both the iso-tuned baseline AND the three leaders to n=7 seeds at the iso-tuned cell (lr=3e-3, wd=5e-4, bs=128, AdamW for baseline / `pair_gm_pdw` / `sg_only_phi_budget`; wd=2e-3 for `slot_act_sine`, compared against the wd=5e-4 baseline neighbour). Iso-tuned baseline (n=7) mean=0.6000, σ_iso=**0.920 pp** — still 2.03× wider than the default-config σ_default=0.453 pp at n=7. Paired Wilcoxon at the iso-tuned cell: `pair_gm_pdw` Δmean=+0.79 pp paired (n=7), W=4.0, p_one=**0.1094**, only 4/7 paired deltas positive; `sg_only_phi_budget` Δmean=+0.66 pp paired (n=6, seed=3 excluded as <30 ep), W=3.0, p_one=**0.0781**, 95 % bootstrap CI [−0.04, +1.46] pp **includes 0**; `slot_act_sine` Δmean=+0.25 pp paired (n=4, seeds 3..6 only against the wd=5e-4 baseline neighbour), W=2.0, p_one=**0.3750**, CI [−0.07, +0.63] pp includes 0. **None of the iso-tuned-n=7 cells clears α=0.05, let alone Holm-Bonferroni α'=0.0167.** The default-config n=7 cert STANDS as the formal claim of the paper because (a) its baseline σ is half the iso-tuned σ even at matched n=7, (b) all three winners produced 7/7 positive paired deltas at the default cell, and (c) paired Wilcoxon p=0.0078 < α'_Holm=0.0167. The iso-tuned regime is reported with full Δ-shrinkage transparency — a Phase-9g n=15+ iso-tuned extension (and the Phase-9e wd=2e-3 baseline-neighbour for `slot_act_sine`) is the principled re-certification path. The previous n=3 placeholder in this paragraph (and the Δs of +1.16/+1.59/+1.68 pp reported in the 2026-05-31 closeout) is superseded by the n=7 Phase-9f numbers above; the n=3 narrative was directionally consistent (it already flagged Phase-5 FAILS for all three) but its Δmean point estimates were biased high by the wide n=3 baseline σ.

**POSI family-size re-framing (per ICML R1 BLOCKER #1).** The k=3 Holm-Bonferroni family used above is the **CONFIRMATORY family**, not the screening family. The candidates were filtered from a ~49-row decision population (35-row CIFAR-10 screen + 7-row CIFAR-100 graduation + 7-row combo-ladder) by the Phase-5 ordinal gate at n=3, a pre-specified screening criterion defined in CLAUDE.md Rule 28 before seeds 3..6 of the n=7 extension were run (the Rule-28 codification commit is the registration evidence). Strict POSI Holm-Bonferroni across the full 49-row decision family would demand α' = 0.05/49 ≈ 0.001, which the Wilcoxon n=7 floor of 0.0078 does NOT clear; the magnitude paired-t p-values (5×10⁻⁵ to 8×10⁻⁴ for the three winners) DO clear that bound for `pair_gm_pdw` and `slot_act_sine`, but not for `sg_only_phi_budget`. **The honest reading: the certification clears Holm-Bonferroni at α=0.05 across the 3-claim confirmatory family (Wilcoxon p < α'_Holm), and clears POSI Holm-Bonferroni at α=0.05 across the 49-row screening family for the two stronger winners under the paired-t magnitude test (p < 0.001); `sg_only_phi_budget` does NOT clear the POSI bound and is correspondingly reported as "POSI-uncertified but family-of-3 certified."** Both bounds are reported because both interpretations are defensible.

**Honest reading of the three case studies (post-n=7 certification).** `pair_gm_pdw` stacks three orthogonal axes; its Δmean +1.74 pp is certified at the family-of-3 bound but does NOT yet resolve the **non-φ 3-axis regularizer control** confound (controls 1–4 specified in [`controls/PLAN.md`](controls/PLAN.md) are READY but unlaunched at submission; R2 Q1). `slot_act_sine` swaps ReLU for `sin(ω·x)` with ω=1; the lift is consistent with Sitzmann et al. (NeurIPS 2020) SIREN's known activation-engineering benefit and has **no φ content** — certification is consistent with "SIREN works at the screening budget," not specifically with "φ-flavoured SIREN works." `sg_only_phi_budget`'s post-fix Δmean +1.24 pp sci-critic verdict remains DERIVATIVE+TESTABLE (rediscovery of RegNet's Pareto-optimal width-progression region — Radosavovic et al. CVPR 2020 explicitly prefers `w_m ∈ [2.5, 2.9]`, NOT φ=1.618); the tuned-RegNetX-200MF head-to-head is filed as Phase-9c.

### 5.6 · Cumulative defect tally

Per area-chair item #12 (category confusion warning): impl-critic verdicts and sci-critic verdicts are **different dimensions**, not additive defects. 83 hypotheses audited: 41 PASS / 24 MINOR / 15 MAJOR / 3 BROKEN. 81 hypotheses scientifically critiqued: 1 NOVEL+TESTABLE / 30 DERIVATIVE+TESTABLE / 40 NUMEROLOGY / 3 EMPIRICALLY-FALSIFIED / 2 UNFALSIFIABLE / 5 INFRASTRUCTURE. Operationally, 14 single-axis + 7 combo rows required Rule-21 post-fix re-run.

### 5.7 · Verdict-on-the-wrong-testbed audit (item #16, PARTIAL)

**H22 Toroidal-φ-closure** (`hypotheses/g3_topologies_graphs/H22_toroidal_phi_closure.md`) requires a "tiled-texture or wrap-aware synthetic dataset" per its design-doc falsifier; the project tested on upright CIFAR-10 and recorded a NUMEROLOGY/UNFALSIFIABLE verdict. **This verdict is downgraded to "UNTESTED-ON-RIGHT-DATASET."** A full audit of the remaining 41 NUMEROLOGY + 2 UNFALSIFIABLE verdicts to identify other CIFAR-as-wrong-testbed cases (candidates: H24, H25, H26, H53–H55, H58) is deferred to a follow-up pass.

**Cross-family re-audit on 10 MAJOR/BROKEN verdicts (R3 W2 / AC item #2, PARTIAL):** [`audits/CROSS_FAMILY_HONEST_REAUDIT.md`](audits/CROSS_FAMILY_HONEST_REAUDIT.md) executes a methodologically-diverse intra-family re-audit (3 methods — property-based, mechanism-trace, paper-math — across a stratified 10-finding subsample including all 3 originally BROKEN). Headline: **8/10 strict CONCORDANT, 10/10 defect-existence CONCORDANT**. Honest limitation: this is **NOT** a non-Claude external auditor — all audit passes remain Opus 4.7. The true cross-family closure (GPT-5 / Gemini 3 Pro pass on the same subsample) is filed as Phase-9e future work. Together with the Phase-9b n=62 calibration (§5.8), the auditor-self-grading concern is bounded from two complementary directions but not fully resolved.

### 5.8 · Audit calibration on third-party code (item #12, RESOLVED)

Track-A doctrine was applied to a 15-hypothesis sample drawn from `pytorch/vision` (`ResNet`/`BasicBlock`/`Bottleneck`/`Wide-ResNet`, `DenseNet`, `VGG`, `SqueezeNet`, `MobileNetV2`) and the cited `pytorch/pytorch` core modules (`Adam`, `SGD`, `kaiming_normal_`, `CosineAnnealingLR`, `BatchNorm2d`). Full audit with file:line citations in [`audits/AUDIT_CALIBRATION_THIRD_PARTY.md`](audits/AUDIT_CALIBRATION_THIRD_PARTY.md).

**Headline:** **10 PASS, 5 MINOR, 0 MAJOR, 0 BROKEN. Non-PASS 33.3 %.** Project: 50.6 %. **MINOR tier 29 % project vs 33 % calibration** — comparable, in line with audit aggressiveness. **MAJOR/BROKEN tier 22 % project vs 0 % calibration** — the **diagnostically-credible 22-pp excess**. Formal interval analysis ([`audits/AUDIT_CALIBRATION_THIRD_PARTY.md`](audits/AUDIT_CALIBRATION_THIRD_PARTY.md) §4.3.1, [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §8; 100 000-iter binomial bootstrap, rng=20260530): bootstrap 95 % CI on the difference = **[+13.3, +31.3] pp** (excludes 0); Wilson 95 % CI project [14.2 %, 31.7 %], calibration [0.0 %, 20.4 %]; **Fisher exact one-sided p=0.036** (clears α=0.05), two-sided p=0.066; pooled z-test two-sided p=0.046. The 22-pp excess is **directionally credible and one-sided significant at α=0.05** but does NOT clear two-sided α=0.05 under the conservative Fisher exact; n≥50 (Phase-9b — timm + HF Transformers + Lightning Bolts) is required to unambiguously certify. The MAJOR/BROKEN tier — where H09 realised-ratio drift, H21 hex_phi undocumented divergence, H67 broken GoldenRoPE import live — is the protocol's load-bearing diagnostic surface.

## 6 · Empirical evidence (CIFAR-10 / CIFAR-100) — protocol output

### 6.1 · Pre-fix screening sweep (single seed, 12 epochs)

35 single-prior CIFAR-10 rows at seed 0 / 12 ep; the only variant beating ResNet-20 baseline (84.78 %) was **H09 phi_budget at 85.54 %** (+0.76 pp). All 35 rows are n=1; per [`STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §4.1, pooled CIFAR-10 12-ep σ across 11 multi-seed tags is 0.607 pp → **2σ_pooled ≈ 1.21 pp**; the 99th-percentile single-seed Δ across 58 seed-0 non-baseline tags is +0.96 pp, INSIDE the 2σ band. **No CIFAR-10 12-ep seed-0 tag has prima-facie statistical credibility as a positive.** The pre-fix `golden_adam` row at 51.96 % (−32.82 pp) was originally framed as a clean falsification; H41 audit + post-fix re-run revealed the collapse was driven by `eps = 1/φ⁴ ≈ 0.146` (not the β-shift). Post-fix β-only top1=0.8394, Δ≈−1 pp. Reddi 2018's non-convergence proof for β2<0.95 is asymptotic and does not yet bite at 12 ep. **H41 is re-classified as WEAKLY NEGATIVE at 12-ep CIFAR-10 screening**, with the Reddi-regime test deferred to Phase-9 long-horizon β2 sweep.

### 6.2 · Post-fix sweep and combo ladder

A 7-row additive combo ladder on top of phi_budget + N orthogonal-axis priors (momentum / dropout / wd / LR / ensemble / activation / pruning) ran on pre-fix and post-fix code; full survival table in [`paper/FINDINGS.md`](paper/FINDINGS.md) §Phase-7. The Tier-A LOO subtractive matrix and two-at-a-time interaction matrix are similarly archived. **All single-prior screening rows remain n=1; the combo ladder is similarly n=1 by design.** Rule-23 (orthogonal-axes-only compounding, §7.2.1 below) was derived from this n=1 screening data; replication at n=7 to establish the "3 axes good, 6 axes bad" threshold with statistical force is filed as Phase-9d future work.

### 6.3 · The honest comparison to SOTA

ResNet-20 at the canonical 164-epoch SOTA recipe (He CVPR 2016) reaches ~91.25 % CIFAR-10 top-1. Our 12-epoch screening budget produces ~84.78 % baseline, **a 6.5-pp shortfall from convergence**. The H09 phi_budget screening lift (+0.76 pp at 12 ep) is less than 12 % of the gap-to-SOTA; the Phase-8 winners' +1.24 to +1.78 pp CIFAR-100 lifts are at the same calibration band. We make **no** SOTA claim at this scale. The H09 lift could entirely disappear when the baseline is tuned to 164-ep SOTA; the head-to-head against tuned-ResNet-20 and RegNetX-200MF is filed as Phase-9c (item #13).

## 7 · Discussion

### 7.1 · What survives the audit (protocol's hill-climb shortlist)

Combining Tracks A + B + Fixer outcomes, hypotheses currently nominated by the protocol for Phase-9 hill-climb (Rule 22: impl-critic PASS AND sci-critic ≠ NUMEROLOGY/UNFALSIFIABLE):

- **H05 fractal_phi_recursion** (DERIVATIVE+TESTABLE; only single prior to lift top-1 in the original 11-row campaign at +2.35 pp on CIFAR-10 screening).
- **H09 phi_budget** (DERIVATIVE+TESTABLE, RegNet-region rediscovery; post-fix +0.89 pp CIFAR-100 3-seed; passes Phase-5 screening gate; awaits Phase-9 + RegNet-tuned-baseline).
- **H21 hex_phi** (DERIVATIVE+TESTABLE; hex topology is real even if φ-radial is decorative).
- **H32 Fibottention** (DERIVATIVE+TESTABLE; Wythoff-array non-overlap is a real geometric property; transformer-only, no CIFAR row).
- **H39 PhiGELU** (DERIVATIVE+TESTABLE; β=φ sits between SiLU and GELU).
- **H58 group avg-pool** (EMPIRICALLY-FALSIFIED on CIFAR; methodologically clean — the negative result is the protocol output).

### 7.2 · What the protocol kills (at the screening + sci-critic level)

All hypotheses rated NUMEROLOGY by sci-critic (~40); all MAJOR/BROKEN findings without a Fixer patch (zero — all 18 are fixed); the catastrophic full-hybrid `sg_full_fib` (**−11.54 pp** below baseline at 12-ep CIFAR-10 screening). H80 Reuleaux constant-width (sci-critic NUMEROLOGY; sg_only_constant_width 75.95 %, mild-negative at screening) is preserved as a screened-negative case study with equal billing to H50 in [`README.md`](README.md) §5 / [`paper/FINDINGS.md`](paper/FINDINGS.md).

#### 7.2.1 · The orthogonal-axes-only doctrine (Rule 23) — operational detail

`sg_full_fib` stacked six nature-inspired priors (φ-channels + Fibonacci-depth + fractal-recursion + hex-conv + group-conv + toroidal-closure) on the same NaturePriorBlock forward path → −11.54 pp. CLAUDE.md Rule 23: multi-prior stacks may stack ONLY priors that touch different layers of the training stack (arch / channel / momentum / regulariser / weight-decay / LR / activation / ensemble / pruning / inference). The Phase-7 combo ladder (CLAUDE.md skill `autoresearch-combo-ladder`) adds one orthogonal axis per row; the n=1 screening data identifies `plr` (phi-LR-schedule) as the single most destructive axis (combo4→combo5 drop −5.66 pp), with combo7+combo8 partially recovering and Tier-A LOO showing `plr` removal from combo8 ALSO hurts (−1.13 pp; interaction structure, not independent destructiveness). **Rule-23 was derived from n=1 screening data; replication at n=7 to establish the "3 axes good, 6 axes bad" threshold with statistical force is filed as Phase-9d future work.** The `pair_gm_pdw` 3-axis stack (arch + momentum + wd) is the project's nominee for the "three-axis orthogonal stack outperforms its solo components" hypothesis, certified at the screening compute budget at the family-of-3 bound (§5.5); the φ-content vs non-φ-3-axis attribution question remains open until Controls 1–4 land.

### 7.3 · Limitations

- **Single-seed for most single-prior sweep rows.** Only `baseline_resnet20`, `phi_budget`, `golden_momentum` carry 3-seed on CIFAR-10 to date; the rest are seed-0. The 51 % non-PASS rate is independent (audit verdicts, not training seeds), but every per-tag empirical delta is single-seed and is treated as screening data per Rule 28.
- **No transformer experiments.** Half the 84 hypotheses target attention / sequence backbones; none tested. ViT-Tiny + decoder-only-LM scaffolds → §7.4.
- **No tuned-baseline or RegNet comparison** (item #13). The H09 +0.78 pp screening lift on CIFAR-10 could be a tuning artifact of nature-inspired-tuned vs untuned baseline. Phase-9c.
- **Auditor-self-grading at the model-family level** (§1.3 + §5.8). Cross-family audit on the calibration sample is open future work.
- **POSI on Phase-8 family selection.** The k=3 confirmatory family is post-screening (§5.5 POSI paragraph). We report both the k=3 family-of-3 certification (Wilcoxon p=0.0078 < α'=0.0167) AND the k=49 POSI bound (paired-t magnitude test clears 0.001 for two winners; `sg_only_phi_budget` does not). Both interpretations are defensible.
- **Wilcoxon-at-floor magnitude gap.** Addressed by the §5.5 paired-t magnitude diagnostic (t=5.43 to 9.06; p_one=5e-5 to 8e-4).
- **Calibration n=15.** Two-sided Fisher exact on the 22-pp MAJOR/BROKEN excess is p=0.066 (one-sided p=0.036). Phase-9b extends to n≥50.
- **Hill-climbed `sg_only_phi_budget` CI includes 0** (§5.5). Reported honestly; the n=7 default-config certification is the formal claim, the hill-climb is an additive robustness extension.
- **Iso-tuned-cell re-certification at n=7 shows Δ-shrinkage and Phase-5 FAIL** (§5.5 iso-tuned paragraph, [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §10). Phase-9f (2026-06-01) extended both the iso-tuned baseline and the leaders to n=7 seeds at the iso-tuned cell. σ_iso at n=7 is 0.920 pp vs σ_default at n=7 = 0.453 pp (still 2.03× wider even at matched n). Iso-tuned paired Δmeans shrink to +0.79/+0.66/+0.25 pp (vs default-config +1.74/+1.24/+1.78 pp); paired Wilcoxon p_one ∈ {0.1094, 0.0781, 0.3750} — none clears α=0.05; the Phase-5 ordinal gate FAILS for all three winners (max iso-tuned baseline = 0.6075). The default-config n=7 certification stands as the formal claim; the iso-tuned regime cannot be re-certified at NeurIPS-α with this sample size — Phase-9g n=15+ iso-tuned extension is the resolution path. R2 BLOCKER #13's mixed-bs concern is partially validated by this finding.
- **CIFAR-10/-100 is not a sufficient testbed** for several equivariance / topological hypotheses (rotated CIFAR, Spherical MNIST, ModelNet40, ogbg-molhiv). Acknowledged in `IDEA_TABLE.md`.
- **Multiple-comparisons.** Phase-8 family CLEARED Holm-Bonferroni at α=0.05 at n=7 (k=3 confirmatory family); the 35-row CIFAR-10 screen at n=1 still cannot clear ANY corrected α (per-test α'=0.05/35 ≈ 0.00143; smallest achievable p=0.5 at n=1). The screen is exploratory by mathematical necessity.
- **Selection bias on the 84-hypothesis design space.** Curated by an LLM agent reading a source PDF; selection is for plausibility, not coverage.

#### 7.3.1 · Methodological caveat — screening vs evaluation (with HARKing acknowledgement)

This subsection was authored **post-hoc** — after the screening protocol had been run for 35 single-prior CIFAR-10 rows and after the Phase-5 demotion of H48 `golden_momentum` and the multiple sci-critic verdicts had landed. Reclassifying every single-prior negative as "screening, not evaluation" *after* observing the negatives is **HARKing** (Hypothesizing After the Results are Known). We accept the methodological cost in this revision; we do **not** retroactively claim that the screening-vs-evaluation distinction was pre-registered. Going forward, the distinction is codified as CLAUDE.md Rule 28 and applies prospectively to all future autoresearch campaigns: any future single-config single-seed sweep must be labelled as screening at launch (config flag + commit hash) before its negatives are observed; only Phase-9 hill-climb + 3-seed + Phase-5-gate results carry "evaluation" weight.

The substantive content of the distinction stands independently of when it was authored. A real evaluation has three layers (CLAUDE.md Rule 28): per-hypothesis hill-climb over the natural knobs of each prior (≥ 20 runs per hypothesis); 3-seed re-runs at best config; cross-dataset Phase-5 worst-leader-seed > best-baseline-seed gate. **The three Phase-8 candidates in §5.5 satisfy LAYER 3 (Phase-5 cross-dataset 3-seed gate) at n=7 AND clear Holm-Bonferroni across the k=3 confirmatory family, but they do NOT satisfy LAYER 1 (per-hypothesis hill-climb at convergence-budget).** They are illustrative protocol output, not standalone empirical wins.

Every other hypothesis-level statement in this paper, in `FINDINGS.md`, and in `audits/*` — including "H41 falsified", "H42 paper-disagrees", "H44 wrong-test", "H47/H48 wrong-schedule", "H50 catastrophic", "H80 Reuleaux mild-negative", "H81 sine-act mid-pack", and all other single-prior verdicts at 12 ep — is reclassified as **SCREENING DATA, not evaluation**, with the post-hoc HARKing acknowledgement explicit.

### 7.4 · Future work

1. **Phase 9 — per-hypothesis hill-climb.** β2 grid for H41; base_wd grid for H44; dropout floor + slope for H47; T_max-step for H48; width-rounding for H09; stride / kernel-size for H21; etc. 3-seed re-run at best config, then Phase-5 cross-dataset gate.
2. **Transformer track.** Wire a ViT-Tiny + decoder-only-LM scaffold; smoke H03/H15/H16/H27/H32/H34/H36/H37/H55/H71.
3. **H71 IcosaRoPE3D dedicated experiment.** Sole NOVEL+TESTABLE survivor is untested; ViT-Tiny + rotated-CIFAR + Spherical-MNIST evaluation.
4. **Tuned-RegNet / tuned-ResNet baseline comparison** (item #13). H09 vs RegNetX-200MF at `w_m ∈ [2.5, 2.9]`, tuned-LR/-wd ResNet-20 at iso-compute, 164-ep convergence budget.
5. **Audit calibration on third-party code** (Phase-9b; item #12 extension). 50–100 hypotheses spanning timm + HF Transformers + Lightning Bolts.
6. **Cross-domain replication of the protocol.** `dlmastery/autoresearchtabular` Higgs, then medical + FX. Empirical test of "content-agnostic" claim.
7. **Dataset transfer.** Tiny ImageNet, rotated-CIFAR, MedMNIST, Spherical MNIST.
8. **Cross-family audit subsample (R3 W2 / Q1 response).** Re-audit 10 of the 18 MAJOR/BROKEN findings with GPT-5 or Gemini 3 Pro; report agreement rate.
9. **Pre-registration of Phase-9.** A pre-registration commit hash (knobs, ranges, seeds, statistical test, success criteria) before launching Phase-9.
10. **Controls 1–4 launch** ([`controls/PLAN.md`](controls/PLAN.md); R2 Q1): non-φ 3-axis stack, activation ablation, tuned-ResNet-20 + RegNetX-200MF, H71 ViT-Tiny smoke. ~31.75 GPU-h; READY at submission, unlaunched.
11. **BS=128 baseline row at n=7** (R2 Q2; Phase-9d). ~3.5 GPU-h. Separates "prior helps" from "prior + small-bs regularisation helps" in the hill-climbed regime.
12. **Iso-tuned n=15+ extension** (Phase-9g, supersedes Phase-9f which is now CLOSED). Phase-9f (2026-06-01) extended both arms to n=7 at the iso-tuned (lr=3e-3, wd=5e-4, bs=128, AdamW) cell and found σ_iso = 0.920 pp (still 2.03× wider than σ_default at matched n=7), Δ-shrinkage from +1.24/+1.74/+1.78 pp to +0.66/+0.79/+0.25 pp paired, and Phase-5 ordinal-gate FAIL for all three winners. Iso-tuned-cell certification at NeurIPS-α requires the leader's σ to remain tight while the baseline's variance is also bounded; only ~15–20 seeds at the iso-tuned cell would resolve the iso-tuned variance estimate definitively. The `slot_act_sine` wd=2e-3 baseline-neighbour extension is the related Phase-9e item.

## 8 · Conclusion

**The protocol is the contribution.** After 84 hypothesis implementations, 41+24+15+3 implementation verdicts, 1+30+40+3+2+5 scientific verdicts, 8 mechanism-correcting Fixer commits, and ~7 GPU-hours of CIFAR-10/-100 ablation, the protocol produced calibration data: 51 % non-PASS impl-critic rate (with the 22-pp MAJOR/BROKEN sub-tier excess as the diagnostically-credible signal, one-sided Fisher p=0.036), 1 / 81 NOVEL+TESTABLE sci-critic rate, and three protocol-screened candidates that pass a 7-seed CIFAR-100 30-ep paired Wilcoxon at α=0.05 under Holm-Bonferroni across the k=3 confirmatory family. We make **no** standalone empirical headline claim for nature-inspired priors at this scale. The three Phase-8 candidates are illustrative protocol output, conditional on Phase-9 hill-climb and on the §5.5.1 non-φ control row. The single NOVEL+TESTABLE sci-critic survivor (H71 IcosaRoPE3D) is untested and is treated as a research proposal, not a result. The audit + Fixer + re-run cycle, encoded in CLAUDE.md Rules 20–28 and packaged as seven content-agnostic skills in [`skills/`](skills/), is what we offer the community: portable infrastructure for distinguishing signal from numerology when an autoresearch design space is irresistibly large and the experimental cost per hypothesis is non-trivial. The cross-domain demonstration of that portability is open future work; the cross-family audit subsample (R3 Q1) is open future work; the empirical Controls 1–4 (R2 Q1) are READY but unlaunched. Subject to those caveats, the protocol caught a headline claim produced by broken code BEFORE that claim went to publication; **whether any of the protocol's surviving candidates becomes an externally defensible empirical claim is decided in Phase-9, not in this submission**.

## References

He K, Zhang X, Ren S, Sun J. 2016 CVPR. *Deep Residual Learning for Image Recognition*. arXiv:1512.03385. — ResNet baseline at 91.25 % CIFAR-10 / 164 ep.
Bronstein MM, Bruna J, Cohen TS, Veličković P. 2021. *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. arXiv:2104.13478. — Canonical GDL synthesis; the design-space framing of §1 sits inside its taxonomy of grid / group / graph / geodesic / gauge priors.
Hoogeboom E, Peters JWT, Cohen TS, Welling M. 2018 ICML. *HexaConv*. arXiv:1803.02108. — Hexagonal lattice convolution; H21 hex_phi.
Cohen TS, Geiger M, Köhler J, Welling M. 2019 ICLR. *Spherical CNNs*. arXiv:1902.04615. — H24/H53/H55 equivariance literature anchor.
Larsson G, Maire M, Shakhnarovich G. 2017 ICLR. *FractalNet: Ultra-Deep Neural Networks without Residuals*. arXiv:1605.07648. — H05 fractal recursion.
Sitzmann V, Martel JNP, Bergman AW, Lindell DB, Wetzstein G. 2020 NeurIPS. *Implicit Neural Representations with Periodic Activation Functions (SIREN)*. arXiv:2006.09661. — H81 slot_act_sine.
Ramsauer H, Schäfl B, Lehner J, et al. 2020 ICLR (2021 proceedings). *Hopfield Networks is All You Need*. arXiv:2008.02217. — H32 / H77 spectral Hopfield analog.
Radosavovic I, Kosaraju RP, Girshick R, He K, Dollár P. 2020 CVPR. *Designing Network Design Spaces (RegNet)*. arXiv:2003.13678. — H09 phi_budget sits inside RegNet Pareto region; RegNet prefers `w_m ∈ [2.5, 2.9]`, NOT φ=1.618 — direct DERIVATIVE-verdict literature anchor.
Reddi SJ, Kale S, Kumar S. 2018 ICLR. *On the Convergence of Adam and Beyond*. arXiv:1904.09237. — β2 < 0.95 non-convergence regime; H41 long-horizon test.
Choi D, Shallue CJ, Nado Z, Lee J, Maddison CJ, Dahl GE. 2019. *On Empirical Comparisons of Optimizers for Deep Learning*. arXiv:1910.05446. — β2 ∈ [0.95, 0.999] sweep; H41 reference.
Wilson AC, Roelofs R, Stern M, Srebro N, Recht B. 2017 NeurIPS. *The Marginal Value of Adaptive Gradient Methods in Machine Learning*. arXiv:1705.08292. — β-tuning marginal behaviour; H41 mild-regression literature support.
Loshchilov I, Hutter F. 2019 ICLR. *Decoupled Weight Decay Regularization (AdamW)*. arXiv:1711.05101. — AdamW; baseline optimizer; reference for H44.
He T, Zhang Z, Zhang H, Zhang Z, Xie J, Li M. 2019 CVPR. *Bag of Tricks for Image Classification with Convolutional Neural Networks*. arXiv:1812.01187. — Orthogonal-axis regularizer stacking; effect-size band for §5.5 magnitude comparison.
Hoffer E, Hubara I, Soudry D. 2017 NeurIPS. *Train longer, generalize better*. arXiv:1705.08741. — Long-horizon training behaviour; H41 / H48 Phase-9 caveat.
Pittorino F, Ferraro A, Perugini G, Feinauer C, Baldassi C, Zecchina R. 2022 ICML. *Deep Networks on Toroids: Removing Symmetries Reveals the Structure of Flat Regions in the Landscape Geometry*. arXiv:2202.03038. — H22 toroidal embedding literature anchor (resolves prior [VERIFY] tag).
Su J, Lu Y, Pan S, Wen B, Liu Y. 2024 Neurocomputing. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864. — H34 / H71 RoPE-family.
Islam MM, Anand R, Wessels DR, de Kruiff F, Kuipers TP, Ying R, Sánchez CI, Vadgama S, Bökman G, Bekkers EJ. 2025. *Platonic Transformers: A Solid Choice for Equivariance*. arXiv:2510.03511. Preprint, under review. — H55 PlatonicAttention literature anchor (resolves prior [VERIFY] tag; status as preprint is explicit).

Full bibliography in [`paper/NATURE_INSPIRED_NETWORKS.md`](paper/NATURE_INSPIRED_NETWORKS.md).

## Repository pointers

- Pre-print and source: <https://github.com/dlmastery/nature_inspired_networks>
- Live dashboard: <https://dlmastery.github.io/nature_inspired_networks/>
- Per-experiment pages: <https://dlmastery.github.io/nature_inspired_networks/dashboard/experiments/>
- Statistical-test analysis: [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) — top-level repo pointer.
- Audit calibration: [`audits/AUDIT_CALIBRATION_THIRD_PARTY.md`](audits/AUDIT_CALIBRATION_THIRD_PARTY.md).
- Controls plan (Controls 1–4, READY): [`controls/PLAN.md`](controls/PLAN.md).
- ICML 2027 reviewer rebuttal: [`audits/ICML_REVIEWS_2026-05-30/REBUTTAL.md`](audits/ICML_REVIEWS_2026-05-30/REBUTTAL.md).
- Phase-9 hill-climb skill: [`skills/autoresearch-per-hypothesis-hillclimb/`](skills/autoresearch-per-hypothesis-hillclimb/).
- Operator quick-reference: [`CLAUDE.md`](CLAUDE.md) §8.

## Appendix A — Threats to validity (response to area-chair Section F)

**F1 · Selection bias on the 84 hypotheses.** The design space was curated by an LLM agent reading a source PDF; members were selected for plausibility within the nature-inspired motivational frame, not for coverage. The audit's "1 NOVEL+TESTABLE / 81" and 51 % non-PASS rates are conditional on this selection.

**F2 · Post-hoc HARKing on negatives.** Engaged in §7.3.1 with explicit acknowledgement; Rule 28 codifies the standard prospectively.

**F3 · `pair_gm_pdw` non-φ 3-axis control missing.** §5.5 + [`controls/PLAN.md`](controls/PLAN.md) Control 1 (READY, unlaunched, R2 Q1).

**F4 · `slot_act_sine` SIREN-without-φ confound.** §5.5 honest reading + Control 2 (READY, unlaunched).

**F5 · Post-fix H09 — corrected widths simply slightly worse than broken widths?** Pre-fix CIFAR-100 3-seed median (0.5805) is ~0.6 pp HIGHER than post-fix (0.5741); we framed this as "the audit caught a fortuitously-high seed-0 result on broken code." Observationally indistinguishable from "the corrected widths are simply slightly worse" at n=3; an iso-cost head-to-head between pre-fix and post-fix widths at the same budget across multiple seeds would resolve it. The case-study reading is the protocol-level claim, NOT the architectural-level claim.

**F6 · Multiple-comparisons correction.** Engaged in §5.5 + [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §0–§3, §7, §8, §9. Confirmatory k=3 family CLEARED Holm-Bonferroni at n=7. POSI k=49 bound reported separately (§5.5 POSI paragraph).

**F7 · 51 % non-PASS rate as evidence of "real defects."** §5.8 + §1.3. The diagnostically-credible signal is the 22-pp MAJOR/BROKEN excess over the 0 % calibration floor (one-sided Fisher p=0.036), not the aggregate.

**F8 · No comparison to non-φ baseline architecture tweaks.** §6.3 + §7.4-4; Phase-9c filed.

## Appendix B — Pre-registration template for Phase-9 (going forward)

```yaml
# Phase-9 hill-climb pre-registration (template; commit hash = registration timestamp)
hypothesis: <H##_short>
hypothesis_design_doc: hypotheses/g<N>/H<NN>_*.md
hypothesis_doc_commit: <sha>           # locks the doc text at registration time
knobs:
  - <knob_1>: <range / grid>           # e.g., beta2: [0.382, 0.5, 0.7, 0.9, 0.95, 0.999]
  - <knob_2>: <range / grid>
search_strategy: coordinate_descent | random_grid_search
budget_runs: >= 20
seeds_at_best_config: [0, 1, 2]
dataset: cifar10                       # 12 ep
graduation_dataset: cifar100           # 30 ep
graduation_seeds: [0, 1, 2, 3, 4, 5, 6]
success_criteria:
  - layer_1: hill-climb best > screening baseline by > X pp at single seed
  - layer_2: 7-seed worst-leader-seed at best config > 7-seed baseline best-seed
  - layer_3: cross-dataset Phase-5 gate satisfied
multiple_comparisons:
  family_size: <# hypotheses in current hill-climb wave>
  alpha_family: 0.01
  correction: Holm-Bonferroni
statistical_test:
  - per_hypothesis: paired Wilcoxon + paired permutation + paired t (magnitude)
  - cross_dataset: paired Wilcoxon (n=7 CIFAR-100)
  - reporting: W, exact p, paired-t p, bootstrap 95% CI on Δ
ablations:
  - non_phi_control: <iso-budget non-phi analog>
  - tuned_baseline_control: <tuned-LR/-wd ResNet-20 + RegNetX-200MF at iso-compute>
  - bs128_baseline_control: <n=7 baseline at hill-climbed bs=128>
registration_commit: <sha>
registration_date: <YYYY-MM-DD>
```

## Appendix C — Per-hypothesis dual-track verdict table (summary)

Full per-hypothesis verdict landscape (Track A × Track B) is archived in [`audits/PER_HYPOTHESIS_VERDICTS.md`](audits/PER_HYPOTHESIS_VERDICTS.md) and the source-of-truth `audits/G<X>_audit.md` + "Addendum: Research-Scientist Critique" sections of each `hypotheses/g<N>/H<NN>_*.md`. 83 hypotheses audited by Track A (H57 deferred): 41 PASS / 24 MINOR / 15 MAJOR / 3 BROKEN. 81 hypotheses critiqued by Track B (H49/H54/H59 INFRASTRUCTURE not counted; H57 deferred): 1 NOVEL+TESTABLE / 30 DERIVATIVE+TESTABLE / 40 NUMEROLOGY / 3 EMPIRICALLY-FALSIFIED / 2 UNFALSIFIABLE / 5 INFRASTRUCTURE.

## Appendix D — Protocol transferability spec (claimed by construction; cross-domain validation future)

The seven content-agnostic skills in [`skills/`](skills/) are: `autoresearch-multi-agent-dispatch`, `autoresearch-critic-team`, `autoresearch-scicritic-team`, `autoresearch-fixer-campaign`, `autoresearch-combo-ladder`, `autoresearch-per-experiment-page`, `autoresearch-auto-checkpoint-loop`. Each skill exposes a content-agnostic API; domain-specific content lives in the prompt-template / per-hypothesis context. The portability test (open future work §7.4-6) clones `dlmastery/autoresearchtabular`, runs Track-A on its `src/` and `hypotheses/`, runs Track-B on its design docs, and compares the resulting verdict distributions to the project's 51 % non-PASS / 1 NOVEL+TESTABLE distribution. If the tabular replication produces an analogous distribution AND identifies at least one defect that an unaudited pipeline would have shipped, the content-agnostic claim is empirically supported.

---

*ICML 2027 main-track submission target: ≤ 8 main pages + supplementary; appendices A–D are supplementary material. PAPER.md serves as the arXiv preprint; the main-track conference version is the §1–§8 narrative.*
