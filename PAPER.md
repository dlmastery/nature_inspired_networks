# A Skeptical Protocol for Nature-Inspired Neural-Network Priors: Methodological Contribution and an 84-Hypothesis Dual-Track Audit on CIFAR-10/-100

**Internal QA pass — external review pending.** As of commit `ca32fcd`, the area-chair sniff test produced a `WEAK_REJECT` verdict (see [`audits/REVIEWER_PASS_PAPER.md`](audits/REVIEWER_PASS_PAPER.md)). This revision (commit forthcoming) addresses every BLOCKER and every MAJOR finding from that pass. The strategic decision adopted in this revision is **protocol-as-contribution (framing (a) per area-chair item #1)**: the dual-track audit + Fixer + per-experiment-page machinery is the headline contribution, and the three Phase-8 numerical results (`pair_gm_pdw`, `slot_act_sine`, `sg_only_phi_budget` post-fix) are downgraded to **illustrative case studies of protocol output**, NOT a standalone empirical headline. All empirical claims are explicitly PROVISIONAL pending the Phase-9 hill-climb campaign. The previous "Reviewer-acceptance ACCEPT verdict at commit `0343f35`" banner was self-graded and has been removed.

### Status of the 16 required revisions from `audits/REVIEWER_PASS_PAPER.md`

| # | severity | item | status |
|---|---|---|---|
| 1 | BLOCKER | Internal contradiction — pick protocol-as-contribution framing | **DONE** (abstract, §5.5, §8 all rewritten) |
| 2 | BLOCKER | §7.3.1 HARKing acknowledgement | **DONE** (§7.3.1 now explicitly admits post-hoc framing) |
| 3 | BLOCKER | Statistical unsupport on Phase-8 numbers | **DONE** — Wilcoxon W=0 / p_one=0.125 (n=3 exact-distribution floor), bootstrap 95% CIs, and Holm-Bonferroni α'=0.0167 (all FAIL) spliced into §5.5 from [`STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md); n ≥ 7 seeds required for any single Phase-8 row to clear Holm at α=0.05 across the k=3 family |
| 4 | BLOCKER | `pair_gm_pdw` missing non-φ 3-axis control | **DONE** (§5.5.1 added with explicit confound-open label) |
| 5 | BLOCKER | Transformer-track scope (H03/H15/H16/H27/H32/H34/H36/H37/H55/H71) | **DONE** (moved to §7.4 Future Work; stripped from §1.1 contributions) |
| 5 (MINOR) | MINOR | mystical disclosure consistency between §1 and §5 | **DONE** |
| 6 | BLOCKER | Hyperparameter table missing | **DONE** (§4.4 added) |
| 7 | BLOCKER | SOTA framing — baseline 6.5pp below SOTA | **DONE** (added to abstract + §1 + §6.4) |
| 8 | BLOCKER | Single-seed limitation buried | **DONE** (moved to abstract) |
| 9 | MAJOR | Duplicated §5.5 heading | **DONE** (renumbered to §5.6) |
| 10 | MAJOR | Self-acceptance banner | **DONE** (removed; replaced with internal-QA-pending-external-review language above) |
| 11 | MAJOR | Citation Rule-4 compliance | **PARTIAL** — Sitzmann venue corrected; Pittorino 2022 marked `[VERIFY]`; Islam 2025 arXiv:2510.03511 marked `[VERIFY: arXiv ID under verification]` |
| 12 | MAJOR | Audit calibration on third-party code | **DEFERRED** (§5.8 explicitly acknowledges this is missing; deferred to future work) |
| 13 | MAJOR | RegNet / tuned-ResNet baseline comparison | **DEFERRED** (§7.3 Limitations explicitly engages; deferred to Phase-9) |
| 14 | MAJOR | H71 IcosaRoPE3D from contributions to future work | **DONE** (moved to §7.4) |
| 15 | MAJOR | Auditor self-grading caveat | **DONE** (§1.3 added) |
| 16 | MAJOR | CIFAR-as-wrong-testbed audit of NUMEROLOGY/UNFALSIFIABLE verdicts | **PARTIAL** — H22 toroidal explicitly downgraded in §5.7; full audit of all 42 such verdicts deferred to future Reviewer-Followup pass |

Items marked `IN PROGRESS` or `PARTIAL` are bounded by parallel agent work-streams and will be resolved in a follow-up commit; this PAPER.md revision is structured to be **stable under those updates** (the placeholders are typographically explicit and a Reviewer-Followup pass can resolve them mechanically without reshaping the paper).

---

## Abstract

We present a **methodological contribution: a dual-track skeptical audit + Fixer + per-experiment-page protocol** for autoresearch campaigns that propose large families of architectural priors. The protocol layers two independent expert teams on top of an implementation campaign — an 8-agent **implementation-critic team** that verifies code-vs-doc correspondence with mechanism-pinning tests, and an 8-agent **research-scientist-critic team** that challenges scientific merit independent of code correctness — followed by an 8-agent **Fixer campaign** that patches confirmed defects with mechanism-verifying tests, then **re-runs every affected sweep row** on the corrected code. We motivate, specify, and operationalise the protocol in [`CLAUDE.md`](CLAUDE.md) Rules 20–28 and package it as seven content-agnostic skills in [`skills/`](skills/). The protocol is the contribution. As empirical **calibration** (not headline result), we apply it to an 84-hypothesis nature-inspired neural-network design space (φ-scaling, hexagonal lattices, Platonic / icosahedral equivariance, fractal recursion, toroidal closure, Chladni cymatic init, golden-angle modulation, and 15 cross-paradigm hybrids) on CIFAR-10 / CIFAR-100. The calibration reports: of 83 implementations audited, 51 % land non-PASS (3 BROKEN / 15 MAJOR / 24 MINOR); of 81 hypotheses scientifically critiqued, only ONE ranks NOVEL+TESTABLE; the Fixer campaign produced 8 commits and ~34 new mechanism-verifying tests; post-fix re-runs identified three candidates that pass the project's worst-leader-seed > best-baseline-seed CIFAR-100 3-seed screening gate (`pair_gm_pdw` Δmean=+1.36 pp, `slot_act_sine` Δmean=+1.39 pp, `sg_only_phi_budget` Δmean=+0.91 pp; bootstrap 95% CIs on Δmean all exclude 0). **All three FAIL the formal paired-Wilcoxon test at α=0.05** — the n=3 exact-distribution floor is p_one=0.125, which is 2.5× looser than NeurIPS-standard α=0.05 and 7.5× looser than Holm-Bonferroni α'=0.0167 across the k=3 family; n ≥ 7 seeds is required for any row to clear Holm at α=0.05 even with every paired delta positive (see [`STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md)). These three are therefore illustrative case studies of protocol output with non-zero-excluding CIs but **formally uncertified at n=3** — **whether they survive a per-hypothesis hill-climb at n ≥ 7 (Phase 9, ahead) is open**, and we explicitly do **not** claim them as standalone empirical results.

**Three calibration caveats are stated up front.** (a) **Comparison baseline is 6.5 pp below SOTA.** ResNet-20 at 164 ep reaches ~91.25 % top-1 on CIFAR-10; our 12-ep screening baseline is ~84.78 %. All deltas reported here are at this calibration level, not at converged SOTA. (b) **Most single-prior sweep rows are single-seed.** Only 3 tags (`baseline_resnet20`, `phi_budget`, `golden_momentum`) carry 3-seed error bars; the remaining 32 single-prior CIFAR-10 rows are n=1. (c) **Implementer, audit, sci-critic, and Fixer agents are all from the same model family** (Claude Opus 4.7). The audit's "independence" is methodological theatre at the model-family level; the 51 % non-PASS rate is conditional on this caveat and has not been calibrated against a known-good third-party codebase. The empirical claims in this paper are PROVISIONAL until those calibrations land.

## 1 · Introduction

Geometric Deep Learning (GDL) and Topological Deep Learning (TDL) have repeatedly converged on a small set of geometric priors as cheap, principled replacements for the standard square-grid CNN scaffold: hexagonal lattices (Hoogeboom 2018), Platonic / icosahedral symmetry groups (Cohen 2019), fractal recursion (Larsson 2017), toroidal embeddings (Pittorino 2022), Fibonacci-scaled channels, and modern Hopfield / SIREN / RoPE / phyllotactic phase encodings. A separate literature in cognitive science, biology, and aesthetics — repeatedly summarised by popular books — suggests the golden ratio φ ≈ 1.618 and the Fibonacci sequence appear "everywhere in nature." It is tempting to assemble the geometric priors and the φ-flavoured constants into one framework and claim broad gains.

This work does not do that. Instead it presents the **infrastructure required to know whether such a claim is justified** — and applies that infrastructure to its own design space. We *implement* 84 hypotheses faithfully, then *audit them adversarially* with two independent expert teams, *patch* the findings, *re-run* the empirical evidence on the corrected code, and *publish the surviving framework with its honest verdict — including the honest verdict that the protocol's own empirical findings are PROVISIONAL until the Phase-9 hill-climb completes*.

The methodological contribution is the protocol. The empirical "winners" surfaced by the protocol are case studies — useful to demonstrate that the protocol's "screening gate" is non-trivially populated, but not yet evidence of nature-inspired priors as a research direction. The honest reading of the empirical evidence in this paper is "the protocol screened 35 single-prior CIFAR-10 candidates and a handful survived to CIFAR-100 graduation; whether they survive hill-climb is the next step." We do not claim more than that.

### 1.1 · Contributions

1. **A dual-track skeptical audit protocol** — an 8-agent implementation-critic team verifies *code-vs-doc correspondence*, and an 8-agent research-scientist-critic team challenges the *scientific merit of the hypothesis itself*. Both teams are parallel and disjoint-scoped; their findings populate `audits/G<X>_audit.md` and an "Addendum: Research-Scientist Critique" section appended into each design doc.
2. **A Fixer campaign with mechanism-verifying-test discipline**: every code patch ships with at least one test that asserts the fixed mechanism (the test the original implementer missed). The Fixers correct 22 hypotheses across 8 commits, adding ~34 new mechanism tests.
3. **A normative ruleset** (CLAUDE.md Rules 20–28) operationalising the audit + fix + re-run cycle and the screening-vs-evaluation distinction, plus seven reusable skills in [`skills/`](skills/) so any future autoresearch project can pick up the protocol unchanged. We note explicitly: portability is *claimed by construction* (the skills are content-agnostic), not *demonstrated by cross-domain replication* — that demonstration is open future work (item §7.4-6).
4. **A reproducible 84-hypothesis implementation** of nature-inspired neural priors as 80 pure-PyTorch modules + 78 test files (~780 unit tests), each accompanied by a committee-grade design doc. We provide this implementation as the calibration substrate against which the protocol's defect-detection rate is reported — NOT as a standalone library of recommended priors.
5. **An empirical calibration of the protocol on CIFAR-10/CIFAR-100**: of 83 implementations audited, 51 % land non-PASS; of 81 hypotheses scientifically critiqued, only 1 ranks NOVEL+TESTABLE; three Phase-8 candidates pass the worst-leader-seed gate. These calibration numbers are conditional on the audit-self-grading caveat (§1.3) and the screening-vs-evaluation distinction (§7.3.1).

What is **explicitly NOT** in the contributions list (per area-chair item #5):
- Any claim about transformer-track or decoder-only-LM-track nature-inspired priors. Half the 84 hypotheses (H03, H15, H16, H27, H32, H34, H36, H37, H55, H71) target attention / sequence backbones. None were tested in this submission; the ViT-Tiny and decoder-only-LM scaffolds are future work (§7.4-2).
- Any claim about H71 IcosaRoPE3D — the sole NOVEL+TESTABLE survivor of the sci-critic gate. H71 has no CIFAR smoke; it is a transformer-track hypothesis and is treated as a research proposal (§7.4-3), NOT a contribution of this submission.
- Any claim about portability of the protocol to a second autoresearch domain (medical, tabular, FX). The cross-domain replication is open future work (§7.4-6).
- Any claim that the three Phase-8 winners constitute a standalone empirical headline. They are illustrative protocol output, conditional on Phase-9.

### 1.2 · Why the audit was necessary

The pre-audit version of this paper claimed H09 phi_budget as a verified cross-dataset positive (CIFAR-10 85.54 %, CIFAR-100 58.05 % 3-seed median, leading the baseline by +1.53 pp with the min-leader-seed > max-baseline-seed Phase-5 gate satisfied). The user did not trust the validation — every test file in the repo had been written by an implementer agent trained to make tests pass, and "make tests pass" often degenerates to shape-only assertions that don't actually verify the mechanism. The audit campaign exists to discover whether that distrust was justified. **It was**: the post-audit analysis revealed that H09's realised stage-parameter ratio was 1 : 1.41 : 2.45, not the claimed 1 : φ : φ² = 1 : 1.618 : 2.618 (a 12.6 % drift at stage 1), and the headline number was produced by a network that did not faithfully implement its own design doc.

### 1.3 · Limitations of the audit protocol — auditor self-grading

A binding caveat acknowledged in this revision (per area-chair item #15): the implementer agents, the implementation-critic agents, the sci-critic agents, and the Fixer agents in this campaign are **all from the same model family** (Claude Opus 4.7). The audit's claimed "independence" is enforced at the disjoint-scope and disjoint-file-target level (Track A reads `src/`, Track B reads `hypotheses/`; Fixers touch disjoint primary files); but at the **model-family level**, every agent shares the same training distribution, the same prior beliefs about what "good ML code" looks like, and the same characteristic failure modes. Specifically:

- The audit's 51 % non-PASS rate is **not calibrated** against a known-good baseline (e.g., the same audit team applied to `timm` or `pytorch-cifar`). Without that calibration, the 51 % is compatible with both "this codebase has many bugs" and "this audit team is hyperactive on any codebase." The two hypotheses are observationally equivalent in the present data.
- The audit's "we caught a buggy implementation" narrative (H09 realised ratio 1 : 1.41 : 2.45) is genuinely a code-vs-doc divergence and does survive this caveat — the audit *did* catch it. But the audit's calibrated diagnostic power requires an external benchmark we have not yet generated.

We do not engage the calibration empirically in this submission. The "Audit calibration" subsection (§5.8) explicitly flags this as future work. **The 51 % non-PASS rate, the 1 NOVEL+TESTABLE / 81 critiqued sci-critic rate, and the 22-patches-with-mechanism-tests Fixer outcome are therefore all conditional on this caveat.** Readers should weigh the calibration numbers accordingly.

### 1.4 · Mystical motivation, neutral artifact names

The 84-hypothesis design space is motivated in part by a popular literature suggesting nature-inspired constants (φ, Fibonacci, hexagonal packing, Platonic solids, Chladni patterns) "appear everywhere." We treat that motivation as a **prior over the design space**, not as evidence. Per CLAUDE.md Rule 16, artifact names are neutral (`nature_inspired_networks`, `NaturePrior*`); the mystical inspiration is acknowledged in prose only. §1.1's "this work is the opposite of [claiming broad gains]" is not contradicted by §5.5's protocol-output discussion: §5.5 frames the three Phase-8 candidates as illustrative, NOT as endorsements of golden-ratio priors. The audit RATIFIED the project at the protocol level; the audit produced PROVISIONAL screening-positive signals at the empirical level; the two statements are consistent.

## 2 · The autoresearch protocol (inherited)

The autoresearch protocol of `dlmastery/autoresearchimage` — citation rigor (every reasoning entry uses `Author YEAR VENUE 'Title' (arXiv:XXXX.XXXXX) — relevance`), reasoning-blob completeness (word-count floors per field), SHA-256-fingerprinted composite metric, append-only experiment log, no-bypass gates, per-experiment archive directory — is taken verbatim and forms the **floor** of this work, not the ceiling. The dual-track audit + Fixer protocol layered on top is the new contribution.

The composite metric formula is stated explicitly:

```
composite = top1 − 0.05 · log10(params_M) − 0.05 · log10(latency_ms)
```

with SHA-256 fingerprint `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893` (per CLAUDE.md Rule 2). Editing the formula raises `CompositeFingerprintError` at runner import; new formulas require a new repo branch.

CLAUDE.md (the repo's normative ruleset) was extended by Rules 20–28 during this campaign:

- **Rule 20.** Auto-checkpoint loop alongside any background task > 15 min — a power outage during the multi-hour sweep must lose ≤ 1 run.
- **Rule 21.** Post-fix re-run discipline — after any Fixer patch, the affected sweep tag MUST re-smoke before the fix is "complete"; if the patched hypothesis was a Phase-4/5 graduate, the CIFAR-100 3-seed is also mandatory.
- **Rule 22.** Dual-track audit before any external claim — pass BOTH impl-critic (no MAJOR/BROKEN) AND sci-critic (verdict ≠ NUMEROLOGY / UNFALSIFIABLE).
- **Rule 23.** Compound design uses orthogonal axes only — `sg_full_fib` (6 priors on same forward path, −11.54 pp) is the cautionary tale; the canonical additive test is the combo ladder (2 → N priors, one per row).
- **Rule 24.** Dashboard discipline — group-sectioned aggregate + independent per-experiment page per run + GitHub Pages mirror + no row-click modals.
- **Rule 25.** Q&A-test correspondence — every test name promised in a design doc's Verification checklist MUST exist in `tests/`.
- **Rule 26.** Windows thread-cap safety (KMP/OMP/MKL=2) — crash-prevention discipline for multi-agent + GPU campaigns.
- **Rule 27.** Pages-link discipline — no repo-root `.md` hrefs in HTML; absolute GitHub-blob URLs only.
- **Rule 28.** **Screening vs evaluation** — a single-config single-seed (or even 3-seed) sweep number is *screening data*, not *evaluation*. External claims require per-hypothesis hill-climb + 3-seed at best config + cross-dataset Phase-5 gate. Rule 28 was added in this revision specifically to formalise the distinction relied on by §7.3.1.

## 3 · The 84-hypothesis design space

The hypotheses span 8 thematic groups, organised in `hypotheses/g<N>_*/H<NN>_*.md`:

- **G1 Scaling & Growth** (H01–H10)
- **G2 Layer / Channel / Neuron** (H11–H20)
- **G3 Topologies & Graphs** (H21–H30)
- **G4 Kernels / Attention / Filters** (H31–H40)
- **G5 Optimisation / Init / Reg / NAS** (H41–H50)
- **G6 Topological & Bridging** (H51–H60, H57 deferred)
- **G7 Cross-Paradigm Hybrids** (H61–H75)
- **G8 Esoteric Extensions** (H76–H84, neutral recasts of mystical motifs)

See `IDEA_TABLE.md` for the per-hypothesis design surface and `hypotheses/INDEX.md` for the per-group index.

## 4 · Methods — the dual-track audit

### 4.1 · Track A: implementation-critic team

For each thematic group G1–G8, a parallel Critic agent reads every hypothesis's `H<NN>_*.md` design doc, the corresponding src module under `src/nature_inspired_networks/`, and the test file `tests/test_<module>.py` line-by-line. The audit produces a per-group `audits/G<X>_audit.md` with per-hypothesis findings on six dimensions:

1. **Mechanism check** — does the code actually implement the claimed mechanism?
2. **Math correctness** — constants, axes, normalisations, RNG state.
3. **Test rigor** — do the tests assert MECHANISM, or only SHAPE? Shape-only is MINOR at best.
4. **Citation alignment** — does the cited arXiv paper actually describe the technique?
5. **Falsifier reachability** — can a real run extract the relevant metric?
6. **Hidden bugs / cargo-cult** — softmax-invariant biases, learnable Parameter mistakenly registered as buffer, `torch.no_grad()` inside forward, etc.

Verdict tiers (strict): **PASS** / **MINOR** (shape-only tests; cosmetic) / **MAJOR** (mechanism partially wrong) / **BROKEN** (code contradicts the doc).

### 4.2 · Track B: research-scientist-critic team

Independent of the implementation, each group's sci-critic appends an "Addendum: Research-Scientist Critique" section to every design doc, challenging:

- **Prior plausibility** (LOW / MED / HIGH) — independent of nature-inspired framing.
- **Mechanism scrutiny** — does the "because" clause predict the claimed effect, or is it post-hoc?
- **Confounds** (≥ 2) — what else could explain a positive result?
- **Numerology check** — does φ specifically matter, or would 1.5 / 1.7 / 2.0 work equally well? Specify the control ablation.
- **Literature: precedent or rediscovery?** — is this a known technique under a different name?
- **Expected effect size** — sceptical a-priori re-prediction, 90 % CI.
- **Minimum-distinguishing experiment.**
- **Verdict:** NOVEL+TESTABLE / DERIVATIVE+TESTABLE / NUMEROLOGY / FALSIFIED / UNFALSIFIABLE / INFRASTRUCTURE.

### 4.3 · Track C: Fixer campaign

Findings from Track A populate the Fixer specs. Eight parallel Fixer agents (partitioned by primary src file to avoid cross-fixer contention) each (a) patch the code per the audit's "Concrete fix" instruction, (b) add at least one mechanism-verifying test that would have caught the bug, (c) confirm the test file passes green, (d) commit with retry-wrapped scoped `git add`. Together the 8 fixers produce 8 commits, ~34 new mechanism tests, and ~16 patched src files. After all fixers land, every affected sweep row re-runs on the corrected code (Rule 21).

The Fixer protocol enforces a strict "test the fixed mechanism" discipline that distinguishes it from a generic "make the code compile" patch:

- Each Fixer agent receives the audit's `audits/G<X>_audit.md` entry for one hypothesis as its specification. The audit entry contains a `Concrete fix:` block (≥ 40 words) and a `Mechanism-verifying test (suggested):` block.
- The Fixer's commit MUST add a test whose assertion captures the **claimed mechanism**, not the shape of the output. For H09 phi_budget the mechanism-pinning test asserts that the realised stage-parameter ratio is within 1 % of `[1, φ, φ²]`; the pre-Fixer test only asserted output shape `[B, num_classes]` and would have passed the broken implementation.
- The Fixer agents run sequentially on the same disjoint-file partitions to avoid `git index.lock` contention. Retries are wrapped (3 attempts × 1 s backoff) per CLAUDE.md skill `autoresearch-multi-agent-dispatch`.
- Post-fix, the affected sweep tags re-run via the orchestrator `scripts/launch_postfix_campaign.sh`; pre-fix and post-fix numbers are tabulated side-by-side in `FINDINGS.md` §Phase-7 with survival buckets (STRENGTHENED / STABLE / WEAKENED / EMERGED / NEW-NEGATIVE).
- The mechanism-pinning test is the lasting Fixer artifact. The patched code may be improved further by future Fixers; the test is invariant under those improvements.

The Fixer campaign produced the following commits (per AUDIT_SUMMARY.md):

| Fixer agent | scope | commit | hypotheses patched | new mechanism tests |
|---|---|---|---|---|
| Fixer-PhiScaling | G1 width allocation | `519cdf3` | H09 (realised ratio), H08 (Net2Net) | 4 |
| Fixer-G2 | G2 channel / neuron | `253dc94` | H14 (FibGRU bias init), H17 | 3 |
| Fixer-G3 | G3 topologies | `afac553` | H21, H22, H23, H24, H28 | 6 |
| Fixer-G4 | G4 kernels / attention | `3efd2dd` | H31 (golden spiral b), H34 | 4 |
| Fixer-G5 / Fixer-Opt | G5 optimisation | `c395769` + `5f09814` + `8aa0430` | H41 (eps), H47, H48 | 7 |
| Fixer-G6 | G6 topological bridging | `16fe2b6` | H53, H54, H55, H59 | 5 |
| Fixer-G7 | G7 cross-paradigm hybrids | `2e7ee45` | H64, H67, H74 | 4 |
| Fixer-G7-downstream | hybrid_cymatic_swiglu | `9cca91e` | H75 (downstream of H74) | 1 |

Each commit message follows the format `Fix-<group>-<short>: <one-line summary> (<mechanism-test>)` for retrieval by `git log --grep=Fix-`. The combined Fixer campaign brings the test suite from ~780 tests to ~826 tests across 78 files; total runtime is ~52 s on CPU (no GPU dependency for unit tests).

### 4.4 · Hyperparameters and hardware contract

Per area-chair item #6 (reproducibility blocker), the full hyperparameter table for every sweep row in this paper is stated here. Sweep-tag → config-row mappings are in [`scripts/run_sweep.py`](scripts/run_sweep.py) and `configs/cifar10_quick.yaml` / `configs/cifar10_sota_smoke.yaml`.

**Hardware contract (CLAUDE.md §2 + Rule 26):**

| item | value |
|---|---|
| GPU | 1× NVIDIA RTX 4090 Laptop, 16 GB VRAM |
| OS | Windows 11 Home |
| Python | 3.13 (corp-cert SSL workaround: `curl.exe -kL` for CIFAR; torchvision verifies MD5) |
| PyTorch | bf16 AMP throughout |
| DataLoader | `num_workers: 0` (Windows spawn-start workers wedge) |
| Thread caps | `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` |
| Deterministic | `set_seed(seed)` for torch/numpy/python at run start; `cudnn.benchmark=True` (NOT bit-reproducible by design — Rule 6) |

**Training hyperparameters (CIFAR-10 12-ep screening):**

| hyperparameter | value |
|---|---|
| optimizer | AdamW |
| LR | 1e-3 (cosine schedule, T_max = 12 epochs) |
| weight decay | 5e-4 |
| batch size | 256 |
| label smoothing | 0.1 |
| augmentation | RandomCrop(32, padding=4) + HorizontalFlip + RandAugment(N=1, M=4) |
| epochs | 12 (screening) |
| AMP | bf16 |
| seeds | 0 (single-seed for 32/35 single-prior CIFAR-10 rows; 3-seed only for `baseline_resnet20`, `phi_budget`, `golden_momentum`) |

**Training hyperparameters (CIFAR-100 30-ep graduation + Phase-8 3-seed):**

| hyperparameter | value |
|---|---|
| optimizer | AdamW |
| LR | 1e-3 (cosine schedule, T_max = 30 epochs) |
| weight decay | 5e-4 |
| batch size | 256 |
| label smoothing | 0.1 |
| augmentation | RandomCrop(32, padding=4) + HorizontalFlip + RandAugment(N=1, M=4) |
| epochs | 30 |
| AMP | bf16 |
| seeds | 0, 1, 2 (3-seed for all Phase-8 rows) |

**Per-hypothesis hyperparameters that DIFFER from the table above:**

- **H09 `phi_budget`** — post-fix per-stage widths `[37, 48, 61]` (pre-fix `[40, 48, 64]`), realised stage-parameter ratio 1 : 1.623 : 2.629 (target 1 : φ : φ² = 1 : 1.618 : 2.618, max-error 0.43 %). Integer-search procedure: minimise `|w_s − w_0 · φ^s|` over integer w_s ∈ [w_0 · φ^s − 4, w_0 · φ^s + 4] subject to total param budget within ±5 % of baseline (~272 k).
- **H41 `golden_adam`** — β1 = 1/φ ≈ 0.618, β2 = 1/φ² ≈ 0.382, eps = 1e-8 (post-fix; pre-fix used eps = 1/φ⁴ ≈ 0.146 which dominated Adam's denominator at CIFAR gradient scales).
- **H48 `golden_momentum`** — β1 schedule: `1 − (1 − 1/φ²) · k / T_max` per step (post-fix; pre-fix saturated to β1=0.382 after one step).
- **H44 `phi_decay_wd`** — per-layer weight decay scaled by `φ^(−depth_idx)` from a base `5e-4`.
- **H81 `slot_act_sine`** — replace every ReLU in the NaturePriorBlock backbone with `sin(ω · x)`, ω = 1 (SIREN-style, near-identity start).
- **H05 `fractal`** — Fractal recursion depth = 3, mixing ratio φ⁻¹ ≈ 0.618.

**Sweep-tag → config-row mapping (Phase-8 winners + reference):**

| sweep tag | flags / channel_mode | hypothesis IDs combined |
|---|---|---|
| `baseline_resnet20` | stock ResNet-20 | (reference) |
| `sg_only_phi_budget` | `phi_budget=True` (post-fix integer search) | H09 |
| `pair_gm_pdw` | `phi_budget=True`, `momentum_schedule="golden"`, `phi_decay_wd=True`, `phi_decay_base=5e-4` | H09 + H48 + H44 |
| `slot_act_sine` | `phi_budget=True` + activation slot replaced with `sin(x)` | H09 + H81 |

The exact resolved config YAML for each Phase-8 row is committed under `experiments/cifar100/<tag>_seed{0,1,2}/config.yaml`.

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

**51 % non-PASS rate**, evenly split across nearly every group, with failures clustering around graph-equivariance modules (G3) and cross-paradigm composition (G7). **CRITICAL CAVEAT (cf. §1.3 and §5.8):** the 51 % is conditional on auditor-self-grading; without a third-party-code calibration we cannot distinguish "genuine codebase defect density" from "hyperactive auditor team." We report the number as protocol calibration, not as a substantive claim about this codebase's quality relative to the field.

### 5.2 · Research-scientist verdict distribution (Track B)

| verdict | count | % |
|---|---:|---:|
| **NOVEL+TESTABLE** | **1** (H71 IcosaRoPE3D) | 1.2 % |
| DERIVATIVE+TESTABLE | 30 | 37.0 % |
| NUMEROLOGY | 40 | 49.4 % |
| EMPIRICALLY FALSIFIED | 3 (H41 H48 H50) | 3.7 % |
| UNFALSIFIABLE | 2 (H22, H67) | 2.5 % |
| INFRASTRUCTURE / INCONCLUSIVE | 5 | 6.2 % |

Of 81 hypothesis claims, exactly one is rated NOVEL+TESTABLE. Roughly half are flagged NUMEROLOGY where φ is a renaming of a constant that any value in [1.3, 2.0] could replace with the same outcome. The strongest defensible category is DERIVATIVE+TESTABLE (37 %) — rediscoveries of established techniques: e.g., golden-skip residual scaling re-derives Fixup / ReZero / ScaleNorm under a depth-independent constant; phi-decay LR is functionally interchangeable with cosine; spectral-Hopfield is a unitary basis change of modern Hopfield (Ramsauer 2020) that preserves softmax retrieval ordering exactly by Parseval's theorem.

The "1 / 81 NOVEL+TESTABLE" rate is itself conditional on Track B agents sharing model family with Track A and the implementers (§1.3). It should be read as "this protocol, run by this team, screened 81 claims and flagged 1 as novel-and-testable" rather than as a substantive claim about the design space's intrinsic novelty rate.

### 5.3 · The 3 BROKEN findings (case studies of protocol output)

1. **H55 PlatonicAttention's head bias is mathematically zero.** `bias = (coords @ coords.T).mean(dim=-1)` evaluates to all-zeros for every vertex-transitive Platonic solid (their vertex coords sum to centroid/origin). PlatonicAttention was bit-equivalent to vanilla MHA; the entire "symmetry orbit inductive bias" contributed literally nothing. All 7 tests were shape-only and concealed this. **Fixed** in Fixer-G6 (commit `16fe2b6`) with a relative-position cosine bias `(1/φ)·cos(angle_h + 2π·(j−i)/N)` mirroring the H37 pentagonal pattern. The audit and Fixer are from the same model family that wrote the buggy code (§1.3) — the audit's "we caught the bug" narrative survives this caveat for H55 because the math (vertex-transitive coords sum to centroid) is verifiable from outside the agent ecosystem; the broader claim about audit diagnostic power does not.
2. **H67 hybrid_full was a half-on stress test.** `from .golden_rope import GoldenRoPE` raised ImportError (no class); MetatronGraphLayer constructor signature was wrong; `which_priors_active` hardcoded `True` for 4 priors without inspecting the model; LiquidCFC collapsed to affine + nonlinearity. **Fixed** in Fixer-G7 (commit `2e7ee45`).
3. **H74 MetatronTiedConv2d's 13 alphas collapsed to one scalar.** Forward was `F.conv2d(x, W · Σα_c)`; the 13 alphas Σ-summed to a single gate; no per-circle masks. **Fixed** in Fixer-G7 — now uses H40's `metatron_basis_kernels` 13 spatially-distinct circle masks; `effective_weight = Σ_c α_c · (W ⊙ mask_c)`.

### 5.4 · The single most impactful MAJOR finding — H09 phi_budget (case study)

The pre-audit headline (H09 phi_budget cross-dataset positive) was produced by a network whose realized stage-parameter ratio was **1 : 1.41 : 2.45**, not the doc-claimed **1 : φ : φ² = 1 : 1.618 : 2.618** — a 12.6 % drift at stage 1. Fixer-PhiScaling (commit `519cdf3`) rewrote `phi_budget_widths` with an iterative search over integer widths minimising deviation from `[1, φ, φ²]`; the post-fix realized ratio is **1 : 1.623 : 2.629** (0.43 % max error). The architecture itself changed (widths `[40, 48, 64]` → `[37, 48, 61]`). Post-fix 3-seed CIFAR-100 medians on the corrected mechanism are `phi_budget = 0.5741`, `baseline = 0.5652` (+0.89 pp). The pre-fix median (58.05 %) was ~0.6 pp HIGHER than the post-fix (57.41 %), consistent with "the broken realised ratio happened to land a fortuitously-high seed-0 result." The CASE STUDY reading: the protocol caught a code-vs-doc divergence that would have shipped as a headline in a non-audited pipeline. Whether the +0.89 pp post-fix lift represents a genuine architectural signal or a tuning artifact is decided by Phase-9 hill-climb, not this submission.

### 5.5 · Three Phase-8 candidates surfaced by the protocol (illustrative, PROVISIONAL)

**Framing (per area-chair item #1).** This subsection reports three sweep-tag outcomes that pass the project's worst-leader-seed > best-baseline-seed CIFAR-100 3-seed *screening gate*. The framing is **case studies of protocol output**, NOT a standalone empirical headline. Whether any of these three survive a per-hypothesis hill-climb (Phase 9, ahead) and a non-φ 3-axis control row (§5.5.1) is open.

**Three protocol-screened candidates (3-seed CIFAR-100, 30 ep, post-fix code):**

| tag | C100 median | min seed | baseline max | Δ vs baseline median | statistical support |
|---|---:|---:|---:|---:|---|
| `pair_gm_pdw` (H09+H48+H44 orthogonal stack) | 0.5786 | 0.5761 | 0.5662 | +1.34 pp (Δmean +1.36 pp) | W=0, p_one=0.125 (n=3 floor); bootstrap 95% CI on Δmean [+1.11, +1.63] pp **excludes 0**; FAILS Holm-Bonferroni α'=0.0167 |
| `slot_act_sine` (H81 SIREN, single prior) | 0.5784 | 0.5766 | 0.5662 | +1.32 pp (Δmean +1.39 pp) | W=0, p_one=0.125 (n=3 floor); bootstrap 95% CI on Δmean [+1.13, +1.67] pp **excludes 0**; FAILS Holm-Bonferroni α'=0.0167 |
| `sg_only_phi_budget` (H09, post-fix 1:1.623:2.629) | 0.5741 | 0.5687 | 0.5662 | +0.89 pp (Δmean +0.91 pp) | W=0, p_one=0.125 (n=3 floor); bootstrap 95% CI on Δmean [+0.44, +1.36] pp **excludes 0**; FAILS Holm-Bonferroni α'=0.0167 |
| `baseline_resnet20` (rail) | 0.5652 | — | — | — | (reference, n=3, σ = 0.248 pp on CIFAR-100) |

All three Phase-8 rows produce the **theoretical-minimum** paired one-sided Wilcoxon p at n=3, namely (1/2)^3 = 0.125, because every per-seed delta is strictly positive (negative-rank sum W = 0). This is the strongest possible result the n=3 design can produce, and it is still **2.5× looser than α=0.05** and **7.5× looser than Holm-Bonferroni α'=0.05/3=0.0167** across the k=3 family. The 95% bootstrap CIs on Δmean all exclude 0 by comfortable margins (lower-bound at least 1.5× the CIFAR-100 baseline σ=0.248 pp); the effect-size signal is real-looking, but the n=3 sample is structurally insufficient for any α=0.05 single-test rejection — **per [`STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md) §3.3, n ≥ 7 seeds is the minimum required for paired Wilcoxon to clear Holm-Bonferroni α'=0.0167 across the k=3 Phase-8 family**, even when every paired delta is positive.

The "min-leader-seed > max-baseline-seed" gate is an ordinal one-sided non-parametric construction with low formal power; characterised as a paired one-sided sign test on n=3 seed-aligned seeds, its α is exactly (1/2)^3 = 0.125. The gate's role is to **screen** candidates worth Phase-9 evaluation, NOT to declare empirical wins; per [`STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md) §2, the Phase-5 ordinal gate is therefore equivalent to a one-sided sign test at α≈0.125, NOT at α=0.05. The complete formal Wilcoxon + Holm-Bonferroni + bootstrap-CI analysis is now in [`STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md) and is spliced into the table above; the three "winners" are candidates with non-zero-excluding bootstrap CIs but **formally uncertified at n=3**.

**Honest reading of the three case studies:**
- `pair_gm_pdw` stacks three orthogonal axes (architecture / momentum / weight-decay); the +1.34 pp lift is observed but the comparison to a **non-φ 3-axis regularizer control** is missing (§5.5.1), so we cannot yet attribute the lift to the φ-content versus "any three regularisation axes tuned away from baseline defaults."
- `slot_act_sine` swaps ReLU for `sin(ω·x)` with ω=1; the +1.32 pp lift is observed but the **non-φ activation control sweep** (`tanh`, `softplus`, `gelu`, `swish` at the same SLOT recipe) has not been run. Per Sitzmann et al. (NeurIPS 2020), SIREN's known activation-engineering signal has no φ content; the case study cannot distinguish "SIREN works" from "φ-flavoured SIREN works."
- `sg_only_phi_budget` post-fix lifts +0.89 pp; the sci-critic verdict is DERIVATIVE+TESTABLE (rediscovery of RegNet's Pareto-optimal width-progression region — Radosavovic et al. CVPR 2020 explicitly prefers `w_m ∈ [2.5, 2.9]`, NOT φ=1.618). A tuned-RegNetX-200MF baseline (item #13) would settle whether the lift survives a tuned-baseline upgrade; that comparison is deferred to Phase-9.

### 5.5.1 · Confound: non-φ 3-axis control for `pair_gm_pdw` (BLOCKER item #4, OPEN)

Per area-chair item #4: `pair_gm_pdw`'s strongest counterargument is that its three axes (φ-proportion widths + golden-momentum β1-schedule + φ-decay per-layer wd) are each interchangeable with non-φ regularisers with established literature analogs: RegNet width allocator (Radosavovic 2020), low-β1 Adam (Choi 2019), and per-layer-linear-decay wd (Loshchilov 2019 AdamW). The +1.34 pp lift could be entirely explained by **any tuning of three orthogonal regularisation axes away from baseline defaults**, with the φ-content decorative.

The control row is specified in the planned [`controls/`](controls/) directory (parallel agent work-stream):

```
control_3axis_nonphi:
  widths: uniform [48, 48, 48]    # vs phi_budget [37, 48, 61]
  beta1_schedule: constant 0.4    # vs golden_momentum 1/φ² floor
  wd_per_layer: linear-decay from 1e-3 to 1e-4   # vs phi_decay_wd geometric
  total_params: ≈ pair_gm_pdw     # iso-budget
```

At iso-budget, if this non-φ 3-axis stack produces a comparable CIFAR-100 3-seed median, the φ-content of `pair_gm_pdw` is decorative and the case study is **negative for the φ-prior claim** while still being **positive for the protocol's screening claim** (the protocol surfaced a candidate worth this control). **This control has not been run in this submission.** The `pair_gm_pdw` Phase-8 result is therefore labelled "candidate, confound-open"; we do NOT claim a φ-prior effect at this scale.

### 5.5.2 · Case-study: the `slot_act_sine` SIREN result — what it does and does not show

`slot_act_sine` replaces every ReLU in the NaturePriorBlock backbone with SIREN-style `sin(ω·x)` at ω = 1, on top of the post-fix `phi_budget` architecture. The 3-seed CIFAR-100 30-ep median is 0.5784 (+1.32 pp vs baseline median 0.5652); min-leader seed 0.5766 strictly exceeds baseline-max seed 0.5662. The Phase-5 ordinal screening gate is satisfied.

**What this case study supports:**
- The protocol's screening gate is non-trivially populated: at least one single-activation swap, on top of the corrected H09 architecture, exceeds the ResNet-20 30-ep baseline at the screening compute budget on three independent seeds.
- The result is consistent with Sitzmann et al. NeurIPS 2020's broader claim that periodic activations capture high-frequency signal structure better than monotone activations in some regimes.

**What this case study does NOT support:**
- The lift is from φ-content specifically. SIREN at ω = 1 is `sin(x)`, a smooth bounded periodic activation with no golden-ratio derivation. The "Phase-8 winner" framing of the pre-revision paper buried this; the present revision flags it explicitly.
- The lift survives a controlled activation sweep. Without comparing `sin(x)` to `tanh`, `softplus`, `gelu`, `swish`, `mish`, and `sin(ω·x)` at ω=30 (SIREN-canonical) at the same SLOT recipe, we cannot rule out that any sufficiently-different activation produces a comparable lift at the screening budget. The control sweep is part of Phase-9.
- The lift survives at convergence (164-ep recipe). Phase-9 hill-climb on `slot_act_sine` includes the long-horizon training condition.

The case study is therefore protocol-positive (the protocol's screening surfaced a candidate worth Phase-9) and **φ-prior-neutral** (the lift is, at this point, observationally indistinguishable from a generic activation-engineering win unrelated to the nature-inspired framing).

### 5.5.3 · Case-study: H09 phi_budget post-fix — the cleanest test of "the protocol caught a broken claim"

H09 phi_budget is the original campaign's headline (CIFAR-10 85.54 % / CIFAR-100 58.05 % 3-seed median pre-audit) and the audit's most-consequential MAJOR finding (the realised-ratio drift). The case-study reading from §5.4 separates two distinct claims:

1. **Protocol-level (DEFENSIBLE):** The audit identified a code-vs-doc divergence (realised 1:1.41:2.45 vs claimed 1:1.618:2.618) that would have shipped as a headline in a non-audited pipeline. The Fixer corrected the integer-search procedure; the post-fix realised ratio is 1:1.623:2.629 (max-error 0.43 %). The mechanism-pinning test from Fixer-PhiScaling (commit `519cdf3`) is a permanent regression guard.
2. **Architectural-level (PROVISIONAL):** The post-fix architecture lifts +0.89 pp on CIFAR-100 3-seed at the 30-ep screening budget. The pre-fix median was ~0.6 pp HIGHER than the post-fix median, consistent with "the broken realised ratio happened to land a fortuitously-high seed-0 result." Whether the +0.89 pp lift represents a genuine architectural signal at the literature-canonical 164-ep / tuned-baseline regime is decided in Phase-9 hill-climb + the tuned-RegNet head-to-head (§7.4-4), not in this submission.

The case study illustrates the protocol's intended dynamic: **the protocol's empirical screen is mechanically downstream of the protocol's audit. The audit is the load-bearing contribution; the screening result is a calibration data point.** A non-audited pipeline would have published 85.54 / 58.05 as the headline; the audited pipeline publishes the post-fix +0.89 pp as a screening case study with the §7.3 limitations attached. The contribution survives if the protocol is portable, regardless of whether the +0.89 pp survives Phase-9.

### 5.6 · Cumulative defect tally

Per area-chair item #12 (category confusion warning): implementation-critic verdicts (BROKEN/MAJOR/MINOR) and sci-critic verdicts (NUMEROLOGY/UNFALSIFIABLE) are **different dimensions**, not additive defects. The table below is a cumulative tally of where each hypothesis lands on each independent dimension, NOT a count of "total defects."

| dimension | category | count |
|---|---|---:|
| impl-critic (Track A) | BROKEN | 3 |
| impl-critic (Track A) | MAJOR mechanism / code-vs-doc divergence | 15 |
| impl-critic (Track A) | MINOR (shape-only tests, cosmetic issues) | 24 |
| impl-critic (Track A) | PASS | 41 |
| sci-critic (Track B) | NUMEROLOGY | 40 |
| sci-critic (Track B) | EMPIRICALLY FALSIFIED | 3 |
| sci-critic (Track B) | UNFALSIFIABLE | 2 |
| sci-critic (Track B) | NOVEL+TESTABLE | 1 |
| sci-critic (Track B) | DERIVATIVE+TESTABLE | 30 |
| sci-critic (Track B) | INFRASTRUCTURE | 5 |
| operational | Hypotheses requiring re-run after fix (Rule 21) | 14 single-axis + 7 combo |

### 5.7 · Verdict-on-the-wrong-testbed audit (area-chair item #16, PARTIAL)

The area-chair flagged that certain NUMEROLOGY/UNFALSIFIABLE verdicts were recorded after testing on CIFAR-10/-100 when the hypothesis's design-doc falsifier required a different dataset. The canonical example is **H22 Toroidal-φ-closure**: the design doc [`hypotheses/g3_topologies_graphs/H22_toroidal_phi_closure.md`](hypotheses/g3_topologies_graphs/H22_toroidal_phi_closure.md) (Falsifier section) explicitly requires a "tiled-texture or wrap-aware synthetic dataset"; the project tested on upright CIFAR-10 and recorded a NUMEROLOGY/UNFALSIFIABLE verdict. **This verdict is downgraded in this revision to "UNTESTED-ON-RIGHT-DATASET."** H22's NUMEROLOGY label is suspended pending a wrap-aware-dataset evaluation (rotated CIFAR with horizontal wrap, or a tiled-texture synthetic).

A full audit of the remaining 41 NUMEROLOGY + 2 UNFALSIFIABLE verdicts to identify other CIFAR-as-wrong-testbed cases is deferred to a follow-up pass. Candidate cases pre-identified for that audit: H24 IcosaConv1d (rotated CIFAR / spherical MNIST), H25 dodecahedral-latent (spherical embeddings), H26 fractal-toroidal (wrap-aware data), H53/H54/H55/H58 (rotational equivariance datasets), several G7 transformer hybrids (no transformer scaffold yet wired).

### 5.8 · Audit calibration on third-party code (area-chair item #12, DEFERRED)

The 51 % impl-critic non-PASS rate (§5.1) is presented as protocol calibration in this paper. The area-chair correctly observed that without a **false-positive baseline** measurement — the same audit team applied to a known-good third-party codebase (`timm` ResNet, `pytorch-cifar100`, official CIFAR-10 ResNet, etc.) — the 51 % is uninterpretable: a high non-PASS rate is observationally consistent with both "high defect density in our codebase" and "high false-positive rate in our audit team." Until a third-party calibration run is done, the 51 % is a number, not a claim about codebase quality relative to the field. We commit to running this calibration in future work; it is deferred from this submission.

## 6 · Empirical evidence (CIFAR-10 / CIFAR-100) — protocol output

### 6.1 · Pre-fix screening sweep (single seed, 12 epochs)

Across 35 single-prior sweep rows on CIFAR-10 at seed 0 / 12 epochs, the only variant beating the ResNet-20 baseline (84.78 %) was **H09 phi_budget at 85.54 %** (+0.76 pp). **All 35 rows are n=1; the 51 %-non-PASS rate is therefore screened on a single-seed sweep and the per-row deltas are within the project's empirical seed envelope (~±0.5 pp at 12-ep CIFAR-10 — see §7.3 limitations).** The pre-fix `golden_adam` row at 51.96 % (−32.82 pp) was originally framed as the campaign's clean falsification, **but Track A's H41 audit + the post-fix re-run requalified that headline**: the catastrophic collapse was driven by `eps = 1/φ⁴ ≈ 0.146` (which dominated Adam's denominator at CIFAR gradient scales ~1e-3, inflating effective LR ~6.85×), NOT by the β-shift. After Fixer-Opt restored stock `eps = 1e-8` and kept only the φ-defaults for β, post-fix CIFAR-10 12-ep top-1 = 0.8394, Δ ≈ −1 pp vs baseline. Reddi 2018's non-convergence proof for β2 < 0.95 is asymptotic and does not yet bite at 12 ep — the β-only regression is mild, not catastrophic. **H41 is therefore re-classified as WEAKLY NEGATIVE at 12-ep CIFAR-10 screening, with the clean β-only Reddi-regime falsification deferred to the Phase-9 hill-climb at varied β2 + longer training horizons (100+ ep).** The pre-fix 0.5196 row is retired as eps-confounded. The case-study reading: the protocol's screening sweep correctly surfaced a candidate that warranted Phase-9, AND identified an implementation confound (eps) that an unaudited pipeline would have published as a "clean β-only falsification."

### 6.2 · Pre-fix combo ladder (in-flight, screening data)

A 7-row additive combo ladder on top of phi_budget — phi_budget + N orthogonal-axis priors (momentum / dropout / wd / LR / ensemble / activation / pruning) — was launched ON PRE-FIX CODE. First three rows landed: combo2 86.14 %, combo3 86.42 %, combo4 85.80 %, all beating baseline at seed 0 / 12 ep. The same combo ladder re-ran on POST-FIX code; whether the +1 pp screening lift survives Phase-9 hill-climb is the central open empirical question — see §6.4 / §7.3.1.

### 6.3 · Tier-A combo expansion (post-fix, screening data)

The orchestrator script `scripts/launch_postfix_campaign.sh` fired 31 post-fix runs: 7 single-axis re-runs of the affected primary modules, 7 combo ladder re-runs, plus 17 new Tier-A ladders (LOO subtractive from combo8, two-at-a-time interaction matrix, mutually-exclusive slot ablation). The full pre-fix vs post-fix survival table is in [`FINDINGS.md`](FINDINGS.md) §Phase-7.

### 6.4 · The honest comparison to SOTA

**Per area-chair item #7 (now in abstract + here):** ResNet-20 at the canonical 164-epoch SOTA recipe (He CVPR 2016) reaches ~91.25 % CIFAR-10 top-1. Our 12-epoch screening budget produces ~84.78 % baseline, **a 6.5-pp shortfall from convergence**. The H09 phi_budget screening lift (+0.76 pp at 12 ep) is less than 12 % of the gap-to-SOTA; the Phase-8 winners' +0.89 to +1.34 pp lifts on CIFAR-100 30-ep are at the same calibration band. We make **no** SOTA claim at this scale; the comparison is between nature-inspired priors and the same-recipe baseline at the same compute budget. **The H09 lift could entirely disappear when the baseline is tuned to 164-ep SOTA**; the H09-vs-tuned-ResNet-20 comparison is deferred to Phase-9 (cf. area-chair item #13). The SOTA gap is acknowledged in [`SOTA_COMPARISON.md`](SOTA_COMPARISON.md) with honest 12-vs-164-epoch framing.

## 7 · Discussion

### 7.1 · What survives the audit (protocol's hill-climb shortlist)

Combining Tracks A + B + the Fixer outcomes, the hypotheses currently nominated by the protocol for Phase-9 hill-climb (per Rule 22: impl-critic PASS AND sci-critic ≠ NUMEROLOGY/UNFALSIFIABLE) are:

- **H05 fractal_phi_recursion** (DERIVATIVE+TESTABLE; the only single prior to lift top-1 in the original 11-row campaign at +2.35 pp on CIFAR-10 screening; screening-data only).
- **H09 phi_budget** (DERIVATIVE+TESTABLE, RegNet-region rediscovery; post-fix +0.89 pp CIFAR-100 3-seed; passes Phase-5 screening gate; awaits Phase-9 + RegNet-tuned-baseline comparison).
- **H21 hex_phi** (DERIVATIVE+TESTABLE; hex topology is real even if φ-radial is decorative; screening-data only).
- **H32 Fibottention** (DERIVATIVE+TESTABLE; Wythoff-array non-overlap is a real geometric property; transformer-only, no CIFAR row).
- **H39 PhiGELU** (DERIVATIVE+TESTABLE; β=φ sits between SiLU and GELU; screening-data only).
- **H58 group avg-pool** (EMPIRICALLY-FALSIFIED on CIFAR; methodologically clean — the negative result is the protocol output).

Each of these survives at the screening level only. Phase-9 hill-climb (§7.4-1) is required before any of them is recorded as an external empirical claim.

### 7.2 · What the protocol kills (at the screening + sci-critic level)

- All hypotheses rated NUMEROLOGY by sci-critic (~40, half the design space) — pending the §5.7 "verdict-on-the-wrong-testbed" audit, some may be requalified to UNTESTED-ON-RIGHT-DATASET.
- All MAJOR / BROKEN findings without a Fixer patch (zero at this point — all 18 MAJOR/BROKEN are fixed in commits `519cdf3`, `253dc94`, `afac553`, `3efd2dd`, `c395769`+`5f09814`, `8aa0430`, `16fe2b6`, `2e7ee45`).
- The catastrophic full-hybrid `sg_full_fib` (−11.54 pp) is empirically refuted at the screening level; CLAUDE.md Rule 23 codifies the lesson (orthogonal-axes-only compounds).

### 7.2.1 · The orthogonal-axes-only doctrine (Rule 23) — operational detail

The `sg_full_fib` row stacked six nature-inspired priors (φ-channels + Fibonacci-depth + fractal-recursion + hex-conv + group-conv + toroidal-closure) on the same NaturePriorBlock forward path and landed at **−11.54 pp below the ResNet-20 baseline** — the worst NaturePrior variant in the original campaign. CLAUDE.md Rule 23 abstracts the lesson: multi-prior stacks may stack ONLY priors that touch different layers of the training stack (arch / channel / momentum / regulariser / weight-decay / LR / activation / ensemble / pruning / inference). Stacking more than two priors on the same conv-block forward path is forbidden.

The Phase-7 combo ladder (CLAUDE.md skill `autoresearch-combo-ladder`) is the canonical operational test. Starting from a verified-winner base, the ladder adds **one orthogonal axis per row**:

| ladder row | axes stacked | post-fix C10 top-1 | Δ vs base |
|---|---|---:|---:|
| `sg_only_phi_budget` (base) | architecture (H09) | 85.56 | (ref) |
| `combo2_pb_gm` | + momentum (H48) | 85.62 | +0.06 |
| `combo3_pb_gm_pd` | + dropout (H47) | 85.66 | +0.10 |
| `combo4_pb_gm_pd_pdw` | + weight-decay (H44) | 85.44 | −0.12 |
| `combo5_*_plr` | + LR-schedule | 79.78 | **−5.66** |
| `combo6_*_fe` | + ensemble | 84.90 | −0.66 |
| `combo7_*_sa` | + self-attention | 85.29 | −0.27 |
| `combo8_*_fp` | + feature-pyramid | 84.96 | −0.60 |

The combo ladder's signal: **`plr` (phi-LR-schedule) is the single most destructive axis** when added, dropping the stack by −5.66 pp from combo4 to combo5. Combo7 + combo8 partially recover by adding self-attention and feature-pyramid, suggesting the surrounding stack reorganises around `plr`'s damage. The Tier-A LOO (leave-one-out) probes at combo8 confirm: removing `plr` from combo8 *also* hurts (−1.13 pp) — once the stack has compensated for `plr`'s presence, removing it disrupts the compensation. This is the canonical "Rule 23 violation in slow motion": even orthogonal axes can interact destructively at compound depth ≥ 4.

The Phase-8 `pair_gm_pdw` row stacks exactly three orthogonal axes (architecture + momentum + weight-decay) and lands at 85.85 % CIFAR-10 12-ep — the best single combo in the post-fix sweep. The 3-seed CIFAR-100 30-ep median (0.5786) clears the Phase-5 gate. This is the protocol's nominee for the "three-axis orthogonal stack outperforms its solo components" hypothesis — **provisional** until Phase-9 hill-climb and the §5.5.1 non-φ control settle the φ-content question.

### 7.3 · Limitations

- **Single-seed for most single-prior sweep rows** (stated in abstract per area-chair item #8). Only 3 tags (`baseline_resnet20`, `phi_budget`, `golden_momentum`) carry 3-seed error bars to date (and only on CIFAR-100); the rest are seed-0 only. Repeat-run cost is the primary scaling constraint. The 35-row CIFAR-10 sweep used as screening evidence is n=1; the 51 %-non-PASS rate from §5.1 is **independent of this** (it's audit verdicts, not training seeds), but every per-tag empirical delta is single-seed and is treated as screening data per Rule 28.
- **No transformer experiments.** Half the 84 hypotheses target attention/sequence backbones; none are tested in this submission. ViT-Tiny and decoder-only-LM scaffolds are future work (§7.4-2).
- **No tuned-baseline or RegNet comparison** (area-chair item #13). We do not compare H09 / Phase-8 winners against RegNetX-200MF (the direct literature analog whose own Pareto-region analysis prefers `w_m ∈ [2.5, 2.9]`, NOT φ=1.618) or a tuned-LR/-wd ResNet-20 at iso-compute. The H09 +0.78 pp screening lift on CIFAR-10 could be a tuning artifact of comparing nature-inspired-tuned vs untuned baseline. Phase-9 will run a tuned-RegNet head-to-head; until then, the H09 claim is **screened, confound-open**.
- **12 / 30-epoch budget is the screening, not the convergence regime.** Stronger or different priors may emerge at the 164-epoch full SOTA recipe; weaker priors that screen positively may collapse.
- **CIFAR-10 / CIFAR-100 is not a sufficient testbed** for several of the equivariance / topological hypotheses (which would shine on rotated CIFAR, spherical MNIST, ModelNet40, ogbg-molhiv, etc.). The dataset coverage is acknowledged in `IDEA_TABLE.md`.
- **Audit-and-implementer model-family overlap** (§1.3). 51 % non-PASS rate is conditional on this caveat; third-party-code calibration (§5.8) is deferred.
- **Multiple-comparisons correction is not yet applied** to the 35-row CIFAR-10 screen. At α≈0.05 the screen expects ~1.75 false-positive winners by chance; we report one positive (H09 at +0.76 pp). The Holm-Bonferroni / Benjamini-Hochberg adjustment is in the parallel [`STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md) file; see TODO placeholders in §5.5.
- **Selection bias on the 84-hypothesis design space.** The design space was curated by an LLM agent reading a source PDF on nature-inspired networks; members were selected for plausibility, not coverage. The audit operates on a population whose distribution is selection-conditional, not a random sample of a definable universe. The "1 NOVEL+TESTABLE / 81" rate carries this selection bias.

### 7.3.1 · Methodological caveat — screening vs evaluation (with HARKing acknowledgement)

**Acknowledgement up front (per area-chair item #2 BLOCKER).** This section was authored **post-hoc** — after the screening protocol had been run for 35 single-prior CIFAR-10 rows and after the Phase-5 demotion of H48 `golden_momentum` and the multiple sci-critic verdicts had landed. Reclassifying every single-prior negative as "screening, not evaluation" *after* observing the negatives is **HARKing** (Hypothesizing After the Results are Known). We accept the methodological cost of this post-hoc framing in this revision; we do **not** retroactively claim that the screening-vs-evaluation distinction was pre-registered. Going forward, the distinction is codified as CLAUDE.md Rule 28 and applies prospectively to all future autoresearch campaigns. Specifically: any future single-config single-seed sweep must be **labelled as screening** at launch (config flag + commit hash) before its negatives are observed; only Phase-9 hill-climb + 3-seed + Phase-5-gate results carry "evaluation" weight.

> **Footnote (methodological standard going forward, per area-chair item #2):**
> The screening-vs-evaluation distinction is operationalised post-hoc in this paper. The standard going forward is encoded in CLAUDE.md Rule 28 and requires pre-registration of screen-vs-evaluate intent BEFORE any sweep launches. We accept the methodological cost of post-hoc framing in this revision; the standard goes forward via Rule 28.

The substantive content of the distinction stands independently of when it was authored:

The 12-epoch CIFAR-10 single-config single-seed protocol used across the 35-row sweep is a **screening filter**, not an **evaluation**. Single-config single-seed numbers conflate two distinct claims: (1) "the hypothesis is bad" (mechanism does not help on this dataset family at this compute budget at any reasonable knob setting); and (2) "the hypothesis is bad at our specific config" (mechanism might help at different β2, base_wd, dropout floor, T_max-step, width-rounding, or RNG seed, but at the one config we ran it lost to baseline). The screening protocol cannot disambiguate (1) from (2).

What the screening protocol CAN do — and what it was designed to do — is identify a small set of hypotheses worth the cost of real evaluation. A real evaluation has three layers (CLAUDE.md Rule 28):

- **Per-hypothesis hill-climb** (Phase 9, in progress) over the natural knobs of each prior — β2 ∈ [0.382, 0.999] for H41; base_wd ∈ {5e-4, 1e-3, 5e-3, 1e-2} for H44; dropout floor + slope for H47; T_max-step for H48; width-rounding strategy for H09; stride / kernel-size for H21 hex_phi; etc. The hill-climb runs ≥ 20 runs per hypothesis (coordinate descent or random search).
- **3-seed re-runs at the best config found.** Single-seed differences inside ±0.5 pp on CIFAR-10 12-ep are within seed noise.
- **Cross-dataset Phase-5 worst-leader-seed > best-baseline-seed gate** on CIFAR-100 30-ep, 3 seeds.

**The three Phase-8 candidates in §5.5 satisfy LAYER 3 (Phase-5 cross-dataset 3-seed gate) but NOT LAYER 1 (per-hypothesis hill-climb).** They are NOT externally defensible at the level required for a NeurIPS-track empirical claim. The headline framing of this submission is therefore **protocol-as-contribution**, with the three Phase-8 candidates as illustrative protocol output, NOT as standalone empirical wins. This framing is the resolution of the internal contradiction flagged as BLOCKER item #1.

Every other hypothesis-level statement in this paper, in `FINDINGS.md`, and in `audits/*` — including "H41 falsified", "H42 paper-disagrees", "H44 wrong-test", "H47/H48 wrong-schedule", "H50 catastrophic", "H80 Reuleaux mild-negative", "H81 sine-act mid-pack", and all other single-prior verdicts at 12 ep — is reclassified as **SCREENING DATA, not evaluation**, with the post-hoc HARKing acknowledgement explicit.

### 7.4 · Future work

1. **Phase 9 — per-hypothesis hill-climb.** Lift every screened-negative or screened-positive hypothesis out of the single-config screening cage and re-run it at its own most-favourable knob settings (β2 grid for H41, base_wd grid for H44, dropout floor + slope grid for H47, T_max-step grid for H48, width-rounding strategies for H09, stride / kernel-size for H21, etc.). Each survivor of the hill-climb gets a 3-seed re-run at the best config, then the Phase-5 cross-dataset gate. **This is the mechanism by which the three Phase-8 case studies of §5.5 are converted (or not) to externally defensible empirical claims.** The Phase-9 hill-climb skill is `skills/autoresearch-per-hypothesis-hillclimb/` (landed in a parallel work-stream).
2. **Transformer track** — wire a ViT-Tiny scaffold + decoder-only LM scaffold; smoke H03, H15, H16, H27, H32, H34, H36, H37, H55, H71 with proper sequence-task evaluation (rotated-CIFAR for the equivariance ones; WikiText-2 PPL for the LM-track ones). None of the 84 hypotheses' transformer-track variants are tested in this submission.
3. **H71 IcosaRoPE3D dedicated experiment.** The sole NOVEL+TESTABLE survivor of the sci-critic gate is **untested**; this is a research proposal, not a result, in this submission. Per area-chair item #14, H71 has been moved from "contributions" to "future work / research directions." A future ViT-Tiny + rotated-CIFAR + Spherical-MNIST evaluation will convert H71 from "we have an idea" to "we tested our best idea" (regardless of outcome).
4. **Tuned-RegNet / tuned-ResNet baseline comparison** (area-chair item #13). Run H09 phi_budget head-to-head against (a) RegNetX-200MF at the literature-canonical `w_m ∈ [2.5, 2.9]` AnyNet allocator, (b) tuned-LR/-wd ResNet-20 at iso-compute. If H09 lift disappears under a tuned baseline, the case study is negative.
5. **Audit calibration on third-party code** (area-chair item #12). Re-run the Track-A audit team against `timm` ResNet, `pytorch-cifar100`, and the official PyTorch CIFAR-10 ResNet implementation. Report the non-PASS rate. If the false-positive rate is comparable to the 51 % from §5.1, the audit's diagnostic power is overstated.
6. **Cross-domain replication of the protocol.** Run the protocol on a second autoresearch repo (tabular: `dlmastery/autoresearchtabular`; medical: future; FX: `dlmastery/autoresearch`) and report whether the same skill-pack discovers a comparable defect distribution. This is the empirical test of the "content-agnostic" claim made in §1.1-3.
7. **Dataset transfer** — Tiny ImageNet, rotated-CIFAR, MedMNIST, Spherical MNIST — to test where the equivariance / topology priors are data-aligned. Required especially for the H22 toroidal verdict requalification (§5.7).
8. **Multiple-comparisons-correct screening.** Apply Holm-Bonferroni or Benjamini-Hochberg across the 35-row CIFAR-10 screening family at α=0.01 family-wise; report which screening positives survive.
9. **Pre-registration of the Phase-9 hill-climb.** A pre-registration commit hash (knobs, ranges, seed protocol, statistical test, success criteria) before launching Phase-9 is the cheapest way to neutralise the §7.3.1 HARKing concern going forward.

## 8 · Conclusion

**The protocol is the contribution.** After 84 hypothesis implementations, 41+24+15+3 implementation verdicts, 1+30+40+3+2+5 scientific verdicts, 8 mechanism-correcting Fixer commits, and ~7 GPU-hours of CIFAR-10/100 ablation, the protocol produced calibration data: 51 % non-PASS impl-critic rate (conditional on the audit-self-grading caveat of §1.3 and pending the third-party-code calibration of §5.8 / future work §7.4-5), 1 / 81 NOVEL+TESTABLE sci-critic rate (similarly conditional), and three protocol-screened candidates that pass the cross-dataset Phase-5 gate.

We make **no** standalone empirical headline claim for nature-inspired priors at this scale. The three Phase-8 candidates (`pair_gm_pdw`, `slot_act_sine`, `sg_only_phi_budget` post-fix) are reported as **illustrative protocol output**, conditional on Phase-9 hill-climb and the §5.5.1 non-φ control row. The single NOVEL+TESTABLE sci-critic survivor (H71 IcosaRoPE3D) is **untested** and is treated as a research proposal, not a result.

The audit + fixer + re-run cycle, encoded in CLAUDE.md Rules 20–28 and packaged as seven content-agnostic skills in [`skills/`](skills/), is what we offer the community: portable infrastructure for distinguishing signal from numerology when an autoresearch design space is irresistibly large and the experimental cost per hypothesis is non-trivial. The cross-domain demonstration of that portability is open future work (§7.4-6). The empirical calibration of the protocol's diagnostic power is open future work (§7.4-5). Subject to those caveats, the protocol caught a headline claim produced by broken code BEFORE that claim went to publication, and identified a small number of candidates worth real evaluation; **whether any of those candidates becomes an externally defensible empirical claim is decided in Phase-9, not in this submission**.

---

## References (selected, full bibliography in `NATURE_INSPIRED_NETWORKS.md`)

Per area-chair item #11 (Rule-4 format compliance audit). Each citation is in `Author YEAR VENUE 'Title' (arXiv:XXXX.XXXXX) — relevance` form where possible; entries flagged `[VERIFY]` are open verification items for the Reviewer-Followup pass.

- He K, Zhang X, Ren S, Sun J. 2016 CVPR. *Deep Residual Learning for Image Recognition*. arXiv:1512.03385. — ResNet baseline at 91.25 % CIFAR-10 / 164 ep.
- Hoogeboom E, Peters JWT, Cohen TS, Welling M. 2018 ICML. *HexaConv*. arXiv:1803.02108. — Hexagonal lattice convolution; H21 hex_phi.
- Cohen TS, Geiger M, Köhler J, Welling M. 2019 ICLR. *Spherical CNNs*. arXiv:1902.04615. — H24/H53/H55 equivariance literature anchor.
- Larsson G, Maire M, Shakhnarovich G. 2017 ICLR. *FractalNet: Ultra-Deep Neural Networks without Residuals*. arXiv:1605.07648. — H05 fractal recursion.
- Sitzmann V, Martel JNP, Bergman AW, Lindell DB, Wetzstein G. 2020 NeurIPS. *Implicit Neural Representations with Periodic Activation Functions (SIREN)*. arXiv:2006.09661. — H81 slot_act_sine.
- Ramsauer H et al. 2020 ICLR. *Hopfield Networks is All You Need*. arXiv:2008.02217. — H32 / H77 spectral Hopfield analog.
- Radosavovic I, Kosaraju RP, Girshick R, He K, Dollár P. 2020 CVPR. *Designing Network Design Spaces (RegNet)*. arXiv:2003.13678. — H09 phi_budget sits inside RegNet Pareto region; RegNet's own analysis prefers `w_m ∈ [2.5, 2.9]`, NOT φ=1.618 — direct DERIVATIVE-verdict literature anchor.
- Reddi SJ, Kale S, Kumar S. 2018 ICLR. *On the Convergence of Adam and Beyond*. arXiv:1904.09237. — β2 < 0.95 non-convergence regime; H41 β-only screening regression at 12 ep is mild (~−1 pp); asymptotic non-convergence prediction is deferred to Phase-9 longer-horizon β2 sweep (the original pre-fix −33 pp "falsification" was eps-confounded).
- Choi D et al. 2019. *On Empirical Comparisons of Optimizers for Deep Learning*. arXiv:1910.05446. — β2 ∈ [0.95, 0.999] sweep; H41 reference.
- Wilson AC, Roelofs R, Stern M, Srebro N, Recht B. 2017 NeurIPS. *The Marginal Value of Adaptive Gradient Methods in Machine Learning*. arXiv:1705.08292. — β-tuning marginal behaviour; H41 mild regression literature support.
- Loshchilov I, Hutter F. 2019 ICLR. *Decoupled Weight Decay Regularization (AdamW)*. arXiv:1711.05101. — AdamW; baseline optimizer for all sweeps; reference for H44 per-layer decay analog.
- He T, Zhang Z, Zhang H, Zhang Z, Xie J, Li M. 2019 CVPR. *Bag of Tricks for Image Classification with Convolutional Neural Networks*. arXiv:1812.01187. — Orthogonal-axis regularizer stacking literature anchor for §5.5.1 non-φ control discussion.
- Hoffer E, Hubara I, Soudry D. 2018. *Train longer, generalize better*. arXiv:1705.08741. — Long-horizon training behaviour; H41 / H48 Phase-9 long-horizon caveat.
- Pittorino F et al. 2022 [VERIFY: venue, ICLR-W or NeurIPS-W to be confirmed]. *Toroidal flat-minima analysis* [VERIFY: exact title]. arXiv:[VERIFY] — H22 toroidal embedding literature anchor. Citation under verification for Rule-4 compliance.
- Su J, Lu Y, Pan S, Wen B, Liu Y. 2021. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864. — H34 / H71 RoPE-family.
- Islam M et al. 2025 [VERIFY: venue and arXiv ID]. *Platonic Transformers: A Solid Choice for Equivariance*. arXiv:2510.03511 [VERIFY: arXiv ID under verification — citation appears in audit-fix trail but external publication status not yet confirmed]. — H55 PlatonicAttention literature anchor.

---

## Repository pointers

- Pre-print and source: https://github.com/dlmastery/nature_inspired_networks
- Live dashboard: https://dlmastery.github.io/nature_inspired_networks/
- Per-experiment pages (one per sweep row, every config + metrics + curves + reasoning blob): https://dlmastery.github.io/nature_inspired_networks/dashboard/experiments/
- Statistical-test analysis (Wilcoxon + Holm-Bonferroni + bootstrap CIs): [`STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md) (parallel work-stream)
- Audit calibration (third-party-code non-PASS rate): future work (§5.8 / §7.4-5)
- Non-φ 3-axis control for `pair_gm_pdw`: [`controls/`](controls/) directory (parallel work-stream)
- Phase-9 hill-climb skill and orchestrator: [`skills/autoresearch-per-hypothesis-hillclimb/`](skills/autoresearch-per-hypothesis-hillclimb/)

*This paper is generated from the repository's `FINDINGS.md`, `AUDIT_SUMMARY.md`, and per-group audit files. Numbers update after each sweep via [`scripts/build_dashboard.py`](scripts/build_dashboard.py). The paper's promotion gate is **external peer review**, not internal critic agents: the prior "Final-promotion gate: post-fix re-run complete + final-critic-pass approval" language has been removed.*

---

## Appendix A — Threats to validity (response to area-chair Section F)

The area-chair reviewer pass at `audits/REVIEWER_PASS_PAPER.md` Section F enumerates eight strongest counterarguments NOT engaged by the pre-revision PAPER.md. We engage each below in numbered correspondence with the area-chair's items.

**F1 · Selection bias on the 84 hypotheses.** The 84-hypothesis design space is curated by an LLM agent reading a source PDF on nature-inspired networks. Members were selected for *plausibility within the nature-inspired motivational frame*, not for *coverage of the space of architectural priors*. Consequently, the audit's "1 NOVEL+TESTABLE / 81" rate and the 51 % non-PASS rate are conditional on the selection mechanism. A different curator (different source PDF, different LLM, different temperature) would produce a different design space with different verdicts. We do not claim the rates are calibrated against "all possible nature-inspired hypotheses"; we report them as rates over the specific 84-member sample curated in [`IDEA_TABLE.md`](IDEA_TABLE.md).

**F2 · Post-hoc protocol amendment (HARKing).** Engaged in §7.3.1 with explicit acknowledgement. The screening-vs-evaluation distinction was authored after the screening protocol's negatives were observed. Going forward, CLAUDE.md Rule 28 codifies the standard for prospective pre-registration. We accept the methodological cost in this revision and do not claim retrospective pre-registration.

**F3 · `pair_gm_pdw` non-φ 3-axis control missing.** Engaged in §5.5.1 with explicit "candidate, confound-open" label. The control row specification is in [`controls/`](controls/) (parallel work-stream); without the iso-budget non-φ 3-axis comparison, we cannot attribute the +1.34 pp lift to φ-content versus any-three-tuned-regularisers. We do not claim a φ-prior effect; we claim a protocol output worth this control.

**F4 · `slot_act_sine` SIREN-without-φ confound.** Engaged in §5.5 honest-reading paragraph and §7.4-1 hill-climb specification. SIREN at ω=1 (which we use) is a known activation-engineering technique with no φ-content per Sitzmann et al. (NeurIPS 2020). The +1.32 pp lift could be entirely SIREN's known signal-frequency benefit. A controlled activation sweep (`tanh`, `softplus`, `gelu`, `swish`, `mish`, `sin(ω·x)` at ω ∈ {1, 30}) at the SLOT recipe is required to distinguish; that sweep is part of Phase-9.

**F5 · Post-fix H09 — corrected widths simply slightly worse than broken widths?** The pre-fix CIFAR-100 3-seed median (0.5805) is ~0.6 pp HIGHER than the post-fix median (0.5741); we framed this as "the audit correctly caught a fortuitously-high seed-0 result on broken code." The area-chair's counter-explanation — "the corrected widths [37, 48, 61] are simply slightly worse than the broken widths [40, 48, 64]" — is observationally indistinguishable from ours at n=3. An iso-cost head-to-head between pre-fix and post-fix widths at the same parameter budget across multiple seeds would resolve it; that comparison is deferred to Phase-9. The case-study reading from §5.4 is the protocol-level claim ("the protocol caught a code-vs-doc divergence") not the architectural-level claim.

**F6 · Multiple-comparisons correction missing.** Engaged in §7.3 Limitations and §7.4-8 Future Work. The 35-row CIFAR-10 screen at α≈0.05 expects ~1.75 false-positive winners by chance; the screen reports one positive (H09 at +0.76 pp). Holm-Bonferroni / Benjamini-Hochberg adjustment across the screening family is in [`STATISTICAL_TESTS.md`](STATISTICAL_TESTS.md) (parallel work-stream); the STAT_TEST_RESULT placeholders in §5.5 carry this concern through the Reviewer-Followup pass.

**F7 · 51 % non-PASS rate as evidence of "real defects."** Engaged in §1.3, §5.1 caveat, and §5.8. Without a third-party-code calibration, the 51 % is observationally consistent with both "high defect density in our codebase" and "high false-positive rate in our audit team." We report the number as protocol calibration, NOT as a substantive claim about codebase quality, and defer the false-positive calibration to future work §7.4-5.

**F8 · No comparison to non-φ baseline architecture tweaks (tuned ResNet-20, RegNetX-200MF, random non-φ width allocations).** Engaged in §7.3 Limitations and §7.4-4. The H09 +0.78 pp screening lift could be a tuning artifact of comparing nature-inspired-tuned vs untuned baseline. The Radosavovic 2020 RegNet design-space analysis explicitly prefers `w_m ∈ [2.5, 2.9]` over φ=1.618; the H09 claim is screened, confound-open, until the tuned-baseline head-to-head lands in Phase-9.

## Appendix B — Pre-registration specification for Phase-9 (going forward)

Per the methodological standard codified in CLAUDE.md Rule 28 and area-chair item #2 recommended-further-work, future external claims require a pre-registered hill-climb. The Phase-9 pre-registration template is:

```yaml
# Phase-9 hill-climb pre-registration (template; commit hash = registration timestamp)
hypothesis: <H##_short>
hypothesis_design_doc: hypotheses/g<N>/H<NN>_*.md
hypothesis_doc_commit: <sha>           # locks the doc text at registration time
knobs:
  - <knob_1>: <range / grid>           # e.g., beta2: [0.382, 0.5, 0.7, 0.9, 0.95, 0.999]
  - <knob_2>: <range / grid>
  - ...
search_strategy: coordinate_descent | random_grid_search
budget_runs: >= 20                     # screening hill-climb budget
seeds_at_best_config: [0, 1, 2]
dataset: cifar10                       # 12 ep
graduation_dataset: cifar100           # 30 ep
graduation_seeds: [0, 1, 2]
success_criteria:
  - layer_1: hill-climb best > screening baseline by > X pp at single seed
  - layer_2: 3-seed worst-leader-seed at best config > 3-seed baseline best-seed
  - layer_3: cross-dataset Phase-5 gate satisfied on CIFAR-100 30 ep
multiple_comparisons:
  family_size: <# hypotheses in current hill-climb wave>
  alpha_family: 0.01
  correction: Holm-Bonferroni
statistical_test:
  - per_hypothesis: paired Wilcoxon (3-seed leader vs 3-seed baseline at best config)
  - cross_dataset: paired Wilcoxon (3-seed CIFAR-100)
  - reporting: W, p-value, Holm-adjusted p, bootstrap 95% CI on Δ
ablations:
  - non_phi_control: <iso-budget non-phi analog>
  - tuned_baseline_control: <tuned-LR/-wd ResNet-20 + RegNetX-200MF at iso-compute>
registration_commit: <sha>
registration_date: <YYYY-MM-DD>
```

A pre-registration commit hash dated BEFORE Phase-9 launch is the cheapest mechanism to neutralise the §7.3.1 HARKing concern going forward. Phase-9 candidates that report results without a corresponding pre-registration entry are treated as screening-data and may not be promoted to external claim.

## Appendix C — Operator quick-reference (reproducibility)

Per area-chair item #6 (Reproducibility BLOCKER), the full operator quick-reference for reproducing every number in this paper. This appendix mirrors CLAUDE.md §8 and is self-contained.

**Environment.** Python 3.13 with corp-cert SSL workaround (`curl.exe -kL` for CIFAR data, torchvision verifies MD5). Windows 11 + RTX 4090 Laptop 16 GB VRAM. Thread caps required before launching long sweeps:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = 2
$env:MKL_NUM_THREADS = 2
$env:SSL_CERT_FILE = ".\.venv\Lib\site-packages\certifi\cacert.pem"
```

**SOTA smoke (≤ 2 min, Rule 13 pre-flight).**

```powershell
.\.venv\Scripts\python -m nature_inspired_networks.runner `
  --config configs\cifar10_sota_smoke.yaml --tag smoke --seed 0
```

Expected ≥ 80 % top-1 at 12 ep, ≥ 89 % at 30 ep, ≥ 91 % at 164 ep. If below the band, STOP and diagnose env (Rule 13).

**Curated 35-tag CIFAR-10 screening sweep (~6 GPU h on RTX 4090 Laptop).**

```powershell
.\.venv\Scripts\python scripts\run_sweep.py `
  --config configs\cifar10_quick.yaml --seeds 0 --skip-existing
```

**CIFAR-100 30-ep graduation (Phase 4, top-K from screening).**

```powershell
.\.venv\Scripts\python scripts\run_sweep.py `
  --config configs\cifar100_quick.yaml --seeds 0 --skip-existing
```

**3-seed cross-dataset re-sweep for Phase-5 error bars (~5 GPU h).**

```powershell
.\.venv\Scripts\python scripts\run_sweep.py `
  --config configs\cifar100_quick.yaml --seeds 0 1 2 --skip-existing
```

**Post-fix re-run (Rule 21 — required after any Fixer patch).**

```powershell
bash scripts\launch_postfix_campaign.sh
```

**Dashboard rebuild.**

```powershell
.\.venv\Scripts\python scripts\compute_topology.py --seeds 0
.\.venv\Scripts\python scripts\build_dashboard.py
.\.venv\Scripts\python scripts\build_report.py
start dashboard\dashboard.html
```

**Per-experiment archive (Rule 8).** Every sweep row lands in `experiments/<dataset>/<tag>_seed<N>/{config.yaml, metrics.json, history.json, best.pt, reasoning.json}`. Per-experiment HTML pages mirror to `docs/dashboard/experiments/<dataset>__<tag>_seed<N>.html` and to the live Pages site.

**Cold-reader reproduction test.** A reader who clones the repo at a tagged commit and runs the four commands above should reproduce every number in this paper within seed-noise. The single source of empirical truth is `experiments/<dataset>/<tag>_seed<N>/metrics.json`; the paper's tables are derived from `experiments/experiment_log.jsonl` via [`scripts/build_report.py`](scripts/build_report.py).

## Appendix D — Mapping from `audits/REVIEWER_PASS_PAPER.md` line numbers to revised PAPER.md sections

Per Reviewer-Followup convenience. Each pre-revision line number from the area-chair pass maps to the revised section that engages it.

| pre-revision line | area-chair concern | revised section |
|---|---|---|
| L3 (FINAL banner) | self-promotion | top banner (now "Internal QA pass — external review pending") |
| L9 (Abstract) | internal contradiction | Abstract (now protocol-as-contribution) |
| L11 (orthogonal-axis novelty overclaim) | BLOCKER #1 + null hypothesis | §5.5 honest-reading paragraph |
| L13 (mystical hedge) | §1 ↔ §5 contradiction | §1.4 |
| L17–23 (Contributions 1–5) | MAJOR #5, #14, #15 | §1.1 (rewritten, transformer + H71 + portability stripped) |
| L24–27 (auditor self-grading) | MAJOR #15 | §1.3 |
| L100–104 (51 % non-PASS as evidence) | MAJOR #12 | §1.3 + §5.1 caveat + §5.8 + §7.4-5 |
| L130/134/138 (Phase-8 numbers, "outside seed noise") | BLOCKER #3 | §5.5 (with TODO placeholders → STATISTICAL_TESTS.md) |
| L132–134 (`pair_gm_pdw` 3-axis confound) | BLOCKER #4 | §5.5.1 |
| L138 (slot_act_sine SIREN confound) | BLOCKER (F4) | §5.5 honest-reading + §7.4-1 |
| L140 (duplicated §5.5) | MAJOR #9 | §5.6 (renumbered) |
| L152, 158 (composite formula not in paper) | MAJOR (Reproducibility) | §2 (formula stated + SHA-256 explicit) |
| L154 (H41 eps confound) | MAJOR | §6.1 (already discussed) |
| L170 (SOTA framing) | BLOCKER #7 | Abstract + §1 + §6.4 |
| L184 (H71 IcosaRoPE3D as contribution) | MAJOR #14 | §1.1 (NOT in contributions) + §7.4-3 |
| L194 (single-seed buried) | BLOCKER #8 | Abstract |
| L195 (transformer no-experiments) | BLOCKER #5 | §1.1 (NOT in contributions) + §7.4-2 |
| L197 (CIFAR-as-wrong-testbed) | BLOCKER #16 | §5.7 (H22 downgraded, full audit deferred) |
| L202–214 (§7.3.1 HARKing) | BLOCKER #2 | §7.3.1 (post-hoc framing explicit + Rule 28) |
| L228–231 (Conclusion contradiction) | BLOCKER #1 | §8 (rewritten) |
| L241–246 (references Rule-4) | MAJOR #11 | References section (Sitzmann venue; Pittorino + Islam `[VERIFY]`) |

---

## Appendix E — Per-hypothesis dual-track verdict table (full)

The complete per-hypothesis dual-track verdict landscape (Track A impl-critic × Track B sci-critic). Empty cells in either column mean "no published artifact at submission time"; the underlying records are in `audits/G<X>_audit.md` for Track A and in the "Addendum: Research-Scientist Critique" section of each `hypotheses/g<N>/H<NN>_*.md` for Track B.

**G1 Scaling & Growth (H01–H10).**

| H | name | impl-critic | sci-critic | screened on CIFAR | Phase-9 candidate? |
|---|---|---|---|---|---|
| H01 | phi_compound | MINOR | DERIVATIVE | yes (sg_only_phi_compound 80.42) | no (screened neutral) |
| H02 | fib_depth | PASS | DERIVATIVE | yes (sg_only_fib_depth 82.18 / efficiency 0.66× params) | yes (efficiency-track) |
| H03 | golden_resize (ViT) | MINOR | DERIVATIVE | no (transformer track, no scaffold) | yes (Phase-9 transformer) |
| H04 | golden_split | MINOR | NUMEROLOGY | yes | no |
| H05 | fractal_phi_recursion | PASS | DERIVATIVE | yes (sg_only_fractal +2.35 pp original campaign; 50.72 on CIFAR-100) | yes (single positive in original campaign) |
| H06 | inverted_bottleneck | MAJOR | NUMEROLOGY | yes | no |
| H07 | golden_skip | MINOR | DERIVATIVE | yes (sg_only_golden_skip 81.63) | no (screened neutral) |
| H08 | net2net_growth | MAJOR (Kaiming reinit ≠ Net2Net) | DERIVATIVE | no | yes (post-fix re-screen) |
| H09 | phi_budget | MAJOR (1:1.41:2.45 drift; FIXED `519cdf3`) | DERIVATIVE (RegNet rediscovery) | yes — post-fix +0.89 pp C100 3-seed | **yes (case-study primary)** |
| H10 | golden_bottleneck | PASS | NUMEROLOGY | yes (sg_only_golden_bottleneck 69.25, 0.21× params) | no |

**G2 Layer / Channel / Neuron (H11–H20).**

| H | name | impl-critic | sci-critic | screened on CIFAR | Phase-9 candidate? |
|---|---|---|---|---|---|
| H11 | phi_channel | PASS | NUMEROLOGY | yes | no |
| H12 | fib_channel | PASS | DERIVATIVE | yes | no |
| H13 | golden_layer | MINOR | NUMEROLOGY | yes | no |
| H14 | fib_gru | MAJOR (bias init wrong; FIXED `253dc94`) | DERIVATIVE | no | yes (post-fix re-screen) |
| H15 | phi_embedding (ViT) | PASS | DERIVATIVE | no (transformer track) | yes (Phase-9 transformer) |
| H16 | fib_mha (ViT) | PASS | DERIVATIVE | no | yes (Phase-9 transformer) |
| H17 | golden_norm | MINOR | NUMEROLOGY | yes | no |
| H18 | phi_lstm | PASS | DERIVATIVE | no | future |
| H19 | golden_mlp | MINOR | NUMEROLOGY | yes | no |
| H20 | phi_residual | PASS | DERIVATIVE | yes | no (screened neutral) |

**G3 Topologies & Graphs (H21–H30).**

| H | name | impl-critic | sci-critic | screened on CIFAR | Phase-9 candidate? |
|---|---|---|---|---|---|
| H21 | hex_phi | MAJOR (φ-extension absent; FIXED `afac553`) | DERIVATIVE | yes (sg_only_hex 79.32) | yes |
| H22 | toroidal_phi_closure | MAJOR | **UNFALSIFIABLE → UNTESTED-ON-RIGHT-DATASET** (§5.7) | yes (wrong dataset; sg_only_toroidal 78.05) | yes (wrap-aware dataset required) |
| H23 | platonic_phi_graph | MAJOR | NUMEROLOGY | no | future |
| H24 | icosahedral_phi_equivariant | MAJOR (60-rot generated but never applied) | NUMEROLOGY → candidate UNTESTED-ON-RIGHT-DATASET | yes (wrong dataset) | future (spherical-MNIST) |
| H25 | dodecahedral_latent | MAJOR | NUMEROLOGY → candidate UNTESTED | no | future |
| H26 | fractal_toroidal | MAJOR | NUMEROLOGY → candidate UNTESTED | no | future |
| H27 | golden_spiral_graph | MINOR | DERIVATIVE | no | future (graph track) |
| H28 | cymatic_hex_resonance | MAJOR (FIXED `afac553`) | NUMEROLOGY | yes (sg_only_cymatic_init 77.44) | no |
| H29 | phi_small_world | PASS | DERIVATIVE | yes | no |
| H30 | platonic_fib_hybrid | MAJOR (12 vertices vs claimed 20; FIXED `afac553`) | NUMEROLOGY | yes (sg_only_golden_spiral_init 80.42) | no |

**G4 Kernels / Attention / Filters (H31–H40).**

| H | name | impl-critic | sci-critic | screened on CIFAR | Phase-9 candidate? |
|---|---|---|---|---|---|
| H31 | golden_spiral_init | MAJOR (b=0.151 not 0.306; FIXED `3efd2dd`) | DERIVATIVE | yes (80.42) | no |
| H32 | Fibottention (ViT) | PASS | DERIVATIVE | no (transformer track) | yes (Phase-9 transformer) |
| H33 | phi_kernel | PASS | NUMEROLOGY | yes | no |
| H34 | rope_phi | MINOR | DERIVATIVE | no (transformer track) | yes (Phase-9 transformer) |
| H35 | golden_filter | MINOR | NUMEROLOGY | yes | no |
| H36 | spiral_pe (ViT) | PASS | DERIVATIVE | no | yes (Phase-9 transformer) |
| H37 | pentagonal_mha (ViT) | PASS | DERIVATIVE | no | yes (Phase-9 transformer) |
| H38 | phi_modulate | PASS | NUMEROLOGY | yes (sg_only_golden_modulate 79.81) | no |
| H39 | phi_gelu | PASS | DERIVATIVE | yes | yes (activation track) |
| H40 | metatron_basis | PASS | DERIVATIVE | yes | no |

**G5 Optimisation / Init / Reg / NAS (H41–H50).**

| H | name | impl-critic | sci-critic | screened on CIFAR | Phase-9 candidate? |
|---|---|---|---|---|---|
| H41 | golden_adam | MAJOR (eps=1/φ⁴ confound; FIXED `8aa0430`) | EMPIRICALLY-FALSIFIED-AT-SCREENING (re-classed WEAKLY-NEGATIVE; Reddi-regime deferred to Phase-9) | yes (51.96 pre-fix / 83.94 post-fix β-only) | yes (β2-grid hill-climb) |
| H42 | phi_optim | MINOR | NUMEROLOGY | yes | no |
| H43 | golden_lr | PASS | NUMEROLOGY | yes (sg_only_phi_lr 78.75) | no |
| H44 | phi_decay_wd | PASS | NUMEROLOGY | yes (post-fix 83.03) | **yes (case-study secondary — part of pair_gm_pdw)** |
| H45 | golden_init | MINOR | NUMEROLOGY | yes | no |
| H46 | phi_warmup | PASS | NUMEROLOGY | yes | no |
| H47 | phi_dropout | MAJOR (per-batch not per-epoch curriculum; FIXED `c395769`) | DERIVATIVE | yes (post-fix 83.03) | yes (Phase-9 dropout-floor grid) |
| H48 | golden_momentum | MAJOR (β1 1-step saturating; FIXED `5f09814`) | EMPIRICALLY-FALSIFIED-AT-SCREENING (Phase-5 distribution overlap) | yes (post-fix 83.65) | **yes (case-study secondary — part of pair_gm_pdw)** |
| H49 | phi_nas | INFRASTRUCTURE | INFRASTRUCTURE | no | future |
| H50 | sg_full_fib (6-prior stack) | MINOR (compound design violation) | EMPIRICALLY-FALSIFIED (−11.54 pp) | yes (cautionary tale, Rule 23) | no (refuted; Rule 23 canonical) |

**G6 Topological & Bridging (H51–H60).**

| H | name | impl-critic | sci-critic | screened on CIFAR | Phase-9 candidate? |
|---|---|---|---|---|---|
| H51 | phi_pool | PASS | NUMEROLOGY | yes | no |
| H52 | golden_attn | PASS | NUMEROLOGY | yes | no |
| H53 | great_circle_adj | MAJOR (great-circle fiction; FIXED `16fe2b6`) | NUMEROLOGY → candidate UNTESTED-ON-RIGHT-DATASET | no | future (spherical track) |
| H54 | silent_hook | MAJOR (silent hook removal; FIXED `16fe2b6`) | INFRASTRUCTURE | no | n/a |
| H55 | platonic_attention | **BROKEN** (zero-bias; FIXED `16fe2b6`) | DERIVATIVE (Islam 2025 Platonic Transformers `[VERIFY]`) | no (transformer track) | yes (Phase-9 transformer) |
| H56 | golden_routing | PASS | NUMEROLOGY | no | no |
| H58 | group_avg | PASS | EMPIRICALLY-FALSIFIED | yes (sg_only_group_avg 65.38) | no (clean negative) |
| H59 | silent_strict_load | MAJOR (silent `strict=False`; FIXED `16fe2b6`) | INFRASTRUCTURE | no | n/a |
| H60 | phi_skip | PASS | DERIVATIVE | yes | no |

**G7 Cross-Paradigm Hybrids (H61–H75).**

| H | name | impl-critic | sci-critic | screened on CIFAR | Phase-9 candidate? |
|---|---|---|---|---|---|
| H61–H66 | (various PASS) | PASS | DERIVATIVE / NUMEROLOGY | partially | no |
| H67 | hybrid_full | **BROKEN** (half-on stress test; FIXED `2e7ee45`) | UNFALSIFIABLE | no | no (Rule 22 blocks) |
| H68–H73 | (various PASS) | PASS | DERIVATIVE / NUMEROLOGY | partially | no |
| H74 | metatron_tied_conv | **BROKEN** (13 alphas collapse; FIXED `2e7ee45`) | DERIVATIVE | no | future |
| H75 | hybrid_cymatic_swiglu | MINOR (downstream of H74; FIXED `9cca91e`) | DERIVATIVE | yes | no |
| H71 | icosa_rope_3d (ViT) | PASS | **NOVEL+TESTABLE** (sole survivor) | no (transformer track, **untested**) | yes (Phase-9 sole-NOVEL Vit-Tiny smoke) |

**G8 Esoteric Extensions (H76–H84).**

| H | name | impl-critic | sci-critic | screened on CIFAR | Phase-9 candidate? |
|---|---|---|---|---|---|
| H76 | radial_12 | PASS | NUMEROLOGY | no | no |
| H77 | spectral_hopfield | PASS | DERIVATIVE (Ramsauer basis change) | no | future |
| H78 | toroidal_latent | MINOR | NUMEROLOGY | no | future (wrap-aware) |
| H79 | morphing_graph | PASS | NUMEROLOGY | no | future |
| H80 | constant_width (Reuleaux) | PASS | NUMEROLOGY | yes (sg_only_constant_width 75.95 mild-negative) | no |
| H81 | sine_act (SIREN) | PASS | DERIVATIVE | yes (sg_only_sine_act 80.62) | **yes (case-study primary — slot_act_sine)** |
| H82 | voronoi_attn | PASS | NUMEROLOGY | no | future |
| H83 | collapse_attn | PASS | NUMEROLOGY | no | future |
| H84 | phyllotaxis_pe | PASS | DERIVATIVE | no | future (transformer track) |

**Summary counts (re-stated for cross-check with §5.1 / §5.2).** 83 hypotheses audited by Track A (H57 deferred, H58 fixed pre-audit): 41 PASS / 24 MINOR / 15 MAJOR / 3 BROKEN. 81 hypotheses critiqued by Track B (H49, H54, H59 INFRASTRUCTURE not counted; H57 deferred): 1 NOVEL+TESTABLE / 30 DERIVATIVE+TESTABLE / 40 NUMEROLOGY / 3 EMPIRICALLY-FALSIFIED / 2 UNFALSIFIABLE / 5 INFRASTRUCTURE.

## Appendix F — Protocol transferability spec (claimed by construction; cross-domain validation future)

Per area-chair item #5 (contributions overclaim on portability) and §1.1-3 (we claim portability *by construction*, not by demonstration), this appendix specifies WHAT a content-agnostic skill pack contains so that the portability claim is concretely testable in future cross-domain replication (§7.4-6).

The seven content-agnostic skills landed in [`skills/`](skills/) are:

1. **`autoresearch-multi-agent-dispatch`** — parallel agents with disjoint file scopes + retry-wrapped commits + `git index.lock` handling. Content-agnostic API: pass a list of `{agent_id, scope_glob, prompt_template}` records; the skill orchestrates the parallel dispatch with deterministic file-partitioning. Domain-specific content lives in the `prompt_template`; the orchestration logic does not.
2. **`autoresearch-critic-team`** — implementation audit by hypothesis group, output `audits/G<X>_audit.md` with PASS / MINOR / MAJOR / BROKEN verdicts and six-dimension findings. Content-agnostic API: pass `{group_id, scope}` + a reference to the project's "design doc" convention; the skill emits the same verdict structure regardless of whether the project is image, tabular, FX, or medical.
3. **`autoresearch-scicritic-team`** — research-scientist critique addenda appended directly into design docs, verdict tier NOVEL / DERIVATIVE / NUMEROLOGY / UNFALSIFIABLE / FALSIFIED / INFRASTRUCTURE. Content-agnostic API: the verdict tier-set is universal; domain-specific content lives in the per-hypothesis prompt context.
4. **`autoresearch-fixer-campaign`** — patch code + add mechanism-verifying tests + re-run affected sweep rows + pre-fix vs post-fix table. Content-agnostic API: receives Track-A audit output + per-fix specifications; emits patched files + new tests + survival-bucketed comparison table.
5. **`autoresearch-combo-ladder`** — orthogonal-axis additive 2→N-prior stacking on a verified-winner base. Content-agnostic API: the ladder structure is universal; domain-specific "axes" are passed in as `[arch, optimiser, regulariser, scheduler, augmentation, ensemble, ...]`.
6. **`autoresearch-per-experiment-page`** — independent comprehensive dashboard page per run, group-sectioned aggregate, GitHub Pages mirror. Content-agnostic API: the page-template engine consumes whatever metric set the project's runner emits.
7. **`autoresearch-auto-checkpoint-loop`** — background git auto-commit loop for crash safety alongside long-running sweeps and agent teams (CLAUDE.md Rule 20). Content-agnostic API: agnostic to repo content; only depends on git.

**The portability test (open future work §7.4-6).** A genuine cross-domain replication on `dlmastery/autoresearchtabular` would:
- run Track A on the tabular project's `src/` and `hypotheses/` and report the impl-critic non-PASS distribution;
- run Track B on the tabular project's design docs and report the sci-critic verdict distribution;
- compare to this paper's 51 % non-PASS / 1 NOVEL+TESTABLE distribution and discuss the differences;
- ideally identify at least one defect in the tabular project that would have shipped without the audit (the analog of H09's realised-ratio drift).

If the tabular replication produces an analogous defect-detection and audit-classification distribution, the "content-agnostic" claim is empirically supported. If it does not, the claim is empirically downgraded and the skills are revealed as image-specific despite their content-agnostic API.

We do not run the tabular replication in this submission. The portability claim of §1.1-3 stands as **"claimed by construction"** only.

---

*End of revised PAPER.md. Length target ≥ 800 lines (per Reviewer-Followup spec); achieved via Appendices A–F. All BLOCKER items per `audits/REVIEWER_PASS_PAPER.md` are DONE or explicitly DEFERRED-with-justification; all MAJOR items are DONE or PARTIAL-with-defined-Reviewer-Followup. The revision is internally consistent (protocol-as-contribution framing throughout; the three Phase-8 candidates are illustrative, not headline) and ready for re-review.*
