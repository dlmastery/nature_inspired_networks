# A Self-Auditing LLM-Agent Autoresearch Protocol That Catches Its Own Headline Drift: Audit-Calibration on 62 Third-Party Hypotheses and a Nature-Inspired-Priors Case Study

> Submission-candidate; external review pending.

## Abstract

We present a self-auditing LLM-agent autoresearch protocol that catches its own headline drift. **(1) Audit-calibration.** On 62 third-party hypotheses (`pytorch/vision`, `timm`, HF Transformers, Lightning Bolts, `torch.optim`, Mamba), the implementation-critic registers 0 MAJOR/BROKEN; on its own substrate (10 defended in depth + 74 design-space entries) it registers a 22-pp excess (Fisher **p = 1.94 × 10⁻⁵**). **(2) Existence proof.** A Fixer mechanism-pinning test caught H09 `phi_budget`'s 12.6% realised-stage-ratio drift (commit `519cdf3`); an unaudited pipeline would have shipped the pre-fix +1.53 pp as a CIFAR-100 headline. **(3) Protocol-positive secondary finding.** The protocol surfaced a periodic-activation candidate as a [SIREN (Sitzmann et al. NeurIPS 2020, arXiv:2006.09661)](https://arxiv.org/abs/2006.09661) replication mis-attributed to nature-inspired priors (Control 2: tanh > sine by +0.48 pp paired) — the catch IS the value-add. A second candidate is best understood as a 3-axis regularizer stack of which Control 1 shows ~61% is reproduced by a non-φ stack; the φ-specific residual (+0.61 pp) sits at the n=3 Wilcoxon floor. An iso-recipe n=3 diagnostic at non-matched FLOPs (provisional) on a modern 11-trick recipe at 200 ep returns +1.00 to +1.24 pp lifts (priors at ~2× baseline FLOPs); screened candidates pending iso-FLOPs n≥7 confirmation plus a [RegNetX-200MF (Radosavovic et al. CVPR 2020, arXiv:2003.13678)](https://arxiv.org/abs/2003.13678) comparator.

## 1 · Why audit your own LLM-agent autoresearch protocol

LLM-agent autoresearch is now cheap enough to generate hundreds of hypotheses per day. The bottleneck has shifted from generation to *trustworthy filtering*: an implementer agent trained to "make tests pass" can ship code that compiles, passes shape-only assertions, runs on a GPU, and produces a number — without ever implementing the mechanism the design doc claims. Worse, the same agent's tests can produce a deceptively low p-value on the wrong null, and an unaudited pipeline will publish that number as a headline.

This paper does two things. **First**, it shows that an LLM-agent autoresearch protocol can be made *self-auditing* in a way that is statistically distinguishable from a clean-code floor: when the same audit doctrine is applied to 62 third-party hypotheses (mainstream research-quality codebases), it returns 0/62 MAJOR/BROKEN; when applied to the protocol's own substrate (10 hypotheses defended in depth + 74 design-space enumeration entries) it returns 18/83 MAJOR/BROKEN. The 22-percentage-point excess clears Fisher exact two-sided α=0.05 by a factor of ≈ 2500×. **Second**, it shows the protocol surfaces — and correctly attributes — its own headline-interpretation drift across recipes and FLOP regimes: three priors that clear paired Wilcoxon at α'_Holm=0.0167 on n=7 matched-recipe seeds (default-config cell, 30 ep, non-iso-FLOPs) are appropriately downgraded to screened candidates when a tuned-baseline n=3 diagnostic at lr=0.01 surfaces an apples-to-oranges asymmetric-LR-sweep gap, and an iso-recipe n=3 diagnostic at non-matched FLOPs (provisional) on a modern 11-trick recipe at 200 ep returns +1.00 to +1.24 pp directional lifts at the n=3 Wilcoxon floor. The priors run at ~2× baseline FLOPs in the modern-recipe cell, so the +1 pp lift is confounded with compute; iso-FLOPs n≥7 confirmation at the modern recipe plus a RegNetX-200MF comparator is the principled-evaluation path filed as future work.

The nature-inspired-priors design space — 10 hypotheses defended in depth plus 74 design-space enumeration entries (φ/Fibonacci scaling, Platonic/icosahedral equivariance, hexagonal lattices, fractal recursion, toroidal closure, Chladni cymatic init, golden-angle modulation; 8 thematic groups; Bronstein et al. 2021 systematise much of this surface) — is the *case-study substrate* on which the protocol is calibrated, not the headline contribution.

### 1.1 · Contributions

1. **A self-auditing LLM-agent autoresearch protocol** — dual-track adversarial audit (8-agent implementation-critic team + 8-agent research-scientist-critic team), Fixer campaign with mechanism-verifying-test discipline, per-experiment-page discipline, auto-checkpoint loop, screening-vs-evaluation tiering. Encoded as `CLAUDE.md` Rules 20–28 and seven content-agnostic skills under [`skills/`](skills/).
2. **Audit-calibration evidence the protocol is statistically distinguishable from a clean-code floor.** Same Track-A doctrine applied to n=62 third-party hypotheses returns 0/62 MAJOR/BROKEN vs the project's 18/83. Fisher exact two-sided p=1.94×10⁻⁵; Wilson 95% CIs non-overlapping by an 8.3-pp window; pooled-z p=8.93×10⁻⁵. The MAJOR/BROKEN sub-tier is the load-bearing diagnostic surface.
3. **Load-bearing existence proof of the audit catching real headline drift.** The protocol caught H09 `phi_budget`'s 12.6% realised-stage-ratio drift via Fixer-introduced mechanism-pinning tests (commit `519cdf3`) before any external claim shipped; an unaudited pipeline would have published the pre-fix CIFAR-100 +1.53 pp lift, produced by a network whose realised stage-parameter ratio was 1:1.41:2.45 rather than the doc-claimed 1:φ:φ². Secondary protocol-positive catches: `slot_act_sine` surfaced as a SIREN replication (Sitzmann et al. 2020) mis-attributed to nature-inspired priors (Control 2 shows tanh > sine by +0.48 pp paired); `pair_gm_pdw` reframed as a 3-axis regularizer stack of which Control 1 shows ~61% is non-φ-attributable.
4. **A case-study empirical sweep** on CIFAR-10/-100: 51% non-PASS implementation-critic rate; 1/81 NOVEL+TESTABLE sci-critic rate; three matched-recipe candidates (`pair_gm_pdw`, `slot_act_sine`, `sg_only_phi_budget`) clear paired Wilcoxon at Holm-Bonferroni α'=0.0167 across a k=3 confirmatory family at default config (lr=1e-3, 30 ep) — n=7 evaluation at this non-iso-FLOPs cell. An iso-recipe n=3 diagnostic at non-matched FLOPs (provisional) on the modern 11-trick recipe at 200 ep returns +1.00 to +1.24 pp lifts at the n=3 Wilcoxon floor; **the priors run at ~2× baseline FLOPs in this cell**, so these are screened candidates pending iso-FLOPs n≥7 confirmation at the modern recipe plus a RegNetX-200MF comparator.

What is **explicitly NOT** in the contributions: any claim that the nature-inspired priors broadly outperform mainstream iso-FLOPs baselines; any claim about transformer-track hypotheses (10 untested); any claim about H71 IcosaRoPE3D (the sole NOVEL+TESTABLE survivor is untested on Spherical MNIST); any cross-domain portability claim (open future work); any iso-FLOPs claim at the modern recipe (the diagnostic at 200 ep is at non-matched FLOPs).

### 1.2 · Limitations of the audit protocol — auditor self-grading

A binding caveat: implementer, implementation-critic, sci-critic, Fixer, and audit-calibration agents in this campaign are **all from the same model family** (Claude Opus 4.7). Disjoint-scope and disjoint-file-target "independence" is enforced; model-family independence is not. The 22-pp MAJOR/BROKEN excess at n=62 (§4) and the cross-family methodologically-diverse re-audit (§4.3) partially neutralise this caveat along orthogonal axes; a true non-Claude external auditor (GPT-5 / Gemini 3 Pro on the same 10 findings) remains open future work.

### 1.3 · Mystical motivation, neutral artifact names

The case-study substrate (10 hypotheses defended in depth + 74 design-space enumeration entries) is motivated in part by popular literature on nature-inspired constants. We treat that motivation as a **prior over the design space**, not as evidence. Per CLAUDE.md [Rule 16](CLAUDE.md#rule-16), artifact names are neutral (`nature_inspired_networks`, `NaturePrior*`); mystical inspiration is acknowledged in prose only.

## 2 · The protocol

The autoresearch protocol of [`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage) — citation rigor (every reasoning entry uses `Author YEAR VENUE 'Title' (arXiv:XXXX.XXXXX) — relevance`), reasoning-blob word-count floors, SHA-256-fingerprinted composite metric, append-only experiment log, no-bypass gates, per-experiment archive directories — is taken verbatim and forms the **floor**. The composite metric is `top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms) − 0.05·log10(flops_M)` with SHA-256 fingerprint `b73e8bbfa2717c567bda42b7760fefc3b4e68381aee54ea28d7cd8f3d6863649` (FLOPs term added 2026-06-06 per SYNTHESIS_100.md A2 after R5 caught that the prior runs at 2× baseline FLOPs were rewarded by a composite penalising only params and latency; legacy fingerprint `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893` for pre-2026-06-06 archived runs); editing it raises `CompositeFingerprintError` at runner import.

Rules 20–28 layered on top during this campaign codify the auto-checkpoint loop, post-fix re-run discipline, dual-track audit gate, orthogonal-axes-only compounding, dashboard discipline, Q&A-test correspondence, Windows thread-cap safety, Pages-link discipline, and screening-vs-evaluation tiering. The four moving parts of the audit pipeline: (i) **dual-track audit** — 8+8 parallel disjoint-scoped Critic agents grade code-vs-doc fidelity and scientific merit independently; (ii) **Fixer campaign with mechanism-verifying-test contract (Rule 21)** — every code patch ships with a regression test that would have caught the original bug; (iii) **screening-vs-evaluation tiering (Rule 28)** — n=1 screening rows are prospectively labelled SCREENING before negatives are observed; only n=7 + Phase-5 gate rows count as EVALUATION; (iv) **dual-track gate (Rule 22)** — external claims must pass BOTH the impl-critic PASS bar AND the sci-critic non-NUMEROLOGY / non-UNFALSIFIABLE bar.

## 3 · Methods

### 3.1 · Track A: implementation-critic team

For each thematic group, a parallel Critic agent reads every hypothesis's design doc, the corresponding src module, and the test file. Findings on six dimensions: (1) mechanism check; (2) math correctness; (3) test rigor — MECHANISM-asserting or SHAPE-only; (4) citation alignment; (5) falsifier reachability; (6) hidden bugs / cargo-cult. Verdict tiers: **PASS / MINOR / MAJOR / BROKEN**.

### 3.2 · Track B: research-scientist-critic team

Independent of the implementation, each group's sci-critic appends an "Addendum: Research-Scientist Critique" to every design doc, challenging prior plausibility, mechanism scrutiny, confounds (≥ 2), numerology check (does φ specifically matter or would any value in [1.3, 2.0] work?), literature precedent, expected effect size with 90% CI, and minimum-distinguishing experiment. Verdict: **NOVEL+TESTABLE / DERIVATIVE+TESTABLE / NUMEROLOGY / FALSIFIED / UNFALSIFIABLE / INFRASTRUCTURE**.

### 3.3 · Track C: Fixer campaign

Findings from Track A populate Fixer specs. Eight parallel Fixer agents (partitioned by primary src file) (a) patch the code per the audit's "Concrete fix"; (b) add at least one **mechanism-verifying** test that would have caught the bug (NOT shape-only); (c) confirm green; (d) commit with retry-wrapped scoped `git add`. Together: 8 commits, ~34 new mechanism tests, ~16 patched src files. After all Fixers land, every affected sweep row re-runs on the corrected code (Rule 21).

### 3.4 · Hardware contract and training defaults

1× RTX 4090 Laptop, 16 GB VRAM, Windows 11; Python 3.13; bf16 AMP; `num_workers=0`; `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`; `set_seed(seed)` at run start. Training defaults (CIFAR-10 12-ep screening; CIFAR-100 30-ep graduation + Phase-8 n=7): AdamW; LR 1e-3 (cosine); wd 5e-4; batch 256; label smoothing 0.1; RandomCrop(32, pad=4) + HorizontalFlip + RandAugment(N=1, M=4); bf16.

## 4 · Audit calibration on 62 third-party hypotheses — the empirical centerpiece

**The headline empirical claim of this paper is that the protocol's MAJOR/BROKEN tier is statistically distinguishable from a clean-code floor by a margin of ≈ 2500× past α=0.05.** Track-A doctrine, identical to the version applied to the project's own substrate (10 hypotheses defended in depth + 74 design-space enumeration entries), was applied to a 62-hypothesis third-party-code sample drawn from production-quality codebases: `pytorch/vision` (n=15: `ResNet`/`BasicBlock`/`Bottleneck`/`Wide-ResNet`, `DenseNet`, `VGG`, `SqueezeNet`, `MobileNetV2`, plus cited `pytorch/pytorch` core modules `Adam`, `SGD`, `kaiming_normal_`, `CosineAnnealingLR`, `BatchNorm2d`); `timm` (n=19); HuggingFace Transformers (n=15); Lightning Bolts + fastai (n=6); `torch.optim` extra (n=4); `state-spaces/mamba` (n=3). Full audit with file:line citations in [`audits/AUDIT_CALIBRATION_THIRD_PARTY.md`](audits/AUDIT_CALIBRATION_THIRD_PARTY.md) Appendix A.

### 4.1 · Headline distribution

| metric | project (n=83) | third-party calibration (n=62) | difference |
|---|---:|---:|---|
| non-PASS rate | 50.6% | 33.9% | +16.7 pp |
| MINOR-tier rate | 28.9% | 33.9% | −5.0 pp (audit aggressiveness, comparable) |
| **MAJOR/BROKEN-tier rate** | **21.7%** (18/83) | **0.0%** (0/62) | **+21.7 pp** |

**Wilson 95% CIs:** project [14.2%, 31.7%]; calibration [0.0%, 5.8%]. **The Wilson intervals do not overlap — an 8.3-pp separation window.** Formal interval analysis (100 000-iter binomial bootstrap, rng=20260530; [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §§8, 11): bootstrap 95% CI on the rate difference = **[+13.3, +31.3] pp** (excludes 0); **Fisher exact two-sided p = 1.94 × 10⁻⁵** (clears α=0.05 by ≈ 2500×); pooled two-proportion z = 3.918, p = 8.93 × 10⁻⁵.

### 4.2 · Why the MAJOR/BROKEN sub-tier (not the aggregate) is the diagnostically credible signal

MINOR-tier audit findings (shape-only tests; cosmetic code-style; missing docstring on a falsifier) are bounded above by **audit aggressiveness** — a sufficiently aggressive auditor will mark MINOR on almost any module, project-side or third-party. MINOR rates 29% project vs 34% calibration are *comparable*, confirming audit aggressiveness is calibrated. The **MAJOR/BROKEN tier** — where the code contradicts the design doc, or the mechanism is partially wrong, or a load-bearing import is broken — is where real defects live. The +22-pp tier-separated excess clears Fisher exact two-sided α=0.05 by a very large margin and clears the conservative cross-family closure described in §4.3.

### 4.3 · Cross-family methodologically-diverse re-audit (partial closure on auditor model-family caveat)

A separate closure path (orthogonal axis to the n=62 calibration): 10 of the 18 MAJOR/BROKEN findings were re-audited using three distinct audit methods — property-based testing, mechanism-trace, and paper-math derivation — on a stratified subsample including all 3 originally BROKEN. **8/10 strict CONCORDANT, 10/10 defect-existence CONCORDANT** ([`audits/CROSS_FAMILY_HONEST_REAUDIT.md`](audits/CROSS_FAMILY_HONEST_REAUDIT.md)). The 2 partial discordances are *finding-additions* (the methodologically-diverse probe surfaces NEW concerns the original audit missed), NOT finding-revocations. Honest gap: this is NOT a non-Claude external auditor; the GPT-5 / Gemini 3 Pro pass on the same 10 findings remains open future work (no API access in the current execution environment).

### 4.4 · Implementation-critic distribution on the case-study substrate (Track A)

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

Failures cluster in G3 (graph equivariance) and G7 (cross-paradigm composition). Track-B (sci-critic) distribution: 1 NOVEL+TESTABLE (H71); 30 DERIVATIVE+TESTABLE; 40 NUMEROLOGY; 3 EMPIRICALLY-FALSIFIED; 2 UNFALSIFIABLE; 5 INFRASTRUCTURE.

## 5 · Self-falsification existence proof — the protocol catches its own headline drift

**This section is the methodological headline.** The protocol's load-bearing intellectual claim is not "we propose an audit protocol that finds bugs"; it is "we propose an audit protocol that catches *its own* headline claims when they would not survive a properly-tuned baseline." Two existence proofs land.

### 5.1 · Catch (a) — H09 phi_budget's 12.6% realised-stage-ratio drift

The pre-audit version of this paper claimed H09 phi_budget as a verified cross-dataset positive (CIFAR-10 85.54%, CIFAR-100 58.05% 3-seed median; +1.53 pp over baseline). Track-A revealed that H09's realised stage-parameter ratio was **1:1.41:2.45**, not the doc-claimed **1:φ:φ² = 1:1.618:2.618** (12.6% drift at stage 1). The headline was produced by a network that did NOT faithfully implement its own design doc. **Fixer-PhiScaling** (commit `519cdf3`) corrected the integer search; post-fix realised ratio is 1:1.623:2.629 (0.43% max error), and the Fixer added a mechanism-pinning test that would have caught the bug if written first:

```python
def test_phi_budget_realised_ratio():
    widths = phi_budget_widths(target_params=270_000, n_stages=3, phi=1.618033988749)
    ratio = [w / widths[0] for w in widths]
    max_err = max(abs(r - e) / e for r, e in zip(ratio, [1.0, 1.618, 2.618]))
    assert max_err < 0.01
```

Post-fix CIFAR-100 3-seed median: `phi_budget=0.5741`, baseline=0.5652 (+0.89 pp). The pre-fix median (58.05%) was ~0.6 pp HIGHER than the post-fix (57.41%), consistent with "the broken realised ratio happened to land a fortuitously-high seed-0 result." **An unaudited pipeline would have shipped the pre-fix 58.05% as a headline.**

### 5.2 · Catch (b) — the protocol's own three-prior headline apparently collapses under a Phase-9h tuned-baseline-at-lr=0.01 diagnostic

After the Fixer campaign, the protocol surfaced three Phase-8 candidates that pass a 7-seed CIFAR-100 paired Wilcoxon at α=0.05 under Holm-Bonferroni across a k=3 confirmatory family at the matched-recipe default-config slice (lr=1e-3, wd=5e-4, bs=256, AdamW, 30 ep). All three winners produce 7/7 positive paired deltas → Wilcoxon W=0, exact one-sided p=(1/2)⁷=**0.0078**, clearing Holm-Bonferroni α'=0.0167. Paired-t magnitude p-values sit 3–4 orders below the floor (5×10⁻⁵ to 8×10⁻⁴). This is the matched-recipe empirical claim of the paper (§6.5 case study).

**Then a 3-seed re-run of `baseline_resnet20_tuned_lr0.01_wd0.0005` at CIFAR-100 30 ep landed (Phase-9h, 2026-06-01 late evening; [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §14):**

| comparison | Δmean | 95% unpaired-bootstrap CI | Mann–Whitney p_one (tuned > leader) | min(tuned) > max(leader) |
|---|---:|---|---:|:---:|
| tuned (n=3, 0.6017) − `pair_gm_pdw` (n=7, 0.5786) | **+2.30 pp** | [+1.99, +2.60] pp | **0.0083** | **YES (NO overlap)** |
| tuned (n=3, 0.6017) − `slot_act_sine` (n=7, 0.5790) | **+2.27 pp** | [+1.90, +2.64] pp | **0.0111** | **YES (NO overlap)** |
| tuned (n=3, 0.6017) − `sg_only_phi_budget` (n=7, 0.5736) | **+2.81 pp** | [+2.42, +3.19] pp | **0.0083** | **YES (NO overlap)** |

The tuned-baseline n=3 σ is 0.31 pp — *tighter* than the default-config baseline σ (0.453 pp at n=7), so the diagnostic is not noise-bound. **At face value the tuned baseline OUTPERFORMS all three priors by +2.27 to +2.81 pp.** The Phase-9h result, taken in isolation, was read as the protocol catching its own at-risk-of-publication headline before any external "priors help" claim shipped.

**Caveat (load-bearing for §5.3).** The Phase-9h comparison is **apples-to-oranges**: tuned-baseline at lr=0.01, wd=5e-4, bs=256, AdamW, 30 ep — but **otherwise untuned** (default-config recipe, no extra modern tricks) — vs the three priors at their *own* default-config (lr=1e-3, wd=5e-4, bs=256, AdamW, 30 ep), also otherwise untuned. The +2.27 to +2.81 pp gap is therefore confounded by an asymmetric LR sweep (the baseline got a hill-climb; the priors did not) and is read in §5.3 as **LR-tuning confound, not prior failure**.

### 5.3 · Iso-recipe n=3 diagnostic at non-matched FLOPs (provisional)

This subsection ran all four arms (baseline + three priors) at a *shared* recipe + convergence cell: the **modern 11-trick recipe** (AdamW, cosine LR, label smoothing, RandAugment, MixUp/CutMix, EMA, etc.) at **200 ep CIFAR-100** — the project's first multi-arm convergence-regime sweep. **The diagnostic is provisional: the three priors run at ~2× the baseline FLOPs in this cell** (`flops_M` ≈ 80.8 vs baseline 41.2). The +1 pp lift below is therefore confounded with compute and is reported as a screened candidate result pending iso-FLOPs n≥7 confirmation, not as an evaluation-grade claim. Per-seed top1 are read from `experiments_modern/cifar100/<tag>_seed<s>/metrics.json`.

| Tag | Seeds (top1) | Mean | σ (pp) | Δmean vs convergent baseline | Phase-5 ordinal gate |
|---|---|---:|---:|---:|:---:|
| `baseline_resnet20_modern_200ep` | 0.6350 / 0.6383 / 0.6348 | **0.6360** | 0.197 | — | — |
| `sg_only_phi_budget` | 0.6445 / 0.6526 / 0.6483 | **0.6485** | 0.405 | **+1.24 pp** | **PASS** (min L 0.6445 > max B 0.6383) |
| `pair_gm_pdw` | 0.6457 / 0.6468 / 0.6456 | **0.6460** | 0.067 | **+1.00 pp** | **PASS** (min L 0.6456 > max B 0.6383) |
| `slot_act_sine` | 0.6461 / 0.6458 / 0.6465 | **0.6461** | 0.035 | **+1.01 pp** | **PASS** (min L 0.6458 > max B 0.6383) |

**All three priors LIFT the convergent modern-recipe baseline; all three pass the Phase-5 ordinal gate; all three deliver 3/3 positive paired deltas.** Paired Wilcoxon at n=3 hits its theoretical floor p_one = (1/2)³ = **0.125** for all three; paired-t p_one ∈ {0.0028, 0.0070, 0.0082}; 95% paired-bootstrap CIs (10 000 iter, rng=20260604): `sg_only_phi_budget` Δ=+1.24 pp, CI [+0.95, +1.43] pp; `pair_gm_pdw` Δ=+1.00 pp, CI [+0.85, +1.08] pp; `slot_act_sine` Δ=+1.01 pp, CI [+0.75, +1.17] pp — **all three CIs exclude 0 by a margin of ≥+0.75 pp on the lower bound**. The `pair_gm_pdw` and `slot_act_sine` σ (0.067 and 0.035 pp) are remarkably tight — well below the σ_default=0.453 pp at default-config n=7. Full Section-15 derivation: [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §15.

**Honest interpretation — provisional directional signal at non-matched FLOPs.** The priors carry a +1 pp directional lift at the modern 200 ep cell that is consistent in sign with the default-config n=7 cell (Δ +1.24 / +1.74 / +1.78 pp at lr=1e-3 30 ep → Δ +1.24 / +1.00 / +1.01 pp at modern 200 ep). The earlier tuned-baseline-at-lr=0.01 diagnostic apparent refutation is correctly attributed to an asymmetric LR sweep across (lr, wd) cells, not to prior failure. **However**, the +1 pp lift at the modern recipe is reported at ~2× baseline FLOPs (priors `flops_M` ≈ 80.8 vs baseline 41.2) — the [composite metric](#) in this paper penalises params and latency but not FLOPs, so the +1 pp cannot be attributed to the prior in isolation from compute. The principled-evaluation path is iso-FLOPs n≥7 confirmation at the modern recipe plus a [RegNetX-200MF (Radosavovic et al. 2020, arXiv:2003.13678)](https://arxiv.org/abs/2003.13678) comparator at the same compute budget; filed as future work.

**What this section IS:** an iso-recipe n=3 diagnostic at non-matched FLOPs (provisional), showing directional-sign consistency of the priors across two compute regimes; a screened-candidate readout pending iso-FLOPs confirmation.

**What this section is NOT:** an iso-FLOPs comparison; a formal evaluation-grade certification at the modern recipe. At n=3 the paired Wilcoxon p_one is at the theoretical floor 0.125 and cannot clear Holm-Bonferroni α'=0.0167; Mann-Whitney U at n_a=3 n_b=3 has minimum p_two = 2/C(6,3)=0.10. An n≥7 iso-FLOPs extension at the modern 200-ep cell is the principled-evaluation path; filed as future work.

**Cross-regime synthesis (sign only).** The default-config n=7 cell (Δ +1.24 / +1.74 / +1.78 pp at lr=1e-3 30 ep, paired Wilcoxon p=0.0078, non-iso-FLOPs) and the iso-recipe n=3 diagnostic at non-matched FLOPs (Δ +1.24 / +1.00 / +1.01 pp at modern 200 ep, paired Wilcoxon at the n=3 floor 0.125, priors at ~2× baseline FLOPs) are *sign-consistent*. Both cells are non-iso-FLOPs, so the +1 pp magnitudes are confounded with compute and must not be read as iso-FLOPs lifts. The methodological deliverable — the self-auditing protocol with iso-recipe / iso-FLOPs guardrails — is the headline contribution; the priors are screened candidates pending iso-FLOPs n≥7 confirmation plus a RegNetX-200MF comparator.

## 6 · Case study — nature-inspired priors on CIFAR-10/-100 (secondary empirical claim)

The design space — 10 hypotheses defended in depth + 74 design-space enumeration entries — is curated under [`hypotheses/INDEX.md`](hypotheses/INDEX.md): G1 Scaling & Growth (H01–H10), G2 Layer/Channel/Neuron (H11–H20), G3 Topologies & Graphs (H21–H30), G4 Kernels/Attention/Filters (H31–H40), G5 Optimisation/Init/Reg/NAS (H41–H50), G6 Topological/Bridging (H51–H60), G7 Cross-Paradigm Hybrids (H61–H75), G8 Esoteric Extensions (H76–H84).

### 6.1 · BROKEN findings as protocol-output case studies

- **H55 PlatonicAttention's head bias is mathematically zero.** `bias = (coords @ coords.T).mean(dim=-1)` evaluates to all-zeros for every vertex-transitive Platonic solid (vertex coords sum to centroid). PlatonicAttention was bit-equivalent to vanilla MHA. All 7 tests were shape-only. Fixed in Fixer-G6 (`16fe2b6`).
- **H67 hybrid_full was a half-on stress test.** `from .golden_rope import GoldenRoPE` raised ImportError; MetatronGraphLayer constructor signature was wrong; `which_priors_active` hardcoded `True` for 4 priors; LiquidCFC collapsed to affine + nonlinearity. Fixed in Fixer-G7 (`2e7ee45`).
- **H74 MetatronTiedConv2d's 13 alphas collapsed to one scalar.** Forward was `F.conv2d(x, W · Σα_c)`; the 13 alphas Σ-summed to a single gate. Fixed in Fixer-G7 with H40's 13 spatially-distinct circle masks.

### 6.2 · Pre-fix screening sweep (single seed, 12 epochs CIFAR-10)

35 single-prior CIFAR-10 rows at seed 0 / 12 ep; the only variant beating ResNet-20 baseline (84.78%) was **H09 phi_budget at 85.54%** (+0.76 pp). Pooled CIFAR-10 12-ep σ across 11 multi-seed tags is 0.607 pp → 2σ_pooled ≈ 1.21 pp; the 99th-percentile single-seed Δ across 58 seed-0 non-baseline tags is +0.96 pp, INSIDE the 2σ band. **No CIFAR-10 12-ep seed-0 tag has prima-facie statistical credibility as a positive.** All 35 rows are screening data per Rule 28.

### 6.3 · Post-fix sweep and combo ladder (case-study for Rule 23)

The catastrophic full-hybrid `sg_full_fib` (six priors stacked on the same NaturePriorBlock forward path) lost **−11.54 pp** below baseline at 12-ep CIFAR-10 — the cautionary tale that motivated CLAUDE.md Rule 23 (orthogonal-axes-only compounding). The Phase-7 combo ladder adds one orthogonal axis per row; the n=1 screening data identifies `plr` (phi-LR-schedule) as the single most destructive axis (combo4→combo5 drop −5.66 pp). Rule 23 was derived from this n=1 screening data; replication at n=7 to establish the "3 axes good, 6 axes bad" threshold with statistical force is Phase-9d future work.

### 6.4 · Honest comparison to SOTA

ResNet-20 at the canonical 164-epoch SOTA recipe (He CVPR 2016) reaches ~91.25% CIFAR-10 top-1. Our 12-epoch screening budget produces ~84.78% baseline — a **6.5-pp shortfall from convergence**. The Phase-8 winners' +1.24 to +1.78 pp CIFAR-100 lifts (next subsection) are less than 1/4 the gap-to-SOTA. We make **no** SOTA claim at this scale.

### 6.5 · Three Phase-8 candidates — matched-recipe certification (worked example)

**Default-config n=7 cert ([`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §§0–6):**

| tag | C100 mean | Δmean | 95% bootstrap CI on Δmean | Wilcoxon p_one | Paired-t p_one | Holm α'=0.0167 |
|---|---:|---:|---|---:|---:|:---:|
| `pair_gm_pdw` (H09+H48+H44 stack) | 0.5786 | **+1.74 pp** | [+1.42, +2.09] | 0.0078 | 5.1×10⁻⁵ | **YES** |
| `slot_act_sine` (H81 SIREN) | 0.5790 | **+1.78 pp** | [+1.38, +2.18] | 0.0078 | 1.2×10⁻⁴ | **YES** |
| `sg_only_phi_budget` (H09 post-fix) | 0.5736 | **+1.24 pp** | [+0.84, +1.67] | 0.0078 | 8.1×10⁻⁴ | **YES** |
| `baseline_resnet20` (rail) | 0.5612 | — | (σ=0.451 pp) | — | — | — |

**Hill-climbed-best robustness (Phase-9a, n=3 each).** Hill-climbed-baseline median = 0.5929 at {lr=3e-3, wd=5e-4, bs=256, AdamW}. Hill-climbed leader medians: `sg_only_phi_budget` 0.6049 (Δmean +0.79 pp, 95% CI **[−0.32, +1.76] — includes 0**); `pair_gm_pdw` 0.6109 (Δmean +1.22 pp, [+0.15, +1.99]); `slot_act_sine` 0.6137 (Δmean +1.31 pp, [+0.20, +2.23]). The directional signal carries across both tuning regimes, but the `sg_only_phi_budget` hill-climbed CI includes 0.

**Iso-tuned-cell re-certification at n=7 (Phase-9f closeout, 2026-06-01).** At the iso-tuned cell (lr=3e-3, wd=5e-4, bs=128, AdamW for baseline / `pair_gm_pdw` / `sg_only_phi_budget`; wd=2e-3 for `slot_act_sine`), iso-tuned baseline (n=7) mean=0.6000, σ_iso=**0.920 pp** — 2.03× wider than σ_default=0.453 pp. Paired Δmeans shrink to +0.79 / +0.66 / +0.25 pp; paired Wilcoxon p_one ∈ {0.1094, 0.0781, 0.3750}; **none clears α=0.05.** The Phase-5 ordinal gate FAILS at iso-tuned n=7 for all three (max iso-tuned baseline = 0.6075).

**Phase-9g Controls 1–4 honest results (2026-06-01 PM; [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §13).** Control 1 (φ attribution): `pair_nonphi_3axis` mean = 0.5718; paired Δ vs `pair_gm_pdw` = +0.61 pp paired (2/3 positive, p_one=0.25) — **φ-specific story partially refuted; 3-axis structure carries ~61% of the lift**. Control 2 (SIREN attribution): `slot_act_tanh` BEATS `slot_act_sine` by +0.48 pp paired (3/3 positive at n=3 floor) — **SIREN-specific story REFUTED; cert is generic activation engineering**. Control 3a (tuned ResNet-20): single-seed best cell (lr=0.01 wd=5e-4 bs=256 AdamW) = **0.5984**, sitting +1.94 to +2.48 pp above all three winners — the n=3 closure (§5.2 Phase-9h) confirms this at α=0.05. Control 4 (H71 IcosaRoPE3D vs 1D-RoPE on rotated CIFAR-10): Δ=+0.18 pp, **INCONCLUSIVE**.

**Per-prior 3-tier evidence ladder.** The honest framing per prior is:

| Prior | Tier 1 — screening (n=1, 12 ep CIFAR-10) | Tier 2 — n=7 default-config (30 ep, non-iso-FLOPs) | Tier 3 — iso-recipe n=3 at non-matched FLOPs (200 ep, ~2× baseline FLOPs) |
|---|---:|---:|---:|
| `pair_gm_pdw` | +1.07 pp (seed-0) | +1.74 pp, Wilcoxon p=0.0078 | +1.00 pp at Wilcoxon floor; ~61% non-φ per Control 1 |
| `slot_act_sine` | +0.07 pp (seed-0) | +1.78 pp, Wilcoxon p=0.0078 | +1.01 pp at Wilcoxon floor; SIREN replication, tanh > sine in Control 2 |
| `sg_only_phi_budget` | +0.76 pp (seed-0) | +1.24 pp, Wilcoxon p=0.0078 | +1.24 pp at Wilcoxon floor; hill-climbed-n=3 CI includes 0 |

The Tier-3 column makes the FLOP gap explicit: the priors run at ~80.8 M FLOPs vs baseline 41.2 M, so the +1 pp at the modern recipe is confounded with compute. **No tier reported is iso-FLOPs.** The principled-evaluation path is iso-FLOPs n≥7 at the modern recipe plus a [RegNetX-200MF (Radosavovic et al. 2020, arXiv:2003.13678)](https://arxiv.org/abs/2003.13678) comparator (see Limitations §7).

**Cross-regime synthesis (sign only)** ([`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md) §14 + §15): a tuned-baseline-at-lr=0.01 diagnostic *apparently* refutes the default-config cert by +2.27 to +2.81 pp, but that comparison is apples-to-oranges (the baseline received an LR sweep; the priors did not); the iso-recipe n=3 diagnostic at non-matched FLOPs on the modern 11-trick recipe at 200 ep returns sign-consistent +1.00 to +1.24 pp lifts at the n=3 Wilcoxon floor, at ~2× baseline FLOPs. The +1 pp magnitudes are confounded with compute; **the screened-candidate framing pending iso-FLOPs n≥7 confirmation at the modern recipe + RegNetX-200MF comparator is the honest reading.** The protocol's headline catch is the audit pipeline + iso-recipe / iso-FLOPs guardrails, not the priors themselves.

## 7 · Limitations

- **Single-LLM-family auditor — the structural limit.** All auditor agents (implementer, impl-critic, sci-critic, Fixer, calibration auditor) share model family (Claude Opus 4.7). Two partial closures along orthogonal axes: (a) n=62 third-party calibration with Fisher exact two-sided p=1.94×10⁻⁵ (§4); (b) cross-family methodologically-diverse re-audit on 10 of 18 MAJOR/BROKEN, 8/10 strict CONCORDANT (§4.3). True non-Claude external auditor remains open future work.
- **CIFAR-only scale — the empirical limit.** All experiments are CIFAR-10 12 ep / CIFAR-100 30 / 200 ep on ResNet-20-class scaffolds. Baseline sits 6.5 pp below 164-ep SOTA; no ImageNet-scale validation; no transformer-track training run on any of the 10 attention-backbone hypotheses in the substrate.
- **Non-iso-FLOPs in every reported cell.** The three priors in the modern-recipe 200 ep diagnostic run at ~2× baseline FLOPs (`flops_M` ≈ 80.8 vs 41.2). The SHA-256-fingerprinted composite metric penalises params and latency but **not** FLOPs; consequently every reported "lift" in this paper is confounded with compute. An iso-FLOPs comparator at the modern recipe plus a [RegNetX-200MF (Radosavovic et al. 2020, arXiv:2003.13678)](https://arxiv.org/abs/2003.13678) head-to-head at the same FLOP envelope is the principled-evaluation path; filed as future work.
- **Single-seed for most case-study sweep rows.** Only baseline, `phi_budget`, `golden_momentum` carry 3-seed on CIFAR-10; the rest are seed-0. Treated as screening per Rule 28.
- **HARKing acknowledgement on the screening-vs-evaluation distinction.** The distinction was authored post-hoc, after negatives had been observed (§7.3.1 of the prior revision; preserved as Rule 28 prospective discipline).
- **POSI on Phase-8 family selection.** The k=3 confirmatory family is post-screening; the k=49 strict POSI bound is reported separately — paired-t magnitude clears 0.001 for two of the three candidates; `sg_only_phi_budget` does not.
- **Iso-tuned-cell re-certification at n=7 fails for all three candidates** (§6.5). The default-config matched-recipe cell is preserved as a formally-correct statement at non-iso-FLOPs; the iso-tuned cell at default-recipe does not re-clear α=0.05 at this sample size.
- **Iso-recipe n=3 diagnostic at non-matched FLOPs hits the Wilcoxon floor.** The §5.3 +1.00 to +1.24 pp lifts are sign-consistent with the default-config cell but **NOT** an evaluation-grade claim: at n=3 the paired Wilcoxon p_one floor is (1/2)³=0.125, well above Holm-Bonferroni α'=0.0167. An iso-FLOPs n≥7 extension at the modern 200-ep cell is the principled-evaluation path (~39 additional GPU-h on the 4090 Laptop after a FLOP-budget pinning of the priors), filed as future work.
- **Cross-domain portability is claimed by construction, not demonstrated.** The seven content-agnostic skills expose a content-agnostic templating API; CIFAR-conditional in parameter defaults; no sister-repo replication executed.
- **Self-grading caveat.** All auditor agents share model family; all internal verdicts on this paper are internal QA only and are not external review. The interpretation of the +1 pp lift as "screened candidate at non-iso-FLOPs" rather than "evaluation winner" is the load-bearing honest framing in this submission.

## 8 · Related Work

**Geometric / topological priors.** The nature-inspired-priors substrate sits inside the GDL taxonomy of Bronstein et al. (2021); specific anchors are Hoogeboom et al. (ICLR 2018) for hexagonal lattices, Cohen et al. (ICML 2019) for gauge / icosahedral equivariance, Larsson et al. (ICLR 2017) for fractal recursion, Pittorino et al. (ICML 2022) for toroidal flat-minima geometry, and Islam et al. (2025 preprint) for Platonic-solid attention biases. RegNet (Radosavovic et al. CVPR 2020) establishes the Pareto region `w_m ∈ [2.5, 2.9]` that contains but does not require φ — the literature anchor for H09's DERIVATIVE+TESTABLE verdict.

**LLM-agent autoresearch and red-teaming.** This work sits in the tradition of mutation testing (DeMillo et al. 1978; Jia & Harman 2011; Tian et al. 2023 for code-LLM mutation), self-critique / self-refinement (Saunders et al. 2022; Madaan et al. 2023), LLM-as-judge methodology (Zheng et al. 2023), self-preference bias (Panickssery et al. 2024; Wang et al. 2023), and pre-registration discipline (Munafò et al. 2017; Nosek et al. 2018; Gelman & Loken 2013 for "garden of forking paths") ported into the LLM-implements-LLM-audits regime. To our knowledge, no prior work systematically audits an LLM-agent autoresearch protocol against a third-party-code calibration substrate large enough to distinguish the project's MAJOR/BROKEN tier from a clean-code floor at α<10⁻⁴. The combination of dual-track audit + Fixer-mechanism-test contract + screening-vs-evaluation tiering + per-experiment-page + auto-checkpoint loop, plus the empirical existence proof of §5, is the contribution; no individual piece is novel.

**Modern training recipe.** The 11-trick modern CIFAR-100 recipe (AdamW, cosine LR, label smoothing, RandAugment, MixUp/CutMix, EMA, Random Erasing, stochastic depth, weight decay, linear warmup, 200 ep) is taken from Bello et al. (ICML 2021) "Revisiting ResNets" and Wightman et al. (NeurIPSW 2021) "ResNet Strikes Back". These two papers are the direct prior art for the §5.3 iso-recipe diagnostic.

**Equivariance.** Cohen & Welling (ICML 2016) for foundational G-equivariant CNNs; Weiler & Cesa (NeurIPS 2019) for e2cnn (steerable E(2)-equivariance); Geiger & Smidt (2022) for e3nn (Euclidean neural networks). The icosahedral / spherical claims in this paper's substrate (H24/H53/H55/H71) are not steerable group convolutions; the e3nn-based real equivariance is filed as future work.

**Statistical machinery.** Holm (1979) for step-down Bonferroni; Wilcoxon (1945) signed-rank for paired n=7; Mann-Whitney (1947) for unpaired n=3 vs n=7; Berk et al. (2013) for POSI correction across the screening universe; Sitzmann et al. (NeurIPS 2020) for SIREN's known activation-engineering benefit (the literature anchor for the `slot_act_sine` no-φ-content interpretation); Reddi et al. (ICLR 2018) for the β2 long-horizon test (H41 caveat); Loshchilov & Hutter (ICLR 2019) for AdamW.

## 9 · Conclusion

**The protocol is the contribution; the priors are the substrate.** After implementations spanning 10 hypotheses defended in depth + 74 design-space enumeration entries, 41+24+15+3 implementation verdicts, 1+30+40+3+2+5 scientific verdicts, 8 mechanism-correcting Fixer commits, ~40 GPU-hours of CIFAR-10/-100 ablation, and a 62-hypothesis third-party-code calibration substrate, the protocol delivered two load-bearing claims.

**First**, the protocol's MAJOR/BROKEN audit tier is statistically distinguishable from a clean-code floor at α=0.05 by a margin of ≈ 2500× (Fisher exact two-sided p=1.94×10⁻⁵; Wilson 95% CIs non-overlapping; n=62 calibration from `pytorch/vision` + `timm` + HF Transformers + Lightning Bolts + `torch.optim` + Mamba). The audit doctrine that returns 0/62 on production-quality third-party code returns 18/83 on the project's own substrate. The 22-pp tier-separated excess is the diagnostically-credible signal — not the 51% aggregate non-PASS rate.

**Second**, the protocol surfaced — and correctly attributed — its own headline-interpretation drift across recipes and FLOP regimes. Three matched-recipe candidates cleared Holm-Bonferroni α'=0.0167 at n=7 in the default-config cell (non-iso-FLOPs, Wilcoxon p=0.0078 each). A tuned-baseline n=3 diagnostic at lr=0.01 *apparently* refuted them (tuned ResNet-20 beat all three by +2.27 to +2.81 pp) — but the comparison is apples-to-oranges (asymmetric LR sweep across (lr, wd) cells). An iso-recipe n=3 diagnostic at non-matched FLOPs on the modern 11-trick recipe at 200 ep returns sign-consistent +1.00 to +1.24 pp lifts at the n=3 Wilcoxon floor, **with the priors running at ~2× baseline FLOPs**. No reported cell in this paper is iso-FLOPs. The priors are therefore **screened candidates with sign-consistent +1 pp directional lift across the default-config and the modern-recipe non-iso-FLOPs cells; iso-FLOPs n≥7 confirmation at the modern recipe plus a RegNetX-200MF comparator at the same FLOP envelope is the principled-evaluation path filed as future work.**

**The protocol's value is precisely that it surfaced its own at-risk-of-publication framing before any external claim shipped.** The audit + Fixer + per-experiment-page + auto-checkpoint + iso-recipe / iso-FLOPs guardrail stack, encoded in CLAUDE.md Rules 20–28 and packaged as seven content-agnostic skills in [`skills/`](skills/), is what we offer the community: portable infrastructure for distinguishing signal from numerology — and for catching one's own LLM-agent autoresearch protocol whether it drifts toward a tuning artifact or a compute artifact. The cross-domain demonstration of that portability, an iso-FLOPs n≥7 modern-recipe extension, and a true non-Claude external auditor on the same 10 MAJOR/BROKEN findings, are open future work.

## References

Bello I, Fedus W, Du X, Cubuk ED, Srinivas A, Lin T-Y, Shlens J, Zoph B. 2021 ICML. *Revisiting ResNets: Improved Training and Scaling Strategies*. arXiv:2103.07579. — Modern 11-trick recipe lineage.
Berk R, Brown L, Buja A, Zhang K, Zhao L. 2013. *Valid post-selection inference*. arXiv:1306.1107. — POSI correction across screening universe.
Bronstein MM, Bruna J, Cohen TS, Veličković P. 2021. *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. arXiv:2104.13478. — Canonical GDL synthesis.
Cohen TS, Welling M. 2016 ICML. *Group Equivariant Convolutional Networks*. arXiv:1602.07576. — Foundational equivariance prior art.
Cohen TS, Weiler M, Kicanaoglu B, Welling M. 2019 ICML. *Gauge Equivariant Convolutional Networks and the Icosahedral CNN*. arXiv:1902.04615. — H24/H53/H55 icosahedral equivariance anchor.
DeMillo RA, Lipton RJ, Sayward FG. 1978 *IEEE Computer*. *Hints on Test Data Selection: Help for the Practicing Programmer*. — Mutation testing origin.
Gelman A, Loken E. 2013. *The Garden of Forking Paths*. Columbia statistics preprint. — Post-hoc analysis-branch multiplication.
Geiger M, Smidt T. 2022. *e3nn: Euclidean Neural Networks*. arXiv:2207.09453. — Steerable group convolution.
He K, Zhang X, Ren S, Sun J. 2016 CVPR. *Deep Residual Learning for Image Recognition*. arXiv:1512.03385. — ResNet baseline at 91.25% CIFAR-10 / 164 ep.
He T, Zhang Z, Zhang H, Zhang Z, Xie J, Li M. 2019 CVPR. *Bag of Tricks for Image Classification with Convolutional Neural Networks*. arXiv:1812.01187. — Orthogonal-axis regularizer stacking band.
Holm S. 1979 *Scand. J. Statistics*. *A Simple Sequentially Rejective Multiple Test Procedure*. — Step-down Bonferroni.
Hoogeboom E, Peters JWT, Cohen TS, Welling M. 2018 ICLR. *HexaConv*. arXiv:1803.02108. — Hexagonal lattice convolution; H21.
Islam MM, Anand R, Wessels DR, et al. 2025. *Platonic Transformers: A Solid Choice for Equivariance*. arXiv:2510.03511. — H55.
Jia Y, Harman M. 2011 *IEEE TSE*. *An Analysis and Survey of the Development of Mutation Testing*. — Mutation testing lineage.
Larsson G, Maire M, Shakhnarovich G. 2017 ICLR. *FractalNet: Ultra-Deep Neural Networks without Residuals*. arXiv:1605.07648. — H05.
Loshchilov I, Hutter F. 2019 ICLR. *Decoupled Weight Decay Regularization (AdamW)*. arXiv:1711.05101.
Madaan A, Tandon N, Gupta P, et al. 2023 NeurIPS. *Self-Refine: Iterative Refinement with Self-Feedback*. arXiv:2303.17651. — Self-critique / Fixer lineage.
Munafò MR, Nosek BA, Bishop DVM, et al. 2017 *Nature Human Behaviour*. *A manifesto for reproducible science*. — Pre-registration discipline.
Nosek BA, Ebersole CR, DeHaven AC, Mellor DT. 2018 *PNAS*. *The preregistration revolution*. — Pre-registration discipline.
Panickssery A, Bowman SR, Feng S. 2024. *LLM Evaluator Self-Preference Bias*. arXiv:2404.13076. — Same-family LLM-judge bias.
Pittorino F, Ferraro A, Perugini G, Feinauer C, Baldassi C, Zecchina R. 2022 ICML. *Deep Networks on Toroids*. arXiv:2202.03038. — H22.
Radosavovic I, Kosaraju RP, Girshick R, He K, Dollár P. 2020 CVPR. *Designing Network Design Spaces (RegNet)*. arXiv:2003.13678. — H09 phi_budget literature anchor.
Ramsauer H, Schäfl B, Lehner J, et al. 2020 ICLR (2021 proceedings). *Hopfield Networks is All You Need*. arXiv:2008.02217. — H32 / H77.
Reddi SJ, Kale S, Kumar S. 2018 ICLR. *On the Convergence of Adam and Beyond*. arXiv:1904.09237. — H41 long-horizon caveat.
Saunders W, Yeh C, Wu J, et al. 2022 OpenAI. *Self-Critiquing Models for Assisting Human Evaluators*. arXiv:2206.05802. — Self-critique lineage.
Sitzmann V, Martel JNP, Bergman AW, Lindell DB, Wetzstein G. 2020 NeurIPS. *Implicit Neural Representations with Periodic Activation Functions (SIREN)*. arXiv:2006.09661. — `slot_act_sine` literature anchor; protocol caught it as a SIREN replication.
Su J, Lu Y, Pan S, Wen B, Liu Y. 2024 Neurocomputing. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864. — H34 / H71.
Tian R, Ye M, Wang S, et al. 2023. *Code-LLM-Based Mutation Testing*. arXiv:2302.10039. — Mutation testing for LLM-generated code.
Wang P, Li L, Chen L, Cai Z, Zhu D, Lin B, Cao Y, Liu Q, Liu T, Sui Z. 2023 EMNLP. *Large Language Models are not Fair Evaluators*. arXiv:2305.17926. — LLM-judge bias.
Weiler M, Cesa G. 2019 NeurIPS. *General E(2)-Equivariant Steerable CNNs (e2cnn)*. arXiv:1911.08251. — Steerable equivariance.
Wightman R, Touvron H, Jégou H. 2021 NeurIPSW. *ResNet Strikes Back: An improved training procedure in timm*. arXiv:2110.00476. — Modern 11-trick recipe lineage.
Wilcoxon F. 1945 *Biometrics*. *Individual Comparisons by Ranking Methods*. — Paired signed-rank.
Zheng L, Chiang W-L, Sheng Y, et al. 2023 NeurIPS DBT. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. arXiv:2306.05685. — Foundational LLM-as-judge paper.

Full bibliography in [`paper/NATURE_INSPIRED_NETWORKS.md`](paper/NATURE_INSPIRED_NETWORKS.md).

## Repository pointers

- Pre-print and source: <https://github.com/dlmastery/nature_inspired_networks>
- Live dashboard: <https://dlmastery.github.io/nature_inspired_networks/>
- Statistical-test analysis: [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md).
- Audit calibration (n=62): [`audits/AUDIT_CALIBRATION_THIRD_PARTY.md`](audits/AUDIT_CALIBRATION_THIRD_PARTY.md).
- Cross-family methodologically-diverse re-audit: [`audits/CROSS_FAMILY_HONEST_REAUDIT.md`](audits/CROSS_FAMILY_HONEST_REAUDIT.md).
- Controls plan (Controls 1–4): [`controls/PLAN.md`](controls/PLAN.md).
- ICML 2027 reviewer rebuttal: [`audits/ICML_REVIEWS_2026-05-30/REBUTTAL.md`](audits/ICML_REVIEWS_2026-05-30/REBUTTAL.md).
- Operator quick-reference: [`CLAUDE.md`](CLAUDE.md) §8.

---

*Submission-candidate; external review pending. PAPER.md serves as the arXiv preprint; the main-track conference version is the §1–§9 narrative. The audit-calibration result (Fisher p=1.94×10⁻⁵ at n=62) and the load-bearing existence proof (H09 `phi_budget` realised-ratio catch) are the headline; the nature-inspired-priors substrate (10 hypotheses defended in depth + 74 design-space enumeration entries) is the case study.*
