# ICML 2027 Reviewer R3 — Methodology / Protocol / Software-Engineering Track

**Paper:** *A Skeptical Protocol for Nature-Inspired Neural-Network Priors: Methodological Contribution and an 84-Hypothesis Dual-Track Audit on CIFAR-10/-100*
**Repository snapshot:** `dlmastery/nature_inspired_networks` @ post-`ca32fcd`
**Reviewer persona:** R3 — methodology / reproducibility infrastructure / agent-based-research-correctness specialist
**Date:** 2026-05-30
**Track:** ICML main conference (methods / infrastructure / meta-research)

---

## 0. Summary recommendation

- **Overall rating:** **5 — Marginally below acceptance threshold** (ICML 1–10 scale).
- **Recommendation:** **WEAK REJECT for ICML main track. STRONG ACCEPT (≥7) for ICML Datasets & Benchmarks track or a Reproducibility / Meta-Research workshop**, conditional on revisions in §6.
- **Confidence:** 4/5 (I have read the load-bearing protocol artefacts directly and spot-checked five skills; I have not re-executed the sweep).

**One-line verdict.** This is a real methodological contribution — the dual-track audit + Fixer + per-experiment-page + mechanism-pinning-test discipline is a substantive, well-engineered port of code-review-and-mutation-testing ideas into the LLM-driven autoresearch setting — but the *novelty over existing literature is incremental*, the *content-agnostic transferability claim is by construction, not demonstrated*, and the *auditor-self-grading circularity* is a load-bearing weakness the paper now acknowledges but does not yet solve. As an ICML main-track paper, the empirical case studies (even at n=7 Holm-cleared) are insufficient infrastructure novelty for the main track. As a D&B / methods-workshop paper, this is well above bar.

---

## 1. Paper-level summary (as R3 understands it)

The authors propose a 29-skill, 28-rule, software-engineered protocol for running adversarial audits on autoresearch campaigns whose implementations are themselves written by LLM agents. The protocol layers (a) an 8-agent implementation-critic team that grades code-vs-doc faithfulness on a PASS/MINOR/MAJOR/BROKEN tier system, (b) an 8-agent scientific-critic team that issues a NOVEL+TESTABLE / DERIVATIVE+TESTABLE / NUMEROLOGY / FALSIFIED / UNFALSIFIABLE / INFRASTRUCTURE verdict against the *idea* (not the code), and (c) an 8-agent Fixer campaign whose patches must each ship with a mechanism-verifying test that would have failed the pre-fix implementation. The protocol is operationalised as Rules 20–28 of `CLAUDE.md` and packaged as 29 content-agnostic skills under `skills/`. Empirical calibration runs the protocol on an 84-hypothesis nature-inspired neural-prior design space (CIFAR-10/-100). Headline calibration: 51 % of audited implementations land non-PASS (3 BROKEN / 15 MAJOR / 24 MINOR of 83); 1/81 hypotheses rate NOVEL+TESTABLE; three protocol-screened Phase-8 candidates pass a Holm-Bonferroni-corrected paired Wilcoxon at α=0.05 under n=7 (`pair_gm_pdw`, `slot_act_sine`, `sg_only_phi_budget`).

R3 is reviewing **the protocol as the contribution** (per the authors' own framing in §1.1 and §8). I treat the empirical headline numbers as illustrative protocol output and grade them under §3.

---

## 2. Top three strengths

### S1. The Fixer-with-mechanism-verifying-test discipline is a genuine and well-engineered idea

Rule 21 + the `autoresearch-fixer-campaign` skill require every patch to ship with at least one test that (a) asserts the *claimed mechanism* (not the output shape), (b) fails on the pre-fix code, and (c) passes on the post-fix code. The H09 phi_budget case study makes the value tangible: the pre-Fixer test only asserted `output.shape == (B, num_classes)` and would have passed an implementation whose realised stage-parameter ratio was 1:1.41:2.45 (a 12.6 % drift from the doc-claimed 1:φ:φ²). The post-Fixer test asserts the realised ratio is within 1 % of `[1, φ, φ²]`. This is, in the ML research setting, a port of *mutation-testing-as-test-strength-evaluator* (Jia & Harman 2011 TSE, Petrović & Ivanković 2018 ICSE) implemented at the API of "claimed mechanism in the design doc." Mechanism-pinning tests are durable artefacts — the patched code may improve further, but the test invariant remains. This is the strongest software-engineering contribution in the paper and the one that most clearly transfers to non-ML settings.

### S2. The dual-track separation of "is the code faithful" from "is the idea defensible" is methodologically clean

Track A (impl-critic, file-scoped to `src/`) and Track B (sci-critic, file-scoped to `hypotheses/`) attack orthogonal failure modes. The verdict-tier systems are disjoint by construction (PASS/MINOR/MAJOR/BROKEN is a code-fidelity dimension; NOVEL/DERIVATIVE/NUMEROLOGY/FALSIFIED/UNFALSIFIABLE/INFRASTRUCTURE is an idea-merit dimension). The skill files (`autoresearch-critic-team/SKILL.md`, `autoresearch-scicritic-team/SKILL.md`) explicitly tabulate this orthogonality ("can both apply? yes — they're orthogonal — code can be correct (impl-PASS) while idea is numerology (sci-NUMEROLOGY)"). The separation is *not* novel in the abstract (peer review has always had a "is the code right" track via reviewers like ICLR Code-Reproducibility and a "is the idea interesting" track via standard reviewing), but its implementation as two independent agent teams with disjoint file-target scopes and parallel-but-blind execution is, to my knowledge, a new operationalisation. Compare: AutoML-Zero (Real et al. 2020 ICML) runs an evolutionary search but does not audit; AutoGluon and Auto-PyTorch run AutoML but do not separate code-fidelity from idea-merit; the ML4Code literature (e.g., Bird et al. 2011 FSE on code review at scale) is the methodological precedent but has not been ported into the autoresearch loop.

### S3. The third-party calibration on pytorch/vision is a serious move and the resulting re-framing is honest

The 2026-05-30 calibration (`audits/AUDIT_CALIBRATION_THIRD_PARTY.md`) applies Track-A doctrine to 15 mechanism claims spanning `torchvision.models.{resnet,densenet,vgg,squeezenet,mobilenetv2}` and `torch.optim.{Adam,SGD}` + `torch.nn.{init,BatchNorm2d}` + `torch.optim.lr_scheduler.CosineAnnealingLR`, yielding 10 PASS / 5 MINOR / 0 MAJOR / 0 BROKEN (33 % non-PASS). The paper now reports both the aggregate 51 % AND the MAJOR/BROKEN-tier 22-pp excess over the calibration's 0 % floor, and explicitly recommends that the 22-pp MAJOR/BROKEN-tier delta be the diagnostically-credible number rather than the aggregate. This is the most credible move in the revision; without it the audit's "we found 51 % defects" headline is observationally equivalent to "hyperactive auditor team". The honesty about n=15 vs n=83 and the chi-squared p≈0.22 caveat (Appendix B) is rare in ML papers and should be praised. R3 nevertheless flags two concerns in §3.

---

## 3. Top three weaknesses

### W1. Protocol novelty over existing literature is incremental, not transformational

The 28-rule + 29-skill formalisation is impressive *as engineering* but is, when decomposed, a port of established practices into the LLM-agent setting:

| Protocol piece | Established precedent |
|---|---|
| Append-only experiment log (Rule 3) | MLflow tracking, Weights & Biases append-only runs, scientific lab notebooks (Klein et al. 2014) |
| SHA-256-fingerprinted composite (Rule 2) | Git content-addressing; ML papers since at least Pineau et al. 2020 NeurIPS reproducibility checklist explicitly fingerprint scoring code |
| Mechanism-pinning tests (Rule 21) | Mutation testing (Jia & Harman 2011 TSE), property-based testing (Hughes 2007 QuickCheck), `hypothesis` library culture |
| Dual-track code-vs-idea separation | NeurIPS / ICML reviewing has always separated code reproducibility (Code-Reproducibility Checklist, Pineau et al. 2020) from scientific merit |
| Per-experiment-page archive (Rule 24) | Papers-with-code, OpenReview supplementary, MLflow UI |
| Auto-checkpoint loop (Rule 20) | Cron + post-commit hooks; CI/CD time-based commit cadence is decades old |
| Q&A-test correspondence (Rule 25) | Contract-driven design (Meyer 1992); requirements-traceability matrices (Ramesh & Jarke 2001) |
| Compose-orthogonal-axes-only (Rule 23) | ANOVA-style factorial design; Box & Hunter 1957 |
| Pre-registration of screening vs evaluation (Rule 28) | Nosek et al. 2018 PNAS pre-registration literature; AsPredicted.org; psychology's "open science framework" |
| Triple-check data-split audit (`autoresearch-data-split-audit`) | WILDS benchmark protocols (Koh et al. 2021 ICML); medical-ML domain-shift literature (Castro et al. 2020 NatureComms) |
| Shuffle-test (`autoresearch-shuffle-test`) | Permutation tests in ML go back to Ojala & Garriga 2010 JMLR |

What *is* novel:
- The integration of all these pieces into a single executable protocol gated by a `CompositeFingerprintError`-style refuse-to-launch runner.
- The Fixer-must-add-mechanism-test contract (S1 above).
- The operationalisation in the *LLM-agent-implements-LLM-agent-audits* regime.

What is *not* novel:
- The individual pieces.
- The "protocol is the contribution" framing per se (cf. CheckList for NLP, Ribeiro et al. 2020 ACL; HELM, Liang et al. 2023 TMLR).

For ICML main track, R3 expected either (a) a transformative single idea or (b) a substantial empirical demonstration that the protocol moves the field's reproducibility/defect-detection frontier. The protocol does (b) only in a single project on itself (CIFAR-10/-100); §7.4-6 explicitly defers cross-domain demonstration to future work.

### W2. Auditor-self-grading at the model-family level is unresolved and load-bearing

Implementer agents, impl-critic agents, sci-critic agents, and Fixer agents are *all* Claude Opus 4.7 (§1.3). The "independence" enforced at the disjoint-scope / disjoint-file-target level is real *as software engineering* but is, at the inferential level, methodological theatre. The third-party calibration on pytorch/vision (S3) helps — it shows MAJOR/BROKEN is empty on a peer-trusted reference codebase — but does not resolve the deeper concern: the same model family that *wrote* the H09 broken realised-ratio code is the model family that *audited* it and *patched* it. Two specific risks:

1. **Audit blind spots are correlated with implementation blind spots.** If Opus 4.7 has a systematic gap (e.g., misunderstanding `register_buffer` vs `register_parameter`, or always-zero biases under vertex-transitive coordinate symmetry), both the implementer and the auditor share that gap. The H55 PlatonicAttention bug (`(coords @ coords.T).mean = 0` for vertex-transitive solids) was caught because the math is verifiable *outside* the agent ecosystem (the centroid identity is independently provable); the broader claim "the audit team is diagnostic" extrapolates from a finding that survives the caveat to findings that may not.
2. **The pytorch/vision calibration auditor is also Opus 4.7.** The 0 % MAJOR/BROKEN on torchvision could mean torchvision is clean, OR it could mean Opus-as-auditor has a systematic reluctance to grade well-known canonical code harshly (anchoring on its prior over "code that ships in `torch.nn`"). The calibration document's Appendix B-1 acknowledges "confirmation bias risk", but the only mitigation is a verbatim-doctrine recitation. A genuine resolution requires either (a) a *different* model family auditing the same calibration sample, or (b) human expert audit on a subsample to bound the Opus-as-auditor false-negative rate.

R3 is sympathetic to the cost of (a) and (b), but the paper currently treats the third-party calibration as resolving the issue (§5.8 "RESOLVED"). It does not; it tightens the credible-interval but does not close the circularity.

A related concern: the published 51 % non-PASS rate is reported *over the project's design space*, which the paper acknowledges is itself a selection-biased LLM-curated set of 84 hypotheses (§7.3 F1). Conditioning on a curated design space audited by the curator's model family produces non-PASS rates that are jointly conditional on both biases. The paper acknowledges this but does not measure it.

### W3. Content-agnostic transferability is claimed by construction but not demonstrated, and the parity audit is weaker evidence than the paper implies

§1.1 contribution 3 states the 29 skills are "content-agnostic so any future autoresearch project can pick up the protocol unchanged." The empirical evidence offered is `audits/SKILLS_PARITY_AUDIT.md`, which compares the local skill library against five sister repos (`autoresearch`, `autoresearchspy`, `autoresearchimage`, `autoresearchtabular`, `autoresearchindexstock`). R3 notes three problems with this evidence:

1. **None of the five sister repos has a `skills/` directory.** The parity audit's own methodology section (§1) admits the sister repos codify their protocols in `CLAUDE.md` prose + `scripts/*.py` + `core/*.py`. The comparison is therefore `local skill ↔ sister CLAUDE.md prose`, not `skill ↔ skill`. This is the weaker direction of the comparison and the parity audit explicitly flags it as a methodological asymmetry (honesty note #1).
2. **The "parity" measured is "can I find a paragraph in the sister CLAUDE.md that intent-matches my skill"**, not "have I run my skill on the sister project and seen it work." The audit's "INTENT MATCHED" verdicts (rows in §3.3, §3.4) are protocol-piece-level analogies, not executions. A genuine content-agnostic test would clone `autoresearchtabular`, drop in the 29 skills, run `autoresearch-critic-team` against the Higgs implementation, and report the resulting non-PASS rate vs the local non-PASS rate. This has not been done.
3. **Several skills are explicitly project-conditional even at the prose level.** The `autoresearch-per-hypothesis-hillclimb` 5-axis cube (lr × wd × batch × seed × optimizer) is a CIFAR-scale image-classification cube; tabular GBMs (`autoresearchtabular`) have a different knob set (n_estimators, max_depth, min_child_weight, eta); FX trading models (`autoresearch`) add Sharpe-budget and walk-forward windows. The skills are *content-agnostic in their templating layer* but their *parameter defaults are CIFAR-conditional*. The paper does not flag this nuance.

§7.4-6 future work calls out the cross-domain replication as the empirical test. R3 agrees but would not let the paper claim content-agnosticism as a present contribution without that demonstration.

---

## 4. R3-specific methodological commentary (deeper notes)

### 4.1 The 22-pp MAJOR/BROKEN excess: how diagnostically-credible is it really?

Sample sizes: project n=83, calibration n=15. Two-sample chi-squared p≈0.22 (not significant at α=0.05); one-sided Fisher exact on the MAJOR/BROKEN sub-tier (15/83 vs 0/15) p≈0.07 (marginal). The paper's recommended re-framing (cite both numbers, treat 22 pp as the diagnostically-credible signal) is *directionally correct* but *statistically marginal*. A larger calibration sample (the paper proposes Phase-9b on timm + Hugging Face transformers + Lightning Bolts at n=50-100) would tighten the CI. R3 considers the current calibration sufficient for the *re-framing* (S3 above) but insufficient for a *headline claim* like "the audit detected 22 pp of real defect-density excess." Recommendation in §6.

### 4.2 Mechanism-pinning tests: do they generalise the bug class or only the literal bug?

Question (D) from the brief. R3's read of the H09 Fixer test (`assert abs(realised_ratio[s] - phi**s) / phi**s < 0.01 for s in [0, 1, 2]`) is that it pins the *exact realised-ratio mechanism*, which is one step better than shape-only but one step worse than property-based. The bug class — "integer-rounding under a budget constraint can silently shift the implemented ratio away from the doc-claimed ratio" — would have been better caught by a property-based test that *sweeps* the budget and the target ratio and asserts the realised ratio is monotone in the target. The mechanism-pinning test as implemented catches *this* H09 bug; it does not catch the broader class of "budget-constrained integer-rounding mechanism drift" that might recur in (say) H08 Net2Net width-growth or H05 fractal-depth integer rounding. The Fixer-PhiScaling commit `519cdf3` did patch both H08 and H09 with mechanism tests, but the H08 test (per AUDIT_SUMMARY.md row 1, 4 new tests including Net2Net) is similarly bug-specific. R3 considers this a real but addressable weakness: the rule could be sharpened to "every Fixer patch ships with at least one *property-based* test where the assertion holds over a parameter grid", at modest cost.

### 4.3 The SHA-256-fingerprinted composite metric (Rule 2)

R3 considers this *useful infrastructure innovation* but *modest in novelty*. Git content-addressing has done this for decades; ML papers since Pineau et al. 2020 NeurIPS Reproducibility Checklist have explicitly fingerprinted scoring code. What is genuinely new is the *refuse-to-launch* enforcement: editing the formula raises `CompositeFingerprintError` at runner import. This is a useful pattern and worth advertising; it is not novel in the abstract but is novel as ergonomics. The composite formula itself (`top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms)`) is a reasonable Pareto-tradeoff scalarisation; R3 would not have chosen the same weights but is willing to grade on the *protocol* rather than the *formula choice* (the formula is a project-specific decision, fingerprinting it is the methodological contribution).

### 4.4 Per-experiment page (Rule 24): is the 10-section template a contribution?

Question (B) from the brief. R3 read `skills/autoresearch-per-experiment-page/SKILL.md` (not quoted here for brevity) and considers the 10-section template *good engineering hygiene* but *not a research contribution*. It is the autoresearch analog of Datasheets for Datasets (Gebru et al. 2018) and Model Cards (Mitchell et al. 2019) — useful, well-engineered, but a port of established documentation discipline into the autoresearch setting. The Pages-link discipline (Rule 27, absolute GitHub-blob URLs) is a tactical fix for a specific failure mode (404s on Pages-served HTML) and is appropriately not claimed as a research contribution.

### 4.5 Skill 29 in §11: is the library complete?

Question (F) from the brief. From the ICML-methodology reviewer perspective, R3 sees three skill-shaped gaps:

1. **No cross-model-family audit skill.** The single biggest open methodological question (W2) would be addressed by a skill that dispatches a *different* model family to re-audit a subsample. Even ad-hoc — "ask a human / different LLM / a hand-coded heuristic checker to re-grade 10 audit verdicts and report agreement" — would be informative. Suggested name: `autoresearch-cross-family-audit`.
2. **No property-based-test skill.** The Fixer mechanism-pinning tests (§4.2) would be sharpened by a skill that defaults to `hypothesis`-style property tests over a parameter grid. Suggested name: `autoresearch-property-based-test` (companion to `autoresearch-fixer-campaign`).
3. **No empirical cross-domain replication skill.** The content-agnostic transferability claim (W3) would be operationalised by a skill that clones a sister repo, drops in the local skill-pack, runs `autoresearch-critic-team` against the sister implementation, and reports the diff in non-PASS rates. Suggested name: `autoresearch-cross-domain-replicate`.

Net: the 29-skill library is *comprehensive over the within-project autoresearch loop* but *thin on the meta-question of audit-the-audit*. The Phase-9b future-work item partially recognises this.

### 4.6 Risk inversion on the 51 % non-PASS rate (Question E from the brief)

The brief poses the question: is the high non-PASS rate due to the framework's own implementer agents being more bug-prone than typical ML research code, or due to the protocol being more diagnostic? R3's reading is **both, with the second effect dominant in the MAJOR/BROKEN tier and the first effect dominant in the MINOR tier**. The calibration's 33 % MINOR-tier rate on torchvision (close to the project's 29 %) tells us the audit is *aggressive* on documented intentional deviations — this is a *protocol diagnostic* effect, not a *codebase-quality* effect. The MAJOR/BROKEN tier (22 % project vs 0 % calibration) — if the gap holds at higher n — *is* a codebase-quality signal: defects an unaudited pipeline would have shipped. The case studies (H09 realised-ratio drift, H55 PlatonicAttention zero-bias, H67 broken GoldenRoPE import, H74 13-alpha collapse) are concrete and verifiable independent of agent ecology; these are the strongest evidence the audit caught real defects, not phantom defects.

R3 is mildly persuaded that LLM-agent-written ML research code is *more bug-prone* than human-authored library code in this regime, primarily because the implementer agents optimise for "tests pass" rather than "mechanism realised", and the shape-only-test pathology is a corner the agents systematically cut. This is itself a *research finding* worth highlighting: it argues for the mechanism-pinning-test discipline (S1) being not just useful but *necessary* when LLM agents are in the implementation loop.

### 4.7 Pytorch/vision audit: did the auditor miss known issues?

The brief asks: are MAJOR/BROKEN = 0 on pytorch/vision an audit blind spot or genuine cleanliness? R3 spot-checked the calibration verdicts against the pytorch/vision GitHub issue tracker (via the references in `audits/AUDIT_CALIBRATION_RAW_NOTES.md`). The TP12 finding (`CosineAnnealingLR` is named for SGDR but implements the no-restart degenerate case) is a real, well-known documentation issue (pytorch/pytorch issue #20028 and forum posts) — the auditor correctly flagged it. The TP15 BatchNorm momentum semantics issue is also real and well-documented (and well-known to be a recurring source of Keras-vs-PyTorch porting bugs). The TP2 V1.5 stride-placement deviation from He 2016 is the canonical NGC reformulation and is correctly graded MINOR. R3 finds no obvious miss in the 15-row audit; if anything, the auditor was *appropriately aggressive* on documented intentional deviations and *appropriately lenient* on undocumented but truly minor cosmetic issues. The 0 MAJOR/BROKEN is, in R3's spot-check, genuine pytorch/vision cleanliness, not auditor blindness.

This *modestly strengthens* the diagnostic-credibility argument (S3 / §4.1).

---

## 5. Specific clarifying questions for the authors

1. **(W2 resolution).** Have you considered re-auditing a 5-10 row subsample of the project's own MAJOR/BROKEN findings with a *different* model family (GPT-5, Gemini 3 Pro) and reporting the agreement rate? This would be the cheapest single move to neutralise the auditor-self-grading concern; absent it, the protocol-as-contribution claim is conditional on a caveat the paper acknowledges but cannot dispatch. The cost is bounded (≤ 20 audit verdicts × 5 minutes per verdict).
2. **(W3 / §4.5).** The cross-domain replication is the empirical test of content-agnosticism. Have you run *any* of the 29 skills against the `autoresearchtabular` Higgs implementation, even informally? A 1-day exercise running `autoresearch-critic-team` against tabular GBM modules would be a much stronger contribution than the prose-level parity audit currently in `audits/SKILLS_PARITY_AUDIT.md`.
3. **(§4.2).** Why are the Fixer mechanism-verifying tests value-specific rather than property-based? The H09 test asserts the realised ratio is within 1 % of `[1, φ, φ²]` for a *specific* budget; a property-based variant would assert the realised ratio approaches the target for any budget in `[64k, 1M]`. The latter would catch the broader bug class. Is this a deliberate choice (e.g., property-based tests too slow) or an accident of skill drafting?
4. **(§4.1).** Phase-9b proposes a 50-100-hypothesis calibration on timm + HuggingFace transformers + Lightning Bolts. What is the timeline? If it lands within the camera-ready window, the audit-diagnostic-power claim graduates from "marginal" to "statistically credible" and R3's rating moves from 5 to 7.

---

## 6. Required revisions for ICML main-track acceptance

R3's current rating (5, WEAK REJECT for main track) would graduate to 7+ (ACCEPT) on the following revisions in priority order:

1. **Cross-domain replication of at least ONE skill on at least ONE sister repo.** The cheapest credible move: run `autoresearch-critic-team` against `autoresearchtabular/core/*.py` and report the non-PASS rate. Even a single-skill, single-domain replication operationalises the content-agnostic claim from prose-level analogy to empirical demonstration. This addresses W3.
2. **Cross-family audit on at least 10 verdicts.** Re-audit 10 of the 18 MAJOR/BROKEN findings with GPT-5 or Gemini 3 Pro; report verdict-agreement. This addresses W2 head-on.
3. **Phase-9b calibration extension** to ≥ 50 third-party hypotheses (timm + HuggingFace + Lightning Bolts). The 22-pp MAJOR/BROKEN-tier excess goes from marginal (Fisher p≈0.07) to credible (likely p<0.01 at n=50). Addresses §4.1.
4. **Promote at least one Fixer mechanism-pinning test to a property-based test** and document the diff. Demonstrates the protocol can sharpen itself. Addresses §4.2.
5. **Add a "Limitations of the protocol" section** that explicitly tabulates the seven open concerns (auditor-self-grading at model-family level, selection bias on design space, content-agnosticism unmeasured, mechanism-pinning vs property-based, n=15 calibration, screening-vs-evaluation HARKing, no cross-model-family audit). The paper currently scatters these across §1.3, §5.8, §7.3, §7.3.1, Appendix A; consolidating them in one explicit subsection helps the reviewer trust the authors.

Revisions for D&B / methods-workshop track acceptance: items 1 and 5 only. The paper is already above bar for that track.

---

## 7. Minor / typographic / suggestion comments

- §5.5.4's distinction between Δmedian (+1.20 / +1.80 / +2.08 pp) and Δmean (+0.79 / +1.22 / +1.31 pp) under hill-climb is honest and helpful but visually buried; consider lifting it into a per-claim table in the abstract.
- The "screening vs evaluation" Rule 28 wording is good but inconsistently applied: §6.1 calls H41 "WEAKLY NEGATIVE at 12-ep CIFAR-10 screening" which mixes the tier (SCREENING) with a verdict (NEGATIVE) — pick one. Recommended: "SCREENING tier, mild-negative".
- The `paper/STATISTICAL_TESTS.md` reference is load-bearing and should be a top-level repository link in the paper's repository pointers (Appendix C), not just an inline citation.
- The "Audit calibration on third-party code" section is currently §5.8 (numbered after §5.7's "verdict-on-the-wrong-testbed audit"). Given how load-bearing it is for the protocol-as-contribution claim, R3 recommends moving it to immediately follow §5.1 (the implementation-critic distribution).
- The H71 IcosaRoPE3D framing in §1.1 ("we explicitly NOT" list) is good; the parallel "NOT" framing should be replicated for the transformer-track hypotheses (e.g., H32 Fibottention, H34 golden-angle RoPE) where the impl-critic verdict was rendered without a transformer scaffold to test on. The §7.4-2 future-work item already promises a ViT-Tiny scaffold; the §1.1 NOT list should make clear the impl-critic verdicts on transformer-track hypotheses are also conditional.

---

## 8. Final decision and rating rationale

- **Rating: 5 — Marginally below acceptance threshold (main track).**
- **Rating: 7 — Accept (D&B / methods-workshop track).**
- **Confidence: 4/5.**

The paper makes a *real* methodological contribution (the dual-track audit + Fixer + mechanism-pinning-test protocol is novel as integration and well-engineered as software), is *honest* about its caveats (the auditor-self-grading circularity, the post-hoc HARKing, the single-domain calibration), and is *competent* in its third-party calibration extension. It fails the ICML main-track novelty bar because (a) the individual protocol pieces are ports of established practices, (b) the cross-domain transferability claim is by construction not by demonstration, and (c) the auditor-self-grading circularity is acknowledged but not dispatched. Three of the five revisions in §6 would lift this above the main-track bar; revisions 1 and 5 alone would push it firmly above the D&B / methods-workshop bar.

R3 specifically wants to flag that the *Fixer-with-mechanism-verifying-test contract* (S1) is the load-bearing intellectual contribution and is *underemphasised* in the abstract. The abstract leads with the dual-track audit; R3 would lead with the Fixer-mechanism-test discipline, which is the move that most directly addresses the LLM-agent-writes-shape-only-tests pathology and is the most portable single idea in the paper.

---

*Reviewer R3 pass complete. Methodology track. Independent of R1 (novelty), R2 (empirical), R4 (presentation/scope) passes — running in parallel per the area-chair instruction.*
