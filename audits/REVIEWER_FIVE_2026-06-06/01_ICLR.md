# ICLR 2027 Reviewer Critique — R1
Date: 2026-06-06 · Reviewer lens: theory + representation-learning + novelty

## Verdict
**REJECT.** The paper has no theoretical contribution, no novel representation-learning insight, no architecture comparison that a 2026 ICLR reviewer would accept (ResNet-20 only, ≤200 ep, CIFAR-only), and its headline empirical claim collapses into "three regularizers tuned on one config beat an untuned baseline by ~1 pp at n=3 on a 6-year-old toy benchmark." The pivot to "the protocol is the contribution" rests on a self-graded n=62 audit-calibration study performed by the same LLM family that wrote, audited, fixed, and re-audited the code — not a methodological contribution ICLR's representation-learning track is structured to evaluate.

## Top 3 fatal flaws
1. **No representation-learning content of any kind.** No theorem, no inductive-bias analysis, no kernel/equivariance proof, no information-geometric statement, no scaling-law claim. The paper is purely empirical at CIFAR scale with a base architecture (ResNet-20, 0.27M params, 2016) that ICLR 2027 reviewers will treat as obsolete by default. ICLR is not a CIFAR-tweak venue.
2. **The "audit-calibration" headline is a Claude-grades-Claude artifact.** The 62 third-party "calibration" hypotheses, the 84 in-house hypotheses, the implementer agents, the impl-critic, sci-critic, Fixer, and the cross-family "re-audit" are all Claude Opus 4.7. The "Fisher p=1.94×10⁻⁵" is the test statistic of a single LLM family grading code it considers in-distribution (`pytorch/vision`, `timm`) vs code it doesn't. This is **not external validation**; it is a confounded measurement of model-prior agreement, presented as if it were an objective defect-detection benchmark. PAPER.md §4.3 admits this in one sentence and then proceeds to headline the result.
3. **The empirical headline is incoherent across §5.2 / §5.3 / §6.5.** Section 5.2 shows the tuned baseline beats the three "winners" by +2.27 to +2.81 pp (Mann–Whitney p ≤ 0.011). Section 5.3 then reframes this as "apples-to-oranges LR-tuning confound" by running an *unmatched* iso-modern-recipe sweep at n=3 where the Wilcoxon p-floor is 0.125 — i.e., a test that *cannot* clear α=0.05 by construction. The paper's own STATISTICAL_TESTS.md §3 derives n≥7 as the minimum for Holm-Bonferroni; §15 then ships an n=3 result and calls it "qualitatively binding." This is moving the goalposts mid-paper.

## Honest summary of the contribution
A laptop-scale (RTX 4090, 16 GB) ablation study of ~70 CIFAR-scale interventions on a ResNet-20 backbone, framed as nature-inspired (φ, hexagonal, Platonic, fractal, toroidal, Chladni, golden-angle) but reducing empirically to three things: (a) a moderate-width-allocation tweak that the paper itself concedes is a rediscovery of RegNet's Pareto region (Radosavovic et al. 2020); (b) replacing ReLU with sin(ωx) (SIREN, Sitzmann 2020); (c) stacking three orthogonal regularizers (Adam β₁ schedule + layer-graded weight decay + width allocator), of which Control 1 (PAPER.md §6.5) shows ~61% of the lift is reproduced by a *non-φ* 3-axis stack. The "protocol" contribution is a set of LLM-agent disciplines (dual-track audit, Fixer-with-mechanism-test, screening-vs-evaluation tier, auto-checkpoint loop) implemented entirely with Claude Opus 4.7, with no cross-family validation.

## 25+ concrete improvements

### Improvement 01 — Strip every theoretical claim that isn't there and reframe as empirical-only
- **What's wrong:** PAPER.md §1.1 lists "self-auditing protocol" as Contribution 1 and the 84-hypothesis substrate as Contribution 4, but there is no theorem, no convergence-rate result, no equivariance proof, no spectral analysis of NaturePriorBlock, no Lipschitz bound, no information-bottleneck argument. ICLR's representation-learning expectations are minimum theorem-or-novel-mechanism; this paper has neither.
- **What to do:** Either (a) add a real theoretical contribution (e.g., prove the φ-width-allocator minimises a specific FLOPs-per-effective-rank objective; or derive an equivariance statement for HexConv at the actually-realised stride pattern; or analyse SIREN's spectral bias on 32×32 inputs); or (b) submit to ICLR's reproducibility/benchmark venues, not the main track. Trying to thread the empirical-only needle at main-track ICLR is the wrong venue choice.
- **Severity:** BLOCKER · **Effort:** L · **Priority:** P0

### Improvement 02 — Acknowledge ResNet-20 is not a 2026 ICLR base architecture
- **What's wrong:** Every empirical claim runs on ResNet-20 (0.27M params, He 2016). PAPER.md §6.4 admits the baseline is 6.5 pp below converged ResNet-20 SOTA, and the headline lifts (~1 pp) are <1/6 of that gap. ICLR 2027 reviewers will not accept "+1 pp at CIFAR-100 30-ep" on a 2016 architecture as evidence for any prior.
- **What to do:** Add at least one modern baseline at CIFAR scale: ConvNeXt-Tiny (28M, CIFAR-transferred), MobileNetV3-Small, RegNetX-200MF (the direct literature comparator for the φ-width claim, which is *still* missing despite being requested in `audits/REVIEWER_PASS_PAPER.md` item 13 from 2026-05-29), or a CIFAR-tuned ViT-Tiny. Re-test the three priors on ≥2 of these. If the lift evaporates, the paper's empirical contribution is "the prior helps on ResNet-20 specifically," which is a much weaker claim than the abstract sells.
- **Severity:** BLOCKER · **Effort:** M (each model ~10 GPU-h on a 4090; 3 priors × 2 backbones × 7 seeds × 200 ep ≈ 60–80 GPU-h) · **Priority:** P0

### Improvement 03 — The audit-calibration substrate is not independent of the auditor's training data
- **What's wrong:** `pytorch/vision`, `timm`, HuggingFace Transformers are *in-distribution* for Claude Opus 4.7 (publicly available on GitHub for years, almost certainly in the training corpus). The "0/62 MAJOR/BROKEN" finding could be entirely an artefact of in-distribution memorisation: Claude has effectively *seen* the correct implementations and is grading against memory rather than against the design doc. The 18/83 finding on the project's own substrate could be entirely an out-of-distribution penalty.
- **What to do:** Run the audit on (a) deliberately-buggy versions of the same third-party modules (mutation testing: introduce a known bug into `BatchNorm2d.forward` — does the audit flag it?); (b) third-party code published *after* the model's training cutoff (Sep 2024 papers' repos that Claude has not memorised); (c) a non-Claude auditor (GPT-5, Gemini 3 Pro, even Llama 4 405B) on the same 18 MAJOR/BROKEN findings. Without (a) or (c), the 22-pp excess is not interpretable.
- **Severity:** BLOCKER · **Effort:** S–M (mutation testing: ~1 person-day, no GPU. Cross-family audit: ~$30 API + 1 day) · **Priority:** P0

### Improvement 04 — Phase-9i n=3 cannot "correct" Phase-9h; it can only fail to falsify
- **What's wrong:** PAPER.md §5.3 claims "Phase-9i resolves the confound" by running n=3 at modern-recipe 200ep and getting +1.00 to +1.24 pp lifts at p=0.125 (the n=3 floor). This is *not* a correction; it is a re-test with insufficient power. The Phase-9h tuned-baseline finding (n=3, σ=0.31 pp, Mann–Whitney p≤0.011, no rank overlap) is *statistically stronger* than the Phase-9i corrective binding (n=3, paired Wilcoxon at floor 0.125). The paper is preferring a weaker-evidence test that supports its narrative.
- **What to do:** Run Phase-9j n≥7 at iso-modern-recipe 200ep on a 4090 (~39 GPU-h per arm × 4 arms × 4 extra seeds = ~150 GPU-h; budget over 5–6 overnight sessions). Until that lands, drop §5.3's "corrects" framing and report Phase-9h and Phase-9i as two underpowered diagnostics that *disagree* — not as a self-correction cycle.
- **Severity:** BLOCKER · **Effort:** M (150 GPU-h) · **Priority:** P0

### Improvement 05 — SIREN's "slot_act_sine" does not belong in a nature-inspired paper
- **What's wrong:** PAPER.md §6.5 Control 2 honestly reports `slot_act_tanh` *beats* `slot_act_sine` by +0.48 pp (3/3 positive paired). This means SIREN is **not** the source of the lift — any smooth bounded activation works equally or better. The paper then continues to list `slot_act_sine` as one of three Phase-8 winners in the abstract, §1.1, §5, §6, and the conclusion, despite the cite-by-cite admission that the result has nothing to do with golden-ratio / nature-inspired priors and is in fact *outperformed* by tanh.
- **What to do:** Demote `slot_act_sine` from the abstract, §1.1, §5.3 binding, conclusion, and conclusion headline. Replace it with `slot_act_tanh` if you want to keep an "activation-substitution" finding, but label it explicitly as "the protocol surfaced a generic activation-engineering signal mistaken for nature-inspired" — this is a *protocol-positive* finding, not a prior-positive one. R1 of the ICML 2026 review pass (`audits/ICML_REVIEWS_2026-05-30/`) already flagged this; the paper still hasn't fixed it.
- **Severity:** BLOCKER · **Effort:** S (writing) · **Priority:** P0

### Improvement 06 — H09 phi_budget is a re-discovery of RegNet's Pareto region; abstract sells it anyway
- **What's wrong:** PAPER.md §8 "Related Work" admits "RegNet (Radosavovic et al. CVPR 2020) establishes the Pareto region w_m ∈ [2.5, 2.9] that contains but does not require φ." The sci-critic addendum in `hypotheses/g1_*/H09_*.md` §"Numerology check" lists 5 confounds and concludes φ-specificity is "almost certainly" not the mechanism. Yet H09 is listed as a Phase-8 winner in the abstract, §1.1, §5, §6 — and the RegNetX-200MF control row remains UNLAUNCHED at submission, despite being item 13 of the 2026-05-29 internal review's required revisions.
- **What to do:** Either run RegNetX-200MF at iso-params (~270k) with 3-seed 30-ep CIFAR-100 (the paper already says it is laptop-feasible; ~10 GPU-h) and report the comparison, or remove H09 from the headlines and present it as "we replicated the RegNet Pareto-region finding using a φ-parameterisation." Item 13 from the prior reviewer pass is still open.
- **Severity:** BLOCKER · **Effort:** S (~10 GPU-h) · **Priority:** P0

### Improvement 07 — Cross-family re-audit is intra-family methodologically diverse, not cross-family
- **What's wrong:** `audits/CROSS_FAMILY_HONEST_REAUDIT.md` §1.3 explicitly admits "This is NOT a non-Claude external auditor. It is the same model family re-running the audit through a different methodological lens." The paper's §4.3 then describes this as a "cross-family methodologically-diverse re-audit (partial closure on auditor model-family caveat)," which is misleading to a fast reviewer. The 8/10 strict concordant rate is Claude-vs-Claude.
- **What to do:** Rename "cross-family" to "intra-family methodologically diverse" throughout PAPER.md, FINDINGS.md, REVIEWER_CHECKLIST.md (currently 6+ instances). Section 4.3's "partial closure" framing should become "no cross-family closure delivered; intra-family method-diverse re-audit reports 8/10 concordant which is a within-family robustness check." Then run *actual* cross-family on at least 5 findings via GPT-5/Gemini 3 Pro API ($20–30 total).
- **Severity:** MAJOR · **Effort:** S · **Priority:** P0

### Improvement 08 — Citation accuracy: arXiv:1902.04615 is NOT Cohen et al. "Spherical CNNs"
- **What's wrong:** PAPER.md line 233 cites `Cohen TS, Geiger M, Köhler J, Welling M. 2019 ICLR. *Spherical CNNs*. arXiv:1902.04615`. This is the wrong attribution: arXiv:1902.04615 is "Gauge Equivariant Convolutional Networks and the Icosahedral CNN" (Cohen, Weiler, Kicanaoglu, Welling — ICML 2019, *not* ICLR). Cohen et al.'s "Spherical CNNs" is arXiv:1801.10130 (ICLR 2018). This citation conflates two distinct papers from the same author group. Used in H24/H53/H55 design docs.
- **What to do:** Correct the citation to either arXiv:1801.10130 (Spherical CNNs, ICLR 2018) or arXiv:1902.04615 (Gauge/Icosahedral CNN, ICML 2019), depending on which one the actual H55 PlatonicAttention mechanism implements. Audit *every* arXiv ID in PAPER.md References and the 84 hypothesis docs for the same class of error — when an LLM agent writes citations from memory at scale, this error type clusters.
- **Severity:** MAJOR · **Effort:** S (1 hour for PAPER.md; 1 day for all 84 docs) · **Priority:** P0

### Improvement 09 — Citation accuracy: Hoogeboom HexaConv is ICLR 2018, not ICML 2018
- **What's wrong:** PAPER.md line 232: `Hoogeboom E, Peters JWT, Cohen TS, Welling M. 2018 ICML. *HexaConv*. arXiv:1803.02108.` HexaConv was published at ICLR 2018, not ICML 2018. Small but indicative; CLAUDE.md Rule 4 explicitly requires correct venue.
- **What to do:** Fix the venue. Then run a Rule-4 conformance pass on the entire References section and the 84 hypothesis docs' `## 4. Citations` blocks.
- **Severity:** MINOR · **Effort:** S · **Priority:** P1

### Improvement 10 — Pittorino 2022 venue is ICML, but the paper is "Deep networks on toroids: removing symmetries reveals the structure of flat regions"
- **What's wrong:** PAPER.md line 241 cites the title as *Deep Networks on Toroids* — the full title is *"Deep networks on toroids: removing symmetries reveals the structure of flat regions in the loss landscape"*. This is the H22 toroidal-prior anchor; the abridged title obscures that the paper is about *loss-landscape symmetry* (a flat-minima study), not a constructive toroidal architecture prior. The H22 sci-critic should engage that the cited paper does *not* in fact propose imposing toroidal closure on activations.
- **What to do:** Use the full title; re-read H22's sci-critic addendum and verify the mechanism claim is consistent with what Pittorino et al. actually argue.
- **Severity:** MINOR · **Effort:** S · **Priority:** P1

### Improvement 11 — Islam et al. 2025 "Platonic Transformers" arXiv:2510.03511 — verify this paper exists
- **What's wrong:** PAPER.md line 243 cites a future-dated arXiv ID 2510.03511 (October 2025). The prior internal review (`audits/REVIEWER_PASS_PAPER.md` item 11) flagged this as "verify Islam 2025 arXiv:2510.03511 publication status." LLM agents are known to hallucinate plausible-sounding arXiv IDs at high rates. This paper is the load-bearing citation for the H55 PlatonicAttention sci-critic verdict and the Fixer-G6 patch.
- **What to do:** Have a human Google-Scholar-verify the arXiv ID *before* resubmission. If the paper doesn't exist or doesn't say what's claimed, remove the citation and the H55 mechanism justification it props up. Do the same for *every* citation post-2024 — these are highest-risk for hallucination because they're least likely to be in Claude's training data.
- **Severity:** BLOCKER (if hallucinated) / MAJOR (if real but mis-cited) · **Effort:** S · **Priority:** P0

### Improvement 12 — Composite metric algebraically rewards param-count reduction, confounding every "lift"
- **What's wrong:** `top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms)` (PAPER.md §3). Going from baseline 0.272M to phi_budget 0.284M params changes the composite by only ~0.001 — but going from 0.27M to 0.16M would be +0.011, *worth +1.1 pp top1 at zero predictive gain*. None of the headline figures explicitly separate the params-discount from the actual top1 lift. `paper/LIMITATIONS.md` §6 admits a FLOPs-normalised metric "would weight differently" but the paper still uses the composite to rank.
- **What to do:** Report raw top1 deltas (already done in §6.5) AND make sure every dashboard/table/figure that ranks by "composite" also shows raw top1, params, latency separately. The composite is suitable for screening only; do not use it as the headline ranking statistic.
- **Severity:** MAJOR · **Effort:** S · **Priority:** P1

### Improvement 13 — n=3 paired Wilcoxon has floor p=0.125; STATISTICAL_TESTS.md §15 ships it as "qualitatively binding"
- **What's wrong:** Phase-9i §5.3 reports paired Wilcoxon p_one = 0.125 (= (1/2)³, the *theoretical floor*) and labels it "qualitatively binding." A test at its own floor is not "binding"; it is uninformative under any standard interpretation. The §3 sample-size derivation explicitly says n ≥ 7 is the minimum for Holm-Bonferroni clearance. The paper is reporting a result at half the pre-registered sample-size requirement.
- **What to do:** Drop "qualitatively binding"; replace with "underpowered (n=3 < n_required=7)." Run Phase-9j n=7 before claiming any iso-modern-recipe result.
- **Severity:** BLOCKER · **Effort:** S (writing) / M (n=7 GPU-h) · **Priority:** P0

### Improvement 14 — POSI correction is missing despite a 35-row screening sweep
- **What's wrong:** PAPER.md §6.2 screened 35 tags on CIFAR-10 at seed 0, identified `phi_budget` as the "only variant beating baseline." STATISTICAL_TESTS.md §5 reports the 99th-percentile single-seed Δ across 58 non-baseline tags is +0.96 pp, inside 2σ_pooled. Then the Phase-8 family of size 3 is presented with Holm-Bonferroni k=3 — but the *honest* family for post-selection inference is the full screening set, k=35–58, so α' ≈ 0.001. None of the three winners' p=0.0078 clears that bar.
- **What to do:** Either (a) pre-register the k=3 family with a verifiable git commit *before* the screening sweep ran, or (b) acknowledge in the abstract that Holm-Bonferroni at α'=0.0167 is conditional on a *post-screening* family selection, which inflates the effective α to ~5%×35/3 ≈ 50%. The current framing is HARKing. The prior reviewer (`audits/REVIEWER_PASS_PAPER.md` §B/§F item 73) called this out; no correction landed.
- **Severity:** BLOCKER · **Effort:** S · **Priority:** P0

### Improvement 15 — Self-graded "Internal QA pass" banners survive on every external-facing artifact
- **What's wrong:** `paper/REVIEWER_CHECKLIST.md` line 1 reads "ACCEPTANCE STATUS (2026-06-04 update): INTERNAL QA PASS." PAPER.md line 3 mentions "Reviewer-acceptance ACCEPT verdict at commit `0343f35`" via reference. The prior reviewer pass (item 10) demanded this be removed; it's still there. CLAUDE.md Rule 37 forbids self-grading banners on external artifacts; the project violates its own rule on its most-read documents.
- **What to do:** Strip every "ACCEPT" / "PASS" / "Internal QA" banner from PAPER.md, README.md, paper/*.md. Reviewers will read these as marketing, not engineering. The truthful banner is "Submission-candidate; not externally reviewed."
- **Severity:** MAJOR · **Effort:** S · **Priority:** P0

### Improvement 16 — The 84-hypothesis design space is itself a selection-biased LLM-generated list
- **What's wrong:** The 84 hypotheses were curated by an LLM agent reading source PDFs on nature-inspired networks. The "1 NOVEL+TESTABLE / 81 critiqued" rate is presented as evidence about the field; it is in fact evidence about the LLM's selection bias. Any cross-cohort comparison ("0/62 on third-party vs 18/83 on project") is comparing two LLM-curated samples drawn under different prompts.
- **What to do:** Add a sentence in §1 acknowledging the 84-hypothesis cohort is LLM-selected from a specific reading list, not a random or expert-curated sample of the field. The prior reviewer (REVIEWER_PASS_PAPER.md §F item 68) called this a BLOCKER; still unaddressed.
- **Severity:** MAJOR · **Effort:** S · **Priority:** P1

### Improvement 17 — H71 IcosaRoPE3D is the sole NOVEL+TESTABLE survivor and is untested
- **What's wrong:** PAPER.md §1.1 caveats explicitly say "the sole NOVEL+TESTABLE survivor is untested." The paper still lists "84-hypothesis design space" as a contribution. A design space whose single novel idea has zero empirical evidence is a literature review, not a contribution.
- **What to do:** Run H71 on rotated CIFAR-10 with a ViT-Tiny scaffold (the controls/PLAN.md Control 4 already attempted this — Δ=+0.18 pp INCONCLUSIVE per §6.5). Extend Control 4 to n=7. If H71 doesn't show real signal, drop it from contributions entirely.
- **Severity:** MAJOR · **Effort:** M (ViT-Tiny CIFAR-10 ~10–15 GPU-h × 7 seeds × 2 arms = ~150 GPU-h) · **Priority:** P1

### Improvement 18 — `pair_gm_pdw`'s components include the previously-falsified `golden_momentum`
- **What's wrong:** `golden_momentum` was originally falsified as a 1-step-saturating schedule (FINDINGS evidence + Fixer commit `1c98226`). `pair_gm_pdw` uses the post-fix `golden_momentum` (β decays to a floor) plus `phi_budget` plus `phi_decay_wd`. The paper does not isolate which of the three axes is doing work. The prior reviewer (`audits/REVIEWER_PASS_PAPER.md` §B item 27) flagged this; still unaddressed.
- **What to do:** Add a "leave-one-out" ablation: `pair_gm_pdw` minus `golden_momentum`, minus `phi_decay_wd`, minus `phi_budget`, each at n=3 30-ep CIFAR-100. Without this, the +1.74 pp claim is a 3-feature bag-of-tricks, not a specific prior.
- **Severity:** MAJOR · **Effort:** M (~15 GPU-h) · **Priority:** P1

### Improvement 19 — Control 1 already shows ~61% of "pair_gm_pdw" lift is from non-φ stacking; demote in abstract
- **What's wrong:** STATISTICAL_TESTS.md §13.1 and PAPER.md §6.5 both report `pair_nonphi_3axis` (non-φ 3-axis stack) achieves +1.06 pp vs baseline — 61% of `pair_gm_pdw`'s +1.74 pp. The residual +0.61 pp φ-attributable lift is at the n=3 Wilcoxon floor (one-sided p=0.25). One seed went *negative* (nonphi BEATS phi by 0.24 pp on seed 2). This is *partial refutation of φ-specificity* and should be in the abstract; instead it is buried in §6.5 and §13.1.
- **What to do:** Add a sentence in the abstract: "Control 1 shows ~61% of the `pair_gm_pdw` lift is reproduced by a non-φ 3-axis stack; the φ-specific residual (+0.61 pp) is at the n=3 Wilcoxon floor." Demote `pair_gm_pdw` from "Phase-8 winner" to "3-axis regulariser stack of which ~61% is generic and ~39% is φ-specific at the n=3 floor."
- **Severity:** MAJOR · **Effort:** S · **Priority:** P0

### Improvement 20 — No comparison to Bello 2021 "Revisiting ResNets" or Wightman 2021 "ResNet Strikes Back"
- **What's wrong:** The convergence/PLAN.md correctly identifies Bello 2021's 11-trick recipe as the right baseline. But the paper never reports a Bello-recipe ResNet-20 *without* the priors at converged 200 ep at n=7 — only n=3 (§5.3). The reference number for "what a modern recipe + the same backbone gets you" is not in the paper. The +1.0 pp lift over a *one-seed* (n=3 mean) modern-recipe baseline could simply be the prior compensating for a 1σ-low baseline seed sample.
- **What to do:** Run the baseline modern-recipe 200ep at n=7, fold into Phase-9j. Cite Bello 2021 and Wightman 2021 (timm) numbers explicitly in §6.4.
- **Severity:** MAJOR · **Effort:** M (~25 GPU-h for n=7 modern baseline) · **Priority:** P1

### Improvement 21 — EMA decay 0.9999 + 200ep + bs=256 + CIFAR-100 50k = ~10k steps; warmup tells half the story
- **What's wrong:** `src/nature_inspired_networks/ema.py` defaults `warmup=True` with the timm/TF formula `min(decay, (1+s)/(10+s))`. At 200 ep × (50000/256) ≈ 39k steps, the warmup is irrelevant after step ~10000 (40% of training). But the *eval* statistic is the EMA shadow's top1; if any of the 4 arms (baseline / 3 priors) has more numerical instability early in training (e.g., the SIREN sin activation has different gradient magnitudes), the warmup-decay interaction is different per arm. The paper does not engage this.
- **What to do:** Report both raw-weight and EMA-weight top1 separately in §5.3. If the SIREN arm is helped disproportionately by EMA, the +1.01 pp lift is partially an EMA-smoothing artefact, not a representation-learning artefact.
- **Severity:** MAJOR · **Effort:** S (data already on disk; re-evaluate) · **Priority:** P1

### Improvement 22 — Mixup α=0.2 + CutMix α=1.0 50/50 alternation is a hyperparameter the paper never tunes
- **What's wrong:** `convergence/PLAN.md` lists the 11 tricks but the paper doesn't ablate any of them. Mixup α=0.2 + CutMix α=1.0 at 50/50 alternation is *one* recipe out of dozens. The +1 pp lift attributed to each prior at modern-recipe could be *much smaller* under a Mixup-α=0.5 or RandAugment-M=9 recipe.
- **What to do:** Either (a) commit to *the* Bello 2021 recipe exactly (cite their exact hyperparameters) and acknowledge no ablation; or (b) do at least a 3-cell Mixup-α ablation at n=3 to show the lift is robust to recipe details. Without (a) or (b), the §5.3 result is recipe-specific.
- **Severity:** MAJOR · **Effort:** M (~9 GPU-h for 3-cell ablation) · **Priority:** P1

### Improvement 23 — "Self-falsification existence proof" is one (1) H09 case; one bug ≠ generalisable detection rate
- **What's wrong:** PAPER.md §5.1 / Conclusion lift the H09 12.6% realised-stage-ratio drift as the "load-bearing existence proof" of protocol self-falsification. This is n=1. A single case where the protocol caught a bug is anecdote; it tells you nothing about the protocol's recall or precision. The Conclusion sells it as Contribution-grade.
- **What to do:** Report at least the *count* of bugs caught vs the *count* of bugs missed (every BROKEN/MAJOR finding is also a bug the *initial* implementer-team missed). If 18/83 audit-tier bugs slipped past the first-pass implementer, the protocol's "self" detection rate is bounded; this is information the abstract should carry.
- **Severity:** MAJOR · **Effort:** S · **Priority:** P1

### Improvement 24 — README.md is 200+ lines and lists 8+ section headers; ICLR reviewers read 30 lines
- **What's wrong:** README.md (and PAPER.md, at 263 lines) signals the author's attention is on the dashboard layer, not on conveying the central claim in 1 paragraph. The "Elevator pitch" is 13 lines and mentions 7 priors before stating any result.
- **What to do:** Reduce README's elevator pitch to ≤4 sentences and lead with the *honest* contribution ("we propose an LLM-agent autoresearch protocol with mechanism-pinning Fixer tests; calibrated on a 62-hypothesis third-party substrate where the same protocol returns 0/62 MAJOR; demonstrated on a 84-hypothesis case study where 3 candidates lift baseline by ~1 pp at CIFAR-100 200ep at n=3, formal n≥7 cert pending"). One paragraph; no badges in the first screen.
- **Severity:** MINOR · **Effort:** S · **Priority:** P1

### Improvement 25 — The "FINAL_STATE" / "Phase-9i convergence-regime corrective binding" naming is unreadable to reviewers
- **What's wrong:** The paper / FINDINGS / commit log uses internal-jargon names (Phase-9h, Phase-9i, "convergence-regime corrective binding," "iso-modern-recipe + iso-convergence n=3 binding") that have no meaning outside the project. An ICLR reviewer reading PAPER.md §5.3 must learn the phase-letter taxonomy before parsing the result.
- **What to do:** Replace every "Phase-9X" label in PAPER.md, FINDINGS.md, README.md with descriptive names: "tuned-baseline diagnostic at n=3," "iso-modern-recipe 200ep at n=3," "n=7 default-config certification." Keep the phase labels in commit logs and internal docs only.
- **Severity:** MINOR · **Effort:** S · **Priority:** P1

### Improvement 26 — No CIFAR-100 baseline at >90% of converged accuracy
- **What's wrong:** The paper's 30-ep CIFAR-100 baseline is 0.5612 (PAPER.md §1.1). Converged ResNet-20 on CIFAR-100 is ~0.69–0.72 (He 2016 recipe at 200+ ep). The modern-recipe 200-ep baseline in §5.3 is 0.6360 — still ~6–8 pp below typical converged CIFAR-100 ResNet-20. The "lift" is being measured against a sub-converged baseline, exactly the regime where regularisation tricks (and any feature-bottleneck reduction) look disproportionately strong.
- **What to do:** Either (a) train baseline to convergence (250–400 ep) and report the gap to literature CIFAR-100 ResNet-20 numbers; or (b) state explicitly in the abstract that lifts are at "30 ep default" or "200 ep 11-trick modern" — *neither* of which is the He 2016 SOTA recipe. The current SOTA_COMPARISON.md only addresses CIFAR-10, not CIFAR-100.
- **Severity:** MAJOR · **Effort:** M (~30 GPU-h to add a fully-converged baseline at n=3) · **Priority:** P1

### Improvement 27 — Mann–Whitney at n_a=3, n_b=7 is at its own minimum 2-sided p=0.0167; that's not a strong-test
- **What's wrong:** PAPER.md §5.2 reports Mann–Whitney p_one ∈ [0.0083, 0.0111] and labels this a strong rejection. But at n_a=3, n_b=7 the minimum achievable two-sided p is 2/C(10,3) = 0.0167 (achieved whenever all 3 are on one side of all 7). The reported one-sided p=0.0083 is at the floor. This is the same kind of "test at its own floor" problem as Phase-9i §5.3 — but in §5.2 the paper interprets the floor as decisive evidence *against* the priors, and in §5.3 the floor as merely "qualitatively binding" *in favour* of the priors. Inconsistent treatment of equivalent statistical conditions.
- **What to do:** Report all "p at the floor" results with the floor explicitly named. Treat both directions symmetrically.
- **Severity:** MAJOR · **Effort:** S · **Priority:** P1

### Improvement 28 — Composite metric fingerprint is intellectual-property theatre, not a contribution
- **What's wrong:** PAPER.md §3 makes much of the SHA-256-fingerprinted composite metric ("editing it raises CompositeFingerprintError at runner import"). Reviewers will read this as a build-system trick, not a research contribution. The fact that you've made the metric uneditable doesn't make the metric *good* — and the metric itself (top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms)) is arbitrary, with no theoretical justification.
- **What to do:** Move the fingerprint mention to an appendix on engineering discipline. Don't lead §3 with it.
- **Severity:** MINOR · **Effort:** S · **Priority:** P2

### Improvement 29 — Repository "11 normative rules" → "28 normative rules" → "38 normative rules" is mission creep
- **What's wrong:** CLAUDE.md grew from 11 → 19 → 28 → 38 rules across the campaign. Each new rule was added in response to a failure the protocol missed. This is a maturity signal *and* a Goodhart signal: the protocol now has 38 rules and 17 skills, much of which is post-hoc rationalisation of fixes. A reviewer will ask "if you needed 38 rules to handle 84 hypotheses, what is the rule-count for a 1000-hypothesis design space?" The answer is "scaling sublinearly," but the paper doesn't engage this.
- **What to do:** Add a §"Rule-set evolution" or appendix showing the cumulative rule count over time and the trigger that caused each addition. This makes the rule-set growth a feature (visible learning) rather than a bug (post-hoc rationalisation).
- **Severity:** MINOR · **Effort:** S · **Priority:** P2

### Improvement 30 — Submit to D&B / Reproducibility venue, not main ICLR
- **What's wrong:** The ICML 2027 area chair synthesis (`audits/ICML_REVIEWS_2026-05-30/AC_synthesis.md`) explicitly recommended Datasets & Benchmarks / Reproducibility & Meta-Research workshop. The protocol-as-contribution framing fits there; the main-track-grade representation-learning bar does not.
- **What to do:** Resubmit to ICLR's Tiny Papers / Blog Posts track, or to a workshop on LLM-agent autonomous research, or to the Datasets & Benchmarks track at NeurIPS / ICLR. Stop pushing main-track ICLR until the empirical envelope (modern architectures, ImageNet, transformer baselines, n=7 at iso-recipe) is closed.
- **Severity:** MAJOR · **Effort:** S (different submission) · **Priority:** P0

## Closing note

What would change my mind from REJECT → ACCEPT (or even WEAK_ACCEPT)?

(1) A genuine cross-family audit: at least 5 of the 18 MAJOR/BROKEN findings independently confirmed by GPT-5 or Gemini 3 Pro, with non-trivial concordance rate reported and any disagreements engaged. (2) Phase-9j n=7 at iso-modern-recipe 200ep landing with paired Wilcoxon clearing Holm-Bonferroni α'=0.0167 at the modern-recipe cell — and the same gate on a *modern* baseline architecture (ConvNeXt-Tiny or RegNetX-200MF), not just ResNet-20. (3) A real theoretical contribution: either an equivariance proof for one of the priors, or a scaling-law-style information-bottleneck analysis showing *why* the φ allocation should dominate uniform-allocation under some explicit constraint. (4) Demotion of `slot_act_sine` and honest reframing of `pair_gm_pdw` as "3-axis regularizer stack of which 61% is generic" — both already established empirically in the paper's own controls, but not reflected in the abstract or conclusion. Without items (1)–(2) the empirical envelope is too narrow; without (3) the venue choice is wrong; without (4) the paper is internally contradictory.

The protocol-as-contribution framing is genuinely interesting and could publish at a Reproducibility / Meta-Research workshop today. As a main-track ICLR representation-learning submission, it is not close.
