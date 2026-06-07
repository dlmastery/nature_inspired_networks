# NeurIPS 2027 Reviewer Critique — R3
Date: 2026-06-06 · Reviewer lens: novelty + significance + broader impact

## Verdict

**Reject (3/10).** I would not advocate for this paper at NeurIPS 2027 under any framing. The "self-auditing LLM-agent autoresearch protocol" framing is a re-skin of well-known prior art (LLM-as-judge, mutation testing, two-stage clinical trials, pre-registration) bolted onto a CIFAR-100 ablation whose headline (+1pp over a deliberately under-tuned ResNet-20 at a recipe ~6.5pp below SOTA) cannot survive a serious significance bar. The "audit-calibration" headline (Fisher p=1.94×10⁻⁵) compares 18/83 self-graded findings on the authors' own buggy code against 0/62 self-graded findings on `pytorch/vision` — that is a calibration of the auditor against itself, not external validity. The paper itself contains the AC synthesis from the May-30 ICML pass concluding "Weak Reject for main; Accept-with-revisions for D&B"; the June-01 → June-04 Phase-9h/9i pivot has made the empirical headline weaker, not stronger, by openly admitting the original certification was confounded. The author is right that this reads as "AI slop" — not because the engineering is bad (it is impressively organized), but because the writing exhibits every textbook tell of LLM-generated post-hoc rationalization: section numbers that grow without bound (Phase-9a/9f/9g/9h/9i/9j), abstracts that average 440 words, banners stacked atop banners ("preserved with Phase-9h honest demotion above"), and ~30 references where the authors cite themselves citing themselves.

## Top 3 fatal flaws

1. **The "audit-calibration" headline (Fisher p=1.94×10⁻⁵) is a comparison artifact, not an external-validity result.** Same Opus-4.7 agent class graded both 83 hypotheses it (or its siblings) wrote AND 62 third-party modules it did not write. A 0% MAJOR/BROKEN rate on `pytorch/vision` is what you would *expect* from any auditor with no incentive to find bugs in PyTorch core — finding a MAJOR bug in `BatchNorm2d` is career-defining. The 22pp gap measures how aggressively the auditor flags code from a recently-written agent codebase relative to mature production code, not how diagnostically credible the protocol is. Authors concede the model-family caveat (§1.2) but treat it as a side note; it is the entire experiment.

2. **The "priors carry ~+1pp signal across recipes" claim is methodological whiplash, not a finding.** The chronology is: (a) declare Phase-8 winners CERTIFIED at α=0.05 Holm-Bonferroni (May 29 PM); (b) discover a tuned baseline beats them by +2.27 to +2.81pp at lr=0.01 (June 1 LE — "honestly demoted from winners"); (c) run a different recipe (modern 11-trick, 200ep) where they lift +1pp (June 4 — "honestly restored"); (d) ship the abstract claiming the priors carry "robust directional signal across recipes." Each pivot is described as the protocol "catching itself." A protocol that swings between "certified" → "refuted" → "restored" inside 96 hours, with every pivot framed as victory, has not converged on a finding; it has produced a Rorschach test. The June-04 Phase-9i n=3 result has paired Wilcoxon at its theoretical floor (p_one=0.125) — not significant at any defensible α — and uses a non-pre-registered recipe chosen *after* the June-01 result was unfavorable.

3. **There is no novelty here that would survive a literature search.** The "modern 11-trick recipe" (§convergence/PLAN.md) is a verbatim re-implementation of Bello 2021 (arXiv:2103.07579) and Wightman 2021 (arXiv:2110.00476 / `timm`). "LLM-as-judge" is the standard term for the dual-track audit (Zheng 2023, arXiv:2306.05685). "Mechanism-verifying tests" are mutation testing (DeMillo 1978). "Two-stage screening + confirmation" is pre-registration (Munafò 2017). "Icosahedral equivariance" without a faithful group representation is not equivariance (Cohen-Welling 2016 arXiv:1602.07576; e2cnn arXiv:1911.08251). SIREN (Sitzmann 2020 arXiv:2006.09661) is cited only in passing — but `slot_act_sine` IS SIREN, as the AC synthesis explicitly notes. The author has done one (1) thing the literature has not: applied this stack to CIFAR-100 with a φ-themed naming convention. That is not a contribution.

## Honest summary

The repository is unusually well-organized — 80+ src modules, 780+ tests, an auto-checkpoint loop, a SHA-256-fingerprinted composite metric, a per-experiment-page dashboard discipline. The engineering quality is genuinely above the median NeurIPS submission. But NeurIPS rewards *findings*, not *organization*, and the empirical envelope here is far too narrow to support the paper's claims. CIFAR-10/100 at ResNet-20 scale on a single laptop GPU was acceptable for a NeurIPS workshop in 2018; in 2027 it is below the bar for the main track regardless of statistical rigor. The "we built a protocol" framing would land at a Reproducibility workshop or D&B (as the May-30 ICML AC concluded), but the June-01 → June-04 pivots have damaged even that case: a protocol whose primary value-add is "catching its own headline drift" cannot demonstrate that value without a stable headline. The 84-hypothesis design space is the kind of breadth-over-depth move that signals "AI-generated taxonomy" to any experienced reviewer. The mystical motivation (acknowledged honestly in MANIFESTO) is not the problem; the problem is that *every* "nature-inspired" prior — φ-budget, golden momentum, Platonic attention — is either a rediscovery (RegNet's `w_m ∈ [2.5, 2.9]` band for φ-budget, per the paper's own §A audit of itself), a re-skin (SIREN as "slot_act_sine"), or BROKEN (H55's mathematically-zero head bias). The negative results are honest; the positive results, after Phase-9h, do not hold up.

## 25+ concrete improvements

### 1. Reframe as a single-claim methods-workshop paper, not a NeurIPS main-track submission
**What's wrong:** PAPER.md is 263 lines of abstract + 9 sections + 7 phases (9a/9f/9g/9h/9i/9j) of statistical narrative attempting to support two simultaneous headlines (protocol + priors). The ICML AC already concluded "Weak Reject for main; Accept-with-revisions for D&B" on a strictly stronger draft.
**What to do:** Pick the H09 realised-ratio-drift catch as the single existence proof. 4 pages: motivation, mechanism, the catch, what an unaudited pipeline would have shipped. Submit to a Reproducibility workshop, ICML D&B, or NeurIPS Datasets & Benchmarks track. Drop the audit-calibration §4 entirely.
**Severity:** BLOCKER **Effort:** 1 week **Priority:** P0

### 2. Drop the audit-calibration headline entirely
**What's wrong:** Fisher p=1.94×10⁻⁵ comparing 18/83 self-audits of self-written code vs 0/62 self-audits of `pytorch/vision` is not measuring what the paper claims. The null hypothesis "production-quality third-party code is bug-free" is *also* false (cf. CVE history of `torch`), so 0/62 measures auditor reluctance to flag canonical code, not actual defect rate. **BLOCKER (prior art not cited):** Liang 2023 *Holistic Evaluation of Language Models* (arXiv:2211.09110) and Zheng 2023 *Judging LLM-as-a-Judge* (arXiv:2306.05685) both document this exact bias (LLM judges defer to authority signals). The paper does not engage either.
**What to do:** Remove §4 from the abstract and §4.1-4.4 from the body. If you must keep it, frame as "auditor calibration bound" not "MAJOR/BROKEN diagnostic credibility."
**Severity:** BLOCKER **Effort:** 4 hours **Priority:** P0

### 3. Acknowledge the Phase-9h → Phase-9i pivot as evidence against the protocol, not for it
**What's wrong:** PAPER.md §5.3 / FINDINGS.md banners frame a 96-hour swing from "certified" → "refuted" → "restored" as the protocol working as designed. Reviewers will read it as a researcher-degrees-of-freedom catastrophe. The Phase-9i recipe (modern 11-trick, 200ep) was chosen *after* Phase-9h came in unfavorable; that is the textbook definition of garden-of-forking-paths (Gelman & Loken 2013).
**What to do:** Either (a) commit to Phase-9h as the binding result and demote the priors to "screening only, do not survive a tuned baseline at α=0.05," OR (b) pre-register a Phase-9j n≥7 confirmation BEFORE seeing the result, with the pre-registration commit hash in the paper. Currently you have neither — you have a moving target.
**Severity:** BLOCKER **Effort:** 40 GPU-h (Phase-9j) + 2 weeks **Priority:** P0

### 4. Either run a real equivariant CNN or drop "icosahedral equivariance" terminology
**What's wrong:** `src/nature_inspired_networks/icosa.py` constructs 60 rotation matrices and applies them as a max-pool over orbits. This is a one-shot orientation augmentation, not a steerable group convolution. **BLOCKER (prior art not faithfully cited):** Cohen-Welling 2016 *Group Equivariant CNNs* (arXiv:1602.07576); Cohen 2019 *Gauge Equivariant CNNs* (arXiv:1902.04615 — cited but not implemented); Weiler 2019 *General E(2)-Equivariant Steerable CNNs* / `e2cnn` (arXiv:1911.08251 — mentioned only as "no Py 3.13 wheel"); Geiger 2022 `e3nn` (arXiv:2207.09453). H55 PlatonicAttention's bias was provably zero (your own audit found this). H71 IcosaRoPE3D is untested. The paper has zero working equivariance.
**What to do:** Drop "equivariant" from any hypothesis name. Replace with "orientation-pooled." Or: implement one (1) hypothesis using `e3nn` (works on Python 3.10/3.11; use a separate venv) and report a real equivariance gap on rotated CIFAR-10 / Spherical MNIST.
**Severity:** BLOCKER **Effort:** 2 weeks (e3nn integration) or 2 hours (rename) **Priority:** P0

### 5. Concede `slot_act_sine` is SIREN and drop it from "nature-inspired priors"
**What's wrong:** Sitzmann 2020 *SIREN* (arXiv:2006.09661) is cited in §8 References as the literature anchor for `slot_act_sine`. The paper's own §5.5 admits the lift is "generic activation engineering." The AC synthesis says it should be demoted from the abstract. It is still there. **BLOCKER:** advertising a SIREN replication as a "nature-inspired prior winner" without disclosing the SIREN anchor in the abstract is misleading.
**What to do:** Remove `slot_act_sine` from the abstract triple. Mention only as "the protocol surfaced a SIREN replication mis-attributed to φ-content; the catch is the protocol-positive finding."
**Severity:** BLOCKER **Effort:** 1 hour **Priority:** P0

### 6. Run datasets beyond CIFAR or do not claim "general protocol"
**What's wrong:** §1 contribution 1 frames seven content-agnostic skills as "portable infrastructure." §7 Limitations admits no cross-domain replication exists. NeurIPS 2027 will not accept CIFAR-10/100 as the sole image-classification testbed. **Imagenette (arXiv: n/a, `fastai/imagenette`, 10 classes, 160px, ~1h/run on 4090 laptop)**, **Tiny-ImageNet (Le & Yang 2015, 200 classes, 64px, ~4h/run)**, **CIFAR-100-LT (Cao 2019 arXiv:1906.07413, long-tail)** are all feasible on a single 4090 laptop within months.
**What to do:** Run the three Phase-8 winners on Imagenette + Tiny-ImageNet at n=3 each (~30 GPU-h total). Report whether the +1pp lift holds. Drop "content-agnostic" claim if it doesn't.
**Severity:** MAJOR **Effort:** 30 GPU-h + 1 week **Priority:** P1

### 7. Pre-register Phase-9j BEFORE running it; cite the commit hash
**What's wrong:** §5.3 announces Phase-9j n≥7 as "filed as future work" without a pre-registration commit hash. After the Phase-9h → 9i pivot, any unblinded result will read as "another moving target."
**What to do:** Commit a `pre-registration/phase9j_2026-06-06.md` file listing the exact recipe, seeds, decision rule, and analysis plan. Cite that commit hash in PAPER.md before launching seeds 3-6. Include the AsPredicted-style document (Nosek 2018 arXiv:1709.05064-equivalent template).
**Severity:** MAJOR **Effort:** 4 hours + Phase-9j compute **Priority:** P0

### 8. Get external (non-Claude) audit on the 18 MAJOR/BROKEN findings
**What's wrong:** §1.2 acknowledges all auditors are Claude Opus 4.7. §4.3 partial-closure with "three distinct audit methods" still runs on the same model. Without GPT-5 / Gemini-3-Pro replication, the audit-calibration headline is unfalsifiable. The AC synthesis explicitly flagged this as the #2 actionable item.
**What to do:** Pay for ~$50 in OpenAI/Google API credits. Run a 10-finding audit using GPT-5 and Gemini 3 Pro with the same skill-template. Report verdict-agreement rate. If < 60% agreement, the entire audit-calibration headline is invalid.
**Severity:** MAJOR **Effort:** 1 week, $50 **Priority:** P1

### 9. Compute Holm-Bonferroni across the full screening family, not just k=3
**What's wrong:** §5 of STATISTICAL_TESTS applies Holm to k=3 confirmatory tests. But the k=3 was selected post-screening from a family of ~84 single-prior + 6+ combo + multiple slot configurations. Honest POSI bound (Berk 2013 arXiv:1306.1107) is k≥40-50; α'_Holm ≈ 0.001. Wilcoxon-at-floor 0.0078 (n=7) does not clear this. The AC synthesis acknowledged R1's POSI objection as "technically correct."
**What to do:** Either (a) report the POSI-corrected p-values explicitly and concede no claim clears at corrected α; (b) extend to n=10-12 seeds at which p_one_min = (1/2)^10 = 0.00098 clears POSI; OR (c) use FDR (Benjamini-Hochberg 1995) and report q-values instead of family-wise.
**Severity:** MAJOR **Effort:** Option (b) ~80 GPU-h; option (a/c) 4 hours **Priority:** P1

### 10. Cite Bello 2021 and Wightman 2021 in the paper body, not just convergence/PLAN.md
**What's wrong:** The "modern 11-trick recipe" is Bello 2021 *Revisiting ResNets* (arXiv:2103.07579) and Wightman 2021 *ResNet Strikes Back* (arXiv:2110.00476). Neither appears in PAPER.md References §8. The Phase-9i "convergence regime" headline reuses their methodology without crediting them. **BLOCKER (prior art not cited).**
**What to do:** Add both to References §8. Reframe Phase-9i as "evaluated on the Bello-Wightman modern recipe at 200ep" not "iso-modern-recipe."
**Severity:** BLOCKER **Effort:** 1 hour **Priority:** P0

### 11. Compare against a tuned RegNetX-200MF baseline
**What's wrong:** §6.4 compares against ResNet-20 only. RegNetX-200MF (Radosavovic 2020 arXiv:2003.13678) is the literature's direct competitor at ~0.27M params and is cited as the H09 anchor. The paper's own §5.5 says H09 is a "rediscovery of RegNet's Pareto region" but does not compare numbers.
**What to do:** Train RegNetX-200MF on CIFAR-100 with the Bello-Wightman recipe at 200ep × 3 seeds (~12 GPU-h). Report whether H09 phi_budget beats it. If not, drop the H09 contribution and report only the protocol catch.
**Severity:** MAJOR **Effort:** 12 GPU-h + 2 days **Priority:** P1

### 12. Reduce the abstract from ~440 words to ~200
**What's wrong:** The abstract is 440 words and contains four parenthetical caveats, three explicit p-values, and Phase-9h/9i/9j cross-references. A NeurIPS abstract is 150-250 words. The current text reads as a defensive blog post.
**What to do:** Cut to 200 words. Lead with H09 phi_budget realised-ratio drift as the single existence proof. Move statistical machinery to §3.
**Severity:** MAJOR **Effort:** 4 hours **Priority:** P1

### 13. Run the H22 toroidal_phi_closure hypothesis on its pre-registered dataset
**What's wrong:** §6 hypothesis docs (e.g., H22) explicitly pre-register "tiled-texture or wrap-aware synthetic dataset" as the falsifier. The paper tested on upright CIFAR-10 and assigned NUMEROLOGY/UNFALSIFIABLE verdict. AC synthesis flagged this as "testing on the wrong dataset and then concluding the hypothesis fails." This contaminates the §4.4 distribution table.
**What to do:** Either run H22 on tiled-CIFAR-10 / wrap-augmented CIFAR (~3 GPU-h) or relabel the verdict UNTESTED_ON_RIGHT_DATASET (Rule 36).
**Severity:** MAJOR **Effort:** 3 GPU-h + 1 day **Priority:** P1

### 14. Drop H71 IcosaRoPE3D from contributions until it is empirically tested
**What's wrong:** §1.1 lists H71 as the sole NOVEL+TESTABLE survivor. §7 admits it is untested. AC synthesis: "an untested novel idea is a research proposal, not a result."
**What to do:** Run H71 on rotated CIFAR-10 vs 1D-RoPE control. If not feasible, move H71 to §8 Future Work and remove from contributions list.
**Severity:** MAJOR **Effort:** 8 GPU-h (ViT-Tiny scaffold) **Priority:** P1

### 15. Add a real broader-impact / dual-use discussion
**What's wrong:** ETHICS_STATEMENT §3 says "Dual-use risk: negligible" three times. This is boilerplate. The actual dual-use risk of an autoresearch protocol is *automated p-hacking and HARKing at scale* — exactly what the Phase-9h → 9i pivot demonstrates. A NeurIPS Ethics-AC will catch this.
**What to do:** Write 1 page on: (a) automated researcher-degrees-of-freedom amplification; (b) LLM-judge cascades producing self-confirming "rigor theatre"; (c) what mitigations the protocol offers and which it does not. Cite Forde 2018 (arXiv:1806.07261) on selection bias in ML research, and Hullman 2022 *Worst-Case Analysis is Self-Defeating* (arXiv:2210.13691).
**Severity:** MAJOR **Effort:** 1 day **Priority:** P1

### 16. Concede `pair_gm_pdw` is a 3-axis regularization stack with no φ content in 2/3 axes
**What's wrong:** Phase-9g Control 1 showed `pair_nonphi_3axis` lifts +0.61pp paired (2/3 positive), about 61% of `pair_gm_pdw`'s lift. The paper's §6.5 admits this but does not propagate to abstract/conclusion. The φ-content story is partially refuted on the authors' own evidence.
**What to do:** Rewrite the `pair_gm_pdw` framing as "any 3-axis orthogonal stack" and drop "φ-budget + golden momentum + φ-decay-WD" mechanism story. Or run a control where ONLY the H09 phi_budget axis varies between phi and non-phi at iso-everything-else, n=7.
**Severity:** MAJOR **Effort:** 1 hour write + optionally 10 GPU-h confirm **Priority:** P1

### 17. Add a hyperparameter table to the paper body
**What's wrong:** REVIEWER_PASS_PAPER.md C1 flagged "no hyperparameter table" as a BLOCKER. PAPER.md still does not have one. Reviewers cannot reproduce without one. The README §2 quick-start is not the same as a hyperparameter table.
**What to do:** Add a §3.5 table listing per-hypothesis: model, optimizer, LR, scheduler, wd, batch, label smoothing, augmentation, AMP, epochs, seeds, hardware.
**Severity:** MAJOR **Effort:** 4 hours **Priority:** P1

### 18. Discuss the LLM-judge bias literature in §8 Related Work
**What's wrong:** §8 Related Work cites Holm 1979 and Wilcoxon 1945 but not the LLM-as-judge literature that bounds the protocol's validity. Zheng 2023 *Judging LLM-as-a-Judge* (arXiv:2306.05685), Wang 2023 *Large Language Models are not Fair Evaluators* (arXiv:2305.17926), Liu 2023 *G-Eval* (arXiv:2303.16634), Chen 2024 *Humans or LLMs as the Judge?* (arXiv:2402.10669) — all directly relevant.
**What to do:** Add 1 paragraph in §8 with these four citations. Discuss self-preference bias (Panickssery 2024 arXiv:2404.13076) and how it applies to same-model-family auditor self-grading.
**Severity:** MAJOR **Effort:** 1 day **Priority:** P1

### 19. Engage the mutation-testing literature for "mechanism-verifying tests"
**What's wrong:** §3.3 Track-C Fixer campaign requires mechanism-verifying tests. This is mutation testing (DeMillo 1978; Jia 2011 *An Analysis and Survey of Mutation Testing*). LLM-based mutation testing is an active research area (Tian 2023 arXiv:2302.10039; Tip 2024). The paper does not cite any of it.
**What to do:** Add a paragraph in §8 connecting "mechanism-verifying test contract" to mutation testing literature and discussing where the protocol differs (LLM-generated mutants vs LLM-generated tests).
**Severity:** MAJOR **Effort:** 1 day **Priority:** P1

### 20. Drop the "first formally-certified empirical claims" language entirely
**What's wrong:** §4.3.1 of README and §0 of STATISTICAL_TESTS use "first formally-certified empirical claims at NeurIPS-standard α." After Phase-9h demoted them and Phase-9i restored them at iso-modern-recipe with p_one=0.125 (above α=0.05), this language is unjustifiable. Effect sizes are also < 1/4 of the 6.5pp gap-to-SOTA (§6.4).
**What to do:** Replace with "screened candidates with consistent +1pp directional lift across two recipes; formal certification pending n≥7 at the modern recipe."
**Severity:** MAJOR **Effort:** 1 hour **Priority:** P1

### 21. Move the methodological-courage signaling to the cover letter, not the paper
**What's wrong:** The paper contains 5+ instances of "honest framing" / "the protocol caught itself" / "the load-bearing methodological catch." NeurIPS reviewers find this distracting; rigor should be demonstrated, not announced.
**What to do:** Cut every instance of "honest" as a self-descriptor. Cut "the protocol catches itself" framing from abstract and §1. Let the audit catches speak for themselves in §5.
**Severity:** MAJOR **Effort:** 2 hours **Priority:** P1

### 22. Cite self-distillation / self-tutoring literature for "self-auditing"
**What's wrong:** "Self-auditing" is a specific term in the LLM literature (Zelikman 2024 *Quiet-STaR*, arXiv:2403.09629; Saunders 2022 *Self-Critiquing Models*, arXiv:2206.05802; Madaan 2023 *Self-Refine*, arXiv:2303.17651; Yuan 2024 *Self-Rewarding LMs*, arXiv:2401.10020). The paper invents the term de novo. **MAJOR (prior art not cited).**
**What to do:** Add a paragraph in §8 situating "self-auditing protocol" within the Self-Refine / Self-Critique / Constitutional-AI lineage. Acknowledge the protocol differs by operating on *research artifacts* not *model outputs*.
**Severity:** MAJOR **Effort:** 1 day **Priority:** P1

### 23. Replace "audit-calibration" framing with "concurrent-validity check"
**What's wrong:** §4 frames the 0/62 vs 18/83 comparison as "audit-calibration." This is psychometric terminology that does not apply — there is no gold-standard external rating to calibrate against. The correct term is *concurrent validity* (do two measurements of the same construct agree?), and concurrent validity requires that the two samples be drawn from the same population. They are not.
**What to do:** Either replace the framing or run the audit on a matched-population sample (e.g., 62 hypotheses from autoresearchimage, autoresearchtabular sister repos — code from the same agent class but a different project).
**Severity:** MAJOR **Effort:** 2 weeks (sample matching) or 2 hours (rename) **Priority:** P1

### 24. Add a "what could falsify this paper" section
**What's wrong:** No section explicitly states what a future result would have to look like to refute the paper's claims. After Phase-9h showed +2.3pp baseline-beats-prior and Phase-9i was run to recover, the implicit answer is "nothing — we will always find a recipe where it works." This is the failure-to-falsify pattern.
**What to do:** Write a §7.5 listing 3-5 results that would refute the paper: e.g., "If Phase-9j n≥7 at the modern 200ep recipe returns Δmean < +0.3pp with paired-bootstrap CI including 0, the priors are refuted." Commit this before running Phase-9j.
**Severity:** MAJOR **Effort:** 4 hours **Priority:** P1

### 25. Reduce the hypothesis count from 84 to 10
**What's wrong:** The 84-hypothesis design space is the single largest "AI slop" tell. Real papers test 3-10 hypotheses with depth. 84 hypotheses on CIFAR-10 with 80% single-seed coverage reads to any reviewer as LLM-generated breadth-without-depth. The §1 contribution "84-hypothesis case study" is itself a liability.
**What to do:** Pick 10 hypotheses that map cleanly to 10 distinct mechanism families. Drop the rest into supplementary as "design-space enumeration, not tested." Defend each of the 10 with a 1-page mechanism + ablation. Cut MANIFESTO references to "84 hypotheses" everywhere.
**Severity:** MAJOR **Effort:** 1 week **Priority:** P1

### 26. Add a per-claim cost / value table
**What's wrong:** Reviewers cannot quickly assess whether each claim is worth the compute spent. ETHICS §2 says 50 GPU-h total, but the per-claim breakdown is buried.
**What to do:** Add a §3.6 table: claim, GPU-h, n_seeds, statistical power achieved, Δmean, whether it cleared the pre-registered α.
**Severity:** MINOR **Effort:** 2 hours **Priority:** P2

### 27. Cite Munafò 2017 and Nosek 2018 for pre-registration
**What's wrong:** Rule 28 / Rule 36 invoke pre-registration as a concept without citing the meta-science literature. Munafò 2017 *A manifesto for reproducible science* (Nature Human Behaviour; arXiv:1709.05064-equivalent), Nosek 2018 *The preregistration revolution* (PNAS).
**What to do:** Add 2 citations in §2 where Rule 28 is introduced.
**Severity:** MINOR **Effort:** 1 hour **Priority:** P2

### 28. Acknowledge the CIFAR-100-at-65% top-1 baseline is non-competitive
**What's wrong:** Phase-9i baseline 0.6360 top-1 on CIFAR-100. Literature SOTA on CIFAR-100 with similar param-counts is >75% (e.g., DenseNet-BC-100 at 77.7%, Pyramidal ResNet at 80%). The "modern recipe" baseline is 10+ points below SOTA. Lifts at this level may not transfer to converged regimes.
**What to do:** Add a sentence in §6.4 / §6.5 acknowledging the 200ep ResNet-20 ceiling and noting that +1pp at 64% top-1 is not equivalent to +1pp at 80% top-1 (effect-size compression near accuracy ceiling).
**Severity:** MINOR **Effort:** 1 hour **Priority:** P2

### 29. Document the dataloader seed protocol
**What's wrong:** REVIEWER_PASS C7 flagged "no random-seed protocol description." `cudnn.benchmark=True` is non-deterministic. Per-seed reproducibility is not bit-exact.
**What to do:** Add a §3.5 paragraph listing: torch seed, numpy seed, python random seed, dataloader generator seed, cudnn settings, AMP determinism.
**Severity:** MINOR **Effort:** 2 hours **Priority:** P2

### 30. Move the dashboard from headline to supplementary
**What's wrong:** README §11 and PAPER.md repeatedly cite the live dashboard URL. A reviewer who clicks the link sees 199 generated HTML pages and 340+ historical broken links (Rule 27 origin). This is not a NeurIPS deliverable; it is a project artifact.
**What to do:** Move dashboard references to supplementary. Inline 3-5 hand-curated figures in the paper body.
**Severity:** MINOR **Effort:** 4 hours **Priority:** P2

## Closing note

There is a paper here, but it is not this one. The H09 realised-ratio-drift catch — an LLM-implementer wrote code claiming `1:φ:φ²` widths and actually produced `1:1.41:2.45`, detected by an LLM-auditor adding a mechanism-pinning test — is a real, interesting, single-existence-proof contribution to the emerging LLM-agent-research literature. A 4-page workshop paper built tightly around that one catch, citing Saunders 2022 / Madaan 2023 / Zheng 2023 / mutation-testing prior art, and explicitly *not claiming the priors work*, would be publishable at a Reproducibility or Trustworthy-ML workshop. Everything else in the current submission — the 84 hypotheses, the audit-calibration headline, the Phase-9 alphabet soup, the "the priors carry +1pp signal" claim — is either prior-art rediscovery, statistical theatre, or post-hoc rationalization. The author's frustration that this reads as AI slop is justified; the root cause is the LLM-generated impulse to maximize surface area (more hypotheses, more sections, more phases, more banners) when the actual finding is one paragraph long. Cut 90% and rewrite from the H09 catch outward.
