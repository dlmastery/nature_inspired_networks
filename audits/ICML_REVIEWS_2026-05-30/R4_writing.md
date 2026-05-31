# ICML 2027 Reviewer R4 — Writing / Clarity / Framing

> Reviewer profile: ICML 2027 program committee; primary area Nature ML / Communications-of-the-ACM-style prose review.
> Focus: abstract claim discipline, internal contradictions, hedging discipline, negative-result framing, accessibility, length, citation rigor, framing-creep.
> Reviewer date: 2026-05-30.
> Reviewer commit context: PAPER.md at line count 848; FINDINGS.md PROMOTION block at line 3; README.md headline §4 carrying the n=7 + Phase-9a table.

---

## Summary of review

The revision is a credible response to the area-chair WEAK_REJECT pass. The "protocol-as-contribution" framing finally lands: the abstract leads with the protocol, the empirical winners are explicitly called "illustrative case studies / calibration data," and the n=7 Holm-Bonferroni certification provides genuine statistical backbone for the three Phase-8 numbers. Negative results (H50 −11.54 pp, H41 requalified, H80 Reuleaux −8.83 pp) are acknowledged. The HARKing concession in §7.3.1 is a methodologically courageous move.

However, the paper still suffers from non-trivial writing-discipline defects that an ICML camera-ready cannot ignore: the abstract is overlong and hedge-heavy to the point of obscuring the value proposition, the framing of the n=7 result oscillates between "case study" and "headline" within the same section, the area-chair-revision-table at the top of the paper consumes 27 lines before the abstract even begins, three [VERIFY] citation tags remain unresolved in the References section, and the §5.5 / §5.5.1–§5.5.4 chain reads as a defensive transcript of a reviewer dialogue rather than a research narrative. Length is a problem: 848 lines for an ICML 8-page main submission is implausible without aggressive cuts; the paper appears to be aiming at arXiv preprint + supplementary, which should be stated explicitly.

---

## Overall rating

**Score: 5 / 10 (Borderline — leaning weak accept as arXiv preprint / methods note; weak reject as ICML main-track submission without further compression and length adjustment).**

**Recommendation: Major Revision.** The methodological contribution is genuinely interesting and the n=7 certification is a real upgrade. With the writing-discipline cuts in Section E below, the paper is ICML-publishable as a methods note. As written, prose quality and length problems would prompt at least one reviewer to reject on framing alone.

---

## A. Abstract discipline — PARTIAL PASS

**The abstract carries every material limitation:** all three caveats (baseline 6.5 pp below SOTA, single-seed for most rows, audit/implementer/Fixer same-model-family circularity) are now in the abstract, explicitly labelled `(a)`, `(b)`, `(c)`. This is a substantive improvement over the prior version.

**The abstract carries the certified claim with proper hedging:** "as of 2026-05-29, these three were extended to n=7 seeds and now FORMALLY CERTIFY at α=0.05 under Holm-Bonferroni" with paired Wilcoxon p=0.0078, bootstrap CIs, and the explicit "at the screening compute budget" framing. The Holm-Bonferroni-cleared n=7 status is the strongest hedged claim the paper can make, and it is made cleanly.

**The abstract picks ONE story — protocol-as-contribution — but the prose discipline wavers:**
- L1 of the abstract: "We present a methodological contribution: a dual-track skeptical audit + Fixer + per-experiment-page protocol" — good, on-brand.
- L1–L3 of abstract: 130 words about the protocol — appropriate dwell time.
- L4 onward: "As empirical calibration, we apply it to an 84-hypothesis nature-inspired neural-network design space..." — good, but then the abstract spends ~210 of its ~440 words on the n=7 certified numbers and their CIs.

The numerical share of the abstract exceeds the protocol share. A reader skimming the abstract comes away with "Holm-Bonferroni-certified Phase-8 winners on CIFAR-100" as the load-bearing finding, not "we built a protocol." The abstract should compress the n=7 result to a single sentence and reclaim 100+ words for the protocol's value proposition (what does the protocol do that an unaudited pipeline doesn't — concretely, in one paragraph).

**MAJOR (writing).** Abstract is 440 words; ICML conventional abstract length is 150–200 words. The Internal QA Pass banner above the abstract (lines 3–27, 27 lines, a 16-row revision-status table) is also pre-abstract real estate that no external reviewer should have to read before the abstract itself. Move the revision table to Appendix D (already exists for similar mapping content); cut the abstract to ≤ 200 words.

**Score on A:** 3 / 5.

---

## B. Internal-contradiction scan — PASS (with three residual line-level inconsistencies)

**Trace of the most-certifiable claim from abstract → §5.5 → §8 conclusion:**

- Abstract: "`pair_gm_pdw` Δmean=+1.74 pp (95% CI [+1.42, +2.09])"
- §5.5 line 285: "`pair_gm_pdw` (H09+H48+H44 orthogonal stack) ... 0.5786 ... Δmean +1.74 pp ... bootstrap 95% CI on Δmean [+1.42, +2.09] pp **excludes 0**; **CLEARS Holm-Bonferroni α'=0.0167**"
- §7.2.1 line 456: "The 7-seed CIFAR-100 30-ep mean (0.5786, Δmean +1.74 pp) is **CERTIFIED at α=0.05 under Holm-Bonferroni**"
- §8 line 506: "the three Phase-8 candidates (`pair_gm_pdw`, `slot_act_sine`, `sg_only_phi_budget` post-fix) are reported as **illustrative protocol output**, conditional on Phase-9 hill-climb and the §5.5.1 non-φ control row."

Verdict: **the claim survives the trace** — the abstract, §5.5, §7.2.1, and §8 agree the number is +1.74 pp at n=7 Holm-cleared. Good.

**However**, the conclusion still contains an unresolved tension: "We make **no** standalone empirical headline claim for nature-inspired priors at this scale" (§8 L506) vs the abstract's heavy weighting on the certified Phase-8 numbers. A reader could legitimately complain that the paper is having it both ways: the abstract spends most of its real estate on the certified empirical claim, then the conclusion disavows it as "illustrative." Either the certification is a headline (and the conclusion should own it) or it is illustrative (and the abstract should not be 50% Phase-8 numbers). The current split-message is a soft contradiction.

**16-row revision-status table line-level scan (lines 9–25):**

- Row 11 (Citation Rule-4 compliance) — marked **PARTIAL**, but no clear path to DONE in the body. The three `[VERIFY]` tags in the References section (lines 529, 531) are still open at submission time. This is consistent with the table's PARTIAL status; not a contradiction, but a defect.
- Row 4 (`pair_gm_pdw` non-φ 3-axis control) — marked **PARTIAL → toward DONE (2026-05-30)** with §5.5.4 cited. §5.5.4 itself is candid that it is a tuned-vs-tuned robustness check, NOT a non-φ control. The table's "toward DONE" framing is slightly over-optimistic relative to §5.5.1's "OPEN" label.
- Row 13 (RegNet baseline) — marked **PARTIAL → toward DONE (2026-05-30)**, but §7.4-4 still lists "Tuned-RegNet / tuned-ResNet baseline comparison" as future work. Consistent under "PARTIAL → toward DONE" but not yet DONE.

These are minor narrative-tension items; none reaches BLOCKER. Internal consistency is in better shape than the area-chair-pass version, and visibly better than the pre-revision version.

**Old "FAILS Holm-Bonferroni" framing — fully retired.** Grep on the repo finds zero remaining occurrences of "FAILS Holm-Bonferroni" or "FAIL.*Holm." The retirement is clean.

**README.md headline vs PAPER.md §5.5:** README §4 ("Headline claims (CERTIFIED, n=7)") and PAPER.md §5.5 / §7.2.1 agree on the same three winners with the same Δmean / CI / p=0.0078 values. The README is more confident in tone ("first formally-certified empirical claims") than PAPER.md (which still uses "illustrative case study" language). This is a tone gap, not a numerical contradiction — but it is the kind of gap a careful reviewer will note. **MINOR.**

**Score on B:** 4 / 5.

---

## C. Negative-result framing — PASS

**H50 sg_full_fib (−11.54 pp):** Appears in §7.2 ("the catastrophic full-hybrid `sg_full_fib` (−11.54 pp) is empirically refuted at the screening level") and is given Rule-23-codifying prominence in §7.2.1 with a 9-row combo-ladder table showing exactly which axis killed the stack (`plr` LR-schedule, −5.66 pp). The negative is load-bearing — it generates Rule 23, an explicit normative invariant in CLAUDE.md. This is exemplary negative-result discipline.

**H41 golden_adam (REQUALIFIED):** Treated with appropriate prominence in §6.1 ("WEAKLY NEGATIVE at 12-ep CIFAR-10 screening" after the pre-fix −33 pp number was eps-confounded). The requalification is also surfaced in FINDINGS.md and in README.md §5 negative-results table. The framing is honest: the original headline was wrong (eps-confound), the requalification is mild, the asymptotic Reddi-regime test is deferred to Phase-9. This is a case study of the protocol doing what it should — and the paper presents it as such.

**H80 Reuleaux constant-width (−8.83 pp):** Listed in README.md §5 "Negative results (first-class citizens)" table with equal prominence to H50 and H41. In PAPER.md §7.3.1 line 488 H80 is mentioned only as one of the post-hoc-reclassified-as-screening tags. The paper proper gives H80 less treatment than the README. **MINOR.** It would strengthen the negative-result discipline to give H80 its own one-paragraph case-study mention in PAPER.md §5.X (analogous to §5.4's H09 treatment), so the three falsifications get equal prominence with the three certified positives.

**Negative results in abstract:** The abstract does NOT explicitly mention the three negative results by name. It mentions "3 BROKEN / 15 MAJOR / 24 MINOR" implementation defects and "ONE ranks NOVEL+TESTABLE" sci-critic verdicts, but the H50 −11.54 pp catastrophic-stack falsification (the canonical Rule-23 motivator) is not in the abstract. Given Rule 9's "Negative results are first-class citizens" and the paper's own claim to honest framing, the abstract should give the single most-load-bearing negative result one sentence. The H50 case is arguably more important to the protocol-as-contribution story than any of the three certified positives — it is the empirical cautionary tale that generates Rule 23.

**Score on C:** 4 / 5.

---

## D. Hedging discipline — PASS (with one residual occurrence)

**"outside seed noise" red-flag scan:**
- PAPER.md: 0 occurrences in main body; 1 occurrence at line 684 in Appendix D's mapping table, used historically ("L130/134/138 (Phase-8 numbers, 'outside seed noise')") to mark what the OLD framing was. This is acceptable — it is meta-commentary on the area-chair's flag, not a live claim.
- FINDINGS.md: 2 occurrences. Line 85 ("leads are outside seed noise") inside the historical 2026-05-29 AM n=3 verdict block. Line 371 inside the older content. **The n=7 promotion block at the top of FINDINGS.md does NOT use the phrase — it uses proper "paired Wilcoxon W=0, p=0.0078, Holm-Bonferroni-cleared" language.** But the older content retains "outside seed noise" wording.

**Recommendation:** Either (a) prefix the line-85 occurrence with "(superseded — the n=7 PROMOTION block above is the current statistical claim)" so a reader landing there knows the framing is historical, or (b) edit the line to remove the phrase and rely on the post-Wilcoxon framing. Currently a reader could quote line 85's "outside seed noise" as a live claim. **MINOR.**

**"Phase-8 winner" framing:** Used 38 times in PAPER.md. The framing is appropriately hedged in §5.5 ("case studies of protocol output that DID survive Phase-9-level statistical rigor at the screening compute budget"), and the paper consistently flags the open Phase-9a converged-budget question. The hedging discipline on "Phase-8 winner" is solid.

**"Provisional" language:** Used 5 times in PAPER.md. Used appropriately in the Phase-9a context and the H22 verdict requalification. Not over-applied.

**"Certified" language:** Used 18 times. Always paired with "at α=0.05 Holm-Bonferroni at n=7 default-config." Always carries the "at the screening compute budget" hedge. Excellent discipline.

**Score on D:** 4 / 5.

---

## E. Length / structure — FAIL

**PAPER.md is 848 lines.** For ICML 2027 (8-page main + unlimited supplementary), 848 lines is implausibly long. The paper appears to be targeting an arXiv preprint + supplementary structure, but this is never stated. A reviewer should be told up front: "this is the arXiv version; the ICML main-track will be the first 200 lines + headline tables, with §5.5.1–5.5.4, §5.6–5.8, §7.2.1, and all Appendices A–F deferred to supplementary." Without that scaffolding, the paper reads as an over-long camera-ready that nobody triaged for length.

**Section-order issues:**

- §1.3 "Limitations of the audit protocol — auditor self-grading" is the single most important caveat in the paper (the 51 % non-PASS rate is conditional on it). Placing it in §1.3 BEFORE methods is the right call, BUT it consumes 8 lines of dense prose before the reader has any concept of what the audit is. A 200-word preview is hard to absorb when the reader doesn't yet know what Track A or Track B does. **Recommendation:** keep a 2-line teaser in §1.3 ("Implementer, auditor, sci-critic, and Fixer agents are all from the same model family; the 51 % non-PASS rate is conditional on this caveat — see §1.3 and §5.8") and move the full discussion to a §5.0 "Audit calibration caveats" block before the §5.1 distribution tables.

- §5.5 is structured as §5.5 → §5.5.1 → §5.5.2 → §5.5.3 → §5.5.4. Five sub-sub-sections on the same topic (the three certified winners). §5.5.1 is a confound, §5.5.2 is a SIREN case-study, §5.5.3 is an H09 case-study, §5.5.4 is the hill-climbed-best regime. This reads as a transcript of a reviewer dialogue (BLOCKER #4 → §5.5.1; F4 SIREN concern → §5.5.2; BLOCKER #13 → §5.5.4) rather than a research narrative. **Recommendation:** consolidate §5.5.1–5.5.3 into one §5.5 narrative paragraph; promote §5.5.4 (hill-climbed-best) to its own §5.6, since it carries genuinely new data, not a reviewer rebuttal.

- **§5.5 comes too early.** The paper's headline numerical result appears in §5.5 before the reader has seen the §6 empirical-evidence section (the full sweep context). A camera-ready should put the headline result AFTER the methodology / sweep description, not interleaved with the audit calibration. Currently §5 (results — audit) and §6 (empirical evidence) are essentially two separate result chapters; a unified §5 with a clean ordering (audit verdicts → screening sweep → Phase-8 certification → hill-climb robustness) would read better.

**Appendices A–F:** 6 appendices is a lot.
- Appendix A (response to area-chair Section F threats): justified — it's an explicit point-by-point rebuttal.
- Appendix B (Phase-9 pre-registration spec): justified — operational artifact.
- Appendix C (operator quick-reference): borderline — duplicates CLAUDE.md §8. **Recommendation:** drop Appendix C and link to CLAUDE.md §8.
- Appendix D (mapping from REVIEWER_PASS_PAPER.md line numbers to revised PAPER.md sections): this is reviewer-correspondence content, not paper content. **Recommendation:** move to `audits/REVIEWER_PASS_PAPER_RESPONSE.md` (or keep here only for the camera-ready). It is not a thing an external reader needs.
- Appendix E (per-hypothesis dual-track verdict table, 117 lines): justified as full disclosure, but consider compressing to a single 84-row CSV in supplementary materials.
- Appendix F (protocol transferability spec): justified — it specifies the content-agnostic contract.

**Net length impact:** cuts to abstract (−200 words), §1.3 (−4 lines preview), §5.5.1–5.5.3 consolidation (−~30 lines), Appendix C drop (−~60 lines), Appendix D move (−~30 lines), Appendix E compression (−~60 lines). Estimated 200-line reduction is achievable without losing content.

**Score on E:** 2 / 5.

---

## F. Citation rigor (Rule 4) — PARTIAL

**Spot-check of 5 references (lines 516–531):**

1. **He K et al. 2016 CVPR. *Deep Residual Learning for Image Recognition*. arXiv:1512.03385.** — Rule-4-compliant.
2. **Sitzmann V et al. 2020 NeurIPS. *Implicit Neural Representations with Periodic Activation Functions (SIREN)*. arXiv:2006.09661.** — Rule-4-compliant (venue + arXiv ID + title + relevance present).
3. **Loshchilov I, Hutter F. 2019 ICLR. *Decoupled Weight Decay Regularization (AdamW)*. arXiv:1711.05101.** — Rule-4-compliant.
4. **Pittorino F et al. 2022 [VERIFY: venue, ICLR-W or NeurIPS-W to be confirmed]. *Toroidal flat-minima analysis* [VERIFY: exact title]. arXiv:[VERIFY] — H22 toroidal embedding literature anchor.** — **NOT Rule-4-compliant.** Three [VERIFY] tags remaining: venue, title, and arXiv ID are all open. A foundational reference for H22 is unverified.
5. **Islam M et al. 2025 [VERIFY: venue and arXiv ID]. *Platonic Transformers: A Solid Choice for Equivariance*. [arXiv:2510.03511] [VERIFY: arXiv ID under verification].** — **NOT Rule-4-compliant.** Cited as the literature anchor for the H55 PlatonicAttention fix; venue and arXiv ID still under verification.

**Foundational citations check:**
- Bronstein 2021 GDL — NOT present in PAPER.md References section. The paper mentions "Geometric Deep Learning (GDL) and Topological Deep Learning (TDL)" in §1, but Bronstein 2021's *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges* is the canonical citation and is absent. The full literature survey is in `paper/NATURE_INSPIRED_NETWORKS.md`, which the References section points to, but the GDL foundational reference should be in the main paper's References. **MAJOR (citation completeness).**
- He 2016 ResNet — present and correct.
- Loshchilov 2019 AdamW — present and correct.
- Reddi 2018 Adam — present and correct.
- Sitzmann 2020 SIREN — present and correct.
- Radosavovic 2020 RegNet — present and correct.

**[VERIFY] tag status:** 3 [VERIFY] tags remain unresolved (the Pittorino 2022 reference and the Islam 2025 reference at 2 tags each, counted in 2 references). The reviewer-revision table at line 20 marks item #11 as **PARTIAL**, consistent with this. But "PARTIAL" status on Rule-4-compliance is a defect — the paper's own Rule 4 forbids parenthetical-only citations and the [VERIFY] tag is functionally a placeholder. Either resolve the verification (one search on arXiv would close Pittorino + Islam) or drop the citations and rewrite the dependent sections.

**Recommendation:** Before resubmission, resolve all [VERIFY] tags. Two hours of arXiv search would close the gap. The paper's own normative rules require it.

**Score on F:** 3 / 5.

---

## G. Reader accessibility — PASS

**Cold-reader 5-click navigation test (README → PAPER → FINDINGS → STATISTICAL_TESTS → one cited claim's metrics.json):**

1. README §4 (Headline claims) → click on "STATISTICAL_TESTS.md" — 1 click.
2. STATISTICAL_TESTS.md §0 (PROMOTION banner) → click on `experiments/cifar100/pair_gm_pdw_seed{0..6}/metrics.json` — would be 2 clicks if the per-seed paths are hyperlinked.

This is achievable in ≤ 5 clicks under the canonical reading path. The README → PAPER → FINDINGS → STATISTICAL_TESTS → metrics.json path is the documented "reviewer's reading order" and it works.

**The "screening vs evaluation" framing for a first-time reader:** The framing is defined in §7.3.1 (HARKing acknowledgement) and is codified as CLAUDE.md Rule 28. A first-time reader who reads §1 → §5.5 → §7.3.1 in order will encounter the distinction at the right moment (after the headline numbers but before the limitations). The framing is clear. **PASS.**

**Protocol-as-contribution value proposition in first 200 words of abstract:** The protocol's value is stated in the first sentence ("a dual-track skeptical audit + Fixer + per-experiment-page protocol for autoresearch campaigns that propose large families of architectural priors"). The "8-agent implementation-critic team + 8-agent research-scientist-critic team + 8-agent Fixer campaign + re-runs every affected sweep row on the corrected code" descriptive sentence at L1 is dense but informative. **PASS.**

**However**, the abstract's first 200 words do NOT make clear "what does the protocol do that an unaudited pipeline cannot do." The H09 phi_budget realised-ratio drift (1:1.41:2.45 vs claimed 1:1.618:2.618 — a 12.6 % drift caught by Track A) is the single most-load-bearing example of the protocol's diagnostic power, and it does not appear until §5.4 / §1.2. **Recommendation:** the abstract should have one sentence like "the protocol caught an unaudited pipeline's headline claim (CIFAR-100 +1.53 pp lift attributed to φ-budget) that turned out to be produced by a network whose realised stage-parameter ratio was 1:1.41:2.45, not the doc-claimed 1:1.618:2.618 — a code-vs-doc divergence the protocol's Track A audit detected and the Fixer corrected." That is the elevator pitch for the protocol's value proposition.

**Score on G:** 4 / 5.

---

## H. Mystical-motivation transparency — PASS

CLAUDE.md Rule 16 mandates neutral artifact names (`nature_inspired_networks`, `NaturePrior*`) and mystical motivation acknowledged in prose only. The discipline is maintained:

- Repo name: `nature_inspired_networks`. ✓
- Package name: `nature_inspired_networks`. ✓
- Class names: `NaturePriorBlock`, `NaturePriorFlags`. ✓
- Block names in `IDEA_TABLE.md`: neutral throughout (`phi_budget`, `hex_phi`, `golden_modulate`, ...). ✓
- §1.4 "Mystical motivation, neutral artifact names" — explicit acknowledgement that the mystical inspiration is "a prior over the design space, not as evidence."
- Abstract: "(φ-scaling, hexagonal lattices, Platonic / icosahedral equivariance, fractal recursion, toroidal closure, Chladni cymatic init, golden-angle modulation, and 15 cross-paradigm hybrids)" — descriptive of the design space, does not lean on mystical motivation as evidence.

**The discipline does NOT leak into headline claims:** The certified Phase-8 numbers are framed as "candidates" with explicit "non-φ control still open" hedging in §5.5.1. The φ-content vs any-three-axes attribution question is explicitly flagged as open. **PASS.**

One residual style issue: §7.2.1 line 456 still uses the framing "the protocol's nominee for the 'three-axis orthogonal stack outperforms its solo components' hypothesis — certified at the screening compute budget." This is neutral and academic. No mystical leakage. **PASS.**

**Score on H:** 5 / 5.

---

## Top 3 strengths

1. **Honest framing throughout.** The HARKing acknowledgement (§7.3.1), the auditor-self-grading caveat (§1.3), the SOTA-gap caveat (§6.4), the "non-φ 3-axis control OPEN" label (§5.5.1), the "claimed by construction, not demonstrated" portability caveat (§1.1 / Appendix F) — every load-bearing claim is explicitly hedged with the right caveat. This is the kind of methodological courage NeurIPS / ICML reviewers reward, even when the empirical evidence is modest.

2. **The n=7 Holm-Bonferroni certification is a real upgrade.** The pre-revision paper had a fragile statistical position; the n=7 extension + Holm-Bonferroni + bootstrap CIs gives the three Phase-8 numbers a defensible statistical floor. The paper now CLEARS the area-chair's BLOCKER #3 cleanly: "outside seed noise" is gone; "paired Wilcoxon W=0, p=0.0078 < α'_Holm=0.0167" replaces it. The retirement of the "FAILS Holm-Bonferroni" old framing is complete (0 occurrences in repo).

3. **Negative-result discipline (especially H50).** The 9-row combo-ladder table in §7.2.1 showing `plr` LR-schedule as the single-axis stack-killer (−5.66 pp) is exemplary engineering forensics. The negative result generates Rule 23. The H41 eps-confound requalification is also handled with appropriate honesty (the pre-fix −33 pp number is explicitly retired as confounded). H80 Reuleaux is given equal billing in the README. This is what "negative results are first-class citizens" should look like in practice.

---

## Top 3 weaknesses

1. **Length and structure problem (848 lines).** The paper as written is implausibly long for ICML main-track. Without explicit "arXiv preprint + supplementary" scaffolding, a reader confronts a wall of text. The §5.5 → §5.5.1 → §5.5.2 → §5.5.3 → §5.5.4 chain reads as a transcript of reviewer rebuttals rather than a research narrative. The pre-abstract 27-line revision-status table is reviewer correspondence, not paper content. The Appendix D mapping table is also reviewer correspondence. Aggressive cuts (estimated 200 lines achievable) are needed.

2. **Abstract is too long and over-numerical.** 440 words; ICML convention is 150–200. The protocol's value proposition (what does it CATCH that an unaudited pipeline doesn't) is buried until §5.4. The H09 realised-ratio drift is the single most compelling example of the protocol's diagnostic power and should appear in the abstract. The certified Phase-8 numbers should be compressed to one sentence; the protocol's elevator pitch should be expanded.

3. **Three unresolved [VERIFY] citation tags.** The Pittorino 2022 reference has three [VERIFY] tags (venue, title, arXiv ID — all open). The Islam 2025 reference has two [VERIFY] tags. The paper's own normative Rule 4 forbids non-compliant citations; PARTIAL is not a passing grade. Two hours of arXiv search would close all three [VERIFY] tags. Bronstein 2021 GDL is the canonical foundational reference for the GDL framing in §1 and is absent from the References section (present only via the `NATURE_INSPIRED_NETWORKS.md` survey link).

---

## Questions for the authors

1. **Q1 — abstract framing.** The abstract spends ~210 of its ~440 words on the n=7 certified Phase-8 numbers, but the conclusion (§8) says "We make no standalone empirical headline claim for nature-inspired priors at this scale." A reader could fairly conclude the abstract and conclusion are arguing different cases. Will you compress the n=7 result to a single sentence in the abstract and expand the protocol's elevator pitch (specifically, the H09 realised-ratio drift as the protocol's signature catch) to claim the recovered word budget?

2. **Q2 — § 5.5.4 hill-climbed Δmean vs Δmedian.** The §5.5.4 table reports both Δmedian and Δmean at n=3 in the hill-climbed regime, with the explanation "Δmedian and Δmean differ because the hill-climbed baseline has notably higher seed-variance (σ=0.97 pp)." This is a methodologically defensible reporting choice, but at n=3 with high baseline variance the median ranking is sensitive to a single high baseline seed (0.6085). What is the probability under the bootstrap that re-running the hill-climbed baseline at a different 3-seed selection would produce a baseline median > one of the leader medians? Without that diagnostic, the "+1.20 / +1.80 / +2.08 pp Δmedian" headline at n=3 is more fragile than the n=7 default-config certification suggests.

3. **Q3 — venue / length declaration.** Is the target venue ICML 2027 main track (8 pages + unlimited supplementary, blind review) or arXiv preprint + ICML methods note? At 848 lines, the current paper cannot fit in ICML main-track without extensive supplementary structure. A camera-ready scaffolding statement (in the abstract or in a footnote) would help reviewers understand which content is being claimed as main-track contribution and which is supplementary.

---

## Commit SHA

Review composed against PAPER.md, paper/FINDINGS.md, paper/AUDIT_SUMMARY.md, README.md, paper/NATURE_INSPIRED_NETWORKS.md, paper/REVIEWER_CHECKLIST.md, paper/SELF_AUDIT_CHECKLIST.md, paper/ETHICS_STATEMENT.md, paper/LIMITATIONS.md, audits/REVIEWER_PASS_PAPER.md at the state of the working tree on 2026-05-30 PM. Commit SHA for this review: filled in by the retry-wrapped commit after this file lands.

---

*R4 Writing/Clarity/Framing review complete. Parallel reviewer passes R1 (novelty), R2 (empirical rigor), R3 (reproducibility) are in flight under `audits/ICML_REVIEWS_2026-05-30/`.*
