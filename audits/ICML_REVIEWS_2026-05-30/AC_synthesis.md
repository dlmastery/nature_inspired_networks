# ICML 2027 Area Chair Meta-Review

**Submission:** *A Skeptical Protocol for Nature-Inspired Neural-Network Priors: Methodological Contribution and an 84-Hypothesis Dual-Track Audit on CIFAR-10/-100*

**AC Recommendation:** **Weak Reject (main track)** with explicit invitation to resubmit to the **Datasets & Benchmarks track** with revisions (effective acceptance there).

**Track recommendation:** **Datasets & Benchmarks / Reproducibility & Meta-Research workshop.** Main-track resubmission requires the W2 + W3 revisions in R3 §6 and the n=7-hillclimbed + non-φ control runs in R2 Q1–Q2.

**Confidence:** AC has read all four review files in full, the PAPER.md abstract, §5.4, §5.5–§5.5.4, the Rule 20–28 codification in CLAUDE.md, `STATISTICAL_TESTS.md` §0–§3 / §7, and `AUDIT_CALIBRATION_THIRD_PARTY.md` §1 + Appendix B. Confidence 4/5.

---

## Consensus across reviewers

**Strengths (≥ 3 reviewers agree):**

- **Honest framing of caveats is unusually courageous.** R1, R2, R3, R4 all explicitly cite the §1.3 auditor-self-grading caveat, the §7.3.1 HARKing acknowledgement, the §5.5.1 non-φ control OPEN label, and the abstract's `(a)/(b)/(c)` caveat triple as the kind of disclosure that NeurIPS / ICML reviewers reward. R3: "rare in ML papers and should be praised." R4: "methodological courage NeurIPS / ICML reviewers reward."
- **The Fixer + mechanism-pinning-test discipline is the load-bearing intellectual contribution.** R1 (H09 case as "single existence proof"), R2 ("reproducibility scaffolding is unusually complete"), R3 ("strongest software-engineering contribution … the most portable single idea"), R4 (S1 in negative-result framing). R3 explicitly says the abstract underemphasises this — the abstract leads with the dual-track audit when it should lead with the Fixer-mechanism-test contract.
- **The n=7 Holm-Bonferroni promotion is a real upgrade since the area-chair pass.** All four reviewers acknowledge the arithmetic is correct (Wilcoxon W=0, p=(1/2)⁷=0.0078, Holm α′=0.0167, k=3, bootstrap CIs ≥ 2σ_baseline). R2: "real upgrade over the n=3 framing." R4: "FAILS Holm-Bonferroni" framing fully retired. R1 acknowledges the internal mechanics are correct conditional on the inputs.
- **The third-party pytorch/vision audit calibration is the most diagnostically credible single number in the paper but is statistically marginal.** R1, R2, R3 all note that n=15 with χ²p≈0.22 / Fisher p≈0.07 on the MAJOR/BROKEN sub-tier does not clear α=0.05 — the paper's own convention. The 22-pp delta is "directionally credible" (R3) but cannot be a headline at this n. R3 spot-checked the calibration verdicts and found no auditor blind spots; the 0 % MAJOR/BROKEN on torchvision appears genuine.
- **The paper is over-scoped and under-replicated.** R1 (84-hypothesis design space is selection-biased, transformer track untested), R2 (controls READY but NOT RUN; BS confound on hill-climb), R3 (content-agnostic claim by construction not demonstration; no cross-domain replication), R4 (848 lines too long for ICML 8-page main; abstract 440 words vs convention 150–200) all converge on the same diagnosis: the contribution is real but the empirical envelope around it has not been closed.

**Weaknesses (≥ 3 reviewers agree):**

- **Auditor-self-grading at the model-family level is acknowledged but not dispatched** (R1 §F, R2 §1.3 reading, R3 W2). Third-party calibration auditor is also Opus 4.7. No cross-family agreement check has been run.
- **Effect sizes are dwarfed by the 6.5-pp gap-to-SOTA** (R1 §B, R2 weakness 3, R4 §C abstract framing). +1.24 to +2.08 pp at 30-ep CIFAR-100 is inside the typical Bag-of-Tricks intervention envelope; "first formally-certified empirical claim at NeurIPS-standard α" framing is not commensurate with magnitude.
- **Controls flagged in the area-chair pass remain unlaunched** (R1 BLOCKER on POSI, R2 §1 weakness, R3 §4.5 cross-domain replicate skill missing). `controls/PLAN.md` Controls 1–4 are wired but no sweep has fired.

---

## Divergence

- **Severity of the post-selection family size problem.** R1 treats POSI as a **BLOCKER** ("honest family is on the order of 40–50 row-level screening decisions; per-test α′ ≈ 0.001 cannot be cleared by Wilcoxon-at-floor 0.0078"). R2, R3, R4 treat it as a hedging-discipline issue rather than a statistical fatality. **AC's take:** R1 is right that the k=3 family is post-screening and Holm is mis-applied; R2/R3/R4 are right that the paper *acknowledges* the screening-vs-evaluation split (Rule 28) and so reads as "we use Phase-5 ordinal-gate-at-n=3 to *nominate* and confirm at n=7" — which is the standard two-stage clinical-trial pattern, not pure HARKing. R1's POSI objection is technically correct, but a paper that explicitly pre-commits to Rule 28 going forward and acknowledges the retrospective HARKing in §7.3.1 is closer to the honest end of the rigor spectrum than R1's BLOCKER tier suggests. AC weights this as a **MAJOR**, not a BLOCKER.

- **What §5.5.4 hill-climb refutes vs introduces.** R2 reads the hill-climb as **introducing a new confound** (BS=128 leaders vs BS=256 baseline default; the hill-climbed `sg_only_phi_budget` CI [−0.32, +1.76] includes 0). R1 reads it as **undercutting the n=7 certification** for the case-study primary. R3 treats it as a robustness pass. R4 treats it as a Δmedian-vs-Δmean reporting choice that needs better visibility. **AC's take:** R2's BS=128 baseline missing-row observation is the single most actionable empirical gap in the paper. One BS=128 baseline row × 7 seeds (~3.5 GPU-h on a laptop 4090) settles the question. The hill-climbed `sg_only_phi_budget` CI including 0 is real and should be acknowledged in the abstract as a *named* limitation of the case-study primary, not merely tabulated in §5.5.4. R1's framing is sharper than R3/R4's; the paper's case for `sg_only_phi_budget` is genuinely weaker than the case for `pair_gm_pdw` and `slot_act_sine` and the abstract should not list them as a uniform triple.

- **Whether the paper belongs at ICML main track at all.** R1 says workshop or D&B; R2 says borderline reject / D&B if controls land; R3 says strong accept (≥7) for D&B / methods-workshop, weak reject for main; R4 says arXiv preprint / methods note. **AC's take:** R3's bifurcated rating is the right diagnosis. The protocol-as-contribution headline is **not transformative enough for main track** under the current novelty conventions (every individual piece is a port of an established practice — R3 W1 enumerates them). It **is** well above bar for ICML D&B or a Reproducibility / Meta-Research workshop. AC concurs with R3's split rating: 5/10 main, 7/10 D&B.

- **Whether SIREN-style `slot_act_sine` belongs in a "nature-inspired" paper at all.** R1 flags this as a **category error in the headline framing** (§F: "a SIREN replication does not belong in a paper whose thesis space is nature-inspired (φ/Platonic/fractal/toroidal) priors"). R4 notes it is appropriately labelled "φ-prior-neutral" in §5.5.2. **AC's take:** R1 is right that listing `slot_act_sine` on equal footing with `pair_gm_pdw` and `sg_only_phi_budget` in the abstract's "first formally certified" triple is a framing defect. The paper's own §5.5.2 explicitly concedes the lift is φ-prior-neutral. The honest move is to demote it from the abstract triple to a case study of "the protocol surfaced a known-good SIREN-class candidate that an unaudited pipeline would have mis-attributed to φ-content" — that is a protocol-positive finding, not a nature-inspired-prior finding.

---

## Highest-leverage actionable items for camera-ready

1. **Launch Controls 1+2 + BS=128 baseline at single-seed-to-n=7.** (R2 Q1 + R2 Q2.) `controls/PLAN.md` Controls 1 (non-φ 3-axis stack) and 2 (activation ablation) at n=3 plus a BS=128 baseline at n=7 close the two largest empirical gaps. Cost: **~12–15 GPU-h on a laptop 4090** (Control 1 ≈ 4.5 GPU-h, Control 2 activation sweep across 5 alternatives × 3 seeds ≈ 5 GPU-h, BS=128 baseline × 7 seeds ≈ 3.5 GPU-h). Without these, `pair_gm_pdw` remains "candidate, confound-open" and `slot_act_sine` remains "SIREN replication misclassified as nature-inspired." This is the single highest-leverage piece of work.

2. **Cross-family audit on 10 verdicts.** (R3 §6 item 2.) Re-audit 10 of the 18 MAJOR/BROKEN findings using GPT-5 or Gemini 3 Pro; report verdict-agreement rate. Cost: **~5 person-hours, ≈ $20 in API credits** — no GPU. Neutralises W2 head-on. Without it, the 22-pp MAJOR/BROKEN diagnostic-credibility number remains conditional on Opus-as-auditor.

3. **Extend audit calibration to n ≥ 50 third-party hypotheses (Phase-9b).** (R1 Q4, R2 Q3, R3 §6 item 3, R4 §F.) Run the audit doctrine on `timm` + HuggingFace Transformers + Lightning Bolts for an additional 35–50 mechanisms. Cost: **~10 person-hours over 2 days** — no GPU. At n=50 the MAJOR/BROKEN 22-pp delta likely clears Fisher p<0.01 and the headline graduates from "directionally credible" to "statistically credible."

4. **Camera-ready compression to ≤ 250 lines main + supplementary scaffolding.** (R4 §E.) Cut the pre-abstract 27-line revision-status table (move to Appendix), drop Appendix C (duplicates CLAUDE.md §8), consolidate §5.5.1–5.5.3 into a single §5.5 narrative paragraph, compress the abstract from 440 → 200 words, lift the H09 realised-ratio-drift case study into the abstract as the protocol's signature catch (R4 G + R3 closing flag). Cost: **~6 person-hours, no GPU.**

5. **Resolve the three [VERIFY] citation tags + add Bronstein 2021 GDL.** (R4 §F.) Pittorino 2022 venue/title/arXiv, Islam 2025 arXiv, plus the missing Bronstein 2021 *Geometric Deep Learning* foundational reference. Cost: **~2 person-hours of arXiv search.** The paper's own Rule 4 forbids `[VERIFY]` placeholders.

**Stretch (if time permits before camera-ready):**

6. **One cross-domain skill replication** — run `autoresearch-critic-team` against `autoresearchtabular/core/*.py` and report the non-PASS rate. (R3 §6 item 1.) Operationalises content-agnosticism from prose-analogy to empirical demonstration. Cost: **~1 day of agent-time, ≈ $30 API.**

---

## Items the authors should defend in rebuttal

- **The protocol-as-contribution framing.** The contribution is the integration of dual-track audit + Fixer-mechanism-test + per-experiment-page + Rule-28 screening-vs-evaluation into a single executable refuse-to-launch protocol. The individual pieces are ports of established practices (R3 W1 table), but the integration is novel and well-engineered. The paper should own this and lead the abstract with the *integration* claim, not the n=7 numbers.

- **The H09 realised-ratio-drift case study (12.6 % drift caught by Track A) as the protocol's signature value-add.** This is the cleanest single demonstration in the paper that an unaudited pipeline would have shipped a wrong headline. R1 acknowledges this as a "single existence proof," R3 as the "value tangible" case, R4 as the elevator-pitch example. The authors should lift H09 into the abstract.

- **The n=7 Holm-Bonferroni result conditional on Rule 28's two-stage screening-then-confirmation framing.** The pattern (n=3 ordinal-gate screen nominates → n=7 paired-Wilcoxon confirms) is the standard two-stage clinical-trial pattern and is defensible if framed as such. R1's POSI objection lands if the paper claims the n=7 family is a fresh independent test; the defensible framing is "the screening tier nominated three candidates; we confirmed them at NeurIPS-α at the screening compute budget under the screening recipe."

- **The third-party calibration at n=15 as a *re-framing tool*, not a headline.** R3 spot-checked the verdicts and found them well-calibrated (the TP12 `CosineAnnealingLR` no-restart issue, TP15 BatchNorm momentum, TP2 V1.5 stride-placement are all real). The calibration's role is to bound the MINOR-tier auditor-aggressiveness effect, not to prove the project codebase has more bugs.

- **The §5.5.4 hill-climb as a robustness pass, not a re-certification.** The paper is already honest that n=3 cannot clear Holm; the framing is correct. The defensible refinement is to acknowledge that `sg_only_phi_budget` has the weakest hill-climbed CI (includes 0) and to demote it from the abstract triple.

---

## Items the authors should concede

- **`slot_act_sine` is a SIREN replication, not a nature-inspired prior result.** Concede in the abstract; demote from the certified-triple framing to a separately-labelled "protocol surfaced a SIREN-class candidate" case study. The paper's own §5.5.2 already says this; the abstract should match.

- **The k=3 family for Holm is a post-screening selection.** Concede that the certified claim is "conditional on the Rule 28 two-stage screening-then-confirmation pattern" rather than "a fresh family of three." A pre-registration footnote — a commit hash dated before any seeds 3–6 were run — is the cleanest artifact (R1 Q5). Add a single sentence to §5.5: "n=7 was pre-specified at commit `<hash>` before seeds 3–6 launched, to clear Holm at the Wilcoxon floor."

- **Content-agnostic portability is claimed by construction, not demonstrated.** Concede in §1.1 contribution 3. R3 W3 is correct that the skills are CIFAR-conditional in their parameter defaults. The §1.1 wording is already close to honest; tighten it to "the skills are content-agnostic in their templating layer; cross-domain transferability is open future work."

- **The Wilcoxon-at-floor is informationally identical to a sign test.** Concede in `STATISTICAL_TESTS.md` §2 (the paper already says this); add a paired-permutation-test-on-Δmean column (~2 hours of analysis on existing data) so the magnitude-test gap R1 §A.1 raises is closed at zero additional GPU cost.

- **`sg_only_phi_budget`'s hill-climbed CI includes 0.** Concede in the abstract — the §5.5.4 table already shows it. Acknowledge that the case-study primary has the weakest evidence under the more honest analysis condition.

- **The 22-pp MAJOR/BROKEN-tier excess is not statistically significant at n=15 (χ² p≈0.22 / Fisher p≈0.07).** Concede in §5.8; treat as a "credible-interval re-framing" not a "diagnostically-credible signal." The phrase should be "tightens the credible interval on the audit's diagnostic floor" rather than "establishes excess defect density."

- **The 848-line PAPER.md cannot fit ICML 8-page main without explicit supplementary scaffolding.** Concede with an explicit footnote or §0 declaration: "This is the arXiv preprint version; the ICML camera-ready will be §1–§5.5 + §7.2.1 + §8, with §5.5.1–5.5.4, §5.6–5.8, all Appendices, and `STATISTICAL_TESTS.md` / `FINDINGS.md` / `AUDIT_CALIBRATION_THIRD_PARTY.md` as supplementary."

---

## AC recommendation justification (≤ 400 words)

**The submission's strongest defensible claim.** The Fixer-with-mechanism-verifying-test contract (Rule 21 + `autoresearch-fixer-campaign` skill), demonstrated concretely on H09 phi_budget's 12.6 % realised-ratio drift, on H55 PlatonicAttention's mathematically-zero head bias (caught by a math-verifiable centroid identity), on H67's silent ImportError, and on H74's 13-α collapse. This is a port of mutation testing into the LLM-agent-implements-LLM-agent-audits regime; it is novel as integration, well-engineered as software, and durable as artifact. It addresses the specific pathology that LLM-implementer-agents optimise for "tests pass" and so cut the shape-only-test corner (R3 §4.6) — a pathology R3 considers a research finding in its own right.

**The submission's weakest claim.** The "first formally-certified empirical claim at NeurIPS-standard α under Holm-Bonferroni" abstract banner. The k=3 family is a post-screening selection (R1 BLOCKER); the Wilcoxon-at-n=7-floor is informationally a sign test (R1); the case-study primary `sg_only_phi_budget` has a hill-climbed CI that includes 0 (R1, R2); `slot_act_sine` is acknowledged as φ-prior-neutral SIREN (R1, paper §5.5.2); the BS=128 baseline confound is unresolved (R2); the controls flagged by the area-chair pass are READY but NOT LAUNCHED (R2 Q1); effect sizes are inside Bag-of-Tricks envelope and < 1/8 of the gap-to-SOTA (R1 §B). The headline rests on a stack of acknowledged-but-unrebutted caveats.

**Why D&B / methods-workshop is the right fit.** The contribution is a *protocol* with an *empirical calibration on a single design space*. ICML D&B exists for benchmark-and-methodology papers whose empirical envelope is narrower than main-track but whose methodological contribution is publishable in its own right. R3's verdict — strong accept (≥7) for D&B, weak reject for main — is the correct AC reading. The protocol's seven content-agnostic skills, the 28-rule normative codification, the SHA-256-fingerprinted composite, the per-experiment-page discipline, the Pages-link convention — all are exactly the kind of reusable methodological infrastructure D&B was created to publish.

**What would push it to a higher track.** (i) Cross-domain replication of even ONE skill on ONE sister repo (R3 §6 item 1), (ii) cross-family audit on 10 verdicts (R3 §6 item 2), (iii) Controls 1+2 + BS=128 baseline launched and reported (R2 Q1+Q2). With those three additions, AC's rating moves from 5/10 main to 7/10 main; without them, the paper is firmly at D&B / workshop scale, which is **not** a downgrade — it is a correct categorisation.

---

## Score

- **Soundness median across reviewers:** 2/4 (R1 2/4; R2 implicit 2/4; R3 implicit 2/4; R4 implicit 3/4 — protocol-mechanics-correctness high, statistical-headline-soundness low).
- **Presentation median:** 2/4 (R1 2/4; R2 implicit 3/4 — reproducibility scaffolding excellent; R3 implicit 3/4; R4 explicit 2/5 → 2/4 — length and structure problem load-bearing).
- **Contribution median:** 2/4 (R1 2/4; R2 implicit 2/4; R3 implicit 2/4 main / 3/4 D&B; R4 implicit 3/4 — protocol-as-contribution lands).
- **Rating mean across reviewers:** (4 + 5 + 5 + 5) / 4 = **4.75 / 10** (R1: 4, R2: 5, R3: 5 main / 7 D&B, R4: 5).
- **AC final score:** **5 / 10 main track** (concurring with R2/R3/R4 above R1; the protocol contribution is real and the n=7 statistical apparatus is internally correct; R1's POSI objection is a MAJOR but not a BLOCKER given the Rule 28 two-stage pre-commitment). **7 / 10 D&B track** (concurring with R3's bifurcated rating; the paper is well above bar for the D&B / Reproducibility & Meta-Research category).

**AC final decision:** **Weak Reject for main; Accept-with-revisions for D&B / Reproducibility & Meta-Research workshop**, conditional on the five camera-ready punchlist items above. The protocol contribution is genuine and worth publishing; the empirical headline cannot carry main-track weight at the current envelope. Authors should be encouraged to resubmit to D&B promptly — this is a paper the field benefits from having on the record.

---

*AC synthesis complete. Composed against R1, R2, R3, R4 review files and PAPER.md abstract + §5.4 + §5.5–§5.5.4 + §5.8, on 2026-05-30 PM. Parallel agent work-streams (Final-Sniff-Test, etc.) remain in flight under disjoint scope.*
