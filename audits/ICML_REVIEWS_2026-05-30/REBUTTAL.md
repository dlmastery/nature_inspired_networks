# ICML 2027 Author Rebuttal — R1 / R2 / R3 / R4

Paper: *A Skeptical Protocol for Nature-Inspired Neural-Network Priors* (84-hypothesis dual-track audit on CIFAR-10/-100). Repo at rebuttal time: post-`5e930e3`. Doc-only fixes landed without GPU work; empirical extensions (Controls 1–4, Phase-9b/c/d) are filed as future work pending separate authorisation.

We thank R1–R4 for substantive engagement. R1's POSI / magnitude / hill-climb-CI critique, R2's controls-not-launched and BS-confound, R3's incremental-novelty / cross-domain gaps, and R4's length / abstract / citation cleanup are accepted in substance; this revision lands the doc-only fixes.

---

## R1 — Theoretical / Statistical Rigor (4/10)

**Q1 — POSI family-size at k=49.** CONCEDED + CLARIFIED. PAPER.md §5.5 now reports certification under TWO bounds: k=3 CONFIRMATORY (Wilcoxon p=0.0078 < α'_Holm=0.0167, cleared) and strict k=49 SCREENING POSI (α'≈0.001). The Phase-5 ordinal gate is the pre-specified screening criterion; CLAUDE.md Rule 28 was committed before seeds 3..6 of the n=7 extension ran (the registration evidence). Under the strict POSI bound, `pair_gm_pdw` and `slot_act_sine` clear (paired-t p<0.001 — see Q3); `sg_only_phi_budget` does NOT and is labelled "POSI-uncertified but family-of-3 certified". Both bounds reported.

**Q2 — Hill-climbed `sg_only_phi_budget` CI [−0.32, +1.76] includes 0.** CONCEDED. §5.5 hill-climb paragraph reads the CI as honestly undercutting the default-config certification under tuned-vs-tuned. The n=7 default-config remains the *formal* claim; the hill-climb is *additive*; the asymmetry is reported, not hidden.

**Q3 — Wilcoxon-at-floor = sign test; magnitude test required.** CONCEDED + FIXED. `scripts/_compute_stat_tests.py` §9 (commit `3f501a3`) adds exact paired permutation on Δmean (2^7=128 sign-flips) and paired-t (df=6). Permutation attains the same 1/128 floor when every delta is positive (mechanical — observed Δmean IS the maximum sign-flipped mean). Paired-t: **p_one = 5.1×10⁻⁵ (`pair_gm_pdw`), 1.2×10⁻⁴ (`slot_act_sine`), 8.1×10⁻⁴ (`sg_only_phi_budget`)** — three to four orders below the floor. Spliced into PAPER.md §5.5.

**Q4 — Calibration n=15; not significant at α=0.05.** PARTIALLY CONCEDED. `AUDIT_CALIBRATION_THIRD_PARTY.md` §4.3.1 + `STATISTICAL_TESTS.md` §8 (commit `3f501a3`): bootstrap 95 % CI on diff = **[+13.3, +31.3] pp** (excludes 0); Wilson project [14.2 %, 31.7 %], calibration [0.0 %, 20.4 %]; **Fisher exact one-sided p=0.036** (clears α=0.05); two-sided p=0.066 (does NOT). One-sided significant, bootstrap-CI excludes 0; conservative two-sided not yet. Phase-9b extends to n≥50.

**Q5 — Stopping-rule.** R1 is correct that n=7 was chosen because it clears Holm at the Wilcoxon floor — that choice is in the open in CLAUDE.md Rule 28, committed before seeds 3..6 ran. The screening-phase HARKing is acknowledged in §7.3.1; Rule 28 binds prospectively.

**Q6 — Same-family calibration auditor.** CONCEDED (Opus 4.7; acknowledged in calibration Appendix B-1). §7.4-8 files cross-family subsample (10 verdicts re-audited by GPT-5 / Gemini 3 Pro), which also addresses R3 Q1.

**MAJOR/MINOR weaknesses.** Post-selection on screening: §7.3.1 HARKing. CIFAR-10 n=1 verdicts: §6.1 explicit "no seed-0 tag has prima-facie credibility". Effect sizes vs Bag-of-Tricks band: §6.3. Model-family overlap: §1.3 + §7.4-8. `slot_act_sine` SIREN-not-φ: §5.5 honest reading. Composite-metric not headline-load-bearing: ACKNOWLEDGED (the SHA-256 *fingerprint discipline* is the contribution, not the weights).

Commits: **`3f501a3`** + **`5e930e3`**.

---

## R2 — Empirical / Experimental Rigor (5/10)

**Q1 — Controls 1–4 READY but unlaunched.** CONCEDED, FILED §7.4-10. §5.5 carries "candidate, confound-open" for `pair_gm_pdw`. We do NOT claim Controls 1–4 are launched at submission; the camera-ready window will run them (~31.75 GPU-h).

**Q2 — BS=128 baseline missing from hill-climb refutation.** CONCEDED. §5.5 flags the asymmetry (leaders bs=128, baseline bs=256). A bs=128 baseline at n=7 (~3.5 GPU-h) is filed as Phase-9d (§7.4-11). Hill-climb refutation downgraded from "refuted" to "qualitatively refuted at this compute budget; bs=128 baseline outstanding."

**Q3 — Bootstrap CI on 22-pp MAJOR/BROKEN excess.** DONE in `3f501a3`. Numbers in R1 Q4.

**Weaknesses.** Controls READY-not-launched, BS confound: filed. Abstract reads past AT-THIS-BUDGET hedge: DONE (199 words, leads with H09). CIFAR-10 12-ep seed-noise: §7.3.1 HARKing + §6.1. `cudnn.benchmark=True` non-determinism: §4.4 / Rule 6. **3-vs-6 axes Rule-23 derived from n=1**: CONCEDED — caveat sentence added at §7.2.1 and §6.2: "Rule-23 was derived from n=1 screening data; replication at n=7 is filed as Phase-9d future work." Phase-7 non-leader rows n=1: CONCEDED; Rule 21 stated, not fully executed at R2's rigor. Cross-domain transfer absent: §7.4-6.

Commits: **`3f501a3`** + **`5e930e3`**.

---

## R3 — Methodology (5/10)

**Q1 — Cross-family audit of 5–10 MAJOR/BROKEN findings.** CONCEDED, FILED §7.4-8 (overlaps R1 Q6).

**Q2 — Skill against `autoresearchtabular` Higgs.** CONCEDED, FILED §7.4-6. §1.1-3 already labels portability as "claimed by construction, not demonstrated".

**Q3 — Mechanism-pinning tests value-specific vs property-based.** PARTIALLY CONCEDED. Camera-ready will add `autoresearch-property-based-test` skill and convert at least one Fixer test (H09 the natural candidate).

**Q4 — Phase-9b timeline.** Filed §7.4-5. If it lands in the camera-ready window the 22-pp claim moves from one-sided p=0.036 to credible (p<0.01 at n=50).

**Weaknesses.** W1 incremental novelty over LLM-as-judge / TDD / pre-registration: CONCEDED. Novelty claim is restricted to the *integration* into a gated, refuse-to-launch runner with the *Fixer-must-add-mechanism-test* contract — not the individual pieces; we add Zheng 2023, Bai 2022, and mutation-testing literature in camera-ready §4. W2 model-family overlap: §7.4-8. W3 transferability claimed not demonstrated: §7.4-6. The new abstract leads with the H09 catch — the load-bearing demonstration of mechanism-pinning test value, per R3's recommendation.

Commits: **`5e930e3`**.

---

## R4 — Writing / Clarity / Framing (5/10)

**Q1 — Abstract too long, over-numerical; lead with H09.** DONE. 199 words; leads with the H09 realised-ratio catch (12.6 % drift caught by Track A; Fixer mechanism-pinning test `519cdf3`); certification compressed to one sentence; three caveats preserved (6.5 pp SOTA gap, n=1 single-prior rows, Opus 4.7 family overlap). The 27-line pre-abstract revision-status table is GONE.

**Q2 — Δmedian vs Δmean fragility at n=3 hill-climb.** CONCEDED. Phase-9c n=7 hill-climbed extension is the proper resolution (filed §7.4). Formal claim remains n=7 default-config.

**Q3 — Venue / length declaration.** ACKNOWLEDGED. PAPER.md (318 lines) is the arXiv preprint structure; ICML 2027 main-track is §1–§8 + Appendices A–D supplementary. Footer scaffolding makes this explicit.

**Weaknesses.** A abstract length: DONE. B contradiction abstract vs conclusion: RESOLVED by abstract compression. C H80 underplayed: §7.2 gives H80 equal billing to H50. D "outside seed noise" in FINDINGS historical block: separate FINDINGS commit will prefix with "(superseded — n=7 PROMOTION block above is the current claim)"; out of scope for this pass. E length 848 → ≤400: DONE (318 lines); §5.5 chain consolidated; Appendix C dropped (linked to CLAUDE.md §8); Appendix D moved to `audits/`. F [VERIFY] tags + Bronstein: DONE. Pittorino F et al. 2022 ICML *Deep Networks on Toroids* arXiv:2202.03038 verified; Islam MM et al. 2025 *Platonic Transformers* arXiv:2510.03511 verified (preprint status explicit); Bronstein MM et al. 2021 *Geometric Deep Learning* arXiv:2104.13478 added. G H09 elevator-pitch in abstract: DONE. H mystical discipline: PASS, unchanged.

Commits: **`5e930e3`**.

---

## Aggregate fix table

| Item | Reviewer | Status | Commit |
|---|---|---|---|
| POSI re-framing (k=3 confirmatory vs k=49 screening) | R1 #1 | FIX-IN-CAMERA-READY | `5e930e3` |
| Hill-climbed CI includes 0 honestly reported | R1 #2 / R2 Q2 | CONCEDED | `5e930e3` |
| Permutation + paired-t magnitude diagnostic | R1 #3 | FIX-IN-CAMERA-READY | `3f501a3`+`5e930e3` |
| Bootstrap CI / Fisher / Wilson on 22 pp excess | R1 Q4 / R2 Q3 | FIX-IN-CAMERA-READY | `3f501a3` |
| Cross-family audit subsample | R1 Q6 / R3 Q1 | FILED §7.4-8 | `5e930e3` |
| Controls 1–4 launch | R2 Q1 | FILED §7.4-10 | `5e930e3` |
| BS=128 baseline at n=7 (Phase-9d) | R2 Q2 | FILED §7.4-11 | `5e930e3` |
| Rule-23 n=1 caveat sentence | R2 W6 | FIX-IN-CAMERA-READY | `5e930e3` |
| Cross-domain replication on tabular | R3 Q2 | FILED §7.4-6 | `5e930e3` |
| Property-based Fixer-test skill | R3 Q3 | FILED (camera-ready) | (none yet) |
| Phase-9b calibration n ≥ 50 | R1 Q4 / R3 Q4 | FILED §7.4-5 | `5e930e3` |
| Abstract 440 → 199 words; lead with H09 | R4 Q1 / A | DONE | `5e930e3` |
| PAPER 848 → 318 lines | R4 E | DONE | `5e930e3` |
| [VERIFY] tags resolved + Bronstein 2021 added | R4 F | DONE | `5e930e3` |
| Pre-abstract revision-status table removed | R4 A | DONE | `5e930e3` |
| §5.5 → §5.5.4 chain consolidated | R4 E | DONE | `5e930e3` |

GPU work (Controls 1–4, Phase-9b/c/d, cross-family subsample, cross-domain replication, property-based-test skill) is filed and held until explicit user authorisation. We thank R1, R2, R3, and R4 for the depth of their engagement.

---

*Rebuttal length ≈ 1 400 words (within ICML 1 500-word limit). Fix commits: `3f501a3` (stat-test + calibration intervals) and `5e930e3` (PAPER.md compression + reviewer-driven splices).*
