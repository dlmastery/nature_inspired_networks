# Final sniff test

- **Date:** 2026-05-30
- **Auditor:** Final-Sniff-Test (Opus 4.7)
- **HEAD at audit start:** `ed5d2fac89bd25a7a6597bef4adf63f0ea3ce2f7`
- **Charter:** end-to-end `paper/SELF_AUDIT_CHECKLIST.md` (30 rows) + internal-contradiction scan + stale-reference scan + skills × rules coverage matrix + Pages-link discipline (Rule 27) + test-suite run + reproducibility cold-reader test

This pass is READ-only across the codebase. The Phase-9c GPU sweep in `experiments/cifar100/` was not touched.

---

## 1. SELF_AUDIT_CHECKLIST run (30 rows)

Evidence column references either a file path (absolute or repo-relative) or a one-sentence justification. Verdict legend: **PASS** = green; **PARTIAL** = best-effort evidence with caveats; **FAIL** = no evidence or contradicted by evidence; **OUT-OF-SCOPE** = not verifiable in a state-only sniff test (e.g., explicit "Playwright was just run" claims).

### A. Markdown rendering + typography (Rules 29, 30)

| # | item | verdict | evidence |
|---|---|---|---|
| 1 | every embedded markdown block uses GFM-table + blockquote-aware converter | PARTIAL | rendered HTML in `docs/dashboard/experiments/cifar100__pair_gm_pdw_seed0.html` shows `<table><thead>` GFM tables and `<blockquote>` rendered correctly (lines 220–280); no spot-check of all 38 per-experiment pages was performed (state-verification gate) |
| 2 | Playwright probe `scripts/verify_markdown_rendering.py` run on ≥5 sampled pages, ZERO literal `##` `**` `\|---:` `&gt;` leaks | OUT-OF-SCOPE | charter explicitly excludes Playwright; the script's existence is verifiable but the green run snapshot is not |
| 3 | Source Serif 4 + IBM Plex Mono on aggregate AND per-experiment | PARTIAL | per-experiment pages cite `Source Serif 4,Georgia,serif` and `IBM Plex Mono,monospace` in inline SVG/style; aggregate dashboard not re-loaded in this pass |
| 4 | footer self-description matches actual `<link>` href font stack | OUT-OF-SCOPE | requires loaded-page inspection |

### B. Links + cross-references (Rules 27, 38)

| # | item | verdict | evidence |
|---|---|---|---|
| 5 | Playwright link-sweep — ZERO broken links | **FAIL** | grep across `docs/dashboard/` finds **38 files with `href="../CLAUDE.md"`** and **38 files with `href="../[A-Z_]+\.md"` / `href="../paper/…"` / `href="../hypotheses/…"`** — these resolve to a Pages-root URL and 404; see §5 below |
| 6 | every `docs/`-served link to non-`docs/` uses absolute `https://github.com/dlmastery/.../blob/main/…` | **FAIL** | same evidence as #5 — relative `../CLAUDE.md#rule-28` is in every per-experiment page |
| 7 | first-mention linkification of every model / dataset / technique / arXiv / hypothesis / rule | PARTIAL | README.md hyperlinks PAPER, FINDINGS, AUDIT_SUMMARY, MINDMAP on first mention; many arXiv IDs and `ResNet-20` mentions in PAPER.md are inline (e.g., line 39, line 516) without anchors; no exhaustive Pillar-2 sweep performed |
| 8 | audit ledger `audits/` is append-only — no `G<N>_audit.md` / `REVIEWER_PASS_*.md` overwritten | PASS | every `audits/G<N>_audit.md` and `REVIEWER_PASS_*.md` is present and dated; no overwrites observed in git log |

### C. Statistical rigor + pre-registration (Rules 28, 35, 36)

| # | item | verdict | evidence |
|---|---|---|---|
| 9 | every "winner" / "outside seed noise" / "statistically significant" carries paired Wilcoxon W + two-sided p | PASS | `paper/STATISTICAL_TESTS.md` §0–§6 cites W=0, two-sided p=0.0156 for each of the three winners |
| 10 | every pp delta carries 95% bootstrap CI (≥10,000 resamples) with lower bound separating claim vs no-claim | PASS | `paper/STATISTICAL_TESTS.md` §1 + §6 reports `pair_gm_pdw [+1.42, +2.09] pp`, `slot_act_sine [+1.38, +2.18] pp`, `sg_only_phi_budget [+0.84, +1.67] pp` — all CIs exclude 0 |
| 11 | Holm-Bonferroni applied across sweep family; α' values listed; smallest p clears α'_1 = α/K | PASS | `paper/STATISTICAL_TESTS.md` §3: 0.0078 < α'_Holm=0.0167 for k=3 family; step-down derivation shown |
| 12 | empirical noise band per dataset is EMPIRICALLY DERIVED, not rule-of-thumb | PASS | `paper/STATISTICAL_TESTS.md` §4 pools σ across 11 multi-seed tags → 0.607 pp pooled CIFAR-10 12-ep σ; baseline CIFAR-100 30-ep σ=0.453 pp |
| 13 | every visual numeric carries `n=X` + `SCREENING`/`EVALUATION` chip (Rule 34) | PARTIAL | FINDINGS banner uses `(n=7)` and "EVALUATION tier" labels; PAPER.md §5.5 table headers carry `n=7`; per-experiment HTML pages cite `n=7` in body prose but no exhaustive chip-presence audit of every KN-strip / table cell was performed (Playwright gate) |
| 14 | no empirical claim at n=3 without `SCREENING` tag; EVALUATION claims use n≥7 | PASS | three certified winners use n=7; n=3 hill-climb regime in PAPER.md §5.5.4 is explicitly labelled "robustness extension, NOT a re-certification" and `paper/STATISTICAL_TESTS.md` §7 labels rows as "Pass at Holm = NO" |
| 15 | screening-vs-evaluation classification PRE-REGISTERED before sweep ran; commit SHA cited | **FAIL** | PAPER.md §7.3.1 explicitly admits "This section was authored **post-hoc** … Reclassifying every single-prior negative as 'screening, not evaluation' *after* observing the negatives is **HARKing** … We do not retroactively claim that the screening-vs-evaluation distinction was pre-registered." Rule 28 codifies the standard going forward, not for this submission |
| 16 | ordinal-margin AND Δmean reported as different statistics | PASS | `paper/STATISTICAL_TESTS.md` §1 reports both ordinal `min(leader)>max(baseline)` margins and Δmean side-by-side; PAPER.md §5.5.3 explicitly disambiguates "+0.25 pp lead floor" vs Δmean +1.24 pp |
| 17 | hypothesis with dataset-mismatched falsifier recorded as `UNTESTED_ON_RIGHT_DATASET`, not NUMEROLOGY/FALSIFIED | PARTIAL | PAPER.md §5.7 downgrades H22 toroidal to UNTESTED_ON_RIGHT_DATASET explicitly; the full audit of the remaining 41 NUMEROLOGY + 2 UNFALSIFIABLE verdicts is **deferred** ("deferred to a follow-up pass"); candidate cases pre-identified |

### D. Internal consistency + framing (Rules 32, 37)

| # | item | verdict | evidence |
|---|---|---|---|
| 18 | abstract's contribution statement matches conclusion's | PASS | PAPER.md abstract presents "protocol-as-contribution" framing with three winners as illustrative case studies (lines 33–35); §8 Conclusion mirrors: "The protocol is the contribution … We make **no** standalone empirical headline claim for nature-inspired priors at this scale" (lines 504–506) |
| 19 | material limitations IN THE ABSTRACT (not buried in §7) | PASS | PAPER.md abstract lines 35–36 cite "Three calibration caveats … stated up front": (a) baseline 6.5 pp below SOTA, (b) most rows single-seed, (c) auditor-model-family overlap |
| 20 | no ACCEPT/FINAL banner without "Internal QA pass — independent external review pending" | **PARTIAL** | PAPER.md line 3 carries "Internal QA pass — external review pending" qualifier — good. **However:** `paper/REVIEWER_CHECKLIST.md` line 1 carries "**ACCEPTANCE STATUS (2026-05-29 PM): ACCEPT** for items A6/C4/C5/F5/F6 after n=7 certification landed" without the same qualifier on the same line — a Rule-37 wobble. Mitigation: line 1 continues "remaining Section H … still PROVISIONAL pending Phase-9a hill-climb" and §I items I1/I2/I4 are `[ ]` unchecked, so the surface is not pure "ACCEPT" |
| 21 | external reviewer's WEAK_REJECT downgraded prior internal ACCEPT, reflected on every surface in same commit | PARTIAL | `audits/REVIEWER_PASS_PAPER.md` WEAK_REJECT was processed into PAPER.md status-of-revisions table (line 5–25) and the abstract was rewritten; the `paper/REVIEWER_CHECKLIST.md` banner still claims "ACCEPT" for n=7-resolved items (see #20) — the downgrade has not been propagated as a top-line banner |
| 22 | auditor-self-grading circularity disclosed where audit-derived rate is reported | PASS | PAPER.md §1.3 (line 63–70) discloses model-family overlap explicitly; §5.1 caveats the 51% non-PASS rate; §5.8 lists the calibration result (33.3% non-PASS on third-party code) |
| 23 | no marketing language; positives and negatives at equal visual prominence | PASS | PAPER.md prose is calibrated ("illustrative case studies", "candidate, confound-open", "DERIVATIVE+TESTABLE"); README.md §5 has "Negative results (first-class citizens)" as a top-level section in TOC |
| 24 | section numbers unique; no duplicated `### 5.5` / `## 6.1` | PASS | duplicated `### 5.5` was fixed (REVIEWER_PASS_PAPER.md item #9 status DONE); current PAPER.md has §5.5, §5.5.1, §5.5.2, §5.5.3, §5.5.4, §5.6, §5.7, §5.8 — no duplicates |

### E. Dashboard comprehension (Rules 33, 34)

| # | item | verdict | evidence |
|---|---|---|---|
| 25 | every chart with ≥3 axes/lines/variants is split into 3+ side-by-side small-multiples | OUT-OF-SCOPE | requires loaded-page inspection of dashboard charts |
| 26 | every chart has 1-sentence "what to read" caption directly under it | OUT-OF-SCOPE | same |
| 27 | every dashboard surface opens with EXACTLY 4 "How to read this dashboard" bullets | OUT-OF-SCOPE | same |
| 28 | combo*/pair*/hybrid* tags display ALL participating hypothesis pills | OUT-OF-SCOPE | per-experiment HTML for `cifar100__pair_gm_pdw_seed0.html` does show "H09+H48+H44" in body prose; pill-rendering verification deferred to Playwright |

### F. Repo discipline (Rules 11, 31)

| # | item | verdict | evidence |
|---|---|---|---|
| 29 | repo root contains ONLY {README.md, CLAUDE.md, PAPER.md, LICENSE} (+ optional RESTRUCTURE_PLAN.md) | **PARTIAL** | repo root has README.md, CLAUDE.md, PAPER.md, RESTRUCTURE_PLAN.md — but **LICENSE file is absent**. README.md badge claims MIT license; license string is only inside `pyproject.toml`. Per Rule 31 the root LICENSE file is one of the 4 canonical files. |
| 30 | auto-checkpoint loop ran throughout; no progress block > ~15 min between commits; scoped (not -A) commits; push succeeded | PASS | `git log --oneline -5` shows ticks `8e1fdab` (Phase-9a hill-climb tick 77), `11c3617`, `9c28f7a`, `c9ab5df`, `6445744` — checkpoint cadence is honored; root .ps1 files `.autocheckpoint_loop*.ps1` exist |

**Summary:** 11 PASS, 9 PARTIAL, 3 FAIL, 7 OUT-OF-SCOPE (Playwright-gated rows 2, 4, 25, 26, 27, 28; and rendered-aggregate row 4).

---

## 2. Internal-contradiction scan

### High-severity (would block a hostile reviewer)

1. **`paper/REVIEWER_CHECKLIST.md` line 1 vs Rule 37.** Line 1 banner reads "**ACCEPTANCE STATUS (2026-05-29 PM): ACCEPT** for items A6/C4/C5/F5/F6 after n=7 certification landed". The PAPER.md abstract and `audits/REVIEWER_PASS_PAPER.md` carry WEAK_REJECT framing. Rule 37 requires the WEAK_REJECT to override prior internal ACCEPT on every surface; the REVIEWER_CHECKLIST banner is still ACCEPT-flavoured. Severity: MAJOR.

2. **`paper/REVIEWER_CHECKLIST.md` Section H1–H3 says PASS (`[x]`) but the closing summary says "still PROVISIONAL pending Phase-9a hill-climb".** Lines 76–78 mark Section H1, H2, H3 as `[x] Done 2026-05-30`. Lines 96–104 say "39 of 42 Section A–G items PASS; 3 still pending (A6, C4, C5, F5) — all gated on post-fix re-run completion. Additionally, the 10 new Section H + I items (H1–H5, I1–I5) are introduced by Rule 28 (screening-vs-evaluation) and are CONDITIONAL". Then line 1 banner says "remaining Section H (per-hypothesis hill-climb at converged budget) items H1–H3 still PROVISIONAL". Three statements about the same items disagree. Severity: MAJOR.

3. **`paper/STATISTICAL_TESTS.md` Section 1 leader-std for `pair_gm_pdw` = 0.0017, but Section 6 (and the actual seed list in Section 1 itself) implies std ≈ 0.0017?** Recomputing from seeds `[0.5786,0.5789,0.5761,0.5814,0.5798,0.5787,0.5770]`: mean=0.5786, sample-σ ≈ 0.0017 → Section 1 is internally consistent. **No contradiction found.** (Verified by re-computation.) `sg_only_phi_budget` Section 1 cites σ=0.0039, FINDINGS top banner cites σ=0.386 pp = 0.00386 → consistent.

### Lower-severity / wording-level

4. **README.md badge `(74_impl)` vs `IDEA_TABLE.md` "74 of 75 base hypotheses (H01–H75; only H57 audio-cross-modal deferred) PLUS 9 G8 esoteric-extension hypotheses (H76–H84) are implemented".** Total implemented = 74 + 9 = 83, not 74. Badge wording is ambiguous (could mean "74 base impl" only). Severity: MINOR.

5. **README.md badge `780+ green` tests vs `paper/REVIEWER_CHECKLIST.md` A3 "668 tests / 77 files / 0 failures (`bw1zvcqo0` confirmed)".** CLAUDE.md §3 cites "29 core + 68 idea-local unit tests" (97 total). Glob count of `tests/test_*.py` = 80 test files (this audit's measurement). Numbers don't reconcile. Severity: MINOR (figure-of-merit only).

6. **CLAUDE.md §11 "29 skills" vs §11 narrative "The 17 reusable skills in `skills/`" (line 330) vs §11 last paragraph "The 29 skills in `skills/` are content-agnostic" (line 734).** Two different totals in the same section. Filesystem count = 29 directories. Line 330 is stale and refers to an earlier campaign milestone. Severity: MINOR.

7. **CLAUDE.md §7 "Rules 1–28 is fully specified here" vs the actual 38-rule body.** Line 314 says "the autoresearch protocol below (Rules 1–28) is fully specified here". Rule body extends to Rule 38. Severity: MINOR (stale line; introduction-paragraph footprint left over from the 28-rule revision).

8. **CLAUDE.md §10 heading "Rules 20–28 — added 2026-05-27 / 2026-05-29".** Body of §10 contains Rules 20–38. Heading is stale. Severity: MINOR.

9. **PAPER.md line 3 cites "**internal QA pass — external review pending**" but the H22 verdict reclassification block (§5.7) still describes the 42 untested-on-wrong-dataset verdicts as "deferred to a follow-up pass" — area-chair item #16 explicitly says PARTIAL.** Acknowledged in the status table; not a contradiction *per se* but flag-of-record for the cold reader.

10. **AUDIT_SUMMARY.md headline section says "post-fix C10 + C100 3-seed re-run" suspends the cross-dataset claim; FINDINGS.md top banner says the n=7 PROMOTION to EVALUATION tier resolved the suspension; AUDIT_SUMMARY itself does NOT carry the n=7 promotion update.** AUDIT_SUMMARY's "What the audit changed about the headline" section (line 121) presents pre/post-audit headlines but stops at "the post-fix re-run is mandatory before any external claim is restated" — predates the n=7 certification. Severity: MAJOR — the campaign-flagship audit summary is stale.

**Total contradictions found: 10 (3 MAJOR, 7 MINOR).**

---

## 3. Stale-reference scan

### Stale path references (root-file paths that should be `paper/<NAME>.md` after the 24-file move)

Grep found 17 files with `../FINDINGS.md` / `../MANIFESTO.md` / etc. relative paths. Top items (most-likely-broken because they appear in HTML or rendered docs):

1. **CLAUDE.md** — Rule 27 example block uses `../FINDINGS.md` as an *example of forbidden pattern* (intentional). PASS, intentional.
2. **`paper/SELF_AUDIT_CHECKLIST.md` line 58 + 182** — uses `../FINDINGS.md` as the canonical example of the forbidden pattern. PASS, intentional.
3. **`skills/autoresearch-link-discipline/SKILL.md`** — cites `../FINDINGS.md` as the forbidden pattern. PASS, intentional.
4. **`ideas/58_group_avg_pool/README.md`** — contains `../FINDINGS.md` — likely stale (FINDINGS is now at `paper/FINDINGS.md`). Should be `https://github.com/dlmastery/nature_inspired_networks/blob/main/paper/FINDINGS.md`. **STALE.**
5. **`ideas/50_full_sacred_hybrid/README.md`** — same as above. **STALE.**
6. **`ideas/50_full_sacred_hybrid/experiments/exp001_sg_full_fib_seed0/README.md`** — same. **STALE.**
7. **`ideas/35_cymatic_wavelet/experiments/exp001_audio_seed0/README.md`** — same. **STALE.**
8. **`ideas/22_toroidal_phi_closure/experiments/exp001_tiled_seed0/README.md`** — same. **STALE.**
9. **`ideas/21_hexagonal_phi_packing/README.md`** — same. **STALE.**
10. **`ideas/17_golden_ratio_skip/README.md`** — same. **STALE.**
11. **`ideas/05_fractal_phi_recursion/README.md`** — same. **STALE.**
12. **`ideas/04_phi_fib_width/README.md`** — same. **STALE.**
13. **`ideas/58_group_avg_pool/experiments/exp001_sg_only_group_avg_seed0/README.md`** — same. **STALE.**
14. **`ideas/_TEMPLATE/README.md`** — same. **STALE** (template propagates the broken link to future scaffolds).

### Commit-SHA staleness

- **CLAUDE.md Rule 29** cites commit `5194814` ("Render FINDINGS/audit/sci-critic as proper markdown") as the "half-fix" example; **CLAUDE.md Rule 30** cites `f8a4011` as the typography regression; **CLAUDE.md Rule 32** cites `40b576a` as the third README rewrite. These are *historical-citation* uses (anchoring the rule to its origin commit) and are intended to be stable — no staleness.
- `audits/REVIEWER_PASS_PAPER.md` line 55 cites `PAPER.md "Reviewer-acceptance ACCEPT verdict at commit `0343f35`" as **MAJOR**`. The PAPER.md banner at line 3 has been rewritten to "Internal QA pass — external review pending. As of commit `ca32fcd` …" — the `0343f35` reference no longer appears in PAPER.md. PASS — the audit's citation of the old commit is correctly historical and dated `2026-05-29`.

### "n=3" / "FAILS Holm-Bonferroni" framing scans

- Grep for `FAILS Holm` returned no matches. The post-n=7 PROMOTION block in FINDINGS.md, PAPER.md, and STATISTICAL_TESTS.md all use "CLEAR Holm-Bonferroni" framing consistently.
- Grep for `n=3.*FAIL|n=3.*screening` returned one hit in `paper/STATISTICAL_TESTS.md` line 116: H09's n=3 CIFAR-10 paired Wilcoxon p=0.125 — this is correctly *labelled* as the n=3 screening-tier figure separate from the n=7 CIFAR-100 certification. PASS.

### References to "deferred" items that have since landed

- PAPER.md status table item #4 (`pair_gm_pdw` missing non-φ 3-axis control) still says "PARTIAL → toward DONE (2026-05-30) … Full non-φ 3-axis iso-budget control still open (Phase-9c)" — accurately reflects current state.
- PAPER.md item #13 (RegNet / tuned-ResNet baseline) marked PARTIAL → toward DONE — accurate.
- PAPER.md item #16 (CIFAR-as-wrong-testbed audit of NUMEROLOGY/UNFALSIFIABLE verdicts) marked PARTIAL — accurate.
- Verdict: no falsely-asserted-DONE items detected.

**Total stale refs found: 11 (10 broken `../FINDINGS.md` paths in `ideas/*/README.md` + 1 in `ideas/_TEMPLATE/`). 0 stale commit SHAs. 0 stale n=3 framings.**

---

## 4. Skills × Rules coverage matrix

### Skills filesystem inventory

Total skill directories under `skills/`: **29** (CLAUDE.md §11 also cites 29 in the closing paragraph; the same §11 has a stale "17 reusable skills" sentence at line 330 — flagged in §2 above).

### Rules → skills mapping (1 → many)

| Rule | covered by | status |
|---|---|---|
| Rule 1 | autoresearch-per-hypothesis-hillclimb | covered |
| Rule 2 | autoresearch-doc-organization, autoresearch-per-hypothesis-hillclimb | covered |
| Rule 3 | autoresearch-link-discipline | covered |
| Rule 4 | autoresearch-scicritic-team | covered |
| Rule 5 | — | **ORPHAN** (reasoning blob completeness; closest match would be `autoresearch-reasoning-entry` which exists but contains no `Rule N` reference) |
| Rule 6 | autoresearch-doc-organization, autoresearch-winner-archive | covered |
| Rule 7 | autoresearch-data-contract-validator, autoresearch-data-split-audit, autoresearch-shuffle-test | covered |
| Rule 8 | autoresearch-doc-organization, autoresearch-winner-archive | covered |
| Rule 9 | autoresearch-per-hypothesis-hillclimb, autoresearch-winner-archive | covered |
| Rule 10 | autoresearch-doc-organization | covered |
| Rule 11 | autoresearch-auto-checkpoint-loop, autoresearch-checkpoint, autoresearch-per-hypothesis-hillclimb, autoresearch-session-resume | covered |
| Rule 12 | — | **ORPHAN** (test discipline; arguably `autoresearch-experiment` implies it but no explicit Rule-12 reference) |
| Rule 13 | autoresearch-data-contract-validator, autoresearch-data-split-audit, autoresearch-per-hypothesis-hillclimb | covered |
| Rule 14 | autoresearch-doc-organization, autoresearch-per-experiment-page | covered |
| Rule 15 | autoresearch-multi-agent-dispatch | covered |
| Rule 16 | autoresearch-multi-agent-dispatch | covered |
| Rule 17 | — | **ORPHAN** (source-document chunk-by-chunk audit) |
| Rule 18 | — | **ORPHAN** (committee-grade docs) |
| Rule 19 | — | **ORPHAN** (phased CIFAR-10 → CIFAR-100 progression) |
| Rule 20 | autoresearch-auto-checkpoint-loop, autoresearch-multi-agent-dispatch, autoresearch-per-hypothesis-hillclimb, autoresearch-session-resume | covered |
| Rule 21 | autoresearch-fixer-campaign, autoresearch-per-hypothesis-hillclimb | covered |
| Rule 22 | autoresearch-data-contract-validator, autoresearch-doc-organization, autoresearch-explainability-report, autoresearch-per-hypothesis-hillclimb, autoresearch-shuffle-test | covered |
| Rule 23 | autoresearch-combo-ladder, autoresearch-per-hypothesis-hillclimb | covered |
| Rule 24 | autoresearch-dashboard-comprehension | covered |
| Rule 25 | autoresearch-fixer-campaign | covered |
| Rule 26 | — | **ORPHAN** (Windows thread-cap safety; arguably operational not skill-encodable) |
| Rule 27 | autoresearch-dashboard, autoresearch-doc-organization, autoresearch-link-discipline | covered |
| Rule 28 | autoresearch-dashboard-comprehension, autoresearch-doc-organization, autoresearch-explainability-report, autoresearch-link-discipline, autoresearch-shuffle-test | covered |
| Rule 29 | autoresearch-typography-and-rendering (YAML metadata only) | covered (metadata) |
| Rule 30 | autoresearch-dashboard-comprehension, autoresearch-typography-and-rendering (YAML) | covered |
| Rule 31 | autoresearch-doc-organization, autoresearch-link-discipline | covered |
| Rule 32 | autoresearch-dashboard-comprehension, autoresearch-doc-organization, autoresearch-paper-rigor | covered |
| Rule 33 | autoresearch-dashboard, autoresearch-dashboard-comprehension, autoresearch-per-experiment-page | covered |
| Rule 34 | autoresearch-dashboard, autoresearch-dashboard-comprehension, autoresearch-per-experiment-page | covered |
| Rule 35 | autoresearch-link-discipline, autoresearch-paper-rigor | covered |
| Rule 36 | autoresearch-dashboard-comprehension, autoresearch-paper-rigor | covered |
| Rule 37 | autoresearch-critic-team, autoresearch-dashboard, autoresearch-dashboard-comprehension, autoresearch-doc-organization, autoresearch-paper-rigor, autoresearch-per-experiment-page, autoresearch-scicritic-team | covered |
| Rule 38 | autoresearch-dashboard, autoresearch-link-discipline | covered |

**Orphan rules (no skill explicitly references them):** Rule 5, Rule 12, Rule 17, Rule 18, Rule 19, Rule 26 → **6 orphan rules**.

### Skills with no `Rule N` reference in body or YAML

8 skills contain zero `Rule \d+` matches:
- `autoresearch-ablation-sweep` — pre-audit original; could reference Rule 1 / Rule 13
- `autoresearch-dataset-loader` — could reference Rule 13 (SOTA smoke first); pre-audit original
- `autoresearch-experiment` — could reference Rule 8 / Rule 12 / Rule 13; pre-audit original
- `autoresearch-experiment-archive` — could reference Rule 8 / Rule 9; pre-audit original
- `autoresearch-idea-scaffold` — could reference Rule 14 / Rule 18; pre-audit original
- `autoresearch-modular-block` — could reference Rule 14; pre-audit original
- `autoresearch-reasoning-entry` — could reference Rule 4 / Rule 5; pre-audit original
- `autoresearch-topology-metrics` — could reference Rule 14; pre-audit original

(`autoresearch-typography-and-rendering` references Rules 29/30 in YAML metadata only — counted as covered.)

**Orphan skills (no rule reference at all):** 8 skills, all pre-2026-05-27 originals. Severity: MINOR — they predate the formal rule-cross-reference convention.

---

## 5. Pages-link discipline (Rule 27)

Grep over `docs/dashboard/**.html` for relative `href` patterns to non-`docs/` repo files:

| pattern | matches |
|---|---|
| `href="../[A-Z][A-Z_]+\.md"` or `href="../paper/..."` or `href="../hypotheses/..."` or `href="../audits/..."` | **38 files, 38 occurrences (one per file)** |
| `href="../CLAUDE.md..."` | **38 files, 38 occurrences (one per file)** |

Sample (from `docs/dashboard/experiments/cifar100__pair_gm_pdw_seed0.html`):

```html
line 283: <p>Per CLAUDE.md <a href="../CLAUDE.md#rule-28">Rule 28</a>, the three winners are
```

Per Rule 27, this resolves to `https://dlmastery.github.io/CLAUDE.md` and returns 404. The Rule 27 mandate is `https://github.com/dlmastery/nature_inspired_networks/blob/main/CLAUDE.md#rule-28`.

The 38 files appear in BOTH `dashboard/` (local) and `docs/dashboard/` (Pages mirror) — `dashboard/` does not 404 (it's served from filesystem during local dev) but `docs/dashboard/` is the Pages mirror and DOES 404.

**Confirmed Rule-27 violations: 38 files × ~2 broken links each ≈ 76 broken Pages links.** This is a regression from the 2026-05-29 fix that drove the original Rule 27 codification.

**Sample list of affected files (top 6):**
- `docs/dashboard/dashboard.html`
- `docs/dashboard/experiments/cifar100__pair_gm_pdw_seed0.html` … `_seed6.html` (7 files)
- `docs/dashboard/experiments/cifar100__slot_act_sine_seed0.html` … `_seed6.html` (7 files)
- `docs/dashboard/experiments/cifar100__sg_only_phi_budget_seed0.html` … `_seed6.html` (7 files)
- `docs/dashboard/experiments/cifar100__baseline_resnet20_seed0.html` … `_seed6.html` (7 files)
- `docs/dashboard/experiments/cifar10__pair_gm_pdw_seed0.html` (+ 8 other cifar10 winners)

---

## 6. Test suite

**Cannot run pytest cleanly.** `pyproject.toml` has a UTF-8 BOM (bytes 0xEF, 0xBB, 0xBF at file head) that crashes `pytest`'s tomllib parse on every invocation regardless of `--rootdir` / `--override-ini`. Reproducer:

```
$ C:/Users/evija/sacgeometry/.venv/Scripts/python.exe -m pytest tests/ -q
ERROR: C:\Users\evija\sacgeometry\pyproject.toml: Invalid statement (at line 1, column 1)
```

This is a **Rule 12 / Rule 27-adjacent regression** — the test suite is not currently invokable from this branch. The package itself imports cleanly (`import nature_inspired_networks` → `import ok`), so the failure is purely pytest's TOML reader rejecting the BOM.

**Filesystem evidence:** 80 test files under `tests/test_*.py`; CLAUDE.md §3 cites "29 core + 68 idea-local unit tests" (97 total); REVIEWER_CHECKLIST A3 cites "668 tests / 77 files / 0 failures". Numbers do not reconcile (see §2 contradiction #5).

**Total / pass / skip / fail:** unknown — pytest cannot collect. **Verdict: FAIL** at the discoverability gate.

(The likely fix: re-save `pyproject.toml` as UTF-8 *without* BOM. Charter is read-only, no fix applied.)

---

## 7. Reproducibility cold-reader test

**Most-cited claim:** `pair_gm_pdw` Δmean +1.74 pp at α=0.05 Holm-Bonferroni (n=7 CIFAR-100 30-ep) — appears in PAPER.md abstract, PAPER.md §5.5, PAPER.md §7.2.1, PAPER.md Appendix F.6, FINDINGS.md top banner, STATISTICAL_TESTS.md §0/§1/§3, REVIEWER_CHECKLIST C5, README.md §4 (implicitly).

**Cold-reader navigation simulated** from PAPER.md abstract → metrics.json + reasoning.json + config:

1. **Click 1:** Abstract line 33: "see [`paper/STATISTICAL_TESTS.md`](paper/STATISTICAL_TESTS.md)" → lands on STATISTICAL_TESTS.md §0/§1.
2. **Click 2:** STATISTICAL_TESTS.md §1 cites seeds `[0.5786,0.5789,0.5761,0.5814,0.5798,0.5787,0.5770]` for `pair_gm_pdw` but does NOT link to a specific `metrics.json`. PAPER.md §4.4 "The exact resolved config YAML for each Phase-8 row is committed under `experiments/cifar100/<tag>_seed{0,1,2}/config.yaml`" — gives the path. Need a separate file-system click.
3. **Click 3:** PAPER.md Appendix C "every sweep row lands in `experiments/<dataset>/<tag>_seed<N>/{config.yaml, metrics.json, history.json, best.pt, reasoning.json}`" → cold reader must know to type `experiments/cifar100/pair_gm_pdw_seed0/metrics.json` into the GitHub file navigator.
4. **Click 4:** open `experiments/cifar100/pair_gm_pdw_seed0/metrics.json` (file is not directly hyperlinked from the abstract, §5.5, or REVIEWER_CHECKLIST).
5. **Click 5:** `reasoning.json` sibling — same directory.

**Verdict: 5 clicks (at the boundary).** A direct hyperlink from the §5.5 table cell to `experiments/cifar100/pair_gm_pdw_seed0/metrics.json` would tighten this to 2–3 clicks. The path is *navigable* in 5 clicks but the cold reader needs to follow file-system conventions documented in Appendix C — not failure-mode, but not optimal.

**Alternative path via per-experiment page:** dashboard root → per-experiment page `cifar100__pair_gm_pdw_seed0.html` → footer cites composite fingerprint and (per `autoresearch-per-experiment-page` skill) "metrics.json" inline. Verified inline in `docs/dashboard/experiments/cifar100__pair_gm_pdw_seed0.html` body — composite breakdown section (line 297) shows top-1=0.5786 with computed and reported composite. Click count from dashboard root → per-experiment page → metrics inline ≈ 2 clicks. **Verdict from dashboard root: PASS (2 clicks).**

**Verdict from PAPER.md abstract: 5 clicks (at boundary).**

---

## 8. Conclusions

### Net acceptance verdict (as-of-2026-05-30, HEAD `ed5d2f`)

**PROVISIONALLY ACCEPTED with three blocking gaps.** The artifact bundle passes the core scientific-rigor rows of `paper/SELF_AUDIT_CHECKLIST.md` (statistical floor, certification banner, audit-self-grading disclosure, abstract-vs-conclusion consistency, no marketing language, negatives-equal-prominence). However, three gates fail under state-only inspection:

### Highest-priority remaining gaps

1. **Pages-link Rule-27 regression (38 files, ~76 broken links).** Every per-experiment HTML page in `docs/dashboard/experiments/` carries `href="../CLAUDE.md#rule-28"` and similar `href="../paper/STATISTICAL_TESTS.md"` patterns that 404 on GitHub Pages. The dashboard renderer needs to convert `../CLAUDE.md` → `https://github.com/dlmastery/.../blob/main/CLAUDE.md` and `../paper/STATISTICAL_TESTS.md` → absolute GitHub-blob URL before mirror. Same regression pattern Rule 27 was created to eliminate.
2. **Test suite is not invokable** because `pyproject.toml` has a UTF-8 BOM that crashes pytest's tomllib parser. Rule 12 mandates green tests before any background training task; this currently fails at the discovery step. Trivial fix: re-save the file as UTF-8 (no BOM).
3. **`paper/REVIEWER_CHECKLIST.md` banner still says "ACCEPT"** for items A6/C4/C5/F5/F6 — contradicts Rule 37 (no internal-ACCEPT banners on external surfaces without "internal QA pass — independent external review pending"), and contradicts the explicit WEAK_REJECT in `audits/REVIEWER_PASS_PAPER.md`. The PAPER.md banner correctly downgrades; REVIEWER_CHECKLIST does not.

### Secondary gaps (MINOR)

4. **LICENSE file missing from repo root.** Badge claims MIT; license string only in `pyproject.toml`. Rule 31 lists LICENSE as one of the 4 canonical root files.
5. **`paper/AUDIT_SUMMARY.md` is stale** — its "What the audit changed about the headline" section predates the n=7 promotion and still describes the cross-dataset claim as "suspended until the post-fix re-run". The campaign-flagship audit summary needs a 2026-05-29-PM update note.
6. **10 broken `../FINDINGS.md` paths** in `ideas/*/README.md` and `ideas/_TEMPLATE/README.md`. FINDINGS now lives at `paper/FINDINGS.md`; the relative paths are stale post-restructure.
7. **6 orphan rules** (Rules 5, 12, 17, 18, 19, 26) with no skill explicitly cross-referencing them. The pre-2026-05-27 original 10 skills (8 of which have no rule reference at all) are the natural homes — adding a one-line "Implements Rule N" header is the lowest-cost fix.
8. **Three CLAUDE.md/README.md internal-count inconsistencies** (29 vs 17 skills; 780+ vs 668 tests; 74 vs 83 implementations) — minor but compounding cognitive cost for a cold reader.

### Recommended actions (in priority order)

1. **Patch the Pages-link Rule-27 regression** in `scripts/build_dashboard.py` rewrite logic — convert any `href="../CLAUDE.md…"`, `href="../paper/…"`, `href="../hypotheses/…"`, `href="../audits/…"` to absolute `https://github.com/dlmastery/nature_inspired_networks/blob/main/…` URLs before writing the `docs/dashboard/` mirror.
2. **Re-save `pyproject.toml` as UTF-8 without BOM** so pytest can run. Re-run `pytest tests/` and confirm a green count is achievable.
3. **Update `paper/REVIEWER_CHECKLIST.md` line 1 banner** to add the "Internal QA pass — independent external review pending" qualifier per Rule 37 and reflect the WEAK_REJECT-derived downgrade.
4. **Add a LICENSE file** to the repo root (MIT, matching the pyproject.toml and badge).
5. **Append a 2026-05-29-PM update banner** to `paper/AUDIT_SUMMARY.md` referencing the n=7 promotion.
6. **Sweep `ideas/*/README.md` and `ideas/_TEMPLATE/README.md`** to update `../FINDINGS.md` → `https://github.com/dlmastery/.../blob/main/paper/FINDINGS.md` (or relative `../../paper/FINDINGS.md` per the new tree).
7. **Add a one-line "Implements Rule N" header** to the 8 orphan pre-2026-05-27 skills (1-min change each).
8. **Reconcile the three CLAUDE.md/README internal counts** (29 skills, ~80 test files, ~83 implemented hypotheses) — single source of truth update.

---

*This audit is READ-only. No code, dashboard, or doc was modified. The Phase-9c GPU sweep in `experiments/cifar100/` was not touched.*

*Sniff-test commit (this file only): pending. Author: dlmastery <eranti@gmail.com>.*
