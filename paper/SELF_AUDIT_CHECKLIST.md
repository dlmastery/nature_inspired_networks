# Self-audit checklist — before claiming "done" on any paper, dashboard, or finding

> Codifies the lessons of the 2026-05-29 reviewer-pass campaign
> (`audits/REVIEWER_PASS_PAPER.md`, `audits/REVIEWER_PASS_DASHBOARD.md`).
> A future Claude / operator MUST tick all 30 boxes before
> claiming an externally-facing artefact is finished. Rebut with
> evidence (commit SHA, Playwright probe output) — never with
> "should be fine" or "I checked it earlier."
>
> Linked rules in CLAUDE.md: 27–38. Linked skills:
> `autoresearch-typography-and-rendering`,
> `autoresearch-doc-organization`,
> `autoresearch-link-discipline`,
> `autoresearch-dashboard-comprehension`,
> `autoresearch-paper-rigor`, plus the augmented
> `autoresearch-critic-team`, `autoresearch-scicritic-team`,
> `autoresearch-fixer-campaign`, `autoresearch-dashboard`,
> `autoresearch-per-experiment-page`.

## How to use

For each row: tick `[x]` ONLY after the evidence column is
filled with a commit SHA, a Playwright probe output snippet, a
file path, or a 1-sentence justification. An empty evidence
column means the row is NOT ticked.

If any row is not satisfied, the artefact is NOT done — block the
"done" claim until the row clears.

---

## A. Markdown rendering + typography (Rules 29, 30)

- [ ] **1.** Every embedded markdown block (FINDINGS verdict,
  sci-critic, impl-critic, hypothesis digest, headline ribbon)
  pipes through the GFM-table + blockquote-aware converter.
  Evidence: ______
- [ ] **2.** Playwright probe `scripts/verify_markdown_rendering.py`
  has been run on at least 5 sampled pages (1 aggregate + 4
  per-experiment, including the headline winner pages) and
  reports ZERO literal `##`, `**`, `|---:|`, `> |`, `&gt;` leaks.
  Evidence: ______
- [ ] **3.** Body font is Source Serif 4 (or Source Serif Pro
  fallback); monospace is IBM Plex Mono; NO Newsreader, NO
  italic-as-emphasis-display. Same family on aggregate AND
  per-experiment pages. Evidence: ______
- [ ] **4.** Footer self-description (e.g., "Brutalist Editorial
  Lab Notebook ... font stack ...") matches the actual `<link>`
  href font stack — driven from the same constant. Evidence: ____

## B. Links + cross-references (Rules 27, 38)

- [ ] **5.** Playwright link-sweep `scripts/verify_links.py` has
  been run on every HTML file in `dashboard/` and `docs/`;
  reports ZERO broken (HTTP ≥ 400) links. Evidence: ______
- [ ] **6.** Every link from `docs/`-served HTML to a non-`docs/`
  repo file uses an absolute `https://github.com/dlmastery/.../blob/main/...`
  URL — NEVER a relative `../FINDINGS.md`. Evidence: ______
- [ ] **7.** First-mention linkification: every model
  (ResNet-20, RegNetX-200MF, ViT-Tiny), dataset (CIFAR-10/100,
  Spherical MNIST), technique (AdamW, SIREN, label smoothing,
  Holm-Bonferroni), arXiv ID, hypothesis ID (H09, H71), and
  CLAUDE.md rule number is a hyperlink on FIRST mention.
  Evidence: ______
- [ ] **8.** Audit ledger `audits/` is append-only: no
  `G<N>_audit.md` / `REVIEWER_PASS_*.md` has been overwritten;
  new passes are new dated files or appended dated sections.
  Evidence: ______

## C. Statistical rigor + pre-registration (Rules 28, 35, 36)

- [ ] **9.** Every "winner" / "outside seed noise" /
  "statistically significant" claim carries a paired Wilcoxon
  W and two-sided p, computed on the actual seed-paired
  differences. Evidence: ______ (cite `paper/STATISTICAL_TESTS.md`
  rows).
- [ ] **10.** Every reported pp delta carries a 95 % bootstrap CI
  (≥ 10,000 resamples), with the lower bound separating "claim"
  from "no claim." Evidence: ______
- [ ] **11.** Holm-Bonferroni correction is applied across the
  sweep family; α' values are listed alongside p values; the
  smallest p clears α'_1 = α / K. Evidence: ______
- [ ] **12.** Empirical noise band (σ_seed) per dataset is
  EMPIRICALLY DERIVED from the project's own multi-seed baseline
  runs — NOT a rule-of-thumb "±0.5 pp". Cited in
  `paper/STATISTICAL_TESTS.md`. Evidence: ______
- [ ] **13.** Every numeric on every visual surface (KN-strip,
  leaderboard cell, headline ribbon) carries `n=X` AND a
  `SCREENING` / `EVALUATION` chip (Rule 34). No bare numerics.
  Evidence: ______
- [ ] **14.** No empirical claim is made at n=3 without the
  `SCREENING` tag. EVALUATION claims use n ≥ 7 (3-hyp family at
  α=0.05 under Holm-Bonferroni). Evidence: ______
- [ ] **15.** The screening-vs-evaluation classification for
  every sweep row was PRE-REGISTERED before the sweep ran —
  commit SHA cited in the paper / FINDINGS entry. No post-hoc
  reclassification of losing rows as "screening." Evidence: ____
- [ ] **16.** Ordinal-margin (`min(leader) − max(baseline)`) and
  Δmean (`mean(leader) − mean(baseline)`) are reported AS
  DIFFERENT STATISTICS alongside each other for any "lead"
  claim. Evidence: ______
- [ ] **17.** Any hypothesis whose pre-registered falsifier
  specifies a dataset NOT in the sweep is recorded as
  `UNTESTED_ON_RIGHT_DATASET`, NOT NUMEROLOGY or FALSIFIED.
  Evidence: ______

## D. Internal consistency + framing (Rules 32, 37)

- [ ] **18.** The paper has been read end-to-end and the
  abstract's contribution statement matches the conclusion's
  contribution statement — no internal contradiction.
  Evidence: ______
- [ ] **19.** Material limitations (single-seed sweep rows,
  baseline-below-SOTA gap, untested NOVEL+TESTABLE hypothesis,
  auditor-self-grading circularity, CIFAR-as-wrong-testbed for
  several equivariance hypotheses) are stated IN THE ABSTRACT —
  not buried in §7. Evidence: ______
- [ ] **20.** No "ACCEPT" / "FINAL" / "Reviewer-acceptance"
  banner on any external surface (README, PAPER, dashboard,
  per-experiment page) without the explicit "Internal QA pass —
  independent external review pending" qualifier. Evidence: ____
- [ ] **21.** When an external reviewer's verdict (e.g.,
  `audits/REVIEWER_PASS_PAPER.md` WEAK_REJECT) downgrades a
  prior internal ACCEPT, the downgrade is reflected on every
  surface in the same commit that processes the audit.
  Evidence: ______
- [ ] **22.** Auditor-self-grading circularity (implementer +
  critic + sci-critic + fixer agents share a model family) is
  disclosed in the paper section that reports any audit-derived
  rate (e.g., "51 % non-PASS"). Calibration plan stated.
  Evidence: ______
- [ ] **23.** No marketing language ("revolutionary",
  "state-of-the-art", "novel framework") in README or PAPER.
  Calibrated positive AND calibrated negative receive equal
  visual prominence. Evidence: ______
- [ ] **24.** Section numbers are unique; no duplicated `### 5.5`
  / `## 6.1` headings. Evidence: ______

## E. Dashboard comprehension (Rules 33, 34)

- [ ] **25.** Every chart with ≥ 3 axes / lines / variants has
  been split into 3+ side-by-side small-multiples (per-group or
  per-tier panels). Evidence: ______
- [ ] **26.** Every chart has a 1-sentence "what to read"
  caption directly under it. Evidence: ______
- [ ] **27.** Every dashboard surface opens with EXACTLY 4
  bullets in the "How to read this dashboard" orientation block
  (what / colour code / screening-vs-evaluation / drill-down).
  Evidence: ______
- [ ] **28.** Multi-hypothesis tags (combo*/pair*/hybrid*)
  display ALL participating hypothesis pills — not just the
  leading H-ID. Evidence: ______

## F. Repo discipline (Rules 11, 31)

- [ ] **29.** Repo root contains ONLY {`README.md`, `CLAUDE.md`,
  `PAPER.md`, `LICENSE`} (plus the transitional
  `RESTRUCTURE_PLAN.md` if mid-restructure — DELETE after
  completion). Every other `.md` is in `paper/`, `docs/`,
  `experiments/`, `hypotheses/`, `audits/`, `skills/`, or
  `memory/`. Verified by `scripts/check_root_files.ps1`.
  Evidence: ______
- [ ] **30.** The auto-checkpoint loop ran throughout this
  campaign (Rule 11 + Rule 20); no progress block is longer than
  ~15 min between commits during active work; commits are
  scoped (not `-A`); push has succeeded for every commit. The
  campaign's most-expensive sweep would lose at most ONE run on
  power outage. Evidence: ______

---

## Failure-mode field guide (extension reading)

The following recurring failure modes have been caught and
codified — each links to its origin audit and the skill that
guards against recurrence:

| failure mode | origin | guard |
|---|---|---|
| Markdown literal `##` leaking to camera | `audits/REVIEWER_PASS_DASHBOARD.md` | Rule 29; `autoresearch-typography-and-rendering` |
| Newsreader italic-as-emphasis on aggregate | `audits/REVIEWER_PASS_DASHBOARD.md` | Rule 30; `autoresearch-typography-and-rendering` |
| 340+ broken `../FINDINGS.md` links from Pages | `audits/REVIEWER_PASS_DASHBOARD.md` | Rules 27 + 38; `autoresearch-link-discipline` |
| Repo root with 18+ `.md` files | cold-reader audit | Rule 31; `autoresearch-doc-organization` |
| README without quick start / negatives-buried | cold-reader audit | Rule 32; `autoresearch-doc-organization` |
| Dense single chart with 20 bars | `audits/REVIEWER_PASS_DASHBOARD.md` | Rule 33; `autoresearch-dashboard-comprehension` |
| KN-strip without `n=X` / tier chip | `audits/REVIEWER_PASS_DASHBOARD.md` | Rule 34; `autoresearch-dashboard-comprehension` |
| n=3 "winner" without Wilcoxon + CI + Bonferroni | `audits/REVIEWER_PASS_PAPER.md` §B | Rule 35; `autoresearch-paper-rigor` |
| Post-hoc "screening" reclassification of losers | `audits/REVIEWER_PASS_PAPER.md` §B BLOCKER | Rule 36; `autoresearch-paper-rigor` |
| NUMEROLOGY verdict on dataset-mismatched H | `audits/REVIEWER_PASS_PAPER.md` §D | Rules 36 + scicritic UNTESTED_ON_RIGHT_DATASET tier |
| "ACCEPT" self-grading banner | `audits/REVIEWER_PASS_PAPER.md` §E | Rule 37; `autoresearch-dashboard-comprehension` |
| Auditor-self-grading circularity not disclosed | `audits/REVIEWER_PASS_PAPER.md` §F | Rule 37; `autoresearch-critic-team` + `autoresearch-scicritic-team` |
| Internal contradiction (abstract vs conclusion) | `audits/REVIEWER_PASS_PAPER.md` §A | Rule 37; `autoresearch-paper-rigor` Pillar 4 |
| Material limitation buried in §7 | `audits/REVIEWER_PASS_PAPER.md` §A | Rule 37; `autoresearch-paper-rigor` Pillar 4 |
| "Fix" claimed without Playwright verification | `audits/REVIEWER_PASS_DASHBOARD.md` | Rule 29; `autoresearch-fixer-campaign` post-fix probe |
| Audit file overwritten with new pass | `audits/REVIEWER_PASS_PAPER.md` (impl) | Rule 38; `autoresearch-link-discipline` ledger discipline |
| Combo* pill showing only leading H-ID | `audits/REVIEWER_PASS_DASHBOARD.md` | Rule 33; `autoresearch-dashboard-comprehension` |
| Ordinal-margin reported AS Δmean | `audits/REVIEWER_PASS_PAPER.md` §E | Rule 36; `autoresearch-paper-rigor` Pillar 2 |
| Empirical noise band stated as "±0.5 pp" rule-of-thumb | `audits/REVIEWER_PASS_PAPER.md` §E | Rule 35; `autoresearch-paper-rigor` Pillar 1 |
| First-mention of ResNet/AdamW/CIFAR not linked | cold-reader audit | Rule 38; `autoresearch-link-discipline` Pillar 2 |

---

*Last updated: 2026-05-29. Rows 1–30 are MANDATORY. Add new rows
when a future audit surfaces a recurring failure mode not covered
above — this file is append-only (per Rule 38 audit-ledger
spirit) but extending the checklist is a normal evolution, not
a rewrite of past rows.*
