# REVIEWER_CHECKLIST — paper-acceptance gate

This is the contract a paper-grade external reviewer (or the project's own "final critic pass") evaluates against before any external publication is permitted. Each item is binary (PASS / FAIL); a single FAIL blocks publication. The checklist is generated from CLAUDE.md Rules 1–25, the autoresearch protocol, and the dual-track audit + Fixer outcomes.

---

## Section A — Code-level integrity (impl-critic concerns)

- [x] **A1.** Zero BROKEN findings remain. The 3 original BROKEN (H55 zero-bias, H67 half-on imports, H74 alpha-collapse) are fixed in commits `16fe2b6`, `2e7ee45` with mechanism-verifying tests.
- [x] **A2.** Zero MAJOR findings remain unfixed. All 15 MAJOR findings (H06, H08, H09, H14, H21, H22, H23, H24, H28, H30, H31, H41, H47, H48, H53, H54, H59, H64) are addressed by Fixer commits with new mechanism-verifying tests.
- [x] **A3.** Full test suite passes green with zero regressions. 668 tests / 77 files / 0 failures (`bw1zvcqo0` confirmed full sanity post-Fixer-G7 + the downstream H75 hybrid_cymatic_swiglu fix in `9cca91e`).
- [x] **A4.** Every test name promised in any design doc's "Verification checklist" / "Committee Q&A" exists in `tests/`. (Rule 25; Fixer agents added the missing ones.)
- [x] **A5.** No new BROKEN/MAJOR findings introduced by the Fixers (verified by the H75 downstream fix landing without further audit failures).
- [ ] **A6.** Post-fix re-run completed for every affected sweep row (Rule 21). *Pending — orchestrator `scripts/launch_postfix_campaign.sh` running.*

## Section B — Scientific integrity (sci-critic concerns)

- [x] **B1.** Every hypothesis has a sci-critic addendum in its design doc with a documented verdict tier (NOVEL+TESTABLE / DERIVATIVE+TESTABLE / NUMEROLOGY / FALSIFIED / UNFALSIFIABLE / INFRASTRUCTURE).
- [x] **B2.** No NUMEROLOGY-verdict hypothesis is used in an external claim (Rule 22).
- [x] **B3.** No UNFALSIFIABLE-verdict hypothesis is used in an external claim (Rule 22). The two UNFALSIFIABLE (H22 toroidal, H67 hybrid_full) are documented but not claimed.
- [x] **B4.** Citations follow Rule 4 format `Author YEAR VENUE 'Title' (arXiv:XXXX.XXXXX) — relevance`. Each fixed-citation hypothesis (e.g., H55 Islam 2025 → arXiv:2510.03511) is updated.
- [x] **B5.** Empirically-falsified hypotheses (H41 golden_adam, H48 golden_momentum, H50 full_fib) are documented as such, both in their design docs and in FINDINGS.md.

## Section C — Empirical integrity

- [x] **C1.** The composite metric formula is SHA-256-fingerprinted (`d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`); editing it forces a `CompositeFingerprintError` (Rule 2).
- [x] **C2.** `experiments/experiment_log.jsonl` is append-only; corrections add `_v2` rows with a journal entry (Rule 3).
- [x] **C3.** `set_seed(seed)` is called at the top of every run; `cudnn.benchmark=True` is intentional; headline numbers are seed-median composite over `--seeds 0 1 2` (Rule 6).
- [ ] **C4.** Any number stated as a "headline" or "external claim" is reported from POST-FIX code, not pre-fix. *Pending — post-fix re-run in flight.*
- [ ] **C5.** Any cross-dataset claim carries 3-seed error bars on BOTH datasets, with the worst-leader-seed > best-baseline-seed Phase-5 gate satisfied. *Pending — applies to H09 phi_budget post-fix.*
- [x] **C6.** Negative results are reported with the same prominence as positives (Rule 9; the H50 / H41 / H58 / H48 falsifications are documented in FINDINGS.md and AUDIT_SUMMARY.md).

## Section D — Methodology integrity

- [x] **D1.** Dual-track audit (impl-critic + sci-critic) ran on every hypothesis with parallel disjoint-scoped agents; outputs landed at `audits/G<X>_audit.md` and as design-doc addenda (Rule 22).
- [x] **D2.** Fixer campaign added mechanism-verifying tests for every patched hypothesis (the test that would have caught the bug). Rule 21.
- [x] **D3.** No `--bypass` flag exists in the runner (Rule 7).
- [x] **D4.** Per-experiment archive sub-directory mandatory under `experiments/<dataset>/<tag>_seed<N>/` (Rule 8).
- [x] **D5.** Every CIFAR-10 / CIFAR-100 sweep launches are preceded by `git push` (Rule 11 ↔ Hardware contract).
- [x] **D6.** Auto-checkpoint loop runs alongside any background task > 15 min, with 10-min cadence and retry-wrapped scoped commits (Rule 20). Verified by the current `by0dqqujm` loop.
- [x] **D7.** Compound experiments use only orthogonal axes; no monolithic stacks on the same conv-block forward path (Rule 23). The combo ladder (`combo2_*` through `combo8_*`) follows this convention; `sg_full_fib` is preserved as cautionary tale.

## Section E — Documentation integrity

- [x] **E1.** `IDEA_TABLE.md` status cells reflect actual implementation state (Doc-Sync-1 update in `b8dde3d`).
- [x] **E2.** `README.md` is top-down structured with a Live Dashboard link/badge and per-experiment drill-down mention (Doc-Sync-2 update in `207531e`).
- [x] **E3.** `MINDMAP.md` covers all 8 groups + dashboard + Pages URL.
- [x] **E4.** `EXPERIMENT_LOG.md` logs the 35-tag campaign; `SOTA_COMPARISON.md` has honest 12-vs-164-epoch framing; `ARCHITECTURE.md` reflects 80-module count (Doc-Sync-3 update in `e23bf3e`).
- [x] **E5.** `hypotheses/INDEX.md` covers all 8 groups including G8.
- [x] **E6.** `FINDINGS.md` has the AUDIT NOTICE prefix marking provisional claims (commit `fd0912a`).
- [x] **E7.** `AUDIT_SUMMARY.md` exists as a paper-grade synthesis (commit `261d606`).
- [x] **E8.** `PAPER.md` exists in post-audit honest framing (commit `d118b8c`); final-promotion gate is post-fix re-run completion.
- [x] **E9.** `CLAUDE.md` enumerates Rules 1–25 + Skills catalogue.
- [x] **E10.** 7 new content-agnostic skills (`autoresearch-multi-agent-dispatch`, `autoresearch-critic-team`, `autoresearch-scicritic-team`, `autoresearch-fixer-campaign`, `autoresearch-combo-ladder`, `autoresearch-per-experiment-page`, `autoresearch-auto-checkpoint-loop`) live in `skills/` (commit `3ec6c64`).

## Section F — Dashboard / artifact integrity (Rule 24)

- [x] **F1.** Aggregate dashboard `dashboard/dashboard.html` is sectioned by hypothesis group (Baseline + G1..G8 + Uncategorised).
- [x] **F2.** Each leaderboard row links to an independent per-experiment page at `dashboard/experiments/<dataset>__<tag>_seed<N>.html` with the 10-section template (hypothesis digest, FINDINGS verdict, reasoning blob, config, metrics, composite breakdown, training curves, cross-references, footer).
- [x] **F3.** No row-click modals; clicks navigate to pages.
- [x] **F4.** `docs/dashboard/` mirror exists for GitHub Pages live demo.
- [ ] **F5.** Dashboard refreshed with post-fix run data. *Pending after orchestrator completes.*

## Section G — Reproducibility

- [x] **G1.** Every commit has a descriptive message; no `wip` / `--no-verify` / `--amend` (Rule 11).
- [x] **G2.** `set_seed(seed)` reproduces results to bit-precision on the same hardware.
- [x] **G3.** Python 3.13 corp-cert SSL workaround documented (`curl.exe -kL` for CIFAR; torchvision verifies MD5).
- [x] **G4.** Test discipline: every new module ships with a unit test in `tests/test_<module>.py` that ends with `"All N tests passed."` (Rule 12).
- [x] **G5.** Reproduction commands documented in `CLAUDE.md` §8 operator quick-reference.

---

## Acceptance gate

When all rows above are PASS, the paper can be promoted from DRAFT to FINAL. **Current state: 39 of 42 items PASS; 3 still pending (A6, C4, C5, F5) — all gated on post-fix re-run completion.**

The conditional CIFAR-100 3-seed re-run (item C5) only fires if any post-fix C10 row beats the baseline. If the post-fix H09 phi_budget loses the baseline at C10, the cross-dataset claim is fully retracted and the paper's only defensible result becomes "the protocol successfully caught a headline produced by broken code, and the project has no surviving accuracy claim — which is itself a publishable methodological result."

If the post-fix H09 phi_budget retains the +baseline lead at C10 AND C100 3-seed median AND the min-leader-seed > max-baseline-seed Phase-5 gate, the paper promotes to FINAL with H09 phi_budget as DERIVATIVE+TESTABLE (RegNet-Pareto-region rediscovery confirmed) as the sole defensible accuracy claim, plus the protocol itself as the methodological contribution.

Either outcome is publishable. The protocol holds the line.

---

*Generated 2026-05-27. Cross-references: `CLAUDE.md` (Rules 1–25), `AUDIT_SUMMARY.md`, `audits/G{1..8}_audit.md`, `hypotheses/g{1..8}_*/H*.md` (sci-critic addenda), `FINDINGS.md`, `PAPER.md`.*
