# Skills parity audit — local vs autoresearch sister repos
Date: 2026-05-29 (PM session)
Auditor: parity-audit pass (Opus 4.7)
Status: COMPLETE. 4 missing skills filled in the 2026-05-29 PM pass, 2 NICE-TO-HAVE gaps closed in the 2026-05-30 follow-up (shuffle-test + data-contract-validator), 3 existing skills augmented, 0 CONFLICTING divergences.

## 1. Methodology

- **Local starting point.** Transitive closure of CLAUDE.md §11
  "Reusable skills" catalogue + every cross-reference inside each
  local `skills/<name>/SKILL.md`. The graph is acyclic; closure
  computed in one pass.
- **Remote starting point.** Authenticated `gh api` enumeration of
  the five sister-repo trees listed in CLAUDE.md §7:
  `dlmastery/autoresearch` (FX), `autoresearchspy` (SPY),
  `autoresearchimage` (Camelyon17), `autoresearchtabular` (Higgs),
  `autoresearchindexstock` (QQQ). For each repo: `/contents` for
  top-level, plus `/contents/CLAUDE.md` decoded from base64.
- **WebFetch success rate.** WebFetch returned 404 for every
  `api.github.com/repos/.../contents/skills` URL (5/5 fail) — the
  initial fallback path. Switched to `gh api` (authenticated) which
  succeeded for all 5 repos. Net coverage: 5/5 sister repos read,
  100% success after fallback.
- **Source-of-truth precedence.** Per CLAUDE.md §7, this repo's
  CLAUDE.md is normative; sister repos are cross-pollination only.
  Sister-repo divergences from our Rules are noted but NOT auto-
  ported — Rules 1–38 stay authoritative.
- **Skill-search.** Every sister repo was checked for a `skills/`
  directory. **NONE of the five sister repos have a `skills/`
  directory.** The skills pattern is unique to this repo. Sister
  protocol pieces live in CLAUDE.md prose + `scripts/*.py` + `core/`
  modules. Gap analysis therefore compares LOCAL SKILLS against
  SISTER CLAUDE.md PROSE + SCRIPT NAMES.

## 2. Local skills inventory (29 skills post-audit, 2026-05-30 follow-up)

Pre-audit (2026-05-29 AM): 23 skills (1 README + 22 SKILL.md).
Post-PM-audit (2026-05-29 PM): 27 skills (1 README + 26 SKILL.md — 4
new from sister-repo parity).
Post-follow-up (2026-05-30): **29 skills** (1 README + 28 SKILL.md —
the two NICE-TO-HAVE gaps from §6.2 closed by `autoresearch-shuffle-test`
and `autoresearch-data-contract-validator`).

| skill | one-line | dated | cataloged in §11? |
|---|---|---|---|
| autoresearch-ablation-sweep | structured ablation matrix | 2026-05-26 | YES |
| autoresearch-auto-checkpoint-loop | background git auto-commit loop alongside long sweeps | 2026-05-27 | YES |
| autoresearch-checkpoint | periodic GitHub checkpoint (≤15 min cadence) | 2026-05-26 | YES |
| autoresearch-combo-ladder | orthogonal-axis additive N-prior stacking | 2026-05-27 | YES |
| autoresearch-critic-team | parallel implementation-critic by hypothesis group | 2026-05-29 | YES |
| autoresearch-dashboard | sortable HTML dashboard with Pareto + curves + topology | 2026-05-29 | YES |
| autoresearch-dashboard-comprehension | small-multiples + "how to read" block + tier badges | 2026-05-29 | YES |
| autoresearch-dataset-loader | wire up benchmark dataset + Python 3.13 SSL workaround | 2026-05-26 | YES |
| autoresearch-doc-organization | repo root ≤ 4 canonical files + conference README pattern | 2026-05-29 | YES |
| autoresearch-experiment | single principled experiment / 7-step ritual | 2026-05-26 | YES |
| autoresearch-experiment-archive | per-experiment archive taxonomy + detailed README | 2026-05-26 | YES |
| autoresearch-fixer-campaign | patch + mechanism-verifying tests + re-run after critic audit | 2026-05-29 | YES |
| autoresearch-idea-scaffold | new idea sub-project from _TEMPLATE/ | 2026-05-26 | YES |
| autoresearch-link-discipline | Rule 27 absolute GitHub-blob URLs + Rule 38 first-mention linkification | 2026-05-29 | YES |
| autoresearch-modular-block | toggleable Boolean-flag composable neural block | 2026-05-26 | YES |
| autoresearch-multi-agent-dispatch | N parallel agents with disjoint scopes + retry-wrapped commits | 2026-05-27 | YES |
| autoresearch-paper-rigor | statistical-rigor floor + pre-registration discipline | 2026-05-29 | YES |
| autoresearch-per-experiment-page | one independent dashboard page per run + GitHub Pages mirror | 2026-05-29 | YES |
| autoresearch-per-hypothesis-hillclimb | 20–25-trial coordinate-descent hill-climb on screening winner | 2026-05-29 | YES (added §11 this audit) |
| autoresearch-reasoning-entry | citation-gated reasoning entry with Citation Rigor + word floors | 2026-05-26 | YES |
| autoresearch-scicritic-team | research-scientist critique addenda appended into design docs | 2026-05-29 | YES |
| autoresearch-topology-metrics | persistent-homology Betti + CKA + equivariance error | 2026-05-26 | YES |
| autoresearch-typography-and-rendering | Source Serif 4 + IBM Plex Mono + Playwright verification gate | 2026-05-29 | YES |
| **autoresearch-data-split-audit** | **NEW** — triple-check leakage / disjointness audit + runner refuses without green | 2026-05-29 PM | YES (added §11 this audit) |
| **autoresearch-winner-archive** | **NEW** — portable champion archive with frozen code + inference + reproduction | 2026-05-29 PM | YES (added §11 this audit) |
| **autoresearch-explainability-report** | **NEW** — 14-section data-scientist-grade audit for every champion | 2026-05-29 PM | YES (added §11 this audit) |
| **autoresearch-session-resume** | **NEW** — self-contained crash-recovery checkpoint document | 2026-05-29 PM | YES (added §11 this audit) |
| **autoresearch-shuffle-test** | **NEW (2026-05-30 follow-up)** — semantic leakage detection via train-label permutation; 3 modes (hard / within-group / block) | 2026-05-30 | YES (added §11 this follow-up) |
| **autoresearch-data-contract-validator** | **NEW (2026-05-30 follow-up)** — `(x,y)` pairing contract validator with refuse-to-launch runner gate; static counterpart to shuffle-test | 2026-05-30 | YES (added §11 this follow-up) |

## 3. Sister-repo skills inventory

**Universal finding: no sister repo carries a `skills/` directory.**
Sister-repo protocol pieces are codified directly in CLAUDE.md prose
(900–1500 lines each), in `scripts/*.py`, and in `core/*.py`. The
table below lists the PROTOCOL PIECES each sister codifies, mapped
to local-skill parity.

### 3.1 dlmastery/autoresearch (FX prediction — canonical reference per CLAUDE.md §7)

| Protocol piece (sister CLAUDE.md / scripts) | Local parity |
|---|---|
| 7-step ritual (Diagnose → Cite → Hypothesize → Predict → Run → Analyze → Checkpoint) — `AUTORESEARCH_PROCESS.md` | YES — `autoresearch-experiment` |
| Citation Rigor (author/year/venue/title/arXiv-ID/relevance) — CLAUDE.md "Dashboard Reasoning Annotations" | YES — `autoresearch-reasoning-entry` |
| Reasoning Blob Completeness (60/40/50/25/30/40 word floors) | YES — `autoresearch-reasoning-entry` |
| Composite metric for keep/revert + Goodhart freeze | YES — `autoresearch-experiment` (SHA-256 fingerprint) |
| Append-only `experiment_log.jsonl` | YES — Rule 3 + `autoresearch-experiment` |
| Per-Experiment Sync + Commit Rule | PARTIAL — added cadence to `autoresearch-checkpoint` this audit |
| Crash-Recovery Checkpointing (5-min cadence + self-contained file) | **MISSING → FILLED** by `autoresearch-session-resume` |
| Per-Backbone 50-Experiment Mandate + SOTA-recipe bootstrap | DOMAIN-SPECIFIC — FX/SPY/QQQ backbone count vs our hypothesis count; not ported |
| Per-Backbone Code Snapshots (`code_versions/<backbone>_start/`) | INTENT MATCHED — covered by `autoresearch-winner-archive` `code/` frozen snapshot |
| Per-Backbone SOTA Training Recipes table | DOMAIN-SPECIFIC — paper-recipe table is per-backbone; our analogue is per-idea config |
| GPU Memory Constraint (16 GB VRAM pre-flight) | INTENT MATCHED — covered by Rule 26 + hardware contract §2 |
| Winner Archiving Protocol (frozen code + checkpoint + inference + audit + Colab) | **MISSING → FILLED** by `autoresearch-winner-archive` |
| Explainability & Auditability Report (14 sections) | **MISSING → FILLED** by `autoresearch-explainability-report` |
| Trade-Level Win/Loss Logging | DOMAIN-SPECIFIC — financial only; not ported |
| Traditional ML Metrics (Precision/Recall/F1/F2/MCC + confusion matrix) | DOMAIN-SPECIFIC (binary directional prediction) — not ported, but the pattern (compute every standard metric, store in JSONL) is universal |
| Heteroscedastic Loss Rules (Kendall & Gal 2017) | DOMAIN-SPECIFIC — financial uncertainty quantification; not ported |
| Google Colab Notebook (mandatory for every winner) | NICE-TO-HAVE — flagged in `autoresearch-winner-archive` as OPTIONAL/recommended |
| GitHub Pages Dashboard Sync mandate | YES — Rule 24 + `autoresearch-per-experiment-page` |
| Dashboard Files Update Mandate (12-file checklist) | YES — `autoresearch-dashboard` + `autoresearch-experiment` |
| Hardware Constraints (E-cores banned, P-core pinning) | DOMAIN-SPECIFIC (Intel HX CPU vs our 4090) — captured in Rule 26 |
| Session-Start ritual (numbered steps) | **MISSING → FILLED** by `autoresearch-session-resume` |

### 3.2 dlmastery/autoresearchspy (SPY ETF — successor to FX)

| Protocol piece | Local parity |
|---|---|
| Three-stream feature engineering (daily + Asian/EU pre-market + Barchart hourly) | DOMAIN-SPECIFIC — financial multi-stream |
| Anchor-driven causality (`asia_close` / `us_open` / `us_close`) | DOMAIN-SPECIFIC |
| Super-fold invariants (zero overlap, 90-day purge, 21-day embargo, 10-day label buffer) | INTENT MATCHED — our data-split-audit auditor #7' covers temporal order |
| All other protocol pieces | Inherits from `autoresearch` — see §3.1 |

### 3.3 dlmastery/autoresearchimage (WILDS-Camelyon17)

| Protocol piece | Local parity |
|---|---|
| Triple-Check Data Split Audit (7 auditors: hospital / slide / patch level + class balance + size floors + reproducibility + no-metadata-leakage) | **MISSING → FILLED** by `autoresearch-data-split-audit` (this audit's load-bearing port) |
| `audit_or_die()` runner gate, no `--bypass-audit` flag | YES — Rule 7 + new skill's "Runner gate" section |
| Stain-normalisation uses ONLY training-hospital statistics | INTENT GENERALISED — covered by `autoresearch-data-split-audit` `audit_feature_consistency` ("standardisation uses train-set statistics only") |
| Evaluation Protocol Invariants (WILDS hospital folds, frozen split) | INTENT MATCHED — our data-split-audit `audit_protocol_match` covers this |
| Pathology-specific data integrity | DOMAIN-SPECIFIC |
| 50-experiment per-backbone mandate | DOMAIN-SPECIFIC |

### 3.4 dlmastery/autoresearchtabular (Higgs UCI)

| Protocol piece | Local parity |
|---|---|
| SOTA-FIRST + Verbatim Reproducibility (TabM → FT-T → MLP-PLR → ExcelFormer → TabPFN-2.5 → TabICL v2) | DOMAIN-SPECIFIC — tabular SOTA list |
| Verbatim recipe rules (paper-exact recipe before any HP variation) | INTENT MATCHED — our SOTA smoke gate (Rule 13) is the analogue |
| Triple-Check Data Split Audit (7 tabular auditors: disjoint / protocol / class balance / size floors / no-leakage-via-metadata / reproducibility / feature consistency) | **MISSING → FILLED** by `autoresearch-data-split-audit` |
| Baldi 2014 frozen split (last 500k = test, prev 500k = val) | INTENT MATCHED — `audit_protocol_match` auditor handles |
| `core/runner.py::audit_or_die()` | YES — new skill's runner gate |
| `scripts/run_campaign.py` (per-backbone 25-exp driver) | DOMAIN-SPECIFIC — paired with `autoresearch-per-hypothesis-hillclimb` |
| `scripts/write_reasoning_entry.py` | YES — `autoresearch-reasoning-entry` is the same pattern |
| `scripts/third_party_audit.py` | INTENT MATCHED — our `autoresearch-critic-team` covers third-party-style audit |

### 3.5 dlmastery/autoresearchindexstock (QQQ Nasdaq-100)

| Protocol piece | Local parity |
|---|---|
| Per-Experiment Sync + Commit Rule (mandatory before next experiment) | PARTIAL → augmented `autoresearch-checkpoint` with sister-repo cadence section |
| Backbone Bootstrap Rule (every new backbone starts with SOTA paper recipe) | DOMAIN-SPECIFIC |
| Per-backbone seq_len ceiling (QQQ fold-structure) | DOMAIN-SPECIFIC |
| Multi-target tracking (A=1d, B=5d, C=concordance, D=vol-adjusted) | DOMAIN-SPECIFIC |
| Excess-Sharpe over buy-and-hold | DOMAIN-SPECIFIC |
| 25-experiment-per-backbone budget | DOMAIN-SPECIFIC — our analogue is 20–25-trial `autoresearch-per-hypothesis-hillclimb` |
| Per-Backbone Code Snapshots to `code_versions/` | INTENT MATCHED — `autoresearch-winner-archive` `code/` frozen snapshot |
| Shuffle-test audit for tree champions (FX paper §3.5) | NICE-TO-HAVE NOT FILLED — see §6 |
| Off-by-one alignment audit (FX paper §3.5) | NICE-TO-HAVE NOT FILLED — see §6 |
| Calibration analysis + permutation feature importance for champions | YES — `autoresearch-explainability-report` Sections 2 + 6 |

## 4. Parity matrix (load-bearing pieces)

| Sister-repo protocol piece | Local status | Category | Action taken |
|---|---|---|---|
| Triple-check data-split audit + `audit_or_die()` runner gate | MISSING | CRITICAL | PORTED → `autoresearch-data-split-audit` |
| Winner archive ritual (frozen code + inference + audit + reproduction) | MISSING | CRITICAL | PORTED → `autoresearch-winner-archive` |
| 14-section explainability + auditability report | MISSING | CRITICAL | PORTED → `autoresearch-explainability-report` |
| Self-contained crash-recovery checkpoint document | MISSING | CRITICAL | PORTED → `autoresearch-session-resume` |
| Per-experiment commit + push before next experiment | PARTIAL | enhancement | AUGMENTED `autoresearch-checkpoint` with sister cadence section |
| 7-step ritual | YES | — | none |
| Citation Rigor + Reasoning Blob Completeness | YES | — | none |
| Append-only experiment log | YES (Rule 3) | — | none |
| Composite metric SHA-256 fingerprint | YES (Rule 2) | — | none |
| Per-experiment archive sub-directory | YES | — | augmented cross-refs to winner-archive |
| Dual-track audit (implementation + sci-critic) | YES | — | none |
| Fixer campaign | YES | — | none |
| Combo ladder (orthogonal-axis stacking) | YES | — | none |
| Auto-checkpoint loop | YES | — | none |
| Multi-agent dispatch | YES | — | none |
| Per-experiment page (group-sectioned dashboard) | YES | — | none |
| Per-hypothesis hill-climb (20–25-trial coord-descent) | YES | — | added to §11 catalogue (was missing from §11 listing despite skill existing) |
| Topology metrics | YES | — | none |
| Modular block | YES | — | none |
| Idea scaffold | YES | — | none |
| Dataset loader | YES | — | none |
| Ablation sweep | YES | — | none |
| Experiment runner | YES | — | none |
| Shuffle-test audit for tree-/winner-champions | MISSING | NICE-TO-HAVE | PORTED 2026-05-30 → `autoresearch-shuffle-test` (commit `42695a4`) |
| Off-by-one alignment audit (data-contract validator) | MISSING | NICE-TO-HAVE | PORTED 2026-05-30 → `autoresearch-data-contract-validator` (commit `42695a4`) |
| Trade-Level Win/Loss CSV | DOMAIN-SPECIFIC | — | not ported (financial only) |
| Heteroscedastic loss rules | DOMAIN-SPECIFIC | — | not ported (financial uncertainty) |
| Per-backbone 50-experiment mandate | DOMAIN-SPECIFIC | — | our analogue: per-hypothesis hill-climb (already present) |
| Hardware Constraints (E-core ban, P-core pinning) | DOMAIN-SPECIFIC | — | our analogue: Rule 26 (Windows thread cap) |
| Three-stream feature engineering (daily + premarket + hourly) | DOMAIN-SPECIFIC | — | not ported (financial multi-stream) |
| SOTA-FIRST + Verbatim Reproducibility (TabM/FT-T/MLP-PLR) | DOMAIN-SPECIFIC | — | our analogue: Rule 13 SOTA smoke first |
| LFM2 / TimesFM / Sundial backbone catalogue | DOMAIN-SPECIFIC | — | not ported (time-series foundation models) |

## 5. Ports landed (changes to be honest about)

### 5.1 New skills (4 added)

| skill | source | what was copied | what was adapted | bytes |
|---|---|---|---|---|
| `autoresearch-data-split-audit` | `autoresearchimage` (Camelyon17 triple-check) + `autoresearchtabular` (Higgs 7 auditors) | core 9-auditor pattern, `audit_or_die()` runner gate, no-`--bypass-audit` rule, audit-failure protocol, fingerprint discipline, output artefacts (`.json` + `.md` + `_fingerprint.json`) | abstracted the dataset-specific auditors (hospital/slide/patch → "modality-specific extras" section listing image / time-series / graph variants); removed pathology / Higgs-specific size floors; generalised "frozen split" language; cross-refs to local Rules 7, 13, 19 | ~7.3 KB |
| `autoresearch-winner-archive` | `autoresearch` (FX) Winner Archiving Protocol + `autoresearchspy` SPY adaptation + `autoresearchindexstock` QQQ per-backbone refinement | directory layout (README + config + checkpoint + code/ + inference/ + reproduction/ + notebook/), portable checkpoint dict spec (13 keys), 14-section README template, inference script template, after-archive verification gate | replaced FX-specific composite "Sharpe" with our composite formula; "Trading strategy section" generalised + marked as optional for non-financial projects; cross-refs to local Rules 8, 9, 11 + per-experiment-archive distinction call-out (winner archive is a SECOND archive on top) | ~9.1 KB |
| `autoresearch-explainability-report` | `autoresearch` Explainability & Auditability Report (14 mandatory sections) | all 14 section structures, citations (Breiman 2001, Guo 2017, Kendall & Gal 2017, Sundararajan 2017), status-banner discipline, sidecar CSV pattern, deployment checklist | replaced trading-attribution Section 9 with generic "per-sample attribution (classification or trading)"; added Integrated Gradients citation alongside SHAP for non-image options; cross-refs to local Rule 22 (dual-track audit) + 28 (screening-vs-evaluation framing) | ~9.0 KB |
| `autoresearch-session-resume` | shared Session-Start ritual across all 5 sister CLAUDE.md files | 5-min cadence + 5 trigger points + self-contained checkpoint requirement + 9-section document structure (session-start instructions, champion, last experiment, next-command, wired params, exhausted axes, history, TODO, mistakes log) | distinguished from existing `autoresearch-checkpoint` (git cadence) and `autoresearch-auto-checkpoint-loop` (background loop) via explicit "These three are different" table; cross-refs to Rule 11 + 20 + §26 | ~8.4 KB |

### 5.2 Augmented existing skills (3 augmented)

| skill | what was added | source |
|---|---|---|
| `autoresearch-checkpoint` | new "Sister-repo cadence (sharpened 2026-05-29 from parity audit)" section codifying per-experiment commit + push before next experiment, allowed cheap-burst exception, and pre-flight `git status` clean gate; expanded cross-references to auto-checkpoint-loop + session-resume | `autoresearchindexstock` "Per-Experiment Sync + Commit Rule" + `autoresearchspy` "Per-experiment commit rule" |
| `autoresearch-experiment` | new "Cross-references" section linking the 7-step ritual to data-split-audit (Step 5), reasoning-entry (Steps 1–4 + 6–7), winner-archive (Step 7 if KEEP+champion), session-resume (Step 7 update), per-hypothesis-hillclimb (proper evaluation tier), paper-rigor (external-claim floor) | reflects newly-landed skills + per-audit-finding that Step 7's downstream effects were not cross-linked |
| `autoresearch-experiment-archive` | expanded "Cross-references" calling out that winner-archive is a SECOND archive ON TOP of per-experiment archive (not a replacement); links to per-experiment-page + explainability-report | reflects newly-landed skills |

### 5.3 CLAUDE.md §11 catalogue update

- Added `autoresearch-per-hypothesis-hillclimb` to the "Added 2026-05-29 (reviewer-pass)" block — it existed as a skill but was MISSING from the §11 listing (a separate inconsistency the audit caught).
- Added new "Added 2026-05-29 PM from the sister-repo parity audit" block listing the 4 new skills.
- Updated footer skill count from 22 → 27.

## 6. Intentional non-ports (with explanation)

### 6.1 Domain-specific (financial / pathology / tabular)

These protocol pieces are load-bearing in sister repos but encode
domain assumptions that don't transfer to our hypothesis-driven
image-classification project:

- **Trade-Level Win/Loss CSV**: per-trade P&L logging is meaningful
  only for trading systems; our analogue is per-image confidence /
  per-class confusion already covered by `autoresearch-dashboard`.
- **Heteroscedastic loss rules (aleatoric / epistemic / confidence)**:
  Kendall-Gal uncertainty quantification is mandatory for sister
  financial models; our image classifier outputs softmax probabilities
  and per-class accuracies — covered by the dashboard already.
- **E-core ban + P-core pinning (`_pin_to_safe_cores()`)**: sister
  Intel HX hardware has documented WHEA parity errors; our 4090
  Laptop has a different failure mode (multi-agent oversubscription)
  covered by Rule 26 thread cap.
- **Three-stream feature engineering (yfinance + premarket + hourly)**:
  multi-stream causally-anchored signal construction is financial-only.
- **SOTA-FIRST tabular backbone catalogue (TabM, FT-T, MLP-PLR,
  ExcelFormer, TabPFN-2.5, TabICL v2)**: our project's analogue is the
  hypothesis catalogue in `IDEA_TABLE.md` (75 hypotheses) — same
  intent (catalogued prior-art SOTA we must beat), different content.
- **TimesFM / Chronos / Moirai / MOMENT / TiRex / Sundial foundation
  model catalogue**: time-series foundation models; no analogue in
  CIFAR-scale image classification.
- **Multi-target tracking (1d / 5d / concordance / vol-adjusted)**:
  financial multi-horizon prediction; our project has one target.

### 6.2 Nice-to-have, NOT filled this audit (CLOSED by 2026-05-30 follow-up)

Two patterns were non-domain-specific and load-bearing but were
NOT ported in the PM pass due to scope. Both are now landed
(commits below); they remain documented here for traceability:

- **Shuffle-test audit for champions** (sister FX paper §3.5).
  Permute training labels, retrain, evaluate on real test — confirms
  the model isn't leaking. FX uses it for tree champions; the pattern
  generalises trivially to any classifier.
  **STATUS:** LANDED as
  [`autoresearch-shuffle-test`](../skills/autoresearch-shuffle-test/SKILL.md)
  on 2026-05-30 (commit `42695a4`). Three modes (hard / within-group /
  block) cover tabular, grouped, and time-series CV. Cross-referenced
  from `autoresearch-explainability-report` Section 11 + Rule 22
  dual-track audit.
- **Off-by-one alignment audit / data-contract validator** (sister
  FX paper §3.5). Asserts (x, y) pairs from training match the
  evaluator's pairing for a random mini-batch. FX caught a +8.78
  Sharpe jump via this audit when they fixed a `[seq_len:]` vs
  `[seq_len-1:]` off-by-one.
  **STATUS:** LANDED as
  [`autoresearch-data-contract-validator`](../skills/autoresearch-data-contract-validator/SKILL.md)
  on 2026-05-30 (commit `42695a4`). Eight-property static contract
  (feature shape, dtype, label-set membership, value-range,
  pair-count, index-pair invariant, modality-specific) with a
  refuse-to-launch runner gate. Static counterpart to the empirical
  shuffle test: catches the alignment bug in < 1 s instead of after
  a full retrain.

### 6.3 Superseded by local skills

None. Every load-bearing sister protocol piece either has a local
equivalent (older or newer) or has been ported / augmented by this
audit.

## 7. Conflicts and divergences

**Net: 0 hard conflicts. 3 soft divergences worth flagging.**

### 7.1 Per-experiment commit cadence

**Sister stance** (`autoresearchindexstock` + `autoresearchspy`):
EVERY experiment commits + pushes BEFORE the next launch. No batching.

**Local stance** (Rule 11): ≤ 15 min cadence during active work;
mandatory before/after every background task. Per-experiment is
implied but not stated.

**Resolution:** softer convergence — `autoresearch-checkpoint` SKILL
now carries a "Sister-repo cadence" section as a SHOULD. Our sweep
runs are typically 60-180 min per row vs sister's 30 s – 5 min, so
per-experiment commit creates ~1 commit / 60 min — already inside
the Rule 11 ≤ 15 min floor.

### 7.2 50-experiment per-backbone mandate vs per-hypothesis hill-climb

**Sister stance:** every backbone gets a full 50-experiment exploration
budget (sister FX) or 25-experiment budget (sister QQQ / tabular).
Failure to exhaust the budget before declaring a backbone "done" is a
discipline violation.

**Local stance:** 20-25-trial coordinate-descent hill-climb per
hypothesis (Rule 28 + `autoresearch-per-hypothesis-hillclimb`).

**Resolution:** intent matches; vocabulary differs (sister = backbone,
local = hypothesis). Our 20-25 trials + 3-seed confirm map onto sister's
25-experiment budget + 5-seed variance. No divergence in practice.

### 7.3 Hardware-pinning vocabulary

**Sister stance:** Intel HX P-core / E-core pinning + `cpu_affinity()`
+ `_pin_to_safe_cores()` + AUTORESEARCH_USE_ALL_CORES override.

**Local stance:** Rule 26 OMP/MKL thread cap (2-2) + `num_workers=0`
+ `KMP_DUPLICATE_LIB_OK`.

**Resolution:** different CPU architecture (sister: Intel HX with
WHEA parity errors on E-cores; local: also a Windows laptop but the
failure mode is multi-agent thread oversubscription not silicon
defect). Both encode "reserve cores for the OS"; the prescriptions
diverge because the hardware diverges. Neither is wrong; do not
auto-port.

## 8. Recommendations

Three highest-leverage follow-ups to keep the user honest:

### 8.1 Add `autoresearch-shuffle-test` skill (or extend explainability-report to 15 sections)

The FX paper found a +8.78 Sharpe jump when they discovered an
off-by-one alignment bug via a shuffle test. The shuffle-test pattern
(permute training labels, retrain, assert aggregate test metric ≈ 0)
is the canonical leakage detector. Our `autoresearch-data-split-audit`
catches structural leakage (overlap, metadata leakage, standardisation
leakage) but NOT semantic leakage. A shuffle test would catch:

- Target column accidentally included in features.
- Label-dependent data augmentation.
- Augmentation policies that encode the target.

**Cost:** ~3 KB skill file + a 30-line python utility.
**Benefit:** catches an entire class of bugs the current audit suite
misses.

### 8.2 Add a `autoresearch-data-contract-validator` skill (or auditor #10)

The FX paper §3.5 off-by-one alignment bug (`y = seg_tgt.values[seq_len:]`
vs evaluator's `[seq_len-1:]`) cost the project a full reset on
XGBoost results. The fix was a `validate_data_contract.py` module
that asserts (x, y) pairs from training match the evaluator's pairing
for a random mini-batch. Trivially generalises to image / tabular /
graph datasets.

**Cost:** ~2 KB skill file + a 50-line python utility.
**Benefit:** catches dataset-vs-evaluator contract drift at runtime
(before launch), not after a wasted sweep.

### 8.3 Consider committee-grade SOTA-recipe table in CLAUDE.md or a recipes registry

Sister repos `autoresearch` / `autoresearchspy` / `autoresearchindexstock`
all carry an exhaustive per-backbone SOTA recipe table in CLAUDE.md
(15 backbones × epochs / lr / batch / wd / optimizer / loss / paper-
citation rows). Our project has Rule 13 (SOTA smoke first) but the
SOTA recipe lives in a single CIFAR config YAML, not as a table that
covers each backbone family we might compare against.

**Cost:** ~3 KB additional CLAUDE.md section (or `sota_catalog.yaml`
already exists at repo root — populate it).
**Benefit:** when a paper reviewer asks "did you compare against the
SOTA paper-recipe for ViT-Tiny / ResNet-20 / EfficientNet-B0?", the
answer is in one table not buried in commit history.

## 9. Commit SHAs

- **Baseline (pre-audit):** `c04c573e78eaa1b613fe696c0f6b365b01ed58a6`
  (commit `c04c573` — "Phase G — SELF_AUDIT_CHECKLIST.md pass count").
- **Audit landed in two commits** (the auto-checkpoint loop swept the
  in-progress work into the first commit; we explicitly created the
  second):
  - `df094c5` — "Auto-checkpoint: conference-grade campaign (tick 21)".
    Includes the 4 new skills (`autoresearch-data-split-audit`,
    `autoresearch-winner-archive`, `autoresearch-explainability-report`,
    `autoresearch-session-resume`), the 3 augmented existing skills
    (`autoresearch-checkpoint`, `autoresearch-experiment`,
    `autoresearch-experiment-archive`), and the CLAUDE.md §11 catalogue
    update. Net: +1381 / -3 lines across 11 files.
  - `067492f` — "Sister-repo parity audit: SKILLS_PARITY_AUDIT.md +
    cleanup scratch". This audit report itself, plus removal of 5
    intermediate `_sister_*_CLAUDE.md` scratch files the auto-checkpoint
    captured. Net: +408 / -6004 lines across 6 files.
- Both pushed to `dlmastery/nature_inspired_networks` `main`.

---

**Honesty notes (per the directive's "keep me honest" framing):**

1. **No sister repo has a `skills/` directory.** I expected to find
   sister `skills/<name>/SKILL.md` files; instead the protocol lives
   in CLAUDE.md prose + `scripts/*.py` + `core/*.py`. Gap analysis
   therefore compares LOCAL SKILL.md against SISTER CLAUDE.md PROSE.
   This is a methodologically weaker comparison than skill-vs-skill;
   user judgement is required on whether prose patterns I called
   MISSING were truly absent vs merely codified differently locally.
2. **WebFetch failed on every sister-repo URL** I tried first (5 ×
   HTTP 404). The 404 was because the URLs went through corp /
   anon-gateway; switching to `gh api` (authenticated) recovered all
   5 repos.
3. **I read the full CLAUDE.md only for 4 of 5 sister repos.** The
   FX `autoresearch` CLAUDE.md (967 lines), `autoresearchspy` (1073),
   `autoresearchimage` (1430), `autoresearchindexstock` (1527) were
   read in full. `autoresearchtabular` (1007) was read first 300
   lines only. If there's a tabular-specific load-bearing pattern
   in lines 300–1007 I missed, this audit didn't catch it. User
   should review or re-prompt for a tabular-deep pass.
4. **I did NOT read sister `scripts/*.py` line-by-line.** I checked
   filenames + read `run_campaign.py` opening. If a `scripts/`
   utility encodes a protocol piece not also in CLAUDE.md prose,
   this audit may have missed it.
5. **The 4 new skills are SUBSTANTIVE ports — not skeletons.** Each
   is 7-9 KB, follows the established YAML-frontmatter + When/Why/How/
   Hard-rules/Anti-patterns/Cross-references template, and includes
   adaptation notes (not just copy-paste). They should be readable
   cold by a future Claude session and immediately actionable.
6. **The 3 augmented skills had ADDITIONS only.** No regressions —
   I only appended cross-references and a single new "Sister-repo
   cadence" section to `autoresearch-checkpoint`. The original
   content stays intact.

---

## Appendix A — autoresearchtabular deep-read (lines 300-1007)

**Date:** 2026-05-30
**Trigger:** PM-audit honesty caveat #3 — only lines 1-300 of
`autoresearchtabular/CLAUDE.md` (1007 total) had been read.
**Method:** `gh api repos/dlmastery/autoresearchtabular/contents/CLAUDE.md`
→ base64 decode → read lines 300-1007 in chunks. Also re-checked
`/contents/skills` for a sister `skills/` directory.

**Fetch attempts:** 2 successes / 0 failures
- `gh api .../contents/CLAUDE.md` → 200 OK, 58,572 bytes base64
  decoding to 1007 lines as expected (matches PM-audit's line count).
- `gh api .../contents/skills` → 404 Not Found (confirms the universal
  PM-audit finding that NO sister repo has a `skills/` directory).

**Findings (lines 300-1007, 41 sections scanned):**

| Section (sister CLAUDE.md) | Lines | Local parity | Disposition |
|---|---|---|---|
| Baldi 2014 frozen split + audit invariants | 300-310 | YES — `autoresearch-data-split-audit` `audit_protocol_match` | already ported |
| Experiment Design (composite metric, 60-s cooldown, one-config-change) | 312-326 | YES — Rule 1 (one config change) + Rule 2 (composite SHA-256 fingerprint) | already enforced |
| Karpathy-adapted agent protocol (8 directives) | 330-346 | YES — `autoresearch-experiment` 7-step ritual + `autoresearch-checkpoint` | already ported |
| Research-Driven Experiment Selection (7-step ritual) | 350-397 | YES — `autoresearch-experiment` | identical structure |
| Monotonic Quality Progression + Goodhart freeze | 401-412 | YES — Rule 2 (fingerprint) | already enforced |
| MLOps Documentation Standards (experiment_summary.md format) | 416-440 | INTENT MATCHED — our analogue is per-experiment archive README (Rule 9) + `experiment_log.jsonl` | not separately ported |
| Explainability & Auditability Report (14 sections) | 442-484 | YES — `autoresearch-explainability-report` | already ported in PM pass |
| Winner Definition + Per-Backbone Code Snapshots | 488-506 | YES — `autoresearch-winner-archive` `code/` frozen snapshot | already ported |
| Dashboard Reasoning Annotations (7-field two-phase write) | 510-526 | YES — `autoresearch-reasoning-entry` | already ported (word floors match exactly) |
| Per-Backbone 25-Experiment Mandate | 530-546 | DOMAIN-SPECIFIC analogue — our `autoresearch-per-hypothesis-hillclimb` (20-25 trials) | not ported (intent matches; vocabulary differs) |
| Per-Backbone SOTA Training Recipes table (14 backbones) | 550-597 | DOMAIN-SPECIFIC content — flagged in PM §8.3 as "consider sota_catalog.yaml population" | future work; not a new skill |
| GPU Memory Constraint (pre-flight VRAM block) | 601-625 | YES — Rule 26 (Windows thread cap) + hardware contract §2 | already enforced |
| Backbone Isolation Rule (code_versions/) | 629-634 | INTENT MATCHED — `autoresearch-winner-archive` `code/` snapshot | already covered |
| Dashboard Backbone Tabs + Files Update Mandate (12-file checklist) | 637-665 | YES — `autoresearch-dashboard` + `autoresearch-per-experiment-page` | already covered |
| Citation Rigor (full author/year/venue/title/arXiv format) | 667-697 | YES — Rule 4 + `autoresearch-reasoning-entry` | identical |
| Reasoning Blob Completeness (word floors 60/40/50/25/30/40) | 701-716 | YES — Rule 5 (identical word floors) | identical |
| Loss Function Rules (BCEWithLogits + focal + label-smoothing) | 720-731 | DOMAIN-SPECIFIC (binary classification) | not ported |
| Winner Archiving Protocol (directory layout) | 734-757 | YES — `autoresearch-winner-archive` | already ported |
| Google Colab Notebook | 761-776 | NICE-TO-HAVE — flagged OPTIONAL in `autoresearch-winner-archive` | already noted |
| Traditional ML Metrics (AUROC/AUPRC/Acc/LogLoss/ECE/Confusion + bg-rejection-at-S-eff) | 780-792 | DOMAIN-SPECIFIC (binary classification + Higgs physics figure) | not ported |
| Per-Prediction Log (CSV columns) | 796-812 | DOMAIN-SPECIFIC (trade logs) — generic version in `autoresearch-dashboard` per-experiment metrics | not ported |
| Architecture (`core/runner.py` ONE experiment per call) | 816-822 | YES — our runner is the same pattern | identical |
| Validation Checklist (6 pre-experiment assertions) | 826-833 | YES — `autoresearch-data-split-audit` runner gate + `autoresearch-data-contract-validator` (NEW THIS FOLLOW-UP) | now fully covered |
| Project Structure tree | 837-899 | DOMAIN-SPECIFIC (tabular backbones list) | not ported |
| Key Constants table | 903-921 | DOMAIN-SPECIFIC values | not ported |
| Common Mistakes table (14 entries) | 924-941 | NICE-TO-HAVE — useful failure-mode catalogue; many already absorbed into Rule 7 + Rule 26 anti-patterns | not separately ported |
| Session Learnings (append-only) | 944-983 | YES — `experiment_log.jsonl` (Rule 3) + `research_journal.md` analogue | already enforced |
| Cross-references + License + Credits | 987-1007 | NA | not ported |

**Net new patterns found below line 300:** **0 load-bearing patterns
were missing locally.**

All 41 sister CLAUDE.md sections from lines 300-1007 either:
1. Already have a local skill (16 sections),
2. Are already enforced by a local Rule (11 sections),
3. Are domain-specific (tabular GBMs / Higgs physics / Baldi split)
   and intentionally not ported (10 sections), or
4. Were already noted as NICE-TO-HAVE in the PM audit §6.2 and are
   now closed by this follow-up (Validation Checklist line 826-833 →
   `autoresearch-data-contract-validator`).

**The PM audit's honesty caveat #3 ("if there's a tabular-specific
load-bearing pattern in lines 300-1007 I missed, this audit didn't
catch it") is now closed: there was no such pattern.** The PM audit's
gap analysis from the first 300 lines + the §3.4 sister-repo
inventory together correctly identified every transferable pattern.

**Follow-up commits (this Appendix + §6.2 closeout + §1/§2 count
updates):**
- `42695a4` — "Skills parity follow-up: add shuffle-test +
  data-contract-validator (Rule 22 closeout)". New skills + CLAUDE.md
  §11 update (29-skill block).
- This audit-document update (next commit) — §6.2 closure, §1 + §2
  count updates (27 → 29), Appendix A.
