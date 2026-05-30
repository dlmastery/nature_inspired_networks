---
name: autoresearch-explainability-report
description: Use whenever a new champion is archived (via autoresearch-winner-archive) and an external claim is about to be made about it. Produces a 14-section data-scientist-grade audit report covering executive summary, feature importance (permutation), top-feature analysis, SHAP-style local explanations, per-fold drift, calibration, uncertainty sanity, prediction distribution, per-trade attribution (where applicable), risk audit, data-pipeline reassertion, model-config dump, known-limitations, deployment checklist. Ported from autoresearch (FX), autoresearchspy (SPY), and autoresearchindexstock (QQQ) where the rule is "a model without explainability is un-deployable".
metadata:
  rules_enforced: [22, 28]
  added: 2026-05-29
  origin: dlmastery/autoresearch (Explainability & Auditability Report)
---

# Skill — 14-section explainability + audit report

## When to use

- Whenever [`autoresearch-winner-archive`](../autoresearch-winner-archive/SKILL.md)
  is invoked — the report lives at the archive's `audit_report.md`.
- Whenever an external claim references a model (FINDINGS headline,
  PAPER abstract, README badge, blog post). The dual-track audit
  (Rule 22) requires both the implementation-critic verdict AND this
  report to be green before the claim ships.
- Before any deployment decision (production trading, clinical use,
  shipping in a product). A trading or clinical model without
  explainability is un-deployable, per the sister-repo discipline.

## Why

The sister `autoresearch` (FX) project codified this rule after losing
a multi-week investigation to a champion whose number turned out to be
driven by ONE leaked feature. The 14 sections force the explainability
+ auditability surface to be measured BEFORE the model goes external.

The four hardest failures this catches:

1. **Single-feature-driven wins** (one column carries the headline; the
   model is a dressed-up if/else).
2. **Per-regime brittleness hidden by aggregates** (test AUC 0.95 hides
   that fold 3 is 0.55).
3. **Mis-calibration** (the model is confident at 0.9 but accurate at
   0.6).
4. **Data-pipeline regressions** (the supposed test set quietly
   started including some training rows).

## The 14 sections (mandatory, in this order)

The report lives at `<winner-archive>/audit_report.md`. Each section
opens with a one-line PASS/FLAG status banner so the reader can scan
the report in 30 seconds.

### 1. Executive summary

- Headline metric (composite, top-1, AUC, Sharpe — whatever the
  project chose).
- Per-fold / per-regime breakdown table.
- Status banner: GREEN (ready) / AMBER (caveat) / RED (do not
  deploy).
- One paragraph: what this model is, what it does, where it fails.

### 2. Feature importance (permutation method)

For every input feature, shuffle that column in the test set and
re-evaluate. Record the drop in the headline metric. Rank by drop.

- Citation: Breiman 2001 'Random Forests' (variable importance
  section).
- Output: `feature_importance.csv` next to the report, with columns
  `[feature_name, metric_drop, rank, domain_category]`.
- Status: GREEN if the top feature contributes ≤ 30% of the total
  drop; AMBER 30–60%; RED > 60% (the model is a dressed-up single
  feature).

### 3. Top-N feature analysis

For the top 10 most-impactful features, explain:

- What the feature measures (cite the dataset card or feature module
  docstring).
- Why it matters economically / clinically / physically (domain
  rationale).
- Per-fold impact: is feature X strong in regime A but weak in B?

This section is where domain expertise lands. Reviewers read this to
decide whether the model has learned signal or learned to memorise.

### 4. SHAP-style local explanations

For 10 random test-set predictions, compute per-feature contribution.
If full SHAP is too expensive, use `gradient × input` (Simonyan, Vedaldi
& Zisserman 2014 'Deep Inside Convolutional Networks' arXiv:1312.6034)
or Integrated Gradients (Sundararajan, Taly & Yan 2017 'Axiomatic
Attribution for Deep Networks' arXiv:1703.01365) as cheap
approximations.

- Output: `shap_local.csv` — one row per (sample, feature) pair.
- Status: GREEN if the explanations align with domain intuition on
  ≥ 8/10 samples; AMBER 5–7; RED < 5.

### 5. Per-fold / per-regime feature drift

For each fold, compute mean / std of each feature vs the training set.
Features with Z-score > 2 indicate distribution shift. Report the top
5 drifted features per fold with a one-line explanation.

- Status: GREEN if drifted features make domain sense (e.g., VIX is
  drifted in the GFC fold); RED if a "stable" feature drifts on a
  fold (probably a loader bug).

### 6. Calibration analysis

- Plot predicted-quantile vs realised-mean. Ideal: monotonic.
- Report calibration error (mean absolute deviation from the
  monotonic ideal).
- Citation: Guo, Pleiss, Sun & Weinberger 2017 ICML 'On Calibration of
  Modern Neural Networks' (arXiv:1706.04599).
- Status: GREEN if ECE ≤ 0.02; AMBER 0.02–0.05; RED > 0.05.

### 7. Uncertainty sanity

- Plot aleatoric vs prediction absolute error — should be monotonic.
- Plot confidence vs hit-rate — should be monotonic.
- Citation: Kendall & Gal 2017 NeurIPS 'What Uncertainties Do We Need
  in Bayesian Deep Learning for Computer Vision?' (arXiv:1703.04977).
- Status: GREEN if both curves are monotonic ≥ 70% of buckets;
  RED if either inverts.

### 8. Per-regime / per-fold prediction distribution

For each fold, plot a histogram of predictions. Identify if the model
is systematically biased (always predicting near a constant) vs
appropriately reactive.

- Status: GREEN if histogram width per fold ≥ 50% of train histogram
  width; RED if any fold collapses to a near-constant.

### 9. Per-trade / per-sample attribution

Decompose cumulative test performance:

- For classification: top-5 confidently-right and top-5 confidently-
  wrong predictions. Pattern analysis.
- For trading: top-5 winning trades and top-5 losing trades per fold
  (date, symbol, predicted, actual, P&L).

Status: GREEN if losses are not concentrated on a single date /
sample / regime; RED if > 30% of losses come from one fold-day.

### 10. Risk audit

- Max drawdown (trading) OR worst-case prediction error (classification).
- VaR-95 / CVaR-95 per fold (trading) OR error distribution tails per
  fold (classification).
- Skewness, kurtosis of returns / residuals.

Status: GREEN if drawdown < project's risk budget; RED if a single
trade / sample could exceed budget.

### 11. Data-pipeline audit (re-assertion)

Re-run the [`autoresearch-data-split-audit`](../autoresearch-data-split-audit/SKILL.md)
on the winner's data and paste the output verbatim. Confirm:

- Zero train / val / test overlap.
- Purge gap / embargo / label-horizon buffer (where applicable).
- Standardisation stats from train only.

Status: GREEN if all auditors PASS with matching fingerprint; RED if
any FAIL.

### 12. Model + config complete dump

Every hyperparameter + Python version + torch version + numpy version
+ random seed. Reviewer should be able to reproduce from this section
alone, even if `config.json` is missing.

Include the composite formula SHA-256 fingerprint here so a reader
knows the headline number was computed against a frozen formula.

### 13. Known limitations and risks

- What regimes / distributions has the model NEVER been tested on
  (hyperinflation, CB digital currencies, pathology slides from a
  hospital with a different scanner, etc.)?
- Where will it most likely fail in deployment?
- What's the seed variance (single-seed cherry vs robust)?

Status: GREEN if limitations are bounded; RED if any limitation
overlaps with the intended deployment surface.

### 14. Deployment checklist

- Monitoring: what metrics to track in production (input drift,
  prediction drift, calibration drift)?
- Kill-switch criteria: max drawdown threshold, consecutive loss
  count, distribution-shift flag.
- Retraining cadence: monthly / quarterly / triggered.
- Reference to `inference/predict.py` in the same archive.

Status: GREEN if every checklist item has a concrete plan; AMBER if
2+ items are TBD; RED if no monitoring plan exists.

## Implementation note

The report can be generated by a script — recommended pattern:

```bash
python -m <pkg>.audit.explainability \
    --winner-dir winners/<tag>_exp<N>_<desc>/ \
    --out winners/<tag>_exp<N>_<desc>/audit_report.md
```

The script:

1. Loads the winner's model + scaler.
2. Computes the 14 sections via the same evaluator the runner uses.
3. Writes the report + sidecar CSVs (`feature_importance.csv`,
   `shap_local.csv`).
4. Exits 1 if any section is RED so CI catches un-deployable winners.

For projects that don't have a script yet, the report can be
hand-authored, but every section's status banner must be backed by a
sidecar file (e.g., `feature_importance.csv`) showing the numbers.

## What "good" looks like

- The report opens with the status banner; reader sees in 5 seconds
  whether the model is deployable.
- Every section has the citation footnote, the sidecar CSV, and the
  domain-language interpretation paragraph.
- The "Known limitations" section is HONEST. A report that says "no
  limitations" has not done the work.
- The deployment checklist references concrete monitoring metrics
  (NOT "we'll figure out monitoring later").

## Anti-patterns

- "Feature importance ranked by gain alone" without permutation —
  gain rewards features the tree split on early, not features that
  generalise.
- SHAP on 10 cherry-picked test points that look good — random
  sampling is mandatory.
- Calibration plot with 10 buckets and "looks monotonic-ish" — report
  the ECE number.
- "Risk audit: max drawdown is fine" without naming the drawdown
  band.
- Treating GREEN as the default — if you find yourself green-stamping
  without computing, you've broken the rule.

## Cross-references

- [`autoresearch-winner-archive`](../autoresearch-winner-archive/SKILL.md)
  — the report lives at the archive's `audit_report.md`.
- [`autoresearch-data-split-audit`](../autoresearch-data-split-audit/SKILL.md)
  — Section 11 re-runs this audit.
- [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md)
  — statistical-rigor floor; this skill's calibration + drift
  sections feed into the paper-rigor self-audit.
- [`autoresearch-critic-team`](../autoresearch-critic-team/SKILL.md)
  + [`autoresearch-scicritic-team`](../autoresearch-scicritic-team/SKILL.md)
  — dual-track audit (Rule 22); this report is the third leg.
- CLAUDE.md Rule 22 — dual-track audit before any external claim.
- CLAUDE.md Rule 28 — leaderboard reflects the screening tier vs
  evaluation tier honestly.
