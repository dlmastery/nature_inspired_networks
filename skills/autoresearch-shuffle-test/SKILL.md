---
name: autoresearch-shuffle-test
description: Semantic leakage detection by shuffling targets in train and re-evaluating — if the model's validation score moves toward chance, the train→val alignment is real; if it stays at non-chance, the eval-side is reading the targets through some leakage channel. Catches the bug class the structural data-split audit cannot see: target-encoded features, label-dependent augmentation, inadvertent label echoes in the feature matrix, off-by-one alignment that lets the evaluator "see" a shifted version of y. Ported from autoresearch (FX) §3.5 where a shuffle test caught a +8.78-Sharpe alignment bug that the structural split audit had stamped green.
metadata:
  type: skill
  group: autoresearch-rigor
  rules_enforced: [7, 22, 28]
  added: 2026-05-30
  origin: dlmastery/autoresearch (FX §3.5 shuffle audit)
---

# Skill — Shuffle-test audit (semantic leakage gate)

## When to use

- **Before any external claim** that names a model: FINDINGS headline,
  PAPER abstract, README badge, blog post, or deployment decision.
  Dual-track audit (Rule 22) is implementation-critic + sci-critic +
  data-split structural + **shuffle test** (this skill). All four must
  be green.
- **When tabular / time-series alignment is non-trivial** — groupby
  aggregations, walk-forward CV, custom block splits, leave-one-group-
  out. These are the alignment bugs that the structural audit
  (`autoresearch-data-split-audit`) cannot detect.
- **When cross-validation uses custom split logic** that the project
  authored (not a vetted `sklearn.model_selection` splitter).
- **When the structural audit is green but the headline number "feels
  too good"** — the +8.78 Sharpe gap on FX was the canary. A win that
  exceeds the published SOTA by > 5% relative is automatically
  shuffle-test-required.
- **After any change to the feature-engineering pipeline** that touches
  the target column (target encoding, leave-one-out encoding, k-fold
  target encoding, rank-encoding by target).
- **After a fixer-campaign patch** to any data-loader, augmenter, or
  evaluator module — the shuffle test is the re-smoke gate.

## Why

The structural data-split audit (`autoresearch-data-split-audit`)
catches the bug class **"a row appears in both train and val"** —
overlap, hospital leakage, slide-ID leakage, temporal lookahead. It
**cannot** catch the bug class **"the evaluator is reading the target
through a feature channel"**:

1. **Target column leaked into the feature matrix.** A column named
   `target_shifted` was kept by accident; structural audit sees zero
   row overlap and stamps green; eval AUC is 0.999.
2. **K-fold target encoding computed on the full dataset** instead of
   train-only. Every val row's feature vector encodes the val row's
   own target. Eval is meaningless.
3. **Augmentation policy is label-conditional.** Class-A images get
   one transform, class-B images get another, and the magnitude of
   the transform leaks the label.
4. **Off-by-one alignment between training and evaluator.** Training
   pairs `(x[i], y[i])` and evaluator pairs `(x[i], y[i+1])`; on a
   dataset with autocorrelated targets, this leaks future information.
   This is the FX §3.5 bug — +8.78 Sharpe before fix, 0.0 Sharpe
   after.
5. **Inadvertent label echo in derived features.** A `mean_target`
   feature computed over a window that included the row itself.

The shuffle test is the canonical semantic-leakage detector: **permute
the training labels, retrain from scratch, evaluate on the unshuffled
validation set. The aggregate validation metric MUST collapse to the
chance baseline.** If it doesn't, the evaluator is reading the targets.

## The 3 shuffle modes

### Mode A — Hard shuffle (default)

Random permutation of `y_train` with a fixed seed. Refit the model.
Evaluate on the **untouched** validation set.

**Expected outcome.** Aggregate val metric drops to the chance
baseline:

- Binary classification: AUROC → 0.50 ± 0.02; AUPRC → positive-prevalence
- Multi-class classification: accuracy → 1/n_classes ± 0.02
- Regression: R² → 0.0 ± 0.05; MSE → var(y_val)
- Trading: Sharpe → 0.0 ± 0.2; cumulative return → 0 ± noise

**Verdict.**
- **PASS** — shuffled-val metric within tolerance of chance baseline.
  The original headline number reflects a real (train, val) alignment.
- **WEAK** — shuffled-val metric is > 2× chance baseline distance but
  < 5×. Investigate: a small feature is encoding the target weakly.
- **FAIL** — shuffled-val metric is > 5× chance baseline distance OR
  is statistically indistinguishable from the unshuffled headline.
  STOP. The evaluator is reading the targets. Fix the alignment /
  remove the leaking feature / re-audit before any external claim.

### Mode B — Within-group shuffle (grouped CV)

For grouped / blocked CV (leave-one-group-out, leave-one-hospital-out,
leave-one-subject-out): permute `y_train` **within each group**, then
retrain.

**Why this mode exists.** Hard-shuffle Mode A on grouped data may
artificially destroy class-group correlations and produce a false
PASS. Within-group shuffle preserves the group-level class prior and
forces the model to find within-group signal — if no within-group
signal exists, the within-group shuffled metric drops to chance.

**Expected outcome.** Same chance baselines as Mode A; PASS / WEAK /
FAIL boundaries identical.

### Mode C — Block shuffle (time-aware)

For time-series / walk-forward CV: permute `y_train` **within each
walk-forward fold**, then refit. Evaluate on the unshuffled future
fold(s).

**Why this mode exists.** Hard-shuffle on time-series destroys the
autocorrelation structure that ALL time-series models exploit; a
PASS under hard-shuffle is essentially "this is not a constant model"
not "this model uses real target alignment". Block-shuffle preserves
within-fold autocorrelation and forces the test to be about the
**predictive direction** of the alignment.

**Expected outcome.** Same chance baselines per fold; the verdict
table is per-fold.

## Operational pattern

```python
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle


def shuffle_test(
    fit_fn,                    # callable: (X_train, y_train) -> fitted model
    score_fn,                  # callable: (model, X_val, y_val) -> scalar metric
    X_train, y_train,
    X_val,   y_val,
    *,
    mode: str = "hard",        # "hard" | "within_group" | "block"
    groups=None,               # array-like, required for within_group / block
    n_repeats: int = 3,        # average over seeds for stability
    seed: int = 0,
    chance_baseline: float | None = None,  # None → auto-derive
    tolerance: float = 0.05,   # PASS if |shuffled - chance| <= tolerance
):
    """Permute y_train, refit, evaluate on (X_val, y_val). Return verdict."""
    if chance_baseline is None:
        chance_baseline = _auto_chance_baseline(y_val, score_fn)

    real_score = score_fn(fit_fn(X_train, y_train), X_val, y_val)

    shuffled_scores = []
    rng = np.random.RandomState(seed)
    for r in range(n_repeats):
        if mode == "hard":
            y_shuf = shuffle(y_train, random_state=rng)
        elif mode == "within_group":
            y_shuf = _shuffle_within_groups(y_train, groups, rng)
        elif mode == "block":
            y_shuf = _shuffle_within_blocks(y_train, groups, rng)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        shuffled_scores.append(score_fn(fit_fn(X_train, y_shuf), X_val, y_val))

    shuffled_mean = float(np.mean(shuffled_scores))
    shuffled_std  = float(np.std(shuffled_scores))
    distance      = abs(shuffled_mean - chance_baseline)

    verdict = (
        "PASS" if distance <= tolerance
        else "WEAK" if distance <= 5 * tolerance
        else "FAIL"
    )
    return {
        "verdict": verdict,
        "real_score": real_score,
        "shuffled_mean": shuffled_mean,
        "shuffled_std":  shuffled_std,
        "chance_baseline": chance_baseline,
        "distance_to_chance": distance,
        "tolerance": tolerance,
        "n_repeats": n_repeats,
        "mode": mode,
    }
```

The auto chance baseline:

- Binary classification, score = AUROC → `0.5`.
- Binary, score = accuracy → `max(p, 1-p)` where `p = mean(y_val)`.
- Multi-class, score = accuracy → `max(class_priors)`.
- Regression, score = R² → `0.0`.
- Trading, score = Sharpe → `0.0`.

## Where the result lands

The shuffle-test report is a sidecar file next to the explainability
report inside the winner archive:

- `winners/<tag>_exp<N>_<desc>/shuffle_test.json` — machine-readable
  per-mode results (verdict, real_score, shuffled_mean, baseline, etc.).
- `winners/<tag>_exp<N>_<desc>/shuffle_test.md` — human-readable
  summary with the PASS / WEAK / FAIL banner and one-line per mode.

`autoresearch-explainability-report` Section 11 ("Data-pipeline audit
re-assertion") MUST cite the shuffle-test verdict alongside the
structural split-audit fingerprint. A green structural audit + RED
shuffle test = the champion is NOT deployable.

## Verdict tiers (binding)

| Verdict | Criterion | Action |
|---|---|---|
| PASS | shuffled metric within tolerance of chance baseline on all required modes | OK to ship the external claim |
| WEAK | 2× – 5× tolerance distance; not statistically indistinguishable from chance but not at chance | Investigate the largest feature-importance contributor; the most likely culprit is a column with a partial label echo |
| FAIL | shuffled metric > 5× tolerance OR within 2 stddev of the real (unshuffled) score | STOP. No external claim. Fix the leak; re-audit |

For grouped/temporal datasets, the **Mode B** or **Mode C** verdict
overrides the Mode A verdict — hard-shuffle on grouped/temporal data
is unreliable.

## Anti-patterns

- **Shuffling only `y_test`** instead of `y_train`. Shuffling `y_test`
  catches a different bug class (the evaluator is doing something
  pathological) but does NOT catch the canonical leakage bug. Shuffle
  `y_train`, refit, evaluate on the **unshuffled** validation.
- **Single-shuffle verdict.** One permutation has too much variance.
  Use `n_repeats ≥ 3` and average. A single shuffled-AUC of 0.52 looks
  like PASS but a 3-repeat mean of 0.56 ± 0.04 is WEAK.
- **Wrong chance baseline.** On imbalanced classification, the chance
  AUROC is still 0.5, but the chance accuracy is `max(p, 1-p)`, not
  `0.5`. Auto-derive from the val distribution.
- **Shuffling features instead of targets.** Permuting `X_train` rows
  destroys the row-feature pairing for the model but preserves the
  global `(X, y)` joint distribution — that's a different audit
  (sanity-check on feature dependence), not a leakage test.
- **Skipping the shuffle test "because the data-split audit is green"**.
  The structural audit is necessary but NOT sufficient. The FX §3.5
  alignment bug had a green structural audit for 3 weeks before the
  shuffle test caught it.
- **Hard-shuffle on grouped data without checking Mode B.** False PASS
  is the failure mode — the hard-shuffle destroys group-class
  correlations and looks like the model lost its signal when actually
  it lost the group prior. Use Mode B for grouped CV.
- **Hard-shuffle on time-series without checking Mode C.** Same false
  PASS — autocorrelation is what time-series models exploit; hard-
  shuffle destroys it.
- **Stopping at PASS without recording the shuffled metric.** The
  shuffled metric value is the audit artefact. Future Claude sessions
  re-run with the same seed and expect the same shuffled metric — if
  it drifts, the loader changed.
- **Treating shuffle-test failure as a hyperparameter to tune around**.
  A FAIL verdict means **the alignment is wrong**. Tuning the model
  on a leaky pipeline is fitting to a corrupted signal.

## Cross-references

- [`autoresearch-data-split-audit`](../autoresearch-data-split-audit/SKILL.md)
  — structural leakage audit; the shuffle test is the **semantic**
  complement. Both must be green.
- [`autoresearch-data-contract-validator`](../autoresearch-data-contract-validator/SKILL.md)
  — `(x, y)` pairing contract; off-by-one alignment is the canonical
  bug that data-contract + shuffle-test both detect (one statically,
  one empirically).
- [`autoresearch-explainability-report`](../autoresearch-explainability-report/SKILL.md)
  — Section 11 cites the shuffle-test verdict; the report cannot be
  green if shuffle test is RED.
- [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md)
  — statistical-rigor floor for external claims; shuffle-test PASS
  is one of the gating criteria.
- [`autoresearch-winner-archive`](../autoresearch-winner-archive/SKILL.md)
  — `shuffle_test.{json,md}` lives inside the archive alongside the
  audit report.
- [`autoresearch-critic-team`](../autoresearch-critic-team/SKILL.md)
  + [`autoresearch-scicritic-team`](../autoresearch-scicritic-team/SKILL.md)
  — dual-track audit (Rule 22); shuffle test is the fourth leg
  (implementation-critic, sci-critic, structural audit, semantic
  audit).
- CLAUDE.md Rule 7 — "no `--bypass` flag"; shuffle-test FAIL has no
  bypass.
- CLAUDE.md Rule 22 — dual-track audit before any external claim.
- CLAUDE.md Rule 28 — leaderboard reflects the screening vs evaluation
  tier honestly; shuffle test is the canary that the evaluation tier's
  number is real, not a leak.

## Provenance

- **FX §3.5 retrospective.** A tree champion on FX showed +8.78 Sharpe
  uplift over the prior best. The structural data-split audit was
  green. The shuffle-test was added retrospectively and FAILED
  immediately (shuffled Sharpe = +7.2 vs chance baseline 0.0).
  Root cause: an off-by-one alignment between `seg_tgt.values[seq_len:]`
  (training) and the evaluator's `[seq_len-1:]`. After fix: real Sharpe
  collapsed to ~0.1, the +8.78 evaporated, and ~3 weeks of downstream
  experiments had to be discarded. The shuffle test is the lesson.
