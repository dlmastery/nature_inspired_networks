---
name: autoresearch-data-split-audit
description: Use when wiring up a new dataset OR before launching ANY experiment on a dataset whose splits drive an external claim (OOD generalisation, benchmark-comparable test set, fixed-fold protocol). Runs a multi-auditor triple-check on the split — disjointness, protocol-conformance, class balance, size floors, reproducibility fingerprint, no-leakage-via-metadata — and writes a machine + human-readable audit report. The runner refuses to launch if the audit is missing, stale, or red. There is no `--bypass-audit` flag. Ported from autoresearchimage (Camelyon17 hospital split) and autoresearchtabular (Higgs Baldi 2014 frozen split).
metadata:
  rules_enforced: [7, 13, 19]
  added: 2026-05-29
  origin: dlmastery/autoresearchimage + dlmastery/autoresearchtabular
---

# Skill — Triple-check data-split audit gate

## When to use

- Wiring up a NEW dataset into the project (image / tabular / graph /
  sequence — any modality).
- BEFORE the first experiment on any dataset whose splits drive an
  external claim — OOD AUC, benchmark-comparable test, fixed-fold
  protocol (Baldi 2014 Higgs, WILDS Camelyon17 hospital, etc.).
- After ANY edit to the dataset loader, split config, transforms, or
  data-mode switch.
- On every session start that intends to run experiments — the audit
  is the canary that catches silent regressions.
- After a crash recovery — state may be inconsistent.

## Why

A single leaked row, slide, or hospital can mask the entire benchmark
claim. For datasets whose POINT is cross-distribution generalisation
(Camelyon17 hospitals 0–4) or whose published numbers all use a single
frozen split (Baldi 2014 Higgs last 500k = test), sloppy splits are
not a bug — they're a publication retraction.

This skill enforces the rule the sister `autoresearchimage` and
`autoresearchtabular` repos discovered: **the runner refuses to launch
without a green audit fingerprint** matching the live data loader.
There is no `--bypass-audit` flag. Period.

## The N auditors

Implement each as a callable in the project's evaluation module (e.g.,
`src/<pkg>/eval/audit.py` or `core/evaluation/audit.py`). Each returns
a status (`PASS` / `FAIL`), a violation list, and a SHA-256 fingerprint
of its canonical artefact.

### Core disjointness auditors (every dataset)

1. **`audit_index_disjoint`** — every row / sample / patch index
   appears in at most one fold. Pairwise intersection of all (train,
   val, test, ood_val, ood_test) sets must be empty. Assertion form:
   `set(train) ∩ set(val) == ∅`, and the symmetric pairings.
2. **`audit_protocol_match`** — the canonical published protocol for
   this dataset is honoured within tolerance. Examples:
   - Higgs UCI: last 500k rows = test, prev 500k = val, rest = train
     (Baldi, Sadowski & Whiteson 2014 Nature Comm). Sizes must match
     within ±0.5%.
   - Camelyon17: hospitals 0,1,2 → train+id_val; hospital 3 → val_ood;
     hospital 4 → test_ood. NO hospital may appear in two folds.
   - CIFAR-10: 50000 train / 10000 test verbatim from torchvision.
3. **`audit_size_floors`** — every fold meets a minimum row count
   (configurable; e.g., `train ≥ 9.5M`, `val ≥ 450k`, `test ≥ 450k`
   for Higgs).

### Sanity auditors

4. **`audit_class_balance`** — every classification fold has both
   classes present (AUROC is otherwise undefined) AND positive
   prevalence ∈ `[low, high]` for the dataset's known range.
5. **`audit_no_leakage_via_metadata`** — confirms columns used for
   split assignment (hospital ID, slide ID, date, row index) are NOT
   among the model's inputs. No peeking at split labels at inference.
6. **`audit_feature_consistency`** — same feature names + dtypes
   across splits; no NaN / Inf in feature columns; standardisation (if
   applied) uses TRAIN-set statistics only (computing mean/std from
   val/test is leakage).

### Modality-specific extras

For images (especially WSI / multi-slide datasets):

7. **`audit_slide_level`** — every `slide_id` appears in at most one
   fold; pairwise slide intersections empty. Patches drawn from the
   same slide cannot be split across train and test.
8. **`audit_subgroup_level`** — every subgroup (hospital, country,
   day, source-domain) appears in exactly the folds the spec assigns
   to it.

For time series / financial:

7'. **`audit_temporal_order`** — train end-time < val start-time and
   val end-time < test start-time, with the project's required purge
   gap and embargo (e.g., 90-day purge + 21-day embargo + 10-day
   label-horizon buffer for FX walk-forward).
8'. **`audit_no_lookahead`** — no feature in row[T] uses information
   from any time > T. Verified by re-deriving each feature using only
   the lagged window and checking equality.

For graphs:

7''. **`audit_node_edge_disjoint`** — depending on the split mode
    (transductive / inductive), confirm either edge-set or node-set
    disjointness across folds, with the chosen mode logged.

### Reproducibility auditor (every dataset)

9. **`audit_reproducibility`** — running the audit twice with the
   same seed must produce identical `(sizes, sample_set_fingerprints,
   feature_names)` tuples, captured as a SHA-256. Any change between
   runs is a regression — most likely a non-deterministic loader.

## Output artefacts

The audit writes to a project-relative path (recommended
`<results_root>/audits/data_split_audit/`):

| file | content | format |
|---|---|---|
| `data_split_audit.json` | machine-readable per-auditor results | JSON |
| `data_split_audit.md` | human-readable summary + violations | Markdown |
| `data_split_audit_fingerprint.json` | SHA-256 of the canonical split | JSON |

The `.md` report is checked into git so a reviewer 6 months later can
read it without re-running the audit.

## Runner gate (the load-bearing rule)

The project's training runner MUST call `audit_or_die()` before any
model build. Pseudocode:

```python
def audit_or_die(cfg, results_root) -> None:
    audit_path = results_root / "audits" / "data_split_audit.json"
    if not audit_path.exists():
        raise SystemExit("REFUSED: data-split audit missing. Run "
                         "`python -m <pkg>.eval.audit --config <cfg>`.")
    audit = json.loads(audit_path.read_text())
    if audit["age_hours"] > 24:
        raise SystemExit("REFUSED: data-split audit is stale (> 24 h).")
    failed = [a for a in audit["auditors"] if a["status"] != "PASS"]
    if failed:
        raise SystemExit(f"REFUSED: auditor(s) failed: "
                         f"{[a['name'] for a in failed]}. "
                         f"Read {results_root}/audits/data_split_audit.md.")
    live_fp = compute_split_fingerprint(load_data(cfg))
    if live_fp != audit["fingerprint"]:
        raise SystemExit(
            f"REFUSED: live data fingerprint {live_fp} != "
            f"audited fingerprint {audit['fingerprint']}. Re-run audit."
        )
```

`SystemExit` (not return-code-1) so calling shell scripts surface the
violation. There is no `--bypass-audit` flag — fixing the violation is
the only path forward.

## When to re-run the audit

| event | re-run? |
|---|---|
| Session start | YES (cheap; catches drift) |
| Edit to dataset loader, transforms, split config | YES |
| Switch `--data-mode` (sim ↔ real, subset ↔ full) | YES |
| Upgrade dataset library version (wilds, torchvision, etc.) | YES |
| Crash mid-experiment | YES (state may be inconsistent) |
| Within an unchanged session, run #2+ on same data | NO (fingerprint will match) |

## Audit failure protocol

When an auditor reports `FAIL`:

1. **Read the violation list** in `data_split_audit.md` — it names
   the specific assertion and counter-example.
2. **DO NOT silence the audit.** The audit is the canary, not the bug.
   Suppressing the auditor with a try/except is a Rule-7-violating
   `--bypass`.
3. **Fix the underlying** loader / config / metadata / standardisation
   mismatch.
4. **Re-run the audit** until green.
5. **Commit the fix + the green audit report together** so the green
   row in git history is traceable to the patch.

## What "good" looks like

- The `data_split_audit.md` report opens with a single PASS/FAIL
  banner and a 6-row table of (auditor, status, fingerprint snippet).
- The fingerprint is short (12 hex chars) and stable across re-runs.
- The runner's refusal message names the specific failed auditor.
- A pre-commit hook (optional but recommended) refuses to commit
  changes to `data/loader.py` without a fresh green audit.

## Anti-patterns

- **Silencing an auditor with `# noqa` or a try/except.** That's a
  `--bypass` by another name.
- **Lowering a size floor** to make a `FAIL` pass. Either the dataset
  shrank legitimately (update the floor in a config and document why)
  or you broke the loader.
- **"I'll add the audit later"** for "just-prototyping" experiments —
  the prototyping runs are the ones most likely to bake the bug into
  later analysis. Run the audit from day one; it's cheap.
- **Computing standardisation stats from val/test** because it gave
  better numbers. That's data leakage, period.
- **Treating Kaggle's public test as your held-out evaluator.** It's
  not held-out if you tuned hyperparameters on it.

## Cross-references

- [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md) —
  the runner's `audit_or_die()` precedes the 7-step ritual.
- [`autoresearch-dataset-loader`](../autoresearch-dataset-loader/SKILL.md)
  — implementing the loader; the audit auditors live next to it.
- [`autoresearch-reasoning-entry`](../autoresearch-reasoning-entry/SKILL.md)
  — pre-run reasoning entry should mention the audit fingerprint as
  the provenance of the train/val/test split.
- CLAUDE.md Rule 7 — "no `--bypass` flag".
- CLAUDE.md Rule 13 — "SOTA smoke first"; the audit precedes the smoke.
