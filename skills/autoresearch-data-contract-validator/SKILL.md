---
name: autoresearch-data-contract-validator
description: Assert that the training pipeline's (x_train, y_train) pairing matches the evaluator's (x_val, y_val) pairing in shape, dtype, label encoding, value ranges, and modality-specific invariants. Catches the bug class where training uses one labelling / shaping convention and the evaluator silently uses another — the off-by-one alignment, the inverted-channel-order, the dropped-column, the swapped-label-class-zero/one. Ported from autoresearch (FX) §3.5 where this contract would have caught the alignment bug retrospectively three weeks earlier than the shuffle test did. The runner refuses to launch if the contract fails. There is no bypass flag.
metadata:
  type: skill
  group: autoresearch-rigor
  rules_enforced: [7, 22]
  added: 2026-05-30
  origin: dlmastery/autoresearch (FX §3.5 off-by-one alignment audit)
---

# Skill — `(x, y)` pairing data-contract validator

## When to use

- **Anywhere the training-side data loader and the eval-side data
  loader live in different functions / files / processes.** This is
  the single most common bug surface in autoresearch: a refactor moves
  the eval loader to a new module and silently drops a transform, or
  the eval-loader uses a different indexing convention. The data
  contract is the static assertion that keeps the two in sync.
- **Before any deployment, paper submission, or external claim.** The
  contract validator runs as part of the runner's pre-flight (same
  layer as `audit_or_die()` from
  [`autoresearch-data-split-audit`](../autoresearch-data-split-audit/SKILL.md)).
- **After any change to the data loader, transforms, augmentation
  pipeline, or evaluator code path.** The contract validator's failure
  mode is the loud one — it refuses to launch — which is the right
  failure mode for a class of bugs that otherwise silently corrupts
  every downstream artefact.
- **Before the SOTA-smoke pre-flight (Rule 13).** If the contract
  fails, the SOTA smoke is meaningless because train and eval don't
  share a contract.

## Why

The structural data-split audit (`autoresearch-data-split-audit`)
asserts **rows don't overlap across folds**. The shuffle-test audit
(`autoresearch-shuffle-test`) asserts **the evaluator isn't reading
the target through a leak**. Neither catches the bug class:

> "Training expects `x ∈ float32[N, 224, 224, 3]` with labels in
> `{0, 1, ..., 9}`. Evaluator silently provides `x ∈ uint8[N, 3, 224,
> 224]` with labels in `{1, ..., 10}`."

The model trains fine, the evaluator runs fine, and the headline
number is GARBAGE. The bug is not in the data, the splits, or the
features — it's in the **contract** between the two halves of the
pipeline. Three concrete real-world variants of this bug:

1. **Off-by-one alignment.** Training pairs `(x[i], y[i])`; eval
   pairs `(x[i], y[i+1])`. On any dataset with autocorrelated y, eval
   gets future-information leak. This is the FX §3.5 bug (+8.78
   Sharpe → 0.0 Sharpe after fix).
2. **Channel-order divergence.** Train uses `NCHW`, eval uses `NHWC`,
   numpy silently broadcasts to a degenerate shape and the model's
   eval predictions are noise.
3. **Label-encoding divergence.** Train uses `{0, 1}`, eval uses
   `{-1, +1}`, the metric library treats `-1` as a class label and
   accuracy is 0 — but the user thinks the model is broken, not the
   loader.

The contract validator runs ONCE at runner startup and ONCE more on
the first batch from each loader, asserts the contract holds, and
refuses-to-launch on any violation.

## The contract (what is asserted)

For every (training loader, eval loader) pair, the validator asserts:

| Property | Assertion | Failure mode it catches |
|---|---|---|
| Feature shape (excluding batch dim) | `x_val.shape[1:] == x_train.shape[1:]` | Channel-order divergence, downsample mismatch, max-length divergence |
| Feature dtype | `x_val.dtype == x_train.dtype` | uint8 vs float32 silent broadcast |
| Label shape | `y_val.shape[1:] == y_train.shape[1:]` | Scalar vs one-hot encoding mismatch |
| Label dtype | `y_val.dtype == y_train.dtype` | int8 vs int64 with overflow surprise |
| Label-set membership | `set(np.unique(y_val_sample)) ⊆ set(np.unique(y_train_sample))` | Unseen class in val (label-shift) OR label remap divergence |
| Feature value range | `min(x_val_sample) ≥ p1(x_train_sample) - δ` and `max(x_val_sample) ≤ p99(x_train_sample) + δ` | Silent renormalisation difference |
| Pair count match | `len(x_val) == len(y_val)` AND `len(x_train) == len(y_train)` | Off-by-one alignment (the FX §3.5 bug) |
| Index-pair invariant (when applicable) | `loader.get_pair(i)` returns the same `(x, y)` that iterating the loader yields at position `i` | Sampler swaps target indices |
| Modality-specific contract | see modality table below | dataset-specific drift |

### Modality-specific contracts (extend as needed)

| Modality | Additional contract |
|---|---|
| Image | `x.shape[-3:] == (C, H, W)` (or `(H, W, C)`) consistently across train and val; same H, W on both |
| Tabular | column-name list matches between train and val (no silent drop / reorder); column dtypes match |
| Time-series | sequence length matches OR per-row length is wrapped in a documented padding scheme that the contract names |
| Text | tokenizer vocabulary matches; same `pad_token_id` and `bos_token_id`; max-length consistent |
| Graph | node-feature dimension matches; edge-feature dimension matches; directionality convention matches |

### Index-pair invariant (the FX §3.5 trap)

This is the assertion that would have caught the +8.78 Sharpe bug
three weeks earlier than the shuffle test did. For any loader that
exposes a deterministic `__getitem__` AND an `__iter__`:

```python
for i in random_sample(range(len(loader)), k=8):
    x_from_iter, y_from_iter = list(loader)[i]
    x_from_getitem, y_from_getitem = loader[i]
    assert torch.equal(x_from_iter, x_from_getitem), \
        f"Loader index {i}: __iter__ vs __getitem__ x diverges"
    assert torch.equal(y_from_iter, y_from_getitem), \
        f"Loader index {i}: __iter__ vs __getitem__ y diverges"
```

The FX bug surfaced as exactly this: `y = seg_tgt.values[seq_len:]`
was wired to the iterator while `[seq_len-1:]` was wired to the
indexer. The two were never compared in code, and the structural
split audit had no reason to flag it.

## Where to call it (runner gate)

The validator runs in the runner's pre-flight, immediately AFTER
`audit_or_die()` from
[`autoresearch-data-split-audit`](../autoresearch-data-split-audit/SKILL.md):

```python
def runner_preflight(cfg):
    audit_or_die(cfg)                     # structural split audit
    validate_data_contract(               # this skill's gate
        train_loader=build_train_loader(cfg),
        val_loader=build_val_loader(cfg),
        test_loader=build_test_loader(cfg),
        modality=cfg.modality,
    )
    sota_smoke_or_die(cfg)                # Rule 13 SOTA pre-flight
    # ... onward to the 7-step ritual
```

`validate_data_contract` raises `SystemExit` on failure (NOT
`AssertionError`, which test frameworks swallow). The exit message
names the SPECIFIC contract that failed and the SPECIFIC counterexample
batch index — so a future Claude session can fix the loader without
re-running the contract.

There is no `--skip-contract` flag, no `--allow-contract-drift`, no
environment variable that disables the check. The contract is a hard
gate (Rule 7).

## Code pattern

```python
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class ContractViolation:
    property_name: str
    expected: str
    actual: str
    sample_index: int | None = None

    def __str__(self):
        idx = f" (batch sample {self.sample_index})" if self.sample_index is not None else ""
        return f"Contract '{self.property_name}'{idx}: expected {self.expected}, got {self.actual}"


def validate_data_contract(
    train_loader,
    val_loader,
    *,
    test_loader=None,
    modality: str = "image",          # "image" | "tabular" | "sequence" | "graph"
    n_batches_to_check: int = 4,      # sample multiple batches, not just first
    feature_range_tolerance: float = 0.05,
) -> None:
    """Refuse-to-launch if the (x, y) contract is violated. Raises SystemExit on FAIL."""
    violations: list[ContractViolation] = []

    train_samples = _sample_batches(train_loader, n_batches_to_check)
    val_samples   = _sample_batches(val_loader,   n_batches_to_check)

    x_train, y_train = train_samples[0]
    x_val,   y_val   = val_samples[0]

    # 1. Feature shape (excluding batch dim)
    if x_train.shape[1:] != x_val.shape[1:]:
        violations.append(ContractViolation(
            "feature_shape", f"{x_train.shape[1:]}", f"{x_val.shape[1:]}"
        ))

    # 2. Feature dtype
    if x_train.dtype != x_val.dtype:
        violations.append(ContractViolation(
            "feature_dtype", f"{x_train.dtype}", f"{x_val.dtype}"
        ))

    # 3 + 4. Label shape + dtype
    if y_train.shape[1:] != y_val.shape[1:]:
        violations.append(ContractViolation(
            "label_shape", f"{y_train.shape[1:]}", f"{y_val.shape[1:]}"
        ))
    if y_train.dtype != y_val.dtype:
        violations.append(ContractViolation(
            "label_dtype", f"{y_train.dtype}", f"{y_val.dtype}"
        ))

    # 5. Label-set membership
    train_labels = set(np.unique(_to_numpy(y_train)).tolist())
    val_labels   = set(np.unique(_to_numpy(y_val)).tolist())
    unknown = val_labels - train_labels
    if unknown:
        violations.append(ContractViolation(
            "label_set", f"⊆ {sorted(train_labels)}",
            f"val has unseen labels {sorted(unknown)}"
        ))

    # 6. Feature value range — checked across N batches for robustness
    train_p1, train_p99 = _robust_range(train_samples)
    val_min, val_max    = _robust_range(val_samples)
    delta = feature_range_tolerance * (train_p99 - train_p1)
    if val_min < train_p1 - delta or val_max > train_p99 + delta:
        violations.append(ContractViolation(
            "feature_range",
            f"[{train_p1:.3f}, {train_p99:.3f}] ± {delta:.3f}",
            f"val [{val_min:.3f}, {val_max:.3f}]"
        ))

    # 7. Pair-count
    if len(x_train) != len(y_train):
        violations.append(ContractViolation(
            "train_pair_count", f"{len(x_train)}", f"{len(y_train)}"
        ))
    if len(x_val) != len(y_val):
        violations.append(ContractViolation(
            "val_pair_count", f"{len(x_val)}", f"{len(y_val)}"
        ))

    # 8. Index-pair invariant (when applicable) — the FX §3.5 trap
    if hasattr(val_loader, "__getitem__") and hasattr(val_loader, "__iter__"):
        violations.extend(_check_index_pair_invariant(val_loader, k=8))

    # 9. Modality-specific
    violations.extend(_modality_contract(modality, x_train, y_train, x_val, y_val))

    if violations:
        msg = "REFUSED: data-contract violations:\n" + "\n".join(f"  - {v}" for v in violations)
        raise SystemExit(msg)
```

The helpers (`_sample_batches`, `_robust_range`, `_to_numpy`,
`_check_index_pair_invariant`, `_modality_contract`) are
project-specific but should live next to the validator in
`src/<pkg>/eval/data_contract.py` (or `core/evaluation/data_contract.py`)
alongside the data-split audit.

## Where the result lands

The validator writes its (passing) result to
`<results_root>/audits/data_contract.json`:

```json
{
  "status": "PASS",
  "checked_at": "2026-05-30T14:30:00Z",
  "n_batches_sampled": 4,
  "modality": "image",
  "feature_shape": "(3, 32, 32)",
  "feature_dtype": "torch.float32",
  "label_dtype": "torch.int64",
  "label_set_size_train": 10,
  "label_set_size_val": 10,
  "feature_range_train_p1_p99": [-1.99, 1.99],
  "feature_range_val_min_max":  [-1.97, 1.96],
  "index_pair_invariant_checked": 8,
  "violations": []
}
```

On FAIL, the JSON includes the violation list verbatim. The companion
markdown file `data_contract.md` is the human-readable summary and
should be committed alongside the structural split-audit report.

## Anti-patterns

- **Checking only the first batch.** Many augmentation pipelines
  produce contract-honouring first batches but contract-breaking
  later batches (e.g., a stochastic transform that occasionally
  produces a wrong shape). Sample `n_batches_to_check ≥ 4`.
- **`assert` instead of `raise SystemExit`.** Python `-O` strips
  asserts; pytest swallows them. The validator must crash the runner
  loudly.
- **Silent BatchNorm running-stat differences treated as a contract
  violation.** BN running stats differ between train (updated) and
  eval (frozen) — that's intended. The contract is about input
  shape / dtype / range, NOT about the model's internal state.
- **Treating explicit min-max scaling differences as "fine".** If
  train uses `(x - mean) / std` with one mean and eval uses a
  different mean, that's data leakage in the structural-audit sense
  AND a contract violation (feature range divergence) in this sense.
- **Skipping the index-pair invariant on PyTorch `Dataset` /
  `DataLoader`.** This is the assertion that catches the FX bug. The
  cost is negligible (8 indexer-vs-iterator pairs); the benefit is
  the single highest-leverage assertion in the autoresearch stack.
- **"I'll add the contract later, after I'm sure the pipeline works."**
  The pipeline only "works" if the contract holds. Run the validator
  from day one; it's cheap.
- **One contract file per modality, with project-specific assumptions
  hard-coded.** The validator should be modality-aware (a parameter)
  not modality-specific (one validator per modality). Reuse the same
  validator across image / tabular / sequence / graph by dispatching
  on `modality`.
- **Adding a `--skip-contract` flag for "smoke runs".** Rule 7 — no
  bypass. Smoke runs are exactly when the contract matters most.

## What "good" looks like

- The contract validator runs in < 2 seconds for typical autoresearch
  workloads (a few sampled batches per loader).
- The runner's FAIL message names ONE specific violation with a
  numeric counterexample — not a generic "data contract failed".
- The `data_contract.md` file is checked into git alongside the
  structural audit report so a reviewer 6 months later sees the
  contract was honoured at the time of the experiment.
- A pre-commit hook (optional) re-runs the validator if any of
  `data/loader.py`, `data/transforms.py`, `evaluation/*.py` changes.
- The validator's modality dispatch is **explicit** (`modality="image"`)
  not auto-detected — auto-detection is the exact place a silent
  divergence sneaks in.

## Cross-references

- [`autoresearch-data-split-audit`](../autoresearch-data-split-audit/SKILL.md)
  — structural row-disjointness audit; this contract validator is the
  complementary **shape/dtype/encoding** assertion. Both gate the
  runner. Recommended ordering: structural audit first (cheaper,
  catches the simpler bugs), then contract validator (catches the
  harder bugs).
- [`autoresearch-shuffle-test`](../autoresearch-shuffle-test/SKILL.md)
  — semantic leakage detector; the contract validator catches the
  off-by-one alignment **statically** (cheap pre-flight), the shuffle
  test catches it **empirically** (expensive but more general). Both
  must pass before any external claim.
- [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md)
  — the 7-step ritual's "Run" step (Step 4) starts with the runner
  pre-flight; the contract validator is part of that pre-flight.
- [`autoresearch-dataset-loader`](../autoresearch-dataset-loader/SKILL.md)
  — implementing the loader; the contract validator lives next to
  the loader in `src/<pkg>/eval/` or `core/evaluation/`.
- [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md)
  — statistical-rigor floor for external claims; data-contract PASS
  is one of the gating criteria.
- CLAUDE.md Rule 7 — "no `--bypass` flag"; the contract is a hard
  gate.
- CLAUDE.md Rule 22 — dual-track audit before any external claim;
  data-contract is the static leg, shuffle-test is the empirical leg.

## Provenance

- **FX §3.5 retrospective.** The same +8.78-Sharpe bug that motivated
  the shuffle-test skill was caused by an off-by-one alignment
  between the iterator (`y = seg_tgt.values[seq_len:]`) and the
  indexer (`y = seg_tgt.values[seq_len-1:]`). The shuffle test
  detected the **symptom** (validation metric did not collapse on
  shuffle); the data-contract validator would have detected the
  **cause** (index-pair invariant failed in the first 8 sampled
  indices, taking < 1 second). The autoresearchtabular CLAUDE.md
  §40 codifies this as `audit.py::audit_split_disjoint()` only
  catching half of the bug class. The other half is this skill.
