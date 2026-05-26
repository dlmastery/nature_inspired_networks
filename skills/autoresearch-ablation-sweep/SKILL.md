---
name: autoresearch-ablation-sweep
description: Use when running a structured ablation matrix across many configs (baselines + single-prior on + leave-one-out + full hybrid). Produces JSONL log + per-run archives + global dashboard. Default cadence is a curated subset that fits in ~60 min on a single GPU; full sweep is a flag.
---

# Skill — Run a structured ablation sweep

## When to use

Any time you want to attribute headline effects to specific design
choices. The minimum useful ablation matrix has at least:

- 1 strong literature baseline (e.g., ResNet-20 on CIFAR-10)
- 1 vanilla version of your block with all priors / flags OFF
- N single-flag-on rows (one per design choice you want to attribute)
- 1 full hybrid row (all flags on)

Curated (~10–15 rows × 1 seed × short epochs) fits in ~60 min on a
4090 Laptop. Full sweep (add leave-one-out + extra channel modes +
3 seeds) takes 3-5×.

## Structure of a sweep driver

```python
def build_matrix(curated: bool = True) -> list[dict]:
    base = {flag: False for flag in FLAGS}
    full = {flag: True  for flag in FLAGS}
    rows = []

    # Baselines (always)
    rows.append(dict(tag="baseline_literature",
                     overrides=dict(model="<lit_model>")))
    rows.append(dict(tag="baseline_vanilla",
                     overrides=dict(model="<your>", flags=base.copy())))

    # Single-prior on rows
    for k in FLAGS:
        f = base.copy(); f[k] = True
        rows.append(dict(tag=f"only_{k}",
                         overrides=dict(model="<your>", flags=f)))

    # Full hybrid
    rows.append(dict(tag="full",
                     overrides=dict(model="<your>", flags=full.copy())))

    if not curated:
        # Leave-one-out from full
        for k in FLAGS:
            f = full.copy(); f[k] = False
            rows.append(dict(tag=f"loo_no_{k}",
                             overrides=dict(model="<your>", flags=f)))
    return rows
```

## What every row writes

Per run (`<root>/<dataset>/<tag>_seed<S>/`):

- `metrics.json` (RunMetrics dataclass)
- `history.json` (per-epoch list)
- `best.pt` (final state_dict — required for trained-feature
  post-hoc analysis like Betti / CKA)

Global (`<root>/`):

- `experiment_log.jsonl` (append-only, one row per run)

## Hard rules

1. **`--skip-existing` is the default.** Restart-friendly. If you
   want a re-run, remove the directory first.
2. **Per-run wall-clock printed.** If a single row takes > 1.5× the
   median, log a warning — likely a config error.
3. **Failures keep the sweep going.** A run that crashes writes
   `<root>/<dataset>/<tag>_seed<S>/error.txt` and the next row
   starts.
4. **Each row is one experiment per the [autoresearch-experiment]
   skill.** Reasoning entries can be batch-authored beforehand, but
   each row still needs its own gated entry.

## Recommended cadence

| stage | runs | seeds | epochs | wall-clock (4090 Laptop) |
|---|---|---|---|---|
| Smoke | 1 | 0 | 3 | < 2 min |
| Curated ablation | 10-15 | 0 | 12-20 | 60-90 min |
| 3-seed re-sweep | 10-15 | 0 1 2 | 12-20 | 3-4× curated |
| Leave-one-out | + 6-10 | 0 1 2 | 12-20 | 2× extra |
| Scale-up (best row only) | 3-5 | 0 1 2 | full epochs | hours-days |

## What "good" looks like

- The curated matrix produces a *Pareto plot* + an *ablation bar
  chart* + *training curves* — the standard
  [autoresearch-dashboard] outputs.
- Each row's verdict (KEEP / NEAR-MISS / DISCARD) is filled in
  `reasoning.json` within 1 working day of run completion.
- The headline finding is rephrased in `FINDINGS.md` with concrete
  Δ-vs-reference numbers — not "improves significantly."

## Anti-patterns

- Running the full sweep before validating the curated matrix.
- Adding new flags to `FLAGS` mid-sweep (the comparisons across rows
  break).
- Sorting rows by accuracy and reporting "top-3" — sort by composite
  to honour the Goodhart fingerprint.
