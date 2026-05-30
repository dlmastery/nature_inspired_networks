---
name: autoresearch-experiment-archive
description: Use when archiving an experiment for publication / reproducibility / future-Claude pickup. Defines the taxonomy directory structure and the very-detailed-design README that every archived experiment must carry.
---

# Skill — Archive an experiment in its own sub-directory

## When to use

After every experiment that produced a numeric result you want to
preserve (i.e., almost always). The point of archiving is that someone
3 months from now can pick up just that sub-directory and reproduce
the experiment from cold.

## The unit of archive

`ideas/<NN_idea_name>/experiments/expNNN_<short_name>/` — that's it.
Everything for ONE experiment lives below this path. No global
side-files, no symlinks into other ideas.

## Mandatory contents

```
expNNN_<short>/
├── README.md          ← very detailed design doc (see below)
├── config.yaml        ← exact config passed to the runner
├── reasoning.json     ← citation-gated reasoning entry
├── run_seed0/
│   ├── metrics.json
│   ├── history.json
│   └── best.pt
├── run_seed1/         ← if multi-seed
├── run_seed2/
└── dashboard/
    ├── dashboard.html       ← local self-contained dashboard
    ├── plot_pareto.png
    ├── plot_curves.png
    └── plot_betti.png       ← if topology was computed
```

## The very-detailed README.md

The README is the deliverable. It must include all the following
sections — even if some are short, they must exist:

```markdown
# expNNN — <one-line title>

## TL;DR

<2-3 sentences with the headline number and verdict>.

## 1. Motivation

Why this experiment exists. What broader idea (H<NN>) it is testing.
Reference the [idea README](../../README.md) for the parent context.

## 2. Hypothesis

The mechanism we are claiming. What flips, what we think it does in
the model. Quote the cited paper. Word count ≥ 50.

## 3. Citations

Full citation block, one per line, in the Citation Rigor format
(see autoresearch-reasoning-entry).

## 4. Pre-registered prediction

Numeric range on the composite + at least one sub-metric. Stored
BEFORE the run started. Word count ≥ 25.

## 5. Method

- Architecture: <link to implementation.py and class name>
- Dataset: <name, train/test sizes>
- Optimiser: <name, hyperparameters>
- Epochs / batch / precision / seeds
- Composite formula + SHA-256 fingerprint

## 6. Results

| seed | top-1 | params | latency_ms | composite |
|------|-------|--------|-----------|-----------|
| 0    | ...   | ...    | ...       | ...       |
| 1    | ...   | ...    | ...       | ...       |
| 2    | ...   | ...    | ...       | ...       |

Insert any relevant plots inline as `![](dashboard/<plot>.png)`.

## 7. Verdict

KEEP / NEAR-MISS / DISCARD + reasoning. Word count ≥ 30.

## 8. Learning

What we now believe; what's next. Word count ≥ 40.

## 9. How to reproduce

```bash
python -m <project>.runner \
  --config config.yaml --tag expNNN_<short> --seed 0 \
  --root run_seed0
python scripts/build_dashboard.py --root . --out dashboard
```

## 10. Open issues

Anything that surprised us, any caveat. Bullet list.
```

## Hard rules

1. **Archives are immutable after verdict-write.** Append-only: if a
   result needs re-interpretation, create `expNNN_<short>_v2/` with a
   link to the original.
2. **No symlinks.** Every artifact is a real file in the dir.
3. **Total dir size ≤ 100 MB** (model weights can be large; if so,
   compress or store on HF Hub and link).
4. **README.md is the source of truth.** If `dashboard/dashboard.html`
   disagrees with README results, README wins until reconciled.
5. **Local dashboard, not just global.** Each archive carries its own
   `dashboard/dashboard.html` so the directory is self-contained.

## Cross-references

- The parent `ideas/<NN_idea>/README.md` carries the idea-level
  design.
- The global `experiment_log.jsonl` at the repo root has one summary
  row per archive sub-directory.
- [`autoresearch-winner-archive`](../autoresearch-winner-archive/SKILL.md)
  — when an experiment in this archive becomes the GLOBAL champion, a
  SECOND archive (the winner archive) is created on top with frozen
  code, inference script, 14-section audit report, and reproduction
  log. This per-experiment archive stays in place; the winner archive
  is additive.
- [`autoresearch-per-experiment-page`](../autoresearch-per-experiment-page/SKILL.md)
  — the dashboard page generated from this archive's metrics +
  reasoning blob.
- [`autoresearch-explainability-report`](../autoresearch-explainability-report/SKILL.md)
  — for champion-archives, the 14-section audit ships at
  `<winner>/audit_report.md`.

## Anti-patterns

- "I'll fill in the README later." Archives without READMEs are dead
  weight; close the PR if they're missing.
- Mixing two experiments in one archive sub-directory ("multi-seed +
  multi-config"). Split them.
- Storing best.pt in git LFS without recording its SHA in the README.
  If LFS goes away the result is unreproducible.
