---
name: autoresearch-per-experiment-page
description: Use to emit one independent dashboard page per experiment with hypothesis content inlined, FINDINGS verdict, reasoning blob, full config, metrics, composite breakdown, per-epoch training curves, cross-references, run footer. Aggregate dashboard becomes group-sectioned (Baseline + G1..G8) instead of one cluttered table. Mirror to `docs/` for GitHub Pages. NO row-click modals — clicks navigate to pages.
---

# Skill — Comprehensive per-experiment dashboard pages

## When to use

- After ≥ 30 experiment runs accumulate, a single aggregate dashboard
  becomes too cluttered to be useful.
- When the user wants to drill into a single experiment and see the
  hypothesis design doc + reasoning + curves all in one place.
- When the existing dashboard's row-click "modal" is too cramped to
  carry the full hypothesis context.

## The two halves of the deliverable

### A. The aggregate dashboard — group-sectioned

```
[ribbon: headline finding from FINDINGS.md]
[best-per-dataset summary]
[composite formula chip + fingerprint]

## Baselines (n rows)
  table of baseline_resnet20, baseline_sg_vanilla rows, sortable.

## G1 — Scaling & Growth (n rows)
  <one-line group description>
  table of G1 runs, sortable composite-desc.

## G2 — Layer / Channel / Neuron
  ...

## G8 — Esoteric Extensions
  ...

## Uncategorised (empty if all tags map)
```

Every leaderboard row click navigates to its per-experiment page.
NO modals.

### B. The per-experiment page — 10 sections

`dashboard/experiments/<dataset>__<tag>_seed<N>.html` contains, in
order:

1. **Header** — title (tag + seed + dataset), pills for hypothesis-ID +
   group + dataset + seed, "← Back to dashboard" link.
2. **Hypothesis** — INLINE digest from the matching
   `hypotheses/g<N>_*/H<NN>_*.md`: title, one-liner, motivation,
   formal hypothesis, mechanism, numeric falsifier, predicted Δ, first
   arXiv citation, "📄 Full design doc →" link to GitHub.
3. **Verdict** — the matching paragraph from `FINDINGS.md`; link to
   the full FINDINGS section.
4. **Reasoning** — if `experiments/.../reasoning.json` exists, render
   diagnosis / citations / hypothesis / prediction / verdict / learning
   fields. Else a discreet "(no reasoning blob)".
5. **Configuration** — `config.yaml` content if present; else
   inferred overrides from `experiment_log.jsonl`.
6. **Metrics** — full metrics.json as a table.
7. **Composite breakdown** — term-by-term: `top1`, `−0.05·log10(params_M)`,
   `−0.05·log10(latency_ms)`, sum (must match reported `composite`).
8. **Training curves** — inline SVG, no external JS. One chart for
   loss, one for accuracy (train + val if both present).
9. **Cross-references** — links to other seeds of the same tag + the
   same tag on the other dataset.
10. **Footer** — composite_fingerprint SHA-256, epochs run,
    train_seconds, generalization_gap, run dir path.

## Implementation pattern

- The renderer LOGIC lives in `src/.../dashboard.py` (per Rule 14 —
  shared primitive). `scripts/build_dashboard.py` is a thin caller.
- Filename collision: tags repeat across datasets (e.g.,
  `sg_only_phi_budget` exists on both cifar10 and cifar100). Use
  `<dataset>__<rundir>.html` to disambiguate. Otherwise the later
  dataset overwrites the earlier.
- The tag→hypothesis mapping (`TAG_TO_HYP`) is a hardcoded dict in
  `dashboard.py`. Maintain it as new tags are added.
- Aggregate HTML rows: `<tr class='row-link'
  onclick="location.href='experiments/<file>.html'">` with
  `cursor:pointer`. The tag cell can additionally wrap in `<a>` with
  `event.stopPropagation()` so the tag link also navigates.

## Mirror for GitHub Pages

Pages serves from `docs/`. After building under `dashboard/`, mirror
byte-identically into `docs/dashboard/`:

```python
for fname in exp_pages:
    (docs_exp_dir / fname).write_text(
        (exp_out / fname).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
```

The live URLs become
`https://<user>.github.io/<repo>/dashboard/dashboard.html` and
`...html/dashboard/experiments/<file>.html`. Link them from `README.md`
near the top with a badge.

## Anti-patterns

- A "reasoning modal" that pops up on row click — too cramped to carry
  10 sections of context. Kill it; rows navigate.
- An aggregate dashboard with one giant 60+-row table — unreadable
  beyond ~20 rows. Section by group.
- Per-experiment pages that don't inline the hypothesis content —
  reader has to open three tabs to understand a single run.
- Updating `dashboard/dashboard.html` without re-mirroring `docs/` —
  GitHub Pages stays stale.

## Cross-references

- CLAUDE.md Rule 24 — full dashboard discipline.
- `autoresearch-dashboard` — the original aggregate-dashboard skill;
  this extends it with per-experiment pages + group sectioning.
- `dashboard/dashboard.html` example output on this repo.
