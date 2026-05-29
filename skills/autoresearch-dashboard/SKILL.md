---
name: autoresearch-dashboard
description: Use when generating the sortable static HTML dashboard with Pareto plots, ablation matrix, training curves, Betti / CKA panels, and a sortable/filterable runs table. Output is self-contained for GitHub Pages.
---

# Skill — Generate an autoresearch dashboard

## When to use

After any sweep, mid-campaign progress check, or before publishing
a checkpoint to a public repo. Re-run any time `experiment_log.jsonl`
or per-run `metrics.json` changes.

## Required inputs

- `<root>/<dataset>/<tag>_seed<S>/metrics.json` — per-run metrics
- `<root>/<dataset>/<tag>_seed<S>/history.json` — per-epoch list
  with at least `epoch`, `test_top1` (or domain-equivalent)
- Optional: `<root>/betti.json` — topology curves
- Optional: `<root>/cka.json` — between-variant alignment

## Outputs

| file | purpose |
|---|---|
| `dashboard/dashboard.html` | self-contained sortable/filterable HTML |
| `dashboard/plot_pareto.png` | top-1 vs. {params, FLOPs, latency}, baselines as stars |
| `dashboard/plot_ablation.png` | bar chart of top-1 per tag with seed-std error bars |
| `dashboard/plot_curves.png` | per-tag training curves |
| `dashboard/plot_betti.png` | β₀ / β₁ collapse per stage (if betti.json exists) |
| `docs/dashboard/dashboard.html` | identical, mirrored for GitHub Pages |
| `docs/index.html` | tiny landing page linking to the dashboard |

## What "good" looks like

- The Pareto plot has at least one **dominated** SacredGeo / your-block
  variant — proves the framework discriminates, not just plots noise.
- The ablation chart is sorted by composite descending; best run is
  highlighted (green background row in the HTML table).
- The runs table supports type-to-filter and click-to-sort *without*
  JavaScript dependencies — a single inline `<script>` block, no CDN.
- File sizes: full HTML ≤ 50 KB; each PNG ≤ 200 KB.

## Sortable-table JS contract

The HTML emits:

```html
<table id='runs' data-dir='asc'>
  <thead>
    <tr>
      <th onclick='sortTable(<col_idx>)'>Column label</th>
      ...
    </tr>
  </thead>
  <tbody>
    <tr><td data-v='<raw>'>display</td>...</tr>
  </tbody>
</table>
```

`data-v` attribute carries the raw numeric value so sort works on
float-as-string. `data-dir` flips between `asc`/`desc`. The
`filterTable()` function does a `textContent.toLowerCase()` substring
match against the `#q` input.

## Hard rules

1. **Self-contained HTML.** No external script tags, no CSS frameworks.
   GitHub Pages must serve it correctly without network.
2. **PNG, not SVG, for plots.** Browsers handle them faster and they
   travel better in Markdown embed.
3. **`composite` column is the default sort.** Other columns require
   clicking the header.
4. **No emoji in the dashboard** unless the user explicitly asked.

## How to extend

To add a new panel:

1. Compute the data into `<root>/<your_panel>.json`.
2. Add a `plot_<your_panel>(rows, out_png)` function to
   `dashboard.py`.
3. Add a card block in `HTML_HEAD`:
   ```python
   if (p / "your_panel.png").exists():
       html.append(f"<div class='card' style='grid-column:1/3'>"
                   f"<h3>Your panel</h3>"
                   f"<img src='your_panel.png'/></div>")
   ```

## Anti-patterns

- Pulling fonts from Google Fonts WITHOUT a `Charter, Georgia,
  serif` fallback chain (breaks offline + Pages-edge-cache
  misses).
- Using Plotly / Bokeh / D3 — too heavy for a checkpoint dashboard.
- Auto-refreshing the dashboard from JS — keep it static.
- A single dense overlaid chart when 3 side-by-side small-
  multiples convey the same info (per Rule 33 + sibling skill
  `autoresearch-dashboard-comprehension`).
- Numeric cells without `n=X` + tier chip (Rule 34 violation).
- A leading-hypothesis-only pill on multi-hypothesis tags
  (combo/pair/hybrid).
- A self-graded ACCEPT banner without the "Internal QA pass —
  external review pending" qualifier (Rule 37).

## Mandatory companion skills (added 2026-05-29)

A dashboard.html commit is INCOMPLETE without ALL of:

1. **Typography + markdown rendering** —
   `skills/autoresearch-typography-and-rendering/`. Source Serif 4
   palette + GFM-table + blockquote markdown converter; Playwright
   asserts no literal `##`/`**`/`|---|` leak through any embedded
   block. ALSO covers per-experiment pages — same converter, same
   font stack, asserted across BOTH surfaces.
2. **Comprehension** —
   `skills/autoresearch-dashboard-comprehension/`. Small-multiples
   over dense charts; mandatory 4-bullet "how to read"
   orientation block; multi-hypothesis pills; no self-grading
   banners.
3. **Link discipline** — `skills/autoresearch-link-discipline/`.
   Absolute GitHub-blob URLs (Rule 27); first-mention
   linkification (Rule 38); Playwright HEAD-test every href.
4. **Seed-count + tier badge** — `n=X` + `SCREENING`/`EVALUATION`
   chip on every numeric (Rule 34, enforced via
   `autoresearch-dashboard-comprehension`).
5. **Per-experiment page mirror** —
   `skills/autoresearch-per-experiment-page/`. Each
   leaderboard row navigates to its own page; pages mirror to
   `docs/dashboard/experiments/`.

After every dashboard generator change, run the Playwright
verification gate (typography + markdown + links) BEFORE
marking the change done. The 2026-05-29 audit caught the same
markdown-rendering regression three times in one session because
this gate was skipped.
