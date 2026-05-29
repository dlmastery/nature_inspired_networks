# Per-hypothesis dashboard — contract

This directory holds the **per-hypothesis** dashboard produced by the
`autoresearch-per-hypothesis-hillclimb` skill (the hill-climb tier of
evaluation that sits between the master screening sweep and the
external Phase-4 / Phase-5 claims).

The page at `dashboard/index.html` is independent of, and linked from,
the master aggregate `dashboard/dashboard.html` at the repository
root. It is the deliverable for the hill-climb campaign on a given
hypothesis.

## Required sections (in order)

1. **Best-config callout.** Top of the page, monospace block:
   ```
   lr=3e-3  wd=5e-4  bs=256  opt=AdamW
   top1 = 0.8617  (median, 3 seeds; min 0.8580, max 0.8650)
   composite = 0.6209  (median, 3 seeds)
   beats baseline_max(3-seed): YES / NO
   ```
2. **Cells table.** Sortable. One row per `(config, seed)` cell. Columns:
   `config_id` · `lr` · `wd` · `batch` · `opt` · `seed` · `top1` ·
   `composite` · `Δ top1 vs best` · `wallclock_s` · link to the
   per-experiment page (the `autoresearch-per-experiment-page` skill's
   per-cell output at
   `dashboard/experiments/<dataset>__<tag>__hc_<config_id>_seed<S>.html`).
3. **Per-axis Pareto plot.** Small-multiples SVG, one panel per axis
   in `[lr, wd, batch, optimizer]`. x = axis value, y = top-1.
   Series coloured by the other axes' fixed values. Reveals
   non-monotone responses (= the signal to switch from
   `--algorithm coordinate` to `--algorithm random`).
4. **Seed-stability bar chart.** At the best config, bars for seeds
   {0, 1, 2}; std written above. Used to argue the seed-min-vs-
   baseline-max gate.
5. **Footer.** Base config, hill-climb algorithm + budget, total
   wall-clock, COMPOSITE_FORMULA fingerprint, links to:
   - master aggregate `dashboard/dashboard.html`
   - `ideas/<NN>/hillclimb_results.json` (the machine-readable
     summary)
   - the hypothesis design doc `hypotheses/g<N>/H<NN>_*.md`
   - the FINDINGS.md row for this hypothesis (pre-fix vs post-fix
     if applicable per Rule 21)

## Input artefacts

The renderer reads:

- `ideas/<NN>/hillclimb_results.json` — produced by
  `scripts/run_hillclimb.py` (the summary).
- `experiments/<dataset>/<tag>__hc_*` — per-cell metrics.json,
  history.json, best.pt.
- `hypotheses/g<N>/H<NN>_*.md` — design-doc digest for the page
  header.
- `FINDINGS.md` — verdict row (KEEP / NEAR-MISS / DISCARD) to embed.

## Output mirror

The page is mirrored byte-identically to
`docs/dashboard/ideas/<NN>/index.html` for the GitHub Pages live
demo at
`https://dlmastery.github.io/nature_inspired_networks/dashboard/ideas/<NN>/`.

## Anti-patterns

- A pretty page that hides cells with crashed runs — keep them
  (status = `ERROR`) so the budget is auditable.
- Per-axis plots that fix the "other" axes at the BASE rather than
  the hill-climbed best — that masks why the climb advanced.
- Mixing screening rows with hill-climb cells in the same table.
  Screening cells (`<tag>_seed<N>/`) and hill-climb cells
  (`<tag>__hc_*_seed<N>/`) are evaluated under different recipes;
  the table must restrict itself to hill-climb cells.
- Skipping the design-doc digest — the reviewer needs the
  hypothesis + mechanism + numeric falsifier on the SAME page as the
  numbers; do not force a click-through to the design doc.

## Cross-references

- `skills/autoresearch-per-hypothesis-hillclimb/SKILL.md` — produces
  the inputs.
- `skills/autoresearch-per-experiment-page/SKILL.md` — produces the
  per-cell sub-pages this dashboard links to.
- `skills/autoresearch-dashboard/SKILL.md` — produces the master
  aggregate this dashboard sits beneath.
- CLAUDE.md Rule 9 — the dashboard / README is the deliverable; the
  weights are secondary.
- CLAUDE.md Rule 24 — dashboard discipline (group sectioning,
  per-experiment pages, NO modals).
