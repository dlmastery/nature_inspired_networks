# Dashboard walkthrough — 2026-05-29 (Phase A / Phase B audit)

Reviewer: dashboard-redesign engineer (post-commit `914f382`, post-fix `paper/FINDINGS.md` path).
Tool: Playwright MCP server, viewport 1400×900, page served from `http://localhost:8765/dashboard/dashboard.html`.

This walkthrough is the deferred Phase 10 (live-page audit) from the prior Repo-Restructure agent. Screenshots capture the page as a hostile NeurIPS supplementary-material reviewer would encounter it. Captions below cite the actual DOM state, not the source code's intended state.

## Page-level summary (Playwright `document.body.scrollHeight = 11,657 px`)

| region | present | notes |
|---|---|---|
| Masthead (h1 + sub + 4 CTA pills) | YES | Source Serif 4 body confirmed (computed style). |
| Headline ribbon (`.headline-ribbon`) | YES | **Markdown renders correctly** — `<h2>`, `<table>`, `<strong>`, `<code>` all parsed. ZERO literal `##`, `**`, `|---:|`, `&gt;`, `> >` leaks (Playwright `textContent` probe). |
| "How to read this dashboard" 4-bullet block | **MISSING** | Required by Rule 33 (autoresearch-dashboard-comprehension SKILL.md Pillar 2). Phase E will add. |
| Formula chip | YES | `.formula-chip` with composite SHA-256 fingerprint. |
| PNG grid: Pareto + Ablation + Curves + Betti + summary + hyp-grid | YES (6 cards) | Old monolithic PNGs — Phase C/D will REPLACE Pareto and Ablation with SVG small-multiples. |
| Group sections G1..G8 + Baseline | YES (9 sections, Uncategorised correctly hidden) | Group counts (runs/group): Baseline 10, G1 45, G2 8, G3 6, G4 8, G5 12, G6 4, G7 0, G8 6. |
| Footer (.meta) with commit SHA stamp | YES | Build-stamp + live-Pages link present (added in prior Phase 6 fix). |

## Three most-ugly findings (PRE Phase C/D/E fixes)

1. **No orientation block** — a reviewer landing on the page is dropped straight into a dense headline-ribbon table without any 4-bullet "what / colour / SCREENING-vs-EVALUATION / drill-down" guide. Rule 33 mandatory.

2. **Pareto PNG is one matplotlib 1×3 panel with overlapping 22-tag annotations** (`plot_pareto.png`) — at the dashboard's 1340px max-width the tag labels collide into illegibility. The user explicitly requires 3 SVG small-multiples panels with shared y-axis, group-coloured points, Phase-8 winner stars, and frontier overlay (Phase C).

3. **Ablation matrix PNG is one matplotlib barh chart with ~25 horizontal bars in arbitrary "composite" order** (`plot_ablation.png`) — the per-group reading is impossible without panning the chart. Per CLAUDE.md Rule 33 + skill SKILL.md Pillar 1, this MUST be 8 group panels in a 2×4 grid with divergent Δ-vs-baseline colouring and noise-band reference line (Phase D).

## Region screenshots

- `01_masthead.png` — top of page. Masthead title in Source Serif 4 + 4 CTA pills (source / background / paper / live). Sub-row of repo-document links present.
- `02_headline_ribbon.png` — the `.headline-ribbon` block as rendered. Confirms markdown renders properly: h2 heading, GFM table with 4 columns + 4 data rows (3 winners + baseline), strong/code spans intact. **No literal `##`, `**`, `|---|`, `> >` leaks visible to the camera.**
- `03_pareto_and_ablation.png` — the PNG grid Pareto + Ablation cards. **Both targeted for SVG-small-multiples replacement.**
- `04_group_g1.png` — first group section (G1 Scaling & Growth, 45 runs). Shows the existing per-group SVG visualisations: top-1 horizontal-bar chart + composite-vs-params scatter; compact runs table with `n=X tier` chips (Rule 34 partially landed via prior fix). Tag pills include n-chip badges showing screening/eval. Phase E orientation block will inform reading.
- `05_group_g5.png` — G5 Optimisation/Init/Regularisation/NAS (12 runs). Same per-group SVG visualisation pattern.
- `06_footer.png` — `.meta` footer with commit-SHA build stamp + live-demo Pages link. Source Serif 4 typography confirmed.

## Phase B — headline ribbon definitive verification

**Status: RENDERS CORRECTLY. No fix needed.** The prior Phase 6 path-fix (FINDINGS.md → paper/FINDINGS.md) is sufficient. Playwright `textContent` probe of `.headline-ribbon` returns zero literal markdown chars:

```
has_double_pound: false
has_double_star: false
has_table_sep: false (no |---)
has_gt_entity: false
has_literal_blockquote: false
```

`.headline-ribbon` `outerHTML` confirms structured rendering: `<h2>✅ PHASE-8 FINAL VERDICT...</h2>`, `<table><thead><tr><th>tag</th>...</tr></thead><tbody><tr><td><code>pair_gm_pdw</code> ...</td>...</tr>...</tbody></table>`, `<strong>+1.34 pp</strong>`. `_md_to_html` + `_strip_blockquote_markers` + CSS rules in `.headline-ribbon { h2, table, th, td, strong, code, p, blockquote }` together do the right thing.

The screenshot `headline_ribbon_AFTER.png` captures the rendered ribbon as a reviewer sees it after the prior path-fix.

## Outstanding for Phases C / D / E (this campaign)

1. Phase C — replace `plot_pareto.png` card with `_pareto_panels_svg(...)` triple-panel SVG (top1 vs params/flops/latency, log x-axes, group colours, Phase-8 winner stars, frontier overlay, caption per panel).
2. Phase D — replace `plot_ablation.png` card with `_ablation_group_panels_svg(...)` 2×4 grid of 8 group panels (Δ vs baseline horizontal bars, divergent colour, ±1.21pp noise band reference line, caption per panel).
3. Phase E — insert 4-bullet "How to read this dashboard" `<section class='how-to-read'>` block AFTER masthead + sub-row but BEFORE `.headline-ribbon`. Style consistent with the existing Source Serif 4 / IBM Plex Mono palette.
