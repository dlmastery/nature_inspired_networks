# NeurIPS-grade reviewer pass — dashboard.html + per-experiment pages
Reviewer: hostile UX/info-density auditor, 2026-05-29.
Verdict: **WEAK_REJECT** — the per-experiment template is on the right track, but two of the three big "we fixed it" claims from commit 5194814 are not actually delivered, and the aggregate dashboard is in a more degraded state than the per-experiment pages it links to. Without the fixes below this would be embarrassing as NeurIPS supplementary material.

Overall summary: the per-experiment pages render their sci-critic + impl-critic markdown blocks correctly (h3 / em / strong / ol / code / blockquote all present), the 10-section template is structurally complete, the footer carries a clickable commit SHA, cross-references work, and SVG curves are inline + self-contained. **But** the FINDINGS-verdict block on every per-experiment page is *NOT* rendered as markdown — `>` block-quote markers leak through as literal `&gt;` text, GFM tables collapse into a single `<p>` with `|` pipes, and on the `pair_gm_pdw_cifar100_seed0` page the whole FINDINGS body is jammed into a single runaway `<h2>` heading. The aggregate dashboard is worse: the "Headline finding" ribbon shows *raw markdown* (`##`, `**`, `|---:|`) untouched, and the page is still styled in `Newsreader / IBM Plex Serif`, not the promised Source Serif 4. The font-swap and markdown-rendering fix landed for only ~half of the surface area.

## Aggregate dashboard (C:\Users\evija\sacgeometry\dashboard\dashboard.html)

- **Source Serif 4 NEVER applied** (CRITICAL fail of 5194814).
  - Line 5 imports the font from Google Fonts.
  - Lines 28, 34, 37, 48, 56, 130, 141, 154 still set `font-family: 'IBM Plex Serif','Charter',Georgia,serif` for body and `'Newsreader',serif;font-style:italic` for h1/h2/.kpi/.headline-ribbon/.group-empty/side-panel.
  - Line 234 footer self-describes the dashboard as "*Brutalist Editorial Lab Notebook v3 · Newsreader / IBM Plex Serif / IBM Plex Mono*" — explicitly confessing the swap was never made.
  - **Regression** vs. the per-experiment pages which DO use Source Serif 4.

- **Markdown rendering broken on the headline ribbon** (CRITICAL fail of 5194814).
  - Line 190: the `<div class='headline-ribbon'>` contains raw markdown: literal `##`, `**`, `|---:|---:|`, table pipes, `<b>Headline finding</b>` followed by `## ✅ PHASE-8 FINAL VERDICT ...`. None of the markdown is parsed.
  - This is the single most prominent block on the page — visible above the fold to every reviewer who clicks the Pages link.

- **Pages-live URL not linked anywhere** (Rule 27 spirit violated).
  - The dashboard header (line 188) links to GitHub repo, FINDINGS.md, EXPERIMENT_LOG.md, PAPER.md, AUDIT_SUMMARY.md — but `https://dlmastery.github.io/nature_inspired_networks/` is nowhere.
  - A reviewer who downloads the supplementary zip and opens `dashboard.html` locally has no way to discover that a live Pages mirror exists.

- **No footer commit-SHA / build-stamp** on the aggregate dashboard.
  - Per-experiment pages have a clean `metrics.json git commit: <sha>` link in their `.meta` footer; the aggregate dashboard has only the generator-name string (line 234). Reviewers cannot pin "which commit produced this dashboard."

- **No seed-count framing on aggregate KPIs / leaderboards.**
  - No "n=1 seed (screening)" vs "n=3 seeds (evaluation)" qualifiers anywhere. The `## ✅ PHASE-8 FINAL VERDICT` ribbon (raw markdown) does mention "3 seeds" inside its broken-text body, but there is no UI-level visual badge / column / pill on the leaderboard tables.
  - This directly violates **Rule 28 (screening-vs-evaluation)** added in commit 2cb0147.

- **Dead UI** (minor cleanliness): `<div id='side-panel'>` (line 236) is rendered as DOM but no code path opens it; `openHypothesis(...)` (line 264) always `window.open`s the GitHub blob URL. The orphan `<button class='close-btn' onclick='closeSide()'>` is unreachable.

- **Group-section "Uncategorised" with 0 runs is rendered** (line 233) — a cosmetic empty section reviewers will fixate on. Hide when count == 0.

- **Top-level title** (line 188): `NaturePriorBlock &mdash; autoresearch dashboard`. The project is `nature_inspired_networks`, not `NaturePriorBlock`. Minor branding inconsistency.

- **Hypothesis-grid summary line** (line 200) reads `84 hypotheses · 30 done · 51 impl · 2 queued · 1 deferred` — sums to 84 ✅ but the project's source-of-truth IDEA_TABLE.md is referenced as 75-hypothesis (per CLAUDE.md §0). Either the IDEA_TABLE expanded to 84 or the count is wrong. (Not blocking; flagging for the user.)

- **No JS errors detected by static inspection** (Playwright file:// was blocked so a live console pass was not possible). Both `sortTable` and `openHypothesis` are well-formed.

## Per-experiment pages sampled (5)

### pair_gm_pdw_cifar100_seed0
- **CRITICAL**: FINDINGS block-quote (line 199) is mis-rendered. The opening `> > | tag | C100 median | ...` table is wrapped inside an `<h2>` instead of being parsed as markdown — the heading becomes a 200-word run-on with literal `&gt;` and `|` characters. The whole "PHASE-8 FINAL VERDICT" body is consumed by ONE giant `<h2>`. **The 5194814 fix is partial: the inner-text markdown converter handles inline `**bold**` and `` `code` `` but not block-quotes containing tables.**
- **MINOR**: line 203 trailing `**` literal at end of sci-critic preamble: `Hardest scrutiny applied — this is the CIFAR-10/100 cross-dataset survivor.</em>**` — stray asterisks unrendered (mid-block escape issue).
- **MINOR**: reasoning blob shows `No reasoning.json for this run directory.` — this is the campaign's three-positive headline experiment; a missing reasoning entry on the Phase-8 winner is a Rule-5 gap.
- **MINOR**: KN-strip says `+1.34 pp Δ vs baseline` with no `n=3` qualifier — a hostile reviewer will assume seed-0 only (it isn't; the dashboard claim is the 3-seed median, but the UI doesn't say so).
- **PASS**: 10/10 sections of the per-experiment template present; sci-critic + impl-critic markdown both render correctly with `<h3>`, `<ol>`, `<strong>`, `<code>` etc.; footer has clickable commit SHA `e4f286f`; cross-references navigate to existing files.

### slot_act_sine_cifar100_seed0
- **CRITICAL**: same FINDINGS block-quote bug — same 200-word `<h2>` runaway. Identical text to pair_gm_pdw; both pages share the same broken FINDINGS body.
- **MINOR**: reasoning blob also missing (`No reasoning.json` — Phase-8 winner, Rule 5 gap).
- **PASS**: impl-critic + sci-critic markdown render correctly, Source Serif 4 active on body+h2, training curves inline, footer SHA `e2ad9dc`.

### sg_only_phi_budget_cifar100_seed0
- **CRITICAL**: same FINDINGS markdown bug.
- **PASS**: rest of template intact. This is the H09 post-fix run; commit SHA in footer should anchor reviewers to the fix commit.

### sg_only_golden_adam_cifar10_seed0
- **MAJOR**: FINDINGS shows the requalification table mangled into a single `<p>` — `| tag | hyp | top-1 | verdict | |---|---|---|---| | sg_only_golden_adam | H41 | 51.96 % | falsified — worst run in the entire project |` — pipes survive as literal characters; reader cannot read the table.
- **OK**: the verdict ("falsified — worst run in the entire project") IS visible as bold via inline `<strong>` — partial rendering happens, just no GFM tables.
- **MINOR**: 12-epoch screening row labelled as "falsified" without `(n=1 seed, screening)` framing — Rule 28 gap.

### combo2_pb_gm_cifar10_seed0 (5th sample — from the combo-ladder / orthogonal-stack family per Rule 23)
- **MAJOR**: FINDINGS table is a 1500-character single `<p>` element with raw `|`, `&mdash;`, and `&mdash;:` separators. Unreadable.
- **MAJOR**: the page's H09 pill (line 189) is misleading — combo2 is H09+H48 stacked; the multi-hypothesis nature is invisible.
- **MINOR**: cross-references section likely missing pair_gm_pdw / pair_gm_plr as "nearest in G1" given they share priors. (Not verified exhaustively.)

## Cross-cutting issues
- **Markdown rendering**: **FAIL** on FINDINGS sections of all 5 sampled per-experiment pages AND on the aggregate dashboard's headline ribbon. **PASS** on sci-critic addenda + impl-critic excerpts on per-experiment pages. The 5194814 fix is half-delivered: the inline-markdown converter handles `**`, `*`, `` ` ``, `_` but not block-quotes containing GFM tables, and the aggregate dashboard's headline ribbon was never wired through the markdown pipeline at all.
- **Font swap (Source Serif 4)**: **PASS** on per-experiment pages (line 26: `body{font-family:'Source Serif 4',...}`). **FAIL** on aggregate dashboard.html (still `IBM Plex Serif` body + `Newsreader` italic display; line 234 self-confesses). Inconsistent typography across the two surfaces is itself a finding — looks unfinished.
- **Console errors**: not measurable (file:// blocked by Playwright); static inspection shows two well-formed scripts (`sortTable`, `openHypothesis`) with no syntax errors.
- **Broken links**: 0 broken at the URL level — all GitHub-blob links are absolute, all cross-reference `experiments/*.html` targets exist on disk. **PASS on Rule 27.**
- **Missing seed-count framing**: count = ALL per-experiment-page KN-strips + aggregate dashboard table cells (~80+ rows). Single biggest "this isn't a top-tier dashboard yet" gap.
- **Aggregate dashboard footer self-describes wrong font stack** — confessional but inconsistent with per-experiment pages.
- **Mojibake / encoding**: no `&amp;amp;` double-escape spotted; `&mdash;` and HTML entities decode correctly in modern browsers. Two stray `**` literals found inside otherwise-rendered markdown blocks (sci-critic preamble on pair_gm_pdw).

## Required fixes (priority order)

1. **Wire the FINDINGS block through the same markdown converter** used for sci-critic and impl-critic. Specifically: the converter currently turns `*` / `_` / `` ` `` into inline tags but does NOT handle block-quotes (`> `) or GFM tables (`|...|---|...|`). Add either `markdown-it`-style table support or a small custom parser that recognises `>` block-quotes and `|...|` table rows. Apply to both the per-experiment Verdict section AND the aggregate dashboard's `.headline-ribbon`.
2. **Apply Source Serif 4 to the aggregate dashboard** consistent with per-experiment pages. Replace `'Newsreader',serif;font-style:italic` headings + `'IBM Plex Serif'` body with `'Source Serif 4'` to match `dashboard/experiments/*.html`. Update the line-234 footer self-description string.
3. **Add seed-count framing** as a visible pill / column / badge wherever a numerical claim is presented: KN-strip on per-experiment pages (`+1.34 pp Δ vs baseline (n=3, evaluation)`), aggregate leaderboard tables (a `Seeds` or `n` column), and the headline ribbon (`Phase-8 winners (n=3 seeds, evaluation gate)`). This is Rule 28 compliance.
4. **Add a footer commit-SHA / build-stamp to the aggregate dashboard** matching the per-experiment-page `.meta` pattern. Link the SHA to its GitHub commit URL.
5. **Link the Pages-live URL prominently** from the aggregate dashboard header — e.g., a `📡 live` pill next to the existing `GitHub` / `FINDINGS.md` row.
6. **Hide the `Uncategorised` group-section when count == 0** (line 233). Or merge it into a single grey "0 unmapped" badge above the leaderboard.
7. **Remove the dead `<div id='side-panel'>` + `closeSide()`** or restore it to a working role. Currently orphaned DOM.
8. **Add reasoning.json for the three Phase-8 winners** (pair_gm_pdw, slot_act_sine, sg_only_phi_budget — all three currently show `No reasoning.json for this run directory`). Per Rule 5, the post-run `verdict + learning` fields should exist for headline-defending runs.
9. **Fix the stray `**` literal in the sci-critic preamble** on pair_gm_pdw (line 203) — single-line markdown-converter regression on bold delimiter pairing.
10. **Show multi-hypothesis pills on combo* pages**: combo2 is H09+H48, combo3 adds H44, … — the H-pill row should show all stacked hypotheses, not just the leading one.

## Verdict reasoning
A NeurIPS supplementary-material dashboard must look final. The per-experiment pages get ~80 % of the way there; the aggregate dashboard is still in mixed-state with the most prominent block (the headline ribbon) showing raw markdown to the camera. Two of the three deliverables claimed by commit `5194814` — "render FINDINGS as proper markdown" and "academic-restrained font swap" — landed only partially. A hostile reviewer landing on the Pages mirror will see literal `## ✅ PHASE-8 FINAL VERDICT` text and stop reading. Fix items 1–3 above and this becomes **WEAK_ACCEPT**; fix 1–5 and it becomes **ACCEPT**.
