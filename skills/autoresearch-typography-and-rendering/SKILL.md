---
name: autoresearch-typography-and-rendering
description: Use when generating ANY externally-facing HTML artefact (dashboard, per-experiment page, GitHub Pages mirror, paper supplementary site) that embeds markdown sourced from .md files. Enforces the academic Source Serif 4 + IBM Plex Mono palette, the GFM-table + blockquote markdown pipeline, and the Playwright verification gate that the same regression has been claimed-fixed three times in one session.
metadata:
  rules_enforced: [29, 30]
  added: 2026-05-29
  origin_audit: audits/REVIEWER_PASS_DASHBOARD.md
---

# Skill — Academic typography + verified markdown rendering

## When to use

- Generating `dashboard/dashboard.html` or any
  `dashboard/experiments/*.html` per-experiment page.
- Writing the GitHub Pages landing page (`docs/index.html`).
- Embedding any sourced-from-.md content (FINDINGS verdict block,
  sci-critic addendum, impl-critic excerpt, hypothesis digest,
  paper abstract) into HTML.
- After ANY change to the dashboard generator (`scripts/build_dashboard.py`,
  `dashboard/build_dashboard.py`), re-run the Playwright verification
  pass — do NOT mark the change "done" without it.

## The two pillars

### Pillar 1 — Academic typography palette

Use ONLY:

```css
body {
  font-family: 'Source Serif 4', 'Source Serif Pro', Charter, Georgia, serif;
  font-feature-settings: 'kern','liga','onum';
}
h1, h2, h3, h4, h5, h6 {
  font-family: 'Source Serif 4', 'Source Serif Pro', Charter, Georgia, serif;
  font-weight: 600; /* never italic-as-emphasis */
}
code, pre, .mono, .commit-sha {
  font-family: 'IBM Plex Mono', ui-monospace, 'Cascadia Code', Consolas, monospace;
}
```

**Forbidden font families on research artefacts:**
- `Newsreader` — magazine display face, italic-as-emphasis.
- `Playfair Display`, `Lora`, `Crimson Pro` — editorial display.
- `Inter`, `Roboto`, `Open Sans` — generic SaaS sans (acceptable
  for utility UI chrome only, NOT body / headings / KPI labels).
- `font-style: italic` for headings or KPI labels — italic is for
  emphasis WITHIN running text, never as display chrome.

**Why a uniform stack across all surfaces matters.**
Per-experiment pages used Source Serif 4 before the aggregate
dashboard was migrated; the visual mismatch between the two
surfaces of the same artefact reads as unfinished work. Reviewer
audit cites this as a CRITICAL fail (`audits/REVIEWER_PASS_DASHBOARD.md`).

### Pillar 2 — Markdown rendering pipeline

Source-of-truth markdown blocks (FINDINGS verdict, sci-critic,
impl-critic, hypothesis digests, paper abstract) MUST go through
a converter that handles ALL of:

| construct | required | example |
|---|---|---|
| `**bold**` | yes | strong |
| `*em*` / `_em_` | yes | em |
| `` `code` `` | yes | code |
| `# H1` ... `###### H6` | yes | hN |
| `- bullet` / `1. ordered` | yes | ul / ol |
| `> blockquote` | yes | blockquote |
| `\| col1 \| col2 \|` GFM tables | yes | table / thead / tbody |
| `> > nested blockquote` | yes | nested blockquote |
| `> \| t1 \| t2 \|` table inside blockquote | yes | the bug class |
| triple-backtick fenced code | yes | pre > code |
| inline links `[txt](url)` | yes | a |

A converter that handles `**` and `*` but NOT GFM tables or
`>`-block-quotes is the bug class that has been "fixed" three
times this session (`5194814`, `f8a4011`, plus an earlier dashboard
pass). Use a tested library (e.g. `marked.parse(src, {gfm:true})`,
`markdown-it` with `markdown-it-attrs` + `markdown-it-table`) or a
custom parser that has the test fixtures below.

## Playwright verification gate (the binding contract)

After EVERY change to the dashboard generator, run this verification
BEFORE committing:

```python
# scripts/verify_markdown_rendering.py
from playwright.sync_api import sync_playwright
from pathlib import Path

DASHBOARD_ROOT = Path("dashboard")
LITERALS_THAT_MUST_NOT_APPEAR = ["## ", "**", "|---", "|---:|", "&gt;", "> |"]
EMBEDDED_MD_SELECTORS = [
    ".headline-ribbon",
    ".findings-verdict",
    ".sci-critic",
    ".impl-critic",
    ".hypothesis-digest",
]

def verify(page_path: Path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file:///{page_path.absolute().as_posix()}")
        for sel in EMBEDDED_MD_SELECTORS:
            el = page.query_selector(sel)
            if not el: continue
            text = el.inner_text()
            for lit in LITERALS_THAT_MUST_NOT_APPEAR:
                assert lit not in text, (
                    f"{page_path.name}: literal {lit!r} leaked in {sel}"
                )
        browser.close()

for html in DASHBOARD_ROOT.rglob("*.html"):
    verify(html)
print("Markdown rendering verified across", len(list(DASHBOARD_ROOT.rglob("*.html"))), "pages.")
```

Run this as the LAST step of any "I fixed the markdown rendering"
turn. The fix is not done until this passes.

## Font-loading hardening

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="preload" as="style"
  href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<link rel="stylesheet"
  href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;600&display=swap">
```

The `Charter, Georgia, serif` fallback chain renders without network
on the local dashboard.html when GitHub Pages is unreachable.

## Footer self-description must match reality

If the footer claims "Source Serif 4 / IBM Plex Mono" but the CSS
imports `Newsreader`, that is a Rule-30 violation AND a footer
self-description bug. Drive the footer text from the actual
`<link>` href so the two cannot drift.

## Anti-patterns

- **Claiming "markdown rendering fixed" without Playwright proof.**
  This is the regression that has shipped THREE times. Treat any
  PR / commit that touches the markdown path as suspect until
  Playwright verification has been re-run on at least 5 sampled
  pages (1 aggregate + 4 per-experiment, including the headline
  Phase-8 winner pages).
- **Different font stacks on aggregate vs per-experiment pages.**
  The single CSS variables block in `_shared.css` is the only
  acceptable place to set the family — both generators consume it.
- **Inline `font-style: italic` for headings / KPI labels.**
  Use weight (600) for emphasis, never italic.
- **CDN-only font loading without a serif fallback chain.** Offline
  / corporate-proxy / GitHub-Pages-edge-cache misses must still
  render in `Charter, Georgia, serif`.

## Cross-references

- CLAUDE.md Rules 29 (markdown rendering), 30 (typography palette),
  27 (Pages-link discipline — the link arm of the same audit).
- `audits/REVIEWER_PASS_DASHBOARD.md` — origin findings.
- `skills/autoresearch-dashboard/` — augmented to call out this
  skill as a prerequisite for any dashboard.html commit.
- `skills/autoresearch-per-experiment-page/` — page template now
  routes FINDINGS through the same converter.
- `skills/autoresearch-link-discipline/` — sibling skill for the
  link arm of the same Playwright verification pass.
