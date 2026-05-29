---
name: autoresearch-link-discipline
description: Use to enforce link hygiene across all externally-facing artefacts (README, PAPER.md, dashboard HTML, per-experiment pages, hypothesis docs). Combines Rule 27 (no repo-root .md hrefs from Pages-served HTML — use absolute GitHub-blob URLs), Rule 38 (every link Playwright-HEAD-tested + first-mention linkification of models/datasets/techniques/arXiv IDs), and the append-only audit ledger discipline. The 2026-05-29 Playwright sweep found 340+ broken links.
metadata:
  rules_enforced: [27, 38]
  added: 2026-05-29
  origin_audit: audits/REVIEWER_PASS_DASHBOARD.md
---

# Skill — Link discipline + first-mention linkification

## When to use

- AFTER any restructure that moves files (e.g., root → `paper/`)
  — rewrite every cross-reference.
- AFTER any change to the dashboard generator or per-experiment
  page template — re-run the Playwright link sweep.
- BEFORE marking README / PAPER / FINDINGS "done" — every model
  name, dataset name, technique, arXiv ID, hypothesis ID, and
  CLAUDE.md rule must be hyperlinked on its FIRST mention.
- WHENEVER adding a new audit file — append, never overwrite.

## Three pillars

### Pillar 1 — Absolute GitHub-blob URLs from Pages HTML (Rule 27)

GitHub Pages publishes ONLY the `docs/` directory. Any link in
generated HTML that points to a non-`docs/` file using a relative
path `../FINDINGS.md` resolves on Pages to `/FINDINGS.md` and
returns **404**. The fix:

```python
# scripts/_link_helpers.py
GITHUB_BLOB = "https://github.com/dlmastery/nature_inspired_networks/blob/main"

def repo_link(path: str) -> str:
    """Convert a repo-rooted path to an absolute GitHub-blob URL.

    Use for any file OUTSIDE docs/ that is referenced from HTML.
    Use a relative path ONLY for files inside dashboard/ (mirrored
    under docs/dashboard/).
    """
    return f"{GITHUB_BLOB}/{path.lstrip('/')}"

# usage in build_dashboard.py
findings_url = repo_link("paper/FINDINGS.md")  # after Rule 31 restructure
paper_url    = repo_link("PAPER.md")           # stays at root
hyp_url      = repo_link("hypotheses/g1_scaling_growth/H09_phi_budget.md")
```

**Forbidden HTML patterns:**

```html
<!-- BROKEN on Pages: -->
<a href="../FINDINGS.md">FINDINGS</a>
<a href="../../hypotheses/g1_scaling_growth/H09.md">H09</a>
<script>fetch('../FINDINGS.md').then(...)</script>

<!-- CORRECT: -->
<a href="https://github.com/dlmastery/nature_inspired_networks/blob/main/paper/FINDINGS.md"
   target="_blank" rel="noopener">FINDINGS</a>
<a href="https://github.com/dlmastery/nature_inspired_networks/blob/main/hypotheses/g1_scaling_growth/H09_phi_budget.md"
   target="_blank" rel="noopener">H09</a>
```

Relative paths are permitted ONLY for siblings under `dashboard/`
(e.g., per-experiment pages linking to `experiments/<tag>.html`).

### Pillar 2 — First-mention linkification (Rule 38)

The first time any of the following appears in any artefact, it
MUST be a hyperlink:

| category | examples | link target |
|---|---|---|
| model architectures | ResNet-20, RegNetX-200MF, ViT-Tiny, ConvNeXt | arXiv abstract URL |
| datasets | CIFAR-10, CIFAR-100, Tiny-ImageNet, Spherical MNIST, MedMNIST | dataset homepage / arXiv |
| techniques | AdamW, SIREN, label smoothing, RandAugment, Mixup, CutMix, EMA | original paper arXiv URL |
| arXiv IDs | arXiv:2006.09661 | `https://arxiv.org/abs/2006.09661` |
| hypothesis IDs | H09, H41, H71 | `hypotheses/g<N>_*/H<NN>_*.md` (via `repo_link`) |
| CLAUDE.md rules | Rule 28, Rule 35 | `CLAUDE.md#rule-28-screening-vs-evaluation` (via `repo_link` + anchor) |
| skills | `autoresearch-paper-rigor` | `skills/<name>/SKILL.md` (via `repo_link`) |
| audit files | `audits/G1_audit.md`, `audits/REVIEWER_PASS_PAPER.md` | `repo_link("audits/...")` |

Common first-mention offenders flagged in the 2026-05-29 audit:
- "ResNet-20" mentioned 47 times in README+PAPER, never linked.
- "AdamW" mentioned 12 times, never linked.
- "SIREN activation" mentioned 6 times, only linked once
  (mid-paragraph in §5.5, not on first mention in the abstract).
- "Holm-Bonferroni" mentioned 8 times, never linked.
- "CIFAR-100" linked once to the wrong dataset version.

### Pillar 3 — Playwright link sweep (the binding contract)

```python
# scripts/verify_links.py
from playwright.sync_api import sync_playwright
from pathlib import Path
import urllib.parse, requests

DASHBOARD_ROOT = Path("dashboard")
DOCS_ROOT      = Path("docs")
TIMEOUT_S      = 10

def collect_hrefs(page) -> list[str]:
    hrefs = page.eval_on_selector_all(
        "a[href]",
        "els => els.map(e => e.getAttribute('href'))"
    )
    return [h for h in hrefs if h and not h.startswith("#")]

def head_test(url: str) -> tuple[bool, int]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=TIMEOUT_S)
        return r.status_code < 400, r.status_code
    except Exception:
        return False, -1

broken = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    for html in list(DASHBOARD_ROOT.rglob("*.html")) + list(DOCS_ROOT.rglob("*.html")):
        page = browser.new_page()
        page.goto(f"file:///{html.absolute().as_posix()}")
        for href in collect_hrefs(page):
            if href.startswith("http"):
                ok, status = head_test(href)
                if not ok:
                    broken.append((str(html), href, status))
        page.close()
    browser.close()

if broken:
    for h, link, status in broken:
        print(f"BROKEN {status} {link} in {h}")
    raise SystemExit(1)
print("Link sweep PASS: 0 broken across", len(broken), "links.")
```

Run as the LAST step of any "I fixed the links" turn. The fix is
not done until this passes.

## Append-only audits ledger (Rule 3 spirit + Rule 38)

Audit files in `audits/` are APPEND-ONLY. New audit passes:

- NEVER overwrite an existing `G<N>_audit.md` or
  `REVIEWER_PASS_*.md` — create a new file with a date suffix
  (`audits/REVIEWER_PASS_PAPER_2026-05-29.md`) or append a new
  dated section to the existing file.
- The PRIOR audit's verdict is preserved; the new pass references
  the prior verdict, never claims it was wrong without an
  inline-quoted rebuttal.
- The audit ledger index lives at `audits/AUDIT_SUMMARY.md` (after
  restructure: `paper/AUDIT_SUMMARY.md`) — list every audit file
  with its date + verdict + rebuttal status.

## Rewriting links after a restructure (Rule 31 transition)

```powershell
# scripts/rewrite_links_post_restructure.ps1
$pairs = @(
  @{old="](FINDINGS.md)";        new="](paper/FINDINGS.md)"},
  @{old="](../FINDINGS.md)";     new="](../paper/FINDINGS.md)"},
  @{old="](MANIFESTO.md)";       new="](paper/MANIFESTO.md)"},
  @{old="](STATISTICAL_TESTS.md)"; new="](paper/STATISTICAL_TESTS.md)"},
  @{old="](paper_abstract.md)";  new="](paper/paper_abstract.md)"},
  @{old="](AUDIT_SUMMARY.md)";   new="](paper/AUDIT_SUMMARY.md)"},
  @{old="](PARADIGM_COMPARISON.md)"; new="](paper/PARADIGM_COMPARISON.md)"},
  @{old="](NEURIPS_CHECKLIST.md)";   new="](paper/NEURIPS_CHECKLIST.md)"},
  @{old="](LIMITATIONS.md)";     new="](paper/LIMITATIONS.md)"},
  @{old="](ETHICS_STATEMENT.md)";    new="](paper/ETHICS_STATEMENT.md)"},
  @{old="](REVIEWER_CHECKLIST.md)";  new="](paper/REVIEWER_CHECKLIST.md)"},
  @{old="](SOTA_COMPARISON.md)"; new="](paper/SOTA_COMPARISON.md)"},
  @{old="](NATURE_INSPIRED_NETWORKS.md)"; new="](paper/NATURE_INSPIRED_NETWORKS.md)"}
)
# DO NOT run blindly; preview with -WhatIf, then commit a SINGLE scoped patch.
```

After the rewrite, run BOTH:
1. The Playwright link sweep (Pillar 3 above).
2. `scripts/check_root_files.ps1` (from `autoresearch-doc-organization`).

## Anti-patterns

- **Inline `fetch('../some.md')` to render markdown client-side.**
  Forbidden by Rule 27. Use `window.open(repo_link("..."), "_blank")`.
- **"I'll add the link the next time I'm in the file."** First-
  mention linkification at the moment of writing is the discipline;
  retrofitting takes longer than doing it once correctly.
- **Linking to the GitHub view (`/github.com/.../blob/`) for files
  INSIDE `docs/`.** Those files ARE on Pages — use the Pages URL
  for stylistic consistency (`https://dlmastery.github.io/...`).
- **Overwriting an audit file with a "v2" pass.** Append, don't
  overwrite. The prior verdict is part of the historical record.

## Cross-references

- CLAUDE.md Rules 27 (Pages-link discipline), 38 (Playwright-HEAD
  + first-mention linkification + audit ledger).
- `audits/REVIEWER_PASS_DASHBOARD.md` — 0 broken links cited but
  the Pages URL was missing entirely (Rule 38 first-mention gap).
- `skills/autoresearch-typography-and-rendering/` — sibling skill,
  same Playwright verification pass covers both arms.
- `skills/autoresearch-doc-organization/` — pairs with this skill
  during any restructure.
