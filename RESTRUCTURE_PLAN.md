# Root-Level Restructure Plan — 2026-05-29

> Drafted before any `git mv`. This file documents the intended
> repository layout after the 2026-05-29 root-cleanup pass. The actual
> moves and cross-reference updates are committed in the following
> commits; revert via `git revert` if the move surfaces a regression.

## Goal

Reduce the ~26 root-level `.md` files to a clean front-door of 3
canonical files plus standard repo-hygiene files. Move every other
research / documentation artifact into a topical subdirectory so a
first-time visitor sees only the documents they actually need first.

## What stays at the repo root

| File | Why it stays |
|---|---|
| `README.md` | Front door — first thing GitHub renders. |
| `CLAUDE.md` | Project operator instructions, normative 28 rules. Must stay at root so `claude` picks it up automatically. |
| `PAPER.md` | The primary research deliverable; reviewers expect to find it at the root. |
| `pyproject.toml` | Build / install config. |
| `.gitignore`, `.gitattributes` | Standard git hygiene. |
| `.github/` (dir) | GitHub Actions, issue templates. |

No `LICENSE` file currently exists; one will be added separately and
will live at root for GitHub recognition.

## What moves

### → `paper/` (research deliverables)

| From | To |
|---|---|
| `MANIFESTO.md` | `paper/MANIFESTO.md` |
| `FINDINGS.md` | `paper/FINDINGS.md` |
| `AUDIT_SUMMARY.md` | `paper/AUDIT_SUMMARY.md` |
| `NATURE_INSPIRED_NETWORKS.md` | `paper/NATURE_INSPIRED_NETWORKS.md` |
| `PARADIGM_COMPARISON.md` | `paper/PARADIGM_COMPARISON.md` |
| `SOTA_COMPARISON.md` | `paper/SOTA_COMPARISON.md` |
| `NEURIPS_CHECKLIST.md` | `paper/NEURIPS_CHECKLIST.md` |
| `ETHICS_STATEMENT.md` | `paper/ETHICS_STATEMENT.md` |
| `LIMITATIONS.md` | `paper/LIMITATIONS.md` |
| `REVIEWER_CHECKLIST.md` | `paper/REVIEWER_CHECKLIST.md` |
| `STATISTICAL_TESTS.md` | `paper/STATISTICAL_TESTS.md` |
| `paper_abstract.md` | `paper/paper_abstract.md` |

### → `hypotheses/` (hypothesis-table and index)

| From | To |
|---|---|
| `IDEA_TABLE.md` | `hypotheses/IDEA_TABLE.md` |

### → `experiments/` (experiment registry and results)

| From | To |
|---|---|
| `EXPERIMENT_LOG.md` | `experiments/EXPERIMENT_LOG.md` |
| `EXPERIMENT_LEDGER.md` | `experiments/EXPERIMENT_LEDGER.md` |
| `RESULTS.md` | `experiments/RESULTS.md` |
| `NEXT_SWEEP_PRIORITY.md` | `experiments/NEXT_SWEEP_PRIORITY.md` |
| `SMOKE_PLAN.md` | `experiments/SMOKE_PLAN.md` |

### → `docs/` (project documentation, GitHub-Pages-served)

| From | To |
|---|---|
| `MINDMAP.md` | `docs/MINDMAP.md` |
| `ARCHITECTURE.md` | `docs/ARCHITECTURE.md` |
| `SETUP.md` | `docs/SETUP.md` |
| `AUTORESEARCH_PROCESS.md` | `docs/AUTORESEARCH_PROCESS.md` |
| `MEDIUM.md` | `docs/MEDIUM.md` |
| `sota_catalog.yaml` | `docs/sota_catalog.yaml` |

## Resulting root after the moves

```
nature_inspired_networks/
├── README.md                ← front door (rewritten in same campaign)
├── CLAUDE.md                ← normative 28 rules
├── PAPER.md                 ← main research deliverable
├── RESTRUCTURE_PLAN.md      ← this file (kept as a paper trail; can be moved to docs/ later)
├── pyproject.toml
├── .github/  .claude/  .gitignore  .gitattributes
├── paper/                   ← 12 research-deliverable .md files
├── hypotheses/              ← 84 design docs + IDEA_TABLE.md + INDEX.md
├── experiments/             ← sweep results + 5 registry .md files
├── docs/                    ← project documentation + sota_catalog.yaml + GitHub Pages root
├── ideas/                   ← modular sub-projects (1 per implemented hypothesis)
├── src/, scripts/, tests/, skills/, audits/, controls/, configs/, dashboard/
```

## Reference-update strategy

Every reference to any moved file is updated in the same commit as the
moves to keep the working tree internally consistent. Reference forms
to search for:

- bare filename (`FINDINGS.md`)
- `./FILE.md`, `../FILE.md`, `../../FILE.md`
- markdown link `[label](FILE.md)`
- absolute GitHub-blob `https://github.com/dlmastery/nature_inspired_networks/blob/main/FILE.md`

Each occurrence is rewritten to the new path, preserving the link form
(absolute GitHub-blob is preferred from generated HTML per Rule 27;
relative paths inside `paper/` or `experiments/` use sibling-relative
form).

## Rationale

- **Reviewer-front-door discipline** — a top-tier conference reviewer
  shouldn't have to scroll past 23 capital-letters `.md` files to find
  the README.
- **Topical grouping** — `paper/` cleanly identifies the research
  deliverables; `experiments/` collocates the registry markdown with
  the run archives it documents; `docs/` is what GitHub Pages serves.
- **Rule-27 compliance** — every cross-reference from generated HTML to
  a moved file uses the new absolute GitHub-blob URL, so Pages stays
  link-clean.
- **Reversibility** — `git mv` preserves history; `git revert` of the
  single restructure commit restores the previous layout.

## Out-of-scope for this campaign

- `audits/` — immutable record per Rule 3 spirit; do not move or
  rewrite.
- `hypotheses/g{1..8}_*/H*.md` — content unchanged; only cross-references
  to moved files are updated.
- `skills/*/SKILL.md` — content unchanged; only cross-references updated.
- `src/`, `tests/` — code untouched except for `dashboard.py` link
  construction.
- `experiments/cifar100/` — GPU sweep in flight; do not write.
