---
name: autoresearch-doc-organization
description: Use when planning the repo's top-level layout, drafting / rewriting README.md, or moving root-level markdown into subdirectories. Enforces the ≤ 4 canonical root files rule, the conference front-door README pattern, and the paper/docs/experiments/hypotheses subdir contract. The 2026-05-29 cold-reader audit found 18+ markdown files in the root, drowning the conference signal.
metadata:
  rules_enforced: [31, 32]
  added: 2026-05-29
  origin_audit: audits/REVIEWER_PASS_PAPER.md, RESTRUCTURE_PLAN.md
---

# Skill — Repo-root discipline + conference-grade README

## When to use

- BEFORE creating any new `.md` file at the repo root — confirm it
  belongs in `paper/`, `docs/`, `experiments/`, `hypotheses/`, or
  `memory/` instead.
- When the README has accumulated marketing language, hides the
  negative-results section, lacks a quick start, or contains
  un-rebuked "ACCEPT" self-grades.
- After ANY restructure that moves root-level files into subdirs
  (e.g., 2026-05-29 `RESTRUCTURE_PLAN.md`) — verify nothing else
  is added to the root afterwards.

## The 4-canonical-files rule (Rule 31)

Repo root contains ONLY:

| file | purpose |
|---|---|
| `README.md` | conference front-door (see Pillar 2 below) |
| `CLAUDE.md` | normative rules for any future operator |
| `PAPER.md` | the paper itself (or `paper.pdf` once typeset) |
| `LICENSE` | license file |

Every OTHER markdown file lives in a subdirectory:

| subdir | what lives here |
|---|---|
| `paper/` | FINDINGS, MANIFESTO, PARADIGM_COMPARISON, paper_abstract, NEURIPS_CHECKLIST, LIMITATIONS, ETHICS_STATEMENT, STATISTICAL_TESTS, REVIEWER_CHECKLIST, SOTA_COMPARISON, SELF_AUDIT_CHECKLIST, AUDIT_SUMMARY, NATURE_INSPIRED_NETWORKS |
| `docs/` | GitHub Pages root; dashboard mirror |
| `experiments/` | per-run archives (per Rule 8) |
| `hypotheses/g<N>_*/` | the 75-hypothesis design docs |
| `ideas/<NN>_*/` | sub-projects (per Rule 14) |
| `audits/` | implementation + sci + reviewer audits (per Rule 22) |
| `skills/` | content-agnostic skills (per Rule 10) |
| `memory/` | per-session checkpoint markdown |
| `scripts/` | runner / sweep / dashboard / build scripts |
| `src/nature_inspired_networks/` | shared infrastructure |
| `tests/` | core unit tests |
| `configs/` | shared YAML configs |
| `RESTRUCTURE_PLAN.md` | **transitional only** — delete after the move completes |

## Cross-reference rewriting after a restructure

When a file moves from `<root>/FINDINGS.md` to `paper/FINDINGS.md`:

1. Grep the whole repo for `](FINDINGS.md)` and `(FINDINGS.md)` —
   rewrite to `](paper/FINDINGS.md)` for files at the root, or
   `](../paper/FINDINGS.md)` for files in a sibling subdir.
2. For HTML emitted to `docs/`: per Rule 27, the link must be
   `https://github.com/dlmastery/nature_inspired_networks/blob/main/paper/FINDINGS.md`
   (absolute GitHub-blob URL, NOT a relative Pages path).
3. For CLAUDE.md cross-references in the rule body, rewrite to
   `paper/FINDINGS.md`.
4. For `skills/<name>/SKILL.md` cross-refs, same treatment.
5. After all rewrites, run the Playwright link-sweep from
   `skills/autoresearch-link-discipline/` to confirm 0 broken links.

## Pillar 2 — README is the conference front-door (Rule 32)

The required structure, in EXACTLY this order:

```markdown
# nature_inspired_networks
> One-paragraph elevator pitch (≤ 80 words): what the project is,
> what the contribution is (protocol-as-contribution per
> audits/REVIEWER_PASS_PAPER.md), and one calibrated negative result
> alongside one calibrated positive — never the positive alone.

<!-- badges row: build, paper-pdf, license, Pages-live -->
[![paper](badges/paper.svg)](paper/PAPER.pdf)
[![pages](badges/pages.svg)](https://dlmastery.github.io/nature_inspired_networks/)
[![license](badges/mit.svg)](LICENSE)

## Quick start (≤ 4 commands)

```powershell
git clone https://github.com/dlmastery/nature_inspired_networks
cd nature_inspired_networks
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m nature_inspired_networks.runner --config configs\cifar10_sota_smoke.yaml --tag smoke --seed 0
```

## Headline findings

- **Positive:** ... (n=3, EVALUATION — paired Wilcoxon W=0, p=0.125,
  Holm-Bonferroni α'=0.0167; lift +1.34 pp).
- **Negative (equal prominence):** ... (n=1, SCREENING; falsifier
  pre-registered before sweep; not rebuttable post-hoc).

## Methodological notes

- Seed protocol: ... (Rule 6)
- Hardware contract: ... (Rule 2 §2)
- Screening vs evaluation framing (Rule 28): ...

## Repo map (one screen)

```
nature_inspired_networks/
├── README.md   ← you are here
├── CLAUDE.md   ← normative rules for any future operator
├── PAPER.md
├── LICENSE
├── paper/      ← FINDINGS, MANIFESTO, STATISTICAL_TESTS, ...
├── docs/       ← GitHub Pages mirror
├── ...
```

## Citation

```bibtex
@misc{nature_inspired_networks_2026,
  title  = {...},
  author = {...},
  year   = {2026},
  note   = {arXiv:XXXX.XXXXX}
}
```

## License + acknowledgements

MIT. ...
```

### What the README MUST NOT contain

- Marketing language ("revolutionary", "state-of-the-art", "novel
  framework"). The audits found such language as recently as
  PAPER.md commit `48df219`.
- A self-grading banner of the form "Reviewer-acceptance ACCEPT"
  or "FINAL" without the "Internal QA pass — external review
  pending" qualifier (Rule 37).
- Negative results buried below the fold. Positive and negative
  receive equal visual weight.
- A `Quick start` block that doesn't run from a clean clone. If
  the venv setup needs extra steps, document them. The cold-reader
  audit caught a 3-line quick start that silently required a `pip
  install torch==2.x` that wasn't shown.

## Repo-root cleanup script

```powershell
# scripts/check_root_files.ps1
$allowed = @("README.md","CLAUDE.md","PAPER.md","LICENSE","RESTRUCTURE_PLAN.md")
$rootMd = Get-ChildItem -Path . -Filter *.md | Where-Object { $_.Name -notin $allowed }
if ($rootMd) {
  Write-Error "Rule 31 violation: extra markdown at repo root: $($rootMd.Name -join ', ')"
  exit 1
}
Write-Host "Rule 31 PASS: repo root markdown is clean."
```

Run as a pre-push hook or in CI.

## Anti-patterns

- **Creating a new `.md` at root "just for now."** That "now" lasts
  six months and the cold-reader audit finds it.
- **README starts with project history / motivation instead of the
  elevator pitch.** First sentence MUST name the contribution.
- **Marketing the positive without the calibrated negative.** The
  hostile reviewer reads the README front-to-back; an absent
  negative-results section is a red flag, not a feature.
- **Linking to `paper/FINDINGS.md` from the README while the README
  is at root.** Use the relative path `paper/FINDINGS.md` — it works
  on GitHub blob view AND on a local clone.

## Cross-references

- CLAUDE.md Rules 31 (root cleanup), 32 (README front-door), 27
  (Pages-link discipline), 28 (screening-vs-evaluation framing
  surfaced in the README headline section).
- `audits/REVIEWER_PASS_PAPER.md` — origin findings for the
  no-self-grading + negative-results-equal-prominence rules.
- `RESTRUCTURE_PLAN.md` — the in-flight restructure transitioning
  the root files into `paper/` (delete this file once complete).
- `skills/autoresearch-link-discipline/` — link-rewriting protocol
  that pairs with any restructure.
- `skills/autoresearch-typography-and-rendering/` — the visual
  layer of the front-door surface.
