---
name: autoresearch-fixer-campaign
description: Use after a critic team has identified BROKEN/MAJOR findings. Dispatch parallel Fixer agents to patch code AND add mechanism-verifying tests (the ones the original implementer missed). After all fixers report, re-run the affected smoke rows and record a pre-fix vs post-fix table in FINDINGS.md.
---

# Skill — Fixer campaign with mechanism-verifying tests + post-fix re-run

## When to use

- Following an `autoresearch-critic-team` audit that surfaced ≥ 1
  BROKEN or MAJOR finding.
- Whenever the user says "fix the audit findings" / "patch and re-run".

## The contract — every fix has three parts

A Fixer agent CANNOT just patch code. Every fix is THREE deliverables:

1. **Patch** — change the code to implement the documented mechanism.
2. **Mechanism-verifying test** — add to the matching `tests/test_*.py`
   a new test that ASSERTS the mechanism (the one the original test
   missed). The new test must FAIL on the pre-fix code and PASS on
   the post-fix code.
3. **Confirm green** — run the test file via `__main__`; must end
   "All N tests passed."

If any part is missing, the fix is incomplete.

## Partition by primary src file

Plan the boundary BEFORE dispatch. When two findings share a file
(e.g., H21+H22+H28+H35 all in `priors.py`), merge them into ONE Fixer
agent. The repo's fixer breakdown (May 2026):

| fixer | findings | primary file(s) |
|---|---|---|
| PhiScaling | H06, H09 | `phi_scaling.py` |
| Priors | H21, H22, H28, H35 | `priors.py` + `cymatic_hex.py` |
| Growth | H08 | `dynamic_growth.py` |
| Graphs | H14, H23, H24, H30 | 4 graph modules |
| InitFilter | H31, H38 | `inits.py` + `fractal_filter.py` |
| Opt | H41, H47, H48 | `optimizers/regularizers/schedulers` + minimal `train.py` |
| G6 | H53, H54, H55, H59 | 4 G6 modules |
| G7 | H64, H67, H74 | 3 hybrid modules + `golden_rope` + `train.py` |

## Output discipline

- Each fixer: scoped `git add <its specific src + test files>`. Never
  `-A`.
- Retry-wrapped commit + push (5 attempts, pull-rebase fallback).
- The combo / sweep that's running in parallel: its Python process has
  ALREADY imported the old modules; your fix won't disturb the
  in-flight run. Post-fix re-runs see the new code.

## Post-fix re-run discipline (Rule 21)

After ALL fixers report:

1. Run the full test suite. Zero regressions tolerated.
2. Identify the affected sweep tags (the tags whose primary module
   was patched).
3. Launch a **re-smoke sweep** at seed 0 with `--skip-existing` and
   `--only <affected tags>`. The `--skip-existing` means already-run
   rows are skipped; only the patched-module rows re-train. Some tags
   may need to be DELETED from `experiments/<dataset>/` first so the
   re-smoke replaces them.
4. If a patched hypothesis was a Phase-4 / Phase-5 graduate (the
   project's "winner"), the re-smoke is NOT enough — also re-run the
   CIFAR-100 3-seed campaign for that hypothesis.

## The pre-fix vs post-fix table (mandatory in FINDINGS.md)

```markdown
| hypothesis | metric | pre-fix | post-fix | Δ | survives? |
|---|---|---|---|---|---|
| H09 phi_budget | CIFAR-10 top1 | 0.8554 | 0.????? | ???? | Y/N |
| H09 phi_budget | CIFAR-100 3-seed median | 0.5805 | 0.????? | ???? | Y/N |
| H06 golden_bottleneck | CIFAR-10 top1 | 0.6925 | 0.????? | ???? | Y/N |
| ...
```

A claim that doesn't survive its own bugfix is **retracted** in
FINDINGS.md, not silently re-stated.

## Anti-patterns

- Patching code without adding a mechanism-verifying test → the next
  agent will silently re-introduce the same bug.
- Patching without re-running the affected sweep → the old numbers
  remain in FINDINGS.md, invisibly stale.
- Catching the patch into a Fixer commit that ALSO updates dashboard
  HTML / FINDINGS — keep concerns scoped: Fixer commits code + tests,
  later commits update narratives.

## Playwright-verify the specific complaint (added 2026-05-29)

The single most-repeated mistake in the 2026-05-29 campaign was
claiming a fix shipped without verifying the SPECIFIC complaint
no longer reproduces. The markdown-rendering bug was "fixed"
THREE times (`5194814`, `f8a4011`, and an earlier dashboard pass)
before Playwright verification caught that block-quote tables
still leaked literal `|---|` to the camera.

The discipline: after every fix, run the verification PROBE that
targets the exact complaint, not a generic smoke. Patterns:

| complaint class | verification probe |
|---|---|
| markdown rendering on dashboard | `scripts/verify_markdown_rendering.py` (Playwright assert no literal `##`/`**`/`\|---\|` on the embedded blocks) |
| broken links in published HTML | `scripts/verify_links.py` (Playwright HEAD-test every href) |
| typography mismatch across surfaces | Playwright `getComputedStyle(el).fontFamily` assert on body of aggregate AND per-experiment pages |
| hypothesis mechanism bug | the mechanism-verifying test added in step 2 of "three parts" above MUST fail on pre-fix and pass on post-fix |
| Q&A test name missing (Rule 25) | grep `tests/` for the promised test name; create if absent |
| sweep row eps-confound (H41 class) | re-run at iso-config with the suspect knob held; report Δ |
| dataset-mismatched verdict | re-classify as UNTESTED_ON_RIGHT_DATASET; do NOT delete the data |

If the verification probe doesn't exist for the complaint class,
WRITE one as part of the fix. A fix without a probe regresses on
the next session.

The Fixer commit message MUST cite the probe and its output:

```
Fix <complaint> — patch + mechanism test + probe-verified

- patch: <file>:<lines>
- mechanism test: tests/test_<module>.py::test_<assertion> (FAIL pre, PASS post)
- probe: scripts/verify_<thing>.py (output below)
  <paste 1-3 lines of the verifier's PASS output>
```

## Cross-references

- CLAUDE.md Rules 21 (post-fix re-run), 25 (Q&A-test contract),
  29 (Playwright markdown verification), 38 (Playwright link
  sweep + audit ledger append-only).
- `audits/REVIEWER_PASS_DASHBOARD.md` — the half-fix that prompted
  the Playwright-probe discipline.
- `autoresearch-critic-team` — the source of fix specs; post-fix
  the Fixer routes back through the critic for verdict update.
- `autoresearch-multi-agent-dispatch` — the dispatch mechanism.
- `autoresearch-typography-and-rendering` — the probe pattern for
  markdown-rendering complaints.
- `autoresearch-link-discipline` — the probe pattern for
  broken-link complaints.
