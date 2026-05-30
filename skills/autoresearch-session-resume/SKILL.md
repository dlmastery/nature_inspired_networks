---
name: autoresearch-session-resume
description: Use to maintain the self-contained crash-recovery checkpoint file (`memory/project_autoresearch_checkpoint.md`) that a fresh Claude Code session reads on startup to resume the autoresearch loop without re-reading any other file. Distinct from `autoresearch-checkpoint` (git commit cadence) and `autoresearch-auto-checkpoint-loop` (background commit loop for long sweeps). This skill is about the DOCUMENT that survives sessions — it carries the current champion, last experiment, exact next command, and full experiment history, so a new session picks up at the same experiment number, same champion, same next-experiment rationale. Ported from all five autoresearch sister repos, which each enforce a session-start checkpoint-read ritual.
metadata:
  rules_enforced: [11, 20]
  added: 2026-05-29
  origin: dlmastery/autoresearch + autoresearchspy + autoresearchimage + autoresearchtabular + autoresearchindexstock
---

# Skill — Self-contained crash-recovery checkpoint document

## When to use

- Session start: READ the checkpoint to recover context without
  re-reading logs / dashboards / READMEs.
- After every experiment: UPDATE the checkpoint with the new champion
  (if applicable) and the EXACT next-experiment command.
- During long reasoning / analysis (no experiment running): SAVE
  current thinking, diagnosis, and plan every 5 minutes. If you've
  been thinking for 3+ minutes without saving, STOP and checkpoint.
- Before starting any code change: SAVE so a crash during the edit
  doesn't lose experiment context.
- After any code change: SAVE the new code state + what was changed.

The Windows host this project targets (RTX 4090 Laptop, Intel HX) has
documented BSOD history under sustained compute (CLAUDE.md §26).
Treating the checkpoint as ephemeral has cost the sister repos
multi-hour campaigns; this skill makes the checkpoint a first-class
artefact.

## Why

This skill is DISTINCT from the two other checkpoint skills:

| skill | scope | trigger |
|---|---|---|
| [`autoresearch-checkpoint`](../autoresearch-checkpoint/SKILL.md) | git commit + push cadence | every milestone (file edit, test green, run-folder, dashboard refresh) |
| [`autoresearch-auto-checkpoint-loop`](../autoresearch-auto-checkpoint-loop/SKILL.md) | BACKGROUND git commit loop | alongside any sweep > 15 min |
| **`autoresearch-session-resume`** (this) | the CHECKPOINT DOCUMENT | every experiment + every 5 min of reasoning |

The first two are about git history. This one is about the SINGLE
markdown file a fresh Claude Code session reads at startup to fully
recover state — without reading any other file.

The sister repos (autoresearch FX, autoresearchspy SPY, autoresearchimage,
autoresearchtabular, autoresearchindexstock QQQ) ALL enforce a Session
Start ritual where step 1 is reading this document. Without it, a
fresh session re-reads the JSONL, the dashboard HTML, and several
markdown files just to figure out "what experiment number am I on, who
is the current champion, and what was I about to try next?"

## The checkpoint file

Lives at `memory/project_autoresearch_checkpoint.md` (or
`<results_root>/project_autoresearch_checkpoint.md` if the project's
`memory/` directory is reserved for skill / Claude memory).

The file is the source of truth for session-resumption. It is NOT a
diary — past entries get summarised + compressed; the document stays
≤ ~1500 lines so it loads quickly. The append-only experiment log
(`experiment_log.jsonl`) keeps the raw history; the checkpoint is the
distilled summary.

## Mandatory sections (in this order)

A fresh Claude Code session reads top-to-bottom. The order is chosen
so the reader can stop at any point and still take a productive next
action.

### 1. Session-start instructions (numbered steps)

The first thing a new session sees. Copy-pasteable. Example (adapt to
project):

```markdown
## Session Start — Do These In Order

1. Read this checkpoint (you're already here).
2. Tail the experiment log: `tail -5 experiment_log.jsonl`.
3. Confirm git status clean: `git status`.
4. Confirm data-split audit is green: run
   [`autoresearch-data-split-audit`](../skills/autoresearch-data-split-audit/SKILL.md).
5. Resume from section "Next Experiment" below — the command is ready
   to paste.
```

### 2. Current champion + composite score

```markdown
## Current Champion

- **Tag**: `<dataset>__<tag>_seed<N>`
- **Composite**: <X.XXXX>
- **Composite SHA-256 fingerprint**: `<frozen-fingerprint>`
- **Headline metric**: <metric_name> = <value>
- **Per-fold / per-regime**: <one-line summary>
- **Frozen on**: <date> at git SHA `<sha>`
- **Winner archive**: `winners/<tag>_exp<N>_<desc>/`
```

### 3. Last experiment result

```markdown
## Last Experiment (exp<N>)

- **Config delta from champion**: <what changed>
- **Composite**: <X.XXXX> (Δ vs champion: <±Y.YYYY>)
- **Per-fold deltas**: <fold-by-fold>
- **Verdict**: KEEP / DISCARD / NEAR-MISS
- **Why**: <one-sentence verdict explanation>
```

### 4. The EXACT next-experiment command

Copy-pasteable bash / PowerShell command. Includes ALL flags. If the
project uses YAML configs, include the config path AND the override
flags.

```markdown
## Next Experiment (exp<N+1>) — READY TO LAUNCH

```powershell
.\.venv\Scripts\python -m <pkg>.runner `
  --config ideas\<NN>\configs\<config>.yaml `
  --tag exp<N+1>_<short> --seed 0 `
  --root ideas\<NN>\experiments\exp<N+1>_<short>\run
```

**Rationale (≥ 50 words)**: <diagnosis + citation + hypothesis>.

**Predicted composite**: <range>; predicted per-fold: <one-line>.

**Pre-flight gates**:
- [ ] Unit tests green (`pytest tests/test_<module>.py`)
- [ ] Data-split audit green
- [ ] SOTA smoke green (if dataset switched)
- [ ] Reasoning entry authored AND validated
- [ ] git status clean + push
```

### 5. All wired parameters and their CLI flags

A table of every config knob the runner accepts. Prevents the common
bug of "I tried `--learning-rate` but the runner only accepts `--lr`"
(which the sister `autoresearch` flagged as a recurring waste of
runs).

### 6. Key learnings from exhausted axes

```markdown
## Exhausted Axes (don't re-try without new evidence)

- `seq_len` for backbone X: champion is N, tested {a, b, c, N, d, e} →
  N is monotonically best within fold-size constraint.
- `weight_decay` for backbone X: champion 7e-4, tested 5e-4/1e-3/2e-3
  within ±0.2 composite — axis closed.
- `lr` warmup: 1, 3, 5 epochs all hurt — axis closed.
```

This is the project's most valuable single artefact: it prevents
re-trying dead ends. If a new session re-runs `lr` warmup 1/3/5
because the checkpoint didn't say "closed", you've burned ~3 GPU-hours
recovering knowledge that already lived in a previous session.

### 7. Full experiment history table (compressed)

```markdown
## Experiment History

| # | tag | config delta | composite | per-fold | verdict |
|---|---|---|---|---|---|
| 1 | baseline_lit | — | 5.83 | [+, +, +, +, +, +, +] | baseline |
| 2 | residual_skip | + residual | 6.74 | [+, +, +, +, +, +, +] | KEEP |
| ... | | | | | |
| N | <last> | <delta> | <composite> | <folds> | <status> |
```

Old rows can be summarised after ~50 experiments (e.g., "exps 30–45:
weight-decay axis ablated; champion stayed at exp 28; all DISCARD").
Goal: a fresh session reads this table and immediately sees the
champion lineage.

### 8. Cumulative TODO list (with priorities)

Things known to be needed but not yet done. Each row has a priority
(P0 / P1 / P2) and a one-line rationale. Examples:

- P0 — multi-seed re-run of champion exp<N> (single-seed wins are
  often luck)
- P1 — feature ablation: top-3 permutation-importance features
- P2 — calibration analysis (winner archive Section 6)

### 9. Open mistakes / corrections log

Append-only. Each row records a mistake the user corrected so future
sessions don't re-make it. Example from sister `autoresearchindexstock`:

- Tried seq_len=120 without checking fold-size constraint (would have
  skipped 6/7 val folds). Corrected by user.
- Treated seed instability as deficiency when actual deficiency was
  per-fold regime failure. Corrected.

## Hard rules

1. **Self-contained.** A fresh session reading ONLY `CLAUDE.md` + this
   checkpoint must be able to resume. No "see also" references that
   require reading another file just to take the next step.
2. **Single source of truth.** If the checkpoint and the JSONL
   disagree on the current champion, the JSONL wins (it's the
   append-only ground truth) and the checkpoint is wrong — fix it.
3. **Update every experiment.** The 5-minute cadence is for
   reasoning; the per-experiment update is non-negotiable.
4. **Compress, don't grow.** When the file passes ~1500 lines,
   compress old experiment rows. The point is fast load + fast read.
5. **Never edit retroactively except to fix factual errors.** If a
   past verdict was wrong, add a "Correction:" note rather than
   silently rewriting.

## What "good" looks like

- A fresh session prompted with "continue the autoresearch loop"
  produces the NEXT experiment command + rationale within 30 seconds,
  having read ONLY this file + CLAUDE.md.
- The checkpoint's "Next Experiment" rationale matches the previous
  session's "Open question" — there is no discontinuity.
- The "Exhausted axes" section grows monotonically, never resets.

## Anti-patterns

- "I'll update the checkpoint at the end of the campaign." → No.
  Per-experiment.
- A checkpoint that references the JSONL by row index without
  summarising — the reader still has to open the JSONL.
- A checkpoint with TODOs but no "Next Experiment" command — the
  reader doesn't know what to do.
- An "Experiment History" table missing the last 10 experiments —
  if the table can't be trusted, the file is dead weight.
- Treating the checkpoint as the only history — the JSONL is the
  ground truth; the checkpoint is the summary.

## Cross-references

- [`autoresearch-checkpoint`](../autoresearch-checkpoint/SKILL.md) —
  the git-commit cadence that persists this file off-machine.
- [`autoresearch-auto-checkpoint-loop`](../autoresearch-auto-checkpoint-loop/SKILL.md)
  — the background commit loop that pushes this file every 10 min
  during long sweeps.
- [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md) —
  the 7-step ritual whose Step 7 (Checkpoint) updates this file.
- [`autoresearch-winner-archive`](../autoresearch-winner-archive/SKILL.md)
  — when this file's "Current Champion" updates, that triggers
  winner archiving.
- CLAUDE.md Rule 11 — periodic GitHub checkpoint discipline.
- CLAUDE.md Rule 20 — auto-checkpoint loop discipline.
- CLAUDE.md §26 — Windows thread-cap safety; the BSOD risk this skill
  defends against.
