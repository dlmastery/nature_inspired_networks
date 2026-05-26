---
name: autoresearch-checkpoint
description: Use whenever progress has been made — code edit, test passing, run folder produced, dashboard refreshed, ledger updated. Commits and pushes a checkpoint to GitHub so a power outage / session crash never loses progress. Default cadence ≤ 15 min during active work; mandatory before and after every background training task.
---

# Skill — Periodic GitHub checkpoint

## When to use

Continuously. This is not a one-shot skill — it's the heartbeat of
the project. Specifically:

| trigger | priority | what to commit |
|---|---|---|
| File edit complete and tests green | high | the edited files + test output |
| Background training task about to launch | mandatory | code + configs as they will run |
| Background training task completed | mandatory | run-folder artifacts + log + dashboard refresh |
| New / updated `IDEA_TABLE.md`, `EXPERIMENT_LEDGER.md`, `FINDINGS.md` | high | the document |
| Skill / `CLAUDE.md` / `ARCHITECTURE.md` updated | high | the document |
| Every ~15 min of active editing | medium | whatever changed |
| Wake-up from `ScheduleWakeup` | mandatory | check `git status` first thing |

## Why

- Power outage on a Windows laptop with no UPS.
- OS killing background tasks on sleep / lid-close.
- The conversation's own context window exhausting before the work is
  committed.
- Recovery should always be from a GitHub remote, never from a local
  WIP state.

## How — the command

```powershell
cd <repo>
git status --short                       # quick sanity check first
git add -A
git -c user.name="<name>" -c user.email="<email>" commit -m "<msg>"
git push
```

For HEREDOC-style commit messages (multi-line):

```powershell
git commit -m @'
<title>

<body line 1>
<body line 2>
'@
```

## Commit-message contract

A good checkpoint commit message:

- **Title (≤ 70 chars)**: what changed at the highest level
- **Body**: what's recoverable from this checkpoint. If it's mid-flight
  work (training still running, dashboard partial), say so.
- **No `--no-verify`, no `--amend`.** Always a fresh commit.

Example bad message: `wip`
Example good message:

```
Mid-sweep checkpoint 3/11: baselines + sg_chan_* complete

Recoverable from this commit:
- baseline_resnet20  top-1 84.78%
- baseline_sg_vanilla top-1 82.16%
- sg_chan_fib        top-1 80.11%
Sweep still training sg_chan_phi seed=0; remaining 8 runs queued.
```

## Hard rules

1. **Push every commit.** Local commits on a dead laptop are
   worthless. `git push` is part of the checkpoint, not a follow-up.
2. **Many small commits beat one big commit.** Granularity matters
   for selective revert and bisect.
3. **Always glance at `git status` first.** Don't blindly `git add -A`
   if you suspect a secret or a large binary slipped in.
4. **Never `--no-verify`.** Hooks exist to catch bugs.
5. **Background-task launch is a checkpoint trigger** — commit
   BEFORE the launch (so the launch state is recoverable) AND AFTER
   the task completes (so the artifacts are recoverable).

## Anti-patterns

- "I'll commit at the end of the turn." → No. Commit on every
  milestone.
- "I'll squash these into one nice commit later." → No. Granular
  commits preserve a useful history.
- Adding `.env`, `.venv/`, dataset tarballs, raw checkpoints larger
  than ~100 MB. → Use `.gitignore`; if a large binary must travel,
  use LFS or HF Hub.
- Pushing to a branch nobody else uses without rebasing first if
  there's an upstream change. → On a single-owner repo this is fine;
  on shared repos, `git pull --rebase` before pushing.

## Cross-references

- `memory/feedback_checkpoint_discipline.md` — the user directive
  that introduced this skill.
- `CLAUDE.md` § "Always-true assertions" rule 11 — the normative
  invariant.
- Every other SKILL.md assumes this discipline is active in the
  background.
