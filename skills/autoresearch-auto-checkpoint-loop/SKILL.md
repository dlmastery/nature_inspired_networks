---
name: autoresearch-auto-checkpoint-loop
description: Use when launching any background task expected to run > 15 min (GPU sweep, parallel agent team). Run a companion background loop that auto-commits + pushes new artifacts every ~10 min with retry-wrapped scoped commits. A power outage during the multi-hour task must lose at most ONE run / ONE agent's work. Stop the loop when the task completes.
---

# Skill — Background auto-checkpoint loop for crash safety

## When to use

- BEFORE launching any background task expected to run > 15 min:
  CIFAR sweep, Phase-4 / Phase-5 GPU campaign, 8-agent parallel
  audit, fixer campaign, etc.
- This is the COMPANION to `autoresearch-checkpoint` (which is the
  one-shot commit skill). The auto-loop runs continuously while the
  long task runs.
- After a fresh recovery from a crash, restart the loop as soon as
  the new long task is launched.

## The loop — Bash background variant (cross-platform via Git Bash on Windows)

```bash
# 10-min cadence; up to 10 ticks = ~100 min. Adjust 60-180 min as needed.
for i in $(seq 1 10); do
  sleep 600
  if [ -n "$(git status -s experiments/<dataset>/)" ]; then
    git add experiments/<dataset>/
    git -c user.name="..." -c user.email="..." \
      commit -m "Auto-checkpoint: <campaign> results (tick $i)" \
      >/dev/null 2>&1 \
      && git push >/dev/null 2>&1 \
      && echo "tick $i: committed + pushed"
  else
    echo "tick $i: no new results"
  fi
done
echo "<campaign> auto-checkpoint loop finished"
```

Launch with `run_in_background: true`. The loop is bounded (10
iterations) so it terminates naturally if you forget to stop it; size
the iteration count to slightly exceed your expected campaign wall-
clock.

## PowerShell variant (Windows-native)

```powershell
for ($i = 1; $i -le 10; $i++) {
  Start-Sleep -Seconds 600
  $status = git status -s experiments/<dataset>/
  if ($status) {
    git add experiments/<dataset>/
    git -c user.name="..." -c user.email="..." `
      commit -m "Auto-checkpoint: <campaign> (tick $i)" *>$null
    if ($?) {
      git push *>$null
      if ($?) { Write-Output "tick $i: committed" }
    }
  } else {
    Write-Output "tick $i: no new results"
  }
}
```

## Stop discipline

When the foreground task finishes, EXPLICITLY stop the auto-loop with
`TaskStop` (Claude Code's `TaskStop` tool, or `kill` the PID). The
loop terminates on its own at the iteration count, but stopping early
prevents redundant ticks.

## Disjoint commit scope

The auto-loop commits ONLY `experiments/<dataset>/`. If you also
spawn parallel agents committing src/ files concurrently, the scopes
are disjoint — index.lock races are handled by each writer's retry
wrapper.

## Crash recovery (validates the discipline)

After a power outage / OS sleep / kernel panic:

1. `git fetch origin && git status -sb` — confirm what's on the remote.
2. Identify the surviving artifacts: should be ALL but the in-flight
   run when the crash hit.
3. Resume the campaign: `python scripts/run_sweep.py --skip-existing
   --only <campaign tags>` — only the missing run(s) re-train.
4. Restart the auto-loop alongside.

The discipline was validated in this repo on 2026-05-27 when the
machine crashed mid-Phase-2 smoke; ZERO source code was lost
(everything was on origin/main); the sweep resumed cleanly with one
re-run.

## Anti-patterns

- Foreground `sleep` — blocks the agent. Always background.
- An unbounded loop (`while true`) — survives forgotten stops but
  pollutes the commit log if forgotten for days. Bounded loops are
  safer.
- `git add -A` inside the auto-loop — would sweep in concurrent
  agents' in-flight src/ edits. Scope to `experiments/<dataset>/`.
- A single auto-loop committing for multiple unrelated campaigns —
  the commit messages become useless. One loop per campaign.

## Cross-references

- CLAUDE.md Rule 11 — periodic GitHub checkpoint (the foundation).
- CLAUDE.md Rule 20 — auto-checkpoint loop alongside long background
  tasks (the normative requirement to use THIS skill).
- `autoresearch-checkpoint` — the one-shot variant.
- `autoresearch-multi-agent-dispatch` — what runs the parallel agents
  whose work this loop is preserving.
