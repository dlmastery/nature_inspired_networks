---
name: autoresearch-multi-agent-dispatch
description: Use when 8+ independent code/doc tasks can run in parallel — pattern for dispatching N agents with disjoint file scopes, retry-wrapped commits, and index.lock contention handling. Each agent edits only its scoped files, commits with a 5-attempt retry loop (add→commit→push; on push fail pull-rebase + retry).
---

# Skill — Parallel multi-agent dispatch with disjoint scopes

## When to use

- Any time the workload partitions into 2+ independent groups (audits
  by hypothesis group, fixers per module family, doc-sync per file
  family).
- Sweep / GPU work is sequential by hardware constraint; ONLY docs /
  code / research / audit / critique work parallelises.

## The contract per dispatched agent

Each agent gets:

1. **Disjoint file scope** — list the exact files this agent is
   allowed to edit. Other agents own other files; do NOT touch them.
2. **Scoped `git add <specific paths>`** — NEVER `git add -A` in a
   multi-agent dispatch (would sweep in another agent's in-flight
   changes).
3. **Retry-wrapped commit + push** — wrap every commit in a 5-attempt
   loop: `git add` → `git commit` → `git push`; on push failure
   `git pull --rebase` then retry; 3-second wait between attempts.
   This handles both `index.lock` races and non-fast-forward
   collisions.
4. **`git -c user.name=… -c user.email=…`** for identity (no
   `--global` per CLAUDE.md Rule 16 fallout).
5. **Bounded structured return** ≤ 200-250 words: commit SHA + tier
   counts + top findings.

## The PowerShell retry pattern

```powershell
$ok = $false
for ($i = 0; $i -lt 5 -and -not $ok; $i++) {
  git add <scoped paths>
  if (-not $?) { Start-Sleep 3; continue }
  git -c user.name="..." -c user.email="..." commit -m "..."
  if (-not $?) { Start-Sleep 3; continue }
  git push
  if ($?) { $ok = $true } else { git pull --rebase 2>$null }
}
```

The Bash equivalent:

```bash
for i in 1 2 3 4 5; do
  git add <paths> && \
    git -c user.name="..." -c user.email="..." commit -m "..." && \
    git push && break
  git pull --rebase || true
  sleep 3
done
```

## Disjoint-scope design — how to partition

Plan the agent boundary BEFORE dispatch. The boundaries that worked in
practice:

| dispatch | partition axis | example |
|---|---|---|
| Implementation | thematic group (G1..G8) | one agent per `hypotheses/g<N>_*/` |
| Audit | thematic group + audit type | `audits/G<N>_audit.md` per agent |
| Doc-sync | doc family | one agent for IDEA_TABLE+INDEX; one for README+MINDMAP; one for EXPERIMENT_LOG+SOTA+ARCHITECTURE |
| Fixer | primary src file | one agent per primary `src/.../<module>.py` family |
| Dashboard | layer | one for build_dashboard.py + renderer, one for README links |

If two agents would touch the SAME file, MERGE their scopes into one
agent. (E.g. when fixer-A wanted `priors.py` for H21/H22/H28 and
fixer-B wanted `priors.py` for H35, both moved into Fixer-Priors.)

## Why retry-wrapped is mandatory

- Concurrent `git commit` on the same repo races on `.git/index.lock`.
  Without retry, one of the agents fails its commit.
- Concurrent `git push` races on remote refs. Without `pull --rebase
  + retry`, one of the agents permanently aborts.
- An auto-checkpoint loop (Rule 20) committing experiments/ also races
  with the agents committing src/. Retry-wrapped is the only safe
  multi-writer pattern.

## Anti-patterns

- `git add -A` from any agent → sweeps another agent's mid-write files
  into the wrong commit. **Always use scoped `git add <paths>`.**
- Dispatching N agents and waiting for ALL → cache cost is high; the
  prompt cache only lasts 5 min. Either commit per-hypothesis (many
  small) or batch within ONE agent's scope.
- Agents committing without retry → first push collision permanently
  loses one agent's work.

## Cross-references

- CLAUDE.md Rule 15 — hierarchical agent teams with SMEs.
- `autoresearch-auto-checkpoint-loop` — the background companion when
  the dispatched agents are long-running.
- `autoresearch-critic-team`, `autoresearch-scicritic-team`,
  `autoresearch-fixer-campaign` — concrete instantiations of this
  skill.
