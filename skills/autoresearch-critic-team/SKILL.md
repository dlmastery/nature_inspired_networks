---
name: autoresearch-critic-team
description: Use when the user does not trust the existing test coverage — dispatch a parallel team of skeptical implementation-critic agents to audit hypothesis modules line-by-line. Each agent owns one thematic group, reads design doc + src + test, emits a structured PASS/MINOR/MAJOR/BROKEN verdict to `audits/G<X>_audit.md`. The goal is to catch shape-only tests, math errors, citation mismatches, and code-vs-doc divergence the implementer agents missed.
---

# Skill — Parallel implementation-critic team

## When to use

- The user says something like "I don't believe enough validation happened"
- A large agent-written codebase is about to be used for external
  claims (paper, headline, dashboard).
- Before any Fixer campaign — the audit IS the fixer's spec.
- Roughly every 50+ hypothesis implementations or 1000+ test lines.

## The doctrine — be skeptical, not confirmatory

Implementer agents trained to "make tests pass" often write **shape-only
tests** (assert `out.shape == (B, C, H, W)`) that don't verify the
mechanism. A passing test suite is **not** evidence the hypothesis is
implemented correctly. The critic's job is to find:

1. **Mechanism check** — does the code actually implement what the
   formal hypothesis claims? (PASS only if a test asserts mechanism,
   not just shape.)
2. **Math correctness** — constants right? off-by-one indexing? wrong
   axis/dim? wrong norm? `(1+√5)/2 = 1.6180339887` vs hardcoded 1.618
   drift? RNG/seed state? log/sqrt of possibly-negative?
3. **Test rigor** — shape-only test = MINOR (at best). PASS requires
   at least one mechanism-verifying assertion.
4. **Citation alignment** — does the cited arXiv paper actually
   describe the technique, or is it a name-drop?
5. **Falsifier reachability** — can you read off the relevant metric
   from a real run?
6. **Hidden bugs / cargo-cult** — softmax-invariant constant biases?
   learnable Parameter registered as buffer (or vice versa)?
   `torch.no_grad()` inside forward? mask buffer that's a Parameter
   (silently trainable)?

## Verdict tiers (strict)

- **PASS** — mechanism implemented correctly AND ≥ 1 mechanism-verifying
  test
- **MINOR** — mechanism correct, but tests are shape-only OR cosmetic
  issue (wrong year in citation, typo in docstring)
- **MAJOR** — mechanism partially wrong OR critical test missing OR
  documented behaviour differs from code
- **BROKEN** — code contradicts the hypothesis (e.g., zero-bias when
  the doc claims a non-trivial bias) OR doesn't run OR test asserts
  the wrong thing

Be conservative with PASS. Most agent-written code lands MINOR or worse.

## Partition by thematic group

Dispatch N agents in parallel, one per hypothesis group (G1..G8 in
this repo, or whatever the project's grouping is). Each agent:

- Reads `hypotheses/g<N>_*/H<NN>_*.md`, `src/.../<module>.py`,
  `tests/test_<module>.py` for every hypothesis in its scope (NO
  skimming — line-by-line).
- Writes a single `audits/G<N>_audit.md` with this template:

```markdown
# G<N> audit — <theme>
Reviewer: Critic-G<N> (expert critic)
Date: <YYYY-MM-DD>

## Summary
PASS: <list>   MINOR: <list>   MAJOR: <list>   BROKEN: <list>

## Per-hypothesis findings

### H<NN> — <name>
**Module:** `src/.../<module>.py`
**Verdict:** <PASS|MINOR|MAJOR|BROKEN>
**Mechanism check:** <1 paragraph, quote line numbers>
**Math correctness:** <findings or "verified">
**Test rigor:** <quote shape-only test asserts if any>
**Citation alignment:** <findings>
**Bugs / cargo-cult:** <specific line refs>
**Concrete fix (if needed):** <patch or instruction for Fixer>

## Group-level concerns
<patterns across the group>

## Recommended follow-ups (prioritized)
1. ...
```

## Output discipline

- Scoped `git add audits/G<N>_audit.md` — never `-A`.
- Retry-wrapped commit + push (see `autoresearch-multi-agent-dispatch`).
- DO NOT modify any `.py` or test file — critics LOG findings, they
  do not patch. Patching is the Fixer's job.
- DO NOT touch `experiments/`, `dashboard/`, or another group's docs.

## Return-to-coordinator format (≤ 250 words)

- Counts per verdict tier
- The 3 most damning findings (H-ID + 1 sentence each)
- Commit SHA

## Cross-references

- CLAUDE.md Rule 22 — dual-track audit before external claim.
- `autoresearch-scicritic-team` — the science-critic counterpart (does
  the hypothesis itself make scientific sense?).
- `autoresearch-fixer-campaign` — consumes the audits as fixer specs.
