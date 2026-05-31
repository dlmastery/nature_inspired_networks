---
name: autoresearch-idea-scaffold
description: Use when scaffolding a new idea sub-project from the _TEMPLATE/. Each idea = one hypothesis = own implementation, own tests, own audit, own experiments. Modular, mix-and-matchable. Domain-agnostic.
---

# Skill — Scaffold a new idea sub-project

## When to use

When introducing a *new hypothesis* into the project (i.e., a new
inductive bias, optimisation trick, regulariser, dataset, or
architectural primitive that deserves its own ablation).

Do NOT use this for trivial cleanups or refactors — those go in
existing modules.

## What is an "idea" exactly?

An idea is a falsifiable claim that maps to:

- a peer-reviewed citation (Author YEAR VENUE 'Title' arXiv:…)
- a single mechanism (one parameter or structural choice that
  changes between baseline and variant)
- a measurable outcome (composite metric Δ, sub-metric Δ)

If you can't write the citation, you don't have an idea, you have a
hunch — write it up as a `BACKLOG.md` line instead.

## Directory layout (copied from `ideas/_TEMPLATE/`)

```
ideas/NN_<short_name>/
├── README.md            — idea statement + lit review + hypothesis
├── IDEA.md              — formal claim + falsifier + predicted Δ range
├── implementation.py    — standalone module
├── tests.py             — correctness tests
├── AUDIT.md             — self-audit (weaknesses found)
├── IMPROVEMENTS.md      — fix log
├── VERIFY.md            — verification log: tests green, sanity checks
├── experiment.py        — idea-specific experiment driver
├── configs/             — YAML configs
├── experiments/         — per-experiment archives (see
│                          autoresearch-experiment-archive skill)
├── results.md           — auto-generated summary
└── dashboard/           — idea-level dashboard
```

## Scaffold recipe

```bash
NN=42; SHORT="my_idea"
cp -r ideas/_TEMPLATE ideas/${NN}_${SHORT}
# Now edit ideas/${NN}_${SHORT}/README.md and IDEA.md to fill in:
# - the citation
# - the hypothesis (≥ 50 words)
# - the falsifier (what observation would kill this idea?)
# - the predicted Δ range on composite
```

## README.md template (top-level for the idea)

```markdown
# H<NN> — <one-line idea title>

## TL;DR

<2-3 sentences>

## Citation

<Full citation in Citation Rigor format>

## Hypothesis

<Mechanism + expected Δ. Word count ≥ 50.>

## Falsifier

<What single observation would kill this idea? Be specific:
"If composite Δ ≤ -0.005 on <dataset>, this idea is DISCARDED.">

## Predicted Δ range

| metric | Δ vs baseline | rationale |
|---|---|---|
| composite | [+0.01, +0.03] | <one sentence> |
| top-1     | [+0.5, +2.0 pp] | <one sentence> |
| params    | [-20%, -5%]     | <one sentence> |

## Status

| stage | done? |
|---|---|
| implementation.py written | [ ] |
| tests.py green | [ ] |
| AUDIT.md filed | [ ] |
| IMPROVEMENTS.md addressed | [ ] |
| VERIFY.md sealed | [ ] |
| First experiment archived | [ ] |
| Verdict authored | [ ] |

## How to test this idea (idea-specific experiment strategy)

<What is the SINGLE BEST dataset / setup that would prove this idea?
Don't reuse the common rail unless it really is the best test.
Examples:
- For a rotation-equivariant idea: rotated CIFAR-10.
- For a convergence-speed idea: small data subset, plot epochs-to-target.
- For a robustness idea: corrupted-CIFAR.
- For an attention-sparsity idea: long-context perplexity.>

## Cross-idea interactions

<Which other ideas does this compose well with? Which conflict?
Where in `ideas/99_mix_all/` does it slot?>
```

## IDEA.md (formal contract)

```markdown
# Formal claim for H<NN>

## Claim

<one sentence, present tense, falsifiable>

## Falsifier

<one observation that DISCARDS this claim. Quote the threshold.>

## Pre-registered prediction

<exact numeric range on the composite + sub-metrics>

## Composite fingerprint at time of registration

<SHA-256 of the composite formula in `eval.py`>

## Signed-off by

<your name + date>
```

## AUDIT.md template (after first impl, before first run)

```markdown
# AUDIT — H<NN>

> Adversarial self-critique. Treat the implementation as if you are
> reviewing it for ICLR.

## Weaknesses found

1. **<one-line summary>** — <why it matters, where in the code>
2. ...

## Bugs caught by tests

- ...

## Bugs NOT caught by tests but suspected

- ...

## Mitigations queued for IMPROVEMENTS.md

- ...
```

## VERIFY.md template (before first archive)

```markdown
# VERIFY — H<NN>

## Tests

- [ ] tests.py passes all assertions (`python ideas/NN/tests.py`)
- [ ] No new pylint / ruff warnings on `implementation.py`
- [ ] No new type errors from `mypy ideas/NN/implementation.py`

## Sanity

- [ ] Vanilla flag combo produces identical output to literature baseline
- [ ] All flags combos forward without shape errors (smoke test)
- [ ] Param count within ±10% of predicted in README

## Signed off by

<your name + date>
```

## Hard rules

1. **No new idea without a citation.** Hunches go in BACKLOG.md.
2. **No experiment until VERIFY.md is signed.** This catches the
   bugs the previous CIFAR-10 sweep found (silent mask gradient
   leakage, group-conv shape error at stride 2, etc.).
3. **The IDEA.md falsifier is the contract.** If the observation
   actually fires, the idea is closed — no goalpost-shifting.
4. **Cross-idea interactions are documented.** When two ideas
   compose, both READMEs link to each other and to
   `ideas/99_mix_all/`.

## What "good" looks like

- A new contributor can read just `ideas/<NN>/README.md` and
  understand the idea, the citation, the falsifier, and the
  predicted Δ range — in 5 minutes.
- The AUDIT.md has at least 3 weaknesses listed; if it's empty, the
  audit was lazy.
- VERIFY.md is signed on a real date, not "TBD".

## Anti-patterns

- Numbering ideas sequentially without leaving gaps — leaves no room
  for inserting closely related ideas later. Use 10-spacing
  (`10`, `20`, `30`).
- Mixing two ideas in one sub-directory because they "feel related."
  Split them; the modular contract is they must be independently
  ablatable.
- README that doesn't link to the parent IDEA_TABLE.md row — orphan.


---

## Cross-references to CLAUDE.md rules

This skill implements Rules 8, 14: Rule 8 (mandatory per-experiment archive sub-directory) and Rule 14 (modular & reusable — thin composition wrapper, no duplicated code). See
[](https://github.com/dlmastery/nature_inspired_networks/blob/main/CLAUDE.md)
for the canonical rule statements.
