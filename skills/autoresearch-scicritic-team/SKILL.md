---
name: autoresearch-scicritic-team
description: Use to critique the SCIENTIFIC MERIT of each hypothesis (not the code). Different from the implementation critic — sci-critic asks whether the hypothesis is novel, derivative, or numerology, independent of whether the implementation is correct. Output is an "Addendum: Research-Scientist Critique" section appended directly into each design doc.
---

# Skill — Parallel research-scientist critic team

## When to use

- After a large body of hypothesis docs has been written (≥ 30 docs).
- Before publication / external claim, to filter out numerology
  rediscoveries and unfalsifiable claims.
- Whenever the user asks "is this idea actually novel, or did someone
  already publish this?"

## How it differs from `autoresearch-critic-team`

| dimension | impl-critic | sci-critic |
|---|---|---|
| audits | the CODE — does it implement the doc? | the IDEA — is the doc itself defensible? |
| output | `audits/G<X>_audit.md` | addendum appended into each design doc |
| verdict | PASS / MINOR / MAJOR / BROKEN | NOVEL / DERIVATIVE / NUMEROLOGY / FALSIFIED / UNFALSIFIABLE |
| can both apply? | yes — they're orthogonal | yes — code can be correct (impl-PASS) while idea is numerology (sci-NUMEROLOGY) |

## Sci-verdict tiers (strict)

- **NOVEL+TESTABLE** — never previously studied; a falsifier exists.
- **DERIVATIVE+TESTABLE** — rediscovery of a known technique under a
  new name; a falsifier exists. (Usually the strongest defensible
  rating.)
- **NUMEROLOGY** — the φ / Fib / Platonic constant is decorative; any
  constant in a similar range would give the same effect. Suggest the
  specific control ablation that would prove it.
- **FALSIFIED** — already refuted by an existing experiment in the
  project's own data.
- **UNFALSIFIABLE** — too many simultaneously-changing variables; no
  outcome can be attributed.
- **INFRASTRUCTURE** — methodology improvement (e.g., 3-seed error
  bars), not a hypothesis.

## Per-hypothesis addendum template

The sci-critic appends THIS exact section to the END of each
`H<NN>_*.md` design doc:

```markdown

---

## Addendum: Research-Scientist Critique (<YYYY-MM-DD>)

*Reviewer: SciCritic-G<X> (elite-research-scientist critic). Critiquing the IDEA, not the implementation (impl audit at `audits/G<X>_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)
<3 sentences specific to this hypothesis>

### Mechanism scrutiny — is the "because" clause real or post-hoc?
<quote the doc's mechanism claim and critique>

### Confounds — what else could explain a positive (or negative) result?
<≥ 2 alternative explanations>

### Numerology check — does φ specifically matter?
<would 1.5, 1.7, 2.0 work equally well? specify the control ablation>

### Literature: precedent or rediscovery?
<has this appeared under another name? cite real arXiv>

### Expected effect size — skeptical a-priori re-prediction
<your 90% CI, not the doc's optimistic claim>

### Minimum-distinguishing experiment
<cheapest experiment that distinguishes this from a numerology placebo>

### Verdict
<NOVEL+TESTABLE | DERIVATIVE+TESTABLE | NUMEROLOGY | UNFALSIFIABLE | FALSIFIED | INFRASTRUCTURE + 1 sentence>
```

## Doctrine — disagree with the author

- Quote claims and challenge them. Citations like `(He 2016)` without
  arXiv ID violate Rule 4 — flag them.
- No hedging on numerology — call it when you see a φ that any other
  constant could replace.
- Cite REAL arXiv for "this is already known" claims. Use the format
  `Author YEAR VENUE 'Title' (arXiv:XXXX.XXXXX) — relevance`.

## Output discipline

- Scoped `git add hypotheses/g<N>_*/*.md` — never `-A`.
- ONE commit per group (not per hypothesis); the addendum is text-only
  and small.
- Retry-wrapped commit + push.
- DO NOT modify `.py` files, tests, audits/, dashboard/, experiments/,
  or other groups' docs.

## Return-to-coordinator format (≤ 200 words)

- Counts per verdict tier
- The single most damning critique (with H-ID)
- Commit SHA

## Dataset-aware verdict — UNTESTED_ON_RIGHT_DATASET (added 2026-05-29)

A hypothesis whose pre-registered falsifier specifies a dataset
that ISN'T in the sweep cannot earn a NUMEROLOGY / FALSIFIED
verdict from that sweep. Add to the tier list:

- **UNTESTED_ON_RIGHT_DATASET** — falsifier specified dataset X
  (e.g., tiled-texture, Spherical MNIST, WikiText-2); sweep ran on
  dataset Y (e.g., upright CIFAR-10); verdict deferred until the
  pre-registered dataset is available.

Example: H22 (toroidal φ-closure) pre-registers "tiled-texture or
wrap-aware synthetic dataset" as the falsifier. The May-2026
sweep ran on upright CIFAR-10. The correct verdict is
`UNTESTED_ON_RIGHT_DATASET`, **not** NUMEROLOGY — and the
reviewer audit (`audits/REVIEWER_PASS_PAPER.md` BLOCKER section D)
called the original NUMEROLOGY labelling out as "testing on the
wrong dataset and then concluding the hypothesis fails."

When in doubt, default to `UNTESTED_ON_RIGHT_DATASET` rather than
NUMEROLOGY. The latter is a strong claim; the former is the
honest verdict.

## Auditor-self-grading circularity disclosure (added 2026-05-29)

Per CLAUDE.md Rule 37, the sci-critic agents in this project
share a model family with the implementer + impl-critic + fixer
agents. The sci-verdict — NOVEL+TESTABLE in particular — is
therefore an **internal sci-QA verdict, not independent external
review**. When referenced externally:

- The "0 NOVEL+TESTABLE-AND-impl-PASS" rate in the May-2026
  campaign is internal; an external reviewer might revise upward
  (some "DERIVATIVE+TESTABLE" verdicts could be NOVEL+TESTABLE
  upon deeper literature review) OR downward (some "NOVEL+TESTABLE"
  verdicts could be DERIVATIVE upon discovery of prior art).
- The verdict tier is reported as a snapshot at commit-SHA, not
  as a permanent classification.
- External reviewer findings (e.g., `audits/REVIEWER_PASS_PAPER.md`
  showing H09 is a RegNet rediscovery the project's own §5.4
  admits) **override** the internal sci-verdict within the same
  commit that processes the audit.

## Cross-references

- CLAUDE.md Rules 22 (dual-track audit), 36 (dataset-aware
  verdicts + pre-registration), 37 (no self-grading + circularity
  disclosure).
- `audits/REVIEWER_PASS_PAPER.md` — origin of the
  UNTESTED_ON_RIGHT_DATASET tier and circularity disclosure.
- `autoresearch-critic-team` — the code-side counterpart, same
  circularity caveat applies.
- `autoresearch-paper-rigor` — the statistical arm: paper-level
  enforcement of these verdict tiers.
- The original sci-critic dispatch on this repo (May 2026)
  produced 0 NOVEL / 31 DERIVATIVE / 41 NUMEROLOGY / 3 FALSIFIED
  across 81 hypotheses — interpret these as internal-snapshot,
  not authoritative.
