---
name: autoresearch-reasoning-entry
description: Use when authoring a citation-gated reasoning entry for an experiment (pre-run diagnosis/citations/hypothesis/prediction or post-run verdict/learning). Enforces Citation Rigor format and Reasoning Blob Completeness word-count floors.
---

# Skill — Author a reasoning entry

## When to use

Before launching ANY experiment that lands in `experiment_log.jsonl`.
Optionally amend post-run with the verdict + learning fields.

## Schema

```json
{
  "experiment_id": "exp001_<short_name>",
  "title": "One-sentence description",
  "diagnosis": "...",
  "citations": ["...", "..."],
  "hypothesis": "...",
  "prediction": "...",
  "verdict":  "",
  "learning": "",
  "composite": 0.0,
  "composite_fingerprint": "<SHA-256 of composite formula>"
}
```

## Word-count floors

| field | floor | post-run required? |
|---|---|---|
| `diagnosis` | 60 words | pre-run |
| `citations` | 40 words single-paper / 80 multi-paper (total) | pre-run |
| `hypothesis` | 50 words | pre-run |
| `prediction` | 25 words | pre-run |
| `verdict` | 30 words | post-run only |
| `learning` | 40 words | post-run only |

## Citation format

Every citation entry must contain (in any order):

- 4-digit year (1900–2099)
- A single-quoted `'Title'` of length ≥ 4 chars
- An `arXiv:NNNN.NNNNN` or `bioRxiv:NNNN.NNNNN` identifier
- A relevance note separated by `--` or `—` from the bibliographic
  prefix

The validator regex set (in the project's `reasoning.py`):

```python
CITATION_RE_YEAR  = re.compile(r"\b(?:19|20)\d{2}\b")
CITATION_RE_TITLE = re.compile(r"'[^']{4,}'")
CITATION_RE_ID    = re.compile(r"(?:arXiv|bioRxiv):\s*\d{4}\.\d{4,5}")
CITATION_RE_DASH  = re.compile(r"(?:—|--)\s*\S")
```

## Canonical citation example

```
He, Zhang, Ren, Sun 2015 CVPR 'Deep Residual Learning for Image
Recognition' (arXiv:1512.03385) -- Source of the canonical ResNet-20
CIFAR variant we use as the literature anchor.
```

## Examples that get rejected

- `(He2016)` — no title, no venue, no arXiv ID, no relevance note.
- `He et al. 2015 'Deep Residual Learning'` — no arXiv ID, no
  relevance.
- `He et al. 2015 'Deep Residual Learning' (arXiv:1512.03385)` —
  no relevance note after `--`.

## Field templates

### `diagnosis` (60+ words)

> exp<N-1> (<previous_tag>) showed <specific observation, with number>.
> The weakest cell of the current champion is <which fold / which
> metric>. Prior experiments ruled out <X>. The open question is
> <specific>. We need this experiment to test <claim>.

### `hypothesis` (50+ words)

> Because <flag X> changes <Y> in the model, mechanism-wise the
> network now <Z>. Per <cited paper>, we expect <effect> on <metric>.
> The cited result on <their dataset> reports <number>; on our
> 12-epoch CIFAR-10 budget that scales to <our number>.

### `prediction` (25+ words)

> composite in [<lo>, <hi>]; top-1 in [<lo>, <hi>]; params <delta>;
> latency <delta> ms; rotation equivariance error <delta>.

### `verdict` (30+ words)

> <KEEP|NEAR-MISS|DISCARD>. composite=<exact>, delta vs. global best
> <±X>. Per-fold: <fold1>=<value>, <fold2>=<value>. <one sentence
> explaining why the prediction held / didn't>.

### `learning` (40+ words)

> Axis <X> is <closed/open>. <one sentence on what we now believe>.
> Next try: <concrete next experiment tag>. Risk: <one risk that
> survived>.

## Anti-patterns

- Padding to hit the word floor with adverbs and qualifiers.
- Writing the verdict before reading the actual numbers.
- Citing a paper you have not opened.
- Predicting a range so wide it cannot be falsified.


---

## Cross-references to CLAUDE.md rules

This skill implements Rules 4, 5: Rule 4 (citation format) and Rule 5 (reasoning-blob word-count floors). See
[](https://github.com/dlmastery/nature_inspired_networks/blob/main/CLAUDE.md)
for the canonical rule statements.
