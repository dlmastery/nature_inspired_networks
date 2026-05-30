---
name: autoresearch-winner-archive
description: Use whenever an experiment becomes the new GLOBAL champion (composite > previous best AND status=KEEP). Archives a fully portable, self-contained subdirectory containing README, exact config, model checkpoint, frozen code snapshot, inference script, per-fold results, audit report, and (where applicable) a self-contained Colab/Jupyter reproduction notebook. The archive must be copy-pasteable to another machine and runnable end-to-end without the rest of the repo.
metadata:
  rules_enforced: [8, 9, 11]
  added: 2026-05-29
  origin: dlmastery/autoresearch (FX) + dlmastery/autoresearchindexstock (QQQ) + dlmastery/autoresearchspy (SPY)
---

# Skill — Archive a champion winner (fully portable)

## When to use

- A new experiment lands with `status == KEEP` AND `composite >
  previous_global_best`. The previous champion's archive stays
  immutable; this skill creates a NEW archive.
- A multi-seed re-run on an existing champion is finalised (the
  archive's `reproduction/seed_variance.json` is the deliverable).
- A baseline experiment is being "frozen" as the reference for a long
  campaign (`baseline` is itself a champion of the empty set).

This skill is DISTINCT from
[`autoresearch-experiment-archive`](../autoresearch-experiment-archive/SKILL.md),
which archives EVERY experiment in `ideas/<NN>/experiments/exp<NNN>_*`.
That skill is per-experiment; this skill is per-champion. A winner
archive is a SECOND archive on top of the per-experiment one — it adds
the frozen code snapshot, the inference script, the 14-section audit
report (via [`autoresearch-explainability-report`](../autoresearch-explainability-report/SKILL.md)),
and the reproduction record.

## Why

The sister `autoresearch` (FX) and `autoresearchindexstock` (QQQ) repos
discovered that a one-row leaderboard entry is not enough for a
deployable champion: the artefact that proves the win must be
*portable*. Someone who copies just the winner directory to another
machine and `pip install`s the listed deps must be able to:

1. Reproduce the headline number to within the published seed band.
2. Run inference on a fresh sample.
3. Read why this config won and what it superseded.

If any of those three breaks, the win is unfalsifiable.

## Directory layout (the unit of archive)

A winner archive lives at a project-relative path. The sister repos
use `autoresearch_results/winners/<tag>_exp<NN>_<desc>/`; this repo
uses `winners/<tag>_seed<N>/` under the per-idea directory OR
`winners/<dataset>__<tag>_seed<N>/` for cross-idea baselines. Either
way the structure inside is identical:

```
winners/<backbone-or-tag>_exp<N>_<short-desc>/
├── README.md                    ← full description (template below)
├── config.json (or .yaml)       ← exact config that produced the win
├── model_checkpoint.pt          ← portable weights + scaler + provenance
├── experiment_log_entry.json    ← the JSONL row for this run
├── per_fold_results.json        ← per-fold val + test breakdown
├── audit_report.md              ← 14-section explainability audit
│                                  (see autoresearch-explainability-report)
├── code/                        ← FROZEN snapshot of source at time of win
│   ├── implementation.py        ← or backbone.py / model.py / etc.
│   ├── train.py                 ← or runner.py
│   ├── features.py              ← or transforms.py
│   ├── splits.py                ← or dataset.py
│   ├── metrics.py               ← composite formula + fingerprint
│   ├── CLAUDE.md                ← the project rules AT TIME OF WIN
│   └── pyproject.toml           ← OR requirements.txt — exact deps
├── inference/
│   ├── predict.py               ← standalone inference (see template)
│   └── README_inference.md      ← how to load + run predictions
├── reproduction/
│   ├── reproduce_log.txt        ← output of the verification re-run
│   └── seed_variance.json       ← cross-seed results (3-7 seeds)
└── notebook/                    ← OPTIONAL (mandatory for FX/SPY/QQQ)
    └── colab_train_and_infer.ipynb  ← self-contained reproduction
```

**Total size budget: ≤ 200 MB.** If `model_checkpoint.pt` is larger,
either compress, store on HF Hub and link, or use Git LFS with the
SHA recorded in `README.md`.

## The model checkpoint (must be portable)

`model_checkpoint.pt` is a torch.save dict containing EVERYTHING needed
to rebuild + use the model without the source repo. Required keys:

| key | content |
|---|---|
| `model_state_dict` | trainable weights |
| `config` | hyperparameters dict (matches the run command) |
| `scaler_mean`, `scaler_scale` | normalisation parameters (for tabular / FX) |
| `feature_columns` | list of feature names in order (schema validation) |
| `target_columns` | list of target names (e.g., `['y']` or `['ret_1d', 'ret_5d']`) |
| `n_features` | int feature count |
| `composite` | the winning composite score |
| `composite_fingerprint` | SHA-256 of the composite formula at time of win |
| `description` | one-line summary |
| `backbone` (or `arch`) | model family identifier |
| `experiment_num` (or `tag`) | provenance |
| `git_sha` | repo HEAD at time of win |
| `seed` | the seed that produced the headline number |

The check: someone with ONLY this `.pt` file plus the `code/` snapshot
must be able to import the class, load weights, apply the scaler,
and predict.

## README.md template

```markdown
# Winner — <tag> (exp<N>) on <dataset>

## TL;DR

<2-3 sentences>. **Composite <X.XXXX>** (Δ vs prior champion: +Y.YYYY).
Headline metric: <top-1 / AUC / Sharpe / etc.>. Frozen `<date>`, git SHA
`<sha>`, seed <N>.

## 1. Scoreboard

| metric | seed | value | vs prior champion |
|---|---|---|---|
| composite | 0 | X.XXXX | +Y |
| <headline metric> | 0 | X.XXX | +Y |
| <fold-1 metric> | 0 | X.XXX | +Y |
| ... | | | |

## 2. Per-fold breakdown

| fold | regime / name | train | val | test | status |
|---|---|---|---|---|---|

## 3. Configuration

Full hyperparameter dump as a fenced YAML or JSON block. Anyone reading
this can paste it into a `--config` and re-run.

## 4. Architecture description

Layers, activations, skip connections, parameter count. For nature-prior
blocks: which flags are set, which mechanism each toggles.

## 5. Key insight

What CHANGED from the previous champion. Reference exp<N-K> and quote
the citation that motivated the change. The single most important
sentence in this README.

## 6. Training details

Epochs run, early-stopping epoch, training time on the recorded hardware
(GPU, VRAM, CPU pin), final LR, final loss.

## 7. Uncertainty / calibration metrics (where applicable)

Aleatoric, epistemic, confidence per fold. Calibration error (ECE).

## 8. Reproduction status

Seeds tested: [0, 1, 2, ...]. Variance observed: composite std, range.
Cross-seed median. Whether the headline is a single-seed cherry or the
across-seeds-median.

## 9. Inference (sample)

```python
from code.implementation import Model
ckpt = torch.load("model_checkpoint.pt", map_location="cpu")
model = Model(**ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
x = torch.randn(1, ckpt["n_features"])
print(model(x))
```

## 10. Trading / deployment strategy (for FX/SPY/QQQ-style projects)

Signal generation rules, entry/exit thresholds, position sizing (Kelly
fraction), per-regime expected performance, risk controls (max drawdown,
kill-switch), caveats and warnings.

## 11. Known limitations

What regimes / distributions has this model NEVER seen? Where will it
most likely fail in deployment?

## 12. Citations

Same Citation Rigor format as the per-experiment reasoning entry. The
primary citation here is the paper that motivated the WINNING change,
NOT the architecture's original paper.

## 13. Audit report

See `audit_report.md` for the full 14-section explainability audit
(autoresearch-explainability-report).

## 14. Provenance

- Repo: <url>
- Git SHA at time of win: `<sha>`
- Source experiment archive: `ideas/<NN>/experiments/exp<NNN>_<short>/`
- Pre-fix vs post-fix: <if applicable, ref FINDINGS.md row>
- Composite SHA-256: `<fingerprint>`
```

## Inference script template

`inference/predict.py` MUST:

1. Import only from `../code/` (the frozen snapshot) — no live repo
   imports.
2. Load the checkpoint, validate `n_features` matches input.
3. Apply the scaler (if any).
4. Run the model.
5. Print prediction + uncertainty (where applicable) in a clear table.
6. Include a `__main__` block with a worked example using `torch.randn`
   or a CSV sample committed alongside.

```python
"""Inference script for <tag> winner. Self-contained — only deps are
torch and what's in ../code/."""
import sys, json, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from implementation import Model

CKPT = Path(__file__).resolve().parents[1] / "model_checkpoint.pt"

def predict(x):
    ckpt = torch.load(CKPT, map_location="cpu")
    assert x.shape[-1] == ckpt["n_features"], \
        f"expected {ckpt['n_features']} features, got {x.shape[-1]}"
    if "scaler_mean" in ckpt:
        x = (x - torch.tensor(ckpt["scaler_mean"])) / torch.tensor(ckpt["scaler_scale"])
    model = Model(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        return model(x)

if __name__ == "__main__":
    ckpt = torch.load(CKPT, map_location="cpu")
    x = torch.randn(2, ckpt["n_features"])
    print(predict(x))
```

## After-archive verification (mandatory)

A winner archive is not "done" until the reproduction passes. Run:

```bash
cd winners/<tag>_exp<N>_<desc>/
python -m venv .venv && source .venv/Scripts/activate
pip install -r code/requirements.txt   # OR pip install -e code/
python inference/predict.py            # exits 0, prints prediction
# Re-run training from the frozen config:
python -c "from code.train import main; main('config.json', seed=0)" \
    > reproduction/reproduce_log.txt 2>&1
```

The reproduction's headline metric must match the published value to
within `± 0.5%` (single-seed runs) or `± 1 σ` (multi-seed median). If
not, FLAG IT in `README.md` § 8 and investigate BEFORE moving on.

## Hard rules

1. **Archives are immutable after creation.** A new champion creates a
   NEW archive; the old one is never edited.
2. **No symlinks.** Every file in the archive is a real file (so
   copying the directory copies the whole archive).
3. **Frozen code is FROZEN.** `code/` is a snapshot, not a symlink to
   live source. Editing `code/implementation.py` post-archive is a
   provenance violation.
4. **Self-contained.** No imports from the live repo. The archive
   must run on a clean machine with only the deps in
   `code/requirements.txt` (or `pyproject.toml`).
5. **Audit report is mandatory** for any winner that drives an
   external claim (FINDINGS headline, paper abstract, README badge).
   See [`autoresearch-explainability-report`](../autoresearch-explainability-report/SKILL.md).
6. **Reproduction log lives in the archive.** The verification re-run
   output goes to `reproduction/reproduce_log.txt`. If the re-run
   diverges > 0.5%, the archive is INVALID — fix or unwin.

## Anti-patterns

- "I'll add the README later." → No. The archive is the README plus
  the artefacts; without README, the artefacts are mystery files.
- Single-seed wins promoted to champion without a multi-seed
  variance check (Rule 6 of the project + sister-repo learnings:
  single-seed champions are often luck).
- `model_checkpoint.pt` without the scaler — inference silently runs
  on raw features and gives garbage.
- Reusing one `winners/` subdir for two different winners (e.g.,
  same tag, different seeds). Each champion is its own dir.
- "It reproduced within 2%" — the band is 0.5% single-seed / 1σ
  multi-seed. Wider gives a false sense of security.

## Cross-references

- [`autoresearch-experiment-archive`](../autoresearch-experiment-archive/SKILL.md)
  — per-experiment archive (preserved for every run); the winner archive
  is a SECOND archive on top.
- [`autoresearch-explainability-report`](../autoresearch-explainability-report/SKILL.md)
  — the 14-section audit report that ships inside `audit_report.md`.
- [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md) —
  the 7-step ritual that decides whether a run is KEEP.
- [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md) —
  the statistical-rigor floor that a winner archive's seed-variance
  block must satisfy before any external claim references it.
- CLAUDE.md Rule 8 — per-experiment archive sub-directory mandate
  (this skill extends the rule to champion-archives).
- CLAUDE.md Rule 9 — every experiment archive has a very detailed
  README.md (this skill extends the README to 14 numbered sections).
