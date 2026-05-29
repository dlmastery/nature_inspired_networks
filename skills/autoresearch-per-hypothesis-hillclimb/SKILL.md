---
name: autoresearch-per-hypothesis-hillclimb
description: Use after a single-config screening sweep has produced a candidate hypothesis, before any external claim is made about it. Runs a 20-25-trial coordinate-descent hill-climb over the (lr × wd × batch × seed × optimizer) cube on top of the screening config, generates a per-hypothesis dashboard at ideas/<NN>/dashboard/index.html, and statistically confirms the best config with a 3-seed re-run.
---

# Skill — Per-hypothesis hill-climb (the proper evaluation tier)

## When to use

- **AFTER** an `autoresearch-ablation-sweep` (or `scripts/run_sweep.py`)
  screening pass has surfaced a candidate hypothesis with a notable
  single-config result on CIFAR-10 — positive (KEEP) or even mildly
  negative (NEAR-MISS).
- **BEFORE** any external claim is made (FINDINGS.md headline, paper
  abstract, README badge, CIFAR-100 graduation). Screening alone is
  NOT sufficient for an external claim.
- After an `autoresearch-fixer-campaign` Fixer agent patches a
  hypothesis: re-screen first, then hill-climb to re-establish the
  tuned ceiling.
- Whenever the user asks to "tune", "hill-climb", "properly evaluate",
  "confirm with seeds", or "promote H## to a winner".

## The lesson this skill encodes (project history)

The project's first 35 hypothesis evaluations were **screening-only**:
each H## ran at the project-default
`lr=1e-3, wd=5e-4, batch=256, seed=0, AdamW` recipe with all other
axes frozen. That conflates "bad hypothesis" with "good hypothesis at
the wrong config". A negative screening result is a hypothesis about
the default recipe — not about the prior itself. The May 2026 audit
campaign added Rule 21 (post-fix re-run), Rule 22 (dual-track audit),
and Rule 23 (orthogonal-axis compounding); this skill closes the
remaining gap by REQUIRING a tuned-config evaluation before any
external claim. Treat screening + hill-climb as a two-stage funnel:
~80 cheap screening configs at default → ~5 hill-climbed candidates
at tuned configs → top-K Phase-4 CIFAR-100 graduates.

## The 5-axis search cube (default)

| axis | values | rationale |
|---|---|---|
| `lr` | {3e-4, 1e-3, 3e-3} | log-uniform 1-decade window centred on AdamW default 1e-3; cited Loshchilov & Hutter 2019 (AdamW), Smith 2017 (LR-range test) |
| `weight_decay` | {1e-4, 5e-4, 2e-3} | log-spaced; brackets the AdamW default 5e-4 by a factor of 4 each side |
| `batch_size` | {128, 256} | batch-size–LR coupling per Smith et al. 2017; halves and stays at the project default 256 |
| `seed` | {0, 1, 2} | reserved for the final confirmation; coordinate descent stays on seed 0 to keep the budget bounded |
| `optimizer` | {AdamW, SGD+mom0.9} | the two families with documented CIFAR-10 SOTA recipes (He 2016 SGD; Loshchilov 2019 AdamW) |

Full Cartesian grid = 3 · 3 · 2 · 3 · 2 = **108** runs ≈ 14 GPU h on
an RTX 4090 Laptop at 12 epochs / run. The hill-climb finishes in
**20–25 runs ≈ 3.3 GPU h** with negligible regret (the literature on
coordinate descent for HP tuning is unanimous: ≤ 10 % composite gap
versus full grid on smooth landscapes; cite Bergstra & Bengio 2012
JMLR — random search beats grid in fewer trials; Hutter et al. 2014
Sequential model-based optimisation).

## Coordinate-descent algorithm (default)

```python
# Pseudocode
state = base_config_from_run_sweep(tag)        # lr=1e-3, wd=5e-4, bs=256, opt=AdamW, seed=0
best_top1 = run(state)                          # 1 run

for axis in [lr, wd, optimizer, batch_size]:    # NB: seed is the confirmation, not an HC axis
    candidates = grid[axis] - {state[axis]}     # skip the value already used
    for v in candidates:
        cand = state.copy(); cand[axis] = v
        top1 = run(cand)                        # +1 run per candidate
        if top1 > best_top1:                    # strict-> (Bergstra & Bengio 2012 §3.3)
            state, best_top1 = cand, top1
    # move on to the next axis at the (new) best

# Final 3-seed confirmation at the hill-climbed best config
for seed in {0, 1, 2}:
    if seed not in seeds_run_at(state):
        run({**state, seed=seed})
```

Total runs = 1 (start) + 2 (lr alt) + 2 (wd alt) + 1 (opt alt)
+ 1 (bs alt) + 2 (3-seed confirm) = **~9 runs minimum**. With a
revisit of the lr-and-wd axes after the optimizer flip (a
mini-second-pass is recommended when SGD wins, since SGD's LR scale
differs from AdamW's by ~10×), the typical campaign is **20-25 runs**.

The strict-`>` champion rule (a tied later candidate does NOT
replace the champion) protects against seed-noise being mistaken for
real improvement — same discipline as `autoresearch_dsbench`'s
`framework/hill_climb.py`.

## Alternative algorithms

`scripts/run_hillclimb.py` supports three flavours via `--algorithm`:

| algorithm | when | runs |
|---|---|---|
| `coordinate` | DEFAULT. Smooth landscape, axis-decomposable. | 20-25 |
| `random` | Suspected interactions between axes (e.g., LR × WD). Draws `--budget` uniform samples from the cube. | `--budget` (default 25) |
| `grid` | Validation / sanity check. Visits all `lr × wd × batch × optimizer` cells at the chosen seed; rarely worth the cost. | up to 36 |

`random` is the recommended fallback when coordinate descent surfaces
a multi-modal landscape (best lr changes when the optimizer flips —
common AdamW↔SGD behaviour). When in doubt, run coordinate first,
inspect the per-axis Pareto plot in the per-hypothesis dashboard,
and re-launch with `--algorithm random` only if the coordinate sweep
shows non-monotone responses.

## Output contract

### Per cell

Each (lr, wd, bs, opt, seed) cell writes to:

```
experiments/<dataset>/<tag>__hc_lr<LR>_wd<WD>_bs<BS>_opt<OPT>_seed<SEED>/
├── metrics.json     # RunMetrics, with extra fields: config_id, hc_axis, hc_step
├── history.json
└── best.pt
```

`config_id` is a deterministic hash `f"lr{LR}_wd{WD}_bs{BS}_opt{OPT}"`
so the per-experiment page renderer
(`autoresearch-per-experiment-page`) can group seeds under the same
config.

### Per hypothesis

Top-level summary at `ideas/<NN_short>/hillclimb_results.json`:

```jsonc
{
  "tag": "sg_only_phi_budget",
  "algorithm": "coordinate",
  "budget": 25,
  "cube": {"lr": [3e-4, 1e-3, 3e-3], ...},
  "base_config": {"lr": 1.0e-3, "wd": 5.0e-4, "bs": 256, "opt": "AdamW"},
  "best_config":  {"lr": 3.0e-3, "wd": 5.0e-4, "bs": 256, "opt": "AdamW"},
  "best_top1_median": 0.8617,
  "best_top1_min":    0.8580,
  "best_top1_range":  0.0061,
  "seeds_confirmed":  [0, 1, 2],
  "cells": [
    {"config": {...}, "seed": 0, "top1": 0.8554, "composite": 0.6109, "wallclock_s": 487.3},
    ...
  ],
  "fingerprint": "<COMPOSITE_FORMULA sha256 first 12>"
}
```

### Per-hypothesis dashboard

`ideas/<NN_short>/dashboard/index.html` — independent page,
NOT a modal off the master dashboard. Must contain:

1. **Best-config callout** — top of page, monospace box, "lr=3e-3
   wd=5e-4 bs=256 opt=AdamW → 0.8617 ± 0.0030 (3 seeds)".
2. **Cells table** — sortable, one row per cell, columns:
   config_id · lr · wd · bs · opt · seed · top1 · composite · Δ-vs-best ·
   wallclock · link to per-experiment page.
3. **Per-axis Pareto plot** — small-multiples SVG, one panel per
   axis (lr, wd, bs, opt), top-1 on y, axis value on x, colour-coded
   by other axes' fixed values. Reveals non-monotone responses.
4. **Seed-stability std** — bar chart at the best config: top-1 for
   seeds {0, 1, 2}, the std written above the bars.
5. **Footer** — base config, hill-climb algorithm, total wall-clock,
   COMPOSITE_FORMULA fingerprint, link back to
   `dashboard/dashboard.html` (the master aggregate).

## Statistical confirmation gate (Phase-5)

A claim is `EXTERNAL-READY` only when:

```
best_top1_min(3-seed) > baseline_top1_max(3-seed)
```

i.e., the WORST seed at the tuned config beats the BEST seed of the
project-default baseline. The single-number metric is `top1`; the
audit-time metric is `composite` (Goodhart-resistant, see Rule 2 +
`eval.py:COMPOSITE_FORMULA`). The seed-min-vs-baseline-max gate is
the same Phase-5 gate the repo already uses on the CIFAR-100 3-seed
re-run. A hill-climb best that fails this gate is recorded as
`NEAR-MISS` in `FINDINGS.md` and is NOT graduated to Phase-4.

## Cost discipline

| step | runs | wall-clock (4090 Laptop, 12 ep CIFAR-10) | cumulative |
|---|---|---|---|
| Screening (single config) | 1 | 8 min | 8 min |
| HC: lr axis | +2 | 16 min | 24 min |
| HC: wd axis | +2 | 16 min | 40 min |
| HC: optimizer axis | +1 | 8 min | 48 min |
| HC: batch axis | +1 | 8 min | 56 min |
| HC: second-pass revisit (optional) | +4 | 32 min | 88 min |
| 3-seed confirmation at best | +2 | 16 min | **104 min** ≈ 1.7 h |

The full default budget of **25 runs ≈ 3.3 GPU h** absorbs all of
the above plus 10 cells of slack. Hill-climbing **5** hypotheses
≈ **17 GPU h** ≈ overnight on a single laptop. Schedule the
campaign in waves: 2 hypotheses per evening, results commit-and-push
after each.

## Auto-checkpoint integration (Rule 11 / Rule 20)

The runner calls `_commit_checkpoint(tag, cell_id)` after every cell.
The commit is scoped to:

```
experiments/<dataset>/<tag>__hc_*/
ideas/<NN>/hillclimb_results.json
ideas/<NN>/dashboard/
```

so concurrent agent edits in `src/` are not swept up. The retry
wrapper (5 attempts, pull-rebase fallback) is identical to
`autoresearch-multi-agent-dispatch`. Running a separate background
auto-loop (Rule 20 / `autoresearch-auto-checkpoint-loop`) IS still
recommended for crash safety — the per-cell commits cover the happy
path; the auto-loop covers the case where `run_one` crashes mid-cell.

## Anti-patterns

1. **Full Cartesian grid (108 runs).** Wastes ~10 GPU h per
   hypothesis for an expected composite gain of < 0.5 pp over
   coordinate descent. Use only as a one-time validation on a single
   pilot hypothesis (e.g., `phi_budget`) to confirm the coordinate
   descent didn't miss a corner of the cube.
2. **Single-seed report at the best config.** Top-1 std across seeds
   on CIFAR-10 is routinely 0.3–0.6 pp at 12 epochs. A 1-seed
   "best" can be 0.5 pp noise. The 3-seed gate is mandatory.
3. **Cherry-picking the highest seed.** Reporting `max(top1)` over
   the three confirmation seeds is a Goodhart violation; the headline
   number is `median` and the gate uses `min`.
4. **Reporting hill-climb best WITHOUT the seed-noise gate.** A
   tuned config that beats the screening baseline by 0.1 pp on
   seed 0 alone is in the noise; do not claim it.
5. **Hill-climbing a BROKEN hypothesis.** If the
   `autoresearch-critic-team` audit gave a BROKEN verdict for this
   hypothesis's implementation, FIX FIRST
   (`autoresearch-fixer-campaign`), then hill-climb. Tuning a buggy
   prior produces a spurious tuned ceiling.
6. **Tuning the screening config in place.** The screening config
   in `scripts/run_sweep.py:build_matrix()` is the documented base.
   Edit it and every downstream comparison breaks (Rule 1). The
   hill-climb writes NEW directories under
   `experiments/<dataset>/<tag>__hc_*/`; the original screening row
   stays put.
7. **Skipping the per-hypothesis dashboard.** The dashboard is the
   deliverable; the weights are secondary (Rule 9). A hill-climb
   without `ideas/<NN>/dashboard/index.html` is incomplete.
8. **Hill-climbing on CIFAR-100 directly.** CIFAR-100 12-ep runs are
   ~30 min; a 25-run hill-climb is ~12 h. Hill-climb on CIFAR-10
   first, graduate the tuned config to Phase-4 CIFAR-100 at 3-seed.

## Cross-references

- **`autoresearch-ablation-sweep`** — the screening counterpart. The
  ablation gives the candidate; this skill gives the verdict.
- **`autoresearch-fixer-campaign`** — when hill-climb reveals an
  impl bug (e.g., the tuned ceiling is suspiciously below the
  baseline at the same config), defer to the fixer; do NOT keep
  re-tuning a broken module.
- **`autoresearch-per-experiment-page`** — the per-cell renderer
  this skill consumes. Every hill-climb cell gets its own page.
- **`autoresearch-auto-checkpoint-loop`** — companion crash-safety
  loop; pair it with the per-cell commits.
- **`autoresearch-combo-ladder`** — the orthogonal-axis stacking
  pattern. Hill-climb FIRST at the single-prior level, THEN combo;
  do not combo-stack untuned priors.
- **`autoresearch-scicritic-team`** — a hill-climbed winner with a
  NUMEROLOGY sci-verdict is NOT external-ready (Rule 22).
- **CLAUDE.md Rule 21** — post-fix re-run discipline (hill-climb is
  one of the re-runs).
- **CLAUDE.md Rule 22** — dual-track audit before any external
  claim (the hill-climbed tuned ceiling is the basis of the claim).
- **CLAUDE.md Rule 23** — orthogonal-axis compounding (hill-climb
  is per-prior; compound only after each prior is hill-climbed).
- **CLAUDE.md Rule 13** — SOTA smoke first (re-run the SOTA smoke
  if any environmental variable changes during the hill-climb).

## Operator commands

```powershell
# Hill-climb a single candidate hypothesis (default coordinate descent).
.\.venv\Scripts\python scripts\run_hillclimb.py `
  --tag sg_only_phi_budget --idea 04_phi_fib_width `
  --algorithm coordinate --budget 25 `
  --config configs\cifar10_quick.yaml

# Sanity-check the runner end-to-end (4 cells, 2 epochs each).
.\.venv\Scripts\python scripts\run_hillclimb.py `
  --tag baseline_resnet20 --idea _TEMPLATE `
  --algorithm grid --budget 4 --lr 3e-4 1e-3 `
  --wd 5e-4 --batch 256 --optimizer AdamW `
  --seeds 0 --epochs 2

# Random search alternative when coordinate hit a non-monotone axis.
.\.venv\Scripts\python scripts\run_hillclimb.py `
  --tag sg_only_phi_budget --idea 04_phi_fib_width `
  --algorithm random --budget 25 --seeds 0
```

## Closing reminder

The hill-climb is not "just more sweep". It is the protocol step that
distinguishes *"this prior helped at one config"* from *"this prior
helps when it has been given a fair shake"*. Every external claim
the project will make — paper headline, README badge, FINDINGS bold
row, CIFAR-100 graduation — must rest on a hill-climbed, 3-seed-
confirmed tuned ceiling. Anything weaker is screening, and the
project has already learned the hard way (35 screening-only verdicts
that had to be re-litigated) that screening alone is not enough.
