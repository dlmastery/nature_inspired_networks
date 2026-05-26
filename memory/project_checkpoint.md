# Project checkpoint — nature_inspired_networks

Last updated: 2026-05-25 (auto-bumped by scripts/build_dashboard.py once it's wired).

## Status

- Project skeleton mirrors `dlmastery/autoresearchimage` layout.
- `NaturePriorBlock` implementation with six toggleable priors + three channel modes.
- Baseline ResNet-20 + NaturePriorNet stacks.
- Training pipeline (bf16 AMP, cosine LR, label smoothing, target-acc tracking).
- Metrics: top-1/5, params, FLOPs (fvcore), batch=1 GPU latency, rotation-equivariance error.
- Composite metric with SHA-256 Goodhart fingerprint.
- Citation Rigor + Reasoning Blob Completeness gates (`src/nature_inspired_networks/reasoning.py`).
- Sortable HTML dashboard generator with Pareto + ablation + curve + Betti plots.
- Persistent-homology Betti curve + linear CKA utilities.
- **Curated 11-row CIFAR-10 ablation COMPLETE** (seed 0, 12 epochs, 64 min wall-clock).
- `RESULTS.md` + `FINDINGS.md` auto-generated; pushed to GitHub.
- Pages live at https://dlmastery.github.io/nature_inspired_networks/dashboard/dashboard.html
- Next-campaign axes:
  - 3-seed re-sweep for error bars
  - leave-one-out from full hybrid (`scripts/run_sweep.py --full`)
  - rotated-CIFAR / IcoMNIST where C4 group prior should pay off
  - MedMNIST PathMNIST (loader is ready)
  - swap C4 max-pool for average-pool to recover the lost 75 % signal
  - trained-feature Betti curves (checkpoint saving is wired for the next sweep)

## Headline finding (CIFAR-10, seed 0, 12 epochs)

| rank | tag | top-1 | params | composite |
|---|---|---|---|---|
| 1 | `baseline_resnet20` | 84.78 % | 272 k | 0.8458 |
| 2 | `baseline_sg_vanilla` | 82.16 % | 186 k | 0.8258 |
| 5 | `sg_only_fractal` | 82.46 % | 259 k | 0.8104 |
| 10 | `sg_only_group` | 69.84 % | 127 k | 0.6937 |
| 11 | `sg_full_fib` | 73.24 % | 259 k | 0.6966 |

The full hybrid is the **worst** NaturePrior variant — the priors
conflict at this scale. The C4 group prior accounts for most of
the damage; max-pool over the 4-rotation orbit throws away 75 %
of the signal. Fractal is the only single prior that lifts top-1.

## Next-experiment command

```powershell
.\.venv\Scripts\python -u scripts\run_sweep.py `
   --config configs\cifar10_quick.yaml --seeds 0 1 2 --full --skip-existing
.\.venv\Scripts\python scripts\compute_topology.py --seeds 0 1 2
.\.venv\Scripts\python scripts\build_dashboard.py
.\.venv\Scripts\python scripts\build_report.py
```

## Key files (line-of-truth)

- `src/nature_inspired_networks/priors.py` — nature-inspired primitives
- `src/nature_inspired_networks/blocks.py` — `NaturePriorBlock`
- `src/nature_inspired_networks/models.py` — `NaturePriorNet`, `ResNet20`
- `src/nature_inspired_networks/runner.py` — single-experiment runner
- `scripts/run_sweep.py` — ablation matrix driver
- `experiments/experiment_log.jsonl` — append-only run log
- `experiments/reasoning_annotations.json` — citation-gated reasoning entries
- `dashboard/dashboard.html` — sortable run table + plots
