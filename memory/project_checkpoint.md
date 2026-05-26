# Project checkpoint — sacgeometry

Last updated: 2026-05-25 (auto-bumped by scripts/build_dashboard.py once it's wired).

## Status

- ✅ Project skeleton mirrors `dlmastery/autoresearchimage` layout.
- ✅ `SacredGeoBlock` implementation with six toggleable priors + three channel modes.
- ✅ Baseline ResNet-20 + SacredGeoNet stacks.
- ✅ Training pipeline (bf16 AMP, cosine LR, label smoothing, target-acc tracking).
- ✅ Metrics: top-1/5, params, FLOPs (fvcore), batch=1 GPU latency, rotation-equivariance error.
- ✅ Composite metric with SHA-256 Goodhart fingerprint.
- ✅ Citation Rigor + Reasoning Blob Completeness gates (`src/sacgeo/reasoning.py`).
- ✅ Sortable HTML dashboard generator with Pareto + ablation + curve plots.
- ✅ Persistent-homology Betti curve + linear CKA utilities.
- 🟡 Sweep currently running (curated 11-row CIFAR-10 ablation, 12 epochs, seed 0).
- ⬜ Push to `dlmastery/sacgeometry` after sweep completes.
- ⬜ MedMNIST PathMNIST run (stretch).
- ⬜ 3-seed re-sweep for error bars.
- ⬜ Trained-checkpoint Betti / CKA (current Betti is fresh-init).

## Next-experiment command

```powershell
.\.venv\Scripts\python -u scripts\run_sweep.py `
   --config configs\cifar10_quick.yaml --seeds 0 1 2 --skip-existing
.\.venv\Scripts\python scripts\compute_topology.py --seeds 0
.\.venv\Scripts\python scripts\build_dashboard.py
```

## Key files (line-of-truth)

- `src/sacgeo/priors.py` — sacred-geometry primitives
- `src/sacgeo/blocks.py` — `SacredGeoBlock`
- `src/sacgeo/models.py` — `SacredGeoNet`, `ResNet20`
- `src/sacgeo/runner.py` — single-experiment runner
- `scripts/run_sweep.py` — ablation matrix driver
- `experiments/experiment_log.jsonl` — append-only run log
- `experiments/reasoning_annotations.json` — citation-gated reasoning entries
- `dashboard/dashboard.html` — sortable run table + plots
