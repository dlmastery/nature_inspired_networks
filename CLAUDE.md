# CLAUDE.md — normative rules for the nature_inspired_networks project

> This file tells future Claude/operator how to **safely** add an
> experiment, what may never change, and how the gates compose. It is
> the canonical reference — README/PROCESS/ARCH summarise it.

## Always-true assertions

1. **One config change per experiment.** Either a single flag in
   `flags:` flips, or `channel_mode` switches. No silent compounding.

2. **Composite formula is frozen.** Edit it and the next run fails
   with a fingerprint mismatch. If you have a *principled* reason to
   change the formula, open a new repo or new branch with a new
   fingerprint constant; do not silently rewrite history.

3. **`experiment_log.jsonl` is append-only.** Never edit a past row.
   To correct a row, append a *new* row with the same tag plus
   `_v2` and a journal entry explaining why.

4. **Citations follow the strict format.** Author list, year, venue,
   single-quoted title, arXiv/bioRxiv ID, relevance note. The validator
   in `src/nature_inspired_networks/reasoning.py` rejects everything else.

5. **No silent randomness.** `set_seed(seed)` is called at the top of
   `run_one()`. `torch.backends.cudnn.benchmark = True` is intentional
   (small per-run jitter is acceptable for ablations; the headline
   number is the seed-median composite over `--seeds 0 1 2`).

6. **No bypass flags.** If a gate refuses, fix the entry, do not
   disable the gate.

## Hardware contract

- Target: **1× RTX 4090 Laptop, 16 GB VRAM, Windows 11**.
- Batch sizes default to **256 with bf16 AMP**. If you change this,
  bump a new config file, do not edit the existing one — diffs need
  to be visible.
- `num_workers: 0` by default on Windows because workers' spawn
  start-method occasionally hangs in this environment. Raise it on
  Linux/WSL.

## Adding an experiment — checklist

1. Add a row to `scripts/run_sweep.py::build_matrix()` with a unique
   tag.
2. (Optional) Add a `reasoning_annotations.json` entry **before**
   training, with the 4 pre-run fields (`diagnosis`, `citations`,
   `hypothesis`, `prediction`).
3. Run:
   ```powershell
   .\.venv\Scripts\python scripts\run_sweep.py `
     --config configs\cifar10_quick.yaml `
     --only <your_tag> --seeds 0 --skip-existing
   ```
4. Inspect the row in `experiments/experiment_log.jsonl`.
5. Append the post-run fields (`verdict`, `learning`) to the same
   reasoning entry.
6. Rebuild the dashboard.

## When the runner refuses

| symptom | likely cause | fix |
|---|---|---|
| `ValueError: Reasoning entry rejected` | a word-count floor or citation format violation | rewrite the failing field; do **not** soften the floors |
| `CompositeFingerprintError` | someone edited the composite formula | revert the edit (or start a new branch with a new constant) |
| `MD5 mismatch on CIFAR tarball` | corp proxy injected something | `Remove-Item data/cifar-10-python.tar.gz` and re-download with `curl.exe -k` |
| `RuntimeError: CUDA out of memory` on >batch=256 | 16 GB laptop limit | drop batch to 128 in a new config; do not edit `cifar10_quick.yaml` |

## What may never go in this repo

- Real-name PII or PHI.
- Pre-trained ImageNet weights re-uploaded under our license (we link
  upstream).
- Closed datasets requiring registration (Tiny ImageNet, WILDS-C17)
  — load on demand at runtime, never check in.

## Sister projects

- [`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage)
  — protocol source-of-truth, OOD pathology classification.
- [`dlmastery/autoresearch`](https://github.com/dlmastery/autoresearch)
  — FX-prediction autoresearch loop (the original).
- [`dlmastery/autoresearchtabular`](https://github.com/dlmastery/autoresearchtabular)
  — tabular ML autoresearch (Higgs UCI).

If you change a gate or composite formula here, **also open an issue
on the source repo (`autoresearchimage`) explaining why** — the gates
are inherited and divergence should be deliberate.
