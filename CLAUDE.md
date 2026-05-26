# CLAUDE.md — normative rules for `nature_inspired_networks`

> Canonical rules for any future Claude / operator working on this repo.
> Captures **every operational directive the user has given across the
> design-and-build session**. README, IDEA_TABLE, ARCHITECTURE, and
> AUTORESEARCH_PROCESS summarise this file.

---

## 0. What this project is

A modular auto-research framework for studying nature-inspired
inductive biases (φ-scaling, Fibonacci channels, hexagonal lattices,
Platonic equivariance, fractal recursion, toroidal closure, cymatic
init, golden-angle modulation, Betti loss, etc.) as drop-in residual /
attention blocks in image classification and graph learning.

The framing is engineering: each "sacred" / "nature" prior maps to a
peer-reviewed Geometric / Topological Deep Learning paper. The mystical
inspiration is acknowledged in the source PDF but **never appears in
artifact names**.

---

## 1. Session-derived directives (the full list)

In order received from the user during the design conversation:

1. **"Read the PDF + the autoresearch skills + write code + ablation/
   feasibility on popular benchmarks + RTX 4090 + detailed dashboards
   like autoresearch + principled."**
   → driven by `C:\Users\evija\Downloads\sacred geometry and neural
   networks.pdf` and the `dlmastery/autoresearchimage` template.
2. **"Checkpoint progress in a public dlmastery repo and write detailed
   docs in autoresearch style."**
   → repo: `dlmastery/nature_inspired_networks` (Pages on `/docs`).
3. **"Each idea is an independent task with its own experiment strategy;
   then a mix-of-all-ideas; each implementation audited, critiqued,
   improved, verified for correctness; each archived in its own
   directory; modular to mix-and-match."**
   → `ideas/NNN_<name>/` taxonomy with one self-contained sub-project
   per hypothesis; `ideas/99_mix_all/` for the composed model.
4. **"Verify you went through all 4 source files chunk by chunk and
   build an incremental design-space table."**
   → `IDEA_TABLE.md`: 60 hypotheses in 6 thematic groups.
5. **"Use neutral / academic naming instead of `sacred*`."**
   → project renamed `sacgeometry → nature_inspired_networks`; classes
   `SacredGeo* → NaturePrior*`. Local working dir
   `C:\Users\evija\sacgeometry` is the legacy path; GitHub repo is
   `dlmastery/nature_inspired_networks`. **Operator may rename the
   local dir off-session** with `Move-Item sacgeometry
   nature_inspired_networks`.
6. **"Nice taxonomy directory. Every experiment's artifacts (including
   very very detailed design README) archived in a separate
   sub-directory on GitHub. All code modular and reusable by anyone."**
   → `ideas/<idea>/experiments/expNNN_<short-name>/` is the archive
   unit. Each archive sub-directory contains: README.md (very
   detailed design + hypothesis + prediction + actual + verdict),
   `config.yaml`, `reasoning.json`, per-seed `run_seed<N>/` with
   metrics/history/best.pt, and a local `dashboard/`.
7. **"Update CLAUDE.md with all the things asked + create a `skills/`
   directory with modular skills (agnostic to nature stuff) that can
   reproduce these kinds of experiments."**
   → this file + `skills/SKILL_NAME/SKILL.md` set. Skills must be
   *content-agnostic* — they reference the autoresearch protocol, not
   the specific nature-inspired hypotheses.

---

## 2. Always-true assertions (the gates)

The following are **non-negotiable** unless the user explicitly waives:

1. **One config change per experiment.** Either a single flag flips,
   one channel mode changes, or one optimizer/init differs. No silent
   compounding.
2. **Composite formula is SHA-256 fingerprinted.** Edit the string and
   the next run refuses to launch with `CompositeFingerprintError`.
   New formulas require a new repo / branch.
3. **`experiment_log.jsonl` is append-only.** Never edit a past row.
   Corrections append a new row with the same tag plus `_v2` and a
   journal entry explaining why.
4. **Citation Rigor.** Every reasoning entry needs an `Author1, Author2,
   …, YEAR VENUE 'Title' (arXiv:XXXX.XXXXX) — relevance` line. The
   validator in `src/nature_inspired_networks/reasoning.py` rejects
   parenthetical-only tags like `(He2016)`.
5. **Reasoning Blob Completeness.** Word-count floors per field:
   `diagnosis ≥ 60`, `citation ≥ 40 (single) / 80 (multi)`,
   `hypothesis ≥ 50`, `prediction ≥ 25`, `verdict ≥ 30`,
   `learning ≥ 40`.
6. **No silent randomness.** `set_seed(seed)` is called at the top of
   every run; `cudnn.benchmark = True` is intentional. Headline
   numbers are seed-median composite over `--seeds 0 1 2`.
7. **No `--bypass` flag.** If a gate refuses, fix the entry, do not
   disable the gate.
8. **Per-experiment archive sub-directory is mandatory.** Even single
   smoke runs land in `ideas/<idea>/experiments/expNNN_<name>/`.
9. **Every experiment archive has a very detailed README.md** that
   stands alone — anyone reading just that directory should be able to
   reproduce the experiment.
10. **Skills under `skills/` are content-agnostic.** A skill that
    mentions sacred-geometry-specifics is leaking domain detail and
    must be rewritten.
11. **Periodic GitHub checkpoint is mandatory.** Commit + push to
    `dlmastery/nature_inspired_networks` on **every** milestone:
    code edit, test green, ledger update, run-folder created, dashboard
    refreshed. Default cadence ≤ 15 min during active work; before AND
    after every background training task. Many small commits beat one
    big commit. The checkpoint is the deliverable — a power outage
    must never lose progress. (See
    `memory/feedback_checkpoint_discipline.md` for the trigger table.)
12. **Test discipline.** Every new module/class/function ships with a
    unit test in `tests/test_<module>.py` exercising shape, branches,
    and the bug class it was added to fix. Tests must pass before
    any training-loop background task is launched. (See
    `memory/feedback_test_discipline.md`.)

---

## 3. Repository layout (canonical)

```
nature_inspired_networks/
├── README.md                       — master entry point
├── IDEA_TABLE.md                   — 60-hypothesis design-space table
├── ARCHITECTURE.md                 — module + shape tables
├── AUTORESEARCH_PROCESS.md         — 7-step protocol ritual
├── CLAUDE.md                       — this file
├── PAPER.md                        — draft paper (auto-fills from runs)
├── FINDINGS.md                     — campaign verdict
├── RESULTS.md                      — auto-generated per-experiment narrative
├── SOTA_COMPARISON.md              — honest literature comparison
├── SETUP.md                        — bring-up steps
├── MEDIUM.md                       — blog-style narrative
├── paper_abstract.md
├── sota_catalog.yaml               — prior-art (single source of truth)
├── pyproject.toml
├── src/nature_inspired_networks/   — shared infra (training, eval,
│                                     dashboard, gates, topology)
├── ideas/                          — TAXONOMY: 60 hypothesis sub-projects
│   ├── _TEMPLATE/                  — copy-and-fill scaffold
│   │   ├── README.md               — idea statement
│   │   ├── IDEA.md                 — formal claim + falsifier
│   │   ├── implementation.py
│   │   ├── tests.py
│   │   ├── AUDIT.md
│   │   ├── IMPROVEMENTS.md
│   │   ├── VERIFY.md
│   │   ├── experiment.py           — idea-specific experiment driver
│   │   ├── configs/
│   │   ├── experiments/            — PER-EXPERIMENT ARCHIVES
│   │   │   └── exp001_<short>/
│   │   │       ├── README.md       — very detailed
│   │   │       ├── config.yaml
│   │   │       ├── reasoning.json
│   │   │       ├── run_seed0/{metrics.json,history.json,best.pt}
│   │   │       ├── run_seed1/…
│   │   │       └── dashboard/
│   │   ├── results.md
│   │   └── dashboard/
│   ├── 01_phi_compound_scaling/    ← H01
│   ├── 02_fib_depth_progression/   ← H02
│   ├── … (H03–H60)
│   └── 99_mix_all/                 ← composed hybrid
├── skills/                         — REUSABLE / CONTENT-AGNOSTIC
│   ├── autoresearch-experiment/
│   ├── autoresearch-ablation-sweep/
│   ├── autoresearch-dashboard/
│   ├── autoresearch-reasoning-entry/
│   ├── autoresearch-modular-block/
│   ├── autoresearch-dataset-loader/
│   ├── autoresearch-topology-metrics/
│   ├── autoresearch-experiment-archive/
│   └── autoresearch-idea-scaffold/
├── configs/                        — shared config templates
├── scripts/                        — top-level runners
├── experiments/                    — LEGACY 11-run sweep (kept for
│                                     historical comparison)
├── dashboard/                      — latest aggregated dashboard
├── docs/                           — GitHub Pages root
└── memory/                         — checkpoint markdown
```

---

## 4. Hardware contract

- Target: **1× RTX 4090 Laptop, 16 GB VRAM, Windows 11**.
- Default batch 256 with bf16 AMP. If you change this, add a new
  config file.
- `num_workers: 0` on Windows because spawn-start workers wedge.
- Python 3.13 corp-cert SSL: download CIFAR with `curl.exe -kL`;
  torchvision's MD5 still verifies content.

---

## 5. Adding an experiment — checklist

For ANY new experiment, regardless of idea:

1. Pick or create `ideas/<NN_idea>/` from `ideas/_TEMPLATE/`.
2. Create `ideas/<NN_idea>/experiments/expNNN_<short_name>/` archive
   dir.
3. Write `experiments/expNNN_*/README.md`: hypothesis, mechanism,
   prediction, dataset, config delta, expected verdict.
4. Author `experiments/expNNN_*/reasoning.json` with the 4 pre-run
   fields (`diagnosis`, `citations`, `hypothesis`, `prediction`);
   validator must accept.
5. Write or copy `experiments/expNNN_*/config.yaml`.
6. Run:
   ```powershell
   .\.venv\Scripts\python -m nature_inspired_networks.runner `
     --config ideas\<NN>\experiments\expNNN_<short>\config.yaml `
     --tag expNNN_<short> --seed 0 `
     --root ideas\<NN>\experiments\expNNN_<short>\run
   ```
7. Append post-run `verdict` + `learning` to `reasoning.json`.
8. Regenerate dashboards: `scripts/build_dashboard.py` walks all
   `ideas/**/experiments/**/run_seed*/metrics.json`.

---

## 6. When the runner refuses

| symptom | cause | fix |
|---|---|---|
| `ValueError: Reasoning entry rejected` | word-count / citation format | rewrite the failing field |
| `CompositeFingerprintError` | someone edited composite formula | revert / new branch |
| `MD5 mismatch on CIFAR tarball` | corp proxy injection | re-download w/ `curl.exe -kL` |
| `CUDA out of memory` | over 16 GB | drop batch in a new config file |

---

## 7. What may never go in this repo

- Real-name PII or PHI.
- Pre-trained ImageNet weights re-uploaded under our license (link upstream).
- Closed datasets requiring registration; load at runtime only.

---

## 8. Sister projects

- [`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage)
  — protocol source-of-truth, OOD pathology.
- [`dlmastery/autoresearch`](https://github.com/dlmastery/autoresearch)
  — FX-prediction (the original).
- [`dlmastery/autoresearchtabular`](https://github.com/dlmastery/autoresearchtabular)
  — tabular ML autoresearch (Higgs UCI).

If you change a gate or composite formula here, also open an issue on
`autoresearchimage` explaining why.

---

## 9. Operator quick-reference

```powershell
# Smoke test (≤ 2 min)
.\.venv\Scripts\python -m nature_inspired_networks.runner `
  --config configs\cifar10_smoke.yaml --tag smoke --seed 0

# Curated 11-row ablation (~60 min)
.\.venv\Scripts\python scripts\run_sweep.py `
  --config configs\cifar10_quick.yaml --seeds 0 --skip-existing

# Build full dashboard + RESULTS.md
.\.venv\Scripts\python scripts\compute_topology.py --seeds 0
.\.venv\Scripts\python scripts\build_dashboard.py
.\.venv\Scripts\python scripts\build_report.py
```

The protocol is the deliverable. The model weights are secondary.
