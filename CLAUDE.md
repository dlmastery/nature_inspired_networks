# CLAUDE.md — normative rules for `nature_inspired_networks`

> Canonical rules for any future Claude / operator working on this repo.
> **Every rule below is strictly enforced.** Failure to follow these
> rules is a defect, not a style choice. The rules consolidate every
> operational directive the user has given across the session. Each
> rule has a section in `memory/` with the original directive,
> rationale, and how-to-apply pattern.

---

## 0. What this project is

A modular autoresearch framework for studying nature-inspired
inductive biases (φ/Fibonacci scaling, Platonic / icosahedral
equivariance, hexagonal lattices, fractal self-similarity, toroidal
closure, Chladni cymatic init, golden-angle modulation) as drop-in
residual / attention blocks in CIFAR-scale image classification and
decoder-only LLMs. Each prior maps to a peer-reviewed GDL/TDL paper.
The mystical motivation is acknowledged in prose only; artifact names
are academic / neutral.

The single source of truth for the design space is
[`IDEA_TABLE.md`](IDEA_TABLE.md) (75 hypotheses across 7 thematic
groups). Per-hypothesis design documents live under
`hypotheses/g<N>_<group>/H<NN>_<short>.md`. Each implemented
hypothesis has a self-contained sub-project under `ideas/<NN>_<short>/`.

---

## 1. The 18 strict rules

### Rule 1 — One config change per experiment
Either a single flag in `flags:` flips, one `channel_mode` switches,
or one optimizer/init differs. No silent compounding.

### Rule 2 — Composite formula is SHA-256 fingerprinted
The string in `src/nature_inspired_networks/eval.py:COMPOSITE_FORMULA`
is hashed at import time. Editing it makes the next run refuse to
launch with `CompositeFingerprintError`. New formulas require a new
branch / repo.

### Rule 3 — `experiment_log.jsonl` is append-only
Never edit a past row. Corrections add a `_v2` row with a journal
entry explaining why.

### Rule 4 — Citation Rigor
Every reasoning entry needs the format
`Author1, Author2, … YEAR VENUE 'Title' (arXiv:XXXX.XXXXX) — relevance`.
The validator in `src/nature_inspired_networks/reasoning.py` rejects
parenthetical-only tags like `(He2016)`.

### Rule 5 — Reasoning Blob Completeness
Word-count floors per field: `diagnosis ≥ 60`, `citation ≥ 40 single
/ 80 multi`, `hypothesis ≥ 50`, `prediction ≥ 25`, `verdict ≥ 30`,
`learning ≥ 40`. Padding to hit the floor is a defect — the floor
exists to force substantive content.

### Rule 6 — No silent randomness
`set_seed(seed)` is called at the top of every run.
`cudnn.benchmark = True` is intentional. Headline numbers are
seed-median composite over `--seeds 0 1 2`.

### Rule 7 — No `--bypass` flag
If a gate refuses, fix the entry, do not disable the gate. There is
no emergency-bypass mode.

### Rule 8 — Per-experiment archive sub-directory is mandatory
Even single smoke runs land in
`ideas/<idea>/experiments/exp<NNN>_<short>/`. The archive
sub-directory carries `README.md` (very detailed design + result +
verdict), `config.yaml`, `reasoning.json`, `run_seed<N>/{metrics.json,
history.json, best.pt}`, and a local `dashboard/`.

### Rule 9 — Every experiment archive has a very detailed README.md
The README stands alone — anyone reading just that sub-directory must
be able to reproduce the experiment from cold. The README is the
deliverable; the weights are secondary.

### Rule 10 — Skills under `skills/` are content-agnostic
A skill that mentions sacred-geometry-specifics is leaking domain
detail and must be rewritten. The skills exist so that ANY future
autoresearch project (tabular, medical, FX, …) can pick them up
unchanged.

### Rule 11 — Periodic GitHub checkpoint is mandatory
Commit + push to `dlmastery/nature_inspired_networks` on EVERY
milestone: file edit, test green, ledger update, run-folder created,
dashboard refreshed. Default cadence ≤ 15 min during active work;
BEFORE AND AFTER every background training task. Many small commits
beat one big commit. The checkpoint is the deliverable — a power
outage must never lose progress.
See [`memory/feedback_checkpoint_discipline.md`](../.claude/projects/C--Users-evija-sacgeometry/memory/feedback_checkpoint_discipline.md)
for the trigger table.

### Rule 12 — Test discipline
Every new module / class / function ships with a unit test in
`tests/test_<module>.py` (or `ideas/<NN>/tests.py`) exercising shape,
every Boolean-flag combination, and the bug class it was added to
fix. Tests must pass before any training-loop background task is
launched. End every test file with `"All N tests passed."` or fail
loudly.
See [`memory/feedback_test_discipline.md`](../.claude/projects/C--Users-evija-sacgeometry/memory/feedback_test_discipline.md).

### Rule 13 — SOTA smoke first
Every experiment workflow on a given dataset runs a known-good
baseline at the **SOTA recipe** as a pre-flight smoke BEFORE any
nature-inspired variant. For CIFAR-10 the canonical smoke is
`configs/cifar10_sota_smoke.yaml` (ResNet-20 + AdamW + cosine + label
smoothing + RandomCrop+HFlip; expected ≥ 80 % top-1 at 12 ep,
≥ 89 % at 30 ep, ≥ 91 % at 164 ep). If the smoke falls below the
expected band, STOP and diagnose the environment — do not run any
variant. `scripts/run_sweep.py` enforces this by gating subsequent
rows on the baseline result.
See [`memory/feedback_sota_smoke_first.md`](../.claude/projects/C--Users-evija-sacgeometry/memory/feedback_sota_smoke_first.md).

### Rule 14 — Modular & reusable architecture
Shared primitives live in `src/nature_inspired_networks/` (single
import surface). Each `ideas/<NN>_<short>/implementation.py` imports
from there — it does NOT duplicate code. The idea sub-project glue is
a thin composition wrapper. Hypothesis documentation lives under
`hypotheses/g<N>_<group>/H<NN>_*.md` (75 docs, 7 thematic groups).

### Rule 15 — Hierarchical agent teams with SMEs
When the workload is parallelizable (writing many design docs,
building many idea sub-projects, conducting research surveys), use
parallel `Agent` calls with hierarchical SME roles. Each agent gets
a disjoint scope, references the shared `_TEMPLATE`, and reports back
with file byte counts + recommended refinements. Experiments
themselves run serially on the single 4090; **only docs / code /
research run in parallel.**

### Rule 16 — Neutral / academic naming
Artifact names are neutral; mystical inspiration is acknowledged only
in prose. The renames from the original `sacgeometry` / `SacredGeo*`
project to `nature_inspired_networks` / `NaturePrior*` are normative,
not cosmetic.
See [`memory/feedback_naming_preference.md`](../.claude/projects/C--Users-evija-sacgeometry/memory/feedback_naming_preference.md).

### Rule 17 — Source-document audit must be chunk-by-chunk
When the user provides multiple source documents (PDFs, transcripts),
they are read in ≤ 250-line chunks with an
[`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md) row appended after
each chunk. No source document is treated as read-once-and-summarised
from memory; the chunked audit is the deliverable.

### Rule 18 — Documentation must be committee-grade
Every hypothesis file (`hypotheses/g<N>/H<NN>_*.md`), every experiment
archive README, every paradigm-comparison entry must be detailed
enough to satisfy a hostile NeurIPS / ICML reviewer. Sections per
template are MANDATORY: motivation (≥ 100w), formal hypothesis (≥ 50w,
"mechanism" or "because"), numeric falsifier, multi-paper citations,
mechanism (CNN-track AND LLM-track), predicted Δ table, 3-part
experimental protocol, cross-references, ≥ 4 Committee Q&A,
verification checklist, status journal. No padding; substantive depth.

### Rule 19 — Phased CIFAR-10 → CIFAR-100 progression
Strict phase order. No hypothesis jumps phases.

- **Phase 0** — Unit tests. `ideas/<NN>/tests.py` green
  (≥ 4 assertions + at least one regression test).
- **Phase 1** — CIFAR-10 SOTA-smoke pre-flight
  (`configs/cifar10_sota_smoke.yaml`). ResNet-20 baseline must hit
  ≥ 80 % top-1 at 12 ep. Failure → STOP, fix env.
- **Phase 2** — CIFAR-10 hypothesis smoke for **ALL** hypotheses.
  12-epoch ablation row per hypothesis. Commit + push after each.
- **Phase 3** — Check-in dashboard + RESULTS + FINDINGS refresh.
  Identify top-K performers by composite. Commit + push.
- **Phase 4** — CIFAR-100 heavy hitters. **Only** the top-K from
  Phase 3 graduate. ≥ 30 epochs (quality) or 100+ (convergence).
  Each run carries its own `ideas/<NN>/experiments/expNNN_*_cifar100/`
  archive.
- **Phase 5** — 3-seed re-run on Phase-4 winners. Add error bars
  before any external claim.

The cheap broad scan precedes the expensive deep dive. CIFAR-100 GPU
budget is too scarce to spend on hypotheses whose 12-epoch CIFAR-10
number would have already disproved them.
See [`memory/feedback_phased_workflow.md`](../.claude/projects/C--Users-evija-sacgeometry/memory/feedback_phased_workflow.md).

---

## 2. Hardware contract

- Target: **1× RTX 4090 Laptop, 16 GB VRAM, Windows 11**.
- bf16 AMP + cosine LR + label smoothing + RandAugment.
- Default batch 256.
- `num_workers: 0` on Windows because spawn-start workers wedge.
- Python 3.13 corp-cert SSL workaround: `curl.exe -kL` for CIFAR;
  torchvision verifies MD5.
- Background-task launches **always** preceded by `git push`
  (per Rule 11).

---

## 3. Repository layout (canonical)

```
nature_inspired_networks/
├── README.md               operator quick-start
├── MINDMAP.md              one-page link map of every artifact
├── MANIFESTO.md            research argument (committee-grade)
├── CLAUDE.md               this file
├── ARCHITECTURE.md         module + shape tables
├── AUTORESEARCH_PROCESS.md 7-step ritual + gates
├── IDEA_TABLE.md           75-hypothesis status table
├── EXPERIMENT_LOG.md       master long-list (Tiers 0-6)
├── EXPERIMENT_LEDGER.md    chunk-by-chunk audit of source docs
├── PARADIGM_COMPARISON.md  8-chunk Liquid/JEPA/KAN/Transformer/GNN
├── NATURE_INSPIRED_NETWORKS.md   state-of-the-field (May 2026)
├── FINDINGS.md             campaign verdicts (incl. negatives)
├── RESULTS.md              auto-generated per-run narratives
├── SOTA_COMPARISON.md      honest map to the literature
├── PAPER.md / paper_abstract.md
├── SETUP.md / MEDIUM.md
├── sota_catalog.yaml
├── pyproject.toml
├── hypotheses/             75 docs in 7 thematic group subdirs
│   ├── _TEMPLATE.md
│   ├── INDEX.md
│   ├── g1_scaling_growth/
│   ├── g2_layer_channel_neuron/
│   ├── g3_topologies_graphs/
│   ├── g4_kernels_attention_filters/
│   ├── g5_optimization_init_reg_nas/
│   ├── g6_topological_bridging/
│   └── g7_cross_paradigm_hybrids/
├── ideas/                  modular sub-projects (1 per impl. hypothesis)
│   ├── _TEMPLATE/
│   └── NN_<short>/
│       ├── README.md
│       ├── IDEA.md
│       ├── implementation.py     ← imports from nature_inspired_networks
│       ├── tests.py
│       ├── AUDIT.md
│       ├── IMPROVEMENTS.md
│       ├── VERIFY.md
│       ├── experiment.py
│       ├── configs/<config>.yaml
│       ├── experiments/expNNN_<short>/
│       │   ├── README.md                ← very detailed
│       │   ├── config.yaml
│       │   ├── reasoning.json
│       │   ├── run_seed0/{metrics, history, best.pt}
│       │   └── dashboard/
│       ├── results.md
│       └── dashboard/
├── src/nature_inspired_networks/   shared infrastructure
├── scripts/                run_sweep / build_dashboard / build_report / compute_topology
├── skills/                 11 content-agnostic auto-research skills
├── tests/                  29 core + 68 idea-local unit tests
├── configs/                shared YAML configs (smoke / quick / sota_smoke)
├── experiments/            legacy CIFAR-10 archive
├── dashboard/              latest aggregated dashboard
├── docs/                   GitHub Pages root
└── memory/                 project checkpoint markdown
```

---

## 4. Adding an experiment — checklist (Rule 13 + 8 + 11)

1. **Pre-flight SOTA smoke (Rule 13).** Run
   `configs/cifar10_sota_smoke.yaml` (or the dataset-specific SOTA
   smoke) and verify top-1 ≥ expected band. If not, STOP.
2. Pick or create `ideas/<NN_idea>/` from `ideas/_TEMPLATE/`.
3. Create `ideas/<NN_idea>/experiments/exp<NNN>_<short>/` archive dir.
4. Write `experiments/exp<NNN>_*/README.md`: hypothesis, mechanism,
   prediction, dataset, config delta, expected verdict.
5. Author `experiments/exp<NNN>_*/reasoning.json` with the 4 pre-run
   fields (`diagnosis`, `citations`, `hypothesis`, `prediction`).
   Validator must accept (Rules 4 + 5).
6. Run unit tests (Rule 12); confirm green.
7. `git commit + push` BEFORE launch (Rule 11).
8. Launch:
   ```powershell
   .\.venv\Scripts\python -m nature_inspired_networks.runner `
     --config ideas\<NN>\configs\<config>.yaml `
     --tag exp<NNN>_<short> --seed 0 `
     --root ideas\<NN>\experiments\exp<NNN>_<short>\run
   ```
9. Append post-run `verdict` + `learning` to `reasoning.json`.
10. Regenerate dashboards (`scripts/build_dashboard.py` +
    `build_report.py`).
11. `git commit + push` AFTER completion (Rule 11).

---

## 5. When the runner refuses

| symptom | cause | fix |
|---|---|---|
| `ValueError: Reasoning entry rejected` | word-count / citation format (Rule 4/5) | rewrite the failing field |
| `CompositeFingerprintError` | composite formula edited (Rule 2) | revert / new branch |
| `MD5 mismatch on CIFAR tarball` | corp proxy injection | re-download with `curl.exe -kL` |
| `CUDA out of memory` | over 16 GB | drop batch in a new config (do NOT edit existing config — Rule 1) |
| SOTA smoke below expected band | env drift / torch version / corrupted data | STOP, diagnose, do not run variants (Rule 13) |

---

## 6. What may never go in this repo

- Real-name PII or PHI.
- Pre-trained ImageNet weights re-uploaded under our license (link
  upstream).
- Closed datasets requiring registration; load at runtime only.
- Secrets, `.env` files, API keys.

---

## 7. Sister projects

- [`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage) — protocol source-of-truth
- [`dlmastery/autoresearch`](https://github.com/dlmastery/autoresearch) — FX prediction
- [`dlmastery/autoresearchtabular`](https://github.com/dlmastery/autoresearchtabular) — Higgs UCI
- [`dlmastery/autoresearchspy`](https://github.com/dlmastery/autoresearchspy) — SPY ETF
- [`dlmastery/autoresearchindexstock`](https://github.com/dlmastery/autoresearchindexstock) — QQQ

If you change a gate or composite formula here, also open an issue on
`autoresearchimage` explaining why.

---

## 8. Operator quick-reference

```powershell
# SOTA smoke (≤ 2 min — Rule 13 pre-flight)
.\.venv\Scripts\python -m nature_inspired_networks.runner `
  --config configs\cifar10_sota_smoke.yaml --tag smoke --seed 0

# Curated 13-row ablation (~90 min)
.\.venv\Scripts\python scripts\run_sweep.py `
  --config configs\cifar10_quick.yaml --seeds 0 --skip-existing

# Trained-feature Betti + dashboard + report
.\.venv\Scripts\python scripts\compute_topology.py --seeds 0
.\.venv\Scripts\python scripts\build_dashboard.py
.\.venv\Scripts\python scripts\build_report.py

# 3-seed re-sweep for error bars (~5 hr)
.\.venv\Scripts\python scripts\run_sweep.py `
  --config configs\cifar10_quick.yaml --seeds 0 1 2 --skip-existing
```

---

## 9. The protocol is the deliverable

The model weights are secondary. The artifact pack — manifesto +
84 hypothesis docs + 7-step protocol entries + dashboards + per-run
archives + critic audits + sci-critic addenda + committed history — is
what survives. Train fewer, document more. Negative results are
first-class citizens (see `FINDINGS.md`).

Every rule above is enforced. There are no exceptions.

---

## 10. Rules 20–25 — added 2026-05-27 after the audit campaign

Rules 1–19 were authored before the parallel critic / sci-critic / fixer
campaign exposed how easy it is for an implementer agent to ship code
that compiles, passes shape-only tests, and silently doesn't implement
the hypothesis. Rules 20–25 close those loopholes.

### Rule 20 — Auto-checkpoint loop alongside long background tasks
ANY background task expected to run > 15 min (GPU sweep, parallel agent
team, multi-agent dispatch) MUST be paired with a background git
auto-commit loop that pushes new artifacts every ~10 min, with
retry-wrapped scoped commits. Stop the loop when the task completes.
A power outage during a multi-hour sweep must lose at most ONE run.
See [`skills/autoresearch-auto-checkpoint-loop/`](skills/autoresearch-auto-checkpoint-loop/).

### Rule 21 — Post-fix re-run discipline
After any Fixer agent patches a hypothesis module (audits identified
the bug), the affected sweep tag MUST be re-smoked at seed 0 before
the fix is considered complete. If the patched hypothesis was a
Phase-4 / Phase-5 graduate (e.g., `phi_budget`), the CIFAR-100 3-seed
re-run is also mandatory before any external claim is updated. The
pre-fix vs post-fix number is recorded explicitly in `FINDINGS.md`.
See [`skills/autoresearch-fixer-campaign/`](skills/autoresearch-fixer-campaign/).

### Rule 22 — Dual-track audit before any external claim
A hypothesis used in an external claim (FINDINGS headline, paper
abstract, README badge) MUST pass BOTH:
- (a) the implementation-critic audit — no MAJOR / BROKEN verdict in
  `audits/G<X>_audit.md`, AND
- (b) the sci-critic addendum — verdict ≠ NUMEROLOGY / UNFALSIFIABLE
  in the design doc's "Addendum: Research-Scientist Critique" section.
A "winner" with a DERIVATIVE+TESTABLE sci-verdict is permitted; a
NUMEROLOGY-verdict winner is NOT. See
[`skills/autoresearch-critic-team/`](skills/autoresearch-critic-team/) and
[`skills/autoresearch-scicritic-team/`](skills/autoresearch-scicritic-team/).

### Rule 23 — Compound design uses orthogonal axes
Multi-prior stacks (combo / hybrid hypotheses) stack ONLY priors that
touch different layers of the training stack (arch / channel /
momentum / regulariser / weight-decay / LR / activation / ensemble /
pruning / inference). Stacking more than two priors on the same conv-
block forward path is forbidden — `sg_full_fib` (−11.54 pp on
CIFAR-10) is the cautionary tale. The canonical additive test is the
"combo ladder": N = 2, 3, 4, … rows where each next row adds exactly
ONE new orthogonal prior. See
[`skills/autoresearch-combo-ladder/`](skills/autoresearch-combo-ladder/).

### Rule 24 — Dashboard discipline
- Aggregate dashboard `dashboard/dashboard.html` is sectioned by
  hypothesis group (Baseline + G1..G8 + Uncategorised) with one-line
  group descriptions and per-group sortable mini-tables; NOT one big
  cluttered table.
- Every leaderboard row links to an INDEPENDENT per-experiment page
  at `dashboard/experiments/<dataset>__<tag>_seed<N>.html` containing:
  hypothesis-doc digest, FINDINGS verdict, reasoning blob (if any),
  config dump, metrics, composite breakdown, per-epoch SVG training
  curves, cross-references to other seeds + the same tag on the other
  dataset, run footer with composite fingerprint.
- Per-experiment pages are byte-identically mirrored to
  `docs/dashboard/experiments/` for the GitHub Pages live demo at
  `https://dlmastery.github.io/nature_inspired_networks/`.
- Row click → navigate to per-experiment page. NO modals.
See [`skills/autoresearch-per-experiment-page/`](skills/autoresearch-per-experiment-page/).

### Rule 25 — Q&A-test correspondence
Every test name listed in a design doc's "Verification checklist" /
"Committee Q&A" block MUST exist in `tests/`. The G3 audit found 3
hypotheses where promised test names (`test_hex_phi_radial_factor`,
`test_phi_scaling`, `test_h08_function_preserving_growth`) were
mentioned in the design doc Q&A but were **absent** from the actual
test suite — the code never delivered what the doc promised. Treat
the Q&A as a binding contract: every Q&A test name not in `tests/`
is a MAJOR audit finding.

### Cumulative checkpoint cadence (reinforcement of Rule 11)
The auto-checkpoint loop discipline is **mandatory** during any
multi-hour campaign. "I'll commit at the end of the turn" is a Rule-11
violation. "I'll squash these later" is a Rule-11 violation. Many
small retry-wrapped commits beat one big commit. A power outage during
the project's most expensive sweep — the H08 fixer landed mid-Phase-6
combo-ladder run — must NEVER lose more than a single training run.

---

## 11. Reusable skills

The `skills/` directory hosts content-agnostic auto-research skills
(Rule 10). Any future autoresearch project (tabular, medical, FX, …)
can pick them up unchanged. The current catalogue:

**Original 10 skills** (pre-audit):
`autoresearch-ablation-sweep` · `autoresearch-checkpoint` ·
`autoresearch-dashboard` · `autoresearch-dataset-loader` ·
`autoresearch-experiment` · `autoresearch-experiment-archive` ·
`autoresearch-idea-scaffold` · `autoresearch-modular-block` ·
`autoresearch-reasoning-entry` · `autoresearch-topology-metrics`.

**Added 2026-05-27 from this campaign** (audit-aware):
- [`autoresearch-multi-agent-dispatch`](skills/autoresearch-multi-agent-dispatch/) — parallel agents with disjoint file scopes + retry-wrapped commits + index.lock handling.
- [`autoresearch-critic-team`](skills/autoresearch-critic-team/) — implementation audit by hypothesis group, output `audits/G<X>_audit.md` with PASS / MINOR / MAJOR / BROKEN verdicts.
- [`autoresearch-scicritic-team`](skills/autoresearch-scicritic-team/) — research-scientist critique addenda appended directly into design docs, verdict tier NOVEL / DERIVATIVE / NUMEROLOGY / UNFALSIFIABLE / FALSIFIED.
- [`autoresearch-fixer-campaign`](skills/autoresearch-fixer-campaign/) — patch code + add mechanism-verifying tests + re-run affected sweep rows + pre-fix vs post-fix table.
- [`autoresearch-combo-ladder`](skills/autoresearch-combo-ladder/) — orthogonal-axis additive 2→N-prior stacking on a verified-winner base.
- [`autoresearch-per-experiment-page`](skills/autoresearch-per-experiment-page/) — independent comprehensive dashboard page per run, group-sectioned aggregate, GitHub Pages mirror.
- [`autoresearch-auto-checkpoint-loop`](skills/autoresearch-auto-checkpoint-loop/) — background git auto-commit loop for crash safety alongside long-running sweeps and agent teams.

*Last updated: 2026-05-27. Rules 1–25 are normative invariants.*
