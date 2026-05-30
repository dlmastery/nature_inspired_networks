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
[`IDEA_TABLE.md`](hypotheses/IDEA_TABLE.md) (75 hypotheses across 7 thematic
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
[`EXPERIMENT_LEDGER.md`](experiments/EXPERIMENT_LEDGER.md) row appended after
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

## 7. Sister projects (cross-links, NOT dependencies)

This repository is **self-contained**. The autoresearch protocol below
(Rules 1–28) is fully specified here; nothing in this file requires
reading another repo to implement. Sister repositories are listed only
as related autoresearch projects on different domains — useful for
cross-pollination but never as authoritative sources:

- [`dlmastery/autoresearch`](https://github.com/dlmastery/autoresearch) — FX prediction
- [`dlmastery/autoresearchimage`](https://github.com/dlmastery/autoresearchimage) — earlier image-classification autoresearch project that helped seed the protocol; this repo's rules supersede any divergence.
- [`dlmastery/autoresearchtabular`](https://github.com/dlmastery/autoresearchtabular) — Higgs UCI tabular
- [`dlmastery/autoresearchspy`](https://github.com/dlmastery/autoresearchspy) — SPY ETF
- [`dlmastery/autoresearchindexstock`](https://github.com/dlmastery/autoresearchindexstock) — QQQ

Composite formula `top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms)`
is **defined here** (Rule 2) and SHA-256-fingerprinted as
`d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`.
A new branch / repo is required to change it — no parent-repo dependency.

The 17 reusable skills in `skills/` (catalogued in §11) are also
self-contained and content-agnostic: any future autoresearch project
can pick them up unchanged.

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

## 10. Rules 20–28 — added 2026-05-27 / 2026-05-29 after the audit campaign

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

### Rule 26 — Windows thread-cap safety (crash prevention)
The autoresearch hardware contract (§2) requires `num_workers: 0` for
DataLoader on Windows (spawn-start workers wedge). After a 2026-05-28
machine crash during a multi-hour multi-agent sweep, the same hardware
contract is extended to ALSO set:
```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = 2
$env:MKL_NUM_THREADS = 2
```
in front of every long-running sweep launch. Without the OMP/MKL caps,
PyTorch + numpy + scipy each spin up `os.cpu_count()` threads, oversubscribing
the CPU when 5+ agents + a GPU sweep + an auto-checkpoint loop run
concurrently. The cap leaves enough cores for OS/IDE/Claude Code to
continue and prevents the system-wide hang the May-28 crash exhibited.
Rule 20's auto-checkpoint loop is then sufficient to lose ≤ 1 run on
any future crash (validated empirically on 2026-05-28; 24 of 31 runs
preserved by auto-checkpoint, the remaining 7 resumed cleanly via
`--skip-existing`).

### Rule 27 — Pages-link discipline (no repo-root .md hrefs in HTML)
GitHub Pages publishes ONLY the `docs/` directory. Any link in the
dashboard HTML to a repo-root file (`FINDINGS.md`, `EXPERIMENT_LOG.md`,
`hypotheses/g*/H*.md`, `audits/G*_audit.md`, `PAPER.md`, etc.) using a
relative path like `../FINDINGS.md` or `../../hypotheses/...` resolves
to a Pages-root URL that returns **404**. The 2026-05-29 Playwright
audit found 340+ such broken links across 199 generated HTML files.

The fix codifies the discipline: every link from generated dashboard
HTML to a non-`docs/` repo file MUST be the absolute GitHub-blob URL
`https://github.com/dlmastery/nature_inspired_networks/blob/main/<path>`.
Relative paths are reserved for files INSIDE `dashboard/` (e.g., per-
experiment pages link to siblings via `experiments/<tag>.html` — that
works because those files are mirrored under `docs/dashboard/`).

Inline `fetch('../hypotheses/...')` JavaScript pattern is forbidden for
the same reason; click handlers must `window.open` the GitHub-blob URL
in a new tab rather than try to inline-render a Pages-blocked file.

A Playwright link sweep (extract all `<a href>`, HEAD-test each suspect,
report 404s) is the canonical verification — run after every renderer
change.

### Rule 28 — Screening vs evaluation
A single-config, single-seed (or even 3-seed) sweep number is SCREENING DATA, not evaluation. It identifies candidates worth further investigation. The headline number for an EXTERNAL claim (FINDINGS, PAPER, README, badge) requires:
- (a) per-hypothesis hill-climb (≥20 runs over lr/wd/optimizer/batch, coordinate descent or random search) at the candidate's design-spec scale; AND
- (b) 3-seed at the best config; AND
- (c) the Phase-5 worst-leader-seed > best-baseline-seed cross-dataset gate.
A screening number MAY be reported in interim docs but only with the explicit "screened, not evaluated" qualifier. Negative results have the same standard: a candidate may not be claimed FALSIFIED on screening alone if its design literature predicts effects only manifest at longer horizons or different hyperparameter regimes (H41's β-only requalification is the cautionary tale: −33pp on screening was eps-confound, not the β-falsification predicted by Reddi 2018; β-only at 12-ep is only −1pp, and Reddi's non-convergence prediction needs ≥100 epochs to manifest).
See [`skills/autoresearch-per-hypothesis-hillclimb/SKILL.md`](skills/autoresearch-per-hypothesis-hillclimb/SKILL.md) for the operational protocol.

### Rule 29 — Markdown rendering MUST be Playwright-verified
Any artefact (dashboard HTML, per-experiment page, README, paper)
that embeds markdown sourced from `.md` files (FINDINGS verdict
blocks, sci-critic addenda, hypothesis digests, headline ribbons)
MUST pipe that content through a markdown converter that handles
GFM tables and `>` block-quotes — NOT just inline `**bold**` / `*em*`
/ `` `code` ``. Verify by Playwright: load the file, assert that
`textContent` of the embedded block contains no literal `##`, `**`,
`|---:|`, or `&gt;` strings. The 2026-05-29 dashboard audit
(`audits/REVIEWER_PASS_DASHBOARD.md`) caught the same regression
THREE times in one session — twice the "fix" had been claimed
without Playwright verification.
**Why:** commit `5194814` ("Render FINDINGS/audit/sci-critic as
proper markdown") shipped a half-fix — block-quote tables still
leaked literal `|---|` to the camera on every per-experiment page.
**How to apply:** see
[`skills/autoresearch-typography-and-rendering/`](skills/autoresearch-typography-and-rendering/).

### Rule 30 — Academic typography for research artefacts
Externally-facing artefacts (dashboards, paper, README rendered via
GitHub Pages) use Source Serif 4 (body) + IBM Plex Mono (code).
**Forbidden:** Newsreader (italic-as-emphasis display face), any
"editorial" or "magazine" face, font-style:italic for headings,
any Google Font CDN not pinned via `<link rel="preload">` with a
local-CSS fallback. Per-experiment pages and the aggregate
dashboard MUST share the same stack — typography mismatch across
two surfaces of the same artefact reads as unfinished work.
**Why:** commit `f8a4011` admits the aggregate dashboard self-
described as "Newsreader / IBM Plex Serif" while per-experiment
pages already used Source Serif 4 — visible regression cited as a
CRITICAL fail in `audits/REVIEWER_PASS_DASHBOARD.md`.
**How to apply:** see
[`skills/autoresearch-typography-and-rendering/`](skills/autoresearch-typography-and-rendering/).

### Rule 31 — Repo root limited to ≤ 4 canonical files
The repo root contains ONLY: `README.md`, `CLAUDE.md`, `PAPER.md`,
`LICENSE`. Every other markdown artefact lives in a subdirectory:
- `paper/` — FINDINGS, MANIFESTO, PARADIGM_COMPARISON,
  paper_abstract, NEURIPS_CHECKLIST, LIMITATIONS, ETHICS_STATEMENT,
  STATISTICAL_TESTS, REVIEWER_CHECKLIST, SOTA_COMPARISON,
  SELF_AUDIT_CHECKLIST.
- `docs/` — GitHub Pages root + dashboard mirror.
- `experiments/`, `hypotheses/`, `ideas/`, `audits/`, `skills/`,
  `memory/`, `scripts/`, `src/`, `tests/`, `configs/`.
A cluttered root is the single most off-putting signal to a first-
time reader landing from a paper citation or a tweet.
**Why:** the 2026-05-29 README cold-reader audit found 18+ markdown
files in repo root, drowning the conference front-door signal.
**How to apply:** see
[`skills/autoresearch-doc-organization/`](skills/autoresearch-doc-organization/).

### Rule 32 — README is the conference front-door
`README.md` is the FIRST and OFTEN ONLY artefact an external reader
sees. It MUST contain in order: (1) one-paragraph elevator pitch
naming the contribution, (2) status badges (build, paper, license,
Pages-live), (3) a 4-command quick start that actually runs from a
clean clone, (4) headline FINDINGS + negative-results section with
EQUAL prominence (negatives never buried), (5) methodological notes
(seed protocol, hardware contract, screening-vs-evaluation framing
per Rule 28), (6) a one-screen repo map, (7) BibTeX citation block,
(8) license + acknowledgements. NO marketing language; NO unrebuked
"ACCEPT" self-grades (Rule 36).
**Why:** commit `40b576a` was the third README rewrite of the
session — earlier passes hid the H09 rediscovery framing below
the fold and lacked a Quick Start that actually worked.
**How to apply:** see
[`skills/autoresearch-doc-organization/`](skills/autoresearch-doc-organization/).

### Rule 33 — Dashboard comprehension: small-multiples + "how to read"
Dense charts with 3+ overlaid axes are forbidden when 3 side-by-
side small-multiples convey the same information. Every chart
carries a 1-sentence "what to read" caption directly under it.
Every dashboard surface (aggregate, per-experiment, paper figure
panel) opens with a "How to read this dashboard" orientation block
of EXACTLY 4 bullets: (a) what this page shows, (b) how to
interpret colour coding, (c) what counts as screening vs evaluation
(Rule 28), (d) where to click for drill-down. Multi-hypothesis
tags (combo*/pair*/hybrid*) display ALL participating hypothesis
pills — the leading H-id alone is misleading.
**Why:** `audits/REVIEWER_PASS_DASHBOARD.md` flagged combo2 page's
H09-only pill (combo2 = H09+H48) as MAJOR, and the dense ablation
chart as unreadable at submission resolution.
**How to apply:** see
[`skills/autoresearch-dashboard-comprehension/`](skills/autoresearch-dashboard-comprehension/)
and the augmented
[`skills/autoresearch-dashboard/`](skills/autoresearch-dashboard/).

### Rule 34 — Seed-count + tier label on every visual numeric
Every numeric on every dashboard / paper figure / README badge
carries an `n=X` qualifier and a `SCREENING` / `EVALUATION` chip
(per Rule 28). KN-strip: `+1.34 pp Δ vs baseline (n=3, EVALUATION)`.
Aggregate leaderboard: dedicated `n` and `tier` columns. Headline
ribbon: `Phase-8 winners (n=3 seeds, EVALUATION gate)`. A bare
"+1.34 pp" without the qualifier is a Rule-28 violation surfacing
as a Rule-34 visual gap.
**Why:** `audits/REVIEWER_PASS_DASHBOARD.md` identified 80+ KN-
strips and table cells missing the framing, cited as the "single
biggest 'not yet top-tier' gap."
**How to apply:** see
[`skills/autoresearch-dashboard-comprehension/`](skills/autoresearch-dashboard-comprehension/)
and the augmented
[`skills/autoresearch-per-experiment-page/`](skills/autoresearch-per-experiment-page/).

### Rule 35 — Statistical rigor floor for empirical claims
Any empirical claim using "outside seed noise" / "statistically
significant" / "winner" framing MUST report: (a) paired Wilcoxon
signed-rank (or paired t-test with explicit normality justification);
(b) 95 % bootstrap CI on the pp delta (≥10,000 resamples); (c) a
multiple-comparisons correction across the sweep family — Holm-
Bonferroni for K known comparisons. **n=3 seeds is SCREENING only.**
For α=0.05 Holm-Bonferroni on a 3-hypothesis family the minimum
seed count is **n≥7** (Wilcoxon needs ≥6 non-tied pairs for two-
sided p < 0.05; Holm correction raises the bar). Empirical noise
band (e.g., "±0.5 pp") must be EMPIRICALLY DERIVED from the
project's own multi-seed data per dataset — NEVER a rule-of-thumb.
The empirically-derived band is reported in `paper/STATISTICAL_TESTS.md`.
**Why:** `audits/REVIEWER_PASS_PAPER.md` BLOCKER section B graded
the "min-leader > max-baseline" ordinal gate as a one-sided sign
test with α≈0.125 — well above any defensible threshold.
**How to apply:** see
[`skills/autoresearch-paper-rigor/`](skills/autoresearch-paper-rigor/).

### Rule 36 — Pre-registration discipline; no post-hoc HARKing
The classification of any sweep row as SCREENING vs EVALUATION
(Rule 28) MUST be pre-registered BEFORE the sweep runs — committed
to git with a hash referenced in the paper / FINDINGS entry.
Retroactively reclassifying a row as "screening" after seeing it
lose is HARKing (Hypothesizing After Results are Known) and is a
BLOCKER-level finding. Ordinal-margin (`min(leader)−max(baseline)`)
is a different statistic from Δmean (`mean(leader)−mean(baseline)`)
and both MUST be reported alongside each other for any "lead"
claim. If a hypothesis's pre-registered falsifier specifies a
dataset that isn't in the sweep, the verdict is
`UNTESTED_ON_RIGHT_DATASET` — never NUMEROLOGY / FALSIFIED.
**Why:** `audits/REVIEWER_PASS_PAPER.md` flagged §7.3.1 of PAPER.md
as the paper's most dangerous BLOCKER — retroactive reclassification
of every single-prior negative as "screening, not evaluation."
**How to apply:** see
[`skills/autoresearch-paper-rigor/`](skills/autoresearch-paper-rigor/).

### Rule 37 — No self-grading banners on external-facing artefacts
"ACCEPT" / "WEAK_ACCEPT" / "Reviewer-acceptance" verdicts produced
by same-model-family agents grading their own project's output are
**INTERNAL QA passes, not external review.** Banners on README,
PAPER, dashboard, per-experiment pages MUST explicitly say
"Internal QA pass — independent external review pending." An
external reviewer's WEAK_REJECT (e.g., `audits/REVIEWER_PASS_PAPER.md`)
**overrides** any prior internal ACCEPT — internal verdicts are
downgraded immediately and the downgrade is logged in the artefact.
Auditor-self-grading circularity (implementer + critic + sci-critic
+ fixer agents share a model family) MUST be disclosed in any
paper section reporting audit-derived rates (e.g., the 51 % non-
PASS finding), with a calibration plan against third-party code.
**Why:** `audits/REVIEWER_PASS_PAPER.md:55` flagged the PAPER.md
"Reviewer-acceptance ACCEPT verdict at commit `0343f35`" as
self-referential and misleading.
**How to apply:** see
[`skills/autoresearch-scicritic-team/`](skills/autoresearch-scicritic-team/)
and augmented
[`skills/autoresearch-critic-team/`](skills/autoresearch-critic-team/).

### Rule 38 — Link discipline + first-mention linkification + audit ledger
Every link in any externally-facing artefact (README, paper,
dashboard HTML, per-experiment page) MUST be HEAD-tested via
Playwright before claiming "done." Repo-root `.md` files
referenced from `docs/`-served HTML use absolute GitHub-blob URLs
(Rule 27); relative `../FINDINGS.md` from `docs/` returns 404.
The FIRST mention of any model name (ResNet-20, RegNetX-200MF,
ViT-Tiny), dataset (CIFAR-10/100, Spherical MNIST), technique
(SIREN, AdamW, label smoothing), arXiv ID, hypothesis ID (H09,
H71), or CLAUDE.md rule number in any artefact MUST be a hyperlink
to its canonical reference. Audit files in `audits/` are
APPEND-ONLY (Rule 3 spirit) — new audits never overwrite previous
ones; each pass creates a dated file or a new section.
**Why:** the 2026-05-29 Playwright link sweep found 340+ broken
links (Rule 27 origin); cold-reader audits flagged un-linkified
first mentions of ResNet, AdamW, CIFAR across README/PAPER.
**How to apply:** see
[`skills/autoresearch-link-discipline/`](skills/autoresearch-link-discipline/).

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

**Added 2026-05-29 from the reviewer-pass + cold-reader campaign**
(Rules 29–38, external-review-aware):
- [`autoresearch-typography-and-rendering`](skills/autoresearch-typography-and-rendering/) — markdown rendering with Playwright-verification + Source Serif 4 / IBM Plex Mono academic palette; no editorial display faces.
- [`autoresearch-doc-organization`](skills/autoresearch-doc-organization/) — repo root ≤ 4 canonical files; README front-door pattern; paper/docs/experiments subdirs.
- [`autoresearch-link-discipline`](skills/autoresearch-link-discipline/) — every link Playwright-HEAD-tested; absolute GitHub-blob URLs per Rule 27; first-mention linkification of models/datasets/techniques/arXiv IDs.
- [`autoresearch-dashboard-comprehension`](skills/autoresearch-dashboard-comprehension/) — small-multiples over dense charts; "how to read" orientation block; seed-tier propagation; multi-hypothesis pill display.
- [`autoresearch-paper-rigor`](skills/autoresearch-paper-rigor/) — statistical-rigor floor (paired Wilcoxon + bootstrap CI + Holm-Bonferroni); pre-registration of screening-vs-evaluation; empirical noise band derivation; UNTESTED_ON_RIGHT_DATASET verdict tier.
- [`autoresearch-per-hypothesis-hillclimb`](skills/autoresearch-per-hypothesis-hillclimb/) — 20–25-trial coordinate-descent hill-climb over (lr × wd × batch × seed × optimizer) on the screening winner, per-hypothesis dashboard, 3-seed statistical confirmation before any external claim.

**Added 2026-05-29 PM from the sister-repo parity audit**
(`audits/SKILLS_PARITY_AUDIT.md` — porting load-bearing patterns from
the five `dlmastery/autoresearch*` sister repos that were absent
locally):
- [`autoresearch-data-split-audit`](skills/autoresearch-data-split-audit/) — triple-check leakage / disjointness / protocol-conformance / size-floor / reproducibility-fingerprint audit; runner refuses to launch without a green report. Ported from `autoresearchimage` (Camelyon17 hospital split) + `autoresearchtabular` (Higgs Baldi 2014 frozen split).
- [`autoresearch-winner-archive`](skills/autoresearch-winner-archive/) — fully-portable champion archive (README + config + checkpoint + frozen code + inference script + per-fold results + audit report + reproduction log + optional Colab notebook). Ported from `autoresearch` (FX) + `autoresearchspy` + `autoresearchindexstock`.
- [`autoresearch-explainability-report`](skills/autoresearch-explainability-report/) — 14-section data-scientist-grade audit (executive summary, permutation feature importance, top-N analysis, SHAP-style local, per-fold drift, calibration, uncertainty sanity, prediction distribution, per-sample attribution, risk audit, data-pipeline re-assertion, model-config dump, known limitations, deployment checklist). Ported from `autoresearch` Explainability & Auditability Report rule.
- [`autoresearch-session-resume`](skills/autoresearch-session-resume/) — self-contained crash-recovery checkpoint document (`memory/project_autoresearch_checkpoint.md`) that a fresh session reads on startup to resume without re-reading logs / dashboards. Ported from the Session-Start ritual common to all five sister repos.

**Added 2026-05-30 from the parity-audit follow-up + autoresearchtabular
deep-read** (closing the two NICE-TO-HAVE gaps deferred in
`audits/SKILLS_PARITY_AUDIT.md` §6.2):
- [`autoresearch-shuffle-test`](skills/autoresearch-shuffle-test/) — semantic leakage detection by permuting `y_train`, refitting, and asserting the validation metric collapses to the chance baseline; three modes (hard / within-group / block) covering tabular, grouped, and time-series CV. Catches the bug class the structural data-split audit cannot see (target-encoded features, label-conditional augmentation, off-by-one alignment). Ported from `autoresearch` (FX) §3.5 shuffle audit where a +8.78-Sharpe alignment bug had a green structural audit for three weeks before this test caught it.
- [`autoresearch-data-contract-validator`](skills/autoresearch-data-contract-validator/) — `(x, y)` pairing contract validator: asserts feature shape / dtype / label-set / value-range / pair-count / index-pair invariant match between training and evaluator loaders. Static counterpart to the shuffle test — catches the off-by-one alignment in < 1 second instead of after a full retrain. Refuse-to-launch at runner pre-flight, no `--skip-contract` flag. Ported from the same FX §3.5 retrospective.

*Last updated: 2026-05-30. Rules 1–38 are normative invariants. CLAUDE.md
is self-contained: no parent-repo dependency required to implement the
protocol. The 29 skills in `skills/` are content-agnostic.*
