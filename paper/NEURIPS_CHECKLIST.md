# NeurIPS Paper Checklist — `nature_inspired_networks`

> Filled-in NeurIPS-style Paper Checklist for the
> `nature_inspired_networks` submission. Every answer is **[Yes / No /
> N/A]** followed by a 1-2 sentence justification with a file path or
> commit reference. The checklist mirrors the canonical NeurIPS Paper
> Checklist (Conferences/2023/PaperChecklist and subsequent years);
> any minor question-set drift between conference years is resolved in
> favour of the broader, more inclusive question.
>
> **Submission stance:** the campaign is a *methodology + screening*
> contribution. The three Phase-5 graduates carry seed-median + error
> bars; the bulk of the 84-hypothesis design space is screening data
> by construction (see [`LIMITATIONS.md`](../paper/LIMITATIONS.md)).
>
> **Date filled:** 2026-05-29. **Author:** dlmastery
> (`eranti@gmail.com`), anonymised for blind review.

---

## 1. Claims & contributions

### 1.1 — Does the abstract and introduction accurately reflect the paper's contributions and scope?
**[Yes]** The abstract in [`paper_abstract.md`](../paper/paper_abstract.md) and
introduction in [`PAPER.md`](PAPER.md) frame the work as a
*screening-grade, audit-traced* empirical study of nature-inspired
inductive biases; both call out the three Phase-5 graduates and the
three first-class negatives without overclaiming.

### 1.2 — Are the contributions clearly enumerated?
**[Yes]** [`PAPER.md`](PAPER.md) §1.3 enumerates four contributions
(protocol, 84-hypothesis design space, 35-tag screening sweep,
dual-track audit). [`README.md`](README.md) §1 mirrors them.

---

## 2. Limitations

### 2.1 — Does the paper discuss the limitations of the work performed by the authors?
**[Yes]** A dedicated [`LIMITATIONS.md`](../paper/LIMITATIONS.md) enumerates
single-hardware scope, screening-only single-seed bulk, 12-epoch
horizon, CIFAR-only scale, image-classification-only empirical track,
composite-metric specificity, and absence of multiplicity correction.
[`PAPER.md`](PAPER.md) imports the same content verbatim into its
Limitations section.

---

## 3. Theoretical results

### 3.1 — Does the paper provide the full set of assumptions and a complete (and correct) proof for all theorems?
**[N/A]** The submission is an **empirical** methodology paper; no
formal theorems are claimed. The composite-metric definition in
[`src/nature_inspired_networks/eval.py`](src/nature_inspired_networks/eval.py)
is a design choice, not a theorem.

### 3.2 — Are the assumptions used in the proofs stated?
**[N/A]** No proofs.

---

## 4. Experimental result reproducibility

### 4.1 — Does the paper fully disclose all the information needed to reproduce the main experimental results?
**[Yes]** Every run lands in a per-experiment archive
(`ideas/<NN>/experiments/expNNN_*/`) carrying the seed, config,
reasoning blob, metrics, history, and `best.pt`, per [`CLAUDE.md`](CLAUDE.md)
Rules 8 & 9. The append-only `experiments/experiment_log.jsonl` is
the master ledger. [`REVIEWER_CHECKLIST.md`](../paper/REVIEWER_CHECKLIST.md)
maps every paper claim to its reproduction command + log file.

### 4.2 — Are the exact commands to reproduce results provided?
**[Yes]** [`README.md`](README.md) §2 (4-command clone → smoke),
[`SETUP.md`](../docs/SETUP.md) §7–§9, and per-archive READMEs all include the
PowerShell command line.

---

## 5. Open access to data and code

### 5.1 — Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results?
**[Yes]** The entire codebase is public at
https://github.com/dlmastery/nature_inspired_networks under MIT
licence (see [`pyproject.toml`](pyproject.toml)). The live dashboard
mirroring every result is at
https://dlmastery.github.io/nature_inspired_networks/. Data is
CIFAR-10 / CIFAR-100, both public benchmark datasets with
permissive licences (§9.1).

### 5.2 — Are the datasets and pre-trained models referenced?
**[Yes]** CIFAR-10 / CIFAR-100 ([Krizhevsky 2009]) are downloaded at
runtime from the canonical Toronto host (see [`SETUP.md`](../docs/SETUP.md)
§4). No pre-trained weights are used; every run trains from scratch.

---

## 6. Hyperparameter details

### 6.1 — Does the paper specify the training details (e.g., the hyperparameters used)?
**[Yes]** Every config lives in `configs/*.yaml` and is bundled into
each experiment archive (Rule 8). Headline configs:
[`configs/cifar10_sota_smoke.yaml`](configs/cifar10_sota_smoke.yaml)
(Rule-13 pre-flight) and
[`configs/cifar10_quick.yaml`](configs/cifar10_quick.yaml) (12-epoch
ablation matrix). Per-experiment configs are immutable thanks to
Rule 1 (one config change per experiment, no silent compounding).

### 6.2 — Are hyperparameter search ranges disclosed?
**[Yes]** The current campaign is *not* a hyperparameter search —
each hypothesis is run at its pre-registered config. The planned
Phase-9 per-hypothesis hill-climb (see
[`skills/autoresearch-per-hypothesis-hillclimb/`](skills/autoresearch-per-hypothesis-hillclimb/))
will pre-register its ranges before launch.

---

## 7. Statistical significance

### 7.1 — Does the paper report error bars or other forms of variability for the main results?
**[Partial — Yes for graduates, No for bulk screening]** The three
Phase-5 graduates (`phi_budget`, `fib_depth`, `golden_momentum`)
carry seed-median + per-seed numbers across seeds {0, 1, 2}; see
[`FINDINGS.md`](../paper/FINDINGS.md) and the per-experiment dashboard pages.
**~80 % of the 35-tag screening sweep is single-seed by design**
(CLAUDE.md Rule 19, Phase 2) — these are not publication claims and
are explicitly framed as screening data in
[`LIMITATIONS.md`](../paper/LIMITATIONS.md). No multiplicity correction is
applied to the 84-hypothesis design space; this is the headline
limitation.

---

## 8. Compute used

### 8.1 — Does the paper disclose the type of compute workers and total amount of compute used?
**[Yes]** Hardware: 1× NVIDIA RTX 4090 Laptop, 16 GB VRAM, Windows 11
([`CLAUDE.md`](CLAUDE.md) §2). Total compute: ≈ **50 GPU-hours** across
the 35-tag screening sweep + Phase-5 3-seed re-runs (see
[`ETHICS_STATEMENT.md`](../paper/ETHICS_STATEMENT.md) and per-run `metrics.json`
files which record wall-clock).

### 8.2 — Per-run compute disclosed?
**[Yes]** Each `metrics.json` carries wall-clock, FLOPs, params, and
latency; the composite metric uses log10(params_M) + log10(latency_ms)
so the per-run cost is reflected in the headline number.

---

## 9. Datasets

### 9.1 — Are dataset licences and conditions of use disclosed?
**[Yes]** CIFAR-10 / CIFAR-100 (Krizhevsky 2009, University of
Toronto; permissively licensed for non-commercial research, in
universal use across the ML benchmark literature). MedMNIST
PathMNIST (Yang et al. 2023, CC-BY-4.0) is downloaded but not part
of the headline campaign; usage is gated by the upstream licence. See
[`ETHICS_STATEMENT.md`](../paper/ETHICS_STATEMENT.md).

### 9.2 — Is responsible use of the data demonstrated?
**[Yes]** Both CIFAR datasets are derived from labelled natural
photographs with no personally identifying information; usage is
restricted to image-classification benchmarking. No data is
re-distributed from this repo; only download instructions to the
canonical upstream hosts are provided ([`SETUP.md`](../docs/SETUP.md) §4).

---

## 10. License of code

### 10.1 — Is the code released under an appropriate licence?
**[Yes]** MIT licence, declared in
[`pyproject.toml`](pyproject.toml) `[project]` table:
`license = { text = "MIT" }`. This is a permissive licence allowing
academic and commercial reuse.

---

## 11. Crowdsourcing & human subjects

### 11.1 — Does the paper involve crowdsourcing or research with human subjects?
**[N/A]** No crowdsourcing, no human subjects, no human-annotation
campaigns. The work is pure benchmark-image classification on
pre-existing public datasets.

### 11.2 — IRB approval?
**[N/A]** No human-subjects research; IRB is not applicable. See
[`ETHICS_STATEMENT.md`](../paper/ETHICS_STATEMENT.md).

---

## 12. Computational resources

### 12.1 — Are computational resource requirements disclosed enough that others can attempt reproduction?
**[Yes]** §8.1 above. Concretely: 1× RTX 4090 Laptop / 16 GB / bf16
AMP. The four-command quick-start in [`README.md`](README.md) §2
reproduces the SOTA-baseline smoke (~2 min wall-clock) on identical
hardware; the full 35-tag screening sweep takes ≈ 90 min and the
3-seed Phase-5 re-sweep ≈ 5 hr.

---

## 13. Safeguards

### 13.1 — Does the paper describe safeguards for responsible release of data or models that have a high risk of misuse?
**[N/A]** Image classification on benchmark CIFAR datasets produces
no models with a misuse risk distinguishable from any other CIFAR
classifier. No models are released; only training code +
reproduction instructions.

---

## 14. Broader impacts

### 14.1 — Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
**[Yes]** [`ETHICS_STATEMENT.md`](../paper/ETHICS_STATEMENT.md) §3 explicitly
addresses dual-use: the contribution is methodological (a protocol
for auditing geometric inductive biases) and the empirical artefacts
(CIFAR-scale classifiers) have negligible direct societal impact
beyond the existing benchmark literature. Indirectly, the protocol
itself (audit-traced screening) is a small positive contribution to
reproducibility hygiene.

---

## 15. Author identification

### 15.1 — Does the paper preserve the anonymity of authors (where applicable for double-blind review)?
**[Yes — for submission]** The submission camera in
[`PAPER.md`](PAPER.md) and [`paper_abstract.md`](../paper/paper_abstract.md)
is anonymised for blind review. The codebase author in git history
(`dlmastery <eranti@gmail.com>`) is disclosed in
[`ETHICS_STATEMENT.md`](../paper/ETHICS_STATEMENT.md) for the camera-ready /
final-decision phase only.

---

## 16. Ethical conduct

### 16.1 — Have the authors read and considered the NeurIPS Code of Ethics?
**[Yes]** The codebase strictly avoids the prohibited content listed
in [`CLAUDE.md`](CLAUDE.md) §6: no PII, no PHI, no closed datasets,
no re-uploaded pre-trained weights, no secrets / API keys in the
repo. The dual-use, IRB, and data-licensing posture is covered in
[`ETHICS_STATEMENT.md`](../paper/ETHICS_STATEMENT.md).

---

## 17. Reproducibility — explicit
*(included as the final question because reproducibility is the headline
contribution.)*

### 17.1 — Is the full pipeline from raw data to reported numbers reproducible by a third party with the released artefacts?
**[Yes]** A reviewer with the [`README.md`](README.md) §2 four-command
quick-start on a 4090-class GPU can:

1. Run the SOTA smoke and verify top-1 ≥ 80 %.
2. Run the 35-tag CIFAR-10 sweep and observe the same leaderboard.
3. Re-generate the live dashboard locally and verify it matches the
   published GitHub Pages version byte-for-byte.

Per-claim evidence pointers (file + line + reproduction command) are
in [`REVIEWER_CHECKLIST.md`](../paper/REVIEWER_CHECKLIST.md). The append-only
`experiments/experiment_log.jsonl` is the canonical ground truth that
all dashboards and reports derive from.

---

*Last reviewed: 2026-05-29. Re-fill this checklist before any
camera-ready submission; flag any drift from this snapshot in the
status journal of [`PAPER.md`](PAPER.md).*
