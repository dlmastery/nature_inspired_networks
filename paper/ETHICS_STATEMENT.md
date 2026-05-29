# Ethics Statement — `nature_inspired_networks`

> Companion to [`PAPER.md`](PAPER.md), [`LIMITATIONS.md`](../paper/LIMITATIONS.md),
> and [`NEURIPS_CHECKLIST.md`](../paper/NEURIPS_CHECKLIST.md). One page; designed
> to satisfy a NeurIPS / ICML / ICLR Ethics-AC pass without further
> reading.

## 1. Datasets

The empirical campaign uses two **public benchmark image-classification
datasets**:

| Dataset | Source | Licence / use terms | PII / PHI |
|---|---|---|---|
| **CIFAR-10** | Krizhevsky 2009 (Univ. of Toronto) | Permissively licensed for non-commercial research; in universal use across the ML benchmark literature | None — labelled natural photographs (animals, vehicles) with no human identifiers |
| **CIFAR-100** | Krizhevsky 2009 (Univ. of Toronto) | Same as CIFAR-10 | None — same provenance |

Neither dataset is re-distributed from this repository; only download
instructions to the canonical upstream hosts are provided in
[`SETUP.md`](../docs/SETUP.md) §4. torchvision verifies tarball MD5 integrity
on every download.

**(Optional, not part of headline campaign):** MedMNIST PathMNIST
(Yang et al. 2023, CC-BY-4.0) is loaded by
[`src/nature_inspired_networks/data.py`](src/nature_inspired_networks/data.py)
for a planned medical-imaging side study. It is licensed CC-BY-4.0
and contains no PHI per its upstream description (de-identified
colon-pathology patches). It plays no role in the headline claims.

No closed, registration-gated, or proprietary datasets are used.
No data is collected from human subjects by this project.

## 2. Compute

- **Hardware:** 1× NVIDIA RTX 4090 Laptop GPU, 16 GB VRAM, Windows 11.
- **Software:** Python 3.13, PyTorch 2.x (CUDA 12.x), bf16 AMP.
- **Total compute:** approximately **50 GPU-hours** across the entire
  campaign — 35-tag CIFAR-10 screening sweep (≈ 25 GPU-hours,
  single-seed, 12 epochs each), Phase-5 3-seed re-sweep on the three
  Phase-4 graduates (≈ 15 GPU-hours), Phase-4 CIFAR-100 quality runs
  on the same graduates (≈ 10 GPU-hours). Per-run wall-clock is
  recorded in every `metrics.json` and surfaced in the live dashboard.

This is a deliberately small compute footprint. The campaign is
designed to be **reproducible on commodity laptop-class hardware** —
no 8-GPU node, no multi-day training, no closed cluster.

## 3. Dual-use & broader impact

The contribution is **methodological**: a protocol for
audit-traced, refusal-to-launch screening of geometric / topological
inductive biases on benchmark image classifiers. The artefacts
produced are:

1. CIFAR-scale ResNet-20-class classifiers with composite-metric
   characterisations. **Dual-use risk: negligible** — these are
   strictly weaker than any standard CIFAR baseline in absolute
   capability terms and confer no novel misuse-enabling capabilities.
2. A reusable autoresearch protocol (refusal-to-launch gates, audit
   tracks, append-only logging). **Dual-use risk: negligible** — a
   methodology paper on reproducibility hygiene.
3. 84 committee-grade hypothesis design docs grounded in published
   geometric-deep-learning literature. **Dual-use risk: negligible.**

We are aware of no foreseeable misuse pathway that is distinguishable
from the misuse pathway of CIFAR classification in general. Indirect
positive impact: the protocol itself nudges the field toward more
rigorously audited screening pipelines for inductive-bias claims —
see [`MANIFESTO.md`](../paper/MANIFESTO.md) for the long-form argument.

## 4. Anonymity / author identification

For **blind review** the submission cameras
([`PAPER.md`](PAPER.md), [`paper_abstract.md`](../paper/paper_abstract.md)) are
anonymised. The codebase author, recoverable from git history (and
disclosed here for the camera-ready / final-decision phase), is:

> `dlmastery <eranti@gmail.com>`

The GitHub identity `dlmastery` and the associated repository
https://github.com/dlmastery/nature_inspired_networks would
deanonymise the submission if linked during the blind-review window;
reviewers are asked not to search for project-specific identifiers
during review.

## 5. Honest disclosure of motivation

This work was motivated, in part, by an exploratory reading of an
earlier private PDF (*Sacred Geometry and Neural Networks*) that
framed the program in mystical / aesthetic terms. The honest position
of the project — set out at length in [`MANIFESTO.md`](../paper/MANIFESTO.md) —
is:

- The **inspiration** is acknowledged in prose only.
- **Every empirical claim** is grounded in (a) a peer-reviewed
  geometric-deep-learning paper for the prior, (b) a pre-registered
  numeric falsifier, and (c) the SHA-256-fingerprinted composite
  metric. The mystical framing has **zero load-bearing role** in any
  reported number.
- **Artifact names are neutral / academic** (CLAUDE.md Rule 16): the
  package is `nature_inspired_networks`, the core block is
  `NaturePriorBlock`, the hypothesis groups are G1–G8. No
  "sacred-geometry"-named class exists in the codebase.

We disclose this background so a reviewer reading
[`MANIFESTO.md`](../paper/MANIFESTO.md) is not surprised. The empirical claims
in [`FINDINGS.md`](../paper/FINDINGS.md), [`PAPER.md`](PAPER.md), and the live
dashboard stand on the composite metric + dual-track audit alone,
**independently of the inspiration**.

## 6. IRB / human-subjects ethics review

**Not applicable.** No human subjects, no crowdsourcing, no clinical
data, no user studies. The work is entirely on pre-existing public
benchmark datasets of labelled natural photographs.

## 7. Code-of-conduct compliance

- No PII / PHI handled or stored.
- No closed datasets used.
- No re-uploaded pre-trained weights (links to upstream sources only).
- No secrets, API keys, or `.env` files in the repository (see
  [`CLAUDE.md`](CLAUDE.md) §6 for the prohibition).
- All experiment artefacts (configs, metrics, reasoning blobs,
  dashboards) are published in the open under the MIT licence in
  [`pyproject.toml`](pyproject.toml).
- The campaign follows the project's own 27 normative rules
  ([`CLAUDE.md`](CLAUDE.md)) without exception; the refusal-to-launch
  gate stack is enforced in code, not by author discretion.

---

*Last reviewed: 2026-05-29.*
