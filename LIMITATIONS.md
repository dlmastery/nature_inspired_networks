# Limitations — `nature_inspired_networks`

> Honest, reviewer-facing enumeration of the campaign's scope, design
> choices, and statistical caveats. Companion to [`PAPER.md`](PAPER.md),
> [`ETHICS_STATEMENT.md`](ETHICS_STATEMENT.md), and
> [`NEURIPS_CHECKLIST.md`](NEURIPS_CHECKLIST.md). Every limitation is
> stated up front rather than left for a reviewer to find.

The contribution is a *methodology + screening* paper, **not** a SOTA
claim. The limitations below shape every empirical result and define
the boundary of what may legitimately be concluded.

## 1. Single hardware platform

All results were produced on **one specific GPU**: a single NVIDIA
RTX 4090 Laptop with 16 GB VRAM running Windows 11. We have **not
ablated**:

- A100 / H100 / desktop-class GPUs (e.g., RTX 4090 24 GB) where
  larger batches or different numerical behaviours might shift the
  composite metric.
- AMD / Apple-Silicon backends.
- TPU / non-CUDA hardware.

The composite metric includes `log10(latency_ms)`, so a hardware
shift would algebraically shift composites; we do not claim
hardware-invariant rankings.

## 2. Screening-only single-seed bulk

Per [`CLAUDE.md`](CLAUDE.md) Rule 19 and the phased workflow,
**approximately 80 % of the 35-tag screening sweep was run at a
single seed × a single config × 12 epochs**. This is a screening
regime by design — its purpose is to decide which few hypotheses
graduate to expensive evaluation, not to publish numbers.

Only the **three Phase-5 graduates** (`phi_budget`, `fib_depth`,
`golden_momentum`) carry seed-median + per-seed numbers across seeds
{0, 1, 2} (see [`FINDINGS.md`](FINDINGS.md) and the live dashboard).
A single-seed number for any other tag is **not a publication claim**
and should be read as calibration evidence, not as a competitive
benchmark.

## 3. 12-epoch CIFAR-10 horizon for screening

The screening sweep uses a 12-epoch ResNet-20 schedule (Rule 13
expected-band: top-1 ≥ 80 % @ 12 epochs). This horizon may
**underexpress** effects that need ≥ 100 epochs to surface:

- H41 `golden_adam` is a textbook example — the original prediction
  was based on Reddi-2018 (ICLR) convergence behaviour at long
  horizons, and the 12-epoch falsification (−33 pp) was later
  re-qualified to a much milder −1 pp under a Reddi-2018-style
  β-only test (see [`FINDINGS.md`](FINDINGS.md)).
- Any "late convergence" prior (e.g., curriculum-style growth
  schedules, slow-emerging regularisers) is therefore at risk of a
  premature negative verdict.

The Phase-9 hill-climb plan
([`skills/autoresearch-per-hypothesis-hillclimb/`](skills/autoresearch-per-hypothesis-hillclimb/))
addresses this for the three graduates only.

## 4. CIFAR scale; no ImageNet or downstream evaluation

All headline numbers are at **CIFAR-10 / CIFAR-100 (32×32, 50 k / 10 k
splits)**. We do not claim transfer to:

- ImageNet (full or subsets — Tiny, 100, 1k, 21k).
- Downstream tasks (object detection, segmentation, fine-tuning to
  medical / satellite imagery).
- Self-supervised pre-training regimes.

The dual-CNN+LLM design of `NaturePriorFlags` is implemented
end-to-end but **no LLM-track empirical data** has been collected.
The LLM-track mechanism is documented in every hypothesis design doc
(committee-grade per CLAUDE.md Rule 18) as a falsifiable prediction,
not as a result. This is the single largest gap between the design
space and the empirical campaign.

## 5. Image classification only

The campaign is entirely image-classification on benchmark CIFAR.
**No language-modelling, no generative-modelling, no time-series, no
RL** empirical data has been collected. Cross-modal claims rest on
the literature citations in each hypothesis doc, not on our own
training runs.

## 6. Composite metric is project-specific

The headline ranking metric is
`top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms)`
(SHA-256 fingerprint
`d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`).
This is a **project-specific aggregation**. Reviewers should be aware:

- A raw-top-1 ranking would rank H09 `phi_budget` strictly above the
  baseline (85.54 % vs 84.78 %) — agreeing with the composite — but
  may **rank some efficiency-leaning rows differently** to the
  composite.
- A FLOPs-normalised or latency-normalised metric would weight H02
  `fib_depth` differently.
- Multi-metric Pareto presentation (top-1 × params × latency) is on
  the live dashboard but is **not** the headline ranking.

The composite is a single SHA-256-fingerprinted design choice; an
edit of the formula refuses the next launch (Rule 2). A new formula
is a new science and requires a new branch / repo.

## 7. Multiplicity correction not applied

The 84-hypothesis design space induces a **multiple-comparisons
problem** that **no Bonferroni / Holm / Benjamini-Hochberg FDR
correction has been applied to**. The three Phase-5 graduates beat
the baseline composite with raw seed-median p-values that would
likely **not survive an FDR-correction at α = 0.05 across 84
hypotheses**. This is the single most important statistical limitation:

- It is the reason the paper frames itself as *methodology +
  screening* rather than *SOTA*.
- The Phase-9 hill-climb is the planned remedy: by promoting
  graduates through a per-hypothesis hill-climb with explicit
  pre-registration, the effective hypothesis count for the *evaluation
  phase* shrinks from 84 to 3, dramatically loosening the correction
  burden.

We recommend any reader who wishes to make a hardline statistical
claim apply the appropriate correction to the raw numbers in
[`FINDINGS.md`](FINDINGS.md) and the live dashboard.

## 8. Honest disclosure of motivation

The campaign was *inspired* by an exploratory private PDF on sacred
geometry. Per CLAUDE.md Rule 16 every artifact name is neutral /
academic, and per [`MANIFESTO.md`](MANIFESTO.md) every empirical claim
is grounded in a peer-reviewed paper and a pre-registered falsifier.
**The mystical framing has zero load-bearing role in any reported
number** — but readers deserve to know it influenced *which*
hypotheses were enumerated in the first place. See
[`ETHICS_STATEMENT.md`](ETHICS_STATEMENT.md) §5.

## 9. Audit traceability rests on author-internal review

The dual-track audit (CLAUDE.md Rule 22) is run by Claude agents
following pre-registered audit skills
([`skills/autoresearch-critic-team/`](skills/autoresearch-critic-team/),
[`skills/autoresearch-scicritic-team/`](skills/autoresearch-scicritic-team/)).
It is **not** an external double-blind audit. While the audit
artifacts (`audits/G*_audit.md` and the design-doc Addendum
sections) are public and replayable, an independent third-party
audit has not been performed.

## 10. Negative results may reflect implementation, not hypothesis

Per the Fixer campaign (Rule 21), several initial negatives were
later requalified after an implementation bug was patched (e.g.,
H08, H41). Some currently-DISCARD verdicts in
[`FINDINGS.md`](FINDINGS.md) **may flip** under a future Fixer pass
or a Phase-9 hill-climb. We err on the side of reporting both the
pre-fix and post-fix numbers explicitly so a reader can see the
trajectory; the post-fix verdict is the operative one.

---

*Last reviewed: 2026-05-29. These limitations are mirrored verbatim
into [`PAPER.md`](PAPER.md)'s Limitations section to satisfy
NEURIPS_CHECKLIST.md item 2.1.*
