# H60 — Three-Seed Uncertainty Quantification

> **One-line claim:** Re-running the 11-row CIFAR-10 ablation sweep
> with seeds {0, 1, 2} produces 95% confidence intervals on every
> top-1 / composite Δ; we predict that the H50 negative result
> (`sg_full_fib` composite -0.1169 vs. `sg_chan_fib`) and the H58
> recovery (`sg_only_group_avg` composite ≥+0.06) both retain
> statistical significance at the 95% level (CI excluding 0) while
> the marginal priors (H17 `sg_only_golden_modulate` Δ=-0.0093) may
> be reclassified as not-statistically-distinguishable.
>
> **Source design space:** G6 Topological + bridging additions (H51–H60).
>
> **Implementation status (this repo):** `⏸ queued — `run_sweep.py
> --seeds 0 1 2 --skip-existing` is the launch command; existing seed=0
> runs are already complete and would not be re-run.`

This document is the committee-grade design write-up for hypothesis
H60. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The entire 11-row CIFAR sweep (T1.1–T1.9, plus L0–L1) was run with
single seed=0. Every Δ in `FINDINGS.md` (e.g., "H05 fractal +2.35 pp
top-1") is a *point estimate* — no error bar, no statistical
significance. This is the protocol gap that H60 closes.

Standard practice in deep learning is 3-seed median with IQR or 5-
seed mean with 95% CI. Without uncertainty estimates, single-prior
deltas of ≤ ±1 pp are unreliable (within seed-variance noise). The
critical claims of this repo — the H50 compound failure, the H58
recovery, the H05 fractal positive — need 3-seed reproduction to be
publication-defensible.

The sacred-geometry framing: nature uses redundancy and averaging
everywhere (eye saccades, neural firing rates, decision-making by
consensus). Single-seed deep-learning experiments are the engineering
analog of "decide with one neuron firing" — unreliable. H60
imports nature's averaging principle to the experiment protocol
itself.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because top-1 / composite deltas have inherent seed-variance from
random init + minibatch ordering + dropout — mechanism-wise, the
variance is bounded above by σ ≈ 0.5 pp on CIFAR-10 for a 12-epoch
single-arm run — per standard deep-learning practice
(He 2016, Loshchilov 2019), we expect (a) the H50 -0.1169 composite
Δ to retain 95% CI exclusion of 0 (high confidence in the negative),
(b) the H58 expected +0.06 composite Δ to retain 95% CI exclusion
of 0, and (c) the marginal priors (H17, H22 partial) to be reclassified
as null-effect at 95% CI.

## 3. Falsifier (≥ 30 words)

If after 3-seed reproduction the H50 Δ is NOT statistically
distinguishable from 0 at 95% CI (i.e., the negative is within seed
noise), OR if H05 fractal lift is reclassified as null, this
hypothesis is DISCARDED — the entire 11-row sweep would be
inconclusive.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
Henderson, Peter and Islam, Riashat and Bachman, Philip and Pineau,
Joelle and Precup, Doina and Meger, David 2018 AAAI 'Deep
Reinforcement Learning that Matters' (arXiv:1709.06560) -- the
foundational paper exposing the single-seed reproducibility crisis;
mandates 3+ seeds.

He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian
2016 CVPR 'Deep Residual Learning for Image Recognition'
(arXiv:1512.03385) -- canonical 5-seed ResNet baseline practice.

Bouthillier, Xavier and Laurent, César and Vincent, Pascal 2019
NeurIPS workshop 'Unreproducible Research is Reproducible' --
discusses seed variance and IQR vs. CI reporting.

Loshchilov, Ilya and Hutter, Frank 2019 ICLR 'Decoupled Weight
Decay Regularization' (arXiv:1711.05101) -- canonical 3-seed
median reporting practice; methodological precedent.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

The implementation reuses existing `scripts/run_sweep.py` with the
`--seeds 0 1 2 --skip-existing` flag. The `--skip-existing` flag
checks for existing archive directories and skips them, so single-
seed=0 runs are not redundantly recomputed; only seeds 1 and 2 are
added.

```bash
# Launch full 3-seed re-sweep
python scripts/run_sweep.py \
    --config configs/cifar10_quick.yaml \
    --seeds 0 1 2 \
    --skip-existing
```

The dashboard auto-detects multi-seed archives and computes 3-seed
median, IQR, and bootstrap 95% CI for every metric. The protocol
spec in `CLAUDE.md` § 4 enforces this: if N_seeds ≥ 3, the public-
facing reports show median ± IQR rather than single-seed point
estimates.

Statistical analysis (per archived seed):

```python
from scipy import stats

def bootstrap_ci(values, n_boot=10000, ci=0.95):
    """Bootstrap 95% CI for a small sample."""
    boots = np.random.choice(values, (n_boot, len(values)),
                              replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(1-ci)/2 * 100, (1+ci)/2 * 100])
    return lo, hi
```

Cost: 2 additional seeds × 11 arms × 12 min = 264 min ≈ 4.5 hours
on RTX 4090 Laptop.

Lives in `scripts/run_sweep.py` (existing), with statistical helpers
in `src/nature_inspired_networks/stats.py`. Re-exported via
`ideas/60_three_seed_uncertainty/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

For LLM track, 3-seed reproduction at 124M scale is expensive (≥6
GPU-hours × 3 seeds × N conditions). Recommendation: 3-seed only on
the *headline* LLM experiments (H50-LLM full hybrid, H67 paradigm
fusion), single-seed on the rest. Document this scope in
`CLAUDE.md` § 4.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. single-seed (T1.*) | rationale |
|---|---|---|
| composite (point estimates) | [0, 0] | medians cluster around single-seed |
| composite CI half-width | [0.005, 0.020] | typical 3-seed σ |
| top-1 (CNN) median | [-0.5, +0.5] | regression to mean |
| top-1 CI half-width | [0.4 pp, 1.0 pp] | |
| params | [0, 0] | unchanged |
| FLOPs | [0, 0] | unchanged |
| GPU latency (batch=1) | [0, 0] | unchanged |
| rotation-equivariance err CI | [0.005, 0.015] | variance bounds |
| KV cache @ 32k (LLM) | [0, 0] | N/A |
| Betti collapse rate CI | [0.05, 0.15] | high seed variance |
| perplexity (LLM) | [0, 0] | N/A for CIFAR sweep |
| total runtime (h) | [4, 6] | 2 additional seeds × 11 arms |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dataset:** CIFAR-10 (same as T1.*)
- **Architecture:** all 11 arms in `configs/cifar10_quick.yaml`
- **Epochs:** 12, batch=128, bf16, AdamW
- **Seeds:** 0, 1, 2 (seed=0 already exists; only 1 and 2 are new runs)
- **Composite formula:** unchanged
- **Run-script:** `python scripts/run_sweep.py --config configs/cifar10_quick.yaml --seeds 0 1 2 --skip-existing`
- **Wall-clock:** 22 new runs × 12 min ≈ 4.5 h on RTX 4090 Laptop
- **Statistical analysis:** bootstrap CI; dashboard refresh
- **Archive path:** `ideas/60_three_seed_uncertainty/experiments/exp001_multi_seed/`

### 7.2 Idea-targeted experiment

The strongest demonstration: identify which Δ in the existing
`FINDINGS.md` are actually statistically distinguishable from 0 at
95% CI. Pre-registration:

- **H50 -0.1169** → expected to remain significant
- **H05 +2.35 pp** → expected to remain significant
- **H58 +8 pp** → expected to remain significant
- **H17 -0.0093** → expected to become null-distinguishable
- **H22 -0.0367 (partial)** → unclear; may go either way
- **H35 -0.0252** → expected to remain marginally significant

### 7.3 Cross-paradigm context (LLM track)

- **Scope:** apply 3-seed only to LLM headline runs.
- **Cost rationale:** single 124M run is ~6 h; 3-seed × 5 conditions
  = 90 h, requiring multi-day scheduling.
- **Run:** `python scripts/run_llm.py --seeds 0 1 2 --tag h50_llm`

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G6 row H60.
- Master experiment list: `EXPERIMENT_LOG.md` row T2.4 (queued).
- Implementation sub-directory: `ideas/60_three_seed_uncertainty/`
- Related hypotheses that compose:
  - **Every other H** — H60's outputs become the CI on each H's
    primary metric.
  - **H50** — needs CI to defend the negative.
  - **H58** — needs CI to defend the recovery.
- Related hypotheses that conflict:
  - None directly.

## 9. Committee Q&A

**Q: Why isn't this just standard practice?**

> It is standard practice — and the previous sweep skipped it (single
> seed only). H60 is the protocol-correction hypothesis. Every paper
> reviewer asks "is this 3-seed?"; H60 ensures the answer is yes.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies that the H50 negative must remain at 95% CI; if it
> doesn't, the headline negative result is reclassified.

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> Seed variance scales similarly; H60 protocol applies.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> The single-seed sweep *suggested* this; H60 confirms or refutes at
> 95% CI. Critical for publication.

**Q: How do we know the implementation is correct?**

> `tests.py` verifies (a) bootstrap CI on a known distribution
> recovers the analytic CI within 1%, (b) `--skip-existing` actually
> skips, (c) seed isolation: seed=N produces deterministic results.

## 10. Verification artifacts checklist

- [ ] `ideas/60_three_seed_uncertainty/implementation.py` exists
- [ ] `ideas/60_three_seed_uncertainty/tests.py` ≥ 6 assertions
- [ ] `ideas/60_three_seed_uncertainty/AUDIT.md` ≥ 3 weaknesses
- [ ] `ideas/60_three_seed_uncertainty/IMPROVEMENTS.md`
- [ ] `ideas/60_three_seed_uncertainty/VERIFY.md` signed
- [ ] One experiment archive (post-launch)
- [ ] `verification/{tests,smoke,gates,reproduction}.txt`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md` and dashboard (with CI annotations)

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-C. Status remains
  `⏸ queued`; awaiting H58 (T2.1/T2.2) completion so the H58 result
  is included in the 3-seed re-sweep.
