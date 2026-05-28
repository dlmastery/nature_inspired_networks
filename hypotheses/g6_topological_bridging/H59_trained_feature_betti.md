# H59 — Trained-Feature Betti Curves

> **One-line claim:** Recomputing the per-stage Betti curves
> (`compute_topology.py`) on **trained** `best.pt` checkpoints
> (rather than the current fresh-init features) reveals the actual
> topological simplification the priors achieve; we predict that
> single-prior models trained with cymatic-init or fractal recursion
> show ≥40% faster β₀ collapse on trained features than the
> reference, validating that the priors *do* shape topology at
> training time even when they fail on top-1, per Naitzat 2020.
>
> **Source design space:** G6 Topological + bridging additions (H51–H60).
>
> **Implementation status (this repo):** `~ partial — checkpoint
> saving is wired in `src/nature_inspired_networks/runner.py:save_run` but
> legacy T1.* runs lack `best.pt` (saving turned on later). Depends on
> H58 (T2.1/T2.2) producing the first ever saved checkpoints.`

This document is the committee-grade design write-up for hypothesis
H59. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The previous CIFAR sweep computed Betti curves but did so on
*fresh-init* features (the `compute_topology.py` script extracted
features from a freshly instantiated model, never from a trained
one). This means the topology measurements in the existing dashboard
reflect only the *initial* topological structure imposed by the
priors — not the *trained* structure. Yet the interesting question
is the latter: does cymatic-init produce topologically simpler
*trained* features than He-init, even if both achieve similar
top-1?

This distinction matters because top-1 is a single scalar; the
topology of intermediate features is a multi-dimensional descriptor
that may reveal *why* a prior was negative. For example: cymatic-
init produced T1.7 (-2.67 pp top-1) — was the topology actually
simpler (and the simplicity hurt the discriminative power), or was
the topology disrupted (and the disruption hurt)? Trained-feature
Betti curves answer this.

The sacred-geometry framing: topology is the "shape of the
representation manifold" — natural systems (folded proteins, brain
networks) have characteristic topologies that distinguish well-
folded vs. mis-folded structures. Measuring topology on trained
features is the engineering equivalent of measuring the folded
state, not just the unfolded primary structure. H59 is the
diagnostic enabler for *every* downstream Betti claim (H51, H54,
H65).

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because trained representations differ structurally from fresh-init
representations — mechanism-wise, training drives features toward
class-cohesive clusters (β₀ → n_classes) while a fresh-init
network's features are essentially random — per Naitzat 2020 and
Hofer 2019, we expect that trained-feature Betti curves of the
H58-saved checkpoints show ≥40% faster β₀-collapse during training
for the fractal-prior arm (T1.5) and ≥20% faster for the cymatic-
init arm (T1.7) vs. the reference (sg_chan_fib), even when their
top-1 differs only marginally.

## 3. Falsifier (≥ 30 words)

If the trained-feature Betti curves are NOT measurably different
between arms (≤10% variation across all priors), the topology-as-
discriminator hypothesis fails, and H59 (along with the broader
Betti-loss program H51/H54/H65) is DISCARDED.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
Naitzat, Gregory and Zhitnikov, Andrey and Lim, Lek-Heng 2020 JMLR
'Topology of deep neural networks' (arXiv:2004.06093) -- shows
trained networks topologically simplify while fresh-init networks
do not; the empirical motivation for trained-feature analysis.

Hofer, Christoph and others 2017 NeurIPS 'Deep Learning with
Topological Signatures' (arXiv:1707.04041) -- foundational work
on PH-based representation analysis.

Edelsbrunner, Herbert and Letscher, David and Zomorodian, Afra
2002 Discrete Comput Geom 'Topological persistence and
simplification' -- the foundational PH algorithm reference.

Bauer, Ulrich 2021 JACM 'Ripser: efficient computation of Vietoris-
Rips persistence barcodes' (arXiv:1908.02518) -- the
implementation library we use (PyRipser binding) for PH
computation on trained features.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

H59 is a *post-hoc analysis* pipeline. Pipeline:

```python
from src.nature_inspired_networks.topology import betti_curve
from src.nature_inspired_networks.runner import load_checkpoint

def compute_trained_betti(checkpoint_path, dataset, n_samples=512):
    model = load_checkpoint(checkpoint_path)
    model.eval()
    # extract penultimate features for n_samples random test examples
    feats = []
    for x, y in dataset.take(n_samples):
        with torch.no_grad():
            feat = model.extract_penultimate(x)
        feats.append(feat)
    feat_matrix = torch.cat(feats, dim=0)  # (n_samples, D)
    # compute PH from the resulting point cloud
    return betti_curve(feat_matrix, max_scale=2.0, max_dim=1)
```

Cost: ≈ 5 min per checkpoint on RTX 4090 (FFI to Ripser for the
PH computation is the bottleneck). Total cost for the 11-row sweep:
≈ 55 min. Inference cost: zero (analysis pipeline only).

Lives in `scripts/compute_trained_topology.py`, with helper code in
`src/nature_inspired_networks/topology.py:trained_betti`. Re-exported via
`ideas/59_trained_betti/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

For decoder LLMs, the analog is to compute Betti curves on the
*residual-stream activations* at each layer for a sample of
TinyStories sentences. This requires saving LLM checkpoints (which
the LLM-track does by default). The pipeline is symmetric to CNN.

FA2 compatibility: extraction is post-FA2, unaffected. Causal mask
preservation: unaffected.

Expected: at 124M, the trained Betti curves show monotonic
simplification across layers — the staircase signature predicted by
Naitzat 2020.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ (trained vs. fresh-init Betti) | rationale |
|---|---|---|
| composite | [0, 0] | analysis only |
| top-1 (CNN) | [0, 0] | analysis only |
| params | [0, 0] | analysis only |
| FLOPs | [0, 0] | analysis only |
| GPU latency (batch=1) | [0, 0] | analysis only |
| rotation-equivariance err | [0, 0] | analysis only |
| KV cache @ 32k (LLM) | [0, 0] | analysis only |
| Betti collapse rate trained | [-0.4, -0.2] | core observable |
| Betti collapse rate fresh-init | [0, +0.05] | reference; should not change |
| perplexity (LLM) | [0, 0] | analysis only |
| analysis wall-clock (s) | [200, 400] | depends on n_samples |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dependency:** H58 must have saved `best.pt` checkpoints (T2.1, T2.2).
- **Dataset:** CIFAR-10 test set
- **Samples:** 512 random test examples per checkpoint
- **Checkpoints to process:** all `best.pt` files in `experiments/cifar10/*/`
  (currently only T2.1, T2.2 available; later T2.4 re-sweep adds more)
- **Run-script:** `python scripts/compute_trained_topology.py --all`
- **Wall-clock:** ≈ 5 min × 11 rows = 55 min (after H58 completes)
- **Output:** Per-checkpoint `trained_betti.json` with β₀/β₁ curves +
  comparison plot in dashboard
- **Archive path:** `ideas/59_trained_betti/experiments/exp001_post_t2_topology/`

### 7.2 Idea-targeted experiment

Compare trained Betti curves across the 11 prior arms after the
3-seed re-sweep (T2.4) and Sacred-NAS (H45) runs complete. Identify
which priors produce the simplest trained topology AND whether that
correlates with top-1.

### 7.3 Cross-paradigm context (LLM track)

After H67 (full-paradigm hybrid) runs, compute residual-stream
trained Betti at every transformer layer. Map the staircase.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G6 row H59.
- Master experiment list: `EXPERIMENT_LOG.md` row T2.3 (queued
  behind T2.1/T2.2).
- Implementation sub-directory: `ideas/59_trained_betti/`
- Related hypotheses that compose:
  - **H51** Betti loss — uses trained-feature Betti as eval metric.
  - **H54** PH activation reg — uses trained-feature Betti as eval.
  - **H58** Group avg-pool — provides the first ever saved
    checkpoints.
  - **H65** PH Betti-collapse loss term (LLM-track sibling).
- Related hypotheses that conflict:
  - None directly. Pure analysis.

## 9. Committee Q&A

**Q: Why isn't this just a metric, not a hypothesis?**

> The *hypothesis* is that trained-feature Betti is a more
> discriminating signal than fresh-init Betti for understanding
> which priors are helping. Pre-registering the prediction
> (cymatic-init has cleaner trained topology than reference) is
> the falsifiable element.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies that trained Betti curves must differ by ≥10% across
> priors. If all priors produce the same trained topology, the
> measurement is uninformative and the hypothesis fails.

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> N/A — H59 is analysis, not prior. The hypothesis is about whether
> the *measurement* discriminates, not whether a prior helps.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H59 is the diagnostic that explains *why* the compound failed.
> Different priors may disrupt topology in different ways; the
> trained-feature analysis reveals this.

**Q: How do we know the implementation is correct?**

> `tests.py` verifies (a) Betti curve on a known point cloud (n
> clusters → β₀ = n at appropriate scale), (b) checkpoint loading
> produces matching forward output, (c) feature extraction matches
> the saved penultimate.

## 10. Verification artifacts checklist

- [ ] `ideas/59_trained_betti/implementation.py` exists, tests green
- [ ] `ideas/59_trained_betti/tests.py` ≥ 6 assertions
- [ ] `ideas/59_trained_betti/AUDIT.md` ≥ 3 weaknesses
- [ ] `ideas/59_trained_betti/IMPROVEMENTS.md`
- [ ] `ideas/59_trained_betti/VERIFY.md` signed
- [ ] At least one experiment archive (post-H58)
- [ ] `verification/{tests,smoke,gates,reproduction}.txt`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-C. Status remains
  `⏸ queued`; depends on H58 producing first `best.pt` checkpoints.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G6 (elite-research-scientist critic). Critiquing the IDEA, not the implementation (audit at `audits/G6_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)
HIGH (for the methodology) but the "hypothesis" framing is wrong. Computing Betti curves on *trained* features rather than *fresh-init* features is straightforward methodology hygiene — Naitzat et al. 2020 JMLR (arXiv:2004.06093), Hofer 2017 NeurIPS (arXiv:1707.04041), and every PH-on-NN paper computes Betti on trained features. The original CIFAR sweep using fresh-init Betti was a bug, not a baseline; fixing it is correct but is not a *hypothesis*. This is infrastructure / methodology work mislabeled as a falsifiable scientific claim.

### Mechanism scrutiny
The doc claims "≥40% faster β₀-collapse for fractal arm vs. reference" — but the comparison is between *priors* via their *trained-feature Betti rate*. This means H59 is actually proposing a new evaluation metric (rate-of-β₀-collapse), then predicting that the metric ranks priors in a specific order. The mechanism for *why* fractal recursion would yield faster β₀-collapse is missing — fractal recursion is about parameter sharing across depth, which is orthogonal to feature-cluster topology. Why would shared weights produce simpler trained Betti? No mechanism given.

### Confounds (≥2)
1. **Trained-feature Betti depends on n_samples**: 512 samples → high variance in β₀ estimate; need to report CI. Comparing prior-A at 512-sample β₀ to prior-B at 512-sample β₀ confounds the topology with the sample noise.
2. **Penultimate-feature scale varies by prior**: cymatic-init produces high-norm features early; fractal produces variable-norm. Persistence threshold ε is feature-scale-coupled (same as H51 confound), so cross-prior Betti comparisons are not normalized.
3. **The hypothesis depends on H58 producing checkpoints — but H58 was empirically falsified** (T2.1 -4.46 pp), so the trained-feature analysis is now on a checkpoint of a regression, not a recovery.

### Numerology / specificity check
"≥40% faster for fractal, ≥20% faster for cymatic" — arbitrary thresholds with no derivation. Why 40% vs 20%? Why not 50/30 or 25/15?

### Literature precedent
Naitzat 2020 (arXiv:2004.06093); Hofer 2017 (arXiv:1707.04041); Moor 2020 "Topological Autoencoders" (arXiv:1906.00722). Trained-feature Betti is standard; the methodology is real and reproducible.

### Expected effect size (90% CI a priori)
Trained Betti curves WILL differ from fresh-init Betti curves by >20% (high confidence — this is well established). Whether fractal-prior specifically yields 40% faster collapse than reference: [-30%, +30%], i.e., null is well within CI.

### Minimum-distinguishing experiment
Already specified: compute trained Betti on all 11 archived checkpoints + report 3-seed CI on collapse-rate. The discriminating test is whether priors *order differently* under trained-Betti vs. top-1 — if order is identical, trained-Betti adds no information.

### Verdict
INFRASTRUCTURE-NOT-HYPOTHESIS — should be reclassified as a tooling/methodology fix (replace fresh-init Betti with trained-feature Betti in `compute_topology.py`) rather than a falsifiable hypothesis. The downstream prediction (40% / 20%) is unmotivated numerology layered on top of a sensible methodology fix.
