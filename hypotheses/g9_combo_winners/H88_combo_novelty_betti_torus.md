# H88 — `combo_novelty_betti_torus` (Phi-Budget + Toroidal Closure + Betti Topological Loss)

> **One-line claim:** Stacking the certified H09 phi_budget allocator
> with two C-novelty-pocket priors — H22 toroidal closure (A3
> wrap-padding) and H51 topological Betti loss (A16 PH regularisation)
> — delivers Δ +0.5 to +1.5 pp over the un-tuned CIFAR-100 30-ep
> baseline at n=3 SCREENING, testing two priors with NO mainstream
> analog on the certified φ-budget base.
>
> **Source design space:** G9 combo winners (new group, 2026-05-30) —
> Phase-9d novelty-pocket wave from
> `audits/COMBINATIONS_RESEARCH/D_new_hypotheses.md` §3.4.
>
> **Implementation status (this repo):** `○ planned` — H09 certified;
> H22 has CIFAR-10 row (`sg_only_toroidal` 0.7805 on broken scaffold);
> H51 `✓ impl` only.

This document is the committee-grade design write-up for hypothesis
H88.

---

## 1. Motivation (≥ 100 words)

Agent C's literature survey identified 14-18 hypotheses in this
project occupying "novelty pockets" — design axes the mainstream
image-classification literature has not visited. Two strong members
of that list are H22 (toroidal closure, wrap-around boundary
condition for conv padding — no mainstream classification recipe
uses topological closure) and H51 (topological Betti loss, persistent-
homology-based auxiliary loss — tangentially explored in TDL but
not as a recipe add-on). Both target topologically-defined
inductive biases — H22 enforces a periodic boundary that respects
SO(2) translation symmetry on the wrapped manifold; H51 directly
regularises the feature-map's persistent homology toward target
Betti numbers, encouraging features that respect the data's
topological structure. The natural-/sacred motivation is that
nature's topological objects (DNA helices, cell membranes, neural
manifolds) all carry persistent-homology signatures that classic
networks ignore. The certified H09 phi_budget base is the
substrate; H22 and H51 are the novelty-pocket additions. This
hypothesis explicitly tests whether two untested novelty-pocket
priors deliver real signal or are decorative when added to a
known-positive base.

## 2. Formal hypothesis (≥ 50 words)

Because the certified H09 phi_budget allocator delivers +1.24 pp
on CIFAR-100 by re-allocating params across stages, **mechanism**-
wise adding H22's toroidal wrap-padding (giving each conv a
periodic boundary condition) should preserve the feature-map
geometry on the wrapped manifold and not damage representation
quality on canonical CIFAR-100 (where the wrap is a small
distortion at boundaries); per Hofer et al. 2017 (Deep Learning
with Topological Signatures) **mechanism**-wise adding H51's
persistent-homology Betti loss should drive feature-map β₀ toward
1 (a single connected component per class) and improve
generalisation; we predict Δ ∈ [+0.5, +1.5] pp at n=3 CIFAR-100
30-ep over `baseline_resnet20`.

## 3. Falsifier (≥ 30 words)

If at n=3 CIFAR-100 30-ep `combo_novelty_betti_torus` median top-1
is **less than** `sg_only_phi_budget` (the certified H09 base
alone, n=7 mean 0.5736) **−** 0.3 pp = 0.5706, the novelty-pocket
priors H22 + H51 do NOT add value on the certified base —
DECORATIVE per Rule 22 sci-critic.

## 4. Citations (≥ 80 words)

Hofer, Kwitt, Niethammer, Uhl 2017 NeurIPS 'Deep Learning with
Topological Signatures' (arXiv:1707.04041) — the persistent-
homology-as-loss reference; establishes the differentiable PH
machinery H51's Betti loss is built on.

Hoogeboom, Peters, Cohen, Welling 2018 NeurIPS 'HexaConv'
(arXiv:1803.02108) — establishes hex / non-square lattice convs
including the wrap-padding boundary condition that H22 implements;
the closest published cousin to H22's toroidal closure.

Tan, Le 2019 ICML 'EfficientNet: Rethinking Model Scaling for
Convolutional Neural Networks' (arXiv:1905.11946) — the
compound-scaling reference underpinning H09 phi_budget (certified
at n=7 in this project, +1.24 pp).

Edelsbrunner, Harer 2010 AMS 'Computational Topology: An
Introduction' (book) — the persistent-homology foundational
reference; β-numbers (β₀ = connected components, β₁ = loops, β₂ =
voids) are the targets of H51's regularisation.

Khrulkov, Oseledets 2018 ICML 'Geometry Score: A Method for
Comparing Generative Adversarial Networks' (arXiv:1802.02664) —
applies persistent-homology to network features; complementary
motivation for H51's feature-space regularisation.

## 5. Mechanism

### 5.1 CNN track

The hypothesis composes three primitives:

**Component 1 — H09 phi_budget (A7).** Per-stage 1:φ:φ² channel
widths under a fixed param budget. Same allocator as the certified
n=7 winner.

**Component 2 — H22 toroidal closure (A3 kernel).**
`src/nature_inspired_networks/priors.py:toroidal_pad`. Replaces
the default `Conv2d(padding='zeros')` with `padding_mode='circular'`,
giving each conv a periodic boundary on the (H, W) image grid.
**FLOPs:** ~0 % overhead (PyTorch's circular padding is native).
**Init:** unchanged.

**Component 3 — H51 Betti loss (A16 loss-aux).**
`src/nature_inspired_networks/betti_loss.py`. After every stage,
compute the persistent-homology β₀ of the spatial feature map
(treated as a point cloud in C-dim activation space, threshold
filtration) and add the loss term λ_PH × (β₀ − 1)² to the
classification loss. **Per-batch cost:** ~10-15 % per epoch
(differentiable PH library overhead). **Default:** λ_PH = 0.01
(must be tuned).

**Composition order:**
1. Build PhiBudgetNet (H09 base).
2. Replace all `Conv2d` to use `padding_mode='circular'` (H22).
3. Register a PH-loss hook on every stage output (H51).
4. Train with the standard cross-entropy + λ_PH × PH-loss
   composite objective.

Rule 23 forward-path check: H09 (A7) + H22 (A3) = 2 on-path
priors. H51 (A16 loss-aux) is gradient-only, off-path. Total
on-path = 2 (cap honored). B's matrix flags A7×A3 = F (both on
path, allowed pairwise); A7×A16, A3×A16 = O.

```python
def make_combo_novelty_betti_torus(cfg):
    model = PhiBudgetNet(num_classes=100, B=cfg.budget)
    apply_toroidal_padding_(model)  # H22
    register_betti_loss_hooks_(model, lambda_ph=0.01)  # H51
    return model
```

### 5.2 LLM track

Not applicable as primary test (CIFAR-only). H51 generalises to
attention head outputs in transformers; H22 toroidal closure is
LLM-track-irrelevant.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline_resnet20 (CIFAR-100 30-ep) | rationale |
|---|---|---|
| composite | [+0.004, +0.014] | top-1 lift dominates; latency cost minor |
| top-1 (CIFAR-100 30-ep, n=3) | [+0.5 pp, +1.5 pp] | H09(+1.24) + H22(modulating ~0) + H51(~0 to +0.5) |
| params | [+0 %, +0 %] | H09 preserves budget; H22/H51 add no params |
| FLOPs | [+10 %, +20 %] | Betti loss is the cost driver |
| GPU latency (batch=1) | [+10 %, +20 %] | dominated by PH library overhead |
| Betti collapse rate | [−0.20, −0.40] | strongly reduced by construction (the H51 regulariser directly drives β₀ → 1) |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** CIFAR-100 (standard split, RandomCrop + HFlip).
- **Architecture:** PhiBudgetNet (H09 base) with circular-padding
  Conv2d and Betti loss hooks on every stage.
- **Epochs / batch / precision / seeds:** 30 ep / 128 / bf16
  AMP / seeds 0..2 (n=3 SCREENING per Rule 28; promote to n=7
  EVALUATION on positive screen).
- **Optimizer:** AdamW LR 3e-3, WD 5e-4 (matching certified
  config).
- **Composite formula:** Rule-2 fingerprint preserved.
- **Run-script invocation:**
  ```powershell
  .\.venv\Scripts\python -m nature_inspired_networks.runner `
    --config configs\cifar100_combo_novelty_betti_torus.yaml `
    --tag combo_novelty_betti_torus --seed <0..2> `
    --root ideas\51_betti_loss\experiments\exp_combo_novelty_betti_torus\run
  ```
- **Wall-clock estimate:** ~25 min/seed × 3 = ~1.25 GPU-h on
  RTX 4090. Promotion to n=7 adds ~3 GPU-h. Total ~3 GPU-h if
  promoted.
- **Archive path:** `ideas/51_betti_loss/experiments/exp_combo_novelty_betti_torus/`.

### 7.2 Idea-targeted experiment (where this prior should SHINE)

The hypothesis predicts neutral-to-positive on canonical CIFAR-100
(no rotation distribution shift); the natural where-it-shines
dataset is **MedMNIST PathMNIST** (cell-image classification with
real topological feature signatures). A 12-ep MedMNIST smoke run
post-screen-success is filed as a future-work follow-up.

### 7.3 Cross-paradigm context (LLM track)

Not the primary test. H51 Betti loss in transformer context is
described in [H51's hypothesis doc](../g6_topological_bridging/H51_betti_loss.md).

## 8. Cross-references

- Parent design space row: NEW — Group G9, hypothesis H88.
- Parent winners: [H09 phi_budget](../g1_scaling_growth/H09_golden_param_budget.md).
- Parent novelty-pocket priors: [H22 toroidal_phi_closure](../g3_topologies_graphs/H22_toroidal_phi_closure.md), [H51 betti_loss](../g6_topological_bridging/H51_betti_loss.md).
- Research basis: `audits/COMBINATIONS_RESEARCH/[A](../../audits/COMBINATIONS_RESEARCH/A_empirical_stackability.md)` (H22 solo on broken scaffold, H51 untested), `[B](../../audits/COMBINATIONS_RESEARCH/B_theoretical_orthogonality.md)` §1 (A7×A3 = F allowed pairwise; A16 off-path), `[C](../../audits/COMBINATIONS_RESEARCH/C_literature_survey.md)` §6 (H22 + H51 in 14-18 novelty pocket).
- Synthesis doc: `audits/COMBINATIONS_RESEARCH/[D](../../audits/COMBINATIONS_RESEARCH/D_new_hypotheses.md)` §3.4.
- Composes with: H81 SIREN (post-screen-success Phase-9e); not with H10 phi_lr (universal antagonist).

## 9. Committee Q&A

**Q: Why isn't this just a re-packaging of Hofer 2017 (PH-loss) + circular-pad CNN?**

> Hofer 2017 demonstrates differentiable PH on dataset-level
> signatures (the loss target is a fixed distribution); H51
> applies PH at the per-stage feature-map level with the loss
> driven toward β₀ → 1 per class (a different formulation).
> Circular-padding CNN is a published primitive (HexaConv 2018),
> but combined with H09 phi_budget allocation and H51 Betti loss
> the composition is novel.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies the numeric falsifier: median top-1 < 0.5706 at
> n=3 CIFAR-100 30-ep refutes the claim. The decision is
> mechanical at the `metrics.json` write time. § 6 gives the
> +0.5 to +1.5 pp prediction window.

**Q: What if H22 toroidal padding hurts on CIFAR-100 (which doesn't wrap)?**

> The empirically-documented `sg_only_toroidal` Δ −6.73 pp on
> CIFAR-10 was on the NaturePrior-fib base scaffold (now known
> to be a broken substrate, per Agent A §1's verdict table). On
> the certified PhiBudgetNet base scaffold, toroidal padding has
> not been tested; the prediction is near-neutral (the wrap
> introduces a small distortion at the image edge, but
> downstream BN + ReLU absorb the small DC shift). § 3's
> falsifier captures the test cleanly.

**Q: Won't the differentiable PH library add unacceptable per-step overhead?**

> Estimated +10-15 % per epoch (~3 minutes total on a 22-minute
> 30-epoch run). The cost is real but bounded; if H88 lands
> +0.5 pp on the certified base, the publishable result justifies
> the compute investment. If the PH-loss overhead exceeds 25 %
> in practice we will downsample the loss to every 2nd stage.

**Q: How do we know the implementation is correct?**

> H22 `toroidal_pad` ships with mechanism tests (verifies wrap
> distance preservation on synthetic checkerboard inputs);
> H51 `betti_loss` ships with mechanism tests (verifies the loss
> drives β₀ down on a synthetic 2-component input). New test
> `tests/test_combo_novelty_betti_torus.py` (TO BE ADDED)
> verifies (a) all three components active, (b) wrap-padding
> preserved through the network, (c) PH-loss hook fires once
> per stage per batch, (d) H09 budget preserved post-padding-
> mode-swap.

## 10. Verification artifacts checklist

- [ ] `ideas/51_betti_loss/implementation.py` extended to expose `make_combo_novelty_betti_torus()` factory
- [ ] `tests/test_combo_novelty_betti_torus.py` ≥ 5 assertions
- [ ] `ideas/51_betti_loss/AUDIT.md` ≥ 3 self-found weaknesses (PH library compute overhead, λ_PH tuning sensitivity, H22 boundary distortion)
- [ ] `ideas/51_betti_loss/IMPROVEMENTS.md` records fixes
- [ ] `ideas/51_betti_loss/VERIFY.md` signed
- [ ] Experiment archive `ideas/51_betti_loss/experiments/exp_combo_novelty_betti_torus/`
- [ ] Per-seed verification {tests.txt, smoke.txt, gates.txt, reproduction.txt}
- [ ] On-screen-success: promote to n=7 EVALUATION (additional 3 GPU-h)
- [ ] Row added to `experiments/experiment_log.jsonl`
- [ ] Result reflected in `paper/FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-30 — Created from synthesis of A/B/C research; H88 design doc drafted as Phase-9d wave-1 rank-3 priority.
