# H91 — `combo_domain_icosa_rotation` (Icosahedral-Equivariant Phi-Budget on Rotated CIFAR-100)

> **One-line claim:** On a rotation-augmented CIFAR-100 benchmark
> (random ±180° rotations), the combination of H09 phi_budget +
> H24 icosa-equivariant conv + H71 IcosaRoPE3D positional encoding
> delivers Δ ≥ +5 pp over an un-equivariant `pair_gm_pdw` baseline
> at n=7 — testing the project's only NOVEL+TESTABLE prior (H71) on
> the dataset where icosahedral equivariance should actually win.
>
> **Source design space:** G9 combo winners (new group, 2026-05-30) —
> Phase-9d domain-stretch wave from
> `audits/COMBINATIONS_RESEARCH/D_new_hypotheses.md` §3.7.
>
> **Implementation status (this repo):** `○ planned` — H09 certified;
> H24 + H71 are `✓ impl` (no CIFAR row yet). Rotated-CIFAR-100
> dataloader requires a new wrapper in `src/nature_inspired_networks/datasets.py`.

This document is the committee-grade design write-up for hypothesis
H91. Every section below is mandatory.

---

## 1. Motivation (≥ 100 words)

The project has spent 84 hypotheses' worth of design space on a
single dataset (CIFAR-10/100 canonical orientation) — a dataset
where rotational priors are EXPECTED to be roughly neutral because
the test set carries no rotation distribution shift. The project's
only NOVEL+TESTABLE sci-critic verdict (B §5) belongs to H71
IcosaRoPE3D, which has never been evaluated because no rotated /
3D benchmark sits in the pipeline. The lesson from Bello 2021
(C §1.16) is that recipes must be evaluated on the dataset where
their inductive bias is informative — not where it is decorative.
The natural extension test is therefore: **evaluate the icosahedral
priors on a rotated-CIFAR-100 benchmark, where 60-vertex icosahedral
group equivariance is the canonical inductive bias**. The
icosahedron has 60 rotational symmetries, the largest finite
rotation subgroup of SO(3) embeddable in 2D feature operations
without combinatorial blow-up. Mathematically: under random
SO(2) image rotations, a non-equivariant CNN has training-set
error inflated by the unaugmented portion of the rotation group,
while an icosa-equivariant CNN has matched error to the full
rotation orbit. Empirically (Cohen 2019): rotation-equivariant
CNNs deliver 5-20 pp lift on rotated benchmarks. The bet of H91
is that the project's certified H09 base + icosa-equivariant H24
+ IcosaRoPE3D H71 stack delivers an equivalent lift on
rotated-CIFAR-100.

## 2. Formal hypothesis (≥ 50 words)

Because random in-plane rotation introduces an SO(2) symmetry the
canonical ResNet-20 does not exploit, **mechanism**-wise a
network that explicitly factorises out the icosahedral rotation
group (H24's icosa-equivariant conv lift + H71's IcosaRoPE3D
positional bias) on top of the certified H09 phi_budget allocator
should match its rotation-orbit at train time and at test time;
per Cohen et al. 2019 (Spherical CNNs / Icosahedral CNNs) we
predict Δ top-1 ∈ [+5, +15] pp on rotated-CIFAR-100 30-ep vs the
canonical `pair_gm_pdw` baseline (which has no rotation prior).

## 3. Falsifier (≥ 30 words)

If at n=7 on rotated-CIFAR-100 (random ±180° rotations applied at
train + test) the `combo_domain_icosa_rotation` median top-1 is
**less than** `pair_gm_pdw` median + 0.0 pp, icosa equivariance
does NOT transfer to rotated-CIFAR-100. This is a strong falsifier
— rotation-equivariant CNNs are the canonical benchmark winners
on rotation tasks (Cohen 2019).

## 4. Citations (≥ 80 words)

Cohen, Geiger, Köhler, Welling 2018 ICLR 'Spherical CNNs'
(arXiv:1801.10130) — the foundational reference for spherical-/
icosahedral-equivariant CNNs; demonstrates the 5-20 pp lift
window on rotated spherical benchmarks that this hypothesis
attempts to reproduce on the rotated-CIFAR-100 adaptation.

Cohen, Weiler, Kicanaoglu, Welling 2019 ICML 'Gauge Equivariant
Convolutional Networks and the Icosahedral CNN' (arXiv:1902.04615)
— the icosa-specific equivariant-conv construction that H24's
implementation in `src/nature_inspired_networks/icosa.py`
parallels (60-vertex group lift + max-pool over orbit).

Su, Lu, Pan, Murtadha, Wen, Liu 2021 arXiv 'RoFormer: Enhanced
Transformer with Rotary Position Embedding' (arXiv:2104.09864) —
the RoPE reference; H71 IcosaRoPE3D generalises RoPE's per-axis
rotation matrices to icosahedral-3-axis rotations.

Tan, Le 2019 ICML 'EfficientNet: Rethinking Model Scaling for
Convolutional Neural Networks' (arXiv:1905.11946) — the
compound-scaling reference whose φ-tuned channel-width ratios
the H09 phi_budget allocator implements (certified at n=7 in
this project, +1.24 pp).

## 5. Mechanism

### 5.1 CNN track

The hypothesis composes three primitives plus a new dataset
wrapper:

**Component 1 — H09 phi_budget (A7).** Same allocator as the
certified n=7 winner; preserves param budget at 1:φ:φ² per-stage
widths.

**Component 2 — H24 icosa equivariant conv (A2 arch-block).**
`src/nature_inspired_networks/icosa.py`. Each conv layer lifts
its kernel to the 60-vertex icosahedral group orbit; the forward
pass computes 60 rotated copies and the response is max-pooled
over the orbit. **FLOPs:** 60× per conv (substantial — the
biggest cost in the stack); the orbit-max-pool aggregates back
to a single channel-feature map preserving spatial dimensions.
**Init:** He on the kernel, replicated across the 60 orbit
copies.

**Component 3 — H71 IcosaRoPE3D positional encoding (A5+A6).**
`src/nature_inspired_networks/hybrid_icosa_rope.py`. Applies
icosa-vertex-aligned 3D rotary positional embeddings to the
feature-map's spatial coordinates. **FLOPs:** O(C × H × W) per
encoding, applied once at the input stage; trivial cost.
**Init:** fixed frequencies derived from icosa-vertex angles.

**Component 4 — Rotated-CIFAR-100 dataloader.** A new wrapper
`RotatedCIFAR100(angle_range=(-180,180), train=True)` in
`src/nature_inspired_networks/datasets.py`. Train-time: random
rotation per sample. Test-time: random rotation per sample
(NOT the same rotation as train — tests true rotation
equivariance, not "memorised rotations"). Implementation uses
`torchvision.transforms.RandomRotation(180, expand=False)`.

**Composition order:**
1. Build PhiBudgetNet (H09 base).
2. Replace `Conv2d` layers with `IcosaEquivariantConv2d` from
   H24 module.
3. Apply IcosaRoPE3D positional embedding at the network input.
4. Train on rotated-CIFAR-100 with the standard runner.

Rule 23 forward-path check: H09 (A7) + H24 (A2) = 2 on-path
priors. H71 (A5/A6 PE) is applied at the input only, off the
conv-block forward path. Total on-path = 2 (cap honored).

```python
def make_combo_domain_icosa(cfg):
    model = PhiBudgetNet(num_classes=100, B=cfg.budget)
    model = replace_conv_with_icosa_(model)  # H24
    model = add_icosa_rope_input_(model)  # H71
    return model
```

### 5.2 LLM track

Not applicable as primary test (rotated-CIFAR-100 is image-only).
The H71 IcosaRoPE3D is itself LLM-track-friendly when applied to
3D point-cloud transformers; that is a separate future-work
direction.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. rotated-CIFAR-100 `pair_gm_pdw` baseline | rationale |
|---|---|---|
| composite | [+0.04, +0.12] | top-1 lift dominates |
| top-1 (rotated-CIFAR-100 30-ep, n=7) | [+5 pp, +15 pp] | Cohen 2019 reports 5-20 pp on rotated benchmarks |
| params | [+0 %, +5 %] | H24 adds 60 orbit copies but most weights are tied |
| FLOPs | [+100 %, +200 %] | H24 orbit-conv is the cost driver |
| GPU latency (batch=1) | [+50 %, +120 %] | orbit-conv FLOPs amortised over batching |
| rotation-equivariance err | [−0.05, −0.20] | strongly reduced by construction |
| param budget preserved | yes | H09 + H24's tied weights |
| Betti collapse rate | [no change] | no topological prior |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** Rotated-CIFAR-100. Generated on-the-fly by
  `RotatedCIFAR100(angle_range=(-180, 180))` wrapper. 50 K train,
  10 K test (same split as canonical CIFAR-100).
- **Architecture:** PhiBudgetNet (H09 base) with all Conv2d
  layers replaced by `IcosaEquivariantConv2d`, IcosaRoPE3D
  applied at input.
- **Epochs / batch / precision / seeds:** 30 ep / 128 / bf16
  AMP / seeds 0..6 (n=7 EVALUATION).
- **Optimizer:** AdamW LR 3e-3, WD 5e-4 (matching the certified
  `pair_gm_pdw` config — held fixed for direct comparison).
- **Composite formula:** Rule-2 fingerprint preserved.
- **Run-script invocation:**
  ```powershell
  .\.venv\Scripts\python -m nature_inspired_networks.runner `
    --config configs\cifar100rot_combo_domain_icosa.yaml `
    --tag combo_domain_icosa_rotation --seed <0..6> `
    --root ideas\24_icosa_phi_equivariant\experiments\exp_combo_domain_icosa_rotation\run
  ```
- **Wall-clock estimate:** ~30 min/seed × 7 = ~3.5 GPU-h. Plus
  `baseline_resnet20` and `pair_gm_pdw` on rotated-CIFAR-100 at
  n=3 each (~2 GPU-h). Total ~5.5 GPU-h.
- **Archive path:** `ideas/24_icosa_phi_equivariant/experiments/exp_combo_domain_icosa_rotation/`.

### 7.2 Idea-targeted experiment (where this prior should SHINE)

The primary experiment IS the idea-targeted experiment — rotated-
CIFAR-100 is exactly the dataset where icosahedral equivariance
should win. Confirmatory secondary experiments: (a) IcoMNIST
(spherical MNIST on icosa unfolding, Cohen 2019); (b) canonical
CIFAR-100 (where equivariance should be NEUTRAL — sanity-check
that the prior doesn't HURT when its inductive bias is
unrewarded).

### 7.3 Cross-paradigm context (LLM track)

H71 IcosaRoPE3D's natural LLM-track home is the point-cloud
transformer / molecular-property regression — out of v0.1
scope. Filed in `ideas/71_icosa_rope_3d/AUDIT.md`.

## 8. Cross-references

- Parent design space row: NEW — Group G9, hypothesis H91.
- Parent winners: [H09 phi_budget](../g1_scaling_growth/H09_golden_param_budget.md) (certified n=7).
- Parent priors (untested on rotation): [H24 icosahedral](../g3_topologies_graphs/H24_icosa_equivariant.md), [H71 IcosaRoPE3D](../g7_cross_paradigm_hybrids/H71_icosa_rope_3d.md).
- Research basis: `audits/COMBINATIONS_RESEARCH/[A](../../audits/COMBINATIONS_RESEARCH/A_empirical_stackability.md)` (no existing combination tested on rotation), `[B](../../audits/COMBINATIONS_RESEARCH/B_theoretical_orthogonality.md)` §5 (H71 NOVEL+TESTABLE), `[C](../../audits/COMBINATIONS_RESEARCH/C_literature_survey.md)` §6 (icosa equivariance is a published GDL pocket).
- Synthesis doc: `audits/COMBINATIONS_RESEARCH/[D](../../audits/COMBINATIONS_RESEARCH/D_new_hypotheses.md)` §3.7.
- Composes with: H51 Betti loss (if topological signal also relevant on rotated data — future Phase-9e).
- Conflicts with: H10 phi_lr (A §3 universal antagonist).

## 9. Committee Q&A

**Q: Why isn't this just a re-packaging of Cohen 2019 Icosahedral CNNs?**

> Cohen 2019 demonstrates icosa equivariance on Spherical MNIST and
> Climate Patterns benchmarks; H91 evaluates on rotated-CIFAR-100
> (an in-plane 2D rotation task) at fixed ResNet-20-class budget
> with the H09 phi-budget allocator and H71 IcosaRoPE3D positional
> encoding. The combination is new and the test substrate is new;
> the lift PATTERN (5-20 pp on rotated benchmarks) is what Cohen
> establishes as the literature anchor for this kind of result.

**Q: How is this falsifiable rather than aesthetic?**

> § 3: median top-1 ≤ `pair_gm_pdw` on rotated-CIFAR-100 + 0.0 pp
> refutes the claim. The numeric is the experiment archive's
> `metrics.json`. § 6 gives the +5 to +15 pp prediction window.

**Q: What if the prior helps on rotated CIFAR-100 but hurts on canonical CIFAR-100?**

> The hypothesis is SCOPED to rotated-CIFAR-100. § 7.2's
> confirmatory secondary experiment on canonical CIFAR-100
> directly tests the "does the equivariance prior HURT when its
> inductive bias is unrewarded" question. Both results are
> part of the deliverable; we expect a small negative or zero
> on canonical CIFAR-100 (the orbit-pool max-aggregation has
> a documented information-loss penalty on un-rotated inputs).

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> The H24 icosa prior has not yet been tested in this project on
> any dataset; the `sg_only_group` (C4 max-pool proxy) was a
> degenerate 4-rotation orbit, NOT the 60-vertex icosa group, and
> the test was on canonical CIFAR-10 where C4 equivariance is
> unrewarded. H91 corrects both: full 60-vertex orbit + dataset
> with active rotation symmetry.

**Q: Won't the 60× FLOP cost make this unaffordable?**

> Yes, the orbit-conv is the dominant cost at +100-200 % FLOPs vs
> the un-equivariant baseline. Wall-clock estimate (§ 7.1) bounds
> the cost at ~30 min/seed on RTX 4090, total 5.5 GPU-h. The cost
> is real but bounded; if H91 lands +5 pp on rotated-CIFAR-100,
> the publishable result justifies the compute investment.

**Q: How do we know the implementation is correct?**

> H24's `icosa.py` ships with mechanism-verifying tests
> (`tests/test_icosa.py`): 60-vertex orbit completeness, max-pool
> aggregation, gauge-equivariance check on synthetic rotations.
> The new test `tests/test_combo_domain_icosa_rotation.py` (TO
> BE ADDED) verifies (a) rotated-input → identical features
> (within tolerance) on the orbit-equivariant network, (b) H09
> budget preserved post-icosa-substitution, (c) IcosaRoPE3D
> applied at input only.

## 10. Verification artifacts checklist

- [ ] `src/nature_inspired_networks/datasets.py` extended with `RotatedCIFAR100` wrapper
- [ ] `ideas/24_icosa_phi_equivariant/implementation.py` extended to expose `make_combo_domain_icosa()` factory
- [ ] `tests/test_combo_domain_icosa_rotation.py` ≥ 5 assertions
- [ ] `ideas/24_icosa_phi_equivariant/AUDIT.md` ≥ 3 self-found weaknesses (FLOP overhead, gauge convention, IcosaRoPE3D 2D ↔ 3D interpretation)
- [ ] `ideas/24_icosa_phi_equivariant/IMPROVEMENTS.md` records mechanism-coupling fixes
- [ ] `ideas/24_icosa_phi_equivariant/VERIFY.md` signed
- [ ] Experiment archive `ideas/24_icosa_phi_equivariant/experiments/exp_combo_domain_icosa_rotation/`
- [ ] Per-seed verification {tests.txt, smoke.txt, gates.txt, reproduction.txt}
- [ ] Confirmatory secondary on canonical CIFAR-100 (n=3 sanity)
- [ ] Row added to `experiments/experiment_log.jsonl`
- [ ] Result reflected in `paper/FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-30 — Created from synthesis of A/B/C research; H91 design doc drafted as Phase-9d wave-1 rank-2 priority.
