# H76 — TetrahedralDualPathBlock (dual complementary C4 reductions + learnable convex merge)

> **One-line claim:** A residual block with two complementary C4
> group-equivariant convolutional paths — one with max-pool orbit
> reduction, one with mean-pool — fused by a learnable convex
> coefficient `beta ∈ [0,1]` recovers within ±0.3 pp of the better
> single-reduction path on CIFAR-10 (12 ep) while letting the network
> *learn* the orientation-aggregation mixture instead of having it
> hard-coded, and never underperforms the worse single path.
>
> **Source design space:** G8 esoteric extensions; concretely
> implementable bonus idea. Esoteric origin (acknowledged once): the
> "Merkaba" star-tetrahedron figure of two interpenetrating tetrahedra
> of opposite polarity motivated pairing two *complementary* orbit
> reductions and letting a balanced merge decide their mixture.
>
> **Implementation status (this repo):** `● implemented` —
> `src/nature_inspired_networks/tetra_dualpath.py` +
> `tests/test_tetra_dualpath.py` (7 assertions, green).

---

## 1. Motivation (≥ 100 words)

The H58 follow-up established a robust, slightly counter-intuitive
finding on this repo: over a C4 rotation orbit, max-pool reduction
beats mean-pool by 4-6 pp top-1 on CIFAR-10, because max acts as a
soft argmax that preserves the strongest orientation response at every
spatial location while mean dilutes discriminative features (see
`priors.GroupConv2d` docstring and `FINDINGS.md`). That result fixed a
single reduction operator for all downstream blocks. But "max is
better on average" is not the same as "max is better everywhere": some
channels/locations may genuinely benefit from orientation averaging
(e.g. textures, low-frequency backgrounds), and committing globally to
one pooling discards that flexibility. Group-equivariant CNNs (Cohen &
Welling 2016) leave the orbit-pooling choice as a fixed design
hyper-parameter; multi-branch residual designs (ResNeXt, Xie et al.
2017) show that *parallel* complementary transformations of the same
input, recombined, outperform a single branch at matched compute. This
hypothesis tests whether a learnable convex blend of the two C4
reductions — the strongest-orientation path and the averaged-orientation
path — lets the network adaptively interpolate between selectivity and
invariance per layer, recovering the best of both without a hand-tuned
switch.

## 2. Formal hypothesis (≥ 50 words)

Because the max and mean C4-orbit reductions encode complementary
inductive biases (selectivity vs. invariance) and the H58 evidence
shows their optimum is data-dependent rather than universal,
**mechanism**-wise a residual block that runs both reductions in
parallel over a shared input and fuses them with a single learnable
convex coefficient `beta = sigmoid(beta_raw)` as `beta·max + (1-beta)·mean`
should converge to a mixture that is at least as good as the better
single path, *because* gradient descent can drive `beta → 1` to recover
pure-max whenever max dominates and toward 0 otherwise, so the block
strictly dominates the worse fixed reduction in expectation.

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on CIFAR-10 (12 ep), the dual-path block scores
**more than 0.3 pp below** the better single-reduction baseline
(pure-max `GroupConv2d`), **OR** if the learned `beta` collapses to a
degenerate boundary (`beta < 0.02` or `beta > 0.98`) on every seed —
indicating the second path contributes nothing and the merge is pure
overhead — this hypothesis is **DISCARDED**. The merge coefficient is
a single scalar in v0; per-channel `beta` is the targeted follow-up.

## 4. Citations (Citation Rigor format, ≥ 40 words)

```
Cohen, Welling 2016 ICML 'Group Equivariant Convolutional Networks'
(arXiv:1602.07576) -- the C4/D4 group-conv formulation whose orbit
pooling step (max vs. mean) this block runs in parallel and blends;
the orbit-reduction operator is exactly the design choice we make
learnable.

Xie, Girshick, Dollar, Tu, He 2017 CVPR 'Aggregated Residual
Transformations for Deep Neural Networks (ResNeXt)'
(arXiv:1611.05431) -- precedent that parallel complementary
transformations of a shared input, summed back, beat a single branch
at matched FLOPs; the convex merge is a 2-branch instance with a
learned mixing weight.

Weiler, Cesa 2019 NeurIPS 'General E(2)-Equivariant Steerable CNNs'
(arXiv:1911.08251) -- general steerable framework that treats the
group-pooling as a tunable layer, supporting the claim that the
reduction operator is a learnable degree of freedom rather than a
fixed constant.
```

## 5. Mechanism

### 5.1 CNN track

Two `GroupConv2d(in, out, k, group="c4")` instances share the block
input `x` of shape `(B, C, H, W)`. Path A uses `reduce="max"`, path B
uses `reduce="mean"`; both emit `(B, out, H', W')`. A single scalar
parameter `beta_raw` (init `logit(0.5)=0`) is squashed by a sigmoid and
convex-combines the paths: `out = beta·A + (1-beta)·B`. When
`in==out` and `stride==1`, an identity skip is added so the block is a
drop-in residual block; otherwise the merged output is returned
directly. Param overhead vs. a single C4 path: exactly one extra conv
kernel (the second path) plus one scalar — the second path is the cost,
not the merge. FLOPs ≈ 2× a single `GroupConv2d` (both orbits are
convolved). The block preserves `(B, C, H, W)` so it slots into any
ResNet stage.

### 5.2 LLM track

Not the primary axis (this is a spatial-equivariance prior). The
analogue for a Transformer would be two attention reductions over a
rotated/permuted key orbit (max-similarity vs. mean-similarity) blended
by a learned `beta`; out of scope for v0, noted for completeness only.

## 6. Predicted Δ on CIFAR-10 (pre-registered)

| metric | Δ vs. pure-max baseline | rationale |
|---|---|---|
| top-1 (12 ep) | [-0.3, +1.0] pp | learns to match or beat best fixed path |
| learned beta (final) | [0.55, 0.85] | leans toward max but uses mean |
| params | [+0.0 %, +0.1 %] | one extra scalar + second conv kernel |
| FLOPs | [+90 %, +100 %] | second orbit convolution |

## 7. Experimental protocol

1. **Unit tests (Phase 0):** `tests/test_tetra_dualpath.py` — 7
   assertions: forward shape preserved, `beta=1` recovers pure max
   path, `beta=0` recovers pure mean path, the two reductions produce
   different activations on shared weights, gradient flows to
   `beta_raw`, residual skip auto-disables on shape change, `beta_init`
   respected. All green.
2. **CIFAR-10 SOTA smoke (Phase 1):** unchanged ResNet-20 baseline must
   hit ≥ 80 % @ 12 ep before any variant (Rule 13).
3. **CIFAR-10 hypothesis smoke (Phase 2):** one sweep row flips a single
   `tetra_dualpath` flag swapping the residual block for
   `TetrahedralDualPathBlock`; compare to the pure-max `GroupConv2d`
   row. Log final `beta` to `history.json`.

## 8. Cross-references

- Builds directly on `priors.GroupConv2d` and the H58 max-vs-mean
  finding in `FINDINGS.md`.
- Composes with H24 (Platonic equivariance) — the dual-path block is a
  drop-in for any C4 stage.

## 9. Verification checklist

- [x] `src/nature_inspired_networks/tetra_dualpath.py` exists, imports
      `GroupConv2d` from shared primitives (Rule 14, no code dup).
- [x] `tests/test_tetra_dualpath.py` ≥ 4 assertions (has 7), green via
      the `if __name__ == "__main__"` runner.
- [x] `beta=0` / `beta=1` mechanism asserted; gradient-to-beta asserted;
      different-activation (edge) asserted; residual fallback (regression)
      asserted.
- [x] `# TODO runner wiring:` block describes flag integration without
      touching `models.py` / `runner.py`.
- [ ] Phase-2 CIFAR-10 sweep row (lead wires the flag).

## 10. Status journal

- 2026-05-27 — Implemented + tested (7/7 green) as a standalone G8
  primitive. Runner wiring deferred to lead per task scope.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G8 (elite-research-scientist critic). Critiquing
the IDEA, not the implementation (audit at `audits/G8_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)

MED. The neutral recast — a learnable convex mixture of two parallel
group-conv branches differing only in their orbit-reduction operator
— is a vanilla 2-branch architecture (Xie, Girshick, Dollar, Tu, He
2017 CVPR 'Aggregated Residual Transformations for Deep Neural
Networks' (arXiv:1611.05431) — same template). The plausibility is
capped, however, by the H58 empirical finding *on this very repo*
that mean-pool C4 attains 65.38% vs. max-pool C4 at 69.84% — a
~4.5 pp gap. A convex merge has no mechanism to *exceed* the better
endpoint; at best it ties max (with `beta→1`) at 2× FLOPs.

### Mechanism scrutiny — does the NEUTRAL recast match the cited real technique?

The recast is a strict subset of ResNeXt-style multi-branch
aggregation (Xie 2017 arXiv:1611.05431) with cardinality=2 and a
learned mixing scalar. It also reduces, for `beta=1`, exactly to
existing max-pool group conv (Cohen, Welling 2016 ICML 'Group
Equivariant Convolutional Networks' (arXiv:1602.07576) — orbit
pooling). So the technique is real; the question is whether the
*Merkaba-motivated* β=0.5 init is principled. The author concedes
the optimum is data-dependent (§1), but the dual-tetrahedra "opposite
polarity" framing offers no defensible reason to start at the
midpoint of an empirically asymmetric landscape.

### Does the esoteric origin contaminate the implementation or framing?

YES, mildly — through the β=0.5 init choice. If the author had run
H58 first (they did) and *still* initialised β at the symmetric
midpoint rather than `logit(0.85)` (biased toward the known-better
max path), that is the Merkaba "dual-polarity balance" framing
leaking into the optimiser's starting point. A data-driven design
would init β toward the better path and let descent shift it down if
mean-pool helps locally.

### Confounds (≥2)

1. **Doubled FLOPs.** Path B doubles the conv work of the block; any
   accuracy parity vs. pure-max conflates the merge benefit with the
   matched-compute advantage already documented for any 2-branch
   block (ResNeXt cardinality scaling, Xie 2017 arXiv:1611.05431).
   The fair baseline is a 2× width pure-max GroupConv2d, not a 1×
   pure-max baseline.
2. **Initialisation bias.** β init at 0.5 starts the network at a
   point H58 already showed is suboptimal. A pure-max baseline
   begins at the better endpoint; the dual-path must *recover* it,
   which adds training-trajectory noise.
3. **Tied weights vs. independent.** The two paths use independent
   GroupConv2d kernels; the merge benefit could come from kernel
   ensembling, not from the max/mean complementarity claim.

### Numerology / specificity check

The "dual tetrahedra of opposite polarity" framing is decorative.
Group-conv orbit reduction has two canonical operators (max, mean);
a learnable mixture is the obvious generalisation regardless of any
Merkaba motif. The "12 vertices of a stellated octahedron" or
similar Merkaba arithmetic does not appear anywhere in the
mechanism — only the integer 2 (two paths) does, and 2 is not
specific to the Merkaba.

### Literature precedent — was the neutral recast already known?

YES. Adaptive pooling / mixed-pooling has been studied repeatedly:
Lee, Gallagher, Tu 2016 AISTATS 'Generalizing Pooling Functions in
Convolutional Neural Networks: Mixed, Gated, and Tree' (arXiv:
1509.08985) — *exactly* a learnable convex mixture of max and mean
pooling, predating the Merkaba framing by a decade. This is not
novel; it is mixed-pooling applied to a C4 orbit instead of a
spatial window.

### Expected effect size (90% CI a priori)

CIFAR-10 12-ep top-1 vs. pure-max C4 GroupConv2d: [−0.6 pp, +0.2 pp]
(re-centred down from the doc's [-0.3, +1.0] because the H58 gap
makes a positive Δ implausible without per-channel β, and the doubled
FLOPs do not benefit a single block). Learned β at convergence:
expected median ≈ 0.85–0.95 (close to pure-max).

### Minimum-distinguishing experiment

Three-arm seed-median 12-ep CIFAR-10: (a) pure-max GroupConv2d at 1×
width, (b) pure-max GroupConv2d at 2× channels (matched FLOPs), (c)
the dual-path block at β-init `logit(0.85)`. The hypothesis is
genuinely live only if (c) > (b) at 3-seed median. If (c) ≤ (b), the
"merge" is just ResNeXt-style ensembling and the Merkaba framing
adds nothing.

### Verdict

DERIVATIVE+TESTABLE — mixed-pooling (Lee 2016 arXiv:1509.08985)
re-applied to C4 orbits; falsifier is sharp and the matched-FLOPs
control disambiguates the merge claim from ResNeXt cardinality.
