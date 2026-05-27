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
