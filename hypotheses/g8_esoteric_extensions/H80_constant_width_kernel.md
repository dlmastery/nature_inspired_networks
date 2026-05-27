# H80 — ConstantWidthKernelConv (Reuleaux-masked isotropic convolution)

> **One-line claim:** Masking a k×k convolution kernel with a
> constant-width (Reuleaux-triangle) soft mask, suppressing the corner
> taps, yields a more isotropic receptive field that holds CIFAR-10
> top-1 within ±0.5 pp of a same-size square kernel at **equal trainable
> parameter count**, and improves robustness to input rotation (lower
> top-1 drop under ±15° test-time rotation) by ≥1 pp.
>
> **Source design space:** G8 esoteric extensions; concretely
> implementable bonus idea. Esoteric origin (acknowledged once): the
> Reuleaux triangle as a "sacred constant-width form" motivated using a
> constant-width support instead of either a square or a plain disk.
>
> **Implementation status (this repo):** `● implemented` —
> `src/nature_inspired_networks/constant_width_kernel.py` +
> `tests/test_constant_width_kernel.py` (6 assertions, green).

---

## 1. Motivation (≥ 100 words)

A square k×k convolution kernel treats the four corner taps — which sit
at distance `(k-1)/√2·... ` farther from the centre and only on the
diagonal axes — with the same status as the axis-aligned neighbours.
This bakes a subtle anisotropy into every conv layer: the effective
receptive field is square, so diagonal structure is weighted more than a
rotationally symmetric operator would. Classical signal processing
prefers isotropic kernels (Gaussians, disks) precisely to avoid
orientation bias, and this repo already exploits shaped-kernel masking
for the hexagonal prior (`hex_kernel_mask`), which suppresses corner
taps to emulate a honeycomb lattice. A Reuleaux triangle is the
canonical *non-circular* curve of constant width: the intersection of
three disks centred at the vertices of an equilateral triangle, each
with radius equal to the side length. Masking the kernel to this
support gives an isotropic-leaning receptive field (corners gone) that,
unlike a plain disk, retains the constant-width geometric property. The
hypothesis is that removing corner anisotropy at no extra parameter cost
(masked taps are held at zero) is at worst neutral on upright CIFAR-10
and beneficial under rotation, where the square kernel's diagonal bias
hurts most.

## 2. Formal hypothesis (≥ 50 words)

Because a square kernel's corner taps inject a diagonal-axis anisotropy
into the receptive field, **mechanism**-wise replacing the kernel
support with a constant-width Reuleaux mask (corners → 0, centre → 1)
removes that anisotropy *because* the masked support is closer to
rotationally symmetric, so the layer's response varies less with input
orientation; we therefore predict equal upright CIFAR-10 accuracy at
equal trainable params and a measurably smaller top-1 drop under small
test-time rotations than the square-kernel baseline.

## 3. Falsifier (≥ 30 words)

If, at 3-seed median, the Reuleaux-masked conv scores **more than
0.5 pp below** the same-size square-kernel baseline on upright CIFAR-10
(12 ep), **OR** if its top-1 drop under ±15° test-time rotation is **not
smaller** (within noise) than the square kernel's drop, the isotropy
benefit is absent and the hypothesis is **DISCARDED**. The mask is a
fixed (non-learned) buffer in v0; a learnable-radius mask is the
targeted follow-up.

## 4. Citations (Citation Rigor format, ≥ 40 words)

```
Cohen, Welling 2016 ICML 'Group Equivariant Convolutional Networks'
(arXiv:1602.07576) -- frames orientation bias as an inductive-bias
choice; a constant-width (near-isotropic) kernel support reduces the
orientation anisotropy that a square kernel introduces, complementary
to explicit group equivariance.

Hoogeboom, Peters, Cohen, Welling 2018 'HexaConv'
(arXiv:1803.02108) -- precedent for masking a square kernel into a
non-square (hexagonal) support to obtain a more isotropic lattice; the
Reuleaux mask is the same masking technique with a constant-width
support instead of a hexagon.

Worrall, Garbin, Turmukhambetov, Brostow 2017 CVPR 'Harmonic
Networks: Deep Translation and Rotation Equivariance'
(arXiv:1612.04642) -- motivates rotationally-symmetric filter supports
for rotation robustness, the property the constant-width mask targets.
```

## 5. Mechanism

### 5.1 CNN track

`reuleaux_mask(k)` builds the intersection of three disks (radius = the
inscribed equilateral triangle's side length, centred at the grid
centre) and normalises the peak to 1.0. A soft variant uses a sigmoid
of the signed distance to each disk boundary so boundary gradients are
smooth; a hard `{0,1}` variant is available. `ConstantWidthConv2d`
holds an `nn.Conv2d` (padding handled manually) and multiplies the mask
into the weight at every forward pass, so masked taps stay at exactly
zero and gradients flow only to in-support taps. Input `(B, C, H, W)` →
output `(B, C_out, H, W)` at `stride=1, padding=k//2`. Trainable
parameter count is **identical** to a same-size `Conv2d` (the mask is a
buffer); the masked taps simply contribute nothing. For `k=5` the
support covers ~10 of 25 cells (between a tiny disk and the full
square).

### 5.2 LLM track

Not the primary axis. A 1-D analogue (constant-width masked depthwise
conv in a conv-augmented Transformer block, e.g. Conformer) is possible
but out of scope for v0.

## 6. Predicted Δ on CIFAR-10 (pre-registered)

| metric | Δ vs. square-kernel baseline | rationale |
|---|---|---|
| top-1 upright (12 ep) | [-0.5, +0.5] pp | corner taps carry little upright signal |
| top-1 drop @ ±15° rot | [-2.0, -0.5] pp smaller | isotropic support |
| trainable params | [0 %, 0 %] | mask is a buffer |
| FLOPs | [-30 %, 0 %] | masked taps still convolved (dense), so ~0 |

## 7. Experimental protocol

1. **Unit tests (Phase 0):** `tests/test_constant_width_kernel.py` — 6
   assertions: mask shape `(k,k)` in `[0,1]`; non-square (corners ~0,
   centre 1); masked forward preserves shape at stride 1, padding k//2,
   with corner taps zeroed; coverage strictly between a tiny disk and
   the full square; gradient flows to in-support taps and is smaller at
   corners; trainable param count equals a plain `Conv2d`. All green.
2. **CIFAR-10 SOTA smoke (Phase 1):** unchanged baseline ≥ 80 % @ 12 ep.
3. **CIFAR-10 hypothesis smoke (Phase 2):** one sweep row flips a
   `constant_width_kernel` flag swapping stage convs (k=5) for
   `ConstantWidthConv2d`; report upright accuracy and a rotated-test-set
   accuracy column.

## 8. Cross-references

- Same masking pattern as `priors.hex_kernel_mask` / `HexConv2d`
  (Rule 14: shaped-kernel masking lives in shared primitives).
- Complements H24 (Platonic equivariance) and H21 (hex lattice) on the
  isotropy axis.

## 9. Verification checklist

- [x] `src/nature_inspired_networks/constant_width_kernel.py` exists.
- [x] `reuleaux_mask(k)` returns `(k,k)` in `[0,1]`, corners ~0, centre 1.
- [x] `tests/test_constant_width_kernel.py` ≥ 4 assertions (has 6), green.
- [x] Shape (mechanism), non-square (mechanism), coverage bounds (edge),
      gradient + equal-param (regression) asserted.
- [x] `# TODO runner wiring:` block describes flag integration without
      touching `models.py` / `blocks.py`.
- [ ] Phase-2 CIFAR-10 sweep row (lead wires the flag).

## 10. Status journal

- 2026-05-27 — Implemented + tested (6/6 green) as a standalone G8
  primitive. Hard and soft mask variants both validated.
