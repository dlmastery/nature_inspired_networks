"""H80 — ConstantWidthKernelConv.

A convolution whose k×k kernel is masked by a CONSTANT-WIDTH
(Reuleaux-triangle-style) soft mask, giving a more isotropic receptive
field than a bare square kernel — closer to a disk, but with the
constant-width property of a Reuleaux triangle (a convex shape whose
width is the same in every direction).

Neutral framing
---------------
A square k×k kernel weights the four corners as heavily as the axis-aligned
neighbours, biasing the receptive field toward the diagonal axes. Masking the
kernel toward an isotropic support (a disk, or here a Reuleaux triangle) removes
that anisotropy. A Reuleaux triangle is the canonical non-circular
curve of constant width: the intersection of three disks, each centred at one
vertex of an equilateral triangle with radius equal to the triangle's side
length. The esoteric origin (the "constant-width form" as a sacred shape) is
acknowledged only as the source intuition; the operational object is a standard
masked convolution, mathematically identical to the masking used by hex-kernel
and other shaped-kernel convs already in this repo.

Functions / classes
--------------------
- :func:`reuleaux_mask(k)` — return a ``(k, k)`` float mask in ``[0, 1]``
  approximating a Reuleaux triangle centred on the grid.
- :class:`ConstantWidthConv2d` — a ``Conv2d`` whose effective weight is
  ``weight * mask`` at every forward pass.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def reuleaux_mask(k: int = 5, soft: bool = True, sharpness: float = 4.0) -> torch.Tensor:
    """Return a ``(k, k)`` Reuleaux-triangle mask in ``[0, 1]``.

    A Reuleaux triangle is the intersection of three disks. Each disk is
    centred at a vertex of an equilateral triangle inscribed so the figure is
    centred on the kernel grid, and each disk has radius equal to the triangle's
    side length ``s``. A grid cell is inside the Reuleaux triangle iff it lies
    inside ALL THREE disks.

    Parameters
    ----------
    k : int
        Kernel size (k×k). Must be ``>= 3``.
    soft : bool
        If ``True`` (default), return a soft mask in ``[0, 1]`` whose value
        falls off smoothly across each disk boundary (a sigmoid of the signed
        distance), so gradients near the boundary are well-behaved. If
        ``False``, return a hard ``{0, 1}`` mask.
    sharpness : float
        Controls the soft-boundary steepness (larger = closer to hard). Only
        used when ``soft=True``.

    Returns
    -------
    torch.Tensor
        ``(k, k)`` mask, centre value ``== 1`` (or ~1 for soft), corner values
        ``0`` (or ~0), normalised so the maximum is exactly ``1.0``.
    """
    assert k >= 3, "reuleaux_mask requires k >= 3"
    # Grid coordinates centred at the kernel centre, in "cells".
    c = (k - 1) / 2.0
    ys = torch.arange(k, dtype=torch.float32) - c
    xs = torch.arange(k, dtype=torch.float32) - c
    Y, X = torch.meshgrid(ys, xs, indexing="ij")

    # Equilateral triangle inscribed in the grid. Place its centroid at the
    # grid centre. Choose side length s so the figure spans the kernel: the
    # circumradius R = s / sqrt(3); we set R so the vertices reach the grid
    # half-extent c, giving s = c * sqrt(3). Each Reuleaux disk has radius s.
    R = c
    s = R * math.sqrt(3.0)  # side length == disk radius

    # Three vertices at 120-degree spacing (point-up triangle).
    verts = []
    for a_deg in (90.0, 210.0, 330.0):
        a = math.radians(a_deg)
        verts.append((R * math.cos(a), R * math.sin(a)))  # (vx, vy)

    # For each vertex, inside-disk indicator = (dist to vertex) <= s.
    # Reuleaux interior = intersection of all three disks.
    if soft:
        mask = torch.ones(k, k)
        for (vx, vy) in verts:
            dist = torch.sqrt((X - vx) ** 2 + (Y - vy) ** 2)
            # signed margin: positive inside the disk, negative outside.
            margin = s - dist
            disk = torch.sigmoid(sharpness * margin)
            mask = mask * disk
    else:
        inside = torch.ones(k, k, dtype=torch.bool)
        for (vx, vy) in verts:
            dist = torch.sqrt((X - vx) ** 2 + (Y - vy) ** 2)
            inside = inside & (dist <= s + 1e-6)
        mask = inside.float()

    # Normalise so the peak (centre) is exactly 1.0.
    peak = mask.max()
    if peak > 0:
        mask = mask / peak
    return mask


class ConstantWidthConv2d(nn.Module):
    """Conv2d whose effective weight is ``weight * reuleaux_mask`` each forward.

    Drop-in replacement for ``nn.Conv2d`` with the same constructor signature.
    The mask is registered as a non-trainable buffer (it moves with ``.to()``
    and is saved in ``state_dict``). Multiplying the mask in at every forward
    (rather than once at init) keeps gradients flowing to all kernel taps and
    keeps the masked taps at exactly zero.

    The constant-width mask gives a near-isotropic receptive field: corners are
    suppressed (corners of a square kernel bias toward diagonals), while the
    constant-width property keeps the support from being a plain disk.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
        soft_mask: bool = True,
        sharpness: float = 4.0,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, bias=bias,
        )
        mask = reuleaux_mask(kernel_size, soft=soft_mask, sharpness=sharpness)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, [self.padding] * 4, mode="constant", value=0.0)
        w = self.conv.weight * self.mask  # broadcast (O, I, k, k) * (k, k)
        return F.conv2d(x, w, self.conv.bias, stride=self.stride, padding=0)


def apply_constant_width(model: nn.Module, min_kernel: int = 3,
                         soft_mask: bool = True, sharpness: float = 4.0) -> nn.Module:
    """Recursively replace every square ``nn.Conv2d`` (kernel ≥ ``min_kernel``)
    with a weight-preserving :class:`ConstantWidthConv2d` of the same shape.

    This is the H80 runner-wiring hook: it turns any CNN into its
    constant-width-kernel ablation in place. 1×1 projection/skip convs
    (kernel < ``min_kernel``) are left untouched so the residual contract is
    preserved. The replaced conv's trained-from-scratch weight + bias are
    copied verbatim, and the conv's own ``padding`` is forwarded so the spatial
    shape is unchanged. Because the Reuleaux mask is a fixed buffer, the
    trainable parameter count is identical to the original conv (masked taps
    are simply held at zero), keeping this a clean one-flag ablation (Rule 1).

    Operates in place and returns ``model`` for convenience.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d):
            kh, kw = (child.kernel_size if isinstance(child.kernel_size, tuple)
                      else (child.kernel_size, child.kernel_size))
            sh = (child.stride if isinstance(child.stride, tuple)
                  else (child.stride, child.stride))[0]
            ph = (child.padding if isinstance(child.padding, tuple)
                  else (child.padding, child.padding))[0]
            # Only square kernels at/above the threshold, groups==1.
            if kh == kw and kh >= min_kernel and child.groups == 1:
                new = ConstantWidthConv2d(
                    child.in_channels, child.out_channels, kernel_size=kh,
                    stride=sh, padding=ph, bias=child.bias is not None,
                    soft_mask=soft_mask, sharpness=sharpness,
                )
                with torch.no_grad():
                    new.conv.weight.copy_(child.weight)
                    if child.bias is not None:
                        new.conv.bias.copy_(child.bias)
                setattr(model, name, new)
            else:
                apply_constant_width(child, min_kernel=min_kernel,
                                     soft_mask=soft_mask, sharpness=sharpness)
        else:
            apply_constant_width(child, min_kernel=min_kernel,
                                 soft_mask=soft_mask, sharpness=sharpness)
    return model
