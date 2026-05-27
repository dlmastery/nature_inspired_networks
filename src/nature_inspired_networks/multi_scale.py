"""Golden-spiral multi-resolution scaling (H03).

H03 grows the input resolution through phi-multiplicative stages along
the golden-angle 137.508 degree lattice. For a CNN-track ablation the
simplest, fastest, and most faithful drop-in is an input-resize wrapper
that resizes a stock CIFAR-10 tensor to a phi-scaled resolution before
the existing NaturePriorNet stem. The full ViT multi-scale variant is
out of scope for the smoke sweep; the helpers below are designed so a
future ViT idea can re-import them unchanged.

Refs:
  Vogel 1979 Math Biosciences 'A better way to construct the sunflower
    head'
  Dosovitskiy et al. 2021 ICLR 'An Image is Worth 16x16 Words'
    (arXiv:2010.11929)
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI


GOLDEN_ANGLE_DEG = 360.0 / (PHI ** 2)        # ~137.50776 deg
GOLDEN_ANGLE_RAD = 2 * math.pi * (1 - 1 / PHI)  # equivalent in radians


def golden_spiral_resolutions(
    base: int = 28,
    n_stages: int = 4,
    align: int = 1,
) -> list[int]:
    """Return a sequence of resolutions scaled by phi**k.

    With ``base=28, n_stages=5`` the canonical H03 schedule is
    ``[28, 45, 73, 118, 191]``. ``align`` rounds each resolution up to
    the nearest multiple (e.g., ``align=4`` keeps tensor cores happy at
    higher resolutions). Resolutions are monotonically non-decreasing
    after alignment.
    """
    if base < 1 or n_stages < 1 or align < 1:
        raise ValueError("base, n_stages, align must all be >= 1")
    out: list[int] = []
    last = 0
    for k in range(n_stages):
        r_raw = base * (PHI ** k)
        r = int(round(r_raw))
        if align > 1:
            r = max(align, align * int(round(r / align)))
        # ensure strict monotonicity so each stage truly upscales
        if r <= last:
            r = last + align
        out.append(r)
        last = r
    return out


def golden_spiral_lattice(n: int) -> torch.Tensor:
    """Return ``n`` 2-D points on a unit disk along the golden spiral.

    Following Vogel 1979: ``theta_k = k * golden_angle``,
    ``r_k = sqrt(k / n)``. The output shape is ``(n, 2)``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    idx = torch.arange(n, dtype=torch.float32)
    theta = idx * GOLDEN_ANGLE_RAD
    r = torch.sqrt(idx / float(n))
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)


class GoldenSpiralResize(nn.Module):
    """Drop-in input wrapper that resizes ``(B, C, H, W)`` to ``size``.

    The H03 CNN-track ablation uses this as a stem-level resize so the
    rest of the NaturePriorNet sees a phi-scaled spatial resolution.
    ``size`` is a single int (square) or a tuple; ``mode`` is the
    interpolation mode (default ``bilinear``). The module is stateless
    and adds no parameters.
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        mode: str = "bilinear",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(size, int):
            if size < 1:
                raise ValueError("size must be >= 1")
            self.size = (size, size)
        else:
            if len(size) != 2 or any(s < 1 for s in size):
                raise ValueError("size must be int or (H, W) >= 1")
            self.size = (int(size[0]), int(size[1]))
        if mode not in {"bilinear", "bicubic", "nearest", "area"}:
            raise ValueError(f"unsupported interp mode '{mode}'")
        self.mode = mode
        # align_corners only meaningful for bilinear/bicubic
        self.align_corners = align_corners if mode in {"bilinear", "bicubic"} else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W) input, got {tuple(x.shape)}")
        kwargs = dict(size=self.size, mode=self.mode)
        if self.align_corners is not None:
            kwargs["align_corners"] = self.align_corners
        return F.interpolate(x, **kwargs)
