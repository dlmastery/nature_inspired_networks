"""H38 - Fractal Golden Filter (multi-scale 3 + 5 + 8 stacked).

Fractal self-similarity at phi-scales is a near-universal organising
principle in nature: cortical folding, vascular branching, lung-alveoli
structure, river networks, and coastline statistics all exhibit fractal
dimension between 2.0 and 2.6 with self-similar substructure at
successive scales (Mandelbrot 1982). The mathematical signature is that
the structure looks similar at every magnification with a fixed scale
ratio - and that ratio is often phi or close to it.

The FractalGoldenFilter applies, at every conv layer, three parallel
kernels of Fibonacci sizes ``{3, 5, 8}`` simultaneously, projecting each
back to the channel count via a 1x1 and summing with learnable per-path
scales ``alpha_k``. This is the fractal generalisation of the Vesica
Piscis filter (H33) with explicit Fibonacci kernel-size progression.

Refs (Citation Rigor):
    Larsson, G., Maire, M., Shakhnarovich, G. 2017 ICLR 'FractalNet'
    (arXiv:1605.07648) - primary fractal-network reference.
    Mandelbrot, B. 1982 'The Fractal Geometry of Nature' - fractal
    self-similarity in nature.
    Szegedy, C., et al. 2015 - Inception multi-branch reference.
    Chen, Y., et al. 2019 - OctConv multi-scale comparator.

Public surface
--------------
- :data:`FIB_KERNELS`                  default ``(3, 5, 8)`` schedule
- :class:`FractalGoldenFilter`         multi-path conv with learnable
                                       per-path scales (phi-decaying init)
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI


FIB_KERNELS: tuple[int, int, int] = (3, 5, 8)


class FractalGoldenFilter(nn.Module):
    """Multi-path conv with Fibonacci-spaced kernel sizes.

    Each path ``k`` (kernel size ``kernel_sizes[k]``) applies an in-place
    small Conv2d at that kernel size with ``padding='same'`` semantics
    (i.e. ``padding=kernel_size // 2``), then a 1x1 projection back to
    ``out_channels``. The per-path outputs are weighted by a learnable
    scale ``alpha_k`` (initialised at ``1 / PHI ** k`` then normalised so
    the alphas sum to 1) and summed.

    Parameters
    ----------
    in_channels, out_channels : int
        Conv2d channels.
    kernel_sizes : Sequence[int]
        Kernel sizes for each path. Default ``(3, 5, 8)``. Must be a
        non-empty sequence of positive odd-or-even integers. For
        ``kernel_size=8`` (even) we use ``padding=4`` and crop one
        column / row of the output on the trailing axis so the spatial
        shape is preserved at stride=1 (PyTorch lacks a true
        ``padding='same'`` for even kernels - we emulate it).
    mid_channels : int | None
        Number of channels in the in-place small kernel before the
        1x1 projection. ``None`` (default) means ``mid_channels =
        out_channels``.
    bias : bool
        Bias on the inner Conv2d and the 1x1 projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = FIB_KERNELS,
        mid_channels: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        kernel_sizes = tuple(int(k) for k in kernel_sizes)
        if len(kernel_sizes) < 1:
            raise ValueError("kernel_sizes must have at least one element")
        if any(k < 1 for k in kernel_sizes):
            raise ValueError(f"kernel sizes must be >= 1, got {kernel_sizes}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        mid = out_channels if mid_channels is None else mid_channels
        self.mid_channels = mid

        # Per-path small conv + 1x1 projection.
        self.path_convs = nn.ModuleList()
        self.path_projs = nn.ModuleList()
        self.path_paddings: list[int] = []
        self.path_even_crops: list[bool] = []
        for k in kernel_sizes:
            # 'same'-style padding: floor(k/2). For even k we will crop
            # one pixel off the trailing axes after the conv to preserve
            # input spatial shape.
            pad = k // 2
            self.path_paddings.append(pad)
            self.path_even_crops.append(k % 2 == 0)
            self.path_convs.append(
                nn.Conv2d(in_channels, mid, k, padding=pad, bias=bias)
            )
            self.path_projs.append(
                nn.Conv2d(mid, out_channels, 1, bias=bias)
            )

        # Learnable phi-decaying scales, normalised at init so sum=1.
        init = torch.tensor([1.0 / (PHI ** k) for k in range(len(kernel_sizes))])
        init = init / init.sum()
        self.alpha = nn.Parameter(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
        B, _, H, W = x.shape
        out: torch.Tensor | None = None
        for i, (conv, proj) in enumerate(zip(self.path_convs, self.path_projs)):
            y = conv(x)
            # Crop one pixel on each trailing axis if the kernel was even
            # (we used floor(k/2) padding, so the output is +1 too big).
            if self.path_even_crops[i]:
                y = y[..., :H, :W]
            y = proj(y)
            term = self.alpha[i] * y
            out = term if out is None else out + term
        assert out is not None
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"kernel_sizes={self.kernel_sizes}, mid={self.mid_channels}"
        )
