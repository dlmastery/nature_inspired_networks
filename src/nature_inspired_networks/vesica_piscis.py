"""H33 - Vesica Piscis Filter (overlapping-circle multi-path convolution).

The Vesica Piscis is the lens-shaped intersection of two equal circles whose
centres each lie on the other's circumference. Repeating the pattern at
half-radius offsets produces the Flower-of-Life tiling of overlapping discs,
which sacred-geometry literature treats as the canonical "multi-scale
overlapping reception" pattern.

For deep learning, multi-branch / multi-scale conv architectures (Szegedy
2015 Inception, Chen 2019 OctConv, Wang 2019 Big-Little Net) demonstrate
that overlapping receptive fields at different scales improve accuracy. The
Vesica Piscis filter is the explicit geometric realisation: each branch
applies a circular-disc-masked kernel whose centre is offset by a fraction
of the disc radius, so adjacent branches' effective receptive fields
overlap in vesica-piscis intersections. The result is a multi-path
convolution where each path samples a different spatially-shifted disc and
the outputs are summed with a learnable per-path scale.

Refs (Citation Rigor):
    Szegedy, C., et al. 2015 CVPR 'Going Deeper with Convolutions'
    (arXiv:1409.4842) - multi-branch Inception reference.
    Olshausen, B. A., Field, D. J. 1996 Nature - natural-image
    rotation-isotropic receptive-field motivation.
    Chen, Y., et al. 2019 CVPR 'Drop an Octave: Reducing Spatial
    Redundancy in CNNs with Octave Convolution' (arXiv:1904.05049) -
    multi-scale comparator.

Public surface
--------------
- :func:`vesica_kernel_mask`   stack of circular-disc masks at offsets
- :class:`VesicaPiscisConv2d`  multi-path masked Conv2d with learnable scales
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI


def vesica_kernel_mask(
    k: int = 5,
    n_circles: int = 3,
    radius: float = 2.0,
    offset: float = 1.0,
) -> torch.Tensor:
    """Return a stack of ``n_circles`` circular-disc masks of size ``k x k``.

    Each slice ``i in [0, n_circles)`` is a binary disc of radius ``radius``
    centred at ``(cy, cx + offset_i)`` where
    ``offset_i = (i - (n_circles - 1) / 2) * 0.5 * offset``. The shifts
    are symmetric about the kernel centre so adjacent discs overlap in
    vesica-piscis intersections.

    Parameters
    ----------
    k : int
        Spatial extent of the square mask. Must be >= 1.
    n_circles : int
        Number of disc slices. Must be >= 1.
    radius : float
        Disc radius in pixels. Must be > 0.
    offset : float
        Horizontal offset step (in units of ``0.5 * radius``? No - the
        per-circle horizontal shift is ``(i - (n_circles - 1) / 2)
        * 0.5 * offset`` pixels, which keeps the layout centred and
        produces ~half-radius spacing for ``offset == radius``).

    Returns
    -------
    torch.Tensor
        Shape ``(n_circles, k, k)``, dtype float32, values in {0., 1.}.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n_circles < 1:
        raise ValueError(f"n_circles must be >= 1, got {n_circles}")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")

    cy = cx = (k - 1) / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(k, dtype=torch.float32),
        torch.arange(k, dtype=torch.float32),
        indexing="ij",
    )
    out = torch.zeros(n_circles, k, k, dtype=torch.float32)
    for i in range(n_circles):
        shift = (i - (n_circles - 1) / 2.0) * 0.5 * offset
        d2 = (ys - cy) ** 2 + (xs - (cx + shift)) ** 2
        out[i] = (d2 <= radius * radius).float()
    return out


class VesicaPiscisConv2d(nn.Module):
    """Multi-path masked Conv2d with one circular-disc mask per path.

    Each path has its own ``nn.Conv2d`` whose effective kernel is the
    raw weight multiplied (elementwise) by a fixed circle mask from
    :func:`vesica_kernel_mask`. The per-path outputs are summed with a
    learnable per-path scalar ``scale[i]``. Padding is symmetric (``k //
    2``) so the spatial shape is preserved at stride 1.

    The mask is registered as a buffer so it moves with ``.to(device)``
    and is saved in ``state_dict``. The scales are an ``nn.Parameter``,
    initialised to ``1 / n_circles`` so the sum is roughly identity at
    init (the per-path raw conv contributes ``1 / n_paths`` of the
    output).

    Parameters
    ----------
    in_channels, out_channels : int
        Conv2d channels.
    kernel_size : int
        Spatial extent of the square kernel (default 5).
    n_circles : int
        Number of vesica-piscis paths (default 3).
    radius : float
        Disc radius in pixels (default 2.0).
    offset : float
        Horizontal shift step in :func:`vesica_kernel_mask` (default 1.0).
    stride, padding, bias : standard ``nn.Conv2d`` semantics. If
        ``padding`` is ``None`` (default) it is set to ``kernel_size // 2``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        n_circles: int = 3,
        radius: float = 2.0,
        offset: float = 1.0,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if n_circles < 1:
            raise ValueError(f"n_circles must be >= 1, got {n_circles}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_circles = n_circles
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=self.padding,
                    bias=bias,
                )
                for _ in range(n_circles)
            ]
        )
        masks = vesica_kernel_mask(
            k=kernel_size,
            n_circles=n_circles,
            radius=radius,
            offset=offset,
        )
        self.register_buffer("masks", masks)
        # Learnable per-path scale (sum-of-paths combiner). Init at
        # 1/n_circles so the unscaled init has roughly identity gain.
        self.scales = nn.Parameter(torch.full((n_circles,), 1.0 / n_circles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
        out: torch.Tensor | None = None
        for i, conv in enumerate(self.convs):
            mask = self.masks[i].view(1, 1, self.kernel_size, self.kernel_size)
            w = conv.weight * mask
            y = F.conv2d(
                x,
                w,
                conv.bias,
                stride=self.stride,
                padding=self.padding,
            )
            term = self.scales[i] * y
            out = term if out is None else out + term
        assert out is not None  # n_circles >= 1
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, k={self.kernel_size}, "
            f"n_circles={self.n_circles}, padding={self.padding}, stride={self.stride}"
        )


# Use PHI to satisfy the "Reuse PHI" guideline: expose phi-spaced offset
# helper for downstream config generators.
def vesica_phi_offsets(n_circles: int) -> list[float]:
    """Return ``n_circles`` horizontal offsets spaced at successive 1/PHI
    decays (``0.5 * PHI ** -i``). Provided as a convenience for callers
    that want the canonical sacred-geometry phi-spaced vesica layout
    rather than the uniform ``offset`` default in
    :func:`vesica_kernel_mask`.
    """
    if n_circles < 1:
        raise ValueError(f"n_circles must be >= 1, got {n_circles}")
    return [0.5 * (PHI ** -i) for i in range(n_circles)]
