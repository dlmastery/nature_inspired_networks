"""H40 - Metatron Kernel Overlap (shared kernel basis).

Metatron's Cube is the 13-circle diagram formed by drawing circles at
the 13 vertices of the Fruit-of-Life pattern: one central circle, six
inner-hex circles, and six outer circles. Projected onto 2-D, the
overlapping discs produce a maximally-symmetric 13-element basis on a
hex-centered lattice whose intersections contain projections of all
five Platonic solids.

For deep learning, the Metatron kernel basis provides **weight-shared
sparse filters**: instead of learning K independent dense kernels, we
construct 13 fixed circular-disc basis kernels (one per Metatron
vertex) and learn (a) per-circle scalar mixing coefficients
``alpha_c``, and (b) a per-circle shared kernel ``W_c``. The effective
filter is

    ``W_eff = sum_c alpha_c * basis_c * W_c``

This is the parameter-saving / low-rank kernel factorisation of
DCFNet (Qiu 2018) with the Metatron-13 fixed geometric basis rather
than a learned PCA basis.

Refs (Citation Rigor):
    Qiu, Q., Cheng, X., Calderbank, R., Sapiro, G. 2018 'DCFNet:
    Deep Convolutional Filter Network with Decomposed Convolutional
    Filters' (arXiv:1802.04145) - kernel-basis factorisation.
    Cohen, T. S., Welling, M. 2016 ICML 'Group Equivariant
    Convolutional Networks' (arXiv:1602.07576) - group-equivariance
    framework Metatron's Cube specialises.
    Hoogeboom, E., et al. 2018 - HexaConv hex-lattice reference.
    Mandelbrot, B. 1982 - multi-scale fractal structure.

Public surface
--------------
- :func:`metatron_basis_kernels`   (13, k, k) overlapping circular masks
- :class:`MetatronConv2d`          shared-basis Conv2d with learnable
                                   alpha and per-circle W
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI


def metatron_basis_kernels(k: int = 7, n_circles: int = 13) -> torch.Tensor:
    """Return ``n_circles`` overlapping circular-disc masks on a ``k x k``
    grid, arranged as the Metatron-Cube vertex layout.

    Canonical layout (n_circles = 13):
        - 1 centre circle at ``(k/2, k/2)``.
        - 6 inner-hex circles at distance ``r_inner`` from centre, at
          angles ``i * pi / 3 for i in range(6)``.
        - 6 outer circles at distance ``r_outer`` from centre, at
          angles ``i * pi / 3 + pi / 6 for i in range(6)``.

    All 13 circles have the same radius ``r_disc = k / 4`` and overlap
    on the central pixel by construction (each circle's centre lies
    within ``r_outer + r_disc`` of the centre; the central pixel is
    inside every disc when ``r_outer + r_disc <= k/2``).

    Other ``n_circles`` values are supported as truncations / extensions
    of the canonical 13 (1 + 6 + 6); ``n_circles in {1, 7, 13}`` are the
    natural choices.

    Parameters
    ----------
    k : int
        Spatial size of each square mask. Must be >= 3.
    n_circles : int
        Number of Metatron circles to keep (default 13).

    Returns
    -------
    torch.Tensor
        Shape ``(n_circles, k, k)``, dtype float32, values in [0, 1]
        (binary masks, but returned as float32 for convenient
        elementwise multiplication with conv weights).
    """
    if k < 3:
        raise ValueError(f"k must be >= 3, got {k}")
    if n_circles < 1:
        raise ValueError(f"n_circles must be >= 1, got {n_circles}")

    cy = cx = (k - 1) / 2.0
    # Concentric layout radii scaled so all outer circles fit. r_disc is
    # the disc radius; r_inner / r_outer are the centre offsets from the
    # kernel centre.
    r_disc = k / 4.0
    r_inner = k / 4.0
    r_outer = k / 2.5  # slightly less than k/2 so outer discs fit

    centres: list[tuple[float, float]] = [(cy, cx)]
    for i in range(6):
        theta = i * math.pi / 3.0
        centres.append((cy + r_inner * math.sin(theta), cx + r_inner * math.cos(theta)))
    for i in range(6):
        theta = i * math.pi / 3.0 + math.pi / 6.0
        centres.append((cy + r_outer * math.sin(theta), cx + r_outer * math.cos(theta)))

    # If a non-canonical n_circles was requested, truncate / cycle.
    if n_circles <= len(centres):
        centres = centres[:n_circles]
    else:
        # cycle the canonical 13 to fill more slots (preserves overlap
        # structure but adds duplicates - useful only for stress tests)
        reps = (n_circles + len(centres) - 1) // len(centres)
        centres = (centres * reps)[:n_circles]

    ys, xs = torch.meshgrid(
        torch.arange(k, dtype=torch.float32),
        torch.arange(k, dtype=torch.float32),
        indexing="ij",
    )
    out = torch.zeros(n_circles, k, k, dtype=torch.float32)
    for i, (ccy, ccx) in enumerate(centres):
        d2 = (ys - ccy) ** 2 + (xs - ccx) ** 2
        out[i] = (d2 <= r_disc * r_disc).float()
    return out


class MetatronConv2d(nn.Module):
    """Conv2d where the effective filter is a learnable mixture over a
    fixed 13-circle Metatron basis.

    Per the H40 design, each output filter is constructed as

        ``W_eff[o, i, y, x] = sum_c alpha[c] * basis[c, y, x] * W[c, o, i, y, x]``

    where:
      - ``basis`` is the fixed ``(n_circles, k, k)`` tensor from
        :func:`metatron_basis_kernels` (registered as a buffer);
      - ``alpha`` is a learnable ``(n_circles,)`` mixing vector;
      - ``W`` is a learnable ``(n_circles, out_channels, in_channels,
        k, k)`` per-circle kernel - "shared" in the sense that all
        circles draw from the same parameter array indexed by ``c``.

    The 1/PHI scaling in alpha-init follows the H38 phi-decay
    convention so that the central circle dominates and outer circles
    contribute decreasing share at initialisation.

    Parameters
    ----------
    in_channels, out_channels : int
        Conv2d channels.
    kernel_size : int
        Spatial extent of each basis kernel (default 7).
    n_circles : int
        Number of Metatron circles (default 13).
    stride, padding, bias : standard Conv2d semantics. If ``padding`` is
        ``None`` (default) it is set to ``kernel_size // 2``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        n_circles: int = 13,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size < 3:
            raise ValueError(f"kernel_size must be >= 3, got {kernel_size}")
        if n_circles < 1:
            raise ValueError(f"n_circles must be >= 1, got {n_circles}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_circles = n_circles
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding

        basis = metatron_basis_kernels(k=kernel_size, n_circles=n_circles)
        self.register_buffer("basis", basis)  # (n_circles, k, k)

        # Per-circle phi-decaying alpha (learnable mixing).
        alpha_init = torch.tensor([1.0 / (PHI ** c) for c in range(n_circles)])
        self.alpha = nn.Parameter(alpha_init)

        # Per-circle shared kernel array W (learnable).
        self.W = nn.Parameter(
            torch.empty(n_circles, out_channels, in_channels, kernel_size, kernel_size)
        )
        # He-style init scaled by sqrt(1 / n_circles) so the summed
        # effective kernel has variance comparable to a standard Conv2d.
        fan_in = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in) / math.sqrt(n_circles)
        with torch.no_grad():
            self.W.normal_(0.0, std)

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def effective_kernel(self) -> torch.Tensor:
        """Construct the effective ``(out_c, in_c, k, k)`` filter by
        summing alpha * basis * W over the circle axis.
        """
        # basis: (n_circles, k, k) -> (n_circles, 1, 1, k, k)
        b = self.basis.view(self.n_circles, 1, 1, self.kernel_size, self.kernel_size)
        # alpha: (n_circles,) -> (n_circles, 1, 1, 1, 1)
        a = self.alpha.view(self.n_circles, 1, 1, 1, 1)
        # Sum over circle axis: (out, in, k, k)
        return (a * b * self.W).sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
        w = self.effective_kernel()
        return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"k={self.kernel_size}, n_circles={self.n_circles}"
        )
