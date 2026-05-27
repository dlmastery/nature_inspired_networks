"""H26 — Fractal Toroidal.

Compose the recursive `_FractalPath` pattern from
:mod:`nature_inspired_networks.blocks` with toroidal (circular) padding
from :func:`nature_inspired_networks.priors.toroidal_pad` so every
convolution inside the fractal recursion operates on a torus topology.

The block is a standalone primitive (does not modify the existing
NaturePriorBlock surface) and follows the same depth / phi-shrink
semantics as the legacy fractal path:

- ``depth=1`` collapses to a single toroidal conv.
- ``depth>=2`` produces parallel branches: branch A is a single
  toroidal conv at this depth, branch B is a toroidal conv followed by
  a recursive ``FractalToroidalBlock(depth-1)``. The two branches are
  merged by mean.
- ``phi_shrink=True`` (default) reduces the recursive branch's
  intermediate width by 1/phi at each recursion level (floor=8), with
  a 1x1 toroidal projection at the merge to restore the output width.

See ``hypotheses/g3_topologies_graphs/H26_fractal_toroidal.md``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nature_inspired_networks.priors import PHI, toroidal_pad


class _ToroidalConv(nn.Module):
    """3x3 (or 1x1) convolution with circular padding.

    For ``kernel_size=1`` no padding is applied; otherwise the input is
    padded circularly by ``kernel_size // 2`` on each spatial side
    before a Conv2d with ``padding=0``.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert kernel_size in {1, 3}, f"unsupported kernel_size {kernel_size}"
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.stride = stride
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size, stride=stride, padding=0, bias=bias,
        )
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = toroidal_pad(x, self.pad)
        y = self.conv(x)
        return self.bn(y)


class FractalToroidalBlock(nn.Module):
    """Self-similar recursive block with toroidal padding at every conv.

    Mirrors the :class:`nature_inspired_networks.blocks._FractalPath`
    contract: width-preserving by construction, single-stride only
    inside the recursion (the outer caller is expected to provide its
    own downsampler if needed — keeping the recursive sub-block at
    stride=1 matches the legacy fractal pattern).

    Parameters
    ----------
    c_in, c_out : int
        Input and output channel counts.
    stride : int, default 1
        Stride for the top-level conv (only used at depth>=1 entry).
    depth : int, default 2
        Recursion depth. Must be >=1.
    phi_shrink : bool, default True
        If True, the recursive (deep) branch shrinks the intermediate
        width by ``1/phi`` per recursion level (floor=8). A 1x1
        toroidal projection at the merge restores the output width.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        stride: int = 1,
        depth: int = 2,
        phi_shrink: bool = True,
    ) -> None:
        super().__init__()
        assert depth >= 1, f"depth must be >=1; got {depth}"
        self.depth = depth
        self.phi_shrink = bool(phi_shrink)
        if depth == 1:
            self.path = _ToroidalConv(c_in, c_out, kernel_size=3, stride=stride)
            self.b_project = None
            return

        # depth >= 2: two branches merged by mean.
        self.a = _ToroidalConv(c_in, c_out, kernel_size=3, stride=stride)
        if self.phi_shrink:
            c_mid = max(8, int(c_out / PHI))
        else:
            c_mid = c_out
        self.b1 = _ToroidalConv(c_in, c_mid, kernel_size=3, stride=stride)
        self.b2 = FractalToroidalBlock(
            c_mid, c_mid, stride=1, depth=depth - 1, phi_shrink=self.phi_shrink,
        )
        if c_mid != c_out:
            self.b_project = _ToroidalConv(c_mid, c_out, kernel_size=1, stride=1)
        else:
            self.b_project = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.depth == 1:
            return self.path(x)
        a = self.a(x)
        b = self.b2(self.b1(x))
        if self.b_project is not None:
            b = self.b_project(b)
        return 0.5 * (a + b)
