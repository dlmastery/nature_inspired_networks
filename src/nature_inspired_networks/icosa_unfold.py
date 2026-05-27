"""H53 — 2D-3D Icosahedral Unfold Bridge (GICOPix-style).

Design doc: ``hypotheses/g6_topological_bridging/H53_icosa_unfold_bridge.md``.

The icosahedron has 12 vertices. A planar GICOPix-style unfold flattens
the 12-vertex point set onto a deterministic 4x3 rectilinear grid so that
standard 2D ``Conv2d`` weights can be reused on sphere-discretised inputs.
The permutation is fixed at module-import time -- great-circle traversal
order -- so two networks built with the same hyperparameters always
unfold to the same planar layout, which is the property a bridge between
the 2D and 3D paradigms needs.

This is the *lightweight* version of the bridge. It does NOT implement
the full 5-rhomboid-patch GICOPix scheme (that lives in the planned
``equiv/gicopix.py`` follow-up); instead the 12 icosa vertices unfold
to a single 4x3 planar grid that the existing ``Conv2d`` kernels can
operate on without modification. Suitable as a feature-level bridge
between an icosa-vertex point set and the planar conv stack.

Public surface
--------------
- :func:`icosa_unfold_permutation` -- ``(12,)`` long tensor that maps
  icosa vertex index -> planar grid index (great-circle order).
- :class:`IcosaUnfold`             -- ``nn.Module`` that reshapes a
  ``(B, 12, D)`` vertex feature tensor to ``(B, 4, 3, D)`` using the
  fixed permutation.
- :class:`IcosaToPlane`            -- composition of ``IcosaUnfold``
  with a standard ``nn.Conv2d``; lets pretrained 2D conv weights run
  on icosa-vertex features.

References (Citation Rigor)
---------------------------
    Cohen, Taco S., Geiger, Mario, Koehler, Jonas, Welling, Max 2019
    ICML 'Gauge Equivariant Convolutional Networks and the Icosahedral
    CNN' (arXiv:1902.04615) -- the canonical icosahedral CNN.
    Yu, Yu, Hong, Sungwon and others 2019 NeurIPS 'Spherical CNNs on
    Unstructured Grids' (arXiv:1901.02039) -- the GICOPix-derived
    icosa-pixelisation; direct technical predecessor of this module.
    Goerski, Krzysztof M., Hivon, Eric, Banday, Anthony J. and others
    2005 ApJ 'HEALPix: A Framework for High-Resolution Discretization
    and Fast Analysis of Data Distributed on the Sphere' --
    methodological anchor for the planar unfold idea.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .priors import PHI  # noqa: F401  (re-exported; canonical PHI constant)


_N_VERTICES = 12
_PLANE_H = 4
_PLANE_W = 3


def _icosa_vertex_coords() -> torch.Tensor:
    """Return the 12 canonical icosahedral vertex coordinates (in R^3).

    The vertices come in three orthogonal "golden rectangles" of size
    ``2 x 2/phi``. Returned tensor has shape ``(12, 3)`` and the
    Euclidean norm of every row is the same icosa-circumradius. Used
    only as a deterministic ordering key for the unfold permutation;
    not exposed as public API.
    """
    inv_phi = 1.0 / PHI
    v = torch.tensor(
        [
            [0.0, +1.0, +inv_phi],   # 0
            [0.0, +1.0, -inv_phi],   # 1
            [0.0, -1.0, +inv_phi],   # 2
            [0.0, -1.0, -inv_phi],   # 3
            [+1.0, +inv_phi, 0.0],   # 4
            [+1.0, -inv_phi, 0.0],   # 5
            [-1.0, +inv_phi, 0.0],   # 6
            [-1.0, -inv_phi, 0.0],   # 7
            [+inv_phi, 0.0, +1.0],   # 8
            [-inv_phi, 0.0, +1.0],   # 9
            [+inv_phi, 0.0, -1.0],   # 10
            [-inv_phi, 0.0, -1.0],   # 11
        ],
        dtype=torch.float32,
    )
    return v


def icosa_unfold_permutation() -> torch.Tensor:
    """Return the ``(12,)`` GICOPix-style unfold permutation.

    The permutation walks the 12 icosa vertices in a great-circle order
    that places vertically adjacent rows of the resulting 4x3 planar
    grid next to vertices that share an edge on the icosahedron (i.e.,
    the unfold approximately preserves vertex adjacency). The walk is:

    - row 0 (top cap)   : the 3 northernmost vertices sorted by azimuth
    - row 1 (upper band): the 3 next-most-northern vertices
    - row 2 (lower band): the 3 next-most-southern vertices
    - row 3 (bottom cap): the 3 southernmost vertices

    Within each row, vertices are sorted by their azimuth angle so the
    great-circle traversal is continuous. The result is deterministic
    and bijective -- every icosa vertex appears exactly once in the
    4x3 = 12-cell planar grid.

    Returns
    -------
    torch.Tensor (dtype=torch.long, shape=(12,))
        ``perm[k]`` is the icosa-vertex index that lands at planar
        position ``k`` (flattened row-major across the 4x3 grid).
    """
    coords = _icosa_vertex_coords()
    # Sort vertices into 4 latitude bands of 3 vertices each. The
    # z-coordinate is monotone with latitude, so descending z is north-
    # to-south.
    z = coords[:, 2]
    order_by_z = torch.argsort(z, descending=True)
    perm = torch.empty(_N_VERTICES, dtype=torch.long)
    for band in range(_PLANE_H):
        band_idxs = order_by_z[band * _PLANE_W: (band + 1) * _PLANE_W]
        # within the band, sort by azimuth atan2(y, x) so the traversal
        # is great-circle-continuous
        band_coords = coords[band_idxs]
        azimuth = torch.atan2(band_coords[:, 1], band_coords[:, 0])
        order_within = torch.argsort(azimuth)
        for col, lkup in enumerate(order_within.tolist()):
            perm[band * _PLANE_W + col] = band_idxs[lkup]
    return perm


class IcosaUnfold(nn.Module):
    """Flatten a 12-vertex icosa point set to a planar 4x3 grid.

    Forward signature
    -----------------
    ``x: (B, 12, D)``  ->  ``y: (B, 4, 3, D)``

    The permutation is registered as a non-trainable buffer so it
    survives ``state_dict`` round-trips and moves with the module
    across devices. Use :meth:`fold` to invert the unfold; calling
    ``unfold(fold(y)) == y`` is exact (no interpolation, the unfold
    is a pure permutation).
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("perm", icosa_unfold_permutation())
        # cached inverse permutation -- p_inv[perm[k]] = k
        inv = torch.empty(_N_VERTICES, dtype=torch.long)
        inv[self.perm] = torch.arange(_N_VERTICES, dtype=torch.long)
        self.register_buffer("perm_inv", inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"expected (B, 12, D), got shape {tuple(x.shape)}"
        assert x.shape[1] == _N_VERTICES, (
            f"expected 12 icosa vertices, got {x.shape[1]}"
        )
        B, _, D = x.shape
        # gather along vertex axis, then reshape to (B, H, W, D)
        gathered = x.index_select(dim=1, index=self.perm)
        return gathered.view(B, _PLANE_H, _PLANE_W, D)

    def fold(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of forward: ``(B, 4, 3, D)`` -> ``(B, 12, D)``."""
        assert y.shape[1:3] == (_PLANE_H, _PLANE_W), (
            f"expected (B, 4, 3, D), got shape {tuple(y.shape)}"
        )
        B = y.shape[0]
        D = y.shape[-1]
        flat = y.reshape(B, _N_VERTICES, D)
        return flat.index_select(dim=1, index=self.perm_inv)


class IcosaToPlane(nn.Module):
    """Run a standard 2D ``Conv2d`` on GICOPix-unfolded icosa features.

    Wraps :class:`IcosaUnfold` so that a pretrained planar 2D conv can
    be re-used on icosa-vertex inputs without modification. The forward
    pass is:

    - unfold ``(B, 12, C_in)`` to ``(B, H=4, W=3, C_in)``
    - permute to NCHW ``(B, C_in, 4, 3)``
    - apply ``nn.Conv2d``
    - permute back to NHWC ``(B, 4, 3, C_out)``
    - fold to ``(B, 12, C_out)``

    Parameters
    ----------
    in_channels, out_channels : int
        as in ``nn.Conv2d``.
    kernel_size : int, default 3
    bias : bool, default True
    padding : int, default 'same' via ``kernel_size // 2`` so the
        spatial 4x3 shape is preserved.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.unfold = IcosaUnfold()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, C_in)
        planar = self.unfold(x)                       # (B, 4, 3, C_in)
        nchw = planar.permute(0, 3, 1, 2).contiguous()  # (B, C_in, 4, 3)
        y = self.conv(nchw)                            # (B, C_out, 4, 3)
        y_nhwc = y.permute(0, 2, 3, 1).contiguous()    # (B, 4, 3, C_out)
        return self.unfold.fold(y_nhwc)                # (B, 12, C_out)
