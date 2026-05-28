"""H53 — 2D-3D Icosahedral Unfold Bridge (lightweight bijection).

Design doc: ``hypotheses/g6_topological_bridging/H53_icosa_unfold_bridge.md``.

The icosahedron has 12 vertices. This module exposes a *deterministic
bijection* between the 12-vertex point set and a 4x3 rectilinear grid
so that standard 2D ``Conv2d`` weights can be reused on icosa-vertex
inputs. The bijection is fixed at module-import time so two networks
built with the same hyperparameters always unfold to the same planar
layout, which is the property a bridge between the 2D and 3D paradigms
needs.

Honest scope (audit note). The bijection is built by sorting vertices
by z-coordinate and slicing into 4 contiguous groups of 3. Because the
12 icosa vertex z-coordinates are ``{+-1, +-1, +-0.618, +-0.618, 0, 0,
0, 0}`` -- i.e. an equatorial band of 4 vertices at z=0 -- the slicing
does NOT recover four geometrically meaningful "latitude bands" or
preserve great-circle vertex adjacency. The permutation is a valid
bijection (every icosa vertex appears exactly once in the planar grid)
and round-trips exactly under ``fold(unfold(x)) == x``; downstream
``Conv2d`` weight reuse is well-defined. But the planar grid does NOT
honour icosa edge structure, and callers should treat this layer as a
deterministic re-shaping bridge rather than a geometric unfold. The
true GICOPix 5-rhomboid-patch scheme is the planned follow-up in
``equiv/gicopix.py``.

Public surface
--------------
- :func:`icosa_unfold_permutation` -- ``(12,)`` long tensor giving a
  deterministic bijection from icosa-vertex index to planar grid
  index. NOT a geometric great-circle traversal; see the audit note
  above.
- :class:`IcosaUnfold`             -- ``nn.Module`` that reshapes a
  ``(B, 12, D)`` vertex feature tensor to ``(B, 4, 3, D)`` using the
  fixed bijection.
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
    """Return the ``(12,)`` deterministic unfold bijection.

    Construction. Sort vertex indices by z-coordinate (descending),
    then slice the sorted index list into 4 contiguous groups of 3 and
    within each group sort by azimuth ``atan2(y, x)``. The result is a
    deterministic permutation of ``{0, ..., 11}``.

    Honest scope (audit note). This is a BIJECTION, NOT a geometric
    great-circle traversal. The 12 icosa vertex z-coordinates are
    ``{+-1, +-1, +-0.618, +-0.618, 0, 0, 0, 0}``: an equatorial band
    of four vertices at z=0 plus two-vertex bands at z = +-1 and
    +-0.618. The 4-group slicing therefore mixes vertices from
    different real latitudes (e.g. group 0 contains one z=1, one z=1
    and one z=0.618 vertex), so the planar grid does NOT honour icosa
    edge structure. Downstream callers should treat the layer as a
    deterministic 12 -> (4, 3) reshape rather than a geometric unfold.

    Returns
    -------
    torch.Tensor (dtype=torch.long, shape=(12,))
        ``perm[k]`` is the icosa-vertex index that lands at planar
        position ``k`` (flattened row-major across the 4x3 grid).
    """
    coords = _icosa_vertex_coords()
    # Sort vertex indices by z (descending); then within each contiguous
    # group of 3, sort by azimuth. The resulting permutation is a pure
    # bijection -- see docstring audit note about why it is NOT a
    # latitude-band / great-circle traversal.
    z = coords[:, 2]
    order_by_z = torch.argsort(z, descending=True)
    perm = torch.empty(_N_VERTICES, dtype=torch.long)
    for band in range(_PLANE_H):
        band_idxs = order_by_z[band * _PLANE_W: (band + 1) * _PLANE_W]
        # within the contiguous group, sort by azimuth atan2(y, x) for
        # determinism (NOT for geometric continuity)
        band_coords = coords[band_idxs]
        azimuth = torch.atan2(band_coords[:, 1], band_coords[:, 0])
        order_within = torch.argsort(azimuth)
        for col, lkup in enumerate(order_within.tolist()):
            perm[band * _PLANE_W + col] = band_idxs[lkup]
    return perm


class IcosaUnfold(nn.Module):
    """Reshape a 12-vertex icosa point set to a planar 4x3 grid via a
    deterministic bijection.

    Forward signature
    -----------------
    ``x: (B, 12, D)``  ->  ``y: (B, 4, 3, D)``

    The permutation is registered as a non-trainable buffer so it
    survives ``state_dict`` round-trips and moves with the module
    across devices. Use :meth:`fold` to invert; calling
    ``unfold(fold(y)) == y`` is exact (no interpolation, the unfold
    is a pure permutation).

    Honest scope (audit note). The bijection does NOT preserve icosa
    edge structure on the planar grid -- see
    :func:`icosa_unfold_permutation` for details. Treat as a
    deterministic reshape, not a geometric unfold.
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
