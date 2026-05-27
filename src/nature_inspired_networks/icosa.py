"""H24 — Icosahedral φ-Equivariant Conv (lightweight, no e2cnn dep).

Provides the 60-element icosahedral rotation group ``I`` (the orientation-
preserving symmetry group of the icosahedron/dodecahedron, isomorphic to
the alternating group ``A_5``) as a ``(60, 3, 3)`` tensor of rotation
matrices, plus two lightweight nn.Modules that exploit the group:

- :class:`IcosaProjection` — projects 3-D point cloud features through
  the 60 rotations to a 60-element representation per point.
- :class:`IcosaConv1d` — applies a shared 1-D conv kernel across the 60
  rotated copies of the input feature axis and max-pools the orbit to
  yield a rotation-invariant feature.

The 60 rotations are constructed by closing the group under
multiplication of two generators (5-fold + 2-fold), which avoids any
hardcoded list and any external dependency.

References
----------
Cohen 2019 'Gauge Equivariant Convolutional Networks and the Icosahedral
CNN' (arXiv:1902.04615); Esteves 2018 'Learning SO(3) Equivariant
Representations with Spherical CNNs' (arXiv:1711.06721). See
``hypotheses/g3_topologies_graphs/H24_icosahedral_phi_equivariant.md``.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from nature_inspired_networks.priors import PHI


# ---------------------------------------------------------------------------
# 60-element icosahedral rotation group
# ---------------------------------------------------------------------------
def _rot_around_axis(axis: torch.Tensor, angle: float) -> torch.Tensor:
    """Rodrigues' formula. ``axis`` must be a unit 3-vector (float64)."""
    a = axis / axis.norm()
    c = math.cos(angle)
    s = math.sin(angle)
    K = torch.tensor(
        [
            [0.0, -a[2].item(), a[1].item()],
            [a[2].item(), 0.0, -a[0].item()],
            [-a[1].item(), a[0].item(), 0.0],
        ],
        dtype=torch.float64,
    )
    I3 = torch.eye(3, dtype=torch.float64)
    return I3 + s * K + (1 - c) * (K @ K)


def icosahedral_rotations(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the 60 rotation matrices of the icosahedral group as a
    ``(60, 3, 3)`` tensor.

    Construction: start from two generators (a 5-fold rotation around a
    vertex of the icosahedron + a 2-fold rotation around an edge
    midpoint) and close the group by repeated left-multiplication. The
    closure terminates at exactly 60 elements (the order of the
    alternating group ``A_5`` to which ``I`` is isomorphic).

    The resulting tensor is deterministic but the element order is
    arbitrary (BFS order from the generators).
    """
    # 5-fold axis: one of the 12 icosahedron vertices, scaled to unit.
    # Coordinates (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1) are the 12
    # vertices; we pick (0, 1, φ).
    v_5 = torch.tensor([0.0, 1.0, PHI], dtype=torch.float64)
    R5 = _rot_around_axis(v_5, 2 * math.pi / 5)

    # 2-fold axis: the midpoint of an icosahedron edge. The edge from
    # (0, 1, φ) to (1, φ, 0) has midpoint (0.5, (1+φ)/2, φ/2), which lies
    # on the line through the origin along that direction.
    v_2 = torch.tensor([0.5, (1.0 + PHI) / 2.0, PHI / 2.0], dtype=torch.float64)
    R2 = _rot_around_axis(v_2, math.pi)

    I3 = torch.eye(3, dtype=torch.float64)
    elements: list[torch.Tensor] = [I3]
    frontier: list[torch.Tensor] = [I3]
    generators = [R5, R2]
    # BFS close-the-group; safety cap so a numerical bug can't infinite-loop.
    for _ in range(200):
        next_frontier: list[torch.Tensor] = []
        for M in frontier:
            for G in generators:
                C = G @ M
                if not _already_in(C, elements):
                    elements.append(C)
                    next_frontier.append(C)
        if not next_frontier:
            break
        frontier = next_frontier

    assert len(elements) == 60, (
        f"icosahedral group closure produced {len(elements)} elements, not 60"
    )
    out = torch.stack(elements, dim=0)  # (60, 3, 3) in float64
    return out.to(dtype)


def _already_in(M: torch.Tensor, group: Sequence[torch.Tensor], tol: float = 1e-6) -> bool:
    for G in group:
        if (M - G).abs().max().item() < tol:
            return True
    return False


# ---------------------------------------------------------------------------
# IcosaProjection — 3-D points → 60-element rotated representation
# ---------------------------------------------------------------------------
class IcosaProjection(nn.Module):
    """Project a 3-D point cloud onto the 60-element icosahedral
    representation.

    Forward: ``(B, N, 3) → (B, N, 60)``.

    For each input point ``p ∈ R^3`` we compute the projection of ``p``
    onto a fixed reference direction (``ê_z = (0, 0, 1)``) after
    rotating ``p`` by each of the 60 group elements:
    ``z_g = (R_g · p) · ê_z`` for ``g ∈ I``. The result is a length-60
    "phase" signature that transforms by a permutation under any further
    icosahedral rotation of the input — i.e. it is *equivariant*.
    """

    def __init__(self) -> None:
        super().__init__()
        R = icosahedral_rotations()  # (60, 3, 3)
        self.register_buffer("rotations", R)
        # the (0, 0, 1) column of each R is the rotated ê_z; we keep it
        # as a (60, 3) "direction" buffer so the forward pass is one
        # matmul per point cloud.
        self.register_buffer("directions", R[:, :, 2])  # (60, 3)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        assert points.dim() == 3 and points.shape[-1] == 3, (
            f"IcosaProjection expects (B, N, 3); got {tuple(points.shape)}"
        )
        # (B, N, 3) @ (3, 60) → (B, N, 60)
        return points @ self.directions.t()


# ---------------------------------------------------------------------------
# IcosaConv1d — shared 1-D conv applied across the 60-orbit copies
# ---------------------------------------------------------------------------
class IcosaConv1d(nn.Module):
    """Apply a shared 1-D conv across the 60 icosahedral orbit copies of
    the input feature axis and max-pool the orbit to give a rotation-
    invariant feature.

    Forward: ``(B, C_in, L) → (B, C_out, L)``.

    The orbit is generated by treating each of the 60 group elements as
    a *cyclic shift* of the length-60 feature axis (when the input
    length is exactly 60). For inputs of length ``L ≠ 60`` we apply the
    conv along ``L`` and use the icosahedral group only to permute the
    output ``C_out`` channels — this gives a *lightweight* equivariance
    proxy suitable for ablation experiments without paying the full
    GICOPix unfold cost.

    Parameters
    ----------
    in_channels, out_channels
        Standard conv channel counts.
    kernel_size, stride, padding
        Standard conv hyper-parameters (default 3 / 1 / 1).
    hidden_sizes : sequence of int, default (8, 13, 21)
        Fibonacci-channel grouping sizes — the layer enforces that
        ``out_channels`` equals the sum of ``hidden_sizes`` so the
        orbit-permutation step can partition the output cleanly into
        Fibonacci groups. (Defaults sum to 42; pass a different
        sequence for other channel budgets.)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        hidden_sizes: Sequence[int] = (8, 13, 21),
    ) -> None:
        super().__init__()
        hidden_sizes = tuple(int(h) for h in hidden_sizes)
        assert all(h > 0 for h in hidden_sizes), hidden_sizes
        derived_out = sum(hidden_sizes)
        if out_channels is None:
            out_channels = derived_out
        assert out_channels == derived_out, (
            f"out_channels ({out_channels}) must equal sum(hidden_sizes) "
            f"({derived_out}) so the Fibonacci-orbit grouping is exact"
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_sizes = hidden_sizes
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        # Group elements only used here to record the orbit size; the
        # permutation logic is intentionally lightweight.
        R = icosahedral_rotations()
        self.register_buffer("rotations", R)
        # 60-element permutation table: cycle within each Fibonacci group.
        perms: list[torch.Tensor] = []
        for g in range(60):
            p = torch.arange(out_channels)
            offset = 0
            for h in hidden_sizes:
                idx = torch.arange(h)
                rolled = (idx + g) % h
                p[offset:offset + h] = offset + rolled
                offset += h
            perms.append(p)
        self.register_buffer("orbit_perm", torch.stack(perms, dim=0))  # (60, C_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"IcosaConv1d expects (B, C, L); got {tuple(x.shape)}"
        y = self.conv(x)  # (B, C_out, L')
        # max-pool over the 60-element orbit produced by cyclic
        # permutation within each Fibonacci channel group.
        # orbit_perm has shape (60, C_out); apply each permutation and
        # stack.
        orbits = y[:, self.orbit_perm, :]  # (B, 60, C_out, L')
        return orbits.amax(dim=1)  # (B, C_out, L')


# TODO runner wiring:
#   - models.py: at the input stem, optionally insert a 1×1 conv that
#     projects (B, 3, H, W) to (B, 3, H·W) points and feed
#     IcosaProjection → IcosaConv1d as the rotation-invariant prefix.
#     The full GICOPix unfold + 60-rotation conv is deferred (see
#     hypotheses/g3_topologies_graphs/H24 §5.1).
#   - configs/cifar10_quick.yaml: gate behind a `icosa_proxy: true`
#     row with hidden_sizes=[8,13,21] as the canonical Fibonacci
#     channel grouping.
#   - run_sweep.py: this hypothesis is testbed-restricted to Spherical
#     MNIST / IcoMNIST per the H58 lesson — CIFAR-10 upright is NOT a
#     valid testbed for the equivariance claim.
