"""H71 — Icosahedral 3-D Rotary Position Embedding (G7 hybrid).

Design doc: ``hypotheses/g7_cross_paradigm_hybrids/H71_icosa_rope_3d.md``.

Generalises :func:`golden_rope.apply_golden_rope` to 3-D. The standard
RoPE rotates **pairs** of channels by a position-dependent angle in a
2-D plane; H71 rotates **triples** of channels by a position-dependent
angle whose axis is one of the 12 icosahedral vertices.

Concretely:

  - The input ``q, k`` tensors are reshaped from ``(B, H, N, D)`` to
    ``(B, H, N, D // 3, 3)``. ``D`` must be a multiple of 3.
  - For each triple, the rotation axis is fixed to icosahedral vertex
    ``a_t = icosa_vertices[t mod 12]`` (so triples cycle through the 12
    icosahedral axes), and the angle is ``θ_p = p · ω_t`` where
    ``ω_t = φ^(-t)`` (golden-ratio frequency decay).
  - The rotation is applied via Rodrigues' formula. This is the natural
    SO(3) generalisation of the 2-D RoPE pair-rotation trick.

References (Citation Rigor)::

    Su, Lu, Pan, Murtadha, Wen, Liu 2021 'RoFormer' (arXiv:2104.09864).

    Cohen, Weiler, Kicanaoglu, Welling 2019 ICML 'Icosahedral CNN'
    (arXiv:1902.04615) — the icosahedral group structure.

    Yartsev, Ulanovsky 2013 Nature — 3-D place-cell basis precedent.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from .priors import PHI


__all__ = ["IcosaRoPE3D", "icosa_vertices_unit"]


def icosa_vertices_unit() -> torch.Tensor:
    """Return the 12 icosahedron vertices as unit vectors, shape (12, 3).

    Coordinates are the canonical ``(0, ±1, ±φ)``, ``(±1, ±φ, 0)``,
    ``(±φ, 0, ±1)`` set, normalised to unit length. Deterministic order.
    """
    verts = torch.tensor(
        [
            [0.0, 1.0, PHI],
            [0.0, -1.0, PHI],
            [0.0, 1.0, -PHI],
            [0.0, -1.0, -PHI],
            [1.0, PHI, 0.0],
            [-1.0, PHI, 0.0],
            [1.0, -PHI, 0.0],
            [-1.0, -PHI, 0.0],
            [PHI, 0.0, 1.0],
            [-PHI, 0.0, 1.0],
            [PHI, 0.0, -1.0],
            [-PHI, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    return verts / verts.norm(dim=-1, keepdim=True)


def _rodrigues_apply(
    x3: torch.Tensor,
    axis: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
) -> torch.Tensor:
    """Apply Rodrigues' rotation formula in a vectorised way.

    Parameters
    ----------
    x3 : torch.Tensor
        Shape ``(..., 3)``.
    axis : torch.Tensor
        Shape ``(..., 3)`` unit vector, broadcastable to x3.
    cos_t, sin_t : torch.Tensor
        Shape broadcastable to ``x3[..., 0]``.

    Returns
    -------
    torch.Tensor
        Same shape as ``x3``.
    """
    # v_rot = v cos + (k × v) sin + k (k · v) (1 - cos)
    cos_t = cos_t.unsqueeze(-1)
    sin_t = sin_t.unsqueeze(-1)
    k_cross_v = torch.cross(axis.expand_as(x3), x3, dim=-1)
    k_dot_v = (axis * x3).sum(dim=-1, keepdim=True)
    return x3 * cos_t + k_cross_v * sin_t + axis * k_dot_v * (1.0 - cos_t)


class IcosaRoPE3D(nn.Module):
    """3-D icosahedral RoPE: position-dependent rotations of channel triples.

    Forward signature: ``q_rot, k_rot = IcosaRoPE3D(...)(q, k, positions)``.

    Parameters
    ----------
    head_dim : int
        Per-head feature dimension. Must be a multiple of 3.
    base : float, default :data:`PHI`
        Base of the per-triple frequency decay: ``ω_t = base^(-t)`` for
        triple index ``t = 0, ..., head_dim // 3 - 1``. Default is φ so
        the radial spacing matches the rest of the φ-RoPE family.
    """

    def __init__(self, head_dim: int, base: float = PHI) -> None:
        super().__init__()
        if head_dim % 3 != 0:
            raise ValueError(
                f"head_dim must be a multiple of 3 for triple-rotation; got {head_dim}"
            )
        if base <= 0:
            raise ValueError(f"base must be positive; got {base}")
        self.head_dim = int(head_dim)
        self.n_triples = head_dim // 3
        self.base = float(base)

        # Per-triple frequencies (D/3,)
        t = torch.arange(self.n_triples, dtype=torch.float32)
        freqs = base ** (-t)  # geometric decay with base φ
        self.register_buffer("freqs", freqs)

        # Per-triple rotation axes — cycle through 12 icosa vertices.
        verts = icosa_vertices_unit()  # (12, 3)
        axes = verts[t.long() % 12]  # (D/3, 3)
        self.register_buffer("axes", axes)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply icosahedral 3-D RoPE to query/key tensors.

        Parameters
        ----------
        q, k : torch.Tensor
            Shape ``(B, H, N, D)`` with ``D == head_dim``.
        positions : torch.Tensor
            Shape ``(N,)``, the per-token position index.

        Returns
        -------
        q_rot, k_rot : torch.Tensor
            Same shape as inputs.
        """
        if q.shape != k.shape:
            raise ValueError(
                f"q/k shape mismatch: {tuple(q.shape)} vs {tuple(k.shape)}"
            )
        if q.dim() != 4:
            raise ValueError(f"expected (B, H, N, D); got {tuple(q.shape)}")
        B, H, N, D = q.shape
        if D != self.head_dim:
            raise ValueError(f"D={D} != head_dim={self.head_dim}")
        if positions.shape != (N,):
            raise ValueError(
                f"positions shape {tuple(positions.shape)} != ({N},)"
            )

        # Angles: (N, D/3) = positions[:, None] * freqs[None, :]
        pos = positions.to(device=q.device, dtype=q.dtype).unsqueeze(-1)  # (N, 1)
        fr = self.freqs.to(device=q.device, dtype=q.dtype).unsqueeze(0)  # (1, D/3)
        angles = pos * fr  # (N, D/3)
        cos_t = angles.cos()
        sin_t = angles.sin()

        # Reshape q, k to (B, H, N, D/3, 3).
        q3 = q.view(B, H, N, self.n_triples, 3)
        k3 = k.view(B, H, N, self.n_triples, 3)

        # Axes: (D/3, 3) → broadcast to (1, 1, 1, D/3, 3).
        axes = self.axes.to(device=q.device, dtype=q.dtype)
        axes_b = axes.view(1, 1, 1, self.n_triples, 3)
        # Cos / sin angles: (1, 1, N, D/3) → unsqueeze last dim in rodrigues.
        cos_b = cos_t.view(1, 1, N, self.n_triples)
        sin_b = sin_t.view(1, 1, N, self.n_triples)
        # Expand to broadcastable forms.
        q_rot3 = _rodrigues_apply(q3, axes_b.expand_as(q3), cos_b, sin_b)
        k_rot3 = _rodrigues_apply(k3, axes_b.expand_as(k3), cos_b, sin_b)
        return q_rot3.reshape(B, H, N, D), k_rot3.reshape(B, H, N, D)
