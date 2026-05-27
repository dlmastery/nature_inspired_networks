"""H34 — Golden-Angle Rotary Position Embedding (RoPE-φ).

A phyllotaxis-inspired variant of RoPE (Su et al. 2021 arXiv:2104.09864)
where each rotary-pair frequency is **shifted** by the golden angle
(137.508° ≈ ``2π·(1 - 1/φ)``) per index. The radial spacing keeps the
exponential RoPE schedule ``freq_k = base^(-2k/dim)`` (``base = φ`` by
default) so frequencies decay geometrically with golden ratio, but the
**phase offset** at frequency ``k`` is ``k · GOLDEN_ANGLE``. The
overall rotation applied at sequence position ``p`` is therefore

    angle(p, k) = p · freq_k + k · GOLDEN_ANGLE

The phase offset is a per-frequency constant — it leaves
*relative* angles between two positions ``p1, p2`` unchanged
(``angle(p2,k) - angle(p1,k) = (p2 - p1) · freq_k``), preserving the
relative-position property RoPE is designed for. The absolute-angle
distribution, however, is forced onto the phyllotactic lattice
(maximally-irrational coverage of the unit circle), which is the
mechanism behind the hypothesised long-context extrapolation gain.

Compare-and-contrast with standard ``apply_rope``
-------------------------------------------------
* Standard RoPE (Su 2021):
    ``freq_k = 10000^(-2k/dim)``, ``angle(p, k) = p · freq_k``.
* RoPE-φ (this module):
    ``freq_k = φ^(-2k/dim)``, ``angle(p, k) = p · freq_k + k · GOLDEN_ANGLE``.

The two differ in:
  1. Base of the frequency schedule (``φ`` vs ``10000``).
  2. The phyllotactic phase offset ``k · GOLDEN_ANGLE`` per pair.

When ``base = PHI`` and the phase offset is dropped, the construction
reduces to standard RoPE with a φ-base.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

from .priors import PHI


__all__ = [
    "GOLDEN_ANGLE",
    "golden_angle_rope_freqs",
    "apply_golden_rope",
]


# Golden angle in radians: 2π · (1 - 1/φ) ≈ 2.39996323 rad ≈ 137.508°.
GOLDEN_ANGLE: float = 2 * math.pi * (1 - 1 / PHI)


def golden_angle_rope_freqs(
    dim: int,
    base: float = PHI,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the frequency and per-frequency phase-offset vectors.

    Parameters
    ----------
    dim : int
        Per-head feature dimension. Must be even (RoPE pairs up
        consecutive components).
    base : float, default ``PHI``
        Base of the exponential frequency schedule. ``freq_k =
        base^(-2k/dim)`` for ``k = 0, ..., dim//2 - 1``. The default
        ``base = φ`` ties the radial spacing to the golden ratio, in
        keeping with the rest of the H34 prior.

    Returns
    -------
    freqs : torch.Tensor
        Shape ``(dim // 2,)`` — the per-pair frequencies.
    phase_offsets : torch.Tensor
        Shape ``(dim // 2,)`` — the per-pair constant phase shifts
        ``k · GOLDEN_ANGLE`` (mod 2π).
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")
    half = dim // 2
    k = torch.arange(half, dtype=torch.float32)
    freqs = base ** (-2.0 * k / dim)
    phase_offsets = (k * GOLDEN_ANGLE) % (2 * math.pi)
    return freqs, phase_offsets


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Pair-wise rotate-half utility (the standard RoPE trick).

    Takes ``x`` of shape ``(..., dim)`` where ``dim = 2 * half`` is
    interpreted as ``half`` 2-D pairs ``(x_{2i}, x_{2i+1})`` and
    returns the half-rotated form ``(-x_{2i+1}, x_{2i})``.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = -x2
    out[..., 1::2] = x1
    return out


def apply_golden_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
    positions: torch.Tensor,
    phase_offsets: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply golden-angle rotary embedding to query and key tensors.

    Parameters
    ----------
    q, k : torch.Tensor
        Shape ``(B, H, N, D)``. ``D`` must be even.
    freqs : torch.Tensor
        Shape ``(D // 2,)``. Per-pair frequencies (typically from
        :func:`golden_angle_rope_freqs`).
    positions : torch.Tensor
        Shape ``(N,)``. Sequence-position indices (usually
        ``torch.arange(N)`` but supports arbitrary position labels for
        relative / interpolated extrapolation experiments).
    phase_offsets : torch.Tensor, optional
        Shape ``(D // 2,)``. Per-pair constant phase shift. If
        ``None``, defaults to ``k · GOLDEN_ANGLE`` matching
        :func:`golden_angle_rope_freqs`. The phase shift is constant
        in position so it does NOT alter the relative-position
        property of RoPE (it cancels in differences) but it changes
        the absolute-angle distribution to the phyllotactic lattice.

    Returns
    -------
    q_rot, k_rot : torch.Tensor
        Same shape as input. ``D``-dimensional rotation applied
        per 2-D pair.
    """
    if q.shape != k.shape:
        raise ValueError(
            f"q and k must have the same shape; got {q.shape} vs {k.shape}"
        )
    if q.dim() != 4:
        raise ValueError(f"expected (B, H, N, D), got {q.shape}")
    B, H, N, D = q.shape
    if D % 2 != 0:
        raise ValueError(f"D must be even; got {D}")
    half = D // 2
    if freqs.shape != (half,):
        raise ValueError(f"freqs shape {tuple(freqs.shape)} != ({half},)")
    if positions.shape != (N,):
        raise ValueError(f"positions shape {tuple(positions.shape)} != ({N},)")
    if phase_offsets is None:
        kidx = torch.arange(half, dtype=torch.float32, device=q.device)
        phase_offsets = (kidx * GOLDEN_ANGLE) % (2 * math.pi)
    else:
        if phase_offsets.shape != (half,):
            raise ValueError(
                f"phase_offsets shape {tuple(phase_offsets.shape)} != ({half},)"
            )

    pos = positions.to(device=q.device, dtype=q.dtype).unsqueeze(-1)  # (N, 1)
    fr = freqs.to(device=q.device, dtype=q.dtype).unsqueeze(0)  # (1, half)
    ph = phase_offsets.to(device=q.device, dtype=q.dtype).unsqueeze(0)  # (1, half)
    angles = pos * fr + ph  # (N, half)
    # Expand pair-wise cos/sin to full D by repeat_interleave along last dim.
    cos_h = angles.cos()  # (N, half)
    sin_h = angles.sin()  # (N, half)
    cos_full = cos_h.repeat_interleave(2, dim=-1)  # (N, D)
    sin_full = sin_h.repeat_interleave(2, dim=-1)  # (N, D)
    # Broadcast to (1, 1, N, D) so it pairs with (B, H, N, D).
    cos_full = cos_full.view(1, 1, N, D)
    sin_full = sin_full.view(1, 1, N, D)
    q_rot = q * cos_full + _rotate_half(q) * sin_full
    k_rot = k * cos_full + _rotate_half(k) * sin_full
    return q_rot, k_rot
