"""H77 — Radial 12-fold Symmetry Attention.

Multi-head attention whose heads are arranged with a 12-fold radial
symmetry: each head ``h`` carries a fixed *relative*-position angular
bias ``cos(2*pi*h/12 + 2*pi*delta/L)`` added to the pre-softmax logits.
This generalises :mod:`pentagonal_attention` (5-fold) to a 12-fold
arrangement, the highest-multiplicity radial tiling that still tiles a
2-D angular plane with integer divisors (12 = clock dial / dodecagonal
rosette). The bias is relative-position-dependent on purpose: a constant
additive per-head bias is invariant under the softmax normalisation, so
only a ``delta = j - i`` dependence actually changes attention weights.

Esoteric origin (acknowledged in one sentence): the "Lotus of Life"
12-petal rosette motivates the 12-fold partition; the implementation is a
neutral, parameter-free relative-positional head bias.

Refs (Citation Rigor):
    Vaswani, A., Shazeer, N., Parmar, N. 2017 NeurIPS 'Attention Is All
    You Need' (arXiv:1706.03762) - the MHA reference this module
    extends with a structural per-head relative bias.
    Shaw, P., Uszkoreit, J., Vaswani, A. 2018 NAACL 'Self-Attention
    with Relative Position Representations' (arXiv:1803.02155) -
    establishes that relative-position biases (not absolute) are what
    alter softmax attention, justifying the delta-dependent form here.

Public surface
--------------
- :func:`radial12_head_angles`  per-head angular phases (period 12)
- :class:`RadialSymmetry12Attention`  MHA with fixed 12-fold angular bias
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI

__all__ = ["radial12_head_angles", "RadialSymmetry12Attention"]

_RADIAL_FOLD = 12


def radial12_head_angles(n_heads: int) -> torch.Tensor:
    """Return a ``(n_heads,)`` tensor of per-head angular phases.

    Head ``h`` is assigned phase ``2*pi*(h % 12)/12``; the phase pattern
    therefore repeats with period 12 along the head axis. ``n_heads``
    must be a positive multiple of 12 so each of the 12 radial sectors
    receives the same number of heads.
    """
    if n_heads < _RADIAL_FOLD or n_heads % _RADIAL_FOLD != 0:
        raise ValueError(
            f"n_heads must be a positive multiple of 12 (got {n_heads})"
        )
    h = torch.arange(n_heads, dtype=torch.float32)
    return 2.0 * math.pi * (h % _RADIAL_FOLD) / float(_RADIAL_FOLD)


class RadialSymmetry12Attention(nn.Module):
    """Multi-head attention with 12 heads (or a multiple) arranged in a
    12-fold radial symmetry.

    Each head ``h`` adds a fixed angular bias to the attention logits at
    relative position ``delta = j - i`` (key minus query):

        ``bias[h, i, j] = (1 / PHI) * cos(2*pi*(h % 12)/12 + 2*pi*delta/L)``

    The bias is a registered buffer (NOT a parameter), so the 12-fold
    structure is purely architectural. Because it depends on ``delta`` it
    is *not* cancelled by the softmax (a constant per-row shift would be),
    and it is exactly periodic with period ``L`` so the attention map is
    equivariant under cyclic position shifts: a shift of ``L/12`` cyclically
    permutes the head phases.

    When ``radial=False`` the bias term is dropped entirely and the module
    behaves as standard multi-head attention.

    Shape convention: input ``x`` is ``(B, N, D)`` with ``D`` divisible by
    ``n_heads`` and ``n_heads`` a positive multiple of 12 (default 12).
    Output is ``(B, N, D)``.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    n_heads : int, default 12
        Number of attention heads; a positive multiple of 12.
    radial : bool, default True
        If False, fall back to plain MHA (no angular bias).
    bias : bool, default False
        Whether the QKV / output projections have a bias term.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 12,
        radial: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if n_heads < _RADIAL_FOLD or n_heads % _RADIAL_FOLD != 0:
            raise ValueError(
                f"n_heads must be a positive multiple of 12 (got {n_heads})"
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.radial = radial
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        # Per-head fixed angular phase (12-fold). Buffer, not a Parameter,
        # so it is structural and moves with .to(device).
        self.register_buffer("head_angles", radial12_head_angles(n_heads))

    def _relative_bias(self, L: int, device, dtype) -> torch.Tensor:
        """Return the ``(n_heads, L, L)`` relative-position angular bias."""
        idx = torch.arange(L, device=device, dtype=dtype)
        rel = idx.view(1, L) - idx.view(L, 1)  # (L, L): j - i
        phase = 2.0 * math.pi * rel / float(L)  # (L, L)
        angles = self.head_angles.to(device=device, dtype=dtype).view(self.n_heads, 1, 1)
        return torch.cos(angles + phase.view(1, L, L)) / float(PHI)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, N, D), got {tuple(x.shape)}")
        B, N, D = x.shape
        if D != self.d_model:
            raise ValueError(f"got D={D}, expected {self.d_model}")
        qkv = self.qkv(x)  # (B, N, 3D)
        qkv = qkv.view(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, h, N, head_dim) each
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.radial:
            bias = self._relative_bias(N, x.device, x.dtype)  # (h, N, N)
            attn = attn + bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, h, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, radial={self.radial}"
        )


# TODO runner wiring:
#   - models.py: add an optional `radial12_attn=True` config branch that
#     swaps a ViT-Tiny block's MHA for RadialSymmetry12Attention (n_heads
#     forced to a multiple of 12). Image track is patch-token sequence.
#   - configs/cifar10_quick.yaml: add a `radial12_attn` flag so the
#     ablation row carries a distinct tag. Not CNN-droppable into
#     ResNet-20 (attention module), so no sweep row is expected by default.
#   - run_sweep.py: gate the row on a positive SOTA-smoke pre-flight (Rule 13).
