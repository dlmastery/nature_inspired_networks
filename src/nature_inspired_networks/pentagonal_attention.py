"""H37 - Pentagonal phi-Attention (multi-head attention with 5-fold heads).

The pentagon / dodecahedron is the canonical 5-fold-symmetry geometric
form: starfish anatomy, the C5 viral capsid axis, dodecahedral
liquid-crystal phases, and Penrose-tile aperiodic packings all live in
D5. Every diagonal of a regular pentagon is exactly phi times its side
and the pentagon's vertex coordinates contain phi in closed form.

For Transformer multi-head attention, the standard practice is uniform
head allocation. **Pentagonal head allocation** groups heads in 5 / 10 /
20 - matching the dodecahedral / icosahedral vertex counts. The shared
primitive here is a fixed (non-learnable) per-head angular bias that adds
``2 * pi * k / 5`` to the attention logits of the heads in group ``k``,
applied as a relative-positional bias along the sequence axis. The
biases are derived purely from each head's group index so the structure
is parameter-free and rotation-symmetric in 5-fold steps.

Refs (Citation Rigor):
    Cohen, T. S., et al. 2019 ICML 'Icosahedral CNN' (arXiv:1902.04615)
    - icosahedral equivariance; dodecahedral is the dual.
    Dosovitskiy, A., et al. 2021 ICLR 'An Image is Worth 16x16 Words'
    (arXiv:2010.11929) - ViT reference H37 modifies.
    Vaswani, A., et al. 2017 NeurIPS 'Attention Is All You Need'
    (arXiv:1706.03762) - original MHA reference.

Public surface
--------------
- :func:`pentagonal_head_groups`  list of head-group indices (5 groups)
- :class:`PentagonalAttention`    MHA with fixed per-head angular bias
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI


def pentagonal_head_groups(n_heads: int = 10) -> list[list[int]]:
    """Partition ``n_heads`` heads into 5 dodeca-vertex-aligned groups.

    ``n_heads`` must be divisible by 5. Head ``h`` is assigned to group
    ``h % 5`` (round-robin), so groups are interleaved rather than
    contiguous; this matches the canonical dodecahedral vertex-orbit
    layout where each of the 5 pentagons of the dodecahedron contains
    one of the 5 "facets" rather than 5 consecutive heads.

    Returns a list of 5 sub-lists; sub-list ``k`` contains the head
    indices assigned to group ``k``. Each sub-list has length
    ``n_heads // 5``.
    """
    if n_heads < 5 or n_heads % 5 != 0:
        raise ValueError(
            f"n_heads must be a positive multiple of 5 (got {n_heads})"
        )
    groups: list[list[int]] = [[] for _ in range(5)]
    for h in range(n_heads):
        groups[h % 5].append(h)
    return groups


def _head_group_angles(n_heads: int) -> torch.Tensor:
    """Return a ``(n_heads,)`` tensor where head ``h`` has angular phase
    ``2 * pi * (h % 5) / 5`` (i.e. its dodeca-vertex angle). The phase
    repeats with period 5 along the head axis.
    """
    groups = pentagonal_head_groups(n_heads)
    angles = torch.zeros(n_heads, dtype=torch.float32)
    for k, members in enumerate(groups):
        angle = 2 * math.pi * k / 5.0
        for h in members:
            angles[h] = angle
    return angles


class PentagonalAttention(nn.Module):
    """Multi-head attention where each head has a fixed angular bias
    proportional to ``2 * pi * (h % 5) / 5``.

    The bias is broadcast across the (query, key) plane and added to the
    pre-softmax logits, so heads in the same pentagonal group see the
    same constant additive shift. The bias is registered as a buffer
    (NOT a parameter) so it is structural, not learned.

    Shape convention: input ``x`` is ``(B, L, d)``. ``d`` must be
    divisible by ``n_heads``; ``n_heads`` must be a positive multiple of
    5 (default 10, matching the Petersen graph / 10 dodeca vertex
    classes). Output is ``(B, L, d)``.

    The per-head bias added to attention logits at relative position
    ``delta = j - i`` (key index minus query index) is

        ``head_bias[h, i, j] = (1 / PHI) * cos(angle_h + 2*pi*delta / L)``

    where ``angle_h = 2*pi * (h % 5) / 5`` is the head's dodeca-vertex
    angle. This is a *relative*-positional bias that varies with
    ``j - i`` and therefore changes softmax outputs (a constant bias
    would be cancelled by the softmax normalisation). It is also
    exactly periodic with period ``L``, so the attention map is
    equivariant under cyclic position shifts of any integer amount.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    n_heads : int
        Number of attention heads; multiple of 5 (default 10).
    bias : bool
        Whether the QKV / output projections have a bias term (default
        False, matching common ViT-Tiny conventions).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 10,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if n_heads < 5 or n_heads % 5 != 0:
            raise ValueError(
                f"n_heads must be a positive multiple of 5 (got {n_heads})"
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        # Per-head fixed angular phase (5-fold). Registered as buffer so
        # it is structural, not learned, and moves with .to(device).
        self.register_buffer("head_bias", _head_group_angles(n_heads))

    def _relative_bias(self, L: int, device, dtype) -> torch.Tensor:
        """Compute the relative-positional bias tensor of shape
        ``(n_heads, L, L)`` from the per-head angular phases. The bias
        at ``(h, i, j)`` is ``(1 / PHI) * cos(angle_h + 2*pi * (j - i) / L)``.
        """
        idx = torch.arange(L, device=device, dtype=dtype)
        rel = idx.view(1, L) - idx.view(L, 1)  # (L, L): j - i
        phase = 2.0 * math.pi * rel / float(L)  # (L, L)
        # head_bias has shape (n_heads,); broadcast over (L, L)
        angles = self.head_bias.to(device=device, dtype=dtype).view(self.n_heads, 1, 1)
        return torch.cos(angles + phase.view(1, L, L)) / float(PHI)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, L, d), got {tuple(x.shape)}")
        B, L, d = x.shape
        if d != self.d_model:
            raise ValueError(f"got d={d}, expected {self.d_model}")
        qkv = self.qkv(x)  # (B, L, 3d)
        qkv = qkv.view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, h, L, head_dim) each
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Relative-positional per-head angular bias, broadcast over batch.
        bias = self._relative_bias(L, x.device, x.dtype)  # (h, L, L)
        attn = attn + bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, h, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, d)
        return self.proj(out)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}, head_dim={self.head_dim}"
