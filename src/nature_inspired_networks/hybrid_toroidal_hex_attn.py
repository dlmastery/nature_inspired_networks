"""H62 — Toroidal-KV Hex Attention (G7 cross-paradigm hybrid).

Design doc: ``hypotheses/g7_cross_paradigm_hybrids/H62_toroidal_kv_hex_attention.md``.

Combines three already-implemented sacred-geometry primitives into one
attention module:

  1. **Toroidal KV cache** — the K and V projections are reshaped to a 2-D
     grid (``H x W``) and wrapped with :func:`toroidal_pad`, so positional
     modular shifts within the cache are exact (no boundary effects).
  2. **Hex attention window** — only positions inside a hex-mask window
     around each query token contribute to attention. The hex mask uses
     :func:`hex_kernel_mask` from priors at the configured radius.
  3. **Query stays standard** — Q is unchanged so the API is compatible
     with downstream layers.

This is the *reference implementation* of the hex+toroidal attention
sub-claim of H62; Fibonacci-ratio pruning is handled separately by
:mod:`nature_inspired_networks.pruning` and is composed at the trainer
level (Rule 14: shared primitives, not duplicated code).

References (Citation Rigor)::

    Hoogeboom, Peters, Cohen, Welling 2018 ECCV 'HexaConv'
    (arXiv:1803.02108) -- the canonical hex-lattice convolution whose
    discrete-Fourier formulation we adopt for the attention window.

    Pittorino, Ferraro, Perugini, Feinauer, Baldassi, Borghesi 2022
    Phys. Rev. E 'Deep networks on toroids' (arXiv:2202.03038) -- why
    wrapping the cache on a torus exploits modular translation invariance.

Public surface
--------------
- :class:`ToroidalKVHexAttention` — multi-head attention with toroidal KV
  reshape and hex window masking. Forward is shape-compatible with the
  Fibottention API: takes ``(B, N, D)`` returns ``(B, N, D)``.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import hex_kernel_mask, toroidal_pad


__all__ = ["ToroidalKVHexAttention"]


def _factor_seq_len(n: int) -> Tuple[int, int]:
    """Return ``(h, w)`` with ``h * w == n`` and ``h`` as close to ``sqrt(n)``.

    Falls back to ``(1, n)`` when no perfect rectangular factorisation
    exists; this keeps the module usable for non-square sequence lengths,
    just at the cost of the toroidal wrap being "1-D circular" instead of
    "2-D toroidal" in that case.
    """
    h = int(round(math.sqrt(n)))
    while h > 1 and (n % h) != 0:
        h -= 1
    if h <= 0:
        h = 1
    w = n // h
    return h, w


class ToroidalKVHexAttention(nn.Module):
    """Hex-window attention with toroidal-padded KV cache.

    Parameters
    ----------
    dim : int
        Embedding dimension ``D``.
    n_heads : int, default 4
        Number of attention heads. Must divide ``dim``.
    hex_kernel_radius : int, default 1
        Hex window radius. ``1`` → 3x3 mask (7 taps, 180-sym). ``2`` →
        5x5 mask (19 taps, true 6-fold isotropic).
    dropout : float, default 0.0
        Attention-softmax dropout.
    bias : bool, default True
        Whether the QKV / output projections carry a bias term.

    Forward shape
    -------------
    ``forward(x): (B, N, D) -> (B, N, D)``. ``N`` is factored to a 2-D
    ``H x W`` grid; non-square ``N`` falls back to ``(1, N)`` (toroidal
    wrap becomes 1-D circular in that case).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        hex_kernel_radius: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        if hex_kernel_radius not in (1, 2):
            raise ValueError(
                f"hex_kernel_radius must be 1 or 2, got {hex_kernel_radius}"
            )
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = dim // n_heads
        self.hex_kernel_radius = int(hex_kernel_radius)
        # 3x3 (radius 1) or 5x5 (radius 2) hex mask.
        k = 3 if hex_kernel_radius == 1 else 5
        self.k = k
        self.register_buffer("hex_mask", hex_kernel_mask(k), persistent=False)
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)

    def _toroidal_neighbour_grid(self, kv: torch.Tensor) -> torch.Tensor:
        """Reshape ``kv`` to ``(B, H_total, k, k, D)`` of hex neighbours.

        ``kv`` shape: ``(B, N, D)``. We reshape to ``(B, h, w, D)``, apply
        :func:`toroidal_pad` with pad ``=k//2`` over the spatial dims,
        then unfold a ``(k, k)`` window centred on each position. The
        resulting tensor has shape ``(B, h*w, k, k, D)``.
        """
        B, N, D = kv.shape
        h, w = _factor_seq_len(N)
        pad = self.k // 2
        # (B, D, h, w) for toroidal_pad which expects 4-D conv layout.
        grid = kv.transpose(1, 2).reshape(B, D, h, w)
        padded = toroidal_pad(grid, pad)  # (B, D, h+2pad, w+2pad)
        # unfold over the two spatial dims → (B, D, h, w, k, k)
        unfolded = padded.unfold(2, self.k, 1).unfold(3, self.k, 1)
        # → (B, h, w, k, k, D) then flatten the spatial axis.
        unfolded = unfolded.permute(0, 2, 3, 4, 5, 1).contiguous()
        unfolded = unfolded.reshape(B, h * w, self.k, self.k, D)
        return unfolded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, N, D), got shape {tuple(x.shape)}")
        B, N, D = x.shape
        if D != self.dim:
            raise ValueError(f"dim mismatch: x has {D}, module has {self.dim}")
        qkv = self.qkv(x)  # (B, N, 3D)
        q, k, v = qkv.split(self.dim, dim=-1)
        # Toroidal-wrapped neighbour grids for K and V. Shape: (B, N, k, k, D).
        k_nbr = self._toroidal_neighbour_grid(k)
        v_nbr = self._toroidal_neighbour_grid(v)
        # Multi-head reshape on K/V neighbours.
        kk = self.k
        # (B, N, kk*kk, n_heads, head_dim)
        k_nbr = k_nbr.reshape(B, N, kk * kk, self.n_heads, self.head_dim)
        v_nbr = v_nbr.reshape(B, N, kk * kk, self.n_heads, self.head_dim)
        # Q: (B, N, n_heads, head_dim) → expand for nbr broadcast.
        q_h = q.reshape(B, N, self.n_heads, self.head_dim)
        # Attention scores: per query position vs its kk*kk neighbours.
        # (B, N, kk*kk, n_heads)
        scores = (q_h.unsqueeze(2) * k_nbr).sum(-1) / math.sqrt(self.head_dim)
        # Hex mask over the kk*kk window — gate out the corner positions.
        mask = self.hex_mask.to(scores.device).reshape(kk * kk)  # (kk*kk,)
        # Additive -inf on masked positions (preserves softmax).
        scores = scores.masked_fill(mask.view(1, 1, kk * kk, 1) == 0, float("-inf"))
        attn = scores.softmax(dim=2)  # softmax over neighbours
        attn = self.attn_dropout(attn)
        # Weighted sum: (B, N, kk*kk, n_heads, head_dim) → (B, N, n_heads, head_dim)
        out = (attn.unsqueeze(-1) * v_nbr).sum(dim=2)
        out = out.reshape(B, N, D)
        return self.proj(out)
