r"""H32 — Fibonacci Dilation Attention (Fibottention).

Implements the *Fibottention* attention pattern of Rajagopalan et al.
2024 (arXiv:2406.19391): a multi-head attention module where each head
uses a **different Fibonacci dilation** ``F_h \in {1, 2, 3, 5, 8, 13,
21, 34, ...}``. Within head ``h``, query position ``i`` attends only
to key positions ``j`` such that ``|i - j| \in {0, F_h, 2·F_h,
3·F_h, ...}``. This realises a sparse multi-scale attention pattern
whose per-head density is ``O(1/F_h)`` and whose union across heads
covers the integer lattice via the **Wythoff array** non-overlap
property (Wythoff 1907).

API
---
:func:`fibonacci_dilations`
    Return the first ``n_heads`` Fibonacci numbers ``[1, 2, 3, 5, 8,
    13, 21, 34, 55, 89]`` as the per-head dilation list.
:func:`wythoff_pattern`
    Given ``(seq_len, n_heads, dilations)`` build per-head sparse
    attention masks of shape ``(n_heads, seq_len, seq_len)`` (bool).
    Head ``h`` attends to position ``j`` from ``i`` iff
    ``(j - i) % dilations[h] == 0`` (with the convention that the
    self-position ``i == j`` is always included). Returned as a
    ``torch.BoolTensor`` where ``True`` means *attend*.
:class:`Fibottention`
    Multi-head attention with per-head dilated sparse masks. When
    ``dilations`` is all-ones the module collapses to standard MHA.

The CIFAR-track entry-point is the ``Fibottention`` module which is
shape-compatible with PyTorch's ``nn.MultiheadAttention``: takes
``(B, N, D)``, returns ``(B, N, D)``.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI  # re-exported for downstream imports


__all__ = [
    "fibonacci_dilations",
    "wythoff_pattern",
    "Fibottention",
]


def fibonacci_dilations(n_heads: int = 6) -> list[int]:
    """Return the first ``n_heads`` Fibonacci numbers starting at 1, 2.

    Examples
    --------
    >>> fibonacci_dilations(6)
    [1, 2, 3, 5, 8, 13]
    >>> fibonacci_dilations(8)
    [1, 2, 3, 5, 8, 13, 21, 34]
    """
    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {n_heads}")
    fib = [1, 2]
    while len(fib) < n_heads:
        fib.append(fib[-1] + fib[-2])
    return fib[:n_heads]


def wythoff_pattern(
    seq_len: int,
    n_heads: int,
    dilations: Sequence[int] | None = None,
) -> torch.Tensor:
    """Build the per-head Wythoff-style sparse attention masks.

    Head ``h`` attends to position ``j`` from query ``i`` iff
    ``(j - i) % dilations[h] == 0``. The self-position ``i == j`` is
    always included (it falls out naturally from the modulo rule since
    ``0 % d == 0`` for any ``d >= 1``).

    Parameters
    ----------
    seq_len : int
        Sequence length ``N``.
    n_heads : int
        Number of attention heads.
    dilations : sequence of int, optional
        Per-head dilation factors. If ``None``, uses
        ``fibonacci_dilations(n_heads)``.

    Returns
    -------
    torch.BoolTensor
        Shape ``(n_heads, seq_len, seq_len)``. ``True`` at
        ``(h, i, j)`` means head ``h`` attends from query ``i`` to
        key ``j``.

    Notes
    -----
    The non-overlap property required by the Wythoff array is
    realised at the *off-diagonal* level: distinct primes / coprime
    dilations produce per-head subsets of ``{1, ..., N-1}`` whose
    pairwise intersections are sparse. Strict disjointness requires
    using consecutive Fibonacci numbers (which are pairwise coprime by
    Lucas's theorem) — the default dilation list satisfies this.
    """
    if dilations is None:
        dilations = fibonacci_dilations(n_heads)
    if len(dilations) != n_heads:
        raise ValueError(
            f"dilations length {len(dilations)} != n_heads {n_heads}"
        )
    if any(d < 1 for d in dilations):
        raise ValueError(f"all dilations must be >= 1, got {list(dilations)}")

    masks = torch.zeros(n_heads, seq_len, seq_len, dtype=torch.bool)
    idx = torch.arange(seq_len)
    diff = idx.unsqueeze(0) - idx.unsqueeze(1)  # (N, N), diff[i, j] = j - i
    for h, d in enumerate(dilations):
        if d == 1:
            masks[h].fill_(True)  # dense head — every position attends to every other
        else:
            masks[h] = (diff.abs() % d) == 0
    return masks


class Fibottention(nn.Module):
    """Multi-head attention with per-head Fibonacci-dilation sparse masks.

    Mirrors the API of ``nn.MultiheadAttention`` with ``batch_first=True``
    semantics: ``forward(x)`` accepts ``(B, N, D)`` and returns
    ``(B, N, D)``. The QKV projection, output projection, and head
    splitting follow the standard Vaswani 2017 recipe.

    Parameters
    ----------
    dim : int
        Embedding dimension ``D``.
    n_heads : int, default 6
        Number of attention heads. Must divide ``dim``.
    dilations : sequence of int, optional
        Per-head dilation factors. Defaults to
        ``fibonacci_dilations(n_heads)``.
    dropout : float, default 0.0
        Attention-softmax dropout.
    bias : bool, default True
        Whether the QKV / output projections carry a bias term.

    Notes
    -----
    * When ``dilations = [1] * n_heads`` every head is dense and the
      module is numerically equivalent to standard MHA (modulo the
      per-head softmax which is identical given an all-True mask).
    * The masks are constructed lazily per ``forward`` against the
      observed ``N`` and cached on the module so repeated calls at the
      same sequence length are O(1).
    * Dropping a sparse mask onto softmax is implemented by additive
      ``-inf`` on disallowed positions, NOT multiplicative — this
      preserves the softmax normalisation across the kept positions.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 6,
        dilations: Sequence[int] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(
                f"dim {dim} not divisible by n_heads {n_heads}"
            )
        if dilations is None:
            dilations = fibonacci_dilations(n_heads)
        dilations = list(dilations)
        if len(dilations) != n_heads:
            raise ValueError(
                f"len(dilations)={len(dilations)} != n_heads={n_heads}"
            )
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dilations = dilations
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self._cached_seq_len: int = -1
        self.register_buffer(
            "_mask_cache",
            torch.zeros(n_heads, 1, 1, dtype=torch.bool),
            persistent=False,
        )

    @property
    def is_dense_fallback(self) -> bool:
        """True iff every head's dilation is 1 (i.e., standard MHA)."""
        return all(d == 1 for d in self.dilations)

    def _build_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._cached_seq_len != seq_len or self._mask_cache.device != device:
            mask = wythoff_pattern(seq_len, self.n_heads, self.dilations).to(device)
            # store on buffer so it travels with .to(device) / state_dict
            self._mask_cache = mask
            self._cached_seq_len = seq_len
        return self._mask_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, N, D), got shape {tuple(x.shape)}")
        B, N, D = x.shape
        if D != self.dim:
            raise ValueError(f"dim mismatch: x has {D}, module has {self.dim}")
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (B, H, N, N)
        mask = self._build_mask(N, x.device)  # (H, N, N)
        if not self.is_dense_fallback:
            # additive -inf on disallowed entries; broadcasts B over H
            attn = attn.masked_fill(~mask.unsqueeze(0), float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, N, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, N, D)
        return self.proj(out)
