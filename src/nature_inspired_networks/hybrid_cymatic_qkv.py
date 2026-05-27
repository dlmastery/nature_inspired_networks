"""H66 — Cymatic QKV Kernel (G7 cross-paradigm hybrid).

Design doc: ``hypotheses/g7_cross_paradigm_hybrids/H66_cymatic_qkv_kernel.md``.

Initialises the Q, K, V projection matrices of an attention module with
**Chladni eigenmode** weights via :func:`cymatic_init_` configured for
``band=(2, 5), orthonormalize=True``. The intuition is that the Chladni
basis is a frequency-band-limited orthogonal basis on the kernel grid;
when it seeds the attention projections it (i) decorrelates the heads at
init (one mode per head) and (ii) injects natural-system standing-wave
structure into the learned attention basis.

The cymatic-init utility was designed for ``nn.Conv2d`` weights of shape
``(out_c, in_c, k, k)``. Linear weights of shape ``(out_dim, in_dim)``
are temporarily reshaped to ``(out_dim, in_dim, 1, 1)`` and then to
``(out_dim, 1, k, k)`` for the cymatic seeding, where ``k = ceil(sqrt(
in_dim))`` is the smallest square that fits ``in_dim`` features. The
filled cymatic kernel is then linearly indexed back into the ``(out_dim,
in_dim)`` matrix; this preserves the Chladni-eigenmode signature in the
matrix's row space while keeping the API a plain ``nn.Linear``.

References (Citation Rigor)::

    Chladni 1787 'Entdeckungen über die Theorie des Klanges' -- the
    sand-on-plate experiments whose discrete eigenmode basis we
    operationalise.

    Vaswani et al. 2017 NeurIPS 'Attention Is All You Need'
    (arXiv:1706.03762) -- the attention QKV projection whose init we
    replace.

    Glorot, Bengio 2010 AISTATS 'Understanding the difficulty of
    training deep feedforward neural networks' -- the standard
    init baseline that H66 replaces with a band-limited Chladni init.

Public surface
--------------
- :class:`CymaticQKVKernel` -- multi-head attention whose QKV projection
  weights are Chladni-orthonormalised at construction time.
- :func:`cymatic_init_linear_` -- helper that applies the cymatic seed
  to an arbitrary ``nn.Linear`` weight matrix.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI, cymatic_init_


__all__ = [
    "CymaticQKVKernel",
    "cymatic_init_linear_",
]


def cymatic_init_linear_(
    linear: nn.Linear,
    band: tuple[int, int] = (2, 5),
    orthonormalize: bool = True,
    seed: int = 0,
) -> None:
    """Apply :func:`cymatic_init_` to an ``nn.Linear`` weight matrix.

    The weight matrix is shape ``(out, in)``. We reshape it to a
    Conv2d-like ``(out, 1, k, k)`` tensor with ``k = ceil(sqrt(in))``,
    seed it via :func:`cymatic_init_`, then copy the (out, in) prefix
    of the flattened result back into the linear weight.

    Parameters
    ----------
    linear : nn.Linear
        Module to re-initialise in place.
    band : (low, high)
        Frequency band passed to :func:`cymatic_init_` (default ``(2, 5)``).
    orthonormalize : bool, default True
        Whether the Chladni basis is Gram-Schmidt orthonormalised across
        the output channels.
    seed : int, default 0
        Forwarded for reproducibility.
    """
    out_dim, in_dim = linear.weight.shape
    k = int(math.ceil(math.sqrt(in_dim)))
    if k < 3:
        # Cymatic basis is only defined for k >= 3 grids; fall back to
        # Xavier for trivially-small layers.
        nn.init.xavier_uniform_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
        return
    # Build a dummy conv whose weight has shape (out, 1, k, k) and seed it.
    dummy = nn.Conv2d(1, out_dim, kernel_size=k, bias=False)
    cymatic_init_(dummy, orthonormalize=orthonormalize, band=band, seed=seed)
    # Flatten the k*k kernel grid and take the leading in_dim entries.
    flat = dummy.weight.reshape(out_dim, k * k)[:, :in_dim]  # (out_dim, in_dim)
    with torch.no_grad():
        linear.weight.copy_(flat)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)


class CymaticQKVKernel(nn.Module):
    """Multi-head attention with cymatic-initialised QKV projections.

    Forward semantics match standard MHA with ``batch_first=True``:
    ``forward(x): (B, N, D) -> (B, N, D)``.

    Parameters
    ----------
    dim : int
        Embedding dimension. Must be divisible by ``n_heads``.
    n_heads : int, default 4
        Number of attention heads.
    band : (low, high), default ``(2, 5)``
        Chladni frequency band forwarded to :func:`cymatic_init_`.
    orthonormalize : bool, default True
        Whether the Chladni basis is Gram-Schmidt orthonormalised.
    dropout : float, default 0.0
        Attention dropout.
    bias : bool, default True
        Whether QKV / output projections carry a bias term.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        band: tuple[int, int] = (2, 5),
        orthonormalize: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = dim // n_heads
        self.band = (int(band[0]), int(band[1]))
        self.orthonormalize = bool(orthonormalize)
        # Standard three-projection layout (kept separate so each gets a
        # distinct Chladni seed via different seeds).
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """(Re-)initialise QKV projections with the cymatic seed.

        Each of Q/K/V receives a *distinct* seed so the three matrices
        are independently band-limited Chladni-orthogonal projections.
        The output projection is left at its default Xavier init.
        """
        cymatic_init_linear_(
            self.q_proj, band=self.band, orthonormalize=self.orthonormalize, seed=0
        )
        cymatic_init_linear_(
            self.k_proj, band=self.band, orthonormalize=self.orthonormalize, seed=1
        )
        cymatic_init_linear_(
            self.v_proj, band=self.band, orthonormalize=self.orthonormalize, seed=2
        )
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, N, D), got shape {tuple(x.shape)}")
        B, N, D = x.shape
        if D != self.dim:
            raise ValueError(f"dim mismatch: x has {D}, module has {self.dim}")
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)
