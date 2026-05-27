"""H27 — Golden Spiral Graph.

Initialise node-embedding tables on a 2-D golden-spiral lattice
(Vogel 1979: ``r_k = sqrt(k+1)``, ``theta_k = k * golden_angle``)
then lift to embedding dimension ``D`` via a seeded random orthonormal
projection. This gives a maximally isotropic discrete sampling of the
disc — every node starts approximately equidistant from its
k-nearest-neighbors with no Gaussian-tail concentration.

This module exposes:

- :func:`golden_spiral_node_init_` — in-place overwrite of an ``(N, D)``
  embedding tensor with the spiral lattice lifted to D dims.
- :class:`SpiralGraphTransformerLayer` — a minimal graph-transformer
  layer (self-attention + FFN + LayerNorm) whose node-position bias is
  drawn from a learnable embedding initialised by the helper.

See ``hypotheses/g3_topologies_graphs/H27_golden_spiral_graph.md``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nature_inspired_networks.priors import PHI


GOLDEN_ANGLE_RAD = 2 * math.pi * (1.0 - 1.0 / PHI)  # ≈ 2.3999632...


def _golden_spiral_2d(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the canonical 2-D golden-spiral lattice as an ``(n, 2)`` tensor.

    Position k = ``(r_k * cos(k*ga), r_k * sin(k*ga))`` with
    ``r_k = sqrt(k+1)`` so the disc is sampled with approximately
    constant Voronoi-cell area (Vogel 1979).
    """
    k = torch.arange(n, dtype=dtype)
    r = torch.sqrt(k + 1.0)
    theta = k * GOLDEN_ANGLE_RAD
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)


def _seeded_orthonormal_projection(
    in_dim: int,
    out_dim: int,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a seeded random orthonormal ``(in_dim, out_dim)`` projection.

    For ``out_dim >= in_dim`` the projection has orthonormal columns
    (an isometry); for ``out_dim < in_dim`` it has orthonormal rows
    (the columns can no longer be orthonormal). Either way the matrix
    has unit singular values, so it preserves Euclidean distances up
    to a global scale.
    """
    g = torch.Generator().manual_seed(int(seed))
    # Always factor the larger axis so we get the appropriate isometry.
    M = max(in_dim, out_dim)
    A = torch.randn(M, M, generator=g, dtype=dtype)
    q, _ = torch.linalg.qr(A, mode="reduced")
    # Slice down to (in_dim, out_dim).
    return q[:in_dim, :out_dim].contiguous()


def golden_spiral_node_init_(
    emb: torch.Tensor,
    seed: int = 0,
    scale: float = 1.0,
) -> torch.Tensor:
    """In-place overwrite ``emb`` (shape ``(N, D)``) with a golden-spiral
    node embedding.

    The 2-D spiral lattice is computed at the input ``N``, normalised
    to unit per-coordinate standard deviation, optionally rescaled
    by ``scale``, then lifted to D dimensions via a seeded random
    orthonormal projection. The lift is deterministic for a given
    ``(seed, N, D)``.

    Returns the same tensor (for chaining).
    """
    assert emb.dim() == 2, f"expected (N, D); got shape {tuple(emb.shape)}"
    N, D = emb.shape
    assert N >= 1 and D >= 1, f"N, D must be >=1; got {N}, {D}"

    spiral = _golden_spiral_2d(N, dtype=emb.dtype)
    # Per-coordinate standardisation so the lift starts from a balanced basis.
    if N > 1:
        mean = spiral.mean(dim=0, keepdim=True)
        std = spiral.std(dim=0, keepdim=True).clamp(min=1e-8)
        spiral = (spiral - mean) / std
    spiral = spiral * scale  # (N, 2)

    proj = _seeded_orthonormal_projection(2, D, seed=seed, dtype=emb.dtype)
    lifted = spiral @ proj  # (N, D)

    with torch.no_grad():
        emb.copy_(lifted.to(device=emb.device))
    return emb


class SpiralGraphTransformerLayer(nn.Module):
    """Minimal graph-transformer layer: pre-norm self-attention + FFN,
    with a learnable node-position embedding initialised on the golden
    spiral (lifted to ``d_model``).

    The layer is dense self-attention over all ``n_nodes`` nodes (no
    sparsity / mask) — appropriate for small fixed-size graphs (e.g.
    icosahedral vertex sets, small molecules); the spiral init is the
    payload, not the attention sparsity pattern.

    Parameters
    ----------
    d_model : int
        Embedding / hidden dimension.
    n_nodes : int
        Number of graph nodes; the position embedding has shape
        ``(n_nodes, d_model)``.
    n_heads : int, default 4
        Number of attention heads. Must divide ``d_model``.
    ffn_mult : int, default 4
        FFN expansion factor (``d_ff = d_model * ffn_mult``).
    spiral_seed : int, default 0
        Seed for the orthonormal projection used by the golden-spiral
        node init.
    dropout : float, default 0.0
        Dropout probability inside attention + FFN.
    """

    def __init__(
        self,
        d_model: int,
        n_nodes: int,
        n_heads: int = 4,
        ffn_mult: int = 4,
        spiral_seed: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model > 0 and n_nodes > 0
        assert d_model % n_heads == 0, (
            f"d_model={d_model} must be divisible by n_heads={n_heads}"
        )
        self.d_model = d_model
        self.n_nodes = n_nodes

        # Golden-spiral node positional embedding.
        pos = torch.empty(n_nodes, d_model)
        golden_spiral_node_init_(pos, seed=spiral_seed)
        self.pos_embedding = nn.Parameter(pos)

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        d_ff = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, N, D) tensor of node features.

        Returns
        -------
        (B, N, D) tensor.
        """
        assert x.dim() == 3, f"expected (B, N, D); got {tuple(x.shape)}"
        B, N, D = x.shape
        assert N == self.n_nodes and D == self.d_model, (
            f"shape mismatch: got (B={B}, N={N}, D={D}); "
            f"expected (B, {self.n_nodes}, {self.d_model})"
        )
        # Add (broadcast) the learnable spiral position embedding.
        h = x + self.pos_embedding.unsqueeze(0)
        # Pre-norm attention + residual.
        h_norm = self.ln1(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        h = h + attn_out
        # Pre-norm FFN + residual.
        h = h + self.ffn(self.ln2(h))
        return h
