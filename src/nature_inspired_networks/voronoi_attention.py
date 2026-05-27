"""H82 — Voronoi Sparse Attention.

Build an irregular sparse attention mask from a Voronoi / Delaunay
tessellation of ``N`` points scattered (deterministically, by seed) in
the unit square. Token ``i`` attends only to tokens whose Voronoi cells
are Delaunay-adjacent to cell ``i`` (plus a self-loop). Delaunay
adjacency is the dual of the Voronoi diagram, so this yields a planar,
locally-irregular sparse connectivity: high-degree hubs near dense point
clusters and low-degree nodes near the boundary — an interlocking
"polygonal-masonry" pattern rather than a regular grid window.

Esoteric origin (acknowledged in one sentence): the interlocking
"polygonal masonry" topology motivates the irregular tessellation; the
implementation is a standard Delaunay-adjacency sparse attention mask.

When SciPy's ``scipy.spatial`` is unavailable, the import is guarded and
a symmetric k-nearest-neighbour adjacency on the same random points is
used as a Delaunay proxy (mean Delaunay degree in 2-D is ~6, so the
default ``knn=6`` matches the expected sparsity).

Refs (Citation Rigor):
    Child, R., Gray, S., Radford, A., Sutskever, I. 2019 'Generating
    Long Sequences with Sparse Transformers' (arXiv:1904.10509) -
    establishes that fixed sparse attention masks recover most dense
    performance at far lower cost; this module supplies a Delaunay mask.
    Beltagy, I., Peters, M. E., Cohan, A. 2020 'Longformer: The
    Long-Document Transformer' (arXiv:2004.05150) - local+global sparse
    attention pattern that the irregular Voronoi mask generalises.

Public surface
--------------
- :func:`voronoi_adjacency`        fixed (N, N) Delaunay-adjacency mask
- :class:`VoronoiSparseAttention`  MHA with the fixed sparse mask applied
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI  # noqa: F401  (shared convention across primitives)

__all__ = ["voronoi_adjacency", "VoronoiSparseAttention"]


def _knn_adjacency(points: torch.Tensor, knn: int) -> torch.Tensor:
    """Symmetric binary kNN adjacency (Delaunay proxy) with self-loops."""
    n = points.shape[0]
    d = torch.cdist(points, points)  # (n, n)
    k = min(knn + 1, n)  # +1 because the nearest point is self
    # indices of the k smallest distances per row (includes self at dist 0)
    idx = d.topk(k, dim=1, largest=False).indices  # (n, k)
    A = torch.zeros(n, n)
    rows = torch.arange(n).unsqueeze(1).expand(-1, k)
    A[rows.reshape(-1), idx.reshape(-1)] = 1.0
    # symmetrise (kNN is not symmetric in general)
    A = ((A + A.t()) > 0).float()
    A.fill_diagonal_(1.0)  # ensure self-loops
    return A


def voronoi_adjacency(n_tokens: int, seed: int = 0, knn: int = 6) -> torch.Tensor:
    """Return a fixed ``(N, N)`` binary Delaunay-adjacency mask.

    ``N`` points are sampled deterministically (seeded) in the unit
    square. The mask is 1 where two tokens' Voronoi cells share a
    Delaunay edge, and on the diagonal (self-loops). The matrix is
    symmetric.

    When ``scipy.spatial`` is importable the true Delaunay triangulation
    is used; otherwise a symmetric ``knn``-nearest-neighbour adjacency on
    the same points is used as a proxy (the import is guarded so the
    primitive has no hard SciPy dependency).

    Parameters
    ----------
    n_tokens : int
        Number of tokens / points ``N``.
    seed : int, default 0
        RNG seed for point placement (determinism).
    knn : int, default 6
        Neighbour count for the SciPy-free fallback (≈ mean 2-D Delaunay
        degree). Ignored when SciPy is available.
    """
    if n_tokens < 2:
        raise ValueError(f"n_tokens must be >= 2 (got {n_tokens})")
    g = torch.Generator().manual_seed(seed)
    points = torch.rand(n_tokens, 2, generator=g)

    A = torch.zeros(n_tokens, n_tokens)
    try:
        from scipy.spatial import Delaunay  # type: ignore

        # Delaunay needs >= 3 non-collinear points; for tiny N fall back.
        if n_tokens >= 4:
            tri = Delaunay(points.numpy())
            for simplex in tri.simplices:
                for a in simplex:
                    for b in simplex:
                        if a != b:
                            A[int(a), int(b)] = 1.0
            A = ((A + A.t()) > 0).float()
            A.fill_diagonal_(1.0)
            return A
    except Exception:
        # SciPy unavailable or triangulation failed → proxy below.
        pass
    return _knn_adjacency(points, knn)


class VoronoiSparseAttention(nn.Module):
    """Multi-head attention restricted to a fixed Voronoi/Delaunay sparse
    mask.

    The mask (a registered buffer, NOT a parameter) is built once from a
    seeded point tessellation; attention logits at masked-out
    (non-adjacent) token pairs are set to ``-inf`` before softmax, so
    each token only attends to its Delaunay neighbours and itself.

    Shape convention: input ``x`` is ``(B, N, D)`` where ``N`` equals the
    ``n_tokens`` the mask was built for, ``D`` divisible by ``n_heads``.
    Output is ``(B, N, D)``.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    n_tokens : int
        Sequence length the sparse mask is built for.
    n_heads : int, default 4
        Number of attention heads.
    seed : int, default 0
        Seed for the Voronoi point placement.
    knn : int, default 6
        Neighbour count for the SciPy-free fallback.
    bias : bool, default False
        Whether the QKV / output projections carry a bias term.
    """

    def __init__(
        self,
        d_model: int,
        n_tokens: int,
        n_heads: int = 4,
        seed: int = 0,
        knn: int = 6,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        # Fixed sparse adjacency mask (1 = allowed to attend).
        self.register_buffer(
            "attn_mask", voronoi_adjacency(n_tokens, seed=seed, knn=knn)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, N, D), got {tuple(x.shape)}")
        B, N, D = x.shape
        if D != self.d_model:
            raise ValueError(f"got D={D}, expected {self.d_model}")
        if N != self.n_tokens:
            raise ValueError(
                f"got N={N}, but mask was built for n_tokens={self.n_tokens}"
            )
        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, h, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Disallowed pairs -> -inf so softmax assigns them zero weight.
        mask = self.attn_mask.to(device=x.device).bool()  # (N, N)
        neg_inf = torch.finfo(attn.dtype).min
        attn = attn.masked_fill(~mask.view(1, 1, N, N), neg_inf)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_tokens={self.n_tokens}, "
            f"n_heads={self.n_heads}, head_dim={self.head_dim}"
        )


# TODO runner wiring:
#   - models.py: add an optional `voronoi_attn=True` config branch that
#     swaps a ViT-Tiny block's dense MHA for VoronoiSparseAttention with
#     n_tokens = num_patches (+1 for the cls token if present).
#   - configs/cifar10_quick.yaml: add a `voronoi_attn` flag so the
#     ablation row carries a distinct tag. This is a sparse-attention
#     module, not CNN-droppable into ResNet-20, so no sweep row is
#     expected by default.
#   - run_sweep.py: gate the row on a positive SOTA-smoke pre-flight (Rule 13).
