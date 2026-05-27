"""H79 — Morphing Polytope Adjacency (Vector-Equilibrium "jitterbug").

Buckminster Fuller's "jitterbug" transformation continuously morphs a
cuboctahedron (the *vector equilibrium*, 12 vertices, 24 edges) into an
icosahedron (12 vertices, 30 edges) by twisting the eight triangular
faces. Both polytopes share the same 12-vertex count, so a single graph
of 12 nodes can interpolate its edge structure between the two extremes.

This module exposes the two 12-vertex sets and a morph that linearly
interpolates their (binary) nearest-neighbour adjacencies:

    ``A(t) = (1 - t) * A_cubocta + t * A_icosa``,  ``t in [0, 1]``.

A learnable, sigmoid-gated ``t`` then drives message passing on the
morphed adjacency, giving a *dynamic topology* whose connectivity the
optimiser can slide along the jitterbug path — useful for continual /
plastic learning where the graph structure should adapt.

Esoteric origin (acknowledged in one sentence): Fuller's vector-
equilibrium / jitterbug motif motivates the morph; the implementation is
a plain convex interpolation of two nearest-neighbour adjacencies.

Refs (Citation Rigor):
    Kipf, T. N., Welling, M. 2017 ICLR 'Semi-Supervised Classification
    with Graph Convolutional Networks' (arXiv:1609.02907) - the
    symmetric-normalised message-passing rule used by MorphingGraphLayer.
    Battaglia, P. W., Hamrick, J. B., Bapst, V. 2018 'Relational
    inductive biases, deep learning, and graph networks'
    (arXiv:1806.01261) - frames learnable adjacency as a relational
    inductive bias, motivating the learnable jitterbug parameter ``t``.

Public surface
--------------
- :func:`cuboctahedron_vertices`  12 cuboctahedron vertices (12, 3)
- :func:`icosahedron_vertices`    12 icosahedron vertices (12, 3)
- :func:`morph_adjacency`         interpolated (12, 12) adjacency
- :class:`MorphingGraphLayer`     message passing with learnable morph t
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .priors import PHI

__all__ = [
    "cuboctahedron_vertices",
    "icosahedron_vertices",
    "morph_adjacency",
    "MorphingGraphLayer",
]

_N_NODES = 12


def cuboctahedron_vertices() -> torch.Tensor:
    """Return the 12 cuboctahedron vertices as a ``(12, 3)`` tensor.

    The cuboctahedron (Fuller's *vector equilibrium*) has its 12 vertices
    at all permutations of ``(+-1, +-1, 0)``: the edge midpoints of a
    cube. All vertices lie on a sphere of radius ``sqrt(2)`` and every
    vertex is equidistant from the centre and from its nearest
    neighbours, hence "vector equilibrium". The vertex order is
    deterministic (by which coordinate slot holds the 0, then signs).
    """
    verts: list[tuple[float, float, float]] = []
    for zero_slot in (0, 1, 2):
        nz = [s for s in range(3) if s != zero_slot]
        for sgn_a in (-1.0, 1.0):
            for sgn_b in (-1.0, 1.0):
                v = [0.0, 0.0, 0.0]
                v[nz[0]] = sgn_a
                v[nz[1]] = sgn_b
                verts.append((v[0], v[1], v[2]))
    assert len(verts) == _N_NODES, len(verts)
    return torch.tensor(verts, dtype=torch.float32)


def icosahedron_vertices() -> torch.Tensor:
    """Return the 12 icosahedron vertices as a ``(12, 3)`` tensor.

    Vertices are the cyclic permutations of ``(0, +-1, +-PHI)`` — the
    canonical golden-ratio construction. All lie on a sphere of radius
    ``sqrt(1 + PHI**2)``. Order is deterministic (by which slot holds the
    0, then signs), matching :func:`cuboctahedron_vertices` so the two
    vertex sets can be morphed index-for-index.
    """
    verts: list[tuple[float, float, float]] = []
    cyclic_slots = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    for s0, s1, s_phi in cyclic_slots:
        for sgn1 in (-1.0, 1.0):
            for sgn_phi in (-1.0, 1.0):
                v = [0.0, 0.0, 0.0]
                v[s0] = 0.0
                v[s1] = sgn1 * 1.0
                v[s_phi] = sgn_phi * PHI
                verts.append((v[0], v[1], v[2]))
    assert len(verts) == _N_NODES, len(verts)
    return torch.tensor(verts, dtype=torch.float32)


def _nearest_neighbour_adjacency(verts: torch.Tensor) -> torch.Tensor:
    """Binary symmetric adjacency: connect each vertex to all others at
    the minimum pairwise distance (the polytope edges).

    For a regular polytope the edges are exactly the minimum-distance
    vertex pairs, so this recovers the true edge set. The result is
    symmetric with a zero diagonal.
    """
    n = verts.shape[0]
    d = torch.cdist(verts, verts)  # (n, n)
    eye = torch.eye(n, dtype=torch.bool)
    d_off = d.masked_fill(eye, float("inf"))
    dmin = d_off.min()
    # tolerance relative to the minimum edge length for float robustness
    tol = 1e-4 * (dmin + 1.0)
    A = ((d_off - dmin).abs() <= tol).to(verts.dtype)
    # enforce exact symmetry (cdist is symmetric but guard against fp drift)
    A = ((A + A.t()) > 0).to(verts.dtype)
    return A


def morph_adjacency(t: float | torch.Tensor) -> torch.Tensor:
    """Return the ``(12, 12)`` morphed adjacency at jitterbug parameter ``t``.

    ``A(t) = (1 - t) * A_cubocta + t * A_icosa`` with ``t in [0, 1]``.
    At ``t = 0`` this is exactly the cuboctahedron adjacency; at
    ``t = 1`` exactly the icosahedron adjacency. Intermediate ``t`` give
    fractional edge weights along the jitterbug path. ``t`` may be a
    Python float or a 0-dim tensor (the latter keeps autograd intact).
    """
    A_c = _nearest_neighbour_adjacency(cuboctahedron_vertices())
    A_i = _nearest_neighbour_adjacency(icosahedron_vertices())
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(float(t), dtype=A_c.dtype)
    t = t.to(A_c.dtype)
    return (1.0 - t) * A_c + t * A_i


class MorphingGraphLayer(nn.Module):
    """Message-passing layer on a learnable jitterbug-morphed topology.

    A raw scalar parameter ``t_raw`` is squashed by a sigmoid to give the
    morph parameter ``t = sigmoid(t_raw) in (0, 1)``; the adjacency is the
    convex interpolation of the cuboctahedron and icosahedron edge sets.
    The optimiser can therefore slide the connectivity continuously along
    the vector-equilibrium ↔ icosahedron jitterbug path.

    Accepts node features ``X: (B, 12, in_dim)`` and produces
    ``(B, 12, out_dim)`` via the symmetric-normalised rule

        ``X' = ReLU( (A_norm @ X) @ W_self + X @ W_skip )``

    where ``A_norm`` is the degree-normalised morphed adjacency.

    Parameters
    ----------
    in_dim : int
        Input feature dimension per node.
    out_dim : int
        Output feature dimension per node.
    init_t : float, default 0.5
        Initial morph parameter (the raw param is set to
        ``logit(init_t)`` so ``sigmoid(t_raw) == init_t`` at start).
    bias : bool, default True
        Whether the internal linear maps carry biases.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        init_t: float = 0.5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert in_dim > 0 and out_dim > 0
        assert 0.0 < init_t < 1.0, "init_t must be in (0, 1)"
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Precompute and cache the two endpoint adjacencies as buffers so
        # they move with .to(device) and the per-forward cost is just the
        # convex blend (no cdist recompute).
        A_c = _nearest_neighbour_adjacency(cuboctahedron_vertices())
        A_i = _nearest_neighbour_adjacency(icosahedron_vertices())
        self.register_buffer("adj_cubocta", A_c)
        self.register_buffer("adj_icosa", A_i)

        logit = torch.log(torch.tensor(init_t) / (1.0 - init_t))
        self.t_raw = nn.Parameter(logit.clone())

        self.w_self = nn.Linear(in_dim, out_dim, bias=bias)
        self.w_skip = nn.Linear(in_dim, out_dim, bias=bias)

    @property
    def t(self) -> torch.Tensor:
        """The current morph parameter ``t = sigmoid(t_raw)`` (0-dim)."""
        return torch.sigmoid(self.t_raw)

    def _morphed_adj(self) -> torch.Tensor:
        t = self.t
        A = (1.0 - t) * self.adj_cubocta + t * self.adj_icosa
        # symmetric normalisation D^{-1/2} A D^{-1/2}
        deg = A.sum(dim=1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        return A * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.shape[1] == _N_NODES, (
            f"MorphingGraphLayer expects (B, 12, D); got {tuple(x.shape)}"
        )
        A = self._morphed_adj()  # (12, 12)
        msg = torch.einsum("ij,bjd->bid", A, x)  # (B, 12, in_dim)
        out = self.w_self(msg) + self.w_skip(x)
        return torch.relu(out)

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"


# TODO runner wiring:
#   - models.py: add an optional `morphing_graph_head=True` config branch
#     that reshapes the pooled feature into 12 node slots and passes it
#     through one or two MorphingGraphLayer instances before the head;
#     the learnable jitterbug t adapts the topology during training.
#   - configs/cifar10_quick.yaml: add a `morphing_graph_dim` flag so the
#     ablation row carries a distinct tag. This is a graph module, not a
#     CNN-droppable conv block, so no sweep row is expected by default.
#   - run_sweep.py: gate the row on a positive SOTA-smoke pre-flight (Rule 13).
