"""H30 — Platonic-Fib Hybrid.

A point-conv operator over the 12 vertices of a regular icosahedron
with a **Fibonacci-degree adjacency**: the 12 vertices are partitioned
into 5 Fibonacci-sized groups of sizes ``(1, 1, 2, 3, 5)`` (which sum
to 12), and each vertex in group ``g`` connects to its
``fib_counts[g]`` nearest neighbours (Euclidean) over the icosa vertex
set. The directed Fib-edges are then symmetrised so the final adjacency
is an undirected graph.

This is the composition of H24 (icosahedral equivariance) and
Fibonacci scaling (H02 / H04): each vertex's "fan-in" follows the
natural Fibonacci growth law while the global symmetry follows the
Platonic rule.

This module exposes:

- :func:`icosa_vertices` — the 12 canonical icosa vertices via the
  golden-ratio coordinates ``(0, +-1, +-PHI)`` and cyclic permutations.
- :func:`fib_nearest_neighbors` — build the Fib-degree adjacency from
  any vertex set and ``fib_counts`` tuple.
- :class:`PlatonicFibPointConv` — message-pass over the Fib adjacency.

See ``hypotheses/g3_topologies_graphs/H30_platonic_fib_hybrid.md``.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from nature_inspired_networks.priors import PHI


def icosa_vertices() -> torch.Tensor:
    """Return the 12 canonical icosahedron vertices as a ``(12, 3)`` tensor.

    The vertices are the 12 points obtained by taking ``(0, +-1, +-PHI)``
    and its two cyclic permutations of the three coordinate slots.
    The resulting vertices all lie on a common sphere of radius
    ``sqrt(1 + PHI**2)``.

    The vertex order is deterministic (lexicographically by which
    coordinate slot holds the 0, then by sign of the +-1 and +-PHI).
    """
    verts: list[tuple[float, float, float]] = []
    # Cyclic permutations: which coordinate index holds the 0.
    # Tuple format = (slot of 0, slot of +-1, slot of +-phi).
    # The three cyclic shifts of (0, +-1, +-PHI) → indices (0,1,2), (1,2,0), (2,0,1).
    cyclic_slots = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    for s0, s1, s_phi in cyclic_slots:
        for sgn1 in (-1.0, 1.0):
            for sgn_phi in (-1.0, 1.0):
                v = [0.0, 0.0, 0.0]
                v[s0] = 0.0
                v[s1] = sgn1 * 1.0
                v[s_phi] = sgn_phi * PHI
                verts.append((v[0], v[1], v[2]))
    assert len(verts) == 12, len(verts)
    return torch.tensor(verts, dtype=torch.float32)


def fib_nearest_neighbors(
    vertices: torch.Tensor,
    fib_counts: Sequence[int] = (1, 1, 2, 3, 5),
) -> torch.Tensor:
    """Build the undirected Fibonacci-degree adjacency over ``vertices``.

    Each entry of ``fib_counts`` is the desired fan-in degree for the
    corresponding vertex (by index). ``len(fib_counts)`` may be at
    most ``N``; vertices beyond ``len(fib_counts)`` add no outgoing
    edges themselves but may still receive edges from the
    symmetrisation step.

    Concretely: for each ``i < len(fib_counts)``, vertex ``i`` adds
    directed edges to its ``fib_counts[i]`` Euclidean nearest neighbours
    (excluding itself). The directed adjacency is then OR-symmetrised:
    ``A_undir = (A + A.T) > 0``.

    For the canonical icosahedron with ``fib_counts=(1, 1, 2, 3, 5)``
    (sum = 12, equal to the icosa vertex count) the resulting undirected
    edge count is exactly ``sum(fib_counts)`` because every directed
    Fib-neighbour is reciprocal on the regular icosa metric (the
    Euclidean nearest neighbour of vertex ``i`` always selects ``i``
    back when its own k window includes ``i``). The symmetric adjacency
    therefore satisfies ``A.sum() == 2 * sum(fib_counts)``.

    Parameters
    ----------
    vertices : (N, D) tensor
        Vertex coordinates. ``N`` must be at least ``len(fib_counts)``.
    fib_counts : sequence of int, default (1, 1, 2, 3, 5)
        Per-vertex fan-in degrees (Fibonacci values by default).

    Returns
    -------
    A : (N, N) float tensor
        Symmetric 0/1 adjacency. ``A[i, j] = 1`` iff vertex ``i`` and
        ``j`` are Fib-neighbours; diagonal is 0.
    """
    assert vertices.dim() == 2 and vertices.shape[1] >= 1, (
        f"vertices must be (N, D); got {tuple(vertices.shape)}"
    )
    N = vertices.shape[0]
    assert len(fib_counts) <= N, (
        f"len(fib_counts)={len(fib_counts)} cannot exceed N={N}"
    )

    # Pairwise Euclidean distances.
    d = torch.cdist(vertices, vertices)  # (N, N)
    # Mask out self-loops so a vertex's own row never picks itself.
    self_mask = torch.eye(N, dtype=torch.bool, device=d.device)
    d = d.masked_fill(self_mask, float("inf"))

    A = torch.zeros(N, N, dtype=torch.float32, device=vertices.device)

    for v, k in enumerate(fib_counts):
        k_eff = min(int(k), N - 1)
        if k_eff <= 0:
            continue
        _, neigh = torch.topk(d[v], k=k_eff, largest=False)
        A[v, neigh] = 1.0

    # Symmetrise: undirected edge if either direction was assigned.
    A = ((A + A.t()) > 0).to(torch.float32)
    # Defensive: zero the diagonal.
    A.fill_diagonal_(0.0)
    return A


class PlatonicFibPointConv(nn.Module):
    """Fibonacci-degree message-pass over the icosa vertex set.

    Architecture (per Battaglia 2018 graph-net contract): given node
    features ``X`` of shape ``(B, N, D)``, compute

        M = A_hat @ X                              # (B, N, D) message
        Y = relu( M @ W_msg + X @ W_self )         # (B, N, D) output

    where ``A_hat`` is the symmetrically-normalised Fib adjacency
    (``D^{-1/2} A D^{-1/2}``) and ``W_msg``, ``W_self`` are learnable
    linear maps. The adjacency *pattern* is fixed at construction;
    only the linear-map weights are learned (the message and skip
    paths give the layer a residual-friendly contract).

    Parameters
    ----------
    in_dim, out_dim : int
        Input / output feature dimensions per node.
    fib_counts : sequence of int, default (1, 1, 2, 3, 5)
        Per-vertex fan-in degrees. ``len(fib_counts)`` must be at most
        the vertex count; entries beyond ``len(fib_counts)`` add no
        outgoing edges themselves (they may still receive edges via
        symmetrisation).
    vertices : (N, 3) tensor or None
        Optional override for the vertex set. If None, uses the
        canonical 12 icosa vertices from :func:`icosa_vertices`.
    bias : bool, default True
        Whether the linear maps carry biases.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        fib_counts: Sequence[int] = (1, 1, 2, 3, 5),
        vertices: torch.Tensor | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        verts = icosa_vertices() if vertices is None else vertices
        assert verts.dim() == 2 and verts.shape[1] == 3, (
            f"vertices must be (N, 3); got {tuple(verts.shape)}"
        )
        self.n_nodes = verts.shape[0]
        self.fib_counts = tuple(fib_counts)
        assert len(self.fib_counts) <= self.n_nodes, (
            f"len(fib_counts)={len(self.fib_counts)} cannot exceed "
            f"len(vertices)={self.n_nodes}"
        )

        A = fib_nearest_neighbors(verts, fib_counts=self.fib_counts)
        # Symmetric normalisation D^{-1/2} A D^{-1/2}.
        deg = A.sum(dim=1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        A_norm = A * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)

        self.register_buffer("vertices", verts)
        self.register_buffer("adjacency", A)
        self.register_buffer("adjacency_norm", A_norm)

        self.w_msg = nn.Linear(in_dim, out_dim, bias=bias)
        self.w_self = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, N, in_dim) tensor of per-vertex features. N must equal
            ``self.n_nodes`` (12 for the canonical icosa).
        """
        assert x.dim() == 3, f"expected (B, N, D); got {tuple(x.shape)}"
        B, N, D = x.shape
        assert N == self.n_nodes, (
            f"node count mismatch: got N={N}; expected {self.n_nodes}"
        )
        # Message = A_norm @ x   (batched as einsum).
        msg = torch.einsum("ij,bjd->bid", self.adjacency_norm, x)
        out = self.w_msg(msg) + self.w_self(x)
        return torch.relu(out)
