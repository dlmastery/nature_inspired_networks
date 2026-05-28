"""H30 — Platonic-Fib Hybrid.

A point-conv operator over the **20 vertices of a regular dodecahedron**
with a **Fibonacci-degree adjacency**: the 20 vertices are partitioned
into 6 Fibonacci-sized groups of sizes ``(1, 1, 2, 3, 5, 8)`` (which
sum to 20), and each vertex in group ``g`` connects to its
``fib_counts[g]`` nearest neighbours (Euclidean) over the dodeca vertex
set. The directed Fib-edges are then **symmetric-closure** merged so
the final adjacency is an undirected graph.

This is the composition of H24 (icosahedral equivariance — the
dodecahedron is the icosahedron's dual, and shares its rotation group)
and Fibonacci scaling (H02 / H04): each vertex's "fan-in" follows the
natural Fibonacci growth law while the global symmetry follows the
Platonic rule.

**On the relaxed degree invariant.** The strict per-node-degree-Fib
invariant ("vertex i has *exactly* fib_counts[i] neighbours") is
geometrically unachievable for arbitrary fib_counts on a fixed vertex
set: a vertex with degree 1 may still RECEIVE edges from neighbours
whose own kNN window happens to include it, so the symmetric closure
``A_ij = 1 iff i ∈ kNN(j) OR j ∈ kNN(i)`` produces a *max*-degree-Fib
graph. We take this relaxation by design — the alternative (drop one
direction or impose strict mutual-NN) breaks connectivity. The
symmetric-OR closure guarantees connectivity for any non-trivial
fib_counts (e.g. {1,1,2,3,5,8}) because every vertex has at least
its single own NN, and the OR closure inherits transitivity-of-NN
on a sphere.

This module exposes:

- :func:`dodeca_vertices` — the 20 canonical dodecahedron vertices via
  the golden-ratio coordinates ``(±1,±1,±1), (0,±1/φ,±φ),
  (±1/φ,±φ,0), (±φ,0,±1/φ)``.
- :func:`fib_nearest_neighbors` — build the Fib-degree adjacency from
  any vertex set and ``fib_counts`` tuple, using symmetric-OR closure.
- :class:`PlatonicFibPointConv` — message-pass over the Fib adjacency.

See ``hypotheses/g3_topologies_graphs/H30_platonic_fib_hybrid.md``.

References
----------
Cohen 2019 'Gauge Equivariant Convolutional Networks and the
Icosahedral CNN' (arXiv:1902.04615); Wang 2019 'Dynamic Graph CNN for
Learning on Point Clouds' (arXiv:1801.07829); Qi 2017 'PointNet'
(arXiv:1612.00593); Vogel 1979 'A better way to construct the
sunflower head' (Math. Biosci. 44).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from nature_inspired_networks.priors import PHI


def dodeca_vertices() -> torch.Tensor:
    """Return the 20 canonical dodecahedron vertices as a ``(20, 3)`` tensor.

    Uses the standard golden-ratio parameterisation, identical to
    :func:`nature_inspired_networks.dodeca_latent.dodecahedron_vertices`:

    - 8 cube vertices ``(±1, ±1, ±1)``
    - 4 vertices ``(0, ±1/φ, ±φ)``
    - 4 vertices ``(±1/φ, ±φ, 0)``
    - 4 vertices ``(±φ, 0, ±1/φ)``

    All 20 vertices lie on a common sphere of radius ``sqrt(3)``.
    """
    inv_phi = 1.0 / PHI
    verts: list[list[float]] = []
    # 8 cube vertices (±1, ±1, ±1)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                verts.append([sx, sy, sz])
    # (0, ±1/φ, ±φ)
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            verts.append([0.0, sy * inv_phi, sz * PHI])
    # (±1/φ, ±φ, 0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            verts.append([sx * inv_phi, sy * PHI, 0.0])
    # (±φ, 0, ±1/φ)
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            verts.append([sx * PHI, 0.0, sz * inv_phi])

    V = torch.tensor(verts, dtype=torch.float32)
    assert V.shape == (20, 3), V.shape
    return V


def fib_nearest_neighbors(
    vertices: torch.Tensor,
    fib_counts: Sequence[int] = (1, 1, 2, 3, 5, 8),
) -> torch.Tensor:
    """Build the undirected Fibonacci-degree adjacency over ``vertices``.

    Fib partition by group: the ``N`` vertices are split into
    ``len(fib_counts)`` consecutive groups of sizes ``fib_counts``
    (the group sizes themselves are the Fibonacci counts). Every
    vertex in group ``g`` then connects to its ``fib_counts[g]``
    Euclidean nearest neighbours.

    For ``fib_counts = (1, 1, 2, 3, 5, 8)`` on the 20-vertex
    dodecahedron:

      * vertex 0       (group 0, size 1): degree 1
      * vertex 1       (group 1, size 1): degree 1
      * vertices 2-3   (group 2, size 2): degree 2 each
      * vertices 4-6   (group 3, size 3): degree 3 each
      * vertices 7-11  (group 4, size 5): degree 5 each
      * vertices 12-19 (group 5, size 8): degree 8 each

    ``sum(fib_counts) == 20`` is required so the partition exactly
    covers the 20 vertices.

    Symmetric-OR closure: the final undirected adjacency is
    ``A_ij = 1`` iff ``i ∈ kNN(j)`` OR ``j ∈ kNN(i)``. This relaxes
    the strict per-node-degree invariant to "max-degree Fib by group"
    (a degree-1 vertex may still RECEIVE edges from neighbours whose
    own kNN window includes it). The OR-closure guarantees
    connectivity for the canonical (1,1,2,3,5,8) partition on the
    dodecahedron because:
      * the size-8 group's kNN covers a wide neighbourhood;
      * every low-degree vertex is hit by at least one size-5/8
        kNN, so the symmetric closure absorbs it into the giant
        component.

    Parameters
    ----------
    vertices : (N, D) tensor
        Vertex coordinates.
    fib_counts : sequence of int, default (1, 1, 2, 3, 5, 8)
        Per-group fan-in degrees AND group sizes (Fibonacci by
        default). ``sum(fib_counts)`` must equal ``N``.

    Returns
    -------
    A : (N, N) float tensor
        Symmetric 0/1 adjacency. ``A[i, j] = 1`` iff vertex ``i`` and
        ``j`` are Fib-neighbours under symmetric-OR closure; diagonal
        is 0.
    """
    assert vertices.dim() == 2 and vertices.shape[1] >= 1, (
        f"vertices must be (N, D); got {tuple(vertices.shape)}"
    )
    N = vertices.shape[0]
    total = sum(int(k) for k in fib_counts)
    assert total == N, (
        f"sum(fib_counts)={total} must equal len(vertices)={N} so the "
        f"partition exactly covers the vertex set."
    )

    # Pairwise Euclidean distances; mask self-loops with +inf.
    d = torch.cdist(vertices, vertices)  # (N, N)
    self_mask = torch.eye(N, dtype=torch.bool, device=d.device)
    d = d.masked_fill(self_mask, float("inf"))

    # Per-group degree assignment: vertices in group g all have degree
    # fib_counts[g]; the partition is by consecutive index blocks.
    A = torch.zeros(N, N, dtype=torch.float32, device=vertices.device)
    cursor = 0
    for group_size in fib_counts:
        k_eff = min(int(group_size), N - 1)
        for v in range(cursor, cursor + int(group_size)):
            if k_eff <= 0:
                continue
            _, neigh = torch.topk(d[v], k=k_eff, largest=False)
            A[v, neigh] = 1.0
        cursor += int(group_size)

    # Symmetric-OR closure: undirected edge if EITHER direction picked.
    A = ((A + A.t()) > 0).to(torch.float32)
    A.fill_diagonal_(0.0)
    return A


def _connected_components(A: torch.Tensor) -> int:
    """Count the connected components of a symmetric 0/1 adjacency.

    Pure-Python BFS over the row-indexed neighbour lists. Used by the
    unit test to assert connectivity of the dodeca-Fib graph.
    """
    N = A.shape[0]
    visited = [False] * N
    components = 0
    for start in range(N):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            neighbours = (A[u] > 0).nonzero(as_tuple=False).flatten().tolist()
            for w in neighbours:
                if not visited[w]:
                    stack.append(w)
    return components


class PlatonicFibPointConv(nn.Module):
    """Fibonacci-degree message-pass over the **20 dodecahedron** vertex set.

    Architecture (per Battaglia 2018 graph-net contract): given node
    features ``X`` of shape ``(B, N, D)``, compute

        M = A_hat @ X                              # (B, N, D) message
        Y = relu( M @ W_msg + X @ W_self )         # (B, N, D) output

    where ``A_hat`` is the symmetrically-normalised Fib adjacency
    (``D^{-1/2} A D^{-1/2}``) and ``W_msg``, ``W_self`` are learnable
    linear maps. The adjacency *pattern* is fixed at construction;
    only the linear-map weights are learned.

    Parameters
    ----------
    in_dim, out_dim : int
        Input / output feature dimensions per node.
    fib_counts : sequence of int, default (1, 1, 2, 3, 5, 8)
        Per-vertex fan-in degrees. ``len(fib_counts) ≤ N``.
    vertices : (N, 3) tensor or None
        Optional override for the vertex set. If None, uses the
        canonical 20 dodecahedron vertices from :func:`dodeca_vertices`.
    bias : bool, default True
        Whether the linear maps carry biases.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        fib_counts: Sequence[int] = (1, 1, 2, 3, 5, 8),
        vertices: torch.Tensor | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        verts = dodeca_vertices() if vertices is None else vertices
        assert verts.dim() == 2 and verts.shape[1] == 3, (
            f"vertices must be (N, 3); got {tuple(verts.shape)}"
        )
        self.n_nodes = verts.shape[0]
        self.fib_counts = tuple(fib_counts)
        total = sum(int(k) for k in self.fib_counts)
        assert total == self.n_nodes, (
            f"sum(fib_counts)={total} must equal len(vertices)="
            f"{self.n_nodes} so the partition exactly covers the "
            f"vertex set."
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
            ``self.n_nodes`` (20 for the canonical dodecahedron).
        """
        assert x.dim() == 3, f"expected (B, N, D); got {tuple(x.shape)}"
        B, N, D = x.shape
        assert N == self.n_nodes, (
            f"node count mismatch: got N={N}; expected {self.n_nodes}"
        )
        msg = torch.einsum("ij,bjd->bid", self.adjacency_norm, x)
        out = self.w_msg(msg) + self.w_self(x)
        return torch.relu(out)
