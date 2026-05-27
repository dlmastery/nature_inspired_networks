"""H23 — Platonic φ-Graph (Metatron's Cube connectivity).

The Metatron-Cube graph is the 2-D projection of all five Platonic solids
onto a single 13-circle diagram. The 13 nodes are: 1 center + 6 inner
hexagon + 6 outer hexagon. Edge weights follow a 2-class partition:

- ``1.0`` on **short-orbit** edges (Vesica Piscis overlap region):
  center↔inner6, and each inner hexagon edge (the inner hex ring).
- ``1/φ`` on **long-orbit** edges (outer rim): each outer hexagon edge
  (outer hex ring), and inner↔outer radial spokes.

The construction is fully deterministic and the adjacency is symmetric
by design. This module provides:

- :func:`metatron_cube_adjacency` — returns the canonical ``(13, 13)``
  weighted adjacency tensor.
- :class:`MetatronGraphLayer` — message-passing layer over the fixed
  adjacency, optionally with learnable edge weights.

References
----------
Battaglia 2018 'Relational inductive biases, deep learning, and graph
networks' (arXiv:1806.01261); Gilmer 2017 'Neural Message Passing for
Quantum Chemistry' (arXiv:1704.01212). See
``hypotheses/g3_topologies_graphs/H23_platonic_phi_graph.md``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from nature_inspired_networks.priors import PHI


# Node indices for the 13-node Metatron-Cube graph:
#   0      → center
#   1..6   → inner hexagon vertices (ring 1)
#   7..12  → outer hexagon vertices (ring 2; outer[i] is colinear with inner[i])
_N_NODES = 13


def metatron_cube_adjacency() -> torch.Tensor:
    """Return the canonical ``(13, 13)`` Metatron-Cube weighted adjacency.

    Layout:
      - node 0          = center
      - nodes 1..6      = inner hexagon ring (6 vertices)
      - nodes 7..12     = outer hexagon ring (6 vertices, radially aligned)

    Edge classes:
      - center↔inner            (6 edges)           weight 1.0
      - inner hex ring          (6 edges)           weight 1.0
      - outer hex ring          (6 edges)           weight 1/φ
      - inner↔outer (radial)    (6 edges)           weight 1/φ

    Total directed edges (symmetric) = 24 undirected edges. All edge
    weights are non-negative reals; the matrix is exactly symmetric and
    has zero diagonal.
    """
    A = torch.zeros(_N_NODES, _N_NODES, dtype=torch.float32)
    inv_phi = 1.0 / PHI

    # center <-> inner hex (weight 1.0)
    for i in range(1, 7):
        A[0, i] = 1.0
        A[i, 0] = 1.0

    # inner hex ring (weight 1.0)
    for i in range(6):
        a = 1 + i
        b = 1 + ((i + 1) % 6)
        A[a, b] = 1.0
        A[b, a] = 1.0

    # outer hex ring (weight 1/phi)
    for i in range(6):
        a = 7 + i
        b = 7 + ((i + 1) % 6)
        A[a, b] = inv_phi
        A[b, a] = inv_phi

    # inner <-> outer radial spokes (weight 1/phi)
    for i in range(6):
        inner = 1 + i
        outer = 7 + i
        A[inner, outer] = inv_phi
        A[outer, inner] = inv_phi

    return A


class MetatronGraphLayer(nn.Module):
    """Message-passing layer over the Metatron-Cube adjacency.

    Accepts node features ``X: (B, 13, D_in)`` and produces ``(B, 13, D_out)``
    via the rule

        X' = ReLU( (A_norm @ X) @ W_self + X @ W_skip )

    where ``A_norm`` is the (symmetric) Metatron adjacency (optionally
    with learnable edge weights gating the fixed pattern) and ``W_self``,
    ``W_skip`` are linear transforms. The adjacency *pattern* (which
    edges exist) is always fixed; only the magnitudes can be learned.

    Parameters
    ----------
    in_dim : int
        Input feature dimension per node.
    out_dim : int
        Output feature dimension per node.
    learnable_edge_weights : bool, default False
        If True, multiply the fixed adjacency element-wise by a
        learnable ``(13, 13)`` parameter (initialised to ones, kept
        symmetric by averaging with its transpose at every forward).
    bias : bool, default True
        Whether the internal linear maps carry biases.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        learnable_edge_weights: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert in_dim > 0 and out_dim > 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.learnable_edge_weights = learnable_edge_weights

        # Fixed adjacency as a buffer so it moves with .to(device).
        A = metatron_cube_adjacency()
        self.register_buffer("adj", A)

        if learnable_edge_weights:
            # one trainable multiplier per (i, j); symmetrised at forward
            self.edge_gate = nn.Parameter(torch.ones_like(A))
        else:
            self.edge_gate = None

        self.w_self = nn.Linear(in_dim, out_dim, bias=bias)
        self.w_skip = nn.Linear(in_dim, out_dim, bias=bias)

    def _effective_adj(self) -> torch.Tensor:
        if self.edge_gate is None:
            A = self.adj
        else:
            gate = 0.5 * (self.edge_gate + self.edge_gate.t())
            A = self.adj * gate
        # symmetric normalisation D^{-1/2} A D^{-1/2}, with eps for the
        # disconnected diagonal (Kipf & Welling 2017 — but skipping the
        # +I self-loop because we already provide an explicit skip path).
        deg = A.sum(dim=1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        return A * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.shape[1] == _N_NODES, (
            f"MetatronGraphLayer expects (B, 13, D); got {tuple(x.shape)}"
        )
        A = self._effective_adj()  # (13, 13)
        # message = A @ X     -- batched as einsum
        msg = torch.einsum("ij,bjd->bid", A, x)  # (B, 13, in_dim)
        out = self.w_self(msg) + self.w_skip(x)
        return torch.relu(out)


# TODO runner wiring:
#   - models.py: add an optional `metatron_head=True` config branch that
#     reshapes the pooled feature vector into 13 node slots (e.g. by
#     channel-wise grouping) and passes through one or two
#     MetatronGraphLayer instances before the final classifier head.
#   - configs/cifar10_quick.yaml: add `metatron_head_dim: 16` so the
#     ablation row carries a distinct tag.
#   - run_sweep.py: gate row on a positive SOTA-smoke pre-flight (Rule 13).
