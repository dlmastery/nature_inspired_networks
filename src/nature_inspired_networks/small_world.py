"""H29 — φ-Small-World GNN.

A Watts-Strogatz small-world graph rewired at probability ``p = 1/φ ≈
0.618`` should achieve the optimal balance of local clustering and
global reach: ``1/φ`` is the most-irrational fraction less than 1, and
cortical connectivity is empirically estimated around p ≈ 0.5–0.7
(Bullmore & Sporns 2009), which brackets ``1/φ``.

This module provides:

- :func:`phi_small_world_adjacency` — deterministic (seeded) Watts-
  Strogatz construction. Returns a symmetric boolean ``(n, n)`` tensor.
- :class:`PhiSmallWorldGNN` — minimal 1-layer GNN that does message
  passing on a fixed small-world adjacency.

Comparison baselines for sweep ablations (NOT implemented here, just
documented for the sweep config row):

- ``p = 0.1``  — Watts-Strogatz "small-world" canonical regime.
- ``p = 0.5``  — half-rewired plateau.
- ``p = 1/φ``  — H29 hypothesised optimum.
- ``p = 0.9``  — near-random regime.

References
----------
Watts, Strogatz 1998 Nature 'Collective dynamics of small-world
networks'; Bullmore, Sporns 2009 Nature Reviews Neuroscience 'Complex
brain networks'; Kipf, Welling 2017 'GCN' (arXiv:1609.02907). See
``hypotheses/g3_topologies_graphs/H29_phi_small_world.md``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from nature_inspired_networks.priors import PHI


def phi_small_world_adjacency(
    n_nodes: int,
    k: int = 4,
    p: float | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Watts-Strogatz small-world adjacency at rewiring probability ``p``.

    Parameters
    ----------
    n_nodes : int
        Number of nodes ``n`` (must be > k).
    k : int, default 4
        Mean degree on the initial ring lattice (must be even). Each
        node is connected to its ``k`` nearest neighbours on a ring.
    p : float or None, default ``1/φ``
        Rewiring probability per edge. ``None`` is interpreted as the
        H29 default ``1/φ ≈ 0.6180``.
    seed : int, default 0
        RNG seed for the rewiring choices — adjacency is deterministic
        for fixed ``(n_nodes, k, p, seed)``.

    Returns
    -------
    A : ``(n, n)`` boolean torch.Tensor
        Symmetric (``A == A.T``), zero-diagonal. Self-loops are never
        introduced; duplicated edges are skipped (a re-wiring attempt
        that would create a duplicate or self-loop leaves the original
        edge in place).
    """
    assert n_nodes > k, f"need n_nodes ({n_nodes}) > k ({k})"
    assert k > 0 and k % 2 == 0, f"k ({k}) must be a positive even integer"
    if p is None:
        p = 1.0 / PHI
    assert 0.0 <= p <= 1.0, p

    gen = torch.Generator().manual_seed(int(seed))
    A = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)

    # Step 1 — ring lattice with degree k.
    half = k // 2
    for i in range(n_nodes):
        for j in range(1, half + 1):
            r = (i + j) % n_nodes
            A[i, r] = True
            A[r, i] = True

    # Step 2 — rewire each "right-side" edge (i, (i+j)%n) with probability p.
    # We iterate j from 1..half and i from 0..n-1; each (i, j) pair is
    # visited once per WS convention.
    for j in range(1, half + 1):
        for i in range(n_nodes):
            r = (i + j) % n_nodes
            if not A[i, r]:
                # was rewired away earlier this pass; leave it.
                continue
            if torch.rand(1, generator=gen).item() < p:
                # pick a new target k_new != i, not already a neighbour.
                # try up to n_nodes times then abandon (keep original).
                rewired = False
                for _ in range(n_nodes):
                    cand = torch.randint(0, n_nodes, (1,), generator=gen).item()
                    if cand == i:
                        continue
                    if A[i, cand]:
                        continue
                    # drop old edge, add new
                    A[i, r] = False
                    A[r, i] = False
                    A[i, cand] = True
                    A[cand, i] = True
                    rewired = True
                    break
                if not rewired:
                    # graph dense enough that no fresh target exists; skip
                    pass

    # zero diagonal (paranoia — never set above, but defensive)
    diag_idx = torch.arange(n_nodes)
    A[diag_idx, diag_idx] = False
    return A


class PhiSmallWorldGNN(nn.Module):
    """Minimal 1-layer GNN over a fixed φ-small-world adjacency.

    Forward: ``(B, N, D_in) → (B, N, D_out)`` via

        h = ReLU( (A_norm @ X) @ W_self + X @ W_skip )

    where ``A_norm`` is the symmetric-normalised small-world adjacency
    (D^{-1/2} A D^{-1/2}, eps-clamped). The adjacency is computed once
    at construction time from ``(n_nodes, k, p, seed)`` and registered
    as a buffer — it does NOT receive gradients.

    Pass ``p=1/φ`` (default via ``None``) for the H29 row;
    ``p ∈ {0.1, 0.5, 0.9}`` for the comparison rows.
    """

    def __init__(
        self,
        n_nodes: int,
        in_dim: int,
        out_dim: int,
        k: int = 4,
        p: float | None = None,
        seed: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert in_dim > 0 and out_dim > 0
        self.n_nodes = n_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim

        A_bool = phi_small_world_adjacency(n_nodes, k=k, p=p, seed=seed)
        A = A_bool.to(torch.float32)
        # symmetric normalisation
        deg = A.sum(dim=1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        A_norm = A * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)
        self.register_buffer("adj", A_norm)

        self.w_self = nn.Linear(in_dim, out_dim, bias=bias)
        self.w_skip = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.shape[1] == self.n_nodes, (
            f"PhiSmallWorldGNN expects (B, {self.n_nodes}, D); got {tuple(x.shape)}"
        )
        msg = torch.einsum("ij,bjd->bid", self.adj, x)
        out = self.w_self(msg) + self.w_skip(x)
        return torch.relu(out)


# TODO runner wiring:
#   - models.py: add an optional `small_world_head` config branch that
#     reshapes the post-pool feature vector into N node slots and runs
#     one or two PhiSmallWorldGNN layers before the classifier.
#   - configs/cifar10_quick.yaml: emit 3 ablation rows
#     `sg_small_world_p_phi`, `sg_small_world_p_01`, `sg_small_world_p_05`
#     covering the H29 baseline comparisons.
#   - For pure node-classification GNN benchmarks (Cora/Citeseer/Pubmed,
#     the H29 falsifier dataset), expose the adjacency-only helper and
#     skip the linear-head wrapper.
