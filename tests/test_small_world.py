"""Unit tests for H29 — phi_small_world_adjacency + PhiSmallWorldGNN.

Run as a script:
    python tests/test_small_world.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.small_world import (  # noqa: E402
    PhiSmallWorldGNN,
    phi_small_world_adjacency,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_adjacency_shape_and_dtype():
    n, k = 32, 4
    A = phi_small_world_adjacency(n, k=k, p=1.0 / PHI, seed=0)
    assert A.shape == (n, n), A.shape
    assert A.dtype == torch.bool, A.dtype
    # zero diagonal — no self-loops
    diag = torch.diagonal(A)
    assert not diag.any().item()


def test_adjacency_symmetric_under_rewiring():
    """Regression: the rewiring step must keep A symmetric (the WS
    construction drops both directed entries of the rewired edge and
    adds both directed entries of the new edge).
    """
    for p in (0.0, 0.1, 1.0 / PHI, 0.5, 0.9, 1.0):
        A = phi_small_world_adjacency(40, k=6, p=p, seed=42)
        assert torch.equal(A, A.t()), f"asymmetric at p={p}"


def test_total_edge_count_is_preserved():
    """Watts-Strogatz rewiring drops one edge and adds another, so the
    total edge count must remain exactly n*k/2 (undirected).
    For zero rewiring (p=0) we further confirm the count matches the
    expected n*k formula on the directed adjacency.
    """
    n, k = 40, 6
    # p = 0 → exactly n*k directed entries
    A0 = phi_small_world_adjacency(n, k=k, p=0.0, seed=0)
    assert A0.sum().item() == n * k, A0.sum().item()
    # arbitrary p preserves the count (each rewire is a 1:1 swap)
    for p in (0.1, 1.0 / PHI, 0.5):
        A = phi_small_world_adjacency(n, k=k, p=p, seed=7)
        assert A.sum().item() == n * k, (p, A.sum().item())


def test_gnn_forward_shape_and_deterministic_adj():
    n, in_dim, out_dim = 16, 8, 12
    gnn = PhiSmallWorldGNN(n_nodes=n, in_dim=in_dim, out_dim=out_dim, k=4, p=None, seed=0)
    x = torch.randn(3, n, in_dim)
    y = gnn(x)
    assert y.shape == (3, n, out_dim), y.shape
    assert torch.isfinite(y).all()

    # Determinism: rebuilding with the same seed must yield the same
    # adjacency buffer.
    gnn2 = PhiSmallWorldGNN(n_nodes=n, in_dim=in_dim, out_dim=out_dim, k=4, p=None, seed=0)
    assert torch.allclose(gnn.adj, gnn2.adj)


def test_gnn_rejects_wrong_node_count():
    gnn = PhiSmallWorldGNN(n_nodes=16, in_dim=8, out_dim=4)
    try:
        gnn(torch.randn(2, 17, 8))
        raise AssertionError("expected AssertionError for mismatched n_nodes")
    except AssertionError as exc:
        if "expected AssertionError" in str(exc):
            raise


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
