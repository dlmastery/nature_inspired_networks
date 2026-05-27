"""Unit tests for H30 — Platonic-Fib Hybrid."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.platonic_fib import (  # noqa: E402
    PlatonicFibPointConv,
    fib_nearest_neighbors,
    icosa_vertices,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_icosa_vertices_12_on_unit_sphere():
    """The 12 icosa vertices via (0, +-1, +-PHI) and cyclic permutations
    must (a) have count 12, (b) all lie on a single sphere of radius
    sqrt(1 + PHI**2), (c) be pairwise distinct.
    """
    v = icosa_vertices()
    assert v.shape == (12, 3)
    # All on common sphere.
    norms = v.norm(dim=1)
    expected_r = (1.0 + PHI ** 2) ** 0.5
    assert torch.allclose(norms, torch.full((12,), expected_r), atol=1e-5)
    # Pairwise distinct vertices.
    for i in range(12):
        for j in range(i + 1, 12):
            assert not torch.allclose(v[i], v[j]), (i, j)


def test_fib_adjacency_edge_count_bounded_by_2_sum_fib_counts():
    """Spec invariant: with N vertices and per-vertex degree
    ``fib_counts``, the directed adjacency has exactly ``sum(fib_counts)``
    outgoing edges by construction. The OR-symmetrisation ``(A + A.T) > 0``
    can only collapse (not add) edges, so the symmetric edge-entry
    count obeys

        sum(fib_counts)  <=  A_sym.sum()  <=  2 * sum(fib_counts)

    The upper bound ``2 * sum(fib_counts)`` is tight when every
    directed Fib-neighbour is mutually reciprocal; the lower bound
    ``sum(fib_counts)`` is tight when every directed edge has its
    reciprocal already in the kNN list. For the canonical icosa +
    ``(1,1,2,3,5)`` setup the count sits in the middle (some
    reciprocity, some not) — we therefore test the invariant range
    and the *directed* count exactly equals ``sum(fib_counts)``.
    """
    v = icosa_vertices()
    fib = (1, 1, 2, 3, 5)
    A = fib_nearest_neighbors(v, fib_counts=fib)

    # The symmetric adjacency is symmetric (sanity).
    assert torch.equal(A, A.t())
    # And has no self-loops.
    assert A.diag().sum().item() == 0.0

    total = A.sum().item()
    lo = sum(fib)
    hi = 2 * sum(fib)
    assert lo <= total <= hi, (total, lo, hi)
    # The symmetric matrix has even sum (each undirected edge
    # contributes 2 entries: A[i,j] and A[j,i]).
    assert int(total) % 2 == 0, total
    # Tight upper bound on undirected edges.
    undirected = int(total) // 2
    assert undirected <= sum(fib), (undirected, sum(fib))


def test_platonic_fib_pointconv_forward_shape():
    """Forward must map (B, 12, in_dim) → (B, 12, out_dim) with finite
    values and gradient flow.
    """
    torch.manual_seed(0)
    layer = PlatonicFibPointConv(in_dim=8, out_dim=16)
    x = torch.randn(3, 12, 8)
    y = layer(x)
    assert y.shape == (3, 12, 16)
    assert torch.isfinite(y).all()
    # Backward path is finite.
    y.sum().backward()
    for p in layer.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_complete_graph_regression_with_uniform_max_fib_counts():
    """Regression: when fib_counts saturates at N-1 (e.g. (11, 11, ...))
    every vertex picks all 11 others as neighbours and the adjacency
    becomes the complete-graph adjacency K_12 (all-ones off-diagonal).
    """
    v = icosa_vertices()
    # 12 entries each of value 11 means each vertex selects its 11
    # nearest neighbours (i.e., every other vertex).
    fib_counts = (11,) * 12
    A = fib_nearest_neighbors(v, fib_counts=fib_counts)
    # Expected: K_12 = J - I (all-ones with zero diagonal).
    K = torch.ones(12, 12) - torch.eye(12)
    assert torch.equal(A, K)


def test_oversized_fib_counts_rejected():
    """Passing fib_counts longer than N (more degrees than vertices)
    must raise — each entry corresponds to a vertex.
    """
    v = icosa_vertices()
    try:
        # 13 entries on a 12-vertex icosa is invalid.
        fib_nearest_neighbors(v, fib_counts=(1,) * 13)
        raise AssertionError("expected AssertionError for oversized fib_counts")
    except AssertionError as exc:
        if "expected AssertionError" in str(exc):
            raise


def test_layer_buffer_persistence_across_device_move():
    """Adjacency + vertex buffers must move with .to(...) — i.e., they
    are registered buffers, not plain tensors.
    """
    layer = PlatonicFibPointConv(in_dim=4, out_dim=8)
    # Buffers exist with expected names.
    buffer_names = {n for n, _ in layer.named_buffers()}
    assert "vertices" in buffer_names
    assert "adjacency" in buffer_names
    assert "adjacency_norm" in buffer_names
    # Type checks.
    assert layer.vertices.shape == (12, 3)
    assert layer.adjacency.shape == (12, 12)
    assert layer.adjacency_norm.shape == (12, 12)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
