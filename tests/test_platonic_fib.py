"""Unit tests for H30 — Platonic-Fib Hybrid (20-vertex dodecahedron)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.platonic_fib import (  # noqa: E402
    PlatonicFibPointConv,
    _connected_components,
    dodeca_vertices,
    fib_nearest_neighbors,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_h30_dodeca_20_vertices_and_connectivity():
    """Post-audit mechanism pin (G3 H30 MAJOR fix).

    Verifies the four things the audit flagged as broken:

      1. The vertex set is the 20 dodecahedron vertices, not 12 icosa.
      2. Vertex coordinates match the canonical dodecahedron set
         {(±1,±1,±1), (0,±1/φ,±φ), (±1/φ,±φ,0), (±φ,0,±1/φ)}
         within 1e-6.
      3. The default Fib partition (1,1,2,3,5,8) sums to 20 (matches
         the doc's Fib-partition claim).
      4. The resulting adjacency (symmetric-OR closure) is connected:
         one connected component over all 20 vertices.
    """
    V = dodeca_vertices()
    # (1) Count.
    assert V.shape == (20, 3), V.shape

    # (2) Canonical vertex set match. Build the expected set as a
    # sorted tuple of triples (to avoid order dependence).
    inv_phi = 1.0 / PHI
    expected: set[tuple[int, int, int]] = set()  # we hash by rounded triple
    cube = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                cube.append((sx, sy, sz))
    others = []
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            others.append((0.0, sy * inv_phi, sz * PHI))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            others.append((sx * inv_phi, sy * PHI, 0.0))
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            others.append((sx * PHI, 0.0, sz * inv_phi))
    expected_pts = cube + others
    assert len(expected_pts) == 20

    # Hash to 6-decimal-rounded triples for set comparison.
    def _key(v):
        return (round(float(v[0]), 6), round(float(v[1]), 6),
                round(float(v[2]), 6))

    have = {_key(V[k]) for k in range(20)}
    want = {_key(p) for p in expected_pts}
    assert have == want, (have ^ want)

    # All on common sphere of radius sqrt(3).
    norms = V.norm(dim=1)
    assert torch.allclose(norms, torch.full((20,), 3.0 ** 0.5), atol=1e-6)

    # (3) Default Fib partition (1,1,2,3,5,8) sums to 20.
    fib = (1, 1, 2, 3, 5, 8)
    assert sum(fib) == 20

    # (4) Symmetric-OR adjacency is connected (one component).
    A = fib_nearest_neighbors(V, fib_counts=fib)
    assert A.shape == (20, 20)
    assert torch.equal(A, A.t()), "adjacency must be symmetric"
    assert A.diag().sum().item() == 0.0
    n_components = _connected_components(A)
    assert n_components == 1, (
        f"dodeca-Fib graph must be connected; got {n_components} "
        f"components"
    )


def test_fib_adjacency_edge_count_bounded():
    """The directed Fib adjacency (before OR closure) has
    sum(degree_per_vertex_in_group) directed entries; the
    symmetric-OR closure has between that count and 2x that count
    of (i, j) entries. The result must be symmetric (even sum).
    """
    V = dodeca_vertices()
    fib = (1, 1, 2, 3, 5, 8)
    # Directed degree per vertex by group: group g has fib[g] vertices
    # each with degree fib[g]. So total directed = sum(fib[g]**2).
    directed = sum(k * k for k in fib)
    A = fib_nearest_neighbors(V, fib_counts=fib)
    total = A.sum().item()
    lo = directed  # OR closure can only collapse, not add (each
                   # symmetric pair counts twice in A[i,j]+A[j,i]).
    hi = 2 * directed
    assert lo <= total <= hi, (total, lo, hi)
    # Symmetric ⇒ even total of nonzero entries.
    assert int(total) % 2 == 0, total


def test_platonic_fib_pointconv_forward_shape():
    """Forward must map (B, 20, in_dim) → (B, 20, out_dim) with finite
    values and gradient flow on the dodeca vertex set.
    """
    torch.manual_seed(0)
    layer = PlatonicFibPointConv(in_dim=8, out_dim=16)
    assert layer.n_nodes == 20
    x = torch.randn(3, 20, 8)
    y = layer(x)
    assert y.shape == (3, 20, 16)
    assert torch.isfinite(y).all()
    y.sum().backward()
    for p in layer.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_complete_graph_regression_with_uniform_max_fib_counts():
    """Regression: when fib_counts is a single group of size 20 whose
    degree saturates at N-1=19, every vertex picks all 19 others and
    the symmetric-OR closure is the complete-graph adjacency K_20.

    The function caps ``k_eff = min(group_size, N-1)`` so a single
    group of size 20 yields k_eff=19 per vertex.
    """
    V = dodeca_vertices()
    fib_counts = (20,)
    A = fib_nearest_neighbors(V, fib_counts=fib_counts)
    K = torch.ones(20, 20) - torch.eye(20)
    assert torch.equal(A, K)


def test_partition_sum_must_equal_n_rejected():
    """Passing fib_counts whose sum != N must raise — the partition
    has to cover the vertex set exactly.
    """
    V = dodeca_vertices()
    try:
        # sum=21 != 20
        fib_nearest_neighbors(V, fib_counts=(1, 1, 2, 3, 5, 9))
        raise AssertionError("expected AssertionError for non-covering partition")
    except AssertionError as exc:
        if "expected AssertionError" in str(exc):
            raise


def test_layer_buffer_persistence_across_device_move():
    """Adjacency + vertex buffers must move with .to(...) — i.e., they
    are registered buffers, not plain tensors.
    """
    layer = PlatonicFibPointConv(in_dim=4, out_dim=8)
    buffer_names = {n for n, _ in layer.named_buffers()}
    assert "vertices" in buffer_names
    assert "adjacency" in buffer_names
    assert "adjacency_norm" in buffer_names
    assert layer.vertices.shape == (20, 3)
    assert layer.adjacency.shape == (20, 20)
    assert layer.adjacency_norm.shape == (20, 20)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
