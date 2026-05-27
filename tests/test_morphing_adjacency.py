"""Unit tests for H79 MorphingPolytopeAdjacency / MorphingGraphLayer."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.morphing_adjacency import (  # noqa: E402
    MorphingGraphLayer,
    cuboctahedron_vertices,
    icosahedron_vertices,
    morph_adjacency,
    _nearest_neighbour_adjacency,
)


def test_vertex_sets_are_12x3():
    cub = cuboctahedron_vertices()
    ico = icosahedron_vertices()
    assert cub.shape == (12, 3), cub.shape
    assert ico.shape == (12, 3), ico.shape


def test_adjacencies_symmetric_zero_diagonal():
    A_c = _nearest_neighbour_adjacency(cuboctahedron_vertices())
    A_i = _nearest_neighbour_adjacency(icosahedron_vertices())
    for A in (A_c, A_i):
        assert torch.allclose(A, A.t()), "adjacency must be symmetric"
        assert torch.equal(A.diagonal(), torch.zeros(12)), "zero diagonal"
    # cuboctahedron has 24 edges, icosahedron 30 edges (undirected) ->
    # directed-entry sums of 48 and 60 respectively (mechanism check).
    assert A_c.sum().item() == 48.0, A_c.sum().item()
    assert A_i.sum().item() == 60.0, A_i.sum().item()


def test_morph_endpoints_match_polytope_adjacencies():
    A_c = _nearest_neighbour_adjacency(cuboctahedron_vertices())
    A_i = _nearest_neighbour_adjacency(icosahedron_vertices())
    assert torch.allclose(morph_adjacency(0.0), A_c), "t=0 must be cubocta"
    assert torch.allclose(morph_adjacency(1.0), A_i), "t=1 must be icosa"
    # midpoint is the average of the two endpoint adjacencies
    mid = morph_adjacency(0.5)
    assert torch.allclose(mid, 0.5 * (A_c + A_i), atol=1e-6)


def test_layer_forward_shape():
    """(B, 12, D_in) -> (B, 12, D_out)."""
    torch.manual_seed(0)
    layer = MorphingGraphLayer(in_dim=5, out_dim=7)
    x = torch.randn(4, 12, 5)
    y = layer(x)
    assert y.shape == (4, 12, 7), y.shape
    # wrong node count must raise
    try:
        layer(torch.randn(4, 11, 5))
        raise AssertionError("expected AssertionError for 11 nodes")
    except AssertionError as exc:
        if "expected AssertionError" in str(exc):
            raise


def test_learnable_t_has_gradient():
    """Mechanism: the morph parameter t must receive a gradient and be
    confined to (0, 1) by the sigmoid gate."""
    torch.manual_seed(0)
    layer = MorphingGraphLayer(in_dim=4, out_dim=4, init_t=0.5)
    assert 0.0 < layer.t.item() < 1.0
    assert abs(layer.t.item() - 0.5) < 1e-5  # init_t honoured
    x = torch.randn(2, 12, 4)
    loss = layer(x).pow(2).sum()
    loss.backward()
    assert layer.t_raw.grad is not None
    assert torch.isfinite(layer.t_raw.grad).all()
    assert layer.t_raw.grad.abs().item() > 0.0


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
