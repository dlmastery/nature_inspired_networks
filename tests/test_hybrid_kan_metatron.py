"""Unit tests for H69 — KAN-Metatron Symbolic Head."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_kan_metatron import (  # noqa: E402
    KANEdge,
    KANMetatronHead,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_kan_edge_silu_at_init_inside_domain():
    """At init the KANEdge knots equal SiLU(x_k); piecewise-linear interp
    over the (-3, 3) domain must match SiLU at the knot positions exactly.
    """
    edge = KANEdge(spline_pts=8, lo=-3.0, hi=3.0)
    xs = torch.linspace(-3.0, 3.0, 8)
    ys = edge(xs)
    silu = xs * torch.sigmoid(xs)
    assert torch.allclose(ys, silu, atol=1e-6), (ys, silu)


def test_kan_edge_clamps_outside_domain():
    """Inputs above ``hi`` produce the same output as ``hi`` itself."""
    edge = KANEdge(spline_pts=4, lo=-1.0, hi=1.0)
    y_in = edge(torch.tensor([1.0]))
    y_out = edge(torch.tensor([100.0]))
    assert torch.allclose(y_in, y_out, atol=1e-6)


def test_kan_edge_differentiable():
    """Gradients must flow through the spline knots."""
    edge = KANEdge(spline_pts=8)
    x = torch.randn(16, requires_grad=True)
    y = edge(x).sum()
    y.backward()
    assert x.grad is not None
    assert edge.knots.grad is not None


def test_kan_metatron_head_forward_shape():
    head = KANMetatronHead(d_in=32, d_out=10, node_dim=4)
    x = torch.randn(2, 32)
    y = head(x)
    assert y.shape == (2, 10)
    assert torch.isfinite(y).all()


def test_kan_metatron_head_3d_input_supported():
    head = KANMetatronHead(d_in=8, d_out=4, node_dim=3)
    x = torch.randn(2, 5, 8)
    y = head(x)
    assert y.shape == (2, 5, 4)


def test_kan_metatron_edge_count_matches_metatron_adjacency_nnz():
    """The Metatron-Cube adjacency has 48 directed non-zero entries
    (= 24 undirected edges from the canonical adjacency)."""
    n = KANMetatronHead.metatron_edge_count()
    assert n == 48, n


def test_kan_metatron_head_param_count_includes_d_out_splines():
    """The head must carry exactly d_out KANEdge sub-modules."""
    head = KANMetatronHead(d_in=8, d_out=7, node_dim=3, spline_pts=5)
    assert len(head.edges) == 7
    for e in head.edges:
        assert e.knots.numel() == 5


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
