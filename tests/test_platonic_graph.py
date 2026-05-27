"""Unit tests for H23 — Metatron-Cube graph adjacency + MetatronGraphLayer.

Run as a script:
    python tests/test_platonic_graph.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.platonic_graph import (  # noqa: E402
    MetatronGraphLayer,
    metatron_cube_adjacency,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_adjacency_shape_and_symmetric():
    A = metatron_cube_adjacency()
    assert A.shape == (13, 13), A.shape
    assert torch.allclose(A, A.t()), "adjacency must be symmetric"
    # zero diagonal — no self-loops in the raw pattern
    assert torch.all(torch.diag(A) == 0.0)


def test_edge_weights_in_canonical_set():
    """Edge weights must come from {0, 1, 1/φ}; no other magnitudes allowed."""
    A = metatron_cube_adjacency()
    inv_phi = 1.0 / PHI
    allowed = torch.tensor([0.0, 1.0, inv_phi])
    flat = A.flatten()
    for v in flat:
        diffs = (allowed - v).abs()
        assert diffs.min().item() < 1e-6, f"unexpected edge weight {v.item()}"
    # And it really uses all three values: we expect 12 ones (center<->6 inner +
    # inner hex ring 6) and 12 inv_phi (outer hex ring 6 + radial 6).
    n_ones = (flat - 1.0).abs().lt(1e-6).sum().item()
    n_invphi = (flat - inv_phi).abs().lt(1e-6).sum().item()
    assert n_ones == 24, n_ones      # 12 undirected ⇒ 24 directed entries
    assert n_invphi == 24, n_invphi  # 12 undirected ⇒ 24 directed entries


def test_layer_forward_shape_fixed_edges():
    layer = MetatronGraphLayer(in_dim=8, out_dim=16, learnable_edge_weights=False)
    x = torch.randn(2, 13, 8)
    y = layer(x)
    assert y.shape == (2, 13, 16), y.shape
    assert torch.isfinite(y).all()
    # edge_gate must not exist on the fixed-edge variant
    assert layer.edge_gate is None


def test_layer_learnable_edges_differs_from_fixed():
    """Regression: with learnable_edge_weights=True the gate must
    influence the output — pushing it away from 1 must produce a
    different forward pass than the fixed-edge layer with the same
    linear weights.
    """
    torch.manual_seed(0)
    fixed = MetatronGraphLayer(8, 16, learnable_edge_weights=False)
    torch.manual_seed(0)
    learn = MetatronGraphLayer(8, 16, learnable_edge_weights=True)
    # copy linear weights so only the edge_gate path differs
    learn.w_self.load_state_dict(fixed.w_self.state_dict())
    learn.w_skip.load_state_dict(fixed.w_skip.state_dict())

    x = torch.randn(2, 13, 8)
    y0 = learn(x)
    # at init, edge_gate is all-ones so outputs should match the fixed layer
    yf = fixed(x)
    assert torch.allclose(y0, yf, atol=1e-5), "ones-gate must match fixed layer"

    # Now perturb the gate and re-run — outputs must differ.
    with torch.no_grad():
        learn.edge_gate.add_(0.5 * torch.randn_like(learn.edge_gate))
    y1 = learn(x)
    assert not torch.allclose(y1, yf, atol=1e-4)
    # gate parameter is trainable
    assert learn.edge_gate.requires_grad


def test_layer_rejects_wrong_node_count():
    layer = MetatronGraphLayer(8, 16)
    bad = torch.randn(2, 12, 8)
    try:
        layer(bad)
        raise AssertionError("expected AssertionError for non-13 node count")
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
