"""Unit tests for H27 — Golden Spiral Graph."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.spiral_graph import (  # noqa: E402
    GOLDEN_ANGLE_RAD,
    SpiralGraphTransformerLayer,
    _golden_spiral_2d,
    golden_spiral_node_init_,
)


def test_golden_spiral_init_preserves_shape():
    """The in-place helper must overwrite the tensor while preserving
    shape and dtype.
    """
    for (N, D) in [(8, 4), (16, 32), (1, 2), (3, 7), (50, 50)]:
        emb = torch.zeros(N, D)
        out = golden_spiral_node_init_(emb, seed=0)
        assert out.shape == (N, D), (N, D, out.shape)
        # Identity return (chaining)
        assert out is emb
        # Not all-zero anymore
        assert emb.abs().sum().item() > 0.0
        assert torch.isfinite(emb).all()


def test_golden_spiral_init_seeded_determinism():
    """Same (N, D, seed) must produce identical embeddings; different
    seeds must produce different embeddings.
    """
    a = torch.zeros(12, 16)
    b = torch.zeros(12, 16)
    c = torch.zeros(12, 16)
    golden_spiral_node_init_(a, seed=42)
    golden_spiral_node_init_(b, seed=42)
    golden_spiral_node_init_(c, seed=43)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_golden_spiral_init_no_nan():
    """No NaNs / infs for a range of N, D combinations including N==1,
    N>D, N<D, and N==D.
    """
    for (N, D) in [(1, 8), (8, 1), (4, 4), (100, 7), (5, 64)]:
        emb = torch.empty(N, D)
        golden_spiral_node_init_(emb, seed=0)
        assert torch.isfinite(emb).all(), (N, D)


def test_golden_spiral_2d_layout_matches_vogel_formula():
    """Spot-check that the underlying 2-D lattice obeys
    r_k = sqrt(k+1) and theta_k = k * golden_angle.
    """
    pts = _golden_spiral_2d(5)
    assert pts.shape == (5, 2)
    for k in range(5):
        r_expected = (k + 1) ** 0.5
        theta_expected = k * GOLDEN_ANGLE_RAD
        x_expected = r_expected * torch.cos(torch.tensor(theta_expected))
        y_expected = r_expected * torch.sin(torch.tensor(theta_expected))
        assert torch.allclose(pts[k, 0], x_expected, atol=1e-5)
        assert torch.allclose(pts[k, 1], y_expected, atol=1e-5)


def test_spiral_graph_transformer_layer_forward_shape():
    """The layer must map (B, N, D) → (B, N, D) with no NaNs and
    must produce a non-trivial change (residuals + attention work).
    """
    torch.manual_seed(0)
    B, N, D = 2, 12, 16
    layer = SpiralGraphTransformerLayer(d_model=D, n_nodes=N, n_heads=4)
    x = torch.randn(B, N, D)
    y = layer(x)
    assert y.shape == (B, N, D)
    assert torch.isfinite(y).all()
    # Spiral position embedding is added, so output cannot equal input.
    assert not torch.allclose(y, x)


def test_spiral_layer_pos_embedding_initialised_via_helper():
    """Constructing the layer must populate pos_embedding with non-zero,
    non-NaN entries (i.e., the helper actually ran on the parameter).
    """
    layer = SpiralGraphTransformerLayer(d_model=8, n_nodes=10, n_heads=2)
    pe = layer.pos_embedding
    assert pe.shape == (10, 8)
    assert torch.isfinite(pe).all()
    assert pe.abs().sum().item() > 0.0
    # And the layer must propagate gradients to pos_embedding.
    x = torch.randn(2, 10, 8, requires_grad=False)
    y = layer(x).sum()
    y.backward()
    assert layer.pos_embedding.grad is not None
    assert torch.isfinite(layer.pos_embedding.grad).all()


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
