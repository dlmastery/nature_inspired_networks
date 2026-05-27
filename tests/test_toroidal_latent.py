"""Unit tests for H78 ToroidalLatent."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.toroidal_latent import (  # noqa: E402
    ToroidalLatent,
    toroidal_distance,
)


def test_forward_shape():
    """(B, in_dim) -> (B, out_dim)."""
    torch.manual_seed(0)
    layer = ToroidalLatent(in_dim=16, out_dim=8)
    x = torch.randn(5, 16)
    y = layer(x)
    assert y.shape == (5, 8), y.shape


def test_embedded_points_lie_on_T2():
    """Each (cos, sin) pair of the T^2 embedding has unit norm."""
    torch.manual_seed(0)
    layer = ToroidalLatent(in_dim=16, out_dim=8)
    x = torch.randn(5, 16)
    angles = layer.angles(x)            # (5, 2)
    embed = layer.embed_angles(angles)  # (5, 4)
    pair1 = embed[:, 0:2].norm(dim=-1)
    pair2 = embed[:, 2:4].norm(dim=-1)
    assert torch.allclose(pair1, torch.ones_like(pair1), atol=1e-5)
    assert torch.allclose(pair2, torch.ones_like(pair2), atol=1e-5)


def test_toroidal_distance_respects_wraparound():
    """distance(0.1, 2*pi - 0.1) must be small (~0.2), NOT ~2*pi."""
    a = torch.tensor([[0.1, 0.1]])
    b = torch.tensor([[2 * math.pi - 0.1, 2 * math.pi - 0.1]])
    d = toroidal_distance(a, b)
    # per-circle wrapped dist = 0.2; combined = sqrt(0.2^2 + 0.2^2)
    expected = math.sqrt(0.2 ** 2 + 0.2 ** 2)
    assert abs(d.item() - expected) < 1e-4, d.item()
    # identical points -> zero distance
    assert toroidal_distance(a, a).item() < 1e-6
    # maximally separated on one circle -> pi on that circle
    c = torch.tensor([[0.0, 0.0]])
    e = torch.tensor([[math.pi, 0.0]])
    assert abs(toroidal_distance(c, e).item() - math.pi) < 1e-5


def test_forward_deterministic_given_seed():
    """Same seed -> identical weights -> identical forward output."""
    torch.manual_seed(123)
    layer1 = ToroidalLatent(in_dim=10, out_dim=4)
    torch.manual_seed(123)
    layer2 = ToroidalLatent(in_dim=10, out_dim=4)
    x = torch.randn(3, 10)
    y1 = layer1(x)
    y2 = layer2(x)
    assert torch.allclose(y1, y2, atol=1e-7)


def test_toroidal_distance_rejects_bad_shape():
    """Regression: trailing dim must be 2 (the two torus angles)."""
    bad = torch.randn(4, 3)
    try:
        toroidal_distance(bad, bad)
        raise AssertionError("expected ValueError for trailing dim != 2")
    except ValueError as exc:
        if "expected ValueError" in str(exc):
            raise


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
