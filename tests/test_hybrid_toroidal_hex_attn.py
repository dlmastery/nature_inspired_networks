"""Unit tests for H62 — Toroidal-KV Hex Attention."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_toroidal_hex_attn import (  # noqa: E402
    ToroidalKVHexAttention,
)


def test_forward_shape_square():
    """Square N=16 = 4x4 grid → forward returns (B, N, D)."""
    attn = ToroidalKVHexAttention(dim=32, n_heads=4, hex_kernel_radius=1)
    x = torch.randn(2, 16, 32)
    y = attn(x)
    assert y.shape == (2, 16, 32)
    assert torch.isfinite(y).all()


def test_forward_shape_radius2():
    """5x5 hex mask still produces correct output shape."""
    attn = ToroidalKVHexAttention(dim=32, n_heads=4, hex_kernel_radius=2)
    x = torch.randn(2, 25, 32)
    y = attn(x)
    assert y.shape == (2, 25, 32)
    assert torch.isfinite(y).all()


def test_hex_mask_buffer_corners_are_zero():
    """The registered hex_mask must zero the two 180-symmetric corners."""
    attn = ToroidalKVHexAttention(dim=8, n_heads=2, hex_kernel_radius=1)
    m = attn.hex_mask
    assert m.shape == (3, 3)
    assert m[0, 2].item() == 0.0
    assert m[2, 0].item() == 0.0
    # 7 cells remain
    assert m.sum().item() == 7


def test_nondivisible_heads_rejected():
    try:
        ToroidalKVHexAttention(dim=10, n_heads=4)
        raise AssertionError("expected ValueError for non-divisible heads")
    except ValueError:
        pass


def test_invalid_radius_rejected():
    try:
        ToroidalKVHexAttention(dim=8, n_heads=2, hex_kernel_radius=3)
        raise AssertionError("expected ValueError for unsupported radius")
    except ValueError:
        pass


def test_non_square_n_falls_back_gracefully():
    """N=7 (prime) → falls back to (1, 7) grid; still produces output."""
    attn = ToroidalKVHexAttention(dim=8, n_heads=2, hex_kernel_radius=1)
    x = torch.randn(1, 7, 8)
    y = attn(x)
    assert y.shape == (1, 7, 8)
    assert torch.isfinite(y).all()


def test_backward_flows_through_qkv():
    """Gradients must reach the QKV projection weights."""
    attn = ToroidalKVHexAttention(dim=16, n_heads=4, hex_kernel_radius=1)
    x = torch.randn(2, 9, 16, requires_grad=True)
    y = attn(x)
    y.sum().backward()
    assert attn.qkv.weight.grad is not None
    assert (attn.qkv.weight.grad.abs().sum() > 0).item()


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
