"""Unit tests for H80 — ConstantWidthKernelConv (Reuleaux-masked conv)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.constant_width_kernel import (  # noqa: E402
    ConstantWidthConv2d,
    reuleaux_mask,
)


def test_mask_shape():
    """Shape: reuleaux_mask(k) returns a (k, k) tensor in [0, 1]."""
    for k in (3, 5, 7):
        m = reuleaux_mask(k)
        assert m.shape == (k, k), m.shape
        assert m.min().item() >= 0.0
        assert m.max().item() <= 1.0 + 1e-6


def test_mask_is_non_square_corners_zero_center_one():
    """Mechanism: corners suppressed, centre at full weight (non-square support)."""
    for soft in (False, True):
        m = reuleaux_mask(5, soft=soft)
        # All four corners are ~0 (outside every Reuleaux disk).
        for (i, j) in ((0, 0), (0, 4), (4, 0), (4, 4)):
            assert m[i, j].item() < 1e-2, (soft, i, j, m[i, j].item())
        # Centre is the peak == 1.0 (normalised).
        assert abs(m[2, 2].item() - 1.0) < 1e-5, (soft, m[2, 2].item())


def test_masked_forward_shape_preserved_stride1():
    """Mechanism: stride=1, padding=k//2 preserves spatial shape."""
    conv = ConstantWidthConv2d(3, 8, kernel_size=5, stride=1)
    x = torch.randn(2, 3, 32, 32)
    y = conv(x)
    assert y.shape == (2, 8, 32, 32), y.shape
    # And the effective kernel really has the corner taps zeroed (hard check
    # against the registered mask for the corner cells).
    eff = conv.conv.weight * conv.mask
    for (i, j) in ((0, 0), (0, 4), (4, 0), (4, 4)):
        assert eff[..., i, j].abs().max().item() < 1e-2, (i, j)


def test_mask_coverage_between_square_and_tiny_disk():
    """Edge case / sanity bounds: a Reuleaux support is smaller than the full
    square (k*k) but larger than a tiny radius-1 disk (~5 cells)."""
    k = 5
    m = reuleaux_mask(k, soft=False)
    coverage = m.sum().item()
    tiny_disk = 5.0   # the 4-neighbour + centre plus star is the floor we beat
    square = float(k * k)
    assert tiny_disk < coverage < square, coverage


def test_gradient_flows_through_masked_kernel():
    """Regression: gradient must flow to the underlying conv weight, and the
    masked (corner) taps must receive (near-)zero gradient because the mask is
    ~0 there — i.e. masking is applied in the forward pass, not just at init."""
    conv = ConstantWidthConv2d(2, 4, kernel_size=5, stride=1)
    x = torch.randn(1, 2, 16, 16)
    conv(x).sum().backward()
    g = conv.conv.weight.grad
    assert g is not None and torch.isfinite(g).all()
    # Corner-tap gradient should be tiny relative to centre-tap gradient.
    corner_g = g[..., 0, 0].abs().max().item()
    center_g = g[..., 2, 2].abs().max().item()
    assert corner_g < center_g, (corner_g, center_g)


def test_param_count_matches_plain_conv():
    """Edge case: masking adds no trainable parameters (mask is a buffer)."""
    cw = ConstantWidthConv2d(3, 8, kernel_size=5, bias=False)
    plain = torch.nn.Conv2d(3, 8, 5, padding=2, bias=False)
    n_cw = sum(p.numel() for p in cw.parameters())
    n_plain = sum(p.numel() for p in plain.parameters())
    assert n_cw == n_plain, (n_cw, n_plain)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
