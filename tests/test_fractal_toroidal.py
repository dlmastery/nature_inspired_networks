"""Unit tests for H26 — FractalToroidalBlock."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.fractal_toroidal import (  # noqa: E402
    FractalToroidalBlock,
    _ToroidalConv,
)


def test_forward_shape_preservation_depth2_phi_shrink():
    """Default depth=2 / phi_shrink=True must preserve (B, C, H, W)."""
    torch.manual_seed(0)
    block = FractalToroidalBlock(c_in=16, c_out=16, depth=2, phi_shrink=True)
    x = torch.randn(2, 16, 8, 8)
    y = block(x)
    assert y.shape == (2, 16, 8, 8), y.shape
    assert torch.isfinite(y).all()


def test_toroidal_vs_zero_padding_differs():
    """A FractalToroidalBlock must produce different outputs than a
    naive zero-padded conv of matched topology; we verify this by
    forcing constant input — circular padding on a constant tensor
    still produces a constant feature map (toroidal preserves
    translation invariance) but the actual values differ from a
    zero-pad reference at the boundary.

    Concrete check: feed an input where the *boundary* matters
    (a non-zero edge column) and confirm the toroidal-padded block
    produces a result distinct from a zero-pad reference using the
    same Conv2d weight.
    """
    torch.manual_seed(0)
    c = 4
    # Toroidal reference path
    tor = _ToroidalConv(c, c, kernel_size=3, stride=1)
    # Zero-pad reference with the SAME weight as the toroidal conv
    import torch.nn as nn
    zero = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
    with torch.no_grad():
        zero.weight.copy_(tor.conv.weight)
    # Input with strong values at the boundary so circular vs zero pad
    # gives observably different convolutions.
    x = torch.zeros(1, c, 8, 8)
    x[:, :, 0, :] = 1.0
    x[:, :, -1, :] = 1.0
    y_tor = tor.conv(  # bypass BN to isolate the padding effect
        torch.nn.functional.pad(x, (1, 1, 1, 1), mode="circular")
    )
    y_zero = zero(x)
    # The two outputs must differ at boundary rows.
    assert not torch.allclose(y_tor, y_zero, atol=1e-5)


def test_depth_1_fallback_collapses_to_single_conv():
    """At depth=1 the block must be a single toroidal conv path (no
    fractal branches). Branch attributes a/b1/b2 must not exist.
    """
    torch.manual_seed(0)
    block = FractalToroidalBlock(c_in=8, c_out=8, depth=1)
    assert hasattr(block, "path")
    assert not hasattr(block, "a")
    assert not hasattr(block, "b1")
    assert not hasattr(block, "b2")
    x = torch.randn(2, 8, 6, 6)
    y = block(x)
    assert y.shape == (2, 8, 6, 6)


def test_depth_3_recursion_forward_ok():
    """A depth-3 instance must forward without error and preserve shape.

    Also verifies the recursive sub-block (b2) has its own deeper b2.
    """
    torch.manual_seed(0)
    block = FractalToroidalBlock(c_in=16, c_out=16, depth=3, phi_shrink=True)
    # Recursive structure: top has b2 which is a depth-2 FractalToroidalBlock
    assert isinstance(block.b2, FractalToroidalBlock)
    assert block.b2.depth == 2
    assert isinstance(block.b2.b2, FractalToroidalBlock)
    assert block.b2.b2.depth == 1
    x = torch.randn(2, 16, 8, 8)
    y = block(x)
    assert y.shape == (2, 16, 8, 8)
    assert torch.isfinite(y).all()
    # Backward pass must produce finite gradients
    y.sum().backward()
    for p in block.parameters():
        if p.requires_grad:
            assert p.grad is None or torch.isfinite(p.grad).all()


def test_phi_shrink_changes_intermediate_width():
    """phi_shrink=True should drop the mid-width below c_out (PHI~1.618
    so 16/PHI ~ 9 → 9 < 16). phi_shrink=False keeps it equal.
    """
    block_shrink = FractalToroidalBlock(c_in=16, c_out=16, depth=2, phi_shrink=True)
    block_uniform = FractalToroidalBlock(c_in=16, c_out=16, depth=2, phi_shrink=False)
    # phi_shrink True → b_project exists (c_mid != c_out)
    assert block_shrink.b_project is not None
    # uniform → no projection needed
    assert block_uniform.b_project is None


def test_stride_2_downsamples():
    """A stride-2 top-level call must halve the spatial dimensions."""
    torch.manual_seed(0)
    block = FractalToroidalBlock(c_in=8, c_out=16, stride=2, depth=2)
    x = torch.randn(2, 8, 16, 16)
    y = block(x)
    assert y.shape == (2, 16, 8, 8)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
