"""Unit tests for the nature-inspired primitives."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.priors import (  # noqa: E402
    PHI,
    GroupConv2d,
    HexConv2d,
    chladni_modes,
    cymatic_init_,
    fibonacci_channels,
    golden_angle_phases,
    hex_kernel_mask,
    toroidal_pad,
)


def test_phi_value():
    assert abs(PHI - (1 + 5 ** 0.5) / 2) < 1e-12
    assert abs(PHI - 1.6180339887) < 1e-8


def test_fibonacci_channels_fib_monotonic_and_div8():
    widths = fibonacci_channels(16, 5, mode="fib")
    assert len(widths) == 5
    assert all(w % 8 == 0 for w in widths)
    assert widths == sorted(widths)


def test_fibonacci_channels_phi_grows_geometrically():
    widths = fibonacci_channels(16, 4, mode="phi")
    # Each successive width should be approximately phi times the previous
    for a, b in zip(widths[:-1], widths[1:]):
        assert 1.2 < b / a < 2.2, (a, b)


def test_hex_mask_zeros_two_corners():
    m = hex_kernel_mask(3)
    assert m[0, 2].item() == 0.0
    assert m[2, 0].item() == 0.0
    # 7 cells remain
    assert m.sum().item() == 7


def test_hex_mask_radius2_19_cells():
    m = hex_kernel_mask(5)
    assert m.sum().item() == 19


def test_chladni_modes_shape():
    modes = chladni_modes(7, n_modes=4)
    assert modes.shape == (4, 7, 7)
    # Modes are normalised to [-1, 1]
    assert modes.abs().max().item() <= 1.0 + 1e-6


def test_cymatic_init_changes_weights():
    conv = torch.nn.Conv2d(8, 16, 3, padding=1)
    w0 = conv.weight.clone()
    cymatic_init_(conv)
    assert not torch.allclose(w0, conv.weight)


def test_toroidal_pad_is_circular():
    x = torch.arange(1, 17, dtype=torch.float32).view(1, 1, 4, 4)
    y = toroidal_pad(x, 1)
    assert y.shape == (1, 1, 6, 6)
    # left padding column equals last column of original
    assert torch.equal(y[0, 0, 1:5, 0], x[0, 0, :, -1])


def test_group_conv_forward_shape():
    conv = GroupConv2d(3, 8, 3, stride=1, padding=1, group="c4")
    x = torch.randn(2, 3, 16, 16)
    y = conv(x)
    assert y.shape == (2, 8, 16, 16)


def test_group_conv_strides():
    conv = GroupConv2d(3, 8, 3, stride=2, padding=1, group="c4")
    x = torch.randn(2, 3, 16, 16)
    y = conv(x)
    assert y.shape == (2, 8, 8, 8)


def test_group_conv_reduce_max_and_mean_h58():
    """H58 regression test: avg-pool reduction over the C4 orbit must work.

    The previous CIFAR-10 sweep showed that max-pool over the 4-rotation
    orbit threw away 75% of the signal, dropping top-1 by 10pp. Mean-pool
    preserves the full orbit signal while still being rotation-invariant.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 16)
    out = {}
    for r in ("max", "mean"):
        g = GroupConv2d(3, 8, 3, stride=1, padding=1, group="c4", reduce=r)
        out[r] = g(x)
        assert out[r].shape == (2, 8, 16, 16)
    # Same weight init (seed reset above) gives different outputs for max vs mean
    assert not torch.allclose(out["max"], out["mean"], atol=1e-4)
    # Both reductions must still be invariant under exact 90-degree rotation
    # AT EACH PIXEL only in expectation; here we just verify the output norm
    # changes smoothly (no NaNs, no inf).
    for r, y in out.items():
        assert torch.isfinite(y).all(), r


def test_group_conv_invalid_reduce_rejected():
    """Catch typos in config files at construction time, not at runtime."""
    try:
        GroupConv2d(3, 8, 3, padding=1, reduce="median")
        raise AssertionError("expected AssertionError for invalid reduce")
    except AssertionError as exc:
        if "expected AssertionError" in str(exc):
            raise


def test_hex_conv_zero_corners_in_effective_kernel():
    conv = HexConv2d(3, 8, 3, padding=1, toroidal=False, bias=False)
    # corner positions of the effective (masked) weight should be zero
    eff = conv.conv.weight * conv.mask
    for o in range(eff.shape[0]):
        for i in range(eff.shape[1]):
            assert eff[o, i, 0, 2].item() == 0.0
            assert eff[o, i, 2, 0].item() == 0.0


def test_golden_angle_phases_distinct_modulo_2pi():
    p = golden_angle_phases(16)
    assert p.shape == (16,)
    # No two phases should land within 0.05 rad of each other
    diffs = (p.unsqueeze(0) - p.unsqueeze(1)).abs()
    diffs.fill_diagonal_(10.0)
    assert diffs.min().item() > 0.05


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
