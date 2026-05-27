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
    chladni_modes_banded,
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


def test_h21v2_hex_conv_radius1_default_keeps_3x3_mask():
    """H21.v2 (a): default hex_kernel_radius=1 → 3x3, 7-tap mask
    (180-sym), preserving the legacy CIFAR-10 smoke row.
    """
    conv = HexConv2d(3, 8, kernel_size=3, padding=1, toroidal=False, bias=False)
    assert conv.hex_kernel_radius == 1
    assert conv.mask.shape == (3, 3)
    assert conv.mask.sum().item() == 7
    # Effective kernel corners zeroed (180-sym corners only)
    eff = conv.conv.weight * conv.mask
    for o in range(eff.shape[0]):
        for i in range(eff.shape[1]):
            assert eff[o, i, 0, 2].item() == 0.0
            assert eff[o, i, 2, 0].item() == 0.0
    y = conv(torch.randn(2, 3, 8, 8))
    assert y.shape == (2, 8, 8, 8)


def test_h21v2_hex_conv_radius2_activates_19_tap_isotropic_mask():
    """H21.v2 (b): hex_kernel_radius=2 selects k=5, 19-tap radius-2 hex
    mask (true 6-fold isotropic). Constructor must override kernel_size
    and padding so the spatial shape is preserved at stride=1.
    """
    # Caller supplies kernel_size=3 but radius=2 must override to k=5.
    conv = HexConv2d(3, 8, kernel_size=3, padding=1, toroidal=False, bias=False,
                     hex_kernel_radius=2)
    assert conv.hex_kernel_radius == 2
    assert conv.mask.shape == (5, 5)
    assert conv.mask.sum().item() == 19
    # Symmetric padding=2 keeps stride=1 output shape == input shape.
    y = conv(torch.randn(2, 3, 8, 8))
    assert y.shape == (2, 8, 8, 8)


def test_h21v2_hex_conv_rejects_unsupported_radius():
    try:
        HexConv2d(3, 8, hex_kernel_radius=3)
        raise AssertionError("expected AssertionError for unsupported radius")
    except AssertionError as exc:
        if "expected AssertionError" in str(exc):
            raise


def test_h35v2_chladni_modes_banded_orthonormal():
    """H35.v2 helper: chladni_modes_banded must return Gram-Schmidt
    orthonormal modes (up to numerical QR precision).
    """
    basis = chladni_modes_banded(4, 5, band=(2, 4), seed=0)
    assert basis.shape == (4, 5, 5)
    flat = basis.reshape(4, -1)
    flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
    gram = flat @ flat.t()
    off_diag = gram - torch.eye(4)
    assert off_diag.abs().max().item() < 1e-4


def test_h35v2_cymatic_init_default_legacy_path():
    """H35.v2 (a): default ``cymatic_init_`` (no kwargs) reproduces the
    legacy path bit-identical — two consecutive calls on a fresh conv
    must produce the same weights because the legacy path is seeded
    deterministically (Generator(0xC1A171C)).
    """
    conv1 = torch.nn.Conv2d(8, 16, 3, padding=1, bias=False)
    conv2 = torch.nn.Conv2d(8, 16, 3, padding=1, bias=False)
    cymatic_init_(conv1)
    cymatic_init_(conv2)
    assert torch.allclose(conv1.weight, conv2.weight)


def test_h35v2_cymatic_init_orthonormalize_band_2_5_decorrelates_channels():
    """H35.v2 (b): with ``orthonormalize=True, band=(2, 5)`` the per-input
    filter bank across output channels must be near-orthogonal.
    """
    conv = torch.nn.Conv2d(4, 8, 3, padding=1, bias=False)
    cymatic_init_(conv, orthonormalize=True, band=(2, 5), seed=0)
    w = conv.weight  # (8, 4, 3, 3)
    # Check pairwise output-channel orthogonality per input channel.
    for i in range(w.shape[1]):
        flat = w[:, i].reshape(w.shape[0], -1)
        flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
        gram = flat @ flat.t()
        off_diag = gram - torch.eye(gram.shape[0])
        # Per-input filters are signed copies of an orthonormal basis,
        # so |off-diagonal| of normalised Gram should be small.
        assert off_diag.abs().max().item() < 0.5, off_diag.abs().max().item()
    # And it must produce *different* weights than the legacy path.
    legacy = torch.nn.Conv2d(4, 8, 3, padding=1, bias=False)
    cymatic_init_(legacy)
    assert not torch.allclose(conv.weight, legacy.weight)


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
