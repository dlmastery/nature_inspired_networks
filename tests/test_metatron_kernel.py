"""Unit tests for H40 - Metatron Kernel Overlap primitives.

Run as ``python tests/test_metatron_kernel.py``; finishes with
``"All N tests passed."`` on success.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.metatron_kernel import (  # noqa: E402
    MetatronConv2d,
    metatron_basis_kernels,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_h40_basis_shape_default_13_k_7():
    """Default basis is (13, 7, 7) float32 with binary values in {0, 1}."""
    b = metatron_basis_kernels(k=7, n_circles=13)
    assert b.shape == (13, 7, 7)
    assert b.dtype == torch.float32
    uniq = set(b.unique().tolist())
    assert uniq.issubset({0.0, 1.0})


def test_h40_basis_center_pixel_hit_by_at_least_3_circles():
    """At least 3 of the 13 Metatron circles cover the central pixel -
    the central circle plus the inner-hex circles whose centres are at
    distance r_inner = k/4 from the centre (well within their own
    radius). This is the canonical "Metatron overlap on the central
    pixel".
    """
    for k in (7, 9, 11):
        b = metatron_basis_kernels(k=k, n_circles=13)
        cy = cx = k // 2
        overlaps = b[:, cy, cx].sum().item()
        assert overlaps >= 3, (k, overlaps)


def test_h40_basis_canonical_circle_count_1_plus_6_plus_6():
    """Truncations at n_circles in {1, 7, 13} preserve the Metatron
    structure (centre, centre+6inner, full)."""
    b1 = metatron_basis_kernels(k=9, n_circles=1)
    b7 = metatron_basis_kernels(k=9, n_circles=7)
    b13 = metatron_basis_kernels(k=9, n_circles=13)
    assert b1.shape == (1, 9, 9)
    assert b7.shape == (7, 9, 9)
    assert b13.shape == (13, 9, 9)
    # The first 1 (centre) and the next 6 (inner hex) are preserved.
    assert torch.allclose(b13[:1], b1)
    assert torch.allclose(b13[:7], b7)


def test_h40_basis_rejects_too_small_k():
    """k must be >= 3."""
    for bad in (0, 1, 2):
        try:
            metatron_basis_kernels(k=bad)
            raise AssertionError(f"expected ValueError for k={bad}")
        except ValueError:
            pass


def test_h40_conv_forward_shape():
    """MetatronConv2d forward preserves (B, C, H, W) at stride=1."""
    conv = MetatronConv2d(3, 8, kernel_size=7, n_circles=13)
    x = torch.randn(2, 3, 16, 16)
    y = conv(x)
    assert y.shape == (2, 8, 16, 16)
    assert torch.isfinite(y).all()


def test_h40_effective_kernel_shape():
    """The effective kernel has the expected (out, in, k, k) shape."""
    conv = MetatronConv2d(4, 6, kernel_size=5, n_circles=13)
    w = conv.effective_kernel()
    assert w.shape == (6, 4, 5, 5)
    assert torch.isfinite(w).all()


def test_h40_learnable_alpha_changes_output():
    """The 13 alpha mixing coefficients are nn.Parameters and changing
    them changes the forward output.
    """
    torch.manual_seed(0)
    conv = MetatronConv2d(3, 4, kernel_size=7, n_circles=13)
    assert isinstance(conv.alpha, torch.nn.Parameter)
    assert conv.alpha.shape == (13,)
    x = torch.randn(1, 3, 12, 12)
    y0 = conv(x)
    with torch.no_grad():
        conv.alpha.data += 1.0  # perturb all alpha
    y1 = conv(x)
    assert not torch.allclose(y0, y1)


def test_h40_alpha_init_is_phi_decay():
    """alpha[c] initialises to 1 / PHI**c (Rule: reuse PHI)."""
    conv = MetatronConv2d(3, 4, kernel_size=7, n_circles=13)
    a = conv.alpha.detach()
    expected = torch.tensor([1.0 / (PHI ** c) for c in range(13)])
    assert torch.allclose(a, expected, atol=1e-6)


def test_h40_basis_is_buffer_not_parameter():
    """The basis is a registered buffer (fixed geometry), not a
    learnable parameter.
    """
    conv = MetatronConv2d(3, 4, kernel_size=7, n_circles=13)
    param_ids = {id(p) for p in conv.parameters()}
    assert id(conv.basis) not in param_ids
    assert isinstance(conv.W, torch.nn.Parameter)
    # state_dict() still contains 'basis' as buffer
    sd = conv.state_dict()
    assert any(k.endswith("basis") for k in sd.keys())


def test_h40_gradient_flows_through_alpha_and_W():
    """Backprop yields non-zero gradient on both alpha and W."""
    torch.manual_seed(0)
    conv = MetatronConv2d(3, 4, kernel_size=5, n_circles=13)
    x = torch.randn(1, 3, 8, 8)
    y = conv(x)
    y.sum().backward()
    assert conv.alpha.grad is not None
    assert conv.alpha.grad.abs().sum().item() > 0
    assert conv.W.grad is not None
    assert conv.W.grad.abs().sum().item() > 0


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
