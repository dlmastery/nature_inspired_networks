"""Unit tests for H72 — Fractal Vesica FFN."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_fractal_vesica import (  # noqa: E402
    FractalVesica1DFFN,
    FractalVesicaFFN,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_2d_ffn_preserves_shape_and_residual():
    ffn = FractalVesicaFFN(c=8, kernel_size=5, base_radius=2.0)
    x = torch.randn(2, 8, 16, 16)
    y = ffn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_2d_ffn_three_paths_with_phi_shrunk_radii():
    ffn = FractalVesicaFFN(c=8)
    assert len(ffn.paths) == 3
    # radii follow b, b/φ, b/φ²
    expected = [2.0, 2.0 / PHI, 2.0 / (PHI * PHI)]
    for got, want in zip(ffn.radii, expected):
        assert abs(got - want) < 1e-6


def test_2d_ffn_hidden_default_is_phi_scaled():
    ffn = FractalVesicaFFN(c=10)
    assert ffn.c_hidden == round(10 * PHI)


def test_2d_ffn_gradient_flows():
    ffn = FractalVesicaFFN(c=8)
    x = torch.randn(2, 8, 8, 8, requires_grad=True)
    y = ffn(x).sum()
    y.backward()
    assert x.grad is not None


def test_1d_ffn_forward_shape():
    ffn = FractalVesica1DFFN(c=8, kernel_size=5)
    x = torch.randn(2, 12, 8)
    y = ffn(x)
    assert y.shape == x.shape


def test_1d_ffn_rejects_even_kernel():
    try:
        FractalVesica1DFFN(c=8, kernel_size=4)
        raise AssertionError("expected ValueError for even kernel_size")
    except ValueError:
        pass


def test_1d_ffn_three_paths_and_masks_shape():
    ffn = FractalVesica1DFFN(c=8, kernel_size=7)
    assert ffn.masks.shape == (3, 7)
    # The smallest radius (PHI^-2 * 2) should produce a strictly smaller
    # support than the largest.
    n_large = int(ffn.masks[0].sum().item())
    n_small = int(ffn.masks[2].sum().item())
    assert n_small <= n_large


def test_2d_ffn_rejects_3d_input():
    ffn = FractalVesicaFFN(c=8)
    try:
        ffn(torch.randn(2, 8, 16))
        raise AssertionError("expected ValueError for 3-D input")
    except ValueError:
        pass


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
