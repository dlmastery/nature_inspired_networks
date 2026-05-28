"""Unit tests for H38 - Fractal Golden Filter primitives.

Run as ``python tests/test_fractal_filter.py``; finishes with
``"All N tests passed."`` on success.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.fractal_filter import (  # noqa: E402
    FIB_KERNELS,
    FractalGoldenFilter,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_h38_forward_shape_unchanged_default():
    """Default FIB_KERNELS = (3, 5, 8) preserves (B, C, H, W) at stride=1."""
    flt = FractalGoldenFilter(3, 8)
    x = torch.randn(2, 3, 16, 16)
    y = flt(x)
    assert y.shape == (2, 8, 16, 16)
    assert torch.isfinite(y).all()


def test_h38_forward_shape_unchanged_odd_size():
    """Spatial size that is not a multiple of 8 must also round-trip
    cleanly through the 8x8 path (we crop the even-kernel output by 1).
    """
    flt = FractalGoldenFilter(3, 4, kernel_sizes=(3, 5, 8))
    for H, W in [(7, 7), (15, 17), (32, 32)]:
        x = torch.randn(1, 3, H, W)
        y = flt(x)
        assert y.shape == (1, 4, H, W), (H, W, y.shape)


def test_h38_all_three_paths_active_by_default():
    """All 3 paths are constructed and contribute to the output: zeroing
    any one path's alpha changes the forward result.
    """
    torch.manual_seed(0)
    flt = FractalGoldenFilter(3, 4, kernel_sizes=(3, 5, 8))
    assert len(flt.path_convs) == 3
    assert len(flt.path_projs) == 3
    x = torch.randn(1, 3, 12, 12)
    y_all = flt(x)
    # Zero out path 0
    with torch.no_grad():
        saved = flt.alpha.data.clone()
        flt.alpha.data[0] = 0.0
    y_no0 = flt(x)
    with torch.no_grad():
        flt.alpha.data.copy_(saved)
    assert not torch.allclose(y_all, y_no0, atol=1e-5)


def test_h38_individual_path_disable_via_kernel_sizes_3():
    """Constructor accepts kernel_sizes=(3,) - a single-path filter has
    exactly one Conv2d + one 1x1 projection and works end-to-end.
    """
    flt = FractalGoldenFilter(3, 6, kernel_sizes=(3,))
    assert len(flt.path_convs) == 1
    assert len(flt.path_projs) == 1
    assert flt.alpha.shape == (1,)
    x = torch.randn(2, 3, 16, 16)
    y = flt(x)
    assert y.shape == (2, 6, 16, 16)


def test_h38_alpha_init_is_phi_decay_normalised():
    """Per-path scales initialise to ``[1, 1/PHI, 1/PHI**2]`` then
    normalise to sum-to-1. They are learnable nn.Parameters.
    """
    flt = FractalGoldenFilter(3, 4, kernel_sizes=(3, 5, 8))
    assert isinstance(flt.alpha, torch.nn.Parameter)
    a = flt.alpha.detach()
    assert a.shape == (3,)
    raw = torch.tensor([1.0, 1.0 / PHI, 1.0 / (PHI ** 2)])
    expected = raw / raw.sum()
    assert torch.allclose(a, expected, atol=1e-6)
    # sums to 1
    assert abs(a.sum().item() - 1.0) < 1e-6
    # strict phi-decay ordering
    assert a[0].item() > a[1].item() > a[2].item()


def test_h38_param_count_roughly_matches_per_path_sum():
    """Parameter count is approximately the sum of per-path
    (Conv2d(in, mid, k) + Conv2d(mid, out, 1)) parameter counts, plus
    the alpha vector. Check exact match (no hidden parameters).
    """
    Cin, Cout, mid = 4, 8, 8
    flt = FractalGoldenFilter(Cin, Cout, kernel_sizes=(3, 5, 8), mid_channels=mid, bias=False)
    expected = 0
    for k in (3, 5, 8):
        expected += Cin * mid * k * k  # path conv
        expected += mid * Cout * 1 * 1  # 1x1 projection
    expected += 3  # alpha
    got = sum(p.numel() for p in flt.parameters())
    assert got == expected, (got, expected)


def test_h38_fib_kernels_constant():
    """The module exposes the canonical Fibonacci-spaced default."""
    assert FIB_KERNELS == (3, 5, 8)


def test_h38_rejects_invalid_kernel_sizes():
    """Empty kernel_sizes and non-positive sizes must be rejected."""
    for bad in [(), (0,), (-1,), (3, 0, 5)]:
        try:
            FractalGoldenFilter(3, 4, kernel_sizes=bad)
            raise AssertionError(f"expected ValueError for kernel_sizes={bad}")
        except ValueError:
            pass


def test_h38_k8_path_is_pixel_aligned():
    """G4 audit MINOR regression test.

    Prior code used symmetric ``padding=k//2`` with a trailing crop,
    which shifts the k=8 path's receptive field by half a pixel
    relative to the k=3 path. Plant a unique-peak input centred on a
    single pixel, set ALL conv weights to a centred Gaussian-ish kernel
    that uniquely maximises at the kernel centre, and assert that the
    k=3 path's argmax and the k=8 path's argmax both sit on the SAME
    spatial index as the input peak — i.e. both receptive fields are
    centred on the input pixel.

    We bypass the 1x1 projection by setting it to identity, and we
    bypass the per-path alpha mixing by using single-path filters.
    """
    import math as _math
    H = W = 13
    cy, cx = H // 2, W // 2

    def _build_centred(k: int) -> FractalGoldenFilter:
        flt = FractalGoldenFilter(1, 1, kernel_sizes=(k,), bias=False)
        with torch.no_grad():
            # Inner conv: weights that uniquely peak at the kernel
            # anchor cell. For odd k this is (k//2, k//2). For even k
            # under our asymmetric pad (left/top = k//2 − 1, right/bot
            # = k//2) the kernel position that covers input pixel
            # (i, j) when computing output (i, j) is (k//2 − 1,
            # k//2 − 1) — that's the "lower" of the two candidate
            # centres for the even kernel. Peak the kernel there.
            anchor = (k - 1) // 2  # k=3 -> 1, k=8 -> 3
            conv = flt.path_convs[0]
            w = conv.weight  # shape (1, 1, k, k)
            for i in range(k):
                for j in range(k):
                    di = i - anchor
                    dj = j - anchor
                    w[0, 0, i, j] = _math.exp(-(di * di + dj * dj))
            # 1x1 projection: identity scalar.
            flt.path_projs[0].weight.fill_(1.0)
            # Alpha is a single-element vector that normalises to 1.
            flt.alpha.data.fill_(1.0)
        return flt

    flt3 = _build_centred(3)
    flt8 = _build_centred(8)

    # Unique-peak input: a small Gaussian bump at (cy, cx).
    x = torch.zeros(1, 1, H, W)
    for i in range(H):
        for j in range(W):
            x[0, 0, i, j] = _math.exp(-((i - cy) ** 2 + (j - cx) ** 2) / 2.0)

    y3 = flt3(x)
    y8 = flt8(x)
    assert y3.shape == (1, 1, H, W) and y8.shape == (1, 1, H, W)
    flat3 = y3[0, 0].flatten().argmax().item()
    flat8 = y8[0, 0].flatten().argmax().item()
    r3, c3 = divmod(flat3, W)
    r8, c8 = divmod(flat8, W)
    assert (r3, c3) == (cy, cx), (
        f"k=3 path argmax at ({r3},{c3}), expected centre ({cy},{cx})"
    )
    assert (r8, c8) == (cy, cx), (
        f"k=8 path argmax at ({r8},{c8}), expected centre ({cy},{cx}) "
        f"— G4 audit MINOR: half-pixel shift NOT fixed"
    )
    # Belt-and-braces: the two paths' argmax indices must match.
    assert (r3, c3) == (r8, c8)


def test_h38_gradient_flows_through_all_paths():
    """Backprop produces non-zero gradient on every path's conv weight
    and on the alpha vector.
    """
    torch.manual_seed(0)
    flt = FractalGoldenFilter(3, 4, kernel_sizes=(3, 5, 8))
    x = torch.randn(1, 3, 8, 8, requires_grad=False)
    y = flt(x)
    y.sum().backward()
    for i, conv in enumerate(flt.path_convs):
        assert conv.weight.grad is not None, i
        assert conv.weight.grad.abs().sum().item() > 0, i
    assert flt.alpha.grad is not None
    assert flt.alpha.grad.abs().sum().item() > 0


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
