"""Unit tests for H74 — Metatron Overlap Weight-Tying."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_metatron_tying import (  # noqa: E402
    METATRON_N_CIRCLES,
    MetatronTiedConv2d,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_metatron_n_circles_is_13():
    assert METATRON_N_CIRCLES == 13
    assert MetatronTiedConv2d.n_circles() == 13


def test_tied_conv_forward_shape():
    conv = MetatronTiedConv2d(3, 8, kernel_size=3)
    x = torch.randn(2, 3, 16, 16)
    y = conv(x)
    assert y.shape == (2, 8, 16, 16)
    assert torch.isfinite(y).all()


def test_tied_conv_param_count_is_kernel_plus_13():
    conv = MetatronTiedConv2d(3, 8, kernel_size=3)
    # weight = 8*3*3*3 = 216; alphas = 13; total = 229
    total = sum(p.numel() for p in conv.parameters())
    assert total == 216 + 13


def test_tied_conv_alphas_init_uniform():
    conv = MetatronTiedConv2d(3, 8, kernel_size=3)
    assert conv.alphas.shape == (13,)
    assert torch.allclose(conv.alphas, torch.full((13,), 1.0 / 13.0), atol=1e-6)


def test_tied_conv_effective_weight_scales_with_alpha_sum():
    conv = MetatronTiedConv2d(2, 4, kernel_size=3, bias=False)
    eff_w = conv.effective_weight()
    expected = conv.weight * conv.alphas.sum()
    assert torch.allclose(eff_w, expected)


def test_tied_conv_compression_vs_untied_bank():
    """An untied 13-conv bank has 13× the kernel params; the tied
    surface has 1× + 13. Ratio should be > 0.9."""
    conv = MetatronTiedConv2d(3, 16, kernel_size=3)
    r = conv.param_compression_ratio()
    assert r > 0.9
    # At init the ratio is 1 - (1/13 + tiny) which exceeds 0.92.
    assert r < 1.0


def test_tied_conv_gradient_flows_to_weight_and_alphas():
    conv = MetatronTiedConv2d(3, 4, kernel_size=3)
    x = torch.randn(2, 3, 8, 8)
    y = conv(x).sum()
    y.backward()
    assert conv.weight.grad is not None
    assert conv.alphas.grad is not None


def test_tied_conv_matches_F_conv2d_with_effective_weight():
    """Forward must equal a manual conv2d using the effective weight."""
    conv = MetatronTiedConv2d(3, 4, kernel_size=3)
    x = torch.randn(2, 3, 8, 8)
    y = conv(x)
    ref = F.conv2d(x, conv.effective_weight(), conv.bias, stride=1, padding=1)
    assert torch.allclose(y, ref)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
