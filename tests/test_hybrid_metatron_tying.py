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


def test_tied_conv_effective_weight_is_masked_circle_mixture():
    """Effective weight = sum_c α_c · (W ⊙ mask_c).

    The G7-audit fix introduces 13 distinct circle MASKS so the alphas
    are not reparameterisation-redundant. The effective weight at every
    spatial position is gated by ``Σ_c α_c · mask_c[y, x]``, which is
    a non-constant 2-D pattern (NOT a single scalar).
    """
    conv = MetatronTiedConv2d(2, 4, kernel_size=5, bias=False)
    eff_w = conv.effective_weight()
    # Reconstruct the expected weighted-mask combo manually.
    combined_mask = (
        conv.alphas.view(13, 1, 1, 1, 1) * conv.masks.view(13, 1, 1, 5, 5)
    ).sum(dim=0)  # (1, 1, k, k)
    expected = conv.weight * combined_mask
    assert torch.allclose(eff_w, expected)
    # And the combined mask is NOT a constant scalar -- the per-spatial
    # contribution genuinely varies, which is what stops the alphas from
    # collapsing.
    cm = combined_mask.view(5, 5)
    assert cm.max().item() > cm.min().item() + 1e-6


def test_h74_alphas_dont_collapse_to_scalar():
    """Setting α = [1, 0, ..., 0] vs α = [0, 1, 0, ..., 0] must produce
    NUMERICALLY DIFFERENT outputs.

    The pre-fix implementation had ``W_eff = W · Σα_c``, so any two
    one-hot alpha vectors yielded identical outputs (Σα_c = 1 in both
    cases). With the 13 distinct circle MASKS, one-hot α picks out one
    spatially-localised circle, and circles 0 (centre) vs 1 (inner-hex)
    occupy different spatial regions of the kernel.
    """
    torch.manual_seed(0)
    conv = MetatronTiedConv2d(2, 4, kernel_size=5, bias=False)
    x = torch.randn(1, 2, 7, 7)
    # α = [1, 0, 0, ..., 0]: only the central circle contributes.
    with torch.no_grad():
        conv.alphas.zero_()
        conv.alphas[0] = 1.0
    y0 = conv(x)
    # α = [0, 1, 0, ..., 0]: only the first inner-hex circle.
    with torch.no_grad():
        conv.alphas.zero_()
        conv.alphas[1] = 1.0
    y1 = conv(x)
    assert not torch.allclose(y0, y1, atol=1e-5), (
        "α=[1,0,...] and α=[0,1,...] produced identical outputs -- "
        "the 13 alphas collapsed to their sum (G7-audit broken-state)."
    )


def test_h74_masks_are_thirteen_distinct_circles():
    """The mask bank has exactly 13 binary circle patterns, and at least
    two of them differ pixel-wise (i.e., the masks aren't degenerate)."""
    conv = MetatronTiedConv2d(2, 4, kernel_size=7)
    assert conv.masks.shape == (13, 7, 7)
    # Each mask has at least one active pixel.
    for c in range(13):
        assert conv.masks[c].sum().item() > 0, f"circle {c} is empty"
    # The 13 masks are not all identical.
    diffs = (conv.masks - conv.masks[0:1]).abs().sum(dim=(1, 2))
    assert (diffs > 0).any().item(), "all 13 masks are identical"


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
