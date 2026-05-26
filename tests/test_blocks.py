"""Smoke tests for NaturePriorBlock and NaturePriorNet."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.blocks import NaturePriorFlags, NaturePriorBlock  # noqa: E402
from nature_inspired_networks.models import ResNet20, NaturePriorConfig, NaturePriorNet  # noqa: E402


def _all_flag_combos():
    """Each of the 6 priors on alone + all-on + all-off."""
    yield NaturePriorFlags(False, False, False, False, False, False)
    for name in ("hex", "group", "fractal", "toroidal", "cymatic_init",
                 "golden_modulate"):
        f = NaturePriorFlags(False, False, False, False, False, False)
        setattr(f, name, True)
        yield f
    yield NaturePriorFlags(True, True, True, True, True, True)


def test_block_forward_shape_keeps_h_w():
    for f in _all_flag_combos():
        blk = NaturePriorBlock(16, 16, stride=1, flags=f)
        x = torch.randn(2, 16, 8, 8)
        y = blk(x)
        assert y.shape == (2, 16, 8, 8), (f.tag(), y.shape)


def test_block_forward_shape_downsamples_on_stride2():
    for f in _all_flag_combos():
        blk = NaturePriorBlock(16, 32, stride=2, flags=f)
        x = torch.randn(2, 16, 8, 8)
        y = blk(x)
        assert y.shape == (2, 32, 4, 4), (f.tag(), y.shape)


def test_resnet20_forward():
    m = ResNet20(num_classes=10)
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_NaturePriorNet_forward_each_channel_mode():
    for mode in ("fib", "phi", "linear"):
        cfg = NaturePriorConfig(num_classes=10, channel_mode=mode,
                              flags=NaturePriorFlags())
        m = NaturePriorNet(cfg)
        y = m(torch.randn(2, 3, 32, 32))
        assert y.shape == (2, 10), mode


def test_NaturePriorNet_stagewise_features_has_4_stages():
    cfg = NaturePriorConfig(num_classes=10)
    m = NaturePriorNet(cfg)
    feats = m.stagewise_features(torch.randn(2, 3, 32, 32))
    assert len(feats) == 4  # stem + 3 stages


def test_resnet20_param_count_in_expected_band():
    m = ResNet20(num_classes=10)
    n = sum(p.numel() for p in m.parameters())
    # Canonical ResNet-20 is ~272 k params
    assert 250_000 < n < 290_000, n


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
