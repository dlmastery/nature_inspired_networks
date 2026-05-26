"""Smoke tests for SacredGeoBlock and SacredGeoNet."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sacgeo.blocks import SacredFlags, SacredGeoBlock  # noqa: E402
from sacgeo.models import ResNet20, SacredGeoConfig, SacredGeoNet  # noqa: E402


def _all_flag_combos():
    """Each of the 6 priors on alone + all-on + all-off."""
    yield SacredFlags(False, False, False, False, False, False)
    for name in ("hex", "group", "fractal", "toroidal", "cymatic_init",
                 "golden_modulate"):
        f = SacredFlags(False, False, False, False, False, False)
        setattr(f, name, True)
        yield f
    yield SacredFlags(True, True, True, True, True, True)


def test_block_forward_shape_keeps_h_w():
    for f in _all_flag_combos():
        blk = SacredGeoBlock(16, 16, stride=1, flags=f)
        x = torch.randn(2, 16, 8, 8)
        y = blk(x)
        assert y.shape == (2, 16, 8, 8), (f.tag(), y.shape)


def test_block_forward_shape_downsamples_on_stride2():
    for f in _all_flag_combos():
        blk = SacredGeoBlock(16, 32, stride=2, flags=f)
        x = torch.randn(2, 16, 8, 8)
        y = blk(x)
        assert y.shape == (2, 32, 4, 4), (f.tag(), y.shape)


def test_resnet20_forward():
    m = ResNet20(num_classes=10)
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_sacredgeonet_forward_each_channel_mode():
    for mode in ("fib", "phi", "linear"):
        cfg = SacredGeoConfig(num_classes=10, channel_mode=mode,
                              flags=SacredFlags())
        m = SacredGeoNet(cfg)
        y = m(torch.randn(2, 3, 32, 32))
        assert y.shape == (2, 10), mode


def test_sacredgeonet_stagewise_features_has_4_stages():
    cfg = SacredGeoConfig(num_classes=10)
    m = SacredGeoNet(cfg)
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
