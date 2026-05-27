"""Unit tests for H33 - Vesica Piscis Filter primitives.

Run as ``python tests/test_vesica_piscis.py``; finishes with
``"All N tests passed."`` on success.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.priors import PHI  # noqa: E402
from nature_inspired_networks.vesica_piscis import (  # noqa: E402
    VesicaPiscisConv2d,
    vesica_kernel_mask,
    vesica_phi_offsets,
)


def test_h33_vesica_kernel_mask_shape_and_dtype():
    """Mask stack has shape (n_circles, k, k) and is float32 binary."""
    m = vesica_kernel_mask(k=5, n_circles=3, radius=2.0, offset=1.0)
    assert m.shape == (3, 5, 5)
    assert m.dtype == torch.float32
    # binary mask
    uniq = set(m.unique().tolist())
    assert uniq.issubset({0.0, 1.0})


def test_h33_vesica_kernel_disc_area_within_tolerance():
    """Each circle's pixel count approximates pi*r^2 within rasterisation
    tolerance. For radius=2, pi*4 ~= 12.57; rasterised disc on a 5x5 grid
    yields 13 pixels (cardinal+diagonal neighbours within distance 2).
    For radius=1.5, pi*2.25 ~= 7.07; on a 5x5 grid we get 9 pixels.
    """
    for r, k in [(2.0, 7), (1.5, 5), (1.0, 5)]:
        m = vesica_kernel_mask(k=k, n_circles=1, radius=r, offset=0.0)
        # Centred disc — count should be roughly pi*r^2 within +/- 4
        # rasterisation pixels (rough but sufficient).
        count = m.sum().item()
        expected = math.pi * r * r
        assert abs(count - expected) <= 4.0, (r, count, expected)


def test_h33_vesica_kernel_offsets_shift_centres():
    """For n_circles >= 2 with positive offset, the leftmost circle's mass
    is biased left of centre and the rightmost is biased right. Tests the
    symmetric shift layout produces vesica-piscis intersections.
    """
    m = vesica_kernel_mask(k=7, n_circles=3, radius=1.5, offset=2.0)
    # x-centre of mass per circle
    xs = torch.arange(7, dtype=torch.float32)
    com_x = [(m[i].sum(dim=0) * xs).sum().item() / max(m[i].sum().item(), 1e-9) for i in range(3)]
    # left disc CoM < middle CoM < right disc CoM
    assert com_x[0] < com_x[1] - 0.1
    assert com_x[1] < com_x[2] - 0.1
    # there is at least one shared 'on' pixel between adjacent discs
    inter01 = (m[0] * m[1]).sum().item()
    inter12 = (m[1] * m[2]).sum().item()
    assert inter01 >= 1
    assert inter12 >= 1


def test_h33_vesica_conv_forward_shape():
    """Forward with default ctor preserves (B, C, H, W) at stride=1."""
    conv = VesicaPiscisConv2d(3, 8, kernel_size=5, n_circles=3, radius=2.0)
    x = torch.randn(2, 3, 16, 16)
    y = conv(x)
    assert y.shape == (2, 8, 16, 16)
    assert torch.isfinite(y).all()


def test_h33_vesica_conv_multi_path_n_circles_2_and_5():
    """The forward works for n_circles in {1, 2, 3, 5} and uses each
    masked kernel (changing path 0's weight changes the output).
    """
    for n in (1, 2, 3, 5):
        conv = VesicaPiscisConv2d(4, 6, kernel_size=5, n_circles=n)
        x = torch.randn(1, 4, 8, 8)
        y0 = conv(x)
        assert y0.shape == (1, 6, 8, 8)
        # perturb conv 0 weight: output must change
        with torch.no_grad():
            conv.convs[0].weight.add_(1.0)
        y1 = conv(x)
        assert not torch.allclose(y0, y1)


def test_h33_vesica_conv_scales_are_trainable_and_aggregate():
    """``scales`` is an nn.Parameter (sum-of-paths combiner) and the
    aggregated forward is exactly the sum-over-paths of scale[i]*conv_i.
    """
    torch.manual_seed(0)
    conv = VesicaPiscisConv2d(3, 4, kernel_size=5, n_circles=3, radius=2.0)
    assert isinstance(conv.scales, torch.nn.Parameter)
    assert conv.scales.requires_grad
    # init: each scale is 1/n_circles
    assert conv.scales.shape == (3,)
    assert torch.allclose(conv.scales.detach(), torch.full((3,), 1.0 / 3.0))
    x = torch.randn(1, 3, 8, 8)
    # manual sum-of-paths must match forward
    import torch.nn.functional as F
    manual = torch.zeros(1, 4, 8, 8)
    for i in range(3):
        w = conv.convs[i].weight * conv.masks[i].view(1, 1, 5, 5)
        manual = manual + conv.scales[i] * F.conv2d(x, w, padding=2)
    fwd = conv(x)
    assert torch.allclose(manual, fwd, atol=1e-5)
    # backprop wrt scales gives non-zero grad
    loss = fwd.sum()
    loss.backward()
    assert conv.scales.grad is not None
    assert conv.scales.grad.abs().sum().item() > 0


def test_h33_vesica_kernel_mask_invalid_inputs():
    """Constructor rejects invalid kernel / circle / radius arguments."""
    for bad in [{"k": 0}, {"n_circles": 0}, {"radius": 0.0}, {"radius": -1.0}]:
        try:
            vesica_kernel_mask(**bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass


def test_h33_vesica_phi_offsets_decay_by_phi():
    """``vesica_phi_offsets`` returns a phi-decaying offset schedule that
    starts at 0.5 and shrinks by 1/PHI per step.
    """
    offs = vesica_phi_offsets(4)
    assert len(offs) == 4
    assert abs(offs[0] - 0.5) < 1e-9
    for a, b in zip(offs[:-1], offs[1:]):
        # b == a / PHI within tolerance
        assert abs(b - a / PHI) < 1e-9


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
