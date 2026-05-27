"""Unit tests for H03 — Golden Spiral Resolution Scaling primitives.

Convention (matches tests/test_priors.py): run as
``python tests/test_multi_scale.py``; the file ends with
``"All N tests passed."`` on success.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.multi_scale import (  # noqa: E402
    GOLDEN_ANGLE_DEG,
    GOLDEN_ANGLE_RAD,
    GoldenSpiralResize,
    golden_spiral_lattice,
    golden_spiral_resolutions,
)
from nature_inspired_networks.models import (  # noqa: E402
    NaturePriorConfig,
    NaturePriorNet,
    build_model,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_h03_golden_angle_constants():
    """Golden angle in degrees ≈ 137.508; in radians ≈ 2.3999."""
    assert abs(GOLDEN_ANGLE_DEG - 137.508) < 0.01
    assert abs(GOLDEN_ANGLE_RAD - 2.399963) < 1e-4
    # The two must agree modulo 2*pi conversion
    assert abs(math.radians(GOLDEN_ANGLE_DEG) - GOLDEN_ANGLE_RAD) < 1e-6


def test_h03_golden_spiral_resolutions_canonical_schedule():
    """The H03 spec schedule from base=28 is [28, 45, 73, 118, 191]."""
    resns = golden_spiral_resolutions(base=28, n_stages=5)
    assert resns == [28, 45, 73, 118, 191]


def test_h03_golden_spiral_resolutions_monotonic_with_align():
    """``align`` rounds to multiples but must keep monotone growth."""
    resns = golden_spiral_resolutions(base=16, n_stages=4, align=4)
    assert all(r % 4 == 0 for r in resns)
    assert resns == sorted(resns)
    assert len(set(resns)) == len(resns), "resolutions must be strictly increasing"


def test_h03_golden_spiral_lattice_shape_and_disk():
    """Lattice must produce ``n`` points all on the closed unit disk."""
    pts = golden_spiral_lattice(50)
    assert pts.shape == (50, 2)
    radii = pts.norm(dim=-1)
    assert (radii <= 1.0 + 1e-6).all()
    # First point is at the origin (k=0)
    assert torch.allclose(pts[0], torch.zeros(2), atol=1e-6)


def test_h03_golden_spiral_lattice_edge_case_n1():
    """Edge case: n=1 must return a single point at origin without error."""
    pts = golden_spiral_lattice(1)
    assert pts.shape == (1, 2)
    assert torch.allclose(pts[0], torch.zeros(2), atol=1e-6)


def test_h03_resize_module_changes_shape_bilinear():
    """GoldenSpiralResize must resize (B, C, H, W) to (B, C, size, size)."""
    m = GoldenSpiralResize(45, mode="bilinear")
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 3, 45, 45)


def test_h03_resize_module_supports_each_interp_mode():
    """Every documented mode (bilinear/bicubic/nearest/area) must run."""
    x = torch.randn(2, 3, 32, 32)
    for mode in ("bilinear", "bicubic", "nearest", "area"):
        m = GoldenSpiralResize(45, mode=mode)
        y = m(x)
        assert y.shape == (2, 3, 45, 45), mode


def test_h03_resize_rejects_invalid_input_dims():
    """3-D input must be rejected (catches loader-stride bugs)."""
    m = GoldenSpiralResize(45)
    try:
        m(torch.randn(3, 32, 32))
        raise AssertionError("expected ValueError on 3-D input")
    except ValueError as exc:
        assert "B, C, H, W" in str(exc)


def test_h03_natureprior_with_input_resolution_forward_shape():
    """End-to-end: NaturePriorNet with input_resolution=45 must forward and
    produce (B, num_classes). This is the H03 sweep-row wiring smoke."""
    m = build_model("NaturePrior", num_classes=10,
                    input_resolution=45)
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_h03_natureprior_default_no_resize():
    """Regression: omitting input_resolution must leave the resize wrapper
    unset so existing sweep rows behave byte-for-byte as before."""
    cfg = NaturePriorConfig(num_classes=10)
    m = NaturePriorNet(cfg)
    assert m.resize is None


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
