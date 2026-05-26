"""H22 - unit tests for the toroidal phi-closure implementation.

Run with:
    python ideas/22_toroidal_phi_closure/tests.py

Output must end with "All N tests passed." or fail loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make project src/ AND this idea's directory importable
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(HERE))

from implementation import (  # noqa: E402
    ToroidalConv2d,
    idea_signature,
    phi_toroidal_pad,
)
from nature_inspired_networks.priors import PHI, toroidal_pad  # noqa: E402


def test_idea_signature_present():
    sig = idea_signature()
    assert isinstance(sig, dict)
    assert sig["hypothesis_id"] == "H22"
    assert "toroidal" in sig["flags_touched"]


def test_toroidal_pad_wraps_corner_to_opposite_corner():
    """Top-left pad row must equal the bottom row of the original (circular)."""
    x = torch.arange(1, 17, dtype=torch.float32).view(1, 1, 4, 4)
    y = toroidal_pad(x, 1)
    assert y.shape == (1, 1, 6, 6)
    # left padding column equals the LAST column of the original
    assert torch.equal(y[0, 0, 1:5, 0], x[0, 0, :, -1])
    # top padding row equals the LAST row of the original
    assert torch.equal(y[0, 0, 0, 1:5], x[0, 0, -1, :])


def test_phi_toroidal_pad_uses_phi_scaled_distance():
    """With phi_scale=True the effective pad is round(PHI*pad)."""
    x = torch.randn(1, 2, 8, 8)
    y_plain = phi_toroidal_pad(x, pad=1, phi_scale=False)
    y_phi = phi_toroidal_pad(x, pad=1, phi_scale=True)
    assert y_plain.shape == (1, 2, 10, 10)  # +1 each side
    eff = int(round(PHI * 1))  # = 2
    assert eff == 2
    assert y_phi.shape == (1, 2, 8 + 2 * eff, 8 + 2 * eff) == (1, 2, 12, 12)


def test_phi_toroidal_pad_zero_pad_is_identity():
    """pad=0 must short-circuit to the input tensor (no copy)."""
    x = torch.randn(2, 3, 8, 8)
    y = phi_toroidal_pad(x, pad=0, phi_scale=True)
    assert torch.equal(x, y)


def test_toroidal_conv_forward_shape_preserves_HW_with_stride1():
    """ToroidalConv2d with stride=1 and odd kernel preserves H, W."""
    conv = ToroidalConv2d(3, 8, kernel_size=3, stride=1, phi_scale=False, bias=False)
    x = torch.randn(2, 3, 16, 16)
    y = conv(x)
    # plain wrap: output shape = (H, W) because pad = kernel_size // 2
    assert y.shape == (2, 8, 16, 16)


def test_toroidal_conv_phi_scaled_grows_spatial_dim():
    """With phi_scale=True the wrap is over-sized so the output is larger."""
    conv = ToroidalConv2d(3, 8, kernel_size=3, stride=1, phi_scale=True, bias=False)
    x = torch.randn(2, 3, 16, 16)
    y = conv(x)
    # phi-scaled wrap adds (2*eff - 2*1) = 2 extra rows/cols vs plain wrap
    assert y.shape[2] > 16 and y.shape[3] > 16


def test_toroidal_pad_is_translation_equivariant_on_torus():
    """A conv with toroidal pad is shift-equivariant on the torus.

    Concretely: rolling the input on the torus then conv-with-toroidal-pad
    equals conv-with-toroidal-pad then rolling the output (the unpadded
    output, which lives on the same HxW torus). This is what makes
    circular padding the "right" choice for cyclic data: ordinary
    zero-padding breaks at the seam.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 2, 6, 6)
    # roll on the torus
    shifts = (2, 3)
    x_shifted = torch.roll(x, shifts=shifts, dims=(2, 3))

    # Plain 1x1 conv (no spatial mixing) under toroidal pad with k=3
    w = torch.randn(3, 2, 3, 3)

    import torch.nn.functional as F
    def conv_tor(t: torch.Tensor) -> torch.Tensor:
        t = toroidal_pad(t, 1)
        return F.conv2d(t, w, padding=0)

    y_a = conv_tor(x_shifted)
    y_b = torch.roll(conv_tor(x), shifts=shifts, dims=(2, 3))
    assert torch.allclose(y_a, y_b, atol=1e-5), (y_a - y_b).abs().max().item()


def test_phi_value_is_golden_ratio():
    """Regression: PHI must remain the golden ratio (1+sqrt(5))/2."""
    assert abs(PHI - (1 + 5 ** 0.5) / 2) < 1e-12


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
