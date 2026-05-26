"""H21 — unit tests for the idea's implementation.

Run with:
    python ideas/21_hexagonal_phi_packing/tests.py

Output must end with "All N tests passed." or fail loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_idea_signature_present():
    """The idea must expose a signature dict so the runner can log it."""
    from implementation import idea_signature
    sig = idea_signature()
    assert isinstance(sig, dict)
    assert sig["hypothesis_id"] == "H21"
    assert sig["active_taps"] == 7
    assert "HexConv2d" in sig["primitives_touched"]


def test_phi_radial_mask_zero_at_corners():
    """The two hex-mask corners ((0,2) and (2,0)) must remain zero after
    applying the phi-radial scale -- the radial scale only modulates
    the active hex taps, never restores the corners."""
    from implementation import hex_phi_radial_mask
    m = hex_phi_radial_mask(3)
    assert m[0, 2].item() == 0.0
    assert m[2, 0].item() == 0.0


def test_phi_radial_mask_centre_to_neighbour_ratio_is_phi():
    """The centre tap weight is 1.0 and each active neighbour is 1/phi.
    The ratio centre/neighbour must equal phi to machine precision."""
    from implementation import hex_phi_radial_mask
    from nature_inspired_networks.priors import PHI
    m = hex_phi_radial_mask(3)
    centre = m[1, 1].item()
    # All six neighbours are 1/phi
    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]:
        n = m[i, j].item()
        assert abs(n - 1.0 / PHI) < 1e-7, (i, j, n)
        assert abs(centre / n - PHI) < 1e-6, (centre, n, centre / n)


def test_phi_radial_mask_seven_active_taps():
    """Same total active-tap count as the uniform hex mask: 7 cells."""
    from implementation import hex_phi_radial_mask
    m = hex_phi_radial_mask(3)
    nonzero = (m != 0).sum().item()
    assert nonzero == 7, nonzero


def test_conv_forward_shape_stride1():
    """PhiRadialHexConv2d at stride=1 preserves H, W."""
    from implementation import PhiRadialHexConv2d
    conv = PhiRadialHexConv2d(3, 8, kernel_size=3, stride=1, padding=1)
    x = torch.randn(2, 3, 16, 16)
    y = conv(x)
    assert y.shape == (2, 8, 16, 16), y.shape


def test_conv_forward_shape_stride2():
    """Stride=2 downsamples H, W by 2."""
    from implementation import PhiRadialHexConv2d
    conv = PhiRadialHexConv2d(3, 8, kernel_size=3, stride=2, padding=1)
    x = torch.randn(2, 3, 16, 16)
    y = conv(x)
    assert y.shape == (2, 8, 8, 8), y.shape


def test_effective_weight_zero_at_corners_after_forward():
    """After forward, the effective weight (self.weight * self.mask) must
    have zero at the two hex corners for every (output, input) channel.
    This is the structural invariant that says 'no information leak
    through the corner taps'."""
    from implementation import PhiRadialHexConv2d
    conv = PhiRadialHexConv2d(3, 8, kernel_size=3, padding=1)
    eff = conv.weight * conv.mask
    for o in range(eff.shape[0]):
        for i in range(eff.shape[1]):
            assert eff[o, i, 0, 2].item() == 0.0
            assert eff[o, i, 2, 0].item() == 0.0


def test_toroidal_padding_changes_output():
    """Toggling `toroidal=True` must produce a different output than
    `toroidal=False` for a non-trivial input (boundary cells see the
    opposite edge instead of zero-padded neighbours)."""
    from implementation import PhiRadialHexConv2d
    torch.manual_seed(0)
    conv_zero = PhiRadialHexConv2d(3, 8, kernel_size=3, padding=1, toroidal=False)
    torch.manual_seed(0)
    conv_torus = PhiRadialHexConv2d(3, 8, kernel_size=3, padding=1, toroidal=True)
    x = torch.randn(2, 3, 16, 16)
    y0 = conv_zero(x)
    y1 = conv_torus(x)
    assert y0.shape == y1.shape == (2, 8, 16, 16)
    assert not torch.allclose(y0, y1), "toroidal must differ from zero-pad on boundaries"


def test_variance_preserving_gain_matches_closed_form():
    """The He-init compensation gain for 1.0 + 6*(1/phi)^2 active mass
    must be sqrt(9 / 3.29...) ~ 1.654. We assert within 1e-4."""
    from implementation import variance_preserving_gain
    import math
    from nature_inspired_networks.priors import PHI
    expected = math.sqrt(9.0 / (1.0 + 6.0 * (1.0 / PHI) ** 2))
    got = variance_preserving_gain(3)
    assert abs(got - expected) < 1e-4, (got, expected)
    # Sanity bound: gain is between 1.0 and 2.0
    assert 1.0 < got < 2.0


def test_t1_3_regression_uniform_vs_phi_radial_outputs_differ():
    """T1.3 used UNIFORM weights (the legacy HexConv2d). H21 uses the
    phi-radial weighting. A regression test that catches accidental
    re-flattening of the radial scale: the H21 conv must produce a
    DIFFERENT output than a uniform-weight hex conv with the same
    weight tensor.
    """
    from implementation import PhiRadialHexConv2d
    from nature_inspired_networks.priors import HexConv2d
    torch.manual_seed(42)
    h17_phi = PhiRadialHexConv2d(3, 8, kernel_size=3, padding=1, bias=False)
    # Build a uniform HexConv2d and force its conv weight to match
    torch.manual_seed(42)
    h_uniform = HexConv2d(3, 8, kernel_size=3, padding=1, bias=False)
    # Sanity: both have shape (8, 3, 3, 3) weight
    assert h17_phi.weight.shape == h_uniform.conv.weight.shape
    # Copy the H21 conv's weight into the uniform one so we isolate the
    # mask difference (not the random init noise).
    with torch.no_grad():
        h_uniform.conv.weight.copy_(h17_phi.weight)
    x = torch.randn(2, 3, 16, 16)
    y_phi = h17_phi(x)
    y_uniform = h_uniform(x)
    assert y_phi.shape == y_uniform.shape
    assert not torch.allclose(y_phi, y_uniform, atol=1e-5), (
        "phi-radial mask did not modulate the output; "
        "the H21 prior is silently inert (regression!)"
    )


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
