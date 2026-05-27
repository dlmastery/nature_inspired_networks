"""Unit tests for H28 — CymaticHexConv."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.cymatic_hex import CymaticHexConv  # noqa: E402
from nature_inspired_networks.priors import HexConv2d  # noqa: E402


def test_cymatic_hex_forward_shape():
    """Forward must preserve spatial shape (stride=1, padding=1)."""
    torch.manual_seed(0)
    conv = CymaticHexConv(3, 8, kernel_size=3, padding=1, toroidal=False, bias=False)
    x = torch.randn(2, 3, 12, 12)
    y = conv(x)
    assert y.shape == (2, 8, 12, 12), y.shape
    assert torch.isfinite(y).all()


def test_t0_reduces_to_static_hex_conv2d():
    """At t=0 the modulation is cos(0)=1 for every channel, so the
    block must produce *exactly* the same output as a HexConv2d that
    shares its weights. We construct the static reference by copying
    weights out of CymaticHexConv.hex.conv.
    """
    torch.manual_seed(0)
    cym = CymaticHexConv(3, 8, kernel_size=3, padding=1, toroidal=False, bias=False)
    # Sanity: t is exactly 0 at init.
    assert cym.t.item() == 0.0
    # Build a static HexConv2d with identical weights and the same mask.
    ref = HexConv2d(3, 8, kernel_size=3, padding=1, toroidal=False, bias=False)
    with torch.no_grad():
        ref.conv.weight.copy_(cym.hex.conv.weight)
    x = torch.randn(2, 3, 8, 8)
    y_cym = cym(x)
    y_ref = ref(x)
    assert torch.allclose(y_cym, y_ref, atol=1e-6), (
        (y_cym - y_ref).abs().max().item()
    )

    # And: bumping t away from 0 must change the output.
    with torch.no_grad():
        cym.t.fill_(0.5)
    y_cym_t = cym(x)
    assert not torch.allclose(y_cym_t, y_ref, atol=1e-4)


def test_omega_and_t_learnable_with_grad_flow():
    """omega and t are nn.Parameters and must receive non-trivial
    gradients when downstream loss depends on the modulation.
    """
    torch.manual_seed(0)
    cym = CymaticHexConv(3, 8, kernel_size=3, padding=1)
    # Move t to a non-zero point so the gradient w.r.t. omega is non-trivial
    # (at t=0, dcos(omega*t)/d omega = -t*sin(omega*t) = 0).
    with torch.no_grad():
        cym.t.fill_(0.3)
    assert cym.t.requires_grad
    assert cym.omega.requires_grad
    x = torch.randn(2, 3, 8, 8, requires_grad=False)
    y = cym(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert cym.t.grad is not None
    assert cym.omega.grad is not None
    assert torch.isfinite(cym.t.grad).item()
    assert torch.isfinite(cym.omega.grad).item()
    # Inner conv weights also receive gradient.
    assert cym.hex.conv.weight.grad is not None
    assert torch.isfinite(cym.hex.conv.weight.grad).all()


def test_hex_corner_mask_preserved_in_effective_kernel():
    """The 7-tap hex mask must zero the two corner taps of the
    effective (modulated) kernel regardless of (t, omega).
    """
    torch.manual_seed(0)
    cym = CymaticHexConv(3, 8, kernel_size=3, padding=1, bias=False)
    # Move (t, omega) away from defaults so modulation is non-trivial.
    with torch.no_grad():
        cym.t.fill_(0.5)
        cym.omega.fill_(2.0)
    mod = cym._modulation()  # (8,)
    eff = cym.hex.conv.weight * cym.hex.mask * mod.view(-1, 1, 1, 1)
    for o in range(eff.shape[0]):
        for i in range(eff.shape[1]):
            assert eff[o, i, 0, 2].abs().item() == 0.0, (o, i)
            assert eff[o, i, 2, 0].abs().item() == 0.0, (o, i)


def test_per_channel_phi_spacing_distinct_modulation():
    """At non-zero t, the per-channel modulation cos(omega*t + phi*c*t)
    must produce distinct values across channels (PHI is irrational, so
    consecutive channels see distinct phases).
    """
    cym = CymaticHexConv(3, 6, kernel_size=3, padding=1, bias=False)
    with torch.no_grad():
        cym.t.fill_(1.0)
        cym.omega.fill_(0.5)
    mod = cym._modulation()
    assert mod.shape == (6,)
    # No two consecutive channels should have identical modulation.
    diffs = (mod[1:] - mod[:-1]).abs()
    assert (diffs > 1e-6).all(), mod


def test_radius2_19tap_mask_path_works():
    """The wrapper must inherit HexConv2d's radius-2 path: hex_kernel_radius=2
    selects the 5x5 19-tap mask and kernel_size override.
    """
    cym = CymaticHexConv(3, 8, hex_kernel_radius=2, padding=1)
    assert cym.hex.hex_kernel_radius == 2
    assert cym.hex.mask.shape == (5, 5)
    assert cym.hex.mask.sum().item() == 19
    y = cym(torch.randn(2, 3, 8, 8))
    assert y.shape == (2, 8, 8, 8)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
