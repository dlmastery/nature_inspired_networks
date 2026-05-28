"""Unit tests for H28 — CymaticHexConv."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import math  # noqa: E402

from nature_inspired_networks.cymatic_hex import (  # noqa: E402
    CymaticHexConv,
    _hex_tap_phases,
)
from nature_inspired_networks.priors import PHI, HexConv2d  # noqa: E402


def test_cymatic_hex_forward_shape():
    """Forward must preserve spatial shape (stride=1, padding=1)."""
    torch.manual_seed(0)
    conv = CymaticHexConv(3, 8, kernel_size=3, padding=1, toroidal=False, bias=False)
    x = torch.randn(2, 3, 12, 12)
    y = conv(x)
    assert y.shape == (2, 8, 12, 12), y.shape
    assert torch.isfinite(y).all()


def test_t0_static_gates_match_cos_phases_and_t_drifts_output():
    """Post-G3-audit semantics: at ``t=0`` the per-tap gate equals
    ``cos(phase_k)`` (constant across t), so the block produces a
    DETERMINISTIC fixed-phase reweighting of the static HexConv2d
    output. We verify the equality holds when we explicitly apply the
    same per-tap cos(phase) reweighting to a static reference, and that
    bumping ``t`` away from 0 changes the output.
    """
    torch.manual_seed(0)
    cym = CymaticHexConv(3, 8, kernel_size=3, padding=1, toroidal=False, bias=False)
    # Sanity: t is exactly 0 at init.
    assert cym.t.item() == 0.0
    # Build a static HexConv2d with identical weights and the same mask.
    ref = HexConv2d(3, 8, kernel_size=3, padding=1, toroidal=False, bias=False)
    with torch.no_grad():
        ref.conv.weight.copy_(cym.hex.conv.weight)
        # Apply the t=0 per-tap gate ( = cos(phase_k) for active taps )
        # to the static reference so the two outputs match.
        gate = torch.cos(cym.tap_phases)
        ref.conv.weight.mul_(gate.view(1, 1, *gate.shape))
    x = torch.randn(2, 3, 8, 8)
    y_cym = cym(x)
    y_ref = ref(x)
    assert torch.allclose(y_cym, y_ref, atol=1e-6), (
        (y_cym - y_ref).abs().max().item()
    )

    # Bumping t away from 0 must change the output.
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
    gate = cym._modulation()  # (k, k)
    eff = cym.hex.conv.weight * cym.hex.mask * gate
    for o in range(eff.shape[0]):
        for i in range(eff.shape[1]):
            assert eff[o, i, 0, 2].abs().item() == 0.0, (o, i)
            assert eff[o, i, 2, 0].abs().item() == 0.0, (o, i)


def test_per_tap_phi_spacing_distinct_active_modulation():
    """Post-G3-audit: per-tap (not per-channel) modulation must produce
    distinct values across the active hex taps when t != 0.
    """
    cym = CymaticHexConv(3, 6, kernel_size=3, padding=1, bias=False)
    with torch.no_grad():
        cym.t.fill_(1.0)
        cym.omega.fill_(0.5)
    gate = cym._modulation()  # (3, 3)
    assert gate.shape == (3, 3)
    # The 7 active taps should have all-distinct gate values (PHI is
    # irrational so the golden-angle-spaced cos values are distinct).
    active_mask = (cym.hex.mask != 0)
    active_vals = gate[active_mask].tolist()
    assert len(active_vals) == 7
    for i in range(len(active_vals)):
        for j in range(i + 1, len(active_vals)):
            assert abs(active_vals[i] - active_vals[j]) > 1e-6, (
                i, j, active_vals
            )


def test_h28_per_tap_phases_are_golden_angle_spaced():
    """H28 (post-G3-audit) — adjacent active-tap phases differ by
    exactly ``2 * pi * PHI / T`` where T is the number of active taps
    (7 for radius=1, 19 for radius=2).
    """
    # Radius=1: T=7.
    cym1 = CymaticHexConv(3, 8, kernel_size=3, padding=1, bias=False)
    assert cym1.n_active_taps == 7
    active_mask = (cym1.hex.mask != 0)
    active_phases = cym1.tap_phases[active_mask].tolist()
    expected_step = 2.0 * math.pi * PHI / 7.0
    # The first three active taps' phases (in row-major scan) must be
    # 0, expected_step, 2*expected_step.
    assert abs(active_phases[0] - 0.0) < 1e-6, active_phases[0]
    assert abs(active_phases[1] - expected_step) < 1e-6, (
        active_phases[1], expected_step
    )
    assert abs(active_phases[2] - 2.0 * expected_step) < 1e-6, (
        active_phases[2], expected_step
    )
    # Radius=2: T=19.
    cym2 = CymaticHexConv(3, 8, hex_kernel_radius=2, padding=1, bias=False)
    assert cym2.n_active_taps == 19
    active_mask2 = (cym2.hex.mask != 0)
    active_phases2 = cym2.tap_phases[active_mask2].tolist()
    expected_step2 = 2.0 * math.pi * PHI / 19.0
    assert abs(active_phases2[0] - 0.0) < 1e-6
    assert abs(active_phases2[1] - expected_step2) < 1e-6
    assert abs(active_phases2[2] - 2.0 * expected_step2) < 1e-6


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
