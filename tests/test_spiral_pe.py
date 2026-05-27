"""Unit tests for H36 — φ-Spiral Positional Encoding.

Asserts:

* :func:`golden_spiral_pe` returns a ``(seq_len, dim)`` tensor
  (default) and ``(seq_len, 3)`` raw trajectory when
  ``learnable=True``.
* Two consecutive calls at the same seed produce bit-identical
  output (Rule 6 — no silent randomness).
* Two consecutive positions ``k`` and ``k+1`` differ — the PE
  discriminates between distinct positions.
* The raw 3-D trajectory's first two coordinates lie on the unit
  circle (``cos² + sin² == 1``) and the z-coordinate is monotone
  increasing.
* :class:`GoldenSpiralPE` ``forward(x)`` preserves ``(B, N, D)``
  shape and adds the PE to ``x``.
* When ``learnable=True``, gradients flow back through the projection
  matrix (the ``nn.Linear(3, dim)`` weight receives a non-zero
  gradient from a downstream loss).
* The module raises loudly on bad ``dim`` (< 3) and on input ``N``
  that exceeds the PE capacity.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.spiral_pe import (  # noqa: E402
    GOLDEN_ANGLE,
    GoldenSpiralPE,
    golden_spiral_pe,
)


# ---------------------------------------------------------------------------
# golden_spiral_pe — helper
# ---------------------------------------------------------------------------
def test_pe_helper_shape_default():
    pe = golden_spiral_pe(seq_len=32, dim=16)
    assert pe.shape == (32, 16)
    assert torch.isfinite(pe).all()


def test_pe_helper_shape_learnable_raw_trajectory():
    """When ``learnable=True`` the helper returns the (N, 3) raw trajectory
    so a caller can wrap it in a learnable Linear(3, D)."""
    traj = golden_spiral_pe(seq_len=16, dim=8, learnable=True)
    assert traj.shape == (16, 3)


def test_pe_helper_two_calls_deterministic():
    pe1 = golden_spiral_pe(seq_len=24, dim=12, seed=0xA17EC)
    pe2 = golden_spiral_pe(seq_len=24, dim=12, seed=0xA17EC)
    assert torch.equal(pe1, pe2), "PE not deterministic across calls"


def test_pe_helper_consecutive_positions_distinct():
    pe = golden_spiral_pe(seq_len=64, dim=16)
    for k in range(pe.shape[0] - 1):
        assert not torch.allclose(pe[k], pe[k + 1], atol=1e-6), k


def test_pe_helper_trajectory_unit_circle_and_monotone_z():
    """The first two coords of the raw trajectory lie on a unit circle
    and the third coord is monotone increasing in position index."""
    traj = golden_spiral_pe(seq_len=64, dim=8, learnable=True)
    # cos² + sin² == 1 within float tolerance.
    r = traj[:, 0] ** 2 + traj[:, 1] ** 2
    assert torch.allclose(r, torch.ones_like(r), atol=1e-6), r.max().item()
    # z monotone non-decreasing
    z = traj[:, 2]
    assert torch.all(z[1:] >= z[:-1])
    # And golden-angle progression in (x, y).
    # angle(k) - angle(k-1) = GOLDEN_ANGLE (mod 2π)
    ang = torch.atan2(traj[:, 1], traj[:, 0])
    diff = (ang[1:] - ang[:-1]) % (2 * math.pi)
    assert torch.allclose(diff, torch.full_like(diff, GOLDEN_ANGLE % (2 * math.pi)), atol=1e-5)


def test_pe_helper_rejects_small_dim():
    try:
        golden_spiral_pe(seq_len=8, dim=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError on dim < 3")


# ---------------------------------------------------------------------------
# GoldenSpiralPE module
# ---------------------------------------------------------------------------
def test_module_forward_shape_preserved():
    mod = GoldenSpiralPE(seq_len=32, dim=16)
    x = torch.randn(2, 32, 16)
    y = mod(x)
    assert y.shape == (2, 32, 16)
    # PE is added (not multiplied): y - x should equal the PE.
    pe = mod.pe()
    assert torch.allclose(y - x, pe.unsqueeze(0).expand_as(y), atol=1e-6)


def test_module_learnable_grad_flows_into_linear():
    """A loss on the module output must propagate gradients into the
    learnable ``nn.Linear(3, dim)`` projection weight."""
    torch.manual_seed(0)
    mod = GoldenSpiralPE(seq_len=8, dim=12, learnable=True)
    x = torch.randn(1, 8, 12, requires_grad=False)
    out = mod(x)
    loss = out.pow(2).mean()
    loss.backward()
    # The learnable Linear weight must have a non-zero gradient.
    assert mod.proj is not None
    assert mod.proj.weight.grad is not None
    assert mod.proj.weight.grad.abs().sum().item() > 0
    # And the scale parameter must too.
    assert mod.scale.grad is not None
    assert mod.scale.grad.abs().sum().item() > 0


def test_module_non_learnable_has_no_linear():
    mod = GoldenSpiralPE(seq_len=8, dim=12, learnable=False)
    assert mod.proj is None
    # The fixed projection must be a buffer, not a Parameter.
    params = {n for n, _ in mod.named_parameters()}
    assert "proj_fixed" not in params
    # Only "scale" is a learnable Parameter on the non-learnable path.
    assert params == {"scale"}


def test_module_rejects_too_long_input():
    mod = GoldenSpiralPE(seq_len=8, dim=12)
    try:
        mod(torch.randn(1, 16, 12))
    except ValueError:
        return
    raise AssertionError("expected ValueError on N > seq_len")


def test_module_rejects_dim_mismatch():
    mod = GoldenSpiralPE(seq_len=8, dim=12)
    try:
        mod(torch.randn(1, 8, 11))
    except ValueError:
        return
    raise AssertionError("expected ValueError on D mismatch")


if __name__ == "__main__":
    import inspect

    fns = [
        v for k, v in globals().items()
        if k.startswith("test_") and inspect.isfunction(v)
    ]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
