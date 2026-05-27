"""Unit tests for H81 — SinusoidalHarmonicActivation (SIREN-style)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.sinusoidal_activation import (  # noqa: E402
    SinusoidalActivation,
    swap_relu_with_sine,
)


def test_forward_shape_preserved():
    """Shape: activation is element-wise, shape unchanged."""
    act = SinusoidalActivation()
    x = torch.randn(4, 16, 8, 8)
    y = act(x)
    assert y.shape == x.shape


def test_sin_zero_at_origin():
    """Mechanism: sin(omega * 0) == 0 for any omega."""
    for omega_init in (1.0, 30.0):
        act = SinusoidalActivation(omega_init=omega_init)
        x = torch.zeros(2, 5)
        assert torch.allclose(act(x), torch.zeros_like(x), atol=1e-7)


def test_omega_is_learnable_parameter_with_grad():
    """Mechanism: omega is an nn.Parameter and receives gradient."""
    act = SinusoidalActivation(omega_init=1.0)
    assert isinstance(act.omega, nn.Parameter)
    x = torch.randn(3, 4, requires_grad=False) + 0.5
    act(x).sum().backward()
    assert act.omega.grad is not None
    assert torch.isfinite(act.omega.grad).all()
    assert act.omega.grad.abs().item() > 0.0


def test_swap_helper_replaces_all_relu_and_forward_runs():
    """Mechanism/regression: swap helper replaces every ReLU (including nested)
    and the resulting tiny CNN still produces the right output shape."""
    cnn = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Sequential(  # nested module with its own ReLU
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(inplace=True),
        ),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 10),
    )
    n_relu_before = sum(isinstance(m, nn.ReLU) for m in cnn.modules())
    assert n_relu_before == 2
    swap_relu_with_sine(cnn, omega_init=1.0)
    n_relu_after = sum(isinstance(m, nn.ReLU) for m in cnn.modules())
    n_sine = sum(isinstance(m, SinusoidalActivation) for m in cnn.modules())
    assert n_relu_after == 0, n_relu_after
    assert n_sine == 2, n_sine
    y = cnn(torch.randn(2, 3, 16, 16))
    assert y.shape == (2, 10)


def test_periodicity_2pi_over_omega():
    """Edge case: act(x) ≈ act(x + 2π/omega) because sin has period 2π."""
    omega = 3.0
    act = SinusoidalActivation(omega_init=omega)
    x = torch.randn(5, 7)
    period = 2 * math.pi / omega
    assert torch.allclose(act(x), act(x + period), atol=1e-5)


def test_per_channel_omega_broadcasts():
    """Edge case: per-channel omega applies a distinct frequency per channel."""
    act = SinusoidalActivation(omega_init=1.0, num_channels=4, dim=1)
    assert act.omega.shape == (4,)
    with torch.no_grad():
        act.omega.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    # Channel 0 has omega=0 => sin(0*x)=0 everywhere on that channel.
    x = torch.randn(2, 4, 6, 6)
    y = act(x)
    assert torch.allclose(y[:, 0], torch.zeros_like(y[:, 0]), atol=1e-7)
    # Channel 3 (omega=3) is generally non-zero on non-zero input.
    assert y[:, 3].abs().sum().item() > 0.0


def test_non_learnable_is_buffer():
    """Regression: learnable=False stores omega as a non-trainable buffer."""
    act = SinusoidalActivation(omega_init=30.0, learnable=False)
    assert not isinstance(act.omega, nn.Parameter)
    assert all(p.requires_grad for p in act.parameters()) or len(list(act.parameters())) == 0
    # forward still works
    y = act(torch.randn(2, 3))
    assert y.shape == (2, 3)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
