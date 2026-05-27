"""Unit tests for the H39 PhiGELU / phi_act activation.

Asserts the four properties called out in the H39 design doc:

* identity at the origin (``phi_act(0) == 0``);
* monotonic on ``x > 0`` (Swish-like family is monotonic for β ≥ 0);
* gradient continuous through ``x = 0`` (auto-diff sanity);
* reduces to standard SiLU when ``beta = 1.0`` (β-sweep control);
* φ value preserved through ``state_dict`` round-trip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.activations import (  # noqa: E402
    PhiGELU,
    phi_act,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_phi_act_zero_at_origin():
    """phi_act(0) must be exactly 0 — required for residual safety."""
    x = torch.zeros(4)
    y = phi_act(x)
    assert torch.allclose(y, torch.zeros_like(y))
    m = PhiGELU()
    y2 = m(x)
    assert torch.allclose(y2, torch.zeros_like(y2))


def test_phi_act_monotonic_for_positive_x():
    """For x > 0, phi_act is strictly increasing.

    Mathematically: d/dx [x·σ(βx)] = σ(βx) + βx·σ(βx)·(1-σ(βx)) > 0 for
    x ≥ 0, β > 0 — both terms non-negative.
    """
    x = torch.linspace(0.0, 5.0, 200)
    y = phi_act(x)
    diffs = y[1:] - y[:-1]
    assert (diffs > 0).all().item(), "phi_act must be monotonic on x>0"


def test_phi_act_reduces_to_swish_with_beta_one():
    """phi_act(x, beta=1) == F.silu(x) within float tolerance."""
    x = torch.linspace(-3.0, 3.0, 50)
    y_phi = phi_act(x, beta=1.0)
    y_silu = F.silu(x)
    assert torch.allclose(y_phi, y_silu, atol=1e-6)


def test_phi_act_matches_formula_with_phi():
    """Spot-check the exact formula at a few points."""
    x = torch.tensor([1.0, -1.0, 2.0, -2.0])
    y = phi_act(x)
    expected = x * torch.sigmoid(x * PHI)
    assert torch.allclose(y, expected, atol=1e-7)


def test_phigelu_module_forward_shape():
    """The module preserves shape and dtype."""
    m = PhiGELU()
    x = torch.randn(4, 8, 16, 16)
    y = m(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_phigelu_gradient_continuous_at_origin():
    """Numerical-gradient sanity: f'(0) ≈ σ(0) = 0.5."""
    x = torch.tensor([0.0], requires_grad=True)
    y = phi_act(x).sum()
    y.backward()
    # f(x) = x·σ(φx); f'(0) = σ(0) + 0·σ'(0) = 0.5
    assert abs(x.grad.item() - 0.5) < 1e-6


def test_phigelu_learnable_beta_has_parameter():
    """When learnable=True the module must expose β as a Parameter."""
    m = PhiGELU(learnable=True)
    params = list(m.parameters())
    assert len(params) == 1
    assert params[0].requires_grad
    # Default init: β = φ
    assert abs(params[0].item() - PHI) < 1e-6


def test_phigelu_buffer_beta_when_not_learnable():
    """When learnable=False the β must NOT appear in m.parameters()."""
    m = PhiGELU(learnable=False)
    assert len(list(m.parameters())) == 0
    assert "beta" in dict(m.named_buffers())


def test_phigelu_drop_in_replacement_for_relu():
    """Replacing nn.ReLU in a Sequential must keep the forward signature
    unchanged (this is the wire-in contract for _GenericConv)."""
    relu_net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
    phi_net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), PhiGELU())
    x = torch.randn(2, 3, 8, 8)
    y1 = relu_net(x)
    y2 = phi_net(x)
    assert y1.shape == y2.shape


def test_phigelu_state_dict_roundtrip():
    """state_dict save+load preserves β regardless of learnable flag."""
    for learnable in (False, True):
        m1 = PhiGELU(learnable=learnable, beta_init=PHI)
        m2 = PhiGELU(learnable=learnable, beta_init=1.0)
        m2.load_state_dict(m1.state_dict())
        b1 = m1.beta.item()
        b2 = m2.beta.item()
        assert abs(b1 - b2) < 1e-7, f"learnable={learnable} β mismatch"


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
