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
    SLOT_ACTIVATION_FACTORIES,
    phi_act,
    swap_relu_with,
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


def test_swap_relu_with_replaces_all_relus_in_resnet20():
    """Control 2: every nn.ReLU submodule in ResNet-20 must be replaced
    by the factory's output. Functional F.relu inside forward bodies
    is documented as out-of-scope (sibling helpers behave identically)."""
    from nature_inspired_networks.models import ResNet20
    m = ResNet20(num_classes=10)
    # Stem has an nn.ReLU; BasicBlocks use F.relu (functional) only.
    n_relu_before = sum(1 for mm in m.modules() if isinstance(mm, nn.ReLU))
    assert n_relu_before >= 1, "expected at least one nn.ReLU in ResNet-20 stem"
    swap_relu_with(m, lambda: nn.Tanh())
    n_relu_after = sum(1 for mm in m.modules() if isinstance(mm, nn.ReLU))
    n_tanh = sum(1 for mm in m.modules() if isinstance(mm, nn.Tanh))
    assert n_relu_after == 0, f"residual nn.ReLU count {n_relu_after}"
    assert n_tanh >= n_relu_before, (n_tanh, n_relu_before)


def test_swap_relu_with_preserves_other_layers():
    """Conv / BN / Linear modules must NOT be touched -- only nn.ReLU."""
    m = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 10),
    )
    # Snapshot id() of every non-ReLU child.
    non_relu_ids_before = {
        id(child) for child in m if not isinstance(child, nn.ReLU)
    }
    swap_relu_with(m, lambda: nn.GELU())
    non_relu_ids_after = {
        id(child) for child in m if not isinstance(child, nn.GELU)
    }
    # Same set: id() preserved for Conv/BN/Linear/Flatten/Pool.
    assert non_relu_ids_before == non_relu_ids_after, (
        "swap_relu_with mutated a non-ReLU module"
    )
    # Forward still runs end-to-end.
    y = m(torch.randn(2, 3, 8, 8))
    assert y.shape == (2, 10)


def _run_slot_activation_branch(slot: str, expected_cls: type) -> None:
    """Helper: build resnet20, run post_build_mutators with the slot,
    assert that every nn.ReLU was replaced by a fresh ``expected_cls``."""
    from nature_inspired_networks.models import ResNet20
    from nature_inspired_networks.runner import post_build_mutators
    m = ResNet20(num_classes=10)
    cfg = {"slot_activation": slot}
    out = post_build_mutators(m, cfg)
    assert out is m  # in-place, returns the same object.
    n_relu = sum(1 for mm in m.modules() if isinstance(mm, nn.ReLU))
    n_target = sum(1 for mm in m.modules() if isinstance(mm, expected_cls))
    assert n_relu == 0, f"{slot}: residual nn.ReLU count {n_relu}"
    assert n_target >= 1, f"{slot}: no {expected_cls.__name__} created"
    # Forward sanity.
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_slot_activation_tanh_dispatches_correctly():
    _run_slot_activation_branch("tanh", nn.Tanh)


def test_slot_activation_softplus_dispatches_correctly():
    _run_slot_activation_branch("softplus", nn.Softplus)


def test_slot_activation_gelu_dispatches_correctly():
    _run_slot_activation_branch("gelu", nn.GELU)


def test_slot_activation_swish_dispatches_correctly():
    """swish is canonically SiLU; both 'swish' and 'silu' must dispatch
    to nn.SiLU (Sitzmann's SIREN-aside in the design doc names ω=1
    SIREN sin(x) as "Swish-1"; we keep the more common engineering
    convention here)."""
    _run_slot_activation_branch("swish", nn.SiLU)
    _run_slot_activation_branch("silu", nn.SiLU)


def test_slot_activation_unknown_rejected():
    """Defensive guard: unrecognised activation alias raises ValueError
    (Rule 7 -- no silent fall-through)."""
    from nature_inspired_networks.models import ResNet20
    from nature_inspired_networks.runner import post_build_mutators
    m = ResNet20(num_classes=10)
    try:
        post_build_mutators(m, {"slot_activation": "bogus_activation"})
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown slot_activation")


def test_slot_activation_factories_table_coverage():
    """SLOT_ACTIVATION_FACTORIES must list every alias required by
    Control 2 (tanh, softplus, gelu, swish) plus the silu alias."""
    required = {"tanh", "softplus", "gelu", "swish", "silu"}
    assert required.issubset(SLOT_ACTIVATION_FACTORIES.keys())
    # Each factory call must return a FRESH module (independent state).
    for name, fac in SLOT_ACTIVATION_FACTORIES.items():
        m1 = fac()
        m2 = fac()
        assert isinstance(m1, nn.Module), name
        assert m1 is not m2, f"factory {name!r} returns the same instance"


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
