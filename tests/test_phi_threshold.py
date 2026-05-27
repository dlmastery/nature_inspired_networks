"""Unit tests for H19 — phi-Neuron Activation Threshold.

Covers:
* Canonical forward of PhiReLU on 2/3/4-D input.
* Branch: PhiReLU tau init equals 1/phi.
* Branch: PhiAdaptiveReLU dynamic threshold (training vs eval modes).
* Regression: forward equals max(0, x - tau) exactly.
* Edge case: gradient flows back to tau.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.priors import PHI  # noqa: E402
from nature_inspired_networks.phi_threshold import (  # noqa: E402
    PHI_RECIPROCAL,
    PhiAdaptiveReLU,
    PhiReLU,
    PhiReLUNaturePriorNet,
)


def test_phi_reciprocal_value():
    assert abs(PHI_RECIPROCAL - 1.0 / PHI) < 1e-12
    assert abs(PHI_RECIPROCAL - 0.6180339887) < 1e-8


def test_phi_relu_tau_init_is_one_over_phi():
    """Canonical: every channel starts at tau = 1/phi."""
    layer = PhiReLU(num_channels=8)
    assert layer.tau.shape == (8,)
    assert torch.allclose(layer.tau, torch.full((8,), 1.0 / PHI), atol=1e-7)


def test_phi_relu_forward_equals_relu_of_x_minus_tau_2d():
    layer = PhiReLU(num_channels=4)
    x = torch.tensor([[0.5, 0.8, 1.0, 2.0]])
    y = layer(x)
    expected = torch.relu(x - 1.0 / PHI)
    assert torch.allclose(y, expected, atol=1e-7), (y, expected)


def test_phi_relu_forward_4d_conv_layout():
    """Branch: 4-D conv layout broadcasts tau as (1, C, 1, 1)."""
    layer = PhiReLU(num_channels=3)
    x = torch.randn(2, 3, 4, 4)
    y = layer(x)
    expected = torch.relu(x - (1.0 / PHI))
    assert torch.allclose(y, expected, atol=1e-7)
    assert y.shape == x.shape


def test_phi_relu_forward_3d_sequence_layout():
    """Branch: 3-D (B, T, C) layout broadcasts tau as (1, 1, C)."""
    layer = PhiReLU(num_channels=5)
    x = torch.randn(2, 7, 5)
    y = layer(x)
    expected = torch.relu(x - (1.0 / PHI))
    assert torch.allclose(y, expected, atol=1e-7)
    assert y.shape == x.shape


def test_phi_relu_rejects_unsupported_ndim():
    """Edge case: 1-D input is not a supported layout."""
    layer = PhiReLU(num_channels=4)
    try:
        layer(torch.randn(4))
        raise AssertionError("expected ValueError on 1-D input")
    except ValueError:
        pass


def test_phi_relu_gradient_flows_to_tau():
    """Regression: tau must accumulate gradient on backward; the
    cortical-interneuron init is only useful if it can adapt."""
    layer = PhiReLU(num_channels=3)
    x = torch.full((1, 3), 1.0)  # all above threshold -> identity-ish path
    y = layer(x).sum()
    y.backward()
    assert layer.tau.grad is not None
    # Each channel got x=1 and threshold=1/phi -> y = 1 - 1/phi for each.
    # d(sum)/d(tau_c) = -1 for each surviving channel.
    assert torch.allclose(layer.tau.grad, torch.full((3,), -1.0), atol=1e-6)


def test_phi_relu_per_channel_independent_thresholds():
    """Branch: per-channel tau values, not a single scalar."""
    layer = PhiReLU(num_channels=4)
    with torch.no_grad():
        layer.tau.copy_(torch.tensor([0.0, 0.5, 1.0, 2.0]))
    x = torch.tensor([[0.7, 0.7, 0.7, 0.7]])
    y = layer(x)
    # Manual: max(0, 0.7-0), max(0, 0.7-0.5), max(0, 0.7-1.0), max(0, 0.7-2)
    expected = torch.tensor([[0.7, 0.2, 0.0, 0.0]])
    assert torch.allclose(y, expected, atol=1e-7), y


def test_phi_adaptive_relu_eval_mode_uses_running_mean():
    """Edge case: in eval mode, no EMA update; threshold = phi *
    running_mean. Catches the bug where eval mode silently consumes the
    batch mean."""
    layer = PhiAdaptiveReLU(num_channels=2)
    # seed running_mean by hand
    with torch.no_grad():
        layer.running_mean.copy_(torch.tensor([0.5, 1.0]))
    layer.eval()
    x = torch.tensor([[[[2.0]], [[2.0]]]])  # (1, 2, 1, 1)
    y = layer(x)
    # tau = phi * mean = (1.618*0.5, 1.618*1.0) ~= (0.809, 1.618)
    # x=2 > both thresholds -> identity passes through
    assert torch.allclose(y, x)
    # If we lower the input below the second channel's threshold:
    x2 = torch.tensor([[[[2.0]], [[1.0]]]])
    y2 = layer(x2)
    # Channel 0: 2.0 > 0.809 -> 2.0; Channel 1: 1.0 < 1.618 -> 0.0
    assert y2[0, 0].item() == 2.0
    assert y2[0, 1].item() == 0.0


def test_phi_adaptive_relu_training_updates_running_mean():
    """Branch: training mode updates the EMA buffer."""
    layer = PhiAdaptiveReLU(num_channels=2, ema_momentum=0.5)
    layer.train()
    before = layer.running_mean.clone()
    x = torch.ones(4, 2, 3, 3)  # per-channel mean = 1.0 for both
    _ = layer(x)
    after = layer.running_mean.clone()
    assert not torch.allclose(before, after)
    # With momentum 0.5: running = 0.5 * 0 + 0.5 * 1 = 0.5
    assert torch.allclose(after, torch.full((2,), 0.5), atol=1e-6)
    assert layer.num_batches_tracked.item() == 1


def test_phi_adaptive_relu_dynamic_rule_matches_design_doc():
    """Regression: out[i] = x[i] where x[i] > phi * mean, else 0.
    Exactly the hint formula from the assignment."""
    layer = PhiAdaptiveReLU(num_channels=1, ema_momentum=1.0)
    layer.train()
    x = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])  # (1, 1, 1, 4)
    # batch_mean = 1.5; threshold = phi * 1.5 ~= 2.427
    # only x=3 survives
    y = layer(x)
    expected = torch.tensor([[[[0.0, 0.0, 0.0, 3.0]]]])
    assert torch.allclose(y, expected, atol=1e-7), y


def test_phi_adaptive_relu_rejects_bad_args():
    try:
        PhiAdaptiveReLU(num_channels=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        PhiAdaptiveReLU(num_channels=4, ema_momentum=0.0)
        raise AssertionError("expected ValueError on momentum=0")
    except ValueError:
        pass
    try:
        PhiAdaptiveReLU(num_channels=4, ema_momentum=1.5)
        raise AssertionError("expected ValueError on momentum>1")
    except ValueError:
        pass


def test_phi_relu_naturepriornet_forward():
    """End-to-end: the variant builds and produces logits of correct shape."""
    torch.manual_seed(0)
    net = PhiReLUNaturePriorNet(num_classes=10, channel_mode="fib")
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    assert y.shape == (2, 10)
    # stem ReLU was replaced
    assert isinstance(net.stem[-1], PhiReLU)


def test_phi_relu_naturepriornet_registered_with_build_model():
    """Regression: the import-time monkey-patch must wire the new name."""
    from nature_inspired_networks.models import build_model
    net = build_model("natureprior_phi_relu", num_classes=10)
    assert isinstance(net, PhiReLUNaturePriorNet)
    assert net(torch.randn(1, 3, 32, 32)).shape == (1, 10)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
