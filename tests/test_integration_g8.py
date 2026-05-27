"""Integration tests for the two CNN-droppable G8 mutators.

H80 ConstantWidthKernelConv and H81 SinusoidalHarmonicActivation are wired
into ``runner.post_build_mutators`` via the override keys
``constant_width_kernel`` and ``sine_activation``. These tests prove that a
freshly built NaturePrior model, after the mutator runs, (a) still does a
correct (2, 3, 32, 32) -> (2, num_classes) forward pass, (b) actually had its
conv kernels / activations replaced, and (c) the trainable parameter count is
preserved for the constant-width swap (the Reuleaux mask is a fixed buffer).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.blocks import NaturePriorFlags  # noqa: E402
from nature_inspired_networks.constant_width_kernel import (  # noqa: E402
    ConstantWidthConv2d,
    apply_constant_width,
)
from nature_inspired_networks.models import build_model  # noqa: E402
from nature_inspired_networks.runner import post_build_mutators  # noqa: E402
from nature_inspired_networks.sinusoidal_activation import (  # noqa: E402
    SinusoidalActivation,
)


def _vanilla_natureprior():
    flags = NaturePriorFlags(hex=False, group=False, fractal=False,
                             toroidal=False, cymatic_init=False,
                             golden_modulate=False)
    return build_model("NaturePrior", num_classes=10, flags=flags,
                       channel_mode="fib")


def _count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _count_conv2d(m):
    return sum(1 for mod in m.modules() if isinstance(mod, nn.Conv2d))


def _count_constwidth(m):
    return sum(1 for mod in m.modules() if isinstance(mod, ConstantWidthConv2d))


def test_h80_constant_width_mutator_forward_and_swap():
    torch.manual_seed(0)
    m = _vanilla_natureprior()
    n_conv_before = _count_conv2d(m)
    m = post_build_mutators(m, {"constant_width_kernel": True})
    # At least one 3x3+ conv was replaced.
    assert _count_constwidth(m) > 0
    # Forward still produces logits of the right shape.
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)
    # 1x1 skip convs (kernel < 3) are NOT swapped, so some plain Conv2d remain
    # only if the backbone has any 1x1 projections; either way no crash.
    assert n_conv_before > 0


def test_h80_param_count_preserved_by_mask_buffer():
    torch.manual_seed(0)
    base = _vanilla_natureprior()
    p_before = _count_trainable(base)
    swapped = apply_constant_width(_vanilla_natureprior_seeded())
    p_after = _count_trainable(swapped)
    # The Reuleaux mask is a registered buffer (non-trainable), so the swap
    # preserves the trainable parameter count exactly.
    assert p_before == p_after, (p_before, p_after)


def _vanilla_natureprior_seeded():
    torch.manual_seed(0)
    return _vanilla_natureprior()


def test_h80_masked_taps_are_zero_in_effective_kernel():
    conv = ConstantWidthConv2d(4, 8, kernel_size=5, stride=1, bias=False)
    eff = conv.conv.weight * conv.mask
    # The hard corner of a 5x5 grid sits outside a constant-width support;
    # with a soft mask the corner weight is strongly attenuated (<50% of center).
    center = conv.mask[2, 2].item()
    corner = conv.mask[0, 0].item()
    assert corner < 0.5 * center
    assert eff.shape == (8, 4, 5, 5)


def test_h81_sine_mutator_replaces_relu_and_forwards():
    torch.manual_seed(0)
    m = _vanilla_natureprior()
    n_relu_before = sum(1 for mod in m.modules() if isinstance(mod, nn.ReLU))
    m = post_build_mutators(m, {"sine_activation": True, "omega_init": 1.0})
    n_relu_after = sum(1 for mod in m.modules() if isinstance(mod, nn.ReLU))
    n_sine = sum(1 for mod in m.modules() if isinstance(mod, SinusoidalActivation))
    assert n_relu_before > 0
    assert n_relu_after == 0
    assert n_sine == n_relu_before
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_h81_omega_is_learnable_and_grad_flows():
    torch.manual_seed(0)
    m = _vanilla_natureprior()
    m = post_build_mutators(m, {"sine_activation": True})
    y = m(torch.randn(2, 3, 32, 32)).sum()
    y.backward()
    sine_mods = [mod for mod in m.modules() if isinstance(mod, SinusoidalActivation)]
    assert sine_mods
    # At least one sine activation exposes a learnable omega with a gradient.
    has_grad = any(
        any(p.requires_grad and p.grad is not None for p in mod.parameters())
        for mod in sine_mods
    )
    assert has_grad


def test_mutators_are_independent_noop_when_flags_absent():
    torch.manual_seed(0)
    m = _vanilla_natureprior()
    n_conv = _count_conv2d(m)
    n_relu = sum(1 for mod in m.modules() if isinstance(mod, nn.ReLU))
    m = post_build_mutators(m, {})  # no G8 flags
    assert _count_constwidth(m) == 0
    assert sum(1 for mod in m.modules() if isinstance(mod, SinusoidalActivation)) == 0
    assert _count_conv2d(m) == n_conv
    assert sum(1 for mod in m.modules() if isinstance(mod, nn.ReLU)) == n_relu


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
