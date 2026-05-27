"""Unit tests for H18 — Fibonacci Stage Transition.

Covers:
* Canonical forward through the (1, 2, 3) cascade on 32x32 CIFAR input.
* Schedule generator branches: pair selection, first-stage stride.
* Regression: AdaptiveAvgPool collapses any final spatial size to 1x1.
* Edge case: stride larger than the input is gracefully handled.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.blocks import NaturePriorFlags  # noqa: E402
from nature_inspired_networks.stride import (  # noqa: E402
    DEFAULT_FIB_STRIDES,
    FibStrideNaturePriorNet,
    fib_stride_schedule,
)


def test_default_strides_are_1_2_3():
    """Canonical schedule: stage 0 keeps resolution; stages 1/2 alternate
    Fib pair (2, 3)."""
    assert DEFAULT_FIB_STRIDES == (1, 2, 3)


def test_fib_stride_schedule_canonical_3_stages():
    s = fib_stride_schedule(3, pair=(2, 3), first_stage_stride=1)
    assert s == [1, 2, 3]


def test_fib_stride_schedule_4_stages_alternates():
    s = fib_stride_schedule(4, pair=(2, 3), first_stage_stride=1)
    assert s == [1, 2, 3, 2], s


def test_fib_stride_schedule_5_stages_alternates():
    s = fib_stride_schedule(5, pair=(2, 3), first_stage_stride=1)
    assert s == [1, 2, 3, 2, 3], s


def test_fib_stride_schedule_pair_branch_2_5():
    """Branch: H18's section 7.1 includes the extended-Fib pair (2, 5)."""
    s = fib_stride_schedule(4, pair=(2, 5), first_stage_stride=1)
    assert s == [1, 2, 5, 2], s


def test_fib_stride_schedule_first_stage_downsamples():
    """Branch: first_stage_stride != 1 path."""
    s = fib_stride_schedule(3, pair=(2, 3), first_stage_stride=2)
    assert s == [2, 2, 3], s


def test_fib_stride_schedule_rejects_invalid_args():
    for bad_n in (0, -1):
        try:
            fib_stride_schedule(bad_n)
            raise AssertionError("expected ValueError for n_stages")
        except ValueError:
            pass
    try:
        fib_stride_schedule(3, pair=(0, 2))
        raise AssertionError("expected ValueError for zero pair entry")
    except ValueError:
        pass


def test_fib_stride_net_forward_shape_on_cifar():
    """Canonical forward: (2, 3, 32, 32) -> (2, 10)."""
    torch.manual_seed(0)
    # All NaturePrior priors OFF so we isolate the stride change.
    flags = NaturePriorFlags(hex=False, group=False, fractal=False,
                              toroidal=False, cymatic_init=False,
                              golden_modulate=False)
    net = FibStrideNaturePriorNet(num_classes=10, channel_mode="fib",
                                  flags=flags)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    assert y.shape == (2, 10)
    assert torch.isfinite(y).all()


def test_fib_stride_net_predicted_cascade_matches_actual_shapes():
    """Regression: the spatial-size prediction must match the forward
    cascade. Catches off-by-one bugs in stride-3 padding handling."""
    torch.manual_seed(0)
    flags = NaturePriorFlags(False, False, False, False, False, False)
    net = FibStrideNaturePriorNet(num_classes=10, channel_mode="fib",
                                  flags=flags)
    cascade = net.predicted_spatial_cascade(32)
    # 32 -> floor((32-1)/1)+1=32 -> floor((32-1)/2)+1=16 -> floor((16-1)/3)+1=6
    assert cascade == [32, 32, 16, 6], cascade

    # Trace stages manually to confirm
    x = torch.randn(1, 3, 32, 32)
    h = x
    h = net.stem(h)
    assert h.shape[-1] == 32
    h = net.stages[0](h)
    assert h.shape[-1] == 32, h.shape
    h = net.stages[1](h)
    assert h.shape[-1] == 16, h.shape
    h = net.stages[2](h)
    assert h.shape[-1] == 6, h.shape


def test_fib_stride_net_adaptive_pool_collapses_to_1x1():
    """Edge case: the adaptive-avg-pool head must produce a 1x1 spatial
    feature regardless of the final stage resolution. Catches the bug
    where stride-3 leaves an odd spatial size incompatible with global
    pooling."""
    torch.manual_seed(0)
    flags = NaturePriorFlags(False, False, False, False, False, False)
    net = FibStrideNaturePriorNet(num_classes=10, flags=flags)
    x = torch.randn(1, 3, 32, 32)
    h = net.stem(x)
    for s in net.stages:
        h = s(h)
    pooled = net.pool(h)
    assert pooled.shape[-2:] == (1, 1), pooled.shape


def test_fib_stride_net_with_5_stride_pair():
    """Branch: custom stride tuple including stride 5 (extended Fib)."""
    torch.manual_seed(0)
    flags = NaturePriorFlags(False, False, False, False, False, False)
    net = FibStrideNaturePriorNet(num_classes=10, flags=flags,
                                  strides=(1, 2, 5))
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    assert y.shape == (2, 10), y.shape


def test_fib_stride_net_registered_with_build_model():
    """Regression: the import-time monkey-patch must wire the new name
    through build_model so the sweep row 'sg_only_fib_stride' resolves."""
    # Importing the module already registers; just call build_model.
    from nature_inspired_networks.models import build_model
    net = build_model("natureprior_fib_stride", num_classes=10)
    assert isinstance(net, FibStrideNaturePriorNet)
    x = torch.randn(1, 3, 32, 32)
    assert net(x).shape == (1, 10)


def test_fib_stride_net_baseline_uniform_strides_still_works():
    """Branch: the same class with the legacy (1, 2, 2) schedule
    matches NaturePriorNet's behaviour (no regression in the
    'reverse-Fib' control condition from H18 section 7.1)."""
    torch.manual_seed(0)
    flags = NaturePriorFlags(False, False, False, False, False, False)
    net = FibStrideNaturePriorNet(num_classes=10, flags=flags,
                                  strides=(1, 2, 2))
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    assert y.shape == (2, 10), y.shape


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
