"""H58 - unit tests for the C4 avg-pool orbit reduction (DISCARDED variant).

Run with:
    python ideas/58_group_avg_pool/tests.py

Output must end with "All N tests passed." or fail loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(HERE))

from implementation import (  # noqa: E402
    idea_signature,
    make_avg_pool_group_conv,
)
from nature_inspired_networks.priors import GroupConv2d  # noqa: E402


def test_idea_signature_marks_discarded():
    sig = idea_signature()
    assert sig["hypothesis_id"] == "H58"
    assert sig["falsifier_status"] == "discarded"
    # The empirical numbers must match the archived FINDINGS.md row.
    assert abs(sig["sg_only_group_avg_top1"] - 0.6538) < 1e-4
    assert abs(sig["sg_only_group_delta_top1"] + 0.0446) < 1e-3
    assert abs(sig["sg_full_fib_avg_top1"] - 0.6686) < 1e-4
    assert abs(sig["sg_full_fib_delta_top1"] + 0.0638) < 1e-3
    assert sig["future_direction"] == "rotated_cifar10"


def test_factory_returns_group_conv_with_reduce_mean():
    conv = make_avg_pool_group_conv(3, 8, kernel_size=3, stride=1, padding=1)
    assert isinstance(conv, GroupConv2d)
    assert conv.reduce == "mean"
    assert conv.group == "c4"


def test_forward_shape_at_stride1():
    conv = make_avg_pool_group_conv(3, 8, stride=1, padding=1)
    x = torch.randn(2, 3, 16, 16)
    y = conv(x)
    assert y.shape == (2, 8, 16, 16)


def test_forward_shape_at_stride2():
    conv = make_avg_pool_group_conv(3, 8, stride=2, padding=1)
    x = torch.randn(2, 3, 16, 16)
    y = conv(x)
    assert y.shape == (2, 8, 8, 8)


def test_mean_pool_differs_from_max_pool_h58_regression():
    """The H58 hypothesis: switching reduce='mean' must materially change
    the forward output. If a future refactor silently collapsed `mean`
    onto `max`, this test catches it."""
    torch.manual_seed(0)
    g_max = GroupConv2d(3, 8, 3, stride=1, padding=1, group="c4", reduce="max")
    torch.manual_seed(0)
    g_mean = GroupConv2d(3, 8, 3, stride=1, padding=1, group="c4", reduce="mean")
    x = torch.randn(2, 3, 16, 16)
    y_max = g_max(x)
    y_mean = g_mean(x)
    assert y_max.shape == y_mean.shape == (2, 8, 16, 16)
    assert not torch.allclose(y_max, y_mean, atol=1e-4)


def test_mean_pool_output_bounded_by_max_pool_per_location():
    """Empirical sanity: mean over the 4-orbit at each spatial location
    must be <= max over the same orbit. (Holds even after the BN/skip
    pipeline downstream may invert it.)"""
    torch.manual_seed(0)
    g_max = GroupConv2d(3, 4, 3, stride=1, padding=1, group="c4", reduce="max")
    torch.manual_seed(0)
    g_mean = GroupConv2d(3, 4, 3, stride=1, padding=1, group="c4", reduce="mean")
    x = torch.randn(2, 3, 8, 8)
    y_max = g_max(x)
    y_mean = g_mean(x)
    # Mean of 4 numbers is <= max of those same 4 numbers at every position.
    assert (y_mean <= y_max + 1e-5).all().item()


def test_finite_outputs():
    """Even though H58 is discarded, the forward pass must be numerically
    finite - the verdict is about TASK metric, not numerical stability."""
    conv = make_avg_pool_group_conv(3, 8, padding=1)
    x = torch.randn(2, 3, 8, 8)
    y = conv(x)
    assert torch.isfinite(y).all().item()


def test_invalid_reduce_string_rejected():
    """A typo in a config that introduces e.g. reduce='avg' (rather than
    the canonical 'mean') must fail loud at construction, not at runtime."""
    try:
        GroupConv2d(3, 8, 3, padding=1, reduce="avg")
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for reduce='avg'")


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
