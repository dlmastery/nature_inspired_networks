"""Unit tests for H08 — Dynamic phi-Growth.

Asserts (per the H08 design doc Q&A section):
  - schedule helper produces the canonical Fibonacci epochs;
  - growth events fire exactly at those epochs and nowhere else;
  - grown model preserves forward output shape;
  - parameter count grows monotonically across the schedule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.blocks import NaturePriorFlags  # noqa: E402
from nature_inspired_networks.dynamic_growth import (  # noqa: E402
    DynamicGrowthCallback,
    fib_growth_schedule,
    grow_model,
)
from nature_inspired_networks.models import NaturePriorConfig, NaturePriorNet  # noqa: E402


def _minimal_factory():
    """A tiny NaturePriorNet factory with all the expensive priors off so
    the test suite stays under a second."""
    flags = NaturePriorFlags(
        hex=False, group=False, fractal=False, toroidal=False,
        cymatic_init=False, golden_modulate=False,
    )
    cfg = NaturePriorConfig(num_classes=10, channel_mode="fib", flags=flags)
    return lambda: NaturePriorNet(cfg)


def test_fib_growth_schedule_default_is_canonical():
    """Default schedule must be the canonical (3, 5, 8, 13) tuple from H08."""
    s = fib_growth_schedule()
    assert s == (3, 5, 8, 13), s


def test_fib_growth_schedule_alt_start_index():
    """start_index=2 gives the (2, 3, 5, 8) Fibonacci slice."""
    s = fib_growth_schedule(n_events=4, start_index=2)
    assert s == (2, 3, 5, 8), s


def test_callback_triggers_only_on_fib_epochs():
    """Test (1) from prompt: growth must fire EXACTLY at Fibonacci-indexed
    epochs and not on any other epoch.
    """
    torch.manual_seed(0)
    cb = DynamicGrowthCallback(_minimal_factory(),
                               fib_schedule=(3, 5, 8, 13))
    p0 = cb.total_param_count()
    history = []
    for epoch in range(0, 15):
        cb.step(epoch)
        history.append((epoch, cb.total_param_count()))
    # Verify the callback recorded events at exactly (3, 5, 8, 13)
    assert cb.fired == {3, 5, 8, 13}, cb.fired
    # Non-Fib epochs must have left the param count unchanged
    expected_growth_epochs = {3, 5, 8, 13}
    for i in range(1, len(history)):
        prev_p = history[i - 1][1]
        cur_e, cur_p = history[i]
        if cur_e in expected_growth_epochs:
            assert cur_p > prev_p, (
                f"expected growth at epoch {cur_e}, "
                f"params {prev_p} -> {cur_p}"
            )
        else:
            assert cur_p == prev_p, (
                f"unexpected growth at non-Fib epoch {cur_e}, "
                f"params {prev_p} -> {cur_p}"
            )
    # And the param count must have strictly increased overall
    assert cb.total_param_count() > p0


def test_grow_model_preserves_forward_shape():
    """Test (2) from prompt: after grow_model the forward output shape is
    bit-identical (B, num_classes)."""
    torch.manual_seed(0)
    factory = _minimal_factory()
    model = factory()
    x = torch.randn(2, 3, 32, 32)
    y_before = model(x)
    assert y_before.shape == (2, 10)
    grow_model(model, n_extra_blocks=2)
    y_after = model(x)
    assert y_after.shape == (2, 10), y_after.shape


def test_param_count_grows_monotonically():
    """Test (3) from prompt: each successive growth event increases the
    total parameter count (strict monotone)."""
    torch.manual_seed(0)
    cb = DynamicGrowthCallback(_minimal_factory(),
                               fib_schedule=(3, 5, 8, 13))
    counts = [cb.total_param_count()]
    for epoch in cb.fib_schedule:
        cb.step(epoch)
        counts.append(cb.total_param_count())
    # Strict monotone: counts[i] < counts[i+1] for all i
    for a, b in zip(counts[:-1], counts[1:]):
        assert a < b, (a, b)


def test_no_growth_at_non_fib_epoch():
    """Test (4) from prompt (negative case): calling step at an epoch
    NOT in the schedule must not change anything.
    """
    torch.manual_seed(0)
    cb = DynamicGrowthCallback(_minimal_factory(),
                               fib_schedule=(3, 5, 8, 13))
    p0 = cb.total_param_count()
    for epoch in (0, 1, 2, 4, 6, 7, 9, 10, 11, 12):  # all non-Fib
        cb.step(epoch)
    assert cb.fired == set()
    assert cb.total_param_count() == p0


def test_callback_idempotent_on_repeated_step():
    """Calling step twice on the same Fib epoch must NOT grow twice."""
    torch.manual_seed(0)
    cb = DynamicGrowthCallback(_minimal_factory(),
                               fib_schedule=(3, 5))
    cb.step(3)
    p_after_first = cb.total_param_count()
    cb.step(3)  # repeat
    p_after_repeat = cb.total_param_count()
    assert p_after_first == p_after_repeat


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
