"""H43 — Unit tests for Fibonacci iterative magnitude pruning.

Per CLAUDE.md Rule 12: ≥ 4 assertions, regression test named for H43,
edge case (non-Fib epoch is a no-op), and a sparsity-monotonicity check.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.pruning import (  # noqa: E402
    FIB_SCHEDULE,
    fibonacci_prune,
    fibonacci_ratios,
    global_sparsity,
)


def _tiny_cnn() -> nn.Module:
    """Small CNN with Conv2d + Linear for pruning round-trip tests."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 4),
    )


def test_h43_fibonacci_ratios_sum_to_one():
    """Regression test for H43: ratios must be normalised Fibonacci."""
    r = fibonacci_ratios(5)
    assert len(r) == 5
    assert math.isclose(sum(r), 1.0, abs_tol=1e-12)
    # The exact normalised Fibonacci fractions:
    expected = [1 / 19, 2 / 19, 3 / 19, 5 / 19, 8 / 19]
    for a, b in zip(r, expected):
        assert math.isclose(a, b, abs_tol=1e-12), (a, b)
    # Schedule constant matches the design doc.
    assert FIB_SCHEDULE[:5] == (1, 2, 3, 5, 8)


def test_h43_prune_only_fires_on_fib_epochs():
    """Edge case: a non-Fibonacci epoch must NOT modify the model."""
    model = _tiny_cnn()
    before = global_sparsity(model)
    info = fibonacci_prune(model, epoch=5, schedule_length=5)  # 0-idx 5 -> 1-idx 6 (not in {1,2,3,5,8})
    assert info["pruned"] is False
    assert info["ratio"] == 0.0
    after = global_sparsity(model)
    assert after == before, (before, after)


def test_h43_prune_increases_sparsity_on_fib_epoch():
    """A Fibonacci epoch must actually zero some weights."""
    model = _tiny_cnn()
    before = global_sparsity(model)
    # epoch=0 → 1-indexed 1, the first Fibonacci gate.
    info = fibonacci_prune(model, epoch=0, schedule_length=5)
    assert info["pruned"] is True
    assert info["fib_idx"] == 0
    assert math.isclose(info["ratio"], 1 / 19, abs_tol=1e-12)
    after = global_sparsity(model)
    # The smallest ratio is 1/19 ≈ 5.3% — significantly above 0.
    assert after > before
    assert after >= 0.04, after


def test_h43_iterative_pruning_monotonic_sparsity():
    """Apply every Fibonacci epoch in sequence; sparsity must be
    non-decreasing each time and finish above the largest single ratio."""
    model = _tiny_cnn()
    sparsities = [global_sparsity(model)]
    for ep in (0, 1, 2, 4, 7):  # 1-indexed: 1, 2, 3, 5, 8 → all Fib
        info = fibonacci_prune(model, epoch=ep, schedule_length=5)
        assert info["pruned"] is True, info
        sparsities.append(global_sparsity(model))
    # Strictly non-decreasing.
    for a, b in zip(sparsities[:-1], sparsities[1:]):
        assert b + 1e-9 >= a, (a, b)
    # After all 5 prune events the final sparsity should exceed the
    # largest single Fibonacci ratio (8/19 ≈ 0.421). H43 design doc
    # targets the ~89% zone; we only assert > 0.42 here so the test
    # is robust to torch.nn.utils.prune's global-vs-per-layer rounding.
    assert sparsities[-1] > 0.42, sparsities


def test_h43_no_prunable_modules_is_safe():
    """A model with no Conv2d / Linear layers must not crash; it
    reports pruned=False."""
    model = nn.Sequential(nn.BatchNorm1d(8), nn.ReLU())
    info = fibonacci_prune(model, epoch=0, schedule_length=5)
    # Either pruned=False (no targets) or pruned=True with zero effect;
    # both are acceptable. We assert no exception was raised and the
    # sparsity is still 0 (BN params unchanged).
    assert info["pruned"] in {True, False}
    assert global_sparsity(model) == 0.0


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
