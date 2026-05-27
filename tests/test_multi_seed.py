"""Unit tests for H60 — Three-Seed Uncertainty Quantification.

Run as a script:
    python tests/test_multi_seed.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.multi_seed import (  # noqa: E402
    aggregate_seeds,
    bootstrap_ci,
    format_seeds_table,
)


def test_aggregate_shape():
    metrics = [
        {"top1": 0.91, "loss": 0.30, "tag": "smoke"},
        {"top1": 0.92, "loss": 0.28, "tag": "smoke"},
        {"top1": 0.90, "loss": 0.31, "tag": "smoke"},
    ]
    agg = aggregate_seeds(metrics)
    # only the numeric keys survive
    assert set(agg.keys()) == {"top1", "loss"}, list(agg.keys())
    for key in ("top1", "loss"):
        stats = agg[key]
        assert set(stats.keys()) == {"mean", "std", "min", "max", "n"}, (key, stats)
        assert stats["n"] == 3
    # mean(top1) = (0.91 + 0.92 + 0.90) / 3 = 0.91
    assert math.isclose(agg["top1"]["mean"], 0.91, abs_tol=1e-9), agg["top1"]["mean"]
    # min/max correct
    assert math.isclose(agg["top1"]["min"], 0.90, abs_tol=1e-9)
    assert math.isclose(agg["top1"]["max"], 0.92, abs_tol=1e-9)


def test_aggregate_handles_empty_input():
    assert aggregate_seeds([]) == {}


def test_aggregate_skips_partial_keys():
    """Keys missing from any seed must not appear in the output."""
    metrics = [
        {"top1": 0.9, "extra": 1.0},
        {"top1": 0.91},                 # no 'extra' here
    ]
    agg = aggregate_seeds(metrics)
    assert "top1" in agg
    assert "extra" not in agg


def test_bootstrap_ci_bounds_and_order():
    """Bootstrap CI must be (lower, upper) with lower <= upper."""
    values = [0.91, 0.92, 0.90, 0.89, 0.93]
    lower, upper = bootstrap_ci(values, n_boot=500, alpha=0.05, seed=0)
    assert lower <= upper, (lower, upper)
    assert math.isfinite(lower) and math.isfinite(upper)
    # CI must be within the convex hull of the data
    assert lower >= min(values) - 1e-9, (lower, min(values))
    assert upper <= max(values) + 1e-9, (upper, max(values))


def test_bootstrap_ci_mean_within_ci():
    """Sample mean lies within the bootstrap CI for any reasonable sample."""
    values = [0.91, 0.92, 0.90, 0.89, 0.93, 0.88, 0.94]
    mean = sum(values) / len(values)
    lower, upper = bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42)
    assert lower <= mean <= upper, (lower, mean, upper)


def test_bootstrap_ci_empty_input():
    lo, hi = bootstrap_ci([], n_boot=10)
    assert math.isnan(lo) and math.isnan(hi)


def test_format_table_contains_mean_and_std():
    metrics = [
        {"top1": 0.91, "loss": 0.30},
        {"top1": 0.92, "loss": 0.28},
        {"top1": 0.90, "loss": 0.31},
    ]
    agg = aggregate_seeds(metrics)
    md = format_seeds_table(agg)
    assert "mean" in md, md
    assert "std" in md, md
    # header has the right markdown skeleton
    assert "|---|" in md
    # one row per metric (here 2 metrics -> at least 4 lines: header, sep, 2 rows)
    assert md.count("\n") >= 3


def test_format_table_empty_returns_blank():
    assert format_seeds_table({}) == ""


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
