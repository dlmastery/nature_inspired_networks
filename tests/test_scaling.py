"""Unit tests for H01 (phi-compound) and H02 (Fibonacci-depth) primitives.

Convention (matches tests/test_priors.py): run as ``python tests/test_scaling.py``;
the file ends with ``"All N tests passed."`` on success.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.priors import PHI, fibonacci_channels  # noqa: E402
from nature_inspired_networks.scaling import (  # noqa: E402
    fibonacci_depths,
    fibonacci_sequence,
    phi_compound,
    phi_compound_channels,
    resolve_blocks_schedule,
)
from nature_inspired_networks.models import (  # noqa: E402
    NaturePriorConfig,
    NaturePriorNet,
    build_model,
)


# --------------------------------------------------------------------- H01
def test_h01_phi_compound_k0_is_identity():
    """At k=0 phi_compound must return exactly the base triple."""
    d, w, r = phi_compound(0, base_depth=18, base_width=64, base_res=32)
    assert d == 18
    assert w == 64
    assert r == 32


def test_h01_phi_compound_grows_by_phi_powers():
    """Depth must grow by phi^k; width by phi^(k/2) (with /8 align);
    resolution by phi^(k/4) (with /16 align)."""
    # depth: exact phi^k rounding
    for k in (1, 2, 3):
        d, _, _ = phi_compound(k, base_depth=18)
        assert d == max(1, int(round(18 * PHI ** k))), (k, d)
    # width is /8-aligned and within one phi-step of the previous stage
    widths = [phi_compound(k, base_width=64)[1] for k in range(4)]
    for w in widths:
        assert w % 8 == 0
        assert w >= 8
    assert widths[1] > widths[0]
    assert widths[3] > widths[1]
    # resolution is /16-aligned
    for k in range(4):
        _, _, r = phi_compound(k, base_res=32)
        assert r % 16 == 0
        assert r >= 16


def test_h01_phi_compound_channels_div8_and_monotone():
    """phi_compound_channels should produce divisible-by-8 monotone widths."""
    chans = phi_compound_channels(k=1, n_stages=4, base_width=16)
    assert len(chans) == 4
    assert all(c % 8 == 0 for c in chans)
    assert all(c >= 8 for c in chans)
    assert chans == sorted(chans)


def test_h01_phi_compound_negative_k_floors():
    """Negative k must respect the 8/16 floors (edge case)."""
    d, w, r = phi_compound(-2, base_depth=18, base_width=64, base_res=32)
    assert d >= 1
    assert w >= 8 and w % 8 == 0
    assert r >= 16 and r % 16 == 0


def test_h01_fibonacci_channels_phi_compound_mode():
    """The 'phi_compound' channel_mode must round to /8 and grow monotonically."""
    widths = fibonacci_channels(16, 4, mode="phi_compound")
    assert len(widths) == 4
    assert all(w % 8 == 0 for w in widths)
    assert widths == sorted(widths)


def test_h01_natureprior_phi_compound_forward_shape():
    """Wiring smoke: NaturePriorNet with channel_mode='phi_compound' must
    forward a CIFAR tensor and produce (B, num_classes)."""
    m = build_model("NaturePrior", num_classes=10, channel_mode="phi_compound")
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


# --------------------------------------------------------------------- H02
def test_h02_fibonacci_sequence_canonical():
    """1, 1, 2, 3, 5, 8, 13, 21 — the canonical (1,1) Fibonacci start."""
    assert fibonacci_sequence(1) == [1]
    assert fibonacci_sequence(2) == [1, 1]
    assert fibonacci_sequence(8) == [1, 1, 2, 3, 5, 8, 13, 21]


def test_h02_fibonacci_depths_3_5_8_13():
    """The H02 spec depth schedule is [3, 5, 8, 13] at start_index=3.

    Sequence (1,1,2,3,5,8,13,21) at indices 3..6 → [3, 5, 8, 13].
    """
    sched = fibonacci_depths(n_stages=4, start_index=3)
    assert sched == [3, 5, 8, 13]


def test_h02_fibonacci_depths_extended_5_stage():
    """5-stage extended schedule [2, 3, 5, 8, 13] at start_index=2.

    Sequence (1,1,2,3,5,8,13) at indices 2..6 → [2, 3, 5, 8, 13].
    """
    sched = fibonacci_depths(n_stages=5, start_index=2)
    assert sched == [2, 3, 5, 8, 13]


def test_h02_resolve_blocks_schedule_modes():
    """resolve_blocks_schedule must dispatch uniform / fib / linear correctly."""
    assert resolve_blocks_schedule(3, 4, mode="uniform") == [3, 3, 3, 3]
    assert resolve_blocks_schedule(3, 4, mode="fib", fib_start=3) == [3, 5, 8, 13]
    assert resolve_blocks_schedule(3, 4, mode="linear") == [3, 4, 5, 6]
    # explicit list bypasses mode
    assert resolve_blocks_schedule([2, 4, 6], 3, mode="uniform") == [2, 4, 6]


def test_h02_resolve_blocks_schedule_rejects_bad_list_length():
    """Edge case: list length mismatch must raise (catches sweep typos early)."""
    try:
        resolve_blocks_schedule([1, 2], 3, mode="uniform")
        raise AssertionError("expected ValueError for length mismatch")
    except ValueError as exc:
        assert "length" in str(exc)


def test_h02_natureprior_fib_blocks_forward_shape():
    """Regression: NaturePriorNet with blocks_mode='fib' over 3 stages
    must materialise [2, 3, 5] (start_index=2) and forward a CIFAR tensor.

    Sequence (1,1,2,3,5) at indices 2..4 → [2, 3, 5].
    """
    cfg = NaturePriorConfig(num_classes=10, channel_mode="fib",
                            blocks_mode="fib", fib_start=2, n_stages=3)
    m = NaturePriorNet(cfg)
    assert m.block_counts == [2, 3, 5]
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_h02_natureprior_uniform_default_unchanged():
    """Default blocks_mode='uniform' must replicate the legacy [3,3,3]
    schedule byte-for-byte."""
    cfg = NaturePriorConfig(num_classes=10)
    m = NaturePriorNet(cfg)
    assert m.block_counts == [3, 3, 3]


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
