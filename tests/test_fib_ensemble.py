"""Unit tests for H20 — Fibonacci Ensemble.

Covers:
* Canonical Fib weight generation.
* Branch: FibEnsemble buffer FIFO + weighted averaging.
* Branch: FibEMA O(1)-memory running average.
* Regression: averaging K copies of the same state-dict equals it.
* Edge case: integer buffers (BN num_batches_tracked) carried through.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.ensemble import (  # noqa: E402
    FibEMA,
    FibEnsemble,
    fib_weights,
    fibonacci,
)


def test_fibonacci_sequence_canonical():
    assert fibonacci(0) == []
    assert fibonacci(1) == [1]
    assert fibonacci(2) == [1, 1]
    assert fibonacci(8) == [1, 1, 2, 3, 5, 8, 13, 21]
    # Sum check (matches H20 design doc total = 54)
    assert sum(fibonacci(8)) == 54


def test_fibonacci_rejects_negative():
    try:
        fibonacci(-1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_fib_weights_normalise_sums_to_one():
    w = fib_weights(8, normalise=True)
    assert abs(sum(w) - 1.0) < 1e-12
    # newest snapshot weight = 21 / 54
    assert abs(w[-1] - 21.0 / 54.0) < 1e-12


def test_fib_weights_unnormalised_match_fibonacci():
    raw = fib_weights(6, normalise=False)
    assert raw == [1.0, 1.0, 2.0, 3.0, 5.0, 8.0]


def test_fib_ensemble_buffer_evicts_oldest():
    """Branch: FIFO behaviour at K+1 updates."""
    ens = FibEnsemble(K=3)
    for i in range(5):
        ens.update(OrderedDict(w=torch.tensor([float(i)])))
    assert len(ens) == 3
    # Newest = 4.0, oldest in buffer = 2.0
    assert ens.checkpoints[-1]["w"].item() == 4.0
    assert ens.checkpoints[0]["w"].item() == 2.0


def test_fib_ensemble_average_of_identical_equals_input():
    """Regression: averaging K identical state-dicts must recover them
    exactly (up to dtype)."""
    base = OrderedDict(
        a=torch.randn(4, 4),
        b=torch.randn(8),
    )
    ens = FibEnsemble(K=8)
    for _ in range(8):
        ens.update(base)
    avg = ens.averaged_state_dict()
    for k, v in base.items():
        assert torch.allclose(avg[k], v, atol=1e-6), k


def test_fib_ensemble_weighted_mean_matches_manual():
    """Canonical: averaged weights equal the Fibonacci-weighted mean."""
    ens = FibEnsemble(K=3)  # weights = [1, 1, 2]; total = 4
    ens.update(OrderedDict(w=torch.tensor([1.0, 2.0])))
    ens.update(OrderedDict(w=torch.tensor([3.0, 4.0])))
    ens.update(OrderedDict(w=torch.tensor([5.0, 6.0])))
    avg = ens.averaged_state_dict()
    # expected = (1*[1,2] + 1*[3,4] + 2*[5,6]) / 4 = ([14, 18]) / 4
    expected = torch.tensor([14.0, 18.0]) / 4.0
    assert torch.allclose(avg["w"], expected, atol=1e-6), avg["w"]


def test_fib_ensemble_partial_buffer_uses_active_weights():
    """Branch: with fewer than K snapshots, the active Fib weights are
    the *trailing* slice so the newest snapshot still dominates."""
    ens = FibEnsemble(K=8)
    # Insert two snapshots — active weights should be the last 2 of
    # [1, 1, 2, 3, 5, 8, 13, 21] = [13, 21]; total = 34.
    ens.update(OrderedDict(w=torch.tensor([10.0])))
    ens.update(OrderedDict(w=torch.tensor([20.0])))
    avg = ens.averaged_state_dict()
    expected = (13 * 10.0 + 21 * 20.0) / 34.0
    assert abs(avg["w"].item() - expected) < 1e-6


def test_fib_ensemble_rejects_empty():
    ens = FibEnsemble(K=4)
    try:
        ens.averaged_state_dict()
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_fib_ensemble_rejects_bad_K():
    try:
        FibEnsemble(K=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_fib_ensemble_integer_buffer_takes_most_recent():
    """Edge case: BatchNorm's `num_batches_tracked` is int64; averaging
    it would round to garbage. The implementation must keep the most
    recent value verbatim."""
    ens = FibEnsemble(K=3)
    for i in range(3):
        ens.update(OrderedDict(
            w=torch.randn(2),
            nbt=torch.tensor(i, dtype=torch.long),
        ))
    avg = ens.averaged_state_dict()
    assert avg["nbt"].dtype == torch.long
    assert avg["nbt"].item() == 2  # most recent


def test_fib_ensemble_load_into_real_model():
    """End-to-end regression: ensemble a real nn.Module's state-dict
    twice over and load it back without shape mismatch."""
    torch.manual_seed(0)
    m = nn.Linear(8, 4)
    sd0 = OrderedDict((k, v.clone()) for k, v in m.state_dict().items())
    # Drift the weights
    with torch.no_grad():
        m.weight.add_(0.1 * torch.randn_like(m.weight))
    sd1 = OrderedDict((k, v.clone()) for k, v in m.state_dict().items())

    ens = FibEnsemble(K=2)  # weights [1, 1] = uniform
    ens.update(sd0)
    ens.update(sd1)
    ens.load_into(m)
    avg_w = (sd0["weight"] + sd1["weight"]) / 2.0
    assert torch.allclose(m.weight, avg_w, atol=1e-6)


def test_fib_ema_running_average_matches_recurrence():
    """Canonical FibEMA: w_new = F_K / sum(F_1..F_K). For K=8 -> 21/54."""
    ema = FibEMA(K=8)
    expected_w_new = 21.0 / 54.0
    assert abs(ema.w_new - expected_w_new) < 1e-12

    # After one update from None -> equals the first state.
    sd0 = OrderedDict(w=torch.tensor([1.0]))
    ema.update(sd0)
    avg0 = ema.averaged_state_dict()
    assert torch.allclose(avg0["w"], sd0["w"], atol=1e-6)

    # After a second update with a different state.
    sd1 = OrderedDict(w=torch.tensor([10.0]))
    ema.update(sd1)
    avg1 = ema.averaged_state_dict()
    expected = (1 - expected_w_new) * 1.0 + expected_w_new * 10.0
    assert abs(avg1["w"].item() - expected) < 1e-6


def test_fib_ema_constant_state_is_idempotent():
    """Regression: feeding the same state-dict repeatedly should
    converge to that state (the steady-state property of any
    convex-combination EMA)."""
    ema = FibEMA(K=8)
    sd = OrderedDict(w=torch.tensor([3.0, 5.0]))
    for _ in range(100):
        ema.update(sd)
    avg = ema.averaged_state_dict()
    assert torch.allclose(avg["w"], sd["w"], atol=1e-5)


def test_fib_ema_rejects_pre_update_query():
    ema = FibEMA(K=4)
    try:
        ema.averaged_state_dict()
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
