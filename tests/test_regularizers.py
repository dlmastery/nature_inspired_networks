"""H47 — Unit tests for PhiDropout.

Per CLAUDE.md Rule 12: ≥ 4 assertions, regression test named for H47,
edge case (eval mode is identity), and probability-range invariant.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.priors import PHI  # noqa: E402
from nature_inspired_networks.regularizers import (  # noqa: E402
    PHI_DROPOUT_SCHEDULE,
    PhiDropout,
)


def test_h47_fib_schedule_normalised():
    """Regression test for H47: 'fib' cycle visits the normalised
    Fibonacci ratios in order."""
    mod = PhiDropout(cycle="fib", length=5)
    expected = [1 / 19, 2 / 19, 3 / 19, 5 / 19, 8 / 19]
    visited = []
    mod.train()
    for _ in range(5):
        visited.append(mod.current_p)
        # advance the counter by running a forward pass.
        _ = mod(torch.zeros(2, 4))
    for a, b in zip(visited, expected):
        assert math.isclose(a, b, abs_tol=1e-6), (a, b)
    # All visited rates are in [0, 1).
    for v in visited:
        assert 0.0 <= v < 1.0


def test_h47_phi_schedule_matches_design_doc():
    """The 'phi' cycle must equal 1/φ^(1+k) for k=0..length-1."""
    mod = PhiDropout(cycle="phi", length=5)
    expected = [1.0 / (PHI ** (1 + k)) for k in range(5)]
    for a, b in zip(PHI_DROPOUT_SCHEDULE, expected):
        assert math.isclose(a, b, abs_tol=1e-12)
    mod.train()
    visited = []
    for _ in range(5):
        visited.append(mod.current_p)
        _ = mod(torch.zeros(2, 4))
    for a, b in zip(visited, expected):
        assert math.isclose(a, b, abs_tol=1e-6), (a, b)


def test_h47_eval_mode_is_identity():
    """Edge case: in eval mode dropout must be a no-op."""
    mod = PhiDropout(cycle="fib", length=5)
    mod.eval()
    x = torch.randn(4, 8)
    y = mod(x)
    assert torch.equal(x, y)
    # And the step counter does NOT advance in eval mode.
    assert int(mod.step_counter.item()) == 0


def test_h47_cycle_wraps_and_stays_in_range():
    """Run many forwards; rate must stay in [0, 1) and the cycle must
    wrap (so a long-running training never escapes the schedule)."""
    mod = PhiDropout(cycle="fib", length=5)
    mod.train()
    rates = []
    for _ in range(20):  # 4 full cycles
        rates.append(mod.current_p)
        _ = mod(torch.randn(4, 8))
    for r in rates:
        assert 0.0 <= r < 1.0, r
    # Step 0 and step 5 (one cycle later) read the same rate.
    assert math.isclose(rates[0], rates[5], abs_tol=1e-9)
    assert math.isclose(rates[1], rates[6], abs_tol=1e-9)


def test_h47_rejects_invalid_args():
    """Constructor must reject bad ``cycle``, ``p_init``, ``length``."""
    raised = 0
    for kwargs in (
        dict(cycle="nope"),
        dict(p_init=1.5),
        dict(p_init=-0.1),
        dict(length=0),
        dict(length=-3),
    ):
        try:
            PhiDropout(**kwargs)
        except ValueError:
            raised += 1
        else:  # pragma: no cover
            pass
    assert raised == 5, f"expected 5 ValueErrors, got {raised}"


def test_h47_dropout_actually_drops_units():
    """A second sanity-check that the module isn't a pass-through in
    training mode at the higher 'phi' rate (≈ 0.618)."""
    torch.manual_seed(42)
    mod = PhiDropout(cycle="phi", length=5)
    mod.train()
    x = torch.ones(1000)
    y = mod(x)
    # At p ≈ 0.618, roughly 38% of entries survive (scaled by 1/(1-p)).
    # We just assert the output is not equal to the input and contains
    # zeros — both invariants of "real" dropout.
    assert not torch.equal(x, y)
    assert (y == 0).any().item()


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
