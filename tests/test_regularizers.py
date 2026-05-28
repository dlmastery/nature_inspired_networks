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
    Fibonacci ratios in order (driven by set_epoch, G5 audit fix)."""
    mod = PhiDropout(cycle="fib", length=5)
    expected = [1 / 19, 2 / 19, 3 / 19, 5 / 19, 8 / 19]
    visited = []
    mod.train()
    for e in range(5):
        mod.set_epoch(e)
        visited.append(mod.current_p)
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
    for e in range(5):
        mod.set_epoch(e)
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


def test_h47_cycle_wraps_and_stays_in_range():
    """Drive across many epochs; rate must stay in [0, 1) and wrap so
    that long-running training never escapes the schedule."""
    mod = PhiDropout(cycle="fib", length=5)
    mod.train()
    rates = []
    for e in range(20):  # 4 full cycles
        mod.set_epoch(e)
        rates.append(mod.current_p)
        _ = mod(torch.randn(4, 8))
    for r in rates:
        assert 0.0 <= r < 1.0, r
    # Epoch 0 and epoch 5 (one cycle later) read the same rate.
    assert math.isclose(rates[0], rates[5], abs_tol=1e-9)
    assert math.isclose(rates[1], rates[6], abs_tol=1e-9)


def test_h47_curriculum_changes_only_on_set_epoch():
    """G5 audit fix: PhiDropout's rate must be constant within an epoch
    and only change when ``set_epoch`` is invoked.

    The original step-keyed implementation advanced once per forward
    pass, cycling the 5-entry schedule ~39× per epoch at batch=256.
    Curriculum must be epoch-driven now.
    """
    mod = PhiDropout(cycle="fib", length=5)
    mod.train()
    # 100 forward passes with no set_epoch call -> rate must stay at
    # schedule[0].
    expected0 = mod.current_p
    rates_e0 = []
    for _ in range(100):
        rates_e0.append(mod.current_p)
        _ = mod(torch.randn(4, 8))
    for r in rates_e0:
        assert math.isclose(r, expected0, abs_tol=1e-12), r
    # Now advance to epoch 1; rate must change to schedule[1].
    mod.set_epoch(1)
    expected1 = float(mod.schedule[1].item())
    assert math.isclose(mod.current_p, expected1, abs_tol=1e-12)
    assert not math.isclose(expected1, expected0, abs_tol=1e-6)
    # And rate is again constant across many forwards within epoch 1.
    for _ in range(50):
        assert math.isclose(mod.current_p, expected1, abs_tol=1e-12)
        _ = mod(torch.randn(4, 8))
    # Advance through several epochs; track the per-epoch rate.
    seen = []
    for e in range(5):
        mod.set_epoch(e)
        seen.append(mod.current_p)
    # Should be 5 distinct values (the full fib schedule).
    assert len(set(round(v, 9) for v in seen)) == 5


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
