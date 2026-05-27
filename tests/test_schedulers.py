"""H48 — Unit tests for GoldenMomentumScheduler.

Per CLAUDE.md Rule 12: ≥ 4 assertions, regression test named for H48,
edge case (SGD with momentum field, floor-clamp behaviour), and a
monotonic-decay invariant.

Also smoke-tests the existing H10 ``PhiDecayLR`` to confirm the new
imports do not regress it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.priors import PHI  # noqa: E402
from nature_inspired_networks.schedulers import (  # noqa: E402
    GOLDEN_MOMENTUM_FLOOR,
    GOLDEN_MOMENTUM_INIT,
    GoldenMomentumScheduler,
    PhiDecayLR,
    golden_momentum_curve,
)


def _tiny_params() -> list[torch.nn.Parameter]:
    """Two leaf tensors for optimizer construction."""
    torch.manual_seed(0)
    return [nn.Parameter(torch.randn(4, 4)) for _ in range(2)]


def test_h48_initial_beta_is_phi_inverse():
    """Regression test for H48: β1 is seeded at 1/φ on construction."""
    opt = torch.optim.AdamW(_tiny_params(), lr=1e-3)
    sched = GoldenMomentumScheduler(opt)
    b1, b2 = opt.param_groups[0]["betas"]
    assert math.isclose(b1, 1.0 / PHI, abs_tol=1e-12), b1
    # β2 must be untouched by the scheduler.
    assert math.isclose(b2, 0.999, abs_tol=1e-12), b2
    assert math.isclose(GOLDEN_MOMENTUM_INIT, 1.0 / PHI, abs_tol=1e-12)
    assert math.isclose(GOLDEN_MOMENTUM_FLOOR, 1.0 / (PHI ** 2), abs_tol=1e-12)
    assert math.isclose(sched.get_last(), 1.0 / PHI, abs_tol=1e-12)


def test_h48_beta_monotonically_decays_toward_floor():
    """Each step() reduces β1 by factor 1/φ until the floor."""
    opt = torch.optim.AdamW(_tiny_params(), lr=1e-3)
    sched = GoldenMomentumScheduler(opt)
    prev = sched.get_last()
    seen = [prev]
    for _ in range(20):
        new = sched.step()
        # monotonic non-increasing
        assert new <= prev + 1e-12, (prev, new)
        # never below the floor
        assert new >= GOLDEN_MOMENTUM_FLOOR - 1e-12, new
        prev = new
        seen.append(new)
    # The last value must equal the floor (we will have decayed past it
    # in fewer than 20 steps because 1/φ → 1/φ² takes exactly one step).
    assert math.isclose(seen[-1], GOLDEN_MOMENTUM_FLOOR, abs_tol=1e-9)


def test_h48_one_step_hits_phi_squared():
    """β1 starts at 1/φ; after one step it equals 1/φ² (the floor)."""
    opt = torch.optim.AdamW(_tiny_params(), lr=1e-3)
    sched = GoldenMomentumScheduler(opt)
    new = sched.step()
    assert math.isclose(new, 1.0 / (PHI ** 2), abs_tol=1e-9), new
    # The optimizer's actual betas[0] must reflect the new value.
    assert math.isclose(opt.param_groups[0]["betas"][0],
                        1.0 / (PHI ** 2), abs_tol=1e-9)


def test_h48_works_with_sgd_momentum():
    """Edge case: SGD param-groups carry ``momentum``, not ``betas``."""
    opt = torch.optim.SGD(_tiny_params(), lr=1e-3, momentum=0.9)
    sched = GoldenMomentumScheduler(opt, init=GOLDEN_MOMENTUM_INIT)
    assert "momentum" in opt.param_groups[0]
    assert math.isclose(opt.param_groups[0]["momentum"], 1.0 / PHI,
                        abs_tol=1e-12)
    new = sched.step()
    assert math.isclose(new, 1.0 / (PHI ** 2), abs_tol=1e-9)
    assert math.isclose(opt.param_groups[0]["momentum"], 1.0 / (PHI ** 2),
                        abs_tol=1e-9)


def test_h48_floor_clamp_is_idempotent():
    """Many step() calls past the floor must not push β below it."""
    opt = torch.optim.AdamW(_tiny_params(), lr=1e-3)
    sched = GoldenMomentumScheduler(opt)
    for _ in range(50):
        sched.step()
    assert math.isclose(sched.get_last(), GOLDEN_MOMENTUM_FLOOR,
                        abs_tol=1e-12)
    # And one more step is still a no-op (no drop below floor).
    final = sched.step()
    assert math.isclose(final, GOLDEN_MOMENTUM_FLOOR, abs_tol=1e-12)


def test_h48_closed_form_curve_bounded():
    """The reference golden_momentum_curve is bounded by [floor, init]
    and monotonically decreasing."""
    curve = golden_momentum_curve(20, tau=5.0)
    assert len(curve) == 20
    assert math.isclose(curve[0], GOLDEN_MOMENTUM_INIT, abs_tol=1e-9)
    for a, b in zip(curve[:-1], curve[1:]):
        assert b <= a + 1e-12, (a, b)
    for v in curve:
        assert GOLDEN_MOMENTUM_FLOOR - 1e-9 <= v <= GOLDEN_MOMENTUM_INIT + 1e-9


def test_h10_phi_decay_lr_still_works():
    """Sanity: the H10 LR scheduler still works (no import regression)."""
    opt = torch.optim.AdamW(_tiny_params(), lr=1.0)
    sch = PhiDecayLR(opt, T_max=1, lr_floor=1e-6)
    lrs = []
    for _ in range(3):
        opt.step()
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    # After T_max=1 step the LR is divided by phi.
    assert math.isclose(lrs[0], 1.0 / PHI, rel_tol=1e-6)
    assert lrs[2] < lrs[1] < lrs[0]


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
