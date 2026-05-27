"""Unit tests for H10 — phi-decay LR scheduler.

Convention (matches tests/test_priors.py): run as
``python tests/test_h10_phi_lr.py``; the file ends with
``"All N tests passed."`` on success.

These tests cover the H10 PhiDecayLR primitive specifically. The
``tests/test_schedulers.py`` file is owned by the H48 (golden momentum)
agent — both test files import PhiDecayLR from the same module without
collision, and a sanity test there confirms PhiDecayLR continues to work.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch.optim import SGD

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.priors import PHI  # noqa: E402
from nature_inspired_networks.schedulers import PhiDecayLR  # noqa: E402
from nature_inspired_networks.train import TrainConfig, _build_scheduler  # noqa: E402


def _make_opt(lr: float = 0.1) -> torch.optim.Optimizer:
    p = torch.nn.Parameter(torch.zeros(4))
    return SGD([p], lr=lr)


def test_h10_phi_decay_initial_lr_is_base_lr():
    """At step 0 the multiplier must be exactly 1.0 (no warmup)."""
    opt = _make_opt(lr=0.1)
    PhiDecayLR(opt, T_max=12)
    assert abs(opt.param_groups[0]["lr"] - 0.1) < 1e-12


def test_h10_phi_decay_step_k_equals_phi_neg_k_over_Tmax():
    """After k steps the LR must equal base_lr * phi^{-k / T_max}.

    This is the H10 hypothesis spec test — the produced LR sequence
    matches the closed form within 1e-9 absolute tolerance.
    """
    opt = _make_opt(lr=0.1)
    T_max = 12
    sched = PhiDecayLR(opt, T_max=T_max)
    seen = [opt.param_groups[0]["lr"]]
    for _ in range(5):
        sched.step()
        seen.append(opt.param_groups[0]["lr"])
    for k, lr_k in enumerate(seen):
        expected = 0.1 * PHI ** (-k / float(T_max))
        assert abs(lr_k - expected) < 1e-9, (k, lr_k, expected)


def test_h10_phi_decay_T_max_1_matches_raw_phi_neg_k():
    """T_max=1 reproduces the raw H10 design-doc rule lr_k = lr_0 * phi^{-k}."""
    opt = _make_opt(lr=0.1)
    sched = PhiDecayLR(opt, T_max=1)
    seen = [opt.param_groups[0]["lr"]]
    for _ in range(4):
        sched.step()
        seen.append(opt.param_groups[0]["lr"])
    expected = [0.1 * PHI ** (-k) for k in range(5)]
    for got, exp in zip(seen, expected):
        assert abs(got - exp) < 1e-9, (got, exp)


def test_h10_phi_decay_floor_clamps_long_horizon():
    """After many steps the LR must clamp to ``lr_floor`` rather than 0."""
    opt = _make_opt(lr=0.1)
    floor = 1e-3
    sched = PhiDecayLR(opt, T_max=1, lr_floor=floor)
    for _ in range(100):
        sched.step()
    assert opt.param_groups[0]["lr"] == floor


def test_h10_phi_decay_invalid_T_max_rejected():
    """Edge case: T_max=0 must raise immediately to catch sweep-typo configs."""
    opt = _make_opt()
    try:
        PhiDecayLR(opt, T_max=0)
        raise AssertionError("expected ValueError for T_max=0")
    except ValueError as exc:
        assert "T_max" in str(exc)


def test_h10_phi_decay_monotonically_non_increasing():
    """Regression: the schedule must never go up between consecutive steps."""
    opt = _make_opt(lr=0.1)
    sched = PhiDecayLR(opt, T_max=12)
    prev = opt.param_groups[0]["lr"]
    for _ in range(30):
        sched.step()
        cur = opt.param_groups[0]["lr"]
        assert cur <= prev + 1e-12, (cur, prev)
        prev = cur


def test_h10_trainer_dispatch_phi_decay():
    """Trainer scheduler dispatch must produce a PhiDecayLR when
    scheduler='phi_decay'; default 'cosine' must NOT (Rule 1 regression)."""
    opt = _make_opt()
    cfg_phi = TrainConfig(epochs=12, scheduler="phi_decay")
    sched_phi = _build_scheduler(opt, cfg_phi)
    assert isinstance(sched_phi, PhiDecayLR)
    cfg_cos = TrainConfig(epochs=12, scheduler="cosine")
    sched_cos = _build_scheduler(opt, cfg_cos)
    assert not isinstance(sched_cos, PhiDecayLR)


def test_h10_trainer_dispatch_rejects_unknown_scheduler():
    """Catch typos in YAML configs at construction time, not at runtime."""
    opt = _make_opt()
    cfg = TrainConfig(epochs=12, scheduler="cosine_dance")
    try:
        _build_scheduler(opt, cfg)
        raise AssertionError("expected ValueError for unknown scheduler")
    except ValueError as exc:
        assert "cosine_dance" in str(exc)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
