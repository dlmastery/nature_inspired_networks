"""Unit tests for H64 — Dynamic Growth + Pruning Cycle."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_growth_pruning import (  # noqa: E402
    DEFAULT_GROW_EPOCHS,
    DEFAULT_PRUNE_EPOCHS,
    GrowthPruningSchedule,
)
from nature_inspired_networks.blocks import NaturePriorBlock, NaturePriorFlags  # noqa: E402


class _ToyNet(nn.Module):
    """Minimal NaturePriorNet-shaped fixture with a ``stages`` ModuleList."""

    def __init__(self, width: int = 8, n_blocks: int = 2):
        super().__init__()
        flags = NaturePriorFlags()
        stage = nn.Sequential(
            *[NaturePriorBlock(width, width, stride=1, flags=flags) for _ in range(n_blocks)]
        )
        self.stages = nn.ModuleList([stage])
        self.head = nn.Linear(width, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for s in self.stages:
            x = s(x)
        return self.head(x.mean(dim=(-2, -1)))


def test_disjoint_schedules_required():
    """Grow and prune epochs must be disjoint."""
    try:
        GrowthPruningSchedule(
            model_factory=lambda: _ToyNet(),
            grow_epochs=(3, 5),
            prune_epochs=(5, 6),  # 5 overlaps
        )
        raise AssertionError("expected ValueError for overlapping schedules")
    except ValueError:
        pass


def test_default_schedules_are_canonical_fibonacci_pairs():
    """Defaults: grow=(3,5,8,13), prune=(4,6,9,14) -- adjacent disjoint sets."""
    assert DEFAULT_GROW_EPOCHS == (3, 5, 8, 13)
    assert DEFAULT_PRUNE_EPOCHS == (4, 6, 9, 14)
    assert set(DEFAULT_GROW_EPOCHS).isdisjoint(set(DEFAULT_PRUNE_EPOCHS))


def test_grow_event_increases_param_count():
    """Calling step at a grow epoch must add fresh blocks (more params)."""
    sched = GrowthPruningSchedule(model_factory=lambda: _ToyNet(), n_extra_blocks=1)
    model = sched.model
    n_before = sum(p.numel() for p in model.parameters())
    model, event = sched.step(epoch=3, model=model)
    assert event == "grow"
    n_after = sum(p.numel() for p in model.parameters())
    assert n_after > n_before


def test_prune_event_increases_sparsity():
    """Calling step at a prune epoch must make some weights zero."""
    from nature_inspired_networks.pruning import global_sparsity
    sched = GrowthPruningSchedule(model_factory=lambda: _ToyNet())
    model = sched.model
    s_before = global_sparsity(model)
    model, event = sched.step(epoch=4, model=model)
    assert event == "prune"
    s_after = global_sparsity(model)
    assert s_after > s_before


def test_unscheduled_epoch_returns_none_event():
    sched = GrowthPruningSchedule(model_factory=lambda: _ToyNet())
    model = sched.model
    n_before = sum(p.numel() for p in model.parameters())
    model, event = sched.step(epoch=0, model=model)
    assert event == "none"
    n_after = sum(p.numel() for p in model.parameters())
    assert n_after == n_before


def test_step_idempotent_within_grow_epoch():
    """Re-firing the same grow epoch must not double-grow (DynamicGrowthCallback
    guarantees idempotency via its ``fired`` set)."""
    sched = GrowthPruningSchedule(model_factory=lambda: _ToyNet(), n_extra_blocks=1)
    model = sched.model
    model, _ = sched.step(epoch=3, model=model)
    n_after_first = sum(p.numel() for p in model.parameters())
    model, _ = sched.step(epoch=3, model=model)
    n_after_second = sum(p.numel() for p in model.parameters())
    assert n_after_first == n_after_second


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
