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


def test_h64_growth_pruning_wired_runs_a_step():
    """Trainer must invoke ``GrowthPruningSchedule.step(epoch, model)``
    at the end of each epoch when ``cfg.growth_pruning_schedule`` is True.

    G7-audit fix: the hybrid was previously orphan code. We instantiate
    a counting wrapper that records every call to ``step`` so the test
    can assert the Trainer fired the schedule at least once.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from nature_inspired_networks.train import TrainConfig, Trainer

    class _CountingSchedule:
        def __init__(self, inner: GrowthPruningSchedule):
            self.inner = inner
            self.n_calls = 0
            self.events = []

        @property
        def model(self):
            return self.inner.model

        def step(self, epoch, model):
            self.n_calls += 1
            new_model, event = self.inner.step(epoch=epoch, model=model)
            self.events.append((epoch, event))
            return new_model, event

    torch.manual_seed(0)
    # Tiny tabular surrogate so we don't need CIFAR. The Trainer is
    # model-agnostic, so any nn.Module that maps (B, 8) -> (B, n_classes)
    # works. Use n_classes >= 5 so topk(5) inside eval() doesn't trip.
    n_classes = 6
    n_train, n_test = 32, 16
    x_tr = torch.randn(n_train, 8)
    y_tr = torch.randint(0, n_classes, (n_train,))
    x_te = torch.randn(n_test, 8)
    y_te = torch.randint(0, n_classes, (n_test,))
    tr_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=8)
    te_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=8)

    class _TinyMLP(nn.Module):
        def __init__(self, n_out=n_classes):
            super().__init__()
            self.fc = nn.Linear(8, n_out)

        def forward(self, x):
            return self.fc(x)

    model = _TinyMLP()
    inner = GrowthPruningSchedule(model_factory=lambda m=model: m, n_extra_blocks=1)
    counter = _CountingSchedule(inner)

    cfg = TrainConfig(
        epochs=5,
        use_bf16=False,
        target_top1=1.0,
        growth_pruning_schedule=True,
        growth_pruning_schedule_obj=counter,
    )
    trainer = Trainer(
        model, tr_loader, te_loader, num_classes=n_classes, cfg=cfg, device="cpu",
    )
    _ = trainer.fit()
    # The Trainer should have invoked the schedule once per epoch.
    assert counter.n_calls >= 1, (
        f"GrowthPruningSchedule.step never called; n_calls={counter.n_calls}"
    )
    # Specifically, 5 epochs → 5 step calls.
    assert counter.n_calls == 5, (
        f"Expected 5 step calls (one per epoch); got {counter.n_calls}"
    )
    # Epoch 4 is in the canonical prune schedule -> should have logged
    # a 'prune' event.
    events = dict(counter.events)
    assert events.get(4) == "prune", f"epoch 4 event was {events.get(4)!r}, expected 'prune'"


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
