"""Unit tests for H54 — Persistent Homology Activation Regularization.

Asserts (per the H54 design doc Q&A section):
  - register_ph_hooks attaches forward hooks on the named stages of a
    small CNN without raising;
  - after a forward pass, reg.loss() returns a finite scalar tensor;
  - the loss is differentiable: .backward() updates parameters of the
    small CNN (non-zero grad on at least one stage's weights);
  - clear_hooks() removes hooks and zeros the activation buffer;
  - register_ph_hooks gracefully skips missing stage attributes;
  - the convenience forward() and loss() return the same value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.ph_reg import (  # noqa: E402
    PHActivationRegularizer,
    register_ph_hooks,
)


class _TinyCNN(nn.Module):
    """3-stage CNN matching the (stage1, stage2, stage3) hook convention."""

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Conv2d(3, 8, 3, padding=1)
        self.stage2 = nn.Conv2d(8, 16, 3, padding=1)
        self.stage3 = nn.Conv2d(16, 16, 3, padding=1)
        self.head = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        h = h.mean(dim=(2, 3))
        return self.head(h)


def test_register_ph_hooks_without_crash():
    """Test (1): hooks attach on the 3 named stages of a small CNN."""
    torch.manual_seed(0)
    model = _TinyCNN()
    reg = register_ph_hooks(model, stage_attr_names=("stage1", "stage2", "stage3"))
    assert len(reg) == 3
    assert len(reg._hooks) == 3
    reg.clear_hooks()


def test_loss_is_scalar_after_forward():
    """Test (2): after a forward pass, reg.loss() is a finite scalar."""
    torch.manual_seed(1)
    model = _TinyCNN()
    reg = register_ph_hooks(model)
    x = torch.randn(4, 3, 8, 8)
    _ = model(x)
    loss = reg.loss()
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert loss.item() >= 0.0
    reg.clear_hooks()


def test_loss_backward_updates_cnn_params():
    """Test (3): the PH loss is fully differentiable end-to-end — backward
    must update at least one stage's parameters."""
    torch.manual_seed(2)
    model = _TinyCNN()
    reg = register_ph_hooks(model)
    x = torch.randn(4, 3, 8, 8)
    _ = model(x)
    loss = reg.loss()
    loss.backward()
    # At least one stage must receive a non-zero gradient.
    found = False
    for stage in (model.stage1, model.stage2, model.stage3):
        if stage.weight.grad is not None and stage.weight.grad.abs().sum().item() > 0.0:
            found = True
            break
    assert found, "no stage received a non-zero PH-loss gradient"
    reg.clear_hooks()


def test_clear_hooks_removes_all():
    """Test (4): clear_hooks empties the hook list and the activation buffer."""
    torch.manual_seed(3)
    model = _TinyCNN()
    reg = register_ph_hooks(model)
    x = torch.randn(2, 3, 8, 8)
    _ = model(x)
    assert reg._captured  # populated
    reg.clear_hooks()
    assert reg._hooks == []
    assert reg._captured == {}
    assert len(reg) == 0
    # After clearing hooks, a fresh forward must NOT repopulate the buffer.
    _ = model(x)
    assert reg._captured == {}


def test_missing_stage_silently_skipped():
    """Test (5): register_ph_hooks tolerates models that only have a
    subset of the requested stage names."""

    class TwoStageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stage1 = nn.Conv2d(3, 8, 3, padding=1)
            self.stage2 = nn.Conv2d(8, 16, 3, padding=1)
            self.head = nn.Linear(16, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.stage2(self.stage1(x))
            return self.head(h.mean(dim=(2, 3)))

    model = TwoStageModel()
    reg = register_ph_hooks(model, stage_attr_names=("stage1", "stage2", "stage3"))
    assert len(reg) == 2  # only 2 of the 3 requested stages exist
    _ = model(torch.randn(2, 3, 8, 8))
    loss = reg.loss()
    assert torch.isfinite(loss).item()
    reg.clear_hooks()


def test_forward_alias_matches_loss():
    """Test (6): the nn.Module forward() returns the same value as loss()."""
    torch.manual_seed(4)
    model = _TinyCNN()
    reg = register_ph_hooks(model)
    _ = model(torch.randn(2, 3, 8, 8))
    a = reg.loss().item()
    b = reg().item()
    assert abs(a - b) < 1e-6, (a, b)
    reg.clear_hooks()


def test_no_capture_yields_zero_loss():
    """Without a forward pass, loss() must return a finite zero scalar
    (no-op contract — the regularizer can be safely added to CE before
    the first batch)."""
    reg = PHActivationRegularizer(stage_targets=[(10.0, 0.0, 0.0)])
    loss = reg.loss()
    assert loss.ndim == 0
    assert loss.item() == 0.0


def test_constructor_validates_inputs():
    """Empty stage_targets / mismatched lambdas are rejected."""
    try:
        PHActivationRegularizer(stage_targets=[])
        raise AssertionError("expected ValueError for empty stage_targets")
    except ValueError as exc:
        if "expected ValueError" in str(exc):
            raise
    try:
        PHActivationRegularizer(
            stage_targets=[(10.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
            lambdas=[0.1],
        )
        raise AssertionError("expected ValueError for mismatched lambdas")
    except ValueError as exc:
        if "expected ValueError" in str(exc):
            raise


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
