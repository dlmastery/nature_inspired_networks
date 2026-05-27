"""Phase-D integration tests: Trainer per-epoch callbacks for H43 / H47
/ H48 / H20 are wired correctly.

The tests cover behaviour, not just construction:

- H43 prune_schedule='fibonacci' -> Trainer wires fibonacci_prune at
  Fib-indexed epochs. We run a 5-epoch training loop on a tiny model +
  a 16-sample TensorDataset, then assert global_sparsity > 0 after
  fit().
- H48 momentum_schedule='golden' -> Trainer attaches a
  GoldenMomentumScheduler that decays beta1 each epoch.
- H20 fib_ensemble={enabled: True, K: 4} -> Trainer attaches a
  FibEnsemble that accumulates state-dict snapshots and loads the
  averaged weights at end of fit().
- TrainConfig backward-compat: the new Phase-D fields default to
  legacy (no prune, no momentum schedule, no ensemble).

Run with the canonical ``__main__`` pattern -- no pytest dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.ensemble import FibEnsemble  # noqa: E402
from nature_inspired_networks.pruning import global_sparsity  # noqa: E402
from nature_inspired_networks.schedulers import (  # noqa: E402
    GOLDEN_MOMENTUM_INIT, GoldenMomentumScheduler,
)
from nature_inspired_networks.train import TrainConfig, Trainer  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny model + tiny dataset (CPU-only, no CUDA assumption).
# ---------------------------------------------------------------------------
class _Tiny(nn.Module):
    """Tiny CIFAR-shaped model with 10 output classes (top-5 needs it)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.fc = nn.Linear(8 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 8)
        return self.fc(x.flatten(1))


def _toy_loaders(num_classes: int = 10):
    torch.manual_seed(0)
    x = torch.randn(16, 3, 8, 8)
    y = torch.randint(0, num_classes, (16,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    return loader, loader


def test_traincfg_phase_d_defaults_back_compat():
    """Newly added TrainConfig fields default to legacy behaviour."""
    cfg = TrainConfig()
    assert cfg.prune_schedule == ""
    assert cfg.prune_length == 5
    assert cfg.momentum_schedule == ""
    assert cfg.fib_ensemble is None


def test_golden_momentum_scheduler_attached_when_flag_set():
    """H48: momentum_schedule='golden' -> Trainer.momentum_sched is a
    GoldenMomentumScheduler initialised at 1/phi.
    """
    tr_loader, te_loader = _toy_loaders()
    cfg = TrainConfig(
        epochs=1, lr=1e-3, weight_decay=0.0, target_top1=1.0,
        use_bf16=False, momentum_schedule="golden",
    )
    tr = Trainer(_Tiny(), tr_loader, te_loader, num_classes=10,
                 cfg=cfg, device="cpu")
    assert isinstance(tr.momentum_sched, GoldenMomentumScheduler)
    # Initial beta1 was overwritten to 1/phi by GoldenMomentumScheduler.
    b1 = tr.opt.param_groups[0]["betas"][0]
    assert abs(b1 - GOLDEN_MOMENTUM_INIT) < 1e-6


def test_golden_momentum_scheduler_decays_beta1_per_epoch():
    """H48: After 1 epoch of training the optimiser's beta1 must be
    smaller than the initial 1/phi (decayed by exactly one factor of
    1/phi, subject to the floor).
    """
    tr_loader, te_loader = _toy_loaders()
    cfg = TrainConfig(
        epochs=1, lr=1e-3, weight_decay=0.0, target_top1=1.0,
        use_bf16=False, momentum_schedule="golden",
    )
    tr = Trainer(_Tiny(), tr_loader, te_loader, num_classes=10,
                 cfg=cfg, device="cpu")
    tr.fit()
    b1_after = tr.opt.param_groups[0]["betas"][0]
    assert b1_after < GOLDEN_MOMENTUM_INIT, b1_after


def test_fibonacci_pruning_induces_sparsity_after_fit():
    """H43: prune_schedule='fibonacci' -> at Fib-indexed epochs (the
    one-indexed schedule {1, 2, 3, 5, 8, ...}) fibonacci_prune runs.
    After a 3-epoch fit we expect non-zero global_sparsity.
    """
    tr_loader, te_loader = _toy_loaders()
    cfg = TrainConfig(
        epochs=3, lr=1e-3, weight_decay=0.0, target_top1=1.0,
        use_bf16=False, prune_schedule="fibonacci", prune_length=5,
    )
    tr = Trainer(_Tiny(), tr_loader, te_loader, num_classes=10,
                 cfg=cfg, device="cpu")
    s_before = global_sparsity(tr.model)
    tr.fit()
    s_after = global_sparsity(tr.model)
    assert s_after > s_before, (s_before, s_after)


def test_fib_ensemble_attached_and_records_snapshots():
    """H20: fib_ensemble={enabled: True, K: 4} -> Trainer.fib_ensemble
    is a FibEnsemble. After 2 epochs of training it has 2 snapshots.
    """
    tr_loader, te_loader = _toy_loaders()
    cfg = TrainConfig(
        epochs=2, lr=1e-3, weight_decay=0.0, target_top1=1.0,
        use_bf16=False,
        fib_ensemble={"enabled": True, "K": 4},
    )
    tr = Trainer(_Tiny(), tr_loader, te_loader, num_classes=10,
                 cfg=cfg, device="cpu")
    assert isinstance(tr.fib_ensemble, FibEnsemble)
    assert tr.fib_ensemble.K == 4
    tr.fit()
    # 2 epochs -> 2 snapshots pushed.
    assert len(tr.fib_ensemble) == 2


def test_fib_ensemble_loads_averaged_weights_into_model():
    """H20: at the end of fit() the model carries the FibEnsemble's
    averaged state_dict. We assert this by checking that the model's
    state_dict matches the averager's output tensor-by-tensor.
    """
    tr_loader, te_loader = _toy_loaders()
    cfg = TrainConfig(
        epochs=2, lr=1e-3, weight_decay=0.0, target_top1=1.0,
        use_bf16=False,
        fib_ensemble={"enabled": True, "K": 4},
    )
    tr = Trainer(_Tiny(), tr_loader, te_loader, num_classes=10,
                 cfg=cfg, device="cpu")
    tr.fit()
    averaged = tr.fib_ensemble.averaged_state_dict()
    for k, v in tr.model.state_dict().items():
        assert torch.allclose(v.detach().float().cpu(),
                              averaged[k].detach().float().cpu(), atol=1e-5), k


def test_no_callbacks_when_flags_off_does_not_change_legacy_fit():
    """Backward-compat: with all Phase-D flags off, fit() runs the
    legacy loop -- no momentum scheduler / no ensemble / no pruning
    side-effects.
    """
    tr_loader, te_loader = _toy_loaders()
    cfg = TrainConfig(
        epochs=1, lr=1e-3, weight_decay=0.0, target_top1=1.0, use_bf16=False,
    )
    tr = Trainer(_Tiny(), tr_loader, te_loader, num_classes=10,
                 cfg=cfg, device="cpu")
    assert tr.momentum_sched is None
    assert tr.fib_ensemble is None
    assert tr._prune_enabled is False
    s_before = global_sparsity(tr.model)
    tr.fit()
    s_after = global_sparsity(tr.model)
    # No prune -> sparsity unchanged.
    assert s_before == s_after


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
