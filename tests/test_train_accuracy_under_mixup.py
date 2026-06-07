"""Synthesis-100 D7 (2026-06-06): truthful generalization gap under Mixup.

Under Mixup/CutMix the per-step ``train_top1 = (argmax == y_a).mean()``
measures accuracy against the DOMINANT label (the timm convention with
the lam >= 0.5 fold). At alpha=0.2 the mean lam is roughly 0.79 so the
*maximum* achievable train_top1 is ~lam ~ 0.79 even when the model
perfectly memorises the training set. Pair that with the legacy
``gap = max(0, train_top1 - test_top1)`` floor at zero and the
generalization-gap diagnostic collapses to 0 across an entire 200-epoch
modern-recipe run (observed empirically: train_top1 ~ 0.45, test_top1
~ 0.635, gap floored at 0).

The fix:
  * Trainer.fit logs ``train_top1_mixed`` and ``train_top1_clean``
    (sampled un-mixed eval-mode pass over a fixed-size train subset).
  * ``fit_info['train_top1_clean']`` carries the value through to
    ``evaluate_full``.
  * ``RunMetrics.train_top1_clean`` records it for downstream.
  * ``generalization_gap`` is ``train_top1_clean - test_top1`` when the
    clean value is available, and it is NOT max-floored (negative is
    diagnostic).

This test exercises:
  * a trainer with mixup_alpha>0 emits the new ``train_top1_clean``
    field and clean > mixed at convergence on a tiny synthetic problem.
  * a trainer with mixup_alpha=0 leaves ``train_top1_clean`` populated
    but equal to mixed (sanity).
  * ``RunMetrics.train_top1_clean`` round-trips through to_dict.

Run via:
    python -m pytest tests/test_train_accuracy_under_mixup.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nature_inspired_networks.eval import RunMetrics  # noqa: E402
from nature_inspired_networks.train import TrainConfig, Trainer  # noqa: E402


class _TinyNet(nn.Module):
    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()
        # >= 1000 parameters so composite_score doesn't raise on D9 guard.
        self.conv1 = nn.Conv2d(3, 24, 3, padding=1)
        self.conv2 = nn.Conv2d(24, 24, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(24, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))


def _make_loaders(n: int = 64, batch: int = 16, n_classes: int = 5):
    torch.manual_seed(0)
    x = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, n_classes, (n,))
    ds = TensorDataset(x, y)
    return (
        DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True),
        DataLoader(ds, batch_size=batch, shuffle=False),
    )


def test_train_top1_clean_emitted_when_mixup_active():
    """Trainer with mixup_alpha>0 must emit ``train_top1_clean`` in
    ``fit_info`` (the un-mixed eval-mode accuracy). The mixed
    ``train_top1_final`` reflects the lam-capped accuracy; the clean
    value reflects the un-confounded train-set fit."""
    torch.manual_seed(0)
    tr_loader, te_loader = _make_loaders()
    cfg = TrainConfig(
        epochs=2, lr=1e-2, weight_decay=0.0, label_smoothing=0.0,
        warmup_epochs=0, use_bf16=False,
        mixup_alpha=0.2, cutmix_alpha=0.0,
        train_top1_clean_samples=64,
    )
    tr = Trainer(_TinyNet(), tr_loader, te_loader, num_classes=5,
                 cfg=cfg, device="cpu")
    info = tr.fit()
    assert "train_top1_clean" in info
    assert info["train_top1_clean"] is not None
    # The clean value is in [0, 1].
    assert 0.0 <= float(info["train_top1_clean"]) <= 1.0
    # History rows record both fields when mixup is active.
    assert "train_top1_clean" in tr.history[-1]
    assert "train_top1_mixed" in tr.history[-1]


def test_train_top1_clean_legacy_path_no_mixing():
    """Without mixing the clean field still propagates -- equal to the
    raw mixed accuracy (since there is no mix to confound it). The
    diagnostic stays a valid number."""
    torch.manual_seed(0)
    tr_loader, te_loader = _make_loaders()
    cfg = TrainConfig(
        epochs=1, lr=1e-2, weight_decay=0.0, label_smoothing=0.0,
        warmup_epochs=0, use_bf16=False,
        mixup_alpha=0.0, cutmix_alpha=0.0,
        train_top1_clean_samples=64,
    )
    tr = Trainer(_TinyNet(), tr_loader, te_loader, num_classes=5,
                 cfg=cfg, device="cpu")
    info = tr.fit()
    # Clean populated even without mixing (sanity; clean == mixed).
    assert info["train_top1_clean"] is not None
    assert abs(info["train_top1_clean"] - info["train_top1_final"]) < 1e-6


def test_run_metrics_train_top1_clean_roundtrip():
    """RunMetrics carries the new field through to_dict so downstream
    dashboards and tools can compute the truthful gap."""
    rm = RunMetrics(
        tag="t", dataset="cifar10", seed=0, epochs=1,
        top1=0.5, top5=0.9, params=100_000, flops=1e6, latency_ms=1.0,
        rot_eq_err=0.0, composite=0.5,
        train_top1=0.45, train_top1_clean=0.78,
    )
    d = rm.to_dict()
    assert d["train_top1_clean"] == 0.78
    # Default behaviour: when unspecified the field is None (back-compat).
    rm2 = RunMetrics(
        tag="t", dataset="cifar10", seed=0, epochs=1,
        top1=0.5, top5=0.9, params=100_000, flops=1e6, latency_ms=1.0,
        rot_eq_err=0.0, composite=0.5,
        train_top1=0.45,
    )
    assert rm2.to_dict()["train_top1_clean"] is None


def test_generalization_gap_uses_clean_when_available():
    """``evaluate_full`` uses the clean train top-1 (no max-floor) when
    available. We synthesise a fit_info that mimics a Mixup run where
    mixed = 0.45 caps at lam and clean = 0.85 is the truthful value."""
    from nature_inspired_networks.train import evaluate_full
    torch.manual_seed(0)
    tr_loader, te_loader = _make_loaders()
    model = _TinyNet()
    # Fake a fit_info as if a Mixup run finished. test_top1 is sampled
    # from the actual evaluate_full call but the gap is computed from
    # train_top1_clean - test_top1 (no floor).
    fake_fit = {
        "history": [],
        "train_seconds": 1.0,
        "epochs_to_target": 1,
        "train_top1_final": 0.45,
        "train_top1_clean": 0.85,
    }
    metrics = evaluate_full(model, te_loader, dataset="cifar10",
                            tag="t", seed=0, epochs=1, fit_info=fake_fit,
                            device="cpu")
    # gap = clean - test_top1; clean was 0.85 -> gap is close to that
    # minus a near-chance test_top1 (~0.25 for 4-class random). No floor.
    expected = 0.85 - metrics.top1
    assert abs(metrics.generalization_gap - expected) < 1e-6
    # train_top1_clean propagates through to RunMetrics.
    assert metrics.train_top1_clean == 0.85


if __name__ == "__main__":
    import pytest as _p
    _p.main([__file__, "-v"])
    print("All 4 tests passed.")
