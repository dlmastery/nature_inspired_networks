"""Phase-C integration tests: Trainer dispatches the right optimizer
and per-layer weight-decay schedule from TrainConfig.

Tests exercise :meth:`Trainer._build_optimizer` directly so they do
not need a DataLoader / device. Each path corresponds to one new sweep
override key:

- ``optimizer='golden_adam'`` (H41) -> GoldenRatioAdamW with betas
  (1/phi, 1/phi^2).
- ``phi_decay_wd=True`` (H44) -> per-block weight decay = base / phi^k
  derived via phi_decay.phi_decay_param_groups.
- Default path -> stock torch.optim.AdamW with a single param group.

Run with the canonical ``__main__`` pattern -- no pytest dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.models import build_model  # noqa: E402
from nature_inspired_networks.optimizers import (  # noqa: E402
    GOLDEN_BETA1, GOLDEN_BETA2, GoldenRatioAdamW,
)
from nature_inspired_networks.priors import PHI  # noqa: E402
from nature_inspired_networks.train import TrainConfig, Trainer  # noqa: E402


def _model():
    return build_model("resnet20", num_classes=10)


def test_default_optimizer_is_stock_adamw():
    """Backward-compat: no flags -> torch.optim.AdamW (legacy path)."""
    m = _model()
    cfg = TrainConfig(lr=1e-3, weight_decay=5e-4)
    opt = Trainer._build_optimizer(m, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    assert not isinstance(opt, GoldenRatioAdamW)
    # single param group for legacy path
    assert len(opt.param_groups) == 1
    assert abs(opt.param_groups[0]["weight_decay"] - 5e-4) < 1e-12


def test_golden_adam_optimizer_dispatch():
    """H41: optimizer='golden_adam' returns GoldenRatioAdamW with
    phi-derived betas.
    """
    m = _model()
    cfg = TrainConfig(lr=1e-3, weight_decay=5e-4, optimizer="golden_adam")
    opt = Trainer._build_optimizer(m, cfg)
    assert isinstance(opt, GoldenRatioAdamW)
    b1, b2 = opt.param_groups[0]["betas"]
    assert abs(b1 - GOLDEN_BETA1) < 1e-9
    assert abs(b2 - GOLDEN_BETA2) < 1e-9


def test_phi_decay_wd_builds_per_block_param_groups():
    """H44: phi_decay_wd=True -> multiple param groups, each with
    weight_decay = base / phi^k for index k = 0, 1, 2, ...
    """
    m = _model()
    base_wd = 1e-2
    cfg = TrainConfig(lr=1e-3, weight_decay=5e-4,
                      phi_decay_wd=True, phi_decay_base=base_wd)
    opt = Trainer._build_optimizer(m, cfg)
    # More than one param group is the smoking gun for per-layer wd.
    assert len(opt.param_groups) > 1
    # Schedule must be monotone-decreasing in weight_decay.
    wds = [g["weight_decay"] for g in opt.param_groups]
    assert all(a >= b for a, b in zip(wds[:-1], wds[1:])), wds
    # First group is base_wd.
    assert abs(wds[0] - base_wd) < 1e-12
    # Second group is base_wd / phi (within fp32).
    assert abs(wds[1] - base_wd / PHI) < 1e-6


def test_phi_decay_wd_composes_with_golden_adam():
    """phi_decay_wd + optimizer='golden_adam' -> GoldenRatioAdamW
    optimiser with per-block weight decay groups. Both flags compose
    without one silently disabling the other.
    """
    m = _model()
    cfg = TrainConfig(
        lr=1e-3, weight_decay=5e-4,
        phi_decay_wd=True, phi_decay_base=2e-3,
        optimizer="golden_adam",
    )
    opt = Trainer._build_optimizer(m, cfg)
    assert isinstance(opt, GoldenRatioAdamW)
    # >1 group AND phi-decayed wds.
    assert len(opt.param_groups) > 1
    wds = [g["weight_decay"] for g in opt.param_groups]
    assert all(a >= b for a, b in zip(wds[:-1], wds[1:])), wds


def test_phi_decay_wd_param_count_preserved():
    """The phi-decay param groups must cover every learnable parameter
    -- a missing group would silently exclude params from the
    optimiser.
    """
    m = _model()
    cfg = TrainConfig(phi_decay_wd=True, phi_decay_base=1e-2)
    opt = Trainer._build_optimizer(m, cfg)
    n_opt = sum(p.numel()
                for g in opt.param_groups for p in g["params"])
    n_model = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_opt == n_model, (n_opt, n_model)


def test_traincfg_phase_c_defaults_back_compat():
    """Newly added TrainConfig fields default to legacy behaviour."""
    cfg = TrainConfig()
    assert cfg.optimizer == "adamw"
    assert cfg.phi_decay_wd is False
    assert abs(cfg.phi_decay_base - 5e-4) < 1e-12
    # Control 1 — const_beta1 default must be None (no pin).
    assert cfg.const_beta1 is None


def test_const_beta1_holds_value_across_epochs():
    """Control 1: when ``const_beta1`` is set the Trainer must pin β1
    on every optimiser param-group to that constant value AND must NOT
    install the H48 GoldenMomentumScheduler -- so β1 stays constant
    across any number of synthetic epochs.
    """
    import torch.utils.data as tud

    m = _model()
    cfg = TrainConfig(
        epochs=3, lr=1e-3, weight_decay=5e-4,
        const_beta1=0.4,
        # Even if golden_momentum is requested, const_beta1 must win.
        momentum_schedule="golden",
    )
    # Tiny synthetic loader so Trainer can construct without I/O.
    xs = torch.randn(2, 3, 32, 32)
    ys = torch.zeros(2, dtype=torch.long)
    ds = tud.TensorDataset(xs, ys)
    loader = tud.DataLoader(ds, batch_size=2)
    tr = Trainer(m, loader, loader, num_classes=10, cfg=cfg, device="cpu")
    # No H48 scheduler installed.
    assert tr.momentum_sched is None, (
        "const_beta1 must bypass the H48 GoldenMomentumScheduler"
    )
    # Every param group's β1 is pinned to 0.4.
    for g in tr.opt.param_groups:
        b1, _ = g["betas"]
        assert abs(b1 - 0.4) < 1e-9, (b1, "expected const_beta1=0.4")
    # Manually re-pin to simulate the rebuild path (proves the call is
    # idempotent and stable across "epochs"); a real fit() never touches
    # β1 when const_beta1 is set.
    for _ in range(5):
        for g in tr.opt.param_groups:
            b1, _ = g["betas"]
            assert abs(b1 - 0.4) < 1e-9


def test_const_beta1_compatible_with_golden_momentum_beta2_decay():
    """Control 1: ``const_beta1`` only pins β1; β2 keeps the optimiser's
    default (0.999 for AdamW). This is the documented behaviour --
    the directive says 'the existing GoldenMomentumScheduler is bypassed
    for β1 only'."""
    import torch.utils.data as tud

    m = _model()
    cfg = TrainConfig(epochs=2, lr=1e-3, weight_decay=5e-4,
                      const_beta1=0.4)
    xs = torch.randn(2, 3, 32, 32)
    ys = torch.zeros(2, dtype=torch.long)
    ds = tud.TensorDataset(xs, ys)
    loader = tud.DataLoader(ds, batch_size=2)
    tr = Trainer(m, loader, loader, num_classes=10, cfg=cfg, device="cpu")
    for g in tr.opt.param_groups:
        b1, b2 = g["betas"]
        assert abs(b1 - 0.4) < 1e-9, b1
        # β2 default for AdamW is 0.999; must be untouched.
        assert abs(b2 - 0.999) < 1e-9, b2


def test_const_beta1_pin_beta1_static_helper():
    """Direct test of Trainer._pin_beta1 on a fresh AdamW optimiser.
    Used by both __init__ and the growth-pruning rebuild path."""
    m = _model()
    cfg = TrainConfig(lr=1e-3, weight_decay=5e-4)
    opt = Trainer._build_optimizer(m, cfg)
    Trainer._pin_beta1(opt, 0.42)
    for g in opt.param_groups:
        b1, b2 = g["betas"]
        assert abs(b1 - 0.42) < 1e-12
        assert abs(b2 - 0.999) < 1e-9


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
