"""Unit tests for nature_inspired_networks.train Trainer extensions.

Phase-9e Wave-1 H88 wiring fix — coverage for the H51 Topological Betti
Loss integration in :class:`train.Trainer`. The Tag-Adder identified
this Trainer-side wiring gap as the blocker for the
``combo_novelty_betti_torus`` (H88) sweep row.

Tests:
- ``test_train_config_accepts_betti_loss_weight``: TrainConfig has the
  new field, defaults to 0.0, and accepts a positive override.
- ``test_train_loop_adds_betti_term_when_weight_positive``: a single
  ``_step`` with ``betti_loss_weight > 0`` produces a different loss
  value than the same step with the weight pinned to 0 — proving the
  auxiliary Betti term is added (not silently swallowed).
- ``test_train_loop_skips_betti_term_when_weight_zero``: with
  ``betti_loss_weight = 0`` the Trainer leaves ``betti_loss_fn = None``
  and the resulting per-step loss matches a hand-computed CE loss
  exactly (regression guard: legacy behaviour byte-for-byte).

Run with the canonical ``__main__`` pattern -- no pytest dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.betti_loss import BettiLoss  # noqa: E402
from nature_inspired_networks.train import TrainConfig, Trainer  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny CIFAR-shaped model that exposes ``stagewise_features`` so the H51
# BettiLoss auxiliary head has a real penultimate feature map to chew on.
# ---------------------------------------------------------------------------
class _TinyWithStages(nn.Module):
    """Tiny multi-stage model that exposes ``stagewise_features``.

    The Trainer's H51 wiring extracts the deepest stage's feature map via
    ``model.stagewise_features(x)[-1]``. A model without that helper is a
    valid configuration (the auxiliary loss is silently skipped) but the
    tests below need to verify the *active* code path, so the toy model
    explicitly implements the protocol.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.stage1 = nn.Conv2d(8, 8, 3, padding=1, bias=False)
        self.stage2 = nn.Conv2d(8, 8, 3, padding=1, bias=False)
        self.fc = nn.Linear(8, num_classes)

    def stagewise_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        x = self.stem(x); feats.append(x)
        x = self.stage1(x); feats.append(x)
        x = self.stage2(x); feats.append(x)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.stagewise_features(x)
        return self.fc(F.adaptive_avg_pool2d(feats[-1], 1).flatten(1))


def _toy_loaders(num_classes: int = 10):
    torch.manual_seed(0)
    x = torch.randn(8, 3, 8, 8)
    y = torch.randint(0, num_classes, (8,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    return loader, loader


# ---------------------------------------------------------------------------
# Test 1 — TrainConfig accepts betti_loss_weight (and exposes defaults).
# ---------------------------------------------------------------------------
def test_train_config_accepts_betti_loss_weight():
    """TrainConfig must:
      (a) default ``betti_loss_weight`` to 0.0 (legacy behaviour),
      (b) accept a positive override + companion hyperparams,
      (c) round-trip the override via :func:`dataclasses.asdict`-free
          attribute access (the runner reads the attribute, not a dict).
    """
    # Default — must NOT crash on legacy configs (back-compat).
    default = TrainConfig()
    assert hasattr(default, "betti_loss_weight"), (
        "TrainConfig is missing the new betti_loss_weight field"
    )
    assert default.betti_loss_weight == 0.0
    assert default.betti_persistence_threshold == 0.1
    assert default.betti_max_pts == 64

    # Override — Phase-9e Wave-1 H88 row uses betti_loss_weight=0.01.
    cfg = TrainConfig(
        betti_loss_weight=0.01,
        betti_persistence_threshold=0.2,
        betti_max_pts=32,
    )
    assert cfg.betti_loss_weight == 0.01
    assert cfg.betti_persistence_threshold == 0.2
    assert cfg.betti_max_pts == 32


# ---------------------------------------------------------------------------
# Test 2 — _step with weight > 0 adds the Betti term to CE.
# ---------------------------------------------------------------------------
def test_train_loop_adds_betti_term_when_weight_positive():
    """A Trainer constructed with ``betti_loss_weight > 0`` must:
      (a) build a ``BettiLoss`` module (``betti_loss_fn is not None``),
      (b) produce a per-step loss that differs from the same step with
          weight=0 by exactly ``betti_loss_weight * BettiLoss(features)``.

    Implementation-level mechanism check (not just a shape smoke):
    we run a single ``_step`` under each config on the SAME batch and
    SAME initial weights, then compare the resulting losses. The
    positive-weight loss must be strictly greater than the weight=0
    loss by a non-zero amount (BettiLoss is non-negative; for a random
    feature cloud the MSE-to-target term is positive in expectation).
    """
    tr_loader, te_loader = _toy_loaders()

    # Build a model once and snapshot the state_dict so both Trainers
    # start from identical weights.
    torch.manual_seed(7)
    model_seed = _TinyWithStages()
    init_sd = {k: v.clone() for k, v in model_seed.state_dict().items()}

    # Trainer A — weight = 0 (legacy CE-only path).
    cfg_a = TrainConfig(
        epochs=1, lr=1e-3, weight_decay=0.0, target_top1=1.0,
        use_bf16=False, betti_loss_weight=0.0,
    )
    m_a = _TinyWithStages()
    m_a.load_state_dict(init_sd)
    tr_a = Trainer(m_a, tr_loader, te_loader, num_classes=10,
                   cfg=cfg_a, device="cpu")
    assert tr_a.betti_loss_fn is None, (
        "betti_loss_fn must remain None when weight=0"
    )

    # Trainer B — weight > 0 (Betti term active).
    cfg_b = TrainConfig(
        epochs=1, lr=1e-3, weight_decay=0.0, target_top1=1.0,
        use_bf16=False, betti_loss_weight=1.0,
    )
    m_b = _TinyWithStages()
    m_b.load_state_dict(init_sd)
    tr_b = Trainer(m_b, tr_loader, te_loader, num_classes=10,
                   cfg=cfg_b, device="cpu")
    assert tr_b.betti_loss_fn is not None, (
        "betti_loss_fn must be wired when weight > 0"
    )
    assert isinstance(tr_b.betti_loss_fn, BettiLoss)

    # Run one step on each from the SAME batch.
    x, y = next(iter(tr_loader))
    loss_a, _ = tr_a._step(x, y)

    # Re-instantiate trainers from the same init weights so the gradient
    # step in tr_a does not influence the parameters of tr_b.
    m_b2 = _TinyWithStages()
    m_b2.load_state_dict(init_sd)
    tr_b2 = Trainer(m_b2, tr_loader, te_loader, num_classes=10,
                    cfg=cfg_b, device="cpu")
    loss_b, _ = tr_b2._step(x, y)

    # The Betti term is non-negative; on a fresh random feature cloud
    # the (β₀, β₁, β₂) estimate is highly unlikely to match the target
    # (1, 0, 0) exactly, so the MSE is strictly > 0 and the active-weight
    # loss must strictly exceed the weight=0 baseline by a tangible
    # margin (we use a small tolerance to guard against fp drift).
    assert loss_b > loss_a + 1e-6, (
        f"betti_loss_weight=1.0 should produce a strictly larger total "
        f"loss; got loss_b={loss_b:.6f} vs loss_a={loss_a:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — _step with weight = 0 matches hand-computed CE exactly.
# ---------------------------------------------------------------------------
def test_train_loop_skips_betti_term_when_weight_zero():
    """Regression guard: ``betti_loss_weight = 0`` (the default) must
    leave the per-step loss byte-for-byte identical to the legacy
    CE-only path. We re-implement the CE computation by hand and assert
    that the Trainer's returned loss matches to fp32 precision.
    """
    tr_loader, te_loader = _toy_loaders()

    torch.manual_seed(11)
    model = _TinyWithStages()
    cfg = TrainConfig(
        epochs=1, lr=1e-3, weight_decay=0.0, target_top1=1.0,
        use_bf16=False, betti_loss_weight=0.0, label_smoothing=0.1,
    )
    tr = Trainer(model, tr_loader, te_loader, num_classes=10,
                 cfg=cfg, device="cpu")
    assert tr.betti_loss_fn is None
    assert tr.betti_loss_weight == 0.0

    x, y = next(iter(tr_loader))
    # Hand-compute CE BEFORE the step (since _step updates the weights).
    with torch.no_grad():
        logits_pre = model(x)
        ce_pre = F.cross_entropy(logits_pre, y, label_smoothing=0.1)
    loss_actual, _ = tr._step(x, y)

    # _step uses the SAME pre-update logits to compute its loss
    # (the gradient step happens AFTER loss.backward() inside _step),
    # so the returned scalar must match our pre-step CE within fp32
    # precision. The Betti term contributes zero (weight=0 path).
    assert abs(loss_actual - float(ce_pre.item())) < 1e-5, (
        f"weight=0 loss should equal pure CE; got {loss_actual:.6f} "
        f"vs CE {float(ce_pre.item()):.6f}"
    )


# ---------------------------------------------------------------------------
# Bonus regression — Trainer with a model that LACKS stagewise_features
# must not crash when betti_loss_weight > 0; the auxiliary loss is
# silently skipped (graceful degradation, per the Trainer wiring doc).
# ---------------------------------------------------------------------------
def test_train_loop_handles_model_without_stagewise_features():
    """A model that does NOT implement ``stagewise_features`` must
    still train when ``betti_loss_weight > 0`` — the auxiliary loss is
    skipped, the CE-only loss is used, and ``_step`` returns a finite
    scalar. This is the documented graceful-fallback behaviour for
    architectures (e.g. ResNet wrappers, ViT) that don't expose a
    multi-stage feature emitter.
    """
    class _NoStages(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
            self.fc = nn.Linear(8, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.adaptive_avg_pool2d(self.conv(x), 1).flatten(1)
            return self.fc(x)

    tr_loader, te_loader = _toy_loaders()
    cfg = TrainConfig(
        epochs=1, lr=1e-3, weight_decay=0.0, target_top1=1.0,
        use_bf16=False, betti_loss_weight=1.0,
    )
    tr = Trainer(_NoStages(), tr_loader, te_loader, num_classes=10,
                 cfg=cfg, device="cpu")
    # betti_loss_fn is wired (weight > 0) but the per-step extractor
    # returns None for this model, so the term is skipped silently.
    assert tr.betti_loss_fn is not None
    x, y = next(iter(tr_loader))
    loss, _ = tr._step(x, y)
    assert torch.isfinite(torch.tensor(loss)), f"non-finite loss {loss}"


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
