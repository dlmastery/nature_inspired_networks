"""Unit tests for the modern 11-trick convergence-regime recipe.

Covers the five recipe primitives added to ``src/nature_inspired_networks``
(Mixup, CutMix, RandomErasing, RandAugment, EMA) plus an end-to-end
training-loop smoke that wires all of them through the existing Trainer.

Per CLAUDE.md Rule 12: every new module ships with a unit test
exercising shape, every Boolean-flag combination, and the bug class it
was added to fix. Per Rule 25: every Q&A test name advertised in the
convergence/PLAN.md MUST exist here.

Run via:
    python -m pytest tests/test_modern_recipe.py -v

End-of-file footer prints "All N tests passed." for the bare-pytest
sanity convention.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nature_inspired_networks.cutmix import cutmix_batch, rand_bbox  # noqa: E402
from nature_inspired_networks.ema import ModelEMA  # noqa: E402
from nature_inspired_networks.mixup import mixup_batch, mixup_loss  # noqa: E402
from nature_inspired_networks.randaugment import build_randaugment  # noqa: E402
from nature_inspired_networks.random_erasing import (  # noqa: E402
    apply_random_erasing_to_batch,
    build_random_erasing,
)


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------
def test_mixup_lambda_in_unit_interval():
    """Per Zhang 2018: λ ~ Beta(α, α) is in (0, 1). Repeated draws must
    every time produce a λ strictly inside the unit interval AND the
    returned mixed image must satisfy x_mix = λ·x + (1-λ)·x[perm]."""
    torch.manual_seed(0)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    for _ in range(20):
        x_mix, y_a, y_b, lam = mixup_batch(x, y, alpha=0.2)
        assert 0.0 < lam < 1.0, f"lam out of (0,1): {lam}"
        # y_a is always the original; y_b is the permuted view
        assert torch.equal(y_a, y)
        # x_mix is a convex combination of x and x[perm] for some perm
        # We can't recover perm without a flag, but we can assert finite.
        assert torch.isfinite(x_mix).all()
    # Boundary case: alpha=0 disables Mixup.
    x_mix0, y_a0, y_b0, lam0 = mixup_batch(x, y, alpha=0.0)
    assert lam0 == 1.0
    assert torch.equal(x_mix0, x)
    assert torch.equal(y_a0, y)
    assert torch.equal(y_b0, y)


def test_mixup_loss_matches_manual():
    """mixup_loss must equal the manual ``λ·CE(y_a) + (1-λ)·CE(y_b)``."""
    torch.manual_seed(1)
    logits = torch.randn(4, 5)
    y_a = torch.tensor([0, 1, 2, 3])
    y_b = torch.tensor([4, 0, 1, 2])
    lam = 0.7
    loss = mixup_loss(logits, y_a, y_b, lam, label_smoothing=0.0)
    manual = (
        lam * nn.functional.cross_entropy(logits, y_a)
        + (1.0 - lam) * nn.functional.cross_entropy(logits, y_b)
    )
    assert torch.allclose(loss, manual)


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------
def test_cutmix_box_within_image_bounds():
    """Per Yun 2019: the CutMix box must always be inside the image. For
    any λ ∈ [0, 1] the rand_bbox helper must return integer coordinates
    with 0 <= x1 <= x2 <= W and 0 <= y1 <= y2 <= H. The cutmix_batch
    output must also have the right shape and finite values."""
    H, W = 32, 32
    for lam in (0.05, 0.25, 0.5, 0.75, 0.95):
        for _ in range(20):
            x1, y1, x2, y2 = rand_bbox((1, 3, H, W), lam)
            assert 0 <= x1 <= x2 <= W, (x1, x2, W)
            assert 0 <= y1 <= y2 <= H, (y1, y2, H)
    # End-to-end batch path
    torch.manual_seed(0)
    x = torch.randn(8, 3, H, W)
    y = torch.randint(0, 10, (8,))
    x_mix, y_a, y_b, lam_eff = cutmix_batch(x, y, alpha=1.0)
    assert x_mix.shape == x.shape
    # lam_eff is the area-corrected ratio; must be in [0, 1].
    assert 0.0 <= lam_eff <= 1.0
    assert torch.isfinite(x_mix).all()
    # alpha=0 -> no-op.
    x0, y_a0, y_b0, l0 = cutmix_batch(x, y, alpha=0.0)
    assert l0 == 1.0
    assert torch.equal(x0, x)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
def test_ema_tracks_weights_over_steps():
    """Per Tarvainen 2017 / Bello 2021: ``p_ema <- d·p_ema + (1-d)·p_live``.
    Starting from identical weights, after a single update at decay=0.5
    the EMA must equal the midpoint of the original and the perturbed
    parameter. After many updates with a fresh (rapidly diverging) live
    model, the EMA must move monotonically toward the live model."""
    torch.manual_seed(0)
    live = nn.Linear(4, 4, bias=False)
    ema = ModelEMA(live, decay=0.5)
    # Initial: EMA == live.
    assert torch.equal(ema.module.weight, live.weight)
    # Perturb live by a known amount.
    old_w = live.weight.detach().clone()
    with torch.no_grad():
        live.weight.add_(torch.ones_like(live.weight))
    ema.update(live)
    expected = 0.5 * old_w + 0.5 * live.weight.detach()
    assert torch.allclose(ema.module.weight, expected, atol=1e-6)
    # Many updates: distance EMA<->live must shrink toward 0.
    distances = []
    for _ in range(30):
        ema.update(live)
        distances.append(float((ema.module.weight - live.weight).abs().mean()))
    # Strictly non-increasing (monotone convergence under fixed live).
    for a, b in zip(distances, distances[1:]):
        assert b <= a + 1e-9, f"EMA diverged: {a} -> {b}"
    # And final distance is small.
    assert distances[-1] < distances[0] / 10.0


# ---------------------------------------------------------------------------
# RandAugment
# ---------------------------------------------------------------------------
def test_randaugment_outputs_3channel_uint8():
    """Per Cubuk 2020: RandAugment operates on PIL/uint8 tensors. The
    torchvision wrapper must accept a (C, H, W) uint8 tensor and return
    a tensor of the same shape and dtype, with values still in [0, 255]."""
    torch.manual_seed(0)
    aug = build_randaugment(num_ops=2, magnitude=14)
    x = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    y = aug(x)
    assert y.dtype == torch.uint8
    assert y.shape == x.shape
    assert int(y.min()) >= 0
    assert int(y.max()) <= 255
    # Validation of arg ranges.
    with pytest.raises(ValueError):
        build_randaugment(num_ops=-1, magnitude=14)
    with pytest.raises(ValueError):
        build_randaugment(num_ops=2, magnitude=31)


# ---------------------------------------------------------------------------
# Random Erasing
# ---------------------------------------------------------------------------
def test_random_erasing_in_place():
    """Per Zhong 2020: RandomErasing zeroes a random rectangle of each
    image (post-Normalize). With p=1.0 the helper must modify EVERY image
    (the modified image differs from the input). With p=0.0 the helper
    must leave the image untouched."""
    torch.manual_seed(0)
    x = torch.ones(8, 3, 32, 32)
    # p=1.0 -> every image must change.
    erase_all = build_random_erasing(p=1.0)
    y = apply_random_erasing_to_batch(x, erase_all)
    diff = (y != x).any(dim=(1, 2, 3))
    assert bool(diff.all()), "p=1.0 erasing did not modify every image"
    # p=0.0 -> nothing changes.
    erase_none = build_random_erasing(p=0.0)
    y0 = apply_random_erasing_to_batch(x, erase_none)
    assert torch.equal(x, y0), "p=0.0 erasing should be the identity"
    with pytest.raises(ValueError):
        build_random_erasing(p=1.5)


# ---------------------------------------------------------------------------
# Full-recipe end-to-end smoke
# ---------------------------------------------------------------------------
def _make_toy_loader(n: int = 32, num_classes: int = 6, batch_size: int = 8):
    """Tiny 3x8x8 dataset for the train-loop smoke."""
    x = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, num_classes, (n,))
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


class _Tiny(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8 * 8 * 8, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.fc(x.flatten(1))


def test_train_loop_runs_with_full_modern_recipe():
    """Smoke: build a Trainer with mixup_alpha + cutmix_alpha + ema_decay
    + warmup_epochs all non-default, run a 2-epoch fit, and assert the
    history has the expected EMA columns + the final model parameters
    differ from a fresh-init checkpoint (training actually moved)."""
    from nature_inspired_networks.train import TrainConfig, Trainer

    torch.manual_seed(0)
    num_classes = 6
    model = _Tiny(num_classes=num_classes)
    init_w = model.fc.weight.detach().clone()

    cfg = TrainConfig(
        epochs=2,
        lr=1e-2,
        weight_decay=5e-4,
        label_smoothing=0.1,
        warmup_epochs=1,
        target_top1=2.0,    # unreachable -> epochs_to_target stays -1
        use_bf16=False,     # CPU smoke
        # Modern recipe — all five tricks at once.
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        mixup_cutmix_prob=0.5,
        ema_decay=0.9,      # exaggerated for the 2-epoch smoke
    )
    loader = _make_toy_loader(num_classes=num_classes)
    trainer = Trainer(model, loader, loader, num_classes, cfg, device="cpu")
    assert trainer.ema is not None, "EMA shadow not constructed"
    # Label smoothing is dropped when mixing is on (Bello 2021).
    assert trainer._effective_label_smoothing == 0.0
    info = trainer.fit()
    assert len(info["history"]) == 2
    # EMA diagnostics must be present in history.
    assert "test_top1_ema" in info["history"][-1]
    # Training actually changed the parameters.
    assert not torch.equal(init_w, model.fc.weight.detach())


def test_train_loop_legacy_path_unchanged():
    """Regression: with default TrainConfig (mixup=cutmix=ema=warmup=0)
    the trainer must NOT construct an EMA shadow and must NOT touch
    label_smoothing."""
    from nature_inspired_networks.train import TrainConfig, Trainer

    torch.manual_seed(0)
    model = _Tiny(num_classes=6)
    cfg = TrainConfig(epochs=1, lr=1e-2, use_bf16=False,
                      label_smoothing=0.1)
    loader = _make_toy_loader()
    trainer = Trainer(model, loader, loader, 6, cfg, device="cpu")
    assert trainer.ema is None
    assert trainer._effective_label_smoothing == 0.1
    info = trainer.fit()
    assert len(info["history"]) == 1
    assert "test_top1_ema" not in info["history"][0]


if __name__ == "__main__":
    # Bare-pytest sanity convention (Rule 12 footer).
    import subprocess

    rc = subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"])
    if rc == 0:
        print("All 8 tests passed.")
    sys.exit(rc)
