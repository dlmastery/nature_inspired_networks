"""Unit tests for H59 — compute_trained_betti.

Run as a script:
    python tests/test_trained_betti.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.trained_betti import (  # noqa: E402
    compute_trained_betti,
    n_points_for,
)


class _ToyStagedNet(nn.Module):
    """Tiny 4-stage CNN exposing the ``stagewise_features`` protocol."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.stage1 = nn.Conv2d(8, 8, 3, padding=1)
        self.stage2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.stage3 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.stagewise_features(x)
        return self.fc(self.pool(feats[-1]).flatten(1))

    def stagewise_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        f0 = torch.relu(self.stem(x))
        f1 = torch.relu(self.stage1(f0))
        f2 = torch.relu(self.stage2(f1))
        f3 = torch.relu(self.stage3(f2))
        return [f0, f1, f2, f3]


def _make_test_loader(n: int = 32) -> DataLoader:
    x = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, 4, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def _save_temp_checkpoint(model: nn.Module) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="h59_test_"))
    ckpt_path = tmpdir / "best.pt"
    # save under the same key the runner uses ("model_state")
    torch.save({"model_state": model.state_dict()}, ckpt_path)
    return ckpt_path


def test_returns_dict_with_four_stages():
    model = _ToyStagedNet()
    ckpt = _save_temp_checkpoint(model)
    loader = _make_test_loader()
    out = compute_trained_betti(ckpt, model, loader, n_samples=24, device="cpu")
    assert isinstance(out, dict), type(out)
    # 4 stages = stem + stage1 + stage2 + stage3 = canonical names
    assert set(out.keys()) == {"stem", "stage1", "stage2", "stage3"}, list(out.keys())


def test_curve_has_b0_b1_per_stage():
    model = _ToyStagedNet()
    ckpt = _save_temp_checkpoint(model)
    loader = _make_test_loader()
    out = compute_trained_betti(ckpt, model, loader, n_samples=24, device="cpu")
    for name, curve in out.items():
        assert "b0" in curve and "b1" in curve, (name, curve)
        assert isinstance(curve["b0"], int), (name, curve["b0"])
        assert isinstance(curve["b1"], int), (name, curve["b1"])


def test_curve_is_nan_free_and_finite():
    model = _ToyStagedNet()
    ckpt = _save_temp_checkpoint(model)
    loader = _make_test_loader()
    out = compute_trained_betti(ckpt, model, loader, n_samples=24, device="cpu")
    for name, curve in out.items():
        for k in ("b0", "b1"):
            v = curve[k]
            assert isinstance(v, int)
            # ints in python are always finite, but the assertion documents
            # the contract; check non-negative.
            assert v >= 0, (name, k, v)


def test_checkpoint_load_handles_plain_state_dict():
    """Plain state_dict (no wrapper) also loads."""
    model = _ToyStagedNet()
    tmpdir = Path(tempfile.mkdtemp(prefix="h59_test_plain_"))
    ckpt = tmpdir / "best.pt"
    torch.save(model.state_dict(), ckpt)
    loader = _make_test_loader()
    out = compute_trained_betti(ckpt, model, loader, n_samples=24, device="cpu")
    assert len(out) == 4


def test_missing_checkpoint_raises():
    model = _ToyStagedNet()
    loader = _make_test_loader()
    bogus = Path(tempfile.gettempdir()) / "does_not_exist_h59.pt"
    try:
        compute_trained_betti(bogus, model, loader, n_samples=8, device="cpu")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_n_points_for_validation():
    assert n_points_for(256) == 256
    try:
        n_points_for(0)
        raise AssertionError("expected ValueError for n_samples=0")
    except ValueError:
        pass


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
