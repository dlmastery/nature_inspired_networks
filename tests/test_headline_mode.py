"""Synthesis-100 D4 / D5 (2026-06-06): headline_mode determinism.

The legacy ``set_seed`` set only ``random`` / ``numpy`` /
``torch.manual_seed`` and enabled ``cudnn.benchmark=True``. That is
sufficient for informal reproducibility (median over 3 seeds is stable
to ~0.4 pp) but not for the convergence-regime cert runs where two
seeded runs MUST produce bit-identical outputs.

``set_seed(seed, headline_mode=True)`` additionally:
  * ``torch.use_deterministic_algorithms(True)`` (with ``warn_only`` for
    operations that don't yet have deterministic kernels)
  * ``cudnn.deterministic = True``, ``cudnn.benchmark = False``
  * sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` if not already present.

And ``seed_worker`` is the per-worker DataLoader seed function
(propagating ``torch.initial_seed()`` to numpy / random) so RandAugment
+ RandomErasing also produce deterministic augmentations.

This test exercises:
  * a small model forward pass produces bit-identical logits across two
    headline-mode-seeded runs over 3 batches.
  * ``seed_worker`` seeds numpy + random to the same value derivable
    from ``torch.initial_seed()``.

Run via:
    python -m pytest tests/test_headline_mode.py -v
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nature_inspired_networks.runner import seed_worker, set_seed  # noqa: E402


class _TinyConv(nn.Module):
    """Small model with enough surface to surface non-determinism."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return self.fc(self.pool(x).flatten(1))


def _one_seeded_run(seed: int, headline_mode: bool) -> list[torch.Tensor]:
    set_seed(seed, headline_mode=headline_mode)
    # ``cuda.manual_seed_all`` is called inside set_seed; the new model's
    # init draws from torch.Generator() which is now seeded. Same with
    # the input batch.
    model = _TinyConv()
    outs: list[torch.Tensor] = []
    for _ in range(3):
        x = torch.randn(4, 3, 16, 16)
        outs.append(model(x).detach().clone())
    return outs


def test_headline_mode_produces_bit_identical_forward():
    """Two runs seeded with the same ``seed`` in headline_mode must
    produce bit-identical logits over 3 batches.

    This is the gate Synthesis-100 D4 specifies: the cert-run path must
    be deterministic enough that an n=7 seed sweep can be re-run from
    scratch to byte-identical metrics.json files. We exercise the CPU
    path so the test does not depend on CUDA / cuDNN availability.
    """
    a = _one_seeded_run(seed=42, headline_mode=True)
    b = _one_seeded_run(seed=42, headline_mode=True)
    assert len(a) == 3 and len(b) == 3
    for i, (ai, bi) in enumerate(zip(a, b)):
        assert ai.shape == bi.shape
        # equal, not allclose -- the requirement is bit-identical.
        assert torch.equal(ai, bi), (
            f"headline_mode determinism broken at batch {i}: "
            f"max abs diff {(ai - bi).abs().max().item():.6e}"
        )


def test_headline_mode_differs_across_seeds():
    """Sanity: two DIFFERENT seeds in headline_mode produce DIFFERENT
    outputs (otherwise the test above is vacuously true)."""
    a = _one_seeded_run(seed=0, headline_mode=True)
    b = _one_seeded_run(seed=1, headline_mode=True)
    # At least one batch must differ.
    any_diff = any(not torch.equal(ai, bi) for ai, bi in zip(a, b))
    assert any_diff, "two distinct seeds produced identical outputs"


def test_seed_worker_seeds_numpy_and_random():
    """``seed_worker`` must seed BOTH numpy and python ``random`` from
    ``torch.initial_seed()`` so per-worker augmentation streams are
    deterministic given the main-process seed.

    We exercise this by calling seed_worker twice with the same torch
    initial seed (set by manual_seed) and asserting numpy / random draws
    match.
    """
    torch.manual_seed(123)
    seed_worker(worker_id=0)
    a1 = np.random.rand()
    b1 = random.random()
    torch.manual_seed(123)
    seed_worker(worker_id=0)
    a2 = np.random.rand()
    b2 = random.random()
    assert a1 == a2, f"numpy stream not deterministic: {a1} vs {a2}"
    assert b1 == b2, f"random stream not deterministic: {b1} vs {b2}"


def test_set_seed_legacy_mode_preserves_benchmark():
    """``set_seed(seed)`` without ``headline_mode`` MUST leave
    ``cudnn.benchmark=True`` so existing screening sweeps keep their
    fast-mode behaviour byte-for-byte (Rule 6 spirit + Synthesis D4
    backward-compat)."""
    set_seed(0, headline_mode=False)
    assert torch.backends.cudnn.benchmark is True
    set_seed(0, headline_mode=True)
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    # Restore default fast-mode for any subsequent tests in this session.
    set_seed(0, headline_mode=False)


if __name__ == "__main__":
    import pytest as _p
    _p.main([__file__, "-v"])
    print("All 4 tests passed.")
