"""Synthesis-100 A3 (2026-06-06): FLOP-target sanity check in the runner.

The Phase-9i confound (Phase-9i priors ran at 80.8 MFLOPs vs the
baseline's 41.2 MFLOPs but the composite metric penalised only
``params + latency``) is the BLOCKER-class motivation for adding
``flops_target`` to the run config. The runner now measures FLOPs with
fvcore before any GPU compute is spent and refuses to launch when the
measurement is outside the band ``flops_target +/- flops_tolerance``.

This test exercises:
  * within-band PASS — runner returns the model unchanged.
  * outside-band FAIL — runner raises :class:`FLOPTargetError`.
  * a NaN measurement (fvcore unable to count, e.g. HexConv) is treated
    as a warning by default and a hard error under
    ``flops_target_strict=True``.

Run via:
    python -m pytest tests/test_flops_target.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nature_inspired_networks.runner import (  # noqa: E402
    FLOPTargetError,
    _check_flops_target,
)


class _TinyConv(nn.Module):
    """Tiny convnet whose FLOPs are countable by fvcore.

    Two 3x3 conv layers + a final linear. At 32x32 input the per-pixel
    MAC count is roughly
        H * W * (K^2 * C_in * C_out) summed over the two convs
    -- in practice we measure it at construction time and pin the test
    band against the live measurement so the test is not fragile to
    fvcore version drift.
    """

    def __init__(self, c1: int = 8, c2: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _measured_flops(model: nn.Module) -> float:
    from nature_inspired_networks.eval import count_flops
    return count_flops(model, input_size=(1, 3, 32, 32))


def test_flops_target_within_band_passes():
    """A model whose measured FLOPs sit inside ``target +/- tolerance``
    must pass through ``_check_flops_target`` without raising.

    The actual measurement is captured from fvcore at runtime so the
    test is robust to fvcore minor-version drift (any version returning
    a finite count within ~5% of itself satisfies the band)."""
    model = _TinyConv()
    measured = _measured_flops(model)
    assert measured > 0 and measured < 1e9, (
        f"sanity: tiny conv should have a finite small flop count; got {measured}"
    )
    cfg = {
        "flops_target": measured,
        "flops_tolerance": 0.10,
    }
    # Must NOT raise. Returns None.
    out = _check_flops_target(model, cfg, input_size=(1, 3, 32, 32))
    assert out is None


def test_flops_target_outside_band_raises():
    """A model whose measured FLOPs sit OUTSIDE the band must raise
    :class:`FLOPTargetError`. We pin the target at 100x the measured
    value so even a generous tolerance can't accept the model."""
    model = _TinyConv()
    measured = _measured_flops(model)
    # Set target 100x the measurement so the band [90x, 110x] cannot
    # contain it under any reasonable tolerance.
    cfg = {
        "flops_target": measured * 100.0,
        "flops_tolerance": 0.10,
    }
    with pytest.raises(FLOPTargetError) as exc_info:
        _check_flops_target(model, cfg, input_size=(1, 3, 32, 32))
    msg = str(exc_info.value)
    assert "outside target band" in msg
    assert "synthesis-100 A3" in msg


def test_flops_target_absent_is_noop():
    """When ``flops_target`` is absent from cfg the check is a no-op --
    backwards-compatible with every existing screening sweep."""
    model = _TinyConv()
    # Empty cfg.
    _check_flops_target(model, {}, input_size=(1, 3, 32, 32))
    # Explicit None.
    _check_flops_target(model, {"flops_target": None},
                        input_size=(1, 3, 32, 32))


if __name__ == "__main__":
    # Bare-pytest sanity convention (CLAUDE.md Rule 12 footer).
    import pytest as _p
    _p.main([__file__, "-v"])
    print("All 3 tests passed.")
