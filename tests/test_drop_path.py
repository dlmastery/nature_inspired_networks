"""Unit tests for H52 — Drop-Path Regularization & Anytime Evaluation.

Asserts (per the H52 design doc Q&A section):
  - DropPath is the identity in eval mode (bit-equivalent);
  - DropPath in train mode drops a non-trivial fraction of samples
    (deterministic with a seeded RNG, regression test);
  - FractalDropPath preserves forward shape across an N-block stack;
  - the per-block drop schedule is monotone non-decreasing in depth;
  - anytime_forward dispatches to set_max_depth at each requested depth
    and returns a dict keyed by depth;
  - anytime_forward raises AttributeError on a model that doesn't
    expose set_max_depth (the additive-integration contract).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.drop_path import (  # noqa: E402
    DropPath,
    FractalDropPath,
    anytime_forward,
)


def test_droppath_identity_in_eval_mode():
    """Test (1): in eval mode DropPath is exactly the identity, regardless
    of p. This is the determinism invariant from the design doc."""
    torch.manual_seed(0)
    dp = DropPath(p=0.5)
    dp.eval()
    x = torch.randn(8, 4, 16, 16)
    y = dp(x)
    assert torch.equal(x, y), "DropPath must be exact identity in eval"


def test_droppath_drops_samples_in_train_mode():
    """Test (2): in train mode with p=0.5 some samples become identically
    zero (deterministic given the seed). This is the regression for the
    "dropping is actually happening" contract.
    """
    torch.manual_seed(123)
    dp = DropPath(p=0.5)
    dp.train()
    x = torch.ones(16, 4, 8, 8)
    y = dp(x)
    # Number of samples with all-zero output should be > 0 and < B.
    per_sample_norm = y.flatten(1).abs().sum(dim=1)
    n_dropped = (per_sample_norm == 0).sum().item()
    assert 0 < n_dropped < 16, n_dropped
    # The non-dropped samples must be rescaled by 1/(1-p) = 2.
    survivor = y[per_sample_norm > 0]
    assert torch.allclose(survivor, torch.full_like(survivor, 2.0))


def test_droppath_p_zero_is_identity_in_training():
    """Test (3): p=0 must be the identity in BOTH eval and train mode.
    This is the "baseline-recovers" check from the design-doc Q&A.
    """
    torch.manual_seed(0)
    dp = DropPath(p=0.0)
    dp.train()
    x = torch.randn(4, 8, 16, 16)
    y = dp(x)
    assert torch.equal(x, y)


def test_droppath_rejects_invalid_p():
    """Constructor must reject p outside [0, 1)."""
    for bad in (-0.1, 1.0, 1.5):
        try:
            DropPath(p=bad)
            raise AssertionError(f"expected ValueError for p={bad}")
        except ValueError as exc:
            if "expected ValueError" in str(exc):
                raise


def test_fractal_drop_path_forward_shape():
    """Test (4): FractalDropPath preserves (B, C, H, W) across N residual
    blocks. We use trivial identity-conv blocks so the residual addition
    is exact."""
    torch.manual_seed(0)

    class IdConv(nn.Module):
        def __init__(self, c: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(c, c, 1, bias=False)
            nn.init.zeros_(self.conv.weight)  # output is zero → residual recovers x

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    blocks = [IdConv(4) for _ in range(5)]
    stack = FractalDropPath(blocks, p_max=0.2)
    stack.eval()  # deterministic at eval
    x = torch.randn(2, 4, 8, 8)
    y = stack(x)
    assert y.shape == x.shape
    # In eval mode every block emits zero, so the residual sum equals x.
    assert torch.allclose(y, x)


def test_fractal_drop_path_schedule_monotone():
    """Test (5): drop probabilities increase from 0 to p_max linearly.
    The schedule must be strictly increasing (or all-zero for n==1).
    """
    blocks = [nn.Identity() for _ in range(6)]
    stack = FractalDropPath(blocks, p_max=0.25)
    probs = stack.drop_probs
    assert probs[0] == 0.0
    assert abs(probs[-1] - 0.25) < 1e-9
    for a, b in zip(probs[:-1], probs[1:]):
        assert a <= b, (a, b)
    # And the last is strictly larger than the first.
    assert probs[-1] > probs[0]


def test_fractal_drop_path_single_block_schedule():
    """N==1 corner case: schedule must be [0.0] (no monotonicity to enforce)."""
    stack = FractalDropPath([nn.Identity()], p_max=0.5)
    assert stack.drop_probs == [0.0]


def test_anytime_forward_dispatches_by_depth():
    """Test (6): the helper calls set_max_depth and returns a dict keyed
    by the requested depths.
    """

    class TinyAnytime(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self._max_depth = 3

            # Initialize linear to identity for clean inspection.
            nn.init.eye_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

        def set_max_depth(self, d: int) -> None:
            self._max_depth = int(d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Output scales with depth so the test can distinguish runs.
            return self.linear(x) * float(self._max_depth)

    model = TinyAnytime()
    x = torch.ones(2, 4)
    out = anytime_forward(model, x, depths=(1, 2, 3))
    assert set(out.keys()) == {1, 2, 3}
    for d in (1, 2, 3):
        assert torch.allclose(out[d], torch.full_like(x, float(d)))
    # State restored.
    assert model._max_depth == 3


def test_anytime_forward_raises_without_hook():
    """Test (7): a model that doesn't expose set_max_depth produces a
    clear AttributeError — this is the "additive integration" contract.
    """
    model = nn.Linear(4, 4)
    try:
        anytime_forward(model, torch.zeros(1, 4), depths=(1,))
        raise AssertionError("expected AttributeError")
    except AttributeError as exc:
        assert "set_max_depth" in str(exc), str(exc)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
