"""Unit tests for H76 — TetrahedralDualPathBlock."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.tetra_dualpath import TetrahedralDualPathBlock  # noqa: E402
from nature_inspired_networks.priors import GroupConv2d  # noqa: E402


def test_forward_shape_preserved():
    """Shape: (B, C, 32, 32) in -> (B, C, 32, 32) out for a residual block."""
    block = TetrahedralDualPathBlock(16, 16, kernel_size=3, stride=1, padding=1)
    x = torch.randn(4, 16, 32, 32)
    y = block(x)
    assert y.shape == (4, 16, 32, 32), y.shape


def test_beta_one_reduces_to_pure_max_path():
    """Mechanism: beta=1 => out == path_max output (+ residual)."""
    block = TetrahedralDualPathBlock(8, 8, beta_init=0.5, residual=False)
    # Force beta -> 1 by setting raw logit large.
    with torch.no_grad():
        block.beta_raw.fill_(50.0)
    x = torch.randn(2, 8, 16, 16)
    out = block(x)
    pure_max = block.path_max(x)
    assert torch.allclose(out, pure_max, atol=1e-4), (out - pure_max).abs().max()
    assert block.beta.item() > 0.999


def test_beta_zero_reduces_to_pure_mean_path():
    """Mechanism: beta=0 => out == path_mean output (+ residual)."""
    block = TetrahedralDualPathBlock(8, 8, beta_init=0.5, residual=False)
    with torch.no_grad():
        block.beta_raw.fill_(-50.0)
    x = torch.randn(2, 8, 16, 16)
    out = block(x)
    pure_mean = block.path_mean(x)
    assert torch.allclose(out, pure_mean, atol=1e-4), (out - pure_mean).abs().max()
    assert block.beta.item() < 0.001


def test_two_paths_produce_different_activations():
    """Mechanism: the max-path and mean-path must genuinely differ.

    They share identical input but use complementary orbit reductions, so on a
    non-degenerate input their outputs cannot be (almost) equal everywhere.
    """
    block = TetrahedralDualPathBlock(8, 8, residual=False)
    # Copy path_max weights into path_mean so ONLY the reduction differs.
    with torch.no_grad():
        block.path_mean.weight.copy_(block.path_max.weight)
    x = torch.randn(2, 8, 16, 16)
    a = block.path_max(x)
    b = block.path_mean(x)
    assert not torch.allclose(a, b, atol=1e-4), "max and mean reductions coincided"


def test_gradient_flows_to_beta():
    """Mechanism/regression: the merge coefficient must receive gradient."""
    block = TetrahedralDualPathBlock(8, 8, residual=False)
    x = torch.randn(2, 8, 16, 16)
    out = block(x)
    out.sum().backward()
    assert block.beta_raw.grad is not None
    assert torch.isfinite(block.beta_raw.grad).all()
    assert block.beta_raw.grad.abs().item() > 0.0


def test_residual_skip_only_when_shape_matches():
    """Edge case: residual skip is disabled when channels or stride change."""
    # Channel change -> no residual skip (cannot add x of different channels).
    block = TetrahedralDualPathBlock(8, 16, residual=True)
    assert block.use_residual is False
    x = torch.randn(2, 8, 16, 16)
    y = block(x)
    assert y.shape == (2, 16, 16, 16)
    # Stride change -> no residual skip (spatial mismatch).
    block2 = TetrahedralDualPathBlock(8, 8, stride=2, padding=1, residual=True)
    assert block2.use_residual is False
    y2 = block2(torch.randn(2, 8, 16, 16))
    assert y2.shape == (2, 8, 8, 8)


def test_beta_init_respected():
    """Edge case: beta property equals beta_init at construction."""
    block = TetrahedralDualPathBlock(8, 8, beta_init=0.7)
    assert abs(block.beta.item() - 0.7) < 1e-5


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
