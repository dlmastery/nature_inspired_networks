"""Unit tests for H51 — Topological Betti Loss.

Asserts (per the H51 design doc Q&A section):
  - persistence surrogate returns the documented shape with grad;
  - loss is a finite scalar tensor;
  - loss.backward() flows non-zero gradients back to the input;
  - the (β₀ = 1, β₁ = 0, β₂ = 0) target lower-bounds the loss on a
    randomly initialised Gaussian point cloud (sanity check that the
    surrogate measures *something*, not just returns zero everywhere);
  - a deliberately two-cluster cloud yields a strictly larger β₀=1 loss
    than a single Gaussian blob (regression test for the connectivity
    proxy);
  - max_pts cap is honoured (no silent shape blow-up).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.betti_loss import (  # noqa: E402
    BettiLoss,
    differentiable_persistence,
)


def test_differentiable_persistence_shape_and_grad():
    """Test (1): output shape is (K, 2) and pairs[:, 1] >= pairs[:, 0]."""
    torch.manual_seed(0)
    x = torch.randn(16, 8, requires_grad=True)
    pairs = differentiable_persistence(x)
    # K = N*(N-1)//2 - 1 (drop last so death is defined).
    expected_k = 16 * 15 // 2 - 1
    assert pairs.shape == (expected_k, 2), pairs.shape
    # Persistence is non-negative by construction (death = next-sorted-edge).
    persistence = pairs[:, 1] - pairs[:, 0]
    assert (persistence >= -1e-6).all(), persistence.min().item()
    # Grad is enabled.
    assert pairs.requires_grad


def test_betti_loss_is_finite_scalar():
    """Test (2): forward returns a 0-d finite tensor."""
    torch.manual_seed(1)
    loss_fn = BettiLoss(persistence_threshold=0.1)
    x = torch.randn(32, 16)
    loss = loss_fn(x, target_betti=(1.0, 0.0, 0.0))
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert loss.item() >= 0.0


def test_betti_loss_backward_nonzero_grad():
    """Test (3): loss.backward() propagates a non-zero gradient to x."""
    torch.manual_seed(2)
    loss_fn = BettiLoss(persistence_threshold=0.1)
    x = torch.randn(24, 8, requires_grad=True)
    loss = loss_fn(x, target_betti=(1.0, 0.0, 0.0))
    loss.backward()
    assert x.grad is not None
    # On a generic random cloud at least *some* point should receive a non-
    # zero gradient -- a fully-zero grad would mean the surrogate has no
    # signal at all, which would be a regression.
    assert x.grad.abs().sum().item() > 0.0


def test_target_beta0_one_lower_bounds_loss_on_gaussian():
    """Test (4): the (1, 0, 0) target gives a finite, *positive* loss on a
    random Gaussian point cloud (the surrogate over-counts components on
    random data, so the loss is bounded below by the over-count squared).
    This is the "non-trivial" sanity check from the design doc.
    """
    torch.manual_seed(3)
    loss_fn = BettiLoss(persistence_threshold=0.05)
    x = torch.randn(48, 12)
    loss = loss_fn(x, target_betti=(1.0, 0.0, 0.0))
    # Loss must be > 0 (the surrogate counts > 1 component on a noisy cloud).
    assert loss.item() > 0.0, loss.item()


def test_two_cluster_higher_loss_than_single_blob():
    """Test (5): a 2-cluster cloud has larger MSE to (β₀=1, 0, 0) than a
    single-blob cloud. This is the central correctness check for the
    connectivity proxy — moving from one blob to two should *increase*
    the distance from the connected-target.
    """
    torch.manual_seed(4)
    loss_fn = BettiLoss(persistence_threshold=0.1)
    blob = torch.randn(48, 4) * 0.1
    cluster_a = torch.randn(24, 4) * 0.1 - 5.0
    cluster_b = torch.randn(24, 4) * 0.1 + 5.0
    two = torch.cat([cluster_a, cluster_b], dim=0)
    loss_blob = loss_fn(blob, target_betti=(1.0, 0.0, 0.0)).item()
    loss_two = loss_fn(two, target_betti=(1.0, 0.0, 0.0)).item()
    assert loss_two > loss_blob, (loss_two, loss_blob)


def test_max_pts_cap_is_honoured():
    """Test (6): max_pts limits the pairwise matrix, no shape blow-up."""
    torch.manual_seed(5)
    x = torch.randn(512, 8)
    pairs = differentiable_persistence(x, max_pts=32)
    expected_k = 32 * 31 // 2 - 1
    assert pairs.shape == (expected_k, 2), pairs.shape


def test_estimate_betti_returns_three_channel_tensor():
    """Test (7): API symmetry — estimate_betti returns (3,) with grad."""
    torch.manual_seed(6)
    loss_fn = BettiLoss(persistence_threshold=0.1)
    x = torch.randn(20, 8, requires_grad=True)
    est = loss_fn.estimate_betti(x)
    assert est.shape == (3,), est.shape
    assert est.requires_grad
    # β₀ >= 1 by construction (the +1 floor in the estimator).
    assert est[0].item() >= 1.0 - 1e-4


def test_handles_degenerate_single_point():
    """Test (8): N < 2 returns a zero-tensor without raising."""
    x = torch.randn(1, 8)
    pairs = differentiable_persistence(x)
    assert pairs.shape == (1, 2)
    # The loss must still be finite (no NaN) on the degenerate case.
    loss_fn = BettiLoss()
    loss = loss_fn(x, target_betti=(1.0, 0.0, 0.0))
    assert torch.isfinite(loss).item()


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
