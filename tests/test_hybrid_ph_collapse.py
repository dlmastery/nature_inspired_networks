"""Unit tests for H65 — Persistent-Homology Betti-Collapse Loss."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_ph_collapse import BettiCollapseLoss  # noqa: E402
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_forward_returns_scalar_loss_and_parts():
    loss_fn = BettiCollapseLoss(persistence_threshold=0.1)
    x = torch.randn(2, 4, 8, 8)
    loss, parts = loss_fn(x)
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"beta_0", "collapse", "spectral"}


def test_default_spectral_weight_is_one_over_phi():
    loss_fn = BettiCollapseLoss()
    assert abs(loss_fn.spectral_weight - 1.0 / PHI) < 1e-12


def test_collapse_is_zero_when_beta_0_matches_target():
    """If we directly hand a single-cluster point cloud, β₀ → 1 and
    the collapse term should be (near) zero."""
    loss_fn = BettiCollapseLoss(persistence_threshold=10.0)  # huge threshold → b0 = 1
    # All points clustered tightly → no persistent gaps → soft β₀ ≈ 1
    x = torch.zeros(16, 4) + 0.001 * torch.randn(16, 4)
    b0 = loss_fn.beta_0_estimate(x)
    # The soft count is in [0, ...]; with a huge threshold and tight cloud
    # the soft count is ~0 and b0 = 1 + ~0 → collapse term near zero.
    assert (b0 - 1.0).abs().item() < 0.5


def test_loss_grad_flows_back():
    loss_fn = BettiCollapseLoss(persistence_threshold=0.1)
    x = torch.randn(2, 4, 6, 6, requires_grad=True)
    loss, _ = loss_fn(x)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_spectral_term_zero_on_non_spatial_inputs():
    """When activations are 2-D, the spectral fallback returns 0
    (only 4-D maps have FFT-spectral meaning)."""
    loss_fn = BettiCollapseLoss(persistence_threshold=0.1)
    x = torch.randn(16, 8)
    loss, parts = loss_fn(x)
    # Non-spatial → spectral contribution is exactly zero.
    assert parts["spectral"] == 0.0


def test_target_beta_0_other_than_one_works():
    """The default target is 1 but the user may want a different target."""
    loss_fn = BettiCollapseLoss(persistence_threshold=0.1)
    x = torch.randn(2, 4, 6, 6)
    loss_a, parts_a = loss_fn(x, target_beta_0=1.0)
    loss_b, parts_b = loss_fn(x, target_beta_0=3.0)
    # Different targets ⇒ different collapse terms (so different total loss).
    assert parts_a["collapse"] != parts_b["collapse"]


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
