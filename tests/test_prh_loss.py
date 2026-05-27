"""H49 unit tests — Platonic representation alignment loss (CKA + PRH)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.prh_loss import (  # noqa: E402
    PRHAlignmentLoss,
    cka_loss,
)


def test_cka_loss_in_unit_interval():
    """cka_loss returns a scalar in [0, 1] for random features."""
    torch.manual_seed(0)
    a = torch.randn(16, 32)
    b = torch.randn(16, 8)
    loss = cka_loss(a, b)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
    assert 0.0 <= loss.item() <= 1.0 + 1e-5


def test_cka_loss_zero_when_features_identical():
    """CKA(X, X) == 1 → cka_loss == 0 (up to numerical tolerance)."""
    torch.manual_seed(0)
    a = torch.randn(20, 16)
    loss = cka_loss(a, a)
    assert loss.item() < 1e-4, f"expected ~0; got {loss.item()}"
    # Also: CKA is invariant under orthogonal linear transform — rotating
    # B should not change CKA.
    Q, _ = torch.linalg.qr(torch.randn(16, 16))
    b = a @ Q
    loss_rot = cka_loss(a, b)
    assert loss_rot.item() < 1e-4, f"expected ~0 under rotation; got {loss_rot.item()}"


def test_cka_loss_symmetric():
    """cka_loss(X, Y) == cka_loss(Y, X)."""
    torch.manual_seed(0)
    a = torch.randn(32, 20)
    b = torch.randn(32, 8)
    l1 = cka_loss(a, b).item()
    l2 = cka_loss(b, a).item()
    assert abs(l1 - l2) < 1e-5, f"symmetry broken: {l1} vs {l2}"


def test_cka_loss_supports_different_feature_dims():
    """CKA via Gram matrices is dim-agnostic — D_a ≠ D_b must work."""
    a = torch.randn(24, 64)
    b = torch.randn(24, 7)
    loss = cka_loss(a, b)
    assert 0.0 <= loss.item() <= 1.0 + 1e-5


def test_prh_alignment_loss_forward_returns_scalar():
    """PRHAlignmentLoss returns a finite scalar."""
    torch.manual_seed(0)
    # 12-vertex Platonic target (dodeca), 4-D embedding
    target = torch.randn(1, 12)
    mod = PRHAlignmentLoss(target=target, feat_dim=64, project=True)
    feat = torch.randn(8, 64)
    loss = mod(feat)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
    assert 0.0 <= loss.item() <= 1.0 + 1e-5


def test_prh_alignment_loss_per_sample_target():
    """When target has shape (B, target_dim) matching feat, no batch_idx
    is needed."""
    torch.manual_seed(0)
    B, target_dim, feat_dim = 8, 12, 64
    target = torch.randn(B, target_dim)
    mod = PRHAlignmentLoss(target=target, feat_dim=feat_dim, project=True)
    feat = torch.randn(B, feat_dim)
    loss = mod(feat)
    assert loss.dim() == 0
    # batch_idx alternative path
    idx = torch.arange(B)
    loss_idx = mod(feat, batch_idx=idx)
    assert abs(loss.item() - loss_idx.item()) < 1e-5


def test_prh_alignment_loss_gradient_flow():
    """Gradients must flow through the projection head into features."""
    torch.manual_seed(0)
    target = torch.randn(1, 12)
    mod = PRHAlignmentLoss(target=target, feat_dim=64, project=True)
    feat = torch.randn(8, 64, requires_grad=True)
    loss = mod(feat)
    loss.backward()
    assert feat.grad is not None
    assert torch.isfinite(feat.grad).all()
    # The projection head should also have a gradient
    assert mod.proj.weight.grad is not None
    assert torch.isfinite(mod.proj.weight.grad).all()


def test_prh_alignment_loss_no_project_requires_dim_match():
    """project=False with mismatched dims must raise."""
    target = torch.randn(1, 12)
    raised = False
    try:
        PRHAlignmentLoss(target=target, feat_dim=64, project=False)
    except AssertionError:
        raised = True
    assert raised, "expected AssertionError when project=False and dims mismatch"
    # And the matched-dim path must succeed:
    mod = PRHAlignmentLoss(target=target, feat_dim=12, project=False)
    feat = torch.randn(8, 12)
    loss = mod(feat)
    assert loss.dim() == 0


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
