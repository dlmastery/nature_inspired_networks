"""Unit tests for H70 — Cymatic Low-Data Curriculum."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_cymatic_curriculum import (  # noqa: E402
    CymaticCurriculumLoss,
    default_cymatic_loss,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_alpha_decays_monotonically_to_zero():
    loss = CymaticCurriculumLoss(warmup_epochs=5, k=8, n_modes=4)
    a = [loss.alpha(e) for e in range(7)]
    # α(0) == 1, α(5) == 0, α non-increasing, clamps at 0 thereafter.
    assert abs(a[0] - 1.0) < 1e-9
    assert abs(a[5] - 0.0) < 1e-9
    assert all(a[i] >= a[i + 1] for i in range(len(a) - 1))
    assert a[6] == 0.0


def test_forward_pure_ce_at_warmup_end():
    """When epoch >= warmup_epochs, forward must equal plain CE."""
    loss = CymaticCurriculumLoss(warmup_epochs=3, k=4, n_modes=2)
    feat = torch.randn(2, 3, 4, 4)
    logits = torch.randn(2, 5)
    targets = torch.tensor([0, 2])
    out = loss(feat, logits, targets, epoch=3)
    ref = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(out, ref, atol=1e-6)


def test_forward_pure_cymatic_at_start():
    """At epoch=0 with warmup>=1, α=1 so the CE term has weight 0."""
    loss = CymaticCurriculumLoss(warmup_epochs=4, k=4, n_modes=2)
    feat = torch.randn(2, 3, 4, 4)
    logits = torch.randn(2, 5)
    targets = torch.tensor([0, 1])
    # CE swap: replacing logits should NOT change loss at epoch=0
    other_logits = torch.randn(2, 5)
    out_a = loss(feat, logits, targets, epoch=0)
    out_b = loss(feat, other_logits, targets, epoch=0)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_default_cymatic_loss_shape_and_finite():
    feat = torch.randn(2, 3, 8, 8)
    modes = torch.randn(4, 8, 8)
    L = default_cymatic_loss(feat, modes)
    assert L.dim() == 0
    assert torch.isfinite(L)


def test_default_cymatic_loss_rejects_bad_shape():
    try:
        default_cymatic_loss(torch.randn(2, 3, 8), torch.randn(4, 8, 8))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        default_cymatic_loss(torch.randn(2, 3, 8, 8), torch.randn(4, 8))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_curriculum_forward_gradient_flows_to_logits_after_warmup():
    """After warmup, gradients must reach the logits via CE."""
    loss = CymaticCurriculumLoss(warmup_epochs=2, k=4, n_modes=2)
    feat = torch.randn(2, 3, 4, 4)
    logits = torch.randn(2, 5, requires_grad=True)
    targets = torch.tensor([1, 0])
    L = loss(feat, logits, targets, epoch=5)
    L.backward()
    assert logits.grad is not None


def test_target_modes_buffer_shape():
    loss = CymaticCurriculumLoss(warmup_epochs=3, k=8, n_modes=4)
    assert loss.target_modes.shape == (4, 8, 8)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
