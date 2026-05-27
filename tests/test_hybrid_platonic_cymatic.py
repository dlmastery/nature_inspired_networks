"""Unit tests for H63 — Platonic Auxiliary Cymatic Teacher."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_platonic_cymatic import (  # noqa: E402
    PlatonicCymaticTeacher,
    cka_distillation_loss,
    cymatic_teacher_target,
)


def test_student_forward_shapes():
    model = PlatonicCymaticTeacher(in_channels=3, width=16, n_blocks=2, n_classes=10)
    x = torch.randn(2, 3, 16, 16)
    logits, act = model(x)
    assert logits.shape == (2, 10)
    assert act.shape == (2, 16, 16, 16)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(act).all()


def test_cymatic_teacher_target_shape_matches_request():
    t = cymatic_teacher_target(batch_size=2, channels=8, height=10, width=12)
    assert t.shape == (2, 8, 10, 12)
    assert torch.isfinite(t).all()


def test_cymatic_teacher_target_is_deterministic():
    """Identical seed → identical target."""
    a = cymatic_teacher_target(2, 4, 5, 5, seed=0)
    b = cymatic_teacher_target(2, 4, 5, 5, seed=0)
    assert torch.allclose(a, b)


def test_cka_distillation_loss_zero_for_identical():
    """CKA distance for identical inputs must be (numerically) zero."""
    x = torch.randn(8, 32)
    loss = cka_distillation_loss(x, x)
    assert loss.abs().item() < 1e-4


def test_cka_distillation_loss_positive_for_orthogonal():
    """Random independent inputs → distance is strictly positive."""
    torch.manual_seed(0)
    x = torch.randn(8, 32)
    y = torch.randn(8, 32)
    loss = cka_distillation_loss(x, y)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_teacher_for_matches_student_act_shape():
    model = PlatonicCymaticTeacher(in_channels=3, width=16, n_blocks=2, n_classes=10)
    x = torch.randn(2, 3, 16, 16)
    _, act = model(x)
    target = model.teacher_for(act)
    assert target.shape == act.shape


def test_distillation_loss_grad_flows_to_student_only():
    """Teacher target is treated as a stop-gradient constant."""
    student = torch.randn(8, 32, requires_grad=True)
    teacher = torch.randn(8, 32, requires_grad=True)
    loss = cka_distillation_loss(student, teacher)
    loss.backward()
    assert student.grad is not None
    # Teacher is detached inside cka_distillation_loss.
    assert teacher.grad is None


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
