"""Unit tests for H71 — Icosahedral 3-D Rotary Position Embedding."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_icosa_rope import (  # noqa: E402
    IcosaRoPE3D,
    icosa_vertices_unit,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_icosa_vertices_count_and_unit_norm():
    v = icosa_vertices_unit()
    assert v.shape == (12, 3)
    norms = v.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(12), atol=1e-5)


def test_icosa_rope_forward_shape():
    rope = IcosaRoPE3D(head_dim=9)
    q = torch.randn(2, 4, 7, 9)
    k = torch.randn(2, 4, 7, 9)
    pos = torch.arange(7).float()
    q_rot, k_rot = rope(q, k, pos)
    assert q_rot.shape == (2, 4, 7, 9)
    assert k_rot.shape == (2, 4, 7, 9)
    assert torch.isfinite(q_rot).all()


def test_icosa_rope_position_zero_is_identity():
    """At position 0, angles are 0 — rotation must be the identity."""
    rope = IcosaRoPE3D(head_dim=6)
    q = torch.randn(2, 3, 1, 6)
    k = torch.randn(2, 3, 1, 6)
    pos = torch.zeros(1)
    q_rot, k_rot = rope(q, k, pos)
    assert torch.allclose(q_rot, q, atol=1e-5)
    assert torch.allclose(k_rot, k, atol=1e-5)


def test_icosa_rope_preserves_norm_per_triple():
    """A pure rotation preserves the L2 norm of each triple of channels."""
    rope = IcosaRoPE3D(head_dim=9)
    q = torch.randn(1, 1, 4, 9)
    k = torch.randn(1, 1, 4, 9)
    pos = torch.tensor([0.0, 1.5, 2.7, 5.0])
    q_rot, _ = rope(q, k, pos)
    # Reshape to triples and compare norms.
    q_tr = q.view(1, 1, 4, 3, 3).norm(dim=-1)
    q_rot_tr = q_rot.view(1, 1, 4, 3, 3).norm(dim=-1)
    assert torch.allclose(q_tr, q_rot_tr, atol=1e-4)


def test_icosa_rope_rejects_non_multiple_of_three_dim():
    try:
        IcosaRoPE3D(head_dim=8)
        raise AssertionError("expected ValueError for head_dim not divisible by 3")
    except ValueError:
        pass


def test_icosa_rope_rejects_shape_mismatch():
    rope = IcosaRoPE3D(head_dim=6)
    q = torch.randn(2, 3, 5, 6)
    k = torch.randn(2, 3, 6, 6)
    try:
        rope(q, k, torch.arange(5).float())
        raise AssertionError("expected ValueError for q/k shape mismatch")
    except ValueError:
        pass


def test_icosa_rope_axes_cycle_through_12():
    """For head_dim > 36, axes must cycle through the 12 icosa vertices."""
    rope = IcosaRoPE3D(head_dim=39)  # 13 triples
    # Triple 12 wraps to vertex 0.
    assert torch.allclose(rope.axes[12], rope.axes[0])
    # Triple 0 must equal vertex 0.
    v = icosa_vertices_unit()
    assert torch.allclose(rope.axes[0], v[0])


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
