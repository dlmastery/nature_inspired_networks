"""Unit tests for H25 — dodecahedron_vertices + DodecaLatentProjector.

Run as a script:
    python tests/test_dodeca_latent.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.dodeca_latent import (  # noqa: E402
    DodecaLatentProjector,
    dodecahedron_vertices,
    vertex_distance_loss,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_dodecahedron_has_20_unique_vertices():
    V = dodecahedron_vertices()
    assert V.shape == (20, 3), V.shape
    # all vertices must be unique (deduplicate by rounding to 6 d.p.)
    rounded = {tuple(v.round(decimals=6).tolist()) for v in V}
    assert len(rounded) == 20, len(rounded)
    # all on a common sphere of radius sqrt(3)
    radii = V.norm(dim=-1)
    expected = math.sqrt(3.0)
    assert (radii - expected).abs().max().item() < 1e-5, radii


def test_vertices_unit_norm_after_normalization():
    """Regression: when `normalize_vertices=True`, every vertex must
    have unit Euclidean norm (within FP tolerance).
    """
    proj = DodecaLatentProjector(in_dim=8, out_dim=10, normalize_vertices=True)
    norms = proj.vertices.norm(dim=-1)
    assert (norms - 1.0).abs().max().item() < 1e-5, norms
    # the buffer is registered (moves with .to(device))
    assert "vertices" in dict(proj.named_buffers())


def test_projector_forward_shapes():
    proj = DodecaLatentProjector(in_dim=16, out_dim=5)
    x = torch.randn(4, 16)
    z_3d, soft, out = proj(x)
    assert z_3d.shape == (4, 3), z_3d.shape
    assert soft.shape == (4, 20), soft.shape
    assert out.shape == (4, 5), out.shape
    # softmax over the 20-vertex axis must sum to ~1 per sample
    assert (soft.sum(dim=-1) - 1.0).abs().max().item() < 1e-5


def test_vertex_distance_loss_zero_on_target_vertex():
    """If z is exactly the target dodeca vertex coordinate, the MSE
    must be zero.
    """
    V = dodecahedron_vertices()
    idx = torch.tensor([0, 5, 13, 19], dtype=torch.long)
    z = V[idx].clone()
    loss = vertex_distance_loss(z, idx)
    assert loss.item() < 1e-10, loss.item()

    # And a perturbed z must give strictly positive loss.
    z2 = z + 0.1
    loss2 = vertex_distance_loss(z2, idx)
    assert loss2.item() > 0.0


def test_projector_rejects_bad_input_shape():
    proj = DodecaLatentProjector(in_dim=8, out_dim=4)
    try:
        proj(torch.randn(2, 6))
        raise AssertionError("expected AssertionError for wrong in_dim")
    except AssertionError as exc:
        if "expected AssertionError" in str(exc):
            raise


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
