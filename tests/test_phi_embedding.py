"""Unit tests for H15 — phi-Initialised Embedding."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.phi_embedding import (  # noqa: E402
    GOLDEN_ANGLE_RAD,
    PhiEmbedding,
    golden_spiral_embedding_init_,
)


def test_phi_embedding_weight_shape_preserved():
    """The init must not change the weight tensor's shape."""
    emb = nn.Embedding(num_embeddings=200, embedding_dim=32)
    shape_before = emb.weight.shape
    golden_spiral_embedding_init_(emb, seed=0)
    assert emb.weight.shape == shape_before


def test_phi_embedding_deterministic_for_same_seed():
    """Two consecutive inits with the same seed must produce identical weights."""
    emb1 = nn.Embedding(num_embeddings=100, embedding_dim=16)
    emb2 = nn.Embedding(num_embeddings=100, embedding_dim=16)
    golden_spiral_embedding_init_(emb1, seed=42)
    golden_spiral_embedding_init_(emb2, seed=42)
    assert torch.allclose(emb1.weight, emb2.weight, atol=1e-7)


def test_phi_embedding_rank_two_lattice_structure():
    """The H15 init projects a 2-D lattice to d_model, so the resulting
    weight matrix is rank-2: the top two singular values dominate.

    This is the *correct* structural property per the design doc -- the
    lattice itself is 2-D and the random orthonormal Q only rotates it
    into d_model dims. The point of the prior is angular separation
    (preserved by Q), not full-rank coverage.
    """
    n, d = 64, 16
    emb = nn.Embedding(num_embeddings=n, embedding_dim=d)
    golden_spiral_embedding_init_(emb, seed=0)
    w = emb.weight.detach().double()
    # Per-row L2 norm has expected value ~1 (default scale heuristic)
    norms = w.norm(dim=1)
    assert 0.5 < norms.mean().item() < 1.5, norms.mean().item()
    # Rank-2 structure: top-2 singular values >> the third.
    s = torch.linalg.svdvals(w)
    assert s[0].item() > 10.0 * s[2].item(), s
    # And the minimum nearest-neighbour angular gap should be > 0 --
    # the golden-angle ensures no two tokens land on the SAME ray.
    w_norm = w / (norms.unsqueeze(1) + 1e-8)
    cos = w_norm @ w_norm.t()
    off_diag = cos.clone()
    off_diag.fill_diagonal_(0.0)
    # No two distinct rows should be identical -- max(off_diag) < 1.
    assert off_diag.max().item() < 1.0 - 1e-6, off_diag.max().item()


def test_phi_embedding_drop_in_forward_shape():
    """PhiEmbedding(x) must return the same shape as nn.Embedding(x).

    For input shape (B,) with B token IDs in [0, num_embeddings),
    output is (B, embedding_dim). For (B, T) it's (B, T, d).
    """
    n, d = 100, 32
    std_emb = nn.Embedding(num_embeddings=n, embedding_dim=d)
    phi_emb = PhiEmbedding(num_embeddings=n, embedding_dim=d, seed=0)
    ids_1d = torch.randint(0, n, (8,))
    ids_2d = torch.randint(0, n, (4, 12))
    assert std_emb(ids_1d).shape == phi_emb(ids_1d).shape == (8, d)
    assert std_emb(ids_2d).shape == phi_emb(ids_2d).shape == (4, 12, d)


def test_phi_embedding_init_changes_weights():
    """The init must actually replace the Xavier-default weight."""
    emb = nn.Embedding(num_embeddings=50, embedding_dim=8)
    w0 = emb.weight.detach().clone()
    golden_spiral_embedding_init_(emb, seed=0)
    assert not torch.allclose(w0, emb.weight)


def test_phi_embedding_rejects_too_small_d_model():
    """Embedding dim must be >= 2 (the lattice is 2-D)."""
    bad = nn.Embedding(num_embeddings=10, embedding_dim=1)
    try:
        golden_spiral_embedding_init_(bad)
        raise AssertionError("expected ValueError for embedding_dim < 2")
    except ValueError:
        pass


def test_golden_angle_value():
    """Golden angle should equal Vogel's 137.5077... degrees."""
    deg = GOLDEN_ANGLE_RAD * 180.0 / 3.141592653589793
    assert 137.5 < deg < 137.51, deg


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
