"""Unit tests for H82 VoronoiSparseAttention."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.voronoi_attention import (  # noqa: E402
    VoronoiSparseAttention,
    voronoi_adjacency,
)


def test_adjacency_shape_symmetric_selfloops():
    N = 32
    A = voronoi_adjacency(N, seed=0)
    assert A.shape == (N, N), A.shape
    assert torch.allclose(A, A.t()), "adjacency must be symmetric"
    # self-loops present (diagonal all ones)
    assert torch.equal(A.diagonal(), torch.ones(N)), "self-loops missing"
    # binary mask
    assert set(A.unique().tolist()) <= {0.0, 1.0}


def test_sparsity_far_below_dense():
    """A Delaunay/kNN mask must be far sparser than the N^2 dense mask."""
    N = 64
    A = voronoi_adjacency(N, seed=0)
    edges = A.sum().item()
    # dense would be N*N; planar Delaunay degree ~6 -> well under 20% density
    assert edges < 0.2 * N * N, edges
    # but every node must attend to at least itself + neighbours
    assert (A.sum(dim=1) >= 2).all(), "every token needs >=1 neighbour"


def test_masked_attention_forward_shape_preserved():
    """(B, N, D) -> (B, N, D) with the sparse mask applied."""
    torch.manual_seed(0)
    N, D = 24, 16
    attn = VoronoiSparseAttention(d_model=D, n_tokens=N, n_heads=4, seed=0)
    x = torch.randn(3, N, D)
    y = attn(x)
    assert y.shape == (3, N, D), y.shape
    assert torch.isfinite(y).all(), "masked softmax produced non-finite output"
    # mismatched N must raise
    try:
        attn(torch.randn(3, N + 1, D))
        raise AssertionError("expected ValueError for wrong N")
    except ValueError as exc:
        if "expected ValueError" in str(exc):
            raise


def test_seed_determinism():
    """Same seed -> identical adjacency; different seed -> (usually) differs."""
    A0a = voronoi_adjacency(40, seed=7)
    A0b = voronoi_adjacency(40, seed=7)
    assert torch.equal(A0a, A0b), "same seed must give identical mask"
    A1 = voronoi_adjacency(40, seed=8)
    assert not torch.equal(A0a, A1), "different seed should give a different mask"


def test_knn_fallback_proxy_matches_sparsity():
    """Regression: the SciPy-free kNN proxy yields a valid symmetric mask
    with self-loops even when invoked directly via small n where Delaunay
    is skipped (n < 4 path)."""
    A = voronoi_adjacency(3, seed=0, knn=2)
    assert A.shape == (3, 3)
    assert torch.allclose(A, A.t())
    assert torch.equal(A.diagonal(), torch.ones(3))


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
