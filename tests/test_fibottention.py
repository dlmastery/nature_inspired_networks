"""Unit tests for H32 — Fibonacci Dilation Attention (Fibottention).

Asserts:

* :func:`fibonacci_dilations` returns the canonical Fibonacci sequence
  ``[1, 2, 3, 5, 8, 13, ...]`` starting at 1, 2.
* :func:`wythoff_pattern` produces a ``(H, N, N)`` boolean mask whose
  per-head density matches ``1 / dilation`` to within a small lattice
  rounding tolerance.
* Non-overlap: for distinct heads ``h1 != h2`` with coprime Fibonacci
  dilations, the off-diagonal mask intersection is sparse (LCM
  governed).
* :class:`Fibottention` forward preserves the ``(B, N, D)`` shape.
* Dense fallback (``dilations=(1, 1, ..., 1)``) reproduces the
  numerical output of a plain dense multi-head attention with the same
  weights.
* Mask cache is reused across forward calls at the same sequence
  length (regression: rebuilding the mask each forward would be a
  perf bug).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.fibottention import (  # noqa: E402
    Fibottention,
    fibonacci_dilations,
    wythoff_pattern,
)


# ---------------------------------------------------------------------------
# fibonacci_dilations
# ---------------------------------------------------------------------------
def test_fibonacci_dilations_default_sequence():
    assert fibonacci_dilations(6) == [1, 2, 3, 5, 8, 13]
    assert fibonacci_dilations(8) == [1, 2, 3, 5, 8, 13, 21, 34]
    assert fibonacci_dilations(10) == [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


def test_fibonacci_dilations_recurrence_holds():
    seq = fibonacci_dilations(10)
    for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
        assert c == a + b, (a, b, c)


def test_fibonacci_dilations_rejects_zero():
    try:
        fibonacci_dilations(0)
    except ValueError:
        return
    raise AssertionError("expected ValueError on n_heads=0")


# ---------------------------------------------------------------------------
# wythoff_pattern
# ---------------------------------------------------------------------------
def test_wythoff_pattern_shape_and_dtype():
    mask = wythoff_pattern(seq_len=32, n_heads=4, dilations=[1, 2, 3, 5])
    assert mask.shape == (4, 32, 32)
    assert mask.dtype == torch.bool


def test_wythoff_pattern_self_position_always_attended():
    """Every head must attend to itself at the diagonal — this is the
    self-loop the Fibottention paper enforces."""
    mask = wythoff_pattern(seq_len=24, n_heads=5, dilations=[1, 2, 3, 5, 8])
    diag = torch.eye(24, dtype=torch.bool)
    for h in range(5):
        assert torch.all(mask[h] & diag == diag), (
            f"head {h} missing self-attention on diagonal"
        )


def test_wythoff_pattern_density_inverse_proportional_to_dilation():
    """Per-head density should be approximately ``1 / dilation`` to within
    boundary effects at sequence length L."""
    L = 256
    dilations = [1, 2, 3, 5, 8, 13]
    mask = wythoff_pattern(seq_len=L, n_heads=len(dilations), dilations=dilations)
    for h, d in enumerate(dilations):
        density = mask[h].float().mean().item()
        expected = 1.0 / d
        # generous tolerance: ±20% relative or ±0.01 absolute
        assert abs(density - expected) <= max(0.2 * expected, 0.01), (
            f"head {h} (dilation {d}): density {density:.4f} vs expected {expected:.4f}"
        )


def test_wythoff_pattern_non_overlap_for_coprime_dilations():
    """For two heads with coprime dilations a and b, the intersection
    of their non-diagonal masks attends to offsets that are multiples
    of lcm(a, b) = a*b — so the *off-diagonal* density is at most
    ~1/(a*b), strictly smaller than either individual density.
    """
    L = 256
    # Pick two coprime Fibonacci dilations: 5 and 8 (gcd = 1).
    mask = wythoff_pattern(seq_len=L, n_heads=2, dilations=[5, 8])
    diag = torch.eye(L, dtype=torch.bool)
    off1 = mask[0] & ~diag
    off2 = mask[1] & ~diag
    intersect = off1 & off2
    d_intersect = intersect.float().mean().item()
    d1 = off1.float().mean().item()
    d2 = off2.float().mean().item()
    # Intersection must be strictly less than either individual density
    # (the Wythoff non-overlap property).
    assert d_intersect < d1, (d_intersect, d1)
    assert d_intersect < d2, (d_intersect, d2)
    # And approximately 1 / lcm(5, 8) = 1/40.
    expected = 1.0 / (5 * 8)
    assert abs(d_intersect - expected) < 0.02, (d_intersect, expected)


def test_wythoff_pattern_dilations_length_validated():
    try:
        wythoff_pattern(seq_len=16, n_heads=4, dilations=[1, 2])
    except ValueError:
        return
    raise AssertionError("expected ValueError on dilations length mismatch")


# ---------------------------------------------------------------------------
# Fibottention forward
# ---------------------------------------------------------------------------
def test_fibottention_forward_shape_preserved():
    torch.manual_seed(0)
    mha = Fibottention(dim=48, n_heads=6)
    x = torch.randn(2, 32, 48)
    y = mha(x)
    assert y.shape == (2, 32, 48)
    assert torch.isfinite(y).all()


def test_fibottention_rejects_indivisible_dim():
    try:
        Fibottention(dim=33, n_heads=6)
    except ValueError:
        return
    raise AssertionError("expected ValueError when dim % n_heads != 0")


def test_fibottention_dense_fallback_matches_standard_mha_numerically():
    """When every head has dilation 1, the mask is all-True and softmax
    over q@k/sqrt(d) must equal an explicit dense computation using the
    same QKV / output weights.
    """
    torch.manual_seed(42)
    dim, n_heads, N, B = 16, 4, 8, 2
    mha = Fibottention(dim=dim, n_heads=n_heads, dilations=[1, 1, 1, 1])
    assert mha.is_dense_fallback
    x = torch.randn(B, N, dim)
    out_fib = mha(x)

    # Reference: dense MHA using the same weights.
    qkv = mha.qkv(x).reshape(B, N, 3, n_heads, dim // n_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) / math.sqrt(dim // n_heads)
    attn = attn.softmax(-1)
    ref = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, dim)
    ref = mha.proj(ref)
    assert torch.allclose(out_fib, ref, atol=1e-6), (out_fib - ref).abs().max()


def test_fibottention_sparse_differs_from_dense():
    """A Fibonacci-dilated forward must produce a numerically different
    output from the dense fallback at the same weights — otherwise the
    masking is not being applied."""
    torch.manual_seed(7)
    dim, n_heads, N, B = 24, 6, 64, 2
    sparse = Fibottention(dim=dim, n_heads=n_heads)
    dense = Fibottention(dim=dim, n_heads=n_heads, dilations=[1] * n_heads)
    # Copy sparse's weights into dense so the only thing differing is the mask.
    dense.qkv.load_state_dict(sparse.qkv.state_dict())
    dense.proj.load_state_dict(sparse.proj.state_dict())
    x = torch.randn(B, N, dim)
    y_sparse = sparse(x)
    y_dense = dense(x)
    assert not torch.allclose(y_sparse, y_dense, atol=1e-4)


def test_fibottention_mask_cache_reused_at_same_seqlen():
    mha = Fibottention(dim=16, n_heads=4)
    x = torch.randn(1, 20, 16)
    _ = mha(x)
    mask1 = mha._mask_cache
    cached_len1 = mha._cached_seq_len
    _ = mha(x)
    assert mha._cached_seq_len == cached_len1
    assert mha._mask_cache.data_ptr() == mask1.data_ptr()
    # Different N forces a rebuild.
    _ = mha(torch.randn(1, 25, 16))
    assert mha._cached_seq_len == 25


if __name__ == "__main__":
    import inspect

    fns = [
        v for k, v in globals().items()
        if k.startswith("test_") and inspect.isfunction(v)
    ]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
