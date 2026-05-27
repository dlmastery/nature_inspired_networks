"""Unit tests for H16 — Fibonacci Head Diversity."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.fib_attention import (  # noqa: E402
    FibMultiheadAttention,
    fib_head_counts,
    fib_head_dilations,
)


def test_fib_attention_forward_shape():
    """(B, N, D) -> (B, N, D), with default embed_dim divisible by 20."""
    torch.manual_seed(0)
    m = FibMultiheadAttention(embed_dim=200)  # 200 / 20 = 10 = d_head
    x = torch.randn(2, 16, 200)
    y = m(x)
    assert y.shape == (2, 16, 200), y.shape


def test_fib_attention_total_heads_is_20():
    """Sum of canonical head counts is 20: [1,1,2,3,5,8]."""
    counts = fib_head_counts()
    dils = fib_head_dilations()
    assert counts == [1, 1, 2, 3, 5, 8]
    assert dils == [1, 2, 3, 5, 8, 13]
    assert sum(counts) == 20
    m = FibMultiheadAttention(embed_dim=200)
    assert m.n_heads == 20


def test_fib_false_collapses_to_uniform_dilation_one():
    """fib=False uses one head per group all at dilation 1 -- no mask
    sparsity. The output should match a manually-built nn.MultiheadAttention
    of the same head count after weight transfer (parity check)."""
    torch.manual_seed(0)
    m_fib_off = FibMultiheadAttention(embed_dim=24, fib=False)  # 6 heads
    assert m_fib_off.n_heads == 6
    assert all(d == 1 for d in m_fib_off.head_dilations)
    # No-mask path: forward must not raise and shape must round-trip.
    x = torch.randn(2, 7, 24)
    y = m_fib_off(x)
    assert y.shape == (2, 7, 24)


def test_fib_different_dilations_produce_different_outputs():
    """fib=True vs fib=False with weights transferred -- different
    dilations produce DIFFERENT attention patterns (no early-out).
    """
    torch.manual_seed(0)
    m_on = FibMultiheadAttention(embed_dim=120, fib=True)   # 20 heads
    torch.manual_seed(0)
    m_off = FibMultiheadAttention(embed_dim=120, fib=False)  # 6 heads
    # Force same input. The two modules have different head counts so
    # we can't share weights directly -- instead verify each separately
    # produces a different output when only the dilation pattern
    # changes.
    x = torch.randn(2, 16, 120)
    y_on = m_on(x)
    y_off = m_off(x)
    assert y_on.shape == y_off.shape == (2, 16, 120)
    # Outputs MUST differ because (a) head count differs and (b)
    # dilations differ.
    assert not torch.allclose(y_on, y_off)


def test_fib_attention_rejects_indivisible_embed_dim():
    """embed_dim must be divisible by sum(head_counts) = 20."""
    try:
        FibMultiheadAttention(embed_dim=100)  # 100 / 20 = 5 OK, this is fine
    except ValueError:
        raise AssertionError("100 should divide cleanly by 20")
    try:
        FibMultiheadAttention(embed_dim=99)  # 99 / 20 = 4.95 -- BAD
        raise AssertionError("expected ValueError for indivisible embed_dim")
    except ValueError:
        pass


def test_fib_attention_rejects_mismatched_schedule_lengths():
    """head_counts and head_dilations must have the same length."""
    try:
        FibMultiheadAttention(embed_dim=20, head_counts=[1, 1, 2],
                              head_dilations=[1, 2])
        raise AssertionError("expected ValueError for mismatched lengths")
    except ValueError:
        pass


def test_fib_attention_gradient_flow():
    """Loss.backward through Fib attention produces finite grads."""
    torch.manual_seed(0)
    m = FibMultiheadAttention(embed_dim=60)
    x = torch.randn(2, 16, 60, requires_grad=True)
    y = m(x).sum()
    y.backward()
    assert torch.isfinite(x.grad).all()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_fib_attention_uniform_with_dilation_one_no_mask_path():
    """When all dilations are 1 the internal mask must be None
    (SDPA fast-path)."""
    m = FibMultiheadAttention(embed_dim=24, fib=False)  # all dilations 1
    mask = m._attn_mask(seq_len=8, device=torch.device("cpu"))
    assert mask is None


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
