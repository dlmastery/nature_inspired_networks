"""Unit tests for H37 - Pentagonal phi-Attention primitives.

Run as ``python tests/test_pentagonal_attention.py``; finishes with
``"All N tests passed."`` on success.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.pentagonal_attention import (  # noqa: E402
    PentagonalAttention,
    pentagonal_head_groups,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_h37_head_groups_partition_into_5():
    """``pentagonal_head_groups`` returns 5 groups of equal size whose
    union is exactly {0, ..., n_heads - 1}.
    """
    for n in (5, 10, 15, 20):
        groups = pentagonal_head_groups(n)
        assert len(groups) == 5
        sizes = {len(g) for g in groups}
        assert sizes == {n // 5}
        all_heads = sorted(h for g in groups for h in g)
        assert all_heads == list(range(n))


def test_h37_head_groups_rejects_non_multiple_of_5():
    """n_heads must be a positive multiple of 5."""
    for bad in (0, 1, 4, 6, 8, 12, -5):
        try:
            pentagonal_head_groups(bad)
            raise AssertionError(f"expected ValueError for n_heads={bad}")
        except ValueError:
            pass


def test_h37_attention_n_heads_must_divide_by_5():
    """The PentagonalAttention constructor rejects n_heads not divisible
    by 5.
    """
    # valid 10-head, d=20
    att = PentagonalAttention(d_model=20, n_heads=10)
    assert att.n_heads == 10
    # invalid
    for bad in (1, 2, 3, 4, 6, 8, 12, 16):
        try:
            PentagonalAttention(d_model=24, n_heads=bad)
            raise AssertionError(f"expected ValueError for n_heads={bad}")
        except ValueError:
            pass
    # d_model not divisible by n_heads
    try:
        PentagonalAttention(d_model=21, n_heads=10)
        raise AssertionError("expected ValueError for d_model not div by n_heads")
    except ValueError:
        pass


def test_h37_attention_forward_shape():
    """Forward preserves (B, L, d)."""
    torch.manual_seed(0)
    att = PentagonalAttention(d_model=20, n_heads=10)
    x = torch.randn(2, 7, 20)
    y = att(x)
    assert y.shape == (2, 7, 20)
    assert torch.isfinite(y).all()


def test_h37_head_bias_is_buffer_not_parameter():
    """The head bias is registered as a buffer and is not learnable. It
    moves with .to(device) but never appears in .parameters().
    """
    att = PentagonalAttention(d_model=20, n_heads=10)
    # Buffer present
    bias = att.head_bias
    assert isinstance(bias, torch.Tensor)
    assert not isinstance(bias, torch.nn.Parameter)
    # Not in parameters()
    param_ids = {id(p) for p in att.parameters()}
    assert id(bias) not in param_ids
    # It appears in state_dict() (because it is a registered buffer)
    sd = att.state_dict()
    assert any(k.endswith("head_bias") for k in sd.keys())


def test_h37_head_bias_repeats_with_period_5():
    """Heads h and h + 5 (when both exist) must have the same bias - the
    bias is purely a function of h % 5. This is the 5-fold symmetry
    constraint.
    """
    for n in (10, 15, 20):
        att = PentagonalAttention(d_model=n * 4, n_heads=n)
        bias = att.head_bias
        for h in range(n - 5):
            assert torch.allclose(bias[h], bias[h + 5]), (n, h, bias[h], bias[h + 5])
        # Within one period the 5 distinct phases are 2*pi*k/5
        expected = torch.tensor(
            [2 * math.pi * k / 5.0 for k in range(5)], dtype=bias.dtype
        )
        # bias values for heads {0, 1, 2, 3, 4} use round-robin so head h
        # is in group h%5; checking the first 5 directly.
        assert torch.allclose(bias[:5], expected, atol=1e-6)


def test_h37_rotational_symmetry_cyclic_shift():
    """Cyclically shifting the sequence by L / 5 must cyclically shift
    the output by the same amount, BECAUSE the per-head additive bias is
    a constant in (q, k) and is therefore translation-equivariant. This
    is the structural pentagonal-symmetry property.

    To remove non-symmetry artifacts from the linear projections (which
    operate per-position and are equivariant under shift trivially) we
    just check that shift then attention equals attention then shift.
    """
    torch.manual_seed(0)
    att = PentagonalAttention(d_model=20, n_heads=10)
    att.eval()
    L = 10  # divisible by 5 so L/5 is integer
    x = torch.randn(1, L, 20)
    shift = L // 5
    y_full = att(x)
    x_shifted = torch.roll(x, shifts=shift, dims=1)
    y_shifted = att(x_shifted)
    y_full_shifted = torch.roll(y_full, shifts=shift, dims=1)
    assert torch.allclose(y_shifted, y_full_shifted, atol=1e-5)


def test_h37_attention_bias_changes_logits():
    """The angular bias is non-zero for groups 1..4 (only group 0 has
    angle 0). Removing the bias must change the forward output, proving
    the bias is wired into the softmax inputs.
    """
    torch.manual_seed(0)
    att = PentagonalAttention(d_model=20, n_heads=10)
    att.eval()
    x = torch.randn(1, 6, 20)
    y_with = att(x)
    # zero out the buffer and forward again
    with torch.no_grad():
        saved = att.head_bias.clone()
        att.head_bias.zero_()
    y_without = att(x)
    # restore for cleanliness (no future test depends on this)
    with torch.no_grad():
        att.head_bias.copy_(saved)
    assert not torch.allclose(y_with, y_without, atol=1e-4)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
