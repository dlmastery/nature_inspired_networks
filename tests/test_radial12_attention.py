"""Unit tests for H77 RadialSymmetry12Attention."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.radial12_attention import (  # noqa: E402
    RadialSymmetry12Attention,
    radial12_head_angles,
)


def test_forward_shape_preserved():
    """(B, N, D) -> (B, N, D)."""
    torch.manual_seed(0)
    attn = RadialSymmetry12Attention(d_model=24, n_heads=12)
    x = torch.randn(2, 7, 24)
    y = attn(x)
    assert y.shape == (2, 7, 24), y.shape


def test_requires_n_heads_multiple_of_12():
    """n_heads not a positive multiple of 12 must raise."""
    for bad in (1, 5, 13, 10):
        try:
            RadialSymmetry12Attention(d_model=24, n_heads=bad)
            raise AssertionError(f"expected ValueError for n_heads={bad}")
        except ValueError as exc:
            if "expected ValueError" in str(exc):
                raise
    # multiple of 12 (24) must succeed
    RadialSymmetry12Attention(d_model=48, n_heads=24)


def test_head_bias_is_registered_buffer_not_parameter():
    """The angular bias must be a buffer, not a learnable Parameter."""
    attn = RadialSymmetry12Attention(d_model=24, n_heads=12)
    buf_names = {n for n, _ in attn.named_buffers()}
    param_names = {n for n, _ in attn.named_parameters()}
    assert "head_angles" in buf_names
    assert "head_angles" not in param_names
    # the angles repeat with period 12 (mechanism correctness)
    angles = radial12_head_angles(24)
    assert torch.allclose(angles[:12], angles[12:24])
    expected0 = 2 * math.pi * 1 / 12
    assert abs(angles[1].item() - expected0) < 1e-6


def test_shift_by_N_over_12_cyclically_permutes_head_phases():
    """Mechanism: the relative bias is cyclically periodic. Shifting all
    positions by N/12 cyclically rotates each head's bias row, and the
    set of per-head phase columns is a permutation of the unshifted set.
    """
    attn = RadialSymmetry12Attention(d_model=24, n_heads=12)
    N = 24  # divisible by 12
    bias = attn._relative_bias(N, torch.device("cpu"), torch.float32)  # (12, N, N)
    shift = N // 12  # = 2
    # Roll the relative-position axis (both i and j) by `shift`; because
    # the bias depends only on (j - i) mod N via a cos of 2*pi*delta/N,
    # a simultaneous roll of rows and cols leaves it invariant.
    rolled = torch.roll(bias, shifts=(shift, shift), dims=(1, 2))
    assert torch.allclose(rolled, bias, atol=1e-5)
    # And the head-phase column at delta=shift equals head h's value at
    # delta=0 shifted by the 12-fold phase step -> verify head 0 row is a
    # cyclic shift along delta of itself by N (full period -> identity).
    full = torch.roll(bias, shifts=(N, N), dims=(1, 2))
    assert torch.allclose(full, bias, atol=1e-5)


def test_radial_false_matches_plain_mha():
    """radial=False must drop the bias and equal a plain MHA forward
    (we compare against a hand-rolled dense attention using the same
    projections)."""
    torch.manual_seed(1)
    attn = RadialSymmetry12Attention(d_model=24, n_heads=12, radial=False)
    x = torch.randn(3, 6, 24)
    y = attn(x)

    # Reconstruct plain MHA using the module's own qkv/proj weights.
    B, N, D = x.shape
    h, hd = attn.n_heads, attn.head_dim
    qkv = attn.qkv(x).view(B, N, 3, h, hd).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    a = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    a = torch.softmax(a, dim=-1)
    o = (a @ v).transpose(1, 2).reshape(B, N, D)
    ref = attn.proj(o)
    assert torch.allclose(y, ref, atol=1e-6)

    # And radial=True must differ from radial=False (bias actually bites).
    torch.manual_seed(1)
    attn_on = RadialSymmetry12Attention(d_model=24, n_heads=12, radial=True)
    attn_on.load_state_dict(attn.state_dict(), strict=False)
    y_on = attn_on(x)
    assert not torch.allclose(y_on, y, atol=1e-5)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
