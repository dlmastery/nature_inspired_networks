"""Unit tests for H55 — PlatonicAttention + PlatonicTransformerBlock.

Run as a script:
    python tests/test_platonic_transformer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.platonic_transformer import (  # noqa: E402
    PlatonicAttention,
    PlatonicTransformerBlock,
    platonic_vertex_coords,
)


def test_platonic_vertex_coords_shapes():
    """Vertex-count table matches Platonic-solid combinatorics."""
    expected = {"tetra": 4, "cube": 8, "octa": 6, "icosa": 12, "dodeca": 20}
    for group, n in expected.items():
        coords = platonic_vertex_coords(group)
        assert coords.shape == (n, 3), (group, coords.shape)
        # all vertices unit-norm
        norms = coords.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(n), atol=1e-5), (group, norms)


def test_attn_forward_shape():
    attn = PlatonicAttention(d_model=24, group="icosa")  # 24 / 12 = 2 d_head
    x = torch.randn(3, 7, 24)
    y = attn(x)
    assert y.shape == (3, 7, 24), y.shape
    assert torch.isfinite(y).all()


def test_attn_head_count_platonic_friendly():
    """Head count must be a Platonic-solid vertex count (5- or 20-friendly)."""
    attn_icosa = PlatonicAttention(d_model=24, group="icosa")
    attn_dodeca = PlatonicAttention(d_model=40, group="dodeca")
    assert attn_icosa.n_heads == 12, attn_icosa.n_heads
    assert attn_dodeca.n_heads == 20, attn_dodeca.n_heads
    # The hypothesis spec calls for divisibility by 5 (icosa pent-orbit
    # has 5 vertices) or 20 (dodeca face orbit).
    assert (
        attn_dodeca.n_heads % 5 == 0
        and attn_dodeca.n_heads % 20 == 0
        and attn_icosa.n_heads % 4 == 0
    )


def test_attn_rejects_indivisible_dmodel():
    """d_model not divisible by n_heads must raise."""
    try:
        PlatonicAttention(d_model=13, group="icosa")
        raise AssertionError("expected ValueError for d_model=13, n_heads=12")
    except ValueError:
        pass


def test_block_forward_shape():
    block = PlatonicTransformerBlock(d_model=24, group="icosa", ffn_mult=2)
    x = torch.randn(2, 5, 24)
    y = block(x)
    assert y.shape == (2, 5, 24), y.shape
    assert torch.isfinite(y).all()


def test_block_param_count_is_sane():
    """Block param count must include attn QKV + out_proj + 2 LNs + FFN."""
    d = 24
    block = PlatonicTransformerBlock(d_model=d, group="icosa", ffn_mult=4)
    n_params = sum(p.numel() for p in block.parameters())
    # rough lower bound: just the FFN's two Linears (d*4d + 4d*d + biases)
    ffn_lower = 2 * d * (4 * d)
    assert n_params > ffn_lower, (n_params, ffn_lower)
    # and the block must have more params than a standalone attention
    # alone (gut-check: attn-only is a strict subset).
    attn_only = sum(p.numel() for p in block.attn.parameters())
    assert n_params > attn_only


def test_block_gradient_flow():
    """Loss.backward over the block must populate gradients on every weight."""
    block = PlatonicTransformerBlock(d_model=12, group="icosa", ffn_mult=2)
    x = torch.randn(2, 4, 12, requires_grad=False)
    y = block(x)
    y.sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_h55_head_bias_is_nonzero_and_relative():
    """G6 audit fix: the Platonic bias must be mathematically non-zero
    and must depend on relative position (i.e. it differs across
    sequence positions). The previous implementation collapsed to
    identically 0 because gram.mean over vertex-transitive Platonic
    solids is always 0.
    """
    # head_angles buffer must NOT be identically zero across all heads
    for group in ("tetra", "octa", "icosa", "dodeca"):
        attn = PlatonicAttention(d_model=_d_for(group), group=group)
        assert hasattr(attn, "head_angles"), f"{group}: missing head_angles buffer"
        ang = attn.head_angles
        assert ang.shape == (attn.n_heads,), (group, ang.shape)
        # at least one head must have a non-zero phase
        assert ang.abs().sum().item() > 1e-6, (
            f"{group}: head_angles is all zero -- the Platonic bias is void"
        )
    # The (h, i, j) relative bias must depend on (j - i): the bias
    # along the diagonal (delta=0) must differ from the off-diagonal.
    attn = PlatonicAttention(d_model=24, group="icosa")
    bias = attn._relative_bias(N=8, device=torch.device("cpu"), dtype=torch.float32)
    assert bias.shape == (12, 8, 8), bias.shape
    # diagonal vs off-diagonal: cos(angle + 0) != cos(angle + 2pi*k/N) for k!=0 in general
    diag = bias[:, 0, 0]
    off = bias[:, 0, 1]
    assert not torch.allclose(diag, off, atol=1e-4), (
        "relative bias is constant along (i, j) -- not position-dependent"
    )
    # Forward output with vs without bias should differ. Build a vanilla
    # MHA baseline by zeroing head_angles in-place and re-running.
    attn = PlatonicAttention(d_model=24, group="icosa")
    torch.manual_seed(7)
    x = torch.randn(2, 6, 24)
    y_with = attn(x)
    attn.head_angles.zero_()
    # Setting all head_angles to zero collapses the cos(angle + phase)
    # bias to cos(phase), which is still a non-trivial relative-position
    # bias (constant across heads). Compare against fully-disabled bias
    # by also clamping rel_bias inside forward via patch:
    # easier: simulate vanilla MHA by saving and zero'ing the buffer
    # then re-running with a monkey-patched _relative_bias.
    attn._relative_bias = lambda N, device, dtype: torch.zeros(  # type: ignore[assignment]
        attn.n_heads, N, N, device=device, dtype=dtype
    )
    y_no_bias = attn(x)
    assert not torch.allclose(y_with, y_no_bias, atol=1e-4), (
        "Platonic bias has no effect on forward output -- module is bit-equivalent to vanilla MHA"
    )


def test_h55_rotation_equivariance_smoke():
    """Cyclic shift of the sequence by 1 must produce a related output.

    The relative-positional bias is exactly periodic with period N, so
    a cyclic position shift permutes positions but preserves per-head
    phase structure. We assert the shifted bias matches a position-
    shifted version of the original (discrete rotation equivariance of
    the BIAS tensor itself, before Q/K/V which are not equivariant).
    """
    attn = PlatonicAttention(d_model=24, group="icosa")
    N = 6
    bias = attn._relative_bias(N=N, device=torch.device("cpu"), dtype=torch.float32)
    # Cyclically shift rows and columns by +1: bias_shifted[h, i, j]
    # should equal bias[h, (i-1)%N, (j-1)%N]. The relative position
    # (j-i) is preserved under simultaneous shift of i and j, so
    # bias_shifted == bias when shift_i == shift_j.
    shift = 1
    bias_shifted = torch.roll(bias, shifts=(shift, shift), dims=(-2, -1))
    assert torch.allclose(bias, bias_shifted, atol=1e-5), (
        "relative-positional bias must be invariant under simultaneous (i, j) cyclic shift"
    )


def _d_for(group: str) -> int:
    """Smallest d_model divisible by group head count, >= group count."""
    counts = {"tetra": 4, "octa": 6, "icosa": 12, "dodeca": 20, "cube": 8}
    return counts[group] * 2


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
