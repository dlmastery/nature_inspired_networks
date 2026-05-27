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


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
