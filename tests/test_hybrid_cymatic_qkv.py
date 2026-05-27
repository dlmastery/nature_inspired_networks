"""Unit tests for H66 — Cymatic QKV Kernel."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_cymatic_qkv import (  # noqa: E402
    CymaticQKVKernel,
    cymatic_init_linear_,
)


def test_forward_shape():
    attn = CymaticQKVKernel(dim=32, n_heads=4)
    x = torch.randn(2, 16, 32)
    y = attn(x)
    assert y.shape == (2, 16, 32)
    assert torch.isfinite(y).all()


def test_qkv_projections_differ_after_init():
    """The three projections must NOT share weights (distinct seeds)."""
    attn = CymaticQKVKernel(dim=32, n_heads=4)
    assert not torch.allclose(attn.q_proj.weight, attn.k_proj.weight)
    assert not torch.allclose(attn.k_proj.weight, attn.v_proj.weight)
    assert not torch.allclose(attn.q_proj.weight, attn.v_proj.weight)


def test_cymatic_init_linear_changes_weights():
    """Calling ``cymatic_init_linear_`` must modify the linear weight."""
    linear = nn.Linear(36, 16)
    w_before = linear.weight.clone()
    cymatic_init_linear_(linear, band=(2, 5), orthonormalize=True, seed=7)
    assert not torch.allclose(w_before, linear.weight)


def test_cymatic_init_linear_zeros_bias():
    linear = nn.Linear(36, 16, bias=True)
    nn.init.uniform_(linear.bias, -1.0, 1.0)
    cymatic_init_linear_(linear, band=(2, 5), orthonormalize=True)
    assert torch.allclose(linear.bias, torch.zeros_like(linear.bias))


def test_cymatic_init_small_in_dim_falls_back_to_xavier():
    """For in_dim < 9 (k < 3) we fall back to Xavier instead of crashing."""
    linear = nn.Linear(4, 8)  # k = ceil(sqrt(4)) = 2 → fallback path
    cymatic_init_linear_(linear)
    # Xavier-init weights are bounded; mostly verify no crash + finite weights.
    assert torch.isfinite(linear.weight).all()


def test_nondivisible_heads_rejected():
    try:
        CymaticQKVKernel(dim=10, n_heads=4)
        raise AssertionError("expected ValueError for non-divisible heads")
    except ValueError:
        pass


def test_backward_pass_flows_to_qkv_weights():
    attn = CymaticQKVKernel(dim=16, n_heads=4)
    x = torch.randn(2, 8, 16, requires_grad=True)
    out = attn(x)
    out.sum().backward()
    assert attn.q_proj.weight.grad is not None
    assert attn.k_proj.weight.grad is not None
    assert attn.v_proj.weight.grad is not None
    assert (attn.q_proj.weight.grad.abs().sum() > 0).item()


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
