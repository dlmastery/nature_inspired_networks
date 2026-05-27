"""Unit tests for H83 — CollapseGatedAttention (learnable-temperature attention)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.collapse_attention import CollapseGatedAttention  # noqa: E402


def test_forward_shape():
    """Shape: (B, N, D) -> (B, N, D)."""
    attn = CollapseGatedAttention(embed_dim=32, num_heads=4)
    x = torch.randn(2, 7, 32)
    y = attn(x)
    assert y.shape == (2, 7, 32), y.shape


def test_low_tau_sharper_than_high_tau():
    """Mechanism: low temperature => peaked attention (higher row-max) than
    high temperature, on the SAME projections / input."""
    torch.manual_seed(0)
    attn = CollapseGatedAttention(embed_dim=16, num_heads=2, tau_init=1.0)
    x = torch.randn(2, 6, 16)
    # Peaked: tau small.
    with torch.no_grad():
        attn.tau_raw.fill_(torch.log(torch.expm1(torch.tensor(0.2))))
    _, a_low = attn(x, need_weights=True)
    # Diffuse: tau large.
    with torch.no_grad():
        attn.tau_raw.fill_(torch.log(torch.expm1(torch.tensor(5.0))))
    _, a_high = attn(x, need_weights=True)
    # Peaked attention has higher max probability per query row.
    assert a_low.amax(dim=-1).mean().item() > a_high.amax(dim=-1).mean().item()


def test_tau_stays_positive_via_softplus():
    """Mechanism: even with a very negative raw parameter, tau > 0."""
    attn = CollapseGatedAttention(embed_dim=16, num_heads=2, tau_init=1.0)
    with torch.no_grad():
        attn.tau_raw.fill_(-100.0)
    assert attn.tau.item() > 0.0
    # And a forward pass with such a tiny tau must not produce NaN/inf.
    y = attn(torch.randn(1, 4, 16))
    assert torch.isfinite(y).all()


def test_gradient_flows_to_tau():
    """Mechanism/regression: the learnable temperature receives gradient."""
    attn = CollapseGatedAttention(embed_dim=16, num_heads=2, tau_init=1.0)
    x = torch.randn(2, 5, 16)
    attn(x).sum().backward()
    assert attn.tau_raw.grad is not None
    assert torch.isfinite(attn.tau_raw.grad).all()
    assert attn.tau_raw.grad.abs().item() > 0.0


def test_attention_rows_sum_to_one():
    """Mechanism: attention is a valid distribution over keys (rows sum to 1)."""
    attn = CollapseGatedAttention(embed_dim=16, num_heads=2)
    x = torch.randn(2, 6, 16)
    _, a = attn(x, need_weights=True)
    row_sums = a.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_collapse_sharpens_and_preserves_simplex():
    """Edge case: collapse=1 puts (almost) all mass on the argmax key while
    rows still sum to 1; collapse=0 leaves the softmax unchanged."""
    attn = CollapseGatedAttention(embed_dim=16, num_heads=2, tau_init=1.0)
    x = torch.randn(2, 6, 16)
    _, a0 = attn(x, collapse=0.0, need_weights=True)
    _, a1 = attn(x, collapse=1.0, need_weights=True)
    # Full collapse => row max is 1.0 (one-hot) and rows still sum to 1.
    assert torch.allclose(a1.amax(dim=-1), torch.ones_like(a1.amax(dim=-1)), atol=1e-5)
    assert torch.allclose(a1.sum(dim=-1), torch.ones_like(a1.sum(dim=-1)), atol=1e-5)
    # collapse=0 is strictly more diffuse than collapse=1.
    assert a0.amax(dim=-1).mean().item() < a1.amax(dim=-1).mean().item()


def test_additive_mask_blocks_positions():
    """Edge case: an additive -inf mask zeroes attention to masked keys
    (e.g. a causal mask)."""
    attn = CollapseGatedAttention(embed_dim=16, num_heads=2)
    N = 5
    x = torch.randn(1, N, 16)
    # Strict causal mask: position i may not attend to j > i.
    mask = torch.triu(torch.full((N, N), float("-inf")), diagonal=1)
    _, a = attn(x, attn_mask=mask, need_weights=True)
    # Upper-triangular (future) entries must be ~0.
    fut = a[..., 0, 1:]  # query 0 attending to keys 1..N-1 (all future)
    assert fut.abs().max().item() < 1e-6


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
