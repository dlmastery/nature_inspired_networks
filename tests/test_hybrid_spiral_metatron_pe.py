"""Unit tests for H73 — Golden Spiral × Metatron Positional Encoding."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_spiral_metatron_pe import (  # noqa: E402
    SpiralMetatronPE,
    fallback_golden_spiral_pe,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_fallback_spiral_pe_shape_and_finite():
    pe = fallback_golden_spiral_pe(16, 8)
    assert pe.shape == (16, 8)
    assert torch.isfinite(pe).all()


def test_fallback_spiral_pe_first_row_is_zero_radius():
    """k=0 → r=√1=1; first row is r*cos(0)=1, r*sin(0)=0."""
    pe = fallback_golden_spiral_pe(2, 4)
    assert abs(pe[0, 0].item() - 1.0) < 1e-6
    assert abs(pe[0, 1].item()) < 1e-6


def test_fallback_spiral_pe_rejects_odd_dim():
    try:
        fallback_golden_spiral_pe(8, 5)
        raise AssertionError("expected ValueError for odd d")
    except ValueError:
        pass


def test_spiral_metatron_pe_forward_shape():
    pe = SpiralMetatronPE(d_model=32, n_max=16, d_spiral=8, d_metatron_node=4)
    positions = torch.arange(10)
    out = pe(positions)
    assert out.shape == (10, 32)
    assert torch.isfinite(out).all()


def test_spiral_metatron_pe_rejects_out_of_range():
    pe = SpiralMetatronPE(d_model=16, n_max=8)
    try:
        pe(torch.tensor([0, 5, 10]))
        raise AssertionError("expected ValueError for index out of range")
    except ValueError:
        pass


def test_spiral_metatron_pe_gradients_reach_seed_and_proj():
    pe = SpiralMetatronPE(d_model=16, n_max=8, d_spiral=4, d_metatron_node=2)
    out = pe(torch.arange(8))
    loss = out.sum()
    loss.backward()
    assert pe.metatron_seed.grad is not None
    assert pe.proj.weight.grad is not None


def test_spiral_metatron_pe_rejects_odd_d_model():
    try:
        SpiralMetatronPE(d_model=15, n_max=8)
        raise AssertionError("expected ValueError for odd d_model")
    except ValueError:
        pass


def test_spiral_pe_buffer_is_precomputed():
    pe = SpiralMetatronPE(d_model=8, n_max=12, d_spiral=4)
    assert pe.spiral_pe.shape == (12, 4)
    # Spiral PE is buffered (deterministic), changes across rows.
    assert not torch.allclose(pe.spiral_pe[0], pe.spiral_pe[1])


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
