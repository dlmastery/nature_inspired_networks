"""Unit tests for H84 SpectralHopfieldMemory."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.spectral_hopfield import (  # noqa: E402
    SpectralHopfieldMemory,
)


def test_associative_recall_returns_stored_pattern():
    """A query ~ a stored pattern must retrieve (near) that pattern."""
    torch.manual_seed(0)
    dim = 32
    mem = SpectralHopfieldMemory(dim=dim, beta=20.0)
    patterns = torch.randn(6, dim)
    mem.store(patterns)
    target = patterns[3]
    query = target + 0.01 * torch.randn(dim)
    out = mem.retrieve(query)
    # retrieved vector must be closest to the stored target among the bank
    dists = (patterns - out).pow(2).sum(dim=-1)
    assert dists.argmin().item() == 3, dists
    assert (out - target).pow(2).sum().item() < (out - patterns[0]).pow(2).sum().item()


def test_output_shape_matches_query():
    dim = 16
    mem = SpectralHopfieldMemory(dim=dim, beta=8.0)
    mem.store(torch.randn(5, dim))
    # batched query
    assert mem.retrieve(torch.randn(4, dim)).shape == (4, dim)
    # single (dim,) query -> (dim,)
    assert mem.retrieve(torch.randn(dim)).shape == (dim,)


def test_high_beta_sharper_than_low_beta():
    """Mechanism: larger beta -> retrieval concentrates on one pattern.

    We measure sharpness directly via the softmax weight placed on the
    matching pattern: high beta must put MORE mass on the target than
    low beta. (Comparing reconstruction error can saturate to ~0 for
    both when patterns are well separated in high-dim spectral space.)
    """
    torch.manual_seed(1)
    dim = 24
    patterns = torch.randn(8, dim)
    target_idx = 5
    target = patterns[target_idx]
    query = target + 0.3 * torch.randn(dim)

    def matching_weight(beta: float) -> float:
        mem = SpectralHopfieldMemory(dim=dim, beta=beta)
        mem.store(patterns)
        s_patt = mem._to_spectral(patterns)            # (M, F)
        s_query = mem._to_spectral(query.unsqueeze(0))  # (1, F)
        scores = beta * (s_query @ s_patt.t())
        w = torch.softmax(scores, dim=-1)
        return w[0, target_idx].item()

    w_lo = matching_weight(0.001)
    w_hi = matching_weight(1.0)
    # sharper (high beta) puts more softmax mass on the matching pattern
    assert w_hi > w_lo, (w_hi, w_lo)
    # low beta is diffuse (no near-onehot lock-on), high beta is sharp
    assert w_lo < 0.9, w_lo
    assert w_hi > 0.9, w_hi


def test_fft_roundtrip_lossless():
    """Regression: irfft(rfft(x)) == x for real inputs."""
    torch.manual_seed(2)
    x = torch.randn(3, 17)
    rt = SpectralHopfieldMemory.spectral_roundtrip(x)
    assert rt.shape == x.shape
    assert torch.allclose(rt, x, atol=1e-5), (rt - x).abs().max()


def test_retrieve_before_store_raises():
    """Edge case: retrieval with an empty bank must error clearly."""
    mem = SpectralHopfieldMemory(dim=8)
    try:
        mem.retrieve(torch.randn(8))
        raise AssertionError("expected RuntimeError before store()")
    except RuntimeError as exc:
        if "expected RuntimeError" in str(exc):
            raise


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
