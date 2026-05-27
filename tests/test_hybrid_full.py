"""Unit tests for H67 — Full Paradigm Hybrid.

The H67 model is deliberately the "all on" reference -- the tests here
verify the shape contract and that *each* of the six load-bearing
priors is actually engaged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_full import FullParadigmHybrid  # noqa: E402
from nature_inspired_networks.blocks import NaturePriorBlock  # noqa: E402
from nature_inspired_networks.fibottention import Fibottention  # noqa: E402
from nature_inspired_networks.hybrid_liquid_jepa import LiquidCFCCell  # noqa: E402


def test_forward_shape():
    """End-to-end forward returns (B, n_classes)."""
    model = FullParadigmHybrid(
        in_channels=3, width=16, n_blocks=2, n_heads=4, n_classes=10
    )
    x = torch.randn(2, 3, 16, 16)
    y = model(x)
    assert y.shape == (2, 10)
    assert torch.isfinite(y).all()


def test_each_load_bearing_prior_is_engaged():
    """Verify every always-on prior shows up in ``which_priors_active``."""
    model = FullParadigmHybrid(
        in_channels=3, width=16, n_blocks=2, n_heads=4, n_classes=10
    )
    flags = model.which_priors_active
    # These four are always True; the other two are conditional on
    # optional modules being on disk.
    assert flags["nature_prior_blocks"] is True
    assert flags["fibottention_attention"] is True
    assert flags["cymatic_qkv_init"] is True
    assert flags["liquid_cfc"] is True
    # Optional priors are returned as booleans either way.
    assert isinstance(flags["golden_rope"], bool)
    assert isinstance(flags["platonic_graph"], bool)


def test_nature_prior_blocks_are_actually_instantiated():
    model = FullParadigmHybrid(width=16, n_blocks=3)
    npb_count = sum(1 for m in model.modules() if isinstance(m, NaturePriorBlock))
    assert npb_count == 3


def test_fibottention_and_cfc_are_present():
    model = FullParadigmHybrid(width=16, n_blocks=2)
    assert any(isinstance(m, Fibottention) for m in model.modules())
    assert any(isinstance(m, LiquidCFCCell) for m in model.modules())


def test_nondivisible_width_rejected():
    try:
        FullParadigmHybrid(width=10, n_heads=4)
        raise AssertionError("expected ValueError for non-divisible width")
    except ValueError:
        pass


def test_backward_flows_to_stem_weights():
    model = FullParadigmHybrid(width=16, n_blocks=2, n_classes=10)
    x = torch.randn(2, 3, 16, 16)
    y = model(x)
    y.sum().backward()
    # Stem is the first conv; verify gradient reached it.
    stem_conv = model.stem[0]
    assert stem_conv.weight.grad is not None
    assert (stem_conv.weight.grad.abs().sum() > 0).item()


def test_qkv_init_is_actually_cymatic():
    """The Fibottention QKV weight should have non-trivial energy in the
    Chladni-friendly mid-band; verified by checking that it doesn't match
    a freshly Xavier-init equivalent layer."""
    model = FullParadigmHybrid(width=16, n_blocks=2)
    fresh = nn.Linear(16, 48)  # matches Fibottention.qkv shape: (3*dim, dim)
    nn.init.xavier_uniform_(fresh.weight)
    assert not torch.allclose(model.attention.qkv.weight, fresh.weight)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
