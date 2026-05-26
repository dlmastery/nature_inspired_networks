"""H50 - unit tests for the full-hybrid composition.

Run with:
    python ideas/50_full_sacred_hybrid/tests.py

Output must end with "All N tests passed." or fail loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(HERE))

from implementation import (  # noqa: E402
    build_full_hybrid_block,
    full_hybrid_flags,
    idea_signature,
)
from nature_inspired_networks.blocks import NaturePriorFlags  # noqa: E402


def test_idea_signature_marks_disproved():
    sig = idea_signature()
    assert sig["hypothesis_id"] == "H50"
    assert sig["falsifier_status"] == "disproved"
    # The empirical numbers must match the archived FINDINGS.md row.
    assert abs(sig["observed_top1"] - 0.7324) < 1e-4
    assert abs(sig["observed_composite"] - 0.6966) < 1e-4
    assert abs(sig["delta_composite"] + 0.1169) < 1e-3


def test_full_hybrid_flags_all_six_on():
    """All six Boolean priors must be ON; group_reduce defaults to 'max'."""
    f = full_hybrid_flags()
    for name in ("hex", "group", "fractal", "toroidal",
                 "cymatic_init", "golden_modulate"):
        assert getattr(f, name) is True, name
    assert f.group_reduce == "max"


def test_full_hybrid_flags_tag_contains_all_priors():
    """The tag must reflect every active prior (used by the runner)."""
    f = full_hybrid_flags()
    tag = f.tag()
    for name in ("hex", "group", "fractal", "toroidal",
                 "cymatic_init", "golden_modulate"):
        assert name in tag, (name, tag)
    assert "(avg)" not in tag  # default uses max-pool


def test_full_hybrid_flags_mean_variant_tag_has_avg_marker():
    """H58 variant: with group_reduce='mean' the tag must end (avg)."""
    f = full_hybrid_flags(group_reduce="mean")
    assert f.group_reduce == "mean"
    assert f.tag().endswith("(avg)")


def test_block_forward_keeps_HW_at_stride1():
    blk = build_full_hybrid_block(16, 16, stride=1)
    x = torch.randn(2, 16, 8, 8)
    y = blk(x)
    assert y.shape == (2, 16, 8, 8), y.shape


def test_block_forward_downsamples_at_stride2():
    blk = build_full_hybrid_block(16, 32, stride=2)
    x = torch.randn(2, 16, 8, 8)
    y = blk(x)
    assert y.shape == (2, 32, 4, 4), y.shape


def test_block_param_count_above_vanilla():
    """Sanity: the full hybrid adds depth + fractal recursion, so the
    parameter count must exceed a vanilla NaturePriorBlock (all flags off)."""
    from nature_inspired_networks.blocks import NaturePriorBlock
    vanilla = NaturePriorBlock(
        16, 32, stride=2,
        flags=NaturePriorFlags(False, False, False, False, False, False),
    )
    full = build_full_hybrid_block(16, 32, stride=2)
    n_vanilla = sum(p.numel() for p in vanilla.parameters())
    n_full = sum(p.numel() for p in full.parameters())
    assert n_full > n_vanilla, (n_vanilla, n_full)


def test_no_nan_no_inf_on_forward():
    """Even in the falsified composition, the forward pass must be finite."""
    blk = build_full_hybrid_block(8, 8, stride=1)
    x = torch.randn(2, 8, 8, 8)
    y = blk(x)
    assert torch.isfinite(y).all().item()


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
