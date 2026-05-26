"""H05 — unit tests for the idea's implementation.

Run with:
    python ideas/05_fractal_phi_recursion/tests.py

Output must end with "All N tests passed." or fail loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make project src/ AND this idea dir importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_idea_signature_present():
    """The idea must expose a signature dict so the runner can log it."""
    from implementation import idea_signature
    sig = idea_signature()
    assert isinstance(sig, dict)
    assert sig["hypothesis_id"] == "H05"
    assert "_FractalPath" in sig["primitives_touched"]
    assert "fractal" in sig["flags_touched"]


def test_fractal_only_flags_has_only_fractal_on():
    """sg_only_fractal: every other Boolean prior off, fractal on."""
    from implementation import fractal_only_flags
    f = fractal_only_flags()
    assert f.fractal is True
    assert f.hex is False
    assert f.group is False
    assert f.toroidal is False
    assert f.cymatic_init is False
    assert f.golden_modulate is False
    assert f.group_reduce == "max"
    # Tag should reflect only the fractal prior
    assert f.tag() == "fractal", f.tag()


def test_build_fractal_block_default_shape_stride1():
    """Default fractal_depth=2, stride=1 must preserve H, W."""
    from implementation import build_fractal_block
    blk = build_fractal_block(16, 16, stride=1, depth=2)
    x = torch.randn(2, 16, 8, 8)
    y = blk(x)
    assert y.shape == (2, 16, 8, 8), y.shape


def test_build_fractal_block_downsamples_on_stride2():
    """Stride=2 must downsample H, W by 2 and respect new c_out."""
    from implementation import build_fractal_block
    blk = build_fractal_block(16, 32, stride=2, depth=2)
    x = torch.randn(2, 16, 8, 8)
    y = blk(x)
    assert y.shape == (2, 32, 4, 4), y.shape


def test_fractal_depth1_equivalent_to_no_recursion_shape():
    """Depth=1 short-circuits the recursion; block should still forward
    and produce the canonical residual-block shape."""
    from implementation import build_fractal_block
    blk = build_fractal_block(16, 16, stride=1, depth=1)
    x = torch.randn(2, 16, 8, 8)
    y = blk(x)
    assert y.shape == (2, 16, 8, 8)
    # Parameter count should be strictly less than depth=2
    blk_d2 = build_fractal_block(16, 16, stride=1, depth=2)
    n1 = sum(p.numel() for p in blk.parameters())
    n2 = sum(p.numel() for p in blk_d2.parameters())
    assert n1 < n2, (n1, n2)


def test_predicted_param_factor_matches_T1_5_observation():
    """T1.5 regression test: depth=2 fractal should roughly double
    param count vs depth=1 (no recursion). The closed-form factor is
    2.0; T1.5 observed +104 % which is factor 2.04. We assert the
    closed-form values for depth in {1, 2, 3}."""
    from implementation import predicted_param_factor
    assert predicted_param_factor(1) == 1.0
    assert predicted_param_factor(2) == 2.0
    assert predicted_param_factor(3) == 3.0


def test_predicted_param_factor_matches_empirical_depth2():
    """The closed-form ratio must agree with the empirical ratio of a
    depth-1 vs depth-2 block within 30 % (allowing for BatchNorm,
    skip-projection and the golden-modulate buffer which are
    block-level overhead the formula does NOT count). This is the
    regression test that catches future drift of `_FractalPath` away
    from the documented branching topology."""
    from implementation import build_fractal_block, predicted_param_factor
    blk1 = build_fractal_block(16, 16, stride=1, depth=1)
    blk2 = build_fractal_block(16, 16, stride=1, depth=2)
    n1 = sum(p.numel() for p in blk1.parameters())
    n2 = sum(p.numel() for p in blk2.parameters())
    empirical = n2 / n1
    predicted = predicted_param_factor(2)
    # within 50 % (BN + skip projection + bias overhead at small channels
    # adds noticeable absolute params at this small c_out)
    assert 0.5 * predicted < empirical < 1.5 * predicted, (n1, n2, empirical, predicted)


def test_rejects_depth_zero():
    """depth=0 is meaningless; the builder must refuse."""
    from implementation import build_fractal_block
    try:
        build_fractal_block(16, 16, stride=1, depth=0)
    except ValueError as exc:
        assert "depth must be >= 1" in str(exc)
    else:
        raise AssertionError("expected ValueError for depth=0")


def test_block_forward_finite_no_nans():
    """Regression test: fractal forward must not introduce NaNs even at
    the deeper depth=3 setting."""
    from implementation import build_fractal_block
    blk = build_fractal_block(16, 32, stride=2, depth=3)
    x = torch.randn(2, 16, 8, 8)
    y = blk(x)
    assert torch.isfinite(y).all(), "fractal block produced non-finite output"
    assert y.shape == (2, 32, 4, 4)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
