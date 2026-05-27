"""Unit tests for H24 — icosahedral rotations + IcosaProjection + IcosaConv1d.

Run as a script:
    python tests/test_icosa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.icosa import (  # noqa: E402
    IcosaConv1d,
    IcosaProjection,
    icosahedral_rotations,
)


def test_60_rotations_each_det_plus_one():
    R = icosahedral_rotations()
    assert R.shape == (60, 3, 3), R.shape
    # determinants must all be +1 (rotation group, no reflections).
    dets = torch.linalg.det(R.double())
    assert (dets - 1.0).abs().max().item() < 1e-5, dets.min().item()
    # orthogonality: R R^T = I per matrix
    I3 = torch.eye(3, dtype=torch.float64)
    err = (R.double() @ R.double().transpose(-1, -2) - I3).abs().max()
    assert err.item() < 1e-5, err.item()


def test_group_closure_under_multiplication():
    """Composition of any two group elements must already be in the
    group (within tolerance). This is the defining property of a group
    and the regression test that the BFS closure was complete.
    """
    R = icosahedral_rotations().double()  # (60, 3, 3)
    # Spot-check: every pair-product R_i @ R_j must equal some R_k.
    # Full 60×60 = 3600 checks is cheap.
    tol = 1e-5
    for i in range(60):
        for j in range(60):
            prod = R[i] @ R[j]
            # find closest element in group
            diffs = (R - prod.unsqueeze(0)).abs().reshape(60, -1).max(dim=1).values
            assert diffs.min().item() < tol, (i, j, diffs.min().item())


def test_icosa_projection_forward_shape():
    proj = IcosaProjection()
    pts = torch.randn(2, 7, 3)
    out = proj(pts)
    assert out.shape == (2, 7, 60), out.shape
    assert torch.isfinite(out).all()


def test_icosa_projection_rejects_bad_input():
    proj = IcosaProjection()
    try:
        proj(torch.randn(2, 7, 4))
        raise AssertionError("expected AssertionError for non-3-channel input")
    except AssertionError as exc:
        if "expected AssertionError" in str(exc):
            raise


def test_icosa_conv1d_forward_shape_default_fib_grouping():
    """Default hidden_sizes=[8,13,21] ⇒ out_channels=42 (sum of three
    consecutive Fibonacci numbers — H24 design intent).
    """
    conv = IcosaConv1d(in_channels=3, kernel_size=3, padding=1)
    assert conv.out_channels == 42
    x = torch.randn(2, 3, 16)
    y = conv(x)
    assert y.shape == (2, 42, 16), y.shape
    assert torch.isfinite(y).all()


def test_icosa_conv1d_rejects_mismatched_out_channels():
    """Regression: out_channels must equal sum(hidden_sizes) so the
    orbit permutation cleanly partitions the output.
    """
    try:
        IcosaConv1d(in_channels=3, out_channels=40, hidden_sizes=(8, 13, 21))
        raise AssertionError("expected AssertionError for mismatched out_channels")
    except AssertionError as exc:
        if "expected AssertionError" in str(exc):
            raise


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
