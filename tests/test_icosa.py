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


def test_h24_rotations_enter_forward_pass():
    """Mechanism pin: the 60 icosa rotations must ACTUALLY influence the
    forward pass — they should change the conv input, not just sit as
    an unused buffer.

    Construction: pick a non-identity rotation ``R`` from the group;
    build an input ``x`` whose channel-triples are NOT R-invariant
    (i.e., ``R · x_triple != x_triple``); compare ``conv(x)`` to
    ``conv(R·x)``. If the 60 rotations are wired into the forward pass
    via mean-pool over the orbit, the two outputs CAN be approximately
    equal (the operator is rotation-equivariant up to channel-triple
    grouping). What we must verify here is that *swapping the input
    triples for their rotated copies does not change the output past
    1e-5 when the orbit is mean-pooled* — i.e., the rotations do
    enter, and the operator behaves like a rotation-equivariant pool.

    Conversely: if we DISABLE rotations (zero them out except identity),
    the output for ``x`` vs ``R·x`` should differ — which proves the
    rotations are load-bearing.
    """
    torch.manual_seed(0)
    conv = IcosaConv1d(in_channels=3, kernel_size=3, padding=1)
    # The first non-identity rotation in the buffered group.
    R = conv.rotations.clone()  # (60, 3, 3)
    # Find a non-identity element.
    I3 = torch.eye(3)
    R_nonid = None
    for g in range(60):
        if (R[g] - I3).abs().max() > 1e-3:
            R_nonid = R[g]  # (3, 3)
            break
    assert R_nonid is not None, "expected at least one non-identity rotation"

    # Input with non-trivial 3-D content per "triple" along L.
    x = torch.randn(2, 3, 16)
    # Apply R_nonid to the channel triple: x'[b, :, l] = R_nonid @ x[b, :, l].
    # The channel axis is exactly the 3-vector here (in_channels=3).
    x_rot = torch.einsum("ij,bjl->bil", R_nonid, x)
    # Sanity: R_nonid actually changes x.
    assert not torch.allclose(x_rot, x, atol=1e-5)

    y = conv(x)
    y_rot = conv(x_rot)
    # Mechanism pin #1: with the 60-rotation mean-pool wired in, the
    # operator is INVARIANT to rotating the input by an element of the
    # icosa group. y(x) ≈ y(R·x) within float tolerance.
    assert torch.allclose(y, y_rot, atol=1e-4), (y - y_rot).abs().max()

    # Mechanism pin #2: now break the equivariance by zeroing all
    # rotations except identity. y(x) and y(R·x) MUST then differ —
    # this confirms the rotations are load-bearing, not theatre.
    with torch.no_grad():
        # Find identity index.
        id_idx = None
        for g in range(60):
            if (conv.rotations[g] - I3).abs().max() < 1e-5:
                id_idx = g
                break
        assert id_idx is not None
        # Replace all non-identity rotations with the identity, so the
        # orbit mean is over 60 identical copies (no rotation in the
        # forward pass).
        for g in range(60):
            if g != id_idx:
                conv.rotations[g].copy_(I3)
    y2 = conv(x)
    y2_rot = conv(x_rot)
    # Now y2 vs y2_rot must differ — the rotations are no longer
    # smoothing them together.
    assert not torch.allclose(y2, y2_rot, atol=1e-3), (y2 - y2_rot).abs().max()


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
