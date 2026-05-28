"""Unit tests for H53 — IcosaUnfold + IcosaToPlane.

Run as a script:
    python tests/test_icosa_unfold.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.icosa_unfold import (  # noqa: E402
    IcosaToPlane,
    IcosaUnfold,
    icosa_unfold_permutation,
)


def test_permutation_length_and_bijective():
    perm = icosa_unfold_permutation()
    assert perm.shape == (12,), perm.shape
    assert perm.dtype == torch.long
    # bijective: each vertex index 0..11 appears exactly once
    sorted_perm, _ = torch.sort(perm)
    assert torch.equal(sorted_perm, torch.arange(12)), (
        f"permutation is not bijective: {perm.tolist()}"
    )


def test_permutation_deterministic():
    """Calling the function twice must return the same permutation."""
    p1 = icosa_unfold_permutation()
    p2 = icosa_unfold_permutation()
    assert torch.equal(p1, p2), "permutation must be deterministic"


def test_unfold_forward_shape():
    unfold = IcosaUnfold()
    x = torch.randn(2, 12, 7)
    y = unfold(x)
    assert y.shape == (2, 4, 3, 7), y.shape
    assert torch.isfinite(y).all()


def test_unfold_fold_roundtrip_is_identity():
    """unfold then fold must reproduce the input exactly (pure permutation)."""
    unfold = IcosaUnfold()
    x = torch.randn(3, 12, 5)
    y = unfold(x)
    x_back = unfold.fold(y)
    assert torch.allclose(x_back, x), "fold(unfold(x)) must equal x exactly"


def test_unfold_rejects_wrong_vertex_count():
    unfold = IcosaUnfold()
    bad = torch.randn(2, 11, 7)
    try:
        unfold(bad)
        raise AssertionError("expected assertion error on non-12 vertex count")
    except AssertionError as exc:
        if "expected assertion error" in str(exc):
            raise


def test_icosa_to_plane_forward_shape():
    layer = IcosaToPlane(in_channels=4, out_channels=8, kernel_size=3)
    x = torch.randn(2, 12, 4)
    y = layer(x)
    assert y.shape == (2, 12, 8), y.shape
    assert torch.isfinite(y).all()


def test_icosa_to_plane_param_count_matches_conv2d():
    """Parameter count must equal a standalone Conv2d(in=4, out=8, k=3)."""
    layer = IcosaToPlane(in_channels=4, out_channels=8, kernel_size=3, bias=True)
    n_params = sum(p.numel() for p in layer.parameters())
    ref = torch.nn.Conv2d(4, 8, 3, padding=1, bias=True)
    n_ref = sum(p.numel() for p in ref.parameters())
    assert n_params == n_ref, f"got {n_params}, expected {n_ref}"


def test_h53_unfold_bijective_no_geometric_claim():
    """G6 audit fix: the unfold is a bijection but does NOT make a
    great-circle / latitude-band adjacency claim. Asserts the
    bijection contract AND that the docstring no longer advertises
    the false geometric claim (post-audit honesty).
    """
    perm = icosa_unfold_permutation()
    # Bijection: permutation length 12, every index 0..11 exactly once.
    assert perm.shape == (12,), perm.shape
    sorted_perm, _ = torch.sort(perm)
    assert torch.equal(sorted_perm, torch.arange(12)), (
        f"permutation must be bijective: {perm.tolist()}"
    )
    # Audit-honesty: docstrings on the module + the function + the
    # IcosaUnfold class must no longer claim "great-circle adjacency"
    # or "latitude bands" (or equivalent geometric-preservation claims).
    import nature_inspired_networks.icosa_unfold as mod

    forbidden_phrases = (
        "great-circle traversal order",
        "approximately preserves vertex adjacency",
        "great-circle traversal is continuous",
    )
    haystacks = (
        mod.__doc__ or "",
        icosa_unfold_permutation.__doc__ or "",
        IcosaUnfold.__doc__ or "",
    )
    for hay in haystacks:
        for phrase in forbidden_phrases:
            assert phrase not in hay, (
                f"docstring still advertises false geometric claim: {phrase!r}"
            )


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
