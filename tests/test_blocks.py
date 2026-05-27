"""Smoke tests for NaturePriorBlock and NaturePriorNet."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.blocks import NaturePriorFlags, NaturePriorBlock  # noqa: E402
from nature_inspired_networks.models import ResNet20, NaturePriorConfig, NaturePriorNet  # noqa: E402


def _all_flag_combos():
    """Each of the 6 priors on alone + all-on + all-off."""
    yield NaturePriorFlags(False, False, False, False, False, False)
    for name in ("hex", "group", "fractal", "toroidal", "cymatic_init",
                 "golden_modulate"):
        f = NaturePriorFlags(False, False, False, False, False, False)
        setattr(f, name, True)
        yield f
    yield NaturePriorFlags(True, True, True, True, True, True)


def test_block_forward_shape_keeps_h_w():
    for f in _all_flag_combos():
        blk = NaturePriorBlock(16, 16, stride=1, flags=f)
        x = torch.randn(2, 16, 8, 8)
        y = blk(x)
        assert y.shape == (2, 16, 8, 8), (f.tag(), y.shape)


def test_block_forward_shape_downsamples_on_stride2():
    for f in _all_flag_combos():
        blk = NaturePriorBlock(16, 32, stride=2, flags=f)
        x = torch.randn(2, 16, 8, 8)
        y = blk(x)
        assert y.shape == (2, 32, 4, 4), (f.tag(), y.shape)


def test_resnet20_forward():
    m = ResNet20(num_classes=10)
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_NaturePriorNet_forward_each_channel_mode():
    for mode in ("fib", "phi", "linear"):
        cfg = NaturePriorConfig(num_classes=10, channel_mode=mode,
                              flags=NaturePriorFlags())
        m = NaturePriorNet(cfg)
        y = m(torch.randn(2, 3, 32, 32))
        assert y.shape == (2, 10), mode


def test_NaturePriorNet_stagewise_features_has_4_stages():
    cfg = NaturePriorConfig(num_classes=10)
    m = NaturePriorNet(cfg)
    feats = m.stagewise_features(torch.randn(2, 3, 32, 32))
    assert len(feats) == 4  # stem + 3 stages


def test_resnet20_param_count_in_expected_band():
    m = ResNet20(num_classes=10)
    n = sum(p.numel() for p in m.parameters())
    # Canonical ResNet-20 is ~272 k params
    assert 250_000 < n < 290_000, n


def test_h58_group_reduce_mean_forward_shape():
    """H58 regression test inside the block.

    Setting group_reduce='mean' in NaturePriorFlags must not change the
    output shape vs. group_reduce='max'; only the values change.
    """
    flags_max = NaturePriorFlags(False, True, False, False, False, False,
                                  group_reduce="max")
    flags_mean = NaturePriorFlags(False, True, False, False, False, False,
                                   group_reduce="mean")
    torch.manual_seed(0)
    blk_max = NaturePriorBlock(16, 32, stride=2, flags=flags_max)
    torch.manual_seed(0)
    blk_mean = NaturePriorBlock(16, 32, stride=2, flags=flags_mean)
    x = torch.randn(2, 16, 8, 8)
    y_max = blk_max(x); y_mean = blk_mean(x)
    assert y_max.shape == y_mean.shape == (2, 32, 4, 4)
    # The values differ because the orbit reduction differs
    assert not torch.allclose(y_max, y_mean, atol=1e-4)


def test_flag_tag_reflects_group_reduce():
    """H58: the tag string must surface (avg) when group_reduce='mean'."""
    f_max = NaturePriorFlags(False, True, False, False, False, False,
                              group_reduce="max")
    f_mean = NaturePriorFlags(False, True, False, False, False, False,
                               group_reduce="mean")
    assert "(avg)" not in f_max.tag()
    assert f_mean.tag().endswith("(avg)")


def test_build_model_accepts_natureprior_casing_and_legacy_alias():
    """Regression test for the rename bug: build_model used to call
    name.lower() then compare to the mixed-case 'NaturePrior' literal,
    which never matched. This test catches that class of typo."""
    from nature_inspired_networks.models import build_model
    # Each of these should construct without error
    for n in ("NaturePrior", "natureprior", "nature_prior",
              "ResNet20", "resnet20", "sacredgeo"):
        m = build_model(n, num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        assert y.shape == (2, 10), f"model {n!r} produced {y.shape}"


def test_h05v2_fractal_phi_shrink_default_unchanged():
    """H05.v2 (a): with fractal_phi_shrink=False (the default), the
    fractal-on block's _FractalPath must keep the legacy uniform-width
    behaviour byte-for-byte — no extra projection module, every recursive
    sub-conv carries the same channel count as the parent.
    """
    from nature_inspired_networks.blocks import _FractalPath
    flags = NaturePriorFlags(False, False, True, False, False, False)
    assert flags.fractal_phi_shrink is False
    fp = _FractalPath(16, 32, stride=1, depth=2, flags=flags)
    # legacy path: no 1x1 projection should be created at the merge.
    assert fp.b_project is None
    # b1 keeps the parent c_out (=32); the recursive b2.path also takes 32 → 32.
    assert fp.b1.conv.weight.shape[0] == 32
    # forward returns expected shape
    x = torch.randn(2, 16, 8, 8)
    y = fp(x)
    assert y.shape == (2, 32, 8, 8)


def test_h05v2_fractal_phi_shrink_activates_when_set():
    """H05.v2 (b): with fractal_phi_shrink=True, the recursive sub-branch
    width shrinks by 1/phi per level (floor 8) and a 1x1 projection is
    added to merge back to c_out. The block's outward shape contract is
    unchanged.
    """
    from nature_inspired_networks.blocks import _FractalPath
    from nature_inspired_networks.priors import PHI
    flags = NaturePriorFlags(False, False, True, False, False, False)
    flags.fractal_phi_shrink = True
    fp = _FractalPath(16, 64, stride=1, depth=2, flags=flags)
    expected_mid = max(8, int(64 / PHI))
    # branch B shrunk to ~64/phi ≈ 39 (or 39 floored at 8)
    assert fp.b1.conv.weight.shape[0] == expected_mid
    # 1x1 projection back to c_out must exist
    assert fp.b_project is not None
    x = torch.randn(2, 16, 8, 8)
    y = fp(x)
    assert y.shape == (2, 64, 8, 8)
    # End-to-end block forward also works with the flag flipped
    blk = NaturePriorBlock(16, 32, stride=2, flags=flags)
    y2 = blk(torch.randn(2, 16, 8, 8))
    assert y2.shape == (2, 32, 4, 4)


def test_h21v2_hex_kernel_radius_default_unchanged():
    """H21.v2 (a): with hex_kernel_radius=1 (default), the hex-on block
    keeps a 3x3 7-tap mask (180-sym), matching the legacy smoke row.
    """
    flags = NaturePriorFlags(True, False, False, False, False, False)
    assert flags.hex_kernel_radius == 1
    blk = NaturePriorBlock(16, 16, stride=1, flags=flags)
    # _GenericConv.conv is the HexConv2d; its mask must be 3x3 with 7 ones.
    hex_conv = blk.conv1.conv
    assert hex_conv.mask.shape == (3, 3)
    assert hex_conv.mask.sum().item() == 7
    y = blk(torch.randn(2, 16, 8, 8))
    assert y.shape == (2, 16, 8, 8)


def test_h21v2_hex_kernel_radius_2_activates_19_tap_mask():
    """H21.v2 (b): hex_kernel_radius=2 selects k=5 (19-tap radius-2 hex)
    for true 6-fold isotropy, without changing the block's outward
    spatial shape contract.
    """
    flags = NaturePriorFlags(True, False, False, False, False, False)
    flags.hex_kernel_radius = 2
    blk = NaturePriorBlock(16, 16, stride=1, flags=flags)
    hex_conv = blk.conv1.conv
    assert hex_conv.mask.shape == (5, 5)
    assert hex_conv.mask.sum().item() == 19
    y = blk(torch.randn(2, 16, 8, 8))
    assert y.shape == (2, 16, 8, 8)
    # Stride=2 downsample path should also be wired through correctly.
    blk2 = NaturePriorBlock(16, 32, stride=2, flags=flags)
    y2 = blk2(torch.randn(2, 16, 8, 8))
    assert y2.shape == (2, 32, 4, 4)


def test_h35v2_cymatic_init_orthonormalize_default_unchanged():
    """H35.v2 (a): with cymatic_init_orthonormalize=False (default), the
    block-level cymatic init reproduces the legacy weights bit-identical.
    """
    flags = NaturePriorFlags(False, False, False, False, True, False)
    assert flags.cymatic_init_orthonormalize is False
    torch.manual_seed(0)
    blk_default = NaturePriorBlock(8, 8, stride=1, flags=flags)
    # Manually construct what the legacy init should produce on the same shape
    from nature_inspired_networks.priors import cymatic_init_
    ref = torch.nn.Conv2d(8, 8, 3, padding=1, bias=False)
    cymatic_init_(ref)
    assert torch.allclose(blk_default.conv1.conv.weight, ref.weight)


def test_h35v2_cymatic_init_orthonormalize_activates_when_set():
    """H35.v2 (b): with cymatic_init_orthonormalize=True, the new
    Gram-Schmidt + band-(2, 5) variant is used. The output channels of
    the basis must be pairwise near-orthogonal (Gram matrix ≈ scaled I).
    """
    flags = NaturePriorFlags(False, False, False, False, True, False)
    flags.cymatic_init_orthonormalize = True
    blk = NaturePriorBlock(8, 8, stride=1, flags=flags)
    w = blk.conv1.conv.weight  # (8, 8, 3, 3)
    # For a fixed input channel, the 8 output-channel filters were built
    # from Gram-Schmidt orthonormal modes (up to a per-(o,i) sign). Their
    # pairwise inner products should be near zero relative to their norm.
    for i in range(w.shape[1]):
        flat = w[:, i].reshape(w.shape[0], -1)  # (out_c, k*k)
        flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
        gram = flat @ flat.t()
        off_diag = gram - torch.eye(gram.shape[0])
        # Most pairs should be near-orthogonal; sign flips don't perturb
        # orthonormality. Allow a generous tolerance for numerical QR.
        assert off_diag.abs().max().item() < 0.5, off_diag.abs().max().item()
    # The forward must still produce the contract shape.
    y = blk(torch.randn(2, 8, 8, 8))
    assert y.shape == (2, 8, 8, 8)


def test_build_model_rejects_unknown():
    from nature_inspired_networks.models import build_model
    try:
        build_model("totally_not_a_model", num_classes=10)
        raise AssertionError("should have raised")
    except ValueError as exc:
        assert "totally_not_a_model" in str(exc)


def test_flag_field_iteration_distinguishes_string_field():
    """Regression test for the compute_topology.py bug.

    NaturePriorFlags has a mix of Boolean fields and non-Boolean fields
    (group_reduce: str, hex_kernel_radius: int). Iterating
    ``__dataclass_fields__`` uniformly and setting every field to a bool
    will assign False or True to those, breaking GroupConv2d/HexConv2d
    asserts. The fix is to iterate only the Boolean field names and pass
    the non-Boolean fields explicitly.
    """
    fields = NaturePriorFlags().__dataclass_fields__
    bool_names = [n for n, f in fields.items() if f.type is bool or f.type == "bool"]
    # Hard-coded canonical Boolean list — if someone adds a new flag they
    # must update this list AND the reconstruction logic in
    # scripts/compute_topology.py.
    expected_bool = {"hex", "group", "fractal", "toroidal",
                     "cymatic_init", "golden_modulate",
                     "fractal_phi_shrink", "cymatic_init_orthonormalize"}
    non_bool = {"group_reduce", "hex_kernel_radius"}
    assert non_bool.isdisjoint(expected_bool)
    # Either f.type was resolved as the type object (bool) or the literal
    # string "bool"; in either case the bool_names list must match
    # expected_bool, or the field-set minus the non-bool fields must.
    assert (
        set(bool_names) == expected_bool
        or set(fields.keys()) - non_bool == expected_bool
    )


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
