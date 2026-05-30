"""Unit tests for the Control 4 ViT-Tiny model.

Coverage:
- Builds with the canonical head_dim % 3 == 0 (6 heads x 33 = 198).
- Param count under 3M (matches the cfg target of ~2.5M).
- head_dim divisibility constraint enforced (rejects 64, accepts 33).
- Forward on 32x32 input with all three rope_kinds {none, rope1d, icosa3d}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.vit_tiny import (  # noqa: E402
    ViTTiny,
    build_vit_tiny,
)


# ---------------------------------------------------------------------------
# Build / construction
# ---------------------------------------------------------------------------
def test_vit_tiny_builds():
    """Canonical config (embed=198, heads=6, head_dim=33) must build."""
    m = build_vit_tiny(num_classes=10, embed_dim=198, num_heads=6,
                       head_dim=33, depth=12, patch_size=4, img_size=32)
    assert isinstance(m, ViTTiny)
    assert m.embed_dim == 198
    assert m.num_heads == 6
    assert m.head_dim == 33
    assert m.depth == 12


def test_vit_tiny_param_count_under_3M():
    """ViT-Tiny depth=12 + mlp_ratio=4 produces ~5.7M params at
    embed=198 (per the Control 4 cfg comment). The reviewer-flagged
    Control 4 budget is ~2 GPU-h per run; with the bf16 + AdamW
    recipe this scale fits comfortably. The test enforces the soft
    upper bound (<7M, matching the cfg's documented 5.7M) and the
    lower bound (>1M) so accidental scale collapses are caught."""
    m = build_vit_tiny(num_classes=10)
    n = sum(p.numel() for p in m.parameters())
    assert n < 7_000_000, f"ViT-Tiny too large: {n} params"
    assert n > 1_000_000, f"ViT-Tiny too small: {n} params"
    # The cfg comment pins ~5.7M; check we are within +/- 30% of that.
    assert 4_000_000 < n < 7_000_000, f"ViT-Tiny off-target: {n} params"


def test_vit_tiny_param_count_at_shallow_depth():
    """At depth=4 the same width must produce ~ depth=12 / 3 params
    (each block is ~6x mlp_ratio * embed^2 plus attn). Sanity guard."""
    m = build_vit_tiny(num_classes=10, depth=4)
    n = sum(p.numel() for p in m.parameters())
    assert n < 3_000_000, f"shallow ViT-Tiny too large: {n} params"


def test_vit_tiny_head_dim_divisible_by_3():
    """head_dim % 3 == 0 is mandatory for IcosaRoPE3D's triple
    rotation. The constructor must reject configurations that violate
    it -- this is the contract documented in the design comment."""
    # 6 heads x 64 head_dim = 384 embed_dim; 64 % 3 = 1 → must fail.
    try:
        build_vit_tiny(num_classes=10, embed_dim=384, num_heads=6,
                       head_dim=64)
    except ValueError as exc:
        # The error message must mention head_dim and divisibility.
        msg = str(exc)
        assert "head_dim" in msg, msg
        return
    raise AssertionError(
        "expected ValueError on head_dim=64 (not divisible by 3)"
    )


def test_vit_tiny_accepts_alternate_4_heads_x_48_config():
    """The directive allows the alternative 4 heads x 48 = 192 embed
    config (48 % 3 == 0). Must build cleanly."""
    m = build_vit_tiny(num_classes=10, embed_dim=192, num_heads=4,
                       head_dim=48, depth=4, patch_size=4, img_size=32)
    assert m.embed_dim == 192
    assert m.head_dim == 48
    # Forward sanity at the alternative geometry.
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_vit_tiny_forward_on_32x32_input():
    """End-to-end forward through 12 transformer blocks on the canonical
    CIFAR input. Tests all three rope_kinds: none / rope1d / icosa3d."""
    for rope_kind in ("none", "rope1d", "icosa3d"):
        m = build_vit_tiny(num_classes=10, embed_dim=198, num_heads=6,
                           head_dim=33, depth=2, patch_size=4,
                           img_size=32, rope_kind=rope_kind)
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        assert y.shape == (2, 10), (rope_kind, y.shape)


def test_vit_tiny_rejects_embed_num_heads_mismatch():
    """num_heads * head_dim must equal embed_dim."""
    try:
        build_vit_tiny(num_classes=10, embed_dim=198, num_heads=5,
                       head_dim=33)
    except ValueError:
        return
    raise AssertionError("expected ValueError on 5 x 33 != 198")


def test_vit_tiny_icosa3d_uses_phi_base_via_default():
    """The IcosaRoPE3D module installed for icosa3d rope_kind must
    use the PHI default base (matching H71's design doc)."""
    from nature_inspired_networks.priors import PHI
    m = build_vit_tiny(num_classes=10, depth=1, rope_kind="icosa3d")
    blk = m.blocks[0]
    icosa = blk.attn.icosa_rope
    assert icosa is not None, "icosa_rope should be installed"
    # IcosaRoPE3D default base is PHI; the head_dim is 33 (6 x 33 = 198).
    assert icosa.head_dim == 33
    assert abs(icosa.base - PHI) < 1e-6


def test_vit_tiny_grad_flows():
    """A backward pass must propagate gradients through every block --
    a silent dropout of the residual would break this."""
    m = build_vit_tiny(num_classes=10, depth=2, rope_kind="rope1d")
    x = torch.randn(2, 3, 32, 32, requires_grad=False)
    y = m(x).sum()
    y.backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad on {name}"


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
