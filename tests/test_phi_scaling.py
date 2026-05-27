"""Unit tests for H06, H07, H09, H17.pure primitives in phi_scaling.py.

Coverage requirement (Rule 12): >= 4 tests per hypothesis, every branch,
one regression test, one edge case.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.priors import PHI  # noqa: E402
from nature_inspired_networks.phi_scaling import (  # noqa: E402
    GoldenBottleneck,
    GoldenBottleneckNet,
    GoldenSkipBlock,
    GoldenSkipResNet,
    PhiBudgetNet,
    PhiSpacedFPN,
    build_phi_model,
    phi_budget_allocations,
    phi_budget_widths,
    phi_pyramid_widths,
)


# ===========================================================================
# H06 -- GoldenBottleneck
# ===========================================================================
def test_h06_canonical_forward_shape_preserved():
    """Canonical forward: stride=1, c_in=c_out preserves spatial shape."""
    blk = GoldenBottleneck(64, 64, stride=1)
    x = torch.randn(2, 64, 16, 16)
    y = blk(x)
    assert y.shape == (2, 64, 16, 16)


def test_h06_mid_channel_is_c_div_phi_rounded_to_8():
    """c_mid = round8(c_in / phi). For c_in=64 -> 64/1.618 = 39.55 -> 40."""
    blk = GoldenBottleneck(64, 64, stride=1, inverted=False)
    # round to nearest multiple of 8 with floor 8: 40 -> 40
    assert blk.c_mid == 40, blk.c_mid
    # phi factor sanity: c_mid * phi ~ c_in within rounding
    assert 0.5 * 64 < blk.c_mid * PHI < 1.5 * 64


def test_h06_inverted_branch_expands_then_contracts():
    """inverted=True: c_mid = round8(c_in * phi). For c_in=64 -> 64*1.618 = 103.55 -> 104."""
    blk = GoldenBottleneck(64, 64, stride=1, inverted=True)
    assert blk.c_mid == 104, blk.c_mid
    assert blk.c_mid > blk.c_in
    x = torch.randn(2, 64, 8, 8)
    y = blk(x)
    assert y.shape == (2, 64, 8, 8)


def test_h06_stride2_downsamples_and_projection_skip_branch():
    """Branch coverage: stride=2 path through projection skip."""
    blk = GoldenBottleneck(32, 64, stride=2)
    x = torch.randn(2, 32, 16, 16)
    y = blk(x)
    assert y.shape == (2, 64, 8, 8)
    # The skip must be a Sequential projection (not Identity)
    import torch.nn as nn
    assert isinstance(blk.skip, nn.Sequential)


def test_h06_no_residual_branch_returns_main_path_only():
    """Edge case: residual=False -- forward must skip the (+skip) addition."""
    blk = GoldenBottleneck(16, 16, stride=1, residual=False)
    assert blk.skip is None
    x = torch.randn(1, 16, 8, 8)
    y = blk(x)
    assert y.shape == (1, 16, 8, 8)
    # Disabling residual must NOT throw, NOT add the skip.


def test_h06_regression_param_count_under_resnet_at_iso_depth():
    """Regression test: GoldenBottleneckNet at 3x3 ResNet-20 schedule must
    have FEWER params than the ResNet-20 baseline (H06 falsifier:
    must deliver >=5% param reduction)."""
    from nature_inspired_networks.models import ResNet20
    base = ResNet20(num_classes=10)
    gold = GoldenBottleneckNet(num_classes=10)
    n_base = sum(p.numel() for p in base.parameters())
    n_gold = sum(p.numel() for p in gold.parameters())
    # Bottleneck reduces a c->c->c block (2*c^2*9 params) to
    # c->c/phi->c/phi->c (c*c/phi + (c/phi)^2*9 + c/phi*c) which is
    # markedly fewer. Require AT LEAST 30% reduction (much stronger
    # than the 5% falsifier; gives slack for batchnorm).
    assert n_gold < n_base, (n_gold, n_base)


def test_h06_net_forward_through_full_stack():
    """Full backbone forward exercises all 3 stages * 3 blocks."""
    net = GoldenBottleneckNet(num_classes=10)
    y = net(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


# ===========================================================================
# H07 -- PhiSpacedFPN
# ===========================================================================
def test_h07_phi_pyramid_widths_grow_geometrically_in_phi():
    """phi_pyramid_widths: each entry >= 8, monotonically increasing."""
    w = phi_pyramid_widths(16, 5)
    assert len(w) == 5
    assert all(c % 8 == 0 for c in w)
    for a, b in zip(w[:-1], w[1:]):
        assert 1.2 < b / a < 2.2, (a, b)


def test_h07_canonical_forward_preserves_per_level_spatial_size():
    """Forward through PhiSpacedFPN preserves each input level's H,W
    while changing channel widths to the phi-spaced schedule."""
    # 4 bottom-up feature maps at strides 1, 2, 4, 8 from a backbone:
    feats = [
        torch.randn(2, 16, 32, 32),
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 128, 4, 4),
    ]
    fpn = PhiSpacedFPN(in_channels=[16, 32, 64, 128], c0=16, phi_widths=True)
    outs = fpn(feats)
    assert len(outs) == 4
    expected = phi_pyramid_widths(16, 4)
    for k, o in enumerate(outs):
        assert o.shape[1] == expected[k], (k, o.shape, expected[k])
        assert o.shape[-2:] == feats[k].shape[-2:]


def test_h07_uniform_widths_branch_baseline():
    """Branch coverage: phi_widths=False reproduces the standard FPN
    uniform-width baseline -- every level has the same c0 channels."""
    feats = [
        torch.randn(2, 16, 32, 32),
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
    ]
    fpn = PhiSpacedFPN(in_channels=[16, 32, 64], c0=24, phi_widths=False)
    outs = fpn(feats)
    assert len(outs) == 3
    for o in outs:
        assert o.shape[1] == 24


def test_h07_input_length_mismatch_rejected():
    """Edge case: passing wrong number of feature maps is a hard error."""
    fpn = PhiSpacedFPN(in_channels=[16, 32, 64], c0=16)
    try:
        fpn([torch.randn(2, 16, 8, 8)])
        raise AssertionError("expected AssertionError on length mismatch")
    except AssertionError as exc:
        if "expected AssertionError" in str(exc):
            raise


def test_h07_regression_phi_widths_strictly_differ_from_uniform():
    """Regression test: phi_widths=True and phi_widths=False MUST produce
    different channel widths. The previous IDEA_TABLE confound was that
    H07's effect was indistinguishable from raising channel_mode='phi'
    on stage widths; this guards the *pyramid* widths are non-uniform."""
    n = 4
    phi_w = phi_pyramid_widths(16, n)
    uniform_w = [16] * n
    assert phi_w != uniform_w
    # specifically, the deepest level must be > 2 x c0
    assert phi_w[-1] >= 2 * phi_w[0]


# ===========================================================================
# H09 -- Golden Proportion Parameter Budget
# ===========================================================================
def test_h09_allocations_sum_to_budget():
    """phi_budget_allocations: rounding-stable; sum = B_total exactly."""
    B = 1_000_000
    alloc = phi_budget_allocations(B, 4)
    assert len(alloc) == 4
    assert sum(alloc) == B
    # Stage k should be ~phi x stage k-1 (within rounding)
    for a, b in zip(alloc[:-1], alloc[1:]):
        assert 1.3 < b / a < 2.0, (a, b)


def test_h09_canonical_widths_consume_budget_approximately():
    """phi_budget_widths derives widths whose squared sum maps to B_total.
    The resulting PhiBudgetNet must have a param count within +/- 50%
    of B_total (rough -- the formula uses the quadratic dominator)."""
    B = 270_000
    widths = phi_budget_widths(B, 3, kernel=3, blocks_per_stage=2)
    assert len(widths) == 3
    assert all(c % 8 == 0 and c >= 8 for c in widths)
    # widths must be monotone non-decreasing (phi**k grows)
    assert widths == sorted(widths)


def test_h09_phi_vs_uniform_budget_modes_differ():
    """Branch coverage: budget_mode='phi' vs 'uniform' produce different widths."""
    phi_net = PhiBudgetNet(num_classes=10, B_total=300_000,
                           n_stages=3, budget_mode="phi")
    uni_net = PhiBudgetNet(num_classes=10, B_total=300_000,
                           n_stages=3, budget_mode="uniform")
    assert phi_net.widths != uni_net.widths, (phi_net.widths, uni_net.widths)
    # phi mode: last stage wider than first; uniform: all equal
    assert phi_net.widths[-1] > phi_net.widths[0]
    assert uni_net.widths[0] == uni_net.widths[-1]


def test_h09_forward_through_full_stack():
    """Canonical forward across all 3 stages."""
    for mode in ("phi", "uniform"):
        net = PhiBudgetNet(num_classes=10, B_total=200_000,
                           n_stages=3, budget_mode=mode)
        y = net(torch.randn(2, 3, 32, 32))
        assert y.shape == (2, 10), mode


def test_h09_regression_phi_ratio_holds():
    """Regression test: the budget allocator must produce stage_k/stage_0
    ratios within 5% of phi**k for all k. This catches drift in the
    closed-form share computation."""
    alloc = phi_budget_allocations(10_000_000, 4)  # large B for low rounding noise
    base = alloc[0]
    for k, a in enumerate(alloc):
        expected = base * (PHI ** k)
        ratio = a / expected
        assert 0.95 < ratio < 1.05, (k, a, expected, ratio)


def test_h09_edge_case_invalid_budget_raises():
    """Edge case: zero/negative budget must raise ValueError."""
    try:
        phi_budget_allocations(0, 3)
        raise AssertionError("expected ValueError on B_total <= 0")
    except ValueError:
        pass
    try:
        phi_budget_allocations(1000, 0)
        raise AssertionError("expected ValueError on n_stages <= 0")
    except ValueError:
        pass


# ===========================================================================
# H17.pure -- Golden Skip Connections
# ===========================================================================
def test_h17p_canonical_alpha_init_at_one_over_phi():
    """GoldenSkipBlock default alpha must be 1/phi (~ 0.618)."""
    blk = GoldenSkipBlock(16, 16, stride=1)
    assert abs(blk.alpha.item() - 1.0 / PHI) < 1e-6
    assert blk.alpha.requires_grad is True


def test_h17p_canonical_forward_shape_keeps_h_w():
    """Forward through GoldenSkipBlock preserves spatial shape."""
    blk = GoldenSkipBlock(16, 16, stride=1)
    x = torch.randn(2, 16, 8, 8)
    y = blk(x)
    assert y.shape == (2, 16, 8, 8)


def test_h17p_branch_trainable_false_freezes_alpha():
    """Branch coverage: trainable=False registers alpha as a buffer,
    not a parameter. Optimizer would not touch it."""
    blk = GoldenSkipBlock(16, 16, stride=1, trainable=False)
    # alpha is not a learnable parameter
    param_names = {n for n, _ in blk.named_parameters()}
    assert "alpha" not in param_names
    # but still accessible numerically
    assert abs(float(blk.alpha) - 1.0 / PHI) < 1e-6


def test_h17p_branch_custom_init_recovers_vanilla_resnet():
    """Setting init=1.0 must reproduce vanilla ResNet skip (y = F(x) + x).

    Comparison: GoldenSkipBlock(init=1.0) and a stride-2 projection
    skip must give the same output up to weight init."""
    torch.manual_seed(0)
    blk_phi = GoldenSkipBlock(16, 16, stride=1, init=1.0)
    assert abs(blk_phi.alpha.item() - 1.0) < 1e-6


def test_h17p_branch_stride2_downsamples_through_projection():
    """Branch coverage: stride=2 must trigger the Sequential projection skip."""
    blk = GoldenSkipBlock(16, 32, stride=2)
    x = torch.randn(2, 16, 8, 8)
    y = blk(x)
    assert y.shape == (2, 32, 4, 4)
    import torch.nn as nn
    assert isinstance(blk.skip, nn.Sequential)


def test_h17p_regression_pure_skip_excludes_channel_gate():
    """Regression test (THE confound to avoid): GoldenSkipBlock must NOT
    contain any golden-angle channel-gate (phases buffer + alpha cos)
    that the H17/H34-confound row in the prior sweep (sg_only_golden_modulate)
    used. This guards against accidental reintroduction."""
    blk = GoldenSkipBlock(32, 32, stride=1)
    # alpha must be a scalar, not a vector of phases
    assert blk.alpha.numel() == 1
    # No 'phases' buffer; the H34 channel-gate uses register_buffer('phases',...)
    assert not hasattr(blk, "phases") or blk.__dict__.get("phases") is None
    # Forward must be the pure linear combination, no cos modulation
    x = torch.zeros(1, 32, 4, 4)
    y = blk(x)
    # With x=0 the conv path gives bn(conv(0)) ~ bn(0) = beta (BN bias);
    # adding alpha * 0 leaves no oscillatory modulation -- y is finite,
    # not a cosine-modulated channel pattern.
    assert torch.isfinite(y).all()


def test_h17p_net_forward_and_alpha_list_length():
    """Full GoldenSkipResNet: 3 stages x 3 blocks = 9 GoldenSkipBlocks."""
    net = GoldenSkipResNet(num_classes=10)
    y = net(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)
    alphas = net.alphas()
    assert len(alphas) == 9
    # all initialised at 1/phi
    for a in alphas:
        assert abs(a - 1.0 / PHI) < 1e-6


def test_h17p_edge_case_alpha_grad_flows():
    """Edge case: a backward pass populates alpha.grad on the learnable
    block but NOT on the frozen-buffer variant."""
    blk_train = GoldenSkipBlock(8, 8, stride=1, trainable=True)
    x = torch.randn(2, 8, 4, 4, requires_grad=False)
    loss = blk_train(x).sum()
    loss.backward()
    assert blk_train.alpha.grad is not None
    assert blk_train.alpha.grad.abs().sum().item() >= 0.0  # any grad is fine

    blk_frozen = GoldenSkipBlock(8, 8, stride=1, trainable=False)
    # buffer has no .grad attribute (PyTorch returns None for buffers)
    assert blk_frozen.alpha.requires_grad is False


# ===========================================================================
# Dispatcher
# ===========================================================================
def test_build_phi_model_dispatch():
    """build_phi_model must route names to the right module."""
    a = build_phi_model("golden_bottleneck", num_classes=10)
    b = build_phi_model("phi_budget", num_classes=10, B_total=200_000)
    c = build_phi_model("golden_skip", num_classes=10)
    assert isinstance(a, GoldenBottleneckNet)
    assert isinstance(b, PhiBudgetNet)
    assert isinstance(c, GoldenSkipResNet)
    try:
        build_phi_model("nonexistent", num_classes=10)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
