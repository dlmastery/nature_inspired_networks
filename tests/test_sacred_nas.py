"""H45 unit tests — Sacred-NAS search-space + controller."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.blocks import (  # noqa: E402
    NaturePriorBlock,
    _FractalPath,
)
from nature_inspired_networks.priors import GroupConv2d, HexConv2d  # noqa: E402
from nature_inspired_networks.sacred_nas import (  # noqa: E402
    SacredNASController,
    phi_small_world,
    platonic_graph_adjacency,
    random_arch_sample,
    sacred_search_space,
)


def test_search_space_keys_present():
    """The descriptor exposes the three documented top-level keys."""
    ss = sacred_search_space()
    assert set(ss.keys()) == {"channel_choices", "block_types", "connectivity"}
    # channel_choices contains the two documented modes
    assert set(ss["channel_choices"].keys()) == {"fib", "phi"}
    # 4 block types in the exact restricted set
    assert ss["block_types"] == (NaturePriorBlock, _FractalPath, HexConv2d, GroupConv2d)
    # connectivity options include both Metatron and phi-small-world
    assert ss["connectivity"][0] is platonic_graph_adjacency
    assert ss["connectivity"][1] is phi_small_world


def test_search_space_channel_choices_are_fib_widths():
    """fib widths should be monotone non-decreasing and div-8."""
    ss = sacred_search_space(c0=16, n_stages=5)
    fib_widths = ss["channel_choices"]["fib"]
    phi_widths = ss["channel_choices"]["phi"]
    assert len(fib_widths) == 5 and len(phi_widths) == 5
    assert all(w % 8 == 0 for w in fib_widths)
    assert all(w % 8 == 0 for w in phi_widths)
    assert fib_widths == sorted(fib_widths)


def test_random_arch_sample_deterministic_with_seed():
    """Same seed → identical sample; different seed → (usually) different."""
    ss = sacred_search_space(c0=16, n_stages=5)
    a1 = random_arch_sample(ss, seed=0)
    a2 = random_arch_sample(ss, seed=0)
    assert a1 == a2
    # Same structural keys present
    assert set(a1.keys()) == {
        "channel_mode", "channel_widths", "block_picks", "connectivity_idx",
    }
    # block picks must be valid indices into the 4-op library
    for pick in a1["block_picks"]:
        assert 0 <= pick < 4
    assert 0 <= a1["connectivity_idx"] < 2
    # Different seeds *generally* differ — at least one of seeds 1..5 must.
    diffs = [random_arch_sample(ss, seed=s) != a1 for s in range(1, 6)]
    assert any(diffs), "expected ≥1 of seeds 1..5 to yield a different sample"


def test_phi_small_world_symmetric_zero_diagonal():
    """phi_small_world output must be symmetric with zero diagonal."""
    A = phi_small_world(13, seed=0)
    assert A.shape == (13, 13)
    assert torch.equal(A, A.t())
    assert torch.equal(A.diag(), torch.zeros(13))
    # Determinism: same seed → same matrix
    A2 = phi_small_world(13, seed=0)
    assert torch.equal(A, A2)
    # Different seed should (usually) differ
    A3 = phi_small_world(13, seed=42)
    assert not torch.equal(A, A3)


def test_controller_forward_shape():
    """Controller forward preserves shape (B, C, H, W) → (B, C, H, W)."""
    torch.manual_seed(0)
    ctrl = SacredNASController(c_in=16)
    x = torch.randn(2, 16, 8, 8)
    y = ctrl(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_controller_alpha_softmax_normalized():
    """Softmax of logits must sum to ~1 and have 4 components."""
    ctrl = SacredNASController(c_in=8)
    a = ctrl.alpha
    assert a.shape == (4,)
    assert abs(a.sum().item() - 1.0) < 1e-6
    # All entries strictly positive (softmax)
    assert (a > 0).all().item()
    # At initialisation logits are zero → uniform 0.25 weights.
    assert torch.allclose(a, torch.full((4,), 0.25), atol=1e-6)


def test_controller_logit_change_alters_alpha():
    """Setting a logit large should push α toward that op (sanity)."""
    ctrl = SacredNASController(c_in=8)
    with torch.no_grad():
        ctrl.logits.copy_(torch.tensor([5.0, 0.0, 0.0, 0.0]))
    a = ctrl.alpha
    assert a[0] > 0.9
    # Sum still 1.
    assert abs(a.sum().item() - 1.0) < 1e-6


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
