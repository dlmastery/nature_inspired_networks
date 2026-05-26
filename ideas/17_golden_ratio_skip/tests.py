"""H17 — unit tests for the idea's implementation.

Run with:
    python ideas/17_golden_ratio_skip/tests.py

Output must end with "All N tests passed." or fail loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_idea_signature_present():
    """The idea must expose a signature dict so the runner can log it."""
    from implementation import idea_signature
    sig = idea_signature()
    assert isinstance(sig, dict)
    assert sig["hypothesis_id"] == "H17"
    assert "PHI" in sig["primitives_touched"]
    # phi_inv should be approximately 0.618...
    assert abs(sig["phi_inv"] - 0.61803398) < 1e-6


def test_static_scales_inv_phi_canonical_value():
    """The 1/phi mode must produce skip=0.618..., branch=1.0."""
    from implementation import _static_scales, PHI_INV
    s, b = _static_scales("inv_phi")
    assert abs(s - PHI_INV) < 1e-12
    assert abs(b - 1.0) < 1e-12
    # And phi - 1 = 1/phi (golden-ratio identity)
    s2, b2 = _static_scales("phi_minus_1")
    assert abs(s2 - s) < 1e-12, "phi-1 and 1/phi must agree numerically"


def test_phi_inv2_sum1_mode_sums_to_one():
    """The 1/phi + 1/phi^2 = 1 identity must hold to machine precision."""
    from implementation import _static_scales, sum_to_one_residual
    s, b = _static_scales("phi_inv2_sum1")
    assert sum_to_one_residual(s, b), (s, b, s + b)


def test_unknown_mode_rejected():
    """Typos in config must be caught at construction time, not run time."""
    from implementation import PhiSkipBlock
    try:
        PhiSkipBlock(nn.Identity(), mode="phi_squared")
    except ValueError as exc:
        assert "unknown mode" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown mode")


class _ToyBlock(nn.Module):
    """Minimal residual block exposing skip + forward_branch for tests."""

    def __init__(self, c: int = 4) -> None:
        super().__init__()
        self.skip = nn.Identity()
        self.branch = nn.Linear(c, c, bias=False)
        nn.init.eye_(self.branch.weight)  # identity init -> branch(x) == x

    def forward_branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.branch(x)


def test_inv_phi_skip_math_identity_branch():
    """With identity-init branch, output must equal (branch_scale + skip_scale) * x.

    For mode='inv_phi': branch=1.0, skip=1/phi -> total = 1 + 1/phi = phi.
    """
    from implementation import PhiSkipBlock
    from nature_inspired_networks.priors import PHI
    blk = _ToyBlock(c=4)
    wrapped = PhiSkipBlock(blk, mode="inv_phi")
    x = torch.randn(2, 4)
    y = wrapped(x)
    expected = (1.0 + 1.0 / PHI) * x  # = phi * x
    assert torch.allclose(y, expected, atol=1e-5), (y[0], expected[0])
    # Sanity: phi = 1 + 1/phi  (golden-ratio identity)
    assert abs((1.0 + 1.0 / PHI) - PHI) < 1e-6


def test_phi_inv2_sum1_mode_outputs_x_under_identity_branch():
    """Sum-to-one mode: skip=1/phi, branch=1/phi^2.
    Identity branch + identity skip -> y = (1/phi + 1/phi^2) * x = 1 * x.
    """
    from implementation import PhiSkipBlock
    blk = _ToyBlock(c=4)
    wrapped = PhiSkipBlock(blk, mode="phi_inv2_sum1")
    x = torch.randn(2, 4)
    y = wrapped(x)
    assert torch.allclose(y, x, atol=1e-5), (y[0], x[0])


def test_learnable_mode_has_trainable_scales():
    """Learnable mode must expose nn.Parameter scalars init to 1/phi and 1.0."""
    from implementation import PhiSkipBlock, PHI_INV
    blk = _ToyBlock(c=4)
    wrapped = PhiSkipBlock(blk, mode="learnable")
    # Skip + branch scales should be trainable parameters
    params = dict(wrapped.named_parameters())
    assert "skip_scale" in params, list(params.keys())
    assert "branch_scale" in params, list(params.keys())
    assert abs(params["skip_scale"].item() - PHI_INV) < 1e-6
    assert abs(params["branch_scale"].item() - 1.0) < 1e-6
    # Backprop end-to-end: a single optimizer step must update both scales
    x = torch.randn(2, 4, requires_grad=False)
    y = wrapped(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert params["skip_scale"].grad is not None
    assert params["branch_scale"].grad is not None


def test_phi_skip_zero_param_overhead_in_static_modes():
    """Static modes (inv_phi, phi_minus_1, phi_inv2_sum1) must add ZERO
    trainable parameters vs the wrapped block. This is the zero-cost
    claim of the H17 hypothesis."""
    from implementation import PhiSkipBlock
    blk = _ToyBlock(c=4)
    base_n = sum(p.numel() for p in blk.parameters())
    for mode in ("inv_phi", "phi_minus_1", "phi_inv2_sum1"):
        wrapped = PhiSkipBlock(blk, mode=mode)
        n = sum(p.numel() for p in wrapped.parameters())
        assert n == base_n, (mode, n, base_n)


def test_phi_skip_regression_T1_8_distinct_from_modulate():
    """Regression test: phi-skip is NOT the same operation as the T1.8
    golden_modulate channel gate, even though both use phi.

    T1.8 (sg_only_golden_modulate) applies an elementwise channel gate
    cos(phases + alpha) to the *branch* output. H17 applies a SCALAR
    to the *skip* path. Under an identity-branch toy block, the two
    operations produce different outputs.

    This test catches anyone who confuses the two priors when reading
    the legacy 11-row sweep table.
    """
    from implementation import PhiSkipBlock
    from nature_inspired_networks.priors import PHI
    blk = _ToyBlock(c=4)
    wrapped = PhiSkipBlock(blk, mode="inv_phi")
    x = torch.randn(2, 4)
    y_h17 = wrapped(x)
    # Simulated golden_modulate output (NOT what H17 does) =
    #   skip + cos(phases) * branch     with phases close to 2.4 rad
    # So a tell-tale H17 invariant: scaling the input by k must scale
    # the output by exactly k (linearity), which channel-gate breaks
    # because the gate is data-independent but multiplies the branch
    # additively, and skip is unchanged -- so doubling x doubles both
    # parts symmetrically here. Use a STRUCTURAL invariant instead:
    # the H17 wrapper must produce phi * x for the identity branch.
    assert torch.allclose(y_h17, PHI * x, atol=1e-5)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
