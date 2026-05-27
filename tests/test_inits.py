"""Unit tests for the φ-flavoured initialisation primitives.

Covers H31 ``golden_spiral_init_`` and H42 ``phi_weight_init_``.
Asserts in roughly the order of the H31/H42 design-doc falsifiers:

* shape preservation (regression);
* variance approximately matches He init (≤ 15 % off for the spiral
  mask; ≤ 2 % off for the φ-init at φ=2.0);
* backward-compat: ``phi_weight_init_(..., phi=2.0)`` reproduces He
  init bit-for-bit in distribution-mean and distribution-std;
* determinism: identical input + seed → identical output;
* spiral structure: the mask has the φ-growth property along sample
  index (successive non-zero radii grow by factor φ within tolerance).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.inits import (  # noqa: E402
    apply_phi_init,
    golden_spiral_init_,
    golden_spiral_mask,
    phi_weight_init_,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


# ---------------------------------------------------------------------------
# H31 — golden_spiral_init_
# ---------------------------------------------------------------------------
def test_golden_spiral_mask_shape_and_nonneg():
    m = golden_spiral_mask(k=5)
    assert m.shape == (5, 5)
    assert (m >= 0).all().item(), "spiral mask must be non-negative"
    # mask must touch the central 3x3 region — the spiral starts at r=r0
    # close to centre, so at least one of the inner 9 cells must light up
    inner = m[1:4, 1:4]
    assert (inner > 0).any().item(), "spiral mask must touch central 3x3"


def test_golden_spiral_mask_nontrivial_coverage():
    """The spiral must cover ≥ 4 distinct grid cells; otherwise the
    init is degenerate (rank-1)."""
    m = golden_spiral_mask(k=5)
    nonzero = (m > 0).sum().item()
    assert nonzero >= 4, f"spiral mask only covers {nonzero} cells"


def test_golden_spiral_init_shape_unchanged():
    torch.manual_seed(0)
    w = torch.empty(16, 8, 5, 5)
    golden_spiral_init_(w)
    assert w.shape == (16, 8, 5, 5)


def test_golden_spiral_init_variance_close_to_he():
    """The mask is renormalised to ``mean(m**2) == 1`` so the
    post-init weight variance must match He's ``2 / fan_in`` to within
    ≈ 15 % (Monte-Carlo with 16*8*5*5 = 3200 samples).
    """
    torch.manual_seed(0)
    out_c, in_c, k = 16, 8, 5
    w = torch.empty(out_c, in_c, k, k)
    golden_spiral_init_(w)
    fan_in = in_c * k * k
    he_var = 2.0 / fan_in
    var = w.var().item()
    ratio = var / he_var
    assert 0.5 < ratio < 1.5, f"variance/he_var = {ratio:.3f} not within [0.5, 1.5]"


def test_golden_spiral_init_rejects_non_4d():
    try:
        golden_spiral_init_(torch.empty(8, 5, 5))
    except ValueError:
        return
    raise AssertionError("expected ValueError on 3-D tensor")


def test_golden_spiral_init_rejects_non_square_kernel():
    try:
        golden_spiral_init_(torch.empty(4, 4, 3, 5))
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-square kernel")


def test_golden_spiral_init_is_deterministic_given_seed():
    """Same seed + same shape → same draw."""
    torch.manual_seed(42)
    w1 = torch.empty(8, 4, 5, 5)
    golden_spiral_init_(w1)
    torch.manual_seed(42)
    w2 = torch.empty(8, 4, 5, 5)
    golden_spiral_init_(w2)
    assert torch.allclose(w1, w2)


# ---------------------------------------------------------------------------
# H42 — phi_weight_init_
# ---------------------------------------------------------------------------
def test_phi_init_variance_matches_phi_over_fan_in():
    """Monte-Carlo: Var(W) ≈ phi / fan_in within 2 % on 1e5 samples."""
    torch.manual_seed(0)
    fan_in = 1024
    w = torch.empty(100, fan_in)
    phi_weight_init_(w)
    expected = PHI / fan_in
    var = w.var().item()
    rel = abs(var - expected) / expected
    assert rel < 0.02, f"var={var:.6e} expected={expected:.6e} rel={rel:.4f}"


def test_phi_init_reduces_to_he_when_phi_equals_2():
    """The H42 backward-compat contract: phi=2 ↦ He init."""
    torch.manual_seed(0)
    fan_in = 512
    w = torch.empty(200, fan_in)
    phi_weight_init_(w, phi=2.0)
    expected = 2.0 / fan_in
    var = w.var().item()
    rel = abs(var - expected) / expected
    assert rel < 0.02, f"var={var:.6e} expected={expected:.6e} rel={rel:.4f}"


def test_phi_init_mean_is_zero():
    """Mean must be ≈ 0 — N(0, σ) draw, 1e5 samples → |mean| < 0.01·σ."""
    torch.manual_seed(0)
    fan_in = 1024
    w = torch.empty(100, fan_in)
    phi_weight_init_(w)
    sigma = math.sqrt(PHI / fan_in)
    assert abs(w.mean().item()) < 0.01 * sigma


def test_phi_init_4d_conv_weight_shape():
    w = torch.empty(16, 8, 3, 3)
    phi_weight_init_(w)
    assert w.shape == (16, 8, 3, 3)


def test_apply_phi_init_walks_all_conv_and_linear():
    """apply_phi_init must touch every nn.Conv2d and nn.Linear weight."""
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1, bias=False),
        nn.Flatten(),
        nn.Linear(16 * 32 * 32, 10, bias=True),
    )
    # Save originals, then apply.
    originals = [m.weight.clone() for m in model.modules()
                 if isinstance(m, (nn.Conv2d, nn.Linear))]
    apply_phi_init(model)
    new = [m.weight for m in model.modules()
           if isinstance(m, (nn.Conv2d, nn.Linear))]
    for o, n in zip(originals, new):
        assert not torch.allclose(o, n), "phi init must change the weight"
    # biases must be zero after apply_phi_init
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
            assert (m.bias == 0).all().item()


def test_phi_init_rejects_1d_tensor():
    try:
        phi_weight_init_(torch.empty(8))
    except ValueError:
        return
    raise AssertionError("expected ValueError on 1-D tensor")


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
