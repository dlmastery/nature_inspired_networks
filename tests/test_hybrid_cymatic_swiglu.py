"""Unit tests for H75 — Harmonic Cymatic SwiGLU."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_cymatic_swiglu import (  # noqa: E402
    HarmonicCymaticSwiGLU,
)
from nature_inspired_networks.activations import PhiGELU  # noqa: E402
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_swiglu_forward_shape_2d_input():
    ffn = HarmonicCymaticSwiGLU(d_in=8)
    x = torch.randn(4, 8)
    y = ffn(x)
    assert y.shape == (4, 8)
    assert torch.isfinite(y).all()


def test_swiglu_forward_shape_3d_input():
    ffn = HarmonicCymaticSwiGLU(d_in=8, d_hidden=16)
    x = torch.randn(2, 5, 8)
    y = ffn(x)
    assert y.shape == (2, 5, 8)


def test_swiglu_uses_phigelu_gate():
    ffn = HarmonicCymaticSwiGLU(d_in=8)
    assert isinstance(ffn.gate_act, PhiGELU)
    # PhiGELU default beta == φ.
    assert abs(ffn.gate_act.beta.item() - PHI) < 1e-6


def test_swiglu_up_a_rows_have_he_equivalent_norm():
    """The cymatic-orthonormal init scales rows to He-equivalent L2."""
    ffn = HarmonicCymaticSwiGLU(d_in=16, d_hidden=8, cymatic_seed=0)
    import math
    he_std = math.sqrt(2.0 / 16)
    row_norms = ffn.up_a.weight.norm(dim=-1)
    expected = he_std * math.sqrt(16)
    assert torch.allclose(row_norms, torch.full_like(row_norms, expected), atol=1e-4)


def test_swiglu_default_hidden_is_phi2_scaled():
    ffn = HarmonicCymaticSwiGLU(d_in=10)
    assert ffn.d_hidden == round(10 * PHI * 2)


def test_swiglu_rejects_wrong_input_dim():
    ffn = HarmonicCymaticSwiGLU(d_in=8)
    try:
        ffn(torch.randn(4, 16))
        raise AssertionError("expected ValueError for wrong input dim")
    except ValueError:
        pass


def test_swiglu_gradient_flows_to_up_a_up_b_down():
    ffn = HarmonicCymaticSwiGLU(d_in=8, d_hidden=12)
    x = torch.randn(2, 8)
    y = ffn(x).sum()
    y.backward()
    assert ffn.up_a.weight.grad is not None
    assert ffn.up_b.weight.grad is not None
    assert ffn.down.weight.grad is not None


def test_swiglu_up_a_differs_from_default_kaiming():
    """The cymatic init must produce different weights than the default
    PyTorch Linear init."""
    ffn = HarmonicCymaticSwiGLU(d_in=8, d_hidden=8, cymatic_seed=0)
    plain = torch.nn.Linear(8, 8, bias=False)
    # The two init schemes should produce different weights (modulo a
    # vanishingly small probability of collision).
    assert not torch.allclose(ffn.up_a.weight, plain.weight, atol=1e-4)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
