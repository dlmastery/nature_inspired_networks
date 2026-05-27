"""Unit tests for H14 — Fibonacci Recurrent."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.fib_recurrent import (  # noqa: E402
    FibGRU,
    default_fib_hidden,
)


def test_fib_gru_forward_shape():
    """(B, T, input_dim) -> (B, T, output_dim)."""
    torch.manual_seed(0)
    m = FibGRU(input_dim=12, output_dim=4)
    x = torch.randn(2, 7, 12)
    y = m(x)
    assert y.shape == (2, 7, 4), y.shape


def test_fib_gru_default_hidden_sizes_are_fibonacci():
    """default hidden sizes are [8, 13, 21, 34]."""
    sizes = default_fib_hidden()
    assert sizes == [8, 13, 21, 34]
    m = FibGRU(input_dim=5)
    assert m.hidden_sizes == [8, 13, 21, 34]


def test_fib_gru_phi_gate_changes_output():
    """phi_gate=True must produce different outputs than phi_gate=False
    (the rescaled update probability changes h_t evolution).
    """
    torch.manual_seed(0)
    m_off = FibGRU(input_dim=8, hidden_sizes=[8, 13], output_dim=3,
                   phi_gate=False)
    torch.manual_seed(0)
    m_on = FibGRU(input_dim=8, hidden_sizes=[8, 13], output_dim=3,
                  phi_gate=True)
    # Copy weights from m_off into m_on so the ONLY difference is the
    # phi-gating rescale (parameter parity is the point of the prior).
    with torch.no_grad():
        for p_a, p_b in zip(m_off.parameters(), m_on.parameters()):
            p_b.copy_(p_a)
    x = torch.randn(2, 5, 8)
    y_off = m_off(x)
    y_on = m_on(x)
    # Same shape ...
    assert y_off.shape == y_on.shape
    # ... but the values must differ: phi-gating divides the update prob
    assert not torch.allclose(y_off, y_on, atol=1e-5), (y_off - y_on).abs().max()


def test_fib_gru_gradient_flow_at_depth():
    """Loss.backward() must produce finite gradients on every parameter
    even when stacked four-cells deep with the deepest hidden = 34.
    """
    torch.manual_seed(0)
    m = FibGRU(input_dim=10, output_dim=4, phi_gate=True)
    x = torch.randn(3, 12, 10)
    target = torch.randint(0, 4, (3, 12))
    logits = m(x)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, 4), target.reshape(-1)
    )
    loss.backward()
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_fib_gru_rejects_wrong_input_dim_and_rank():
    """Inputs with bad rank or feature dim must raise ValueError."""
    m = FibGRU(input_dim=8)
    # Wrong rank
    try:
        m(torch.randn(4, 8))  # 2-D, missing time axis
        raise AssertionError("expected ValueError for 2-D input")
    except ValueError:
        pass
    # Wrong feature dim
    try:
        m(torch.randn(2, 5, 7))  # 7 != 8
        raise AssertionError("expected ValueError for wrong feature dim")
    except ValueError:
        pass


def test_fib_gru_initial_state_is_optional():
    """Passing h0=None must give the same result as passing all zeros."""
    torch.manual_seed(0)
    m = FibGRU(input_dim=6, hidden_sizes=[8, 13])
    x = torch.randn(2, 4, 6)
    y1 = m(x)
    h0 = [torch.zeros(2, 8), torch.zeros(2, 13)]
    y2 = m(x, h0=h0)
    assert torch.allclose(y1, y2)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
