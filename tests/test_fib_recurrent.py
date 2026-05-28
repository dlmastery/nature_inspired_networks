"""Unit tests for H14 — Fibonacci Recurrent."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.fib_recurrent import (  # noqa: E402
    FibGRU,
    LOGIT_PHI_RECIPROCAL,
    default_fib_hidden,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


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
    """phi_gate=True must produce different outputs than phi_gate=False.

    The prior is now a *bias init* on bias_ih[hidden:2*hidden] (the
    update-gate input-to-hidden bias slot). The two models are
    constructed under identical seeds; phi_gate=True additionally
    overwrites the update-gate bias slot with logit(1/phi). The
    forward pass therefore diverges immediately.
    """
    torch.manual_seed(0)
    m_off = FibGRU(input_dim=8, hidden_sizes=[8, 13], output_dim=3,
                   phi_gate=False)
    torch.manual_seed(0)
    m_on = FibGRU(input_dim=8, hidden_sizes=[8, 13], output_dim=3,
                  phi_gate=True)
    # Sanity: m_on's update-gate bias slot is now logit(1/phi) while
    # m_off's is the default PyTorch GRUCell init (~uniform [-1/sqrt(h),
    # +1/sqrt(h)]), so the parameter sets cannot match — and importantly
    # we do NOT overwrite them, because doing so would wipe the bias
    # init that is the whole point of the phi_gate prior.
    x = torch.randn(2, 5, 8)
    y_off = m_off(x)
    y_on = m_on(x)
    # Same shape ...
    assert y_off.shape == y_on.shape
    # ... but the values must differ: phi-gating's bias init biases the
    # update gate toward 1/phi at init, changing h_t evolution.
    assert not torch.allclose(y_off, y_on, atol=1e-5), (y_off - y_on).abs().max()


def test_h14_update_gate_bias_init_is_logit_one_over_phi():
    """Mechanism pin: every cell's update-gate bias slot
    (bias_ih[hidden:2*hidden]) must equal logit(1/phi) = log(phi)
    within float tolerance when phi_gate=True.

    PyTorch's GRUCell flattens biases as [b_r | b_z | b_n], so the
    middle slot of length ``hidden_size`` is the update-gate bias.
    """
    torch.manual_seed(0)
    m = FibGRU(input_dim=6, hidden_sizes=[8, 13, 21], output_dim=2,
               phi_gate=True)
    expected = math.log(PHI)  # logit(1/phi) = log(phi)
    # Numeric match to the module's exposed constant.
    assert abs(LOGIT_PHI_RECIPROCAL - expected) < 1e-12, (
        LOGIT_PHI_RECIPROCAL, expected
    )
    for k, cell in enumerate(m.cells):
        h = cell.hidden_size
        z_bias = cell.bias_ih.data[h:2 * h]
        target = torch.full((h,), expected, dtype=z_bias.dtype)
        assert torch.allclose(z_bias, target, atol=1e-6), (
            k, z_bias[:3].tolist(), expected
        )
        # And the other two slots (reset, candidate) must NOT all equal
        # logit(1/phi) — they should retain the default uniform init,
        # which is overwhelmingly likely to differ.
        r_bias = cell.bias_ih.data[0:h]
        n_bias = cell.bias_ih.data[2 * h:3 * h]
        assert not torch.allclose(r_bias, target, atol=1e-3)
        assert not torch.allclose(n_bias, target, atol=1e-3)

    # And with phi_gate=False the update-gate bias slot must NOT
    # equal logit(1/phi).
    torch.manual_seed(0)
    m2 = FibGRU(input_dim=6, hidden_sizes=[8, 13, 21], output_dim=2,
                phi_gate=False)
    for cell in m2.cells:
        h = cell.hidden_size
        z_bias = cell.bias_ih.data[h:2 * h]
        target = torch.full((h,), expected, dtype=z_bias.dtype)
        assert not torch.allclose(z_bias, target, atol=1e-3)


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
