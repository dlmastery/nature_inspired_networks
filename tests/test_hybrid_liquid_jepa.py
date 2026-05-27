"""Unit tests for H61 — Sacred Liquid JEPA Hybrid."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_liquid_jepa import (  # noqa: E402
    LiquidCFCCell,
    SacredLiquidJEPA,
    jepa_loss,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_cfc_cell_forward_shape():
    cell = LiquidCFCCell(d_in=16, d_hid=32)
    x = torch.randn(4, 16)
    h = cell(x)
    assert h.shape == (4, 32)
    assert torch.isfinite(h).all()


def test_cfc_cell_dt_defaults_to_one_over_phi():
    cell = LiquidCFCCell(d_in=8, d_hid=8)
    assert abs(cell.dt - 1.0 / PHI) < 1e-12


def test_cfc_cell_stateful_h_progresses():
    cell = LiquidCFCCell(d_in=4, d_hid=4)
    x = torch.randn(2, 4)
    h0 = torch.zeros(2, 4)
    h1 = cell(x, h0)
    h2 = cell(x, h1)
    # Two consecutive identical inputs from differing states must yield
    # different outputs (the cell carries state, not just input).
    assert not torch.allclose(h1, h2, atol=1e-6)


def test_cfc_tau_positive():
    cell = LiquidCFCCell(d_in=4, d_hid=6)
    assert (cell.tau > 0).all()


def test_encoder_forward_shape():
    model = SacredLiquidJEPA(in_channels=3, d_latent=32, n_blocks=2)
    x = torch.randn(2, 3, 16, 16)
    z = model.encode(x)
    assert z.shape == (2, 32)


def test_end_to_end_forward_shapes():
    model = SacredLiquidJEPA(in_channels=3, d_latent=32, n_blocks=2)
    x = torch.randn(2, 3, 16, 16)
    z, h, z_pred = model(x)
    assert z.shape == (2, 32)
    assert h.shape == (2, 32)
    assert z_pred.shape == (2, 32)


def test_jepa_loss_is_mse_and_stop_grads_target():
    z_pred = torch.randn(2, 8, requires_grad=True)
    z_target = torch.randn(2, 8, requires_grad=True)
    loss = jepa_loss(z_pred, z_target)
    # MSE-equivalence: same value as torch.nn.functional.mse_loss with detach.
    import torch.nn.functional as F
    ref = F.mse_loss(z_pred, z_target.detach())
    assert torch.allclose(loss, ref)
    # Gradients must NOT flow back to z_target (stop-grad).
    loss.backward()
    assert z_pred.grad is not None
    assert z_target.grad is None


def test_jepa_loss_shape_mismatch_rejected():
    try:
        jepa_loss(torch.randn(2, 4), torch.randn(2, 8))
        raise AssertionError("expected ValueError for shape mismatch")
    except ValueError:
        pass


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
