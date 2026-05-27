"""Unit tests for H68 — On-Device World Model."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.hybrid_world_model import (  # noqa: E402
    OnDeviceWorldModel,
    world_model_jepa_loss,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def test_world_model_encode_shape():
    m = OnDeviceWorldModel(in_channels=3, c0=8, d_latent=16)
    x = torch.randn(2, 3, 16, 16)
    z = m.encode(x)
    assert z.shape == (2, 16)
    assert torch.isfinite(z).all()


def test_world_model_predict_advances_state():
    m = OnDeviceWorldModel(in_channels=3, c0=8, d_latent=8)
    z = torch.randn(2, 8)
    z_pred, h = m.predict(z)
    assert z_pred.shape == (2, 8)
    assert h.shape == (2, 8)
    # A second predict with the carried hidden state must change h.
    z_pred2, h2 = m.predict(z, h)
    assert not torch.allclose(h, h2, atol=1e-6)


def test_world_model_forward_sequence_shapes():
    m = OnDeviceWorldModel(in_channels=3, c0=8, d_latent=12)
    seq = torch.randn(2, 4, 3, 16, 16)
    z_seq, z_pred = m(seq)
    assert z_seq.shape == (2, 4, 12)
    assert z_pred.shape == (2, 3, 12)


def test_world_model_jepa_loss_matches_cosine_and_stops_grad():
    z_pred = torch.randn(2, 8, requires_grad=True)
    z_target = torch.randn(2, 8, requires_grad=True)
    loss = world_model_jepa_loss(z_pred, z_target)
    assert torch.isfinite(loss)
    loss.backward()
    assert z_pred.grad is not None
    # stop-grad on target
    assert z_target.grad is None


def test_world_model_jepa_loss_shape_mismatch_rejected():
    try:
        world_model_jepa_loss(torch.randn(2, 4), torch.randn(2, 8))
        raise AssertionError("expected ValueError for shape mismatch")
    except ValueError:
        pass


def test_world_model_fibonacci_widths_monotonic():
    m = OnDeviceWorldModel(in_channels=3, c0=8, d_latent=16)
    # Stage widths must follow Fibonacci-channel rule (non-decreasing, /8).
    assert m.widths == sorted(m.widths)
    for w in m.widths:
        assert w % 8 == 0


def test_world_model_phi_constant_present_in_skip_block():
    """GoldenSkipBlock by default initialises its skip-path scalar at 1/φ."""
    m = OnDeviceWorldModel(in_channels=3, c0=8, d_latent=8)
    # Each GoldenSkipBlock instance exposes ``alpha``; default init = 1/PHI.
    assert abs(m.stage0.alpha.item() - 1.0 / PHI) < 1e-6
    assert abs(m.stage1.alpha.item() - 1.0 / PHI) < 1e-6
    assert abs(m.stage2.alpha.item() - 1.0 / PHI) < 1e-6


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
