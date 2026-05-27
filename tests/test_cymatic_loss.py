"""H46 unit tests — Fourier-domain cymatic loss."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.cymatic_loss import (  # noqa: E402
    CymaticLossModule,
    chladni_target_spectrum,
    cymatic_loss,
    power_spectrum_2d,
)


def test_power_spectrum_shape_matches_input():
    """power_spectrum_2d preserves (B, C, H, W)."""
    x = torch.randn(2, 3, 8, 8)
    spec = power_spectrum_2d(x)
    assert spec.shape == x.shape
    assert torch.isfinite(spec).all()
    # Power spectrum is non-negative
    assert (spec >= 0).all()


def test_power_spectrum_dc_matches_sum_squared():
    """DC bin of |FFT|^2 equals (sum of input)^2 (Parseval-style sanity)."""
    x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
    spec = power_spectrum_2d(x)
    expected_dc = x.sum().item() ** 2
    assert abs(spec[0, 0, 0, 0].item() - expected_dc) < 1e-3


def test_target_spectrum_shape_and_normalisation():
    """Target spectrum has shape (H, W) and total mass == 1."""
    tgt = chladni_target_spectrum(8, 8, n_modes=6, band=(2, 5), seed=0)
    assert tgt.shape == (8, 8)
    assert torch.isfinite(tgt).all()
    assert (tgt >= 0).all()
    assert abs(tgt.sum().item() - 1.0) < 1e-5
    # Non-square shape supported
    tgt_rect = chladni_target_spectrum(4, 6, n_modes=4, band=(2, 4), seed=1)
    assert tgt_rect.shape == (4, 6)
    assert abs(tgt_rect.sum().item() - 1.0) < 1e-5


def test_target_spectrum_deterministic_with_seed():
    """Same (h, w, modes, band, seed) → bitwise identical target."""
    t1 = chladni_target_spectrum(8, 8, n_modes=8, band=(2, 5), seed=42)
    t2 = chladni_target_spectrum(8, 8, n_modes=8, band=(2, 5), seed=42)
    assert torch.equal(t1, t2)
    t3 = chladni_target_spectrum(8, 8, n_modes=8, band=(2, 5), seed=7)
    assert not torch.equal(t1, t3)


def test_cymatic_loss_is_scalar_and_finite():
    """cymatic_loss returns a finite scalar by default."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 8)
    tgt = chladni_target_spectrum(8, 8, n_modes=6, band=(2, 5), seed=0)
    loss = cymatic_loss(x, tgt)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
    # Non-negative (it's an MSE)
    assert loss.item() >= 0.0


def test_cymatic_loss_zero_when_perfect_match():
    """If the per-(B, C) normalised spectrum equals the target, loss == 0.

    Construct activations whose FFT2 produces, after per-(B, C)
    normalisation, exactly the target tensor: easiest is to set
    activations themselves so their normalised power-spectrum matches
    `target`. We use the inverse FFT of sqrt(target) to engineer this
    (the |FFT|^2 then equals target up to normalisation).
    """
    h, w = 4, 4
    tgt = chladni_target_spectrum(h, w, n_modes=4, band=(2, 4), seed=0)
    # sqrt(target) as |FFT|; ifft gives an activation whose FFT magnitude
    # squared equals target (sum-normalised). Use 1 channel for clarity.
    amp = torch.sqrt(tgt.clamp(min=0.0))
    # Use a real-valued construction: pick activations = ifft(amp) (real part).
    fft_target = amp.to(torch.complex64)
    activation = torch.fft.ifft2(fft_target).real.unsqueeze(0).unsqueeze(0)
    loss = cymatic_loss(activation, tgt)
    assert loss.item() < 1e-5, f"expected near-zero loss, got {loss.item()}"


def test_cymatic_loss_module_forward_caches_target():
    """CymaticLossModule lazily builds target and reuses across calls."""
    mod = CymaticLossModule(n_modes=6, band=(2, 5), seed=0)
    x1 = torch.randn(2, 3, 8, 8)
    l1 = mod(x1)
    assert l1.dim() == 0
    # Same shape → uses cached target
    cached_tgt = mod._target.clone()
    x2 = torch.randn(2, 3, 8, 8)
    l2 = mod(x2)
    assert torch.equal(mod._target, cached_tgt)
    assert l2.dim() == 0
    # Different shape → rebuilds
    x3 = torch.randn(2, 3, 16, 16)
    _ = mod(x3)
    assert mod._target.shape == (16, 16)


def test_cymatic_loss_module_gradient_flow():
    """Loss must be differentiable w.r.t. the activations."""
    torch.manual_seed(0)
    mod = CymaticLossModule(n_modes=4, band=(2, 4), seed=0)
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    loss = mod(x)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # At least one gradient entry must be non-zero
    assert x.grad.abs().sum().item() > 0.0


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
